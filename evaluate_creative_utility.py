#!/usr/bin/env python3
"""
Loads input from .jsonl file or HuggingFace dataset with:
  - query: str
  - path_prediction: list[str]
  - eval_model_name: str (optional; use --model_name to override)

Runs evaluation: strength and factuality only.
Aggregates strength + factuality into the combined creative utility metric

Uses eval_support.py for support (Path, PathEvaluator, utility). Inference via litellm:
  --vllm: use server_url as vLLM endpoint; else use model name only.
"""

import argparse
import json
import os
import sys
from keyhandler import KeyHandler
KeyHandler.set_env_key()
import numpy as np
import pandas as pd
from datasets import load_dataset

from path_evaluator import Path, PathEvaluator
from inference import _create_litellm_generator
from creative_utility import get_utility_dataset


# from eval_support import (
#     get_utility_dataset_v2,
# )


def _extract_answer(text: str) -> str:
    """Extract content from <answer>...</answer> tags, or return full text. Default parser."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else text



def load_and_prepare_data(input_file: str, split: str = "train") -> pd.DataFrame:
    """Load data from JSONL file or HuggingFace dataset. Raises ValueError if unsupported."""
    if input_file.lower().endswith(".jsonl"):
        try:
            data = pd.read_json(input_file, lines=True)
        except Exception as e:
            raise ValueError(f"input file not supported: {e}") from e
    else:
        try:
            dataset = load_dataset(input_file)
            if split in dataset:
                data = dataset[split].to_pandas()
            else:
                raise ValueError(f"dataset not supported: {split}")
        except Exception as e:
            raise ValueError(f"input file not supported: {e}") from e

    required = ["query", "path_prediction"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Dataset must have column '{col}'. Found: {list(data.columns)}")
  
    return data

def compute_creative_utility(
    data: pd.DataFrame,
    patience: float = 0.9,
    factuality_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Aggregate strength and factuality into creative utility via get_utility_dataset_v2.
    """
    data = get_utility_dataset(
        data,
        strength_column="strength_scores",
        paths_column="parsed_paths",
        valid_column='validity_scores',
        patience=patience,
        factuality_threshold=factuality_threshold,
        factuality_column="factuality_scores",
    )
    return data


def main():
    """CLI entry point: load data, run strength/factuality eval, compute creative utility, optionally save."""
    parser = argparse.ArgumentParser(
        description="Evaluate HF dataset (strength + factuality) and compute creative utility"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input path: .jsonl file (local) or HuggingFace dataset (e.g. org/dataset-name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split for HuggingFace (ignored for .jsonl) (default: train)",
    )
    parser.add_argument("--response_column",
                        type=str,
                        default='path_prediction',
                        help="Name of the key/column with the model responses")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="Eval model for strength/factuality. Overrides eval_model_name from dataset if provided.",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM for inference",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="",
        help="Server URL for vLLM/open-source models (optional)",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=0.9,
        help="Patience for creative utility (default: 0.9)",
    )
    parser.add_argument(
        "--factuality_threshold",
        type=float,
        default=1.0,
        help="Factuality threshold for filtering paths (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results (JSONL). If not set, prints summary only.",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading input: {args.input_file}")
    data = load_and_prepare_data(args.input_file, split=args.split)
    if args.response_column not in data.columns:
        raise ValueError("The response column passed to --response_column must be in data.")
    # Determine eval model
    eval_model = args.model_name
    
    # Parse paths
    print("Parsing path predictions...")
    all_parsed = []
    for _, row in data.iterrows():
        preds = row["path_prediction"] # this is a list of model predictions
        if isinstance(preds, str):
            preds = json.loads(preds) if preds.strip() else []
        preds = preds or []
        filtered = []
        for raw in preds:
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                continue
            p = Path(raw_prediction=raw, source_path=None)
            result = p.parse_path_from_text()
            if result:
                filtered.append(result)
        all_parsed.append(filtered if filtered else [[]])

    data["parsed_paths"] = [paths[0] for paths in all_parsed]
    # data['paths_0'] = [paths[0][0] for paths in all_parsed]
    
    # paths_0 is already in correct format: list of paths per instance, each path = list of triples
    # data["parsed_paths_for_eval"] = list(data["paths_0"])

    # Create litellm-based inference generator
    generation_params = {"temperature": 0.0, "max_tokens": 4096}

    evaluator = PathEvaluator( model_name=eval_model,
        use_vllm=args.vllm,
        server_url=args.server_url,
        generation_params=generation_params)

    print(data.head())
    data, eval_stats = evaluator.get_eval(data)

    # Compute creative utility
    print("Computing creative utility...")
    data_eval = compute_creative_utility(
        data,
        patience=args.patience,
        factuality_threshold=args.factuality_threshold,
    )

    # Summary
    util = data_eval["utility_scores"]
    mean_max_strength = data_eval["strength_scores"].apply(
        lambda x: np.max(x) if x else 0.0
    ).mean()

    mean_mean_strength = data_eval["strength_scores"].apply(
        lambda x: np.mean(x) if x else 0.0
    ).mean()

    def _avg_factuality(x):
        if not x:
            return 0.0
        try:
            return np.mean([np.mean(i) if i else 0.0 for i in x])
        except Exception:
            return 0.0

    fact = data_eval["factuality_scores"].apply(_avg_factuality)

    avg_paths_per_instance = data_eval["parsed_paths"].apply(
        lambda x: len(x) if x else 0
    ).mean()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  N instances (with paths): {len(data_eval)}")
    print(f"  Avg paths per instance:    {avg_paths_per_instance:.2f}")
    print(f"  Creative utility (mean):  {util.mean():.4f} (std: {util.std():.4f})")
    print(f"  Avg strength (mean of max):      {mean_max_strength:.4f}")
    print(f"  Avg strength (mean of mean):      {mean_mean_strength:.4f}")
    print(f"  Avg factuality (mean):    {fact.mean():.4f}")
    print(f"  Factuality prompting:     {eval_stats['factuality_prompts']} prompts, ${eval_stats['factuality']:.6f}")
    print(f"  Strength prompting:       {eval_stats['strength_prompts']} prompts, ${eval_stats['strength']:.6f}")
    print(f"  Total eval:                {eval_stats['factuality_prompts'] + eval_stats['strength_prompts']} prompts, ${eval_stats['factuality'] + eval_stats['strength']:.6f}")
    print("=" * 60)

    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        # Serialize list columns for JSONL
        for col in data_eval.columns:
            if data_eval[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                data_eval[col] = data_eval[col].apply(
                    lambda x: json.dumps(x.tolist() if hasattr(x, "tolist") else x)
                    if isinstance(x, (list, np.ndarray))
                    else x
                )
        data_eval.to_json(out_path, lines=True, orient="records")
        print(f"\nResults saved to: {out_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
