# CREATE
Authors: Manya Wadhwa, Tiasa Singha Roy, Harvey Lederman,  Junyi Jessy Li, Greg Durrett

[Benchmark🤗](https://huggingface.co/datasets/wadhma/CREATE) | [Leaderboard](https://manyawadhwa.github.io/projects/create/) |

## Overview
CREATE is a benchmark designed to measure associative reasoning in models. This benchmark evaluates whether models can construct valid, diverse, and insightful paths that connect two concepts through intermediate entities or relationships. We introduce creative utility, a unified metric that captures both the quality and diversity of generated connections. 

<!-- Creative utility includes a patience parameter (p), which controls how utility is distributed across the ranked list of responses. -->


Example query:

>“What are different ways to connect Dakota Johnson to people who starred in fantasy or science-fiction movies?”

We want the model to generate paths like:

>Dakota Johnson co-stars with Chris Evans in Materialists; Chris Evans played Captain America in The Avengers.

>Dakota Johnson is the stepdaughter of Antonio Banderas, who voiced Puss in Boots in Shrek.

These responses illustrate associative creativity: each path is coherent, factually grounded, and offers a distinct conceptual route between the two endpoints.



## 🛠️ Installation

First, install the required dependencies: 
```bash
python3 -m venv create_env 
source create_env/bin/activate
pip install -r requirements.txt 
```


## 🚀 Running CREATE

Our [benchmark](https://huggingface.co/datasets/wadhma/CREATE) is available on huggingface! The following code snippet shows how to access the benchmark. 

```python 
from datasets import load_dataset
data = load_dataset('wadhma/CREATE')['train'].to_pandas() 
print(data['query']) ## the benchmark questions
```

The base prompt we use in the paper for is included in [prompt.py](prompt.py).


## 📊 Evaluation

### Setup

1. Copy `keyhandler_template.py` to `keyhandler.py` and add your API keys (OpenAI, Anthropic, and/or HuggingFace as needed for your evaluator model).
2. Ensure your predictions file has the required format (see below).

### Input Format

Your input must be a `.jsonl` file or a HuggingFace dataset with at least:

| Column | Type | Description |
|--------|------|-------------|
| `query` | str | The benchmark question |
| `path_prediction` | list[str] | Model-generated paths (one string per path) |

### Running Evaluation

The script computes **strength** and **factuality** scores, then aggregates them into the **creative utility** metric. In the paper we use `gpt-oss-120b` for evaluations; you can use any model supported by LiteLLM. Note: we are in the process of updating this to bespoke-curator for more efficient inference.

```bash
python evaluate_creative_utility.py --input_file <path_or_hf_dataset> [options]
```

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--input_file` | (required) | Path to `.jsonl` file or HuggingFace dataset (e.g. `org/dataset-name`) |
| `--split` | `train` | Dataset split (HuggingFace only) |
| `--response_column` | `path_prediction` | Column name for model responses |
| `--model_name` | `gpt-4.1-mini-2025-04-14` | Evaluator model for strength/factuality |
| `--patience` | `0.9` | Patience parameter for creative utility |
| `--factuality_threshold` | `1.0` | Threshold for filtering paths |
| `--output` | None | Output path for results (JSONL) |
| `--vllm` | False | Use vLLM endpoint for inference |
| `--server_url` | `""` | Server URL for vLLM/open-source models |

**Example:**

```bash
python evaluate_creative_utility.py --input_file predictions.jsonl --model_name gpt-4o --output results.jsonl
```

The script prints a summary including mean creative utility, average strength, average factuality, and evaluation cost. See [evaluate_creative_utility.py](evaluate_creative_utility.py) for full details.