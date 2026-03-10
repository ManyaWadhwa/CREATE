
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prompt_bank import CLASS_SIZE_PROMPT, TRIPLE_FACTUAL_CHECKING_PROMPT
from inference import _create_litellm_generator


def json_parsing(response):
    """
    Generic JSON parser that handles various formats:
    - Markdown code blocks (```json ... ```)
    - Raw JSON strings
    - Python dict/list literals
    - Extra whitespace and formatting
    """
    if not response or not isinstance(response, str):
        return None

    import json
    import ast

    # Normalize the response string
    text = response.strip()

    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    text = re.sub(r'```json\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()

    # Remove leading/trailing brackets/quotes if they're part of markdown formatting
    text = text.strip('`').strip()

    # Try JSON parsing first
    try:
        result = json.loads(text)
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Python literal_eval for dict/list structures
    try:
        result = ast.literal_eval(text)
        return result
    except (ValueError, SyntaxError, TypeError):
        # TypeError can occur with unhashable types (e.g., dict in set)
        pass

    # Try to extract JSON from text that might have extra content
    # Look for JSON-like structures in the text
    json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)

    if json_match:
        candidate = json_match.group(0)
        # json_match.group(0) always returns a string, so we can skip the isinstance check
        try:
            result = json.loads(candidate)
            return result
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            result = ast.literal_eval(candidate)
            return result
        except (ValueError, SyntaxError, TypeError):
            # TypeError can occur with unhashable types (e.g., dict in set)
            pass

    return None

# -----------------------------------------------------------------------------
# Path parsing
# -----------------------------------------------------------------------------

def _extract_answer_content(text: str) -> Optional[str]:
    """Extract content between <answer>...</answer> tags; returns last match or None."""
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None


def _parse_triple(s: str) -> Optional[Tuple[str, str, str]] | None:
    """Parse a string like '(s, p, o)' or 's, p, o' into a (subject, predicate, object) tuple."""
    s = s.strip().strip("'").strip('"').strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return None
    return tuple(parts)


def _fix_bad_closing_curly_middle(text: str) -> str:
    """Fix malformed JSON where } appears instead of ] before next key (e.g. "2":)."""
    pattern = re.compile(r'(\]\s*)\}(?=\s*,\s*\n\s*"\d+"\s*:)')
    return pattern.sub(r'\1]', text)


def _fix_bad_closing_curly_end(text: str) -> str:
    """Fix malformed JSON where trailing } should be ]."""
    pattern = re.compile(r'\}(\s*,?\s*\n)')
    return pattern.sub(r']', text)



class Path:
    """Parse raw path predictions into structured triples."""

    def __init__(self, raw_prediction: str | list, source_path=None, verbalized=False):
        """Initialize from raw prediction string or list; extracts answer from </think> or </reason> tags if present."""
        if isinstance(raw_prediction, str):
            self.raw_prediction = raw_prediction
        elif isinstance(raw_prediction, (list, np.ndarray)):
            self.raw_prediction = raw_prediction[0]
        else:
            raise ValueError("raw_prediction must be str or list")
        if self.raw_prediction is None:
            self.predicted_path_str = None
            self.reasoning = None
        elif "</think>" in self.raw_prediction:
            self.predicted_path_str = self.raw_prediction.split("</think>")[-1].strip()
            self.reasoning = self.raw_prediction.split("</think>")[0]
        elif "</reason>" in self.raw_prediction:
            self.predicted_path_str = self.raw_prediction.split("</reason>")[-1].strip()
            self.reasoning = self.raw_prediction.split("</reason>")[0]
        else:
            self.predicted_path_str = self.raw_prediction
            self.reasoning = None
        self.source_path = source_path
        self.parsed_paths = []
        self.verbalized = verbalized

    def parse_path_from_text(self) -> Optional[List[List[Tuple[str, str, str]]]]:
        """
        Parse raw prediction text into structured paths (list of triples per path).

        Handles JSON/dict formats, <answer> tags, and various triple encodings.
        Returns list of paths, each path a list of (subject, predicate, object) tuples.
        """
        if self.predicted_path_str is None:
            return None
        if len(self.predicted_path_str) == 0:
            if "<answer>" in self.raw_prediction and "</answer>" in self.raw_prediction:
                self.predicted_path_str = _extract_answer_content(self.raw_prediction)
        if "<answer>" in self.predicted_path_str or "</answer>" in self.predicted_path_str:
            text_between_answer_tags = _extract_answer_content(self.predicted_path_str)
        else:
            text_between_answer_tags = self.predicted_path_str
        if text_between_answer_tags is None:
            self.parsed_paths = None
            return None
        if '<answer>' in text_between_answer_tags.lower():
            text_between_answer_tags = text_between_answer_tags.replace("<answer>", "").strip()
        if "</answer>" in text_between_answer_tags.lower():
            text_between_answer_tags = text_between_answer_tags.replace("</answer>", "").strip()
        if 'json' in text_between_answer_tags:
            text_after_json = text_between_answer_tags.split("json")[-1].replace("```", "").strip()
        else:
            text_after_json = text_between_answer_tags
        if text_after_json is None:
            self.parsed_paths = None
            return None
        if 'json' in text_after_json:
            text_between_answer_tags = text_after_json.replace("json", "").replace("```", "").strip()
        else:
            text_between_answer_tags = text_after_json
        text_between_answer_tags = text_between_answer_tags.lower().strip()
        if text_between_answer_tags in ("{\n}", "{}"):
            self.parsed_paths = None
            return None
        try:
            text_between_answer_tags = re.sub(r'//.*', '', text_between_answer_tags)
            text_between_answer_tags = re.sub(r'/\*[\s\S]*?\*/', '', text_between_answer_tags)
            text_between_answer_tags = re.sub(r'^\s*#.*$', '', text_between_answer_tags, flags=re.M)
            text_between_answer_tags = re.sub(r'\\\\\"', r'\"', text_between_answer_tags)
            text_between_answer_tags = re.sub(r'\(\s*"', r'["', text_between_answer_tags)
            text_between_answer_tags = re.sub(r'"\s*\)', r'"]', text_between_answer_tags)
            text_between_answer_tags = re.sub(r',(\s*[}\]])', r'\1', text_between_answer_tags)
            text_json = json.loads(text_between_answer_tags.strip())
        except Exception:
            text_json = None
        if not text_json:
            try:
                text_json = ast.literal_eval(text_between_answer_tags)
            except Exception:
                text_json = None
        if text_json is None:
            fixed_text = _fix_bad_closing_curly_middle(text_between_answer_tags)
            fixed_text = _fix_bad_closing_curly_end(fixed_text)
            try:
                text_json = json.loads(fixed_text)
            except Exception:
                self.parsed_paths = None
                return None
        if isinstance(text_json, list):
            return [text_json]
        if text_json is Ellipsis or text_json is ...:
            self.parsed_paths = None
            return None
        predicted_paths = []
        for key in text_json:
            path_data = text_json[key]
            if isinstance(path_data, dict) and "path" in path_data:
                path = path_data["path"]
            else:
                path = path_data
            finalized_path = []
            if isinstance(path, int) or len(path) == 0 or isinstance(path, dict):
                continue
            elif isinstance(path[0], str):
                if '(' not in path[0]:
                    for i in range(0, len(path), 3):
                        if len(path) == i + 3:
                            finalized_path.append((path[i], path[i + 1], path[i + 2]))
                else:
                    for t in path:
                        try:
                            finalized_path.append(ast.literal_eval(t))
                        except Exception:
                            finalized_path.append(_parse_triple(t))
            else:
                for triple in path:
                    if isinstance(triple, dict):
                        finalized_path.append([triple[t] for t in triple])
                    elif isinstance(triple, (list, tuple, set)):
                        finalized_path.append(tuple(triple))
                    elif isinstance(triple, str):
                        triple_p = _parse_triple(triple)
                        if triple_p is not None:
                            finalized_path.append(triple_p)
            predicted_paths.append(finalized_path)
        predicted_paths = list(filter(None, predicted_paths))
        self.parsed_paths = predicted_paths
        return predicted_paths


# -----------------------------------------------------------------------------
# Validity check
# -----------------------------------------------------------------------------

def check_path_validity(
    path: Optional[List],
    entity1: str,
    entity2: str,
) -> Tuple[str, int]:
    """
    Check if a path is valid given start/end entities.

    Path format: list of triples [(s,p,o), (s,p,o), ...]

    Checks:
    (a) entity1 occurs in the first triple and entity2 in the last triple
    (b) last triple does not have subject == object
    (c) consecutive triples have continuity (subject/object of next in prev)

    Returns:
        (valid, reason): valid is 1 or 0, reason is explanation string
    """
    def _check_triple_none(triple):
        for t in triple:
            if t is None:
                return True
        return False

    def _fix_triple(p):
        return [str(i) for i in p]

    if path is None:
        return 0, "path is empty"
    path = [list(t) for t in path]  # copy to avoid mutating input
    for i, triple in enumerate(path):
        if triple is None or len(triple) != 3:
            return 0, f"triple {i} is not length three"
    valid = False
    reason = ""
    if entity1.lower() in str(path[0]).lower() and entity2.lower() in str(path[-1]).lower():
        valid = True
    if len(path[-1]) != 3 or str(path[-1][0]).lower() == str(path[-1][2]).lower():
        valid = False
        reason = "the last triple has the same subject and object"
    for i in range(len(path) - 1):
        path[i] = _fix_triple(path[i])
        path[i + 1] = _fix_triple(path[i + 1])
        if _check_triple_none(path[i]):
            valid = False
            reason = f"One part of the triple at index {i} is None"
        elif _check_triple_none(path[i + 1]):
            valid = False
            reason = f"One part of the triple at index {i + 1} is None"
        elif (
            path[i + 1][0].lower().strip() in path[i][0].lower().strip()
            or path[i + 1][0].lower().strip() in path[i][-1].lower().strip()
            or path[i + 1][-1].lower().strip() in path[i][0].lower().strip()
            or path[i + 1][-1].lower().strip() in path[i][-1].lower().strip()
        ):
            valid = True
        else:
            valid = False
            reason = f"there is no continuity for the pair ({i}, {i+1})"
            break
    return reason, 1 if valid else 0


# -----------------------------------------------------------------------------
# PathEvaluator
# -----------------------------------------------------------------------------

def _json_parsing(response: str) -> Optional[Any]:
    """Parse JSON from LLM response; strips markdown, tries json.loads then ast.literal_eval."""
    if not response or not isinstance(response, str):
        return None
    text = response.strip()
    text = re.sub(r'```json\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip('`').strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        pass
    json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
    if json_match:
        candidate = json_match.group(0)
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError, TypeError):
            pass
    return None



class PathEvaluator:
    """Evaluator for strength and factuality scoring."""

    def __init__(self, model_name, use_vllm, server_url, generation_params):
        """Initialize with LLM generator (litellm) for strength/factuality prompts."""
        self.generator = _create_litellm_generator(model_name, use_vllm, server_url, generation_params)

    def validity_checker(self, path, entity1: str, entity2: str) -> Tuple[int, str]:
        """Check if path is valid between entity1 and entity2; returns (1|0, reason)."""
        return check_path_validity(path, entity1, entity2)

    def strength_scoring_prompt(self, path) -> str:
        """Return formatted prompt for class-size (strength) scoring of a path."""
        return CLASS_SIZE_PROMPT.format(**{"path": path})

    def get_factuality_prompt(self, path) -> str:
        """Return formatted prompt for factuality/hallucination checking of a path."""
        return TRIPLE_FACTUAL_CHECKING_PROMPT.format(**{"path": path})

    def get_factuality(self, paths):
        """Score all paths for factuality; returns (outputs dict, parsed_factuality_scores, total_cost, num_prompts)."""
        print(f"Scoring {len(paths)} paths for factuality!")
        outputs = {"raw_factuality_response": [],
                   "parsed_factuality_response": [],
                   "factuality_prompt": []}
        total_cost = 0.0
        num_prompts = 0
        for index, predicted_path in enumerate(paths):
            prompts = []
            raw_responses = []
            parsed_responses = []
            for path in predicted_path:
                # for each path in the connections predicted
                prompt = self.get_factuality_prompt(path)
                response, cost = self.generator(prompt)
                total_cost += cost
                num_prompts += 1
                parsed_response = self.parse_factuality(response)
                prompts.append(prompt)
                raw_responses.append(response)
                parsed_responses.append(parsed_response)
            outputs['raw_factuality_response'].append(raw_responses)
            outputs['parsed_factuality_response'].append(parsed_responses)
            outputs['factuality_prompt'].append(prompts)

        return outputs, outputs['parsed_factuality_response'], total_cost, num_prompts

    def get_valid(self, paths, entity1, entity2):
        """Check validity of each path between corresponding entity1/entity2 pairs; returns (outputs, parsed)."""
        outputs = {"validity_raw_response": [],
                   "validity_parsed_responses": []}
        for index, (e1, e2, predicted_path) in enumerate(zip(entity1, entity2, paths)):
            raw_responses = []
            parsed_responses = []
            for path in predicted_path:
                valid, reason = self.validity_checker(path, e1, e2)
                raw_responses.append(valid)
                parsed_responses.append(reason)
            outputs['validity_raw_response'].append(raw_responses)
            outputs['validity_parsed_responses'].append(parsed_responses)
        return outputs, outputs['validity_parsed_responses']


    def get_strength(self, paths):
        """Score all paths for strength (class size); returns (outputs dict, min_strength per instance, total_cost, num_prompts)."""
        print(f"Scoring {len(paths)} paths for Strength!")
        outputs = {"raw_strength_response": [],
                   "parsed_strength_response": [],
                   "strength_prompt": [],
                   "min_strength": []}
        total_cost = 0.0
        num_prompts = 0
        for index, predicted_path in enumerate(paths):
            # one instance
            prompts = []
            raw_responses = []
            parsed_responses = []
            for path in predicted_path:
                # for each path in the connections predicted
                prompt = self.strength_scoring_prompt(path)
                response, cost = self.generator(prompt)
                total_cost += cost
                num_prompts += 1
                parsed_response = self.parse_class_size(response)
                prompts.append(prompt)
                raw_responses.append(response)
                parsed_responses.append(parsed_response)

            outputs['raw_strength_response'].append(raw_responses)
            outputs['parsed_strength_response'].append(parsed_responses)
            # we don't consider the last connection for any generated path because that is a prompt constraint rather than a model generated relation
            outputs['min_strength'].append([min(parsed[:-1]) for parsed in parsed_responses])
            outputs['strength_prompt'].append(prompts)

        return outputs, outputs['min_strength'], total_cost, num_prompts

    def get_eval(self, data):
        """Run full evaluation: factuality, strength, validity; adds eval columns to data.
        Returns (data, eval_stats) where eval_stats has 'factuality', 'strength' (costs USD),
        'factuality_prompts', 'strength_prompts' (prompt counts)."""
        paths = list(data['parsed_paths'])
        factuality_eval, parsed_fact, factuality_cost, factuality_prompts = self.get_factuality(paths)
        strength_eval, parsed_strength, strength_cost, strength_prompts = self.get_strength(paths)
        validity_eval, parsed_valid = self.get_valid(paths, data['entity_a'], data['entity_b'])
        data['factuality_eval'] = factuality_eval
        data['strength_eval'] = strength_eval
        data['validity_eval'] = validity_eval
        data['factuality_scores'] = parsed_fact
        data['strength_scores'] = parsed_strength
        data['validity_scores'] = parsed_valid
        eval_stats = {
            "factuality": factuality_cost,
            "strength": strength_cost,
            "factuality_prompts": factuality_prompts,
            "strength_prompts": strength_prompts,
        }
        return data, eval_stats

    def parse_factuality(self, response: str) -> List[List[float]]:
        """Parse factuality response into per-triple scores (1=factual, 0=hallucinated)."""
        result = _json_parsing(response.lower())
        if result is None:
            return []
        if isinstance(result, list):
            return [1 if "not" in str(r).lower() else 0.0 for r in result]
        hallucination = result.get("judgments", [])
        if len(hallucination) == 0:
            return []
        # Per-triple factuality: [1, 0, 1] for 3 triples (1=factual, 0=hallucinated)
        return [1 if "not" in str(h).lower() else 0.0 for h in hallucination]

    def parse_regex_class_size(self, response):
        """
            function that parses class size list from a response like:
           [
      {
        "explanation": "For Class A (screenwriter, Thomas Gilou): There are likely a limited number of works where Thomas Gilou is the screenwriter, so this class is relatively small. For Class B (Retirement Home, screenwriter): 'Retirement Home' is a specific work, and it's common for films or shows to have more than one screenwriter. Thus, the larger class is Class B.",
        "judgment": 2
      },
      {
        "explanation": "For Class A (screenwriter, Kev Adams): Kev Adams is known primarily as an actor, and there are likely very few works where he is listed as a screenwriter, making this class small. For Class B (Retirement Home, screenwriter): As mentioned before, 'Retirement Home' can have multiple screenwriters, making this class larger.",
        "judgment": 2
      },
      {
        "explanation": "For Class A (occupation, voice actor): The occupation 'voice actor' is held by many individuals, making this class large. For Class B (Kev Adams, occupation): Kev Adams has several occupations, but they are limited in number. Therefore, the larger class is Class A.",
        "judgment": 1000
      }
    ]
        Uses regex to extract all "judgment":<int> values from the response.
        """
        if not response or len(response) == 0:
            return []

        # Look for all occurrences of "judgment":<int> pattern
        # Pattern: "judgment" followed by optional whitespace, colon, optional whitespace, and an integer
        pattern = r'"judgment"\s*:\s*(\d+)'
        matches = re.findall(pattern, response, re.IGNORECASE)

        if matches:
            # Convert string matches to integers
            try:
                judgments = [int(match) for match in matches]
                return judgments
            except ValueError:
                return []

        return []

    def parse_class_size(self, response):
        """Parse class-size (strength) response; maps judgment integers to 1–5 scale."""
        def mapping(class_size):
            try:
                class_size = int(class_size)
            except:
                return 0
            if class_size <= 10:
                return 5
            if 10 < class_size < 100:
                return 4
            if 100 <= class_size < 500:
                return 3
            if 500 <= class_size < 5000:
                return 2
            return 1
        result = json_parsing(response.lower())
        if isinstance(result, list):
            return [mapping(r.get("judgment", 0)) for r in result]
        parse_regex = self.parse_regex_class_size(response)
        if len(parse_regex) == 0:
            return []
        return [mapping(j) for j in parse_regex]
