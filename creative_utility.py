import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import json

_sentence_model = None


def _get_sentence_model():
    """Return lazily-loaded SentenceTransformer model (all-MiniLM-L6-v2) for embeddings."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def path_to_string(triples) -> str:
    """
    Convert a list of knowledge-graph triples into a readable string representation.

    Each triple (h, r, t) is formatted as "h [r] t", joined by " | ".
    The first triple omits the head and starts with "[r] t".
    """
    tokens = []
    triples = [t for t in triples if t is not None]
    for i, trip in enumerate(triples):
        if len(trip) == 3:
            h, r, t = trip
            h, r, t = str(h).lower(), str(r).lower(), str(t).lower()
            if i == 0:
                tokens.append(f" [{r}] {t}")
            else:
                tokens.append(f"{h} [{r}] {t}")
        else:
            tokens.append(str(trip))
    return " | ".join(tokens)


def get_similarity(path_list) -> np.ndarray:
    """
    Compute pairwise cosine distances between path embeddings.

    Paths are encoded via sentence-transformers; returns an (n, n) distance matrix
    where larger values indicate more dissimilar paths.
    """
    from sklearn.metrics.pairwise import cosine_distances
    path_string = [path_to_string(p[:-1]) if len(p) > 1 else path_to_string(p) for p in path_list]
    model = _get_sentence_model()
    emb = model.encode(path_string, normalize_embeddings=True)
    return cosine_distances(emb)


def get_lexical_similarity(path1, path2) -> float:
    """
    Compute lexical distance between two paths based on relation overlap.

    Uses Jaccard distance on relation sets (1 - |intersection|/|union|).
    Returns 1.0 if either path is empty; 0.0 if both have no relations.
    """
    path1 = [p for p in path1 if p is not None]
    path2 = [p for p in path2 if p is not None]
    if len(path1) == 0 or len(path2) == 0:
        return 1.0
    relations1 = set(str(r[1]).strip().lower() if len(r) > 1 else str(r).strip().lower() for r in path1[:-1])
    relations2 = set(str(r[1]).strip().lower() if len(r) > 1 else str(r).strip().lower() for r in path2[:-1])
    if len(relations1) == 0 and len(relations2) == 0:
        return 0.0
    intersection = len(relations1 & relations2)
    union = len(relations1 | relations2)
    if union == 0:
        return 1.0
    return 1.0 - (intersection / union)


def get_lexical_similarity_list(path_list) -> np.ndarray:
    """
    Build pairwise lexical distance matrix for a list of paths.

    Returns symmetric (n, n) matrix of get_lexical_similarity values.
    """
    n = len(path_list)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = get_lexical_similarity(path_list[i], path_list[j])
            distance_matrix[i, j] = sim
            distance_matrix[j, i] = sim
    return distance_matrix


def greedy_select_not_optimized(strengths, dist_matrix, n) -> Tuple:
    """
    Greedily select n items that maximize both strength and diversity.

    Starts with the highest-strength item, then iteratively adds items that
    maximize min_distance_to_selected * strength, balancing diversity (far
    from already selected items) with individual strength.

    Args:
        strengths: Per-item strength scores (higher is better).
        dist_matrix: Pairwise distance matrix of shape (m, m), where m is the
            number of items. Larger values indicate more diverse pairs.
        n: Number of items to select.

    Returns:
        Tuple of (selected_indices, selected_scores, selected_strengths,
        selected_distances), where selected_scores are the diversification
        scores (min_dist * strength) used for each selection step.
    """
    strengths = list(map(float, strengths))
    m = len(strengths)
    if dist_matrix.shape != (m, m):
        raise ValueError(f"dist_matrix must be shape {(m, m)}, got {dist_matrix.shape}")
    if n <= 0:
        return [], [], [], []
    if n > m:
        raise ValueError(f"n={n} cannot exceed m={m}")
    first = np.argmax(strengths).item()
    selected = [first]
    remaining = set(range(m))
    remaining.remove(first)
    selected_scores = [strengths[first]]
    selected_distance = [0]
    selected_strengths = [strengths[first]]
    while len(selected) < n:
        best_item = None
        best_score = -float("inf")
        best_dist = None
        best_strength = None
        for item in remaining:
            min_dist = float("inf")
            for k in selected:
                d = float(dist_matrix[item, k])
                if d < min_dist:
                    min_dist = d
            score = min_dist * strengths[item]
            if score > best_score:
                best_score = score
                best_item = item
                best_strength = strengths[item]
                best_dist = min_dist
        selected.append(best_item)
        selected_scores.append(best_score)
        remaining.remove(best_item)
        selected_strengths.append(best_strength)
        selected_distance.append(best_dist)
    return selected, selected_scores, selected_strengths, selected_distance


def saturating_drop(x, ub=0.7, lb=0.4) -> np.ndarray:
    """
    Apply saturating nonlinearity to distance values.

    Values in (0.7, 1.0] map to 1; lower values use a smooth curve
    0.5 * (1 - cos(pi * (x/0.7)^2)).
    """
    x = np.asarray(x)
    return np.where((x > 0.7) & (x <= 1.0), 1, 0.5 * (1 - np.cos(np.pi * (x / 0.7) ** 2)))


def get_utility_dataset(
    data: pd.DataFrame,
    strength_column: str,
    paths_column: str,
    factuality_column: str,
    valid_column: str,
    patience: float = 0.9,
    factuality_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Compute creative utility scores for each row in the dataset.

    Filters paths by validity and factuality, computes semantic distance between
    paths, greedily selects diverse high-strength paths, and aggregates into
    marginal/utility scores with patience-based discounting.

    Adds columns: pairwise_distance_average, raw_utility_scores, utility_scores.
    """
    all_utility = []
    all_pairwise_distance_average = []
    count = 0

    for index, row in data.iterrows():
        paths = row[paths_column]
        if paths is None:
            all_utility.append([])
            all_pairwise_distance_average.append([])
            count += 1
            continue
        if isinstance(row[paths_column], str):
            paths = json.loads(row[paths_column])
        elif isinstance(row[paths_column], np.ndarray):
            paths = [list(p) for p in list(row[paths_column])]
        strength_values = row[strength_column]
        factuality_values = row[factuality_column]
        # filter by valid and factual threshold..
        if valid_column is not None:
            valid_paths = row[valid_column]
            if isinstance(valid_paths, (list, np.ndarray)) and valid_paths is not None:
                valid_paths = [float(v) for v in list(valid_paths)]
            else:
                valid_paths = []
            n = min(len(paths), len(strength_values), len(valid_paths), len(factuality_values))
            indices = [i for i in range(n) if valid_paths[i] == 1]
            paths = [paths[i] for i in indices]
            strength_values = [strength_values[i] for i in indices]
            if factuality_values is not None:
                # filter by factuality threshold
                factuality_values = [factuality_values[i] for i in indices]
                factuality_values = [np.mean(i) for i in list(factuality_values)]
                factuality_values = [1 if f >= factuality_threshold else 0 for f in factuality_values]
                n = min(len(paths), len(strength_values), len(factuality_values))
                indices = [i for i in range(n) if factuality_values[i] == 1]
                paths = [paths[i] for i in indices]
                strength_values = [strength_values[i] for i in indices]
        if len(paths) != len(strength_values):
            continue
        selected_d, selected_s = [], []
        utility_scores, marginal_scores, pairwise_distance_average = [], [], []
        if len(paths) > 0:
            distance_matrix = get_similarity(paths)
            distance_matrix = saturating_drop(distance_matrix)
            # just make sure it is within boundaries..
            distance_matrix[distance_matrix > 1] = 1
            np.fill_diagonal(distance_matrix, 1.0)
            upper_tri = np.triu_indices_from(distance_matrix, k=1)
            pairwise_distance_average.append(float(np.mean(distance_matrix[upper_tri])))

            # sort based on best strength and highest pairwise distance
            _, sorted_scores, selected_s, selected_d = greedy_select_not_optimized(
                strength_values, distance_matrix, n=len(strength_values)
            )
            p = patience
            marginal_scores = [u * (p ** (i - 1)) if i != 0 else u for i, u in enumerate(sorted_scores)]
            utility_scores = [marginal_scores[0]] # first element
            for i in range(1, len(marginal_scores)):
                utility_scores.append(float(np.sum(marginal_scores[: i + 1])))
        else:
            count += 1
        all_utility.append(utility_scores)
        all_pairwise_distance_average.append(pairwise_distance_average)
    data[f"pairwise_distance_average"] = all_pairwise_distance_average
    data[f"raw_utility_scores"] = all_utility
    data[f"utility_scores"] = data[f"raw_utility_scores"].apply(
        lambda x: x[-1] if x else 1
    )
    return data
