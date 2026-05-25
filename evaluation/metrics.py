"""
metrics.py — Evaluation primitives for the two-tier unlearning report.
"""

from __future__ import annotations

import numpy as np


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=np.float32).flatten()
    b = np.asarray(v2, dtype=np.float32).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = set(recommended[:k])
    rel = set(relevant)
    hit = len(rec_k & rel)
    return hit / min(k, len(rel))


def rank_overlap(list1: list, list2: list, k: int) -> float:
    """Jaccard overlap of top-k items in the two ranked lists."""
    a = set(list1[:k])
    b = set(list2[:k])
    if not (a | b):
        return 0.0
    return len(a & b) / len(a | b)


def membership_inference_score(
    model, forget_indices: list[int], retain_indices: list[int],
    edge_index, num_users: int,
) -> float:
    """
    Simple confidence-based MIA: score how strongly the user→forget edges
    are still preferred relative to user→retain edges. 0.5 = random,
    >0.5 means model still "remembers" forgotten items.
    """
    import torch
    model.eval()
    with torch.no_grad():
        emb = model.propagate(edge_index)
        users = emb[:num_users]
        movies = emb[num_users : num_users + model.num_movies]
        if not forget_indices or not retain_indices:
            return 0.5
        f_idx = torch.tensor(forget_indices, dtype=torch.long)
        r_idx = torch.tensor(retain_indices, dtype=torch.long)
        f_scores = (users @ movies[f_idx].T).flatten()
        r_scores = (users @ movies[r_idx].T).flatten()
        # Fraction of forget scores above the median retain score
        median_r = float(torch.median(r_scores).item())
        return float((f_scores > median_r).float().mean().item())


def embedding_reversion_score(
    before: np.ndarray, after: np.ndarray, target: np.ndarray
) -> float:
    """
    For Tier 2: how close the post-erase embedding is to the pre-session target.
    Returns 1.0 - cosine_distance(after, target). Higher = better reversion.
    """
    return 1.0 - cosine_distance(after, target)


# ── extended metrics for the rich CLI evaluation suite ───────────────────────
def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    """Fraction of top-k recs that are in the relevant set."""
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    rel = set(relevant)
    return sum(1 for r in rec_k if r in rel) / len(rec_k)


def hit_rate_at_k(recommended: list, relevant: list, k: int) -> float:
    """1.0 if any relevant item is in top-k, else 0.0."""
    if not relevant:
        return 0.0
    rel = set(relevant)
    return 1.0 if any(r in rel for r in recommended[:k]) else 0.0


def mean_reciprocal_rank(recommended: list, relevant: list) -> float:
    """1 / rank of first relevant item (1-indexed)."""
    rel = set(relevant)
    for i, r in enumerate(recommended, 1):
        if r in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    """Binary-relevance NDCG@k."""
    rel = set(relevant)
    dcg = 0.0
    for i, r in enumerate(recommended[:k], 1):
        if r in rel:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(len(rel), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def coverage(all_recommended_lists: list[list], universe_size: int) -> float:
    """Fraction of catalogue that appears in any user's top-k."""
    if universe_size <= 0:
        return 0.0
    seen: set = set()
    for lst in all_recommended_lists:
        seen.update(lst)
    return len(seen) / float(universe_size)


def intra_list_diversity(recommended: list, item_embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance within a top-k list (higher = more diverse)."""
    if len(recommended) < 2:
        return 0.0
    vecs = item_embeddings[recommended]
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(cosine_distance(vecs[i], vecs[j]))
    return float(np.mean(dists)) if dists else 0.0


def embedding_drift_norm(before: np.ndarray, after: np.ndarray) -> float:
    """L2 norm of (after - before). Magnitude of embedding shift."""
    a = np.asarray(before, dtype=np.float32).flatten()
    b = np.asarray(after, dtype=np.float32).flatten()
    return float(np.linalg.norm(b - a))


def kendall_tau_distance(list1: list, list2: list, k: int) -> float:
    """
    Normalised Kendall-tau distance of two ranked lists' top-k intersection.
    Returns 0 (identical order) to 1 (fully reversed).
    """
    common = [x for x in list1[:k] if x in set(list2[:k])]
    if len(common) < 2:
        return 0.0
    pos2 = {x: i for i, x in enumerate(list2[:k])}
    inversions = 0
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            if pos2[common[i]] > pos2[common[j]]:
                inversions += 1
    max_inv = n * (n - 1) / 2
    return float(inversions / max_inv) if max_inv > 0 else 0.0
