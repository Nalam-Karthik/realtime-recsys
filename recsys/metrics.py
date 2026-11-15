import numpy as np


def recall_at_k(ranked, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / ideal


def summarize(rankings: dict, truth: dict, ks=(10, 50)) -> dict:
    """Mean recall@k / ndcg@k over users that have ground truth."""
    users = [u for u in rankings if truth.get(u)]
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([recall_at_k(rankings[u], truth[u], k) for u in users]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(rankings[u], truth[u], k) for u in users]))
    out["users"] = len(users)
    return out
