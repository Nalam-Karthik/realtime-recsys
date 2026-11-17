"""Ranking features. The same code builds offline training rows and online request rows,
so there is no training/serving skew.

Time-dependent features are computed relative to the request time (the user's latest
interaction), so popularity is "what was popular when this user was watching".
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 86400
FEATURES = [
    "tt_score", "retrieval_rank", "item_log_pop", "item_avg_rating", "pop_30d", "pop_180d",
    "item_age_years", "genre_affinity", "user_log_n", "max_sim_recent", "mean_sim_recent",
]


@dataclass
class ItemStats:
    log_pop: np.ndarray     # log1p(interaction count)
    avg_rating: np.ndarray  # shrunk towards the global mean
    year: np.ndarray        # release year
    ts_values: np.ndarray   # CSR layout: sorted interaction timestamps of all items...
    ts_offsets: np.ndarray  # ...item k owns ts_values[ts_offsets[k]:ts_offsets[k + 1]]

    @classmethod
    def from_interactions(cls, df: pd.DataFrame, n_items: int, years: np.ndarray,
                          prior: float = 10.0) -> "ItemStats":
        counts = np.bincount(df.i, minlength=n_items + 1).astype(np.float32)
        sums = np.bincount(df.i, weights=df.rating, minlength=n_items + 1).astype(np.float32)
        mu = float(df.rating.mean())
        avg = (sums + prior * mu) / (counts + prior)
        order = np.lexsort((df.ts.to_numpy(), df.i.to_numpy()))
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(np.log1p(counts), avg.astype(np.float32), years.astype(np.float32),
                   df.ts.to_numpy()[order].astype(np.int64), offsets)

    def window_counts(self, items: np.ndarray, end_ts: int, days: int) -> np.ndarray:
        """Interactions per item in (end_ts - days, end_ts]."""
        out = np.empty(len(items), dtype=np.float32)
        for k, i in enumerate(items):
            ts = self.ts_values[self.ts_offsets[i]:self.ts_offsets[i + 1]]
            out[k] = np.searchsorted(ts, end_ts, "right") - np.searchsorted(ts, end_ts - days * DAY, "right")
        return out

    def to_npz(self, path) -> None:
        np.savez(path, log_pop=self.log_pop, avg_rating=self.avg_rating, year=self.year,
                 ts_values=self.ts_values, ts_offsets=self.ts_offsets)

    @classmethod
    def from_npz(cls, path) -> "ItemStats":
        z = np.load(path)
        return cls(z["log_pop"], z["avg_rating"], z["year"], z["ts_values"], z["ts_offsets"])


def build_features(history: np.ndarray, request_ts: int, cand_items: np.ndarray,
                   cand_scores: np.ndarray, item_emb: np.ndarray, item_feats: np.ndarray,
                   stats: ItemStats, max_history: int) -> np.ndarray:
    """Feature matrix [n_candidates, len(FEATURES)] for one user."""
    history = np.asarray(history, dtype=np.int64)[-max_history:]
    n = len(cand_items)
    genres = item_feats[:, :-1]
    if len(history):
        profile = genres[history].mean(0)
        cand_g = genres[cand_items]
        denom = np.linalg.norm(cand_g, axis=1) * (np.linalg.norm(profile) + 1e-8) + 1e-8
        affinity = cand_g @ profile / denom
        sims = item_emb[cand_items] @ item_emb[history[-5:]].T
        max_sim, mean_sim = sims.max(1), sims.mean(1)
    else:
        affinity = max_sim = mean_sim = np.zeros(n, dtype=np.float32)
    request_year = 1970 + request_ts / (365.25 * DAY)
    return np.column_stack([
        cand_scores,
        np.arange(n, dtype=np.float32),
        stats.log_pop[cand_items],
        stats.avg_rating[cand_items],
        np.log1p(stats.window_counts(cand_items, request_ts, 30)),
        np.log1p(stats.window_counts(cand_items, request_ts, 180)),
        request_year - stats.year[cand_items],
        affinity,
        np.full(n, np.log1p(len(history)), dtype=np.float32),
        max_sim,
        mean_sim,
    ]).astype(np.float32)
