"""Online inference: encode user -> ANN retrieval -> LambdaRank re-rank -> optional MMR diversification."""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .data import pad_history
from .features import ItemStats, build_features
from .index import load_index, search
from .two_tower import TwoTower

def resolve_version(artifact_root: str | Path) -> Path:
    root = Path(artifact_root)
    return root / (root / "LATEST").read_text().strip()


def mmr(item_ids: np.ndarray, scores: np.ndarray, emb: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: trade relevance against similarity to already-picked items."""
    if lam <= 0 or len(item_ids) <= k:
        return list(range(min(k, len(item_ids))))
    rel = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
    vecs = emb[item_ids]
    picked, max_sim = [], np.full(len(item_ids), -np.inf)
    for _ in range(k):
        mmr_score = (1 - lam) * rel - lam * np.where(np.isfinite(max_sim), max_sim, 0.0)
        mmr_score[picked] = -np.inf
        best = int(mmr_score.argmax())
        picked.append(best)
        max_sim = np.maximum(max_sim, vecs @ vecs[best])
    return picked


class Recommender:
    def __init__(self, artifact_root: str | Path):
        self.dir = resolve_version(artifact_root)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.version = self.manifest["version"]
        cfg = self.manifest["config"]
        self.history_len = cfg["tower"]["history_len"]
        self.max_history = cfg["max_history_feature"]
        self.n_candidates = cfg["ranker"]["n_candidates"]

        maps = json.loads((self.dir / "mappings.json").read_text())
        self.user2idx = {int(k): v for k, v in maps["user2idx"].items()}
        self.item2idx = {int(k): v for k, v in maps["item2idx"].items()}
        movies = pd.read_csv(self.dir / "movies.csv").sort_values("item_idx")
        self.idx2movie = np.zeros(len(movies) + 1, dtype=np.int64)
        self.idx2movie[movies.item_idx] = movies.movieId
        self.meta = {int(r.movieId): {"title": r.title, "genres": r.genres.split("|")}
                     for r in movies.itertuples()}

        self.item_feats = np.load(self.dir / "item_feats.npy")
        self.item_emb = np.load(self.dir / "item_emb.npy")
        ckpt = torch.load(self.dir / "two_tower.pt", weights_only=True)
        self.model = TwoTower(ckpt["n_users"], ckpt["n_items"], self.item_feats, ckpt["dim"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.index = load_index(str(self.dir / "items.faiss"))
        self.ranker = lgb.Booster(model_file=str(self.dir / "ranker.txt"))
        self.stats = ItemStats.from_npz(self.dir / "item_stats.npz")
        self.snapshot_ts = int(self.stats.ts_values.max())
        self.popular_idx = np.argsort(-self.stats.log_pop)
        self.popular_idx = self.popular_idx[self.popular_idx > 0]

    def seed_histories(self) -> dict[str, list]:
        return json.loads((self.dir / "histories.json").read_text())

    def _item(self, idx: int, score: float, **extra) -> dict:
        movie_id = int(self.idx2movie[idx])
        return {"movie_id": movie_id, "score": round(float(score), 5), **self.meta[movie_id], **extra}

    def recommend(self, history: list[tuple[int, int]], user_id: int | None = None, k: int = 10,
                  diversity: float = 0.0, explain: bool = False) -> dict:
        timings = {}
        hist = np.array([self.item2idx[m] for m, _ in history if m in self.item2idx], dtype=np.int64)
        user_idx = self.user2idx.get(user_id, 0) if user_id is not None else 0
        if len(hist) == 0 and user_idx == 0:
            return {"strategy": "popular", "items": self.popular(k), "timings_ms": timings}

        t = time.perf_counter()
        with torch.no_grad():
            u = self.model.encode_users(torch.tensor([user_idx]),
                                        torch.as_tensor(pad_history(hist, self.history_len))[None]).numpy()
        timings["encode_user"] = time.perf_counter() - t

        t = time.perf_counter()
        cands, scores = search(self.index, u, self.n_candidates, [set(hist.tolist())])
        cands, scores = cands[0], scores[0]
        timings["ann_search"] = time.perf_counter() - t

        t = time.perf_counter()
        # Features are computed as of the latest data snapshot the stats cover.
        request_ts = min(max((ts for _, ts in history), default=self.snapshot_ts), self.snapshot_ts)
        feats = build_features(hist, request_ts, cands, scores, self.item_emb, self.item_feats,
                               self.stats, self.max_history)
        rank_scores = self.ranker.predict(feats)
        order = np.argsort(-rank_scores, kind="stable")
        cands, rank_scores = cands[order], rank_scores[order]
        timings["rerank"] = time.perf_counter() - t

        t = time.perf_counter()
        picked = mmr(cands, rank_scores, self.item_emb, k, diversity)
        timings["diversify"] = time.perf_counter() - t

        items = []
        recent = hist[-20:]
        for p in picked:
            extra = {}
            if explain and len(recent):
                anchor = int(recent[np.argmax(self.item_emb[recent] @ self.item_emb[cands[p]])])
                extra["because_you_watched"] = self.meta[int(self.idx2movie[anchor])]["title"]
            items.append(self._item(int(cands[p]), rank_scores[p], **extra))
        strategy = "two_tower+lambdarank" + ("+mmr" if diversity > 0 else "")
        return {"strategy": strategy, "items": items,
                "timings_ms": {k: round(v * 1000, 3) for k, v in timings.items()}}

    def popular(self, k: int, exclude: set[int] = frozenset()) -> list[dict]:
        out = []
        for idx in self.popular_idx:
            if int(self.idx2movie[idx]) not in exclude:
                out.append(self._item(int(idx), self.stats.log_pop[idx]))
            if len(out) == k:
                break
        return out

    def similar(self, movie_id: int, k: int = 10) -> list[dict]:
        idx = self.item2idx[movie_id]
        ids, scores = search(self.index, self.item_emb[idx][None], k, [{idx}])
        return [self._item(int(i), s) for i, s in zip(ids[0], scores[0])]

    def describe(self, movie_id: int) -> dict | None:
        return self.meta.get(movie_id)
