"""Offline pipeline: train two-tower -> build ANN index -> train re-ranker -> evaluate -> version artifacts."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import PipelineConfig
from .data import build_training_examples, load_dataset, pad_history, user_histories, user_last_ts
from .features import ItemStats, build_features
from .index import build_index, save_index, search
from .metrics import summarize
from .ranker import feature_importance, train_ranker
from .two_tower import all_item_embeddings, train_two_tower, user_embeddings


def _user_matrix(model, hist_map, users, history_len):
    hists = np.stack([pad_history(hist_map[u], history_len) for u in users])
    return user_embeddings(model, np.asarray(users, dtype=np.int64), hists)


def exact_topk(model, n_items, hist_map, users, k, history_len):
    """Brute-force retrieval (used for model selection and as the ANN ground truth)."""
    scores = _user_matrix(model, hist_map, users, history_len) @ all_item_embeddings(model, n_items).T
    scores[:, 0] = -np.inf
    for row, u in enumerate(users):
        scores[row, hist_map[u]] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, 1).argsort(1)[:, ::-1]
    return {u: np.take_along_axis(top, order, 1)[row].tolist() for row, u in enumerate(users)}


def retrieve(model, index, hist_map, users, k, history_len):
    U = _user_matrix(model, hist_map, users, history_len)
    return search(index, U, k, [set(hist_map[u].tolist()) for u in users])


def ranking_rows(users, cands, scores, hist_map, ts_map, truth, item_emb, item_feats, stats, max_history):
    X, y, groups = [], [], []
    for u, items, sc in zip(users, cands, scores):
        labels = np.isin(items, list(truth.get(u, ()))).astype(np.float32)
        if labels.sum() == 0:
            continue  # no learning signal for this query
        X.append(build_features(hist_map[u], ts_map[u], items, sc, item_emb, item_feats, stats, max_history))
        y.append(labels)
        groups.append(len(items))
    return np.vstack(X), np.concatenate(y), groups


def run_pipeline(cfg: PipelineConfig, log=print) -> Path:
    t_start = time.time()
    ds = load_dataset(cfg.data_dir, cfg.val_frac, cfg.test_frac)
    tc, L, N = cfg.tower, cfg.tower.history_len, cfg.ranker.n_candidates
    log(f"users={ds.n_users} items={ds.n_items} train={len(ds.train)} val={len(ds.val)} test={len(ds.test)}")

    train_hist = user_histories(ds.train)
    trainval = pd.concat([ds.train, ds.val], ignore_index=True)
    trainval_hist = user_histories(trainval)
    val_truth = {int(u): set(g.i.tolist()) for u, g in ds.val.groupby("u")}
    test_truth = {int(u): set(g.i.tolist()) for u, g in ds.test.groupby("u")}
    val_users = sorted(u for u in val_truth if u in train_hist)
    test_users = sorted(u for u in test_truth if u in trainval_hist)

    # ---- Stage 1: two-tower retrieval model ----
    examples = build_training_examples(ds.train, L)
    item_counts = np.bincount(ds.train.i, minlength=ds.n_items + 1).astype(np.float64)

    def eval_fn(model):
        recs = exact_topk(model, ds.n_items, train_hist, val_users, 100, L)
        return round(summarize(recs, val_truth, ks=(100,))["recall@100"], 4)

    t0 = time.time()
    model, curve = train_two_tower(ds.n_users, ds.n_items, ds.item_feats, examples, item_counts, tc, eval_fn, log)
    tower_secs = time.time() - t0
    item_emb = all_item_embeddings(model, ds.n_items)
    index = build_index(item_emb, cfg.index)

    # ---- Stage 2: LambdaRank re-ranker, trained on the val window (unseen by the tower) ----
    stats_train = ItemStats.from_interactions(ds.train, ds.n_items, ds.item_years)
    cands, scores = retrieve(model, index, train_hist, val_users, N, L)
    train_ts = user_last_ts(ds.train)
    # 20% of queries are held out for early stopping.
    is_es = np.random.default_rng(tc.seed).random(len(val_users)) < 0.2

    def rows(mask):
        pick = np.flatnonzero(mask)
        return ranking_rows([val_users[k] for k in pick], [cands[k] for k in pick], [scores[k] for k in pick],
                            train_hist, train_ts, val_truth, item_emb, ds.item_feats, stats_train,
                            cfg.max_history_feature)

    fit_rows, es_rows = rows(~is_es), rows(is_es)
    log(f"ranker rows={len(fit_rows[1])} queries={len(fit_rows[2])} positives={int(fit_rows[1].sum())}")
    booster = train_ranker(*fit_rows, cfg.ranker, valid=es_rows)

    # ---- Offline evaluation on the held-out test window ----
    stats_tv = ItemStats.from_interactions(trainval, ds.n_items, ds.item_years)
    trainval_ts = user_last_ts(trainval)
    t0 = time.time()
    cands, scores = retrieve(model, index, trainval_hist, test_users, N, L)
    retrieval_ms = (time.time() - t0) * 1000 / len(test_users)
    tt_rank, rerank = {}, {}
    t0 = time.time()
    for u, items, sc in zip(test_users, cands, scores):
        feats = build_features(trainval_hist[u], trainval_ts[u], items, sc, item_emb, ds.item_feats,
                               stats_tv, cfg.max_history_feature)
        tt_rank[u] = items.tolist()
        rerank[u] = items[np.argsort(-booster.predict(feats), kind="stable")].tolist()
    rank_ms = (time.time() - t0) * 1000 / len(test_users)

    pop_order = np.argsort(-np.bincount(trainval.i, minlength=ds.n_items + 1))
    pop_order = pop_order[pop_order > 0]
    popular = {}
    for u in test_users:
        seen = set(trainval_hist[u].tolist())
        popular[u] = [int(i) for i in pop_order[: N + len(seen)] if i not in seen][:N]

    exact = exact_topk(model, ds.n_items, trainval_hist, test_users, N, L)
    ann_recall = float(np.mean([len(set(tt_rank[u]) & set(exact[u])) / N for u in test_users]))
    ks = (10, 50, N)
    metrics = {
        "popularity_baseline": summarize(popular, test_truth, ks),
        "two_tower": summarize(tt_rank, test_truth, ks),
        "two_tower_plus_ranker": summarize(rerank, test_truth, ks),
        "ann_recall_vs_exact@%d" % N: round(ann_recall, 4),
        "offline_latency_ms_per_user": {"retrieval": round(retrieval_ms, 3), "rerank": round(rank_ms, 3)},
    }
    for name in ("popularity_baseline", "two_tower", "two_tower_plus_ranker"):
        log(f"{name:24s} " + " ".join(f"{k}={v:.4f}" for k, v in metrics[name].items() if k != "users"))
    log(f"HNSW recall vs exact search@{N}: {ann_recall:.4f}")

    # ---- Version and persist artifacts ----
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    out = Path(cfg.artifact_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_users": ds.n_users, "n_items": ds.n_items,
                "dim": tc.dim}, out / "two_tower.pt")
    np.save(out / "item_emb.npy", item_emb)
    np.save(out / "item_feats.npy", ds.item_feats)
    save_index(index, str(out / "items.faiss"))
    booster.save_model(str(out / "ranker.txt"))
    all_df = pd.concat([trainval, ds.test], ignore_index=True)
    ItemStats.from_interactions(all_df, ds.n_items, ds.item_years).to_npz(out / "item_stats.npz")
    ds.movies[["movieId", "title", "genres", "item_idx"]].to_csv(out / "movies.csv", index=False)
    idx2item = {v: k for k, v in ds.item2idx.items()}
    idx2user = {v: k for k, v in ds.user2idx.items()}
    histories = {str(idx2user[int(u)]): [[idx2item[int(i)], int(t)] for i, t in zip(g.i, g.ts)]
                 for u, g in all_df.sort_values(["u", "ts", "i"], kind="stable").groupby("u")}
    (out / "histories.json").write_text(json.dumps(histories))
    (out / "mappings.json").write_text(json.dumps({"user2idx": ds.user2idx, "item2idx": ds.item2idx,
                                                    "genres": ds.genres}))
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"users": ds.n_users, "items": ds.n_items, "interactions": len(all_df)},
        "config": cfg.to_dict(),
        "training_curve": curve,
        "tower_train_secs": round(tower_secs, 1),
        "ranker_best_iteration": booster.best_iteration,
        "ranker_feature_importance": feature_importance(booster),
        "metrics": metrics,
        "pipeline_secs": round(time.time() - t_start, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (Path(cfg.artifact_dir) / "LATEST").write_text(version)
    log(f"artifacts -> {out}")
    return out


