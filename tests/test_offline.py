import json

import numpy as np
import pytest
import torch

from recsys.config import IndexConfig
from recsys.data import build_training_examples, load_dataset
from recsys.index import build_index, search
from recsys.metrics import ndcg_at_k, recall_at_k
from recsys.service import mmr
from recsys.two_tower import in_batch_softmax_loss


def test_temporal_split_has_no_leakage(data_dir):
    ds = load_dataset(data_dir)
    for u in ds.train.u.unique():
        tr, va, te = (d[d.u == u].ts for d in (ds.train, ds.val, ds.test))
        assert tr.max() <= va.min() <= va.max() <= te.min()
    assert len(ds.train) + len(ds.val) + len(ds.test) == sum(1 for _ in open(data_dir / "ratings.csv")) - 1


def test_training_examples_only_see_the_past(data_dir):
    ds = load_dataset(data_dir)
    users, hists, targets = build_training_examples(ds.train, history_len=30)
    first = ds.train.sort_values("ts").groupby("u").head(1)
    seen_first = set(zip(first.u, first.i))
    for u, h, t in zip(users[:500], hists[:500], targets[:500]):
        assert t not in h[h > 0]  # target never leaks into its own history
        if (u, t) in seen_first:
            assert (h == 0).all()


def test_loss_masks_duplicate_targets():
    torch.manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    v = u.clone()
    targets = torch.tensor([3, 3, 5, 7])
    log_q = torch.zeros(10)
    loss = in_batch_softmax_loss(u, v, targets, log_q, temperature=0.1)
    assert torch.isfinite(loss)
    # With rows 0 and 1 sharing a target, each must not be penalised for scoring the other.
    unmasked_v = v.clone()
    loss_distinct = in_batch_softmax_loss(u, unmasked_v, torch.tensor([3, 4, 5, 7]), log_q, 0.1)
    assert loss <= loss_distinct + 1e-6


def test_ranking_metrics():
    assert recall_at_k([1, 2, 3, 4], {2, 9}, 2) == 0.5
    assert ndcg_at_k([1, 2], {1}, 2) == 1.0
    assert ndcg_at_k([2, 1], {1}, 2) == pytest.approx(1 / np.log2(3))
    assert recall_at_k([1], set(), 1) == 0.0


@pytest.mark.parametrize("kind", ["flat", "hnsw"])
def test_ann_search_excludes_seen_items(kind):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(101, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    emb[0] = 0
    index = build_index(emb, IndexConfig(kind=kind))
    ids, scores = search(index, emb[[5]], k=10, exclude=[{5, 6, 7}])
    assert len(ids[0]) == 10 and not {5, 6, 7} & set(ids[0].tolist())
    assert 0 not in ids[0] and np.all(np.diff(scores[0]) <= 1e-6)


def test_mmr_increases_diversity():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(2, 8))
    emb = np.vstack([np.zeros(8)] + [base[0] + 0.01 * rng.normal(size=8) for _ in range(10)]
                    + [base[1] + 0.01 * rng.normal(size=8) for _ in range(10)]).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    items = np.arange(1, 21)
    scores = np.linspace(1.0, 0.0, 20)  # every cluster-0 item outranks every cluster-1 item
    assert set(mmr(items, scores, emb, 4, 0.0)) <= set(range(10))
    assert set(mmr(items, scores, emb, 4, 0.7)) & set(range(10, 20))


def test_pipeline_beats_popularity_and_versions_artifacts(artifact_root):
    version = (artifact_root / "LATEST").read_text()
    manifest = json.loads((artifact_root / version / "manifest.json").read_text())
    m = manifest["metrics"]
    assert m["two_tower"]["recall@50"] > m["popularity_baseline"]["recall@50"]
    for f in ("two_tower.pt", "items.faiss", "ranker.txt", "item_stats.npz", "histories.json"):
        assert (artifact_root / version / f).exists()
