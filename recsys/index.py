"""FAISS ANN index over item embeddings (inner product == cosine, embeddings are L2-normalized)."""
import faiss
import numpy as np

from .config import IndexConfig


def build_index(item_emb: np.ndarray, cfg: IndexConfig) -> faiss.Index:
    vectors = np.ascontiguousarray(item_emb[1:])  # row 0 is padding; faiss id k == item_idx k + 1
    dim = vectors.shape[1]
    if cfg.kind == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.ef_construction
        index.hnsw.efSearch = cfg.ef_search
    index.add(vectors)
    return index


def search(index: faiss.Index, queries: np.ndarray, k: int, exclude: list[set] | None = None):
    """Top-k item_idx per query, skipping excluded (already-seen) items."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    max_excl = max((len(e) for e in exclude), default=0) if exclude else 0
    fetch = min(k + max_excl, index.ntotal)
    scores, ids = index.search(queries, fetch)
    out_ids, out_scores = [], []
    for row in range(len(queries)):
        excl = exclude[row] if exclude else set()
        keep_i, keep_s = [], []
        for s, i in zip(scores[row], ids[row]):
            item = int(i) + 1
            if i < 0 or item in excl:
                continue
            keep_i.append(item)
            keep_s.append(float(s))
            if len(keep_i) == k:
                break
        out_ids.append(np.array(keep_i, dtype=np.int64))
        out_scores.append(np.array(keep_s, dtype=np.float32))
    return out_ids, out_scores


def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str, ef_search: int | None = None) -> faiss.Index:
    index = faiss.read_index(path)
    if ef_search is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search
    return index
