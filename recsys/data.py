"""Loading MovieLens, building id maps / item features, and leakage-free temporal splits."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


@dataclass
class Dataset:
    movies: pd.DataFrame          # movieId, title, genres, item_idx
    item2idx: dict[int, int]      # movieId -> idx (0 is padding)
    user2idx: dict[int, int]      # userId -> idx (0 is "unknown user")
    genres: list[str]
    item_feats: np.ndarray        # [n_items + 1, n_genres + 1] multi-hot genres + normalized year
    item_years: np.ndarray        # [n_items + 1] release year
    train: pd.DataFrame           # u, i, rating, ts  (index space)
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def n_items(self) -> int:
        return len(self.item2idx)

    @property
    def n_users(self) -> int:
        return len(self.user2idx)


def _parse_year(title: str) -> float:
    m = YEAR_RE.search(title.strip())
    return float(m.group(1)) if m else np.nan


def build_item_features(movies: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    genres = sorted({g for gs in movies.genres for g in gs.split("|") if g != "(no genres listed)"})
    g2i = {g: k for k, g in enumerate(genres)}
    feats = np.zeros((len(movies) + 1, len(genres) + 1), dtype=np.float32)
    years = movies.title.map(_parse_year)
    years = years.fillna(years.median())
    item_years = np.zeros(len(movies) + 1, dtype=np.float32)
    item_years[movies.item_idx.to_numpy()] = years.to_numpy()
    item_years[0] = years.median()
    year_norm = ((years - years.min()) / max(years.max() - years.min(), 1.0)).to_numpy()
    for row, gs, y in zip(movies.item_idx, movies.genres, year_norm):
        for g in gs.split("|"):
            if g in g2i:
                feats[row, g2i[g]] = 1.0
        feats[row, -1] = y
    return feats, item_years, genres


def temporal_split(df: pd.DataFrame, val_frac: float, test_frac: float):
    """Per-user chronological split: each user's oldest interactions train, newest test.

    Nothing from a user's future can leak into features for the past.
    """
    df = df.sort_values(["u", "ts", "i"], kind="stable")
    rank = df.groupby("u").cumcount()
    size = df.groupby("u")["i"].transform("size")
    n_test = np.maximum(1, np.floor(size * test_frac)).astype(int)
    n_val = np.maximum(1, np.floor(size * val_frac)).astype(int)
    test_start = size - n_test
    val_start = test_start - n_val
    train = df[rank < val_start]
    val = df[(rank >= val_start) & (rank < test_start)]
    test = df[rank >= test_start]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_dataset(data_dir: str | Path, val_frac: float = 0.1, test_frac: float = 0.1) -> Dataset:
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    movies = movies[movies.movieId.isin(ratings.movieId.unique())].reset_index(drop=True)

    item2idx = {int(m): k + 1 for k, m in enumerate(movies.movieId)}
    user2idx = {int(u): k + 1 for k, u in enumerate(sorted(ratings.userId.unique()))}
    movies["item_idx"] = movies.movieId.map(item2idx)
    item_feats, item_years, genres = build_item_features(movies)

    df = pd.DataFrame({
        "u": ratings.userId.map(user2idx).astype(np.int64),
        "i": ratings.movieId.map(item2idx).astype(np.int64),
        "rating": ratings.rating.astype(np.float32),
        "ts": ratings.timestamp.astype(np.int64),
    })
    train, val, test = temporal_split(df, val_frac, test_frac)
    return Dataset(movies, item2idx, user2idx, genres, item_feats, item_years, train, val, test)


def user_histories(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Chronological item sequence per user."""
    df = df.sort_values(["u", "ts", "i"], kind="stable")
    return {int(u): g["i"].to_numpy() for u, g in df.groupby("u")}


def user_last_ts(df: pd.DataFrame) -> dict[int, int]:
    return {int(u): int(t) for u, t in df.groupby("u")["ts"].max().items()}


def pad_history(items, length: int) -> np.ndarray:
    """Most recent `length` items, left-aligned, zero padded."""
    items = list(items)[-length:]
    out = np.zeros(length, dtype=np.int64)
    out[: len(items)] = items
    return out


def build_training_examples(train: pd.DataFrame, history_len: int):
    """One example per interaction: (user, items seen strictly before it, target item)."""
    users, hists, targets = [], [], []
    for u, seq in user_histories(train).items():
        for t in range(len(seq)):
            users.append(u)
            hists.append(pad_history(seq[:t], history_len))
            targets.append(seq[t])
    return np.array(users, dtype=np.int64), np.stack(hists), np.array(targets, dtype=np.int64)
