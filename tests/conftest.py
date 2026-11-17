import recsys  # noqa: F401  (OpenMP env vars before torch/faiss)

import numpy as np
import pandas as pd
import pytest

from recsys.config import PipelineConfig
from recsys.pipeline import run_pipeline

GENRES = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]


def write_synthetic_movielens(path, n_users=80, n_movies=150, seed=0):
    """Small MovieLens-format dataset with genre-clustered tastes, so there is signal to learn."""
    rng = np.random.default_rng(seed)
    movie_genre = rng.integers(0, len(GENRES), n_movies)
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_movies + 1) * 10,
        "title": [f"Movie {i} ({1980 + i % 40})" for i in range(n_movies)],
        "genres": [GENRES[g] for g in movie_genre],
    })
    rows = []
    for u in range(1, n_users + 1):
        fav = rng.integers(0, len(GENRES))
        p = np.where(movie_genre == fav, 8.0, 1.0)
        picks = rng.choice(n_movies, size=int(rng.integers(25, 45)), replace=False, p=p / p.sum())
        t0 = 1_000_000_000 + int(rng.integers(0, 10**7))
        for k, m in enumerate(picks):
            rows.append((u, int(movies.movieId[m]), float(rng.choice([3.0, 4.0, 5.0])), t0 + k * 3600))
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(path / "ratings.csv", index=False)
    movies.to_csv(path / "movies.csv", index=False)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("ml")
    write_synthetic_movielens(path)
    return path


@pytest.fixture(scope="session")
def artifact_root(data_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    cfg = PipelineConfig(data_dir=str(data_dir), artifact_dir=str(root))
    cfg.tower.epochs = 3
    cfg.tower.batch_size = 256
    cfg.ranker.n_candidates = 50
    cfg.ranker.n_estimators = 30
    cfg.ranker.min_child_samples = 5
    run_pipeline(cfg, log=lambda *_: None)
    return root
