"""Second-stage LightGBM LambdaRank re-ranker."""
import lightgbm as lgb
import numpy as np

from .config import RankerConfig
from .features import FEATURES


def train_ranker(X: np.ndarray, y: np.ndarray, groups: list[int], cfg: RankerConfig,
                 valid: tuple | None = None) -> lgb.Booster:
    """LambdaRank; `valid=(X, y, groups)` enables early stopping on held-out queries."""
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(X, y, group=groups, feature_name=FEATURES)
    valid_sets, callbacks = [], []
    if valid is not None:
        valid_sets = [lgb.Dataset(valid[0], valid[1], group=valid[2], reference=train_set)]
        callbacks = [lgb.early_stopping(30, verbose=False)]
    return lgb.train(params, train_set, num_boost_round=cfg.n_estimators,
                     valid_sets=valid_sets, callbacks=callbacks)


def feature_importance(booster: lgb.Booster) -> dict[str, float]:
    gain = booster.feature_importance(importance_type="gain")
    total = gain.sum() or 1.0
    return {f: round(float(g / total), 4) for f, g in sorted(zip(FEATURES, gain), key=lambda t: -t[1])}
