"""Two-tower retrieval model trained with in-batch sampled softmax + logQ correction."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TowerConfig


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, item_feats: np.ndarray, dim: int):
        super().__init__()
        n_feats = item_feats.shape[1]
        self.register_buffer("item_feats", torch.as_tensor(item_feats, dtype=torch.float32))
        # Item id embeddings are shared: they encode the target item and pool the user's history.
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.user_emb = nn.Embedding(n_users + 1, dim)  # row 0 = unknown / cold-start user
        self.item_feat_proj = nn.Linear(n_feats, dim)
        self.item_mlp = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.user_mlp = nn.Sequential(
            nn.Linear(3 * dim + n_feats, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim)
        )
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_emb(items), self.item_feat_proj(self.item_feats[items])], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def encode_users(self, users: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        mask = (history > 0).float().unsqueeze(-1)
        count = mask.sum(1).clamp(min=1.0)
        hist_mean = (self.item_emb(history) * mask).sum(1) / count
        # Recency signal: the last 5 real interactions (history is left-aligned, newest last).
        lengths = (history > 0).sum(1)
        pos = torch.arange(history.size(1), device=history.device).unsqueeze(0)
        recent_mask = ((pos >= (lengths - 5).unsqueeze(1)) & (pos < lengths.unsqueeze(1))).float().unsqueeze(-1)
        recent_mean = (self.item_emb(history) * recent_mask).sum(1) / recent_mask.sum(1).clamp(min=1.0)
        genre_profile = (self.item_feats[history] * mask).sum(1) / count
        x = torch.cat([self.user_emb(users), hist_mean, recent_mean, genre_profile], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)


def in_batch_softmax_loss(u: torch.Tensor, v: torch.Tensor, targets: torch.Tensor,
                          log_q: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sampled softmax over in-batch negatives.

    - logQ correction removes the popularity bias of in-batch sampling (Yi et al., RecSys'19).
    - Duplicate targets in the batch are masked so an item is never its own negative.
    """
    logits = (u @ v.T) / temperature - log_q[targets].unsqueeze(0)
    dup = targets.unsqueeze(0) == targets.unsqueeze(1)
    dup.fill_diagonal_(False)
    logits = logits.masked_fill(dup, float("-inf"))
    labels = torch.arange(len(targets), device=u.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def all_item_embeddings(model: TwoTower, n_items: int) -> np.ndarray:
    model.eval()
    emb = model.encode_items(torch.arange(n_items + 1)).numpy()
    emb[0] = 0.0
    return emb.astype(np.float32)


@torch.no_grad()
def user_embeddings(model: TwoTower, users: np.ndarray, histories: np.ndarray) -> np.ndarray:
    model.eval()
    return model.encode_users(torch.as_tensor(users), torch.as_tensor(histories)).numpy()


def train_two_tower(n_users, n_items, item_feats, examples, item_counts, cfg: TowerConfig,
                    eval_fn=None, log=print) -> tuple[TwoTower, list[dict]]:
    """Train; `eval_fn(model) -> float` (validation recall) selects the best epoch."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    users, hists, targets = (torch.as_tensor(a) for a in examples)
    q = (item_counts + 1.0) / (item_counts + 1.0).sum()
    log_q = torch.as_tensor(np.log(q), dtype=torch.float32)

    model = TwoTower(n_users, n_items, item_feats, cfg.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, best_score, best_state = [], -1.0, None
    n = len(targets)
    for epoch in range(cfg.epochs):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.as_tensor(rng.permutation(n))
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if len(idx) < 2:
                continue
            u = users[idx].clone()
            u[torch.rand(len(u)) < cfg.user_id_dropout] = 0
            loss = in_batch_softmax_loss(
                model.encode_users(u, hists[idx]), model.encode_items(targets[idx]),
                targets[idx], log_q, cfg.temperature,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        record = {"epoch": epoch + 1, "loss": total / n, "secs": round(time.time() - t0, 1)}
        if eval_fn is not None:
            record["val_recall@100"] = eval_fn(model)
            if record["val_recall@100"] > best_score:
                best_score = record["val_recall@100"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        log(record)
        history.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
