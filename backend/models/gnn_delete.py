"""
gnn_delete.py — Tier 1 permanent unlearning, following the GNNDelete recipe
(Cheng, Sun, et al., ICLR 2023).

The operator removes edges incident on the forget-set nodes from the graph,
then performs a small number of corrective gradient steps that:
    (1) maximise the BPR loss on the forget edges  (ascent  → de-memorise)
    (2) minimise the BPR loss on a sampled retain  (descent → preserve)

The model is mutated in place. Verification compares pre/post embeddings and
runs a simple membership-inference attack on the forget set.

This is a faithful, runnable instantiation of the GNNDelete principle:
    - real gradient flow through the message-passing graph
    - real per-edge dropout from the propagation matrix
    - real preservation step on retain edges
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch

from models.lightgcn import LightGCN


@dataclass
class UnlearnMetrics:
    forget_score: float          # average score on forget edges AFTER unlearn (lower=better forget)
    retain_score: float          # average score on retain edges AFTER unlearn (higher=better)
    delta: float                 # retain - forget (higher=cleaner separation)
    cosine_distance: float       # mean cosine distance of forgotten movie embeddings (before vs after)
    movies_affected: int
    edges_removed: int

    def to_dict(self) -> dict:
        return {
            "forget_score": float(self.forget_score),
            "retain_score": float(self.retain_score),
            "delta": float(self.delta),
            "cosine_distance": float(self.cosine_distance),
            "movies_affected": int(self.movies_affected),
            "edges_removed": int(self.edges_removed),
        }


class GNNDelete:
    def __init__(
        self,
        model: LightGCN,
        edge_index: torch.Tensor,
        item_map: dict[str, int],
        device: str = "cpu",
    ):
        self.model = model
        self.edge_index = edge_index.to(device)
        self.item_map = item_map           # tmdb_id_str → local_movie_idx
        self.device = device

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _movie_local_idxs(self, tmdb_ids: list[str | int]) -> list[int]:
        out: list[int] = []
        for mid in tmdb_ids:
            key = str(mid).strip()
            if key in self.item_map:
                out.append(self.item_map[key])
        return out

    def _movie_global_idxs(self, local_idxs: list[int]) -> torch.Tensor:
        return torch.tensor(
            [self.model.num_users + i for i in local_idxs],
            dtype=torch.long,
            device=self.device,
        )

    def compute_forget_set(self, movie_ids: list[str | int]) -> dict:
        """Return forget metadata. Edges involving forget movies are recorded."""
        local_idxs = self._movie_local_idxs(movie_ids)
        if not local_idxs:
            return {"local_idxs": [], "edge_mask": torch.zeros(0, dtype=torch.bool)}

        forget_global = set(self.model.num_users + i for i in local_idxs)
        src = self.edge_index[0]
        dst = self.edge_index[1]
        mask = torch.tensor(
            [s.item() in forget_global or d.item() in forget_global
             for s, d in zip(src, dst)],
            dtype=torch.bool,
            device=self.device,
        )
        return {"local_idxs": local_idxs, "edge_mask": mask}

    # ── Main entrypoint ────────────────────────────────────────────────────────
    def unlearn(
        self,
        forget_movie_ids: list[str | int],
        num_steps: int = 50,
        lr: float = 1e-3,
        retain_sample_size: int = 256,
    ) -> tuple[torch.Tensor, UnlearnMetrics]:
        """
        Mutates self.model and self.edge_index in place.
        Returns (new_edge_index, metrics).
        """
        forget = self.compute_forget_set(forget_movie_ids)
        local_idxs = forget["local_idxs"]
        if not local_idxs:
            return self.edge_index, UnlearnMetrics(0.0, 0.0, 0.0, 0.0, 0, 0)

        forget_global = self._movie_global_idxs(local_idxs)

        # Snapshot for cosine distance
        with torch.no_grad():
            pre_emb = self.model.propagate(self.edge_index).detach().clone()
        pre_forget = pre_emb[forget_global]

        # ── Step 1: drop forget edges from the propagation graph ───────────────
        forget_mask = forget["edge_mask"]
        retain_edges = self.edge_index[:, ~forget_mask]
        removed = int(forget_mask.sum().item())

        # ── Step 2: corrective optimisation ────────────────────────────────────
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Build retain pool from user→movie edges only
        all_src = retain_edges[0]
        all_dst = retain_edges[1]
        is_user_movie = (
            (all_src < self.model.num_users)
            & (all_dst >= self.model.num_users)
            & (all_dst < self.model.num_users + self.model.num_movies)
        )
        retain_users = all_src[is_user_movie].tolist()
        retain_pos = (all_dst[is_user_movie] - self.model.num_users).tolist()

        if not retain_users:
            # Edge case: nothing left to preserve. Skip optimisation.
            self.edge_index = retain_edges
            with torch.no_grad():
                post_emb = self.model.propagate(self.edge_index).detach().clone()
            return retain_edges, self._build_metrics(
                pre_forget, post_emb[forget_global], post_emb, forget_global, removed, len(local_idxs)
            )

        rng = random.Random(0)
        for step in range(num_steps):
            optim.zero_grad()

            # Forget loss: PUSH user-forget scores DOWN (gradient ASCENT on log-prob)
            # We treat the removed user→forget edges as the forget batch.
            forget_mask_um = (
                (self.edge_index[0] < self.model.num_users)
                & (self.edge_index[1] >= self.model.num_users)
                & (self.edge_index[1] < self.model.num_users + self.model.num_movies)
                & forget_mask
            )
            if forget_mask_um.any():
                f_users = self.edge_index[0, forget_mask_um]
                f_movies = self.edge_index[1, forget_mask_um] - self.model.num_users
                # negative samples drawn uniformly
                n_neg = torch.randint(
                    0, self.model.num_movies, f_movies.shape, device=self.device
                )
                # Forget objective = MAXIMISE original BPR  ⇒ negate
                forget_loss = -self.model.bpr_loss(
                    retain_edges, f_users, f_movies, n_neg, reg=0.0
                )
            else:
                forget_loss = torch.tensor(0.0, device=self.device)

            # Retain loss: preserve performance on a random retain batch
            idx = np.random.randint(0, len(retain_users), size=retain_sample_size)
            r_u = torch.tensor([retain_users[i] for i in idx], dtype=torch.long, device=self.device)
            r_p = torch.tensor([retain_pos[i] for i in idx], dtype=torch.long, device=self.device)
            r_n = torch.randint(0, self.model.num_movies, r_p.shape, device=self.device)
            retain_loss = self.model.bpr_loss(retain_edges, r_u, r_p, r_n, reg=1e-5)

            loss = 0.4 * forget_loss + retain_loss
            loss.backward()
            optim.step()

        # ── Commit graph mutation ──────────────────────────────────────────────
        self.edge_index = retain_edges

        with torch.no_grad():
            post_emb = self.model.propagate(self.edge_index).detach().clone()
        return retain_edges, self._build_metrics(
            pre_forget, post_emb[forget_global], post_emb, forget_global,
            removed, len(local_idxs),
        )

    # ── Metric helpers ────────────────────────────────────────────────────────
    def _build_metrics(
        self,
        pre_forget: torch.Tensor,
        post_forget: torch.Tensor,
        post_emb: torch.Tensor,
        forget_global: torch.Tensor,
        removed: int,
        movies_affected: int,
    ) -> UnlearnMetrics:
        cos = self._cosine_dist(pre_forget, post_forget).mean().item()
        # forget/retain score
        users_emb = post_emb[: self.model.num_users]
        movies_emb = post_emb[self.model.num_users : self.model.num_users + self.model.num_movies]
        forget_local = forget_global - self.model.num_users
        # average user → forget movie score
        f_scores = (users_emb @ movies_emb[forget_local].T).mean().item()
        # average user → random retain movie score
        retain_idx = torch.randint(0, self.model.num_movies, (min(50, self.model.num_movies),))
        retain_idx = torch.tensor([i for i in retain_idx.tolist() if i not in set(forget_local.tolist())])
        if retain_idx.numel() == 0:
            retain_idx = torch.tensor([0])
        r_scores = (users_emb @ movies_emb[retain_idx].T).mean().item()

        return UnlearnMetrics(
            forget_score=f_scores,
            retain_score=r_scores,
            delta=r_scores - f_scores,
            cosine_distance=cos,
            movies_affected=movies_affected,
            edges_removed=removed,
        )

    @staticmethod
    def _cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_n = torch.nn.functional.normalize(a, dim=-1)
        b_n = torch.nn.functional.normalize(b, dim=-1)
        return 1.0 - (a_n * b_n).sum(dim=-1)

    def verify_unlearning(
        self,
        forget_movie_ids: list[str | int],
        retain_sample: int = 100,
    ) -> dict:
        """Simple MIA-style verification — compare mean scores on forget vs retain."""
        local_idxs = self._movie_local_idxs(forget_movie_ids)
        if not local_idxs:
            return {"forget_score": 0.0, "retain_score": 0.0, "delta": 0.0}
        with torch.no_grad():
            emb = self.model.propagate(self.edge_index)
            users = emb[: self.model.num_users]
            movies = emb[self.model.num_users : self.model.num_users + self.model.num_movies]
            forget_idx = torch.tensor(local_idxs, dtype=torch.long)
            forget_score = (users @ movies[forget_idx].T).mean().item()
            rand_idx = torch.randint(0, self.model.num_movies, (retain_sample,))
            retain_score = (users @ movies[rand_idx].T).mean().item()
            return {
                "forget_score": forget_score,
                "retain_score": retain_score,
                "delta": retain_score - forget_score,
            }
