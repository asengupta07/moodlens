"""
influence.py — Tier 2 session unlearning via influence functions
(Koh & Liang, ICML 2017).

The session graph is treated as a small set of additional user→movie edges.
For committed sessions we record the pre-session state, then erase by applying
the exact inverse of that session perturbation: remove the temporary topology
and restore the pre-session embedding table. When no session snapshot exists,
we fall back to the first-order influence approximation.

erase_session    → subtract the influence (revert user embedding toward
                   pre-session state).
commit_session   → run a short fine-tune over the session edges, merging
                   them into the permanent embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.lightgcn import LightGCN


@dataclass
class SessionMetrics:
    cosine_distance: float
    user_embedding_before: list[float]
    user_embedding_after: list[float]
    edges_processed: int
    reversion_score: float | None = None
    non_destructive: bool = False

    def to_dict(self) -> dict:
        data = {
            "cosine_distance": float(self.cosine_distance),
            "before_vector": list(self.user_embedding_before),
            "after_vector": list(self.user_embedding_after),
            "edges_processed": int(self.edges_processed),
            "non_destructive": bool(self.non_destructive),
        }
        if self.reversion_score is not None:
            data["reversion_score"] = float(self.reversion_score)
        return data


class SessionUnlearner:
    def __init__(
        self,
        model: LightGCN,
        edge_index: torch.Tensor,
        item_map: dict[str, int],
        device: str = "cpu",
    ):
        self.model = model
        self.edge_index = edge_index.to(device)
        self.item_map = item_map
        self.device = device
        self._snapshots: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _resolve(self, session_edges: list[tuple]) -> list[tuple[int, int, float]]:
        """Normalize session edges to (user_idx, movie_local_idx, weight)."""
        out = []
        for entry in session_edges:
            if len(entry) == 3:
                u, m, w = entry
            else:
                u, m = entry
                w = 1.0
            key = str(m).strip()
            if key not in self.item_map:
                continue
            out.append((int(u), self.item_map[key], float(w)))
        return out

    def _session_key(
        self, session_edges: list[tuple], user_id: int,
    ) -> tuple[Any, ...]:
        """Stable key for matching a commit with its later discard/partial erase."""
        edges = self._resolve(session_edges)
        return (
            int(user_id),
            tuple(sorted((u, m, round(w, 6)) for u, m, w in edges)),
        )

    def _user_emb(self, user_id: int) -> torch.Tensor:
        with torch.no_grad():
            emb = self.model.propagate(self.edge_index)
            return emb[user_id].detach().clone()

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a_n = torch.nn.functional.normalize(a, dim=-1)
        b_n = torch.nn.functional.normalize(b, dim=-1)
        return 1.0 - float((a_n * b_n).sum().item())

    def _positive_edges(self, edges: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        """Commit only positive session evidence into the graph."""
        return [e for e in edges if e[2] > 0]

    def has_session_snapshot(self, session_edges: list[tuple], user_id: int) -> bool:
        return self._session_key(session_edges, user_id) in self._snapshots

    def preview_session_embedding(
        self,
        session_edges: list[tuple],
        user_id: int,
        strength: float = 0.35,
    ) -> torch.Tensor:
        """
        Return a temporary, non-mutating user embedding shaped by session edges.

        This is the live Tier 2 recommendation state: mood can steer ranking
        without touching the durable LightGCN parameters. Positive weights pull
        the user vector toward session movies; negative weights push away.
        """
        edges = self._resolve(session_edges)
        base = self._user_emb(self.model.user_idx(user_id))
        if not edges:
            return base

        with torch.no_grad():
            emb = self.model.propagate(self.edge_index)
            movie_ids = torch.tensor([e[1] for e in edges], dtype=torch.long, device=self.device)
            weights = torch.tensor([e[2] for e in edges], dtype=torch.float32, device=self.device)
            movie_vecs = emb[self.model.num_users + movie_ids]
            denom = weights.abs().sum().clamp_min(1e-6)
            target = (movie_vecs * weights.unsqueeze(1)).sum(dim=0) / denom
            return base + float(strength) * (target - base)

    def discard_active_session(
        self,
        session_edges: list[tuple],
        user_id: int,
    ) -> SessionMetrics:
        """
        Clear an active, uncommitted mood without mutating model parameters.

        The before vector is the session-conditioned preview used for live
        ranking; the after vector is the durable profile embedding.
        """
        before = self.preview_session_embedding(session_edges, user_id)
        after = self._user_emb(self.model.user_idx(user_id))
        return SessionMetrics(
            cosine_distance=self._cosine(before, after),
            user_embedding_before=before.tolist(),
            user_embedding_after=after.tolist(),
            edges_processed=len(self._resolve(session_edges)),
            reversion_score=1.0,
            non_destructive=True,
        )

    # ── Influence approximation ───────────────────────────────────────────────
    def compute_session_influence(
        self,
        session_edges: list[tuple],
        user_id: int,
        damping: float = 0.01,
    ) -> torch.Tensor:
        """
        Estimate the embedding shift introduced by `session_edges` on `user_id`.
        Returns the influence vector (same dim as a single embedding row).
        """
        edges = self._resolve(session_edges)
        if not edges:
            return torch.zeros(self.model.embedding_dim, device=self.device)

        key = self._session_key(session_edges, user_id)
        snapshot = self._snapshots.get(key)
        if snapshot is not None:
            row = self.model.user_idx(user_id)
            current = self.model.embedding.weight[row].detach()
            original = snapshot["embedding_weight"][row].to(self.device)
            return current - original

        users = torch.tensor([e[0] for e in edges], dtype=torch.long, device=self.device)
        pos = torch.tensor([e[1] for e in edges], dtype=torch.long, device=self.device)
        # weights treated as importance multipliers via repetition in negatives sampling
        neg = torch.randint(0, self.model.num_movies, pos.shape, device=self.device)

        user_idx = self.model.user_idx(user_id)
        param = self.model.embedding.weight
        # Compute gradient of session loss w.r.t. THIS user's embedding row
        loss = self.model.bpr_loss(self.edge_index, users, pos, neg, reg=0.0)
        grad = torch.autograd.grad(loss, param, retain_graph=False, create_graph=False)[0]
        v = grad[user_idx].detach()

        # First-order Hessian inverse approximation:  H^{-1} v ≈ (1/(1+damping)) * v
        return v / (1.0 + damping)

    # ── Public ops ─────────────────────────────────────────────────────────────
    def erase_session(
        self,
        session_edges: list[tuple],
        user_id: int,
        mode: str = "discard",
        step: float = 1.0,
    ) -> SessionMetrics:
        """
        mode='discard' — fully revert (step=1.0)
        mode='partial' — soft decay (step=0.7)
        """
        scale = 1.0 if mode == "discard" else 0.7 if mode == "partial" else float(step)
        before = self._user_emb(self.model.user_idx(user_id))

        key = self._session_key(session_edges, user_id)
        snapshot = self._snapshots.pop(key, None) if mode == "discard" else self._snapshots.get(key)
        if snapshot is not None:
            with torch.no_grad():
                if mode == "discard":
                    # Exact inverse of the session fine-tune: remove appended session
                    # edges and restore the pre-session embedding table.
                    self.model.embedding.weight.copy_(snapshot["embedding_weight"].to(self.device))
                    self.edge_index = snapshot["edge_index"].to(self.device)
                else:
                    original = snapshot["embedding_weight"].to(self.device)
                    current = self.model.embedding.weight.detach()
                    self.model.embedding.weight.copy_(original + (1.0 - scale) * (current - original))
                    self.edge_index = snapshot["edge_index"].to(self.device)
        else:
            influence = self.compute_session_influence(session_edges, user_id)
            with torch.no_grad():
                self.model.embedding.weight[self.model.user_idx(user_id)] -= scale * influence

        after = self._user_emb(self.model.user_idx(user_id))
        return SessionMetrics(
            cosine_distance=self._cosine(before, after),
            user_embedding_before=before.tolist(),
            user_embedding_after=after.tolist(),
            edges_processed=len(self._resolve(session_edges)),
        )

    def commit_session(
        self,
        session_edges: list[tuple],
        user_id: int,
        num_steps: int = 25,
        lr: float = 5e-3,
    ) -> SessionMetrics:
        """Fine-tune the user embedding on session edges → merges them in."""
        edges = self._resolve(session_edges)
        if not edges:
            v = self._user_emb(self.model.user_idx(user_id))
            return SessionMetrics(0.0, v.tolist(), v.tolist(), 0)

        positive_edges = self._positive_edges(edges)
        if not positive_edges:
            v = self._user_emb(self.model.user_idx(user_id))
            return SessionMetrics(0.0, v.tolist(), v.tolist(), 0)

        key = self._session_key(session_edges, user_id)
        self._snapshots[key] = {
            "embedding_weight": self.model.embedding.weight.detach().clone().cpu(),
            "edge_index": self.edge_index.detach().clone().cpu(),
        }

        users = torch.tensor([e[0] for e in positive_edges], dtype=torch.long, device=self.device)
        pos = torch.tensor([e[1] for e in positive_edges], dtype=torch.long, device=self.device)

        before = self._user_emb(self.model.user_idx(user_id))
        # Only optimise the base embedding table — full param works fine for tiny session
        optim = torch.optim.Adam([self.model.embedding.weight], lr=lr)
        for _ in range(num_steps):
            optim.zero_grad()
            neg = torch.randint(0, self.model.num_movies, pos.shape, device=self.device)
            loss = self.model.bpr_loss(self.edge_index, users, pos, neg, reg=1e-5)
            loss.backward()
            optim.step()

        # Add session edges into permanent edge_index
        new_src = torch.cat([self.edge_index[0], users])
        new_dst = torch.cat([self.edge_index[1], pos + self.model.num_users])
        self.edge_index = torch.stack([new_src, new_dst], dim=0)

        after = self._user_emb(self.model.user_idx(user_id))
        return SessionMetrics(
            cosine_distance=self._cosine(before, after),
            user_embedding_before=before.tolist(),
            user_embedding_after=after.tolist(),
            edges_processed=len(positive_edges),
        )
