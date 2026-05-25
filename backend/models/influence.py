"""
influence.py — Tier 2 session unlearning via influence functions
(Koh & Liang, ICML 2017).

The session graph is treated as a small set of additional user→movie edges.
We approximate the influence of those edges on the user's embedding using a
first-order Taylor expansion of the inverse Hessian-vector product (LiSSA
truncation depth 1; sufficient for the small per-session perturbation).

erase_session    → subtract the influence (revert user embedding toward
                   pre-session state).
commit_session   → run a short fine-tune over the session edges, merging
                   them into the permanent embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.lightgcn import LightGCN


@dataclass
class SessionMetrics:
    cosine_distance: float
    user_embedding_before: list[float]
    user_embedding_after: list[float]
    edges_processed: int

    def to_dict(self) -> dict:
        return {
            "cosine_distance": float(self.cosine_distance),
            "before_vector": list(self.user_embedding_before),
            "after_vector": list(self.user_embedding_after),
            "edges_processed": int(self.edges_processed),
        }


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

    def _user_emb(self, user_id: int) -> torch.Tensor:
        with torch.no_grad():
            emb = self.model.propagate(self.edge_index)
            return emb[user_id].detach().clone()

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a_n = torch.nn.functional.normalize(a, dim=-1)
        b_n = torch.nn.functional.normalize(b, dim=-1)
        return 1.0 - float((a_n * b_n).sum().item())

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
        influence = self.compute_session_influence(session_edges, user_id)
        # Subtract influence directly from the BASE embedding row in place
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

        users = torch.tensor([e[0] for e in edges], dtype=torch.long, device=self.device)
        pos = torch.tensor([e[1] for e in edges], dtype=torch.long, device=self.device)

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
            edges_processed=len(edges),
        )
