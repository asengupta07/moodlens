"""
lightgcn.py — LightGCN (He et al., SIGIR 2020) implemented from scratch in PyTorch.

No PyG dependency. Manual message passing so GNNDelete and influence functions
can directly operate on the embedding matrices.

Tripartite graph: user nodes + movie nodes + genre nodes, indexed contiguously:
    [0, n_users)                              → users
    [n_users, n_users + n_movies)             → movies
    [n_users + n_movies, n_users + n_movies + n_genres) → genres

Edges:
    user ↔ movie  (positive interactions)
    movie ↔ genre (membership)

Embedding dim = 64, layers = 3 (fixed per CLAUDE.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn


EMBEDDING_DIM = 64
NUM_LAYERS = 3


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_genres: int = 0,
        embedding_dim: int = EMBEDDING_DIM,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_genres = num_genres
        self.num_total = num_users + num_movies + num_genres
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.num_total, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    # ── Index helpers ──────────────────────────────────────────────────────────
    def user_idx(self, user_id: int) -> int:
        return int(user_id)

    def movie_idx(self, movie_local_id: int) -> int:
        return self.num_users + int(movie_local_id)

    def genre_idx(self, genre_local_id: int) -> int:
        return self.num_users + self.num_movies + int(genre_local_id)

    # ── Forward / propagation ─────────────────────────────────────────────────
    def _build_norm_adj(self, edge_index: torch.Tensor) -> torch.sparse.Tensor:
        """Symmetric-normalised adjacency:  A_hat = D^{-1/2} (A + A^T) D^{-1/2}"""
        device = edge_index.device
        n = self.num_total

        # Make symmetric
        src, dst = edge_index[0], edge_index[1]
        all_src = torch.cat([src, dst])
        all_dst = torch.cat([dst, src])

        # Degree
        deg = torch.zeros(n, device=device)
        deg.scatter_add_(0, all_src, torch.ones_like(all_src, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        norm = deg_inv_sqrt[all_src] * deg_inv_sqrt[all_dst]
        indices = torch.stack([all_src, all_dst], dim=0)
        return torch.sparse_coo_tensor(indices, norm, (n, n)).coalesce()

    def propagate(self, edge_index: torch.Tensor) -> torch.Tensor:
        """L layers of mean propagation. Returns final embedding matrix (N × D)."""
        adj = self._build_norm_adj(edge_index)
        x = self.embedding.weight
        layer_outputs = [x]
        for _ in range(self.num_layers):
            x = torch.sparse.mm(adj, x)
            layer_outputs.append(x)
        # Mean of all layers (LightGCN uses simple mean)
        return torch.stack(layer_outputs, dim=0).mean(dim=0)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index)

    # ── Convenience getters ───────────────────────────────────────────────────
    def get_all_embeddings(
        self, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.propagate(edge_index)
        users = emb[: self.num_users]
        movies = emb[self.num_users : self.num_users + self.num_movies]
        genres = emb[self.num_users + self.num_movies :]
        return users, movies, genres

    def get_user_embedding(
        self, user_id: int, edge_index: torch.Tensor
    ) -> torch.Tensor:
        emb = self.propagate(edge_index)
        return emb[self.user_idx(user_id)]

    def get_movie_embedding(
        self, movie_local_id: int, edge_index: torch.Tensor
    ) -> torch.Tensor:
        emb = self.propagate(edge_index)
        return emb[self.movie_idx(movie_local_id)]

    # ── BPR loss ──────────────────────────────────────────────────────────────
    def bpr_loss(
        self,
        edge_index: torch.Tensor,
        users: torch.Tensor,      # (B,)
        pos_movies: torch.Tensor, # (B,) — local movie indices
        neg_movies: torch.Tensor, # (B,) — local movie indices
        reg: float = 1e-4,
    ) -> torch.Tensor:
        emb = self.propagate(edge_index)
        u_e = emb[users]
        pos_e = emb[self.num_users + pos_movies]
        neg_e = emb[self.num_users + neg_movies]

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 reg on the BASE embedding (per LightGCN paper)
        base = self.embedding.weight
        reg_loss = reg * (
            base[users].pow(2).sum()
            + base[self.num_users + pos_movies].pow(2).sum()
            + base[self.num_users + neg_movies].pow(2).sum()
        ) / users.size(0)

        return loss + reg_loss

    # ── Scoring helper ────────────────────────────────────────────────────────
    @torch.no_grad()
    def score_movies(
        self,
        user_emb: torch.Tensor,
        movie_embs: torch.Tensor,
    ) -> torch.Tensor:
        """Dot-product scores. user_emb (D,), movie_embs (M, D)."""
        return movie_embs @ user_emb
