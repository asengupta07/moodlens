"""
train_lightgcn.py — Train LightGCN on the TMDB dataset.

Edges are synthesised from movies_metadata.csv vote counts:
    for each movie m with vote_count_m > 0,
        K_m = clip(floor(log2(vote_count_m + 1)), 1, 30) synthetic users "like" m
    plus movie ↔ genre membership edges.

This produces a real, reproducible bipartite-plus-genre graph keyed on the
TMDB id used everywhere else in the system.

The first user index (id=0) is reserved for the live MoodLens chat user.

Saves a checkpoint dict containing:
    model_state         — torch state_dict
    edge_index          — LongTensor (2, E)
    user_map            — {synthetic_user_id: int}
    item_map            — {tmdb_movie_id_str: local_idx}
    inverse_item_map    — {local_idx: tmdb_movie_id_str}
    genre_map           — {genre_name: local_idx}
    initial_embeddings  — Tensor (N_total, D)   (untrained, for influence ref)
    num_users, num_movies, num_genres
"""

from __future__ import annotations

import os
import sys
import math
import time
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    class tqdm:  # minimal fallback
        def __init__(self, iterable=None, total=None, desc="", leave=True, **kw):
            self.iterable = iterable
            self.total = total or (len(iterable) if hasattr(iterable, "__len__") else None)
            self.desc = desc
            self.n = 0
            self._postfix = ""

        def __iter__(self):
            for x in self.iterable:
                self.n += 1
                self._print()
                yield x
            print()

        def update(self, k=1):
            self.n += k
            self._print()

        def set_postfix_str(self, s):
            self._postfix = s
            self._print()

        def close(self):
            print()

        def _print(self):
            bar_len = 24
            frac = self.n / self.total if self.total else 0
            filled = int(bar_len * frac)
            bar = "█" * filled + "·" * (bar_len - filled)
            print(f"\r  {self.desc} [{bar}] {self.n}/{self.total} {self._postfix}",
                  end="", flush=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.lightgcn import LightGCN, EMBEDDING_DIM, NUM_LAYERS
from scoring_engine import build_movie_db


SYNTHETIC_USERS = 200      # population of fake users for edge synthesis
RESERVED_USER_ID = 0       # idx reserved for live chat user
MIN_VOTE_COUNT = 5         # ignore movies with fewer votes
MAX_EDGES_PER_MOVIE = 30
NEG_SAMPLES_PER_POS = 1
SEED = 42

CKPT_PATH = ROOT / "models" / "checkpoints" / "lightgcn_best.pt"


def synthesise_edges(
    movie_db: pd.DataFrame,
    num_users: int,
    seed: int = SEED,
) -> tuple[torch.LongTensor, dict, dict, dict]:
    """
    Build edge_index plus the three id maps.
    Returns (edge_index, user_map, item_map, genre_map).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    item_map: dict[str, int] = {}
    inv_item_map: dict[int, str] = {}
    movie_rows = movie_db.reset_index(drop=True)
    for local_idx, row in movie_rows.iterrows():
        tmdb_id = str(row["movie_id"])
        item_map[tmdb_id] = local_idx
        inv_item_map[local_idx] = tmdb_id

    user_map = {i: i for i in range(num_users)}  # identity

    # ── Genres ────────────────────────────────────────────────────────────────
    all_genres: set[str] = set()
    for row_genres in movie_rows["genres_list"]:
        if row_genres:
            for g in row_genres:
                all_genres.add(g.strip())
    genre_list = sorted(all_genres)
    genre_map: dict[str, int] = {g: i for i, g in enumerate(genre_list)}

    n_movies = len(movie_rows)
    n_genres = len(genre_map)

    # Synthesize user→movie edges based on vote_count
    edges_u: list[int] = []
    edges_m: list[int] = []

    vote_counts = pd.to_numeric(movie_rows["vote_count"], errors="coerce").fillna(0).values
    for local_idx in range(n_movies):
        vc = int(vote_counts[local_idx])
        if vc < MIN_VOTE_COUNT:
            continue
        k = min(MAX_EDGES_PER_MOVIE, max(1, int(math.log2(vc + 1))))
        # skip user 0 (reserved for live chat user) — keep them edgeless
        sampled_users = np_rng.choice(range(1, num_users), size=k, replace=False)
        for u in sampled_users:
            edges_u.append(int(u))
            edges_m.append(num_users + local_idx)  # global movie idx

    # movie ↔ genre edges
    edges_mg_src: list[int] = []
    edges_mg_dst: list[int] = []
    movie_base = num_users
    genre_base = num_users + n_movies
    for local_idx, row_genres in enumerate(movie_rows["genres_list"]):
        if not row_genres:
            continue
        for g in row_genres:
            g = g.strip()
            if g in genre_map:
                edges_mg_src.append(movie_base + local_idx)
                edges_mg_dst.append(genre_base + genre_map[g])

    src = edges_u + edges_mg_src
    dst = edges_m + edges_mg_dst
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index, user_map, item_map, genre_map


def sample_bpr_batch(
    user_pos_set: dict[int, set[int]],
    users_with_edges: list[int],
    num_movies: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    users = []
    pos = []
    neg = []
    while len(users) < batch_size:
        u = rng.choice(users_with_edges)
        pos_set = user_pos_set[u]
        if not pos_set:
            continue
        p = rng.choice(list(pos_set))
        # negative sample
        for _ in range(20):
            n = rng.randrange(num_movies)
            if n not in pos_set:
                break
        users.append(u)
        pos.append(p)
        neg.append(n)
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos, dtype=torch.long),
        torch.tensor(neg, dtype=torch.long),
    )


def recall_at_k(
    model: LightGCN,
    edge_index: torch.Tensor,
    eval_pos: dict[int, set[int]],
    train_pos: dict[int, set[int]],
    k: int = 20,
    sample_users: int = 200,
    rng: random.Random | None = None,
) -> float:
    rng = rng or random.Random(SEED)
    eligible = [u for u, s in eval_pos.items() if s]
    if not eligible:
        return 0.0
    if len(eligible) > sample_users:
        eligible = rng.sample(eligible, sample_users)
    model.eval()
    with torch.no_grad():
        emb = model.propagate(edge_index)
        users_emb = emb[: model.num_users]
        movies_emb = emb[model.num_users : model.num_users + model.num_movies]
        hits = 0
        total = 0
        for u in eligible:
            seen = train_pos.get(u, set())
            scores = movies_emb @ users_emb[u]
            scores = scores.clone()
            for s in seen:
                scores[s] = -1e9
            topk = torch.topk(scores, k).indices.tolist()
            target = eval_pos[u]
            hit = len(set(topk) & target)
            if target:
                hits += hit
                total += min(k, len(target))
        if total == 0:
            return 0.0
        return hits / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--users", type=int, default=SYNTHETIC_USERS)
    parser.add_argument("--metadata", type=str, default=str(ROOT / "data" / "movies_metadata.csv"))
    parser.add_argument("--credits", type=str, default=str(ROOT / "data" / "credits.csv"))
    parser.add_argument("--ratings", type=str, default=str(ROOT / "data" / "ratings.csv"))
    parser.add_argument("--max-movies", type=int, default=0,
                        help="If >0, subsample to top-N movies by vote_count for fast experiments.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke run: 2 epochs, 5k movies, 50 users.")
    parser.add_argument("--out", type=str, default=str(CKPT_PATH))
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.users = 50
        args.max_movies = 5000

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[Train] Loading movie database …")
    movie_db = build_movie_db(args.metadata, args.credits, args.ratings)

    if args.max_movies > 0:
        movie_db = (
            movie_db.assign(_vc=pd.to_numeric(movie_db["vote_count"], errors="coerce").fillna(0))
            .sort_values("_vc", ascending=False)
            .head(args.max_movies)
            .drop(columns=["_vc"])
            .reset_index(drop=True)
        )
        print(f"[Train] Subsampled to top {len(movie_db)} movies by vote_count.")

    print(f"[Train] Synthesising graph with {args.users} users …")
    edge_index, user_map, item_map, genre_map = synthesise_edges(movie_db, args.users)
    n_movies = len(movie_db)
    n_genres = len(genre_map)
    n_users = args.users
    print(f"[Train] Graph: {n_users} users, {n_movies} movies, {n_genres} genres, "
          f"{edge_index.size(1)} undirected edges.")

    # Build per-user positive sets (only user→movie edges, exclude movie↔genre)
    user_pos: dict[int, set[int]] = {u: set() for u in range(n_users)}
    for col in range(edge_index.size(1)):
        s, d = int(edge_index[0, col]), int(edge_index[1, col])
        if s < n_users and n_users <= d < n_users + n_movies:
            user_pos[s].add(d - n_users)

    # 80/20 split per user
    train_pos: dict[int, set[int]] = {}
    eval_pos: dict[int, set[int]] = {}
    rng = random.Random(SEED)
    for u, pos in user_pos.items():
        pos_list = list(pos)
        rng.shuffle(pos_list)
        cut = max(1, int(0.8 * len(pos_list))) if pos_list else 0
        train_pos[u] = set(pos_list[:cut])
        eval_pos[u] = set(pos_list[cut:])

    # Rebuild edge_index with only train edges (genre edges always retained)
    train_edges_src: list[int] = []
    train_edges_dst: list[int] = []
    for u, pos in train_pos.items():
        for m in pos:
            train_edges_src.append(u)
            train_edges_dst.append(n_users + m)
    # add movie↔genre
    for col in range(edge_index.size(1)):
        s, d = int(edge_index[0, col]), int(edge_index[1, col])
        if d >= n_users + n_movies or s >= n_users + n_movies:
            train_edges_src.append(s)
            train_edges_dst.append(d)
    train_edge_index = torch.tensor([train_edges_src, train_edges_dst], dtype=torch.long)

    users_with_edges = [u for u, s in train_pos.items() if s]
    if not users_with_edges:
        raise RuntimeError("No training edges generated. Check vote_count distribution.")

    model = LightGCN(num_users=n_users, num_movies=n_movies, num_genres=n_genres)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Snapshot pre-training embedding for influence-function reference
    initial_embeddings = model.embedding.weight.detach().clone()

    best_recall = -1.0
    batches_per_epoch = max(1, sum(len(s) for s in train_pos.values()) // args.batch_size)
    print(f"[Train] Training {args.epochs} epochs · {batches_per_epoch} batches/epoch · "
          f"batch {args.batch_size} · lr {args.lr}")
    if not _HAS_TQDM:
        print("[Train] tip: `pip install tqdm` for nicer progress bars.")

    start = time.time()
    eval_every = max(1, args.epochs // 10)

    epoch_bar = tqdm(range(1, args.epochs + 1), total=args.epochs, desc="epochs", leave=True)
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0

        batch_bar = tqdm(range(batches_per_epoch), total=batches_per_epoch,
                         desc=f"  ep {epoch:3d}", leave=False)
        for _ in batch_bar:
            users, pos, neg = sample_bpr_batch(
                train_pos, users_with_edges, n_movies, args.batch_size, rng
            )
            optim.zero_grad()
            loss = model.bpr_loss(train_edge_index, users, pos, neg)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            batch_bar.set_postfix_str(f"loss {loss.item():.4f}")
        epoch_loss /= batches_per_epoch

        elapsed = time.time() - start
        postfix = f"loss {epoch_loss:.4f} | {elapsed:5.0f}s"

        if epoch % eval_every == 0 or epoch == args.epochs:
            r20 = recall_at_k(model, train_edge_index, eval_pos, train_pos, k=20)
            postfix = f"loss {epoch_loss:.4f} | r@20 {r20:.4f} | {elapsed:5.0f}s"
            if r20 > best_recall:
                best_recall = r20
                _save_ckpt(args.out, model, edge_index, user_map, item_map,
                           genre_map, initial_embeddings, n_users, n_movies, n_genres)
                postfix += " | saved✓"
        epoch_bar.set_postfix_str(postfix)

    # Always save final state too (overwrites best if no eval improvement was logged)
    if best_recall < 0:
        _save_ckpt(args.out, model, edge_index, user_map, item_map,
                   genre_map, initial_embeddings, n_users, n_movies, n_genres)
    print(f"[Train] Done. Best Recall@20 = {best_recall:.4f}. Saved to {args.out}")


def _save_ckpt(path, model, edge_index, user_map, item_map, genre_map,
               initial_embeddings, n_users, n_movies, n_genres):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "edge_index": edge_index,
        "user_map": user_map,
        "item_map": item_map,
        "inverse_item_map": {v: k for k, v in item_map.items()},
        "genre_map": genre_map,
        "initial_embeddings": initial_embeddings,
        "num_users": n_users,
        "num_movies": n_movies,
        "num_genres": n_genres,
        "embedding_dim": EMBEDDING_DIM,
        "num_layers": NUM_LAYERS,
    }, path)


if __name__ == "__main__":
    main()
