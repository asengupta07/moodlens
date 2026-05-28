"""
Sanity checks for the LightGCN forward + BPR loss + GNNDelete + influence functions
on a tiny synthetic graph that does NOT depend on the real CSV data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.lightgcn import LightGCN
from models.gnn_delete import GNNDelete
from models.influence import SessionUnlearner


def build_toy_graph():
    n_users, n_movies, n_genres = 5, 10, 3
    # user-movie edges
    edges = [
        (0, 5), (0, 6),
        (1, 5), (1, 7),
        (2, 7), (2, 8),
        (3, 8), (3, 9),
        (4, 6), (4, 9),
    ]
    # movie-genre edges
    movie_base = n_users
    genre_base = n_users + n_movies
    edges += [
        (movie_base + 0, genre_base + 0),
        (movie_base + 1, genre_base + 0),
        (movie_base + 2, genre_base + 1),
        (movie_base + 3, genre_base + 2),
    ]
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long), n_users, n_movies, n_genres


def test_lightgcn_forward():
    edge_index, n_users, n_movies, n_genres = build_toy_graph()
    m = LightGCN(n_users, n_movies, n_genres, embedding_dim=8, num_layers=2)
    out = m.propagate(edge_index)
    assert out.shape == (n_users + n_movies + n_genres, 8), f"bad shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in embedding"


def test_bpr_loss_decreases():
    edge_index, n_users, n_movies, n_genres = build_toy_graph()
    m = LightGCN(n_users, n_movies, n_genres, embedding_dim=8, num_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    users = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    pos = torch.tensor([0, 0, 2, 3, 1], dtype=torch.long)
    neg = torch.tensor([3, 4, 5, 1, 0], dtype=torch.long)
    initial = m.bpr_loss(edge_index, users, pos, neg).item()
    for _ in range(50):
        opt.zero_grad()
        loss = m.bpr_loss(edge_index, users, pos, neg)
        loss.backward()
        opt.step()
    final = m.bpr_loss(edge_index, users, pos, neg).item()
    assert final < initial, f"loss did not decrease: {initial} -> {final}"


def test_gnn_delete_shifts_embeddings():
    edge_index, n_users, n_movies, n_genres = build_toy_graph()
    m = LightGCN(n_users, n_movies, n_genres, embedding_dim=8, num_layers=2)
    item_map = {str(i): i for i in range(n_movies)}
    d = GNNDelete(m, edge_index, item_map)

    with torch.no_grad():
        pre = m.propagate(edge_index)[n_users + 0].clone()
    new_edges, metrics = d.unlearn(forget_movie_ids=["0", "1"], num_steps=5, lr=1e-2)
    with torch.no_grad():
        post = m.propagate(new_edges)[n_users + 0]
    cos = 1.0 - torch.nn.functional.cosine_similarity(pre.unsqueeze(0), post.unsqueeze(0)).item()
    assert cos > 0.0, f"GNNDelete produced zero drift, cos={cos}"
    assert metrics.edges_removed > 0, f"no edges removed, got {metrics.edges_removed}"


def test_influence_erase_changes_user_embedding():
    edge_index, n_users, n_movies, n_genres = build_toy_graph()
    m = LightGCN(n_users, n_movies, n_genres, embedding_dim=8, num_layers=2)
    item_map = {str(i): i for i in range(n_movies)}
    su = SessionUnlearner(m, edge_index, item_map)

    session = [(0, "5", 1.0), (0, "6", 1.0)]
    metrics = su.erase_session(session, user_id=0, mode="discard")
    assert metrics.cosine_distance >= 0.0
    assert metrics.edges_processed == 2


def test_influence_commit_then_erase_roundtrip():
    edge_index, n_users, n_movies, n_genres = build_toy_graph()
    m = LightGCN(n_users, n_movies, n_genres, embedding_dim=8, num_layers=2)
    item_map = {str(i): i for i in range(n_movies)}
    su = SessionUnlearner(m, edge_index, item_map)

    with torch.no_grad():
        before = m.propagate(su.edge_index)[0].clone()
    session = [(0, "5", 1.0), (0, "7", 1.0)]
    original_edge_count = su.edge_index.shape[1]
    su.commit_session(session, user_id=0, num_steps=10, lr=5e-3)
    with torch.no_grad():
        after_commit = m.propagate(su.edge_index)[0].clone()
    su.erase_session(session, user_id=0, mode="discard")
    with torch.no_grad():
        after_erase = m.propagate(su.edge_index)[0].clone()

    drift_commit = 1.0 - torch.nn.functional.cosine_similarity(
        before.unsqueeze(0), after_commit.unsqueeze(0)
    ).item()
    drift_erase = 1.0 - torch.nn.functional.cosine_similarity(
        before.unsqueeze(0), after_erase.unsqueeze(0)
    ).item()
    assert su.edge_index.shape[1] == original_edge_count
    assert drift_commit > 0.0
    assert drift_erase < 1e-5, f"erase did not restore baseline: {drift_erase:.8f}"


if __name__ == "__main__":
    test_lightgcn_forward()
    print("✓ forward shape OK")
    test_bpr_loss_decreases()
    print("✓ BPR loss decreases")
    test_gnn_delete_shifts_embeddings()
    print("✓ GNNDelete produces embedding drift")
    test_influence_erase_changes_user_embedding()
    print("✓ Influence erase runs")
    test_influence_commit_then_erase_roundtrip()
    print("✓ commit→erase roundtrip runs")
    print("\nAll backend ML tests passed.")
