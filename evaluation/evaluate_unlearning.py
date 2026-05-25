"""
evaluate_unlearning.py — Produces metrics.json for the project report.

Tier 1 (GNNDelete):
    - forget cosine distance (before vs after) on the forgotten items
    - retain Recall@20 before vs after (should be within ~2%)
    - membership inference attack score on the forget set

Tier 2 (influence functions):
    - embedding reversion: cosine(user_before, user_after_erase) ≈ 0
    - top-20 rank overlap before vs after erase (high = good restoration)

Run from project root:
    python evaluation/evaluate_unlearning.py
"""

from __future__ import annotations

import sys
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from models.lightgcn import LightGCN
from models.gnn_delete import GNNDelete
from models.influence import SessionUnlearner
from evaluation.metrics import (
    cosine_distance,
    recall_at_k,
    rank_overlap,
    membership_inference_score,
    embedding_reversion_score,
)


CKPT = ROOT / "backend" / "models" / "checkpoints" / "lightgcn_best.pt"
OUT = ROOT / "evaluation" / "metrics.json"


def _topk_titles(model, edge_index, item_map_inv, user_id, k=20):
    inv = {v: k for k, v in item_map_inv.items()}  # not used; kept for clarity
    with torch.no_grad():
        emb = model.propagate(edge_index)
        u = emb[user_id]
        movies = emb[model.num_users : model.num_users + model.num_movies]
        scores = movies @ u
        topk = torch.topk(scores, k).indices.tolist()
        return topk  # local movie indices


def run():
    if not CKPT.exists():
        print(f"[Eval] Checkpoint missing: {CKPT}")
        print("Run: python backend/models/train_lightgcn.py --quick")
        sys.exit(1)

    print("[Eval] Loading checkpoint …")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    n_users = ckpt["num_users"]
    n_movies = ckpt["num_movies"]
    n_genres = ckpt["num_genres"]
    model = LightGCN(
        num_users=n_users, num_movies=n_movies, num_genres=n_genres,
        embedding_dim=ckpt.get("embedding_dim", 64),
        num_layers=ckpt.get("num_layers", 3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    edge_index = ckpt["edge_index"]
    item_map = ckpt["item_map"]
    inv_item_map = ckpt["inverse_item_map"]

    rng = random.Random(0)

    # ════ TIER 1 — GNNDelete ════════════════════════════════════════════════
    print("[Eval] === Tier 1: GNNDelete ===")
    forget_local = rng.sample(range(n_movies), k=min(20, n_movies))
    forget_ids = [inv_item_map[i] for i in forget_local]

    # Snapshot before
    with torch.no_grad():
        emb_pre = model.propagate(edge_index).clone()
    pre_forget = emb_pre[n_users + torch.tensor(forget_local)]

    user_pool = list(range(1, n_users))
    eval_users = rng.sample(user_pool, k=min(50, len(user_pool)))
    pre_topk = {u: _topk_titles(model, edge_index, inv_item_map, u, k=20) for u in eval_users}

    deleter = GNNDelete(model, edge_index, item_map)
    new_edges, t1_metrics = deleter.unlearn(forget_ids, num_steps=20, lr=1e-3)

    with torch.no_grad():
        emb_post = model.propagate(new_edges).clone()
    post_forget = emb_post[n_users + torch.tensor(forget_local)]

    forget_cos = float(
        np.mean([cosine_distance(p.numpy(), q.numpy()) for p, q in zip(pre_forget, post_forget)])
    )
    post_topk = {u: _topk_titles(model, new_edges, inv_item_map, u, k=20) for u in eval_users}
    avg_rank_overlap = float(np.mean([rank_overlap(pre_topk[u], post_topk[u], 20) for u in eval_users]))

    # MIA: forget vs random retain
    retain_indices = rng.sample([i for i in range(n_movies) if i not in set(forget_local)], k=50)
    mia = membership_inference_score(model, forget_local, retain_indices, new_edges, n_users)

    tier1 = {
        "forget_cosine_distance": forget_cos,
        "retain_topk_overlap": avg_rank_overlap,
        "membership_inference_score": float(mia),
        "movies_forgotten": len(forget_local),
        "raw_metrics": t1_metrics.to_dict(),
    }
    print(f"  forget cosine: {forget_cos:.4f}  retain overlap: {avg_rank_overlap:.4f}  MIA: {mia:.4f}")

    # ════ TIER 2 — Influence functions ══════════════════════════════════════
    print("[Eval] === Tier 2: Influence functions ===")
    # Re-load original ckpt for a clean test
    model2 = LightGCN(
        num_users=n_users, num_movies=n_movies, num_genres=n_genres,
        embedding_dim=ckpt.get("embedding_dim", 64),
        num_layers=ckpt.get("num_layers", 3),
    )
    model2.load_state_dict(ckpt["model_state"])
    model2.eval()
    su = SessionUnlearner(model2, ckpt["edge_index"], item_map)

    user_id = 0
    with torch.no_grad():
        emb0 = model2.propagate(su.edge_index)
        target_u = emb0[user_id].clone().numpy()

    session_local = rng.sample(range(n_movies), k=10)
    session_tmdb = [inv_item_map[i] for i in session_local]
    session_edges = [(user_id, mid, 1.0) for mid in session_tmdb]

    pre_topk_u = _topk_titles(model2, su.edge_index, inv_item_map, user_id, k=20)
    # commit then erase to simulate full session lifecycle
    su.commit_session(session_edges, user_id, num_steps=10, lr=5e-3)
    mid_topk_u = _topk_titles(model2, su.edge_index, inv_item_map, user_id, k=20)
    erase_metrics = su.erase_session(session_edges, user_id, mode="discard")
    post_topk_u = _topk_titles(model2, su.edge_index, inv_item_map, user_id, k=20)

    with torch.no_grad():
        after_u = model2.propagate(su.edge_index)[user_id].numpy()

    reversion = embedding_reversion_score(target_u, after_u, target_u)
    pre_post_overlap = rank_overlap(pre_topk_u, post_topk_u, 20)
    mid_post_overlap = rank_overlap(mid_topk_u, post_topk_u, 20)
    tier2 = {
        "embedding_reversion_score": float(reversion),
        "rank_overlap_pre_vs_post_erase": float(pre_post_overlap),
        "rank_overlap_mid_vs_post_erase": float(mid_post_overlap),
        "edges_in_session": len(session_edges),
        "raw_metrics": erase_metrics.to_dict(),
    }
    print(f"  reversion: {reversion:.4f}  pre↔post overlap: {pre_post_overlap:.4f}")

    OUT.write_text(json.dumps({"tier1": tier1, "tier2": tier2}, indent=2))
    print(f"[Eval] Wrote {OUT}")


if __name__ == "__main__":
    run()
