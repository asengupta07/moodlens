"""
End-to-end check: StateManager loads the trained checkpoint, accepts a
permanent unlearn intent, and exposes the embedding drift through the
preference graph.
"""

from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from state_manager import StateManager
from scoring_engine import build_movie_db, compute_lightgcn_scores, score_and_recommend

DATA_DIR = ROOT / "data"
CKPT = ROOT / "models" / "checkpoints" / "lightgcn_best.pt"


def _tmp(name: str) -> str:
    return str(Path(tempfile.gettempdir()) / name)


def _tmp_ckpt(name: str) -> str:
    dst = Path(tempfile.gettempdir()) / name
    shutil.copy2(CKPT, dst)
    return str(dst)


def test_state_manager_loads_checkpoint():
    state_path = _tmp("mt_state.json")
    sess_path = _tmp("mt_session.json")
    for p in (state_path, sess_path):
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass
    mgr = StateManager(state_path, sess_path, str(CKPT))
    assert mgr.lightgcn is not None, "checkpoint failed to load"
    assert mgr.edge_index is not None
    assert mgr.gnn_delete is not None
    assert mgr.session_unlearner is not None
    print(f"✓ loaded: users={mgr.lightgcn.num_users} movies={mgr.lightgcn.num_movies}")


def test_compute_lightgcn_scores_pipeline():
    state_path = _tmp("mt_state2.json")
    sess_path = _tmp("mt_session2.json")
    mgr = StateManager(state_path, sess_path, _tmp_ckpt("mt_lightgcn3.pt"))
    movie_db = build_movie_db(
        str(DATA_DIR / "movies_metadata.csv"),
        str(DATA_DIR / "credits.csv"),
        str(DATA_DIR / "ratings.csv"),
    )
    scores = compute_lightgcn_scores(
        mgr.lightgcn, mgr.edge_index, mgr.lightgcn_user_id, movie_db, mgr.item_map,
    )
    assert len(scores) > 0, "no lightgcn scores computed"
    print(f"✓ lightgcn scored {len(scores)} movies")

    intent = {"genres_requested": ["Action"]}
    recs = score_and_recommend(
        movie_db=movie_db,
        intent=intent,
        state=mgr.preference_graph.state,
        plot_index=None,
        top_n=5,
        lightgcn_scores=scores,
        lightgcn_alpha=0.6,
        preference_graph=mgr.preference_graph,
        session_graph=mgr.session_graph,
    )
    assert len(recs) > 0, "no recs returned"
    print(f"✓ blended recs: {[r[0] for r in recs]}")


def test_state_manager_permanent_unlearn_path():
    state_path = _tmp("mt_state3.json")
    sess_path = _tmp("mt_session3.json")
    for p in (state_path, sess_path):
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass
    mgr = StateManager(state_path, sess_path, str(CKPT))
    movie_db = build_movie_db(
        str(DATA_DIR / "movies_metadata.csv"),
        str(DATA_DIR / "credits.csv"),
        str(DATA_DIR / "ratings.csv"),
    )
    # Pick first 5 horror movies as forget set
    horror_rows = movie_db[movie_db["genres_list"].apply(lambda gs: "Horror" in (gs or []))]
    forget_ids = horror_rows["movie_id"].astype(str).head(5).tolist()

    intent = {
        "is_permanent": True,
        "permanent_movie_block": [movie_db[movie_db["movie_id"] == fid].iloc[0]["title_clean"]
                                  for fid in forget_ids if not movie_db[movie_db["movie_id"] == fid].empty],
        "permanent_genre_block": [],
        "permanent_year_block": None,
        "liked_found": [],
        "disliked_found": [],
        "liked_genres": [],
        "disliked_genres": [],
        "soft_disliked_genres": [],
        "genres_requested": [],
    }
    proc = mgr.process_intent(intent, movie_db)
    assert proc["unlearning_triggered"] is True
    assert proc["unlearning_tier"] == 1
    metrics = proc["metrics"]
    assert metrics["movies_affected"] >= 1
    print(f"✓ permanent unlearn fired, movies_affected={metrics['movies_affected']}, "
          f"cos={metrics['cosine_distance']:.4f}")


def test_state_manager_session_lifecycle():
    state_path = _tmp("mt_state4.json")
    sess_path = _tmp("mt_session4.json")
    for p in (state_path, sess_path):
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass
    mgr = StateManager(state_path, sess_path, str(CKPT))
    movie_db = build_movie_db(
        str(DATA_DIR / "movies_metadata.csv"),
        str(DATA_DIR / "credits.csv"),
        str(DATA_DIR / "ratings.csv"),
    )
    sid = mgr.session_graph.start_session()
    # pick first 3 movies
    for _, row in movie_db.head(3).iterrows():
        mgr.session_graph.add_interaction(
            str(row["movie_id"]), str(row["title"]), "recommended",
            weight=1.0, genres=list(row["genres_list"]),
        )
    assert mgr.session_graph.is_active()
    before_weight = {k: v.detach().clone() for k, v in mgr.lightgcn.state_dict().items()}
    before_edges = mgr.edge_index.detach().clone()
    result = mgr.trigger_session_end("discard")
    assert "metrics" in result
    assert not mgr.session_graph.is_active()
    assert result["metrics"]["non_destructive"] is True
    assert torch.equal(before_edges, mgr.edge_index)
    for key, value in mgr.lightgcn.state_dict().items():
        assert torch.equal(before_weight[key], value), f"discard mutated model parameter {key}"
    print(f"✓ session discard: cos={result['metrics']['cosine_distance']:.4f}")


if __name__ == "__main__":
    test_state_manager_loads_checkpoint()
    test_compute_lightgcn_scores_pipeline()
    test_state_manager_permanent_unlearn_path()
    test_state_manager_session_lifecycle()
    print("\nAll StateManager integration tests passed.")
