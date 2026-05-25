"""Tests for the PreferenceGraph, SessionGraph, and graph_builder modules."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph.preference_graph import PreferenceGraph
from graph.session_graph import SessionGraph
from graph.graph_builder import build_viz_payload


def test_preference_graph_likes_and_blocks(tmp_path=None):
    pg = PreferenceGraph()
    pg.add_like("123", "Inception", ["Action", "Sci-Fi"])
    pg.add_permanent_dislike("456", "Bad Movie", ["Comedy"])
    pg.add_permanent_genre_block("Horror")

    assert any("Inception" in (m.get("title") if isinstance(m, dict) else m)
               for m in pg.state["liked_movies"])
    assert "Horror" in pg.state["blocked_genres"]
    blocked = pg.get_blocked_movie_titles()
    assert "bad movie" in blocked
    weights = pg.get_genre_weights()
    assert weights["Action"] > 1.0
    assert weights["Horror"] < 1.0


def test_preference_graph_persistence():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    pg = PreferenceGraph(path=path)
    pg.add_like("1", "X", ["Drama"])
    pg.save()
    pg2 = PreferenceGraph.load(path)
    titles = [m.get("title") if isinstance(m, dict) else m for m in pg2.state["liked_movies"]]
    assert "X" in titles


def test_session_graph_lifecycle():
    sg = SessionGraph()
    assert not sg.is_active()
    sid = sg.start_session()
    assert sg.is_active()
    sg.add_interaction("100", "Movie A", "recommended", weight=0.5, genres=["Action"])
    sg.add_interaction("101", "Movie B", "liked", weight=1.0, genres=["Action", "Adventure"])
    summary = sg.summary()
    assert summary["movie_count"] == 2
    assert summary["dominant_mood"] == "Action"
    edges = sg.get_edges_for_user(user_id=0)
    assert len(edges) == 2
    assert all(e[0] == 0 for e in edges)
    sg.clear()
    assert not sg.is_active()


def test_graph_builder_merges():
    pg = PreferenceGraph()
    pg.add_like("1", "Movie One", ["Action"])
    pg.add_permanent_genre_block("Horror")
    sg = SessionGraph()
    sg.start_session()
    sg.add_interaction("2", "Mood Pick", "recommended", genres=["Thriller"])

    payload = build_viz_payload(pg, sg, last_recommendations={"Mood Pick": 0.9})
    types = {n["type"] for n in payload["nodes"]}
    assert "user" in types
    assert "movie_liked" in types
    assert "movie_session" in types or "recommended" in types
    assert payload["session_active"] is True
    assert payload["session_count"] == 1


if __name__ == "__main__":
    test_preference_graph_likes_and_blocks()
    print("✓ preference graph likes/blocks")
    test_preference_graph_persistence()
    print("✓ preference graph persistence")
    test_session_graph_lifecycle()
    print("✓ session graph lifecycle")
    test_graph_builder_merges()
    print("✓ graph builder merges tiers")
    print("\nAll graph tests passed.")
