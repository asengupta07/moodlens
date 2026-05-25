"""
graph_builder.py — Merge permanent + session graphs into a single viz payload
for GnnVisualizer.

Node types and colors (must match frontend):
    user             — green
    movie_liked      — blue       (permanent likes)
    movie_session    — amber      (session interactions)
    movie_blocked    — red        (permanent erasure)
    genre            — purple
    genre_blocked    — red outline
    recommended      — purple     (current batch of recommendations)
"""

from __future__ import annotations

from graph.preference_graph import PreferenceGraph
from graph.session_graph import SessionGraph


def build_viz_payload(
    preference_graph: PreferenceGraph,
    session_graph: SessionGraph,
    last_recommendations: dict[str, float] | None = None,
) -> dict:
    perm = preference_graph.to_viz_payload()
    sess = session_graph.to_viz_payload()

    nodes = list(perm["nodes"])
    edges = list(perm["edges"])
    seen = {n["id"] for n in nodes}

    # Session overlay
    for n in sess["nodes"]:
        if n["id"] in seen:
            continue
        nodes.append(n)
        seen.add(n["id"])
    for e in sess["edges"]:
        edges.append(e)

    # Current recommendations (purple)
    last_recommendations = last_recommendations or {}
    for title, score in last_recommendations.items():
        n_id = f"rec__{title[:30].replace(' ', '_').replace('/', '_')}"
        if n_id in seen:
            continue
        nodes.append({
            "id": n_id, "label": title, "type": "recommended",
            "weight": round(float(score), 3),
        })
        seen.add(n_id)
        edges.append({
            "source": "user", "target": n_id,
            "weight": round(float(score), 3),
            "type": "recommends",
        })

    state = preference_graph.state
    return {
        "nodes": nodes,
        "edges": edges,
        "genre_weights": {k: round(v, 3) for k, v in state["genre_weights"].items() if v != 1.0},
        "stats": {
            "liked_count": len(state["liked_movies"]),
            "disliked_count": len(state["disliked_movies"]),
            "blocked_count": len(state["blocked_movies"]),
            "liked_genres": list(state["liked_genres"]),
            "disliked_genres": list(state["disliked_genres"]),
            "blocked_genres": list(state.get("blocked_genres", [])),
        },
        "session_active": sess.get("active", False),
        "session_mood": sess.get("mood"),
        "permanent_count": len(state["liked_movies"]) + len(state["blocked_movies"]),
        "session_count": len(session_graph.state.get("interactions", [])),
        "blocked_count": len(state["blocked_movies"]),
    }
