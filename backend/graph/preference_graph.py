"""
preference_graph.py — Long-lived user identity / preference graph.

Persists to user_state.json. Manages:
    liked_movies         — list of {id, title, timestamp}
    disliked_movies      — soft session dislikes (mirrors existing state)
    blocked_movies       — permanently erased movies (Tier 1)
    blocked_genres       — permanently erased genres (Tier 1)
    genre_weights        — multiplicative per-genre score weights
    embedding_drift_history — list of {tier, event_type, timestamp, cosine_distance, ...}
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

DEFAULT_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Sci-Fi", "Thriller", "TV Movie", "War", "Western",
]

LIKE_BOOST = 0.15
LIKE_CAP = 1.5
DISLIKE_DECAY = 0.70


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class PreferenceGraph:
    def __init__(self, state: dict | None = None, path: str | None = None):
        self.path = path
        self.state = state if state is not None else self._default()
        self._normalise()

    # ── Default + io ──────────────────────────────────────────────────────────
    @staticmethod
    def _default() -> dict:
        return {
            "liked_movies": [],
            "disliked_movies": [],
            "blocked_movies": [],
            "liked_genres": [],
            "disliked_genres": [],
            "blocked_genres": [],
            "genre_weights": {g: 1.0 for g in DEFAULT_GENRES},
            "last_recommendations": [],
            "last_recommendation_scores": {},
            "last_requested_genres": [],
            "preference_events": [],
            "embedding_drift_history": [],
            "lightgcn_user_id": 0,
        }

    def _normalise(self) -> None:
        d = self._default()
        for k, v in d.items():
            if k not in self.state:
                self.state[k] = v
        for g in DEFAULT_GENRES:
            self.state["genre_weights"].setdefault(g, 1.0)

    # ── Likes / dislikes ──────────────────────────────────────────────────────
    def add_like(self, movie_id: int | str, title: str, genres: list[str], engagement: float = 1.0):
        rec = {"id": str(movie_id), "title": title, "timestamp": _now(), "engagement": engagement}
        existing_titles = [m.get("title") if isinstance(m, dict) else str(m)
                           for m in self.state["liked_movies"]]
        if title not in existing_titles:
            self.state["liked_movies"].append(rec)
        for g in genres:
            self._boost_genre(g)
        self._event("like", {"movie_id": str(movie_id), "title": title, "genres": genres})

    def add_soft_dislike(self, title: str, genres: list[str]):
        """Session-style soft dislike — just decays genres, doesn't block."""
        if title and title not in self.state["disliked_movies"]:
            self.state["disliked_movies"].append(title)
        for g in genres:
            self._decay_genre(g)
        self._event("soft_dislike", {"title": title, "genres": genres})

    def add_permanent_dislike(self, movie_id: int | str, title: str, genres: list[str]):
        rec = {"id": str(movie_id), "title": title, "timestamp": _now(), "reason": "permanent"}
        existing = [m.get("title") if isinstance(m, dict) else str(m)
                    for m in self.state["blocked_movies"]]
        if title not in existing:
            self.state["blocked_movies"].append(rec)
        # Also mirror to disliked_movies for legacy scoring path
        if title and title not in self.state["disliked_movies"]:
            self.state["disliked_movies"].append(title)
        for g in genres:
            self._decay_genre(g)
        self._event("permanent_dislike_movie", {
            "movie_id": str(movie_id), "title": title, "genres": genres,
        })

    def add_permanent_genre_block(self, genre: str):
        genre = genre.strip()
        if not genre:
            return
        if genre not in self.state["blocked_genres"]:
            self.state["blocked_genres"].append(genre)
        if genre not in self.state["disliked_genres"]:
            self.state["disliked_genres"].append(genre)
        if genre in self.state["liked_genres"]:
            self.state["liked_genres"].remove(genre)
        self._decay_genre(genre)
        self._event("permanent_block_genre", {"genre": genre})

    def add_liked_genre(self, genre: str):
        genre = genre.strip()
        if not genre:
            return
        if genre not in self.state["liked_genres"]:
            self.state["liked_genres"].append(genre)
        if genre in self.state["disliked_genres"]:
            self.state["disliked_genres"].remove(genre)
        if genre in self.state["blocked_genres"]:
            self.state["blocked_genres"].remove(genre)
        self._boost_genre(genre)

    def add_disliked_genre(self, genre: str):
        # session-style, no permanent block
        genre = genre.strip()
        if not genre:
            return
        if genre not in self.state["disliked_genres"]:
            self.state["disliked_genres"].append(genre)
        if genre in self.state["liked_genres"]:
            self.state["liked_genres"].remove(genre)
        self._decay_genre(genre)

    # ── Genre weight ops ──────────────────────────────────────────────────────
    def _boost_genre(self, genre: str):
        genre = genre.strip()
        if not genre:
            return
        cur = self.state["genre_weights"].get(genre, 1.0)
        self.state["genre_weights"][genre] = round(min(cur + LIKE_BOOST, LIKE_CAP), 4)

    def _decay_genre(self, genre: str):
        genre = genre.strip()
        if not genre:
            return
        cur = self.state["genre_weights"].get(genre, 1.0)
        self.state["genre_weights"][genre] = round(cur * DISLIKE_DECAY, 4)

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_blocked_movie_titles(self) -> set[str]:
        out: set[str] = set()
        for m in self.state["blocked_movies"]:
            t = m.get("title") if isinstance(m, dict) else str(m)
            if t:
                out.add(t.lower())
        # legacy disliked_movies also act as blocked
        for t in self.state.get("disliked_movies", []):
            out.add(str(t).lower())
        return out

    def get_blocked_movie_ids(self) -> set[str]:
        out: set[str] = set()
        for m in self.state["blocked_movies"]:
            if isinstance(m, dict) and m.get("id"):
                out.add(str(m["id"]))
        return out

    def get_blocked_genres(self) -> set[str]:
        out = {g.lower() for g in self.state.get("blocked_genres", [])}
        out |= {g.lower() for g in self.state.get("disliked_genres", [])}
        return out

    def get_genre_weights(self) -> dict[str, float]:
        return dict(self.state["genre_weights"])

    # ── Drift history ─────────────────────────────────────────────────────────
    def record_drift(self, tier: int, event_type: str, payload: dict):
        rec = {
            "tier": tier,
            "event_type": event_type,
            "timestamp": _now(),
            **payload,
        }
        hist = self.state.setdefault("embedding_drift_history", [])
        hist.append(rec)
        # keep last 100
        del hist[:-100]

    # ── Event log ─────────────────────────────────────────────────────────────
    def _event(self, ev_type: str, payload: dict):
        events = self.state.setdefault("preference_events", [])
        events.append({"type": ev_type, "timestamp": _now(), **payload})
        del events[:-100]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | None = None) -> None:
        path = path or self.path
        if not path:
            return
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "PreferenceGraph":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    return cls(state=data, path=path)
            except (json.JSONDecodeError, ValueError):
                pass
        pg = cls(state=cls._default(), path=path)
        pg.save()
        return pg

    # ── Viz payload ───────────────────────────────────────────────────────────
    def to_viz_payload(self) -> dict:
        """Returns simple node/edge JSON for the permanent tier."""
        nodes = [{"id": "user", "label": "You", "type": "user", "weight": 1.0}]
        edges = []
        seen = {"user"}

        def nid(prefix: str, label: str) -> str:
            return f"{prefix}__{label[:30].replace(' ', '_').replace('/', '_')}"

        for m in self.state["liked_movies"]:
            t = m.get("title") if isinstance(m, dict) else str(m)
            n = nid("liked", t)
            if n not in seen:
                nodes.append({"id": n, "label": t, "type": "movie_liked", "weight": 1.0})
                seen.add(n)
                edges.append({"source": "user", "target": n, "weight": 0.8, "type": "permanent_like"})

        for m in self.state["blocked_movies"]:
            t = m.get("title") if isinstance(m, dict) else str(m)
            n = nid("blocked", t)
            if n not in seen:
                nodes.append({"id": n, "label": t, "type": "movie_blocked", "weight": 0.0})
                seen.add(n)
                edges.append({"source": "user", "target": n, "weight": 0.0, "type": "blocked"})

        for g, w in self.state["genre_weights"].items():
            if w == 1.0:
                continue
            n = nid("genre", g)
            ntype = "genre_blocked" if g in self.state.get("blocked_genres", []) else "genre"
            nodes.append({"id": n, "label": g, "type": ntype, "weight": round(w, 3)})
            edges.append({"source": "user", "target": n, "weight": round(w, 3), "type": "genre_weight"})

        return {"nodes": nodes, "edges": edges}
