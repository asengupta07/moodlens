"""
session_graph.py — Ephemeral mood session subgraph.

Created on first interaction after the user lands or clears their previous
mood, destroyed by the "New Mood" button. Persisted to session_state.json so
it survives a server restart in the middle of a session.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter

GENRE_DECAY_PER_MIN = 0.0   # placeholder for recency weighting; off by default


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class SessionGraph:
    def __init__(self, state: dict | None = None, path: str | None = None):
        self.path = path
        self.state = state if state is not None else self._empty()

    @staticmethod
    def _empty() -> dict:
        return {
            "active": False,
            "session_id": None,
            "start_time": None,
            "start_ts": None,
            "detected_mood": None,
            "interactions": [],
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start_session(self) -> str:
        sid = str(uuid.uuid4())
        self.state = {
            "active": True,
            "session_id": sid,
            "start_time": _now_iso(),
            "start_ts": _now_ts(),
            "detected_mood": None,
            "interactions": [],
        }
        self.save()
        return sid

    def is_active(self) -> bool:
        return bool(self.state.get("active"))

    def clear(self) -> None:
        self.state = self._empty()
        self.save()

    # ── Interactions ──────────────────────────────────────────────────────────
    def add_interaction(
        self,
        movie_id: int | str,
        title: str,
        interaction_type: str,        # 'recommended' | 'liked' | 'skipped' | 'disliked'
        weight: float = 1.0,
        genres: list[str] | None = None,
    ) -> None:
        if not self.is_active():
            self.start_session()
        interaction = {
            "movie_id": str(movie_id),
            "title": title,
            "type": interaction_type,
            "weight": float(weight),
            "genres": list(genres or []),
            "timestamp": _now_iso(),
        }
        self.state["interactions"].append(interaction)
        self._detect_mood()
        self.save()

    def _detect_mood(self) -> None:
        counter: Counter = Counter()
        for it in self.state["interactions"]:
            for g in it.get("genres", []):
                counter[g] += 1 if it["type"] in ("recommended", "liked") else -1
        positives = {g: c for g, c in counter.items() if c > 0}
        if positives:
            self.state["detected_mood"] = max(positives.items(), key=lambda x: x[1])[0]

    # ── Tensors for influence functions ───────────────────────────────────────
    def get_edges_for_user(self, user_id: int = 0) -> list[tuple[int, str, float]]:
        """Returns (user_id, movie_tmdb_id_str, weight) tuples."""
        out = []
        sign = {"liked": 1.0, "recommended": 0.5, "skipped": -0.3, "disliked": -1.0}
        for it in self.state["interactions"]:
            w = it.get("weight", 1.0) * sign.get(it["type"], 0.5)
            out.append((user_id, it["movie_id"], w))
        return out

    def get_session_movie_titles(self) -> set[str]:
        return {it["title"].lower() for it in self.state["interactions"] if it.get("title")}

    def get_session_movie_ids(self) -> set[str]:
        return {str(it["movie_id"]) for it in self.state["interactions"] if it.get("movie_id")}

    def summary(self) -> dict:
        mins = 0
        if self.state.get("start_ts"):
            mins = max(0, int((_now_ts() - self.state["start_ts"]) / 60))
        liked = sum(1 for i in self.state["interactions"] if i["type"] == "liked")
        return {
            "movie_count": len(self.state["interactions"]),
            "liked_count": liked,
            "dominant_mood": self.state.get("detected_mood"),
            "duration_minutes": mins,
            "session_id": self.state.get("session_id"),
            "start_time": self.state.get("start_time"),
        }

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
    def load(cls, path: str) -> "SessionGraph":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    return cls(state=data, path=path)
            except (json.JSONDecodeError, ValueError):
                pass
        sg = cls(state=cls._empty(), path=path)
        sg.save()
        return sg

    # ── Viz payload ───────────────────────────────────────────────────────────
    def to_viz_payload(self) -> dict:
        nodes = []
        edges = []
        if not self.is_active():
            return {"nodes": nodes, "edges": edges, "active": False, "mood": None}

        def nid(t: str) -> str:
            return f"session__{t[:30].replace(' ', '_').replace('/', '_')}"

        seen = set()
        for it in self.state["interactions"]:
            t = it["title"]
            n = nid(t)
            if n in seen:
                continue
            seen.add(n)
            nodes.append({
                "id": n, "label": t, "type": "movie_session",
                "weight": float(it.get("weight", 1.0)),
                "interaction": it["type"],
            })
            edges.append({
                "source": "user", "target": n,
                "weight": float(it.get("weight", 1.0)),
                "type": "session_interaction",
            })
        return {
            "nodes": nodes,
            "edges": edges,
            "active": True,
            "mood": self.state.get("detected_mood"),
        }
