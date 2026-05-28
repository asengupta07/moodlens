"""
state_manager.py — Two-tier state coordinator.

Owns:
    PreferenceGraph         (permanent, persists user_state.json)
    SessionGraph            (ephemeral, persists session_state.json)
    LightGCN                (loaded from checkpoint if present)
    GNNDelete               (Tier 1 operator, wired to LightGCN)
    SessionUnlearner        (Tier 2 operator, wired to LightGCN)

Backwards-compatible helpers (load_state, save_state, update_liked, ...) are
preserved for the legacy CLI entry point in main.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Legacy helpers (kept so main.py still imports cleanly) ─────────────────────
DEFAULT_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Sci-Fi", "Thriller", "TV Movie", "War", "Western",
]
LIKE_BOOST = 0.15
LIKE_CAP = 1.5
DISLIKE_DECAY = 0.70


def _default_state() -> dict:
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


def initialize_state() -> dict:
    return _default_state()


def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                defaults = _default_state()
                migrated = False
                for k, v in defaults.items():
                    if k not in data:
                        data[k] = v
                        migrated = True
                for genre in DEFAULT_GENRES:
                    if genre not in data["genre_weights"]:
                        data["genre_weights"][genre] = 1.0
                        migrated = True
                _dedupe_list(data, "liked_movies")
                _dedupe_list(data, "disliked_movies")
                _dedupe_list(data, "liked_genres")
                _dedupe_list(data, "disliked_genres")
                _dedupe_list(data, "blocked_genres")
                if migrated:
                    save_state(path, data)
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    state = _default_state()
    save_state(path, state)
    return state


def save_state(path: str, state: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)


def _dedupe_list(state: dict, key: str) -> None:
    seen = set()
    deduped = []
    for item in state.get(key, []):
        if isinstance(item, dict):
            marker = json.dumps(item, sort_keys=True)
        else:
            marker = str(item).strip().lower()
        if not marker:
            continue
        if marker not in seen:
            deduped.append(item)
            seen.add(marker)
    state[key] = deduped


def remember_event(state: dict, event: dict) -> None:
    events = state.setdefault("preference_events", [])
    events.append(event)
    del events[:-100]


def update_liked(state: dict, title: str, genres: list[str]) -> None:
    t = title.strip()
    if not t:
        return
    existing_titles = [m.get("title") if isinstance(m, dict) else str(m)
                       for m in state["liked_movies"]]
    if t not in existing_titles:
        state["liked_movies"].append(t)
    for g in genres:
        boost_genre(state, g)


def update_disliked(state: dict, title: str, genres: list[str]) -> None:
    t = title.strip()
    if t and t not in state["disliked_movies"]:
        state["disliked_movies"].append(t)
    for g in genres:
        decay_genre(state, g)


def boost_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    state["genre_weights"].setdefault(genre, 1.0)
    cur = state["genre_weights"].get(genre, 1.0)
    state["genre_weights"][genre] = round(min(cur + LIKE_BOOST, LIKE_CAP), 4)


def decay_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    state["genre_weights"].setdefault(genre, 1.0)
    cur = state["genre_weights"].get(genre, 1.0)
    state["genre_weights"][genre] = round(cur * DISLIKE_DECAY, 4)


def update_liked_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    if genre not in state["liked_genres"]:
        state["liked_genres"].append(genre)
    if genre in state["disliked_genres"]:
        state["disliked_genres"].remove(genre)
    boost_genre(state, genre)


def update_disliked_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    if genre not in state["disliked_genres"]:
        state["disliked_genres"].append(genre)
    if genre in state["liked_genres"]:
        state["liked_genres"].remove(genre)
    decay_genre(state, genre)


# ══════════════════════════════════════════════════════════════════════════════
#  TWO-TIER MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class StateManager:
    """High-level coordinator used by the FastAPI app."""

    def __init__(
        self,
        state_path: str,
        session_path: str,
        checkpoint_path: str | None = None,
    ):
        from graph.preference_graph import PreferenceGraph
        from graph.session_graph import SessionGraph

        self.state_path = state_path
        self.session_path = session_path
        self.checkpoint_path = checkpoint_path

        self.preference_graph = PreferenceGraph.load(state_path)
        self.session_graph = SessionGraph.load(session_path)

        self.lightgcn = None
        self.edge_index = None
        self.user_map = None
        self.item_map = None
        self.inverse_item_map = None
        self.genre_map = None
        self.initial_embeddings = None
        self.gnn_delete = None
        self.session_unlearner = None
        self.lightgcn_user_id = self.preference_graph.state.get("lightgcn_user_id", 0)

        self._load_lightgcn()

    # ── Bootstrapping ─────────────────────────────────────────────────────────
    def _load_lightgcn(self) -> None:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            logger.warning(
                f"[StateManager] LightGCN checkpoint not found at {self.checkpoint_path}. "
                f"Falling back to Bayesian scoring."
            )
            return
        try:
            import torch
            from models.lightgcn import LightGCN
            from models.gnn_delete import GNNDelete
            from models.influence import SessionUnlearner

            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            n_users = ckpt["num_users"]
            n_movies = ckpt["num_movies"]
            n_genres = ckpt["num_genres"]
            self.lightgcn = LightGCN(
                num_users=n_users,
                num_movies=n_movies,
                num_genres=n_genres,
                embedding_dim=ckpt.get("embedding_dim", 64),
                num_layers=ckpt.get("num_layers", 3),
            )
            self.lightgcn.load_state_dict(ckpt["model_state"])
            self.lightgcn.eval()
            self.edge_index = ckpt["edge_index"]
            self.user_map = ckpt["user_map"]
            self.item_map = ckpt["item_map"]
            self.inverse_item_map = ckpt.get(
                "inverse_item_map", {v: k for k, v in self.item_map.items()}
            )
            self.genre_map = ckpt["genre_map"]
            self.initial_embeddings = ckpt.get("initial_embeddings")

            self.gnn_delete = GNNDelete(self.lightgcn, self.edge_index, self.item_map)
            self.session_unlearner = SessionUnlearner(
                self.lightgcn, self.edge_index, self.item_map
            )
            logger.info(
                f"[StateManager] LightGCN loaded "
                f"({n_users}u/{n_movies}m/{n_genres}g, dim={self.lightgcn.embedding_dim})."
            )
        except Exception as e:
            logger.exception(f"[StateManager] Failed to load LightGCN checkpoint: {e}")
            self.lightgcn = None

    # ── Intent processing ─────────────────────────────────────────────────────
    def process_intent(self, intent: dict, movie_db) -> dict:
        """
        Routes intent → graphs + unlearning ops.
        Returns dict of side-effect metadata for the chat reply.
        """
        result = {
            "unlearning_triggered": False,
            "unlearning_tier": None,
            "metrics": None,
            "state_changed": False,
        }

        # ── Permanent Tier 1 unlearning ───────────────────────────────────────
        movie_ids_to_unlearn: list[str] = []

        if intent.get("permanent_year_block"):
            year = int(intent["permanent_year_block"])
            ids = self._movies_before_year(movie_db, year)
            movie_ids_to_unlearn.extend(ids)
            for m_id in ids:
                title_rows = movie_db[movie_db["movie_id"] == m_id]
                title = str(title_rows.iloc[0]["title"]) if not title_rows.empty else ""
                genres = list(title_rows.iloc[0]["genres_list"]) if not title_rows.empty else []
                self.preference_graph.add_permanent_dislike(m_id, title, genres)
            result["state_changed"] = True

        for genre in intent.get("permanent_genre_block", []):
            self.preference_graph.add_permanent_genre_block(genre)
            ids = self._movies_in_genre(movie_db, genre)
            movie_ids_to_unlearn.extend(ids)
            result["state_changed"] = True

        for raw_title in intent.get("permanent_movie_block", []):
            rows = movie_db[movie_db["title_clean"] == str(raw_title).lower()]
            if rows.empty:
                continue
            m_id = str(rows.iloc[0]["movie_id"])
            title = str(rows.iloc[0]["title"])
            genres = list(rows.iloc[0]["genres_list"])
            self.preference_graph.add_permanent_dislike(m_id, title, genres)
            movie_ids_to_unlearn.append(m_id)
            result["state_changed"] = True

        if movie_ids_to_unlearn and self.gnn_delete is not None:
            metrics = self.trigger_permanent_unlearn(list(set(movie_ids_to_unlearn)))
            result.update({
                "unlearning_triggered": True,
                "unlearning_tier": 1,
                "metrics": metrics,
            })

        # ── Likes (permanent profile signal) ──────────────────────────────────
        for raw_title in intent.get("liked_found", []):
            rows = movie_db[movie_db["title_clean"] == str(raw_title).lower()]
            if rows.empty:
                continue
            m_id = str(rows.iloc[0]["movie_id"])
            title = str(rows.iloc[0]["title"])
            genres = list(rows.iloc[0]["genres_list"])
            self.preference_graph.add_like(m_id, title, genres)
            self.session_graph.add_interaction(m_id, title, "liked", weight=1.0, genres=genres)
            result["state_changed"] = True

        for genre in intent.get("liked_genres", []):
            self.preference_graph.add_liked_genre(genre)
            result["state_changed"] = True

        # ── Soft dislikes (session only) ──────────────────────────────────────
        for raw_title in intent.get("disliked_found", []):
            rows = movie_db[movie_db["title_clean"] == str(raw_title).lower()]
            title = str(rows.iloc[0]["title"]) if not rows.empty else str(raw_title)
            genres = list(rows.iloc[0]["genres_list"]) if not rows.empty else []
            self.preference_graph.add_soft_dislike(title, genres)
            if not rows.empty:
                self.session_graph.add_interaction(
                    str(rows.iloc[0]["movie_id"]), title, "disliked", weight=1.0, genres=genres,
                )
            result["state_changed"] = True

        for genre in intent.get("disliked_genres", []):
            if not intent.get("is_permanent"):
                self.preference_graph.add_disliked_genre(genre)
                result["state_changed"] = True

        for genre in intent.get("soft_disliked_genres", []):
            self.preference_graph._decay_genre(genre)
            result["state_changed"] = True

        # ── Unblock genre if user re-requests it ──────────────────────────────
        for genre in intent.get("genres_requested", []):
            if genre in self.preference_graph.state["disliked_genres"]:
                self.preference_graph.state["disliked_genres"].remove(genre)
                cur = self.preference_graph.state["genre_weights"].get(genre, 1.0)
                self.preference_graph.state["genre_weights"][genre] = max(cur, 1.0)
                result["state_changed"] = True

        if result["state_changed"]:
            self.preference_graph.save()
        return result

    # ── Tier 1 trigger ────────────────────────────────────────────────────────
    def trigger_permanent_unlearn(self, movie_ids: list[str]) -> dict:
        if self.gnn_delete is None:
            return {"error": "lightgcn_not_loaded"}
        import os
        new_edges, metrics = self.gnn_delete.unlearn(
            forget_movie_ids=movie_ids,
            num_steps=int(os.getenv("GNND_STEPS", "50")),
            lr=float(os.getenv("GNND_LR", "0.001")),
        )
        self.edge_index = new_edges
        # Save updated checkpoint so unlearning persists across restarts
        self._save_checkpoint()
        m = metrics.to_dict()
        self.preference_graph.record_drift(tier=1, event_type="permanent_unlearn", payload=m)
        self.preference_graph.save()
        return m

    # ── Tier 2 trigger ────────────────────────────────────────────────────────
    def trigger_session_end(self, mode: str) -> dict:
        if self.session_unlearner is None:
            self.session_graph.clear()
            return {"error": "lightgcn_not_loaded", "session_summary": {}}

        summary = self.session_graph.summary()
        edges = self.session_graph.get_edges_for_user(self.lightgcn_user_id)

        if mode == "commit":
            metrics = self.session_unlearner.commit_session(edges, self.lightgcn_user_id)
            for it in self.session_graph.state.get("interactions", []):
                if it.get("type") not in ("liked", "recommended"):
                    continue
                if float(it.get("weight", 0.0)) <= 0:
                    continue
                self.preference_graph.add_like(
                    it.get("movie_id", ""),
                    it.get("title", ""),
                    list(it.get("genres", [])),
                    engagement=float(it.get("weight", 1.0)),
                )
            event = "session_commit"
        else:
            if self.session_unlearner.has_session_snapshot(edges, self.lightgcn_user_id):
                metrics = self.session_unlearner.erase_session(
                    edges, self.lightgcn_user_id, mode="discard",
                )
            else:
                metrics = self.session_unlearner.discard_active_session(
                    edges, self.lightgcn_user_id,
                )
            event = "session_discard"
        self.edge_index = self.session_unlearner.edge_index
        if self.gnn_delete is not None:
            self.gnn_delete.edge_index = self.edge_index
        self.session_graph.clear()
        m = metrics.to_dict()
        m["mode"] = mode
        m["mood"] = summary.get("dominant_mood")
        self.preference_graph.record_drift(tier=2, event_type=event, payload=m)
        self.preference_graph.save()
        if not getattr(metrics, "non_destructive", False):
            self._save_checkpoint()
        return {"metrics": m, "session_summary": summary}

    # ── Misc helpers ──────────────────────────────────────────────────────────
    def _movies_before_year(self, movie_db, year: int) -> list[str]:
        df = movie_db.copy()
        df["_year"] = (
            df.get("release_date", "")
            .astype(str)
            .str.slice(0, 4)
            .pipe(lambda s: __import__("pandas").to_numeric(s, errors="coerce"))
        )
        ids = df.loc[df["_year"] < year, "movie_id"].astype(str).tolist()
        return ids

    def _movies_in_genre(self, movie_db, genre: str) -> list[str]:
        wanted = genre.strip().lower()
        ids = []
        for _, row in movie_db.iterrows():
            for g in (row.get("genres_list") or []):
                if g.strip().lower() == wanted:
                    ids.append(str(row["movie_id"]))
                    break
        return ids

    def _save_checkpoint(self) -> None:
        if not self.checkpoint_path or self.lightgcn is None:
            return
        try:
            import torch
            from models.lightgcn import EMBEDDING_DIM, NUM_LAYERS
            torch.save({
                "model_state": self.lightgcn.state_dict(),
                "edge_index": self.edge_index,
                "user_map": self.user_map,
                "item_map": self.item_map,
                "inverse_item_map": self.inverse_item_map,
                "genre_map": self.genre_map,
                "initial_embeddings": self.initial_embeddings,
                "num_users": self.lightgcn.num_users,
                "num_movies": self.lightgcn.num_movies,
                "num_genres": self.lightgcn.num_genres,
                "embedding_dim": self.lightgcn.embedding_dim,
                "num_layers": self.lightgcn.num_layers,
            }, self.checkpoint_path)
        except Exception as e:
            logger.warning(f"[StateManager] Failed to persist checkpoint: {e}")

    def get_user_embedding(self):
        if self.lightgcn is None:
            return None
        import torch
        with torch.no_grad():
            emb = self.lightgcn.propagate(self.edge_index)
            return emb[self.lightgcn_user_id]

    def reset_all(self) -> None:
        """Clears both graphs. Reloads original LightGCN checkpoint."""
        # Reset preference graph
        from graph.preference_graph import PreferenceGraph
        from graph.session_graph import SessionGraph
        self.preference_graph = PreferenceGraph(state=PreferenceGraph._default(), path=self.state_path)
        self.preference_graph.save()
        self.session_graph = SessionGraph(state=SessionGraph._empty(), path=self.session_path)
        self.session_graph.save()
        # Reload model from disk if available
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self._load_lightgcn()
