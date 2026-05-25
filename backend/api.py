"""
api.py — FastAPI server for MoodLens (two-tier machine unlearning).

Run:
    cd backend
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from llm_client import GeminiClient as Groq  # Gemini-backed shim, Groq-compatible surface

from state_manager import StateManager, save_state, load_state
from intent_parser import parse_intent
from scoring_engine import (
    build_movie_db,
    score_and_recommend,
    compute_lightgcn_scores,
)
from embedder import setup_embedder, build_plot_index
from graph.graph_builder import build_viz_payload

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))


def _path(env_key, default):
    return os.path.join(ROOT, os.getenv(env_key, default))


METADATA_CSV      = _path("METADATA_CSV",     "data/movies_metadata.csv")
CREDITS_CSV       = _path("CREDITS_CSV",      "data/credits.csv")
RATINGS_CSV       = _path("RATINGS_CSV",      "data/ratings.csv")
STATE_FILE        = _path("STATE_FILE",        "user_state.json")
SESSION_FILE      = _path("SESSION_FILE",      "session_state.json")
EMBEDDINGS_CACHE  = _path("EMBEDDINGS_CACHE", "embeddings_cache.npy")
LIGHTGCN_CKPT     = _path("LIGHTGCN_CHECKPOINT", "models/checkpoints/lightgcn_best.pt")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")
GROQ_MODEL        = os.getenv("GEMINI_MODEL", os.getenv("GROQ_MODEL", "gemini-3.5-flash"))
GROQ_TEMPERATURE  = float(os.getenv("GEMINI_TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "0.8")))
GROQ_MAX_TOKENS   = int(os.getenv("GEMINI_MAX_TOKENS", os.getenv("GROQ_MAX_TOKENS", "1024")))
TOP_N             = int(os.getenv("TOP_N_RESULTS", "5"))
LIGHTGCN_ALPHA    = float(os.getenv("LIGHTGCN_ALPHA", "0.6"))

app_state: dict = {}


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Starting MoodLens backend …")

    has_key = any(os.getenv(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    if not has_key:
        raise RuntimeError("Missing env var: set GEMINI_API_KEY (or GOOGLE_API_KEY).")

    app_state["groq_client"] = Groq()

    setup_embedder(
        cache_path=EMBEDDINGS_CACHE,
        backend=EMBEDDING_BACKEND,
        voyage_key=VOYAGE_API_KEY or None,
    )

    print("[API] Loading movie database …")
    app_state["movie_db"] = build_movie_db(METADATA_CSV, CREDITS_CSV, RATINGS_CSV)

    print("[API] Building plot embedding index …")
    app_state["plot_index"] = build_plot_index(app_state["movie_db"])

    print("[API] Initialising two-tier state manager …")
    app_state["manager"] = StateManager(
        state_path=STATE_FILE,
        session_path=SESSION_FILE,
        checkpoint_path=LIGHTGCN_CKPT,
    )

    app_state["conversation"] = []
    print("[API] Ready ✓")
    yield
    print("[API] Shutting down.")


app = FastAPI(title="MoodLens API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ResetRequest(BaseModel):
    confirm: bool = True


class NewMoodRequest(BaseModel):
    action: str  # "discard" | "commit"


class PermanentUnlearnRequest(BaseModel):
    movie_ids: list[str] | None = None
    genres: list[str] | None = None
    year_before: int | None = None


# ── Prompt builders ────────────────────────────────────────────────────────────
def build_system_prompt(state: dict, session_mood: str | None) -> str:
    liked = [m.get("title") if isinstance(m, dict) else str(m) for m in state.get("liked_movies", [])]
    blocked = [m.get("title") if isinstance(m, dict) else str(m) for m in state.get("blocked_movies", [])]
    blocked_g = state.get("blocked_genres", []) or state.get("disliked_genres", [])
    liked_g = state.get("liked_genres", [])
    genres = state.get("genre_weights", {})
    top_g = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
    top_g_str = ", ".join(f"{g} ({w:.2f})" for g, w in top_g if w != 1.0)

    mood_note = (
        f"- Current mood session: {session_mood} mood active (this is temporary)"
        if session_mood else "- Current mood session: none active"
    )

    return f"""You are MoodLens, a conversational movie recommendation assistant powered by:
  (1) LightGCN graph embeddings learned over a TMDB ~45k movie graph
  (2) GNNDelete — TIER 1 permanent erasure of permanently-blocked movies/genres
  (3) Influence functions — TIER 2 session unlearning, triggered by "New Mood"

PROFILE:
- Liked (permanent)  : {liked or 'none yet'}
- Permanently blocked: {blocked or 'none yet'}
- Liked genres       : {liked_g or 'none yet'}
- Blocked genres     : {blocked_g or 'none yet'}
- Top genre weights  : {top_g_str or 'all neutral'}
{mood_note}

RULES:
1. Numbered list, 1 line per movie, brief reason.
2. NEVER recommend permanently-blocked movies or genres above.
3. 2–3 sentence intro then the list. Warm but concise.
4. Only mention titles from the SYSTEM CONTEXT block.
5. If permanent unlearning fired this turn, acknowledge it explicitly:
   "Got it — I've permanently removed those from your graph (LightGCN embedding shifted)."
6. If the user is in a mood session, gently distinguish session likes from permanent preferences.
"""


def format_reco_context(recs: list, intent: dict) -> str:
    if not recs:
        return "No movies matched. Tell user politely and ask for different criteria."
    parts = []
    if intent.get("directors_requested"): parts.append("director(s): " + ", ".join(intent["directors_requested"]))
    if intent.get("actors_requested"):    parts.append("actor(s): "    + ", ".join(intent["actors_requested"]))
    if intent.get("genres_requested"):    parts.append("genre(s): "    + ", ".join(intent["genres_requested"]))
    if intent.get("plot_description"):    parts.append("plot: "        + intent["plot_description"])
    if intent.get("plot_keywords"):       parts.append("keywords: "    + ", ".join(intent["plot_keywords"]))
    criteria = "; ".join(parts) if parts else "general preference"
    lines = [f"Scored results for [{criteria}]:"]
    for i, (title, score) in enumerate(recs, 1):
        lines.append(f"  {i}. {title}  (score: {score:.2f})")
    lines.append("\nPresent ALL of them as a numbered list. Do not skip or invent titles.")
    return "\n".join(lines)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    mgr: StateManager = app_state.get("manager")
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "lightgcn_loaded": mgr.lightgcn is not None if mgr else False,
        "session_active": mgr.session_graph.is_active() if mgr else False,
    }


@app.get("/state")
async def get_state():
    mgr: StateManager = app_state["manager"]
    return mgr.preference_graph.state


@app.get("/graph")
async def get_graph():
    mgr: StateManager = app_state["manager"]
    return build_viz_payload(
        mgr.preference_graph,
        mgr.session_graph,
        last_recommendations=mgr.preference_graph.state.get("last_recommendation_scores", {}),
    )


@app.post("/reset")
async def reset_state(req: ResetRequest):
    if not req.confirm:
        return {"message": "Reset cancelled"}
    mgr: StateManager = app_state["manager"]
    mgr.reset_all()
    app_state["conversation"] = []
    return {"message": "State reset successfully"}


@app.get("/greet")
async def greet():
    mgr: StateManager = app_state["manager"]
    groq_client = app_state["groq_client"]
    conversation = app_state.get("conversation", [])
    if conversation:
        return {"greeting": "Welcome back! What would you like to watch today?"}

    sys_prompt = build_system_prompt(
        mgr.preference_graph.state,
        mgr.session_graph.state.get("detected_mood"),
    )
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": "Greet me in one friendly sentence and ask what I'm in the mood to watch."},
        ],
        temperature=GROQ_TEMPERATURE,
        max_completion_tokens=150,
        stream=False,
    )
    greeting = resp.choices[0].message.content.strip()
    conversation.append({"role": "user",      "content": "Greet me..."})
    conversation.append({"role": "assistant", "content": greeting})
    app_state["conversation"] = conversation
    return {"greeting": greeting}


@app.get("/session")
async def get_session():
    mgr: StateManager = app_state["manager"]
    sg = mgr.session_graph
    return {
        "active": sg.is_active(),
        "session_id": sg.state.get("session_id"),
        "mood": sg.state.get("detected_mood"),
        "movie_count": len(sg.state.get("interactions", [])),
        "start_time": sg.state.get("start_time"),
        "interactions": sg.state.get("interactions", []),
    }


@app.get("/embedding-drift")
async def embedding_drift():
    mgr: StateManager = app_state["manager"]
    history = mgr.preference_graph.state.get("embedding_drift_history", [])
    permanent = [h for h in history if h.get("tier") == 1]
    session = [h for h in history if h.get("tier") == 2]
    return {
        "has_data": bool(history),
        "permanent_history": permanent,
        "session_history": session,
    }


@app.post("/new-mood")
async def new_mood(req: NewMoodRequest):
    mgr: StateManager = app_state["manager"]
    if not mgr.session_graph.is_active():
        return {
            "success": False,
            "message": "No active mood session.",
            "session_summary": {},
            "embedding_drift": None,
        }
    if req.action not in ("discard", "commit"):
        raise HTTPException(status_code=400, detail="action must be 'discard' or 'commit'")
    result = mgr.trigger_session_end(req.action)
    return {
        "success": True,
        "session_summary": result.get("session_summary", {}),
        "embedding_drift": result.get("metrics", {}),
        "message": (
            "Mood cleared. Taste profile restored."
            if req.action == "discard"
            else "Mood committed into your permanent profile."
        ),
    }


@app.post("/permanent-unlearn")
async def permanent_unlearn(req: PermanentUnlearnRequest):
    mgr: StateManager = app_state["manager"]
    if mgr.lightgcn is None:
        raise HTTPException(status_code=503, detail="LightGCN checkpoint not loaded")

    movie_ids: list[str] = list(req.movie_ids or [])
    movie_db = app_state["movie_db"]

    if req.year_before:
        movie_ids += mgr._movies_before_year(movie_db, int(req.year_before))
    if req.genres:
        for g in req.genres:
            movie_ids += mgr._movies_in_genre(movie_db, g)
            mgr.preference_graph.add_permanent_genre_block(g)
    # Mark each movie as permanently disliked in the profile
    for m_id in movie_ids:
        rows = movie_db[movie_db["movie_id"] == m_id]
        if rows.empty:
            continue
        title = str(rows.iloc[0]["title"])
        genres = list(rows.iloc[0]["genres_list"])
        mgr.preference_graph.add_permanent_dislike(m_id, title, genres)

    movie_ids = list(set(movie_ids))
    if not movie_ids:
        return {
            "success": False,
            "movies_affected": 0,
            "embedding_drift": None,
            "message": "No matching movies.",
        }

    metrics = mgr.trigger_permanent_unlearn(movie_ids)
    return {
        "success": True,
        "movies_affected": metrics.get("movies_affected", len(movie_ids)),
        "embedding_drift": metrics,
        "message": f"GNNDelete erased {metrics.get('movies_affected', 0)} movies from the graph.",
    }


# ── Chat (upgraded SSE) ───────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    movie_db = app_state["movie_db"]
    plot_index = app_state["plot_index"]
    groq_client = app_state["groq_client"]
    conversation = app_state["conversation"]
    mgr: StateManager = app_state["manager"]

    # Ensure session is active when any chat happens (mood session begins implicitly)
    if not mgr.session_graph.is_active():
        mgr.session_graph.start_session()

    async def generate():
        intent = parse_intent(
            user_input=user_input,
            movie_db=movie_db,
            groq_client=groq_client,
            groq_model=GROQ_MODEL,
            last_recs=mgr.preference_graph.state.get("last_recommendations", []),
            recent_history=conversation[-6:],
            user_state=mgr.preference_graph.state,
        )

        proc = mgr.process_intent(intent, movie_db)

        # Stream unlearn event immediately if Tier 1 fired this turn
        if proc.get("unlearning_triggered"):
            yield (
                "data: " + json.dumps({
                    "type": "unlearn",
                    "tier": proc["unlearning_tier"],
                    "metrics": proc["metrics"],
                }) + "\n\n"
            )

        # ── Build effective intent for scoring ─────────────────────────────────
        has_any_filter = any([
            intent.get("genres_requested"),
            intent.get("actors_requested"),
            intent.get("directors_requested"),
            intent.get("plot_description"),
            intent.get("plot_keywords"),
        ])
        pure_sentiment = (
            proc["state_changed"]
            and not has_any_filter
            and not intent.get("wants_different")
            and not proc.get("unlearning_triggered")
        )

        if intent.get("genres_requested"):
            mgr.preference_graph.state["last_requested_genres"] = intent["genres_requested"]
            mgr.preference_graph.save()

        effective_intent = dict(intent)
        if intent.get("wants_different") and not has_any_filter:
            for k in ("genres_requested", "actors_requested", "directors_requested", "plot_keywords"):
                effective_intent[k] = []
        elif not has_any_filter and not pure_sentiment:
            last_genres = mgr.preference_graph.state.get("last_requested_genres", [])
            if last_genres:
                effective_intent["genres_requested"] = last_genres

        # ── LightGCN scores for the live user ─────────────────────────────────
        gcn_scores = None
        if mgr.lightgcn is not None:
            gcn_scores = compute_lightgcn_scores(
                model=mgr.lightgcn,
                edge_index=mgr.edge_index,
                user_id=mgr.lightgcn_user_id,
                movie_db=movie_db,
                item_map=mgr.item_map,
            )

        recommendations = score_and_recommend(
            movie_db=movie_db,
            intent=effective_intent,
            state=mgr.preference_graph.state,
            plot_index=plot_index,
            top_n=TOP_N,
            lightgcn_scores=gcn_scores,
            lightgcn_alpha=LIGHTGCN_ALPHA if gcn_scores else 0.0,
            preference_graph=mgr.preference_graph,
            session_graph=mgr.session_graph,
        )

        shown_recommendations = bool(recommendations) and not pure_sentiment

        if shown_recommendations:
            mgr.preference_graph.state["last_recommendations"] = [r[0] for r in recommendations]
            mgr.preference_graph.state["last_recommendation_scores"] = {r[0]: r[1] for r in recommendations}
            mgr.preference_graph.save()
            # Log recommended movies into the session graph
            for title, _ in recommendations:
                rows = movie_db[movie_db["title_clean"] == title.lower()]
                if rows.empty:
                    continue
                mgr.session_graph.add_interaction(
                    str(rows.iloc[0]["movie_id"]),
                    title,
                    "recommended",
                    weight=0.5,
                    genres=list(rows.iloc[0]["genres_list"]),
                )

        # ── LLM context ───────────────────────────────────────────────────────
        if pure_sentiment:
            reco_context = (
                "The user just expressed a like/dislike. Acknowledge warmly, "
                "confirm preference noted, invite them to ask for recommendations."
            )
        elif proc.get("unlearning_triggered"):
            metrics = proc["metrics"] or {}
            reco_context = (
                f"PERMANENT UNLEARNING FIRED (Tier 1, GNNDelete). "
                f"Movies erased: {metrics.get('movies_affected', '?')}. "
                f"Cosine drift on forgotten embeddings: {metrics.get('cosine_distance', 0):.4f}. "
                "Acknowledge in one warm sentence. Then list current recommendations below.\n"
                + format_reco_context(recommendations, effective_intent)
            )
        elif intent.get("wants_different"):
            reco_context = (
                "User wants different picks from the last batch.\n"
                + format_reco_context(recommendations, effective_intent)
            )
        elif intent.get("sentiment_last_recs") == "negative" and not recommendations:
            reco_context = (
                "User did not enjoy the last suggestions and nothing matched. "
                "Apologise briefly, ask them to describe what they'd prefer differently."
            )
        else:
            reco_context = format_reco_context(recommendations, effective_intent)

        # Session event for the frontend badge
        yield (
            "data: " + json.dumps({
                "type": "session",
                "active": mgr.session_graph.is_active(),
                "mood": mgr.session_graph.state.get("detected_mood"),
                "movie_count": len(mgr.session_graph.state.get("interactions", [])),
            }) + "\n\n"
        )

        # ── Groq streaming ────────────────────────────────────────────────────
        sys_prompt = build_system_prompt(
            mgr.preference_graph.state,
            mgr.session_graph.state.get("detected_mood"),
        )
        conversation.append({"role": "user", "content": user_input})
        injected = (
            "\n\n[SYSTEM CONTEXT — internal only, do not quote verbatim]\n"
            + reco_context
        )
        llm_messages = (
            [{"role": "system", "content": sys_prompt}]
            + conversation[:-1]
            + [{"role": "user", "content": user_input + injected}]
        )

        full_reply = []
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=llm_messages,
            temperature=GROQ_TEMPERATURE,
            max_completion_tokens=GROQ_MAX_TOKENS,
            top_p=1,
            stream=True,
            stop=None,
        )
        for chunk in completion:
            tok = chunk.choices[0].delta.content or ""
            if tok:
                full_reply.append(tok)
                yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"
            await asyncio.sleep(0)

        conversation.append({"role": "assistant", "content": "".join(full_reply)})

        graph_data = build_viz_payload(
            mgr.preference_graph,
            mgr.session_graph,
            last_recommendations=mgr.preference_graph.state.get("last_recommendation_scores", {}),
        )
        yield f"data: {json.dumps({'type': 'graph', 'data': graph_data})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
