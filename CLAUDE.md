# CLAUDE.md — MoodLens: Two-Tier Machine Unlearning for Movie Recommendations

## READ THIS FIRST

This is a final-year BTech (CSE AIML) project on machine unlearning in recommendation systems. The product is called **MoodLens**. It is a conversational movie recommender that implements two fundamentally different types of unlearning — permanent erasure for identity-level dislikes, and session-scoped forgetting for mood-based watches. Every architectural decision must serve these two goals. Do not simplify, stub, or fake either unlearning tier. Both must be real, mathematically grounded, and demonstrable.

The existing codebase (`/`) has a working FastAPI backend, Next.js frontend, TMDB dataset (~45k movies), Groq LLM reply streaming, and a force-directed graph visualizer. You are upgrading and extending this — not replacing it from scratch unless a specific file is called out for replacement.

---

## Project Identity

**Name:** MoodLens

**Tagline:** *"Your taste is yours. Your mood is temporary. The algorithm should know the difference."*

**Core research argument:** Existing machine unlearning papers (GNNDelete, SISA, influence functions) treat all forgetting as the same operation. This project argues permanent preference erasure and session mood drift are fundamentally different problems requiring different techniques — and builds a working system to demonstrate it.

**Two-tier unlearning:**
- **Tier 1 — Permanent unlearning:** GNNDelete applied to LightGCN embeddings. Used when a user declares a permanent dislike (e.g. "I will never watch movies from before 1990", "block horror forever"). Surgically removes the influence of those nodes from learned embeddings without full retraining.
- **Tier 2 — Session unlearning:** Influence functions applied to a session subgraph. Used when the user ends a mood context via the "New Mood" button. Decays or erases the session graph's effect on permanent embeddings so tonight's action binge doesn't contaminate tomorrow's horror recommendations.

**Session end definition:** Explicit user trigger only — the "New Mood" button. This is a deliberate, academically defensible design choice rooted in user sovereignty and GDPR consent framing. Do NOT implement time-based or app-close-based session endings.

**Dataset:** TMDB ~45k movies from existing `backend/data/` CSVs. Do not replace or re-download the dataset.

**LLM:** Groq API (existing). Keep all Groq streaming logic intact.

**Embeddings:** `all-MiniLM-L6-v2` for semantic plot similarity (existing). Keep this.

---

## Repository Structure (target state)

```
moodlens/
├── CLAUDE.md                          ← this file
├── .env                               ← GROQ_API_KEY, (no other secrets needed)
├── backend/
│   ├── api.py                         ← FastAPI app (upgrade existing)
│   ├── main.py                        ← CLI chat (keep existing)
│   ├── data/
│   │   ├── movies_metadata.csv        ← existing, do not touch
│   │   ├── credits.csv                ← existing, do not touch
│   │   └── ratings.csv                ← existing, do not touch
│   ├── models/
│   │   ├── lightgcn.py                ← NEW: LightGCN model definition
│   │   ├── train_lightgcn.py          ← NEW: training script (run once)
│   │   ├── gnn_delete.py              ← NEW: GNNDelete operator
│   │   ├── influence.py               ← NEW: influence function session unlearning
│   │   └── checkpoints/
│   │       └── lightgcn_best.pt       ← saved after training (gitignore this)
│   ├── graph/
│   │   ├── preference_graph.py        ← NEW: permanent preference graph manager
│   │   ├── session_graph.py           ← NEW: session subgraph manager
│   │   └── graph_builder.py           ← NEW: builds viz payload from both graphs
│   ├── scoring_engine.py              ← upgrade existing: plug in LightGCN scores
│   ├── intent_parser.py               ← keep existing, minor additions
│   ├── state_manager.py               ← upgrade existing: two-tier state
│   ├── embedder.py                    ← keep existing
│   ├── embeddings_cache.npy           ← keep existing (regenerated if missing)
│   ├── user_state.json                ← keep existing schema, extend fields
│   ├── session_state.json             ← NEW: ephemeral session graph state
│   └── requirements.txt               ← update with new deps
├── frontend/
│   ├── app/
│   │   ├── page.tsx                   ← landing (keep, update copy to MoodLens)
│   │   ├── chat/
│   │   │   └── page.tsx               ← main demo (upgrade)
│   │   └── architecture/
│   │       └── page.tsx               ← update to reflect real architecture
│   ├── components/
│   │   ├── ChatWindow.tsx             ← keep
│   │   ├── ChatInput.tsx              ← keep
│   │   ├── MessageBubble.tsx          ← keep
│   │   ├── GnnVisualizer.tsx          ← upgrade: show both graph tiers
│   │   ├── NewMoodButton.tsx          ← NEW: session end trigger with confirm modal
│   │   ├── UnlearningPanel.tsx        ← NEW: shows before/after embedding drift
│   │   ├── SessionBadge.tsx           ← NEW: shows current mood context label
│   │   └── EmbeddingDriftChart.tsx    ← NEW: cosine distance visualization
│   ├── api/
│   │   ├── chat/route.ts              ← keep (proxy to Python)
│   │   ├── greet/route.ts             ← keep
│   │   ├── graph/route.ts             ← keep
│   │   ├── reset/route.ts             ← keep
│   │   ├── new-mood/route.ts          ← NEW: triggers session unlearning
│   │   ├── unlearn/route.ts           ← NEW: triggers Tier 1 permanent unlearning
│   │   └── embedding-drift/route.ts  ← NEW: returns before/after vectors for viz
│   └── package.json
└── evaluation/
    ├── evaluate_unlearning.py         ← NEW: evaluation script
    └── metrics.py                     ← NEW: cosine distance, rank overlap, forget rate
```

---

## Backend: Complete Specification

### Python version and dependencies

Python 3.10+. All new deps go in `requirements.txt`:

```
fastapi
uvicorn
groq
pandas
numpy
torch
torch-geometric
sentence-transformers
scikit-learn
python-dotenv
httpx
```

Install PyTorch Geometric carefully — follow https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html for the correct torch version. Do not guess the install command.

### Running the backend

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

First run: LightGCN checkpoint must already exist (run `python models/train_lightgcn.py` first). If checkpoint is missing, fall back to existing scoring engine with a warning log. Never crash on missing checkpoint.

---

## ML Pipeline: LightGCN

### File: `backend/models/lightgcn.py`

Implement a clean LightGCN from scratch using PyTorch (no PyG trainer — manual forward pass for full control). This is important: we need direct access to the embedding matrices for GNNDelete to operate on them.

```python
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        ...
    
    def forward(self, edge_index):
        # Propagate embeddings through L layers without feature transformation
        # Return final user and item embeddings (mean of all layer outputs)
        ...
    
    def get_user_embedding(self, user_id):
        ...
    
    def get_item_embedding(self, item_id):
        ...
    
    def get_all_embeddings(self):
        # Returns (user_embeddings, item_embeddings) as tensors
        # Used by GNNDelete and influence functions
        ...
```

**Graph construction for TMDB data:**
- Users: synthetic users constructed from ratings.csv (use existing user IDs)
- Items: movie IDs from movies_metadata.csv
- Edges: user-movie interaction edges from ratings.csv (rating >= 3.5 = positive interaction)
- Also add movie-genre edges so genre information propagates through the graph
- Build as a bipartite graph: user nodes + movie nodes, no user-user or movie-movie edges in the base graph

**Embedding dim:** 64. **Layers:** 3. **These are not hyperparameters to be tuned — fix them.**

### File: `backend/models/train_lightgcn.py`

Standalone training script. Run once. Saves checkpoint to `models/checkpoints/lightgcn_best.pt`.

```python
# Training loop:
# - BPR (Bayesian Personalized Ranking) loss
# - Adam optimizer, lr=0.001
# - 100 epochs, batch size 1024
# - Evaluate Recall@20 every 10 epochs on a held-out 20% split
# - Save best checkpoint by Recall@20
# - Print progress every epoch
# - Total runtime on CPU: expect 15-30 minutes for TMDB scale

# After training, also save:
# - edge_index tensor (needed by GNNDelete)
# - user/item ID mappings (needed to look up embeddings by TMDB movie ID)
# - initial embedding matrices (needed by influence functions as reference point)
```

Save format: `torch.save({'model_state': ..., 'edge_index': ..., 'user_map': ..., 'item_map': ..., 'initial_embeddings': ...}, path)`

---

## ML Pipeline: Tier 1 — GNNDelete (Permanent Unlearning)

### File: `backend/models/gnn_delete.py`

Implement GNNDelete based on Cheng et al., ICLR 2023. This is the core academic contribution of Tier 1. Do not fake this with simple embedding zeroing.

**What GNNDelete does:** Given a trained GNN and a set of edges/nodes to forget (the "forget set"), it computes a gradient-based correction to the model parameters such that:
1. The influence of the forget set on the remaining embeddings is minimized
2. The model's performance on the retain set is preserved
3. No full retraining is required

**Implementation:**

```python
class GNNDelete:
    def __init__(self, model: LightGCN, edge_index, device='cpu'):
        self.model = model
        self.edge_index = edge_index  # full graph edge index
        self.device = device
    
    def compute_forget_set(self, movie_ids: list[int]) -> torch.Tensor:
        """
        Given a list of TMDB movie IDs to forget, return the edge indices
        involving those nodes that need to be removed from the graph.
        """
        ...
    
    def unlearn(self, forget_movie_ids: list[int], num_steps: int = 50, lr: float = 1e-3):
        """
        Main unlearning operator. Modifies model parameters in-place.
        
        Algorithm:
        1. Identify forget set edges from movie_ids
        2. Compute gradient of model loss w.r.t. forget set edges
        3. Apply gradient ascent on forget set (increase loss → reduce memorization)
        4. Apply gradient descent on retain set (maintain performance)
        5. Repeat for num_steps
        6. Return updated model + metrics dict
        """
        ...
    
    def verify_unlearning(self, forget_movie_ids: list[int]) -> dict:
        """
        Compute membership inference attack score on forget set.
        Lower score = better unlearning.
        Returns: {'forget_score': float, 'retain_score': float, 'delta': float}
        """
        ...
```

**When Tier 1 is triggered:**
- User says "I hate oldies" / "never show me pre-1990 films" → extract all TMDB movie IDs with release year < 1990, call `gnn_delete.unlearn(those_ids)`
- User says "block horror forever" → extract all horror genre movie IDs, call `gnn_delete.unlearn(those_ids)`
- User says "never show me [specific movie]" → call `gnn_delete.unlearn([that_movie_id])`
- After unlearning, save the updated model checkpoint
- Log the before/after embedding cosine distances for the evaluation panel

**Intent parser must detect these patterns** (see intent_parser.py section).

---

## ML Pipeline: Tier 2 — Influence Functions (Session Unlearning)

### File: `backend/models/influence.py`

Implement influence function-based unlearning for session graph erasure.

**What influence functions do here:** Given the session subgraph (temporary interactions from this mood context), compute an approximation of how much those interactions shifted the user embedding, then apply the inverse update to restore the embedding to its pre-session state.

This is mathematically grounded in Koh & Liang (2017) "Understanding Black-Box Predictions via Influence Functions."

```python
class SessionUnlearner:
    def __init__(self, model: LightGCN, device='cpu'):
        self.model = model
        self.device = device
    
    def compute_session_influence(self, session_edges: list[tuple], user_id: int) -> torch.Tensor:
        """
        Estimate how much each session interaction shifted the user embedding.
        Uses first-order Taylor approximation of the inverse Hessian-vector product.
        
        Args:
            session_edges: list of (user_id, movie_id, weight) tuples from session
            user_id: the user whose embedding to correct
        
        Returns:
            influence_vector: estimated embedding shift to reverse
        """
        # Approximate H^{-1} v using conjugate gradient or LiSSA
        # H = Hessian of training loss w.r.t. user embedding
        # v = gradient of session loss w.r.t. user embedding
        ...
    
    def erase_session(self, session_graph, user_id: int, mode: str = 'discard'):
        """
        Apply session unlearning.
        
        mode='discard': fully erase session influence (user chose "don't keep")
        mode='partial': decay session influence by 70% (softer erasure)
        
        Returns: {'embedding_before': tensor, 'embedding_after': tensor, 'cosine_distance': float}
        """
        ...
    
    def commit_session(self, session_graph, user_id: int):
        """
        Merge session interactions INTO permanent graph (user chose "keep this mood").
        This is the opposite of erasure — call LightGCN fine-tune step.
        """
        ...
```

**When Tier 2 is triggered:**
- User clicks "New Mood" button → modal appears with two options:
  - "Keep in my profile" → call `session_unlearner.commit_session()`
  - "Forget this mood" → call `session_unlearner.erase_session(mode='discard')`
- After erasure, clear `session_state.json`
- Log before/after cosine distance for the drift panel

---

## Graph Management

### File: `backend/graph/preference_graph.py`

Manages the permanent preference graph state. This is the persistent, long-lived representation of the user's identity.

```python
class PreferenceGraph:
    """
    Permanent preference graph. Persists to user_state.json.
    Nodes: user, liked movies, genre nodes
    Edges: user→movie (weight = engagement strength), movie→genre (weight = 1.0)
    Genre nodes carry a weight multiplier (boosted on like, decayed on dislike)
    """
    
    def add_like(self, movie_id: int, engagement: float = 1.0):
        # Add movie to permanent likes, boost genre weights by +0.15, cap at 1.5
        ...
    
    def add_permanent_dislike(self, movie_id: int):
        # Add to hard block list. Triggers GNNDelete for this movie.
        ...
    
    def add_permanent_genre_block(self, genre: str):
        # Add genre to blocked set. Triggers GNNDelete for all movies in genre.
        ...
    
    def get_blocked_movies(self) -> set[int]:
        # Returns all hard-blocked movie IDs (for scoring engine filter)
        ...
    
    def get_genre_weights(self) -> dict[str, float]:
        # Returns per-genre score multipliers
        ...
    
    def to_viz_payload(self) -> dict:
        # Returns graph JSON for GnnVisualizer
        # Nodes: {id, label, type: 'user'|'movie'|'genre', weight}
        # Edges: {source, target, weight, type: 'like'|'block'|'genre'}
        ...
    
    def save(self):
        # Persist to user_state.json
        ...
    
    def load(self):
        # Load from user_state.json
        ...
```

### File: `backend/graph/session_graph.py`

Manages the ephemeral session subgraph. Lives only for one mood context. Stored in `session_state.json` but wiped on "New Mood".

```python
class SessionGraph:
    """
    Session mood graph. Ephemeral — created on first interaction after New Mood,
    destroyed when user presses New Mood again.
    
    Nodes: movies watched/interacted with this session
    Edges: user→movie (weight = engagement, recency-decayed)
    Metadata: session_id (UUID), start_time, mood_label (detected or user-set)
    """
    
    def __init__(self):
        self.session_id = None
        self.edges = []           # list of (movie_id, weight, timestamp)
        self.movie_ids = set()
        self.detected_mood = None # e.g. "action", "thriller", inferred from genres
        self.start_time = None
    
    def start_session(self):
        # Generate new session_id, set start_time, clear edges
        ...
    
    def add_interaction(self, movie_id: int, interaction_type: str, weight: float):
        # interaction_type: 'recommended', 'liked', 'skipped', 'disliked'
        # Infer/update detected_mood from movie genres
        ...
    
    def get_edges_tensor(self) -> list[tuple]:
        # Returns list of (user_id, movie_id, weight) for influence functions
        ...
    
    def is_active(self) -> bool:
        # Returns True if a session is currently live
        ...
    
    def to_viz_payload(self) -> dict:
        # Returns session graph nodes/edges for overlay on GnnVisualizer
        # Session nodes styled in amber, with 'session' type flag
        ...
    
    def clear(self):
        # Wipe session. Called after commit or discard.
        # Save empty state to session_state.json
        ...
    
    def save(self):
        # Persist to session_state.json
        ...
    
    def load(self):
        # Load from session_state.json (resume in-progress session after restart)
        ...
```

### File: `backend/graph/graph_builder.py`

Builds the combined visualization payload from both graphs.

```python
def build_viz_payload(preference_graph: PreferenceGraph, session_graph: SessionGraph) -> dict:
    """
    Merges permanent and session graph into a single payload for GnnVisualizer.
    
    Node types and colors (must match frontend GnnVisualizer):
    - 'user'     → green
    - 'movie_liked'    → blue (permanent likes)
    - 'movie_session'  → amber (session interactions, this mood only)
    - 'movie_blocked'  → red (hard blocked / erased)
    - 'genre'    → purple (with weight shown as edge thickness)
    - 'genre_blocked'  → red outline (blocked genre)
    
    Edge types:
    - permanent like → solid blue line
    - session interaction → dashed amber line
    - genre membership → thin gray line, weight encoded in thickness
    - blocked → red dashed line with X marker
    
    Returns:
    {
        nodes: [...],
        edges: [...],
        session_active: bool,
        session_mood: str | null,
        permanent_count: int,
        session_count: int,
        blocked_count: int
    }
    """
    ...
```

---

## Scoring Engine (Upgrade)

### File: `backend/scoring_engine.py`

Extend the existing scoring engine to incorporate LightGCN embedding scores alongside the existing Bayesian + bonus scoring.

**New scoring formula:**

```
final_score = (
    alpha * lightgcn_score(user, movie)     # GNN embedding dot product similarity
    + (1 - alpha) * existing_score(movie)   # existing Bayesian + bonuses
) * prod(genre_weights) * session_penalty(movie)
```

Where:
- `alpha = 0.6` when LightGCN checkpoint is available, `0.0` as fallback
- `lightgcn_score` = dot product of user embedding and movie embedding, normalized
- `session_penalty(movie)` = 0.3 if movie was shown in current session (avoid repeats within mood), 1.0 otherwise
- All existing hard filters remain: blocked movies → score = 0, blocked genres → score = 0

**New method to add:**

```python
def get_lightgcn_score(self, user_embedding: torch.Tensor, movie_ids: list[int]) -> dict[int, float]:
    """
    Batch compute LightGCN scores for a list of movie IDs against user embedding.
    Returns dict of {movie_id: score}.
    """
    ...

def score_and_recommend(self, intent, preference_graph, session_graph, top_n=10) -> list[dict]:
    """
    Upgrade signature: now takes both graph objects.
    Returns top_n movies as list of {movie_id, title, score, genres, year, overview}.
    """
    ...
```

---

## Intent Parser (Upgrade)

### File: `backend/intent_parser.py`

Add detection for new intent types. Preserve all existing intents.

**New intents to detect:**

```python
# PERMANENT_DISLIKE_YEAR — "never show me movies before X", "I hate oldies", "no classics"
# Extracts: year_threshold (int), e.g. 1990
# Triggers: Tier 1 GNNDelete on all movies with release_year < year_threshold

# PERMANENT_DISLIKE_GENRE — "block horror forever", "never horror again", "I will never watch [genre]"
# Extracts: genre_name (str)
# Triggers: Tier 1 GNNDelete on all movies in that genre

# PERMANENT_DISLIKE_MOVIE — "never show me [movie]", "I hated [movie] remove it completely"
# Extracts: movie_title (str)
# Triggers: Tier 1 GNNDelete on that specific movie

# SESSION_LIKE — "this was great for tonight", "liked it for now"
# Adds to session graph only, not permanent

# SESSION_DISLIKE — "not in the mood for this", "skip this tonight"
# Adds to session graph as negative edge, not permanent block

# NEW_MOOD — should never come from text (only from button), but handle gracefully if it does
```

The intent parser must distinguish between permanent dislikes ("I hate horror forever") and session dislikes ("not feeling horror tonight"). Use Groq LLM for this disambiguation — include it in the system prompt with clear examples.

**Updated Groq system prompt for intent parsing** (add to existing):

```
You are parsing intent for a movie recommender that distinguishes between:
- PERMANENT preferences: things the user will always feel (hate forever, never show me, block completely)
- SESSION preferences: things the user feels right now in this mood (not tonight, not in the mood, skip for now)

When a user says "I hate [X]", default to SESSION unless they use words like "forever", "never", "always", "ever again", "remove completely", "block".

Return JSON with fields: {intent_type, genres, actors, directors, movie_titles, year_threshold, is_permanent, session_only, raw_query}
```

---

## State Manager (Upgrade)

### File: `backend/state_manager.py`

Extend to manage two-tier state. Must coordinate between `PreferenceGraph`, `SessionGraph`, `GNNDelete`, and `SessionUnlearner`.

```python
class StateManager:
    def __init__(self):
        self.preference_graph = PreferenceGraph()
        self.session_graph = SessionGraph()
        self.lightgcn = None          # loaded on startup if checkpoint exists
        self.gnn_delete = None        # initialized with lightgcn
        self.session_unlearner = None # initialized with lightgcn
        self.embedder = Embedder()
        self._load_models()
        self._load_graphs()
    
    def _load_models(self):
        # Try to load LightGCN checkpoint
        # If missing: log warning, set self.lightgcn = None, continue
        ...
    
    def process_intent(self, intent: dict) -> dict:
        """
        Routes intent to correct graph and/or unlearning operation.
        Returns dict with: {updated_preference_graph, updated_session_graph, 
                           unlearning_triggered: bool, unlearning_tier: int|None,
                           embedding_drift: float|None}
        """
        ...
    
    def trigger_permanent_unlearn(self, movie_ids: list[int]) -> dict:
        """
        Calls GNNDelete. Returns metrics for frontend drift panel.
        """
        ...
    
    def trigger_session_end(self, mode: str) -> dict:
        """
        mode: 'discard' or 'commit'
        Calls SessionUnlearner. Returns metrics for frontend drift panel.
        """
        ...
    
    def get_user_embedding(self) -> torch.Tensor | None:
        """Returns current user embedding from LightGCN, or None if not available."""
        ...
    
    def reset_all(self):
        """Full reset: clears both graphs, reloads original LightGCN checkpoint."""
        ...
```

---

## API Endpoints (Complete)

### File: `backend/api.py`

All endpoints. Preserve all existing endpoints exactly. Add new ones.

#### Existing endpoints (keep, do not break):

```
GET  /health          → {status: "ok", lightgcn_loaded: bool, session_active: bool}
GET  /greet           → SSE stream of greeting message
POST /chat            → SSE stream: tokens + graph updates (upgrade to two-tier state)
GET  /graph           → current graph viz payload
GET  /state           → full preference state
POST /reset           → clear everything, reload original checkpoint
```

**Upgrade `/chat` to:**
- Accept session interactions and route to session graph
- Detect permanent vs session intent and route accordingly
- If permanent unlearning triggered mid-chat, include `{type: "unlearn", tier: 1, metrics: {...}}` SSE event
- Always include `{type: "graph", data: {...}}` SSE event with merged graph payload
- Include `{type: "session", active: bool, mood: str|null}` SSE event

#### New endpoints:

```
POST /new-mood
     Body: {action: "discard" | "commit"}
     Response: {
       success: bool,
       session_summary: {movie_count, dominant_mood, duration_minutes},
       embedding_drift: {cosine_distance: float, before_vector: list[float], after_vector: list[float]},
       message: str
     }
     Side effects: triggers SessionUnlearner, clears session_state.json

POST /permanent-unlearn
     Body: {movie_ids: list[int]?, genres: list[str]?, year_before: int?}
     Response: {
       success: bool,
       movies_affected: int,
       embedding_drift: {cosine_distance: float, forget_score: float, retain_score: float},
       message: str
     }
     Side effects: triggers GNNDelete, saves updated checkpoint

GET  /embedding-drift
     Response: {
       has_data: bool,
       permanent_history: list[{timestamp, event_type, cosine_distance}],
       session_history: list[{timestamp, mood, cosine_distance}]
     }
     Used by: EmbeddingDriftChart component

GET  /session
     Response: {
       active: bool,
       session_id: str | null,
       mood: str | null,
       movie_count: int,
       start_time: str | null,
       interactions: list[{movie_id, title, interaction_type, timestamp}]
     }
```

---

## Frontend: Complete Specification

### Stack
Next.js 14 (App Router), React, TypeScript, Tailwind CSS, framer-motion. Existing stack — do not change.

### Pages

#### `/` — Landing (update existing)
Update hero copy to MoodLens branding and two-tier unlearning pitch. Keep glassmorphism aesthetic. Update the mock chat example to show:
1. User asks for horror → gets recs
2. User presses New Mood → transitions to action
3. User says "never show me oldies" → GNNDelete triggered, graph shows red nodes vanishing

#### `/chat` — Main demo (upgrade existing)
Layout: chat on left (60%), sidebar on right (40%).

Sidebar sections (top to bottom):
1. `SessionBadge` — current mood label + session movie count
2. `NewMoodButton` — prominent, with confirm modal
3. `GnnVisualizer` — force-directed graph (upgrade)
4. `UnlearningPanel` — shows last unlearning event + metrics
5. Existing: liked/blocked genre chips, counts

#### `/architecture` — Update to reflect real system
Show the actual two-tier architecture with LightGCN + GNNDelete + influence functions. Remove false claims about PyG training that were in the old version.

### New Components

#### `NewMoodButton.tsx`
```tsx
// Prominent button in sidebar
// On click: opens modal with two options:
//   "Forget this mood" (red) → POST /new-mood {action: "discard"}
//   "Keep in my profile" (blue) → POST /new-mood {action: "commit"}
// Shows session summary in modal: X movies, dominant mood: [action/thriller/etc], Y minutes
// After action: shows embedding drift result as a brief toast notification
// Disabled if no active session (session_active === false)
```

#### `SessionBadge.tsx`
```tsx
// Displayed at top of sidebar
// Shows: amber pill with mood label (e.g. "Action mood") + movie count ("3 movies")
// If no session: gray pill "No active mood"
// Pulses subtly when session is active (CSS animation, reduced-motion aware)
```

#### `UnlearningPanel.tsx`
```tsx
// Shows the last unlearning event
// Fields:
//   - Event type: "Permanent erasure" (Tier 1) or "Session cleared" (Tier 2)
//   - Timestamp
//   - Cosine distance (embedding drift): shown as a colored number (red = high drift)
//   - For Tier 1: movies affected count, forget score
//   - For Tier 2: session duration, mood label
// If no events yet: empty state with explanation text
```

#### `EmbeddingDriftChart.tsx`
```tsx
// Line chart showing cosine distance over time for both tiers
// Two lines: permanent unlearning events (red), session events (amber)
// X axis: event number or timestamp
// Y axis: cosine distance (0 to 1)
// Tooltip on hover showing event details
// Use recharts library (already in project)
// Falls back to empty state if no events yet
```

#### `GnnVisualizer.tsx` (upgrade existing)
Add session graph overlay on top of permanent graph. New node/edge types:
- Session movie nodes: amber circles, dashed border, labeled "[title] (session)"
- Session edges: amber dashed lines
- When "New Mood" is pressed and mode is "discard": animate session nodes fading out with red flash
- When "New Mood" is pressed and mode is "commit": animate session nodes transitioning to blue (merging into permanent)
- Keep all existing node types and colors

Node color legend (must be visible in UI):
- Green = you (user node)
- Blue = permanent likes
- Amber = session interactions (this mood only)
- Purple = genre nodes
- Red = erased / hard blocked

### API Routes (Next.js)

All routes proxy to `http://localhost:8000`. Add:

```typescript
// app/api/new-mood/route.ts
// POST → Python POST /new-mood
// Pass through body and response

// app/api/unlearn/route.ts  
// POST → Python POST /permanent-unlearn
// Pass through body and response

// app/api/embedding-drift/route.ts
// GET → Python GET /embedding-drift
// Pass through response

// app/api/session/route.ts
// GET → Python GET /session
// Pass through response
```

---

## Evaluation

### File: `evaluation/evaluate_unlearning.py`

Standalone evaluation script. Run after training and implementing unlearning to generate numbers for the report.

```python
"""
Evaluation protocol for two-tier unlearning.

Metrics for Tier 1 (GNNDelete / permanent):
1. Forget quality: cosine distance between embedding before and after unlearning
   for the forgotten items. Higher = more forgotten.
2. Retain quality: Recall@20 on non-forgotten items before vs after.
   Should stay within 2% of original.
3. Forget score: membership inference attack success rate on forget set.
   Lower = better unlearning (system less "remembers" those items).

Metrics for Tier 2 (influence functions / session):
1. Embedding reversion: cosine distance of user embedding after session erase
   vs user embedding before session started. Should be close to 0 (good reversion).
2. Recommendation shift: rank overlap of top-20 recs before and after session,
   before and after erase. Erase should restore pre-session rankings.

Run this to produce a metrics.json for use in the project report.
"""
```

### File: `evaluation/metrics.py`

```python
def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float: ...
def recall_at_k(recommended: list, relevant: list, k: int) -> float: ...
def rank_overlap(list1: list, list2: list, k: int) -> float: ...
def membership_inference_score(model, forget_edges, retain_edges) -> float: ...
def embedding_reversion_score(before: np.ndarray, after: np.ndarray, target: np.ndarray) -> float: ...
```

---

## Data Flow: End-to-End (Every Request)

### Normal chat message (no unlearning)

```
1. User types message → POST /api/chat (Next.js) → POST /chat (FastAPI)
2. intent_parser.parse_intent(message) → Groq LLM → intent dict
3. state_manager.process_intent(intent):
   a. If movie mentioned: add to session_graph (interaction edge)
   b. If genre preference: update preference_graph genre weights
   c. If liked movie: add to preference_graph permanent likes
4. scoring_engine.score_and_recommend(intent, preference_graph, session_graph):
   a. Get user embedding from LightGCN (if loaded)
   b. Compute lightgcn_score for all movies
   c. Compute existing Bayesian + bonus scores
   d. Combine with alpha weighting
   e. Apply hard filters (blocked movies, blocked genres)
   f. Apply session_penalty for already-shown movies
   g. Return top-10
5. Groq streams natural language reply (SSE token events)
6. graph_builder.build_viz_payload(preference_graph, session_graph) → graph JSON
7. SSE events sent: tokens, then graph update, then done
8. Frontend: renders tokens, updates GnnVisualizer, updates SessionBadge
```

### Permanent unlearning triggered (Tier 1)

```
1. User says "never show me movies before 1990"
2. intent_parser detects PERMANENT_DISLIKE_YEAR, year_threshold=1990, is_permanent=True
3. state_manager.trigger_permanent_unlearn(movie_ids=[all pre-1990 movie IDs]):
   a. Record before-embedding of user
   b. gnn_delete.unlearn(movie_ids, num_steps=50)
   c. Record after-embedding
   d. Compute cosine_distance(before, after)
   e. Add to embedding drift history
   f. Save updated checkpoint
   g. Update preference_graph blocked list
4. SSE event: {type: "unlearn", tier: 1, metrics: {cosine_distance, movies_affected, forget_score}}
5. SSE event: {type: "graph", data: {...}} — blocked nodes now appear red
6. Groq streams confirmation reply
7. Frontend: UnlearningPanel updates, GnnVisualizer shows red erased nodes
```

### New Mood button pressed (Tier 2)

```
1. User clicks New Mood → modal opens
2. User chooses "Forget this mood" → POST /api/new-mood {action: "discard"}
3. FastAPI POST /new-mood:
   a. Get session_graph edges
   b. Record current user embedding (before)
   c. session_unlearner.erase_session(session_graph, user_id, mode='discard')
   d. Record new user embedding (after)
   e. Compute cosine_distance(before, after)
   f. Add to session drift history
   g. session_graph.clear()
   h. Save session_state.json (empty)
4. Response: {success, session_summary, embedding_drift}
5. Frontend:
   a. Animate session nodes fading/flashing red in GnnVisualizer
   b. Show embedding drift in UnlearningPanel
   c. SessionBadge resets to "No active mood"
   d. Toast: "Mood cleared. Your taste profile is restored."
```

### Session commit (keep this mood)

```
Same as above but action='commit':
3c. session_unlearner.commit_session(session_graph, user_id)
    → fine-tunes LightGCN on session edges → merges into permanent graph
5a. Animate session nodes transitioning blue (amber → blue)
5d. Toast: "Mood added to your profile."
```

---

## user_state.json Schema

```json
{
  "liked_movies": [{"id": 123, "title": "...", "timestamp": "..."}],
  "disliked_movies": [{"id": 456, "title": "...", "timestamp": "..."}],
  "blocked_movies": [{"id": 789, "title": "...", "timestamp": "...", "reason": "permanent"}],
  "disliked_genres": ["Horror"],
  "genre_weights": {"Action": 1.2, "Horror": 0.5, "Drama": 1.0},
  "preference_events": [
    {"type": "like", "movie_id": 123, "timestamp": "..."},
    {"type": "permanent_unlearn", "movie_ids": [...], "timestamp": "...", "cosine_distance": 0.34}
  ],
  "embedding_drift_history": [
    {"tier": 1, "event_type": "year_block", "timestamp": "...", "cosine_distance": 0.34, "movies_affected": 412},
    {"tier": 2, "event_type": "session_discard", "timestamp": "...", "cosine_distance": 0.12, "mood": "action"}
  ],
  "lightgcn_user_id": 0
}
```

## session_state.json Schema

```json
{
  "active": true,
  "session_id": "uuid-here",
  "start_time": "2024-01-01T20:00:00",
  "detected_mood": "action",
  "interactions": [
    {"movie_id": 123, "title": "John Wick", "type": "liked", "weight": 1.0, "timestamp": "..."},
    {"movie_id": 456, "title": "Mad Max", "type": "recommended", "weight": 0.7, "timestamp": "..."}
  ]
}
```

---

## Environment Variables

```bash
# .env
GROQ_API_KEY=your_key_here
LIGHTGCN_CHECKPOINT=backend/models/checkpoints/lightgcn_best.pt
LIGHTGCN_ALPHA=0.6           # weight of LightGCN score vs Bayesian score
GNND_STEPS=50                # GNNDelete optimization steps
GNND_LR=0.001                # GNNDelete learning rate
SESSION_UNLEARN_MODE=discard # default session end mode
```

---

## Startup Sequence

```
1. Load movies_metadata.csv, credits.csv, ratings.csv into DataFrame
2. Try loading LightGCN from checkpoint:
   - Success: initialize GNNDelete and SessionUnlearner with model
   - Fail: log WARNING "LightGCN checkpoint not found. Using Bayesian scoring only."
3. Load embeddings_cache.npy (or generate and save if missing, ~30-60s)
4. Load user_state.json (or create empty if missing)
5. Load session_state.json (or create empty if missing)
6. Build PreferenceGraph and SessionGraph from loaded state
7. Server ready on port 8000
```

---

## Critical Constraints (Never Violate)

1. **GNNDelete must be real** — do not replace it with embedding zeroing or simple filtering. The mathematical correctness of the forget/retain gradient steps is required for the academic claim.

2. **Influence functions must be real** — do not replace with simple embedding reset. Use the gradient-based approximation.

3. **Session graph is always separate from permanent graph** — never write session interactions directly to user_state.json or permanent graph during an active session.

4. **"New Mood" is the only session end trigger** — no time-based, app-close, or automatic session endings.

5. **Hard-blocked movies and genres must score exactly 0** — not just low. They must never appear in recommendations regardless of LightGCN score.

6. **LightGCN checkpoint absence must not crash the server** — graceful degradation to existing Bayesian scoring is required.

7. **All API endpoints from the original codebase must still work** — do not break `/health`, `/greet`, `/chat`, `/graph`, `/state`, `/reset`.

8. **Embedding drift history must persist across server restarts** — stored in user_state.json, not in memory.

9. **Session state must survive server restart** — resume in-progress session from session_state.json.

10. **The GnnVisualizer must show both graph tiers simultaneously** — permanent nodes and session nodes on the same canvas with distinct visual styling.

---

## What "Done" Looks Like

A complete, working demo where:

1. You open `/chat`, ask for horror recommendations, get horror recs, see graph populate.
2. You say "I hate movies before 1990" — GNNDelete runs, graph shows red blocked nodes, future recs have zero pre-1990 films.
3. You press New Mood, see your session summary, choose "Forget this mood" — session nodes animate out, embedding drift is shown in the panel.
4. You now ask for action recommendations — you get action recs, no oldies, graph shows new amber session nodes building up.
5. You run `evaluation/evaluate_unlearning.py` and get a metrics.json with real numbers: forget score, retain Recall@20, cosine distances, rank overlap.
6. You can present this and explain every component because it's all real.

---

## References (cite these in the report)

- He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.
- Cheng et al. (2023). GNNDelete: A General Strategy for Unlearning in Graph Neural Networks. ICLR 2023.
- Koh & Liang (2017). Understanding Black-Box Predictions via Influence Functions. ICML 2017.
- Bourtoule et al. (2021). Machine Unlearning. IEEE S&P 2021. (SISA training — cite for comparison)
- Cao & Yang (2015). Towards Making Systems Forget with Machine Unlearning. IEEE S&P 2015. (original unlearning framing)
- GDPR Article 17 — Right to erasure ("right to be forgotten"). (policy motivation)