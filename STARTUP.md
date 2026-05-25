# MoodLens — Startup Guide

Two-tier machine unlearning movie recommender. Backend = FastAPI + LightGCN +
GNNDelete + influence functions. Frontend = Next.js 14.

This guide assumes a fresh clone with the TMDB CSVs already present in
`backend/data/` (`movies_metadata.csv`, `credits.csv`, `ratings.csv`).

---

## 0 · Prerequisites

| Tool        | Version  | Notes                                                              |
| ----------- | -------- | ------------------------------------------------------------------ |
| Python      | 3.10+    | Tested with 3.13                                                   |
| Node.js     | 18+      | For the Next.js frontend                                           |
| Disk        | ~1.5 GB  | TMDB CSVs (~ratings.csv is 700 MB) + embedding cache               |
| RAM         | ≥ 8 GB   | LightGCN training peaks around 3 GB                                |
| Gemini key  | required | https://aistudio.google.com/apikey → set `GEMINI_API_KEY` in `.env` |

---

## 1 · Backend — first-time setup

```bash
cd backend

# (Optional but recommended) create a venv
python3 -m venv .venv
source .venv/bin/activate

# Install Python deps
pip install -r requirements.txt
```

### 1.1 · Environment variables

Edit `backend/.env`:

```bash
# Primary LLM: Google Gemini (via the Groq-compatible shim in llm_client.py)
GEMINI_API_KEY=AIza…your_key…
GEMINI_MODEL=gemini-3.5-flash
GEMINI_TEMPERATURE=0.8
GEMINI_MAX_TOKENS=1024

# Legacy Groq config — only read if Gemini is unset (kept for fallback)
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.8
GROQ_MAX_TOKENS=1024

METADATA_CSV=data/movies_metadata.csv
CREDITS_CSV=data/credits.csv
RATINGS_CSV=data/ratings.csv

EMBEDDING_BACKEND=local
EMBEDDINGS_CACHE=embeddings_cache.npy

# Two-tier paths + LightGCN
STATE_FILE=user_state.json
SESSION_FILE=session_state.json
LIGHTGCN_CHECKPOINT=models/checkpoints/lightgcn_best.pt
LIGHTGCN_ALPHA=0.6
GNND_STEPS=50
GNND_LR=0.001

TOP_N_RESULTS=5
```

### 1.2 · Train LightGCN once

The recommender works without the checkpoint (falls back to Bayesian scoring),
but you need it for **both** unlearning tiers.

```bash
# From backend/ ─────────────────────────────────────────────────────────
# QUICK SMOKE TEST (2 epochs, 5k movies, 50 users — ~20 seconds)
python models/train_lightgcn.py --quick

# FULL TRAINING (100 epochs, full dataset; budget 15-30 min on CPU)
python models/train_lightgcn.py

# OR: subsample to N most-popular movies for a faster but still-real run
python models/train_lightgcn.py --max-movies 15000 --users 200
```

This writes:

- `backend/models/checkpoints/lightgcn_best.pt`

It contains: model state, edge_index, user/item/genre maps, and the snapshot of
initial embeddings (needed by influence functions as a reference point).

### 1.3 · Run the backend server

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

First boot will build the plot-embedding index (`embeddings_cache.npy`) if it
isn't already there — that takes 30-60 s and is cached for every future run.

If the LightGCN checkpoint is missing, the server logs:

```
[StateManager] LightGCN checkpoint not found. Falling back to Bayesian scoring.
```

— the server still runs. Unlearning endpoints return 503 until you train.

---

## 2 · Frontend

```bash
cd frontend
npm install      # (or: bun install / pnpm install / yarn)
npm run dev
```

Open <http://localhost:3000> for the landing, <http://localhost:3000/chat> for
the demo, <http://localhost:3000/architecture> for the system description.

The frontend assumes the backend lives at `http://localhost:8000` (override
with `BACKEND_URL` env var in `.env.local`).

---

## 3 · Verify the install

Run the backend test suite (no app boot required):

```bash
cd backend

# Unit tests — toy graph, no real data
python tests/test_lightgcn.py
python tests/test_graphs.py
python tests/test_intent.py

# End-to-end integration (needs the trained checkpoint)
python tests/test_state_manager.py
```

You should see `✓` for every line and `All …  tests passed.` at the bottom.

### 3.1 · Evaluation suite (CLI, visual)

Two runners are available — pick the one that matches what you're doing.

#### A. Rich CLI runner — recommended for demos & reports

The Rich runner is the headline evaluation experience: paneled tables,
color-coded thresholds, sparkline embedding-drift fingerprints, per-tier
verdicts, and a live progress bar. It reports **every** metric we measure
(Tier 1: 12 metrics, Tier 2: 9 metrics).

```bash
# Install the visualisation deps once (rich + tqdm are in requirements.txt,
# but a one-liner for partial installs):
pip install rich tqdm

# Quick eval (~3 s)  — small samples, smoke-tests every metric
python evaluation/cli.py --quick

# Full eval (~30-60 s) — bigger samples, the numbers to put in the report
python evaluation/cli.py

# Other useful flags
python evaluation/cli.py --no-tier2          # Tier 1 only
python evaluation/cli.py --no-tier1          # Tier 2 only
python evaluation/cli.py --json-only         # CI/headless, skip UI
python evaluation/cli.py --seed 42           # reproducibility
python evaluation/cli.py --ckpt path/to.pt   # alternate checkpoint
```

Output:

- A live progress bar while each tier evaluates.
- A coloured **Setup** panel with checkpoint stats (size, users, movies,
  genres, edges, embed dim, layers).
- A **TIER 1 — Permanent Unlearning (GNNDelete)** table:
  - Forget cosine distance ↑
  - Embedding drift L2 norm ↑
  - Retain top-20 overlap ↑
  - Recall@20 / Precision@10 / NDCG@20 / Hit-rate@20 / MRR (retain)
  - Membership-inference score ↓
  - Forget leakage in top-20 ↓
  - Catalogue coverage ↑
  - Intra-list diversity ↑
  - Plus per-movie and per-user sparkline distributions.
- A **TIER 2 — Session Unlearning (Influence Functions)** table:
  - Embedding reversion score ↑
  - Cosine(before, after) ↓ / Cosine(before, mid) ↑ / Cosine(mid, after) ↑
  - Embedding drift L2 ↓
  - Pre↔post top-20 overlap ↑ / Mid↔post top-20 overlap ↓
  - Pre↔post NDCG@20 ↑
  - Kendall-tau distance ↓
- An **Embedding drift fingerprint** panel: 4 sparklines (before / commit /
  erase / |after − before|) over the user-embedding vector.
- A **Final Verdict** panel summarising pass/fail counts per tier.
- `evaluation/metrics.json` is written for the project report.

Each row carries a green ✓ PASS or red ✗ MISS verdict against a sensible
threshold. The arrow next to the metric name shows the desired direction
(↑ = higher is better, ↓ = lower is better).

#### B. Plain runner — legacy / scripting

The original lean runner is still available and writes the same JSON:

```bash
python evaluation/evaluate_unlearning.py
```

Both runners produce `evaluation/metrics.json` — the Rich runner's payload
is a superset with extra fields (per-row pass/fail lists, drift vectors).

---

## 4 · Demo flow

1. Open `/chat`.
2. Ask "recommend some horror movies" → you'll see 5 horror recs and the
   session badge turns amber (mood = Horror).
3. Type "I liked the second one" → that movie goes into your permanent likes,
   and its genres get a +0.15 boost.
4. Type "Block horror forever and never show me movies before 1990." → the
   server fires GNNDelete (Tier 1). You'll see:
   - `unlearn` SSE event in the chat stream
   - red blocked nodes in the live graph
   - the **Unlearning Panel** populates with cosine drift, edges removed, etc.
   - permanent unlearning history accrues a new point in the drift chart.
5. Click the **New Mood** button → modal opens. Choose "Forget this mood"
   to fire Tier 2 influence-function erasure. You'll see:
   - the session badge resets
   - a new amber point on the drift chart
   - the Unlearning Panel updates to show Tier 2 metrics.
6. Ask "recommend something now" → no pre-1990 movies, no horror, and the
   action mood is gone from your taste profile.

---

## 5 · Endpoint map (for debugging)

```
GET  /health              → status, lightgcn_loaded, session_active
GET  /state               → full preference_graph.state
GET  /graph               → merged viz payload (permanent + session)
GET  /session             → current session details
GET  /embedding-drift     → full drift history (tier 1 + tier 2)
POST /chat   (SSE)        → main conversation stream
POST /new-mood            → trigger Tier 2 (action: discard|commit)
POST /permanent-unlearn   → trigger Tier 1 directly (movie_ids, genres, year_before)
POST /reset               → wipe everything
GET  /greet               → opening greeting
```

---

## 6 · Troubleshooting

| Symptom                                       | Fix                                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `Missing env var: GEMINI_API_KEY`             | Edit `backend/.env`. Get a key at https://aistudio.google.com/apikey.                                              |
| `404 model not found` from Gemini             | `gemini-3.5-flash` may not be live for your key — try `gemini-2.5-flash`, `gemini-2.0-flash`, or `gemini-1.5-flash`. |
| `429` / TPM rate-limit from Gemini            | Free-tier quota hit — wait, or set `GEMINI_MODEL=gemini-2.0-flash` (higher RPM), or upgrade your key.              |
| `LightGCN checkpoint not found`               | Run `python backend/models/train_lightgcn.py --quick` (then full training when ready).                            |
| `/permanent-unlearn` returns 503              | Same as above — checkpoint required.                                                                              |
| `embeddings_cache.npy` rebuilds every boot    | The cache size mismatch happens after dataset changes; delete the stale `.npy` and let it rebuild once.            |
| Chat works but no recs                        | Check `disliked_genres`/`blocked_genres` in `user_state.json` — if everything is blocked, no movies pass the filter. |
| Frontend stuck on "Backend Offline"           | Confirm uvicorn is running on port 8000 and CORS allows `http://localhost:3000`.                                  |
| Recs feel identical to Bayesian-only          | Check the logs for `[StateManager] LightGCN loaded`. If missing, the checkpoint failed to load.                   |
