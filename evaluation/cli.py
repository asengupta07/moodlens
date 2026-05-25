"""
evaluation/cli.py — visually rich CLI evaluation runner for MoodLens.

Renders every two-tier unlearning metric in a colourful, paneled CLI report
with live progress, threshold verdicts, sparkline embedding drift charts,
and an exportable JSON snapshot.

Run from project root:
    python evaluation/cli.py
    python evaluation/cli.py --quick          # faster, smaller eval sample
    python evaluation/cli.py --json-only      # just produce metrics.json, no UI
    python evaluation/cli.py --no-tier1
    python evaluation/cli.py --no-tier2

Dependencies:
    pip install rich
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    rank_overlap,
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
    coverage,
    intra_list_diversity,
    embedding_drift_norm,
    kendall_tau_distance,
    membership_inference_score,
    embedding_reversion_score,
)

# ── rich (with graceful fallback) ─────────────────────────────────────────────
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, MofNCompleteColumn,
    )
    from rich.align import Align
    from rich.text import Text
    from rich.rule import Rule
    from rich.columns import Columns
    from rich.box import ROUNDED, HEAVY, DOUBLE
    from rich import print as rprint
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    rprint = print

CKPT = ROOT / "backend" / "models" / "checkpoints" / "lightgcn_best.pt"
OUT = ROOT / "evaluation" / "metrics.json"

SPARK = "▁▂▃▄▅▆▇█"


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════
def sparkline(values: list[float], width: int = 32) -> str:
    """Render a list of floats as a Unicode block sparkline of given width."""
    if not values:
        return ""
    vs = np.asarray(values, dtype=np.float32)
    if len(vs) > width:
        idx = np.linspace(0, len(vs) - 1, width).astype(int)
        vs = vs[idx]
    lo, hi = float(vs.min()), float(vs.max())
    if hi - lo < 1e-9:
        return SPARK[len(SPARK) // 2] * len(vs)
    norm = (vs - lo) / (hi - lo)
    bins = np.clip((norm * (len(SPARK) - 1)).round().astype(int), 0, len(SPARK) - 1)
    return "".join(SPARK[b] for b in bins)


def bar(value: float, vmin: float, vmax: float, width: int = 24) -> str:
    if vmax <= vmin:
        return "·" * width
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    filled = int(round(width * frac))
    return "█" * filled + "·" * (width - filled)


@dataclass
class MetricRow:
    name: str
    value: float
    threshold: str
    direction: str        # "higher" | "lower"
    passed: bool
    fmt: str = "{:.4f}"

    def value_str(self) -> str:
        return self.fmt.format(self.value)


@dataclass
class TierResult:
    title: str
    rows: list[MetricRow] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def verdict(self) -> bool:
        return all(r.passed for r in self.rows)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 — GNNDelete evaluation
# ══════════════════════════════════════════════════════════════════════════════
def _topk_local(model, edge_index, user_id: int, k: int) -> list[int]:
    with torch.no_grad():
        emb = model.propagate(edge_index)
        u = emb[user_id]
        movies = emb[model.num_users: model.num_users + model.num_movies]
        scores = movies @ u
        return torch.topk(scores, k).indices.tolist()


def evaluate_tier1(ckpt: dict, args, progress=None, task_id=None) -> TierResult:
    n_users, n_movies = ckpt["num_users"], ckpt["num_movies"]
    n_genres = ckpt["num_genres"]
    rng = random.Random(args.seed)

    model = LightGCN(
        num_users=n_users, num_movies=n_movies, num_genres=n_genres,
        embedding_dim=ckpt.get("embedding_dim", 64),
        num_layers=ckpt.get("num_layers", 3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    edge_index = ckpt["edge_index"]
    inv_item_map = ckpt["inverse_item_map"]
    item_map = ckpt["item_map"]

    sample_movies = args.tier1_forget
    sample_users = args.tier1_users

    valid_local = list(inv_item_map.keys())
    forget_local = rng.sample(valid_local, k=min(sample_movies, len(valid_local)))
    forget_ids = [inv_item_map[i] for i in forget_local]

    if progress: progress.update(task_id, description="[cyan]Tier 1: snapshot before", advance=1)
    with torch.no_grad():
        emb_pre = model.propagate(edge_index).clone()
    pre_forget = emb_pre[n_users + torch.tensor(forget_local)]

    eval_users = rng.sample(range(1, n_users), k=min(sample_users, n_users - 1))
    pre_topk = {u: _topk_local(model, edge_index, u, 20) for u in eval_users}

    if progress: progress.update(task_id, description="[cyan]Tier 1: running GNNDelete", advance=1)
    deleter = GNNDelete(model, edge_index, item_map)
    new_edges, raw = deleter.unlearn(forget_ids, num_steps=args.gnnd_steps, lr=1e-3)

    if progress: progress.update(task_id, description="[cyan]Tier 1: snapshot after", advance=1)
    with torch.no_grad():
        emb_post = model.propagate(new_edges).clone()
    post_forget = emb_post[n_users + torch.tensor(forget_local)]

    if progress: progress.update(task_id, description="[cyan]Tier 1: computing metrics", advance=1)

    # per-movie forget cosine distances
    forget_cos_per = [
        cosine_distance(p.numpy(), q.numpy())
        for p, q in zip(pre_forget, post_forget)
    ]
    forget_cos = float(np.mean(forget_cos_per))
    drift_norm = float(np.mean([
        embedding_drift_norm(p.numpy(), q.numpy())
        for p, q in zip(pre_forget, post_forget)
    ]))

    post_topk = {u: _topk_local(model, new_edges, u, 20) for u in eval_users}
    overlaps = [rank_overlap(pre_topk[u], post_topk[u], 20) for u in eval_users]
    avg_overlap = float(np.mean(overlaps))

    # build a synthetic "relevant" set per user = original top-5 (pre-unlearn)
    # and measure recall/precision/ndcg AFTER unlearn — i.e. how well the model
    # still surfaces the same preferences when forgotten items are excluded.
    relevant_per_user = {u: pre_topk[u][:5] for u in eval_users}
    recalls = [recall_at_k(post_topk[u], relevant_per_user[u], 20) for u in eval_users]
    precs = [precision_at_k(post_topk[u], relevant_per_user[u], 10) for u in eval_users]
    ndcgs = [ndcg_at_k(post_topk[u], relevant_per_user[u], 20) for u in eval_users]
    hits = [hit_rate_at_k(post_topk[u], relevant_per_user[u], 20) for u in eval_users]
    mrrs = [mean_reciprocal_rank(post_topk[u], relevant_per_user[u]) for u in eval_users]

    forget_set = set(forget_local)
    retain_pool = [i for i in valid_local if i not in forget_set]
    retain_indices = rng.sample(retain_pool, k=min(50, max(1, len(retain_pool))))
    mia = membership_inference_score(model, forget_local, retain_indices, new_edges, n_users)

    # catalogue coverage
    cov = coverage(list(post_topk.values()), n_movies)
    # diversity — use movie embeddings from post graph
    movie_emb_np = emb_post[n_users: n_users + n_movies].numpy()
    diversities = [intra_list_diversity(post_topk[u], movie_emb_np) for u in eval_users]
    avg_div = float(np.mean(diversities))

    # forgotten movies should not appear in any post top-k
    leakage = float(np.mean([
        sum(1 for r in post_topk[u] if r in forget_set) / max(1, len(post_topk[u]))
        for u in eval_users
    ]))

    result = TierResult(title="TIER 1 — Permanent Unlearning (GNNDelete)")
    result.rows = [
        MetricRow("Forget cosine distance",   forget_cos,        "> 0.02",   "higher", forget_cos > 0.02),
        MetricRow("Embedding drift L2 norm",  drift_norm,        "> 0.05",   "higher", drift_norm > 0.05),
        MetricRow("Retain top-20 overlap",    avg_overlap,       "> 0.30",   "higher", avg_overlap > 0.30),
        MetricRow("Recall@20 (retain)",       float(np.mean(recalls)),  "> 0.30", "higher", float(np.mean(recalls)) > 0.30),
        MetricRow("Precision@10 (retain)",    float(np.mean(precs)),    "> 0.15", "higher", float(np.mean(precs)) > 0.15),
        MetricRow("NDCG@20 (retain)",         float(np.mean(ndcgs)),    "> 0.30", "higher", float(np.mean(ndcgs)) > 0.30),
        MetricRow("Hit-rate@20 (retain)",     float(np.mean(hits)),     "> 0.60", "higher", float(np.mean(hits)) > 0.60),
        MetricRow("MRR (retain)",             float(np.mean(mrrs)),     "> 0.15", "higher", float(np.mean(mrrs)) > 0.15),
        MetricRow("Membership-inference",     float(mia),               "< 0.50", "lower",  float(mia) < 0.50),
        MetricRow("Forget leakage in top-20", leakage,                  "< 0.05", "lower",  leakage < 0.05),
        MetricRow("Catalogue coverage",       cov,                      "> 0.01", "higher", cov > 0.01),
        MetricRow("Intra-list diversity",     avg_div,                  "> 0.10", "higher", avg_div > 0.10),
    ]
    result.extras = {
        "forget_cosine_per_movie": forget_cos_per,
        "overlap_per_user": overlaps,
        "recall_per_user": recalls,
        "ndcg_per_user": ndcgs,
        "raw": raw.to_dict(),
        "movies_forgotten": len(forget_local),
        "edges_removed": raw.edges_removed,
        "users_evaluated": len(eval_users),
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Influence functions evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_tier2(ckpt: dict, args, progress=None, task_id=None) -> TierResult:
    n_users, n_movies = ckpt["num_users"], ckpt["num_movies"]
    n_genres = ckpt["num_genres"]
    rng = random.Random(args.seed + 1)

    model = LightGCN(
        num_users=n_users, num_movies=n_movies, num_genres=n_genres,
        embedding_dim=ckpt.get("embedding_dim", 64),
        num_layers=ckpt.get("num_layers", 3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    item_map = ckpt["item_map"]
    inv_item_map = ckpt["inverse_item_map"]

    su = SessionUnlearner(model, ckpt["edge_index"], item_map)
    user_id = 0

    if progress: progress.update(task_id, description="[magenta]Tier 2: pre-session snapshot", advance=1)
    with torch.no_grad():
        emb0 = model.propagate(su.edge_index)
        before_u = emb0[user_id].clone().numpy()
    pre_topk = _topk_local(model, su.edge_index, user_id, 20)

    valid_local = list(inv_item_map.keys())
    session_local = rng.sample(valid_local, k=min(args.tier2_session_size, len(valid_local)))
    session_tmdb = [inv_item_map[i] for i in session_local]
    session_edges = [(user_id, mid, 1.0) for mid in session_tmdb]

    if progress: progress.update(task_id, description="[magenta]Tier 2: committing session", advance=1)
    su.commit_session(session_edges, user_id, num_steps=args.commit_steps, lr=5e-3)
    with torch.no_grad():
        mid_u = model.propagate(su.edge_index)[user_id].clone().numpy()
    mid_topk = _topk_local(model, su.edge_index, user_id, 20)

    if progress: progress.update(task_id, description="[magenta]Tier 2: erasing session", advance=1)
    erase_metrics = su.erase_session(session_edges, user_id, mode="discard")
    with torch.no_grad():
        after_u = model.propagate(su.edge_index)[user_id].clone().numpy()
    post_topk = _topk_local(model, su.edge_index, user_id, 20)

    if progress: progress.update(task_id, description="[magenta]Tier 2: computing metrics", advance=1)

    rev = embedding_reversion_score(before_u, after_u, before_u)
    pre_post_overlap = rank_overlap(pre_topk, post_topk, 20)
    mid_post_overlap = rank_overlap(mid_topk, post_topk, 20)
    cos_before_after = cosine_distance(before_u, after_u)
    cos_before_mid = cosine_distance(before_u, mid_u)
    cos_mid_after = cosine_distance(mid_u, after_u)
    drift = embedding_drift_norm(before_u, after_u)
    kendall = kendall_tau_distance(pre_topk, post_topk, 20)
    pre_post_ndcg = ndcg_at_k(post_topk, pre_topk, 20)

    result = TierResult(title="TIER 2 — Session Unlearning (Influence Functions)")
    result.rows = [
        MetricRow("Embedding reversion score", rev,             "> 0.80", "higher", rev > 0.80),
        MetricRow("Cosine(before, after) ",    cos_before_after,"< 0.05", "lower",  cos_before_after < 0.05),
        MetricRow("Cosine(before, mid)   ",    cos_before_mid,  "> 0.00", "higher", cos_before_mid > 0.0),
        MetricRow("Cosine(mid, after)    ",    cos_mid_after,   "> 0.00", "higher", cos_mid_after > 0.0),
        MetricRow("Embedding drift L2",        drift,           "< 0.20", "lower",  drift < 0.20),
        MetricRow("Pre↔post top-20 overlap",   pre_post_overlap,"> 0.50", "higher", pre_post_overlap > 0.50),
        MetricRow("Mid↔post top-20 overlap",   mid_post_overlap,"< 0.80", "lower",  mid_post_overlap < 0.80),
        MetricRow("Pre↔post NDCG@20",          pre_post_ndcg,   "> 0.50", "higher", pre_post_ndcg > 0.50),
        MetricRow("Kendall-tau distance",      kendall,         "< 0.50", "lower",  kendall < 0.50),
    ]
    result.extras = {
        "before_vector": before_u.tolist(),
        "mid_vector": mid_u.tolist(),
        "after_vector": after_u.tolist(),
        "session_size": len(session_edges),
        "raw": erase_metrics.to_dict(),
        "pre_topk": pre_topk,
        "mid_topk": mid_topk,
        "post_topk": post_topk,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Rich rendering
# ══════════════════════════════════════════════════════════════════════════════
def render_banner(console) -> None:
    title = Text("MoodLens · Two-Tier Unlearning Evaluation Suite", style="bold cyan")
    sub = Text("LightGCN  +  GNNDelete (Tier 1)  +  Influence Functions (Tier 2)",
               style="dim italic")
    inner = Group(Align.center(title), Align.center(sub))
    console.print(Panel(inner, box=DOUBLE, border_style="cyan", padding=(1, 4)))


def render_tier_table(result: TierResult, color: str) -> Panel:
    table = Table(box=ROUNDED, expand=True, border_style=color, pad_edge=False)
    table.add_column("Metric", style=f"bold {color}", no_wrap=True, ratio=3)
    table.add_column("Value", justify="right", no_wrap=True, min_width=9)
    table.add_column("Bar", no_wrap=True, min_width=22)
    table.add_column("Target", justify="right", style="dim", no_wrap=True, min_width=8)
    table.add_column("Verdict", justify="center", no_wrap=True, min_width=8)

    for row in result.rows:
        # bar scaled to 0..1 (clamped) for at-a-glance visual
        clamp = max(0.0, min(1.0, row.value if row.value <= 1.0 else row.value / max(1.0, row.value + 1)))
        bar_str = bar(clamp, 0.0, 1.0, 20)
        verdict = "[green]✓ PASS[/]" if row.passed else "[red]✗ MISS[/]"
        val_color = "green" if row.passed else "yellow"
        arrow = "↑" if row.direction == "higher" else "↓"
        table.add_row(
            f"{row.name} {arrow}",
            f"[{val_color}]{row.value_str()}[/]",
            f"[{color}]{bar_str}[/]",
            row.threshold,
            verdict,
        )

    sub_extras = []
    if "forget_cosine_per_movie" in result.extras:
        spark = sparkline(result.extras["forget_cosine_per_movie"], 48)
        sub_extras.append(Text.from_markup(
            f"[dim]per-movie forget cosine drift[/] [bold {color}]{spark}[/]"
        ))
    if "overlap_per_user" in result.extras:
        spark = sparkline(result.extras["overlap_per_user"], 48)
        sub_extras.append(Text.from_markup(
            f"[dim]per-user retain overlap     [/] [bold {color}]{spark}[/]"
        ))

    body: list = [table]
    if sub_extras:
        body.append(Rule(style=f"dim {color}"))
        body.extend(sub_extras)

    return Panel(
        Group(*body),
        title=f"[bold {color}]{result.title}[/]",
        border_style=color,
        box=HEAVY,
        padding=(1, 2),
    )


def render_tier2_vectors(result: TierResult, color: str = "magenta") -> Panel:
    b = result.extras["before_vector"]
    m = result.extras["mid_vector"]
    a = result.extras["after_vector"]
    delta = [abs(x - y) for x, y in zip(b, a)]
    width = min(64, len(b))
    body = Group(
        Text.from_markup(f"[dim]user embedding · before session [/] [bold {color}]{sparkline(b, width)}[/]"),
        Text.from_markup(f"[dim]user embedding · after  commit  [/] [bold {color}]{sparkline(m, width)}[/]"),
        Text.from_markup(f"[dim]user embedding · after  erase   [/] [bold {color}]{sparkline(a, width)}[/]"),
        Text.from_markup(f"[dim]|after - before| (reversion err) [/] [bold red]{sparkline(delta, width)}[/]"),
    )
    return Panel(
        body,
        title=f"[bold {color}]Embedding drift fingerprint[/]",
        border_style=color, box=ROUNDED, padding=(1, 2),
    )


def render_summary(tier1: TierResult | None, tier2: TierResult | None) -> Panel:
    rows = []
    if tier1:
        v = tier1.verdict()
        rows.append(f"  Tier 1 (Permanent / GNNDelete)        : "
                    f"{'[bold green]✓ PASS[/]' if v else '[bold red]✗ NEEDS REVIEW[/]'}")
        passed = sum(1 for r in tier1.rows if r.passed)
        rows.append(f"  → {passed}/{len(tier1.rows)} metrics within threshold")
    if tier2:
        v = tier2.verdict()
        rows.append(f"  Tier 2 (Session / Influence functions) : "
                    f"{'[bold green]✓ PASS[/]' if v else '[bold red]✗ NEEDS REVIEW[/]'}")
        passed = sum(1 for r in tier2.rows if r.passed)
        rows.append(f"  → {passed}/{len(tier2.rows)} metrics within threshold")
    rows.append("")
    rows.append(f"  [dim]metrics.json written to[/] [cyan]{OUT.relative_to(ROOT)}[/]")
    return Panel(
        Text.from_markup("\n".join(rows)),
        title="[bold]Final Verdict[/]",
        border_style="green" if all((t is None or t.verdict()) for t in (tier1, tier2)) else "yellow",
        box=DOUBLE, padding=(1, 4),
    )


def render_plain(tier1: TierResult | None, tier2: TierResult | None) -> None:
    """Fallback render when rich isn't installed."""
    for t in (tier1, tier2):
        if t is None:
            continue
        print("\n" + "═" * 76)
        print(f"  {t.title}")
        print("═" * 76)
        for r in t.rows:
            verdict = "PASS" if r.passed else "MISS"
            print(f"  {r.name:34s} {r.value_str():>10s}  target {r.threshold:>10s}  [{verdict}]")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="MoodLens rich CLI evaluation")
    p.add_argument("--quick", action="store_true", help="Smaller sample sizes for fast eval")
    p.add_argument("--no-tier1", action="store_true")
    p.add_argument("--no-tier2", action="store_true")
    p.add_argument("--json-only", action="store_true", help="Skip rich UI, just write JSON")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=CKPT)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    if args.quick:
        args.tier1_forget = 10
        args.tier1_users = 20
        args.gnnd_steps = 10
        args.tier2_session_size = 5
        args.commit_steps = 5
    else:
        args.tier1_forget = 30
        args.tier1_users = 75
        args.gnnd_steps = 30
        args.tier2_session_size = 10
        args.commit_steps = 10
    return args


def main():
    args = parse_args()
    console = Console() if _HAS_RICH else None

    if not args.ckpt.exists():
        msg = f"Checkpoint missing: {args.ckpt}\n→ run: python backend/models/train_lightgcn.py --quick"
        if console: console.print(Panel(msg, title="[red]error[/]", border_style="red"))
        else: print(msg)
        sys.exit(1)

    if console and not args.json_only:
        render_banner(console)
        console.print()

    t0 = time.time()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_size_mb = args.ckpt.stat().st_size / 1024 / 1024

    if console and not args.json_only:
        console.print(Panel(
            Text.from_markup(
                f"[bold]checkpoint[/] [cyan]{args.ckpt.relative_to(ROOT)}[/] "
                f"([dim]{ckpt_size_mb:.1f} MB[/])\n"
                f"[bold]users[/]  {ckpt['num_users']}    "
                f"[bold]movies[/] {ckpt['num_movies']}    "
                f"[bold]genres[/] {ckpt['num_genres']}    "
                f"[bold]edges[/]  {ckpt['edge_index'].shape[1] // 2}\n"
                f"[bold]embed dim[/] {ckpt.get('embedding_dim', 64)}    "
                f"[bold]layers[/] {ckpt.get('num_layers', 3)}    "
                f"[bold]mode[/] {'quick' if args.quick else 'full'}"
            ),
            title="[bold cyan]Setup[/]", border_style="cyan", box=ROUNDED,
        ))
        console.print()

    tier1, tier2 = None, None

    if console and not args.json_only:
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console, transient=True,
        ) as progress:
            if not args.no_tier1:
                tid = progress.add_task("[cyan]Tier 1: starting", total=4)
                tier1 = evaluate_tier1(ckpt, args, progress=progress, task_id=tid)
                progress.update(tid, description="[bold green]Tier 1: complete", completed=4)
            if not args.no_tier2:
                tid = progress.add_task("[magenta]Tier 2: starting", total=4)
                tier2 = evaluate_tier2(ckpt, args, progress=progress, task_id=tid)
                progress.update(tid, description="[bold green]Tier 2: complete", completed=4)
    else:
        if not args.no_tier1:
            tier1 = evaluate_tier1(ckpt, args)
        if not args.no_tier2:
            tier2 = evaluate_tier2(ckpt, args)

    # write JSON snapshot
    payload: dict[str, Any] = {}
    if tier1:
        payload["tier1"] = {
            **{r.name: r.value for r in tier1.rows},
            "_passed": [r.name for r in tier1.rows if r.passed],
            "_failed": [r.name for r in tier1.rows if not r.passed],
            **{k: v for k, v in tier1.extras.items()
               if k not in ("forget_cosine_per_movie", "overlap_per_user",
                            "recall_per_user", "ndcg_per_user")},
        }
    if tier2:
        payload["tier2"] = {
            **{r.name: r.value for r in tier2.rows},
            "_passed": [r.name for r in tier2.rows if r.passed],
            "_failed": [r.name for r in tier2.rows if not r.passed],
            "session_size": tier2.extras["session_size"],
            "raw": tier2.extras["raw"],
        }
    args.out.write_text(json.dumps(payload, indent=2, default=float))

    if console and not args.json_only:
        console.print()
        if tier1:
            console.print(render_tier_table(tier1, "cyan"))
            console.print()
        if tier2:
            console.print(render_tier_table(tier2, "magenta"))
            console.print()
            console.print(render_tier2_vectors(tier2))
            console.print()
        console.print(render_summary(tier1, tier2))
        elapsed = time.time() - t0
        console.print(f"\n  [dim]Total elapsed: {elapsed:.1f}s[/]\n")
    elif args.json_only:
        print(f"metrics.json written → {args.out}")
    else:
        render_plain(tier1, tier2)
        print(f"\nmetrics.json written → {args.out}")


if __name__ == "__main__":
    main()
