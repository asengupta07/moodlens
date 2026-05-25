"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  BarChart3,
  Check,
  CircleAlert,
  Gauge,
  Loader2,
  Play,
  ShieldCheck,
  X,
} from "lucide-react";
import { GlassNav } from "@/components/ui/GlassNav";
import { GlassButton } from "@/components/ui/GlassButton";

type Metrics = Record<string, any>;
type Direction = "higher" | "lower" | "info";
type EvalPhase = "idle" | "running" | "complete";

type ResearchMetric = {
  id: string;
  title: string;
  short: string;
  value: number | null;
  direction: Direction;
  target?: number;
  targetLabel: string;
  explanation: string;
  detail: string;
  accent: string;
  scaleMax: number;
  unit?: string;
};

function readMetric(obj: Record<string, any> | undefined, keys: string[]) {
  if (!obj) return null;
  for (const key of keys) {
    const value = obj[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function fmt(value: number | null, unit = "") {
  if (value === null || Number.isNaN(value)) return "-";
  const out = Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(4);
  return `${out}${unit}`;
}

function isPass(metric: ResearchMetric) {
  if (metric.value === null || metric.direction === "info" || metric.target === undefined) return null;
  return metric.direction === "higher" ? metric.value > metric.target : metric.value < metric.target;
}

function quality(metric: ResearchMetric) {
  if (metric.value === null) return 0;
  if (metric.direction === "info") return Math.min(1, Math.abs(metric.value) / metric.scaleMax);
  if (metric.target === undefined) return Math.min(1, Math.abs(metric.value) / metric.scaleMax);
  if (metric.direction === "higher") {
    if (metric.target === 0) return metric.value > 0 ? 1 : 0;
    return Math.min(1, metric.value / metric.target);
  }
  return Math.max(0, Math.min(1, 1 - metric.value / metric.target));
}

function barWidth(metric: ResearchMetric) {
  if (metric.value === null) return 0;
  return Math.max(0.02, Math.min(1, Math.abs(metric.value) / metric.scaleMax));
}

function useResearchMetrics(metrics: Metrics | null) {
  return useMemo(() => {
    const tier1 = metrics?.tier1 ?? {};
    const tier2 = metrics?.tier2 ?? {};
    const metric = (
      id: string,
      title: string,
      short: string,
      value: number | null,
      direction: Direction,
      target: number,
      targetLabel: string,
      explanation: string,
      detail: string,
      accent: string,
      scaleMax = 1,
    ): ResearchMetric => ({
      id,
      title,
      short,
      value,
      direction,
      target,
      targetLabel,
      explanation,
      detail,
      accent,
      scaleMax,
    });

    const t1: ResearchMetric[] = [
      metric("t1-forget-cos", "Forget cosine distance", "Forget drift", readMetric(tier1, ["Forget cosine distance", "forget_cosine_distance"]), "higher", 0.02, "Target: > 0.0200", "Mean cosine movement of forgotten movie embeddings after GNNDelete.", "Higher means the forget set actually moved away from the trained model memory.", "var(--wine)", 0.2),
      metric("t1-drift-l2", "Embedding drift L2 norm", "L2 drift", readMetric(tier1, ["Embedding drift L2 norm"]), "higher", 0.05, "Target: > 0.0500", "Average L2 magnitude of the forget-set embedding update.", "This catches tiny cosmetic changes that would not count as meaningful erasure.", "var(--wine)", 1),
      metric("t1-overlap", "Retain top-20 overlap", "Retain overlap", readMetric(tier1, ["Retain top-20 overlap", "retain_topk_overlap"]), "higher", 0.3, "Target: > 0.3000", "How much non-forgotten recommendation ranking survives the deletion.", "Higher means permanent erasure did not destroy retained utility.", "var(--blue)", 1),
      metric("t1-recall", "Recall@20 (retain)", "Recall@20", readMetric(tier1, ["Recall@20 (retain)"]), "higher", 0.3, "Target: > 0.3000", "Recall on the retained synthetic relevance set after unlearning.", "The retained profile should still recover relevant items.", "var(--blue)", 1),
      metric("t1-precision", "Precision@10 (retain)", "Precision@10", readMetric(tier1, ["Precision@10 (retain)"]), "higher", 0.15, "Target: > 0.1500", "Precision among the top ten retained recommendations.", "A local quality check for the top of the list.", "var(--blue)", 1),
      metric("t1-ndcg", "NDCG@20 (retain)", "NDCG@20", readMetric(tier1, ["NDCG@20 (retain)"]), "higher", 0.3, "Target: > 0.3000", "Ranking quality for retained items after the forget operation.", "High NDCG means relevant retained items stay near the top.", "var(--blue)", 1),
      metric("t1-hit", "Hit-rate@20 (retain)", "Hit-rate@20", readMetric(tier1, ["Hit-rate@20 (retain)"]), "higher", 0.6, "Target: > 0.6000", "Whether at least one retained relevant item appears in the top 20.", "A coarse but important preserved-utility signal.", "var(--green)", 1),
      metric("t1-mrr", "MRR (retain)", "MRR", readMetric(tier1, ["MRR (retain)"]), "higher", 0.15, "Target: > 0.1500", "Mean reciprocal rank for retained recommendations.", "Higher means the first useful retained item appears earlier.", "var(--green)", 1),
      metric("t1-mia", "Membership-inference", "MIA", readMetric(tier1, ["Membership-inference", "membership_inference_score"]), "lower", 0.5, "Target: < 0.5000", "Attack success proxy on forgotten interactions.", "Lower is better; below 0.5 is better than random attack success.", "var(--purple)", 1),
      metric("t1-leakage", "Forget leakage in top-20", "Leakage", readMetric(tier1, ["Forget leakage in top-20"]), "lower", 0.05, "Target: < 0.0500", "How often forgotten items still leak into post-unlearn top-20 lists.", "This should be near zero for a clean permanent erase.", "var(--wine)", 0.2),
      metric("t1-coverage", "Catalogue coverage", "Coverage", readMetric(tier1, ["Catalogue coverage"]), "higher", 0.01, "Target: > 0.0100", "How much of the catalogue appears across recommendation lists.", "A sanity check that recommendations are not collapsing to a tiny set.", "var(--green)", 0.08),
      metric("t1-diversity", "Intra-list diversity", "Diversity", readMetric(tier1, ["Intra-list diversity"]), "higher", 0.1, "Target: > 0.1000", "Mean pairwise distance within recommendation lists.", "Higher means retained recommendations preserve some variety.", "var(--green)", 0.5),
    ];

    const t2: ResearchMetric[] = [
      metric("t2-reversion", "Embedding reversion score", "Reversion", readMetric(tier2, ["Embedding reversion score", "embedding_reversion_score"]), "higher", 0.8, "Target: > 0.8000", "How close the user embedding returns to the pre-session reference after erase.", "This is the core Tier II objective; currently it reveals whether influence erasure is strong enough.", "var(--amber)", 1),
      metric("t2-before-after", "Cosine(before, after)", "Before/after distance", readMetric(tier2, ["Cosine(before, after) ", "Cosine(before, after)"]), "lower", 0.05, "Target: < 0.0500", "Cosine distance between pre-session and post-erase user embeddings.", "Lower means the user vector returned closer to its original state.", "var(--wine)", 1),
      metric("t2-before-mid", "Cosine(before, mid)", "Session shift", readMetric(tier2, ["Cosine(before, mid)   ", "Cosine(before, mid)"]), "higher", 0.0, "Target: > 0.0000", "Distance introduced by committing the temporary session.", "This should be positive, proving the session had measurable influence.", "var(--amber)", 1),
      metric("t2-mid-after", "Cosine(mid, after)", "Erase movement", readMetric(tier2, ["Cosine(mid, after)    ", "Cosine(mid, after)"]), "higher", 0.0, "Target: > 0.0000", "Distance between the committed-session vector and the erased vector.", "A positive value proves the erase operator changed the vector.", "var(--amber)", 0.1),
      metric("t2-drift-l2", "Embedding drift L2", "L2 error", readMetric(tier2, ["Embedding drift L2"]), "lower", 0.2, "Target: < 0.2000", "L2 residual error between pre-session and post-erase embeddings.", "Lower means less session residue remains in the profile.", "var(--wine)", 2),
      metric("t2-pre-post", "Pre-post top-20 overlap", "Rank restore", readMetric(tier2, ["Pre\u2194post top-20 overlap", "rank_overlap_pre_vs_post_erase"]), "higher", 0.5, "Target: > 0.5000", "Top-20 overlap before the session versus after erasing it.", "Higher means recommendations returned toward the original taste state.", "var(--green)", 1),
      metric("t2-mid-post", "Mid-post top-20 overlap", "Session residue", readMetric(tier2, ["Mid\u2194post top-20 overlap", "rank_overlap_mid_vs_post_erase"]), "lower", 0.8, "Target: < 0.8000", "Top-20 overlap between in-session recommendations and post-erase recommendations.", "Lower is better; high overlap means the temporary mood still dominates.", "var(--wine)", 1),
      metric("t2-ndcg", "Pre-post NDCG@20", "Rank NDCG", readMetric(tier2, ["Pre\u2194post NDCG@20"]), "higher", 0.5, "Target: > 0.5000", "NDCG comparing post-erase ranking against the pre-session ranking.", "Higher means the ordering, not just the set, came back.", "var(--green)", 1),
      metric("t2-kendall", "Kendall-tau distance", "Order error", readMetric(tier2, ["Kendall-tau distance"]), "lower", 0.5, "Target: < 0.5000", "Normalized order disagreement on the shared top-k items.", "Lower means common recommendations kept a similar order.", "var(--purple)", 1),
    ];

    return { tier1: t1, tier2: t2 };
  }, [metrics]);
}

function score(metrics: ResearchMetric[]) {
  const checks = metrics.map(isPass).filter((v): v is boolean => v !== null);
  const passed = checks.filter(Boolean).length;
  return { passed, total: checks.length, failed: checks.length - passed };
}

function verdictLabel(passed: number, total: number) {
  if (total === 0) return "Not measured";
  if (passed === total) return "Pass";
  if (passed >= Math.ceil(total / 2)) return "Mixed";
  return "Needs work";
}

function VerdictCard({
  label,
  title,
  metrics,
  accent,
}: {
  label: string;
  title: string;
  metrics: ResearchMetric[];
  accent: string;
}) {
  const s = score(metrics);
  const verdict = verdictLabel(s.passed, s.total);
  const percent = s.total ? s.passed / s.total : 0;

  return (
    <section className="border border-[var(--rule)] bg-[rgba(28,24,18,0.72)] p-6">
      <div className="flex items-start justify-between gap-5">
        <div>
          <div className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">{label}</div>
          <h2 className="font-display mt-2 text-5xl leading-none" style={{ color: accent }}>
            {title}
          </h2>
        </div>
        <div className="relative h-24 w-24 shrink-0">
          <svg viewBox="0 0 100 100" className="-rotate-90">
            <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(242,237,227,0.10)" strokeWidth="8" />
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke={accent}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${percent * 251.2} 251.2`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="font-display text-3xl">{s.passed}</div>
            <div className="font-space-grotesk text-[9px] uppercase tracking-[0.1em] text-[var(--clay)]">of {s.total}</div>
          </div>
        </div>
      </div>
      <div className="mt-6 flex items-center gap-2">
        {s.failed === 0 ? (
          <Check size={16} className="text-[var(--green)]" />
        ) : (
          <CircleAlert size={16} className="text-[var(--wine)]" />
        )}
        <span className="font-space-grotesk text-[11px] uppercase tracking-[0.14em]" style={{ color: accent }}>
          {verdict}
        </span>
      </div>
      <p className="mt-3 text-sm leading-7 text-[var(--ink-2)]">
        {title === "Tier I"
          ? "Permanent erasure is judged by forget movement, retained recommendation quality, and attack resistance."
          : "Session unlearning is judged by whether the user embedding and rankings return toward the pre-session state."}
      </p>
    </section>
  );
}

function MetricCard({ metric }: { metric: ResearchMetric }) {
  const passed = isPass(metric);
  const fill = barWidth(metric);

  return (
    <article className="border border-[var(--rule)] bg-[rgba(20,16,8,0.58)] p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="font-space-grotesk text-[10px] uppercase tracking-[0.13em] text-[var(--clay)]">{metric.short}</div>
          <h3 className="mt-2 text-lg font-medium leading-snug text-[var(--ink)]">{metric.title}</h3>
        </div>
        <span
          className="inline-flex items-center gap-1 border px-2 py-1 font-space-grotesk text-[9px] uppercase tracking-[0.1em]"
          style={{
            color: passed === false ? "var(--wine)" : passed === true ? "var(--green)" : "var(--clay)",
            borderColor: passed === false ? "rgba(216,88,74,0.42)" : "var(--rule)",
          }}
        >
          {passed === true && <Check size={11} />}
          {passed === false && <X size={11} />}
          {passed === null ? "Info" : passed ? "Pass" : "Flag"}
        </span>
      </div>

      <div className="mt-5 flex items-end justify-between gap-4">
        <div className="font-display text-5xl leading-none" style={{ color: metric.accent }}>
          {fmt(metric.value, metric.unit)}
        </div>
        <div className="text-right font-space-grotesk text-[10px] uppercase tracking-[0.1em] text-[var(--clay)]">
          {metric.targetLabel}
        </div>
      </div>

      <div className="mt-4 h-3 border border-[var(--rule)] bg-[rgba(242,237,227,0.04)] p-[2px]">
        <div className="h-full" style={{ width: `${fill * 100}%`, background: metric.accent }} />
      </div>

      <p className="mt-4 text-sm leading-6 text-[var(--ink-2)]">{metric.explanation}</p>
      <p className="mt-2 text-xs leading-6 text-[var(--clay)]">{metric.detail}</p>
    </article>
  );
}

function ResearchSummary({ tier1, tier2 }: { tier1: ResearchMetric[]; tier2: ResearchMetric[] }) {
  const rows = [...tier1, ...tier2].filter((m) => m.direction !== "info");
  return (
    <section className="border border-[var(--rule)] p-6">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
            <BarChart3 size={13} /> Evaluation overview
          </div>
          <h2 className="font-display mt-2 text-4xl">Research checks at a glance.</h2>
        </div>
        <div className="text-sm leading-6 text-[var(--ink-2)]">
          Green checks are within the chosen report thresholds. Red checks need tuning or more training.
        </div>
      </div>

      <div className="grid gap-4">
        {rows.map((metric) => {
          const passed = isPass(metric);
          return (
            <div key={metric.id} className="grid items-center gap-3 md:grid-cols-[170px_1fr_84px_74px]">
              <div className="text-sm text-[var(--ink-2)]">{metric.short}</div>
              <div className="relative h-8 border border-[var(--rule)] bg-[rgba(242,237,227,0.035)] p-1">
                <div className="h-full" style={{ width: `${quality(metric) * 100}%`, background: metric.accent }} />
              </div>
              <div className="font-mono text-xs text-[var(--ink)]">{fmt(metric.value)}</div>
              <div
                className={`flex items-center gap-1 font-space-grotesk text-[10px] uppercase tracking-[0.1em] ${
                  passed ? "text-[var(--green)]" : "text-[var(--wine)]"
                }`}
              >
                {passed ? <Check size={12} /> : <X size={12} />}
                {passed ? "Pass" : "Flag"}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function CountCard({ label, value, accent, valueText }: { label: string; value?: number | null; accent: string; valueText?: string }) {
  return (
    <div className="border border-[var(--rule)] p-4">
      <div className="font-display text-5xl leading-none" style={{ color: accent }}>
        {valueText ?? fmt(value ?? null)}
      </div>
      <div className="mt-2 text-xs leading-5 text-[var(--clay)]">{label}</div>
    </div>
  );
}

function vectorValues(values: number[], width = 64) {
  if (!values.length) return [];
  const step = Math.max(1, Math.floor(values.length / width));
  const sampled = values.filter((_, index) => index % step === 0).slice(0, width);
  const max = Math.max(0.0001, ...sampled.map((v) => Math.abs(v)));
  return sampled.map((value) => Math.max(0.08, Math.min(1, Math.abs(value) / max)));
}

function VectorStrip({ label, values, accent }: { label: string; values: number[]; accent: string }) {
  const bars = vectorValues(values);
  if (!bars.length) return null;
  return (
    <div className="grid gap-2 md:grid-cols-[220px_1fr] md:items-end">
      <div className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">{label}</div>
      <div className="flex h-12 items-end gap-[2px] overflow-hidden border border-[var(--rule)] bg-[rgba(242,237,227,0.03)] px-2 py-2">
        {bars.map((height, index) => (
          <span
            key={index}
            className="w-full min-w-[3px]"
            style={{ height: `${height * 100}%`, background: accent, opacity: 0.42 + height * 0.58 }}
          />
        ))}
      </div>
    </div>
  );
}

function EmbeddingFingerprint({ metrics }: { metrics: Metrics | null }) {
  const raw = metrics?.tier2?.raw ?? {};
  const before = Array.isArray(raw.before_vector) ? raw.before_vector : [];
  const after = Array.isArray(raw.after_vector) ? raw.after_vector : [];
  if (!before.length || !after.length) return null;
  const delta = before.map((value: number, index: number) => Math.abs(value - (after[index] ?? 0)));

  return (
    <section className="section-band wrap pb-16 pt-12">
      <div className="border border-[var(--rule)] bg-[rgba(28,24,18,0.72)] p-6">
        <div className="mb-6 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
          <Gauge size={13} /> Embedding drift fingerprint
        </div>
        <div className="grid gap-5">
          <VectorStrip label="User embedding before session" values={before} accent="var(--blue)" />
          <VectorStrip label="User embedding after erase" values={after} accent="var(--amber)" />
          <VectorStrip label="Absolute reversion error" values={delta} accent="var(--wine)" />
        </div>
      </div>
    </section>
  );
}

function FinalVerdict({ tier1, tier2 }: { tier1: { passed: number; total: number }; tier2: { passed: number; total: number } }) {
  const t1Pass = tier1.total > 0 && tier1.passed === tier1.total;
  const t2Pass = tier2.total > 0 && tier2.passed === tier2.total;
  return (
    <section className="section-band bg-[#0e0a05] py-12">
      <div className="wrap">
        <div className="border border-[var(--rule-strong)] p-7">
          <div className="mb-6 text-center font-space-grotesk text-[10px] uppercase tracking-[0.16em] text-[var(--clay)]">
            Final verdict
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="border border-[var(--rule)] p-5">
              <div className="font-display text-4xl text-[var(--wine)]">Tier I</div>
              <div className={`mt-3 font-space-grotesk text-[11px] uppercase tracking-[0.14em] ${t1Pass ? "text-[var(--green)]" : "text-[var(--wine)]"}`}>
                {t1Pass ? "Pass" : "Needs review"}
              </div>
              <p className="mt-3 text-sm leading-6 text-[var(--ink-2)]">
                {tier1.passed}/{tier1.total} permanent-unlearning metrics within threshold.
              </p>
            </div>
            <div className="border border-[var(--rule)] p-5">
              <div className="font-display text-4xl text-[var(--amber)]">Tier II</div>
              <div className={`mt-3 font-space-grotesk text-[11px] uppercase tracking-[0.14em] ${t2Pass ? "text-[var(--green)]" : "text-[var(--wine)]"}`}>
                {t2Pass ? "Pass" : "Needs review"}
              </div>
              <p className="mt-3 text-sm leading-6 text-[var(--ink-2)]">
                {tier2.passed}/{tier2.total} session-unlearning metrics within threshold.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function OutputPanel({ log, running }: { log: string; running: boolean }) {
  if (!log && !running) return null;
  const lines = (log || "Waiting for Python output...").split("\n");
  return (
    <section className="section-band wrap pb-16 pt-10">
      <div className="border border-[var(--rule)] bg-[var(--bone-2)]">
        <div className="flex items-center justify-between border-b border-[var(--rule)] px-5 py-4">
          <div className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
            Evaluation output
          </div>
          {running && <Loader2 size={14} className="animate-spin text-[var(--wine)]" />}
        </div>
        <div className="grid gap-px bg-[var(--rule)]">
          {lines.map((line, i) => (
            <div key={`${line}-${i}`} className="bg-[var(--bone-2)] px-5 py-3 font-mono text-xs leading-6 text-[var(--ink-2)]">
              {line}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const previewBars = [
  [0.72, "var(--wine)"],
  [0.44, "var(--blue)"],
  [0.91, "var(--green)"],
  [0.33, "var(--amber)"],
  [0.58, "var(--purple)"],
  [0.18, "var(--wine)"],
] as const;

function PlaceholderBars({ active = false }: { active?: boolean }) {
  return (
    <div className="grid gap-4">
      {previewBars.map(([width, color], index) => (
        <div key={index} className="grid items-center gap-3 md:grid-cols-[150px_1fr_64px]">
          <div className="h-3 border border-[var(--rule)] bg-[rgba(242,237,227,0.05)]" />
          <div className="h-8 border border-[var(--rule)] bg-[rgba(242,237,227,0.035)] p-1">
            <div
              className={active ? "h-full motion-safe:animate-pulse" : "h-full opacity-60"}
              style={{
                width: `${width * 100}%`,
                background: color,
                animationDelay: `${index * 140}ms`,
              }}
            />
          </div>
          <div className="h-3 border border-[var(--rule)] bg-[rgba(242,237,227,0.05)]" />
        </div>
      ))}
    </div>
  );
}

function PlaceholderDashboard({ onRun }: { onRun: () => void }) {
  return (
    <>
      <section className="section-band wrap grid gap-6 py-12 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="border border-[var(--rule)] bg-[rgba(28,24,18,0.72)] p-7">
          <div className="eyebrow">Waiting for evaluation</div>
          <h2 className="font-display mt-4 text-5xl leading-tight">No run loaded yet.</h2>
          <p className="mt-5 text-sm leading-7 text-[var(--ink-2)]">
            This page intentionally starts clean. Press Run Evaluation to execute the project evaluation suite and reveal the measured Tier I and Tier II metrics for this browser session.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <GlassButton onClick={onRun}>
              <Play size={15} />
              Run Evaluation
            </GlassButton>
            <Link href="/architecture">
              <GlassButton variant="secondary">View Architecture</GlassButton>
            </Link>
          </div>
        </div>

        <div className="border border-[var(--rule)] p-6">
          <div className="mb-6 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
            <BarChart3 size={13} /> Preview dashboard
          </div>
          <PlaceholderBars />
        </div>
      </section>

      <section className="wrap grid gap-6 pb-16 lg:grid-cols-3">
        {[
          ["Tier I", "GNNDelete checks will appear here after the run.", "var(--wine)"],
          ["Tier II", "Influence-function session metrics will populate after completion.", "var(--amber)"],
          ["Fingerprint", "Embedding vector strips are generated only from real output.", "var(--blue)"],
        ].map(([title, body, color]) => (
          <article key={title} className="border border-[var(--rule)] p-6">
            <div className="font-display text-4xl" style={{ color }}>{title}</div>
            <p className="mt-4 text-sm leading-7 text-[var(--ink-2)]">{body}</p>
            <div className="mt-6 grid grid-cols-5 gap-2">
              {[0.2, 0.55, 0.35, 0.8, 0.45].map((height, index) => (
                <span
                  key={index}
                  className="block border border-[var(--rule)] bg-[rgba(242,237,227,0.04)]"
                  style={{ height: `${30 + height * 60}px` }}
                />
              ))}
            </div>
          </article>
        ))}
      </section>
    </>
  );
}

function RunningEvaluation() {
  const steps = [
    "Loading LightGCN checkpoint",
    "Sampling forget and retain sets",
    "Running GNNDelete correction",
    "Committing temporary session",
    "Erasing session influence",
    "Writing metrics.json",
  ];

  return (
    <>
      <section className="section-band wrap grid gap-6 py-12 lg:grid-cols-[420px_1fr]">
        <div className="border border-[var(--rule)] bg-[rgba(216,88,74,0.08)] p-7">
          <div className="flex items-center gap-3">
            <Loader2 size={22} className="animate-spin text-[var(--wine)]" />
            <div className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--wine)]">
              Evaluation running
            </div>
          </div>
          <h2 className="font-display mt-5 text-5xl leading-tight">Measuring the forget.</h2>
          <p className="mt-5 text-sm leading-7 text-[var(--ink-2)]">
            The backend is running `evaluation/cli.py --json-only`. Results will replace this animation as soon as the metrics payload returns.
          </p>
        </div>

        <div className="border border-[var(--rule)] p-6">
          <div className="mb-6 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
            <Gauge size={13} /> Live metric slots
          </div>
          <PlaceholderBars active />
        </div>
      </section>

      <section className="wrap pb-16">
        <div className="grid gap-px border border-[var(--rule)] bg-[var(--rule)] md:grid-cols-3">
          {steps.map((step, index) => (
            <div key={step} className="bg-[var(--bone-2)] p-5">
              <div className="flex items-center gap-3">
                <span
                  className="h-2.5 w-2.5 rounded-full bg-[var(--wine)] motion-safe:animate-pulse"
                  style={{ animationDelay: `${index * 180}ms` }}
                />
                <span className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
                  Step {String(index + 1).padStart(2, "0")}
                </span>
              </div>
              <p className="mt-4 text-sm leading-6 text-[var(--ink-2)]">{step}</p>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

function CompletedDashboard({
  metrics,
  research,
  counts,
  t1Score,
  t2Score,
  log,
  running,
}: {
  metrics: Metrics | null;
  research: { tier1: ResearchMetric[]; tier2: ResearchMetric[] };
  counts: { moviesForgotten: number | null; edgesRemoved: number | null; usersEvaluated: number | null; sessionEdges: number | null };
  t1Score: { passed: number; total: number; failed: number };
  t2Score: { passed: number; total: number; failed: number };
  log: string;
  running: boolean;
}) {
  return (
    <>
      <section className="section-band wrap grid gap-6 py-12 lg:grid-cols-2">
        <VerdictCard label="GNNDelete permanent erasure" title="Tier I" metrics={research.tier1} accent="var(--wine)" />
        <VerdictCard label="Influence session unlearning" title="Tier II" metrics={research.tier2} accent="var(--amber)" />
      </section>

      <section className="wrap pb-12">
        <ResearchSummary tier1={research.tier1} tier2={research.tier2} />
      </section>

      <section className="wrap grid gap-6 pb-16 lg:grid-cols-2">
        <div>
          <div className="mb-4 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
            <Gauge size={13} /> Tier I metric cards
          </div>
          <div className="grid gap-4">
            {research.tier1.map((metric) => <MetricCard key={metric.id} metric={metric} />)}
          </div>
        </div>
        <div>
          <div className="mb-4 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
            <Gauge size={13} /> Tier II metric cards
          </div>
          <div className="grid gap-4">
            {research.tier2.map((metric) => <MetricCard key={metric.id} metric={metric} />)}
          </div>
        </div>
      </section>

      <section className="section-band bg-[#0e0a05] py-12">
        <div className="wrap grid gap-5 md:grid-cols-4">
          <CountCard label="Forget edges removed" value={counts.edgesRemoved} accent="var(--wine)" />
          <CountCard label="Users evaluated" value={counts.usersEvaluated} accent="var(--blue)" />
          <CountCard label="Tier I passed / total" valueText={`${t1Score.passed}/${t1Score.total}`} accent="var(--green)" />
          <CountCard label="Tier II passed / total" valueText={`${t2Score.passed}/${t2Score.total}`} accent="var(--amber)" />
        </div>
      </section>

      <EmbeddingFingerprint metrics={metrics} />
      <FinalVerdict tier1={t1Score} tier2={t2Score} />
      <OutputPanel log={log} running={running} />
    </>
  );
}

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [phase, setPhase] = useState<EvalPhase>("idle");
  const [log, setLog] = useState("");
  const [error, setError] = useState("");
  const running = phase === "running";
  const research = useResearchMetrics(metrics);
  const t1Score = score(research.tier1);
  const t2Score = score(research.tier2);

  const counts = useMemo(() => {
    const t1 = metrics?.tier1 ?? {};
    const t2 = metrics?.tier2 ?? {};
    return {
      moviesForgotten: readMetric(t1, ["movies_forgotten", "movies_forgotten", "movies_forgotten", "Movies forgotten"]),
      edgesRemoved: readMetric(t1, ["edges_removed", "Edges removed"]),
      usersEvaluated: readMetric(t1, ["users_evaluated", "Users evaluated"]),
      sessionEdges: readMetric(t2, ["session_size", "edges_in_session", "Edges in session"]),
    };
  }, [metrics]);

  const runEvaluation = async () => {
    setPhase("running");
    setMetrics(null);
    setError("");
    setLog("Running evaluation/cli.py --json-only...");
    try {
      const res = await fetch("/api/evaluation/run", { method: "POST" });
      const data = await res.json();
      setMetrics(data.metrics ?? null);
      setLog([data.stdout, data.stderr].filter(Boolean).join("\n").trim());
      if (!res.ok || !data.success) {
        setError(data.message ?? "Evaluation failed.");
        setPhase("idle");
      } else {
        setPhase("complete");
      }
    } catch {
      setError("Evaluation request failed.");
      setPhase("idle");
    }
  };

  return (
    <main className="min-h-screen bg-[var(--bone)] text-[var(--ink)]">
      <GlassNav />

      <header className="wrap py-16 md:py-20">
        <div className="grid gap-10 lg:grid-cols-[1fr_420px]">
          <div>
            <div className="eyebrow">
              <span className="eyebrow-dot" />
              Evaluation console
            </div>
            <h1 className="font-display mt-7 text-[clamp(46px,7vw,104px)] leading-[0.98] tracking-[-0.025em]">
              Prove the forgetting, <em className="text-[var(--wine)]">then read the result.</em>
            </h1>
            <p className="mt-8 max-w-3xl text-lg leading-8 text-[var(--ink-2)]">
              This dashboard runs the exact repository evaluation script, then translates `evaluation/metrics.json`
              into report-ready checks for permanent erasure and session unlearning.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <GlassButton onClick={runEvaluation} disabled={running}>
                {running ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
                {running ? "Running" : "Run Evaluation"}
              </GlassButton>
              <Link href="/chat"><GlassButton variant="ghost">Open Demo</GlassButton></Link>
            </div>
          </div>

          <div className="border border-[var(--rule)] p-6">
            <div className="flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
              <ShieldCheck size={13} /> {phase === "complete" ? "Current verdict" : phase === "running" ? "Run in progress" : "Ready state"}
            </div>
            {phase === "complete" ? (
              <div className="mt-5 grid grid-cols-2 gap-3">
                <CountCard label="Tier I checks passed" valueText={`${t1Score.passed}/${t1Score.total}`} accent="var(--wine)" />
                <CountCard label="Tier II checks passed" valueText={`${t2Score.passed}/${t2Score.total}`} accent="var(--amber)" />
                <CountCard label="Movies in forget set" value={counts.moviesForgotten} accent="var(--blue)" />
                <CountCard label="Session interactions" value={counts.sessionEdges} accent="var(--green)" />
              </div>
            ) : (
              <div className="mt-5 grid grid-cols-2 gap-3">
                <CountCard label="Results on reload" valueText="Clean" accent="var(--blue)" />
                <CountCard label="Stored in page state" valueText="No" accent="var(--wine)" />
                <CountCard label="Runner" valueText="CLI" accent="var(--amber)" />
                <CountCard label="Rubric" valueText="21" accent="var(--green)" />
              </div>
            )}
            <p className="mt-5 text-xs leading-6 text-[var(--ink-2)]">
              {phase === "complete"
                ? "Metrics are visible for this page session. Refreshing returns the console to the clean preview state."
                : phase === "running"
                  ? "The backend is computing the real evaluation now."
                  : "No real metrics are loaded yet. This is a preview state."}
            </p>
            {error && <p className="mt-3 text-xs leading-6 text-[var(--wine)]">{error}</p>}
          </div>
        </div>
      </header>

      {phase === "idle" && <PlaceholderDashboard onRun={runEvaluation} />}
      {phase === "running" && <RunningEvaluation />}
      {phase === "complete" && (
        <CompletedDashboard
          metrics={metrics}
          research={research}
          counts={counts}
          t1Score={t1Score}
          t2Score={t2Score}
          log={log}
          running={running}
        />
      )}
    </main>
  );
}
