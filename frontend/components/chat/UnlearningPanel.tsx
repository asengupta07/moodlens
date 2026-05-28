"use client";

import { CheckCircle2, History, ShieldCheck, Sparkles, Trash2 } from "lucide-react";

export interface DriftEvent {
  tier: number;
  event_type: string;
  timestamp: string;
  cosine_distance?: number;
  movies_affected?: number;
  edges_removed?: number;
  edges_processed?: number;
  forget_score?: number;
  retain_score?: number;
  mood?: string | null;
  mode?: string;
  reversion_score?: number;
  non_destructive?: boolean;
}

interface Props {
  lastEvent: DriftEvent | null;
}

function fmtDate(s?: string) {
  if (!s) return "No event";
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
}

function fmt(v?: number) {
  if (v === undefined || v === null || Number.isNaN(v)) return "-";
  return v.toFixed(4);
}

export function UnlearningPanel({ lastEvent }: Props) {
  if (!lastEvent) {
    return (
      <section className="mood-panel p-4">
        <div className="mb-3 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
          <History size={12} /> Memory surgery log
        </div>
        <p className="text-xs leading-6 text-[var(--ink-2)]">
          When MoodLens forgets something, this panel explains what changed. Permanent erasure means the model carved out a long-term dislike. Session clearing means tonight's mood was wiped.
        </p>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <Mini label="Tier I" value="Identity" color="var(--wine)" />
          <Mini label="Tier II" value="Mood" color="var(--amber)" />
        </div>
      </section>
    );
  }

  const isTier1 = lastEvent.tier === 1;
  const Icon = isTier1 ? Trash2 : Sparkles;
  const color = isTier1 ? "var(--wine)" : "var(--amber)";
  const isCleanSessionErase = !isTier1 && (
    lastEvent.non_destructive || (lastEvent.reversion_score ?? 0) >= 0.98
  );

  return (
    <section className="mood-panel p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em]" style={{ color }}>
          <Icon size={13} />
          {isTier1 ? "Permanent erasure" : "Session cleared"}
        </div>
        <span className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
          Tier {lastEvent.tier}
        </span>
      </div>

      <div className="font-display text-4xl leading-none" style={{ color }}>
        {fmt(lastEvent.cosine_distance)}
      </div>
      <div className="mt-1 text-[11px] text-[var(--clay)]">memory movement score - {fmtDate(lastEvent.timestamp)}</div>
      <p className="mt-3 text-xs leading-6 text-[var(--ink-2)]">
        {isTier1
          ? "This is the size of the embedding shift caused by permanent unlearning. Bigger usually means the forgotten region actually moved."
          : "This is the measured mood rollback. Discarding a live session clears the temporary ranking vector without writing to the permanent checkpoint."}
      </p>

      {!isTier1 && (
        <div className="mt-3 flex items-start gap-2 border border-[rgba(216,168,74,0.28)] bg-[rgba(216,168,74,0.07)] p-3 text-xs leading-5 text-[var(--ink-2)]">
          {isCleanSessionErase ? (
            <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-[var(--green)]" />
          ) : (
            <ShieldCheck size={15} className="mt-0.5 shrink-0 text-[var(--amber)]" />
          )}
          <span>
            {isCleanSessionErase
              ? "Clean session boundary: the mood was removed without residual profile drift."
              : "Session boundary recorded. Check reversion and residual drift before claiming a clean erase."}
          </span>
        </div>
      )}

      <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-[var(--rule)] pt-4 text-xs">
        {isTier1 ? (
          <>
            <Metric label="Movies erased" value={lastEvent.movies_affected ?? "-"} />
            <Metric label="Edges removed" value={lastEvent.edges_removed ?? "-"} />
            <Metric label="Forget score" value={fmt(lastEvent.forget_score)} />
            <Metric label="Retain score" value={fmt(lastEvent.retain_score)} />
          </>
        ) : (
          <>
            <Metric label="Mood" value={lastEvent.mood ?? "-"} />
            <Metric label="Edges" value={lastEvent.edges_processed ?? "-"} />
            <Metric label="Mode" value={lastEvent.mode ?? "-"} />
            <Metric label="Reversion" value={fmt(lastEvent.reversion_score)} />
            <Metric label="Write" value={lastEvent.non_destructive ? "No checkpoint write" : "Model update"} />
          </>
        )}
      </div>
    </section>
  );
}

function Mini({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="border border-[var(--rule)] p-2">
      <div className="font-space-grotesk text-[9px] uppercase tracking-[0.1em] text-[var(--clay)]">{label}</div>
      <div className="mt-1 text-sm" style={{ color }}>{value}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="font-space-grotesk text-[10px] uppercase tracking-[0.1em] text-[var(--clay)]">{label}</div>
      <div className="mt-1 truncate text-[var(--ink)]">{value}</div>
    </div>
  );
}
