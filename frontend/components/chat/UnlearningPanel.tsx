"use client";

import { Sparkles, Trash2, History } from "lucide-react";

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
}

interface Props {
  lastEvent: DriftEvent | null;
}

function fmtDate(s?: string) {
  if (!s) return "—";
  try { return new Date(s).toLocaleString(); } catch { return s; }
}

function fmtDrift(v?: number) {
  if (v === undefined || v === null || isNaN(v)) return "—";
  return v.toFixed(4);
}

export function UnlearningPanel({ lastEvent }: Props) {
  if (!lastEvent) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/5 p-3 text-xs">
        <div className="flex items-center gap-2 text-white/40 uppercase tracking-wider text-[10px] font-medium mb-2">
          <History size={11} /> Unlearning Panel
        </div>
        <p className="text-white/40 text-[11px] leading-relaxed">
          No unlearning events yet. Try "block horror forever" to trigger Tier 1
          (GNNDelete), or press <strong>New Mood</strong> to trigger Tier 2
          (influence functions).
        </p>
      </div>
    );
  }
  const isTier1 = lastEvent.tier === 1;
  const color = isTier1 ? "text-red-300" : "text-amber-300";
  const accent = isTier1 ? "border-red-500/30 bg-red-500/5" : "border-amber-400/30 bg-amber-400/5";
  const Icon = isTier1 ? Trash2 : Sparkles;
  return (
    <div className={`rounded-xl border ${accent} p-3 text-xs space-y-2`}>
      <div className="flex items-center gap-2">
        <Icon size={13} className={color} />
        <span className={`font-medium ${color}`}>
          {isTier1 ? "Permanent erasure" : "Session cleared"}
        </span>
        <span className="ml-auto text-white/40 text-[10px]">
          Tier {lastEvent.tier}
        </span>
      </div>
      <div className="text-white/50 text-[10px]">{fmtDate(lastEvent.timestamp)}</div>

      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-white/10">
        <div>
          <div className="text-white/40 text-[10px]">Cosine drift</div>
          <div className={`font-mono ${color}`}>{fmtDrift(lastEvent.cosine_distance)}</div>
        </div>
        {isTier1 ? (
          <>
            <div>
              <div className="text-white/40 text-[10px]">Movies erased</div>
              <div className="text-white font-mono">{lastEvent.movies_affected ?? "—"}</div>
            </div>
            <div>
              <div className="text-white/40 text-[10px]">Edges removed</div>
              <div className="text-white font-mono">{lastEvent.edges_removed ?? "—"}</div>
            </div>
            <div>
              <div className="text-white/40 text-[10px]">Δ retain−forget</div>
              <div className="text-white font-mono">
                {fmtDrift((lastEvent.retain_score ?? 0) - (lastEvent.forget_score ?? 0))}
              </div>
            </div>
          </>
        ) : (
          <>
            <div>
              <div className="text-white/40 text-[10px]">Mood</div>
              <div className="text-amber-300">{lastEvent.mood ?? "—"}</div>
            </div>
            <div>
              <div className="text-white/40 text-[10px]">Edges</div>
              <div className="text-white font-mono">{lastEvent.edges_processed ?? "—"}</div>
            </div>
            <div>
              <div className="text-white/40 text-[10px]">Mode</div>
              <div className="text-white capitalize">{lastEvent.mode ?? "—"}</div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
