"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bookmark, Loader2, RefreshCw, Trash2, X } from "lucide-react";

interface SessionSummary {
  movie_count?: number;
  liked_count?: number;
  positive_count?: number;
  negative_count?: number;
  mood_confidence?: number;
  dominant_mood?: string | null;
  duration_minutes?: number;
}

interface DriftMetrics {
  cosine_distance?: number;
  edges_processed?: number;
  reversion_score?: number;
  non_destructive?: boolean;
  mode?: string;
}

interface Props {
  active: boolean;
  movieCount: number;
  mood: string | null;
  onCleared: (drift: DriftMetrics | null, summary: SessionSummary) => void;
}

export function NewMoodButton({ active, movieCount, mood, onCleared }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<"discard" | "commit" | null>(null);

  const trigger = async (action: "discard" | "commit") => {
    setBusy(action);
    try {
      const res = await fetch("/api/new-mood", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = await res.json();
      onCleared(data.embedding_drift ?? null, data.session_summary ?? {});
    } finally {
      setBusy(null);
      setOpen(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        disabled={!active}
        className={`tap-highlight flex w-full items-center justify-center gap-2 rounded-full px-4 py-3 font-space-grotesk text-[11px] uppercase tracking-[0.14em] transition-colors ${
          active
            ? "bg-[var(--wine)] text-[var(--bone)] hover:bg-[var(--ink)]"
            : "border border-[var(--rule)] bg-[rgba(242,237,227,0.035)] text-[var(--clay-2)]"
        }`}
      >
        <RefreshCw size={14} />
        New Mood
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm"
            onClick={(e) => {
              if (e.target === e.currentTarget && !busy) setOpen(false);
            }}
          >
            <motion.div
              initial={{ scale: 0.96, opacity: 0, y: 12 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.96, opacity: 0, y: 12 }}
              className="w-full max-w-xl border border-[var(--rule)] bg-[var(--bone-2)] p-6 shadow-2xl"
            >
              <div className="mb-6 flex items-start justify-between gap-5">
                <div>
                  <div className="eyebrow">Session boundary</div>
                  <h3 className="font-display mt-2 text-4xl leading-none text-[var(--ink)]">
                    End this mood?
                  </h3>
                  <p className="mt-3 max-w-md text-sm leading-6 text-[var(--ink-2)]">
                    Choose whether this session becomes part of your permanent taste graph. Discarding clears the temporary ranking vector without writing to the checkpoint.
                  </p>
                </div>
                <button
                  onClick={() => !busy && setOpen(false)}
                  className="tap-highlight rounded-full border border-[var(--rule)] p-2 text-[var(--clay)] hover:text-[var(--ink)]"
                  disabled={!!busy}
                  aria-label="Close"
                >
                  <X size={17} />
                </button>
              </div>

              <div className="mb-5 grid gap-3 border-y border-[var(--rule)] py-4 text-sm sm:grid-cols-3">
                <div>
                  <div className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
                    Dominant mood
                  </div>
                  <div className="mt-1 text-[var(--amber)]">{mood ?? "Unlabelled"}</div>
                </div>
                <div>
                  <div className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
                    Movies
                  </div>
                  <div className="mt-1 font-mono text-[var(--ink)]">{movieCount}</div>
                </div>
                <div>
                  <div className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
                    Tier
                  </div>
                  <div className="mt-1 text-[var(--ink-2)]">Influence functions</div>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <button
                  onClick={() => trigger("discard")}
                  disabled={!!busy}
                  className="tap-highlight border border-[rgba(216,88,74,0.45)] bg-[rgba(216,88,74,0.10)] p-5 text-left text-[var(--ink)] transition-colors hover:bg-[rgba(216,88,74,0.16)] disabled:opacity-50"
                >
                  <div className="flex items-center gap-2 font-space-grotesk text-[11px] uppercase tracking-[0.14em] text-[var(--wine)]">
                    {busy === "discard" ? <Loader2 size={16} className="animate-spin" /> : <Trash2 size={16} />}
                    Forget this mood
                  </div>
                  <p className="mt-3 text-sm leading-6 text-[var(--ink-2)]">
                    Roll back this mood boundary and leave long-term taste unchanged.
                  </p>
                </button>
                <button
                  onClick={() => trigger("commit")}
                  disabled={!!busy}
                  className="tap-highlight border border-[rgba(120,166,200,0.45)] bg-[rgba(120,166,200,0.10)] p-5 text-left text-[var(--ink)] transition-colors hover:bg-[rgba(120,166,200,0.16)] disabled:opacity-50"
                >
                  <div className="flex items-center gap-2 font-space-grotesk text-[11px] uppercase tracking-[0.14em] text-[var(--blue)]">
                    {busy === "commit" ? <Loader2 size={16} className="animate-spin" /> : <Bookmark size={16} />}
                    Keep in my profile
                  </div>
                  <p className="mt-3 text-sm leading-6 text-[var(--ink-2)]">
                    Fine-tune LightGCN and promote positive session signals into identity memory.
                  </p>
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
