"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RefreshCw, X, Trash2, Bookmark, Loader2 } from "lucide-react";

interface SessionSummary {
  movie_count?: number;
  liked_count?: number;
  dominant_mood?: string | null;
  duration_minutes?: number;
}

interface DriftMetrics {
  cosine_distance?: number;
  edges_processed?: number;
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
  const [busy, setBusy] = useState(false);

  const trigger = async (action: "discard" | "commit") => {
    setBusy(true);
    try {
      const res = await fetch("/api/new-mood", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = await res.json();
      onCleared(data.embedding_drift ?? null, data.session_summary ?? {});
    } finally {
      setBusy(false);
      setOpen(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        disabled={!active}
        className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl border text-sm transition-all duration-200
          ${active
            ? "border-amber-400/40 bg-amber-400/10 text-amber-300 hover:bg-amber-400/20"
            : "border-white/10 bg-white/5 text-white/30 cursor-not-allowed"}`}
      >
        <RefreshCw size={14} />
        <span>New Mood</span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md flex items-center justify-center p-4"
            onClick={(e) => {
              if (e.target === e.currentTarget && !busy) setOpen(false);
            }}
          >
            <motion.div
              initial={{ scale: 0.92, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.92, opacity: 0 }}
              className="w-full max-w-md rounded-2xl border border-white/10 bg-bg-base p-6"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-white font-semibold text-lg">End this mood?</h3>
                  <p className="text-white/50 text-xs mt-1">
                    Two-tier unlearning. Your decision controls how this session
                    affects your taste profile.
                  </p>
                </div>
                <button
                  onClick={() => !busy && setOpen(false)}
                  className="text-white/40 hover:text-white"
                  disabled={busy}
                >
                  <X size={18} />
                </button>
              </div>

              <div className="rounded-xl bg-white/5 border border-white/10 p-3 mb-5 text-xs space-y-1.5">
                <div className="flex justify-between text-white/70">
                  <span>Dominant mood</span>
                  <span className="text-amber-300 font-medium">
                    {mood ?? "—"}
                  </span>
                </div>
                <div className="flex justify-between text-white/70">
                  <span>Movies this session</span>
                  <span className="text-white tabular-nums">{movieCount}</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => trigger("discard")}
                  disabled={busy}
                  className="flex flex-col items-center gap-1 p-4 rounded-xl border border-red-500/40 bg-red-500/10 text-red-300 hover:bg-red-500/20 transition-all disabled:opacity-50"
                >
                  {busy ? <Loader2 size={18} className="animate-spin" /> : <Trash2 size={18} />}
                  <span className="text-sm font-medium">Forget this mood</span>
                  <span className="text-[10px] text-red-300/70 text-center">
                    Tier 2 erase via influence functions
                  </span>
                </button>
                <button
                  onClick={() => trigger("commit")}
                  disabled={busy}
                  className="flex flex-col items-center gap-1 p-4 rounded-xl border border-blue-500/40 bg-blue-500/10 text-blue-300 hover:bg-blue-500/20 transition-all disabled:opacity-50"
                >
                  {busy ? <Loader2 size={18} className="animate-spin" /> : <Bookmark size={18} />}
                  <span className="text-sm font-medium">Keep in my profile</span>
                  <span className="text-[10px] text-blue-300/70 text-center">
                    Fine-tune LightGCN with these edges
                  </span>
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
