"use client";

import { motion } from "framer-motion";
import { Moon, Sparkles } from "lucide-react";

interface Props {
  active: boolean;
  mood: string | null;
  movieCount: number;
}

export function SessionBadge({ active, mood, movieCount }: Props) {
  if (!active) {
    return (
      <div className="flex items-center justify-between border border-[var(--rule)] bg-[rgba(242,237,227,0.035)] px-4 py-3 text-xs text-[var(--clay)]">
        <span className="flex items-center gap-2">
          <Moon size={13} /> No active mood
        </span>
        <span className="font-space-grotesk uppercase tracking-[0.12em]">0 movies</span>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex items-center justify-between border border-[rgba(216,168,74,0.36)] bg-[rgba(216,168,74,0.10)] px-4 py-3 text-xs text-[var(--ink)]"
    >
      <span className="flex items-center gap-2">
        <Sparkles size={13} className="text-[var(--amber)]" />
        <span className="font-medium">{mood ? `${mood} mood` : "Mood active"}</span>
      </span>
      <span className="font-space-grotesk uppercase tracking-[0.12em] text-[var(--amber)]">
        {movieCount} movies
      </span>
    </motion.div>
  );
}
