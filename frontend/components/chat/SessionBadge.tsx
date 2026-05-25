"use client";

import { motion } from "framer-motion";
import { Sparkles, Moon } from "lucide-react";

interface Props {
  active: boolean;
  mood: string | null;
  movieCount: number;
}

export function SessionBadge({ active, mood, movieCount }: Props) {
  if (!active) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-white/5 text-white/40 text-xs">
        <Moon size={12} />
        <span>No active mood</span>
      </div>
    );
  }
  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-amber-400/30 bg-amber-400/10 text-amber-300 text-xs motion-safe:animate-pulse"
    >
      <Sparkles size={12} className="text-amber-300" />
      <span className="font-medium">{mood ? `${mood} mood` : "Mood active"}</span>
      <span className="text-amber-200/70">· {movieCount} movies</span>
    </motion.div>
  );
}
