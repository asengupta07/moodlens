"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
}

export function GlassCard({ children, className, hover }: GlassCardProps) {
  return (
    <motion.div
      whileHover={hover ? { y: -3 } : undefined}
      transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
      className={cn("mood-card rounded-[2px]", className)}
    >
      {children}
    </motion.div>
  );
}
