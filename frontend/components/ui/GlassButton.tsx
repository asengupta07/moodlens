"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface GlassButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  children: React.ReactNode;
}

export const GlassButton = React.forwardRef<HTMLButtonElement, GlassButtonProps>(
  ({ className, variant = "primary", size = "md", children, ...props }, ref) => {
    const base =
      "tap-highlight inline-flex items-center justify-center gap-2 rounded-full font-space-grotesk uppercase tracking-[0.14em] transition-colors disabled:pointer-events-none disabled:opacity-45";
    const sizes = {
      sm: "px-4 py-2 text-[10px]",
      md: "px-5 py-3 text-[11px]",
      lg: "px-7 py-4 text-xs",
    };
    const variants = {
      primary: "bg-[var(--ink)] text-[var(--bone)] hover:bg-[var(--wine)]",
      secondary:
        "border border-[var(--rule-strong)] bg-[rgba(242,237,227,0.04)] text-[var(--ink)] hover:border-[var(--ink)]",
      ghost: "text-[var(--ink-2)] hover:text-[var(--wine)] hover:bg-[rgba(242,237,227,0.05)]",
    };

    return (
      <button
        ref={ref}
        className={cn(base, sizes[size], variants[variant], className)}
        {...props}
      >
        {children}
      </button>
    );
  },
);

GlassButton.displayName = "GlassButton";
