"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { GlassButton } from "./GlassButton";

interface GlassNavProps {
  onMobileMenuToggle?: (isOpen: boolean) => void;
}

const navLinks = [
  { name: "Thesis", path: "/#thesis" },
  { name: "Chat", path: "/chat" },
  { name: "Architecture", path: "/architecture" },
  { name: "Evaluation", path: "/evaluation" },
];

export function GlassNav({ onMobileMenuToggle }: GlassNavProps) {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => {
    const next = !isOpen;
    setIsOpen(next);
    onMobileMenuToggle?.(next);
  };

  return (
    <>
      <motion.nav
        initial={{ y: -72 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        className="sticky top-0 z-50 border-b border-[var(--rule)] bg-[rgba(20,16,8,0.78)] backdrop-blur-xl"
      >
        <div className="wrap flex h-16 items-center justify-between gap-6">
          <Link href="/" className="flex items-center gap-3">
            <span className="relative h-[18px] w-[18px] rounded-full border border-[var(--ink)] after:absolute after:inset-[3px] after:rounded-full after:bg-[var(--ink)]" />
            <span className="font-display text-[23px] leading-none tracking-[-0.01em]">
              Mood<em className="text-[var(--wine)]">Lens</em>
            </span>
          </Link>

          <div className="hidden items-center gap-8 lg:flex">
            {navLinks.map((link) => {
              const active =
                link.path === "/"
                  ? pathname === "/"
                  : pathname === link.path || (link.path.includes("#") && pathname === "/");
              return (
                <Link
                  key={link.path}
                  href={link.path}
                  className={cn(
                    "relative text-[13px] text-[var(--ink-2)] transition-colors hover:text-[var(--wine)]",
                    active && "text-[var(--ink)]",
                  )}
                >
                  {link.name}
                </Link>
              );
            })}
          </div>

          <div className="hidden items-center gap-3 lg:flex">
            <span className="inline-flex items-center gap-2 rounded-full border border-[var(--rule-strong)] px-3 py-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--wine)] motion-safe:animate-pulse" />
              Research preview
            </span>
            <Link href="/chat">
              <GlassButton size="sm">Open Demo</GlassButton>
            </Link>
          </div>

          <button
            className="tap-highlight rounded-full border border-[var(--rule)] p-2 text-[var(--ink)] lg:hidden"
            onClick={toggle}
            aria-label="Toggle navigation"
          >
            {isOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </motion.nav>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="fixed left-4 right-4 top-20 z-40 overflow-hidden border border-[var(--rule)] bg-[var(--bone-2)] lg:hidden"
          >
            <div className="grid gap-1 p-4">
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  href={link.path}
                  onClick={() => setIsOpen(false)}
                  className="px-3 py-3 text-sm text-[var(--ink-2)] hover:text-[var(--wine)]"
                >
                  {link.name}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
