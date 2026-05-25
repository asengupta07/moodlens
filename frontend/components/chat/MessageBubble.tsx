"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { ChatMessage } from "./ChatWindow";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28 }}
      className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          "max-w-[88%] border px-4 py-3 text-[14.5px] leading-7 shadow-[inset_0_1px_0_rgba(242,237,227,0.04)] md:max-w-[78%]",
          isUser
            ? "border-[rgba(216,88,74,0.34)] bg-[rgba(216,88,74,0.12)] text-[var(--ink)]"
            : "border-[var(--rule)] bg-[rgba(242,237,227,0.045)] text-[var(--ink-2)]",
        )}
      >
        <div className="mb-1 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
          {isUser ? "You" : "MoodLens"}
        </div>
        <div className={cn("markdown-body", isUser && "markdown-body-user")}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content || ""}
          </ReactMarkdown>
        </div>
      </div>
    </motion.div>
  );
}
