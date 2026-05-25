"use client";

import { MessageBubble } from "./MessageBubble";
import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export function ChatWindow({
  messages,
  isLoading,
}: {
  messages: ChatMessage[];
  isLoading: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto pr-2">
      {messages.length === 0 ? (
        <div className="flex h-full items-center justify-center text-center">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-sm"
          >
            <p className="font-display text-3xl italic text-[var(--ink)]">
              What are we in the mood for?
            </p>
            <p className="mt-3 text-sm leading-6 text-[var(--clay)]">
              MoodLens will separate temporary mood signals from permanent taste.
            </p>
          </motion.div>
        </div>
      ) : (
        <div className="flex flex-col gap-4 pb-4">
          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}
          {isLoading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-1.5 pl-2 pt-2">
              {[0, 160, 320].map((delay) => (
                <span
                  key={delay}
                  className="h-2 w-2 rounded-full bg-[var(--wine)] motion-safe:animate-bounce"
                  style={{ animationDelay: `${delay}ms` }}
                />
              ))}
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}
