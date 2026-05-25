"use client";

import { Loader2, Send } from "lucide-react";
import React, { useEffect, useRef } from "react";

interface ChatInputProps {
  input: string;
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
}

export function ChatInput({
  input,
  handleInputChange,
  handleSubmit,
  isLoading,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 132)}px`;
  }, [input]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if ((input || "").trim() && !isLoading) e.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-end gap-3 border border-[var(--rule)] bg-[var(--bone-3)] p-2 focus-within:border-[var(--rule-strong)]"
    >
      <textarea
        ref={textareaRef}
        value={input || ""}
        onChange={handleInputChange}
        onKeyDown={onKeyDown}
        placeholder="Tell MoodLens what you feel like watching..."
        rows={1}
        className="max-h-[132px] flex-1 resize-none bg-transparent px-2 py-3 text-[15px] leading-6 text-[var(--ink)] outline-none placeholder:text-[var(--clay-2)]"
      />
      <button
        type="submit"
        disabled={!(input || "").trim() || isLoading}
        className="tap-highlight mb-1 flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[var(--ink)] text-[var(--bone)] transition-colors hover:bg-[var(--wine)] disabled:opacity-40"
        aria-label="Send message"
      >
        {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={17} />}
      </button>
    </form>
  );
}
