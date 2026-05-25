"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { GlassNav } from "@/components/ui/GlassNav";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { GlassCard } from "@/components/ui/GlassCard";
import { ChatWindow, ChatMessage } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { GlassButton } from "@/components/ui/GlassButton";
import { GnnVisualizer } from "@/components/chat/GnnVisualizer";
import { SessionBadge } from "@/components/chat/SessionBadge";
import { NewMoodButton } from "@/components/chat/NewMoodButton";
import { UnlearningPanel, DriftEvent } from "@/components/chat/UnlearningPanel";
import { EmbeddingDriftChart } from "@/components/chat/EmbeddingDriftChart";
import { MessageSquare, Circle, GitGraph, Trash2, Loader2 } from "lucide-react";

interface GraphData {
  nodes: any[];
  edges: any[];
  genre_weights: Record<string, number>;
  stats: {
    liked_count: number;
    disliked_count: number;
    liked_genres: string[];
    disliked_genres: string[];
    blocked_count?: number;
    blocked_genres?: string[];
  };
  session_active?: boolean;
  session_mood?: string | null;
  session_count?: number;
  blocked_count?: number;
}

const EMPTY_GRAPH: GraphData = {
  nodes: [],
  edges: [],
  genre_weights: {},
  stats: { liked_count: 0, disliked_count: 0, liked_genres: [], disliked_genres: [] },
  session_active: false,
  session_mood: null,
  session_count: 0,
  blocked_count: 0,
};

export default function ChatPage() {
  const [messages, setMessages]             = useState<ChatMessage[]>([]);
  const [input, setInput]                   = useState("");
  const [isLoading, setIsLoading]           = useState(false);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [graphData, setGraphData]           = useState<GraphData>(EMPTY_GRAPH);
  const [isOnline, setIsOnline]             = useState(false);
  const [isResetting, setIsResetting]       = useState(false);
  const [sessionActive, setSessionActive]   = useState(false);
  const [sessionMood, setSessionMood]       = useState<string | null>(null);
  const [sessionCount, setSessionCount]     = useState(0);
  const [lastDrift, setLastDrift]           = useState<DriftEvent | null>(null);
  const [permanentHistory, setPermanentHistory] = useState<any[]>([]);
  const [sessionHistory, setSessionHistory] = useState<any[]>([]);
  const abortRef                            = useRef<AbortController | null>(null);

  const fetchGraph = useCallback(async () => {
    try {
      const r = await fetch("/api/graph", { cache: "no-store" });
      if (r.ok) {
        const data = await r.json();
        setGraphData(data);
        setSessionActive(!!data.session_active);
        setSessionMood(data.session_mood ?? null);
        setSessionCount(data.session_count ?? 0);
      }
    } catch {}
  }, []);

  const fetchDriftHistory = useCallback(async () => {
    try {
      const r = await fetch("/api/embedding-drift", { cache: "no-store" });
      if (r.ok) {
        const data = await r.json();
        setPermanentHistory(data.permanent_history ?? []);
        setSessionHistory(data.session_history ?? []);
        const all = [...(data.permanent_history ?? []), ...(data.session_history ?? [])];
        if (all.length > 0) {
          all.sort((a, b) =>
            (new Date(b.timestamp).getTime() || 0) - (new Date(a.timestamp).getTime() || 0),
          );
          setLastDrift(all[0]);
        }
      }
    } catch {}
  }, []);

  useEffect(() => {
    const init = async () => {
      try {
        const greetRes = await fetch("/api/greet", { cache: "no-store" });
        if (greetRes.ok) {
          setIsOnline(true);
          const data = await greetRes.json();
          if (data.greeting) {
            setMessages([{ id: "greeting", role: "assistant", content: data.greeting }]);
          }
          await fetchGraph();
          await fetchDriftHistory();
        }
      } catch {
        setIsOnline(false);
      }
    };
    init();
  }, [fetchGraph, fetchDriftHistory]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleSubmit = useCallback(async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    const assistantId = `a-${Date.now()}`;
    setMessages(prev => [...prev, { id: assistantId, role: "assistant", content: "" }]);

    try {
      abortRef.current = new AbortController();
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: abortRef.current.signal,
      });

      if (!res.ok || !res.body) {
        setMessages(prev =>
          prev.map(m => m.id === assistantId
            ? { ...m, content: "⚠️ Failed to reach the backend. Make sure the Python server is running on port 8000." }
            : m
          )
        );
        setIsLoading(false);
        return;
      }

      const reader    = res.body.getReader();
      const decoder   = new TextDecoder();
      let buffer      = "";
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const event = JSON.parse(raw);
            if (event.type === "token") {
              fullContent += event.content;
              setMessages(prev =>
                prev.map(m => m.id === assistantId ? { ...m, content: fullContent } : m)
              );
            } else if (event.type === "graph") {
              setGraphData(event.data);
              setSessionActive(!!event.data.session_active);
              setSessionMood(event.data.session_mood ?? null);
              setSessionCount(event.data.session_count ?? 0);
            } else if (event.type === "session") {
              setSessionActive(!!event.active);
              setSessionMood(event.mood ?? null);
              setSessionCount(event.movie_count ?? 0);
            } else if (event.type === "unlearn") {
              const drift: DriftEvent = {
                tier: event.tier,
                event_type: event.tier === 1 ? "permanent_unlearn" : "session_event",
                timestamp: new Date().toISOString(),
                ...event.metrics,
              };
              setLastDrift(drift);
              fetchDriftHistory();
            }
          } catch {}
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setMessages(prev =>
          prev.map(m => m.id === assistantId
            ? { ...m, content: "⚠️ Connection error. Is the Python backend running on port 8000?" }
            : m
          )
        );
      }
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, fetchDriftHistory]);

  const handleReset = async () => {
    if (!confirm("Reset all preferences, mood session, and conversation history?")) return;
    setIsResetting(true);
    try {
      await fetch("/api/reset", { method: "POST" });
      setMessages([]);
      setGraphData(EMPTY_GRAPH);
      setLastDrift(null);
      setPermanentHistory([]);
      setSessionHistory([]);
      setSessionActive(false);
      setSessionMood(null);
      setSessionCount(0);
      const greetRes = await fetch("/api/greet", { cache: "no-store" });
      if (greetRes.ok) {
        const data = await greetRes.json();
        if (data.greeting) {
          setMessages([{ id: `greeting-${Date.now()}`, role: "assistant", content: data.greeting }]);
        }
      }
    } finally {
      setIsResetting(false);
    }
  };

  const handleNewMoodCleared = (drift: any, summary: any) => {
    if (drift) {
      setLastDrift({
        tier: 2,
        event_type: "session_clear",
        timestamp: new Date().toISOString(),
        cosine_distance: drift.cosine_distance,
        edges_processed: drift.edges_processed,
        mood: summary?.dominant_mood ?? null,
        mode: drift.mode ?? "discard",
      });
    }
    setSessionActive(false);
    setSessionMood(null);
    setSessionCount(0);
    fetchGraph();
    fetchDriftHistory();
  };

  const stats = graphData?.stats;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key="chat"
        variants={pageTransition}
        initial="initial"
        animate="animate"
        exit="exit"
        className="h-screen flex flex-col pt-4 overflow-hidden"
      >
        <GlassNav />

        <main className="flex-1 max-w-7xl w-full mx-auto p-4 flex flex-col lg:flex-row gap-6 mt-4 min-h-0 container">
          <aside className="hidden lg:flex w-80 flex-col gap-3 overflow-y-auto">
            <GlassCard className="p-4 flex flex-col gap-3">
              <SessionBadge active={sessionActive} mood={sessionMood} movieCount={sessionCount} />

              <NewMoodButton
                active={sessionActive}
                movieCount={sessionCount}
                mood={sessionMood}
                onCleared={handleNewMoodCleared}
              />

              <GlassButton
                variant="secondary"
                className="w-full"
                onClick={handleReset}
                disabled={isResetting}
              >
                {isResetting ? (
                  <span className="flex items-center gap-2 justify-center">
                    <Loader2 size={14} className="animate-spin" />
                    Resetting…
                  </span>
                ) : (
                  <span className="flex items-center gap-2 justify-center">
                    <Trash2 size={14} />
                    Full Reset
                  </span>
                )}
              </GlassButton>

              <div className="p-3 bg-white/5 rounded-xl border border-white/10 text-xs font-inter space-y-2">
                <div className="text-white/40 uppercase tracking-wider text-[10px] font-medium mb-2">Graph State</div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Liked (permanent)</span>
                  <span className="text-accent-green font-medium tabular-nums">{stats?.liked_count ?? 0}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Session</span>
                  <span className="text-amber-300 font-medium tabular-nums">{sessionCount}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Blocked</span>
                  <span className="text-red-400 font-medium tabular-nums">{stats?.blocked_count ?? graphData?.blocked_count ?? 0}</span>
                </div>

                {stats?.liked_genres && stats.liked_genres.length > 0 && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-white/40 text-[10px] mb-1.5">Boosted genres</div>
                    <div className="flex flex-wrap gap-1">
                      {stats.liked_genres.slice(0, 5).map(g => (
                        <span key={g} className="px-1.5 py-0.5 bg-accent-green/15 text-accent-green rounded text-[10px]">{g}</span>
                      ))}
                    </div>
                  </div>
                )}

                {((stats as any)?.blocked_genres?.length > 0 || stats?.disliked_genres?.length > 0) && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-white/40 text-[10px] mb-1.5">Blocked genres</div>
                    <div className="flex flex-wrap gap-1">
                      {[...new Set<string>([...((stats as any)?.blocked_genres ?? []), ...(stats?.disliked_genres ?? [])])].slice(0, 5).map(g => (
                        <span key={g} className="px-1.5 py-0.5 bg-red-500/15 text-red-400 rounded text-[10px]">{g}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <UnlearningPanel lastEvent={lastDrift} />

              <EmbeddingDriftChart permanent={permanentHistory} session={sessionHistory} />

              <div className="mt-1 pt-3 border-t border-white/10 flex items-center gap-3">
                <div className="relative">
                  <Circle size={12} className={isOnline ? "text-accent-green fill-accent-green" : "text-red-400 fill-red-400"} />
                  {isOnline && <div className="absolute inset-0 bg-accent-green blur-[4px] rounded-full animate-pulse" />}
                </div>
                <span className="text-sm text-white/70">{isOnline ? "MoodLens Online" : "Backend Offline"}</span>
              </div>
            </GlassCard>
          </aside>

          <section className="flex-1 flex flex-col min-w-0">
            <GlassCard className="flex-1 flex flex-col p-2 md:p-4 overflow-hidden relative">
              <div className="flex items-center justify-between mb-3 shrink-0">
                <div className="flex items-center gap-2 text-white/40 text-xs font-inter">
                  <MessageSquare size={13} />
                  <span>MoodLens — Two-Tier Unlearning Recommender</span>
                </div>
                <button
                  onClick={() => setShowVisualizer(true)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-accent-purple/50 text-white/60 hover:text-white transition-all duration-200 text-xs font-inter group"
                >
                  <GitGraph size={13} className="text-accent-purple group-hover:scale-110 transition-transform" />
                  <span>Live Graph</span>
                  <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse ml-0.5" />
                </button>
              </div>

              <ChatWindow messages={messages} isLoading={isLoading} />

              <div className="mt-2 shrink-0">
                <ChatInput
                  input={input}
                  handleInputChange={handleInputChange}
                  handleSubmit={handleSubmit}
                  isLoading={isLoading}
                />
              </div>
            </GlassCard>
          </section>
        </main>

        <AnimatePresence>
          {showVisualizer && (
            <GnnVisualizer graphData={graphData} onClose={() => setShowVisualizer(false)} />
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
}
