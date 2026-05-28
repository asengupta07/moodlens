"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type { ElementType } from "react";
import { GlassNav } from "@/components/ui/GlassNav";
import { motion, AnimatePresence } from "framer-motion";
import { pageTransition } from "@/lib/animations";
import { ChatWindow, ChatMessage } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { GnnVisualizer } from "@/components/chat/GnnVisualizer";
import { SessionBadge } from "@/components/chat/SessionBadge";
import { NewMoodButton } from "@/components/chat/NewMoodButton";
import { UnlearningPanel, DriftEvent } from "@/components/chat/UnlearningPanel";
import { EmbeddingDriftChart } from "@/components/chat/EmbeddingDriftChart";
import { Activity, BrainCircuit, Cpu, GitGraph, Loader2, MessageSquare, Radar, Trash2 } from "lucide-react";

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

function SignalCard({
  icon: Icon,
  label,
  title,
  body,
  color,
}: {
  icon: ElementType;
  label: string;
  title: string;
  body: string;
  color: string;
}) {
  return (
    <section className="mood-panel p-4">
      <div className="flex items-start gap-3">
        <div className="border border-[var(--rule)] p-2" style={{ color }}>
          <Icon size={17} strokeWidth={1.5} />
        </div>
        <div>
          <div className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">{label}</div>
          <div className="mt-1 text-sm font-medium text-[var(--ink)]">{title}</div>
          <p className="mt-2 text-xs leading-6 text-[var(--ink-2)]">{body}</p>
        </div>
      </div>
    </section>
  );
}

function MemoryMeter({ label, value, color, caption }: { label: string; value: number; color: string; caption: string }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3">
        <span className="font-space-grotesk text-[10px] uppercase tracking-[0.1em] text-[var(--clay)]">{label}</span>
        <span className="font-mono text-[11px]" style={{ color }}>{Math.round(pct * 100)}%</span>
      </div>
      <div className="h-2 border border-[var(--rule)] bg-[rgba(242,237,227,0.035)] p-[1px]">
        <div className="h-full" style={{ width: `${pct * 100}%`, background: color }} />
      </div>
      <p className="mt-1 text-[11px] leading-5 text-[var(--clay)]">{caption}</p>
    </div>
  );
}

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
            ? { ...m, content: "Failed to reach the backend. Make sure the Python server is running on port 8000." }
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
            ? { ...m, content: "Connection error. Is the Python backend running on port 8000?" }
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
        reversion_score: drift.reversion_score,
        non_destructive: drift.non_destructive,
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
        className="flex h-screen flex-col overflow-hidden bg-[var(--bone)] text-[var(--ink)]"
      >
        <GlassNav />

        <main className="wrap grid min-h-0 flex-1 gap-5 py-5 lg:grid-cols-[minmax(0,1fr)_420px]">
          <section className="mood-card flex min-h-0 flex-col p-4 md:p-5">
            <div className="mb-4 flex shrink-0 items-center justify-between gap-3 border-b border-[var(--rule)] pb-4">
              <div>
                <div className="flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
                  <MessageSquare size={13} />
                  Conversational recommender
                </div>
                <h1 className="font-display mt-1 text-3xl tracking-[-0.01em] text-[var(--ink)]">
                  MoodLens chat
                </h1>
              </div>
              <button
                onClick={() => setShowVisualizer(true)}
                className="tap-highlight flex items-center gap-2 rounded-full border border-[var(--rule-strong)] px-4 py-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--ink-2)] transition-colors hover:border-[var(--ink)] hover:text-[var(--ink)]"
              >
                <GitGraph size={14} className="text-[var(--wine)]" />
                Live Graph
              </button>
            </div>

            <ChatWindow messages={messages} isLoading={isLoading} />

            <div className="mt-3 shrink-0">
              <ChatInput
                input={input}
                handleInputChange={handleInputChange}
                handleSubmit={handleSubmit}
                isLoading={isLoading}
              />
            </div>
          </section>

          <aside className="hidden min-h-0 flex-col gap-3 overflow-y-auto lg:flex">
            <SignalCard
              icon={BrainCircuit}
              label="Memory model"
              title="Two brains, one conversation"
              body="Blue is long-term taste. Amber is tonight's mood. Red is anything MoodLens has been told to truly forget."
              color="var(--blue)"
            />

            <section className="mood-panel p-4">
              <div className="mb-4 flex items-center justify-between">
                <div className="eyebrow">Control room</div>
                <div className="flex items-center gap-2 text-xs text-[var(--ink-2)]">
                  <span className={`h-2 w-2 rounded-full ${isOnline ? "bg-[var(--green)]" : "bg-[var(--wine)]"}`} />
                  {isOnline ? "Online" : "Offline"}
                </div>
              </div>
              <SessionBadge active={sessionActive} mood={sessionMood} movieCount={sessionCount} />

              <div className="mt-3">
              <NewMoodButton
                active={sessionActive}
                movieCount={sessionCount}
                mood={sessionMood}
                onCleared={handleNewMoodCleared}
              />
              </div>

              <button
                className="tap-highlight mt-3 flex w-full items-center justify-center gap-2 rounded-full border border-[var(--rule)] px-4 py-3 font-space-grotesk text-[11px] uppercase tracking-[0.14em] text-[var(--ink-2)] transition-colors hover:border-[var(--wine)] hover:text-[var(--wine)] disabled:opacity-50"
                onClick={handleReset}
                disabled={isResetting}
              >
                {isResetting ? (
                  <><Loader2 size={14} className="animate-spin" /> Resetting</>
                ) : (
                  <><Trash2 size={14} /> Full Reset</>
                )}
              </button>
            </section>

            <section className="mood-panel p-4 text-xs">
              <div className="mb-3 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
                <Activity size={12} /> Live memory telemetry
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className="border border-[var(--rule)] p-3">
                  <div className="font-display text-3xl text-[var(--blue)]">{stats?.liked_count ?? 0}</div>
                  <div className="mt-1 text-[var(--clay)]">Permanent</div>
                </div>
                <div className="border border-[var(--rule)] p-3">
                  <div className="font-display text-3xl text-[var(--amber)]">{sessionCount}</div>
                  <div className="mt-1 text-[var(--clay)]">Session</div>
                </div>
                <div className="border border-[var(--rule)] p-3">
                  <div className="font-display text-3xl text-[var(--wine)]">{stats?.blocked_count ?? graphData?.blocked_count ?? 0}</div>
                  <div className="mt-1 text-[var(--clay)]">Erased</div>
                </div>
              </div>

              <div className="mt-4 grid gap-4 border-t border-[var(--rule)] pt-4">
                <MemoryMeter
                  label="identity weight"
                  value={Math.min(1, (stats?.liked_count ?? 0) / 8)}
                  color="var(--blue)"
                  caption="How much permanent taste has been captured."
                />
                <MemoryMeter
                  label="mood heat"
                  value={Math.min(1, sessionCount / 8)}
                  color="var(--amber)"
                  caption="How strongly this temporary session is shaping recs."
                />
                <MemoryMeter
                  label="erase pressure"
                  value={Math.min(1, ((stats?.blocked_count ?? graphData?.blocked_count ?? 0) as number) / 8)}
                  color="var(--wine)"
                  caption="Hard blocks and permanent dislikes in force."
                />
              </div>

              {stats?.liked_genres && stats.liked_genres.length > 0 && (
                <div className="mt-4 border-t border-[var(--rule)] pt-3">
                  <div className="mb-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">Boosted genres</div>
                  <div className="flex flex-wrap gap-1.5">
                    {stats.liked_genres.slice(0, 7).map(g => (
                      <span key={g} className="border border-[rgba(120,166,200,0.35)] px-2 py-1 text-[10px] text-[var(--blue)]">{g}</span>
                    ))}
                  </div>
                </div>
              )}

              {((stats as any)?.blocked_genres?.length > 0 || stats?.disliked_genres?.length > 0) && (
                <div className="mt-4 border-t border-[var(--rule)] pt-3">
                  <div className="mb-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">Blocked genres</div>
                  <div className="flex flex-wrap gap-1.5">
                    {[...new Set<string>([...((stats as any)?.blocked_genres ?? []), ...(stats?.disliked_genres ?? [])])].slice(0, 7).map(g => (
                      <span key={g} className="border border-[rgba(216,88,74,0.42)] px-2 py-1 text-[10px] text-[var(--wine)]">{g}</span>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <SignalCard
              icon={Radar}
              label="Plain-English drift"
              title="When the line jumps, the taste map moved"
              body="Embedding drift is not a user-facing score. It is a lab trace: how far the recommender's internal coordinates moved after forgetting."
              color="var(--amber)"
            />

            <UnlearningPanel lastEvent={lastDrift} />

            <EmbeddingDriftChart permanent={permanentHistory} session={sessionHistory} />

            <SignalCard
              icon={Cpu}
              label="Scoring stack"
              title="LightGCN + semantic filters + hard blocks"
              body="Recommendations are scored by graph similarity, plot/metadata relevance, genre weights, and explicit erasure filters."
              color="var(--green)"
            />
          </aside>
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
