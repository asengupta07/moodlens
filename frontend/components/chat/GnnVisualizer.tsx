"use client";

import { useMemo, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { Brain, CircleHelp, Network, RotateCcw, X } from "lucide-react";

interface GraphNode {
  id: string;
  label: string;
  type:
    | "user"
    | "recommended"
    | "liked"
    | "erased"
    | "genre"
    | "movie_liked"
    | "movie_session"
    | "movie_blocked"
    | "genre_blocked";
  weight: number;
  interaction?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  type:
    | "recommends"
    | "liked"
    | "erased"
    | "has_genre"
    | "permanent_like"
    | "session_interaction"
    | "blocked"
    | "genre_weight";
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
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

interface Props {
  graphData: GraphData | null;
  onClose: () => void;
}

type LayoutNode = GraphNode & {
  x: number;
  y: number;
  r: number;
};

const W = 980;
const H = 620;
const USER = { x: 490, y: 315 };

const NODE_CONFIG: Record<string, { color: string; radius: number; label: string; explain: string }> = {
  user: { color: "#88b38b", radius: 11, label: "You", explain: "The active user profile." },
  recommended: { color: "#b38dcc", radius: 7, label: "Recommendation", explain: "Movie currently being scored." },
  liked: { color: "#78a6c8", radius: 7, label: "Permanent like", explain: "Long-term taste signal." },
  movie_liked: { color: "#78a6c8", radius: 7, label: "Permanent like", explain: "Long-term taste signal." },
  movie_session: { color: "#d8a84a", radius: 7, label: "Session movie", explain: "Temporary mood signal." },
  movie_blocked: { color: "#d8584a", radius: 7, label: "Erased / blocked", explain: "Hard filter or unlearn target." },
  erased: { color: "#d8584a", radius: 7, label: "Erased / blocked", explain: "Hard filter or unlearn target." },
  genre: { color: "#b38dcc", radius: 6, label: "Genre", explain: "Genre bridge node." },
  genre_blocked: { color: "#d8584a", radius: 6, label: "Blocked genre", explain: "Genre-level permanent dislike." },
};

const EDGE_CONFIG: Record<string, { color: string; dash?: string; opacity: number }> = {
  recommends: { color: "#b38dcc", opacity: 0.52 },
  liked: { color: "#78a6c8", opacity: 0.58 },
  permanent_like: { color: "#78a6c8", opacity: 0.58 },
  session_interaction: { color: "#d8a84a", dash: "7 7", opacity: 0.72 },
  blocked: { color: "#d8584a", dash: "4 7", opacity: 0.72 },
  erased: { color: "#d8584a", dash: "4 7", opacity: 0.72 },
  has_genre: { color: "#9a8f7c", opacity: 0.36 },
  genre_weight: { color: "#b38dcc", opacity: 0.36 },
};

function groupFor(type: string) {
  if (type === "user") return "user";
  if (type === "movie_session") return "session";
  if (type === "movie_blocked" || type === "erased" || type === "genre_blocked") return "blocked";
  if (type === "genre") return "genre";
  return "permanent";
}

function truncate(label: string, len = 28) {
  return label.length > len ? `${label.slice(0, len)}...` : label;
}

function placeArc(nodes: GraphNode[], center: { x: number; y: number }, radius: number, start: number, end: number): LayoutNode[] {
  if (nodes.length === 0) return [];
  if (nodes.length === 1) {
    const cfg = NODE_CONFIG[nodes[0].type] ?? NODE_CONFIG.recommended;
    const a = (start + end) / 2;
    return [{ ...nodes[0], x: center.x + Math.cos(a) * radius, y: center.y + Math.sin(a) * radius, r: cfg.radius }];
  }
  return nodes.map((node, i) => {
    const cfg = NODE_CONFIG[node.type] ?? NODE_CONFIG.recommended;
    const t = i / (nodes.length - 1);
    const a = start + (end - start) * t;
    return {
      ...node,
      x: center.x + Math.cos(a) * radius,
      y: center.y + Math.sin(a) * radius,
      r: cfg.radius,
    };
  });
}

function buildLayout(data: GraphData | null) {
  const all = data?.nodes ?? [];
  const userNode = all.find((n) => n.type === "user") ?? { id: "user", label: "You", type: "user" as const, weight: 1 };
  const limited = {
    session: all.filter((n) => groupFor(n.type) === "session").slice(0, 18),
    permanent: all.filter((n) => groupFor(n.type) === "permanent").slice(0, 16),
    genre: all.filter((n) => groupFor(n.type) === "genre").slice(0, 12),
    blocked: all.filter((n) => groupFor(n.type) === "blocked").slice(0, 12),
  };

  const nodes: LayoutNode[] = [
    { ...userNode, x: USER.x, y: USER.y, r: NODE_CONFIG.user.radius },
    ...placeArc(limited.session, USER, 185, -1.45, 1.45),
    ...placeArc(limited.permanent, USER, 280, 2.35, 3.95),
    ...placeArc(limited.genre, USER, 255, -2.3, -0.85),
    ...placeArc(limited.blocked, USER, 310, 0.9, 2.1),
  ];

  const ids = new Set(nodes.map((n) => n.id));
  const edges = (data?.edges ?? [])
    .filter((edge) => ids.has(String(edge.source)) && ids.has(String(edge.target)))
    .slice(0, 90);

  return { nodes, edges };
}

function edgePath(source: LayoutNode, target: LayoutNode, index: number) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const mx = source.x + dx * 0.5;
  const my = source.y + dy * 0.5;
  const normal = Math.sqrt(dx * dx + dy * dy) || 1;
  const curve = ((index % 5) - 2) * 7;
  const cx = mx + (-dy / normal) * curve;
  const cy = my + (dx / normal) * curve;
  return `M ${source.x} ${source.y} Q ${cx} ${cy} ${target.x} ${target.y}`;
}

export function GnnVisualizer({ graphData, onClose }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const layout = useMemo(() => buildLayout(graphData), [graphData]);
  const nodeMap = useMemo(() => new Map(layout.nodes.map((n) => [n.id, n])), [layout.nodes]);
  const selected = selectedId ? nodeMap.get(selectedId) ?? null : null;
  const stats = graphData?.stats;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <motion.div
        initial={{ scale: 0.98, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.98, opacity: 0 }}
        transition={{ duration: 0.18 }}
        className="grid h-[88vh] w-full max-w-7xl overflow-hidden border border-[var(--rule)] bg-[var(--bone)] lg:grid-cols-[minmax(0,1fr)_340px]"
      >
        <section className="grid min-h-0 grid-rows-[112px_1fr]">
          <header className="flex items-center justify-between border-b border-[var(--rule)] px-7">
            <div>
              <div className="flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
                <Network size={13} /> Deterministic memory map
              </div>
              <h2 className="font-display mt-1 text-4xl text-[var(--ink)]">MoodLens live graph</h2>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSelectedId(null)}
                className="rounded-full border border-[var(--rule)] p-2 text-[var(--clay)] transition-colors hover:text-[var(--ink)]"
                title="Clear selection"
              >
                <RotateCcw size={16} />
              </button>
              <button
                onClick={onClose}
                className="rounded-full border border-[var(--rule)] p-2 text-[var(--clay)] transition-colors hover:text-[var(--ink)]"
                title="Close"
              >
                <X size={16} />
              </button>
            </div>
          </header>

          <div className="relative min-h-0 overflow-hidden">
            {layout.nodes.length > 1 ? (
              <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full">
                <defs>
                  <pattern id="graph-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(242,237,227,0.045)" strokeWidth="1" />
                  </pattern>
                </defs>
                <rect width={W} height={H} fill="url(#graph-grid)" />
                <circle cx={USER.x} cy={USER.y} r="185" fill="none" stroke="rgba(216,168,74,0.12)" strokeDasharray="6 9" />
                <circle cx={USER.x} cy={USER.y} r="280" fill="none" stroke="rgba(120,166,200,0.10)" />
                <circle cx={USER.x} cy={USER.y} r="310" fill="none" stroke="rgba(216,88,74,0.10)" strokeDasharray="4 8" />

                {layout.edges.map((edge, index) => {
                  const src = nodeMap.get(String(edge.source));
                  const tgt = nodeMap.get(String(edge.target));
                  if (!src || !tgt) return null;
                  const cfg = EDGE_CONFIG[edge.type] ?? EDGE_CONFIG.recommends;
                  return (
                    <path
                      key={`${edge.source}-${edge.target}-${index}`}
                      d={edgePath(src, tgt, index)}
                      fill="none"
                      stroke={cfg.color}
                      strokeOpacity={cfg.opacity}
                      strokeWidth={Math.max(0.8, Math.min(2.4, edge.weight * 1.2))}
                      strokeDasharray={cfg.dash}
                    />
                  );
                })}

                {layout.nodes.map((node) => {
                  const cfg = NODE_CONFIG[node.type] ?? NODE_CONFIG.recommended;
                  const selected = selectedId === node.id;
                  const showLabel = node.type === "user" || selected;
                  return (
                    <g
                      key={node.id}
                      onMouseEnter={() => setSelectedId(node.id)}
                      onFocus={() => setSelectedId(node.id)}
                      onClick={() => setSelectedId(node.id)}
                      className="cursor-pointer"
                      tabIndex={0}
                    >
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={selected ? node.r + 3 : node.r}
                        fill={cfg.color}
                        stroke={selected ? "var(--ink)" : "rgba(20,16,8,0.78)"}
                        strokeWidth={selected ? 2 : 1.5}
                      />
                      {(node.type === "movie_blocked" || node.type === "erased" || node.type === "genre_blocked") && (
                        <line
                          x1={node.x - node.r}
                          x2={node.x + node.r}
                          y1={node.y}
                          y2={node.y}
                          stroke="var(--bone)"
                          strokeWidth="2"
                        />
                      )}
                      {showLabel && (
                        <g>
                          <rect
                            x={node.x - 62}
                            y={node.y + node.r + 8}
                            width="124"
                            height="22"
                            fill="rgba(20,16,8,0.86)"
                            stroke="rgba(242,237,227,0.14)"
                          />
                          <text
                            x={node.x}
                            y={node.y + node.r + 23}
                            textAnchor="middle"
                            fontSize="11"
                            fill="var(--ink)"
                            fontFamily="Inter, sans-serif"
                          >
                            {truncate(node.label, 18)}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>
            ) : (
              <div className="flex h-full items-center justify-center text-center text-[var(--clay)]">
                <div>
                  <Brain className="mx-auto mb-4 text-[var(--wine)]" size={38} strokeWidth={1.3} />
                  <p className="font-display text-3xl text-[var(--ink)]">No graph yet.</p>
                  <p className="mt-2 text-sm">Start chatting and recommendations will appear as nodes.</p>
                </div>
              </div>
            )}
          </div>
        </section>

        <aside className="min-h-0 overflow-y-auto border-l border-[var(--rule)] bg-[rgba(28,24,18,0.94)] p-5">
          <div className="eyebrow">How to read it</div>
          <p className="mt-4 text-sm leading-7 text-[var(--ink-2)]">
            This is no longer a physics graph. It is a fixed memory map: amber mood nodes stay in the inner ring, blue permanent taste nodes sit farther out, purple genres form bridges, and red erased items stay isolated.
          </p>

          <div className="mt-6 grid gap-2">
            {["user", "movie_session", "movie_liked", "genre", "movie_blocked"].map((type) => {
              const cfg = NODE_CONFIG[type];
              return (
                <div key={type} className="flex gap-3 border border-[var(--rule)] p-3">
                  <span className="mt-1 h-3 w-3 rounded-full border border-[rgba(20,16,8,0.7)]" style={{ background: cfg.color }} />
                  <div>
                    <div className="text-sm text-[var(--ink)]">{cfg.label}</div>
                    <div className="mt-1 text-xs leading-5 text-[var(--clay)]">{cfg.explain}</div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="mt-6 grid grid-cols-3 gap-2">
            <MiniStat label="nodes" value={layout.nodes.length} />
            <MiniStat label="edges" value={layout.edges.length} />
            <MiniStat label="session" value={graphData?.session_count ?? 0} />
          </div>

          <div className="mt-6 border border-[var(--rule)] p-4">
            <div className="flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
              <CircleHelp size={12} /> Selected node
            </div>
            {selected ? (
              <div className="mt-3">
                <div className="text-lg leading-6 text-[var(--ink)]">{selected.label}</div>
                <div className="mt-2 text-sm text-[var(--clay)]">{NODE_CONFIG[selected.type]?.label ?? selected.type}</div>
                <p className="mt-3 text-xs leading-6 text-[var(--ink-2)]">{NODE_CONFIG[selected.type]?.explain}</p>
              </div>
            ) : (
              <p className="mt-3 text-sm leading-6 text-[var(--ink-2)]">Hover or click a node to inspect what part of memory it represents.</p>
            )}
          </div>

          {stats && (
            <div className="mt-6 border border-[var(--rule)] p-4">
              <div className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">Profile state</div>
              <div className="mt-3 grid gap-2 text-xs">
                <Row label="Permanent likes" value={stats.liked_count ?? 0} color="var(--blue)" />
                <Row label="Hard blocks" value={stats.blocked_count ?? graphData?.blocked_count ?? 0} color="var(--wine)" />
                <Row label="Mood active" value={graphData?.session_active ? "yes" : "no"} color="var(--amber)" />
              </div>
            </div>
          )}
        </aside>
      </motion.div>
    </motion.div>
  );
}

function MiniStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="border border-[var(--rule)] p-3 text-center">
      <div className="font-display text-3xl text-[var(--ink)]">{value}</div>
      <div className="font-space-grotesk text-[9px] uppercase tracking-[0.12em] text-[var(--clay)]">{label}</div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: ReactNode; color: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-[var(--ink-2)]">{label}</span>
      <span className="font-mono" style={{ color }}>{value}</span>
    </div>
  );
}
