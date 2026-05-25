"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import katex from "katex";
import {
  Activity,
  ArrowRight,
  Binary,
  Database,
  FileJson,
  GitBranch,
  type LucideIcon,
  Network,
  Route,
  ShieldAlert,
  Sigma,
  Zap,
} from "lucide-react";
import { GlassNav } from "@/components/ui/GlassNav";
import { GlassButton } from "@/components/ui/GlassButton";

const colors = {
  user: "#88b38b",
  permanent: "#78a6c8",
  session: "#d8a84a",
  genre: "#b38dcc",
  erased: "#d8584a",
  line: "rgba(242,237,227,0.28)",
};

const pipeline = [
  ["01", "Intent parse", "Groq classifies the message as recommendation, session mood, or permanent preference."],
  ["02", "Graph update", "PreferenceGraph persists identity memory; SessionGraph stores temporary mood evidence."],
  ["03", "Score blend", "LightGCN dot products are mixed with Bayesian, semantic, genre, and hard-block signals."],
  ["04", "Stream", "FastAPI streams reply tokens, graph payloads, session state, and unlearning events."],
  ["05", "Unlearn", "GNNDelete handles permanent erasure. Influence functions handle New Mood erasure."],
  ["06", "Evaluate", "The CLI and dashboard measure forgetting, retained quality, attack resistance, and reversion."],
];

const endpointGroups = [
  ["Chat loop", [["POST", "/chat"], ["GET", "/greet"], ["GET", "/graph"], ["GET", "/session"]]],
  ["Unlearning", [["POST", "/new-mood"], ["POST", "/permanent-unlearn"], ["GET", "/embedding-drift"]]],
  ["State", [["GET", "/health"], ["GET", "/state"], ["POST", "/reset"]]],
  ["Evaluation", [["POST", "/api/evaluation/run"], ["GET", "/api/evaluation"]]],
];

const graphNodes = [
  { id: "you", label: "You", type: "user", x: 500, y: 325, r: 16 },
  { id: "s1", label: "Tonight: action", type: "session", x: 620, y: 182, r: 10 },
  { id: "s2", label: "Session movie", type: "session", x: 705, y: 266, r: 9 },
  { id: "s3", label: "Temporary skip", type: "session", x: 690, y: 392, r: 9 },
  { id: "p1", label: "Permanent like", type: "permanent", x: 292, y: 182, r: 10 },
  { id: "p2", label: "Drama signal", type: "permanent", x: 222, y: 320, r: 9 },
  { id: "p3", label: "Liked movie", type: "permanent", x: 306, y: 468, r: 9 },
  { id: "g1", label: "Genre: Drama", type: "genre", x: 430, y: 112, r: 9 },
  { id: "g2", label: "Genre: Action", type: "genre", x: 548, y: 102, r: 9 },
  { id: "g3", label: "Genre: Sci-Fi", type: "genre", x: 390, y: 546, r: 8 },
  { id: "b1", label: "Blocked horror", type: "erased", x: 790, y: 132, r: 9 },
  { id: "b2", label: "Oldies erased", type: "erased", x: 812, y: 495, r: 9 },
];

const graphEdges = [
  ["you", "s1", "session"],
  ["you", "s2", "session"],
  ["you", "s3", "session"],
  ["you", "p1", "permanent"],
  ["you", "p2", "permanent"],
  ["you", "p3", "permanent"],
  ["p1", "g1", "genre"],
  ["p2", "g1", "genre"],
  ["s1", "g2", "genre"],
  ["p3", "g3", "genre"],
  ["you", "b1", "erased"],
  ["you", "b2", "erased"],
];

const metricRows = [
  ["Tier I", "Forget cosine distance", "higher", "Forgotten items should move away from their pre-delete representation."],
  ["Tier I", "Retain top-20 overlap", "higher", "Good recommendations outside the forget set should stay stable."],
  ["Tier I", "Membership inference", "lower", "An attacker should be less able to detect deleted interactions."],
  ["Tier II", "Embedding reversion score", "higher", "The erased session should return the user vector toward its pre-session state."],
  ["Tier II", "Pre/post rank overlap", "higher", "After erase, rankings should resemble the pre-session rankings."],
  ["Tier II", "Mid/post rank overlap", "lower", "After erase, rankings should stop looking like the temporary session."],
];

function latex(markup: string) {
  return katex.renderToString(markup, {
    displayMode: true,
    throwOnError: false,
    strict: false,
    trust: false,
  });
}

function SectionHead({ label, title, body }: { label: string; title: ReactNode; body?: string }) {
  return (
    <div className="grid gap-4 md:grid-cols-[180px_1fr] md:gap-12">
      <div className="font-space-grotesk text-[11px] uppercase tracking-[0.12em] text-[var(--clay)]">{label}</div>
      <div>
        <h2 className="font-display max-w-5xl text-4xl leading-[1.05] tracking-[-0.018em] md:text-6xl">{title}</h2>
        {body && <p className="mt-5 max-w-3xl text-sm leading-7 text-[var(--ink-2)]">{body}</p>}
      </div>
    </div>
  );
}

function LatexBlock({ title, math, caption }: { title: string; math: string; caption?: string }) {
  return (
    <div className="border border-[var(--rule)] bg-[rgba(20,16,8,0.74)] p-5">
      <div className="mb-4 flex items-center gap-2 font-space-grotesk text-[10px] uppercase tracking-[0.13em] text-[var(--clay)]">
        <Sigma size={13} /> {title}
      </div>
      <div
        className="no-scrollbar overflow-x-auto overflow-y-visible py-1 text-[var(--ink)] [&_.katex-display]:my-0 [&_.katex-display]:overflow-x-auto [&_.katex-display]:overflow-y-visible [&_.katex-display]:pb-1 [&_.katex-display]:pt-1 [&_.katex]:text-[1.04em]"
        dangerouslySetInnerHTML={{ __html: latex(math) }}
      />
      {caption && <p className="mt-4 text-xs leading-6 text-[var(--ink-2)]">{caption}</p>}
    </div>
  );
}

function TechCard({ icon: Icon, title, body, color }: { icon: LucideIcon; title: string; body: string; color: string }) {
  return (
    <article className="border border-[var(--rule)] bg-[rgba(242,237,227,0.025)] p-6">
      <Icon size={25} style={{ color }} strokeWidth={1.4} />
      <h3 className="font-display mt-5 text-3xl">{title}</h3>
      <p className="mt-4 text-sm leading-7 text-[var(--ink-2)]">{body}</p>
    </article>
  );
}

function NodeLegend({ label, color, body }: { label: string; color: string; body: string }) {
  return (
    <div className="flex gap-3 border border-[var(--rule)] p-3">
      <span className="mt-1 h-3 w-3 rounded-full border border-black/50" style={{ background: color }} />
      <div>
        <div className="text-sm text-[var(--ink)]">{label}</div>
        <div className="mt-1 text-xs leading-5 text-[var(--clay)]">{body}</div>
      </div>
    </div>
  );
}

function edgePath(source: (typeof graphNodes)[number], target: (typeof graphNodes)[number], bend: number) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const mx = source.x + dx * 0.5;
  const my = source.y + dy * 0.5;
  const normal = Math.sqrt(dx * dx + dy * dy) || 1;
  const cx = mx + (-dy / normal) * bend;
  const cy = my + (dx / normal) * bend;
  return `M ${source.x} ${source.y} Q ${cx} ${cy} ${target.x} ${target.y}`;
}

function MemoryGraph() {
  const nodeById = new Map(graphNodes.map((node) => [node.id, node]));

  return (
    <div className="overflow-hidden border border-[var(--rule)] bg-[rgba(20,16,8,0.78)]">
      <div className="flex items-center justify-between border-b border-[var(--rule)] px-5 py-4">
        <div>
          <div className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">Deterministic example graph</div>
          <div className="font-display mt-1 text-3xl">MoodLens memory map</div>
        </div>
        <div className="hidden font-mono text-xs text-[var(--clay)] sm:block">fixed layout, bounded nodes, no force collapse</div>
      </div>
      <svg viewBox="0 0 1000 650" className="h-[460px] w-full">
        <defs>
          <pattern id="arch-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(242,237,227,0.045)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="1000" height="650" fill="url(#arch-grid)" />
        <circle cx="500" cy="325" r="165" fill="none" stroke="rgba(216,168,74,0.16)" strokeDasharray="7 8" />
        <circle cx="500" cy="325" r="285" fill="none" stroke="rgba(120,166,200,0.13)" />
        <circle cx="500" cy="325" r="335" fill="none" stroke="rgba(216,88,74,0.13)" strokeDasharray="5 8" />

        {graphEdges.map(([a, b, type], index) => {
          const source = nodeById.get(a);
          const target = nodeById.get(b);
          if (!source || !target) return null;
          const stroke = colors[type as keyof typeof colors] ?? colors.line;
          const dash = type === "session" ? "8 8" : type === "erased" ? "4 7" : undefined;
          return (
            <path
              key={`${a}-${b}`}
              d={edgePath(source, target, ((index % 5) - 2) * 8)}
              fill="none"
              stroke={stroke}
              strokeDasharray={dash}
              strokeOpacity={type === "genre" ? 0.42 : 0.72}
              strokeWidth={type === "erased" ? 1.7 : 1.3}
            />
          );
        })}

        {graphNodes.map((node) => {
          const fill = colors[node.type as keyof typeof colors] ?? colors.permanent;
          const isUser = node.type === "user";
          return (
            <g key={node.id}>
              <circle cx={node.x} cy={node.y} r={node.r} fill={fill} stroke="rgba(20,16,8,0.85)" strokeWidth={isUser ? 2.4 : 1.7} />
              {node.type === "erased" && <line x1={node.x - node.r} x2={node.x + node.r} y1={node.y} y2={node.y} stroke="var(--bone)" strokeWidth="2" />}
              {(isUser || node.type === "session" || node.type === "erased") && (
                <g>
                  <rect x={node.x - 62} y={node.y + node.r + 9} width="124" height="24" fill="rgba(20,16,8,0.88)" stroke="rgba(242,237,227,0.13)" />
                  <text x={node.x} y={node.y + node.r + 25} textAnchor="middle" fontSize="11" fill="var(--ink)" fontFamily="Inter, sans-serif">
                    {node.label}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        <text x="28" y="52" fontSize="12" fill="var(--clay)" fontFamily="Inter, sans-serif">inner ring = current mood</text>
        <text x="28" y="76" fontSize="12" fill="var(--clay)" fontFamily="Inter, sans-serif">outer ring = permanent profile and erasures</text>
      </svg>
    </div>
  );
}

function SchemaBlock() {
  return (
    <div className="border border-[var(--rule)] bg-[rgba(242,237,227,0.025)] p-5">
      <div className="font-space-grotesk text-[10px] uppercase tracking-[0.13em] text-[var(--clay)]">Graph objects</div>
      <div className="mt-4 grid gap-3 text-sm leading-6 text-[var(--ink-2)]">
        <p><span className="font-mono text-[var(--ink)]">PreferenceGraph</span> stores long-term likes, hard blocks, genre weights, and permanent unlearning history in <span className="font-mono">backend/user_state.json</span>.</p>
        <p><span className="font-mono text-[var(--ink)]">SessionGraph</span> stores the current mood session in <span className="font-mono">backend/session_state.json</span> and is cleared only when the user presses New Mood.</p>
        <p><span className="font-mono text-[var(--ink)]">VizPayload</span> merges both layers into bounded nodes: user, permanent movie, session movie, genre, and blocked nodes.</p>
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="border border-[var(--rule)] p-3">
      <div className="font-space-grotesk text-[9px] uppercase tracking-[0.12em] text-[var(--clay)]">{label}</div>
      <div className="mt-1 text-sm" style={{ color }}>{value}</div>
    </div>
  );
}

export default function ArchitecturePage() {
  return (
    <main className="min-h-screen bg-[var(--bone)] text-[var(--ink)]">
      <GlassNav />

      <header className="wrap py-16 md:py-20">
        <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl">
          <div className="eyebrow">
            <span className="eyebrow-dot" />
            Technical architecture
          </div>
          <h1 className="font-display mt-7 text-[clamp(46px,7vw,108px)] leading-[0.98] tracking-[-0.025em]">
            Two memory systems. <em className="text-[var(--wine)]">Two erasers.</em> One user-controlled recommender.
          </h1>
          <p className="mt-8 max-w-4xl text-lg leading-8 text-[var(--ink-2)]">
            MoodLens separates identity-level taste from temporary mood. Permanent dislikes are erased from the trained LightGCN graph with GNNDelete; session moods are reversed with influence functions at an explicit New Mood boundary.
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            <Link href="/chat"><GlassButton>Open Chat</GlassButton></Link>
            <Link href="/evaluation"><GlassButton variant="secondary">Run Evaluation</GlassButton></Link>
          </div>
        </motion.div>
      </header>

      <section className="section-band py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="I - Thesis"
            title={<>Forgetting old taste and forgetting tonight are mathematically different operations.</>}
            body="The project claim is architectural: one graph keeps identity-level preference, another graph captures temporary mood, and each layer gets a different unlearning operator."
          />

          <div className="mt-12 grid gap-5 lg:grid-cols-4">
            <TechCard icon={Database} title="TMDB substrate" body="~45k movies, genres, ratings, credits, and plot embeddings define the catalogue layer." color="var(--blue)" />
            <TechCard icon={Network} title="LightGCN graph" body="200 synthetic users, 45,080 movie nodes, 20 genre nodes, 64-dimensional embeddings, and 3 propagation layers." color="var(--green)" />
            <TechCard icon={ShieldAlert} title="GNNDelete tier" body="Permanent dislikes trigger forget-set correction while retaining recommendation quality." color="var(--wine)" />
            <TechCard icon={Zap} title="Influence tier" body="New Mood estimates and reverses the session subgraph's effect on the user embedding." color="var(--amber)" />
          </div>
        </div>
      </section>

      <section className="section-band bg-[#0e0a05] py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="II - Graph Schema"
            title={<>A fixed memory map: permanent taste outside, mood inside, erasures isolated.</>}
            body="The example below follows the rebuilt live graph model: deterministic coordinates, bounded rings, no glow, no physics collapse, and labels only where they help."
          />

          <div className="mt-12 grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
            <MemoryGraph />
            <div className="grid gap-4">
              <div className="border border-[var(--rule)] p-5">
                <div className="font-display text-3xl">Legend</div>
                <div className="mt-5 grid gap-3">
                  <NodeLegend label="User" color={colors.user} body="The active profile embedding used for scoring." />
                  <NodeLegend label="Permanent movie" color={colors.permanent} body="A long-term preference kept across sessions." />
                  <NodeLegend label="Session movie" color={colors.session} body="A temporary mood signal erased or committed by New Mood." />
                  <NodeLegend label="Genre bridge" color={colors.genre} body="Catalogue structure and genre multipliers." />
                  <NodeLegend label="Blocked / erased" color={colors.erased} body="Hard filters and GNNDelete targets." />
                </div>
              </div>
              <SchemaBlock />
            </div>
          </div>

          <div className="mt-8 grid gap-6 lg:grid-cols-2">
            <LatexBlock
              title="Permanent graph"
              math={String.raw`G_{\mathrm{perm}}=(V_u \cup V_m \cup V_g,\; E_{\mathrm{like}} \cup E_{\mathrm{genre}} \cup E_{\mathrm{block}})`}
              caption="This graph represents identity-level preference. It persists and is the target of Tier I permanent unlearning."
            />
            <LatexBlock
              title="Session graph"
              math={String.raw`G_{\mathrm{sess}}^{(t)}=(V_u \cup V_{m,t},\; E_{\mathrm{sess}}^{(t)}),\qquad G_{\mathrm{sess}}\xrightarrow{\mathrm{New\ Mood}}\varnothing`}
              caption="The session graph exists only inside the current mood context. It is explicitly cleared or committed by the user."
            />
          </div>
        </div>
      </section>

      <section className="section-band py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="III - LightGCN"
            title={<>The base recommender is deliberately inspectable.</>}
            body="LightGCN is used because propagation is linear graph smoothing without feature transforms. That makes embedding movement and unlearning effects easier to explain, measure, and defend."
          />

          <div className="mt-12 grid gap-6 lg:grid-cols-3">
            <LatexBlock
              title="Propagation"
              math={String.raw`\mathbf{E}^{(0)}=\mathbf{Q},\qquad \mathbf{E}^{(k+1)}=\mathbf{D}^{-\frac12}\mathbf{A}\mathbf{D}^{-\frac12}\mathbf{E}^{(k)}`}
              caption="A is the normalized user-movie-genre graph adjacency. Q is the trainable base embedding table."
            />
            <LatexBlock
              title="Layer aggregation"
              math={String.raw`\mathbf{E}^{*}=\frac{1}{K+1}\sum_{k=0}^{K}\mathbf{E}^{(k)},\qquad K=3,\quad d=64`}
              caption="The final embedding is the mean of layer 0 through layer 3. The checkpoint stores 64-dimensional vectors."
            />
            <LatexBlock
              title="BPR training objective"
              math={String.raw`\mathcal{L}_{\mathrm{BPR}}=-\log\sigma(\hat{y}_{ui}-\hat{y}_{uj})+\lambda\|\Theta\|_2^2,\qquad \hat{y}_{ui}=\mathbf{e}_u^\top\mathbf{e}_i`}
              caption="For user u, positive movie i, and negative movie j, the model learns to rank i above j."
            />
          </div>

          <div className="mt-8 grid gap-5 md:grid-cols-3">
            <TechCard icon={Binary} title="Checkpoint facts" body="The current trained checkpoint contains 200 users, 45,080 movies, 20 genres, and 90,367 directed graph edges." color="var(--green)" />
            <TechCard icon={GitBranch} title="Accessible embeddings" body="The model exposes user, movie, and genre embedding matrices directly for unlearning and drift visualization." color="var(--purple)" />
            <TechCard icon={Activity} title="Fallback mode" body="If the checkpoint is missing, the backend logs a warning and uses the existing recommender path instead of crashing." color="var(--amber)" />
          </div>
        </div>
      </section>

      <section className="section-band bg-[#0e0a05] py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="IV - Scoring"
            title={<>Recommendations are graph-aware, semantic-aware, and still hard-filtered.</>}
            body="LightGCN adds collaborative graph signal, while the older Bayesian and semantic scoring keeps the product useful when the graph is sparse or unavailable."
          />

          <div className="mt-12 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
            <LatexBlock
              title="Final movie score"
              math={String.raw`s(m)=\left(\alpha\,\mathbf{e}_u^\top\mathbf{e}_m+(1-\alpha)\,s_{\mathrm{base}}(m)\right)\cdot \prod_{g\in \Gamma_m} w_g\cdot p_{\mathrm{sess}}(m)`}
              caption="alpha is 0.6 when LightGCN is loaded and 0.0 in fallback mode. Hard-blocked movies and genres bypass this formula and score exactly zero."
            />
            <LatexBlock
              title="Session repeat penalty"
              math={String.raw`p_{\mathrm{sess}}(m)=\begin{cases}0.3,&m\in V_{m,t}\\1.0,&m\notin V_{m,t}\end{cases}`}
              caption="The current mood session should guide the conversation without repeatedly recommending the same items."
            />
          </div>
        </div>
      </section>

      <section className="section-band py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="V - Two Unlearning Tiers"
            title={<>Tier I changes learned memory. Tier II reverses temporary influence.</>}
          />

          <div className="mt-12 grid gap-6 lg:grid-cols-2">
            <div className="border border-[var(--rule)] p-7">
              <div className="flex items-center gap-3">
                <ShieldAlert className="text-[var(--wine)]" size={28} strokeWidth={1.4} />
                <div>
                  <div className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">Tier I - permanent</div>
                  <h3 className="font-display text-4xl">GNNDelete</h3>
                </div>
              </div>
              <p className="mt-5 text-sm leading-7 text-[var(--ink-2)]">
                Used for "never", "forever", "block", and identity-level dislikes. The forget set is removed from influence while retained graph utility is protected.
              </p>
              <div className="mt-5">
                <LatexBlock
                  title="Delete correction"
                  math={String.raw`\theta'=\theta-\eta\nabla_\theta\mathcal{L}_{\mathrm{retain}}(\theta;R)+\eta\nabla_\theta\mathcal{L}_{\mathrm{forget}}(\theta;F)`}
                  caption="The retain gradient preserves useful recommendations. The forget gradient weakens memorization of the forget set F."
                />
              </div>
              <div className="mt-5 grid grid-cols-3 gap-2 text-xs">
                <Stat label="scope" value="global" color="var(--wine)" />
                <Stat label="trigger" value="forever" color="var(--wine)" />
                <Stat label="state" value="persistent" color="var(--wine)" />
              </div>
            </div>

            <div className="border border-[var(--rule)] p-7">
              <div className="flex items-center gap-3">
                <Zap className="text-[var(--amber)]" size={28} strokeWidth={1.4} />
                <div>
                  <div className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">Tier II - session</div>
                  <h3 className="font-display text-4xl">Influence functions</h3>
                </div>
              </div>
              <p className="mt-5 text-sm leading-7 text-[var(--ink-2)]">
                Used only when the user presses New Mood. It estimates how the session interactions shifted the user embedding and subtracts that influence if the user discards the mood.
              </p>
              <div className="mt-5">
                <LatexBlock
                  title="Session influence"
                  math={String.raw`\mathcal{I}_{\mathrm{sess}}\approx \mathbf{H}_{\theta}^{-1}\nabla_{\theta}\mathcal{L}_{\mathrm{sess}},\qquad \mathbf{e}_{u,\mathrm{after}}=\mathbf{e}_{u,\mathrm{mid}}-\beta\mathcal{I}_{\mathrm{sess}}`}
                  caption="The inverse Hessian-vector product is approximated so the app can reverse session influence without full retraining."
                />
              </div>
              <div className="mt-5 grid grid-cols-3 gap-2 text-xs">
                <Stat label="scope" value="local" color="var(--amber)" />
                <Stat label="trigger" value="New Mood" color="var(--amber)" />
                <Stat label="state" value="ephemeral" color="var(--amber)" />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="section-band bg-[#0e0a05] py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="VI - Request Flow"
            title={<>Every chat turn streams text, state, graph updates, and unlearning events.</>}
          />

          <div className="mt-12 grid gap-px overflow-hidden border border-[var(--rule)] bg-[var(--rule)] md:grid-cols-3">
            {pipeline.map(([n, title, body]) => (
              <article key={n} className="bg-[var(--bone)] p-7">
                <div className="font-display text-5xl text-[rgba(216,88,74,0.42)]">{n}</div>
                <h3 className="mt-5 font-display text-3xl">{title}</h3>
                <p className="mt-4 text-sm leading-6 text-[var(--ink-2)]">{body}</p>
              </article>
            ))}
          </div>

          <div className="mt-10 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="grid gap-4">
              <TechCard icon={Route} title="SSE contract" body="The backend emits token, graph, session, and unlearn events so the UI can update progressively without a full refresh." color="var(--blue)" />
              <TechCard icon={FileJson} title="State files" body="user_state.json is durable identity memory; session_state.json is temporary mood memory." color="var(--amber)" />
            </div>
            <div className="grid gap-4">
              {endpointGroups.map(([group, eps]) => (
                <div key={String(group)} className="border border-[var(--rule)]">
                  <div className="border-b border-[var(--rule)] px-5 py-3 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">{String(group)}</div>
                  <div className="grid gap-px bg-[var(--rule)] sm:grid-cols-2">
                    {(eps as string[][]).map(([method, path]) => (
                      <div key={path} className="flex items-center gap-4 bg-[var(--bone-2)] px-5 py-4">
                        <span className={`w-11 font-mono text-[11px] ${method === "GET" ? "text-[var(--green)]" : "text-[var(--wine)]"}`}>{method}</span>
                        <span className="font-mono text-sm text-[var(--ink-2)]">{path}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="section-band py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="VII - Evaluation"
            title={<>The demo is measurable, not just visual.</>}
            body="The evaluation suite generates metrics.json and the dashboard turns those numbers into human-readable evidence."
          />

          <div className="mt-12 overflow-hidden border border-[var(--rule)]">
            <div className="grid grid-cols-[0.8fr_1.3fr_0.7fr_1.4fr] gap-px bg-[var(--rule)] text-sm">
              <div className="bg-[var(--bone-3)] p-4 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">Tier</div>
              <div className="bg-[var(--bone-3)] p-4 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">Metric</div>
              <div className="bg-[var(--bone-3)] p-4 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">Goal</div>
              <div className="bg-[var(--bone-3)] p-4 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">Meaning</div>
              {metricRows.map(([tier, metric, goal, meaning]) => (
                <div key={`${tier}-${metric}`} className="contents">
                  <div className="bg-[var(--bone)] p-4 text-[var(--ink)]">{tier}</div>
                  <div className="bg-[var(--bone)] p-4 text-[var(--ink-2)]">{metric}</div>
                  <div className="bg-[var(--bone)] p-4 text-[var(--amber)]">{goal}</div>
                  <div className="bg-[var(--bone)] p-4 text-[var(--ink-2)]">{meaning}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-8 grid gap-6 lg:grid-cols-3">
            <LatexBlock
              title="Cosine distance"
              math={String.raw`d_{\cos}(\mathbf{x},\mathbf{y})=1-\frac{\mathbf{x}^{\top}\mathbf{y}}{\|\mathbf{x}\|_2\|\mathbf{y}\|_2}`}
              caption="Used for embedding drift and forget movement."
            />
            <LatexBlock
              title="Rank overlap"
              math={String.raw`\mathrm{overlap}@k(A,B)=\frac{|A_{1:k}\cap B_{1:k}|}{k}`}
              caption="Used to compare recommendation lists before, during, and after unlearning."
            />
            <LatexBlock
              title="Reversion score"
              math={String.raw`\mathrm{rev}=1-d_{\cos}(\mathbf{e}_{u,\mathrm{pre}},\mathbf{e}_{u,\mathrm{after}})`}
              caption="Higher means the erased session returned closer to the original user embedding."
            />
          </div>
        </div>
      </section>

      <section className="section-band bg-[#0e0a05] py-16">
        <div className="wrap grid gap-6 lg:grid-cols-[1fr_1fr]">
          <div>
            <div className="eyebrow">Design note</div>
            <h2 className="font-display mt-4 text-5xl leading-tight">The user controls when mood ends.</h2>
            <p className="mt-5 text-sm leading-7 text-[var(--ink-2)]">
              MoodLens deliberately avoids app-close or time-based session endings. New Mood is an explicit consent boundary: the user decides whether to keep or discard the session influence, and the system gets a clean before/after evaluation point.
            </p>
            <div className="mt-7">
              <Link href="/evaluation" className="inline-flex items-center gap-2 text-sm text-[var(--amber)]">
                Inspect the evaluation dashboard <ArrowRight size={15} />
              </Link>
            </div>
          </div>
          <LatexBlock
            title="Consent boundary"
            math={String.raw`\mathrm{NewMood}=\arg\min_{\mathrm{boundary}}\left(\mathrm{ambiguity}\right)\quad\text{subject to explicit user intent}`}
            caption="A manual New Mood trigger is academically cleaner than guessing from timeouts, tab close, or inactivity."
          />
        </div>
      </section>
    </main>
  );
}
