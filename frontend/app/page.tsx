"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import type { Variants } from "framer-motion";
import { GlassNav } from "@/components/ui/GlassNav";
import { GlassButton } from "@/components/ui/GlassButton";
import { ArrowRight, GitBranch, MessageSquare, RefreshCw } from "lucide-react";

const fade: Variants = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] as const } },
};

const tierRows = [
  {
    label: "Mechanism",
    identity: "GNNDelete with forget-set ascent and retain-set preservation.",
    mood: "Influence functions that reverse the session embedding shift.",
  },
  {
    label: "Scope",
    identity: "Global, persistent, applied to future recommendations.",
    mood: "Local to the current mood graph and explicitly cleared.",
  },
  {
    label: "Trigger",
    identity: "Words like never, forever, block, remove completely.",
    mood: "The New Mood button or language such as tonight or for now.",
  },
];

function SectionHead({ label, title }: { label: string; title: React.ReactNode }) {
  return (
    <div className="grid gap-4 md:grid-cols-[160px_1fr] md:gap-12">
      <div className="font-space-grotesk text-[11px] uppercase tracking-[0.12em] text-[var(--clay)]">
        {label}
      </div>
      <h2 className="font-display max-w-4xl text-4xl leading-[1.04] tracking-[-0.018em] text-[var(--ink)] md:text-6xl">
        {title}
      </h2>
    </div>
  );
}

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-[var(--bone)] text-[var(--ink)]">
      <GlassNav />

      <header className="wrap py-16 md:py-20">
        <motion.div initial="hidden" animate="visible" variants={fade}>
          <div className="eyebrow">
            <span className="eyebrow-dot" />
            A conversational recommender with a memory it knows how to forget
          </div>
          <h1 className="font-display mt-9 max-w-6xl text-[clamp(48px,7.4vw,116px)] leading-[0.98] tracking-[-0.022em]">
            Your taste is <em className="text-[var(--wine)]">yours</em>.
            <br />
            Your mood is temporary.
            <br />
            <span className="italic text-[var(--clay)]">The algorithm should know the difference.</span>
          </h1>
          <div className="mt-10 grid items-end gap-9 md:grid-cols-[1.1fr_1fr]">
            <p className="font-display max-w-xl text-[22px] italic leading-[1.4] text-[var(--ink-2)]">
              MoodLens is built on <span className="text-[var(--wine)]">LightGCN</span> with two-tier machine unlearning:
              permanent erasure for identity-level dislikes, and session forgetting for the weather of tonight.
            </p>
            <div className="grid gap-6 text-sm leading-6 text-[var(--ink-2)] sm:grid-cols-2">
              <div>
                <span className="mb-2 block font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
                  Foundation
                </span>
                LightGCN graph collaborative filtering over the user and movie bipartite graph.
              </div>
              <div>
                <span className="mb-2 block font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
                  Forgetting, two ways
                </span>
                GNNDelete for identity. Influence functions for mood.
              </div>
            </div>
          </div>
          <div className="mt-12 flex flex-wrap items-center gap-3">
            <Link href="/chat">
              <GlassButton size="lg">
                Open the demo <ArrowRight size={15} />
              </GlassButton>
            </Link>
            <Link href="/evaluation">
              <GlassButton size="lg" variant="secondary">
                Run evaluation
              </GlassButton>
            </Link>
          </div>
          <div className="mt-14 flex flex-wrap gap-6 border-t border-[var(--rule)] pt-6 font-space-grotesk text-[11px] uppercase tracking-[0.08em] text-[var(--clay)]">
            <span>001 - Recommender</span>
            <span>002 - Conversational</span>
            <span>003 - Principled unlearning</span>
            <span className="md:ml-auto text-[var(--ink-2)]">MoodLens - 2026</span>
          </div>
        </motion.div>
      </header>

      <section id="thesis" className="section-band wrap py-16 md:py-24">
        <SectionHead
          label="I - Thesis"
          title={
            <>
              Every major unlearning paper treats all forgetting as the same operation.{" "}
              <em className="text-[var(--wine)]">It is not.</em>
            </>
          }
        />
        <p className="font-display mt-12 max-w-5xl text-[clamp(24px,3vw,40px)] leading-[1.22] tracking-[-0.012em] text-[var(--ink)]">
          There is a difference between <strong className="font-normal italic text-[var(--wine)]">who you are</strong> and{" "}
          <strong className="font-normal italic text-[var(--wine)]">what you felt like tonight</strong>. MoodLens turns that
          distinction into a working recommender you can inspect, reset, evaluate, and argue for.
        </p>
      </section>

      <section id="two-tier" className="section-band bg-[#0e0a05] py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="II - Two Tiers"
            title={
              <>
                One model. <em className="text-[var(--wine)]">Two</em> kinds of forgetting.
              </>
            }
          />
          <div className="mt-14 grid gap-10 lg:grid-cols-[1fr_1px_1fr]">
            <article>
              <div className="flex items-baseline justify-between gap-4">
                <span className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
                  Tier I - Permanent
                </span>
                <span className="font-space-grotesk text-[11px] text-[var(--clay)]">01 / 02</span>
              </div>
              <h3 className="font-display mt-8 text-6xl tracking-[-0.02em] md:text-7xl">Identity.</h3>
              <p className="mt-6 max-w-md text-[15px] leading-7 text-[var(--ink-2)]">
                For oldies you never want, horror you will never choose, or a specific title that should be erased from the graph.
              </p>
              <div className="mt-8 border-t border-[var(--rule)]">
                {tierRows.map((row) => (
                  <div key={row.label} className="grid gap-3 border-b border-[var(--rule)] py-4 text-sm md:grid-cols-[130px_1fr]">
                    <span className="font-space-grotesk text-[11px] uppercase tracking-[0.08em] text-[var(--clay)]">
                      {row.label}
                    </span>
                    <span className="leading-6 text-[var(--ink-2)]">{row.identity}</span>
                  </div>
                ))}
              </div>
              <div className="mt-10 aspect-[5/3] overflow-hidden border border-[var(--rule)] bg-[var(--bone-2)] [background-image:repeating-linear-gradient(90deg,transparent_0_11px,rgba(242,237,227,0.07)_11px_12px),repeating-linear-gradient(0deg,transparent_0_11px,rgba(242,237,227,0.05)_11px_12px)]">
                <div className="flex h-full items-center justify-center">
                  <div className="h-[38%] aspect-square rounded-full bg-[var(--ink)] shadow-[inset_0_0_0_8px_var(--bone),inset_0_0_0_9px_var(--ink)]" />
                </div>
              </div>
            </article>

            <div className="hidden bg-[var(--rule)] lg:block" />

            <article>
              <div className="flex items-baseline justify-between gap-4">
                <span className="font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--clay)]">
                  Tier II - Session
                </span>
                <span className="font-space-grotesk text-[11px] text-[var(--clay)]">02 / 02</span>
              </div>
              <h3 className="font-display mt-8 text-6xl italic tracking-[-0.02em] text-[var(--wine)] md:text-7xl">Mood.</h3>
              <p className="mt-6 max-w-md text-[15px] leading-7 text-[var(--ink-2)]">
                For tonight. The model absorbs a temporary direction, then the New Mood action clears the session graph without rewriting identity.
              </p>
              <div className="mt-8 border-t border-[var(--rule)]">
                {tierRows.map((row) => (
                  <div key={row.label} className="grid gap-3 border-b border-[var(--rule)] py-4 text-sm md:grid-cols-[130px_1fr]">
                    <span className="font-space-grotesk text-[11px] uppercase tracking-[0.08em] text-[var(--clay)]">
                      {row.label}
                    </span>
                    <span className="leading-6 text-[var(--ink-2)]">{row.mood}</span>
                  </div>
                ))}
              </div>
              <div className="relative mt-10 aspect-[5/3] overflow-hidden border border-[var(--rule)] bg-[var(--bone-2)]">
                {[84, 62, 42, 24].map((w, i) => (
                  <span
                    key={w}
                    className="absolute left-1/2 top-1/2 aspect-square -translate-x-1/2 -translate-y-1/2 rounded-full border border-[var(--wine)]"
                    style={{
                      width: `${w}%`,
                      opacity: [0.16, 0.32, 0.55, 1][i],
                      borderStyle: i === 1 ? "dashed" : i === 0 ? "dotted" : "solid",
                      background: i === 3 ? "var(--wine)" : "transparent",
                    }}
                  />
                ))}
              </div>
            </article>
          </div>
        </div>
      </section>

      <section id="conversation" className="section-band wrap py-16 md:py-24">
        <SectionHead
          label="III - Conversation"
          title={
            <>
              You talk to it. <em className="text-[var(--wine)]">It listens differently</em> depending on what kind of thing you said.
            </>
          }
        />
        <div className="mt-14 grid gap-12 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="max-w-md">
            <h3 className="font-display text-3xl italic leading-tight">The interface is chat. The research lives underneath.</h3>
            <p className="mt-5 text-[15px] leading-7 text-[var(--ink-2)]">
              Permanent language routes to Tier I. Tonight-only language routes to Tier II. The button is the explicit user-owned session boundary.
            </p>
            <div className="mt-8 grid gap-4 text-sm leading-6 text-[var(--ink-2)]">
              {[
                ["A", "Permanent signal", "Edges are removed from the retained graph."],
                ["B", "Session signal", "Influence weights update only this mood."],
                ["C", "New Mood", "Tier II clears while Tier I remains intact."],
              ].map(([n, title, body]) => (
                <div key={n} className="grid grid-cols-[20px_1fr] gap-4">
                  <span className="font-space-grotesk text-[11px] text-[var(--wine)]">{n}</span>
                  <span>
                    <b className="font-medium text-[var(--ink)]">{title}</b> - {body}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="mood-card rounded-[2px] p-6">
            <div className="mb-5 flex items-center justify-between border-b border-[var(--rule)] pb-4 font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
              <span className="inline-flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--wine)]" />
                Session - Sunday, 9:42 PM
              </span>
              <span>Tier I: 14 carved - Tier II: 3 live</span>
            </div>
            <div className="space-y-5 text-[14.5px] leading-6">
              <p><span className="text-[var(--clay)]">MoodLens:</span> Good evening. What are we in the mood for?</p>
              <p><span className="text-[var(--ink)]">You:</span> Something quiet. Not another bleak drama. I had a heavy week.</p>
              <p>
                <span className="text-[var(--clay)]">MoodLens:</span> Noted for tonight. I will set bleak dramas aside and lean warmer.
                <span className="mt-2 block font-space-grotesk text-[10px] uppercase tracking-[0.1em] text-[var(--wine)]">
                  Tier II - influence weight: bleak-drama -0.62
                </span>
              </p>
              <p><span className="text-[var(--ink)]">You:</span> Also, stop recommending horror. Ever. It is not for me.</p>
              <p>
                <span className="text-[var(--clay)]">MoodLens:</span> Understood. That is identity-level, so I am carving it out.
                <span className="mt-2 block font-space-grotesk text-[10px] uppercase tracking-[0.1em] text-[var(--wine)]">
                  Tier I - GNNDelete - genre:horror erased
                </span>
              </p>
              <div className="flex flex-wrap items-center gap-3 border border-[var(--rule-strong)] bg-[var(--bone-3)] p-4">
                <span className="rounded-full bg-[var(--wine)] px-3 py-2 font-space-grotesk text-[10px] uppercase tracking-[0.14em] text-[var(--bone)]">
                  New Mood
                </span>
                <span className="font-display italic text-[var(--ink-2)]">resets tonight, not you.</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="architecture" className="section-band bg-[#0e0a05] py-16 md:py-24">
        <div className="wrap">
          <SectionHead
            label="IV - Architecture"
            title={
              <>
                Three components. <em className="text-[var(--wine)]">One coherent argument.</em>
              </>
            }
          />
          <div className="mt-12 grid gap-6 md:grid-cols-3">
            {[
              [GitBranch, "LightGCN", "A 3-layer graph collaborative filter over users, movies, and genre edges."],
              [RefreshCw, "GNNDelete", "Permanent erasure that minimizes forget influence while retaining utility."],
              [MessageSquare, "Influence Erase", "Session unlearning that reverses mood drift at the embedding level."],
            ].map(([Icon, title, body], i) => {
              const I = Icon as typeof GitBranch;
              return (
                <article key={String(title)} className="border border-[var(--rule)] p-7">
                  <div className="font-space-grotesk text-[11px] uppercase tracking-[0.12em] text-[var(--clay)]">
                    {String(i + 1).padStart(2, "0")} / Component
                  </div>
                  <I className="mt-8 text-[var(--wine)]" size={26} strokeWidth={1.5} />
                  <h3 className="font-display mt-4 text-3xl">{String(title)}</h3>
                  <p className="mt-4 text-sm leading-6 text-[var(--ink-2)]">{String(body)}</p>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <footer className="section-band wrap py-12">
        <div className="flex flex-wrap items-end justify-between gap-8">
          <div className="font-display text-6xl leading-none tracking-[-0.03em]">
            Mood<em className="text-[var(--wine)]">Lens</em>
          </div>
          <div className="text-right font-space-grotesk text-[11px] uppercase tracking-[0.12em] leading-7 text-[var(--clay)]">
            <Link href="/chat" className="hover:text-[var(--wine)]">Chat demo</Link>
            <br />
            <Link href="/evaluation" className="hover:text-[var(--wine)]">Evaluation console</Link>
          </div>
        </div>
      </footer>
    </main>
  );
}
