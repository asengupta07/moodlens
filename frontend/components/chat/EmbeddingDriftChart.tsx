"use client";

import { useMemo } from "react";

interface Point {
  tier: number;
  event_type: string;
  timestamp: string;
  cosine_distance?: number;
}

interface Props {
  permanent: Point[];
  session: Point[];
}

const WIDTH = 320;
const HEIGHT = 112;
const PAD_X = 24;
const PAD_Y = 16;

function buildPath(values: number[], maxValue: number): string {
  if (values.length === 0) return "";
  const max = Math.max(0.01, maxValue);
  const stepX = values.length > 1 ? (WIDTH - PAD_X * 2) / (values.length - 1) : 0;
  return values
    .map((v, i) => {
      const x = PAD_X + stepX * i;
      const y = HEIGHT - PAD_Y - (v / max) * (HEIGHT - PAD_Y * 2);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export function EmbeddingDriftChart({ permanent, session }: Props) {
  const permVals = permanent.map((p) => p.cosine_distance ?? 0);
  const sessVals = session.map((p) => p.cosine_distance ?? 0);
  const maxValue = Math.max(0.01, ...permVals, ...sessVals);
  const permPath = useMemo(() => buildPath(permVals, maxValue), [permVals, maxValue]);
  const sessPath = useMemo(() => buildPath(sessVals, maxValue), [sessVals, maxValue]);
  const empty = permVals.length === 0 && sessVals.length === 0;

  return (
    <section className="mood-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-space-grotesk text-[10px] uppercase tracking-[0.12em] text-[var(--clay)]">
          Taste movement timeline
        </span>
        <div className="flex items-center gap-3 font-space-grotesk text-[9px] uppercase tracking-[0.1em]">
          <span className="flex items-center gap-1 text-[var(--wine)]">
            <span className="h-px w-4 bg-[var(--wine)]" /> Tier I
          </span>
          <span className="flex items-center gap-1 text-[var(--amber)]">
            <span className="h-px w-4 bg-[var(--amber)]" /> Tier II
          </span>
        </div>
      </div>
      {empty ? (
        <div>
          <p className="text-xs leading-6 text-[var(--ink-2)]">
            No memory movement yet. When the system forgets, this becomes a timeline of how strongly the recommender's internal taste map changed.
          </p>
          <div className="mt-3 flex h-16 items-end gap-1 border border-[var(--rule)] bg-[rgba(242,237,227,0.03)] p-2">
            {[0.22, 0.5, 0.32, 0.74, 0.46, 0.62, 0.28, 0.84, 0.36, 0.58].map((h, i) => (
              <span
                key={i}
                className="flex-1 bg-[var(--clay)] opacity-30"
                style={{ height: `${h * 100}%` }}
              />
            ))}
          </div>
        </div>
      ) : (
        <div>
        <svg width="100%" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="overflow-visible">
          {[0.25, 0.5, 0.75, 1].map((t) => (
            <line
              key={t}
              x1={PAD_X}
              x2={WIDTH - PAD_X}
              y1={HEIGHT - PAD_Y - t * (HEIGHT - PAD_Y * 2)}
              y2={HEIGHT - PAD_Y - t * (HEIGHT - PAD_Y * 2)}
              stroke="rgba(242,237,227,0.08)"
              strokeWidth={1}
            />
          ))}
          {permPath && <path d={permPath} fill="none" stroke="var(--wine)" strokeWidth={2} />}
          {sessPath && <path d={sessPath} fill="none" stroke="var(--amber)" strokeWidth={2} strokeDasharray="5 4" />}
          {[...permVals.map((v, i) => ({ v, i, c: "var(--wine)", label: permanent[i]?.event_type ?? "Tier I" })), ...sessVals.map((v, i) => ({ v, i, c: "var(--amber)", label: session[i]?.event_type ?? "Tier II" }))].map((p, idx) => {
            const values = p.c === "var(--wine)" ? permVals : sessVals;
            const stepX = values.length > 1 ? (WIDTH - PAD_X * 2) / (values.length - 1) : 0;
            return (
              <circle
                key={idx}
                cx={PAD_X + stepX * p.i}
                cy={HEIGHT - PAD_Y - (p.v / maxValue) * (HEIGHT - PAD_Y * 2)}
                r={3}
                fill={p.c}
              >
                <title>{`${p.label}: ${p.v.toFixed(4)}`}</title>
              </circle>
            );
          })}
        </svg>
        <p className="mt-2 text-[11px] leading-5 text-[var(--clay)]">
          Red: permanent profile surgery. Amber: temporary mood clearing. Both lines use the same vertical scale.
        </p>
        </div>
      )}
    </section>
  );
}
