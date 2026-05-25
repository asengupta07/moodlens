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

const WIDTH = 280;
const HEIGHT = 100;
const PAD_X = 24;
const PAD_Y = 14;

function buildPath(values: number[]): string {
  if (values.length === 0) return "";
  const max = Math.max(0.01, ...values);
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

  const permPath = useMemo(() => buildPath(permVals), [permVals]);
  const sessPath = useMemo(() => buildPath(sessVals), [sessVals]);

  const empty = permVals.length === 0 && sessVals.length === 0;

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-3 text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="text-white/40 uppercase tracking-wider text-[10px] font-medium">
          Embedding drift
        </span>
        <div className="flex gap-3 text-[10px]">
          <span className="flex items-center gap-1 text-red-300">
            <span className="w-2 h-0.5 bg-red-400" /> Tier 1
          </span>
          <span className="flex items-center gap-1 text-amber-300">
            <span className="w-2 h-0.5 bg-amber-400" /> Tier 2
          </span>
        </div>
      </div>
      {empty ? (
        <p className="text-white/30 text-[11px]">
          No drift events yet. Trigger Tier 1 or Tier 2 unlearning to see the
          chart populate.
        </p>
      ) : (
        <svg width="100%" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="overflow-visible">
          {/* baseline */}
          <line
            x1={PAD_X}
            x2={WIDTH - PAD_X}
            y1={HEIGHT - PAD_Y}
            y2={HEIGHT - PAD_Y}
            stroke="rgba(255,255,255,0.1)"
            strokeWidth={1}
          />
          {permPath && (
            <path d={permPath} fill="none" stroke="#f87171" strokeWidth={1.5} />
          )}
          {sessPath && (
            <path d={sessPath} fill="none" stroke="#fbbf24" strokeWidth={1.5} />
          )}
          {permVals.map((v, i) => {
            const stepX = permVals.length > 1 ? (WIDTH - PAD_X * 2) / (permVals.length - 1) : 0;
            const max = Math.max(0.01, ...permVals);
            const cx = PAD_X + stepX * i;
            const cy = HEIGHT - PAD_Y - (v / max) * (HEIGHT - PAD_Y * 2);
            return <circle key={`p-${i}`} cx={cx} cy={cy} r={2} fill="#f87171" />;
          })}
          {sessVals.map((v, i) => {
            const stepX = sessVals.length > 1 ? (WIDTH - PAD_X * 2) / (sessVals.length - 1) : 0;
            const max = Math.max(0.01, ...sessVals);
            const cx = PAD_X + stepX * i;
            const cy = HEIGHT - PAD_Y - (v / max) * (HEIGHT - PAD_Y * 2);
            return <circle key={`s-${i}`} cx={cx} cy={cy} r={2} fill="#fbbf24" />;
          })}
        </svg>
      )}
    </div>
  );
}
