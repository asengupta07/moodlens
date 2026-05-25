import { NextRequest } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

export async function GET(_req: NextRequest) {
  const root = path.resolve(process.cwd(), "..");
  const metricsPath = path.join(root, "evaluation", "metrics.json");

  try {
    const raw = await readFile(metricsPath, "utf8");
    return Response.json({ success: true, metrics: JSON.parse(raw) });
  } catch (error: any) {
    return Response.json(
      {
        success: false,
        metrics: null,
        message: error?.code === "ENOENT" ? "No metrics.json found yet." : "Could not read metrics.json.",
      },
      { status: error?.code === "ENOENT" ? 200 : 500 },
    );
  }
}
