import { NextRequest } from "next/server";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function POST(_req: NextRequest) {
  const root = path.resolve(process.cwd(), "..");
  const script = path.join(root, "evaluation", "cli.py");
  const metricsPath = path.join(root, "evaluation", "metrics.json");
  const python = process.env.PYTHON_BIN || "python3";

  try {
    const { stdout, stderr } = await execFileAsync(python, [script, "--json-only"], {
      cwd: root,
      maxBuffer: 1024 * 1024 * 20,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    const raw = await readFile(metricsPath, "utf8");
    return Response.json({
      success: true,
      metrics: JSON.parse(raw),
      stdout,
      stderr,
    });
  } catch (error: any) {
    let metrics = null;
    try {
      metrics = JSON.parse(await readFile(metricsPath, "utf8"));
    } catch {}

    return Response.json(
      {
        success: false,
        metrics,
        stdout: error?.stdout ?? "",
        stderr: error?.stderr ?? error?.message ?? "Evaluation failed.",
        message: "Evaluation script failed.",
      },
      { status: 500 },
    );
  }
}
