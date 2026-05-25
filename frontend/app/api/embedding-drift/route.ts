import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(_req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  try {
    const res = await fetch(`${backendUrl}/embedding-drift`, { cache: 'no-store' });
    const data = await res.json();
    return Response.json(data);
  } catch {
    return Response.json({ has_data: false, permanent_history: [], session_history: [] });
  }
}
