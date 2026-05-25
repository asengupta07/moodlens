import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(_req: NextRequest) {
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
  try {
    const res = await fetch(`${backendUrl}/session`, { cache: 'no-store' });
    const data = await res.json();
    return Response.json(data);
  } catch {
    return Response.json({
      active: false,
      session_id: null,
      mood: null,
      movie_count: 0,
      start_time: null,
      interactions: [],
    });
  }
}
