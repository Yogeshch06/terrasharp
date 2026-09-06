import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:10000";

export async function GET(_req: NextRequest) {
  try {
    const res = await fetch(`${API_URL}/health`);
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ status: "unreachable", model_loaded: false }, { status: 503 });
  }
}
