import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:10000";

export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const jobId = searchParams.get("jobId");
  const filename = searchParams.get("filename");

  if (!jobId || !filename) {
    return NextResponse.json({ error: "jobId and filename required" }, { status: 400 });
  }

  const url = `${API_URL}/download/${jobId}/${filename}`;
  const signedUrl = { url };
  return NextResponse.json(signedUrl);
}
