// frontend/next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // In production (Vercel), NEXT_PUBLIC_API_URL is set to the deployed
  // backend URL. In local dev we proxy /api/* to localhost:8000 so the
  // browser never has to deal with CORS at all — the Next.js dev server
  // forwards the request server-side.
  async rewrites() {
    // Only proxy in development; in production the env var handles routing.
    if (process.env.NODE_ENV === "production") return [];
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
