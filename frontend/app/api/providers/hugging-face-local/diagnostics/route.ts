import { fetchFromBackend } from "@/lib/backend";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const response = await fetchFromBackend("/providers/hugging-face-local/diagnostics", {
      cache: "no-store",
      headers: {
        Accept: "application/json",
      },
    });
    const body = await response.text();

    return new Response(body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") ?? "application/json",
      },
    });
  } catch {
    return Response.json(
      {
        detail: "Unable to reach the backend Hugging Face Local diagnostics endpoint.",
      },
      { status: 502 },
    );
  }
}
