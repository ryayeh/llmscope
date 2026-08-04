const DEFAULT_BACKEND_BASE_URLS = [
  "http://127.0.0.1:8000",
  "http://localhost:8000",
  "http://127.0.0.1:8001",
  "http://localhost:8001",
];

function normalizeBackendUrl(baseUrl: string, path: string): string {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const normalizedPath = path.startsWith("/") ? path.slice(1) : path;

  return new URL(normalizedPath, normalizedBase).toString();
}

function getConfiguredBaseUrls(): string[] {
  return [
    process.env.BACKEND_API_BASE_URL,
    process.env.NEXT_PUBLIC_API_BASE_URL,
  ].filter((value): value is string => Boolean(value));
}

export function getBackendUrlCandidates(path: string): string[] {
  return [...new Set([...getConfiguredBaseUrls(), ...DEFAULT_BACKEND_BASE_URLS])].map(
    (baseUrl) => normalizeBackendUrl(baseUrl, path),
  );
}

export async function fetchFromBackend(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  let lastError: unknown;

  for (const url of getBackendUrlCandidates(path)) {
    try {
      return await fetch(url, init);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error("Unable to reach any configured backend URL.");
}
