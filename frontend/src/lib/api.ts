import type { SseEventPayload } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export function normalizeReference(value: string) {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

export function buildMusicPrompt(reference: string, passageText: string | null, prompt: string) {
  const parts = [
    reference ? `Scripture reference: ${reference}.` : null,
    passageText ? `Passage text:\n${passageText}` : null,
    prompt ? `Song direction:\n${prompt}` : null
  ].filter((value): value is string => value !== null);

  return parts.join("\n\n");
}

function buildApiUrl(path: string, params?: Record<string, string>) {
  const base = API_BASE || window.location.origin;
  const url = new URL(path, base);

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
  }

  return url.toString();
}

export async function requestJson<T>(path: string, init?: RequestInit, params?: Record<string, string>) {
  const response = await fetch(buildApiUrl(path, params), {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {})
    }
  });

  const raw = await response.text();
  const payload = raw ? (JSON.parse(raw) as unknown) : null;

  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof payload.detail === "string"
        ? payload.detail
        : `Request failed with status ${response.status}.`;
    throw new Error(detail);
  }

  return payload as T;
}

export async function requestRealtimeAnswer(
  webrtcUrl: string,
  clientSecret: string,
  offerSdp: string
) {
  const response = await fetch(webrtcUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${clientSecret}`,
      "Content-Type": "application/sdp"
    },
    body: offerSdp
  });

  const raw = await response.text();
  if (!response.ok) {
    throw new Error(raw || `Realtime negotiation failed with status ${response.status}.`);
  }

  return raw;
}

function parseSseEvent(rawEvent: string): SseEventPayload | null {
  const lines = rawEvent.split(/\r?\n/);
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  const rawData = dataLines.join("\n");
  try {
    return { event, data: JSON.parse(rawData) as unknown };
  } catch {
    return { event, data: rawData };
  }
}

export async function requestSse(
  path: string,
  init: RequestInit,
  onEvent: (event: SseEventPayload) => void
) {
  const response = await fetch(buildApiUrl(path), {
    ...init,
    headers: {
      Accept: "text/event-stream",
      ...(init.headers ?? {})
    }
  });

  if (!response.ok) {
    const raw = await response.text();
    let payload: unknown = null;
    try {
      payload = raw ? (JSON.parse(raw) as unknown) : null;
    } catch {
      payload = null;
    }
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload &&
      typeof payload.detail === "string"
        ? payload.detail
        : `Request failed with status ${response.status}.`;
    throw new Error(detail);
  }

  if (!response.body) {
    throw new Error("Streaming response body is unavailable.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex !== -1) {
      const rawEvent = buffer.slice(0, separatorIndex).trim();
      buffer = buffer.slice(separatorIndex + 2);
      if (rawEvent) {
        const parsed = parseSseEvent(rawEvent);
        if (parsed) {
          onEvent(parsed);
        }
      }
      separatorIndex = buffer.indexOf("\n\n");
    }
  }

  buffer += decoder.decode();
  const trailing = buffer.trim();
  if (trailing) {
    const parsed = parseSseEvent(trailing);
    if (parsed) {
      onEvent(parsed);
    }
  }
}

export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Unable to read recorded audio."));
    reader.onloadend = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
        return;
      }
      reject(new Error("Unable to encode recorded audio."));
    };
    reader.readAsDataURL(blob);
  });
}
