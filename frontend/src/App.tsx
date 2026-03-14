import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

type TranslationCode = "WEB" | "KJV";
type ImageStyle = "modern_editorial_illustration";
type HealthStatus = "checking" | "online" | "offline";
type PersonaChatRole = "user" | "assistant";
type HymnJobStatus = "queued" | "in_progress" | "completed" | "failed";
type ViewMode = "study" | "hymn" | "discussion";

type BibleVerse = {
  book: string;
  chapter: number;
  verse: number;
  text: string;
};

type BiblePassageResponse = {
  reference: string;
  translation: TranslationCode;
  normalized_reference: string;
  text: string;
  verses: BibleVerse[];
};

type UsageMetrics = {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
};

type StudyPlanResponse = {
  reference: string;
  normalized_reference: string;
  translation: TranslationCode;
  passage_text: string;
  passage_title: string;
  context_points: string[];
  discussion_questions: string[];
  reflection_questions: string[];
  include_question_notes: boolean;
  discussion_question_notes: string[] | null;
  reflection_question_notes: string[] | null;
  model: string;
  usage: UsageMetrics | null;
};

type PassageImageResponse = {
  reference: string;
  translation: TranslationCode;
  style: ImageStyle;
  prompt_used: string;
  image_b64_or_url: string;
  alt_text: string;
};

type PersonaChatMessage = {
  role: PersonaChatRole;
  content: string;
};

type PersonaChatResponse = {
  reply: string;
  model: string;
  usage: UsageMetrics | null;
};

type VoiceTranscriptionResponse = {
  transcript: string;
  model: string;
};

type HymnSection = {
  label: string;
  lyrics: string;
};

type HymnLyrics = {
  title: string;
  theme: string;
  scripture_references: string[];
  sections: HymnSection[];
};

type HymnGenerateResponse = {
  reference: string;
  normalized_reference: string;
  translation: TranslationCode;
  passage_text: string;
  hymn: HymnLyrics;
  job_id: string;
  job_status: HymnJobStatus;
  provider: string;
  model: string;
  usage: UsageMetrics | null;
};

type HymnJobResponse = {
  job_id: string;
  status: HymnJobStatus;
  provider: string;
  audio_url: string | null;
  error: string | null;
};

function normalizeReference(value: string) {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
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

async function requestJson<T>(path: string, init?: RequestInit, params?: Record<string, string>) {
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

type SseEventPayload = {
  event: string;
  data: unknown;
};

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

async function requestSse(
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

function blobToDataUrl(blob: Blob): Promise<string> {
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

export default function App() {
  const [activeView, setActiveView] = useState<ViewMode>("study");

  const [reference, setReference] = useState("Luke 21:5-28");
  const [translation, setTranslation] = useState<TranslationCode>("WEB");
  const [goals, setGoals] = useState("");
  const [userNotes, setUserNotes] = useState("");
  const [includeQuestionNotes, setIncludeQuestionNotes] = useState(false);

  const [healthStatus, setHealthStatus] = useState<HealthStatus>("checking");

  const [passage, setPassage] = useState<BiblePassageResponse | null>(null);
  const [studyPlan, setStudyPlan] = useState<StudyPlanResponse | null>(null);
  const [passageError, setPassageError] = useState("");
  const [studyPlanError, setStudyPlanError] = useState("");
  const [isLoadingPassage, setIsLoadingPassage] = useState(false);
  const [isLoadingStudyPlan, setIsLoadingStudyPlan] = useState(false);

  const [passageImage, setPassageImage] = useState<PassageImageResponse | null>(null);
  const [passageImageError, setPassageImageError] = useState("");
  const [isLoadingPassageImage, setIsLoadingPassageImage] = useState(false);

  const [personaInput, setPersonaInput] = useState("");
  const [personaMessages, setPersonaMessages] = useState<PersonaChatMessage[]>([]);
  const [personaModel, setPersonaModel] = useState<string | null>(null);
  const [personaError, setPersonaError] = useState("");
  const [isSendingPersona, setIsSendingPersona] = useState(false);
  const [isRecordingPersona, setIsRecordingPersona] = useState(false);
  const [isTranscribingPersona, setIsTranscribingPersona] = useState(false);
  const [enableVoiceReply, setEnableVoiceReply] = useState(true);

  const personaRecorderRef = useRef<MediaRecorder | null>(null);
  const personaStreamRef = useRef<MediaStream | null>(null);
  const personaChunksRef = useRef<Blob[]>([]);

  const [hymnStyle, setHymnStyle] = useState("modern worship hymn, acoustic");
  const [hymnMood, setHymnMood] = useState("hopeful");
  const [hymnResult, setHymnResult] = useState<HymnGenerateResponse | null>(null);
  const [hymnJob, setHymnJob] = useState<HymnJobResponse | null>(null);
  const [hymnError, setHymnError] = useState("");
  const [isGeneratingHymn, setIsGeneratingHymn] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        await requestJson<{ status: string }>("/health");
        if (!cancelled) {
          setHealthStatus("online");
        }
      } catch {
        if (!cancelled) {
          setHealthStatus("offline");
        }
      }
    }

    void checkHealth();

    return () => {
      cancelled = true;
    };
  }, []);

  const hymnJobId = hymnResult?.job_id ?? null;
  const hymnJobStatus = hymnJob?.status ?? hymnResult?.job_status ?? null;

  useEffect(() => {
    if (!hymnJobId) {
      return;
    }
    if (hymnJobStatus === "completed" || hymnJobStatus === "failed") {
      return;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const next = await requestJson<HymnJobResponse>(`/api/hymn/jobs/${hymnJobId}`);
        if (!cancelled) {
          setHymnJob(next);
        }
      } catch (error) {
        if (!cancelled) {
          setHymnError(error instanceof Error ? error.message : "Unable to refresh hymn status.");
        }
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [hymnJobId, hymnJobStatus]);

  function releasePersonaMediaResources() {
    if (personaStreamRef.current) {
      personaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    personaStreamRef.current = null;
    personaRecorderRef.current = null;
    personaChunksRef.current = [];
  }

  useEffect(() => {
    return () => {
      if (personaRecorderRef.current && personaRecorderRef.current.state !== "inactive") {
        personaRecorderRef.current.stop();
      }
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      releasePersonaMediaResources();
    };
  }, []);

  async function handlePassageLookup() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setPassageError("Enter a reference.");
      return;
    }

    setIsLoadingPassage(true);
    setPassageError("");

    try {
      const response = await requestJson<BiblePassageResponse>("/api/bible/passage", undefined, {
        reference: trimmedReference,
        translation
      });
      setPassage(response);
    } catch (error) {
      setPassage(null);
      setPassageError(error instanceof Error ? error.message : "Unable to load passage.");
    } finally {
      setIsLoadingPassage(false);
    }
  }

  async function handleStudyPlanGeneration() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setStudyPlanError("Enter a reference.");
      return;
    }

    setIsLoadingStudyPlan(true);
    setStudyPlanError("");

    try {
      const canReusePassage =
        passage !== null &&
        normalizeReference(passage.reference) === normalizeReference(trimmedReference) &&
        passage.translation === translation;

      const response = await requestJson<StudyPlanResponse>("/api/study-plan", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          reference: trimmedReference,
          translation,
          goals: goals.trim() || undefined,
          user_notes: userNotes.trim() || undefined,
          include_question_notes: includeQuestionNotes,
          passage_text: canReusePassage ? passage.text : undefined
        })
      });

      setStudyPlan(response);
      if (!canReusePassage) {
        setPassage({
          reference: response.reference,
          normalized_reference: response.normalized_reference,
          translation: response.translation,
          text: response.passage_text,
          verses: []
        });
      }
    } catch (error) {
      setStudyPlan(null);
      setStudyPlanError(error instanceof Error ? error.message : "Unable to generate study plan.");
    } finally {
      setIsLoadingStudyPlan(false);
    }
  }

  async function handlePassageImageGeneration() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setPassageImageError("Enter a reference.");
      return;
    }

    setIsLoadingPassageImage(true);
    setPassageImageError("");

    try {
      const response = await requestJson<PassageImageResponse>("/api/passage-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          reference: trimmedReference,
          translation,
          style: "modern_editorial_illustration"
        })
      });
      setPassageImage(response);
    } catch (error) {
      setPassageImage(null);
      setPassageImageError(error instanceof Error ? error.message : "Unable to generate image.");
    } finally {
      setIsLoadingPassageImage(false);
    }
  }

  function speakPersonaReply(replyText: string) {
    const cleaned = replyText.trim();
    if (!cleaned || !enableVoiceReply) {
      return;
    }
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleaned);
    utterance.rate = 1;
    window.speechSynthesis.speak(utterance);
  }

  async function sendPersonaMessage(rawMessage: string) {
    const userMessage = rawMessage.trim();
    if (!userMessage) {
      setPersonaError("Type a message.");
      return;
    }

    const nextMessages = [...personaMessages, { role: "user", content: userMessage } as PersonaChatMessage];
    setPersonaMessages([...nextMessages, { role: "assistant", content: "" }]);
    setPersonaError("");
    setIsSendingPersona(true);

    try {
      let streamedModel: string | null = null;
      let receivedChunk = false;
      let assistantReply = "";
      await requestSse(
        "/api/persona-chat/stream",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            messages: nextMessages,
            reference_context: reference.trim() || undefined,
            translation
          })
        },
        ({ event, data }) => {
          if (event === "meta") {
            if (
              typeof data === "object" &&
              data !== null &&
              "model" in data &&
              typeof data.model === "string"
            ) {
              streamedModel = data.model;
            }
            return;
          }

          if (event === "chunk") {
            const delta =
              typeof data === "object" &&
              data !== null &&
              "delta" in data &&
              typeof data.delta === "string"
                ? data.delta
                : null;
            if (delta && delta.length > 0) {
              assistantReply += delta;
              receivedChunk = true;
              setPersonaMessages((current) => {
                if (current.length === 0) {
                  return [{ role: "assistant", content: delta }];
                }
                const next = [...current];
                const last = next[next.length - 1];
                if (last.role !== "assistant") {
                  next.push({ role: "assistant", content: delta });
                  return next;
                }
                next[next.length - 1] = { role: "assistant", content: `${last.content}${delta}` };
                return next;
              });
            }
            return;
          }

          if (event === "error") {
            const detail =
              typeof data === "object" &&
              data !== null &&
              "detail" in data &&
              typeof data.detail === "string"
                ? data.detail
                : "Unable to stream response.";
            throw new Error(detail);
          }
        }
      );

      if (streamedModel) {
        setPersonaModel(streamedModel);
      }
      if (!receivedChunk) {
        setPersonaMessages((current) => {
          if (current.length === 0) {
            return current;
          }
          const last = current[current.length - 1];
          if (last.role === "assistant" && !last.content.trim()) {
            return current.slice(0, -1);
          }
          return current;
        });
      } else {
        speakPersonaReply(assistantReply);
      }
    } catch (error) {
      setPersonaMessages((current) => {
        if (current.length === 0) {
          return current;
        }
        const last = current[current.length - 1];
        if (last.role === "assistant" && !last.content.trim()) {
          return current.slice(0, -1);
        }
        return current;
      });
      setPersonaError(error instanceof Error ? error.message : "Unable to send message.");
    } finally {
      setIsSendingPersona(false);
    }
  }

  async function handlePersonaSend() {
    const userMessage = personaInput.trim();
    if (!userMessage) {
      setPersonaError("Type a message.");
      return;
    }
    setPersonaInput("");
    await sendPersonaMessage(userMessage);
  }

  async function transcribePersonaRecording(blob: Blob) {
    setIsTranscribingPersona(true);
    setPersonaError("");

    try {
      const audioBase64 = await blobToDataUrl(blob);
      const response = await requestJson<VoiceTranscriptionResponse>("/api/voice/transcribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          audio_base64: audioBase64,
          mime_type: blob.type || "audio/webm",
          file_name: "voice_input.webm"
        })
      });

      await sendPersonaMessage(response.transcript);
    } catch (error) {
      setPersonaError(error instanceof Error ? error.message : "Unable to process recorded audio.");
    } finally {
      setIsTranscribingPersona(false);
    }
  }

  async function startPersonaRecording() {
    if (isRecordingPersona || isSendingPersona || isTranscribingPersona) {
      return;
    }
    if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
      setPersonaError("Voice input is not supported in this browser.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setPersonaError("Microphone access is not available in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredMimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = MediaRecorder.isTypeSupported(preferredMimeType)
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);

      personaStreamRef.current = stream;
      personaRecorderRef.current = recorder;
      personaChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          personaChunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setPersonaError("Recording failed.");
        setIsRecordingPersona(false);
        releasePersonaMediaResources();
      };

      recorder.onstop = () => {
        const chunks = [...personaChunksRef.current];
        const blobType = recorder.mimeType || "audio/webm";
        setIsRecordingPersona(false);
        releasePersonaMediaResources();

        if (chunks.length === 0) {
          setPersonaError("No audio captured.");
          return;
        }

        const recording = new Blob(chunks, { type: blobType });
        void transcribePersonaRecording(recording);
      };

      setPersonaError("");
      recorder.start();
      setIsRecordingPersona(true);
    } catch (error) {
      setPersonaError(error instanceof Error ? error.message : "Unable to access microphone.");
      setIsRecordingPersona(false);
      releasePersonaMediaResources();
    }
  }

  function stopPersonaRecording() {
    if (!personaRecorderRef.current || personaRecorderRef.current.state === "inactive") {
      return;
    }
    personaRecorderRef.current.stop();
  }

  function handlePersonaReset() {
    if (personaRecorderRef.current && personaRecorderRef.current.state !== "inactive") {
      personaRecorderRef.current.stop();
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    releasePersonaMediaResources();
    setPersonaMessages([]);
    setPersonaModel(null);
    setPersonaError("");
    setPersonaInput("");
    setIsRecordingPersona(false);
    setIsTranscribingPersona(false);
  }

  async function handleHymnGeneration() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setHymnError("Enter a reference.");
      return;
    }

    setIsGeneratingHymn(true);
    setHymnError("");

    try {
      const canReusePassage =
        passage !== null &&
        normalizeReference(passage.reference) === normalizeReference(trimmedReference) &&
        passage.translation === translation;

      const response = await requestJson<HymnGenerateResponse>("/api/hymn/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          reference: trimmedReference,
          translation,
          style_hint: hymnStyle.trim(),
          mood_hint: hymnMood.trim() || undefined,
          user_notes: userNotes.trim() || undefined,
          passage_text: canReusePassage ? passage.text : undefined
        })
      });

      setHymnResult(response);
      setHymnJob({
        job_id: response.job_id,
        status: response.job_status,
        provider: response.provider,
        audio_url: null,
        error: null
      });

      if (!canReusePassage) {
        setPassage({
          reference: response.reference,
          normalized_reference: response.normalized_reference,
          translation: response.translation,
          text: response.passage_text,
          verses: []
        });
      }
    } catch (error) {
      setHymnResult(null);
      setHymnJob(null);
      setHymnError(error instanceof Error ? error.message : "Unable to generate hymn.");
    } finally {
      setIsGeneratingHymn(false);
    }
  }

  const hasUsage = studyPlan?.usage && studyPlan.usage.total_tokens !== null;
  const hymnStatusLabel = useMemo(() => {
    if (!hymnJobStatus) {
      return "queued";
    }
    return hymnJobStatus;
  }, [hymnJobStatus]);

  return (
    <div className="shell">
      <div className="background-orb background-orb-left" />
      <div className="background-orb background-orb-right" />

      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">yaf-gpt</p>
          <h1>Study Workspace</h1>
        </div>

        <div className={`status-pill ${healthStatus}`}>
          <span className="status-dot" />
          <span>{healthStatus.toUpperCase()}</span>
        </div>
      </header>

      <section className="view-switcher">
        <button
          type="button"
          className={`view-button ${activeView === "study" ? "active" : ""}`}
          onClick={() => setActiveView("study")}
        >
          Study
        </button>
        <button
          type="button"
          className={`view-button ${activeView === "hymn" ? "active" : ""}`}
          onClick={() => setActiveView("hymn")}
        >
          Hymn
        </button>
        <button
          type="button"
          className={`view-button ${activeView === "discussion" ? "active" : ""}`}
          onClick={() => setActiveView("discussion")}
        >
          Discussion
        </button>
      </section>

      <main className={`workspace ${activeView === "discussion" ? "workspace-single" : ""}`}>
        {activeView !== "discussion" ? (
        <section className="panel control-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Inputs</p>
              <h2>{activeView === "study" ? "Study" : activeView === "hymn" ? "Hymn" : "Discussion"}</h2>
            </div>
          </div>

          {activeView !== "hymn" ? (
            <>
              <label className="field">
                <span>Reference</span>
                <input
                  value={reference}
                  onChange={(event) => setReference(event.target.value)}
                  placeholder="Luke 21:5-28"
                />
              </label>

              <label className="field">
                <span>Translation</span>
                <select value={translation} onChange={(event) => setTranslation(event.target.value as TranslationCode)}>
                  <option value="WEB">WEB</option>
                  <option value="KJV">KJV</option>
                </select>
              </label>
            </>
          ) : null}

          {activeView === "study" ? (
            <>
              <label className="field">
                <span>Goals</span>
                <textarea
                  rows={3}
                  value={goals}
                  onChange={(event) => setGoals(event.target.value)}
                  placeholder="Optional"
                />
              </label>

              <label className="field">
                <span>Notes</span>
                <textarea
                  rows={3}
                  value={userNotes}
                  onChange={(event) => setUserNotes(event.target.value)}
                  placeholder="Optional"
                />
              </label>

              <label className="field field-inline">
                <span>Include question notes</span>
                <input
                  type="checkbox"
                  checked={includeQuestionNotes}
                  onChange={(event) => setIncludeQuestionNotes(event.target.checked)}
                />
              </label>

              <div className="action-row action-row-single">
                <button
                  type="button"
                  className="primary-button"
                  onClick={handlePassageLookup}
                  disabled={isLoadingPassage}
                >
                  {isLoadingPassage ? "Loading..." : "Fetch passage"}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleStudyPlanGeneration}
                  disabled={isLoadingStudyPlan}
                >
                  {isLoadingStudyPlan ? "Generating..." : "Generate plan"}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handlePassageImageGeneration}
                  disabled={isLoadingPassageImage}
                >
                  {isLoadingPassageImage ? "Generating..." : "Generate image"}
                </button>
              </div>
            </>
          ) : null}

          {activeView === "hymn" ? (
            <>
              <label className="field">
                <span>Style</span>
                <input
                  value={hymnStyle}
                  onChange={(event) => setHymnStyle(event.target.value)}
                  placeholder="modern worship hymn, acoustic"
                />
              </label>

              <label className="field">
                <span>Mood</span>
                <input
                  value={hymnMood}
                  onChange={(event) => setHymnMood(event.target.value)}
                  placeholder="hopeful"
                />
              </label>

              <label className="field">
                <span>Notes</span>
                <textarea
                  rows={3}
                  value={userNotes}
                  onChange={(event) => setUserNotes(event.target.value)}
                  placeholder="Optional"
                />
              </label>

              <div className="action-row action-row-single">
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleHymnGeneration}
                  disabled={isGeneratingHymn}
                >
                  {isGeneratingHymn ? "Generating..." : "Generate hymn"}
                </button>
              </div>
            </>
          ) : null}

        </section>
        ) : null}

        <section className="results-column">
          {activeView === "study" ? (
            <>
              <article className="panel">
                <div className="panel-heading">
                  <div>
                    <p className="panel-kicker">Passage</p>
                    <h2>{passage?.normalized_reference ?? "No passage"}</h2>
                  </div>
                  {passage && <span className="meta-badge">{passage.translation}</span>}
                </div>

                {passageError ? <p className="error-banner">{passageError}</p> : null}

                {passage ? (
                  <div className="stack">
                    <p className="passage-text">{passage.text}</p>
                    {passage.verses.length > 0 ? (
                      <div className="verse-list">
                        {passage.verses.map((verse) => (
                          <article key={`${verse.book}-${verse.chapter}-${verse.verse}`} className="verse-card">
                            <p className="verse-label">
                              {verse.book} {verse.chapter}:{verse.verse}
                            </p>
                            <p>{verse.text}</p>
                          </article>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <p className="empty-state">Fetch a passage.</p>
                )}
              </article>

              <article className="panel">
                <div className="panel-heading">
                  <div>
                    <p className="panel-kicker">Study plan</p>
                    <h2>{studyPlan?.passage_title ?? "No plan"}</h2>
                  </div>
                  {studyPlan && <span className="meta-badge">{studyPlan.model}</span>}
                </div>

                {studyPlanError ? <p className="error-banner">{studyPlanError}</p> : null}

                {studyPlan ? (
                  <div className="stack">
                    <section>
                      <h3>Context</h3>
                      <ul className="content-list">
                        {studyPlan.context_points.map((point) => (
                          <li key={point}>{point}</li>
                        ))}
                      </ul>
                    </section>

                    <section>
                      <h3>Questions</h3>
                      <ol className="content-list ordered-list">
                        {studyPlan.discussion_questions.map((question, idx) => (
                          <li key={question}>
                            {question}
                            {studyPlan.discussion_question_notes?.[idx] ? (
                              <p className="question-note">{studyPlan.discussion_question_notes[idx]}</p>
                            ) : null}
                          </li>
                        ))}
                      </ol>
                    </section>

                    <section>
                      <h3>Reflection</h3>
                      <ul className="content-list">
                        {studyPlan.reflection_questions.map((question, idx) => (
                          <li key={question}>
                            {question}
                            {studyPlan.reflection_question_notes?.[idx] ? (
                              <p className="question-note">{studyPlan.reflection_question_notes[idx]}</p>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                    </section>

                    {hasUsage ? (
                      <p className="usage-note">
                        Tokens: {studyPlan.usage?.prompt_tokens ?? 0} / {studyPlan.usage?.completion_tokens ?? 0} / {" "}
                        {studyPlan.usage?.total_tokens ?? 0}
                      </p>
                    ) : null}
                  </div>
                ) : (
                  <p className="empty-state">Generate a plan.</p>
                )}
              </article>

              <article className="panel">
                <div className="panel-heading">
                  <div>
                    <p className="panel-kicker">Image</p>
                    <h2>{passageImage ? "Ready" : "No image"}</h2>
                  </div>
                  {passageImage && <span className="meta-badge">{passageImage.style}</span>}
                </div>

                {passageImageError ? <p className="error-banner">{passageImageError}</p> : null}

                {passageImage ? (
                  <div className="stack">
                    <img className="image-preview" src={passageImage.image_b64_or_url} alt={passageImage.alt_text} />
                    <p className="prompt-note">{passageImage.alt_text}</p>
                  </div>
                ) : (
                  <p className="empty-state">Generate an image.</p>
                )}
              </article>
            </>
          ) : null}

          {activeView === "discussion" ? (
            <article className="panel">
              <div className="panel-heading">
                <div>
                  <p className="panel-kicker">Discussion</p>
                  <h2>Mentor chat</h2>
                </div>
                {personaModel ? <span className="meta-badge">{personaModel}</span> : null}
              </div>

              {personaError ? <p className="error-banner">{personaError}</p> : null}

              <div className="stack">
                {personaMessages.length > 0 ? (
                  <div className="chat-thread">
                    {personaMessages.map((message, index) => (
                      <article key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
                        <p className="summary-label">{message.role === "user" ? "You" : "Mentor"}</p>
                        <p>{message.content}</p>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="empty-state">Start the chat.</p>
                )}

                <label className="field">
                  <span>Message</span>
                  <textarea
                    rows={3}
                    value={personaInput}
                    onChange={(event) => setPersonaInput(event.target.value)}
                    placeholder="Ask a question"
                  />
                </label>

                <div className="mini-action-row">
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={handlePersonaSend}
                    disabled={isSendingPersona || isRecordingPersona || isTranscribingPersona}
                  >
                    {isSendingPersona ? "Sending..." : "Send"}
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={isRecordingPersona ? stopPersonaRecording : () => void startPersonaRecording()}
                    disabled={isSendingPersona || isTranscribingPersona}
                  >
                    {isRecordingPersona ? "Stop recording" : "Voice input"}
                  </button>
                  <button type="button" className="secondary-button" onClick={handlePersonaReset}>
                    Reset
                  </button>
                </div>

                <label className="field field-inline">
                  <span>Voice reply</span>
                  <input
                    type="checkbox"
                    checked={enableVoiceReply}
                    onChange={(event) => setEnableVoiceReply(event.target.checked)}
                  />
                </label>

                {isRecordingPersona ? <p className="muted-text">Recording...</p> : null}
                {isTranscribingPersona ? <p className="muted-text">Transcribing audio...</p> : null}
              </div>
            </article>
          ) : null}

          {activeView === "hymn" ? (
            <article className="panel">
              <div className="panel-heading">
                <div>
                  <p className="panel-kicker">Hymn</p>
                  <h2>{hymnResult?.hymn.title ?? "No hymn"}</h2>
                </div>
                {hymnResult ? <span className="meta-badge">{hymnResult.model}</span> : null}
              </div>

              {hymnError ? <p className="error-banner">{hymnError}</p> : null}

              {hymnResult ? (
                <div className="stack">
                  <p className="muted-text">Theme: {hymnResult.hymn.theme}</p>
                  <p className="status-inline">
                    Status: <span className={`job-status ${hymnStatusLabel}`}>{hymnStatusLabel}</span>
                  </p>
                  <p className="muted-text">Provider: {hymnJob?.provider ?? hymnResult.provider}</p>

                  {hymnResult.hymn.scripture_references.length > 0 ? (
                    <section>
                      <h3>References</h3>
                      <ul className="content-list">
                        {hymnResult.hymn.scripture_references.map((ref) => (
                          <li key={ref}>{ref}</li>
                        ))}
                      </ul>
                    </section>
                  ) : null}

                  <section>
                    <h3>Lyrics</h3>
                    <div className="stack">
                      {hymnResult.hymn.sections.map((section) => (
                        <article key={`${section.label}-${section.lyrics}`} className="hymn-section">
                          <p className="summary-label">{section.label}</p>
                          <p className="passage-text">{section.lyrics}</p>
                        </article>
                      ))}
                    </div>
                  </section>

                  {hymnJob?.audio_url ? (
                    <section>
                      <h3>Audio</h3>
                      <audio controls src={hymnJob.audio_url} />
                    </section>
                  ) : (
                    <p className="empty-state">Audio pending.</p>
                  )}
                </div>
              ) : (
                <p className="empty-state">Generate a hymn.</p>
              )}
            </article>
          ) : null}
        </section>
      </main>
    </div>
  );
}
