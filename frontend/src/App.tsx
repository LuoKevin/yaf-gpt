import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

const SUGGESTED_REFERENCES = [
  "Luke 21:5-28",
  "Romans 8:28-39",
  "John 15:1-11",
  "Psalm 23:1-6"
] as const;

type TranslationCode = "WEB" | "KJV";
type ImageStyle = "modern_editorial_illustration";
type HealthStatus = "checking" | "online" | "offline";
type PersonaChatRole = "user" | "assistant";
type HymnJobStatus = "queued" | "in_progress" | "completed" | "failed";

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

export default function App() {
  const [reference, setReference] = useState("Luke 21:5-28");
  const [translation, setTranslation] = useState<TranslationCode>("WEB");
  const [goals, setGoals] = useState(
    "Help our young adult group understand the passage flow and end with concrete reflection."
  );
  const [userNotes, setUserNotes] = useState(
    "Mixed group of newer and longtime Christians. Keep the plan discussion-friendly."
  );

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
          setHymnError(error instanceof Error ? error.message : "Unable to refresh hymn job status.");
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

  async function handlePassageLookup() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setPassageError("Enter a Bible reference before fetching a passage.");
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
      setStudyPlanError("Enter a Bible reference before generating a study plan.");
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
      setPassageImageError("Enter a Bible reference before generating an image.");
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
      setPassageImageError(error instanceof Error ? error.message : "Unable to generate passage image.");
    } finally {
      setIsLoadingPassageImage(false);
    }
  }

  async function handlePersonaSend() {
    const userMessage = personaInput.trim();
    if (!userMessage) {
      setPersonaError("Type a message before sending.");
      return;
    }

    const nextMessages = [...personaMessages, { role: "user", content: userMessage } as PersonaChatMessage];
    setPersonaMessages(nextMessages);
    setPersonaInput("");
    setPersonaError("");
    setIsSendingPersona(true);

    try {
      const response = await requestJson<PersonaChatResponse>("/api/persona-chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          messages: nextMessages,
          reference_context: reference.trim() || undefined,
          translation
        })
      });

      setPersonaMessages((current) => [...current, { role: "assistant", content: response.reply }]);
      setPersonaModel(response.model);
    } catch (error) {
      setPersonaError(error instanceof Error ? error.message : "Unable to send persona message.");
    } finally {
      setIsSendingPersona(false);
    }
  }

  function handlePersonaReset() {
    setPersonaMessages([]);
    setPersonaModel(null);
    setPersonaError("");
  }

  async function handleHymnGeneration() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setHymnError("Enter a Bible reference before generating a hymn.");
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

  const activeVerseCount = passage?.verses.length ?? 0;
  const hasUsage = studyPlan?.usage && studyPlan.usage.total_tokens !== null;
  const hymnStatusLabel = useMemo(() => {
    if (!hymnJobStatus) {
      return "not-started";
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
          <h1>Scripture lookup, planning, imagery, mentoring, and hymn drafting.</h1>
          <p className="hero-text">
            Build full study sessions in one place: fetch a passage, generate discussion plans,
            create passage illustrations, chat with a mentor persona, and draft a hymn with async
            music generation.
          </p>
        </div>

        <div className={`status-pill ${healthStatus}`}>
          <span className="status-dot" />
          <span>
            API {healthStatus === "checking" ? "checking" : healthStatus === "online" ? "online" : "offline"}
          </span>
        </div>
      </header>

      <section className="summary-grid">
        <article className="summary-card">
          <p className="summary-label">Translation</p>
          <strong>{translation}</strong>
        </article>
        <article className="summary-card">
          <p className="summary-label">Loaded verses</p>
          <strong>{activeVerseCount}</strong>
        </article>
        <article className="summary-card">
          <p className="summary-label">Study questions</p>
          <strong>{studyPlan?.discussion_questions.length ?? 0}</strong>
        </article>
      </section>

      <section className="suggestion-strip">
        <span>Quick references</span>
        {SUGGESTED_REFERENCES.map((suggestion) => (
          <button key={suggestion} type="button" className="chip" onClick={() => setReference(suggestion)}>
            {suggestion}
          </button>
        ))}
      </section>

      <main className="workspace">
        <section className="panel control-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Inputs</p>
              <h2>Study setup</h2>
            </div>
          </div>

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

          <label className="field">
            <span>Group goals</span>
            <textarea
              rows={4}
              value={goals}
              onChange={(event) => setGoals(event.target.value)}
              placeholder="What should the group walk away understanding?"
            />
          </label>

          <label className="field">
            <span>Leader notes</span>
            <textarea
              rows={4}
              value={userNotes}
              onChange={(event) => setUserNotes(event.target.value)}
              placeholder="Anything the backend should consider for tone or context?"
            />
          </label>

          <label className="field">
            <span>Hymn style hint</span>
            <input
              value={hymnStyle}
              onChange={(event) => setHymnStyle(event.target.value)}
              placeholder="modern worship hymn, acoustic"
            />
          </label>

          <label className="field">
            <span>Hymn mood hint</span>
            <input
              value={hymnMood}
              onChange={(event) => setHymnMood(event.target.value)}
              placeholder="hopeful"
            />
          </label>

          <div className="action-row action-row-single">
            <button
              type="button"
              className="primary-button"
              onClick={handlePassageLookup}
              disabled={isLoadingPassage}
            >
              {isLoadingPassage ? "Loading passage..." : "Fetch passage"}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={handleStudyPlanGeneration}
              disabled={isLoadingStudyPlan}
            >
              {isLoadingStudyPlan ? "Generating..." : "Generate study plan"}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={handlePassageImageGeneration}
              disabled={isLoadingPassageImage}
            >
              {isLoadingPassageImage ? "Generating image..." : "Generate passage image"}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={handleHymnGeneration}
              disabled={isGeneratingHymn}
            >
              {isGeneratingHymn ? "Composing hymn..." : "Generate hymn"}
            </button>
          </div>

          <p className="panel-note">
            Study-plan and hymn requests reuse fetched passage text when possible so the backend can
            avoid duplicate lookups.
          </p>
        </section>

        <section className="results-column">
          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Passage</p>
                <h2>{passage?.normalized_reference ?? "No passage loaded yet"}</h2>
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
                ) : (
                  <p className="muted-text">
                    Verse-by-verse detail is not available for this result because it came from a
                    generation endpoint rather than the direct Bible lookup endpoint.
                  </p>
                )}
              </div>
            ) : (
              <p className="empty-state">Fetch a passage to see the text and verse breakdown here.</p>
            )}
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Study plan</p>
                <h2>{studyPlan?.passage_title ?? "No study plan generated yet"}</h2>
              </div>
              {studyPlan && <span className="meta-badge">{studyPlan.model}</span>}
            </div>

            {studyPlanError ? <p className="error-banner">{studyPlanError}</p> : null}

            {studyPlan ? (
              <div className="stack">
                <section>
                  <h3>Context points</h3>
                  <ul className="content-list">
                    {studyPlan.context_points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                </section>

                <section>
                  <h3>Discussion questions</h3>
                  <ol className="content-list ordered-list">
                    {studyPlan.discussion_questions.map((question) => (
                      <li key={question}>{question}</li>
                    ))}
                  </ol>
                </section>

                <section>
                  <h3>Reflection questions</h3>
                  <ul className="content-list">
                    {studyPlan.reflection_questions.map((question) => (
                      <li key={question}>{question}</li>
                    ))}
                  </ul>
                </section>

                {hasUsage ? (
                  <p className="usage-note">
                    Token usage: {studyPlan.usage?.prompt_tokens ?? 0} prompt, {" "}
                    {studyPlan.usage?.completion_tokens ?? 0} completion, {" "}
                    {studyPlan.usage?.total_tokens ?? 0} total.
                  </p>
                ) : null}
              </div>
            ) : (
              <p className="empty-state">
                Generate a study plan to render context points, discussion prompts, and reflection
                questions.
              </p>
            )}
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Passage image</p>
                <h2>{passageImage ? "Illustration ready" : "No image generated yet"}</h2>
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
              <p className="empty-state">
                Generate a passage image to visualize a scene or theme anchored to the selected
                Scripture.
              </p>
            )}
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Persona mentor</p>
                <h2>Interactive discussion</h2>
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
                <p className="empty-state">
                  Ask a question about this passage to begin mentor-style discussion.
                </p>
              )}

              <label className="field">
                <span>Message</span>
                <textarea
                  rows={3}
                  value={personaInput}
                  onChange={(event) => setPersonaInput(event.target.value)}
                  placeholder="What is Jesus emphasizing in this part of the passage?"
                />
              </label>

              <div className="mini-action-row">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handlePersonaSend}
                  disabled={isSendingPersona}
                >
                  {isSendingPersona ? "Sending..." : "Send to mentor"}
                </button>
                <button type="button" className="secondary-button" onClick={handlePersonaReset}>
                  Reset chat
                </button>
              </div>
            </div>
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Hymn studio</p>
                <h2>{hymnResult?.hymn.title ?? "No hymn generated yet"}</h2>
              </div>
              {hymnResult ? <span className="meta-badge">{hymnResult.model}</span> : null}
            </div>

            {hymnError ? <p className="error-banner">{hymnError}</p> : null}

            {hymnResult ? (
              <div className="stack">
                <p className="muted-text">Theme: {hymnResult.hymn.theme}</p>
                <p className="status-inline">
                  Music job status: <span className={`job-status ${hymnStatusLabel}`}>{hymnStatusLabel}</span>
                </p>
                <p className="muted-text">Music provider: {hymnJob?.provider ?? hymnResult.provider}</p>

                {hymnResult.hymn.scripture_references.length > 0 ? (
                  <section>
                    <h3>Scripture references</h3>
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
                    <h3>Audio preview</h3>
                    <audio controls src={hymnJob.audio_url} />
                  </section>
                ) : (
                  <p className="muted-text">
                    Audio preview will appear when the async music job reaches completed status.
                  </p>
                )}
              </div>
            ) : (
              <p className="empty-state">
                Generate a hymn to produce Scripture-anchored lyrics and start a background music
                job.
              </p>
            )}
          </article>
        </section>
      </main>
    </div>
  );
}
