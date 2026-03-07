import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

const SUGGESTED_REFERENCES = [
  "Luke 21:5-28",
  "Romans 8:28-39",
  "John 15:1-11",
  "Psalm 23:1-6"
] as const;

type TranslationCode = "WEB" | "KJV";
type HealthStatus = "checking" | "online" | "offline";

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

  const activeVerseCount = passage?.verses.length ?? 0;
  const hasUsage = studyPlan?.usage && studyPlan.usage.total_tokens !== null;

  return (
    <div className="shell">
      <div className="background-orb background-orb-left" />
      <div className="background-orb background-orb-right" />

      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">yaf-gpt</p>
          <h1>Scripture lookup and study prep in one TypeScript workspace.</h1>
          <p className="hero-text">
            Fetch a passage, inspect the text, and generate a young-adult discussion plan against
            the FastAPI backend already in this repo.
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
          <button
            key={suggestion}
            type="button"
            className="chip"
            onClick={() => setReference(suggestion)}
          >
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
            <select
              value={translation}
              onChange={(event) => setTranslation(event.target.value as TranslationCode)}
            >
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

          <div className="action-row">
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
          </div>

          <p className="panel-note">
            The study-plan request reuses the fetched passage text when possible so the backend does
            not need to perform the same lookup twice.
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
                      <article
                        key={`${verse.book}-${verse.chapter}-${verse.verse}`}
                        className="verse-card"
                      >
                        <p className="verse-label">
                          {verse.book} {verse.chapter}:{verse.verse}
                        </p>
                        <p>{verse.text}</p>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted-text">
                    Verse-by-verse detail is not available for this result because it came from the
                    study-plan response rather than the direct Bible lookup endpoint.
                  </p>
                )}
              </div>
            ) : (
              <p className="empty-state">
                Fetch a passage to see the text and verse breakdown here.
              </p>
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
                    Token usage: {studyPlan.usage?.prompt_tokens ?? 0} prompt,{" "}
                    {studyPlan.usage?.completion_tokens ?? 0} completion,{" "}
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
        </section>
      </main>
    </div>
  );
}
