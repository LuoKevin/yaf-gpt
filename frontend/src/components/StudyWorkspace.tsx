import type {
  BiblePassageResponse,
  PassageImageResponse,
  StudyPlanResponse,
  TranslationCode
} from "../types";

type StudyWorkspaceProps = {
  reference: string;
  translation: TranslationCode;
  goals: string;
  userNotes: string;
  includeQuestionNotes: boolean;
  passage: BiblePassageResponse | null;
  studyPlan: StudyPlanResponse | null;
  passageError: string;
  studyPlanError: string;
  passageImage: PassageImageResponse | null;
  passageImageError: string;
  isLoadingPassage: boolean;
  isLoadingStudyPlan: boolean;
  isLoadingPassageImage: boolean;
  hasUsage: boolean | null;
  onReferenceChange: (value: string) => void;
  onTranslationChange: (value: TranslationCode) => void;
  onGoalsChange: (value: string) => void;
  onUserNotesChange: (value: string) => void;
  onIncludeQuestionNotesChange: (value: boolean) => void;
  onFetchPassage: () => void;
  onGeneratePlan: () => void;
  onGenerateImage: () => void;
};

export function StudyWorkspace({
  reference,
  translation,
  goals,
  userNotes,
  includeQuestionNotes,
  passage,
  studyPlan,
  passageError,
  studyPlanError,
  passageImage,
  passageImageError,
  isLoadingPassage,
  isLoadingStudyPlan,
  isLoadingPassageImage,
  hasUsage,
  onReferenceChange,
  onTranslationChange,
  onGoalsChange,
  onUserNotesChange,
  onIncludeQuestionNotesChange,
  onFetchPassage,
  onGeneratePlan,
  onGenerateImage
}: StudyWorkspaceProps) {
  return (
    <>
      <section className="panel control-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-kicker">Inputs</p>
            <h2>Study</h2>
          </div>
        </div>

        <label className="field">
          <span>Reference</span>
          <input value={reference} onChange={(event) => onReferenceChange(event.target.value)} placeholder="Luke 21:5-28" />
        </label>

        <label className="field">
          <span>Translation</span>
          <select value={translation} onChange={(event) => onTranslationChange(event.target.value as TranslationCode)}>
            <option value="WEB">WEB</option>
            <option value="KJV">KJV</option>
          </select>
        </label>

        <label className="field">
          <span>Goals</span>
          <textarea rows={3} value={goals} onChange={(event) => onGoalsChange(event.target.value)} placeholder="Optional" />
        </label>

        <label className="field">
          <span>Notes</span>
          <textarea
            rows={3}
            value={userNotes}
            onChange={(event) => onUserNotesChange(event.target.value)}
            placeholder="Optional"
          />
        </label>

        <label className="field field-inline">
          <span>Include question notes</span>
          <input
            type="checkbox"
            checked={includeQuestionNotes}
            onChange={(event) => onIncludeQuestionNotesChange(event.target.checked)}
          />
        </label>

        <div className="action-row action-row-single">
          <button type="button" className="primary-button" onClick={onFetchPassage} disabled={isLoadingPassage}>
            {isLoadingPassage ? "Loading..." : "Fetch passage"}
          </button>
          <button type="button" className="secondary-button" onClick={onGeneratePlan} disabled={isLoadingStudyPlan}>
            {isLoadingStudyPlan ? "Generating..." : "Generate plan"}
          </button>
          <button type="button" className="secondary-button" onClick={onGenerateImage} disabled={isLoadingPassageImage}>
            {isLoadingPassageImage ? "Generating..." : "Generate image"}
          </button>
        </div>
      </section>

      <section className="results-column">
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
                  Tokens: {studyPlan.usage?.prompt_tokens ?? 0} / {studyPlan.usage?.completion_tokens ?? 0} /{" "}
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
      </section>
    </>
  );
}
