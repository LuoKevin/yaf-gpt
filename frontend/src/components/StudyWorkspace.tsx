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
  const headingParts = reference.trim().split(" ");
  const headingBook = headingParts[0] || "Study";
  const headingRest = reference.trim().slice(headingBook.length).trim();

  return (
    <section className="study-prototype">
      <div className="study-main-column">
        <div className="workspace-header">
          <p className="workspace-kicker">Study</p>
          <h1>
            {headingBook} <span>{headingRest}</span>
          </h1>
          <p className="workspace-copy">
            Build a guided reading flow, keep the text centered, and let the study plan sit beside the passage instead of taking over it.
          </p>
        </div>

        <div className="study-selector-bar">
          <div className="study-selector-field">
            <span className="material-symbols-outlined">menu_book</span>
            <input value={reference} onChange={(event) => onReferenceChange(event.target.value)} placeholder="Luke 21:5-28" />
          </div>
          <div className="study-selector-divider" />
          <select value={translation} onChange={(event) => onTranslationChange(event.target.value as TranslationCode)}>
            <option value="WEB">WEB</option>
            <option value="KJV">KJV</option>
          </select>
          <button type="button" className="primary-button" onClick={onFetchPassage} disabled={isLoadingPassage}>
            {isLoadingPassage ? "Loading..." : "Fetch passage"}
          </button>
        </div>

        <div className="study-goal-bar">
          <label className="field">
            <span>Study goal</span>
            <input value={goals} onChange={(event) => onGoalsChange(event.target.value)} placeholder="Set your intention..." />
          </label>
          <button type="button" className="secondary-button" onClick={onGeneratePlan} disabled={isLoadingStudyPlan}>
            {isLoadingStudyPlan ? "Generating..." : "Get study"}
          </button>
          <button type="button" className="ghost-button" onClick={onGenerateImage} disabled={isLoadingPassageImage}>
            {isLoadingPassageImage ? "Generating..." : "Generate image"}
          </button>
        </div>

        {passageError ? <p className="error-banner">{passageError}</p> : null}

        <article className="study-passage-sheet">
          {passage ? (
            <>
              <div className="study-passage-heading">
                <p className="section-label">Passage</p>
                <span className="surface-pill">{passage.translation}</span>
              </div>
              {passage.verses.length > 0 ? (
                <div className="study-verse-flow">
                  {passage.verses.map((verse) => (
                    <div key={`${verse.book}-${verse.chapter}-${verse.verse}`} className="study-verse-row">
                      <span className="study-verse-number">{verse.verse}</span>
                      <p>{verse.text}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="study-passage-text">{passage.text}</p>
              )}
            </>
          ) : (
            <p className="empty-state">Fetch a passage to begin the study workspace.</p>
          )}
        </article>
      </div>

      <aside className="study-side-column">
        <article className="prototype-card">
          <div className="card-header">
            <div>
              <p className="section-label">Study plan</p>
              <h3>{studyPlan?.passage_title ?? "Waiting for a plan"}</h3>
            </div>
            {studyPlan ? <span className="surface-pill">{studyPlan.model}</span> : null}
          </div>

          {studyPlanError ? <p className="error-banner">{studyPlanError}</p> : null}

          {studyPlan ? (
            <div className="card-stack">
              <section>
                <h4>Context</h4>
                <ul className="prototype-list">
                  {studyPlan.context_points.map((point) => (
                    <li key={point}>{point}</li>
                  ))}
                </ul>
              </section>
              <section>
                <h4>Discussion</h4>
                <ol className="prototype-list ordered">
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
                <h4>Reflection</h4>
                <ul className="prototype-list">
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
            <p className="empty-state">Generate a study plan after setting your passage and goal.</p>
          )}
        </article>

        <article className="prototype-card">
          <div className="card-header">
            <div>
              <p className="section-label">Visual context</p>
              <h3>{passageImage ? "Generated image" : "Awaiting image"}</h3>
            </div>
          </div>
          {passageImageError ? <p className="error-banner">{passageImageError}</p> : null}
          {passageImage ? (
            <div className="card-stack">
              <img className="image-preview" src={passageImage.image_b64_or_url} alt={passageImage.alt_text} />
              <p className="prompt-note">{passageImage.alt_text}</p>
            </div>
          ) : (
            <p className="empty-state">Generate an image to add a visual metaphor for the passage.</p>
          )}
        </article>

        <article className="prototype-card">
          <div className="card-header">
            <div>
              <p className="section-label">Personal notes</p>
              <h3>Marginalia</h3>
            </div>
          </div>
          <label className="field">
            <span>Notes</span>
            <textarea rows={6} value={userNotes} onChange={(event) => onUserNotesChange(event.target.value)} placeholder="Pen your insights here..." />
          </label>
          <label className="toggle-row">
            <span>Include question notes</span>
            <input
              type="checkbox"
              checked={includeQuestionNotes}
              onChange={(event) => onIncludeQuestionNotesChange(event.target.checked)}
            />
          </label>
        </article>
      </aside>
    </section>
  );
}
