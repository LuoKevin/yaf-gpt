import { useEffect, useMemo, useRef, useState } from "react";

import type {
  BiblePassageResponse,
  PassageImageResponse,
  StudyPlanResponse,
  TranslationCode
} from "../types";

const BIBLE_BOOKS = [
  "Genesis",
  "Exodus",
  "Leviticus",
  "Numbers",
  "Deuteronomy",
  "Joshua",
  "Judges",
  "Ruth",
  "1 Samuel",
  "2 Samuel",
  "1 Kings",
  "2 Kings",
  "1 Chronicles",
  "2 Chronicles",
  "Ezra",
  "Nehemiah",
  "Esther",
  "Job",
  "Psalms",
  "Proverbs",
  "Ecclesiastes",
  "Song of Solomon",
  "Isaiah",
  "Jeremiah",
  "Lamentations",
  "Ezekiel",
  "Daniel",
  "Hosea",
  "Joel",
  "Amos",
  "Obadiah",
  "Jonah",
  "Micah",
  "Nahum",
  "Habakkuk",
  "Zephaniah",
  "Haggai",
  "Zechariah",
  "Malachi",
  "Matthew",
  "Mark",
  "Luke",
  "John",
  "Acts",
  "Romans",
  "1 Corinthians",
  "2 Corinthians",
  "Galatians",
  "Ephesians",
  "Philippians",
  "Colossians",
  "1 Thessalonians",
  "2 Thessalonians",
  "1 Timothy",
  "2 Timothy",
  "Titus",
  "Philemon",
  "Hebrews",
  "James",
  "1 Peter",
  "2 Peter",
  "1 John",
  "2 John",
  "3 John",
  "Jude",
  "Revelation"
] as const;

const PASSAGE_RANGE_PATTERN = /^\d{1,3}(?::\d{1,3})?(?:-\d{1,3}(?::\d{1,3})?)?$/;

function normalizeBookInput(value: string) {
  return value.trim().replace(/\s+/g, " ");
}

function parseReferenceParts(reference: string) {
  const cleaned = normalizeBookInput(reference);
  const matchedBook = [...BIBLE_BOOKS]
    .sort((left, right) => right.length - left.length)
    .find((book) => cleaned.toLowerCase() === book.toLowerCase() || cleaned.toLowerCase().startsWith(`${book.toLowerCase()} `));

  if (!matchedBook) {
    return {
      book: "",
      range: cleaned
    };
  }

  return {
    book: matchedBook,
    range: cleaned.slice(matchedBook.length).trim()
  };
}

type StudyWorkspaceProps = {
  reference: string;
  translation: TranslationCode;
  goals: string;
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
  onFetchPassage: () => void;
  onGeneratePlan: () => void;
  onGenerateImage: () => void;
};

export function StudyWorkspace({
  reference,
  goals,
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
  onGoalsChange,
  onFetchPassage,
  onGeneratePlan,
  onGenerateImage
}: StudyWorkspaceProps) {
  const parsedReference = useMemo(() => parseReferenceParts(reference), [reference]);
  const [bookInput, setBookInput] = useState(parsedReference.book);
  const [rangeInput, setRangeInput] = useState(parsedReference.range);
  const [isBookMenuOpen, setIsBookMenuOpen] = useState(false);
  const [bookFilterQuery, setBookFilterQuery] = useState("");
  const [isBookFiltering, setIsBookFiltering] = useState(false);
  const bookSelectorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setBookInput(parsedReference.book);
    setRangeInput(parsedReference.range);
  }, [parsedReference.book, parsedReference.range]);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!bookSelectorRef.current?.contains(event.target as Node)) {
        setIsBookMenuOpen(false);
      }
    }

    window.addEventListener("mousedown", handlePointerDown);
    return () => {
      window.removeEventListener("mousedown", handlePointerDown);
    };
  }, []);

  const normalizedBook = normalizeBookInput(bookInput);
  const hasExactBookMatch = BIBLE_BOOKS.some((book) => book.toLowerCase() === normalizedBook.toLowerCase());
  const normalizedRange = rangeInput.trim();
  const isRangeValid = normalizedRange.length > 0 && PASSAGE_RANGE_PATTERN.test(normalizedRange);
  const canSubmitReference = hasExactBookMatch && isRangeValid;
  const filteredBooks = useMemo(() => {
    if (!isBookFiltering) {
      return [...BIBLE_BOOKS];
    }
    const query = normalizeBookInput(bookFilterQuery).toLowerCase();
    if (!query) {
      return [...BIBLE_BOOKS];
    }
    return BIBLE_BOOKS.filter((book) => book.toLowerCase().includes(query));
  }, [bookFilterQuery, isBookFiltering]);

  function updateReference(nextBook: string, nextRange: string) {
    const cleanedBook = normalizeBookInput(nextBook);
    const cleanedRange = nextRange.trim();

    if (
      BIBLE_BOOKS.some((book) => book.toLowerCase() === cleanedBook.toLowerCase()) &&
      PASSAGE_RANGE_PATTERN.test(cleanedRange)
    ) {
      onReferenceChange(`${cleanedBook} ${cleanedRange}`);
    }
  }

  function handleBookChange(value: string) {
    setBookInput(value);
    setBookFilterQuery(value);
    setIsBookFiltering(true);
    setIsBookMenuOpen(true);
    updateReference(value, rangeInput);
  }

  function handleBookSelect(book: string) {
    setBookInput(book);
    setBookFilterQuery("");
    setIsBookFiltering(false);
    setIsBookMenuOpen(false);
    updateReference(book, rangeInput);
  }

  function handleRangeChange(value: string) {
    setRangeInput(value);
    if (PASSAGE_RANGE_PATTERN.test(value.trim())) {
      updateReference(bookInput, value);
    }
  }

  function toggleBookMenu() {
    setIsBookMenuOpen((current) => {
      const next = !current;
      if (next) {
        setIsBookFiltering(false);
        setBookFilterQuery("");
      }
      return next;
    });
  }

  const headingBook = parsedReference.book || "Study";
  const headingRest = parsedReference.range;

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
          <div className="study-selector-field study-book-selector" ref={bookSelectorRef}>
            <span className="material-symbols-outlined">menu_book</span>
            <input
              value={bookInput}
              onChange={(event) => handleBookChange(event.target.value)}
              onFocus={() => {
                setIsBookMenuOpen(true);
                setIsBookFiltering(false);
                setBookFilterQuery("");
              }}
              placeholder="Select a book"
              aria-expanded={isBookMenuOpen}
              aria-haspopup="listbox"
            />
            <button
              type="button"
              className={`study-selector-caret-button ${isBookMenuOpen ? "open" : ""}`}
              onMouseDown={(event) => event.preventDefault()}
              onClick={toggleBookMenu}
              aria-label={isBookMenuOpen ? "Close book selector" : "Open book selector"}
            >
              <span className="material-symbols-outlined study-selector-caret" aria-hidden="true">
                expand_more
              </span>
            </button>
            {isBookMenuOpen ? (
              <div className="study-book-dropdown" role="listbox" aria-label="Bible books">
                {filteredBooks.length > 0 ? (
                  filteredBooks.map((book) => (
                    <button
                      key={book}
                      type="button"
                      className={`study-book-option ${book.toLowerCase() === normalizedBook.toLowerCase() ? "active" : ""}`}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => handleBookSelect(book)}
                    >
                      {book}
                    </button>
                  ))
                ) : (
                  <p className="study-book-empty">No matching book.</p>
                )}
              </div>
            ) : null}
          </div>
          <input
            className="study-range-input"
            value={rangeInput}
            onChange={(event) => handleRangeChange(event.target.value)}
            placeholder="21:5-28"
            inputMode="numeric"
            pattern={PASSAGE_RANGE_PATTERN.source}
            title="Use formats like 8, 8:1, 8:1-11, or 8:1-9:4."
            aria-invalid={rangeInput.length > 0 && !isRangeValid}
          />
          <button
            type="button"
            className="primary-button"
            onClick={onFetchPassage}
            disabled={isLoadingPassage || !canSubmitReference}
          >
            {isLoadingPassage ? (
              <>
                <span className="loading-spinner" aria-hidden="true" />
                <span>Loading...</span>
              </>
            ) : (
              "Fetch passage"
            )}
          </button>
        </div>

        {!hasExactBookMatch && normalizedBook.length > 0 ? (
          <p className="field-note">Choose a canonical Bible book from the dropdown list.</p>
        ) : null}
        {rangeInput.length > 0 && !isRangeValid ? (
          <p className="field-note">Passage range must match formats like `8`, `8:1`, `8:1-11`, or `8:1-9:4`.</p>
        ) : null}

        <div className="study-goal-bar">
          <label className="field">
            <span>Study goal</span>
            <input value={goals} onChange={(event) => onGoalsChange(event.target.value)} placeholder="Set your intention..." />
          </label>
          <button type="button" className="secondary-button" onClick={onGeneratePlan} disabled={isLoadingStudyPlan}>
            {isLoadingStudyPlan ? (
              <>
                <span className="loading-spinner" aria-hidden="true" />
                <span>Generating...</span>
              </>
            ) : (
              "Get study"
            )}
          </button>
          <button type="button" className="ghost-button" onClick={onGenerateImage} disabled={isLoadingPassageImage}>
            {isLoadingPassageImage ? (
              <>
                <span className="loading-spinner" aria-hidden="true" />
                <span>Generating...</span>
              </>
            ) : (
              "Generate image"
            )}
          </button>
        </div>

        {passageError ? <p className="error-banner">{passageError}</p> : null}

        <article className="study-passage-sheet study-passage-sheet-compact">
          {isLoadingPassage ? (
            <div className="loading-panel">
              <span className="loading-spinner" aria-hidden="true" />
              <p className="empty-state">Loading passage...</p>
            </div>
          ) : passage ? (
            <>
              <div className="study-passage-heading">
                <div>
                  <p className="section-label">Passage reference</p>
                  <h3>{passage.reference}</h3>
                </div>
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

        <div className="study-feature-grid">
          <article className="prototype-card study-plan-card">
          <div className="card-header">
            <div>
              <p className="section-label">Study plan</p>
              <h3>{studyPlan?.passage_title ?? "Waiting for a plan"}</h3>
            </div>
            {studyPlan ? <span className="surface-pill">{studyPlan.model}</span> : null}
          </div>

          {studyPlanError ? <p className="error-banner">{studyPlanError}</p> : null}

          {isLoadingStudyPlan ? (
            <div className="loading-panel">
              <span className="loading-spinner" aria-hidden="true" />
              <p className="empty-state">Generating study plan...</p>
            </div>
          ) : studyPlan ? (
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

          <article className="prototype-card study-image-card">
            <div className="card-header">
              <div>
                <p className="section-label">Visual context</p>
                <h3>{passageImage ? "Generated image" : "Awaiting image"}</h3>
              </div>
            </div>
            {passageImageError ? <p className="error-banner">{passageImageError}</p> : null}
            {isLoadingPassageImage ? (
              <div className="loading-panel">
                <span className="loading-spinner" aria-hidden="true" />
                <p className="empty-state">Generating image...</p>
              </div>
            ) : passageImage ? (
              <div className="card-stack">
                <img className="image-preview" src={passageImage.image_b64_or_url} alt={passageImage.alt_text} />
                <p className="prompt-note">{passageImage.alt_text}</p>
              </div>
            ) : (
              <p className="empty-state">Generate an image to add a visual metaphor for the passage.</p>
            )}
          </article>
        </div>
      </div>
    </section>
  );
}
