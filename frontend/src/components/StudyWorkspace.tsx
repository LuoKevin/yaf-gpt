import { useEffect, useMemo, useRef, useState } from "react";

import type {
  BiblePassageResponse,
  PassageImageResponse,
  StudyPlanResponse
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
const PRESET_PASSAGES = [
  { reference: "Matthew 5", label: "Matthew 5 — Sermon on the Mount" },
  { reference: "Matthew 22:34-40", label: "Matthew 22:34–40 — Greatest Commandments" },
  { reference: "Luke 10:25-37", label: "Luke 10:25–37 — Good Samaritan" },
  { reference: "Luke 15:11-32", label: "Luke 15:11–32 — Prodigal Son" },
  { reference: "John 13:1-17", label: "John 13:1–17 — Servant leadership / washing feet" },
  { reference: "John 15:1-17", label: "John 15:1–17 — Abide in me / love one another" }
] as const;

const TYPING_INTERVAL_MS = 18;
const TYPING_CHARS_PER_TICK = 3;

function normalizeBookInput(value: string) {
  return value.trim().replace(/\s+/g, " ");
}

function normalizeReferenceMatch(value: string) {
  return normalizeBookInput(value).replace(/[–—]/g, "-").toLowerCase();
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

function useTypingProgress(contentKey: string | null, totalChars: number) {
  const [visibleChars, setVisibleChars] = useState(totalChars);

  useEffect(() => {
    if (!contentKey || totalChars <= 0) {
      setVisibleChars(totalChars);
      return;
    }

    setVisibleChars(0);
    const timer = window.setInterval(() => {
      setVisibleChars((current) => {
        if (current >= totalChars) {
          window.clearInterval(timer);
          return totalChars;
        }
        return Math.min(current + TYPING_CHARS_PER_TICK, totalChars);
      });
    }, TYPING_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [contentKey, totalChars]);

  return visibleChars;
}

function sumStrings(values: string[]) {
  return values.reduce((total, value) => total + value.length, 0);
}

function sumStudyPlanListChars(items: string[], notes?: Array<string | null | undefined>) {
  return items.reduce((total, item, index) => total + item.length + (notes?.[index]?.length ?? 0), 0);
}

type StudyWorkspaceProps = {
  reference: string;
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
  onReferenceChange: (value: string) => void;
  onGoalsChange: (value: string) => void;
  onSubmitStudyRequest: () => void;
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
  onReferenceChange,
  onGoalsChange,
  onSubmitStudyRequest,
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
  const canSubmitStudyRequest = hasExactBookMatch && isRangeValid;
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
  const selectedPresetReference =
    PRESET_PASSAGES.find((preset) => normalizeReferenceMatch(preset.reference) === normalizeReferenceMatch(reference))?.reference ?? "";
  const passageTypingKey = useMemo(() => {
    if (!passage) {
      return null;
    }
    return `${passage.normalized_reference}:${passage.translation}:${passage.text}`;
  }, [passage]);
  const studyPlanTypingKey = useMemo(() => {
    if (!studyPlan) {
      return null;
    }
    return JSON.stringify({
      title: studyPlan.passage_title,
      context: studyPlan.context_points,
      discussion: studyPlan.discussion_questions,
      discussionNotes: studyPlan.discussion_question_notes,
      reflection: studyPlan.reflection_questions,
      reflectionNotes: studyPlan.reflection_question_notes,
    });
  }, [studyPlan]);
  const passageTotalChars = useMemo(() => {
    if (!passage) {
      return 0;
    }
    if (passage.verses.length > 0) {
      return sumStrings(passage.verses.map((verse) => verse.text));
    }
    return passage.text.length;
  }, [passage]);
  const studyPlanTitleChars = studyPlan?.passage_title.length ?? 0;
  const studyPlanBodyChars = useMemo(() => {
    if (!studyPlan) {
      return 0;
    }
    return (
      sumStrings(studyPlan.context_points) +
      sumStudyPlanListChars(studyPlan.discussion_questions, studyPlan.discussion_question_notes) +
      sumStudyPlanListChars(studyPlan.reflection_questions, studyPlan.reflection_question_notes)
    );
  }, [studyPlan]);
  const visiblePassageChars = useTypingProgress(passageTypingKey, passageTotalChars);
  const visibleStudyPlanTitleChars = useTypingProgress(
    studyPlanTypingKey ? `${studyPlanTypingKey}:title` : null,
    studyPlanTitleChars,
  );
  const visibleStudyPlanBodyChars = useTypingProgress(
    studyPlanTypingKey ? `${studyPlanTypingKey}:body` : null,
    studyPlanBodyChars,
  );

  function handlePresetChange(value: string) {
    if (!value) {
      return;
    }
    setIsBookMenuOpen(false);
    setBookFilterQuery("");
    setIsBookFiltering(false);
    onReferenceChange(value);
  }

  function renderTypedVerseFlow() {
    if (!passage) {
      return null;
    }

    let remainingChars = visiblePassageChars;
    return passage.verses.map((verse) => {
      if (remainingChars <= 0) {
        return null;
      }

      const visibleText = verse.text.slice(0, remainingChars);
      remainingChars = Math.max(0, remainingChars - verse.text.length);

      if (!visibleText) {
        return null;
      }

      return (
        <span key={`${verse.book}-${verse.chapter}-${verse.verse}`} className="study-verse-inline">
          <sup className="study-verse-number-inline">{verse.verse}</sup>
          <span>{visibleText}</span>{" "}
        </span>
      );
    });
  }

  function renderTypedList(
    items: string[],
    visibleChars: number,
    notes?: Array<string | null | undefined>,
  ) {
    let remainingChars = visibleChars;

    return items.map((item, index) => {
      if (remainingChars <= 0) {
        return null;
      }

      const visibleItem = item.slice(0, remainingChars);
      remainingChars = Math.max(0, remainingChars - item.length);

      if (!visibleItem) {
        return null;
      }

      const note = notes?.[index] ?? null;
      const visibleNote =
        note && visibleItem.length === item.length && remainingChars > 0
          ? note.slice(0, remainingChars)
          : "";

      if (visibleNote) {
        remainingChars = Math.max(0, remainingChars - note.length);
      }

      return (
        <li key={`${index}-${item}`}>
          {visibleItem}
          {visibleNote ? (
            <p className="question-note">{visibleNote}</p>
          ) : null}
        </li>
      );
    });
  }

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

        <div className="study-request-row">
          <div className="study-request-bar">
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
            <input
              className="study-goal-field"
              value={goals}
              onChange={(event) => onGoalsChange(event.target.value)}
              placeholder="Study Goal (Optional)"
            />
            <button
              type="button"
              className="primary-button"
              onClick={onSubmitStudyRequest}
              disabled={(isLoadingPassage || isLoadingStudyPlan) || !canSubmitStudyRequest}
            >
              {isLoadingPassage || isLoadingStudyPlan ? (
                <>
                  <span className="loading-spinner" aria-hidden="true" />
                  <span>Generating...</span>
                </>
              ) : (
                "Generate study"
              )}
            </button>
          </div>
          <div className="study-preset-card">
            <select
              className="study-preset-select"
              value={selectedPresetReference}
              onChange={(event) => handlePresetChange(event.target.value)}
              aria-label="Choose a preset Bible passage"
            >
              <option value="">Suggested passages</option>
              {PRESET_PASSAGES.map((preset) => (
                <option key={preset.reference} value={preset.reference}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {!hasExactBookMatch && normalizedBook.length > 0 ? (
          <p className="field-note">Choose a canonical Bible book from the dropdown list.</p>
        ) : null}
        {rangeInput.length > 0 && !isRangeValid ? (
          <p className="field-note">Passage range must match formats like `8`, `8:1`, `8:1-11`, or `8:1-9:4`.</p>
        ) : null}

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
                  <p className="study-passage-text">{renderTypedVerseFlow()}</p>
                </div>
              ) : (
                <p className="study-passage-text">{passage.text.slice(0, visiblePassageChars)}</p>
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
              <h3>
                {studyPlan
                  ? studyPlan.passage_title.slice(0, visibleStudyPlanTitleChars)
                  : "Waiting for a plan"}
              </h3>
            </div>
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
                  {renderTypedList(studyPlan.context_points, visibleStudyPlanBodyChars)}
                </ul>
              </section>
              <section>
                <h4>Discussion</h4>
                <ol className="prototype-list ordered">
                  {renderTypedList(
                    studyPlan.discussion_questions,
                    Math.max(0, visibleStudyPlanBodyChars - sumStrings(studyPlan.context_points)),
                    studyPlan.discussion_question_notes,
                  )}
                </ol>
              </section>
              <section>
                <h4>Reflection</h4>
                <ul className="prototype-list">
                  {renderTypedList(
                    studyPlan.reflection_questions,
                    Math.max(
                      0,
                      visibleStudyPlanBodyChars -
                        sumStrings(studyPlan.context_points) -
                        sumStudyPlanListChars(studyPlan.discussion_questions, studyPlan.discussion_question_notes),
                    ),
                    studyPlan.reflection_question_notes,
                  )}
                </ul>
              </section>
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
