type PassageQuestionsProps = {
  questions: {
    prompt: string;
    type: "content" | "application";
  }[];
};

export function PassageQuestions({ questions }: PassageQuestionsProps) {
  const grouped = questions.length
    ? {
        content: questions.filter((q) => q.type === "content"),
        application: questions.filter((q) => q.type === "application"),
      }
    : {
        content: [],
        application: [],
      };

  return (
    <section className="panel passage-questions">
      <header>
        <h3>Passage Questions</h3>
      </header>
      {grouped.content.length === 0 && grouped.application.length === 0 ? (
        <p>Questions will appear here once a passage is generated.</p>
      ) : (
        <>
          {grouped.content.length > 0 && (
            <div className="question-block">
              <h4>Content</h4>
              <ul>
                {grouped.content.map((question, index) => (
                  <li key={`content-${index}`}>{question.prompt}</li>
                ))}
              </ul>
            </div>
          )}
          {grouped.application.length > 0 && (
            <div className="question-block">
              <h4>Application</h4>
              <ul>
                {grouped.application.map((question, index) => (
                  <li key={`application-${index}`}>{question.prompt}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </section>
  );
}
