type PassageQuestionsProps = {
  questions: string[];
};

export function PassageQuestions({ questions }: PassageQuestionsProps) {
  const list = questions.length
    ? questions
    : ["Questions will appear here once a passage is generated."];

  return (
    <section className="panel passage-questions">
      <header>
        <h3>Passage Questions</h3>
      </header>
      <ul>
        {list.map((question, index) => (
          <li key={`${question}-${index}`}>{question}</li>
        ))}
      </ul>
    </section>
  );
}
