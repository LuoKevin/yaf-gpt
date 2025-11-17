type IcebreakerQuestionProps = {
  question: string;
};

export function IcebreakerQuestion({ question }: IcebreakerQuestionProps) {
  return (
    <section className="panel icebreaker">
      <header>
        <h3>Icebreaker Question</h3>
      </header>
      <p>{question || "Once you request a passage, an icebreaker will appear here."}</p>
    </section>
  );
}
