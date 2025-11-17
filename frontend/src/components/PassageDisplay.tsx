type PassageDisplayProps = {
  reference: string;
  text: string;
};

export function PassageDisplay({ reference, text }: PassageDisplayProps) {
  return (
    <section className="panel passage-display">
      <header>
        <h2>{reference || "Selected Passage"}</h2>
      </header>
      <p>{text || "Enter a passage above to load the text."}</p>
    </section>
  );
}
