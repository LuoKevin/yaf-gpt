type PassageContextProps = {
  context: string;
};

export function PassageContext({ context }: PassageContextProps) {
  return (
    <section className="panel passage-context">
      <header>
        <h3>Context</h3>
      </header>
      <p>{context || "Historical and literary context will appear here."}</p>
    </section>
  );
}
