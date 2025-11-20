type PassageContextProps = {
  context: string[];
};

export function PassageContext({ context }: PassageContextProps) {
  const list = context.length
    ? context
    : ["Historical and literary context will appear here."];

  return (
    <section className="panel passage-context">
      <header>
        <h3>Context</h3>
      </header>
      <ul>
        {list.map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>
    </section>
  );
}
