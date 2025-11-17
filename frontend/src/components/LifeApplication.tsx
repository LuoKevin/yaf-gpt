type LifeApplicationProps = {
  points: string[];
};

export function LifeApplication({ points }: LifeApplicationProps) {
  const list = points.length > 0 ? points : ["Application ideas will be generated after you submit a passage."];

  return (
    <section className="panel life-application">
      <header>
        <h3>Life Application</h3>
      </header>
      <ul>
        {list.map((point, index) => (
          <li key={`${point}-${index}`}>{point}</li>
        ))}
      </ul>
    </section>
  );
}
