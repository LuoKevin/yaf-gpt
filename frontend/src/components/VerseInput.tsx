import { FormEvent, useState } from "react";

type VerseInputProps = {
  initialReference?: string;
  onSubmit: (reference: string) => void;
};

export function VerseInput({ initialReference = "", onSubmit }: VerseInputProps) {
  const [reference, setReference] = useState(initialReference);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (reference.trim().length === 0) return;
    onSubmit(reference.trim());
  }

  return (
    <form className="panel verse-input" onSubmit={handleSubmit}>
      <label htmlFor="reference">Bible Passage</label>
      <div className="input-row">
        <input
          id="reference"
          name="reference"
          placeholder="e.g., Luke 11:1-13"
          value={reference}
          onChange={(event) => setReference(event.target.value)}
        />
        <button type="submit">Generate</button>
      </div>
    </form>
  );
}
