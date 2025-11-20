import { useState } from "react";
import parables from "./parables.json";

type ParableSelectorProps = {
  label?: string;
  onSelect: (reference: string) => void;
};

type ParablesByBook = Record<
  string,
  Array<{
    title: string;
    reference: string;
  }>
>;

export function ParableSelector({ label = "Choose a parable", onSelect }: ParableSelectorProps) {
  const data = parables as ParablesByBook;
  const books = Object.keys(data);
  const [selectedReference, setSelectedReference] = useState("");

  function handleChange(event: React.ChangeEvent<HTMLSelectElement>) {
    const value = event.target.value;
    setSelectedReference(value);
    // if (value) {
    //   onSelect(value);
    // }
  }

  return (
    <div className="panel parable-selector">
      <label htmlFor="parable-select">{label}</label>
      <select id="parable-select" value={selectedReference} onChange={handleChange}>
        <option value="">-- Select a parable --</option>
        {books.map((book) => (
          <optgroup key={book} label={book}>
            {data[book].map((parable) => (
              <option key={parable.reference} value={parable.reference}>
                {parable.title} ({parable.reference})
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      <button onClick={() => onSelect(selectedReference)} type="submit">Generate</button>
    </div>
  );
}
