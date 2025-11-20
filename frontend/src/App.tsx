import { useState } from "react";
import "./App.css";
import { PassageDisplay } from "./components/PassageDisplay";
import { IcebreakerQuestion } from "./components/IcebreakerQuestion";
import { PassageContext } from "./components/PassageContext";
import { LifeApplication } from "./components/LifeApplication";
import { PassageImage } from "./components/PassageImage";
import { PassageQuestions } from "./components/PassageQuestions";
import { ParableSelector } from "./components/ParableSelector";
import { usePassage } from "./hooks/usePassage";

function App() {
  const [reference, setReference] = useState("");
  const [imageUrl, setImageUrl] = useState<string | undefined>();
  const {
    getStudyNotes,
    passageText,
    icebreaker,
    context,
    questions,
    lifeApplication,
  } = usePassage();

  function handlePassageSelect(ref: string) {
    setReference(ref);
    setImageUrl(undefined);
    getStudyNotes(ref).catch(() => undefined);
  }

  const placeholderPassage =
    passageText ||
    "In progress: connect to the backend Bible service. For now this is a placeholder for the passage text.";
  const placeholderIcebreaker =
    icebreaker ||
    "In progress: connect to the backend Bible service. For now this is a placeholder for the icebreaker question.";
  const placeholderContext =
    context.length > 0
      ? context
      : ["In progress: connect to the backend Bible service. For now this is a placeholder for the passage context."];
  const placeholderQuestions =
    questions.length > 0
      ? questions
      : ["In progress: connect to the backend Bible service. For now this is a placeholder for the passage questions."];
  const placeholderLifeApplication =
    lifeApplication.length > 0
      ? lifeApplication
      : ["In progress: connect to the backend Bible service. For now this is a placeholder for the life application points."];

  return (
    <div className="app-shell">
      <header>
        <h1>YAF-GPT</h1>
        <p>Your one-in-all spiritual learning objective platform.</p>
      </header>

      <ParableSelector onSelect={handlePassageSelect} />

      <main className="grid">
        <PassageDisplay reference={reference} text={placeholderPassage} />
        <IcebreakerQuestion question={placeholderIcebreaker} />
        <PassageContext context={placeholderContext} />
        <PassageQuestions questions={placeholderQuestions} />
        <LifeApplication points={placeholderLifeApplication} />
        <PassageImage imageUrl={imageUrl} description={reference} />
      </main>
    </div>
  );
}

export default App;
