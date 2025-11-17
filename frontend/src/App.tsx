import { useState } from "react";
import "./App.css";
import { VerseInput } from "./components/VerseInput";
import { PassageDisplay } from "./components/PassageDisplay";
import { IcebreakerQuestion } from "./components/IcebreakerQuestion";
import { PassageContext } from "./components/PassageContext";
import { LifeApplication } from "./components/LifeApplication";
import { PassageImage } from "./components/PassageImage";
import { PassageQuestions } from "./components/PassageQuestions";
import { ParableSelector } from "./components/ParableSelector";

function App() {
  const [reference, setReference] = useState("");
  const [passageText, setPassageText] = useState("");
  const [icebreaker, setIcebreaker] = useState("");
  const [context, setContext] = useState("");
  const [lifeApplication, setLifeApplication] = useState<string[]>([]);
  const [imageUrl, setImageUrl] = useState<string | undefined>();
  const [questions, setQuestions] = useState<
    { prompt: string; type: "content" | "application" }[]
  >([]);

  function handleGeneratePassage(ref: string) {
    setReference(ref);
    // Placeholder data while backend integration is in progress
    setPassageText(
      "In progress: connect to the backend Bible service. For now this is a placeholder for the passage text."
    );
    setIcebreaker("Share a moment when prayer felt particularly powerful to you.");
    setContext("These verses highlight how Jesus teaches persistence in prayer and reliance on the Father.");
    setLifeApplication([
      "Identify one area of your life where you can ask, seek, and knock with renewed faith.",
      "Encourage a friend or your small group to practice persistent prayer together this week.",
    ]);
    setImageUrl(undefined);
    setQuestions([
      { prompt: "What does Jesus teach about persistence in prayer?", type: "content" },
      { prompt: "How have you seen persistence answered in your life?", type: "application" },
    ]);
  }

  return (
    <div className="app-shell">
      <header>
        <h1>YAF-GPT</h1>
        <p>Your one-in-all spiritual learning objective platform.</p>
      </header>

      <ParableSelector onSelect={handleGeneratePassage} />

      <main className="grid">
        <PassageDisplay reference={reference} text={passageText} />
        <IcebreakerQuestion question={icebreaker} />
        <PassageContext context={context} />
        <PassageQuestions questions={questions} />
        <LifeApplication points={lifeApplication} />
        <PassageImage imageUrl={imageUrl} description={reference} />
      </main>
    </div>
  );
}

export default App;
