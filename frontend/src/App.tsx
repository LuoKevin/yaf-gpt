import { useState } from "react";
import "./App.css";
import { VerseInput } from "./components/VerseInput";
import { PassageDisplay } from "./components/PassageDisplay";
import { IcebreakerQuestion } from "./components/IcebreakerQuestion";
import { PassageContext } from "./components/PassageContext";
import { LifeApplication } from "./components/LifeApplication";
import { PassageImage } from "./components/PassageImage";

function App() {
  const [reference, setReference] = useState("");
  const [passageText, setPassageText] = useState("");
  const [icebreaker, setIcebreaker] = useState("");
  const [context, setContext] = useState("");
  const [lifeApplication, setLifeApplication] = useState<string[]>([]);
  const [imageUrl, setImageUrl] = useState<string | undefined>();

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
  }

  return (
    <div className="app-shell">
      <header>
        <h1>Bible Study Companion</h1>
        <p>Generate study notes, context, and visuals for any passage.</p>
      </header>

      <VerseInput onSubmit={handleGeneratePassage} />

      <main className="grid">
        <PassageDisplay reference={reference} text={passageText} />
        <IcebreakerQuestion question={icebreaker} />
        <PassageContext context={context} />
        <LifeApplication points={lifeApplication} />
        <PassageImage imageUrl={imageUrl} description={reference} />
      </main>
    </div>
  );
}

export default App;
