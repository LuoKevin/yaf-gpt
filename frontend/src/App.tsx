import { useEffect, useState } from "react";
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

  const {  getStudyNotes, passageText, icebreaker, context, questions, lifeApplication } = usePassage();


  return (
    <div className="app-shell">
      <header>
        <h1>YAF-GPT</h1>
        <p>Your one-in-all spiritual learning objective platform.</p>
      </header>

      <ParableSelector onSelect={(ref) => setReference(ref)} />

      <main className="grid">
        <PassageDisplay reference={reference} text={passageText || "In progress: connect to the backend Bible service. For now this is a placeholder for the passage text."} />
        <IcebreakerQuestion question={icebreaker || "In progress: connect to the backend Bible service. For now this is a placeholder for the icebreaker question."} />
        <PassageContext context={context || "In progress: connect to the backend Bible service. For now this is a placeholder for the passage context."} />
        <PassageQuestions questions={questions.length > 0 ? questions : [{ prompt: "In progress: connect to the backend Bible service. For now this is a placeholder for the passage questions.", type: "content" }]} />
        <LifeApplication points={lifeApplication.length > 0 ? lifeApplication : ["In progress: connect to the backend Bible service. For now this is a placeholder for the life application points."]} />
        <PassageImage imageUrl={imageUrl} description={reference} />
      </main>
    </div>
  );
}

export default App;
