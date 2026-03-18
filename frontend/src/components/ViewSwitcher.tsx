import type { ViewMode } from "../types";

type ViewSwitcherProps = {
  activeView: ViewMode;
  onChange: (view: ViewMode) => void;
};

export function ViewSwitcher({ activeView, onChange }: ViewSwitcherProps) {
  return (
    <section className="view-switcher">
      <button
        type="button"
        className={`view-button ${activeView === "study" ? "active" : ""}`}
        onClick={() => onChange("study")}
      >
        Study
      </button>
      <button
        type="button"
        className={`view-button ${activeView === "music" ? "active" : ""}`}
        onClick={() => onChange("music")}
      >
        Music
      </button>
      <button
        type="button"
        className={`view-button ${activeView === "discussion" ? "active" : ""}`}
        onClick={() => onChange("discussion")}
      >
        Discussion
      </button>
    </section>
  );
}
