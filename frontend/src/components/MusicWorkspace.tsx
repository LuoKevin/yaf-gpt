import type { MusicGenerateResponse, MusicJobResponse, MusicJobStatus, TranslationCode } from "../types";

type MusicWorkspaceProps = {
  reference: string;
  translation: TranslationCode;
  musicTitle: string;
  musicPrompt: string;
  musicStyle: string;
  musicMood: string;
  musicResult: MusicGenerateResponse | null;
  musicJob: MusicJobResponse | null;
  musicError: string;
  musicStatusLabel: MusicJobStatus | "queued";
  isGeneratingMusic: boolean;
  onReferenceChange: (value: string) => void;
  onTranslationChange: (value: TranslationCode) => void;
  onMusicTitleChange: (value: string) => void;
  onMusicPromptChange: (value: string) => void;
  onMusicStyleChange: (value: string) => void;
  onMusicMoodChange: (value: string) => void;
  onGenerateMusic: () => void;
};

export function MusicWorkspace({
  reference,
  translation,
  musicTitle,
  musicPrompt,
  musicStyle,
  musicMood,
  musicResult,
  musicJob,
  musicError,
  musicStatusLabel,
  isGeneratingMusic,
  onReferenceChange,
  onTranslationChange,
  onMusicTitleChange,
  onMusicPromptChange,
  onMusicStyleChange,
  onMusicMoodChange,
  onGenerateMusic
}: MusicWorkspaceProps) {
  return (
    <>
      <section className="panel control-panel">
        <div className="panel-heading">
          <div>
            <p className="panel-kicker">Inputs</p>
            <h2>Music</h2>
          </div>
        </div>

        <label className="field">
          <span>Reference</span>
          <input value={reference} onChange={(event) => onReferenceChange(event.target.value)} placeholder="Luke 21:5-28" />
        </label>

        <label className="field">
          <span>Translation</span>
          <select value={translation} onChange={(event) => onTranslationChange(event.target.value as TranslationCode)}>
            <option value="WEB">WEB</option>
            <option value="KJV">KJV</option>
          </select>
        </label>

        <label className="field">
          <span>Track title</span>
          <input value={musicTitle} onChange={(event) => onMusicTitleChange(event.target.value)} placeholder="Optional" />
        </label>

        <label className="field">
          <span>Prompt</span>
          <textarea
            rows={4}
            value={musicPrompt}
            onChange={(event) => onMusicPromptChange(event.target.value)}
            placeholder="Describe the lyrics or direction for the track"
          />
        </label>

        <label className="field">
          <span>Style</span>
          <input
            value={musicStyle}
            onChange={(event) => onMusicStyleChange(event.target.value)}
            placeholder="modern worship, acoustic"
          />
        </label>

        <label className="field">
          <span>Mood</span>
          <input value={musicMood} onChange={(event) => onMusicMoodChange(event.target.value)} placeholder="hopeful" />
        </label>

        <div className="action-row action-row-single">
          <button type="button" className="primary-button" onClick={onGenerateMusic} disabled={isGeneratingMusic}>
            {isGeneratingMusic ? "Generating..." : "Generate music"}
          </button>
        </div>
      </section>

      <section className="results-column">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">Music</p>
              <h2>{musicResult?.title ?? "No track"}</h2>
            </div>
            {musicResult ? <span className="meta-badge">{musicJob?.provider ?? musicResult.provider}</span> : null}
          </div>

          {musicError ? <p className="error-banner">{musicError}</p> : null}

          {musicResult ? (
            <div className="stack">
              <p className="status-inline">
                Status: <span className={`job-status ${musicStatusLabel}`}>{musicStatusLabel}</span>
              </p>
              <p className="muted-text">Provider: {musicJob?.provider ?? musicResult.provider}</p>

              <section>
                <h3>Prompt</h3>
                <p className="passage-text">{musicResult.prompt}</p>
              </section>

              {musicJob?.error ? <p className="error-banner">{musicJob.error}</p> : null}

              {musicJob?.audio_url ? (
                <section>
                  <h3>Audio</h3>
                  <audio controls src={musicJob.audio_url} />
                </section>
              ) : (
                <p className="empty-state">Audio pending.</p>
              )}
            </div>
          ) : (
            <p className="empty-state">Generate a track.</p>
          )}
        </article>
      </section>
    </>
  );
}
