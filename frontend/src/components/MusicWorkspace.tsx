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
    <section className="music-prototype">
      <div className="workspace-header workspace-header-centered">
        <p className="workspace-kicker">Music</p>
        <h1>Harmony in study</h1>
        <p className="workspace-copy">
          Turn passage direction into a more coherent sonic brief, then follow the draft through to generated audio.
        </p>
      </div>

      <div className="music-generation-card">
        <div className="card-stack">
          <label className="field">
            <span>Reference</span>
            <input value={reference} onChange={(event) => onReferenceChange(event.target.value)} placeholder="Luke 21:5-28" />
          </label>
          <div className="music-grid">
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
          </div>
          <label className="field">
            <span>Sonic prompt</span>
            <textarea
              rows={3}
              value={musicPrompt}
              onChange={(event) => onMusicPromptChange(event.target.value)}
              placeholder="Describe the atmosphere, lyrics, or musical direction..."
            />
          </label>
          <div className="music-grid">
            <label className="field">
              <span>Style</span>
              <input value={musicStyle} onChange={(event) => onMusicStyleChange(event.target.value)} placeholder="modern worship, acoustic" />
            </label>
            <label className="field">
              <span>Mood</span>
              <input value={musicMood} onChange={(event) => onMusicMoodChange(event.target.value)} placeholder="hopeful" />
            </label>
          </div>
          <button type="button" className="primary-button wide-button" onClick={onGenerateMusic} disabled={isGeneratingMusic}>
            {isGeneratingMusic ? "Generating..." : "Generate composition"}
          </button>
        </div>
      </div>

      <section className="music-results">
        <div className="card-header">
          <div>
            <p className="section-label">Recent track</p>
            <h2>{musicResult?.title ?? "No track generated yet"}</h2>
          </div>
          {musicResult ? <span className="surface-pill">{musicJob?.provider ?? musicResult.provider}</span> : null}
        </div>

        {musicError ? <p className="error-banner">{musicError}</p> : null}

        {musicResult ? (
          <div className="card-stack">
            <div className="track-row">
              <div className="track-icon">
                <span className="material-symbols-outlined">library_music</span>
              </div>
              <div className="track-meta">
                <h3>{musicResult.title}</h3>
                <p>
                  {musicStyle} • {musicMood} • {translation}
                </p>
              </div>
              <span className={`job-status ${musicStatusLabel}`}>{musicStatusLabel}</span>
            </div>

            <section>
              <h4>Prompt</h4>
              <p className="study-passage-text">{musicResult.prompt}</p>
            </section>

            {musicJob?.error ? <p className="error-banner">{musicJob.error}</p> : null}

            {musicJob?.audio_url ? (
              <audio controls src={musicJob.audio_url} />
            ) : (
              <p className="empty-state">Audio is still pending from the provider.</p>
            )}
          </div>
        ) : (
          <p className="empty-state">Generate a track to fill this sanctuary with something you can actually play.</p>
        )}
      </section>
    </section>
  );
}
