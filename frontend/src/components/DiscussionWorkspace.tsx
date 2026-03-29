import { useState, type CSSProperties } from "react";

import type { PersonaChatMessage } from "../types";

type DiscussionWorkspaceProps = {
  personaError: string;
  personaMessages: PersonaChatMessage[];
  isProcessingPersona: boolean;
  isRecordingPersona: boolean;
  isPlayingPersonaAudio: boolean;
  voiceVisualLevel: number;
  voiceStatus: string;
  onRecordToggle: () => void | Promise<void>;
};

export function DiscussionWorkspace({
  personaError,
  personaMessages,
  isProcessingPersona,
  isRecordingPersona,
  isPlayingPersonaAudio,
  voiceVisualLevel,
  voiceStatus,
  onRecordToggle
}: DiscussionWorkspaceProps) {
  const [isTranscriptOpen, setIsTranscriptOpen] = useState(false);
  const voiceOrbitStyle = {
    "--voice-level": String(Math.max(0, Math.min(voiceVisualLevel, 1)))
  } as CSSProperties;
  const isVoiceActive = isRecordingPersona || isPlayingPersonaAudio;

  return (
    <section className="discussion-prototype">
      <div className="workspace-header workspace-header-centered">
        <p className="workspace-kicker">Discussion</p>
        <h1>Voice discussion</h1>
        <p className="workspace-copy">
          Record a prompt, tap again to send it, then hear the spoken mentor reply back.
        </p>
      </div>

      <article className="discussion-voice-hero">
        <div className="discussion-voice-meta">
          <div>
            <p className="section-label">Voice session</p>
            <h2>{isVoiceActive ? "Voice session active" : "Mentor chat"}</h2>
            <p className="muted-text">
              This workspace is isolated from text chat and handles spoken input plus spoken replies only.
              {isRecordingPersona ? " Tap the button again to stop and send." : ""}
            </p>
          </div>
          <div className="discussion-status-cluster">
            {voiceStatus ? (
              <span className="surface-pill loading-pill">
                {isProcessingPersona ? <span className="loading-spinner" aria-hidden="true" /> : null}
                <span>{voiceStatus}</span>
              </span>
            ) : null}
          </div>
        </div>

        {personaError ? <p className="error-banner">{personaError}</p> : null}

        <div className="voice-visualizer-shell">
          <div className={`voice-orbit ${isVoiceActive ? "active" : ""}`} style={voiceOrbitStyle}>
            <div className="voice-orbit-inner">
              <div className="voice-bars">
                <span />
                <span />
                <span />
                <span />
                <span />
                <span />
              </div>
            </div>
          </div>
        </div>

        <div className="discussion-controls">
          <button
            type="button"
            className="danger-button call-button"
            onClick={onRecordToggle}
            disabled={isProcessingPersona}
          >
            {isProcessingPersona ? (
              <span className="loading-spinner" aria-hidden="true" />
            ) : (
              <span className="material-symbols-outlined">{isRecordingPersona ? "stop_circle" : "mic"}</span>
            )}
            <span>
              {isProcessingPersona
                ? "Processing..."
                : isRecordingPersona
                  ? "Stop and send"
                  : "Start recording"}
            </span>
          </button>
        </div>

        <div className="discussion-voice-settings">
          {isProcessingPersona ? (
            <div className="loading-inline">
              <span className="loading-spinner" aria-hidden="true" />
              <p className="muted-text">Transcribing and rendering reply...</p>
            </div>
          ) : null}
        </div>
      </article>

      {!isTranscriptOpen ? (
        <div className="discussion-transcript-collapsed">
          <button
            type="button"
            className="ghost-button transcript-icon-button"
            onClick={() => setIsTranscriptOpen(true)}
            data-tooltip="Show transcript"
            aria-label="Show transcript"
          >
            <span className="material-symbols-outlined" aria-hidden="true">
              notes
            </span>
          </button>
        </div>
      ) : (
        <article className="prototype-card discussion-transcript-card">
          <div className="card-header">
            <div>
              <p className="section-label">Transcript</p>
              <h3>Conversation thread</h3>
            </div>
            <button
              type="button"
              className="ghost-button transcript-toggle-button"
              onClick={() => setIsTranscriptOpen(false)}
            >
              <span className="material-symbols-outlined" aria-hidden="true">
                expand_less
              </span>
              <span>Hide transcript</span>
            </button>
          </div>
          {personaMessages.length > 0 ? (
          <div className="discussion-transcript-list">
            {personaMessages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`discussion-transcript-item ${message.role}`}>
                <div className="discussion-transcript-meta">
                  <span className="discussion-transcript-role">
                    {message.role === "user" ? "You" : "YAF-GPT"}
                  </span>
                </div>
                <div className="discussion-transcript-body">
                  <p>{message.content}</p>
                </div>
              </article>
            ))}
            {isProcessingPersona ? (
              <div className="discussion-transcript-status">
                <span className="loading-spinner" aria-hidden="true" />
                <span>Waiting for spoken reply...</span>
              </div>
            ) : null}
          </div>
          ) : (
            <p className="empty-state">Start recording, then tap again to send your spoken prompt.</p>
          )}
        </article>
      )}
    </section>
  );
}
