import { useState, type CSSProperties } from "react";

import type { PersonaChatMessage } from "../types";

type DiscussionWorkspaceProps = {
  personaError: string;
  personaMessages: PersonaChatMessage[];
  isSendingPersona: boolean;
  isTranscribingPersona: boolean;
  isRealtimeVoiceConnecting: boolean;
  isRealtimeVoiceActive: boolean;
  realtimeVoiceLevel: number;
  realtimeVoiceStatus: string;
  onRealtimeVoiceToggle: () => void | Promise<void>;
};

export function DiscussionWorkspace({
  personaError,
  personaMessages,
  isSendingPersona,
  isTranscribingPersona,
  isRealtimeVoiceConnecting,
  isRealtimeVoiceActive,
  realtimeVoiceLevel,
  realtimeVoiceStatus,
  onRealtimeVoiceToggle
}: DiscussionWorkspaceProps) {
  const [isTranscriptOpen, setIsTranscriptOpen] = useState(false);
  const voiceOrbitStyle = {
    "--voice-level": String(Math.max(0, Math.min(realtimeVoiceLevel, 1)))
  } as CSSProperties;

  return (
    <section className="discussion-prototype">
      <div className="workspace-header workspace-header-centered">
        <p className="workspace-kicker">Discussion</p>
        <h1>Live discussion</h1>
        <p className="workspace-copy">
          Keep this workspace voice-first, with isolated recording, live voice, and transcript state.
        </p>
      </div>

      <article className="discussion-voice-hero">
        <div className="discussion-voice-meta">
          <div>
            <p className="section-label">Voice session</p>
            <h2>{isRealtimeVoiceActive ? "Voice session active" : "Mentor chat"}</h2>
            <p className="muted-text">
              Record a prompt or start a live session. This workspace no longer shares chat state with the text workspace.
            </p>
          </div>
          <div className="discussion-status-cluster">
            {realtimeVoiceStatus ? (
              <span className="surface-pill loading-pill">
                {isRealtimeVoiceConnecting ? <span className="loading-spinner" aria-hidden="true" /> : null}
                <span>{realtimeVoiceStatus}</span>
              </span>
            ) : null}
          </div>
        </div>

        {personaError ? <p className="error-banner">{personaError}</p> : null}

        <div className="voice-visualizer-shell">
          <div className={`voice-orbit ${isRealtimeVoiceActive ? "active" : ""}`} style={voiceOrbitStyle}>
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
            onClick={onRealtimeVoiceToggle}
            disabled={isSendingPersona || isTranscribingPersona}
          >
            {isRealtimeVoiceConnecting ? (
              <span className="loading-spinner" aria-hidden="true" />
            ) : (
              <span className="material-symbols-outlined">{isRealtimeVoiceActive ? "call_end" : "wifi_calling_3"}</span>
            )}
            <span>
              {isRealtimeVoiceConnecting
                ? "Connecting..."
                : isRealtimeVoiceActive
                  ? "End discussion"
                  : "Start live voice"}
            </span>
          </button>
        </div>

        <div className="discussion-voice-settings">
          {isTranscribingPersona ? (
            <div className="loading-inline">
              <span className="loading-spinner" aria-hidden="true" />
              <p className="muted-text">Transcribing audio...</p>
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
            {isTranscribingPersona || isSendingPersona || isRealtimeVoiceConnecting ? (
              <div className="discussion-transcript-status">
                <span className="loading-spinner" aria-hidden="true" />
                <span>
                  {isTranscribingPersona
                    ? "Transcribing..."
                    : isRealtimeVoiceConnecting
                      ? "Connecting live voice..."
                      : "Waiting for response..."}
                </span>
              </div>
            ) : null}
          </div>
          ) : (
            <p className="empty-state">Record or start live voice to build the isolated discussion transcript.</p>
          )}
        </article>
      )}
    </section>
  );
}
