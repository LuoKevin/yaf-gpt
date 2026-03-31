import { useState, type CSSProperties } from "react";

import type { ChatMessage } from "../types";

type DiscussionWorkspaceProps = {
  discussionError: string;
  discussionMessages: ChatMessage[];
  isConnectingSession: boolean;
  isSessionActive: boolean;
  isPlayingSessionAudio: boolean;
  voiceVisualLevel: number;
  voiceStatus: string;
  onSessionToggle: () => void | Promise<void>;
};

export function DiscussionWorkspace({
  discussionError,
  discussionMessages,
  isConnectingSession,
  isSessionActive,
  isPlayingSessionAudio,
  voiceVisualLevel,
  voiceStatus,
  onSessionToggle
}: DiscussionWorkspaceProps) {
  const [isTranscriptOpen, setIsTranscriptOpen] = useState(false);
  const voiceOrbitStyle = {
    "--voice-level": String(Math.max(0, Math.min(voiceVisualLevel, 1)))
  } as CSSProperties;
  const isVoiceActive = isSessionActive || isPlayingSessionAudio;

  return (
    <section className="discussion-prototype">
      <div className="workspace-header workspace-header-centered">
        <p className="workspace-kicker">Discussion</p>
        <h1>Voice discussion</h1>
        <p className="workspace-copy">
          Open a live session, speak naturally, and let the mentor respond in real time.
        </p>
      </div>

      <article className="discussion-voice-hero">
        <div className="discussion-voice-meta">
          <div>
            <p className="section-label">Voice session</p>
            <h2>{isVoiceActive ? "Voice session active" : "Mentor chat"}</h2>
            {isSessionActive ? <p className="muted-text">Tap the button again to end the session.</p> : null}
          </div>
          <div className="discussion-status-cluster">
            {voiceStatus ? (
              <span className="surface-pill loading-pill">
                {isConnectingSession ? <span className="loading-spinner" aria-hidden="true" /> : null}
                <span>{voiceStatus}</span>
              </span>
            ) : null}
          </div>
        </div>

        {discussionError ? <p className="error-banner">{discussionError}</p> : null}

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
            onClick={onSessionToggle}
            disabled={isConnectingSession}
          >
            {isConnectingSession ? (
              <span className="loading-spinner" aria-hidden="true" />
            ) : (
              <span className="material-symbols-outlined">{isSessionActive ? "call_end" : "phone_in_talk"}</span>
            )}
            <span>
              {isConnectingSession
                ? "Connecting..."
                : isSessionActive
                  ? "End live session"
                  : "Start live session"}
            </span>
          </button>
        </div>

        <div className="discussion-voice-settings">
          {isConnectingSession ? (
            <div className="loading-inline">
              <span className="loading-spinner" aria-hidden="true" />
              <p className="muted-text">Opening live audio session...</p>
            </div>
          ) : null}
          <p className="composer-disclaimer">
            YAF-GPT is not a source of formal religious advice. For doctrine, pastoral care, or serious spiritual guidance,
            talk with a trusted pastor, elder, or church leader.
          </p>
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
          {discussionMessages.length > 0 ? (
          <div className="discussion-transcript-list">
            {discussionMessages.filter(Boolean).map((message, index) => (
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
            {isConnectingSession ? (
              <div className="discussion-transcript-status">
                <span className="loading-spinner" aria-hidden="true" />
                <span>Connecting live audio...</span>
              </div>
            ) : null}
          </div>
          ) : (
            <p className="empty-state">Start a live session and speak naturally to begin the conversation.</p>
          )}
        </article>
      )}
    </section>
  );
}
