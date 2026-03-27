import type { PersonaChatMessage } from "../types";

type DiscussionWorkspaceProps = {
  personaModel: string | null;
  personaError: string;
  personaMessages: PersonaChatMessage[];
  enableVoiceReply: boolean;
  isSendingPersona: boolean;
  isRecordingPersona: boolean;
  isTranscribingPersona: boolean;
  isRealtimeVoiceConnecting: boolean;
  isRealtimeVoiceActive: boolean;
  realtimeVoiceStatus: string;
  onPersonaVoiceToggle: () => void;
  onRealtimeVoiceToggle: () => void | Promise<void>;
  onPersonaReset: () => void;
  onEnableVoiceReplyChange: (value: boolean) => void;
};

export function DiscussionWorkspace({
  personaModel,
  personaError,
  personaMessages,
  enableVoiceReply,
  isSendingPersona,
  isRecordingPersona,
  isTranscribingPersona,
  isRealtimeVoiceConnecting,
  isRealtimeVoiceActive,
  realtimeVoiceStatus,
  onPersonaVoiceToggle,
  onRealtimeVoiceToggle,
  onPersonaReset,
  onEnableVoiceReplyChange
}: DiscussionWorkspaceProps) {
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
            {personaModel ? <span className="surface-pill">{personaModel}</span> : null}
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
          <div className={`voice-orbit ${isRealtimeVoiceActive ? "active" : ""}`}>
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
            className="secondary-button round-button"
            onClick={onPersonaVoiceToggle}
            disabled={isSendingPersona || isTranscribingPersona || isRealtimeVoiceActive || isRealtimeVoiceConnecting}
          >
            <span className="material-symbols-outlined">{isRecordingPersona ? "stop" : "mic"}</span>
          </button>
          <button
            type="button"
            className="danger-button call-button"
            onClick={onRealtimeVoiceToggle}
            disabled={isSendingPersona || isRecordingPersona || isTranscribingPersona}
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
          <button type="button" className="secondary-button round-button" onClick={onPersonaReset}>
            <span className="material-symbols-outlined">restart_alt</span>
          </button>
        </div>

        <div className="discussion-voice-settings">
          <label className="toggle-row">
            <span>Voice reply</span>
            <input
              type="checkbox"
              checked={enableVoiceReply}
              onChange={(event) => onEnableVoiceReplyChange(event.target.checked)}
            />
          </label>
          {isRecordingPersona ? <p className="muted-text">Recording...</p> : null}
          {isTranscribingPersona ? (
            <div className="loading-inline">
              <span className="loading-spinner" aria-hidden="true" />
              <p className="muted-text">Transcribing audio...</p>
            </div>
          ) : null}
        </div>
      </article>

      <article className="prototype-card">
        <div className="card-header">
          <div>
            <p className="section-label">Transcript</p>
            <h3>Conversation thread</h3>
          </div>
        </div>
        {personaMessages.length > 0 ? (
          <div className="discussion-transcript-list">
            {personaMessages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`chat-message ${message.role}`}>
                <div className="chat-message-body">
                  <p>{message.content}</p>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="empty-state">Record or start live voice to build the isolated discussion transcript.</p>
        )}
      </article>
    </section>
  );
}
