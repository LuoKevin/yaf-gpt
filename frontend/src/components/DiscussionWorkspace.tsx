import type { PersonaChatMessage } from "../types";

type DiscussionWorkspaceProps = {
  personaModel: string | null;
  personaError: string;
  personaMessages: PersonaChatMessage[];
  personaInput: string;
  enableVoiceReply: boolean;
  isSendingPersona: boolean;
  isRecordingPersona: boolean;
  isTranscribingPersona: boolean;
  isRealtimeVoiceConnecting: boolean;
  isRealtimeVoiceActive: boolean;
  realtimeVoiceStatus: string;
  onPersonaInputChange: (value: string) => void;
  onPersonaSend: () => void | Promise<void>;
  onPersonaVoiceToggle: () => void;
  onRealtimeVoiceToggle: () => void | Promise<void>;
  onPersonaReset: () => void;
  onEnableVoiceReplyChange: (value: boolean) => void;
};

export function DiscussionWorkspace({
  personaModel,
  personaError,
  personaMessages,
  personaInput,
  enableVoiceReply,
  isSendingPersona,
  isRecordingPersona,
  isTranscribingPersona,
  isRealtimeVoiceConnecting,
  isRealtimeVoiceActive,
  realtimeVoiceStatus,
  onPersonaInputChange,
  onPersonaSend,
  onPersonaVoiceToggle,
  onRealtimeVoiceToggle,
  onPersonaReset,
  onEnableVoiceReplyChange
}: DiscussionWorkspaceProps) {
  const isBusy =
    isSendingPersona || isRecordingPersona || isTranscribingPersona || isRealtimeVoiceConnecting;

  return (
    <section className="results-column">
      <article className="panel">
        <div className="panel-heading">
          <div>
            <p className="panel-kicker">Discussion</p>
            <h2>Mentor chat</h2>
          </div>
          {personaModel ? <span className="meta-badge">{personaModel}</span> : null}
        </div>

        {personaError ? <p className="error-banner">{personaError}</p> : null}

        <div className="stack">
          {personaMessages.length > 0 ? (
            <div className="chat-thread">
              {personaMessages.map((message, index) => (
                <article key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
                  <p className="summary-label">{message.role === "user" ? "You" : "Mentor"}</p>
                  <p>{message.content}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="empty-state">Start the chat.</p>
          )}

          <label className="field">
            <span>Message</span>
            <textarea
              rows={3}
              value={personaInput}
              onChange={(event) => onPersonaInputChange(event.target.value)}
              placeholder="Ask a question"
            />
          </label>

          <div className="mini-action-row">
            <button
              type="button"
              className="secondary-button"
              onClick={onPersonaSend}
              disabled={isBusy || isRealtimeVoiceActive}
            >
              {isSendingPersona ? "Sending..." : "Send"}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={onPersonaVoiceToggle}
              disabled={isSendingPersona || isTranscribingPersona || isRealtimeVoiceActive || isRealtimeVoiceConnecting}
            >
              {isRecordingPersona ? "Stop recording" : "Voice input"}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={onRealtimeVoiceToggle}
              disabled={isSendingPersona || isRecordingPersona || isTranscribingPersona}
            >
              {isRealtimeVoiceConnecting
                ? "Connecting..."
                : isRealtimeVoiceActive
                  ? "Stop live voice"
                  : "Live voice"}
            </button>
            <button type="button" className="secondary-button" onClick={onPersonaReset}>
              Reset
            </button>
          </div>

          <label className="field field-inline">
            <span>Voice reply</span>
            <input
              type="checkbox"
              checked={enableVoiceReply}
              onChange={(event) => onEnableVoiceReplyChange(event.target.checked)}
            />
          </label>

          {isRecordingPersona ? <p className="muted-text">Recording...</p> : null}
          {isTranscribingPersona ? <p className="muted-text">Transcribing audio...</p> : null}
          {realtimeVoiceStatus ? <p className="muted-text">{realtimeVoiceStatus}</p> : null}
        </div>
      </article>
    </section>
  );
}
