import type { KeyboardEvent } from "react";

import type { PersonaChatMessage, TranslationCode } from "../types";

type TextChatWorkspaceProps = {
  reference: string;
  translation: TranslationCode;
  personaModel: string | null;
  personaError: string;
  personaMessages: PersonaChatMessage[];
  personaInput: string;
  isSendingPersona: boolean;
  onPersonaInputChange: (value: string) => void;
  onPersonaSend: () => void | Promise<void>;
  onPersonaReset: () => void;
};

export function TextChatWorkspace({
  reference,
  translation,
  personaModel,
  personaError,
  personaMessages,
  personaInput,
  isSendingPersona,
  onPersonaInputChange,
  onPersonaSend,
  onPersonaReset
}: TextChatWorkspaceProps) {
  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }

    event.preventDefault();
    void onPersonaSend();
  }

  return (
    <section className="chat-shell">
      <div className="chat-window panel">
        <div className="chat-header">
          <div>
            <p className="panel-kicker">Chat</p>
            <h2>Text conversation</h2>
          </div>
          {personaModel ? <span className="meta-badge">{personaModel}</span> : null}
        </div>

        <div className="chat-toolbar">
          <span className="chat-context">{reference.trim() || "No passage context set"}</span>
          <span className="chat-context">{translation}</span>
          <button type="button" className="ghost-button" onClick={onPersonaReset}>
            New chat
          </button>
        </div>

        {personaError ? <p className="error-banner">{personaError}</p> : null}

        <div className="chat-log">
          {personaMessages.length > 0 ? (
            personaMessages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`chat-message ${message.role}`}>
                <div className="chat-message-body">
                  <p>{message.content}</p>
                </div>
              </article>
            ))
          ) : (
            <div className="chat-empty-state">
              <p className="summary-label">Start a conversation</p>
              <h3>Ask a passage question or explore an idea in plain text.</h3>
              <p className="muted-text">
                This mode keeps the mentor chat focused on a simple thread, without live voice or recording controls.
              </p>
            </div>
          )}
        </div>

        <div className="chat-composer">
          <label className="sr-only" htmlFor="text-chat-input">
            Message
          </label>
          <textarea
            id="text-chat-input"
            rows={1}
            value={personaInput}
            onChange={(event) => onPersonaInputChange(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Message the mentor"
          />
          <div className="chat-composer-actions">
            <p className="muted-text">Enter to send. Shift+Enter for a new line.</p>
            <button
              type="button"
              className="primary-button"
              onClick={onPersonaSend}
              disabled={isSendingPersona}
            >
              {isSendingPersona ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
