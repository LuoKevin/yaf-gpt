import type { KeyboardEvent } from "react";

import type { PersonaChatMessage } from "../types";

type TextChatWorkspaceProps = {
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
    <section className="workspace-chat">
      <div className="workspace-header workspace-header-centered">
        <p className="workspace-kicker">Chat</p>
        <p className="workspace-copy">
          A centered assistant thread built for Scripture questions, reflection, and quick theological exploration.
        </p>
      </div>

      <div className="chat-prototype-layout">
        <div className="chat-feed-card">
          <div className="chat-feed-header">
            <div>
              <p className="section-label">Conversation</p>
              <h2>The assistant is ready</h2>
            </div>
            {personaModel ? <span className="surface-pill">{personaModel}</span> : null}
          </div>

          {personaError ? <p className="error-banner">{personaError}</p> : null}

          <div className="chat-log">
            {personaMessages.length > 0 ? (
              personaMessages.map((message, index) => (
                <article key={`${message.role}-${index}`} className={`chat-message ${message.role}`}>
                  {message.role === "assistant" ? (
                    <div className="assistant-badge-row">
                      <div className="assistant-avatar">
                        <span className="material-symbols-outlined">auto_awesome</span>
                      </div>
                      <span className="assistant-label">YAF-GPT Assistant</span>
                    </div>
                  ) : null}
                  <div className="chat-message-body">
                    <p>{message.content}</p>
                  </div>
                </article>
              ))
            ) : (
              <div className="chat-empty-state">
                <p className="section-label">Start a conversation</p>
                <h3>Ask about a passage, a doctrine, or a question you are still wrestling through.</h3>
                <p className="muted-text">
                  This workspace keeps the exchange quiet and text-first, with the focus staying on the conversation itself.
                </p>
              </div>
            )}
          </div>
        </div>

        <aside className="chat-context-panel">
          <div className="prototype-card">
            <p className="section-label">Mode</p>
            <p className="muted-text">
              Use this view for the cleanest mentoring thread, with fewer controls and less interface noise around the exchange.
            </p>
          </div>
          <button type="button" className="ghost-button wide-button" onClick={onPersonaReset}>
            New chat
          </button>
        </aside>
      </div>

      <div className="chat-composer-shell">
        <label className="sr-only" htmlFor="text-chat-input">
          Message
        </label>
        <div className="chat-composer">
          <textarea
            id="text-chat-input"
            rows={1}
            value={personaInput}
            onChange={(event) => onPersonaInputChange(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Ask the Assistant about Scripture..."
          />
          <div className="chat-composer-actions">
            <div className="composer-hints">
              <span className="material-symbols-outlined">edit_note</span>
              <p className="muted-text">Enter to send. Shift+Enter for a new line.</p>
            </div>
            <button type="button" className="primary-button" onClick={onPersonaSend} disabled={isSendingPersona}>
              {isSendingPersona ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
        <p className="composer-disclaimer">
          YAF-GPT can offer historical and theological help, but it is not a replacement for pastoral care or communal worship.
        </p>
      </div>
    </section>
  );
}
