import { useEffect, useRef, type KeyboardEvent } from "react";

import type { ChatMessage } from "../types";

type TextChatWorkspaceProps = {
  chatError: string;
  chatMessages: ChatMessage[];
  chatInput: string;
  isSendingChat: boolean;
  onChatInputChange: (value: string) => void;
  onChatSend: () => void | Promise<void>;
};

export function TextChatWorkspace({
  chatError,
  chatMessages,
  chatInput,
  isSendingChat,
  onChatInputChange,
  onChatSend
}: TextChatWorkspaceProps) {
  const chatLogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const lastMessage = chatMessages[chatMessages.length - 1];
    if (!lastMessage || lastMessage.role !== "assistant") {
      return;
    }

    const chatLog = chatLogRef.current;
    if (!chatLog) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      chatLog.scrollTop = chatLog.scrollHeight;
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [chatMessages, isSendingChat]);

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) {
      return;
    }

    event.preventDefault();
    void onChatSend();
  }

  return (
    <section className="workspace-chat">
      <div className="workspace-header workspace-header-centered">
        <p className="workspace-kicker">Chat</p>
      </div>

      <div className="chat-prototype-layout">
        <div className="chat-feed-card">
          {chatError ? <p className="error-banner">{chatError}</p> : null}

          <div className="chat-log" ref={chatLogRef}>
            {chatMessages.length > 0 ? (
              <>
                {chatMessages.filter(Boolean).map((message, index) => {
                  const isStreamingAssistantPlaceholder =
                    message.role === "assistant" &&
                    isSendingChat &&
                    index === chatMessages.length - 1 &&
                    message.content.trim().length === 0;

                  return (
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
                      {isStreamingAssistantPlaceholder ? (
                        <div className="loading-inline">
                          <span className="loading-spinner" aria-hidden="true" />
                          <p>Thinking...</p>
                        </div>
                      ) : (
                        <p>{message.content}</p>
                      )}
                    </div>
                  </article>
                )})}
              </>
            ) : (
              <div className="chat-empty-state">
                <p className="section-label">Start a conversation</p>
                <h3>Ask about a passage, a doctrine, or just have a faith-centered chat.</h3>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="chat-composer-shell">
        <label className="sr-only" htmlFor="text-chat-input">
          Message
        </label>
        <div className="chat-composer">
            <textarea
              id="text-chat-input"
              rows={1}
              value={chatInput}
              onChange={(event) => onChatInputChange(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="Ask the Assistant about Scripture..."
            />
          <div className="chat-composer-actions">
            <div className="composer-hints">
              <span className="material-symbols-outlined">edit_note</span>
              <p className="muted-text">Enter to send. Shift+Enter for a new line.</p>
            </div>
            <button type="button" className="primary-button" onClick={onChatSend} disabled={isSendingChat}>
              {isSendingChat ? (
                <>
                  <span className="loading-spinner" aria-hidden="true" />
                  <span>Sending...</span>
                </>
              ) : (
                "Send"
              )}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
