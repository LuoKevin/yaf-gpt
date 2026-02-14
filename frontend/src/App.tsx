import { useMemo, useState } from "react";

const SUGGESTIONS = [
  "What does the Sermon on the Mount teach?",
  "Explain the Trinity in simple terms.",
  "What are the Gospels about?"
];

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "Hi! Ask me a question about Christianity." }
  ]);
  const [input, setInput] = useState("");

  const canSend = input.trim().length > 0;

  const lastUserMessage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === "user") return messages[i].content;
    }
    return "";
  }, [messages]);

  const handleSend = () => {
    if (!canSend) return;
    const nextMessage: ChatMessage = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, nextMessage]);
    setInput("");

    // Placeholder assistant response until API is wired.
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Thanks! I can answer once the backend is connected."
        }
      ]);
    }, 250);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div>
          <p className="kicker">yaf-gpt</p>
          <h1>Christianity Assistant</h1>
        </div>
        <div className="status">
          <span className="dot" />
          <span>Offline</span>
        </div>
      </header>

      <main className="chat">
        {messages.map((message, index) => (
          <article key={`${message.role}-${index}`} className={`bubble ${message.role}`}>
            <p>{message.content}</p>
          </article>
        ))}
      </main>

      <section className="composer">
        <div className="suggestions">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => setInput(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
        <div className="input-row">
          <textarea
            placeholder="Ask about the Bible, theology, or history..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
          />
          <button type="button" onClick={handleSend} disabled={!canSend}>
            Send
          </button>
        </div>
        <p className="hint">
          Press Enter to send, Shift+Enter for a new line.
        </p>
        {lastUserMessage && (
          <p className="last">Last question: {lastUserMessage}</p>
        )}
      </section>
    </div>
  );
}
