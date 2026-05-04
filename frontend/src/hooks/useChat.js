import { useState, useCallback, useRef } from "react";
import { sendAgentChat } from "../api/client";

function makeId() {
  return Math.random().toString(36).slice(2);
}

export function useChat() {
  const sessionId = useRef(makeId()).current;
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(
    (userText) => {
      if (isStreaming || !userText.trim()) return;
      setError(null);

      const userMsg = { role: "user", content: userText };
      // assistant message carries steps[] for tool call display
      const assistantMsg = { role: "assistant", content: "", steps: [] };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      sendAgentChat({
        sessionId,
        message: userText,
        onStep: (text) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (!last || last.role !== "assistant") return prev;
            updated[updated.length - 1] = { ...last, steps: [...last.steps, text] };
            return updated;
          });
        },
        onToken: (char) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (!last || last.role !== "assistant") return prev;
            updated[updated.length - 1] = { ...last, content: last.content + char };
            return updated;
          });
        },
        onDone: () => setIsStreaming(false),
        onError: (err) => {
          setIsStreaming(false);
          setError(err.message || "Something went wrong. Please try again.");
        },
      });
    },
    [messages, isStreaming, sessionId]
  );

  return { messages, isStreaming, error, sendMessage };
}
