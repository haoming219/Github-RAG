import { useState, useCallback } from "react";
import { sendChat } from "../api/client";

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    language: "",
    min_stars: 0,
    topics: [],
  });

  const sendMessage = useCallback(
    (userText) => {
      if (isStreaming || !userText.trim()) return;
      setError(null);

      const userMsg = { role: "user", content: userText };
      const assistantMsg = { role: "assistant", content: "" };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      sendChat({
        messages: [...messages, userMsg],
        filters,
        onChunk: (text) => {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: updated[updated.length - 1].content + text,
            };
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
    [messages, filters, isStreaming]
  );

  return { messages, isStreaming, error, filters, setFilters, sendMessage };
}
