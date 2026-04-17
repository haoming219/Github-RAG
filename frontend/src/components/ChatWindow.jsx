import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";

export function ChatWindow({ messages, error }) {
  const bottomRef = useRef(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4">
      {messages.length === 0 && (
        <p className="text-center text-gray-400 mt-10">
          Ask about GitHub projects...
        </p>
      )}
      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}
      {error && (
        <p className="text-center text-red-500 text-sm mt-2">{error}</p>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
