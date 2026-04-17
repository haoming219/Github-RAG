import { useState } from "react";

export function InputBar({ onSend, isStreaming }) {
  const [text, setText] = useState("");

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const submit = () => {
    if (!text.trim() || isStreaming) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="border-t p-3 flex gap-2">
      <textarea
        className="flex-1 border rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
        rows={2}
        placeholder="Ask about GitHub projects... (Enter to send, Shift+Enter for newline)"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isStreaming}
      />
      <button
        className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
        onClick={submit}
        disabled={isStreaming || !text.trim()}
      >
        Send
      </button>
    </div>
  );
}
