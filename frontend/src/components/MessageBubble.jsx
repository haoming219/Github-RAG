import ReactMarkdown from "react-markdown";

export function MessageBubble({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-[75%] px-4 py-2 rounded-lg text-sm ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-gray-100 text-gray-800"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <ReactMarkdown className="prose prose-sm max-w-none">
            {message.content || "▌"}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
}
