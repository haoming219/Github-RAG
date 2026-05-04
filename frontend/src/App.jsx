import { ChatWindow } from "./components/ChatWindow";
import { InputBar } from "./components/InputBar";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, isStreaming, error, sendMessage } = useChat();

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto border-x">
      <header className="p-4 border-b">
        <h1 className="text-lg font-semibold">GitHub Project Search</h1>
        <p className="text-xs text-gray-500">
          Powered by ReAct Agent — Pinecone + GitHub + GLM
        </p>
      </header>
      <ChatWindow messages={messages} error={error} />
      <InputBar onSend={sendMessage} isStreaming={isStreaming} />
    </div>
  );
}
