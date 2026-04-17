import { useEffect, useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { InputBar } from "./components/InputBar";
import { FilterPanel } from "./components/FilterPanel";
import { useChat } from "./hooks/useChat";
import { getFilterOptions } from "./api/client";

export default function App() {
  const { messages, isStreaming, error, filters, setFilters, sendMessage } = useChat();
  const [filterOptions, setFilterOptions] = useState(null);

  useEffect(() => {
    getFilterOptions()
      .then(setFilterOptions)
      .catch(() => {}); // silently ignore — filters are optional
  }, []);

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto border-x">
      <header className="p-4 border-b">
        <h1 className="text-lg font-semibold">GitHub Project Search</h1>
        <p className="text-xs text-gray-500">
          Powered by RAG — Pinecone + BM25 + GLM
        </p>
      </header>
      <FilterPanel
        filters={filters}
        setFilters={setFilters}
        filterOptions={filterOptions}
      />
      <ChatWindow messages={messages} error={error} />
      <InputBar onSend={sendMessage} isStreaming={isStreaming} />
    </div>
  );
}
