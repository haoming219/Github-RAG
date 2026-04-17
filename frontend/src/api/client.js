const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function getFilterOptions() {
  const res = await fetch(`${API_URL}/api/filters/options`);
  if (!res.ok) throw new Error("Failed to fetch filter options");
  return res.json();
}

/**
 * Send chat request and call onChunk(text) for each streaming chunk.
 * Calls onDone() when stream ends. Calls onError(err) on failure.
 */
export async function sendChat({ messages, filters, onChunk, onDone, onError }) {
  try {
    const res = await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, filters }),
    });

    if (!res.ok) {
      onError(new Error(`Server error: ${res.status}`));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") {
          onDone();
          return;
        }
        try {
          const parsed = JSON.parse(data);
          if (parsed.text) onChunk(parsed.text);
        } catch {
          // ignore malformed lines
        }
      }
    }
    onDone();
  } catch (err) {
    onError(err);
  }
}
