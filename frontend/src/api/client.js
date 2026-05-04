const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function getFilterOptions() {
  const res = await fetch(`${API_URL}/api/filters/options`);
  if (!res.ok) throw new Error("Failed to fetch filter options");
  return res.json();
}

/**
 * Send a message to the ReAct agent endpoint.
 * onStep(text)  — called for each agent_step event
 * onToken(text) — called for each token event
 * onDone()      — called when stream ends
 * onError(err)  — called on failure
 */
export async function sendAgentChat({ sessionId, message, onStep, onToken, onDone, onError }) {
  try {
    const res = await fetch(`${API_URL}/agent/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
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
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();
        if (data === "[DONE]") {
          onDone();
          return;
        }
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === "agent_step") onStep(parsed.content);
          else if (parsed.type === "token") onToken(parsed.content);
          else if (parsed.type === "error") onError(new Error(parsed.content));
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
