import ReactMarkdown from "react-markdown";

const STEP_LABELS = [
  ["正在查询知识库", "🔍"],
  ["正在从 GitHub 获取仓库信息", "⚙️"],
  ["正在搜索代码", "⚙️"],
  ["正在读取文件", "⚙️"],
  ["正在搜索互联网", "🌐"],
  ["正在生成完整报告", "📄"],
  ["正在调用工具", "🔧"],
];

function parseStep(text) {
  for (const [prefix, icon] of STEP_LABELS) {
    if (text.startsWith(prefix)) return { label: prefix, icon };
  }
  return { label: text, icon: "🔧" };
}

function AgentSteps({ steps, hasContent }) {
  if (!steps || steps.length === 0 || hasContent) return null;

  return (
    <div className="mb-2 text-xs text-gray-500">
      <div className="space-y-1 border-l-2 border-gray-200 pl-2">
        {steps.map((step, i) => {
          const { label, icon } = parseStep(step);
          return (
            <div key={i} className="flex items-start gap-1.5 leading-snug">
              <span>{icon}</span>
              <span>{label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const hasSteps = !isUser && message.steps && message.steps.length > 0;
  const hasContent = !isUser && !!message.content;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-[75%] px-4 py-2 rounded-lg text-sm ${
          isUser ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-800"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <>
            {hasSteps && (
              <AgentSteps steps={message.steps} hasContent={hasContent} />
            )}
            <ReactMarkdown className="prose prose-sm max-w-none">
              {message.content || (hasSteps ? "" : "▌")}
            </ReactMarkdown>
            {!hasContent && !hasSteps && null}
            {!hasContent && hasSteps && (
              <span className="text-gray-400 animate-pulse">▌</span>
            )}
          </>
        )}
      </div>
    </div>
  );
}
