import { useState } from "react";
import ReactMarkdown from "react-markdown";

const STEP_ICONS = {
  "正在查询知识库": "🔍",
  "正在从 GitHub 获取仓库信息": "⚙️",
  "正在搜索代码": "⚙️",
  "正在读取文件": "⚙️",
  "正在搜索互联网": "🌐",
  "正在生成完整报告": "📄",
  "正在调用工具": "🔧",
};

function stepIcon(text) {
  for (const [prefix, icon] of Object.entries(STEP_ICONS)) {
    if (text.startsWith(prefix)) return icon;
  }
  return "🔧";
}

function AgentSteps({ steps, hasContent }) {
  const [collapsed, setCollapsed] = useState(false);

  if (!steps || steps.length === 0) return null;

  return (
    <div className="mb-2 text-xs text-gray-500">
      {hasContent && (
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="flex items-center gap-1 mb-1 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <span>{collapsed ? "▶" : "▼"}</span>
          <span>{collapsed ? "查看过程" : "收起过程"}</span>
        </button>
      )}
      {!collapsed && (
        <div className="space-y-1 border-l-2 border-gray-200 pl-2">
          {steps.map((step, i) => (
            <div key={i} className="flex items-start gap-1.5 leading-snug">
              <span>{stepIcon(step)}</span>
              <span>{step}</span>
            </div>
          ))}
        </div>
      )}
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
