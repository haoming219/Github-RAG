export function FilterPanel({ filters, setFilters, filterOptions }) {
  if (!filterOptions) return null;

  const toggleTopic = (topic) => {
    setFilters((prev) => ({
      ...prev,
      topics: prev.topics.includes(topic)
        ? prev.topics.filter((t) => t !== topic)
        : [...prev.topics, topic],
    }));
  };

  return (
    <div className="border-b p-3 flex flex-wrap gap-3 items-center text-sm">
      {/* Language */}
      <select
        className="border rounded px-2 py-1"
        value={filters.language}
        onChange={(e) => setFilters((prev) => ({ ...prev, language: e.target.value }))}
      >
        <option value="">All Languages</option>
        {filterOptions.languages.map((l) => (
          <option key={l} value={l}>{l}</option>
        ))}
      </select>

      {/* Min Stars */}
      <div className="flex items-center gap-1">
        <span className="text-gray-500">Min ★</span>
        <input
          type="number"
          className="border rounded px-2 py-1 w-24"
          min={0}
          value={filters.min_stars}
          onChange={(e) =>
            setFilters((prev) => ({ ...prev, min_stars: parseInt(e.target.value) || 0 }))
          }
        />
      </div>

      {/* Topics (show top 10 most common) */}
      <div className="flex flex-wrap gap-1">
        {filterOptions.topics.slice(0, 10).map((topic) => (
          <button
            key={topic}
            onClick={() => toggleTopic(topic)}
            className={`px-2 py-0.5 rounded-full text-xs border ${
              filters.topics.includes(topic)
                ? "bg-blue-600 text-white border-blue-600"
                : "text-gray-600 border-gray-300"
            }`}
          >
            {topic}
          </button>
        ))}
      </div>
    </div>
  );
}
