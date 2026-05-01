import sys, pathlib, pickle, json, types
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from retriever import _apply_filter, _rrf, _max_pool_vector_results

# --- _apply_filter ---

def test_apply_filter_no_constraints():
    meta = {"language": "Python", "stars": 5000, "topics": ["web"]}
    assert _apply_filter(meta, language="", min_stars=0, topics=[]) is True

def test_apply_filter_language_match():
    meta = {"language": "Python", "stars": 100, "topics": []}
    assert _apply_filter(meta, language="Python", min_stars=0, topics=[]) is True

def test_apply_filter_language_mismatch():
    meta = {"language": "Go", "stars": 100, "topics": []}
    assert _apply_filter(meta, language="Python", min_stars=0, topics=[]) is False

def test_apply_filter_min_stars():
    meta = {"language": "", "stars": 500, "topics": []}
    assert _apply_filter(meta, language="", min_stars=1000, topics=[]) is False
    assert _apply_filter(meta, language="", min_stars=500, topics=[]) is True

def test_apply_filter_topics_or_logic():
    meta = {"language": "", "stars": 0, "topics": ["web", "api"]}
    assert _apply_filter(meta, language="", min_stars=0, topics=["api", "ml"]) is True
    assert _apply_filter(meta, language="", min_stars=0, topics=["ml"]) is False

# --- _rrf ---

def test_rrf_single_list():
    # single list: rank order preserved
    result = _rrf([{"a": 0, "b": 1, "c": 2}], k=60)
    assert result[0] == "a"
    assert result[1] == "b"

def test_rrf_two_lists_boost_overlap():
    # "b" appears in both lists → higher score than "a" (only in list 1)
    list_a = {"a": 0, "b": 1}
    list_b = {"b": 0, "c": 1}
    result = _rrf([list_a, list_b], k=60)
    assert result[0] == "b"

def test_rrf_empty_list_ignored():
    result = _rrf([{"a": 0, "b": 1}, {}], k=60)
    assert "a" in result

def test_rrf_returns_top_k():
    ranked = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
    result = _rrf([ranked], k=60, top_k=3)
    assert len(result) == 3

# --- _max_pool_vector_results ---

def _make_match(parent_id):
    m = types.SimpleNamespace()
    m.metadata = {"parent_id": parent_id}
    return m

def test_max_pool_deduplicates():
    matches = [
        _make_match("repo/a__0"),
        _make_match("repo/a__0"),  # duplicate — same parent
        _make_match("repo/b__0"),
    ]
    result = _max_pool_vector_results(matches, limit=20)
    assert len(result) == 2
    assert "repo/a__0" in result
    assert result["repo/a__0"] == 0  # best rank is 0 (first occurrence)

def test_max_pool_respects_limit():
    matches = [_make_match(f"repo/x__{i}") for i in range(30)]
    result = _max_pool_vector_results(matches, limit=10)
    assert len(result) == 10

def test_max_pool_records_first_seen_rank():
    # First occurrence at rank 0 should win over second at rank 2
    matches = [
        _make_match("repo/a__0"),  # rank 0
        _make_match("repo/b__0"),  # rank 1
        _make_match("repo/a__0"),  # rank 2 — duplicate, should be ignored
    ]
    result = _max_pool_vector_results(matches, limit=20)
    assert result["repo/a__0"] == 0
