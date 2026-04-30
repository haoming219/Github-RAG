import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chunker import clean_readme, split_by_headings, split_into_children

# --- clean_readme ---

def test_clean_readme_removes_badges():
    text = "[![Build](https://img.shields.io/badge/build-passing)](https://example.com)\nHello"
    assert "[![" not in clean_readme(text)
    assert "Hello" in clean_readme(text)

def test_clean_readme_removes_html():
    text = "<div>Some content</div>\nHello"
    assert "<div>" not in clean_readme(text)
    assert "Hello" in clean_readme(text)

def test_clean_readme_removes_image_links():
    text = "![logo](https://example.com/logo.png)\nHello"
    assert "![" not in clean_readme(text)
    assert "Hello" in clean_readme(text)

# --- split_by_headings ---

def test_split_by_headings_basic():
    text = "Intro paragraph here with enough content to pass the threshold.\n\n## Installation\nRun pip install.\n\n## Usage\nImport and use it in your project."
    chunks = split_by_headings(text, "owner/repo")
    assert len(chunks) >= 2
    assert chunks[0]["section_title"] == "__intro__"
    assert chunks[0]["section_index"] == 0
    assert chunks[0]["parent_id"] == "owner/repo__0"
    assert any(c["section_title"] == "## Installation" for c in chunks)

def test_split_by_headings_short_chunk_merged():
    text = "This is a long enough intro paragraph to stand on its own.\n\n## Section A\nThis section has good content here.\n\n## Tiny\nHi"
    chunks = split_by_headings(text, "owner/repo")
    contents = [c["content"] for c in chunks]
    assert not any(c == "Hi" for c in contents)

def test_split_by_headings_no_readme_fallback():
    chunks = split_by_headings("", "owner/repo", description="A cool library")
    assert len(chunks) == 1
    assert chunks[0]["section_title"] == "__description__"
    assert chunks[0]["content"] == "A cool library"
    assert chunks[0]["parent_id"] == "owner/repo__0"

def test_split_by_headings_parent_id_format():
    text = "Intro.\n\n## A\nContent A with enough text to pass threshold.\n\n## B\nContent B with enough text to pass threshold."
    chunks = split_by_headings(text, "facebook/react")
    for i, c in enumerate(chunks):
        assert c["parent_id"] == f"facebook/react__{i}"
        assert c["section_index"] == i

# --- split_into_children ---

def test_split_into_children_short_content_single_child():
    parent = {
        "parent_id": "owner/repo__0",
        "content": "Short content that is well under five hundred twelve tokens."
    }
    children = split_into_children(parent)
    assert len(children) == 1
    assert children[0]["parent_id"] == "owner/repo__0"
    assert children[0]["child_index"] == 0
    assert children[0]["vector_id"] == "owner/repo__0__child0"
    assert "content" in children[0]

def test_split_into_children_long_content_multiple_children():
    long_text = "This is a sentence. " * 100
    parent = {"parent_id": "owner/repo__1", "content": long_text * 2}
    children = split_into_children(parent)
    assert len(children) >= 2
    for i, c in enumerate(children):
        assert c["child_index"] == i
        assert c["vector_id"] == f"owner/repo__1__child{i}"

def test_split_into_children_child_index_sequential():
    long_text = "Word sentence here. " * 150
    parent = {"parent_id": "repo/name__2", "content": long_text}
    children = split_into_children(parent)
    indices = [c["child_index"] for c in children]
    assert indices == list(range(len(children)))
