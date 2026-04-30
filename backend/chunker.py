import re
import tiktoken
from llama_index.core.node_parser import SentenceSplitter

_tokenizer = tiktoken.get_encoding("cl100k_base")
_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50, tokenizer=_tokenizer.encode)

MIN_TOKENS = 3


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def clean_readme(text: str) -> str:
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_by_headings(text: str, full_name: str, description: str = "") -> list[dict]:
    """
    First-level split: divide cleaned README text at Markdown heading boundaries.
    Returns parent chunk dicts with parent_id, section_index, section_title, content.
    If text is empty, falls back to a single description chunk.
    """
    if not text.strip():
        return [{
            "parent_id": f"{full_name}__0",
            "full_name": full_name,
            "section_index": 0,
            "section_title": "__description__",
            "content": description or full_name,
        }]

    lines = text.split('\n')
    raw_chunks = []
    current_title = "__intro__"
    current_lines = []

    for line in lines:
        if re.match(r'^#{1,6}\s', line):
            content = '\n'.join(current_lines).strip()
            if content:
                raw_chunks.append({"section_title": current_title, "content": content})
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = '\n'.join(current_lines).strip()
    if content:
        raw_chunks.append({"section_title": current_title, "content": content})

    # Merge chunks below MIN_TOKENS threshold into previous chunk
    merged = []
    for chunk in raw_chunks:
        if merged and _count_tokens(chunk["content"]) < MIN_TOKENS:
            merged[-1]["content"] += "\n" + chunk["content"]
        else:
            merged.append(chunk)

    result = []
    for i, chunk in enumerate(merged):
        result.append({
            "parent_id": f"{full_name}__{i}",
            "full_name": full_name,
            "section_index": i,
            "section_title": chunk["section_title"],
            "content": chunk["content"],
        })
    return result


def split_into_children(parent: dict) -> list[dict]:
    """
    Second-level split: divide a parent chunk's content using SentenceSplitter.
    Returns child chunk dicts with parent_id, child_index, vector_id, content.
    """
    from llama_index.core.schema import Document
    doc = Document(text=parent["content"])
    nodes = _splitter.get_nodes_from_documents([doc])
    children = []
    for i, node in enumerate(nodes):
        children.append({
            "parent_id": parent["parent_id"],
            "child_index": i,
            "vector_id": f"{parent['parent_id']}__child{i}",
            "content": node.get_content(),
        })
    return children
