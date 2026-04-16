import re

def clean_readme(text: str) -> str:
    """Remove badges, HTML tags, and image links from README markdown."""
    # Remove badge links: [![...](...)(...)]
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    # Remove plain image links: ![...](...)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_by_headings(text: str, parent_id: str) -> list[dict]:
    """
    Split cleaned markdown text into chunks at each heading boundary.
    Returns list of dicts with: parent_id, section_title, content, chunk_index.
    Chunks with fewer than 50 chars of content are merged into the previous chunk.
    """
    lines = text.split('\n')
    chunks = []
    current_title = "__intro__"
    current_lines = []

    for line in lines:
        if re.match(r'^#{1,6}\s', line):
            # Save current chunk before starting new one
            content = '\n'.join(current_lines).strip()
            if content:
                chunks.append({
                    "parent_id": parent_id,
                    "section_title": current_title,
                    "content": content,
                    "chunk_index": len(chunks)
                })
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save final chunk
    content = '\n'.join(current_lines).strip()
    if content:
        chunks.append({
            "parent_id": parent_id,
            "section_title": current_title,
            "content": content,
            "chunk_index": len(chunks)
        })

    # Merge short chunks (< 50 chars) into previous
    merged = []
    for chunk in chunks:
        if merged and len(chunk["content"]) < 50:
            merged[-1]["content"] += "\n" + chunk["content"]
        else:
            merged.append(chunk)

    # Re-index after merge
    for i, chunk in enumerate(merged):
        chunk["chunk_index"] = i

    return merged
