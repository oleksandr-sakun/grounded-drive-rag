"""
Chunking: split Markdown into section-level chunks.

We chunk on headings rather than on a fixed token window. Two reasons:

1. Citations. A heading gives us a stable anchor to point the user at
   ("Expense Policy > Submitting a claim"), which a sliding window does not.
2. Coherence. Policy documents are written in sections. A section is the unit
   a human would quote, so it is the unit we should retrieve.

Sections longer than MAX_CHARS are split on paragraph boundaries, and each
piece keeps the parent heading path so the citation stays correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

MAX_CHARS = 1800
MIN_CHARS = 80  # below this a chunk is noise (a stray heading, an empty section)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: str  # e.g. "Expense Policy > Submitting a claim"
    text: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return self.section_path


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split an oversized section on blank lines, never mid-paragraph."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    size = 0

    for para in text.split("\n\n"):
        para_len = len(para) + 2
        if size + para_len > limit and current:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += para_len

    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(md: str, doc_id: str, doc_title: str) -> list[Chunk]:
    """Turn one Markdown document into a list of Chunks."""
    lines = md.splitlines()

    # heading_stack[i] = current heading text at level i+1
    heading_stack: list[str] = []
    body: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        body.clear()
        if len(text) < MIN_CHARS:
            return

        path = " > ".join(heading_stack) if heading_stack else doc_title
        for piece in _split_long(text):
            if len(piece.strip()) < MIN_CHARS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks):03d}",
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section_path=path,
                    # Prepend the path so the retriever can match on section
                    # words too ("submitting a claim" is a real query).
                    text=f"{path}\n\n{piece.strip()}",
                )
            )

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title)
        else:
            body.append(line)

    flush()
    return chunks
