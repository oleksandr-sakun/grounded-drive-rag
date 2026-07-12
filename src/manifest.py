"""
The manifest.

This is the component that exists because of a real production failure: a
managed connector reported a successful sync and silently indexed a fraction
of the source documents. Nothing errored. Nothing warned. The bot simply
did not know things, and there was no way to tell from the outside.

A manifest makes that failure impossible to hide. Every source file gets a
row: its id, its revision, a hash of its content, and how many chunks it
produced. Reconciliation is then a set comparison, and a missing document is
a loud, specific error rather than a vague sense that the bot "seems worse".

It is also the thing you hand a client who asks "is everything indexed?".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class DocEntry:
    doc_id: str  # Drive file id (or local filename in demo mode)
    title: str
    source_mime: str  # what it was in Drive
    revision: str  # Drive revision / modifiedTime; "-" locally
    content_sha256: str  # hash of the *normalized markdown*
    chunk_count: int
    web_url: str = ""  # link back to the original, for citations
    local_file: str = ""  # the .md we wrote. The join key between the manifest
    #                       and the corpus on disk -- do not derive this from the
    #                       title, which is the heading INSIDE the document and
    #                       need not resemble the Drive filename at all.
    error: str = ""  # non-empty if this document failed to ingest


@dataclass
class Manifest:
    generated_at: str = ""
    source: str = ""
    docs: list[DocEntry] = field(default_factory=list)

    # ---------- construction ----------

    @staticmethod
    def build(source: str, docs: list[DocEntry]) -> "Manifest":
        return Manifest(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=source,
            docs=docs,
        )

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        payload = {
            "generated_at": self.generated_at,
            "source": self.source,
            "docs": [asdict(d) for d in self.docs],
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> "Manifest":
        raw = json.loads(Path(path).read_text())
        return Manifest(
            generated_at=raw["generated_at"],
            source=raw["source"],
            docs=[DocEntry(**d) for d in raw["docs"]],
        )

    # ---------- the point of the whole thing ----------

    @property
    def total_chunks(self) -> int:
        return sum(d.chunk_count for d in self.docs)

    @property
    def failed(self) -> list[DocEntry]:
        return [d for d in self.docs if d.error]

    @property
    def empty(self) -> list[DocEntry]:
        """Ingested without error but produced no chunks. Usually a parser
        failure on a scanned PDF: the file is 'there' and says nothing."""
        return [d for d in self.docs if not d.error and d.chunk_count == 0]

    def reconcile(self, expected_ids: set[str]) -> dict:
        """Compare what the source says exists against what we actually indexed."""
        got = {d.doc_id for d in self.docs if not d.error and d.chunk_count > 0}
        return {
            "expected": len(expected_ids),
            "indexed": len(got),
            "missing": sorted(expected_ids - got),
            "unexpected": sorted(got - expected_ids),
            "failed": [(d.doc_id, d.title, d.error) for d in self.failed],
            "empty": [(d.doc_id, d.title) for d in self.empty],
        }

    def report(self, expected_ids: set[str] | None = None) -> str:
        lines = [
            f"Manifest  source={self.source}  generated={self.generated_at}",
            f"  documents : {len(self.docs)}",
            f"  chunks    : {self.total_chunks}",
        ]
        if self.failed:
            lines.append(f"  FAILED    : {len(self.failed)}")
            for d in self.failed:
                lines.append(f"      - {d.title}: {d.error}")
        if self.empty:
            lines.append(f"  EMPTY     : {len(self.empty)}  (parsed, but no text)")
            for d in self.empty:
                lines.append(f"      - {d.title}")

        if expected_ids is not None:
            r = self.reconcile(expected_ids)
            lines.append(f"  expected  : {r['expected']}   indexed: {r['indexed']}")
            if r["missing"]:
                lines.append(f"  MISSING   : {len(r['missing'])}")
                for m in r["missing"]:
                    lines.append(f"      - {m}")
            else:
                lines.append("  reconciled: OK, nothing missing")

        return "\n".join(lines)
