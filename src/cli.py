#!/usr/bin/env python3
"""
Demo CLI.

  python3 cli.py index                        # build index, print manifest
  python3 cli.py eval --no-llm                # run the test set
  python3 cli.py ask "how many vacation days?" --no-llm

Any command takes --corpus to point at a different document set:

  python3 cli.py eval --no-llm --corpus ../corpus-drive

The Drive-ingested corpus and the local one go through exactly the same code
path. That is the point: if retrieval behaves differently on the two, the
difference is in the documents, not in the system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from answering import answer, format_citations
from chunking import chunk_markdown
from manifest import DocEntry, Manifest, sha256
from retrieval import LocalBM25

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "corpus"


def manifest_path_for(corpus_dir: Path) -> Path:
    """One manifest per corpus, named after it. Keeps the Drive run from
    silently overwriting the local one."""
    if corpus_dir.name == "corpus":
        return ROOT / "manifest.json"
    return ROOT / f"manifest-{corpus_dir.name.replace('corpus-', '')}.json"


def load_corpus(corpus_dir: Path):
    """Local stand-in for Drive ingestion. Same shape: files in, chunks and a
    manifest out. drive_ingest.py produces a directory this can read."""
    if not corpus_dir.is_dir():
        sys.exit(f"no such corpus directory: {corpus_dir}")

    chunks, entries = [], []

    for path in sorted(corpus_dir.glob("*.md")):
        md = path.read_text()
        title = md.splitlines()[0].lstrip("# ").strip() if md else path.stem
        doc_chunks = chunk_markdown(md, doc_id=path.name, doc_title=title)
        chunks.extend(doc_chunks)
        entries.append(
            DocEntry(
                doc_id=path.name,
                title=title,
                source_mime="text/markdown",
                revision="-",
                content_sha256=sha256(md),
                chunk_count=len(doc_chunks),
                web_url=f"file://{path}",
            )
        )

    if not chunks:
        sys.exit(f"corpus is empty: {corpus_dir}")

    return chunks, Manifest.build(source=str(corpus_dir), docs=entries)


def cmd_index(corpus_dir: Path) -> None:
    _, man = load_corpus(corpus_dir)
    out = manifest_path_for(corpus_dir)
    man.save(out)

    expected = {p.name for p in corpus_dir.glob("*.md")}
    print(man.report(expected_ids=expected))
    print(f"\nmanifest written to {out}")


def cmd_ask(corpus_dir: Path, question: str, use_llm: bool, k: int) -> None:
    chunks, _ = load_corpus(corpus_dir)
    retriever = LocalBM25(chunks)

    res = answer(question, retriever, k=k, call_model=use_llm)

    print(f"\nQ: {question}\n")
    print(res.text)

    if res.refused:
        print(f"\n[refused at gate: {res.gate}]")
        explain = {
            "no-match": "nothing in the corpus matched at all",
            "unknown-term": "the key word of the question appears in no document",
            "no-coverage": "top results never mention what was actually asked about",
            "model": "sources were retrieved but did not contain the answer",
        }
        print(f"[{explain.get(res.gate, 'refused')}]")
        if res.hits:
            # Printed precisely because it is often HIGH. That is the lesson:
            # a confident score is not evidence of an answer.
            print(f"[top score was {res.hits[0].score:.2f} — and still wrong]")
        return

    print("\nSources:")
    print(format_citations(res.hits))
    print(f"\n[top score {res.hits[0].score:.2f}  backend={retriever.name}]")


# Questions the corpus CAN answer, and questions it CANNOT.
# The second list is the important one. Any RAG demo can do the first.
EVAL_ANSWERABLE = [
    "how many vacation days do I get?",
    "can I carry unused holiday into next year?",
    "what's the first response time for a critical incident?",
    "how much does the Growth plan cost?",
    "can I deploy on a Friday?",
    "when do I get production access?",
    "what is a KX-Code?",
    "is 2FA required?",
    "can I expense a co-working space?",
    "what happens if an S1 isn't resolved in 4 hours?",
]

EVAL_MUST_REFUSE = [
    "what's our policy on crypto payments?",
    "how do I request a sabbatical?",
    "what's the parental leave allowance?",
    "who is the CEO?",
    "what's the company's revenue?",
    "can I bring a dog to the office?",
]


def cmd_eval(corpus_dir: Path, use_llm: bool) -> None:
    chunks, _ = load_corpus(corpus_dir)
    retriever = LocalBM25(chunks)

    print(f"corpus: {corpus_dir}  ({len(chunks)} chunks)")
    print()
    print("=" * 72)
    print("ANSWERABLE — these must NOT be refused")
    print("=" * 72)

    fails = 0
    for q in EVAL_ANSWERABLE:
        res = answer(q, retriever, call_model=use_llm)
        ok = not res.refused
        fails += 0 if ok else 1
        score = res.hits[0].score if res.hits else 0.0
        top = res.hits[0].chunk.section_path if res.hits else "—"
        print(f"  [{'PASS' if ok else 'FAIL'}] {score:5.2f}  {q}")
        print(f"          -> {top}")

    print()
    print("=" * 72)
    print("UNANSWERABLE — these MUST be refused")
    print("=" * 72)

    for q in EVAL_MUST_REFUSE:
        res = answer(q, retriever, call_model=use_llm)
        ok = res.refused
        fails += 0 if ok else 1
        score = res.hits[0].score if res.hits else 0.0
        print(f"  [{'PASS' if ok else 'FAIL'}] {score:5.2f}  {q}")
        if not ok:
            print(f"          LEAKED -> {res.hits[0].chunk.section_path}")

    total = len(EVAL_ANSWERABLE) + len(EVAL_MUST_REFUSE)
    print()
    print(f"{total - fails}/{total} passed")
    sys.exit(1 if fails else 0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="corpus directory (default: ../corpus)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index")

    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--no-llm", action="store_true")
    a.add_argument("-k", type=int, default=4)

    e = sub.add_parser("eval")
    e.add_argument("--no-llm", action="store_true")

    args = p.parse_args()
    corpus_dir = args.corpus.resolve()

    if args.cmd == "index":
        cmd_index(corpus_dir)
    elif args.cmd == "ask":
        cmd_ask(corpus_dir, args.question, use_llm=not args.no_llm, k=args.k)
    elif args.cmd == "eval":
        cmd_eval(corpus_dir, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
