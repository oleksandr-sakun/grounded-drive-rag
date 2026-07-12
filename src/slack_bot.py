#!/usr/bin/env python3
"""
Slack bot.

Deliberately thin. Every decision that matters -- retrieval, the refusal gates,
grounding -- already happened in answering.py. This file moves text between Slack
and that function, and does nothing else. If a bug in an answer can be reproduced
from the CLI, it does not live here.

That is the test for whether an interface layer is the right size.

    pip install slack-bolt

    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    python3 slack_bot.py

Socket Mode: no public URL, no tunnel, no inbound firewall rule. The bot dials
out. For an internal assistant that is the correct trade -- there is nothing to
expose and nothing to attack.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import sys
from pathlib import Path

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answering import answer  # noqa: E402
from chunking import chunk_markdown  # noqa: E402
from config import Settings  # noqa: E402
from manifest import Manifest  # noqa: E402
from retrieval import build_retriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("slack_bot")


def slugify_title(name: str) -> str:
    """Mirror of drive_ingest.slugify. Kept in step deliberately: the filename it
    produces is the join key between the manifest and the corpus on disk."""
    name = re.sub(r"\.(md|pdf|txt|csv|docx?|xlsx?)$", "", name, flags=re.I)
    s = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    return re.sub(r"[\s_]+", "-", s)[:60] or "untitled"

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(os.environ.get("CORPUS_DIR", ROOT / "corpus-drive"))

# Greetings and meta-comments. These are not questions, and running them through
# retrieval produces a confident answer to a question nobody asked. Cheaper to
# recognise them than to explain the resulting nonsense.
SMALL_TALK = re.compile(
    r"^\s*(hi|hey|hello|yo|thanks|thank you|thx|ta|ok|okay|cool|nice|got it|"
    r"good (morning|afternoon|evening))\s*[!.?]*\s*$",
    re.I,
)

GREETING = (
    "Hi. Ask me anything about the handbook — vacation, expenses, SLAs, "
    "the release process. I answer only from the documents, and I'll tell you "
    "when the answer isn't in them."
)


def load_index():
    """Build the retriever once, at startup.

    Rebuilding per request would burn embedding calls and add a second of latency
    to every message for no benefit. The corpus does not change between requests;
    it changes when someone re-runs ingestion, and that is a restart.
    """
    chunks = []
    urls: dict[str, str] = {}

    # The manifest carries the Drive links. Chunks don't, so citations would
    # otherwise be plain text -- and a citation you cannot click is a citation
    # nobody checks.
    #
    # Keyed on the SLUGIFIED filename, not on the document title. Titles are not
    # stable identifiers: drive_ingest writes "kestrel-pricing.md" from a Sheet
    # whose first heading is "# kestrel-pricing", while a Doc's title and its
    # heading can differ entirely. Matching on title silently produced no link
    # for half the corpus -- and a broken citation is worse than none, because
    # nobody reports it, they just stop clicking.
    manifest_path = ROOT / "manifest-drive.json"
    if manifest_path.exists():
        man = Manifest.load(manifest_path)
        for d in man.docs:
            if not d.web_url or d.web_url.startswith("file://"):
                continue
            if not d.local_file:
                continue
            # Keyed on the file the ingester actually wrote. Reconstructing this
            # from the title does not work: the title is the heading INSIDE the
            # document, the filename comes from Drive, and they routinely differ
            # ("08-glossary.md" contains "# Glossary"). Deriving one from the
            # other silently produced zero links -- and a citation with no link
            # is a citation nobody checks, so nobody reports it.
            urls[pathlib.Path(d.local_file).stem] = d.web_url
    else:
        log.warning(
            "no manifest-drive.json -- citations will have no links. "
            "Run drive_ingest.py first."
        )

    for path in sorted(CORPUS.glob("*.md")):
        md = path.read_text()
        title = md.splitlines()[0].lstrip("# ").strip() if md else path.stem
        chunks.extend(chunk_markdown(md, doc_id=path.stem, doc_title=title))

    if not chunks:
        sys.exit(f"corpus is empty: {CORPUS}")

    settings = Settings.from_env()
    retriever = build_retriever(settings, chunks)

    log.info(
        "indexed %d chunks from %s | backend=%s",
        len(chunks),
        CORPUS,
        retriever.name,
    )
    return retriever, urls


RETRIEVER, DOC_URLS = load_index()
app = App(token=os.environ["SLACK_BOT_TOKEN"])


def format_sources(hits, urls: dict[str, str]) -> str:
    seen, lines = set(), []
    n = 0
    for h in hits:
        key = h.chunk.doc_id  # file stem, matches the manifest key
        if key in seen:
            continue
        seen.add(key)
        n += 1

        url = urls.get(key)
        title = h.chunk.doc_title
        label = f"<{url}|{title}>" if url else title
        lines.append(f"[{n}] {label}")

    return "  ·  ".join(lines)


def handle(text: str, say, thread_ts: str) -> None:
    text = re.sub(r"<@[^>]+>", "", text).strip()

    if not text:
        say(text=GREETING, thread_ts=thread_ts)
        return

    if SMALL_TALK.match(text):
        say(text=GREETING, thread_ts=thread_ts)
        return

    res = answer(text, RETRIEVER, k=4, call_model=True)

    if res.refused:
        log.info("refused | gate=%s | q=%r", res.gate, text)
        say(text=f"{res.text}", thread_ts=thread_ts)
        return

    # Only list what the model actually cited. Dumping every retrieved chunk
    # means a question about pricing arrives with the expense policy attached,
    # which teaches people that the citations are decorative -- and once they
    # believe that, they stop checking them, which defeats the point of having
    # them at all.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", res.text)})
    hits = [res.hits[i - 1] for i in cited if 1 <= i <= len(res.hits)]

    body = res.text
    sources = format_sources(hits or res.hits[:1], DOC_URLS)
    if sources:
        body += f"\n\n_Sources: {sources}_"

    log.info("answered | top=%.2f | q=%r", res.hits[0].score, text)
    say(text=body, thread_ts=thread_ts)


@app.event("app_mention")
def on_mention(event, say):
    handle(event.get("text", ""), say, event.get("thread_ts") or event["ts"])


@app.event("message")
def on_dm(event, say):
    # Channel messages arrive here too; only answer DMs, or the bot talks over
    # every conversation it is a member of.
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id") or event.get("subtype"):
        return
    handle(event.get("text", ""), say, event.get("thread_ts") or event["ts"])


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
