"""
Grounded answering.

The requirement is "answer only from the documents". Most demos implement this
by putting "only use the provided context" in the system prompt and hoping.
That is one gate, it is probabilistic, and it fails quietly.

There are two gates here:

  Gate 1 (deterministic, free).  If retrieval returns nothing above a score
  floor, we refuse without calling the model at all. No model call means no
  opportunity to hallucinate, and no token spend on a question we already know
  we cannot answer.

  Gate 2 (model-level).  The prompt provides numbered sources, requires inline
  citations, and requires an explicit refusal when the sources are insufficient.

Gate 1 is the one that matters. A RAG system that quietly guesses is worse than
no system, because a wrong answer with a confident tone and a citation attached
is more dangerous than no answer at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from retrieval import Hit, Retriever, tokenize

# A floor, but a low one. Its only job is to catch a query that matched nothing
# at all. It is deliberately NOT the grounding gate.
#
# The first version of this file used a score threshold of 2.0 as the gate, and
# it failed in both directions on the very first run:
#
#   "what is a KX-Code?"            score 1.83  -> refused, but the answer WAS
#                                                  in the glossary
#   "what's our crypto policy?"     score 9.60  -> confidently retrieved the
#                                                  vacation policy, because the
#                                                  word "policy" is everywhere
#
# BM25 scores are not comparable across queries -- they scale with query length
# and with the rarity of the words used. A fixed threshold on an incomparable
# number is superstition. The gate below asks a question that actually has a
# stable answer.
SCORE_FLOOR = 0.5

REFUSAL = (
    "I couldn't find anything about that in the documents I have access to. "
    "If you think it should be there, the folder may not be indexed yet."
)

SYSTEM_PROMPT = """You answer questions strictly from the numbered sources provided below.

Rules, in order of importance:

1. Use ONLY the sources. You have no other knowledge. If the sources do not
   contain the answer, say so plainly and stop. Do not fill gaps with general
   knowledge, do not guess, and do not reason from what is "usually" true.
2. Cite the source number inline for every claim, like [1] or [2].
3. If the sources partially answer the question, answer that part and say
   explicitly which part you cannot answer.
4. Be concise. Do not restate the question. Do not pad.
5. Never mention "the sources say" as a hedge for something they do not say.

If nothing in the sources is relevant, reply with exactly:
NO_ANSWER
"""


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    refused: bool
    gate: str = ""  # which gate refused, for debugging


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] {h.chunk.doc_title} — {h.chunk.section_path}\n{h.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def format_citations(hits: list[Hit], used: list[int] | None = None) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        if used and i not in used:
            continue
        lines.append(f"  [{i}] {h.chunk.doc_title} — {h.chunk.section_path}")
    return "\n".join(lines)


def answer(
    question: str,
    retriever: Retriever,
    k: int = 4,
    call_model: bool = True,
) -> Answer:
    hits = retriever.search(question, k=k)

    # ---- Gate 1a: nothing matched at all ----
    if not hits or hits[0].score < SCORE_FLOOR:
        return Answer(text=REFUSAL, hits=hits, refused=True, gate="no-match")

    # ---- Gate 1b: concept coverage ----
    #
    # Take the most distinctive idea in the question -- the word that carries
    # the actual meaning, not the scaffolding around it. Then ask one question:
    #
    #     does that idea appear, in any of its forms, in the text we retrieved?
    #
    # If the user asks about crypto payments and the word "crypto" exists
    # nowhere in the corpus, no similarity score should be able to talk us into
    # answering. If the user asks about a KX-Code and "kx-code" is sitting right
    # there in the glossary, no low score should talk us out of it.
    #
    # This gate is cheap, deterministic, and explains itself in the logs.
    get_concepts = getattr(retriever, "concepts_for", None)
    if get_concepts is not None:
        cs = get_concepts(question)
        if cs:
            key = cs[0]  # most distinctive concept

            if key.oov:
                # The corpus has never seen this word in any document.
                return Answer(
                    text=REFUSAL, hits=hits, refused=True, gate="unknown-term"
                )

            covered = any(
                key.forms & set(tokenize(h.chunk.text)) for h in hits
            )
            if not covered:
                return Answer(
                    text=REFUSAL, hits=hits, refused=True, gate="no-coverage"
                )

    if not call_model:
        return Answer(
            text="(model call skipped — retrieval only)",
            hits=hits,
            refused=False,
        )

    # ---- Gate 2: model ----
    text = _call_model(question, build_context(hits))

    if text.strip().upper().startswith("NO_ANSWER"):
        return Answer(text=REFUSAL, hits=hits, refused=True, gate="model")

    return Answer(text=text, hits=hits, refused=False)


def _call_model(question: str, context: str) -> str:
    """Anthropic by default; set MODEL_PROVIDER=gemini to use Gemini instead."""
    provider = os.environ.get("MODEL_PROVIDER", "anthropic").lower()
    prompt = f"SOURCES:\n\n{context}\n\n---\n\nQUESTION: {question}"

    if provider == "anthropic":
        import requests

        key = os.environ["ANTHROPIC_API_KEY"]
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": os.environ.get("MODEL", "claude-sonnet-4-6"),
                "max_tokens": 800,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        r.raise_for_status()
        return "".join(
            b.get("text", "") for b in r.json()["content"] if b.get("type") == "text"
        ).strip()

    if provider == "gemini":
        import requests

        key = os.environ["GEMINI_API_KEY"]
        model = os.environ.get("MODEL", "gemini-2.5-flash")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": prompt}]}],
                # Reasoning is disabled. Sage leaked raw reasoning into Slack
                # answers; the cheapest fix is to not generate it for a task
                # that is extraction, not deliberation.
                "generationConfig": {"temperature": 0.0},
            },
            timeout=60,
        )
        r.raise_for_status()
        cands = r.json().get("candidates", [])
        if not cands:
            return "NO_ANSWER"
        return "".join(
            p.get("text", "") for p in cands[0]["content"]["parts"]
        ).strip()

    raise ValueError(f"unknown MODEL_PROVIDER: {provider!r}")
