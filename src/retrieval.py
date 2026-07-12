"""
Retrieval, behind an interface.

The whole system talks to `Retriever.search(query) -> list[Hit]`. Nothing else
knows or cares what is underneath. That is deliberate: it is what lets a
production system migrate from a self-hosted index to a managed one (or back)
by changing one config value and restarting, rather than by rewriting the app.

Two backends here:
  LocalBM25  - self-hosted keyword retrieval. No cloud, no bill, no latency.
  VertexAISearch - managed semantic retrieval (Discovery Engine).

BM25 is implemented inline rather than pulled from a library, because the whole
of it is thirty lines and a dependency you understand is worth more than one
you import.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from chunking import Chunk

# ---------------------------------------------------------------------------
# Term normalisation
#
# The single highest-leverage component in a real RAG system, and the one
# everybody skips. Users type the words they use; documents contain the words
# the company wrote down. These are not the same words. In production this
# dictionary grows every time someone asks a question the bot got wrong -- it
# is the cheapest possible feedback loop.
# ---------------------------------------------------------------------------

TERM_DICTIONARY: dict[str, str] = {
    "pto": "paid time off vacation",
    "holiday": "vacation paid time off",
    "holidays": "vacation paid time off",
    "annual leave": "vacation paid time off",
    "days off": "vacation paid time off",
    "wfh": "remote work",
    "work from home": "remote work",
    "expenses": "expense reimbursable claim",
    "reimburse": "reimbursable expense claim",
    "reimbursement": "reimbursable expense claim",
    "2fa": "multi-factor authentication mfa",
    "mfa": "multi-factor authentication",
    "sev1": "s1 critical severity",
    "sev 1": "s1 critical severity",
    "p1": "s1 critical severity",
    "downtime": "production system down critical",
    "deploy": "deployment production release",
    "ship": "deployment production release",
    "rollback": "roll back deployment",
    "prod access": "production access",
    "laptop": "device laptop encryption",
    "cost": "price pricing plan",
    "how much": "price pricing plan",
}

# Function words, and the generic verbs English wraps around every question.
#
# This list has to be thorough, because of how the coverage gate below works: a
# word that is absent from the corpus is treated as strong evidence that we
# cannot answer. That inference is correct for "sabbatical" and nonsense for
# "many". The stoplist is what separates the two.
#
# The first run of the gate refused "how many vacation days do I get?" — because
# "many" appears nowhere in the corpus, and the gate concluded the corpus had
# never heard of the topic. It had. It just doesn't say "many".
STOPWORDS = {
    # articles, pronouns, prepositions, conjunctions
    "a", "an", "the", "i", "we", "you", "they", "it", "he", "she", "my", "our",
    "your", "their", "me", "us", "them", "of", "to", "in", "on", "for", "at",
    "by", "with", "from", "about", "as", "into", "than", "then", "so", "and",
    "or", "if", "but", "not", "no", "any", "some", "all", "each", "every",
    "that", "this", "these", "those", "there", "here",
    # auxiliaries and modals
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "done", "have", "has", "had", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must",
    # question words
    "what", "how", "when", "where", "who", "whom", "which", "why", "whose",
    # generic verbs and quantifiers that carry no topical meaning
    "get", "gets", "give", "take", "happen", "happens", "need", "want", "know",
    "tell", "say", "make", "use", "go", "come", "put", "let", "many", "much",
    "more", "most", "long", "far", "lot", "thing", "things", "please", "does",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")

# Contractions must die before tokenisation, or "isn't" becomes the token "isn",
# which is absent from every corpus on earth and trips the unknown-term gate.
CONTRACTIONS = [
    (re.compile(r"n['’]t\b"), " not "),
    (re.compile(r"['’](s|re|ve|ll|d|m)\b"), " "),
    (re.compile(r"['’]"), " "),
]


def normalise(text: str) -> str:
    text = text.lower()
    for pattern, repl in CONTRACTIONS:
        text = pattern.sub(repl, text)
    return text


def stem(w: str) -> str:
    """A deliberately crude suffix stripper.

    Not linguistically correct, and it does not need to be. It needs to be
    *consistent*: applied identically at index time and at query time, so that
    "carry" finds "carried" and "payments" is not a different word from
    "payment". Without it, the coverage gate below rejects correct answers
    over an -ed.
    """
    if len(w) <= 3 or "-" in w:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "i"
    elif w.endswith("es") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    if w.endswith("y") and len(w) > 3:
        w = w[:-1] + "i"
    return w


def expand(query: str) -> str:
    """Apply the term dictionary. Multi-word keys first, so 'work from home'
    is caught before 'work' is looked at on its own."""
    q = normalise(query)
    for term in sorted(TERM_DICTIONARY, key=len, reverse=True):
        if term in q:
            q = f"{q} {TERM_DICTIONARY[term]}"
    return q


def tokenize(text: str) -> list[str]:
    return [
        stem(t)
        for t in TOKEN_RE.findall(normalise(text))
        if t not in STOPWORDS and len(t) > 1
    ]


@dataclass
class Concept:
    """One idea in the query, with every surface form it might take.

    "2fa" and "multi-factor authentication" are the same concept. The user
    types one, the document contains the other. Grouping them is what makes it
    possible to ask "is this idea actually present in the retrieved text?"
    rather than "did these exact letters appear?".
    """

    forms: set[str]
    idf: float
    oov: bool  # no form of this concept exists anywhere in the corpus


def concepts(query: str, idf: dict[str, float], max_idf: float) -> list[Concept]:
    """Break a query into concepts, ranked by how distinctive each one is."""
    out: list[Concept] = []

    q = normalise(query)
    for raw in TOKEN_RE.findall(q):
        if raw in STOPWORDS or len(raw) <= 1:
            continue

        forms = {stem(raw)}
        for term, expansion in TERM_DICTIONARY.items():
            # A dictionary key can be multi-word ("work from home"); match it
            # against the raw query, not against the single token.
            if term == raw or (" " in term and term in q):
                forms |= {stem(t) for t in TOKEN_RE.findall(expansion)}

        known = [f for f in forms if f in idf]
        if known:
            out.append(Concept(forms=forms, idf=max(idf[f] for f in known), oov=False))
        else:
            # Nothing in the corpus resembles this word. That is the single
            # strongest possible signal that we cannot answer the question --
            # far stronger than any similarity score.
            out.append(Concept(forms=forms, idf=max_idf * 1.5, oov=True))

    out.sort(key=lambda c: c.idf, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[Hit]:
        ...


# ---------------------------------------------------------------------------
# Backend: local BM25
# ---------------------------------------------------------------------------


class LocalBM25:
    name = "local-bm25"

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.docs = [tokenize(c.text) for c in chunks]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0

        self.tf: list[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            for term in set(d):
                df[term] += 1

        # BM25 idf, floored at zero so a term in every document cannot
        # contribute negative score.
        self.idf = {
            t: max(0.0, math.log(1 + (self.n - c + 0.5) / (c + 0.5)))
            for t, c in df.items()
        }
        self.max_idf = max(self.idf.values()) if self.idf else 1.0

    def concepts_for(self, query: str) -> list[Concept]:
        """Query concepts, most distinctive first. Used by the grounding gate."""
        return concepts(query, self.idf, self.max_idf)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(expand(query))
        if not terms:
            return []

        scored: list[Hit] = []
        for i, chunk in enumerate(self.chunks):
            tf = self.tf[i]
            dl = len(self.docs[i]) or 1
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = f + self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                score += idf * (f * (self.K1 + 1)) / denom
            if score > 0:
                scored.append(Hit(chunk=chunk, score=score))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# Backend: Vertex AI Search (Discovery Engine)
#
# Same signature. Nothing above this line changes when you switch to it.
#
# Two things worth knowing before you wire this up, both learned the hard way:
#
#   1. Regional endpoints are real. A data store in `eu` must be called at
#      https://eu-discoveryengine.googleapis.com — the global host returns
#      an empty result set, not an error.
#
#   2. If the data store was created by a connector with ACL enforcement
#      enabled (`aclEnabled: true` — immutable, you cannot turn it off), a
#      service account will authenticate successfully and receive zero
#      results. No error. You need an OAuth identity that actually holds
#      access to the underlying documents.
#
# A Drive folder you own does not have this problem. A corporate Drive folder
# wired through SSO very well might.
# ---------------------------------------------------------------------------


class VertexAISearch:
    name = "vertex-ai-search"

    def __init__(
        self,
        project_id: str,
        data_store_id: str,
        location: str = "eu",
        access_token: str = "",
    ) -> None:
        self.project_id = project_id
        self.data_store_id = data_store_id
        self.location = location
        self.access_token = access_token

        host = (
            "discoveryengine.googleapis.com"
            if location == "global"
            else f"{location}-discoveryengine.googleapis.com"
        )
        self.endpoint = (
            f"https://{host}/v1/projects/{project_id}/locations/{location}"
            f"/collections/default_collection/dataStores/{data_store_id}"
            f"/servingConfigs/default_search:search"
        )

    def search(self, query: str, k: int = 5) -> list[Hit]:
        import requests  # local import: the local backend needs no network

        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.access_token}"},
            json={
                "query": expand(query),
                "pageSize": k,
                "contentSearchSpec": {
                    "extractiveContentSpec": {"maxExtractiveSegmentCount": 2}
                },
            },
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        hits: list[Hit] = []
        for rank, r in enumerate(results):
            doc = r.get("document", {})
            data = doc.get("derivedStructData", {})
            segments = data.get("extractive_segments") or data.get(
                "extractiveSegments", []
            )
            text = "\n\n".join(s.get("content", "") for s in segments).strip()
            if not text:
                continue
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=doc.get("id", f"vertex-{rank}"),
                        doc_id=doc.get("id", ""),
                        doc_title=data.get("title", "Untitled"),
                        section_path=data.get("title", "Untitled"),
                        text=text,
                    ),
                    # Discovery Engine does not return a comparable score, so
                    # rank is turned into one. Do not mix these numbers with
                    # BM25 scores; they are not the same currency.
                    score=1.0 / (rank + 1),
                )
            )
        return hits


def build_retriever(backend: str, chunks: list[Chunk], **kw) -> Retriever:
    if backend == "local":
        return LocalBM25(chunks)
    if backend == "vertex":
        return VertexAISearch(**kw)
    raise ValueError(f"unknown backend: {backend!r}")
