"""
Embeddings, via the Gemini API.

Deliberately not a vector database. For a corpus this size a database would be
a dependency, a service to run, and a thing to explain -- in exchange for
nothing. Cosine similarity over a few hundred vectors is a dot product; numpy
does it in microseconds. Scale is the reason to reach for Pinecone, and scale
is not present here.

That is a judgement to revisit at maybe 50k chunks, not a principle.

Embeddings are cached on disk, keyed by a hash of the chunk text. Re-running the
index does not re-embed unchanged documents, which matters both for cost and for
the patience of whoever is iterating on the corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta/models"

# The API caps batch size; keep well under it.
BATCH = 50


def _key(text: str, model: str, dim: int) -> str:
    h = hashlib.sha256(f"{model}:{dim}:{text}".encode("utf-8")).hexdigest()
    return h[:32]


class Embedder:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dim: int = 768,
        cache_path: str | Path = "../.embeddings.json",
    ) -> None:
        self.model = model
        self.dim = dim
        self.cache_path = Path(cache_path)
        self.cache: dict[str, list[float]] = {}

        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception:
                # A corrupt cache is not worth crashing over. Rebuild it.
                self.cache = {}

    # ---------- persistence ----------

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache))

    # ---------- api ----------

    def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        import requests

        key = os.environ["GEMINI_API_KEY"]
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task,
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        r = requests.post(
            f"{API}/{self.model}:batchEmbedContents",
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return [e["values"] for e in r.json()["embeddings"]]

    # ---------- public ----------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus chunks. Cached; only the misses hit the network."""
        out: list[list[float] | None] = [None] * len(texts)
        todo: list[int] = []

        for i, t in enumerate(texts):
            k = _key(t, self.model, self.dim)
            if k in self.cache:
                out[i] = self.cache[k]
            else:
                todo.append(i)

        for start in range(0, len(todo), BATCH):
            idx = todo[start : start + BATCH]
            vecs = self._embed_batch(
                [texts[i] for i in idx], task="RETRIEVAL_DOCUMENT"
            )
            for i, v in zip(idx, vecs):
                out[i] = v
                self.cache[_key(texts[i], self.model, self.dim)] = v

        if todo:
            self.save()

        return [v for v in out if v is not None]

    def embed_query(self, text: str) -> list[float]:
        """Queries use a different task type than documents. This is not
        cosmetic -- asymmetric embedding measurably improves retrieval, because
        a question and the passage that answers it are not the same shape."""
        return self._embed_batch([text], task="RETRIEVAL_QUERY")[0]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
