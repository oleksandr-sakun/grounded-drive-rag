"""
Configuration.

One environment variable decides how retrieval works. Nothing above the
Retriever interface knows which backend is live -- that is the whole point of
having the interface, and it is what makes a migration a config change rather
than a rewrite.

    RAG_BACKEND=bm25      keyword only. No network, no bill, no latency.
    RAG_BACKEND=hybrid    BM25 + Gemini embeddings, fused. Adds semantics.
    RAG_BACKEND=vertex    Vertex AI Search (Discovery Engine).

The grounding gate is deliberately NOT part of this choice. It runs against the
corpus vocabulary, not against the index, so it behaves identically on all three
backends. A semantic retriever must not be able to smuggle in an answer the
documents do not contain -- that is exactly the failure mode semantic search
makes easier, and exactly the one this system exists to prevent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    backend: str = "bm25"

    # --- hybrid ---
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    # Weight of the semantic ranking in the fusion. 0.0 = pure BM25.
    semantic_weight: float = 0.5
    embed_cache: str = "../.embeddings.json"

    # --- vertex ---
    project_id: str = ""
    data_store_id: str = ""
    location: str = "global"

    # --- generation ---
    model_provider: str = "gemini"
    model: str = "gemini-flash-latest"

    @staticmethod
    def from_env() -> "Settings":
        s = Settings()
        s.backend = os.environ.get("RAG_BACKEND", s.backend).lower()
        s.embed_model = os.environ.get("EMBED_MODEL", s.embed_model)
        s.semantic_weight = float(
            os.environ.get("SEMANTIC_WEIGHT", s.semantic_weight)
        )
        s.project_id = os.environ.get("GCP_PROJECT_ID", "")
        s.data_store_id = os.environ.get("VERTEX_DATA_STORE_ID", "")
        s.location = os.environ.get("VERTEX_LOCATION", s.location)
        s.model_provider = os.environ.get("MODEL_PROVIDER", s.model_provider)
        s.model = os.environ.get("MODEL", s.model)

        if s.backend not in ("bm25", "hybrid", "vertex"):
            raise ValueError(
                f"RAG_BACKEND must be bm25|hybrid|vertex, got {s.backend!r}"
            )
        if s.backend == "vertex" and not (s.project_id and s.data_store_id):
            raise ValueError(
                "RAG_BACKEND=vertex requires GCP_PROJECT_ID and VERTEX_DATA_STORE_ID"
            )
        return s
