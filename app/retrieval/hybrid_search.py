"""
Hybrid retrieval over the inventor disclosure.

Combines:
  - BM25 (rank_bm25): sparse, keyword-precise, catches exact technical terms.
  - Vector search (TF-IDF + cosine by default): catches paraphrase/semantic overlap.

Swap-in note: `VectorIndex` is written as a small interface. In production,
replace `TfidfVectorIndex` with a real embedding-backed index (ChromaDB /
FAISS + sentence-transformers or an API embedding model) by implementing
the same `.query(text, top_k)` method — no other file needs to change.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_disclosure(disclosure: str) -> List[str]:
    """Split disclosure into sentence-level passages. Swap for semantic
    chunking (paragraph-aware, overlap windows) for longer real disclosures."""
    sentences = re.split(r"(?<=[.!?])\s+", disclosure.strip())
    return [s.strip() for s in sentences if s.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class RetrievalHit:
    passage: str
    bm25_score: float
    vector_score: float
    hybrid_score: float


class TfidfVectorIndex:
    """Local, offline-safe stand-in for a dense vector index."""

    def __init__(self, passages: List[str]):
        self.passages = passages
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(passages) if passages else None

    def query(self, text: str) -> List[float]:
        if self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        return sims.tolist()


class HybridRetriever:
    def __init__(self, disclosure: str, bm25_weight: float = 0.5, vector_weight: float = 0.5):
        self.passages = chunk_disclosure(disclosure)
        self.bm25 = BM25Okapi([_tokenize(p) for p in self.passages]) if self.passages else None
        self.vector_index = TfidfVectorIndex(self.passages)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def retrieve(self, query: str, top_k: int = 1) -> List[RetrievalHit]:
        if not self.passages:
            return []

        bm25_raw = self.bm25.get_scores(_tokenize(query)).tolist()
        vector_raw = self.vector_index.query(query)

        bm25_norm = self._normalize(bm25_raw)
        vector_norm = self._normalize(vector_raw)

        hybrid = [
            self.bm25_weight * b + self.vector_weight * v
            for b, v in zip(bm25_norm, vector_norm)
        ]

        ranked = sorted(
            range(len(self.passages)),
            key=lambda i: hybrid[i],
            reverse=True,
        )[:top_k]

        return [
            RetrievalHit(
                passage=self.passages[i],
                bm25_score=bm25_raw[i],
                vector_score=vector_raw[i],
                hybrid_score=hybrid[i],
            )
            for i in ranked
        ]
