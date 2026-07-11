"""
Agent 2: Evidence Retrieval Agent

For each claim element, runs hybrid retrieval (BM25 + vector) over the
inventor disclosure to find the passage(s) most likely to support or
contradict it. Deliberately has NO LLM call — this stage is pure,
cheap, local retrieval, which matters at scale (this is the layer that
avoids paying per-claim-element LLM cost for something a search index
can do).
"""
from __future__ import annotations
from typing import List

from app.retrieval.hybrid_search import HybridRetriever
from app.schemas import ClaimElement, EvidenceMatch


def retrieve_evidence(
    disclosure: str, claim_elements: List[ClaimElement]
) -> List[EvidenceMatch]:
    retriever = HybridRetriever(disclosure)
    matches: List[EvidenceMatch] = []

    for el in claim_elements:
        hits = retriever.retrieve(el.text, top_k=1)
        if hits:
            top = hits[0]
            matches.append(
                EvidenceMatch(
                    element_id=el.element_id,
                    passage=top.passage,
                    bm25_score=top.bm25_score,
                    vector_score=top.vector_score,
                    hybrid_score=top.hybrid_score,
                )
            )
        else:
            matches.append(
                EvidenceMatch(
                    element_id=el.element_id,
                    passage="",
                    bm25_score=0.0,
                    vector_score=0.0,
                    hybrid_score=0.0,
                )
            )
    return matches
