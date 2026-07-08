"""
citation.py — maps raw retrieved chunks into Citation objects.
Single responsibility: construct structured citations. No LLM calls,
no retrieval calls here — pure data shaping, easily unit-testable.
"""
from app.models.response import Citation


def build_citations(chunks: list[dict]) -> list[Citation]:
    """
    Convert raw chunk dicts (from retriever.retrieve()) into Citation objects.

    Each chunk is expected to look like:
        {
            "text": str,
            "source_file": str,
            "page_number": int | None,
            "chunk_index": int,
            "score": float,
        }
    """
    citations = []
    for chunk in chunks:
        snippet = chunk["text"]
        if len(snippet) > 240:
            snippet = snippet[:240].rsplit(" ", 1)[0] + "..."

        citations.append(
            Citation(
                source_file=chunk["source_file"],
                page_number=chunk.get("page_number"),
                chunk_index=chunk["chunk_index"],
                snippet=snippet,
                relevance_score=chunk["score"],
            )
        )
    return citations


def mean_confidence(chunks: list[dict]) -> float:
    """Mean of top-k similarity scores — used as the confidence proxy."""
    if not chunks:
        return 0.0
    return round(sum(c["score"] for c in chunks) / len(chunks), 4)


def top_citation(chunks: list[dict]) -> Citation:
    """
    Return the single highest-scoring chunk as a Citation.
    Used by /contradict, which returns one evidence citation per document.
    """
    if not chunks:
        raise ValueError("Cannot build a citation from an empty chunk list")
    best = max(chunks, key=lambda c: c["score"])
    return build_citations([best])[0]
