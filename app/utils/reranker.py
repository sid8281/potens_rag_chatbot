"""
app/utils/reranker.py
─────────────────────
Semantic reranking for retrieved chunks using a sentence-transformers CrossEncoder.

The CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2) is a small, fast
model that scores (query, chunk) pairs on semantic relevance. It's more
accurate than embedding similarity alone but requires more compute per
reranking call, so it's typically used as a post-processing step on the
top-K retrieved results.

Singleton pattern ensures the model is loaded once and reused across all
reranking calls (no reload per request).

Usage:
    from app.utils.reranker import rerank

    chunks = [
        {"text": "...", "source_file": "...", "page_number": 1, ...},
        ...
    ]
    query = "What is attention?"
    reranked = rerank(query, chunks, top_k=3)
    # → chunks with added "rerank_score" key, top-3 by that score
"""

from functools import lru_cache
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Thin wrapper around a sentence-transformers CrossEncoder model.
    Loads the model once (lazy init) and scores (query, chunk) pairs.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Initialize the reranker with a CrossEncoder model.

        Args:
            model_name: HuggingFace model ID for the CrossEncoder.
                       Default is the fast, accurate ms-marco-MiniLM-L-6-v2.
        """
        logger.info("Loading CrossEncoder model: %s", model_name)
        # Import here so torch/transformers only load when reranker is instantiated
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("CrossEncoder model ready: %s", model_name)

    def rerank_pairs(self, query: str, texts: list[str]) -> list[float]:
        """
        Score a list of (query, text) pairs.

        Args:
            query: The user's query string.
            texts: List of chunk texts to score against the query.

        Returns:
            List of scores (float), one per text, in the same order.
            Higher score = higher semantic relevance to the query.
        """
        if not texts:
            return []

        # Build (query, text) pairs for the model
        pairs = [[query, text] for text in texts]

        # Predict scores — returns a 1D array of floats
        scores = self._model.predict(pairs)

        # Convert numpy array to Python list of floats
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """
    Return the singleton CrossEncoderReranker.

    The model is loaded on first call; cached for all subsequent calls.
    Thread-safe via Python's GIL for the lru_cache lookup.

    Returns:
        CrossEncoderReranker instance.
    """
    return CrossEncoderReranker()


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved chunks using a CrossEncoder model.

    Takes a query and a list of chunks (already retrieved via embedding
    similarity), scores each (query, chunk["text"]) pair using the
    CrossEncoder, adds the score to each chunk, sorts by descending score,
    and returns the top-k.

    Args:
        query: The user's query string (in English, typically already
               translated if multilingual support is enabled).
        chunks: List of chunk dicts, each with:
                {
                    "text": str,          # chunk content
                    "source_file": str,   # document name
                    "page_number": int,   # page in PDF (or None)
                    "chunk_index": int,   # position in document
                    "score": float,       # original embedding similarity
                }
        top_k: Number of top-reranked chunks to return. If top_k >= len(chunks),
               all chunks are returned (sorted).

    Returns:
        List of reranked chunks, each with an added "rerank_score" key:
        {
            "text": str,
            "source_file": str,
            "page_number": int,
            "chunk_index": int,
            "score": float,           # original embedding similarity (unchanged)
            "rerank_score": float,    # CrossEncoder relevance score (new)
        }

        Sorted in descending order by rerank_score.
        Length <= top_k (or < if fewer chunks were input).

        If an exception occurs during reranking, the original chunks
        (with no "rerank_score" added) are returned unchanged, and a
        warning is logged — the application does not crash.

    Example:
        >>> query = "What is attention in neural networks?"
        >>> chunks = [
        ...     {"text": "Attention is a mechanism...", "source_file": "paper.pdf", ...},
        ...     {"text": "Transformers use attention...", "source_file": "paper.pdf", ...},
        ... ]
        >>> reranked = rerank(query, chunks, top_k=2)
        >>> len(reranked)
        2
        >>> reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
        True
    """
    # Edge case: empty input — nothing to rerank
    if not chunks:
        logger.debug("rerank(): received empty chunk list, returning empty list")
        return []

    try:
        # Get the singleton reranker
        reranker = get_reranker()

        # Extract texts in the same order as chunks
        texts = [chunk["text"] for chunk in chunks]

        logger.debug("Reranking %d chunks for query: %s", len(chunks), query[:60])

        # Score all (query, text) pairs
        rerank_scores = reranker.rerank_pairs(query, texts)

        # Attach rerank_score to each chunk, preserving all other keys
        reranked_chunks = []
        for chunk, score in zip(chunks, rerank_scores):
            # Create a shallow copy so we don't modify the input
            chunk_with_score = chunk.copy()
            chunk_with_score["rerank_score"] = float(score)
            reranked_chunks.append(chunk_with_score)

        # Sort by rerank_score in descending order (higher score = better)
        reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Return only top-k
        result = reranked_chunks[:top_k]
        logger.debug(
            "Reranking complete: returned %d/%d chunks (requested top_k=%d)",
            len(result),
            len(chunks),
            top_k,
        )

        return result

    except Exception as e:
        # Graceful degradation: if reranking fails, return the original
        # chunks unchanged instead of crashing the application. Log the
        # error so we can debug it, but don't let /ask or /contradict fail.
        logger.warning(
            "CrossEncoder reranking failed (falling back to original chunks): %s",
            str(e),
            exc_info=True,
        )
        return chunks[:top_k] if top_k < len(chunks) else chunks

