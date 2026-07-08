"""
app/utils/embedder.py
─────────────────────
Singleton wrapper around a HuggingFace sentence-transformers model.

The model is loaded once on first access (lazy init) and reused across
all ingestion and retrieval calls. Loading takes ~2–5 seconds on first
call; subsequent calls are instant.

Default model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~22M parameters — fast on CPU
  - English-optimised, good cross-lingual transfer for common languages

Multilingual swap (no code changes needed beyond .env):
  EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - Same interface, 50+ language support
  - Slightly slower and larger (~117M parameters)

Usage:
    from app.utils.embedder import get_embedder
    embedder = get_embedder()
    vectors = embedder.embed(["text one", "text two"])
    # → list[list[float]], shape (2, 384)
"""

from functools import lru_cache
from typing import Union

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Thin wrapper around SentenceTransformer that standardises the interface
    and isolates the model import to this file.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        # Import here so the heavy torch/transformers import only happens
        # when an Embedder is actually instantiated (not at module load time)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model ready — dim=%d, model=%s",
            self.embedding_dim,
            model_name,
        )

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """
        Embed one or more text strings.

        Args:
            texts: A single string or a list of strings.

        Returns:
            List of embedding vectors. Each vector is a list[float] of
            length self.embedding_dim. Order matches the input order.

        Notes:
            - Empty strings are replaced with a single space to avoid
              zero-vector edge cases in ChromaDB distance calculations.
            - batch_size=64 is a safe default for CPU; the model handles
              batching internally.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Guard: replace empty strings so the model doesn't produce zero vectors
        cleaned = [t if t.strip() else " " for t in texts]

        vectors = self._model.encode(
            cleaned,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity = dot product after L2 norm
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string and return a flat vector.
        Convenience wrapper used by the retriever.
        """
        return self.embed([query])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    Return the singleton Embedder.
    Model is loaded on first call; cached for all subsequent calls.
    Thread-safe via Python's GIL for the lru_cache lookup.
    """
    return Embedder(settings.embedding_model)


def warmup() -> Embedder:
    """
    Force the embedding model to load now rather than on the first request.
    Called from the FastAPI startup lifespan (app/main.py) so the first
    /ask or /ingest call isn't the one paying the ~2-5s model load cost.
    """
    return get_embedder()
