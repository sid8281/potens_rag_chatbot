"""
retriever.py — thin wrapper around ChromaDB.
Single responsibility: take a query string (+ optional doc filter), return
chunks with similarity scores. No prompt logic, no LLM logic here.
"""
from typing import Optional
from app.utils.reranker import rerank
from app.config import settings
from app.core.chroma_client import get_collection as _get_collection
from app.utils.embedder import get_embedder

# NOTE: retriever.py used to open its own `chromadb.PersistentClient`,
# separate from the one ingestion.py created. That meant two live
# connections to the same on-disk store. `_get_collection` is now just
# re-exported from app.core.chroma_client so the whole app shares exactly
# one client/collection instance. Kept as `_get_collection` (rather than
# renaming) since app/api/routes/health.py imports it by that name.


def retrieve(
    query: str,
    top_k: int = None,
    doc_filter: Optional[list[str]] = None,
) -> list[dict]:
    """
    Query ChromaDB for the most relevant chunks.

    Args:
        query: the (already-translated-to-English, if needed) query string.
        top_k: number of chunks to return. Defaults to settings.TOP_K.
        doc_filter: optional list of source_file names to restrict search to.

    Returns:
        List of dicts, each shaped like:
        {
            "text": str,
            "source_file": str,
            "page_number": int | None,
            "chunk_index": int,
            "score": float,   # similarity in [0, 1], higher = more relevant
        }
    """
    
    top_k = top_k or settings.TOP_K
    candidate_count = max(top_k * 4, 20)
    collection = _get_collection()
    embedder = get_embedder()

    # embed_query() returns a single flat vector (list[float]), which is
    # what ChromaDB's query_embeddings=[...] expects one of. embed() (plural)
    # returns a list of vectors and would double-nest the shape here.
    query_embedding = embedder.embed_query(query)

    where = None
    if doc_filter:
        where = {"source_file": {"$in": doc_filter}}
    
    candidate_count = max(top_k * 4, 20)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        where=where,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )
    
    

    # [RERANKER HOOK]
    # To add cross-encoder reranking, pass `results` through a
    # CrossEncoderReranker here before formatting the return value.
    # See: cross-encoder/ms-marco-MiniLM-L-6-v2

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        # Chroma returns cosine distance by default; convert to a
        # similarity score in [0, 1] for use as a confidence proxy.
        similarity = max(0.0, 1.0 - distance)
        chunks.append(
            {
                "text": doc,
                "source_file": meta.get("source_file", "unknown"),
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(similarity, 4),
            }
        )

    chunks = rerank(query=query,chunks=chunks,top_k=top_k)
    return chunks


def retrieve_for_topic(doc_id: str, topic: str, top_k: int = None) -> list[dict]:
    """
    Convenience wrapper for /contradict: retrieve chunks from a single
    document, filtered by doc_id, relevant to a given topic.
    """
    return retrieve(query=topic, top_k=top_k, doc_filter=[doc_id])


def document_exists(doc_id: str) -> bool:
    """
    Check whether a doc_id is present in the index at all. Used for input
    validation (e.g. /contradict should 404 clearly, not silently return
    an empty-evidence response, when a doc_id was mistyped).
    """
    collection = _get_collection()
    result = collection.get(where={"source_file": doc_id}, limit=1)
    return len(result.get("ids", [])) > 0


def list_indexed_documents() -> list[str]:
    """Return the distinct set of source_file names currently indexed."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    sources = {m.get("source_file") for m in result.get("metadatas", []) if m}
    return sorted(sources)
