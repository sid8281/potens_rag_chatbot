from fastapi import APIRouter

from app.config import settings
from app.core.retriever import _get_collection

router = APIRouter()


@router.get("/health")
def health():
    try:
        collection = _get_collection()
        all_docs = collection.get(include=["metadatas"])
        metadatas = all_docs.get("metadatas", [])
        total_chunks = len(metadatas)
        unique_sources = {m.get("source_file") for m in metadatas if m}

        return {
            "status": "ok",
            "chroma_docs": len(unique_sources),
            "total_chunks": total_chunks,
            "embedding_model": settings.EMBEDDING_MODEL,
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
        }
