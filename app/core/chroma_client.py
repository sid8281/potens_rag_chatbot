from functools import lru_cache

import chromadb

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR
    )


@lru_cache(maxsize=1)
def get_collection():
    client = get_client()

    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )