"""
app/core/ingestion.py
─────────────────────
Ingestion pipeline: file → text → chunks → embeddings → ChromaDB.

Supported file types:
  .pdf   — PyPDF page-by-page extraction with page number tracking
  .txt   — plain UTF-8 text
  .md    — Markdown (treated as plain text; structure preserved in chunks)

Pipeline:
  1. load_document()   — extract raw text + build page_map for PDFs
  2. chunk_document()  — RecursiveCharacterTextSplitter → Chunk objects
  3. embed()           — sentence-transformers batch encode
  4. store_chunks()    — upsert to ChromaDB with metadata

ChromaDB upsert semantics:
  Re-ingesting the same file replaces its existing chunks (same chunk_id).
  This means /ingest is idempotent — safe to call multiple times on the
  same document without creating duplicates.

Returns IngestResult with doc_id, chunk count, and any per-file errors.
"""


import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb

from app.config import settings
from app.core.chroma_client import get_collection as get_chroma_collection
from app.utils.chunker import Chunk, chunk_document
from app.utils.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Result model ─────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Returned by ingest_file() and ingest_directory()."""
    doc_id: str
    chunks_created: int
    status: str                        # "success" | "error" | "skipped"
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# ── ChromaDB client ───────────────────────────────────────────────────────────
# `get_chroma_collection` is re-exported from app.core.chroma_client so
# existing callers (scripts/ingest_docs.py, etc.) keep working unchanged.
# There is now exactly ONE PersistentClient for the whole app — see
# app/core/chroma_client.py for the singleton implementation.


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_document(file_path: Path) -> tuple[str, dict[int, int]]:
    """
    Extract raw text from a supported file type.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        (full_text, page_map) where:
          full_text — the entire document as a single string
          page_map  — {char_offset: page_number} for PDFs (empty dict for text files)

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix in {".txt", ".md"}:
        return _load_text(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Supported: .pdf, .txt, .md"
        )


def _load_pdf(file_path: Path) -> tuple[str, dict[int, int]]:
    """Extract text from a PDF, tracking which character offset each page starts at."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages_text: list[str] = []
    page_map: dict[int, int] = {}   # char_offset → 1-based page number
    char_cursor = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        # Some PDFs have hyphenated line breaks — normalise them
        page_text = re.sub(r"-\n(\w)", r"\1", page_text)
        # Collapse excessive blank lines
        page_text = re.sub(r"\n{3,}", "\n\n", page_text)

        page_map[char_cursor] = page_num
        pages_text.append(page_text)
        char_cursor += len(page_text) + 1   # +1 for the newline separator

    full_text = "\n".join(pages_text)

    if not full_text.strip():
        logger.warning(
            "PDF '%s' produced empty text — may be scanned/image-only. "
            "Consider OCR preprocessing.", file_path.name
        )

    logger.info("Loaded PDF '%s': %d pages, %d chars", file_path.name, len(reader.pages), len(full_text))
    return full_text, page_map


def _load_text(file_path: Path) -> tuple[str, dict[int, int]]:
    """Load a plain text or Markdown file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive blank lines
    logger.info("Loaded text file '%s': %d chars", file_path.name, len(text))
    return text, {}


# ── Step 2: Chunk ─────────────────────────────────────────────────────────────
# Delegated to app/utils/chunker.py — see that module for strategy details.


# ── Step 3 + 4: Embed & Store ────────────────────────────────────────────────

def store_chunks(chunks: list[Chunk], collection: chromadb.Collection) -> int:
    """
    Embed chunks in batches and upsert into ChromaDB.

    Uses upsert (not add) so re-ingesting the same document replaces
    existing chunks rather than creating duplicates.

    Args:
        chunks:     Chunk objects from chunk_document().
        collection: The ChromaDB collection to write into.

    Returns:
        Number of chunks successfully stored.
    """
    if not chunks:
        return 0

    embedder = get_embedder()
    BATCH_SIZE = 100   # stay well within ChromaDB's default max_batch_size

    stored = 0
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = [c.metadata for c in batch]

        logger.debug(
            "Embedding batch %d–%d of %d chunks",
            batch_start + 1,
            batch_start + len(batch),
            len(chunks),
        )
        embeddings = embedder.embed(texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        stored += len(batch)

    logger.info("Stored %d chunks in ChromaDB collection '%s'", stored, collection.name)
    return stored


# ── Main pipeline entry point ─────────────────────────────────────────────────

def ingest_file(
    file_path: Path,
    collection: Optional[chromadb.Collection] = None,
) -> IngestResult:
    """
    Full ingestion pipeline for a single file.

    Args:
        file_path:  Path to the document (.pdf, .txt, or .md).
        collection: Existing ChromaDB collection. If None, a new client is
                    created (useful for the CLI script; pass the collection
                    in FastAPI to reuse the connection).

    Returns:
        IngestResult with doc_id, chunk count, and status.
    """
    source_file = file_path.name
    logger.info("=== Ingesting: %s ===", source_file)

    if collection is None:
        collection = get_chroma_collection()

    try:
        # Step 1: Load
        full_text, page_map = load_document(file_path)

        if not full_text.strip():
            return IngestResult(
                doc_id=source_file,
                chunks_created=0,
                status="skipped",
                warnings=["Document produced no extractable text."],
            )

        # Step 2: Chunk
        chunks = chunk_document(
            text=full_text,
            source_file=source_file,
            page_map=page_map if page_map else None,
        )

        if not chunks:
            return IngestResult(
                doc_id=source_file,
                chunks_created=0,
                status="skipped",
                warnings=["Chunker produced 0 chunks — document may be too short."],
            )

        # Steps 3 + 4: Embed & Store
        stored = store_chunks(chunks, collection)

        logger.info("✓ Ingested '%s': %d chunks stored", source_file, stored)
        return IngestResult(
            doc_id=source_file,
            chunks_created=stored,
            status="success",
        )

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return IngestResult(doc_id=source_file, chunks_created=0, status="error", error=str(e))
    except ValueError as e:
        logger.error("Unsupported file: %s", e)
        return IngestResult(doc_id=source_file, chunks_created=0, status="error", error=str(e))
    except Exception as e:
        logger.exception("Unexpected error ingesting '%s'", source_file)
        return IngestResult(doc_id=source_file, chunks_created=0, status="error", error=str(e))


def ingest_document(
    file_path: Path | str,
    doc_id: str,
    collection: Optional[chromadb.Collection] = None,
) -> int:
    """
    Ingest a single document using its original filename (doc_id) for metadata/citations.
    Returns the number of chunks successfully stored in ChromaDB.
    """
    file_path = Path(file_path)
    source_file = doc_id
    logger.info("=== Ingesting: %s ===", source_file)

    if collection is None:
        collection = get_chroma_collection()

    # Step 1: Load
    full_text, page_map = load_document(file_path)

    if not full_text.strip():
        logger.warning("Document '%s' produced no extractable text.", source_file)
        return 0

    # Step 2: Chunk
    chunks = chunk_document(
        text=full_text,
        source_file=source_file,
        page_map=page_map if page_map else None,
    )

    if not chunks:
        logger.warning("Chunker produced 0 chunks for '%s'.", source_file)
        return 0

    # Steps 3 + 4: Embed & Store
    stored = store_chunks(chunks, collection)

    logger.info("✓ Ingested '%s': %d chunks stored", source_file, stored)
    return stored


def ingest_directory(
    directory: Path,
    extensions: tuple[str, ...] = (".pdf", ".txt", ".md"),
) -> list[IngestResult]:
    """
    Ingest all supported documents in a directory (non-recursive).
    Uses a single shared ChromaDB collection for efficiency.

    Args:
        directory:  Path to the folder containing raw documents.
        extensions: File extensions to include.

    Returns:
        List of IngestResult, one per file found.
    """
    files = [f for f in sorted(directory.iterdir()) if f.suffix.lower() in extensions]

    if not files:
        logger.warning("No supported files found in '%s'", directory)
        return []

    logger.info("Found %d file(s) to ingest in '%s'", len(files), directory)
    collection = get_chroma_collection()
    results: list[IngestResult] = []

    for file_path in files:
        result = ingest_file(file_path, collection=collection)
        results.append(result)

    # Summary
    success = sum(1 for r in results if r.status == "success")
    total_chunks = sum(r.chunks_created for r in results)
    logger.info(
        "Ingestion complete: %d/%d files succeeded, %d total chunks",
        success, len(results), total_chunks,
    )
    return results


def delete_document(source_file: str, collection: Optional[chromadb.Collection] = None) -> int:
    """
    Remove all chunks belonging to a document from ChromaDB.
    Used when re-ingesting a changed file cleanly (optional utility).

    Returns: number of chunks deleted.
    """
    if collection is None:
        collection = get_chroma_collection()

    existing = collection.get(where={"source_file": source_file})
    ids_to_delete = existing.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for '%s'", len(ids_to_delete), source_file)
    else:
        logger.info("No existing chunks found for '%s'", source_file)

    return len(ids_to_delete)


def preload_documents(force: bool = False) -> list[IngestResult]:
    """
    Preload documents from data/documents.

    Behaviour:
    - Existing PDFs are skipped.
    - Newly added PDFs are automatically ingested.
    - force=True re-ingests everything.
    """

    logger.info("=" * 70)
    logger.info("PRELOAD STARTED")
    logger.info("=" * 70)

    collection = get_chroma_collection()

    preload_dir = Path(settings.preload_docs_dir)

    if not preload_dir.exists():
        logger.warning("Directory '%s' not found.", preload_dir)
        return []

    pdfs = sorted(
        f for f in preload_dir.iterdir()
        if f.suffix.lower() in {".pdf", ".txt", ".md"}
    )

    if not pdfs:
        logger.warning("No documents found.")
        return []

    # Documents already indexed
    indexed_docs = set()

    if collection.count() > 0:
        data = collection.get(include=["metadatas"])

        for meta in data.get("metadatas", []):
            if meta and "source_file" in meta:
                indexed_docs.add(meta["source_file"])

    logger.info("Already indexed: %d document(s)", len(indexed_docs))

    results = []

    for pdf in pdfs:

        if pdf.name in indexed_docs and not force:
            logger.info("Skipping %s (already indexed)", pdf.name)
            continue

        logger.info("Ingesting %s", pdf.name)

        results.append(
            ingest_file(pdf, collection)
        )

    logger.info("=" * 70)
    logger.info("Preload complete.")
    logger.info("=" * 70)

    return results


def get_collection_stats(collection: Optional[chromadb.Collection] = None) -> dict:
    """
    Return a summary of what is currently stored in ChromaDB.
    Used by the /health endpoint.
    """
    if collection is None:
        collection = get_chroma_collection()

    count = collection.count()

    # Get unique doc names from stored metadata
    if count > 0:
        sample = collection.get(limit=count, include=["metadatas"])
        doc_names = sorted({
            m["source_file"]
            for m in sample.get("metadatas", [])
            if m and "source_file" in m
        })
    else:
        doc_names = []

    return {
        "total_chunks": count,
        "total_documents": len(doc_names),
        "documents": doc_names,
        "collection": collection.name,
    }
