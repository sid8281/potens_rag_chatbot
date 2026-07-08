"""
app/utils/chunker.py
────────────────────
Wraps LangChain's RecursiveCharacterTextSplitter to produce chunks with
fully-populated metadata suitable for citation generation.

Strategy: RecursiveCharacterTextSplitter
  chunk_size=700, chunk_overlap=100  (configurable via settings)

The splitter tries these separators in order before hard-cutting:
  "\n\n"  →  paragraph boundary     (legal articles, policy sections)
  "\n"    →  line boundary          (technical docs, bullet points)
  ". "    →  sentence boundary      (prose paragraphs)
  " "     →  word boundary          (last resort before character cut)

Why 700 / 100:
  700 chars ≈ 120–160 tokens for English — carries a complete legal clause
  or technical explanation without drowning the retrieval signal.
  100-char overlap ensures a governing clause at the end of chunk N
  is still present at the start of chunk N+1.

Metadata attached to every chunk:
  source_file   — filename only (e.g. "gdpr.pdf"), no full path
  page_number   — page where the chunk starts (PDF only; 0 for plain text)
  chunk_index   — 0-based position within this document's chunk sequence
  char_start    — character offset of chunk start in the full document text
  total_chunks  — total number of chunks for this document (set post-split)
  snippet       — first 120 chars of chunk text (for citation previews)

Future enhancement: replace with semantic chunking that splits on
embedding-similarity discontinuities when document structure is irregular.
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and storage in ChromaDB.

    text       — the raw chunk content sent to the embedding model
    metadata   — dict stored alongside the vector in ChromaDB;
                 every field here maps directly to a Citation field
    chunk_id   — unique string ID for ChromaDB (derived from source + index)
    """
    text: str
    metadata: dict
    chunk_id: str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            source = self.metadata.get("source_file", "unknown")
            idx = self.metadata.get("chunk_index", 0)
            # Normalise to a safe ChromaDB key: no spaces, no slashes
            safe_source = source.replace(" ", "_").replace("/", "_")
            self.chunk_id = f"{safe_source}__chunk_{idx:04d}"


# ── Splitter (module-level singleton) ────────────────────────────────────────

def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,          # character-based, no tokenizer dependency
        is_separator_regex=False,
    )


_splitter = _make_splitter()


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    source_file: str,
    page_map: Optional[dict[int, int]] = None,
) -> list[Chunk]:
    """
    Split a document's full text into chunks with complete metadata.

    Args:
        text:        Full extracted text of the document.
        source_file: Filename (not full path) — e.g. "gdpr.pdf".
                     Stored in metadata and used as the doc_id for /contradict.
        page_map:    Optional dict mapping character offset → page number.
                     Built by the PDF loader; None for plain-text files.

    Returns:
        List of Chunk objects, ordered by position in the document.
    """
    if not text or not text.strip():
        logger.warning("chunk_document received empty text for '%s' — skipping", source_file)
        return []

    raw_chunks: list[str] = _splitter.split_text(text)

    if not raw_chunks:
        logger.warning("Splitter produced 0 chunks for '%s'", source_file)
        return []

    chunks: list[Chunk] = []
    char_cursor = 0

    for idx, chunk_text in enumerate(raw_chunks):
        # Approximate character start — find the chunk in the remaining text
        # (split_text doesn't return offsets, so we track position manually)
        try:
            char_start = text.index(chunk_text, max(0, char_cursor - settings.chunk_overlap))
        except ValueError:
            char_start = char_cursor  # fallback if exact match fails (whitespace diff)

        char_cursor = char_start + len(chunk_text)

        # Resolve page number from page_map if available
        page_number = 0
        if page_map:
            # page_map: {char_offset: page_number} — find the highest offset ≤ char_start
            page_number = _resolve_page(page_map, char_start)

        metadata = {
            "source_file": source_file,
            "page_number": page_number,
            "chunk_index": idx,
            "char_start": char_start,
            "total_chunks": len(raw_chunks),   # back-filled correctly since we know total
            "snippet": chunk_text[:120].replace("\n", " ").strip(),
        }

        chunks.append(Chunk(text=chunk_text, metadata=metadata))

    logger.info(
        "Chunked '%s' → %d chunks  (size=%d, overlap=%d)",
        source_file,
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks


def _resolve_page(page_map: dict[int, int], char_offset: int) -> int:
    """
    Given a page_map {char_offset: page_number}, return the page number
    for a given character offset using the last page boundary ≤ char_offset.
    """
    page = 0
    for boundary, page_num in sorted(page_map.items()):
        if boundary <= char_offset:
            page = page_num
        else:
            break
    return page
