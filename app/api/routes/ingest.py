import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.api.errors import IngestionFailedError, UnsupportedFileTypeError
from app.core.ingestion import ingest_document  # from your Hour 2-4 ingestion.py
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB, generous for a 24h build
STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MB — how much of the upload we hold in RAM at once


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Add a document to the existing ChromaDB collection without touching
    already-indexed documents (upsert semantics — see ingestion.py).

    The upload is streamed to a temp file in 1MB chunks rather than read
    fully into memory with a single file.read(), so large PDFs don't spike
    RAM usage.
    """
    if not file.filename:
        raise UnsupportedFileTypeError(extension="(none)", allowed=ALLOWED_EXTENSIONS)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(extension=suffix, allowed=ALLOWED_EXTENSIONS)

    # ── Stream upload to disk ────────────────────────────────────────────────
    total_bytes = 0
    tmp_path: str
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(STREAM_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise IngestionFailedError(
                        reason=f"File exceeds max size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
                    )
                tmp.write(chunk)
    except IngestionFailedError:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        Path(tmp_path).unlink(missing_ok=True)
        raise IngestionFailedError(reason="Uploaded file is empty.")

    logger.info("Received upload '%s' (%d bytes) — starting ingestion...", file.filename, total_bytes)

    try:
        chunks_created = ingest_document(file_path=tmp_path, doc_id=file.filename)
    except Exception as e:
        logger.exception("Ingestion failed for '%s'", file.filename)
        raise IngestionFailedError(reason=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info("Finished ingesting '%s' — %d chunks created.", file.filename, chunks_created)

    return {
        "doc_id": file.filename,
        "chunks_created": chunks_created,
        "status": "success",
    }
