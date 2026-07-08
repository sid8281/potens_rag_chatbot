from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import AppError, app_error_handler, unhandled_exception_handler
from app.api.routes import ask, contradict, health, ingest
from app.core import llm_client
from app.core.chroma_client import get_collection
from app.core.ingestion import preload_documents
from app.utils.embedder import warmup as warmup_embedder
from app.utils.logger import configure_uvicorn_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    configure_uvicorn_logging()

    logger.info("Loading embedding model...")
    warmup_embedder()
    logger.info("Embedding model ready.")

    logger.info("Checking vector database...")
    get_collection()  # creates the singleton client + collection

    logger.info("Initializing Gemini client...")
    if llm_client.warmup():
        logger.info("Gemini client ready.")
    else:
        logger.warning(
            "Gemini client failed to initialize — check GEMINI_API_KEY. "
            "/ask will fail until this is fixed."
        )

    # Preload data/documents/ — no-op if the collection already has chunks.
    preload_documents()

    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    # Nothing to clean up: ChromaDB's PersistentClient and the embedding
    # model don't hold sockets/handles that need explicit closing.


app = FastAPI(
    title="RAG Document Q&A with Citations",
    description="Document Q&A over 5+ documents with citations, contradiction "
    "detection, multilingual support, and hallucination guarding.",
    version="1.0.0",
    lifespan=lifespan,
)

# Consistent error schema across all routes — see app/api/errors.py
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(ask.router, tags=["ask"])
app.include_router(contradict.router, tags=["contradict"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(health.router, tags=["health"])
#app.include_router(status.router, tags=["status"])


@app.get("/")
def root():
    return {"message": "RAG Assessment API — see /docs for endpoints"}
