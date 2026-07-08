"""
errors.py — consistent error response schema across all routes.
Every error returned by the API, regardless of route, has this shape:

{
    "error": {
        "code": "string",       # machine-readable, e.g. "DOC_NOT_FOUND"
        "message": "string",    # human-readable
        "detail": "string | null"
    }
}
"""
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: str | None = None


class AppError(Exception):
    """Base class for all expected application errors."""

    def __init__(self, code: str, message: str, status_code: int = 400, detail: str | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class EmptyQueryError(AppError):
    def __init__(self):
        super().__init__(
            code="EMPTY_QUERY",
            message="Query must not be empty.",
            status_code=400,
        )


class DocumentNotFoundError(AppError):
    def __init__(self, doc_id: str):
        super().__init__(
            code="DOC_NOT_FOUND",
            message=f"Document '{doc_id}' was not found in the index.",
            status_code=404,
            detail=f"doc_id={doc_id}",
        )


class UnsupportedFileTypeError(AppError):
    def __init__(self, extension: str, allowed: set[str]):
        super().__init__(
            code="UNSUPPORTED_FILE_TYPE",
            message=f"File type '{extension}' is not supported.",
            status_code=400,
            detail=f"allowed types: {sorted(allowed)}",
        )


class IngestionFailedError(AppError):
    def __init__(self, reason: str):
        super().__init__(
            code="INGESTION_FAILED",
            message="Document ingestion failed.",
            status_code=500,
            detail=reason,
        )


class LLMUnavailableError(AppError):
    def __init__(self, reason: str):
        super().__init__(
            code="LLM_UNAVAILABLE",
            message="The language model service is currently unavailable.",
            status_code=503,
            detail=reason,
        )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            }
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so raw tracebacks never leak to the client."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "detail": str(exc),
            }
        },
    )
