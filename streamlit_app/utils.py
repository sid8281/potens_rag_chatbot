"""
utils.py — thin httpx client wrapping the FastAPI backend.
Single responsibility: HTTP calls only. No Streamlit rendering logic here,
so it can be reused or tested independently of the UI.
"""
import httpx

API_BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0


class APIError(Exception):
    """Raised when the backend returns a structured {"error": {...}} response."""

    def __init__(self, code: str, message: str, detail: str | None = None):
        self.code = code
        self.detail = detail
        super().__init__(message)


def _unwrap(response: httpx.Response) -> dict:
    """Raise APIError with the backend's message if the response is an error."""
    if response.status_code >= 400:
        try:
            body = response.json()
            err = body.get("error", {})
            raise APIError(
                code=err.get("code", "UNKNOWN_ERROR"),
                message=err.get("message", "Request failed."),
                detail=err.get("detail"),
            )
        except ValueError:
            response.raise_for_status()
    return response.json()


def ask(query: str, top_k: int = 5, doc_filter: list[str] | None = None) -> dict:
    payload = {"query": query, "top_k": top_k}
    if doc_filter:
        payload["doc_filter"] = doc_filter

    response = httpx.post(f"{API_BASE_URL}/ask", json=payload, timeout=TIMEOUT)
    return _unwrap(response)


def contradict(doc_id_1: str, doc_id_2: str, topic: str) -> dict:
    payload = {"doc_id_1": doc_id_1, "doc_id_2": doc_id_2, "topic": topic}
    response = httpx.post(f"{API_BASE_URL}/contradict", json=payload, timeout=TIMEOUT)
    return _unwrap(response)


def ingest(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes)}
    response = httpx.post(f"{API_BASE_URL}/ingest", files=files, timeout=TIMEOUT)
    return _unwrap(response)


def health() -> dict:
    response = httpx.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
    return _unwrap(response)
