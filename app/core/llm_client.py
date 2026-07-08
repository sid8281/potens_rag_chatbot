"""
llm_client.py — Gemini API wrapper.
Single responsibility: send a formatted prompt, get raw text back, and detect
the NO_ANSWER_IN_DOCS sentinel. No citation logic, no prompt-building logic here.
"""
import google.generativeai as genai

from app.config import settings

NO_ANSWER_SENTINEL = "NO_ANSWER_IN_DOCS"

_model = None


def _get_model():
    global _model
    if _model is None:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        _model = genai.GenerativeModel(settings.GEMINI_MODEL)
    return _model


def warmup() -> bool:
    """
    Eagerly configure the Gemini client on startup (rather than on the
    first /ask call). Only initializes the client/model object — does not
    make a network call, so startup stays fast even if Gemini is slow to
    respond.

    Returns:
        True if the client was configured successfully, False otherwise
        (e.g. missing/invalid API key). Never raises — startup should not
        crash the whole app just because Gemini config failed; /status
        will surface the problem instead.
    """
    try:
        _get_model()
        return True
    except Exception:
        return False


def is_connected() -> bool:
    """Whether the Gemini client is currently initialized."""
    return _model is not None


def generate(prompt: str) -> dict:
    """
    Send a prompt to Gemini and return a structured result.

    Returns:
        {
            "text": str,          # raw model output (empty string if no_answer)
            "no_answer": bool,    # True if the model signalled NO_ANSWER_IN_DOCS
            "raw_response": str,  # unmodified model text, for logging/debugging
        }
    """
    model = _get_model()
    response = model.generate_content(prompt)
    raw_text = (response.text or "").strip()


    print("\n" + "=" * 80)
    print("RAW GEMINI RESPONSE:")
    print(raw_text)
    print("=" * 80 + "\n")

    # Hallucination guard: don't pass raw text through if the model
    # signalled that the docs don't cover the question. Being lenient
    # about exact match (sentinel anywhere in a short response) protects
    # against the model wrapping it in a stray sentence.
    no_answer = NO_ANSWER_SENTINEL in raw_text

    return {
        "text": "" if no_answer else raw_text,
        "no_answer": no_answer,
        "raw_response": raw_text,
    }


def generate_contradiction(prompt: str) -> dict:
    """
    Send a contradiction-analysis prompt to Gemini.

    Returns:
        {
            "contradicts": bool,
            "reasoning": str,
        }
    """
    model = _get_model()
    response = model.generate_content(prompt)
    raw_text = (response.text or "").strip()

    # Simple, robust parse: look for a leading yes/no rather than requiring
    # strict JSON output from the model (keeps the prompt simple for a
    # 24-hour build; swap for structured output later if needed).
    lowered = raw_text.lower()
    contradicts = lowered.startswith("yes") or "\nyes" in lowered[:50].lower()

    return {
        "contradicts": contradicts,
        "reasoning": raw_text,
    }

