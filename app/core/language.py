"""
language.py — detect language, translate query to English for retrieval,
and hand back the detected language so the answer can be delivered in kind.
Translation happens at the query boundary only; the vector store stays
monolingual (English). See README for the translation-boundary rationale.
"""
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

_LANG_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh-cn": "Chinese",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
}


def detect_language(text: str) -> str:
    """Return a human-readable language name (falls back to English)."""
    try:
        code = detect(text)
    except LangDetectException:
        return "English"
    return _LANG_CODE_TO_NAME.get(code, code)


def detect_and_translate(query: str) -> tuple[str, str]:
    """
    Detect the query's language and translate it to English for embedding
    lookup against the (English-only) vector store.

    Returns:
        (translated_query, detected_language_name)
    """
    detected_language = detect_language(query)

    if detected_language == "English":
        return query, detected_language

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
    except Exception:
        # Translation service failure shouldn't break the whole request —
        # fall back to the original text and let retrieval do its best.
        translated = query

    return translated, detected_language
