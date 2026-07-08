"""
prompt_builder.py — loads .txt templates and injects variables.
Single responsibility: string assembly. No LLM calls, no retrieval calls here.
"""
from pathlib import Path

from app.config import settings

_TEMPLATE_CACHE: dict[str, str] = {}


def _load_template(filename: str) -> str:
    """Read a prompt template from disk, caching it in memory after first read."""
    if filename not in _TEMPLATE_CACHE:
        path = Path(settings.PROMPTS_DIR) / filename
        _TEMPLATE_CACHE[filename] = path.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[filename]


def build_answer_prompt(chunks: list[dict], query: str, language: str = "English") -> str:
    """
    Build the /ask prompt from retrieved chunks.

    Args:
        chunks: list of chunk dicts from retriever.retrieve()
        query: the user's original (untranslated) query text to answer against.
                Note: retrieval happens on the translated query, but we prefer
                echoing the user's original phrasing back into the prompt where
                possible for clarity in the demo/log trail.
        language: human-readable language name for the model to respond in
                  (e.g. "English", "French", "Hindi").
    """
    template = _load_template("answer_prompt.txt")

    context = "\n\n".join(
        f"[Source: {c['source_file']}, chunk {c['chunk_index']}"
        f"{', page ' + str(c.get('page_number')) if c.get('page_number') else ''}]\n{c['text']}"
        for c in chunks
    )

    return template.format(context=context, query=query, language=language)


def build_contradiction_prompt(
    doc_id_1: str,
    doc_id_2: str,
    topic: str,
    chunks_a: list[dict],
    chunks_b: list[dict],
) -> str:
    """Build the /contradict prompt from two sets of retrieved chunks."""
    template = _load_template("contradiction_prompt.txt")

    context_a = "\n\n".join(
        f"[chunk {c['chunk_index']}] {c['text']}" for c in chunks_a
    )
    context_b = "\n\n".join(
        f"[chunk {c['chunk_index']}] {c['text']}" for c in chunks_b
    )

    return template.format(
        topic=topic,
        doc_id_1=doc_id_1,
        doc_id_2=doc_id_2,
        context_a=context_a,
        context_b=context_b,
    )
