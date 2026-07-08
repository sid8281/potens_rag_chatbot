"""
pipeline_example.py — NOT part of the app; shows how ask.py should call
the four core modules together. Copy this logic into
app/api/routes/ask.py once dependencies.py wires up injection.
"""
from app.config import settings
from app.core import citation, llm_client, prompt_builder, retriever
from app.core.language import detect_and_translate  # from Hour 10-13 block
from app.models.response import AskResponse


def answer_question(query: str, top_k: int = None, doc_filter: list[str] = None) -> AskResponse:
    # 1. Handle multilingual boundary
    translated_query, detected_language = detect_and_translate(query)

    # 2. Retrieve
    chunks = retriever.retrieve(translated_query, top_k=top_k, doc_filter=doc_filter)

    # 3. Confidence + HITL gate (computed before the LLM call so we can
    #    still answer, but flag it, rather than blocking the request)
    confidence = citation.mean_confidence(chunks)
    hitl_required = confidence < settings.CONFIDENCE_THRESHOLD

    # 4. Build prompt
    prompt = prompt_builder.build_answer_prompt(chunks, query=query, language=detected_language)

    # 5. Call LLM (hallucination guard enforced inside llm_client)
    result = llm_client.generate(prompt)

    # 6. Build citations (empty if no_answer, since nothing was actually used)
    citations = [] if result["no_answer"] else citation.build_citations(chunks)

    return AskResponse(
        answer=result["text"] if not result["no_answer"] else "The documents do not contain an answer to this question.",
        language=detected_language,
        citations=citations,
        confidence=confidence,
        hitl_required=hitl_required,
        no_answer=result["no_answer"],
    )
