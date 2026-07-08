from fastapi import APIRouter, Depends
from app.utils.reranker import rerank

from app.api.dependencies import (
    get_citation_module,
    get_language_module,
    get_llm_client,
    get_prompt_builder,
    get_retriever,
)
from app.api.errors import EmptyQueryError, LLMUnavailableError
from app.config import settings
from app.models.request import AskRequest
from app.models.response import AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    retriever=Depends(get_retriever),
    prompt_builder=Depends(get_prompt_builder),
    llm_client=Depends(get_llm_client),
    citation=Depends(get_citation_module),
    detect_and_translate=Depends(get_language_module),
):
    # 0. Input validation — empty/whitespace-only query
    if not request.query or not request.query.strip():
        raise EmptyQueryError()

    # 1. Multilingual boundary: translate query to English for retrieval,
    #    remember the detected language to answer back in kind.
    translated_query, detected_language = detect_and_translate(request.query)

    # 2. Retrieve
    chunks = retriever.retrieve(
        translated_query, top_k=request.top_k, doc_filter=request.doc_filter
    )
    # After: chunks = retriever.retrieve(...)
    chunks = rerank(translated_query, chunks, top_k=request.top_k)

    # 3. Confidence + HITL gate
    confidence = citation.mean_confidence(chunks)
    hitl_required = confidence < settings.CONFIDENCE_THRESHOLD

    if not chunks:
        return AskResponse(
            answer="The documents do not contain an answer to this question.",
            language=detected_language,
            citations=[],
            confidence=0.0,
            hitl_required=True,
            no_answer=True,
        )

    # 4. Build prompt
    prompt = prompt_builder.build_answer_prompt(
        chunks, query=request.query, language=detected_language
    )

    # 5. Call LLM — hallucination guard enforced inside llm_client
    try:
        result = llm_client.generate(prompt)
    except Exception as e:
        raise LLMUnavailableError(reason=str(e))

    # 6. Build citations (skip if the model said the docs don't cover it)
    citations = [] if result["no_answer"] else citation.build_citations(chunks)
    answer_text = (
        "The documents do not contain an answer to this question."
        if result["no_answer"]
        else result["text"]
    )

    return AskResponse(
        answer=answer_text,
        language=detected_language,
        citations=citations,
        confidence=confidence,
        hitl_required=hitl_required,
        no_answer=result["no_answer"],
    )
