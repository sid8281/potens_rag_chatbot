"""
contradiction.py — dual-doc retrieval + contradiction prompt logic.
Single responsibility: given two doc IDs and a topic, retrieve relevant
passages from each and ask the LLM whether they conflict.
"""
from app.core import citation, llm_client, prompt_builder, retriever
from app.models.response import ContradictResponse


def check_contradiction(
    doc_id_1: str, doc_id_2: str, topic: str, top_k: int = 5
) -> ContradictResponse:
    chunks_a = retriever.retrieve_for_topic(doc_id_1, topic, top_k=top_k)
    chunks_b = retriever.retrieve_for_topic(doc_id_2, topic, top_k=top_k)

    if not chunks_a or not chunks_b:
        missing = doc_id_1 if not chunks_a else doc_id_2
        return ContradictResponse(
            contradicts=False,
            reasoning=f"No relevant passages found in '{missing}' for topic '{topic}'.",
            evidence_a=citation.top_citation(chunks_a) if chunks_a else _empty_citation(doc_id_1),
            evidence_b=citation.top_citation(chunks_b) if chunks_b else _empty_citation(doc_id_2),
        )

    prompt = prompt_builder.build_contradiction_prompt(
        doc_id_1, doc_id_2, topic, chunks_a, chunks_b
    )
    result = llm_client.generate_contradiction(prompt)

    return ContradictResponse(
        contradicts=result["contradicts"],
        reasoning=result["reasoning"],
        evidence_a=citation.top_citation(chunks_a),
        evidence_b=citation.top_citation(chunks_b),
    )


def _empty_citation(doc_id: str):
    from app.models.response import Citation

    return Citation(
        source_file=doc_id,
        page_number=None,
        chunk_index=-1,
        snippet="(no relevant passage found)",
        relevance_score=0.0,
    )
