from typing import Optional
from pydantic import BaseModel


class Citation(BaseModel):
    source_file: str
    page_number: Optional[int] = None
    chunk_index: int
    snippet: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    language: str
    citations: list[Citation]
    confidence: float
    hitl_required: bool
    no_answer: bool


class ContradictResponse(BaseModel):
    contradicts: bool
    reasoning: str
    evidence_a: Citation
    evidence_b: Citation
