from typing import Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5
    doc_filter: Optional[list[str]] = None


class ContradictRequest(BaseModel):
    doc_id_1: str
    doc_id_2: str
    topic: str = Field(..., min_length=1)
