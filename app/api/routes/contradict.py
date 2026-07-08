from fastapi import APIRouter, Depends

from app.api.dependencies import get_contradiction_module
from app.api.errors import DocumentNotFoundError, EmptyQueryError
from app.core.retriever import document_exists
from app.models.request import ContradictRequest
from app.models.response import ContradictResponse

router = APIRouter()


@router.post("/contradict", response_model=ContradictResponse)
def contradict(
    request: ContradictRequest,
    contradiction=Depends(get_contradiction_module),
):
    # Validate topic
    if not request.topic.strip():
        raise EmptyQueryError()

    # Validate first document
    if not document_exists(request.doc_id_1):
        raise DocumentNotFoundError(request.doc_id_1)

    # Validate second document
    if not document_exists(request.doc_id_2):
        raise DocumentNotFoundError(request.doc_id_2)

    # Compare both documents
    return contradiction.check_contradiction(
        doc_id_1=request.doc_id_1,
        doc_id_2=request.doc_id_2,
        topic=request.topic,
    )