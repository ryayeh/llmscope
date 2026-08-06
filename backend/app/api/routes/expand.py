from fastapi import APIRouter, HTTPException, status

from app.core.errors import LLMScopeError
from app.schemas.generation import NodeExpansionRequest, NodeExpansionResponse
from app.services.generation_service import generation_service

router = APIRouter(tags=["expansion"])


@router.post(
    "/expand-node",
    response_model=NodeExpansionResponse,
    status_code=status.HTTP_200_OK,
    summary="Return the next-token distribution for a graph node",
)
def expand_node(request: NodeExpansionRequest) -> NodeExpansionResponse:
    try:
        return generation_service.expand_node(request)
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
