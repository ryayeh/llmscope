from fastapi import APIRouter, status

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
    return generation_service.expand_node(request)
