from fastapi import APIRouter, HTTPException, status

from app.core.errors import LLMScopeError
from app.schemas.generation import ContinueGenerationRequest, ContinueGenerationResponse
from app.services.generation_service import generation_service

router = APIRouter(tags=["continuation"])


@router.post(
    "/continue-node",
    response_model=ContinueGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Reveal the next cached token or create a new continuation segment",
)
def continue_node(request: ContinueGenerationRequest) -> ContinueGenerationResponse:
    try:
        return generation_service.continue_node(request)
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
