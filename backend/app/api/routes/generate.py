from fastapi import APIRouter, HTTPException, status

from app.core.errors import LLMScopeError
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.generation_service import generation_service

router = APIRouter(tags=["generation"])


@router.post(
    "/generate",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a response and inferred trace",
)
def generate(request: GenerationRequest) -> GenerationResponse:
    try:
        return generation_service.build_response(request)
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
