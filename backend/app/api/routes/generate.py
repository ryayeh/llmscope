from fastapi import APIRouter, status

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
    return generation_service.build_response(request)
