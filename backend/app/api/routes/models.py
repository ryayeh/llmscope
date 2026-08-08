from fastapi import APIRouter, Query, status

from app.schemas.model_catalog import ModelCatalogResponse
from app.services.generation_service import generation_service

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=ModelCatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="Return available model options",
)
def list_models() -> ModelCatalogResponse:
    return generation_service.list_models(force_refresh=False)


@router.get(
    "/models/refresh",
    response_model=ModelCatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh provider-backed model discovery",
)
def refresh_models(force: bool = Query(default=True)) -> ModelCatalogResponse:
    return generation_service.list_models(force_refresh=force)
