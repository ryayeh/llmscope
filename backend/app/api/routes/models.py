from fastapi import APIRouter, HTTPException, Query, status

from app.core.errors import LLMScopeError
from app.schemas.huggingface_local import (
    HuggingFaceAttentionRequest,
    HuggingFaceAttentionResponse,
    HuggingFaceLocalDiagnosticsResponse,
    HuggingFaceLocalLoadRequest,
    HuggingFaceLocalStatusResponse,
)
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


@router.get(
    "/providers/hugging-face-local",
    response_model=HuggingFaceLocalStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Return Hugging Face Local runtime status",
)
def get_huggingface_local_status() -> HuggingFaceLocalStatusResponse:
    try:
        return generation_service.get_huggingface_local_status()
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc


@router.post(
    "/providers/hugging-face-local/load",
    response_model=HuggingFaceLocalStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Download and load a supported Hugging Face Local model",
)
def load_huggingface_local_model(
    request: HuggingFaceLocalLoadRequest,
) -> HuggingFaceLocalStatusResponse:
    try:
        return generation_service.load_huggingface_local_model(request)
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc


@router.post(
    "/providers/hugging-face-local/unload",
    response_model=HuggingFaceLocalStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Unload the active Hugging Face Local model",
)
def unload_huggingface_local_model() -> HuggingFaceLocalStatusResponse:
    try:
        return generation_service.unload_huggingface_local_model()
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc


@router.get(
    "/providers/hugging-face-local/diagnostics",
    response_model=HuggingFaceLocalDiagnosticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Return Hugging Face Local environment diagnostics",
)
def get_huggingface_local_diagnostics() -> HuggingFaceLocalDiagnosticsResponse:
    try:
        return generation_service.get_huggingface_local_diagnostics()
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc


@router.post(
    "/providers/hugging-face-local/attention",
    response_model=HuggingFaceAttentionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze token attention for the loaded Hugging Face Local model",
)
def analyze_huggingface_local_attention(
    request: HuggingFaceAttentionRequest,
) -> HuggingFaceAttentionResponse:
    try:
        return generation_service.analyze_huggingface_attention(request)
    except LLMScopeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
