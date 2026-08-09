from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.provider import ModelProvider
from app.schemas.provider_capabilities import ProviderCapabilitiesDetail


class HuggingFaceLocalState(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    OOM = "oom"
    ERROR = "error"


class HuggingFaceLocalModelStatus(BaseModel):
    id: str
    label: str
    revision: str | None = None
    resolved_revision: str | None = None
    status: HuggingFaceLocalState
    status_message: str | None = None
    downloaded: bool = False
    loaded: bool = False
    recommended: bool = False


class HuggingFaceLocalLimits(BaseModel):
    context_window_tokens: int = Field(..., ge=1)
    default_output_tokens: int = Field(..., ge=1)
    max_output_tokens: int = Field(..., ge=1)
    stored_top_alternatives: int = Field(..., ge=1)


class HuggingFaceLocalStatusResponse(BaseModel):
    provider: ModelProvider = Field(default=ModelProvider.HUGGING_FACE)
    label: str
    status: HuggingFaceLocalState
    status_message: str | None = None
    capabilities: ProviderCapabilitiesDetail
    cuda_available: bool
    busy: bool = False
    device: str | None = None
    precision: str | None = None
    torch_version: str | None = None
    transformers_version: str | None = None
    gpu_name: str | None = None
    gpu_total_vram_gb: float | None = None
    gpu_free_vram_gb: float | None = None
    active_model_id: str | None = None
    active_model_label: str | None = None
    active_model_revision: str | None = None
    active_model_resolved_revision: str | None = None
    recommended_model_id: str | None = None
    missing_dependencies: list[str] = Field(default_factory=list)
    limits: HuggingFaceLocalLimits
    models: list[HuggingFaceLocalModelStatus] = Field(default_factory=list)


class HuggingFaceLocalLoadRequest(BaseModel):
    model_id: str = Field(..., min_length=1)


class HuggingFaceLocalDiagnosticsResponse(BaseModel):
    cuda_available: bool
    selected_device: str | None = None
    selected_dtype: str | None = None
    torch_version: str | None = None
    transformers_version: str | None = None
    torch_cuda_runtime: str | None = None
    gpu_name: str | None = None
    gpu_total_vram_gb: float | None = None
    gpu_free_vram_gb: float | None = None
    python_version: str
    platform: str
    disk_free_gb: float | None = None
    missing_dependencies: list[str] = Field(default_factory=list)
