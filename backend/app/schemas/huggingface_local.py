from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.provider import ModelProvider
from app.schemas.generation import CanonicalPromptToken, CanonicalTokenSourceCategory
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


class HuggingFaceAttentionAggregationMode(str, Enum):
    SINGLE_HEAD = "single_head"
    AVERAGE_HEADS = "average_heads"
    MAX_HEADS = "max_heads"


class HuggingFaceAttentionAnalysisMode(str, Enum):
    PREDICTION = "prediction"
    REPRESENTATION = "representation"


class HuggingFaceAttentionSequenceScope(str, Enum):
    PROMPT = "prompt"
    GENERATED = "generated"


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
    active_model_num_hidden_layers: int | None = Field(default=None, ge=1)
    active_model_num_attention_heads: int | None = Field(default=None, ge=1)
    active_model_attention_implementation: str | None = None
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


class HuggingFaceAttentionRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    model_revision: str | None = Field(default=None, max_length=255)
    tokenizer_identity: str | None = Field(default=None, max_length=500)
    tokenizer_revision: str | None = Field(default=None, max_length=255)
    prompt_token_ids: list[int] = Field(default_factory=list)
    prompt_tokens: list[CanonicalPromptToken] = Field(default_factory=list)
    generated_token_ids: list[int] = Field(default_factory=list)
    selected_generated_token_index: int = Field(..., ge=0)
    selected_layer: int = Field(..., ge=0)
    selected_head: int | None = Field(default=None, ge=0)
    analysis_mode: HuggingFaceAttentionAnalysisMode = Field(
        default=HuggingFaceAttentionAnalysisMode.PREDICTION
    )
    aggregation_mode: HuggingFaceAttentionAggregationMode = Field(
        default=HuggingFaceAttentionAggregationMode.AVERAGE_HEADS
    )
    comparison_layers: list[int] = Field(default_factory=list, max_length=32)
    journey_layers: list[int] = Field(default_factory=list, max_length=256)
    journey_max_rows: int = Field(default=5, ge=1, le=8)
    max_connections: int = Field(default=8, ge=1, le=512)
    max_context_tokens: int = Field(default=256, ge=1, le=512)
    allow_truncated_recompute: bool = False


class HuggingFaceAttentionTokenInfo(BaseModel):
    token_id: int = Field(..., ge=0)
    raw_token: str
    display_token: str
    decoded_contribution: str
    token_bytes: list[int] = Field(default_factory=list)
    full_position: int = Field(..., ge=0)
    analyzed_position: int = Field(..., ge=0)
    sequence_scope: HuggingFaceAttentionSequenceScope
    source_category: CanonicalTokenSourceCategory
    source_label: str
    special_token: bool = False
    generated_token_index: int | None = Field(default=None, ge=0)
    attention_weight: float | None = Field(default=None, ge=0)
    is_query: bool = False
    is_selected_token: bool = False


class HuggingFaceAttentionSource(BaseModel):
    token_id: int = Field(..., ge=0)
    raw_token: str
    display_token: str
    decoded_contribution: str
    token_bytes: list[int] = Field(default_factory=list)
    full_position: int = Field(..., ge=0)
    analyzed_position: int = Field(..., ge=0)
    sequence_scope: HuggingFaceAttentionSequenceScope
    source_category: CanonicalTokenSourceCategory
    source_label: str
    special_token: bool = False
    generated_token_index: int | None = Field(default=None, ge=0)
    attention_weight: float = Field(..., ge=0)
    rank: int = Field(..., ge=1)


class HuggingFaceAttentionCategoryBreakdown(BaseModel):
    input_context: float = Field(..., ge=0)
    earlier_output: float = Field(..., ge=0)
    system_message: float = Field(..., ge=0)
    user_prompt: float = Field(..., ge=0)
    assistant_prefix: float = Field(..., ge=0)
    template_control: float = Field(..., ge=0)
    exclusive_total: float = Field(..., ge=0)


class HuggingFaceAttentionLayerSummary(BaseModel):
    layer_index: int = Field(..., ge=0)
    depth_ratio: float = Field(..., ge=0, le=1)
    top_meaningful_source: HuggingFaceAttentionSource | None = None
    category_breakdown: HuggingFaceAttentionCategoryBreakdown
    attention_mass_sum: float = Field(..., ge=0)
    top_n_coverage: float = Field(..., ge=0)


class HuggingFaceAttentionJourneyRow(BaseModel):
    row_id: str
    row_kind: str
    label: str
    included_reason: str
    source: HuggingFaceAttentionSource | None = None
    weights: list[float] = Field(default_factory=list)
    max_weight: float = Field(..., ge=0)


class HuggingFaceAttentionLayerJourney(BaseModel):
    layers: list[int] = Field(default_factory=list)
    sampled: bool = False
    scale_max: float = Field(..., ge=0)
    rows: list[HuggingFaceAttentionJourneyRow] = Field(default_factory=list)


class HuggingFaceAttentionResponse(BaseModel):
    provider: ModelProvider = Field(default=ModelProvider.HUGGING_FACE)
    model_id: str
    model_revision: str | None = None
    tokenizer_identity: str | None = None
    tokenizer_revision: str | None = None
    analysis_mode: HuggingFaceAttentionAnalysisMode
    selected_token: HuggingFaceAttentionTokenInfo
    query_token: HuggingFaceAttentionTokenInfo
    analyzed_tokens: list[HuggingFaceAttentionTokenInfo] = Field(default_factory=list)
    sources: list[HuggingFaceAttentionSource] = Field(default_factory=list)
    all_sources: list[HuggingFaceAttentionSource] = Field(default_factory=list)
    selected_layer: int = Field(..., ge=0)
    selected_head: int | None = Field(default=None, ge=0)
    aggregation_mode: HuggingFaceAttentionAggregationMode
    attention_implementation_used: str
    num_layers: int = Field(..., ge=1)
    num_query_heads: int = Field(..., ge=1)
    selected_token_position: int = Field(..., ge=0)
    query_position: int = Field(..., ge=0)
    selected_token_id: int = Field(..., ge=0)
    query_token_id: int = Field(..., ge=0)
    prompt_token_count: int = Field(..., ge=0)
    generated_token_index: int = Field(..., ge=0)
    sequence_length: int = Field(..., ge=1)
    layer_index: int = Field(..., ge=0)
    head_index: int | None = Field(default=None, ge=0)
    average_heads: bool = False
    source_positions: list[int] = Field(default_factory=list)
    attention_weights: list[float] = Field(default_factory=list)
    attention_mass_sum: float = Field(..., ge=0)
    top_n_coverage: float = Field(default=0, ge=0)
    truncated_context: bool = False
    context_truncated: bool = False
    original_full_context_length: int = Field(..., ge=1)
    analyzed_context_length: int = Field(..., ge=1)
    category_breakdown: HuggingFaceAttentionCategoryBreakdown
    comparison_layers: list[HuggingFaceAttentionLayerSummary] = Field(default_factory=list)
    layer_journey: HuggingFaceAttentionLayerJourney | None = None
