from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from app.models.provider import ModelProvider
from app.schemas.provider_capabilities import ProviderCapabilitiesDetail


class ContinuationMode(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


class ApiErrorDetail(BaseModel):
    code: str
    message: str


class AlternativeCandidate(BaseModel):
    node_id: str | None = None
    segment_id: str | None = Field(default=None, min_length=1)
    token: str
    display_token: str | None = None
    token_bytes: list[int] = Field(default_factory=list)
    decoded_contribution: str | None = None
    cumulative_decoded_text: str | None = None
    cumulative_token_ids: list[int] | None = None
    cumulative_log_probability: float | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    raw_probability: float | None = Field(default=None, ge=0, le=1)
    normalized_displayed_probability: float | None = Field(default=None, ge=0, le=1)
    log_probability: float | None = None
    entropy: float | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    token_id: int | None = Field(default=None, ge=0)
    tokenizer_id: int | None = Field(default=None, ge=0)
    rank: int | None = Field(default=None, ge=1)
    text_preview: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    finish_reason: str | None = None
    rationale: str | None = None
    generation_step: int | None = Field(default=None, ge=0)
    continuation_mode: ContinuationMode | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class TokenTrace(BaseModel):
    id: str
    segment_id: str | None = Field(default=None, min_length=1)
    branch_id: str
    parent_node_id: str | None = None
    model: str
    source: Literal["openai", "ollama", "hugging_face", "demo"]
    index: int = Field(..., ge=0)
    position: int = Field(..., ge=0)
    token: str
    display_token: str
    token_bytes: list[int] = Field(default_factory=list)
    decoded_contribution: str
    cumulative_decoded_text: str
    cumulative_token_ids: list[int] | None = None
    cumulative_log_probability: float | None = None
    token_id: int | None = Field(default=None, ge=0)
    tokenizer_id: int | None = Field(default=None, ge=0)
    probability: float | None = Field(default=None, ge=0, le=1)
    raw_probability: float | None = Field(default=None, ge=0, le=1)
    normalized_displayed_probability: float | None = Field(default=None, ge=0, le=1)
    log_probability: float | None = None
    entropy: float | None = Field(default=None, ge=0)
    cumulative_probability: float | None = Field(default=None, ge=0, le=1)
    latency_ms: int = Field(..., ge=0)
    text_preview: str
    context_before: str
    context_after: str
    finish_reason: str | None = None
    alternatives: list[AlternativeCandidate] = Field(default_factory=list)
    generation_step: int = Field(..., ge=0)
    continuation_mode: ContinuationMode = Field(default=ContinuationMode.EXACT)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class TokenTreeNode(BaseModel):
    id: str
    token: str
    display_token: str | None = None
    token_id: int | None = Field(default=None, ge=0)
    tokenizer_id: int | None = Field(default=None, ge=0)
    probability: float | None = Field(default=None, ge=0, le=1)
    raw_probability: float | None = Field(default=None, ge=0, le=1)
    normalized_displayed_probability: float | None = Field(default=None, ge=0, le=1)
    log_probability: float | None = None
    entropy: float | None = Field(default=None, ge=0)
    cumulative_probability: float | None = Field(default=None, ge=0, le=1)
    latency_ms: int = Field(..., ge=0)
    depth: int = Field(..., ge=0)
    rank: int = Field(..., ge=1)
    text_preview: str
    is_selected_path: bool = False
    children: list["TokenTreeNode"] = Field(default_factory=list)


class TreeSummary(BaseModel):
    max_depth: int = Field(..., ge=0)
    branch_width: int = Field(..., ge=1)
    total_nodes: int = Field(..., ge=0)
    explored_paths: int = Field(..., ge=0)
    selected_path_depth: int = Field(..., ge=0)


class PromptInsights(BaseModel):
    detected_intent: str
    focus_terms: list[str] = Field(default_factory=list)
    response_strategy: str
    suggested_follow_ups: list[str] = Field(default_factory=list)


class RequestEcho(BaseModel):
    prompt: str = Field(..., min_length=1)
    provider: ModelProvider | None = None
    model: str
    preset: str = Field(default="general")
    max_tokens: int = Field(..., ge=1, le=4096)
    temperature: float = Field(..., ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    variation: int = Field(default=0, ge=0)
    demo_mode: bool = False


class GenerationStats(BaseModel):
    provider: ModelProvider
    model: str
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    latency_ms: int = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0)
    generated_at: datetime


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    provider: ModelProvider | None = None
    model: str = Field(default="gpt-4.1-mini")
    preset: str = Field(default="general")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    variation: int = Field(default=0, ge=0)
    demo_mode: bool = False

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be blank.")
        return value


class GenerationResponse(BaseModel):
    mode: str = Field(default="live")
    prompt_used: str
    prompt_token_ids: list[int] | None = None
    completion: str
    notes: str
    request: RequestEcho
    insights: PromptInsights
    tokens: list[TokenTrace]
    tree: TokenTreeNode
    tree_summary: TreeSummary
    stats: GenerationStats
    provider_capabilities: ProviderCapabilitiesDetail


class NodeExpansionRequest(BaseModel):
    root_prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        validation_alias=AliasChoices("root_prompt", "prompt"),
        serialization_alias="root_prompt",
    )
    provider: ModelProvider | None = None
    model: str = Field(default="gpt-4.1-mini")
    preset: str = Field(default="general")
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    parent_node_id: str = Field(..., min_length=1)
    parent_token: str = Field(default="")
    assistant_prefix: str = Field(
        default="",
        max_length=50000,
        validation_alias=AliasChoices("assistant_prefix", "parent_text_preview"),
        serialization_alias="assistant_prefix",
    )
    prompt_token_ids: list[int] | None = Field(default=None)
    canonical_prefix_token_ids: list[int] | None = Field(default=None)
    generated_prefix_token_ids: list[int] | None = Field(default=None)
    reconstructed_prompt: str = Field(default="", max_length=60000)
    expected_prompt_length: int | None = Field(default=None, ge=0)
    expected_utf8_length: int | None = Field(default=None, ge=0)
    expected_assistant_prefix_length: int | None = Field(default=None, ge=0)
    expected_assistant_prefix_utf8_length: int | None = Field(default=None, ge=0)
    expected_token_count: int | None = Field(default=None, ge=0)
    selected_token_id: int | None = Field(default=None, ge=0)
    selected_tokenizer_id: int | None = Field(default=None, ge=0)
    model_revision: str | None = Field(default=None, max_length=255)
    tokenizer_identity: str | None = Field(default=None, max_length=500)
    tokenizer_revision: str | None = Field(default=None, max_length=255)
    depth: int = Field(default=0, ge=0)
    cumulative_probability: float = Field(default=1.0, ge=0, le=1)
    variation: int = Field(default=0, ge=0)
    max_children: int = Field(default=4, ge=1, le=20)
    demo_mode: bool = False

    @field_validator("root_prompt")
    @classmethod
    def validate_expansion_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be blank.")
        return value


class ContinueGenerationRequest(NodeExpansionRequest):
    cached_segment_id: str | None = Field(default=None, min_length=1)
    cached_token_index: int | None = Field(default=None, ge=0)


class NodeExpansionCandidate(BaseModel):
    id: str
    segment_id: str | None = Field(default=None, min_length=1)
    branch_id: str
    parent_node_id: str
    model: str
    source: Literal["openai", "ollama", "hugging_face", "demo"]
    token: str
    display_token: str
    token_bytes: list[int] = Field(default_factory=list)
    decoded_contribution: str
    cumulative_decoded_text: str
    cumulative_token_ids: list[int] | None = None
    cumulative_log_probability: float
    token_id: int | None = Field(default=None, ge=0)
    tokenizer_id: int | None = Field(default=None, ge=0)
    probability: float = Field(..., ge=0, le=1)
    raw_probability: float = Field(..., ge=0, le=1)
    normalized_displayed_probability: float = Field(..., ge=0, le=1)
    log_probability: float
    entropy: float = Field(..., ge=0)
    cumulative_probability: float = Field(..., ge=0, le=1)
    latency_ms: int = Field(..., ge=0)
    depth: int = Field(..., ge=0)
    rank: int = Field(..., ge=1)
    text_preview: str
    context_before: str
    context_after: str
    finish_reason: str | None = None
    rationale: str | None = None
    generation_step: int = Field(..., ge=0)
    continuation_mode: ContinuationMode = Field(default=ContinuationMode.EXACT)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class NodeExpansionResponse(BaseModel):
    mode: str = Field(default="live")
    parent_node_id: str
    children: list[NodeExpansionCandidate] = Field(default_factory=list)
    entropy: float = Field(..., ge=0)
    expanded_at: datetime
    notes: str


class ContinueGenerationResponse(NodeExpansionResponse):
    action: Literal["reveal_cached", "new_provider_segment"]
    continuation_mode: ContinuationMode
    provider_capabilities: ProviderCapabilitiesDetail
    segment_id: str | None = None
    revealed_count: int = Field(..., ge=0)
    cached_token_count: int = Field(..., ge=0)
    remaining_cached_tokens: int = Field(..., ge=0)


TokenTreeNode.model_rebuild()
