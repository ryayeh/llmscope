from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.provider import ModelProvider


class AlternativeCandidate(BaseModel):
    token: str
    probability: float = Field(..., ge=0, le=1)
    log_probability: float | None = None
    entropy: float | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    token_id: int | None = Field(default=None, ge=0)
    text_preview: str | None = None
    rationale: str | None = None


class TokenTrace(BaseModel):
    id: str
    token: str
    token_id: int = Field(..., ge=0)
    probability: float = Field(..., ge=0, le=1)
    log_probability: float
    entropy: float = Field(..., ge=0)
    cumulative_probability: float = Field(..., ge=0, le=1)
    latency_ms: int = Field(..., ge=0)
    position: int = Field(..., ge=0)
    text_preview: str
    alternatives: list[AlternativeCandidate] = Field(default_factory=list)


class TokenTreeNode(BaseModel):
    id: str
    token: str
    token_id: int = Field(..., ge=0)
    probability: float = Field(..., ge=0, le=1)
    log_probability: float
    entropy: float = Field(..., ge=0)
    cumulative_probability: float = Field(..., ge=0, le=1)
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
    model: str
    preset: str = Field(default="general")
    max_tokens: int = Field(..., ge=1, le=4096)
    temperature: float = Field(..., ge=0, le=2)
    variation: int = Field(default=0, ge=0)


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
    model: str = Field(default="gpt-4.1-mini")
    preset: str = Field(default="general")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    variation: int = Field(default=0, ge=0)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        prompt = value.strip()
        if not prompt:
            raise ValueError("Prompt must not be blank.")
        return prompt


class GenerationResponse(BaseModel):
    mode: str = Field(default="mock")
    prompt_used: str
    completion: str
    notes: str
    request: RequestEcho
    insights: PromptInsights
    tokens: list[TokenTrace]
    tree: TokenTreeNode
    tree_summary: TreeSummary
    stats: GenerationStats


class NodeExpansionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="gpt-4.1-mini")
    preset: str = Field(default="general")
    temperature: float = Field(default=0.7, ge=0, le=2)
    parent_node_id: str = Field(..., min_length=1)
    parent_token: str = Field(default="")
    parent_text_preview: str = Field(default="")
    depth: int = Field(default=0, ge=0)
    cumulative_probability: float = Field(default=1.0, ge=0, le=1)
    variation: int = Field(default=0, ge=0)
    max_children: int = Field(default=4, ge=1, le=8)

    @field_validator("prompt")
    @classmethod
    def validate_expansion_prompt(cls, value: str) -> str:
        prompt = value.strip()
        if not prompt:
            raise ValueError("Prompt must not be blank.")
        return prompt


class NodeExpansionCandidate(BaseModel):
    id: str
    parent_node_id: str
    token: str
    token_id: int = Field(..., ge=0)
    probability: float = Field(..., ge=0, le=1)
    log_probability: float
    entropy: float = Field(..., ge=0)
    cumulative_probability: float = Field(..., ge=0, le=1)
    latency_ms: int = Field(..., ge=0)
    depth: int = Field(..., ge=0)
    rank: int = Field(..., ge=1)
    text_preview: str
    rationale: str | None = None


class NodeExpansionResponse(BaseModel):
    mode: str = Field(default="inferred")
    parent_node_id: str
    children: list[NodeExpansionCandidate] = Field(default_factory=list)
    entropy: float = Field(..., ge=0)
    expanded_at: datetime
    notes: str


TokenTreeNode.model_rebuild()
