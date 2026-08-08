from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models.provider import ModelProvider


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_token_logprobs: bool
    supports_native_continuation: bool
    minimum_output_tokens: int
    supports_entropy: bool = False
    supports_attention: bool = False
    supports_streaming: bool = False
    supports_branching: bool = False
    supports_continuation: bool = False

    @property
    def supports_logprobs(self) -> bool:
        return self.supports_token_logprobs

    @property
    def supports_exact_continuation(self) -> bool:
        return self.supports_native_continuation


@dataclass(frozen=True)
class DiscoveredModel:
    id: str
    label: str


@dataclass(frozen=True)
class ProviderDiscoveryResult:
    provider_name: ModelProvider
    provider_label: str
    status: str
    status_message: str | None
    recommended_models: list[str]
    capabilities: ProviderCapabilities
    models: list[DiscoveredModel]


@dataclass(frozen=True)
class ProviderGenerationResult:
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    finish_reason: str | None


class LLMProvider(Protocol):
    provider_name: ModelProvider
    provider_label: str
    capabilities: ProviderCapabilities

    def discover_models(self, *, force_refresh: bool = False) -> ProviderDiscoveryResult:
        ...
