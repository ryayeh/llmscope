from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter, time
from urllib import error, request

from app.models.provider import ModelProvider
from app.providers.base import (
    DiscoveredModel,
    LLMProvider,
    ProviderCapabilities,
    ProviderDiscoveryResult,
    ProviderGenerationResult,
)

OLLAMA_RECOMMENDED_MODELS = [
    "qwen2.5:3b",
    "phi3",
    "gemma3",
    "llama3.2",
]
OLLAMA_OFFLINE_MESSAGE = (
    "Ollama is not running.\n\n"
    "Install from https://ollama.com/\n\n"
    "Then run:\n\n"
    "ollama serve"
)
OLLAMA_EMPTY_MESSAGE = (
    "No local Ollama models found.\n\n"
    "Example:\n\n"
    "ollama pull qwen2.5:3b"
)
OLLAMA_TOKEN_PATTERN = re.compile(r"\s*\S+|\s+")


@dataclass(frozen=True)
class OllamaToken:
    raw_token: str
    display_token: str
    decoded_contribution: str
    token_bytes: list[int]


class OllamaProvider(LLMProvider):
    provider_name = ModelProvider.OLLAMA
    provider_label = "Ollama"
    capabilities = ProviderCapabilities(
        supports_token_logprobs=False,
        supports_native_continuation=False,
        minimum_output_tokens=1,
        supports_entropy=False,
        supports_attention=False,
        supports_streaming=True,
        supports_branching=False,
        supports_continuation=False,
    )

    def __init__(self, base_url: str, *, cache_ttl_seconds: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cached_discovery: ProviderDiscoveryResult | None = None
        self._cached_at: float = 0.0

    def warm_cache(self) -> None:
        self.discover_models(force_refresh=True)

    def discover_models(self, *, force_refresh: bool = False) -> ProviderDiscoveryResult:
        now = time()
        if (
            not force_refresh
            and self._cached_discovery is not None
            and (now - self._cached_at) < self._cache_ttl_seconds
        ):
            return self._cached_discovery

        try:
            payload = self._get_json("/api/tags")
        except error.URLError:
            discovery = ProviderDiscoveryResult(
                provider_name=self.provider_name,
                provider_label=self.provider_label,
                status="offline",
                status_message=OLLAMA_OFFLINE_MESSAGE,
                recommended_models=list(OLLAMA_RECOMMENDED_MODELS),
                capabilities=self.capabilities,
                models=[],
            )
        else:
            models = [
                DiscoveredModel(
                    id=str(item.get("name", "")).strip(),
                    label=str(item.get("name", "")).strip(),
                )
                for item in payload.get("models", [])
                if str(item.get("name", "")).strip()
            ]

            if not models:
                discovery = ProviderDiscoveryResult(
                    provider_name=self.provider_name,
                    provider_label=self.provider_label,
                    status="empty",
                    status_message=OLLAMA_EMPTY_MESSAGE,
                    recommended_models=list(OLLAMA_RECOMMENDED_MODELS),
                    capabilities=self.capabilities,
                    models=[],
                )
            else:
                discovery = ProviderDiscoveryResult(
                    provider_name=self.provider_name,
                    provider_label=self.provider_label,
                    status="ready",
                    status_message=None,
                    recommended_models=list(OLLAMA_RECOMMENDED_MODELS),
                    capabilities=self.capabilities,
                    models=models,
                )

        self._cached_discovery = discovery
        self._cached_at = now
        return discovery

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> ProviderGenerationResult:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        start_time = perf_counter()
        payload = self._post_json("/api/generate", body)
        measured_latency_ms = max(1, int((perf_counter() - start_time) * 1000))
        total_duration_ns = int(payload.get("total_duration") or 0)
        latency_ms = (
            max(1, int(round(total_duration_ns / 1_000_000)))
            if total_duration_ns > 0
            else measured_latency_ms
        )
        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        completion_tokens = int(payload.get("eval_count") or 0)
        total_tokens = prompt_tokens + completion_tokens

        return ProviderGenerationResult(
            completion=str(payload.get("response", "")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=str(payload.get("done_reason") or "") or None,
        )

    def tokenize(self, text: str) -> list[OllamaToken]:
        tokens = OLLAMA_TOKEN_PATTERN.findall(text)
        return [
            OllamaToken(
                raw_token=token,
                display_token=self._display_token(token),
                decoded_contribution=token,
                token_bytes=list(token.encode("utf-8")),
            )
            for token in tokens
        ]

    def _get_json(self, path: str) -> dict:
        request_url = f"{self._base_url}{path}"
        http_request = request.Request(
            request_url,
            method="GET",
            headers={"Accept": "application/json"},
        )
        with request.urlopen(http_request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        request_url = f"{self._base_url}{path}"
        encoded_body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            request_url,
            data=encoded_body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(http_request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def _display_token(self, token: str) -> str:
        visible = token.replace("\t", "\u21E5").replace("\n", "\u21B5\n")
        if visible.startswith(" "):
            visible = visible.replace(" ", "\u2420", 1)
        return visible or "\u2205"
