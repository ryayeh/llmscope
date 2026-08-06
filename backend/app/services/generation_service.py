from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
import zlib

from openai import OpenAI

from app.core.config import get_settings
from app.core.errors import LLMScopeError
from app.models.provider import ModelProvider
from app.schemas.generation import (
    AlternativeCandidate,
    ContinueGenerationRequest,
    ContinueGenerationResponse,
    GenerationRequest,
    GenerationResponse,
    GenerationStats,
    NodeExpansionCandidate,
    NodeExpansionRequest,
    NodeExpansionResponse,
    ProviderCapabilitiesDetail,
    PromptInsights,
    RequestEcho,
    TokenTrace,
    TokenTreeNode,
    TreeSummary,
)
from app.schemas.model_catalog import ModelCatalogResponse, ModelOption, PresetOption

logger = logging.getLogger(__name__)

IMPORTANT_SHORT_WORDS = {"ai", "api", "app", "bug", "db", "llm", "ml", "sdk", "ui", "ux"}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "give",
    "help",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "make",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "plain",
    "please",
    "return",
    "returns",
    "sentence",
    "show",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "tell",
    "to",
    "use",
    "want",
    "with",
    "write",
    "you",
    "your",
}

DEFAULT_TOP_LOGPROBS = 6
MIN_PROBABILITY = 0.000001
DEMO_TOKEN_PATTERN = re.compile(r"\s*\S+")

PRESET_OPTIONS = [
    PresetOption(id="general", label="General"),
    PresetOption(id="reasoning", label="Reasoning"),
    PresetOption(id="coding", label="Code"),
    PresetOption(id="coach", label="Coach"),
]

PRESET_INSTRUCTIONS = {
    "general": "Answer directly and keep the wording clean and practical.",
    "reasoning": "Show the logic in a compact way and make trade-offs explicit.",
    "coding": "Prioritize technical clarity, implementation detail, and concrete examples when relevant.",
    "coach": "Use a supportive tone and give the user a usable benchmark, next step, or adjustment.",
}

CONTINUATION_TONE_HINTS = {
    "general": "Match the concise, practical tone already established by the assistant prefix.",
    "reasoning": "Match the compact reasoning style already established by the assistant prefix.",
    "coding": "Match the technical wording and implementation-focused style already established by the assistant prefix.",
    "coach": "Match the supportive coaching tone already established by the assistant prefix.",
}

RESPONSE_STYLE_HINTS = [
    "Lead with the answer, then add one short supporting sentence.",
    "Use a slightly different phrasing pattern from the last attempt and avoid canned wording.",
    "Keep the structure tight, but vary the opening and emphasis naturally.",
    "Prefer a crisp benchmark-first answer when the user is asking for a range or recommendation.",
]

MODEL_OPTIONS = [
    ModelOption(
        id="gpt-4o-mini",
        label="GPT-4o mini",
        provider=ModelProvider.OPENAI,
        group="OpenAI",
    ),
    ModelOption(
        id="gpt-4.1-mini",
        label="GPT-4.1 mini",
        provider=ModelProvider.OPENAI,
        group="OpenAI",
    ),
    ModelOption(
        id="gpt-4o",
        label="GPT-4o",
        provider=ModelProvider.OPENAI,
        group="OpenAI",
    ),
    ModelOption(
        id="gpt-4.1",
        label="GPT-4.1",
        provider=ModelProvider.OPENAI,
        group="OpenAI",
    ),
]

MODEL_PRICING_USD_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
}

MODEL_OPTION_MAP = {option.id: option for option in MODEL_OPTIONS}
PRESET_OPTION_MAP = {option.id: option for option in PRESET_OPTIONS}


@dataclass(frozen=True)
class ContinuationContext:
    root_prompt: str
    assistant_prefix: str
    reconstructed_prompt: str
    character_length: int
    utf8_length: int
    token_count: int


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_assistant_prefill: bool
    supports_token_logprobs: bool
    minimum_output_tokens: int


@dataclass
class GenerationSegment:
    segment_id: str
    source_node_id: str
    request_prompt: str
    context_prefix: str
    model: str
    preset: str
    temperature: float
    top_p: float
    variation: int
    demo_mode: bool
    mode: str
    tokens: list[TokenTrace]
    revealed_count: int = 0
    finish_reason: str | None = None


class GenerationService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: OpenAI | None = None
        self._segments_by_id: dict[str, GenerationSegment] = {}

    def list_models(self) -> ModelCatalogResponse:
        return ModelCatalogResponse(
            default_model="gpt-4.1-mini",
            default_preset="general",
            models=MODEL_OPTIONS,
            presets=PRESET_OPTIONS,
        )

    def build_response(self, request: GenerationRequest) -> GenerationResponse:
        prompt = request.prompt
        keywords = self._extract_keywords(prompt)
        intent, strategy = self._detect_intent(prompt.lower())
        preset = request.preset if request.preset in PRESET_OPTION_MAP else "general"
        model_option = MODEL_OPTION_MAP.get(
            request.model,
            ModelOption(
                id=request.model,
                label=request.model,
                provider=ModelProvider.OPENAI,
                group="Custom",
            ),
        )

        if request.demo_mode:
            completion, response_mode, usage, latency_ms, tokens = self._build_demo_generation(
                request=request,
                prompt=prompt,
                keywords=keywords,
                intent=intent,
                preset=preset,
            )
        else:
            completion, response_mode, usage, latency_ms, tokens = self._build_live_generation(
                request=request,
                prompt=prompt,
                intent=intent,
                preset=preset,
            )

        tree, tree_summary = self._build_tree(tokens)
        prompt_tokens = self._usage_value(usage, "input_tokens") or self._estimate_prompt_tokens(prompt, keywords)
        completion_tokens = self._usage_value(usage, "output_tokens") or len(tokens)
        total_tokens = self._usage_value(usage, "total_tokens") or (prompt_tokens + completion_tokens)
        effective_latency_ms = latency_ms or max(1, len(tokens) * 30)

        stats = GenerationStats(
            provider=model_option.provider,
            model=request.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=effective_latency_ms,
            estimated_cost_usd=self._estimate_cost(
                request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            generated_at=datetime.now(timezone.utc),
        )

        insights = PromptInsights(
            detected_intent=intent.replace("_", " "),
            focus_terms=keywords[:5],
            response_strategy=f"{PRESET_INSTRUCTIONS[preset]} {strategy}",
            suggested_follow_ups=self._build_follow_ups(keywords, intent),
        )

        notes = (
            "Demo data. Token alternatives are synthetic because Demo Mode is enabled."
            if request.demo_mode
            else "Live provider response with per-token logprobs."
        )

        return GenerationResponse(
            mode=response_mode,
            prompt_used=prompt,
            completion=completion,
            notes=notes,
            request=RequestEcho(
                prompt=prompt,
                model=request.model,
                preset=preset,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                variation=request.variation,
                demo_mode=request.demo_mode,
            ),
            insights=insights,
            tokens=tokens,
            tree=tree,
            tree_summary=tree_summary,
            stats=stats,
        )

    def expand_node(self, request: NodeExpansionRequest) -> NodeExpansionResponse:
        context = self._build_continuation_context(request)
        prompt = context.root_prompt
        preset = request.preset if request.preset in PRESET_OPTION_MAP else "general"
        assistant_prefix = context.assistant_prefix
        intent, _ = self._detect_intent(prompt.lower())
        capabilities = self._provider_capabilities_for_model(request.model)

        if self._settings.app_env.lower() == "development":
            logger.debug(
                "CONTINUATION REQUEST %s",
                {
                    "selected_node_id": request.parent_node_id,
                    "selected_token": request.parent_token,
                    "assistant_prefix": assistant_prefix,
                    "reconstructed_prompt": context.reconstructed_prompt,
                },
            )

        if request.demo_mode:
            children, entropy, response_mode = self._build_demo_expansion(
                request=request,
                prompt=prompt,
                assistant_prefix=assistant_prefix,
                intent=intent,
            )
            notes = "Demo data. Token alternatives are synthetic because Demo Mode is enabled."
        else:
            continuation_mode = "native_prefill" if capabilities.supports_assistant_prefill else "approximate"
            if capabilities.supports_assistant_prefill:
                steps, _, latency_ms = self._request_live_steps(
                    request=request,
                    prompt=prompt,
                    preset=preset,
                    intent=intent,
                    assistant_prefix=assistant_prefix,
                    branch_id=request.parent_node_id,
                    parent_node_id=request.parent_node_id,
                    max_output_tokens=1,
                    top_logprobs=request.max_children,
                )
            else:
                steps, _, latency_ms = self._request_approximate_steps(
                    request=request,
                    prompt=prompt,
                    assistant_prefix=assistant_prefix,
                    branch_id=request.parent_node_id,
                    parent_node_id=request.parent_node_id,
                    max_output_tokens=1,
                    top_logprobs=request.max_children,
                )
            if not steps:
                self._raise_logprobs_unavailable()

            source_trace = steps[0]
            latency_value = latency_ms or source_trace.latency_ms
            children = self._build_expansion_children_from_trace(
                request=request,
                trace=source_trace,
                latency_ms=latency_value,
            )
            children = self._apply_continuation_metadata(
                children=children,
                capabilities=capabilities,
                continuation_mode=continuation_mode,
            )
            if self._settings.app_env.lower() == "development":
                logger.debug(
                    "CONTINUATION RESPONSE %s",
                    {
                        "selected_node_id": request.parent_node_id,
                        "selected_token": request.parent_token,
                        "first_returned_token": children[0].token if children else None,
                        "first_returned_probability": children[0].probability if children else None,
                        "first_returned_alternatives": [
                            {
                                "token": child.token,
                                "probability": child.probability,
                                "rank": child.rank,
                            }
                            for child in children[1:]
                        ],
                    },
                )
            entropy = source_trace.entropy
            response_mode = "live"
            notes = (
                "Approximate next-token distribution regenerated from the selected branch context."
                if continuation_mode == "approximate"
                else "Next-token distribution returned by the provider for this exact branch context."
            )

        return NodeExpansionResponse(
            mode=response_mode,
            parent_node_id=request.parent_node_id,
            children=children,
            entropy=entropy,
            expanded_at=datetime.now(timezone.utc),
            notes=notes,
        )

    def continue_node(self, request: ContinueGenerationRequest) -> ContinueGenerationResponse:
        context = self._build_continuation_context(request)
        capabilities = self._provider_capabilities_for_model(request.model)
        cached_segment = (
            self._segments_by_id.get(request.cached_segment_id)
            if request.cached_segment_id
            else None
        )

        if cached_segment and request.cached_token_index is not None:
            if request.cached_token_index < len(cached_segment.tokens):
                return self._reveal_next_cached_token(
                    request=request,
                    context=context,
                    capabilities=capabilities,
                    segment=cached_segment,
                    token_index=request.cached_token_index,
                )

        if capabilities.supports_assistant_prefill:
            return self._generate_native_prefill_segment(
                request=request,
                context=context,
                capabilities=capabilities,
            )

        return self._generate_approximate_segment(
            request=request,
            context=context,
            capabilities=capabilities,
        )

    def _build_live_generation(
        self,
        *,
        request: GenerationRequest,
        prompt: str,
        intent: str,
        preset: str,
    ) -> tuple[str, str, Any | None, int, list[TokenTrace]]:
        steps, usage, latency_ms = self._request_live_steps(
            request=request,
            prompt=prompt,
            preset=preset,
            intent=intent,
            assistant_prefix="",
            branch_id="main",
            parent_node_id="root",
            max_output_tokens=max(16, min(request.max_tokens, 1024)),
            top_logprobs=DEFAULT_TOP_LOGPROBS,
        )

        completion = "".join(step.token for step in steps)
        return completion, "live", usage, latency_ms, steps

    def _provider_capabilities_for_model(self, model: str) -> ProviderCapabilities:
        provider = MODEL_OPTION_MAP.get(
            model,
            ModelOption(
                id=model,
                label=model,
                provider=ModelProvider.OPENAI,
                group="Custom",
            ),
        ).provider

        if provider == ModelProvider.OPENAI:
            return ProviderCapabilities(
                supports_assistant_prefill=False,
                supports_token_logprobs=True,
                minimum_output_tokens=16,
            )

        return ProviderCapabilities(
            supports_assistant_prefill=False,
            supports_token_logprobs=True,
            minimum_output_tokens=1,
        )

    def _serialize_provider_capabilities(
        self,
        capabilities: ProviderCapabilities,
    ) -> ProviderCapabilitiesDetail:
        return ProviderCapabilitiesDetail(
            supports_assistant_prefill=capabilities.supports_assistant_prefill,
            supports_token_logprobs=capabilities.supports_token_logprobs,
            minimum_output_tokens=capabilities.minimum_output_tokens,
        )

    def _make_segment_id(self, *, source_node_id: str, assistant_prefix: str, model: str) -> str:
        checksum = zlib.adler32(f"{model}:{source_node_id}:{assistant_prefix}".encode("utf-8"))
        return f"segment:{source_node_id}:{checksum:08x}"

    def _store_generation_segment(
        self,
        *,
        request: ContinueGenerationRequest,
        context: ContinuationContext,
        steps: list[TokenTrace],
        revealed_count: int,
        mode: str,
    ) -> GenerationSegment:
        segment_id = self._make_segment_id(
            source_node_id=request.parent_node_id,
            assistant_prefix=context.assistant_prefix,
            model=request.model,
        )
        segment = GenerationSegment(
            segment_id=segment_id,
            source_node_id=request.parent_node_id,
            request_prompt=context.root_prompt,
            context_prefix=context.assistant_prefix,
            model=request.model,
            preset=request.preset,
            temperature=request.temperature,
            top_p=request.top_p,
            variation=request.variation,
            demo_mode=request.demo_mode,
            mode=mode,
            tokens=steps,
            revealed_count=revealed_count,
            finish_reason=steps[-1].finish_reason if steps else None,
        )
        self._segments_by_id[segment_id] = segment
        return segment

    def _apply_continuation_metadata(
        self,
        *,
        children: list[NodeExpansionCandidate],
        capabilities: ProviderCapabilities,
        continuation_mode: str,
        segment_id: str | None = None,
        next_cached_token_index: int | None = None,
        cached_token_count: int = 0,
    ) -> list[NodeExpansionCandidate]:
        is_exact = continuation_mode in {"cached_exact", "native_prefill"}
        updated_children: list[NodeExpansionCandidate] = []
        for child in children:
            metadata = {
                **child.metadata,
                "supports_assistant_prefill": capabilities.supports_assistant_prefill,
                "supports_token_logprobs": capabilities.supports_token_logprobs,
                "minimum_output_tokens": capabilities.minimum_output_tokens,
                "continuation_mode": continuation_mode,
                "continuation_mode_label": "Exact" if is_exact else "Approximate",
                "continuation_mode_is_exact": is_exact,
            }
            if not is_exact:
                metadata["continuation_mode_tooltip"] = (
                    "Probabilities are computed from a regenerated continuation because exact "
                    "assistant continuation is unavailable for this provider."
                )
            if child.rank == 1:
                metadata.update(
                    {
                        "cached_segment_id": segment_id,
                        "cached_token_count": cached_token_count,
                        "cached_tokens_remaining": max(
                            0,
                            cached_token_count - (next_cached_token_index or cached_token_count),
                        ),
                        "next_cached_token_index": next_cached_token_index,
                    }
                )
            updated_children.append(child.model_copy(update={"metadata": metadata}))
        return updated_children

    def _log_continue_action(
        self,
        *,
        action: str,
        mode: str,
        request: ContinueGenerationRequest,
        capabilities: ProviderCapabilities,
        segment_id: str | None,
        revealed_count: int,
        cached_token_count: int,
        source_context: str,
        raw_first_token_if_requested: str | None,
        provider_call: bool,
    ) -> None:
        if self._settings.app_env.lower() != "development":
            return

        logger.debug(
            "CONTINUE ACTION %s",
            {
                "action": action,
                "mode": mode,
                "provider": str(MODEL_OPTION_MAP.get(request.model, MODEL_OPTIONS[1]).provider),
                "model": request.model,
                "supports_assistant_prefill": capabilities.supports_assistant_prefill,
                "segment_id": segment_id,
                "revealed_count": revealed_count,
                "cached_token_count": cached_token_count,
                "provider_call": provider_call,
                "source_context": source_context,
                "raw_first_token_if_requested": raw_first_token_if_requested,
            },
        )

    def _reveal_next_cached_token(
        self,
        *,
        request: ContinueGenerationRequest,
        context: ContinuationContext,
        capabilities: ProviderCapabilities,
        segment: GenerationSegment,
        token_index: int,
    ) -> ContinueGenerationResponse:
        if token_index >= len(segment.tokens):
            raise LLMScopeError(
                code="CACHED_SEGMENT_EXHAUSTED",
                message="The cached continuation segment is exhausted.",
                status_code=409,
            )

        trace = segment.tokens[token_index]
        if trace.context_before != context.assistant_prefix:
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The cached continuation segment no longer matches the selected node context.",
                status_code=409,
            )

        children = self._build_expansion_children_from_trace(
            request=request,
            trace=trace,
            latency_ms=trace.latency_ms,
        )
        next_cached_token_index = token_index + 1 if token_index + 1 < len(segment.tokens) else None
        continuation_mode = "cached_exact" if segment.mode == "exact" else "approximate"
        children = self._apply_continuation_metadata(
            children=children,
            capabilities=capabilities,
            continuation_mode=continuation_mode,
            segment_id=segment.segment_id,
            next_cached_token_index=next_cached_token_index,
            cached_token_count=len(segment.tokens),
        )

        segment.revealed_count = max(segment.revealed_count, token_index + 1)
        self._log_continue_action(
            action="reveal_cached",
            mode=continuation_mode,
            request=request,
            capabilities=capabilities,
            segment_id=segment.segment_id,
            revealed_count=segment.revealed_count,
            cached_token_count=len(segment.tokens),
            source_context=context.assistant_prefix,
            raw_first_token_if_requested=None,
            provider_call=False,
        )
        return ContinueGenerationResponse(
            mode="live",
            action="reveal_cached",
            continuation_mode=continuation_mode,
            parent_node_id=request.parent_node_id,
            children=children,
            entropy=trace.entropy,
            expanded_at=datetime.now(timezone.utc),
            notes=(
                "Revealed the next cached token from an approximate continuation segment."
                if continuation_mode == "approximate"
                else "Revealed the next cached token from the existing provider segment."
            ),
            provider_capabilities=self._serialize_provider_capabilities(capabilities),
            segment_id=segment.segment_id,
            revealed_count=segment.revealed_count,
            cached_token_count=len(segment.tokens),
            remaining_cached_tokens=max(0, len(segment.tokens) - segment.revealed_count),
        )

    def _generate_native_prefill_segment(
        self,
        *,
        request: ContinueGenerationRequest,
        context: ContinuationContext,
        capabilities: ProviderCapabilities,
    ) -> ContinueGenerationResponse:
        prompt = context.root_prompt
        preset = request.preset if request.preset in PRESET_OPTION_MAP else "general"
        assistant_prefix = context.assistant_prefix
        intent, _ = self._detect_intent(prompt.lower())
        steps, _, latency_ms = self._request_live_steps(
            request=request,
            prompt=prompt,
            preset=preset,
            intent=intent,
            assistant_prefix=assistant_prefix,
            branch_id=request.parent_node_id,
            parent_node_id=request.parent_node_id,
            max_output_tokens=1,
            top_logprobs=request.max_children,
        )
        if not steps:
            self._raise_logprobs_unavailable()

        segment = self._store_generation_segment(
            request=request,
            context=context,
            steps=steps,
            revealed_count=1,
            mode="exact",
        )
        source_trace = steps[0]
        latency_value = latency_ms or source_trace.latency_ms
        children = self._build_expansion_children_from_trace(
            request=request,
            trace=source_trace,
            latency_ms=latency_value,
        )
        next_cached_token_index = 1 if len(steps) > 1 else None
        children = self._apply_continuation_metadata(
            children=children,
            capabilities=capabilities,
            continuation_mode="native_prefill",
            segment_id=segment.segment_id,
            next_cached_token_index=next_cached_token_index,
            cached_token_count=len(steps),
        )

        self._log_continue_action(
            action="new_provider_segment",
            mode="native_prefill",
            request=request,
            capabilities=capabilities,
            segment_id=segment.segment_id,
            revealed_count=segment.revealed_count,
            cached_token_count=len(segment.tokens),
            source_context=context.assistant_prefix,
            raw_first_token_if_requested=source_trace.token,
            provider_call=True,
        )
        return ContinueGenerationResponse(
            mode="live",
            action="new_provider_segment",
            continuation_mode="native_prefill",
            parent_node_id=request.parent_node_id,
            children=children,
            entropy=source_trace.entropy,
            expanded_at=datetime.now(timezone.utc),
            notes=(
                "Created a new cached provider segment and revealed only its first token."
            ),
            provider_capabilities=self._serialize_provider_capabilities(capabilities),
            segment_id=segment.segment_id,
            revealed_count=segment.revealed_count,
            cached_token_count=len(segment.tokens),
            remaining_cached_tokens=max(0, len(segment.tokens) - segment.revealed_count),
        )

    def _generate_approximate_segment(
        self,
        *,
        request: ContinueGenerationRequest,
        context: ContinuationContext,
        capabilities: ProviderCapabilities,
    ) -> ContinueGenerationResponse:
        steps, _, latency_ms = self._request_approximate_steps(
            request=request,
            prompt=context.root_prompt,
            assistant_prefix=context.assistant_prefix,
            branch_id=request.parent_node_id,
            parent_node_id=request.parent_node_id,
            max_output_tokens=1,
            top_logprobs=request.max_children,
        )
        if not steps:
            self._raise_logprobs_unavailable()

        segment = self._store_generation_segment(
            request=request,
            context=context,
            steps=steps,
            revealed_count=1,
            mode="approximate",
        )
        source_trace = steps[0]
        latency_value = latency_ms or source_trace.latency_ms
        children = self._build_expansion_children_from_trace(
            request=request,
            trace=source_trace,
            latency_ms=latency_value,
        )
        next_cached_token_index = 1 if len(steps) > 1 else None
        children = self._apply_continuation_metadata(
            children=children,
            capabilities=capabilities,
            continuation_mode="approximate",
            segment_id=segment.segment_id,
            next_cached_token_index=next_cached_token_index,
            cached_token_count=len(steps),
        )

        self._log_continue_action(
            action="new_provider_segment",
            mode="approximate",
            request=request,
            capabilities=capabilities,
            segment_id=segment.segment_id,
            revealed_count=segment.revealed_count,
            cached_token_count=len(segment.tokens),
            source_context=context.assistant_prefix,
            raw_first_token_if_requested=source_trace.token,
            provider_call=True,
        )
        return ContinueGenerationResponse(
            mode="live",
            action="new_provider_segment",
            continuation_mode="approximate",
            parent_node_id=request.parent_node_id,
            children=children,
            entropy=source_trace.entropy,
            expanded_at=datetime.now(timezone.utc),
            notes="Created a cached approximate continuation segment and revealed its first token.",
            provider_capabilities=self._serialize_provider_capabilities(capabilities),
            segment_id=segment.segment_id,
            revealed_count=segment.revealed_count,
            cached_token_count=len(segment.tokens),
            remaining_cached_tokens=max(0, len(segment.tokens) - segment.revealed_count),
        )

    def _build_approximate_continuation_prompt(
        self,
        *,
        prompt: str,
        assistant_prefix: str,
    ) -> str:
        return "\n\n".join(
            [
                "Continue the unfinished assistant response below.",
                "Return ONLY the text immediately following the final character.",
                "\n".join(
                    [
                        "Do not repeat the existing response.",
                        "Do not restart the answer.",
                        "Do not introduce yourself.",
                        "Do not add explanations.",
                    ]
                ),
                f"Original user request:\n{prompt}",
                f"Current assistant response:\n{assistant_prefix}",
            ]
        )

    def _request_approximate_steps(
        self,
        *,
        request: NodeExpansionRequest | ContinueGenerationRequest,
        prompt: str,
        assistant_prefix: str,
        branch_id: str,
        parent_node_id: str,
        max_output_tokens: int,
        top_logprobs: int,
    ) -> tuple[list[TokenTrace], Any | None, int]:
        client = self._get_client()

        if client is None:
            raise LLMScopeError(
                code="OPENAI_NOT_CONFIGURED",
                message="OpenAI is not configured on the backend. Enable Demo Mode or add an API key.",
                status_code=503,
            )

        approximate_prompt = self._build_approximate_continuation_prompt(
            prompt=prompt,
            assistant_prefix=assistant_prefix,
        )
        capabilities = self._provider_capabilities_for_model(request.model)
        effective_output_tokens = max(capabilities.minimum_output_tokens, max_output_tokens)
        input_items = [{"role": "user", "content": approximate_prompt}]

        if self._settings.app_env.lower() == "development":
            logger.debug(
                "APPROXIMATE CONTINUATION API INPUT %s",
                {
                    "parent_node_id": parent_node_id,
                    "mode": "approximate",
                    "request_messages": input_items,
                    "assistant_prefix": assistant_prefix,
                    "requested_max_output_tokens": max_output_tokens,
                    "effective_max_output_tokens": effective_output_tokens,
                    "top_logprobs": top_logprobs,
                },
            )

        start_time = perf_counter()
        try:
            response = client.responses.create(
                model=request.model,
                input=input_items,
                include=["message.output_text.logprobs"],
                max_output_tokens=effective_output_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_logprobs=max(1, min(top_logprobs, 20)),
            )
        except Exception as exc:  # pragma: no cover - depends on provider/network.
            message = "The backend could not fetch token alternatives from the provider."
            if self._settings.app_env.lower() == "development":
                message = f"{message} {exc}"
            raise LLMScopeError(
                code="PROVIDER_REQUEST_FAILED",
                message=message,
                status_code=502,
            ) from exc

        latency_ms = max(1, int((perf_counter() - start_time) * 1000))
        output_text, logprob_entries = self._extract_output_text_and_logprobs(response)
        if self._settings.app_env.lower() == "development":
            first_entry = logprob_entries[0] if logprob_entries else None
            logger.debug(
                "APPROXIMATE CONTINUATION API OUTPUT %s",
                {
                    "parent_node_id": parent_node_id,
                    "mode": "approximate",
                    "raw_returned_tokens": [getattr(entry, "token", "") for entry in logprob_entries],
                    "raw_first_token": getattr(first_entry, "token", "") if first_entry else "",
                    "output_text": output_text,
                },
            )

        if not logprob_entries:
            self._raise_logprobs_unavailable()

        steps = self._build_live_token_traces(
            branch_id=branch_id,
            model=request.model,
            source="openai",
            parent_node_id=parent_node_id,
            context_prefix=assistant_prefix,
            logprob_entries=logprob_entries,
            latency_ms=latency_ms,
            finish_reason=getattr(response, "status", None),
        )

        if not steps and output_text:
            self._raise_logprobs_unavailable()

        return steps, getattr(response, "usage", None), latency_ms

    def _request_live_steps(
        self,
        *,
        request: GenerationRequest | NodeExpansionRequest,
        prompt: str,
        preset: str,
        intent: str,
        assistant_prefix: str,
        branch_id: str,
        parent_node_id: str,
        max_output_tokens: int,
        top_logprobs: int,
    ) -> tuple[list[TokenTrace], Any | None, int]:
        client = self._get_client()

        if client is None:
            raise LLMScopeError(
                code="OPENAI_NOT_CONFIGURED",
                message="OpenAI is not configured on the backend. Enable Demo Mode or add an API key.",
                status_code=503,
            )

        instructions = self._build_live_instructions(
            preset=preset,
            intent=intent,
            variation=request.variation,
            assistant_prefix=assistant_prefix,
        )
        input_items: list[dict[str, str]] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        if assistant_prefix:
            input_items.append(
                {
                    "role": "assistant",
                    "content": assistant_prefix,
                }
            )

        capabilities = self._provider_capabilities_for_model(request.model)
        effective_output_tokens = max(capabilities.minimum_output_tokens, max_output_tokens)
        if self._settings.app_env.lower() == "development" and assistant_prefix:
            logger.debug(
                "CONTINUATION API INPUT %s",
                {
                    "parent_node_id": parent_node_id,
                    "exact_prompt": prompt,
                    "assistant_prefix": assistant_prefix,
                    "continuation_instructions": instructions,
                    "request_messages": input_items,
                    "requested_max_output_tokens": max_output_tokens,
                    "effective_max_output_tokens": effective_output_tokens,
                    "top_logprobs": top_logprobs,
                },
            )
        start_time = perf_counter()
        try:
            response = client.responses.create(
                model=request.model,
                input=input_items,
                instructions=instructions,
                include=["message.output_text.logprobs"],
                max_output_tokens=effective_output_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_logprobs=max(1, min(top_logprobs, 20)),
            )
        except Exception as exc:  # pragma: no cover - depends on provider/network.
            message = "The backend could not fetch token alternatives from the provider."
            if self._settings.app_env.lower() == "development":
                message = f"{message} {exc}"
            raise LLMScopeError(
                code="PROVIDER_REQUEST_FAILED",
                message=message,
                status_code=502,
            ) from exc

        latency_ms = max(1, int((perf_counter() - start_time) * 1000))
        output_text, logprob_entries = self._extract_output_text_and_logprobs(response)
        if self._settings.app_env.lower() == "development" and assistant_prefix:
            first_entry = logprob_entries[0] if logprob_entries else None
            raw_top_alternatives = [
                {
                    "token": getattr(candidate, "token", ""),
                    "logprob": float(getattr(candidate, "logprob", 0.0)),
                    "bytes": getattr(candidate, "bytes", []) or [],
                }
                for candidate in (getattr(first_entry, "top_logprobs", []) or [])
            ]
            logger.debug(
                "CONTINUATION API OUTPUT %s",
                {
                    "parent_node_id": parent_node_id,
                    "raw_returned_tokens": [getattr(entry, "token", "") for entry in logprob_entries],
                    "raw_first_token": getattr(first_entry, "token", "") if first_entry else "",
                    "raw_top_logprobs_token_0": raw_top_alternatives,
                    "output_text": output_text,
                },
            )
            logger.debug(
                "CONTINUATION DIAGNOSTIC %s",
                {
                    "exactAssistantPrefix": assistant_prefix,
                    "rawFirstToken": getattr(first_entry, "token", "") if first_entry else "",
                    "rawTopAlternatives": raw_top_alternatives,
                    "requestMessages": input_items,
                    "continuationInstructions": instructions,
                },
            )

        if not logprob_entries:
            self._raise_logprobs_unavailable()

        steps = self._build_live_token_traces(
            branch_id=branch_id,
            model=request.model,
            source="openai",
            parent_node_id=parent_node_id,
            context_prefix=assistant_prefix,
            logprob_entries=logprob_entries,
            latency_ms=latency_ms,
            finish_reason=getattr(response, "status", None),
        )

        if not steps and output_text:
            self._raise_logprobs_unavailable()

        return steps, getattr(response, "usage", None), latency_ms

    def _extract_output_text_and_logprobs(self, response: Any) -> tuple[str, list[Any]]:
        output_text_parts: list[str] = []
        logprob_entries: list[Any] = []

        for output_item in getattr(response, "output", []) or []:
            if getattr(output_item, "type", None) != "message":
                continue

            for content_part in getattr(output_item, "content", []) or []:
                if getattr(content_part, "type", None) != "output_text":
                    continue

                output_text_parts.append(getattr(content_part, "text", "") or "")
                logprob_entries.extend(getattr(content_part, "logprobs", []) or [])

        return "".join(output_text_parts), logprob_entries

    def _build_live_token_traces(
        self,
        *,
        branch_id: str,
        model: str,
        source: str,
        parent_node_id: str,
        context_prefix: str,
        logprob_entries: list[Any],
        latency_ms: int,
        finish_reason: str | None = None,
    ) -> list[TokenTrace]:
        traces: list[TokenTrace] = []
        cumulative_probability = 1.0
        context_before = context_prefix
        cumulative_log_probability = 0.0
        cumulative_token_ids: list[int] | None = []
        per_token_latency = max(1, int(round(latency_ms / max(len(logprob_entries), 1))))

        for index, entry in enumerate(logprob_entries):
            output_step = index
            chosen_token = getattr(entry, "token", "")
            chosen_logprob = float(getattr(entry, "logprob", 0.0))
            chosen_token_bytes = getattr(entry, "bytes", []) or []
            chosen_token_id = self._safe_optional_int(getattr(entry, "token_id", None))
            chosen_tokenizer_id = self._safe_optional_int(getattr(entry, "tokenizer_id", None))
            raw_top_logprobs = getattr(entry, "top_logprobs", []) or []
            ranked_candidates, entropy = self._build_ranked_candidates(
                chosen_token=chosen_token,
                chosen_logprob=chosen_logprob,
                chosen_bytes=chosen_token_bytes,
                chosen_token_id=chosen_token_id,
                chosen_tokenizer_id=chosen_tokenizer_id,
                top_logprobs=raw_top_logprobs,
                context_before=context_before,
                generation_step=output_step,
                latency_ms=per_token_latency,
                parent_cumulative_log_probability=cumulative_log_probability,
                parent_cumulative_token_ids=cumulative_token_ids,
            )

            if not ranked_candidates:
                continue

            chosen_candidate = ranked_candidates[0]
            cumulative_probability = round(
                self._clamp(
                    cumulative_probability * chosen_candidate["raw_probability"],
                    MIN_PROBABILITY,
                    1.0,
                ),
                6,
            )
            step_parent_id = parent_node_id if index == 0 else traces[index - 1].id
            step_id = self._make_step_id(
                branch_id=branch_id,
                token_index=output_step,
                token=chosen_token,
            )
            trace = TokenTrace(
                id=step_id,
                branch_id=branch_id,
                parent_node_id=step_parent_id,
                model=model,
                source=source,
                index=output_step,
                position=output_step,
                token=chosen_token,
                display_token=chosen_candidate["display_token"],
                token_bytes=chosen_candidate["token_bytes"],
                decoded_contribution=chosen_candidate["decoded_contribution"],
                cumulative_decoded_text=chosen_candidate["cumulative_decoded_text"],
                cumulative_token_ids=chosen_candidate["cumulative_token_ids"],
                cumulative_log_probability=chosen_candidate["cumulative_log_probability"],
                token_id=chosen_candidate["token_id"],
                tokenizer_id=chosen_candidate["tokenizer_id"],
                probability=chosen_candidate["raw_probability"],
                raw_probability=chosen_candidate["raw_probability"],
                normalized_displayed_probability=chosen_candidate[
                    "normalized_displayed_probability"
                ],
                log_probability=chosen_candidate["log_probability"],
                entropy=entropy,
                cumulative_probability=cumulative_probability,
                latency_ms=per_token_latency,
                text_preview=chosen_candidate["cumulative_decoded_text"],
                context_before=context_before,
                context_after=chosen_candidate["cumulative_decoded_text"],
                finish_reason=finish_reason,
                generation_step=output_step,
                metadata={
                    "branch_id": branch_id,
                    "parent_node_id": step_parent_id,
                    "source": source,
                },
                alternatives=[
                    AlternativeCandidate(
                        node_id=self._make_step_id(
                            branch_id=step_parent_id,
                            token_index=output_step,
                            token=candidate["token"],
                        ),
                        token=candidate["token"],
                        display_token=candidate["display_token"],
                        token_bytes=candidate["token_bytes"],
                        decoded_contribution=candidate["decoded_contribution"],
                        cumulative_decoded_text=candidate["cumulative_decoded_text"],
                        cumulative_token_ids=candidate["cumulative_token_ids"],
                        cumulative_log_probability=candidate["cumulative_log_probability"],
                        probability=candidate["raw_probability"],
                        raw_probability=candidate["raw_probability"],
                        normalized_displayed_probability=candidate[
                            "normalized_displayed_probability"
                        ],
                        log_probability=candidate["log_probability"],
                        entropy=entropy,
                        latency_ms=per_token_latency,
                        token_id=candidate["token_id"],
                        tokenizer_id=candidate["tokenizer_id"],
                        rank=candidate["rank"],
                        text_preview=candidate["cumulative_decoded_text"],
                        context_before=context_before,
                        context_after=candidate["cumulative_decoded_text"],
                        rationale=None,
                        generation_step=output_step,
                        metadata={
                            "branch_id": step_parent_id,
                            "parent_node_id": step_parent_id,
                            "source": source,
                        },
                    )
                    for candidate in ranked_candidates[1:]
                ],
            )
            traces.append(trace)

            if self._settings.app_env.lower() == "development" and context_prefix:
                logger.debug(
                    "CONTINUATION TOKEN TRACE %s",
                    {
                        "branch_id": branch_id,
                        "output_step": output_step,
                        "raw_api_token": chosen_token,
                        "raw_logprob": chosen_logprob,
                        "raw_top_logprobs": [
                            {
                                "token": getattr(candidate, "token", ""),
                                "logprob": float(getattr(candidate, "logprob", 0.0)),
                                "bytes": getattr(candidate, "bytes", []) or [],
                            }
                            for candidate in raw_top_logprobs
                        ],
                        "parsed_generated_token": trace.token,
                        "parsed_alternatives": [
                            {
                                "token": candidate.token,
                                "logprob": candidate.log_probability,
                                "probability": candidate.raw_probability,
                                "rank": candidate.rank,
                            }
                            for candidate in trace.alternatives
                        ],
                        "context_before": trace.context_before,
                        "context_through": trace.context_after,
                    },
                )

            context_before = trace.cumulative_decoded_text
            cumulative_log_probability = trace.cumulative_log_probability
            cumulative_token_ids = trace.cumulative_token_ids

        self._log_token_steps(branch_id=branch_id, context_suffix=context_before, steps=traces)
        return traces

    def _build_ranked_candidates(
        self,
        *,
        chosen_token: str,
        chosen_logprob: float,
        chosen_bytes: list[int],
        chosen_token_id: int | None,
        chosen_tokenizer_id: int | None,
        top_logprobs: list[Any],
        context_before: str,
        generation_step: int,
        latency_ms: int,
        parent_cumulative_log_probability: float,
        parent_cumulative_token_ids: list[int] | None,
    ) -> tuple[list[dict[str, Any]], float]:
        deduped_candidates: list[tuple[str, float, list[int], int | None, int | None]] = []
        seen_tokens: set[str] = set()

        def add_candidate(
            token: str,
            logprob: float,
            token_bytes: list[int],
            token_id: int | None,
            tokenizer_id: int | None,
        ) -> None:
            if token in seen_tokens:
                return
            seen_tokens.add(token)
            deduped_candidates.append((token, logprob, token_bytes, token_id, tokenizer_id))

        add_candidate(
            chosen_token,
            chosen_logprob,
            chosen_bytes,
            chosen_token_id,
            chosen_tokenizer_id,
        )
        for candidate in top_logprobs:
            add_candidate(
                getattr(candidate, "token", ""),
                float(getattr(candidate, "logprob", 0.0)),
                getattr(candidate, "bytes", []) or [],
                self._safe_optional_int(getattr(candidate, "token_id", None)),
                self._safe_optional_int(getattr(candidate, "tokenizer_id", None)),
            )

        raw_probabilities = [
            self._probability_from_logprob(logprob) for _, logprob, _, _, _ in deduped_candidates
        ]
        normalized_probabilities = self._normalize_probabilities(raw_probabilities)
        entropy = self._calculate_entropy(normalized_probabilities)
        ranked: list[dict[str, Any]] = []

        for index, (
            (token, logprob, token_bytes, token_id, tokenizer_id),
            raw_probability,
            normalized_probability,
        ) in enumerate(
            zip(
                deduped_candidates,
                raw_probabilities,
                normalized_probabilities,
                strict=False,
            ),
            start=1,
        ):
            decoded_contribution = self._decode_token_bytes(token, token_bytes)
            cumulative_decoded_text = f"{context_before}{decoded_contribution}"
            ranked.append(
                {
                    "token": token,
                    "display_token": self._canonical_display_token(decoded_contribution),
                    "token_bytes": token_bytes,
                    "decoded_contribution": decoded_contribution,
                    "cumulative_decoded_text": cumulative_decoded_text,
                    "cumulative_token_ids": self._append_token_history(
                        parent_cumulative_token_ids,
                        token_id,
                    ),
                    "cumulative_log_probability": round(
                        parent_cumulative_log_probability + logprob,
                        6,
                    ),
                    "token_id": token_id,
                    "tokenizer_id": tokenizer_id,
                    "log_probability": round(logprob, 6),
                    "raw_probability": raw_probability,
                    "normalized_displayed_probability": normalized_probability,
                    "rank": index,
                    "latency_ms": latency_ms,
                    "generation_step": generation_step,
                    "context_after": cumulative_decoded_text,
                }
            )

        return ranked, entropy

    def _build_expansion_children_from_trace(
        self,
        *,
        request: NodeExpansionRequest,
        trace: TokenTrace,
        latency_ms: int,
    ) -> list[NodeExpansionCandidate]:
        candidates = [
            {
                "token": trace.token,
                "display_token": trace.display_token,
                "token_bytes": trace.token_bytes,
                "decoded_contribution": trace.decoded_contribution,
                "cumulative_decoded_text": trace.cumulative_decoded_text,
                "cumulative_token_ids": trace.cumulative_token_ids,
                "cumulative_log_probability": trace.cumulative_log_probability,
                "token_id": trace.token_id,
                "tokenizer_id": trace.tokenizer_id,
                "probability": trace.raw_probability,
                "raw_probability": trace.raw_probability,
                "normalized_displayed_probability": trace.normalized_displayed_probability,
                "log_probability": trace.log_probability,
                "rank": 1,
                "generation_step": request.depth,
                "context_after": trace.cumulative_decoded_text,
            },
            *[
                {
                    "token": candidate.token,
                    "display_token": candidate.display_token
                    or self._canonical_display_token(candidate.token),
                    "token_bytes": candidate.token_bytes,
                    "decoded_contribution": candidate.decoded_contribution
                    or self._decode_token_bytes(candidate.token, candidate.token_bytes),
                    "cumulative_decoded_text": candidate.cumulative_decoded_text
                    or candidate.context_after
                    or candidate.text_preview
                    or f"{trace.context_before}{candidate.token}",
                    "cumulative_token_ids": candidate.cumulative_token_ids,
                    "cumulative_log_probability": candidate.cumulative_log_probability
                    or round(
                        trace.cumulative_log_probability
                        - trace.log_probability
                        + (candidate.log_probability or self._safe_log(candidate.probability)),
                        6,
                    ),
                    "token_id": candidate.token_id,
                    "tokenizer_id": candidate.tokenizer_id,
                    "probability": candidate.raw_probability or candidate.probability,
                    "raw_probability": candidate.raw_probability or candidate.probability,
                    "normalized_displayed_probability": candidate.normalized_displayed_probability
                    or candidate.probability,
                    "log_probability": candidate.log_probability
                    or self._safe_log(candidate.probability),
                    "rank": candidate.rank or (index + 2),
                    "generation_step": request.depth,
                    "context_after": candidate.context_after
                    or candidate.text_preview
                    or f"{trace.context_before}{candidate.token}",
                }
                for index, candidate in enumerate(trace.alternatives)
            ],
        ]
        children: list[NodeExpansionCandidate] = []

        for candidate in candidates[: request.max_children]:
            if candidate["generation_step"] != request.depth:
                raise LLMScopeError(
                    code="CONTINUATION_CONTEXT_MISMATCH",
                    message="The returned child generation step does not match the canonical prefix length.",
                    status_code=502,
                )

            if (
                candidate["cumulative_token_ids"] is not None
                and len(candidate["cumulative_token_ids"]) != request.depth + 1
            ):
                raise LLMScopeError(
                    code="CONTINUATION_CONTEXT_MISMATCH",
                    message="The returned child token history does not match the canonical decoding step.",
                    status_code=502,
                )

            cumulative_probability = round(
                self._clamp(
                    request.cumulative_probability * candidate["raw_probability"],
                    MIN_PROBABILITY,
                    1.0,
                ),
                6,
            )
            child_id = self._make_step_id(
                branch_id=request.parent_node_id,
                token_index=request.depth,
                token=candidate["token"],
            )
            children.append(
                NodeExpansionCandidate(
                    id=child_id,
                    branch_id=child_id,
                    parent_node_id=request.parent_node_id,
                    model=request.model,
                    source=trace.source,
                    token=candidate["token"],
                    display_token=candidate["display_token"],
                    token_bytes=candidate["token_bytes"],
                    decoded_contribution=candidate["decoded_contribution"],
                    cumulative_decoded_text=candidate["cumulative_decoded_text"],
                    cumulative_token_ids=candidate["cumulative_token_ids"],
                    cumulative_log_probability=candidate["cumulative_log_probability"],
                    token_id=candidate["token_id"],
                    tokenizer_id=candidate["tokenizer_id"],
                    probability=candidate["probability"],
                    raw_probability=candidate["raw_probability"],
                    normalized_displayed_probability=candidate[
                        "normalized_displayed_probability"
                    ],
                    log_probability=candidate["log_probability"],
                    entropy=trace.entropy,
                    cumulative_probability=cumulative_probability,
                    latency_ms=latency_ms,
                    depth=request.depth + 1,
                    rank=candidate["rank"],
                    text_preview=candidate["cumulative_decoded_text"],
                    context_before=trace.context_before,
                    context_after=candidate["cumulative_decoded_text"],
                    finish_reason=trace.finish_reason,
                    rationale=None,
                    generation_step=candidate["generation_step"],
                    metadata={
                        "parent_node_id": request.parent_node_id,
                        "source": trace.source,
                    },
                )
            )

        self._log_expansion_candidates(
            branch_id=request.parent_node_id,
            context_suffix=trace.context_before,
            children=children,
        )
        return children

    def _build_demo_generation(
        self,
        *,
        request: GenerationRequest,
        prompt: str,
        keywords: list[str],
        intent: str,
        preset: str,
    ) -> tuple[str, str, Any | None, int, list[TokenTrace]]:
        completion = self._build_demo_completion(
            prompt=prompt,
            keywords=keywords,
            intent=intent,
            max_tokens=request.max_tokens,
            preset=preset,
            variation=request.variation,
        )
        traces = self._build_demo_token_traces(
            branch_id="main",
            model=request.model,
            prompt=prompt,
            completion=completion,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return completion, "demo", None, max(1, len(traces) * 32), traces

    def _build_demo_expansion(
        self,
        *,
        request: NodeExpansionRequest,
        prompt: str,
        assistant_prefix: str,
        intent: str,
    ) -> tuple[list[NodeExpansionCandidate], float, str]:
        demo_continuation = self._build_demo_completion(
            prompt=f"{prompt}\n\nContinuation context: {assistant_prefix}",
            keywords=self._extract_keywords(f"{prompt} {assistant_prefix}"),
            intent=intent,
            max_tokens=max(12, request.max_children),
            preset=request.preset,
            variation=request.variation,
        )
        demo_tokens = self._demo_tokenize(demo_continuation)
        if not demo_tokens:
            return [], 0.0, "demo"

        candidate_specs: list[tuple[str, float]] = []
        seen: set[str] = set()
        for index, token in enumerate(demo_tokens):
            if token in seen:
                continue
            seen.add(token)
            probability = round(
                self._clamp(0.68 - (index * 0.08), 0.06, 0.92),
                6,
            )
            candidate_specs.append((token, probability))
            if len(candidate_specs) == request.max_children:
                break

        normalized_probabilities = self._normalize_probabilities(
            [probability for _, probability in candidate_specs]
        )
        entropy = self._calculate_entropy(normalized_probabilities)
        children: list[NodeExpansionCandidate] = []

        for rank, ((token, raw_probability), normalized_probability) in enumerate(
            zip(candidate_specs, normalized_probabilities, strict=False),
            start=1,
        ):
            child_id = self._make_step_id(
                branch_id=request.parent_node_id,
                token_index=request.depth,
                token=token,
            )
            decoded_contribution = self._decode_token_bytes(token, self._token_bytes(token))
            text_preview = f"{assistant_prefix}{decoded_contribution}"
            children.append(
                NodeExpansionCandidate(
                    id=child_id,
                    branch_id=child_id,
                    parent_node_id=request.parent_node_id,
                    model=request.model,
                    source="demo",
                    token=token,
                    display_token=self._canonical_display_token(decoded_contribution),
                    token_bytes=self._token_bytes(token),
                    decoded_contribution=decoded_contribution,
                    cumulative_decoded_text=text_preview,
                    cumulative_token_ids=None,
                    cumulative_log_probability=round(self._safe_log(raw_probability), 6),
                    token_id=None,
                    tokenizer_id=None,
                    probability=raw_probability,
                    raw_probability=raw_probability,
                    normalized_displayed_probability=normalized_probability,
                    log_probability=self._safe_log(raw_probability),
                    entropy=entropy,
                    cumulative_probability=round(
                        self._clamp(
                            request.cumulative_probability * raw_probability,
                            MIN_PROBABILITY,
                            1.0,
                        ),
                        6,
                    ),
                    latency_ms=24 + (rank * 6),
                    depth=request.depth + 1,
                    rank=rank,
                    text_preview=text_preview,
                    context_before=assistant_prefix,
                    context_after=text_preview,
                    finish_reason="demo",
                    rationale="Demo candidate",
                    generation_step=request.depth,
                    metadata={
                        "source": "demo",
                        "parent_node_id": request.parent_node_id,
                    },
                )
            )

        return children, entropy, "demo"

    def _build_demo_token_traces(
        self,
        *,
        branch_id: str,
        model: str,
        prompt: str,
        completion: str,
        temperature: float,
        top_p: float,
    ) -> list[TokenTrace]:
        tokens = self._demo_tokenize(completion)
        pool = self._demo_candidate_pool(prompt=prompt, completion=completion)
        traces: list[TokenTrace] = []
        context_before = ""
        cumulative_probability = 1.0
        cumulative_log_probability = 0.0

        for index, token in enumerate(tokens):
            chosen_probability = round(
                self._clamp(
                    0.86 - (index * 0.02) - max(temperature - 0.7, 0) * 0.05 + (1 - top_p) * 0.03,
                    0.08,
                    0.96,
                ),
                6,
            )
            alternatives: list[tuple[str, float]] = []
            alt_index = 0
            while len(alternatives) < 3 and alt_index < len(pool):
                candidate = pool[alt_index]
                alt_index += 1
                if candidate == token or any(existing[0] == candidate for existing in alternatives):
                    continue
                probability = round(
                    self._clamp(chosen_probability - (0.12 + len(alternatives) * 0.08), 0.03, 0.74),
                    6,
                )
                alternatives.append((candidate, probability))

            raw_probabilities = [chosen_probability, *[probability for _, probability in alternatives]]
            normalized_probabilities = self._normalize_probabilities(raw_probabilities)
            entropy = self._calculate_entropy(normalized_probabilities)
            decoded_contribution = self._decode_token_bytes(token, self._token_bytes(token))
            context_after = f"{context_before}{decoded_contribution}"
            cumulative_probability = round(
                self._clamp(cumulative_probability * chosen_probability, MIN_PROBABILITY, 1.0),
                6,
            )
            cumulative_log_probability = round(
                cumulative_log_probability + self._safe_log(chosen_probability),
                6,
            )
            step_id = self._make_step_id(branch_id=branch_id, token_index=index, token=token)
            traces.append(
                TokenTrace(
                    id=step_id,
                    branch_id=branch_id,
                    parent_node_id="root" if index == 0 else traces[index - 1].id,
                    model=model,
                    source="demo",
                    index=index,
                    position=index,
                    token=token,
                    display_token=self._canonical_display_token(decoded_contribution),
                    token_bytes=self._token_bytes(token),
                    decoded_contribution=decoded_contribution,
                    cumulative_decoded_text=context_after,
                    cumulative_token_ids=None,
                    cumulative_log_probability=cumulative_log_probability,
                    token_id=None,
                    tokenizer_id=None,
                    probability=chosen_probability,
                    raw_probability=chosen_probability,
                    normalized_displayed_probability=normalized_probabilities[0],
                    log_probability=self._safe_log(chosen_probability),
                    entropy=entropy,
                    cumulative_probability=cumulative_probability,
                    latency_ms=26 + (index * 4),
                    text_preview=context_after,
                    context_before=context_before,
                    context_after=context_after,
                    finish_reason="demo",
                    generation_step=index,
                    metadata={
                        "branch_id": branch_id,
                        "source": "demo",
                    },
                    alternatives=[
                        AlternativeCandidate(
                            node_id=self._make_step_id(
                                branch_id="root" if index == 0 else traces[index - 1].id,
                                token_index=index,
                                token=alt_token,
                            ),
                            token=alt_token,
                            display_token=self._canonical_display_token(alt_token),
                            token_bytes=self._token_bytes(alt_token),
                            decoded_contribution=self._decode_token_bytes(
                                alt_token,
                                self._token_bytes(alt_token),
                            ),
                            cumulative_decoded_text=f"{context_before}{alt_token}",
                            cumulative_token_ids=None,
                            cumulative_log_probability=round(
                                cumulative_log_probability
                                - self._safe_log(chosen_probability)
                                + self._safe_log(alt_probability),
                                6,
                            ),
                            probability=alt_probability,
                            raw_probability=alt_probability,
                            normalized_displayed_probability=normalized_probabilities[alt_rank],
                            log_probability=self._safe_log(alt_probability),
                            entropy=entropy,
                            latency_ms=24 + (alt_rank * 5),
                            token_id=None,
                            tokenizer_id=None,
                            rank=alt_rank + 1,
                            text_preview=f"{context_before}{alt_token}",
                            context_before=context_before,
                            context_after=f"{context_before}{alt_token}",
                            rationale="Demo candidate",
                            generation_step=index,
                            metadata={
                                "branch_id": "root" if index == 0 else traces[index - 1].id,
                                "source": "demo",
                            },
                        )
                        for alt_rank, (alt_token, alt_probability) in enumerate(alternatives, start=1)
                    ],
                )
            )
            context_before = context_after

        return traces

    def _demo_candidate_pool(self, *, prompt: str, completion: str) -> list[str]:
        combined = [*self._demo_tokenize(completion), *self._demo_tokenize(prompt)]
        pool: list[str] = []
        seen: set[str] = set()

        for token in combined:
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if not token.startswith(" ") and pool:
                token = f" {token.strip()}"
            pool.append(token)

        return pool

    def _build_live_instructions(
        self,
        *,
        preset: str,
        intent: str,
        variation: int,
        assistant_prefix: str = "",
    ) -> str:
        if assistant_prefix:
            return " ".join(
                [
                    "You are continuing an unfinished assistant response inside a model-inspection UI.",
                    "The original user request already appears earlier in the conversation.",
                    "Treat the supplied assistant prefix as the active assistant turn and continue from its exact endpoint.",
                    "Emit only the immediate next assistant text that follows that exact prefix.",
                    CONTINUATION_TONE_HINTS[preset],
                    "Do not restart the answer, repeat the prefix, summarize it, quote it, or open a new user or assistant turn.",
                    "Preserve whitespace, punctuation, markdown, Unicode, and formatting exactly as continuation context.",
                ]
            )

        style_hint = RESPONSE_STYLE_HINTS[variation % len(RESPONSE_STYLE_HINTS)]
        parts = [
            "You are answering inside a model-inspection UI.",
            "Be accurate, direct, and useful.",
            "Avoid filler, vague generic language, and meta commentary.",
            "If the user asks for a benchmark, range, definition, or recommendation, answer that concrete question first.",
            PRESET_INSTRUCTIONS[preset],
            style_hint,
            f"Detected intent: {intent}.",
        ]

        return " ".join(parts)

    def _build_demo_completion(
        self,
        *,
        prompt: str,
        keywords: list[str],
        intent: str,
        max_tokens: int,
        preset: str,
        variation: int,
    ) -> str:
        prompt_lower = prompt.lower()

        if self._looks_like_track_benchmark(prompt_lower):
            options = [
                (
                    "For a 16-year-old in the 400m, a solid range is often about 55 to 60 seconds, "
                    "low-50s is strong, and sub-50 is elite in many high-school settings."
                ),
                (
                    "A competitive range for a 16-year-old in the 400m is usually around 55 to 60 seconds, "
                    "with faster times standing out more seriously."
                ),
            ]
            return self._truncate_demo_completion(options[variation % len(options)], max_tokens)

        topic = self._topic_phrase(prompt, keywords)
        focus = keywords[0] if keywords else "the main constraint"
        secondary = keywords[1] if len(keywords) > 1 else "the result"

        if intent == "comparison":
            sentence = (
                f"For {topic}, compare control, speed, and reliability first, then decide which option better supports {focus}."
            )
        elif intent == "debugging":
            sentence = (
                f"To debug {topic}, reproduce the smallest failing case first, inspect the step where {focus} diverges, and add one focused check for {secondary}."
            )
        elif intent == "planning":
            sentence = (
                f"For {topic}, start with the smallest version that works, measure {focus}, and only add complexity when it changes {secondary}."
            )
        elif preset == "coach":
            sentence = (
                f"For {topic}, start with a clear baseline, then tighten the target as you get better data about {focus}."
            )
        else:
            sentence = (
                f"For {topic}, start with the direct answer, then add the context that most changes {focus} and {secondary}."
            )

        return self._truncate_demo_completion(sentence, max_tokens)

    def _build_follow_ups(self, keywords: list[str], intent: str) -> list[str]:
        focus = keywords[0] if keywords else "this topic"
        secondary = keywords[1] if len(keywords) > 1 else "the main constraint"

        if intent == "debugging":
            return [
                f"Ask for the smallest reproducible case around {focus}.",
                f"Request a checklist for validating {secondary} step by step.",
                "Ask which test would prevent the same failure.",
            ]

        if intent == "planning":
            return [
                f"Ask for an implementation checklist focused on {focus}.",
                f"Request milestones and metrics for {secondary}.",
                "Ask which part should stay minimal in the first version.",
            ]

        if intent == "comparison":
            return [
                f"Ask for a side-by-side table focused on {focus}.",
                f"Request concrete trade-offs around {secondary}.",
                "Ask which option should be the safer default.",
            ]

        return [
            f"Ask for a deeper explanation of {focus}.",
            f"Request concrete examples tied to {secondary}.",
            "Ask for the benchmark or decision rule in plain language.",
        ]

    def _build_tree(self, tokens: list[TokenTrace]) -> tuple[TokenTreeNode, TreeSummary]:
        max_depth = min(6, len(tokens))
        branch_width = 1 + min(
            3,
            max((len(token.alternatives) for token in tokens[:max_depth]), default=0),
        )
        root = TokenTreeNode(
            id="root",
            token="<start>",
            display_token="<start>",
            token_id=None,
            tokenizer_id=None,
            probability=1.0,
            raw_probability=1.0,
            normalized_displayed_probability=1.0,
            log_probability=0.0,
            entropy=0.0,
            cumulative_probability=1.0,
            latency_ms=0,
            depth=0,
            rank=1,
            text_preview="",
            is_selected_path=True,
            children=self._build_tree_children(
                tokens=tokens,
                position=0,
                parent_cumulative=1.0,
                prefix_text="",
                selected_path=True,
                node_prefix="root",
                max_depth=max_depth,
                branch_width=branch_width,
            ),
        )
        return root, TreeSummary(
            max_depth=max_depth,
            branch_width=branch_width,
            total_nodes=self._count_tree_nodes(root),
            explored_paths=self._count_tree_paths(root),
            selected_path_depth=max_depth,
        )

    def _build_tree_children(
        self,
        *,
        tokens: list[TokenTrace],
        position: int,
        parent_cumulative: float,
        prefix_text: str,
        selected_path: bool,
        node_prefix: str,
        max_depth: int,
        branch_width: int,
    ) -> list[TokenTreeNode]:
        if position >= max_depth:
            return []

        source = tokens[position]
        candidates: list[dict[str, Any]] = [
            {
                "token": source.token,
                "display_token": source.display_token,
                "probability": source.raw_probability,
                "normalized_displayed_probability": source.normalized_displayed_probability,
                "log_probability": source.log_probability,
                "latency_ms": source.latency_ms,
                "token_id": source.token_id,
                "tokenizer_id": source.tokenizer_id,
                "is_main_branch": True,
            }
        ]
        candidates.extend(
            {
                "token": alternative.token,
                "display_token": alternative.display_token
                or self._canonical_display_token(alternative.token),
                "probability": alternative.raw_probability or alternative.probability,
                "normalized_displayed_probability": alternative.normalized_displayed_probability
                or alternative.probability,
                "log_probability": alternative.log_probability
                or self._safe_log(alternative.probability),
                "latency_ms": alternative.latency_ms or source.latency_ms,
                "token_id": alternative.token_id,
                "tokenizer_id": alternative.tokenizer_id,
                "is_main_branch": False,
            }
            for alternative in source.alternatives[: branch_width - 1]
        )

        children: list[TokenTreeNode] = []
        for rank, candidate in enumerate(candidates, start=1):
            preview = f"{prefix_text}{candidate['token']}"
            cumulative_probability = round(
                self._clamp(parent_cumulative * candidate["probability"], MIN_PROBABILITY, 1.0),
                6,
            )
            on_selected_path = selected_path and candidate["is_main_branch"]
            node_id = f"{node_prefix}.{position + 1}.{rank}"
            children.append(
                TokenTreeNode(
                    id=node_id,
                    token=candidate["token"],
                    display_token=candidate["display_token"],
                    token_id=candidate["token_id"],
                    tokenizer_id=candidate["tokenizer_id"],
                    probability=candidate["probability"],
                    raw_probability=candidate["probability"],
                    normalized_displayed_probability=candidate[
                        "normalized_displayed_probability"
                    ],
                    log_probability=candidate["log_probability"],
                    entropy=source.entropy,
                    cumulative_probability=cumulative_probability,
                    latency_ms=candidate["latency_ms"],
                    depth=position + 1,
                    rank=rank,
                    text_preview=preview,
                    is_selected_path=on_selected_path,
                    children=self._build_tree_children(
                        tokens=tokens,
                        position=position + 1,
                        parent_cumulative=cumulative_probability,
                        prefix_text=preview,
                        selected_path=on_selected_path,
                        node_prefix=node_id,
                        max_depth=max_depth,
                        branch_width=branch_width,
                    ),
                )
            )

        return children

    def _count_tree_nodes(self, node: TokenTreeNode) -> int:
        return 1 + sum(self._count_tree_nodes(child) for child in node.children)

    def _count_tree_paths(self, node: TokenTreeNode) -> int:
        if not node.children:
            return 1
        return sum(self._count_tree_paths(child) for child in node.children)

    def _estimate_prompt_tokens(self, prompt: str, keywords: list[str]) -> int:
        return min(max(len(self._demo_tokenize(prompt)) + len(keywords), 14), 256)

    def _estimate_cost(self, model: str, *, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING_USD_PER_1K.get(
            model,
            {"input": 0.0004, "output": 0.0016},
        )
        estimated = (
            (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])
        ) / 1000
        return round(estimated, 6)

    def _usage_value(self, usage: Any | None, field_name: str) -> int | None:
        value = getattr(usage, field_name, None)
        if isinstance(value, int):
            return value
        return None

    def _normalize_probabilities(self, probabilities: list[float]) -> list[float]:
        total = sum(probabilities)

        if total <= 0:
            if not probabilities:
                return []
            even_probability = round(1 / len(probabilities), 6)
            return [even_probability for _ in probabilities]

        return [round(probability / total, 6) for probability in probabilities]

    def _calculate_entropy(self, probabilities: list[float]) -> float:
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        return round(entropy, 6)

    def _probability_from_logprob(self, logprob: float) -> float:
        return round(math.exp(logprob), 6)

    def _safe_log(self, probability: float) -> float:
        return round(math.log(max(probability, MIN_PROBABILITY)), 6)

    def _make_step_id(self, *, branch_id: str, token_index: int, token: str) -> str:
        safe_token = self._encode_token(token)
        return f"{branch_id}:{token_index}:{safe_token}"

    def _encode_token(self, token: str) -> str:
        checksum = zlib.adler32(token.encode("utf-8")) & 0xFFFFFFFF
        return format(checksum, "08x")

    def _display_token(self, token: str) -> str:
        visible = token.replace("\t", "⇥").replace("\n", "↵\n")
        if visible.startswith(" "):
            visible = visible.replace(" ", "␠", 1)
        return visible or "∅"

    def _token_bytes(self, token: str) -> list[int]:
        return list(token.encode("utf-8"))

    def _decode_token_bytes(self, token: str, token_bytes: list[int]) -> str:
        if token_bytes:
            try:
                decoded = bytes(token_bytes).decode("utf-8")
                if decoded:
                    return decoded
            except UnicodeDecodeError:
                pass
        return token

    def _append_token_history(
        self,
        parent_token_ids: list[int] | None,
        token_id: int | None,
    ) -> list[int] | None:
        if parent_token_ids is None or token_id is None:
            return None
        return [*parent_token_ids, token_id]

    def _safe_optional_int(self, value: Any | None) -> int | None:
        return value if isinstance(value, int) and value >= 0 else None

    def _canonical_display_token(self, token: str) -> str:
        visible = token.replace("\t", "\u21E5").replace("\n", "\u21B5\n")
        if visible.startswith(" "):
            visible = visible.replace(" ", "\u2420", 1)
        return visible or "\u2205"

    def _build_continuation_context(
        self,
        request: NodeExpansionRequest,
    ) -> ContinuationContext:
        root_prompt = request.root_prompt
        assistant_prefix = request.assistant_prefix
        reconstructed_prompt = f"{root_prompt}{assistant_prefix}"
        character_length = len(reconstructed_prompt)
        utf8_length = len(reconstructed_prompt.encode("utf-8"))
        token_count = request.depth
        display_markers = ("\u2420", "\u21B5", "\u21E5", "â ", "â†µ", "â‡¥")

        if request.parent_node_id == "root" and assistant_prefix:
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The root prompt cannot include an assistant prefix before continuation.",
                status_code=400,
            )

        if any(marker in assistant_prefix for marker in display_markers):
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The continuation context contains display-only whitespace markers instead of raw decoded text.",
                status_code=400,
            )

        if (
            request.parent_node_id != "root"
            and request.parent_token
            and not assistant_prefix.endswith(request.parent_token)
        ):
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The assistant prefix no longer ends with the selected raw token.",
                status_code=400,
            )

        if request.reconstructed_prompt and request.reconstructed_prompt != reconstructed_prompt:
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The reconstructed continuation prompt no longer matches the raw-token graph context.",
                status_code=400,
            )

        if (
            request.expected_prompt_length is not None
            and request.expected_prompt_length != character_length
        ):
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The reconstructed continuation character length does not match the graph validation data.",
                status_code=400,
            )

        if request.expected_utf8_length is not None and request.expected_utf8_length != utf8_length:
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The reconstructed continuation byte length does not match the graph validation data.",
                status_code=400,
            )

        if request.expected_token_count is not None and request.expected_token_count != token_count:
            raise LLMScopeError(
                code="CONTINUATION_CONTEXT_MISMATCH",
                message="The reconstructed continuation token count does not match the graph validation data.",
                status_code=400,
            )

        return ContinuationContext(
            root_prompt=root_prompt,
            assistant_prefix=assistant_prefix,
            reconstructed_prompt=reconstructed_prompt,
            character_length=character_length,
            utf8_length=utf8_length,
            token_count=token_count,
        )

    def _log_token_steps(
        self,
        *,
        branch_id: str,
        context_suffix: str,
        steps: list[TokenTrace],
    ) -> None:
        if self._settings.app_env.lower() != "development" or not steps:
            return

        for step in steps:
            top_k_snapshot = [
                {
                    "token": step.token,
                    "raw_logprob": step.log_probability,
                    "converted_probability": step.raw_probability,
                    "normalized_probability": step.normalized_displayed_probability,
                },
                *[
                    {
                        "token": candidate.token,
                        "raw_logprob": candidate.log_probability,
                        "converted_probability": (
                            candidate.raw_probability
                            if candidate.raw_probability is not None
                            else candidate.probability
                        ),
                        "normalized_probability": candidate.normalized_displayed_probability,
                    }
                    for candidate in step.alternatives[:4]
                ],
            ]
            probability_sum = round(
                sum(item["converted_probability"] for item in top_k_snapshot),
                6,
            )
            logger.info(
                "probability-audit branch=%s index=%s chosen_token=%r raw_logprob=%s converted_probability=%s normalized_probability=%s top_k_probabilities=%s probability_sum=%s context_suffix=%r",
                branch_id,
                step.index,
                step.token,
                step.log_probability,
                step.raw_probability,
                step.normalized_displayed_probability,
                top_k_snapshot,
                probability_sum,
                context_suffix[-120:],
            )

    def _log_expansion_candidates(
        self,
        *,
        branch_id: str,
        context_suffix: str,
        children: list[NodeExpansionCandidate],
    ) -> None:
        if self._settings.app_env.lower() != "development" or not children:
            return

        probability_sum = round(sum(child.raw_probability for child in children), 6)
        logger.info(
            "node-expansion branch=%s chosen_token=%r raw_logprob=%s converted_probability=%s normalized_probability=%s top_k=%s probability_sum=%s context_suffix=%r",
            branch_id,
            children[0].token,
            children[0].log_probability,
            children[0].raw_probability,
            children[0].normalized_displayed_probability,
            [
                {
                    "token": child.token,
                    "logprob": child.log_probability,
                    "probability": child.raw_probability,
                    "normalized_probability": child.normalized_displayed_probability,
                }
                for child in children
            ],
            probability_sum,
            context_suffix[-120:],
        )

    def _raise_logprobs_unavailable(self) -> None:
        raise LLMScopeError(
            code="TOP_LOGPROBS_UNAVAILABLE",
            message="This model response did not include token alternatives.",
            status_code=409,
        )

    def _looks_like_track_benchmark(self, prompt_lower: str) -> bool:
        return (
            ("good time" in prompt_lower or "fast time" in prompt_lower)
            and any(
                event in prompt_lower
                for event in ("100m", "200m", "400m", "800m", "1600", "mile", "5k")
            )
        )

    def _topic_phrase(self, prompt: str, keywords: list[str]) -> str:
        if keywords:
            return " ".join(keywords[: min(3, len(keywords))])

        sanitized = prompt.strip().rstrip(".?!")
        if len(sanitized) <= 48:
            return sanitized.lower()
        return f"{sanitized[:45].rstrip()}...".lower()

    def _extract_keywords(self, text: str, *, limit: int = 6) -> list[str]:
        keywords: list[str] = []
        seen: set[str] = set()

        for word in re.findall(r"[a-zA-Z0-9][\w-]*", text.lower()):
            if word in seen:
                continue
            if word in STOP_WORDS:
                continue
            if len(word) < 3 and word not in IMPORTANT_SHORT_WORDS:
                continue

            seen.add(word)
            keywords.append(word)

            if len(keywords) == limit:
                break

        return keywords or ["request"]

    def _detect_intent(self, prompt_lower: str) -> tuple[str, str]:
        if any(marker in prompt_lower for marker in ("difference", "compare", "versus", " vs ")):
            return (
                "comparison",
                "Highlight the decision boundary, contrast the trade-offs, and keep the answer grounded in measurable criteria.",
            )
        if any(marker in prompt_lower for marker in ("debug", "bug", "error", "issue", "fix")):
            return (
                "debugging",
                "Reduce the problem to the smallest failing path and isolate where the expected signal diverges.",
            )
        if any(marker in prompt_lower for marker in ("plan", "roadmap", "checklist", "steps")):
            return (
                "planning",
                "Start from the simplest viable path, then add structure only where it changes the result.",
            )
        if any(marker in prompt_lower for marker in ("list", "options", "choices", "bullet")):
            return (
                "enumeration",
                "Surface the main branches cleanly so the next decision is easier to make.",
            )
        if any(marker in prompt_lower for marker in ("summary", "summarize", "overview", "brief")):
            return (
                "summary",
                "Compress the request into the main idea, then preserve only the signals that affect action.",
            )

        return (
            "explanation",
            "Explain the answer directly, then connect it to the practical context that matters most.",
        )

    def _truncate_demo_completion(self, completion: str, max_tokens: int) -> str:
        tokens = self._demo_tokenize(completion)
        if len(tokens) <= max_tokens:
            return completion
        truncated = "".join(tokens[:max_tokens]).rstrip(",;:-")
        return f"{truncated}."

    def _demo_tokenize(self, text: str) -> list[str]:
        return DEMO_TOKEN_PATTERN.findall(text)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))

    def _get_client(self) -> OpenAI | None:
        if not self._settings.openai_api_key:
            return None

        if self._client is None:
            self._client = OpenAI(api_key=self._settings.openai_api_key)

        return self._client


generation_service = GenerationService()
