from __future__ import annotations

import gc
import hashlib
import math
import platform
import re
import shutil
import sys
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from app.core.errors import LLMScopeError
from app.models.provider import ModelProvider
from app.providers.base import (
    DiscoveredModel,
    LLMProvider,
    ProviderCapabilities,
    ProviderDiscoveryResult,
)
from app.schemas.generation import (
    AlternativeCandidate,
    CanonicalPromptToken,
    CanonicalTokenSourceCategory,
    TokenTrace,
)
from app.schemas.huggingface_local import (
    HuggingFaceAttentionAggregationMode,
    HuggingFaceAttentionAnalysisMode,
    HuggingFaceAttentionCategoryBreakdown,
    HuggingFaceAttentionJourneyRow,
    HuggingFaceAttentionLayerJourney,
    HuggingFaceAttentionLayerSummary,
    HuggingFaceAttentionRequest,
    HuggingFaceAttentionResponse,
    HuggingFaceAttentionSequenceScope,
    HuggingFaceAttentionSource,
    HuggingFaceAttentionTokenInfo,
    HuggingFaceLocalDiagnosticsResponse,
    HuggingFaceLocalLimits,
    HuggingFaceLocalModelStatus,
    HuggingFaceLocalState,
    HuggingFaceLocalStatusResponse,
)
from app.schemas.provider_capabilities import ProviderCapabilitiesDetail

try:  # pragma: no cover - optional runtime dependency.
    import torch
except Exception:  # pragma: no cover - optional runtime dependency.
    torch = None

try:  # pragma: no cover - optional runtime dependency.
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional runtime dependency.
    transformers = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:  # pragma: no cover - optional runtime dependency.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )
except Exception:  # pragma: no cover - optional runtime dependency.
    snapshot_download = None

    class GatedRepoError(Exception):
        pass

    class HfHubHTTPError(Exception):
        pass

    class LocalEntryNotFoundError(Exception):
        pass

    class RepositoryNotFoundError(Exception):
        pass


HUGGING_FACE_LOCAL_CAPABILITIES = ProviderCapabilities(
    supports_token_logprobs=True,
    supports_native_continuation=True,
    minimum_output_tokens=1,
    supports_entropy=True,
    supports_attention=True,
    supports_streaming=False,
    supports_branching=True,
    supports_continuation=True,
)
HUGGING_FACE_LOCAL_CAPABILITIES_DETAIL = ProviderCapabilitiesDetail(
    supports_logprobs=True,
    supports_entropy=True,
    supports_attention=True,
    supports_exact_continuation=True,
    supports_streaming=False,
    supports_branching=True,
    supports_continuation=True,
    minimum_output_tokens=1,
)
HUGGING_FACE_LOCAL_PROVIDER_MESSAGE = (
    "Local CUDA analysis is available. Select a supported model and click Load."
)
HUGGING_FACE_LOCAL_MISSING_DEPENDENCY_MESSAGE = (
    "Hugging Face Local Analysis dependencies are missing. "
    "Install transformers, accelerate, and safetensors."
)
HUGGING_FACE_LOCAL_CUDA_UNAVAILABLE_MESSAGE = (
    "CUDA is unavailable. Hugging Face Local Analysis requires an NVIDIA GPU with CUDA enabled."
)
LOCAL_TOP_P_EPSILON = 1e-8
MIN_PROBABILITY = 1e-12
RECOMMENDED_3B_FREE_VRAM_GB = 8.5
DEFAULT_ATTENTION_CONTEXT_LIMIT = 256
MAX_ATTENTION_CONTEXT_LIMIT = 512
ATTENTION_CACHE_SIZE = 48


@dataclass(frozen=True)
class SupportedLocalModel:
    id: str
    label: str
    revision: str | None


@dataclass
class LoadedLocalRuntime:
    model_id: str
    label: str
    revision: str | None
    resolved_revision: str | None
    tokenizer: Any
    model: Any
    device: str
    dtype: str
    num_hidden_layers: int | None
    num_attention_heads: int | None
    attention_implementation: str | None


@dataclass(frozen=True)
class AttentionTokenMetadata:
    token_id: int
    raw_token: str
    display_token: str
    decoded_contribution: str
    token_bytes: list[int]
    full_position: int
    analyzed_position: int
    sequence_scope: HuggingFaceAttentionSequenceScope
    source_category: CanonicalTokenSourceCategory
    source_label: str
    special_token: bool
    generated_token_index: int | None


@dataclass(frozen=True)
class RenderedChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class CanonicalPromptBuildResult:
    prompt_ids: list[int]
    prompt_tokens: list[CanonicalPromptToken]
    raw_context_text: str | None
    system_prompt: str | None


@dataclass(frozen=True)
class HuggingFaceGenerationResult:
    completion: str
    prompt_token_ids: list[int] | None
    prompt_tokens: list[CanonicalPromptToken] | None
    raw_context_text: str | None
    system_prompt: str | None
    tokens: list[TokenTrace]
    prompt_token_count: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    finish_reason: str | None
    resolved_revision: str | None
    device: str
    dtype: str


@dataclass(frozen=True)
class HuggingFaceRuntimeIdentity:
    model_id: str
    model_revision: str | None
    tokenizer_identity: str | None
    tokenizer_revision: str | None


def collect_huggingface_local_diagnostics(
    *,
    selected_device: str | None = None,
    selected_dtype: str | None = None,
) -> HuggingFaceLocalDiagnosticsResponse:
    missing_dependencies: list[str] = []
    torch_version: str | None = None
    transformers_version: str | None = None
    torch_cuda_runtime: str | None = None
    cuda_available = False
    gpu_name: str | None = None
    gpu_total_vram_gb: float | None = None
    gpu_free_vram_gb: float | None = None

    if torch is None:
        missing_dependencies.append("torch")
    else:
        torch_version = getattr(torch, "__version__", None)
        torch_cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            gpu_total_vram_gb = round(total_mem / (1024**3), 2)
            gpu_free_vram_gb = round(free_mem / (1024**3), 2)

    if transformers is None:
        missing_dependencies.append("transformers")
    else:
        transformers_version = getattr(transformers, "__version__", None)

    if AutoModelForCausalLM is None or AutoTokenizer is None:
        if "transformers" not in missing_dependencies:
            missing_dependencies.append("transformers")

    if snapshot_download is None and "huggingface_hub" not in missing_dependencies:
        missing_dependencies.append("huggingface_hub")

    disk_free_gb: float | None = None
    try:
        free_bytes = shutil.disk_usage(Path.cwd().anchor).free
        disk_free_gb = round(free_bytes / (1024**3), 2)
    except Exception:
        disk_free_gb = None

    return HuggingFaceLocalDiagnosticsResponse(
        cuda_available=cuda_available,
        selected_device=selected_device or ("cuda:0" if cuda_available else None),
        selected_dtype=selected_dtype or ("float16" if cuda_available else None),
        torch_version=torch_version,
        transformers_version=transformers_version,
        torch_cuda_runtime=torch_cuda_runtime,
        gpu_name=gpu_name,
        gpu_total_vram_gb=gpu_total_vram_gb,
        gpu_free_vram_gb=gpu_free_vram_gb,
        python_version=sys.version.splitlines()[0],
        platform=platform.platform(),
        disk_free_gb=disk_free_gb,
        missing_dependencies=missing_dependencies,
    )


class HuggingFaceLocalProvider(LLMProvider):
    provider_name = ModelProvider.HUGGING_FACE
    provider_label = "Hugging Face Local"
    capabilities = HUGGING_FACE_LOCAL_CAPABILITIES

    def __init__(
        self,
        *,
        default_model: str,
        model_revisions: dict[str, str | None],
        hf_token: str,
        context_limit: int,
        default_output_tokens: int,
        max_output_tokens: int,
        stored_top_alternatives: int = 5,
    ) -> None:
        self._default_model = default_model
        self._hf_token = hf_token or ""
        self._context_limit = context_limit
        self._default_output_tokens = default_output_tokens
        self._max_output_tokens = max_output_tokens
        self._stored_top_alternatives = stored_top_alternatives
        self._supported_models = [
            SupportedLocalModel(
                id="Qwen/Qwen2.5-3B-Instruct",
                label="Qwen2.5 3B Instruct",
                revision=self._normalize_revision(
                    model_revisions.get("Qwen/Qwen2.5-3B-Instruct")
                ),
            ),
            SupportedLocalModel(
                id="Qwen/Qwen2.5-1.5B-Instruct",
                label="Qwen2.5 1.5B Instruct",
                revision=self._normalize_revision(
                    model_revisions.get("Qwen/Qwen2.5-1.5B-Instruct")
                ),
            ),
        ]
        self._runtime: LoadedLocalRuntime | None = None
        self._runtime_lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._busy_lock = threading.Lock()
        self._attention_cache: OrderedDict[str, HuggingFaceAttentionResponse] = OrderedDict()
        self._attention_cache_lock = threading.Lock()
        self._status_state = HuggingFaceLocalState.READY
        self._status_message = HUGGING_FACE_LOCAL_PROVIDER_MESSAGE

    @property
    def supported_model_ids(self) -> set[str]:
        return {model.id for model in self._supported_models}

    def discover_models(self, *, force_refresh: bool = False) -> ProviderDiscoveryResult:
        status = self.get_status()
        return ProviderDiscoveryResult(
            provider_name=self.provider_name,
            provider_label=self.provider_label,
            status=status.status.value,
            status_message=status.status_message,
            recommended_models=[status.recommended_model_id] if status.recommended_model_id else [],
            capabilities=self.capabilities,
            models=[
                DiscoveredModel(
                    id=model.id,
                    label=model.label,
                    status=model.status.value,
                    status_message=model.status_message,
                )
                for model in status.models
            ],
        )

    def get_status(self) -> HuggingFaceLocalStatusResponse:
        diagnostics = collect_huggingface_local_diagnostics(
            selected_device=self._runtime.device if self._runtime else None,
            selected_dtype=self._runtime.dtype if self._runtime else None,
        )
        missing_dependencies = diagnostics.missing_dependencies
        cuda_available = diagnostics.cuda_available
        recommended_model_id = self._recommended_model_id(
            diagnostics.gpu_free_vram_gb,
            diagnostics.gpu_total_vram_gb,
        )
        runtime = self._runtime

        if missing_dependencies:
            provider_state = HuggingFaceLocalState.ERROR
            provider_message = HUGGING_FACE_LOCAL_MISSING_DEPENDENCY_MESSAGE
        elif not cuda_available:
            provider_state = HuggingFaceLocalState.CUDA_UNAVAILABLE
            provider_message = HUGGING_FACE_LOCAL_CUDA_UNAVAILABLE_MESSAGE
        else:
            provider_state = self._status_state
            provider_message = self._status_message or HUGGING_FACE_LOCAL_PROVIDER_MESSAGE

        if runtime is None and provider_state == HuggingFaceLocalState.READY:
            provider_message = HUGGING_FACE_LOCAL_PROVIDER_MESSAGE

        models = [
            self._build_model_status(
                model_spec=model_spec,
                recommended_model_id=recommended_model_id,
                missing_dependencies=missing_dependencies,
                cuda_available=cuda_available,
                runtime=runtime,
            )
            for model_spec in self._ordered_supported_models(recommended_model_id)
        ]

        return HuggingFaceLocalStatusResponse(
            label=self.provider_label,
            status=provider_state,
            status_message=provider_message,
            capabilities=HUGGING_FACE_LOCAL_CAPABILITIES_DETAIL,
            cuda_available=cuda_available,
            busy=self._busy_lock.locked(),
            device=runtime.device if runtime else diagnostics.selected_device,
            precision=runtime.dtype if runtime else diagnostics.selected_dtype,
            torch_version=diagnostics.torch_version,
            transformers_version=diagnostics.transformers_version,
            gpu_name=diagnostics.gpu_name,
            gpu_total_vram_gb=diagnostics.gpu_total_vram_gb,
            gpu_free_vram_gb=diagnostics.gpu_free_vram_gb,
            active_model_id=runtime.model_id if runtime else None,
            active_model_label=runtime.label if runtime else None,
            active_model_revision=runtime.revision if runtime else None,
            active_model_resolved_revision=runtime.resolved_revision if runtime else None,
            active_model_num_hidden_layers=runtime.num_hidden_layers if runtime else None,
            active_model_num_attention_heads=runtime.num_attention_heads if runtime else None,
            active_model_attention_implementation=runtime.attention_implementation if runtime else None,
            recommended_model_id=recommended_model_id,
            missing_dependencies=missing_dependencies,
            limits=HuggingFaceLocalLimits(
                context_window_tokens=self._context_limit,
                default_output_tokens=self._default_output_tokens,
                max_output_tokens=self._max_output_tokens,
                stored_top_alternatives=self._stored_top_alternatives,
            ),
            models=models,
        )

    def get_diagnostics(self) -> HuggingFaceLocalDiagnosticsResponse:
        runtime = self._runtime
        return collect_huggingface_local_diagnostics(
            selected_device=runtime.device if runtime else None,
            selected_dtype=runtime.dtype if runtime else None,
        )

    def load_model(self, model_id: str) -> HuggingFaceLocalStatusResponse:
        model_spec = self._resolve_model(model_id)
        self._ensure_runtime_dependencies()
        self._ensure_cuda_available()

        if self._busy_lock.locked():
            raise LLMScopeError(
                code="HF_LOCAL_BUSY",
                message="The local analysis model is busy. Wait for the current generation to finish.",
                status_code=503,
            )

        with self._load_lock:
            runtime = self._runtime
            if runtime is not None and runtime.model_id == model_spec.id:
                self._set_status(
                    HuggingFaceLocalState.READY,
                    f"{model_spec.label} is already loaded on {runtime.device}.",
                )
                return self.get_status()

            self.unload_model(raise_if_busy=False)
            was_downloaded = self._is_model_downloaded(model_spec)
            self._set_status(
                HuggingFaceLocalState.LOADING if was_downloaded else HuggingFaceLocalState.DOWNLOADING,
                f"Preparing {model_spec.label} on CUDA.",
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_spec.id,
                    revision=model_spec.revision,
                    trust_remote_code=False,
                    token=self._hf_token or None,
                    use_fast=True,
                )
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = self._load_transformers_model(model_spec)
                model.eval()
                device = str(next(model.parameters()).device)
                dtype = self._normalize_dtype_name(next(model.parameters()).dtype)
                resolved_revision = (
                    getattr(getattr(model, "config", None), "_commit_hash", None)
                    or tokenizer.init_kwargs.get("_commit_hash")
                    or model_spec.revision
                )
                self._runtime = LoadedLocalRuntime(
                    model_id=model_spec.id,
                    label=model_spec.label,
                    revision=model_spec.revision,
                    resolved_revision=resolved_revision,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    dtype=dtype,
                    num_hidden_layers=self._config_int(getattr(model, "config", None), "num_hidden_layers"),
                    num_attention_heads=self._config_int(
                        getattr(model, "config", None), "num_attention_heads"
                    ),
                    attention_implementation=self._attention_implementation_name(model),
                )
                self._set_status(
                    HuggingFaceLocalState.READY,
                    f"{model_spec.label} is loaded on {device} ({dtype}).",
                )
                return self.get_status()
            except Exception as exc:
                self._cleanup_runtime()
                self._raise_load_exception(exc, model_spec)

        return self.get_status()

    def unload_model(self, *, raise_if_busy: bool = True) -> HuggingFaceLocalStatusResponse:
        if raise_if_busy and self._busy_lock.locked():
            raise LLMScopeError(
                code="HF_LOCAL_BUSY",
                message="The local analysis model is busy. Wait for the current generation to finish before unloading.",
                status_code=503,
            )

        with self._runtime_lock:
            self._cleanup_runtime()
            self._set_status(
                HuggingFaceLocalState.READY,
                "The local analysis model has been unloaded. Select a model and click Load.",
            )

        return self.get_status()

    def get_runtime_identity(self, model_id: str) -> HuggingFaceRuntimeIdentity:
        runtime = self._require_loaded_runtime(model_id)
        return HuggingFaceRuntimeIdentity(
            model_id=runtime.model_id,
            model_revision=runtime.resolved_revision or runtime.revision,
            tokenizer_identity=self._tokenizer_identity(runtime.tokenizer, runtime.model_id),
            tokenizer_revision=self._tokenizer_revision(runtime.tokenizer, runtime),
        )

    def _build_canonical_prompt_tokens(
        self,
        tokenizer: Any,
        *,
        prompt: str,
    ) -> CanonicalPromptBuildResult:
        messages = [{"role": "user", "content": prompt}]
        prompt_ids_tensor = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        prompt_ids = [int(value) for value in prompt_ids_tensor[0].tolist()]
        formatted_prompt_text = self._render_chat_template_text(tokenizer, messages)
        rendered_messages = self._extract_rendered_chat_messages(formatted_prompt_text or "")
        special_token_ids = {
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", [])
            if isinstance(token_id, int)
        }
        category_by_position = {
            full_position: CanonicalTokenSourceCategory.TEMPLATE
            for full_position in range(len(prompt_ids))
        }
        search_start = 0

        for rendered_message in rendered_messages:
            category = self._source_category_for_message_role(rendered_message.role)
            if category is None or not rendered_message.content:
                continue

            content_ids = self._tokenizer_content_ids(tokenizer, rendered_message.content)
            content_span = self._find_subsequence_from_index(
                sequence=prompt_ids,
                target=content_ids,
                start_index=search_start,
            )

            if content_span is None:
                continue

            for full_position in range(content_span[0], content_span[1]):
                category_by_position[full_position] = category

            search_start = content_span[1]

        if not rendered_messages:
            prompt_content_ids = self._tokenizer_content_ids(tokenizer, prompt)
            prompt_span = self._find_subsequence(prompt_ids, prompt_content_ids)
            if prompt_span is not None:
                for full_position in range(prompt_span[0], prompt_span[1]):
                    category_by_position[full_position] = (
                        CanonicalTokenSourceCategory.USER_PROMPT
                    )

        prompt_tokens: list[CanonicalPromptToken] = []

        for full_position, token_id in enumerate(prompt_ids):
            raw_token = self._raw_token(tokenizer, token_id)
            decoded_contribution = self._decode_token(tokenizer, token_id)
            special_token = token_id in special_token_ids
            source_category = category_by_position.get(
                full_position,
                CanonicalTokenSourceCategory.TEMPLATE,
            )
            source_label = self._source_label_for_category(source_category)

            prompt_tokens.append(
                CanonicalPromptToken(
                    token_id=token_id,
                    raw_token=raw_token,
                    display_token=self._display_token(decoded_contribution),
                    decoded_contribution=decoded_contribution,
                    token_bytes=self._token_bytes(decoded_contribution, raw_token),
                    full_position=full_position,
                    source_category=source_category,
                    source_label=source_label,
                    special_token=special_token,
                )
            )

        system_messages = [
            rendered_message.content
            for rendered_message in rendered_messages
            if rendered_message.role == "system" and rendered_message.content
        ]
        system_prompt = "\n\n".join(system_messages) if system_messages else None

        return CanonicalPromptBuildResult(
            prompt_ids=prompt_ids,
            prompt_tokens=prompt_tokens,
            raw_context_text=formatted_prompt_text,
            system_prompt=system_prompt,
        )

    def _render_chat_template_text(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
    ) -> str | None:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    return_tensors=None,
                )
            except Exception:
                return None
        except Exception:
            return None

        return rendered if isinstance(rendered, str) else None

    def _extract_rendered_chat_messages(
        self,
        rendered_text: str,
    ) -> list[RenderedChatMessage]:
        if not rendered_text:
            return []

        pattern = re.compile(
            r"<\|im_start\|>(?P<role>[^\n]+)\n(?P<content>.*?)(?:<\|im_end\|>\n?|$)",
            re.DOTALL,
        )
        messages: list[RenderedChatMessage] = []

        for match in pattern.finditer(rendered_text):
            role = match.group("role").strip().lower()
            content = match.group("content")
            messages.append(
                RenderedChatMessage(
                    role=role,
                    content=content,
                )
            )

        return messages

    def _source_category_for_message_role(
        self,
        role: str,
    ) -> CanonicalTokenSourceCategory | None:
        if role == "system":
            return CanonicalTokenSourceCategory.SYSTEM
        if role == "user":
            return CanonicalTokenSourceCategory.USER_PROMPT
        if role == "assistant":
            return CanonicalTokenSourceCategory.ASSISTANT_PREFIX
        return None

    def _source_label_for_category(
        self,
        category: CanonicalTokenSourceCategory,
    ) -> str:
        if category == CanonicalTokenSourceCategory.SYSTEM:
            return "System message"
        if category == CanonicalTokenSourceCategory.USER_PROMPT:
            return "User prompt"
        if category == CanonicalTokenSourceCategory.ASSISTANT_PREFIX:
            return "Assistant prefix"
        return "Template / control"

    def _tokenizer_content_ids(self, tokenizer: Any, text: str) -> list[int]:
        if not text:
            return []

        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except Exception:
            return []

        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if input_ids is None:
            return []
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return [int(value) for value in input_ids]

    def _find_subsequence(
        self,
        sequence: list[int],
        target: list[int],
    ) -> tuple[int, int] | None:
        if not target or len(target) > len(sequence):
            return None

        last_start = len(sequence) - len(target)
        for start in range(last_start + 1):
            if sequence[start : start + len(target)] == target:
                return start, start + len(target)

        return None

    def _find_subsequence_from_index(
        self,
        *,
        sequence: list[int],
        target: list[int],
        start_index: int,
    ) -> tuple[int, int] | None:
        if not target or len(target) > len(sequence):
            return None

        last_start = len(sequence) - len(target)
        effective_start = max(0, min(start_index, last_start))
        for start in range(effective_start, last_start + 1):
            if sequence[start : start + len(target)] == target:
                return start, start + len(target)

        return self._find_subsequence(sequence, target)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        assistant_prefix: str,
        branch_id: str,
        parent_node_id: str,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        max_candidates: int,
        canonical_prefix_token_ids: list[int] | None = None,
        prompt_token_ids: list[int] | None = None,
        repetition_penalty: float | None = None,
        top_k: int | None = None,
    ) -> HuggingFaceGenerationResult:
        model_spec = self._resolve_model(model)
        runtime = self._require_loaded_runtime(model_spec.id)
        self._ensure_cuda_available()
        effective_max_output_tokens = max(1, min(max_output_tokens, self._max_output_tokens))
        effective_repetition_penalty = repetition_penalty if repetition_penalty and repetition_penalty > 0 else 1.0
        effective_top_k = top_k if top_k and top_k > 0 else None
        effective_candidates = max(1, max_candidates)

        if not self._busy_lock.acquire(blocking=False):
            self._set_status(
                HuggingFaceLocalState.BUSY,
                f"{runtime.label} is busy. Wait for the current local generation to finish.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_BUSY",
                message="The local analysis model is busy. Wait for the current generation to finish.",
                status_code=503,
            )

        self._set_status(
            HuggingFaceLocalState.BUSY,
            f"Generating with {runtime.label} on {runtime.device}.",
        )

        try:
            (
                full_prefix_ids,
                prompt_ids_for_result,
                generated_text_prefix,
                prompt_token_metadata,
                raw_context_text,
                system_prompt,
            ) = self._resolve_generation_prefix(
                runtime=runtime,
                prompt=prompt,
                assistant_prefix=assistant_prefix,
                canonical_prefix_token_ids=canonical_prefix_token_ids,
                prompt_token_ids=prompt_token_ids,
            )

            if len(full_prefix_ids) > self._context_limit:
                raise LLMScopeError(
                    code="HF_CONTEXT_TOO_LONG",
                    message=(
                        f"The selected prefix uses {len(full_prefix_ids)} tokens, which exceeds the "
                        f"configured local context limit of {self._context_limit}."
                    ),
                    status_code=400,
                )

            start_time = perf_counter()
            traces, finish_reason = self._run_generation_loop(
                runtime=runtime,
                branch_id=branch_id,
                parent_node_id=parent_node_id,
                full_prefix_ids=full_prefix_ids,
                generated_text_prefix=generated_text_prefix,
                max_output_tokens=effective_max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=effective_top_k,
                repetition_penalty=effective_repetition_penalty,
                max_candidates=effective_candidates,
            )
            latency_ms = max(1, int((perf_counter() - start_time) * 1000))
            per_token_latency = max(1, int(round(latency_ms / max(len(traces), 1))))
            for trace in traces:
                trace.latency_ms = per_token_latency
                for alternative in trace.alternatives:
                    alternative.latency_ms = per_token_latency
            completion = traces[-1].cumulative_decoded_text if traces else ""
            prompt_token_count = len(prompt_ids_for_result or full_prefix_ids)
            completion_tokens = len(traces)

            self._set_status(
                HuggingFaceLocalState.READY,
                f"{runtime.label} is loaded on {runtime.device} ({runtime.dtype}).",
            )
            return HuggingFaceGenerationResult(
                completion=completion,
                prompt_token_ids=prompt_ids_for_result,
                prompt_tokens=prompt_token_metadata,
                raw_context_text=raw_context_text,
                system_prompt=system_prompt,
                tokens=traces,
                prompt_token_count=prompt_token_count,
                completion_tokens=completion_tokens,
                total_tokens=prompt_token_count + completion_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                resolved_revision=runtime.resolved_revision,
                device=runtime.device,
                dtype=runtime.dtype,
            )
        except LLMScopeError:
            self._set_status(
                HuggingFaceLocalState.READY,
                f"{runtime.label} is loaded on {runtime.device} ({runtime.dtype}).",
            )
            raise
        except RuntimeError as exc:
            if self._looks_like_oom(exc):
                self._cleanup_cuda_after_oom()
                self._set_status(
                    HuggingFaceLocalState.OOM,
                    "CUDA ran out of memory. Reduce context/output length or load the 1.5B model.",
                )
                raise LLMScopeError(
                    code="HF_LOCAL_OOM",
                    message=(
                        "CUDA ran out of memory while generating locally. Reduce context/output "
                        "length or load Qwen/Qwen2.5-1.5B-Instruct."
                    ),
                    status_code=503,
                ) from exc
            self._set_status(
                HuggingFaceLocalState.ERROR,
                "The local analysis generation failed unexpectedly.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_GENERATION_FAILED",
                message="The local analysis generation failed unexpectedly.",
                status_code=500,
            ) from exc
        except Exception as exc:
            self._set_status(
                HuggingFaceLocalState.ERROR,
                "The local analysis generation failed unexpectedly.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_GENERATION_FAILED",
                message="The local analysis generation failed unexpectedly.",
                status_code=500,
            ) from exc
        finally:
            self._busy_lock.release()

    def analyze_attention(
        self,
        request: HuggingFaceAttentionRequest,
    ) -> HuggingFaceAttentionResponse:
        assert torch is not None

        model_spec = self._resolve_model(request.model_id)
        runtime = self._require_loaded_runtime(model_spec.id)
        self._ensure_cuda_available()
        runtime_identity = self.get_runtime_identity(model_spec.id)
        self._validate_attention_request(request, runtime, runtime_identity)
        full_sequence_ids = [*request.prompt_token_ids, *request.generated_token_ids]
        selected_token_position, query_position = self._resolve_attention_positions(request)
        cache_key = self._make_attention_cache_key(
            request,
            full_sequence_ids=full_sequence_ids,
            selected_token_position=selected_token_position,
            query_position=query_position,
        )
        cached = self._get_cached_attention(cache_key)
        if cached is not None:
            return cached

        if not self._busy_lock.acquire(blocking=False):
            self._set_status(
                HuggingFaceLocalState.BUSY,
                f"{runtime.label} is busy. Wait for the current local task to finish before computing attention.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_BUSY",
                message="The local analysis model is busy. Wait for the current local task to finish.",
                status_code=503,
            )

        self._set_status(
            HuggingFaceLocalState.BUSY,
            f"Computing attention with {runtime.label} on {runtime.device}.",
        )

        outputs = None
        attention_tuple = None
        selected_attention_row = None
        aggregated_row = None
        try:
            (
                analyzed_ids,
                window_start,
                context_truncated,
            ) = self._select_attention_context_window(
                full_sequence_ids=full_sequence_ids,
                query_position=query_position,
                selected_token_position=selected_token_position,
                max_context_tokens=request.max_context_tokens,
                allow_truncated_recompute=request.allow_truncated_recompute,
            )
            query_position_in_window = query_position - window_start
            selected_position_in_window = selected_token_position - window_start
            original_attention_implementation = self._attention_implementation_name(runtime.model)

            with self._runtime_lock:
                self._set_attention_implementation(runtime.model, "eager")
                try:
                    device = next(runtime.model.parameters()).device
                    input_ids = torch.tensor([analyzed_ids], dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids, device=device)

                    with torch.inference_mode():
                        outputs = runtime.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            output_attentions=True,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    attention_tuple = getattr(outputs, "attentions", None)
                finally:
                    self._set_attention_implementation(runtime.model, original_attention_implementation)

            if not attention_tuple:
                raise LLMScopeError(
                    code="HF_ATTENTION_UNAVAILABLE",
                    message=(
                        "Attention mode is unavailable for the loaded Hugging Face Local runtime. "
                        "Reload the model in an attention-compatible configuration and try again."
                    ),
                    status_code=503,
                )

            num_layers = len(attention_tuple)
            comparison_layers = self._normalize_requested_attention_layers(
                request.comparison_layers,
                num_layers=num_layers,
                error_code="HF_ATTENTION_COMPARISON_LAYER_OUT_OF_RANGE",
                label="comparison layer",
            )
            journey_layers = self._normalize_requested_attention_layers(
                request.journey_layers,
                num_layers=num_layers,
                error_code="HF_ATTENTION_JOURNEY_LAYER_OUT_OF_RANGE",
                label="journey layer",
            )
            if request.selected_layer >= num_layers:
                raise LLMScopeError(
                    code="HF_ATTENTION_LAYER_OUT_OF_RANGE",
                    message=f"Layer {request.selected_layer} is out of range for this model.",
                    status_code=400,
                )

            requested_layers = self._merge_requested_attention_layers(
                request.selected_layer,
                comparison_layers,
                journey_layers,
            )
            layer_rows: dict[int, Any] = {}
            token_metadata = self._build_attention_token_metadata(
                request=request,
                runtime=runtime,
                analyzed_ids=analyzed_ids,
                window_start=window_start,
            )
            num_query_heads = 0

            for layer_index in requested_layers:
                layer_attention = attention_tuple[layer_index]
                if layer_attention is None or layer_attention.ndim != 4:
                    raise LLMScopeError(
                        code="HF_ATTENTION_UNAVAILABLE",
                        message="The loaded Hugging Face Local runtime did not return a usable attention tensor.",
                        status_code=503,
                    )

                current_num_query_heads = int(layer_attention.shape[1])
                if num_query_heads == 0:
                    num_query_heads = current_num_query_heads
                if request.selected_head is not None and request.selected_head >= current_num_query_heads:
                    raise LLMScopeError(
                        code="HF_ATTENTION_HEAD_OUT_OF_RANGE",
                        message=f"Head {request.selected_head} is out of range for layer {layer_index}.",
                        status_code=400,
                    )

                selected_attention_row = (
                    layer_attention[0, :, query_position_in_window, : query_position_in_window + 1]
                    .detach()
                    .to(torch.float32)
                    .cpu()
                )
                self._validate_attention_row(
                    selected_attention_row,
                    request=request,
                    runtime=runtime,
                    query_position=query_position,
                )
                aggregated_candidate = self._aggregate_attention_row(
                    selected_attention_row,
                    aggregation_mode=request.aggregation_mode,
                    selected_head=request.selected_head,
                )
                self._validate_aggregated_attention_row(
                    aggregated_candidate,
                    request=request,
                    runtime=runtime,
                    query_position=query_position,
                )
                layer_rows[layer_index] = aggregated_candidate

            aggregated_row = layer_rows[request.selected_layer]

            response = self._build_attention_response(
                request=request,
                runtime=runtime,
                full_sequence_ids=full_sequence_ids,
                analyzed_ids=analyzed_ids,
                token_metadata=token_metadata,
                selected_token_position=selected_token_position,
                selected_position_in_window=selected_position_in_window,
                query_position=query_position,
                query_position_in_window=query_position_in_window,
                aggregated_attention_row=aggregated_row,
                comparison_layers=comparison_layers,
                comparison_rows=layer_rows,
                journey_layers=journey_layers,
                num_layers=num_layers,
                num_query_heads=num_query_heads,
                context_truncated=context_truncated,
                window_start=window_start,
                attention_implementation_used="eager",
            )
            self._store_attention(cache_key, response)
            self._set_status(
                HuggingFaceLocalState.READY,
                f"{runtime.label} is loaded on {runtime.device} ({runtime.dtype}).",
            )
            return response
        except LLMScopeError:
            self._set_status(
                HuggingFaceLocalState.READY,
                f"{runtime.label} is loaded on {runtime.device} ({runtime.dtype}).",
            )
            raise
        except RuntimeError as exc:
            if self._looks_like_oom(exc):
                self._cleanup_cuda_after_oom()
                self._set_status(
                    HuggingFaceLocalState.OOM,
                    "CUDA ran out of memory during attention analysis. Reduce the context limit or use the 1.5B model.",
                )
                raise LLMScopeError(
                    code="HF_ATTENTION_OOM",
                    message=(
                        "CUDA ran out of memory while computing attention. Try a smaller attention context limit, "
                        "use Qwen/Qwen2.5-1.5B-Instruct, or unload other GPU-heavy applications."
                    ),
                    status_code=503,
                ) from exc
            self._set_status(
                HuggingFaceLocalState.ERROR,
                "The local attention analysis failed unexpectedly.",
            )
            raise LLMScopeError(
                code="HF_ATTENTION_FAILED",
                message="The local attention analysis failed unexpectedly.",
                status_code=500,
            ) from exc
        except Exception as exc:
            self._set_status(
                HuggingFaceLocalState.ERROR,
                "The local attention analysis failed unexpectedly.",
            )
            raise LLMScopeError(
                code="HF_ATTENTION_FAILED",
                message="The local attention analysis failed unexpectedly.",
                status_code=500,
            ) from exc
        finally:
            del outputs
            del attention_tuple
            del selected_attention_row
            del aggregated_row
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._busy_lock.release()

    def _validate_attention_request(
        self,
        request: HuggingFaceAttentionRequest,
        runtime: LoadedLocalRuntime,
        runtime_identity: HuggingFaceRuntimeIdentity,
    ) -> None:
        if request.model_revision and request.model_revision != runtime_identity.model_revision:
            raise LLMScopeError(
                code="HF_ATTENTION_MODEL_REVISION_MISMATCH",
                message="The selected graph was produced by a different loaded model revision.",
                status_code=409,
            )

        if (
            request.tokenizer_identity
            and runtime_identity.tokenizer_identity
            and request.tokenizer_identity != runtime_identity.tokenizer_identity
        ):
            raise LLMScopeError(
                code="HF_ATTENTION_TOKENIZER_MISMATCH",
                message="The selected graph was produced by a different tokenizer identity.",
                status_code=409,
            )

        if (
            request.tokenizer_revision
            and runtime_identity.tokenizer_revision
            and request.tokenizer_revision != runtime_identity.tokenizer_revision
        ):
            raise LLMScopeError(
                code="HF_ATTENTION_TOKENIZER_MISMATCH",
                message="The selected graph was produced by a different tokenizer revision.",
                status_code=409,
            )

        if len(request.prompt_tokens) != len(request.prompt_token_ids):
            raise LLMScopeError(
                code="HF_ATTENTION_PROMPT_TOKEN_METADATA_MISMATCH",
                message=(
                    "Canonical prompt-token metadata is missing or does not match the formatted prompt token-ID sequence."
                ),
                status_code=400,
            )

        for full_position, prompt_token in enumerate(request.prompt_tokens):
            if prompt_token.full_position != full_position:
                raise LLMScopeError(
                    code="HF_ATTENTION_PROMPT_TOKEN_POSITION_MISMATCH",
                    message="Canonical prompt-token positions are not contiguous from the formatted prompt prefix.",
                    status_code=400,
                )
            if prompt_token.token_id != request.prompt_token_ids[full_position]:
                raise LLMScopeError(
                    code="HF_ATTENTION_PROMPT_TOKEN_ID_MISMATCH",
                    message="Canonical prompt-token IDs do not match the formatted prompt token-ID prefix.",
                    status_code=400,
                )

        if not request.generated_token_ids:
            raise LLMScopeError(
                code="HF_ATTENTION_GENERATED_IDS_REQUIRED",
                message="The selected token lacks canonical generated token IDs for attention analysis.",
                status_code=400,
            )

        if request.selected_generated_token_index >= len(request.generated_token_ids):
            raise LLMScopeError(
                code="HF_ATTENTION_INDEX_OUT_OF_RANGE",
                message="The selected generated-token index is outside the canonical generated token sequence.",
                status_code=400,
            )

        if runtime.num_hidden_layers is not None and request.selected_layer >= runtime.num_hidden_layers:
            raise LLMScopeError(
                code="HF_ATTENTION_LAYER_OUT_OF_RANGE",
                message=f"Layer {request.selected_layer} is out of range for the loaded model.",
                status_code=400,
            )

        if request.aggregation_mode == HuggingFaceAttentionAggregationMode.SINGLE_HEAD and request.selected_head is None:
            raise LLMScopeError(
                code="HF_ATTENTION_HEAD_REQUIRED",
                message="An individual attention head must be selected when using single-head attention mode.",
                status_code=400,
            )

        if request.aggregation_mode != HuggingFaceAttentionAggregationMode.SINGLE_HEAD and request.selected_head is not None:
            raise LLMScopeError(
                code="HF_ATTENTION_HEAD_MODE_CONFLICT",
                message="Choose either a single head or an aggregation mode, but not both.",
                status_code=400,
            )

        if runtime.num_attention_heads is not None and request.selected_head is not None:
            if request.selected_head >= runtime.num_attention_heads:
                raise LLMScopeError(
                    code="HF_ATTENTION_HEAD_OUT_OF_RANGE",
                    message=f"Head {request.selected_head} is out of range for the loaded model.",
                    status_code=400,
                )

    def _resolve_attention_positions(
        self,
        request: HuggingFaceAttentionRequest,
    ) -> tuple[int, int]:
        prompt_token_count = len(request.prompt_token_ids)
        selected_token_position = prompt_token_count + request.selected_generated_token_index
        query_position = (
            selected_token_position - 1
            if request.analysis_mode == HuggingFaceAttentionAnalysisMode.PREDICTION
            else selected_token_position
        )
        if query_position < 0:
            raise LLMScopeError(
                code="HF_ATTENTION_QUERY_POSITION_INVALID",
                message=(
                    "Prediction attention for the first generated token requires canonical prompt/chat-template tokens."
                ),
                status_code=400,
            )
        return selected_token_position, query_position

    def _select_attention_context_window(
        self,
        *,
        full_sequence_ids: list[int],
        query_position: int,
        selected_token_position: int,
        max_context_tokens: int,
        allow_truncated_recompute: bool,
    ) -> tuple[list[int], int, bool]:
        prefix_through_selected = full_sequence_ids[: selected_token_position + 1]
        if len(prefix_through_selected) <= max_context_tokens:
            return prefix_through_selected, 0, False

        if not allow_truncated_recompute:
            raise LLMScopeError(
                code="HF_ATTENTION_CONTEXT_TOO_LONG",
                message=(
                    f"Sequence too long for exact attention analysis. The canonical prefix through the selected token "
                    f"uses {len(prefix_through_selected)} tokens, which exceeds the analysis limit of "
                    f"{max_context_tokens}. Enable truncated recomputation explicitly if you want an approximate "
                    "local attention view."
                ),
                status_code=400,
            )

        window_end = len(prefix_through_selected)
        window_start = max(0, window_end - max_context_tokens)
        if query_position < window_start:
            raise LLMScopeError(
                code="HF_ATTENTION_CONTEXT_TOO_LONG",
                message=(
                    f"Sequence too long for attention analysis at the selected query position. The retained "
                    f"{max_context_tokens}-token window would exclude the required canonical query row."
                ),
                status_code=400,
            )
        return prefix_through_selected[window_start:window_end], window_start, True

    def _set_attention_implementation(self, model: Any, implementation: str | None) -> None:
        config = getattr(model, "config", None)
        if config is None:
            return
        if implementation is None:
            return
        setattr(config, "_attn_implementation", implementation)

    def _aggregate_attention_row(
        self,
        attention_row: Any,
        *,
        aggregation_mode: HuggingFaceAttentionAggregationMode,
        selected_head: int | None,
    ) -> Any:
        assert torch is not None
        if aggregation_mode == HuggingFaceAttentionAggregationMode.SINGLE_HEAD:
            if selected_head is None:
                raise LLMScopeError(
                    code="HF_ATTENTION_HEAD_REQUIRED",
                    message="An attention head must be selected for single-head mode.",
                    status_code=400,
                )
            return attention_row[selected_head]
        if aggregation_mode == HuggingFaceAttentionAggregationMode.MAX_HEADS:
            return torch.max(attention_row, dim=0).values
        return torch.mean(attention_row, dim=0)

    def _normalize_requested_attention_layers(
        self,
        layers: list[int],
        *,
        num_layers: int,
        error_code: str,
        label: str,
    ) -> list[int]:
        normalized: list[int] = []
        seen: set[int] = set()

        for layer_index in layers:
            if layer_index < 0 or layer_index >= num_layers:
                raise LLMScopeError(
                    code=error_code,
                    message=f"The requested {label} {layer_index} is out of range for this model.",
                    status_code=400,
                )
            if layer_index in seen:
                continue
            seen.add(layer_index)
            normalized.append(layer_index)

        return normalized

    def _merge_requested_attention_layers(
        self,
        selected_layer: int,
        comparison_layers: list[int],
        journey_layers: list[int],
    ) -> list[int]:
        merged: list[int] = []
        seen: set[int] = set()

        for layer_index in [selected_layer, *comparison_layers, *journey_layers]:
            if layer_index in seen:
                continue
            seen.add(layer_index)
            merged.append(layer_index)

        return merged

    def _build_attention_token_metadata(
        self,
        *,
        request: HuggingFaceAttentionRequest,
        runtime: LoadedLocalRuntime,
        analyzed_ids: list[int],
        window_start: int,
    ) -> list[AttentionTokenMetadata]:
        prompt_length = len(request.prompt_token_ids)
        prompt_token_by_position = {
            token.full_position: token for token in request.prompt_tokens
        }
        metadata: list[AttentionTokenMetadata] = []

        for analyzed_position, token_id in enumerate(analyzed_ids):
            full_position = window_start + analyzed_position
            sequence_scope = (
                HuggingFaceAttentionSequenceScope.PROMPT
                if full_position < prompt_length
                else HuggingFaceAttentionSequenceScope.GENERATED
            )
            generated_token_index = full_position - prompt_length if full_position >= prompt_length else None
            prompt_token = prompt_token_by_position.get(full_position)
            if prompt_token is not None:
                raw_token = prompt_token.raw_token
                display_token = prompt_token.display_token
                decoded_contribution = prompt_token.decoded_contribution
                token_bytes = list(prompt_token.token_bytes)
                source_category = prompt_token.source_category
                source_label = prompt_token.source_label
                special_token = prompt_token.special_token
            else:
                raw_token = self._raw_token(runtime.tokenizer, token_id)
                decoded_contribution = self._decode_token(runtime.tokenizer, token_id)
                display_token = self._display_token(decoded_contribution)
                token_bytes = self._token_bytes(decoded_contribution, raw_token)
                source_category = CanonicalTokenSourceCategory.GENERATED_OUTPUT
                source_label = "Earlier output"
                special_token = False

            metadata.append(
                AttentionTokenMetadata(
                    token_id=token_id,
                    raw_token=raw_token,
                    display_token=display_token,
                    decoded_contribution=decoded_contribution,
                    token_bytes=token_bytes,
                    full_position=full_position,
                    analyzed_position=analyzed_position,
                    sequence_scope=sequence_scope,
                    source_category=source_category,
                    source_label=source_label,
                    special_token=special_token,
                    generated_token_index=generated_token_index,
                )
            )

        return metadata

    def _build_attention_category_breakdown(
        self,
        analyzed_tokens: list[HuggingFaceAttentionTokenInfo],
    ) -> HuggingFaceAttentionCategoryBreakdown:
        totals = {
            "assistant_prefix": 0.0,
            "earlier_output": 0.0,
            "input_context": 0.0,
            "system_message": 0.0,
            "template_control": 0.0,
            "user_prompt": 0.0,
        }

        for token in analyzed_tokens:
            weight = token.attention_weight if token.attention_weight is not None else 0.0

            if token.sequence_scope == HuggingFaceAttentionSequenceScope.PROMPT:
                totals["input_context"] += weight
            else:
                totals["earlier_output"] += weight

            if token.source_category == CanonicalTokenSourceCategory.SYSTEM:
                totals["system_message"] += weight
            elif token.source_category == CanonicalTokenSourceCategory.USER_PROMPT:
                totals["user_prompt"] += weight
            elif token.source_category == CanonicalTokenSourceCategory.ASSISTANT_PREFIX:
                totals["assistant_prefix"] += weight
            elif token.source_category == CanonicalTokenSourceCategory.TEMPLATE:
                totals["template_control"] += weight

        exclusive_total = (
            totals["system_message"]
            + totals["user_prompt"]
            + totals["assistant_prefix"]
            + totals["template_control"]
            + totals["earlier_output"]
        )
        return HuggingFaceAttentionCategoryBreakdown(
            input_context=round(totals["input_context"], 6),
            earlier_output=round(totals["earlier_output"], 6),
            system_message=round(totals["system_message"], 6),
            user_prompt=round(totals["user_prompt"], 6),
            assistant_prefix=round(totals["assistant_prefix"], 6),
            template_control=round(totals["template_control"], 6),
            exclusive_total=round(exclusive_total, 6),
        )

    def _materialize_attention_view(
        self,
        *,
        token_metadata: list[AttentionTokenMetadata],
        aggregated_attention_row: Any,
        query_position_in_window: int,
        selected_position_in_window: int,
        max_connections: int,
    ) -> tuple[
        list[HuggingFaceAttentionTokenInfo],
        list[HuggingFaceAttentionSource],
        list[HuggingFaceAttentionSource],
        HuggingFaceAttentionCategoryBreakdown,
        float,
        float,
    ]:
        assert torch is not None
        analyzed_tokens: list[HuggingFaceAttentionTokenInfo] = []
        ranked_sources: list[tuple[float, HuggingFaceAttentionTokenInfo]] = []

        for token_meta in token_metadata:
            attention_weight = (
                float(aggregated_attention_row[token_meta.analyzed_position].item())
                if token_meta.analyzed_position <= query_position_in_window
                else None
            )
            token_info = HuggingFaceAttentionTokenInfo(
                token_id=token_meta.token_id,
                raw_token=token_meta.raw_token,
                display_token=token_meta.display_token,
                decoded_contribution=token_meta.decoded_contribution,
                token_bytes=list(token_meta.token_bytes),
                full_position=token_meta.full_position,
                analyzed_position=token_meta.analyzed_position,
                sequence_scope=token_meta.sequence_scope,
                source_category=token_meta.source_category,
                source_label=token_meta.source_label,
                special_token=token_meta.special_token,
                generated_token_index=token_meta.generated_token_index,
                attention_weight=round(attention_weight, 6) if attention_weight is not None else None,
                is_query=token_meta.analyzed_position == query_position_in_window,
                is_selected_token=token_meta.analyzed_position == selected_position_in_window,
            )
            analyzed_tokens.append(token_info)
            if attention_weight is not None:
                ranked_sources.append((attention_weight, token_info))

        ranked_sources.sort(key=lambda item: (-item[0], item[1].full_position))
        all_sources = [
            HuggingFaceAttentionSource(
                token_id=token_info.token_id,
                raw_token=token_info.raw_token,
                display_token=token_info.display_token,
                decoded_contribution=token_info.decoded_contribution,
                token_bytes=list(token_info.token_bytes),
                full_position=token_info.full_position,
                analyzed_position=token_info.analyzed_position,
                sequence_scope=token_info.sequence_scope,
                source_category=token_info.source_category,
                source_label=token_info.source_label,
                special_token=token_info.special_token,
                generated_token_index=token_info.generated_token_index,
                attention_weight=round(weight, 6),
                rank=rank,
            )
            for rank, (weight, token_info) in enumerate(ranked_sources, start=1)
        ]
        top_sources = all_sources[: max(max_connections, 1)]
        category_breakdown = self._build_attention_category_breakdown(analyzed_tokens)
        attention_mass_sum = round(float(torch.sum(aggregated_attention_row).item()), 6)
        top_n_coverage = round(float(sum(source.attention_weight for source in top_sources)), 6)
        return (
            analyzed_tokens,
            all_sources,
            top_sources,
            category_breakdown,
            attention_mass_sum,
            top_n_coverage,
        )

    def _find_top_meaningful_source(
        self,
        all_sources: list[HuggingFaceAttentionSource],
    ) -> HuggingFaceAttentionSource | None:
        fallback: HuggingFaceAttentionSource | None = None
        for source in all_sources:
            if source.source_category == CanonicalTokenSourceCategory.TEMPLATE:
                continue
            if fallback is None:
                fallback = source
            token_text = source.display_token or source.decoded_contribution or source.raw_token
            if any(character.isalnum() for character in token_text):
                return source
        return fallback

    def _build_attention_layer_summaries(
        self,
        *,
        comparison_layers: list[int],
        layer_rows: dict[int, Any],
        token_metadata: list[AttentionTokenMetadata],
        query_position_in_window: int,
        selected_position_in_window: int,
        num_layers: int,
        max_connections: int,
    ) -> list[HuggingFaceAttentionLayerSummary]:
        summaries: list[HuggingFaceAttentionLayerSummary] = []

        for layer_index in comparison_layers:
            (
                _analyzed_tokens,
                all_sources,
                top_sources,
                category_breakdown,
                attention_mass_sum,
                top_n_coverage,
            ) = self._materialize_attention_view(
                token_metadata=token_metadata,
                aggregated_attention_row=layer_rows[layer_index],
                query_position_in_window=query_position_in_window,
                selected_position_in_window=selected_position_in_window,
                max_connections=max_connections,
            )
            summaries.append(
                HuggingFaceAttentionLayerSummary(
                    layer_index=layer_index,
                    depth_ratio=0.0 if num_layers <= 1 else round(layer_index / (num_layers - 1), 6),
                    top_meaningful_source=self._find_top_meaningful_source(all_sources),
                    category_breakdown=category_breakdown,
                    attention_mass_sum=attention_mass_sum,
                    top_n_coverage=top_n_coverage,
                )
            )

        return summaries

    def _build_attention_layer_journey(
        self,
        *,
        journey_layers: list[int],
        layer_rows: dict[int, Any],
        token_metadata: list[AttentionTokenMetadata],
        query_position_in_window: int,
        selected_position_in_window: int,
        num_layers: int,
        max_connections: int,
        selected_decoded_contribution: str,
    ) -> HuggingFaceAttentionLayerJourney | None:
        if not journey_layers:
            return None

        all_sources_by_layer: dict[int, list[HuggingFaceAttentionSource]] = {}
        position_weight_by_layer: dict[int, dict[int, float]] = {}
        attention_mass_by_layer: dict[int, float] = {}

        for layer_index in journey_layers:
            (
                _analyzed_tokens,
                all_sources,
                _top_sources,
                _category_breakdown,
                attention_mass_sum,
                _top_n_coverage,
            ) = self._materialize_attention_view(
                token_metadata=token_metadata,
                aggregated_attention_row=layer_rows[layer_index],
                query_position_in_window=query_position_in_window,
                selected_position_in_window=selected_position_in_window,
                max_connections=max_connections,
            )
            all_sources_by_layer[layer_index] = all_sources
            position_weight_by_layer[layer_index] = {
                source.full_position: source.attention_weight for source in all_sources
            }
            attention_mass_by_layer[layer_index] = attention_mass_sum

        def best_source(
            predicate: Callable[[HuggingFaceAttentionSource], bool],
            *,
            prefer_lexical: bool = False,
        ) -> HuggingFaceAttentionSource | None:
            winner: HuggingFaceAttentionSource | None = None
            winning_tuple: tuple[int, float, int, int] | None = None

            for layer_index in journey_layers:
                for source in all_sources_by_layer[layer_index]:
                    if not predicate(source):
                        continue
                    token_text = source.display_token or source.decoded_contribution or source.raw_token
                    lexical_score = 1 if any(character.isalnum() for character in token_text) else 0
                    candidate = (
                        lexical_score if prefer_lexical else 0,
                        source.attention_weight,
                        -source.full_position,
                        -layer_index,
                    )
                    if winning_tuple is None or candidate > winning_tuple:
                        winning_tuple = candidate
                        winner = source
            return winner

        candidate_sources = [
            (
                "strongest_prompt_source",
                best_source(
                    lambda source: (
                        source.sequence_scope == HuggingFaceAttentionSequenceScope.PROMPT
                        and source.source_category != CanonicalTokenSourceCategory.TEMPLATE
                    ),
                    prefer_lexical=True,
                ),
            ),
            (
                "strongest_earlier_output_source",
                best_source(
                    lambda source: source.sequence_scope == HuggingFaceAttentionSequenceScope.GENERATED,
                    prefer_lexical=True,
                ),
            ),
            (
                "strongest_template_source",
                best_source(
                    lambda source: source.source_category == CanonicalTokenSourceCategory.TEMPLATE
                ),
            ),
        ]
        if selected_decoded_contribution:
            candidate_sources.append(
                (
                    "exact_prompt_match",
                    best_source(
                        lambda source: (
                            source.sequence_scope == HuggingFaceAttentionSequenceScope.PROMPT
                            and source.decoded_contribution == selected_decoded_contribution
                        )
                    ),
                )
            )

        rows: list[HuggingFaceAttentionJourneyRow] = []
        used_positions: set[int] = set()

        for included_reason, source in candidate_sources:
            if source is None or source.full_position in used_positions:
                continue
            used_positions.add(source.full_position)
            weights = [
                round(position_weight_by_layer[layer_index].get(source.full_position, 0.0), 6)
                for layer_index in journey_layers
            ]
            rows.append(
                HuggingFaceAttentionJourneyRow(
                    row_id=f"source:{source.full_position}",
                    row_kind="source",
                    label=source.display_token or source.raw_token,
                    included_reason=included_reason,
                    source=source,
                    weights=weights,
                    max_weight=round(max(weights) if weights else 0.0, 6),
                )
            )
            if len(rows) >= max_connections:
                break

        remaining_capacity = max_connections - len(rows)
        if remaining_capacity > 0:
            other_weights: list[float] = []
            for layer_index in journey_layers:
                included_sum = sum(
                    position_weight_by_layer[layer_index].get(source_position, 0.0)
                    for source_position in used_positions
                )
                other_weights.append(
                    round(
                        max(attention_mass_by_layer[layer_index] - included_sum, 0.0),
                        6,
                    )
                )

            if any(weight > 0.02 for weight in other_weights):
                rows.append(
                    HuggingFaceAttentionJourneyRow(
                        row_id="other",
                        row_kind="other",
                        label="Other sources",
                        included_reason="residual_mass",
                        source=None,
                        weights=other_weights,
                        max_weight=round(max(other_weights) if other_weights else 0.0, 6),
                    )
                )

        scale_max = round(max((row.max_weight for row in rows), default=0.0), 6)
        return HuggingFaceAttentionLayerJourney(
            layers=journey_layers,
            sampled=journey_layers != list(range(num_layers)),
            scale_max=scale_max,
            rows=rows,
        )

    def _validate_attention_row(
        self,
        attention_row: Any,
        *,
        request: HuggingFaceAttentionRequest,
        runtime: LoadedLocalRuntime,
        query_position: int,
    ) -> None:
        assert torch is not None
        if not torch.isfinite(attention_row).all():
            logger.warning(
                "hf-attention-invalid-row model=%s revision=%s mode=%s layer=%s head=%s query_position=%s reason=non_finite",
                runtime.model_id,
                runtime.resolved_revision or runtime.revision,
                request.analysis_mode.value,
                request.selected_layer,
                request.selected_head,
                query_position,
            )
            raise LLMScopeError(
                code="HF_ATTENTION_INVALID",
                message="The selected attention row contained non-finite values for valid causal positions.",
                status_code=500,
            )
        if torch.any(attention_row < 0):
            raise LLMScopeError(
                code="HF_ATTENTION_INVALID",
                message="The selected attention row contained negative values.",
                status_code=500,
            )
        if not torch.any(attention_row > 0):
            raise LLMScopeError(
                code="HF_ATTENTION_INVALID",
                message="The selected attention row did not contain any usable attention mass.",
                status_code=500,
            )

    def _validate_aggregated_attention_row(
        self,
        aggregated_row: Any,
        *,
        request: HuggingFaceAttentionRequest,
        runtime: LoadedLocalRuntime,
        query_position: int,
    ) -> None:
        assert torch is not None
        if not torch.isfinite(aggregated_row).all():
            logger.warning(
                "hf-attention-invalid-aggregate model=%s revision=%s mode=%s layer=%s head=%s query_position=%s reason=non_finite",
                runtime.model_id,
                runtime.resolved_revision or runtime.revision,
                request.analysis_mode.value,
                request.selected_layer,
                request.selected_head,
                query_position,
            )
            raise LLMScopeError(
                code="HF_ATTENTION_INVALID",
                message="The aggregated attention row contained non-finite values.",
                status_code=500,
            )
        if torch.any(aggregated_row < 0):
            raise LLMScopeError(
                code="HF_ATTENTION_INVALID",
                message="The aggregated attention row contained negative values.",
                status_code=500,
            )
        if request.aggregation_mode in (
            HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
            HuggingFaceAttentionAggregationMode.SINGLE_HEAD,
        ):
            row_sum = float(aggregated_row.sum().item())
            if abs(row_sum - 1.0) > 0.05:
                raise LLMScopeError(
                    code="HF_ATTENTION_INVALID",
                    message="The aggregated attention row did not sum to one within numerical tolerance.",
                    status_code=500,
                )

    def _make_attention_cache_key(
        self,
        request: HuggingFaceAttentionRequest,
        *,
        full_sequence_ids: list[int],
        selected_token_position: int,
        query_position: int,
    ) -> str:
        sequence_bytes = ",".join(str(token_id) for token_id in full_sequence_ids).encode("utf-8")
        sequence_hash = hashlib.sha1(sequence_bytes).hexdigest()
        return "|".join(
            [
                request.model_id,
                request.model_revision or "",
                request.tokenizer_identity or "",
                request.tokenizer_revision or "",
                sequence_hash,
                request.analysis_mode.value,
                str(selected_token_position),
                str(query_position),
                str(request.selected_generated_token_index),
                str(request.selected_layer),
                str(request.selected_head if request.selected_head is not None else "avg"),
                request.aggregation_mode.value,
                str(request.max_connections),
                str(request.max_context_tokens),
                ",".join(str(layer_index) for layer_index in request.comparison_layers),
                ",".join(str(layer_index) for layer_index in request.journey_layers),
                str(request.journey_max_rows),
                "truncated" if request.allow_truncated_recompute else "exact",
            ]
        )

    def _get_cached_attention(self, cache_key: str) -> HuggingFaceAttentionResponse | None:
        with self._attention_cache_lock:
            cached = self._attention_cache.get(cache_key)
            if cached is None:
                return None
            self._attention_cache.move_to_end(cache_key)
            return cached.model_copy(deep=True)

    def _store_attention(self, cache_key: str, response: HuggingFaceAttentionResponse) -> None:
        with self._attention_cache_lock:
            self._attention_cache[cache_key] = response.model_copy(deep=True)
            self._attention_cache.move_to_end(cache_key)
            while len(self._attention_cache) > ATTENTION_CACHE_SIZE:
                self._attention_cache.popitem(last=False)

    def _build_attention_response(
        self,
        *,
        request: HuggingFaceAttentionRequest,
        runtime: LoadedLocalRuntime,
        full_sequence_ids: list[int],
        analyzed_ids: list[int],
        token_metadata: list[AttentionTokenMetadata],
        selected_token_position: int,
        selected_position_in_window: int,
        query_position: int,
        query_position_in_window: int,
        aggregated_attention_row: Any,
        comparison_layers: list[int],
        comparison_rows: dict[int, Any],
        journey_layers: list[int],
        num_layers: int,
        num_query_heads: int,
        context_truncated: bool,
        window_start: int,
        attention_implementation_used: str,
    ) -> HuggingFaceAttentionResponse:
        prompt_length = len(request.prompt_token_ids)
        source_positions = [
            window_start + analyzed_position
            for analyzed_position in range(query_position_in_window + 1)
        ]
        attention_weights = [
            round(float(aggregated_attention_row[analyzed_position].item()), 6)
            for analyzed_position in range(query_position_in_window + 1)
        ]
        (
            analyzed_tokens,
            all_sources,
            sources,
            category_breakdown,
            attention_mass_sum,
            top_n_coverage,
        ) = self._materialize_attention_view(
            token_metadata=token_metadata,
            aggregated_attention_row=aggregated_attention_row,
            query_position_in_window=query_position_in_window,
            selected_position_in_window=selected_position_in_window,
            max_connections=request.max_connections,
        )
        selected_token = analyzed_tokens[selected_position_in_window]
        query_token = analyzed_tokens[query_position_in_window]
        return HuggingFaceAttentionResponse(
            model_id=runtime.model_id,
            model_revision=runtime.resolved_revision or runtime.revision,
            tokenizer_identity=self._tokenizer_identity(runtime.tokenizer, runtime.model_id),
            tokenizer_revision=self._tokenizer_revision(runtime.tokenizer, runtime),
            analysis_mode=request.analysis_mode,
            selected_token=selected_token,
            query_token=query_token,
            analyzed_tokens=analyzed_tokens,
            sources=sources,
            all_sources=all_sources,
            selected_layer=request.selected_layer,
            selected_head=request.selected_head,
            aggregation_mode=request.aggregation_mode,
            attention_implementation_used=attention_implementation_used,
            num_layers=num_layers,
            num_query_heads=num_query_heads,
            selected_token_position=selected_token_position,
            query_position=query_position,
            selected_token_id=selected_token.token_id,
            query_token_id=query_token.token_id,
            prompt_token_count=prompt_length,
            generated_token_index=request.selected_generated_token_index,
            sequence_length=len(full_sequence_ids),
            layer_index=request.selected_layer,
            head_index=request.selected_head,
            average_heads=request.aggregation_mode == HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
            source_positions=source_positions,
            attention_weights=attention_weights,
            attention_mass_sum=attention_mass_sum,
            top_n_coverage=top_n_coverage,
            truncated_context=context_truncated,
            context_truncated=context_truncated,
            original_full_context_length=len(full_sequence_ids),
            analyzed_context_length=len(analyzed_ids),
            category_breakdown=category_breakdown,
            comparison_layers=self._build_attention_layer_summaries(
                comparison_layers=comparison_layers,
                layer_rows=comparison_rows,
                token_metadata=token_metadata,
                query_position_in_window=query_position_in_window,
                selected_position_in_window=selected_position_in_window,
                num_layers=num_layers,
                max_connections=request.max_connections,
            ),
            layer_journey=self._build_attention_layer_journey(
                journey_layers=journey_layers,
                layer_rows=comparison_rows,
                token_metadata=token_metadata,
                query_position_in_window=query_position_in_window,
                selected_position_in_window=selected_position_in_window,
                num_layers=num_layers,
                max_connections=request.journey_max_rows,
                selected_decoded_contribution=selected_token.decoded_contribution,
            ),
        )

    def _ordered_supported_models(self, recommended_model_id: str | None) -> list[SupportedLocalModel]:
        if recommended_model_id is None:
            return list(self._supported_models)
        preferred = [model for model in self._supported_models if model.id == recommended_model_id]
        others = [model for model in self._supported_models if model.id != recommended_model_id]
        return [*preferred, *others]

    def _build_model_status(
        self,
        *,
        model_spec: SupportedLocalModel,
        recommended_model_id: str | None,
        missing_dependencies: list[str],
        cuda_available: bool,
        runtime: LoadedLocalRuntime | None,
    ) -> HuggingFaceLocalModelStatus:
        downloaded = self._is_model_downloaded(model_spec) if not missing_dependencies else False
        is_loaded = runtime is not None and runtime.model_id == model_spec.id
        state = HuggingFaceLocalState.NOT_DOWNLOADED
        message = "Not downloaded yet."
        resolved_revision = runtime.resolved_revision if is_loaded else None

        if missing_dependencies:
            state = HuggingFaceLocalState.ERROR
            message = HUGGING_FACE_LOCAL_MISSING_DEPENDENCY_MESSAGE
        elif not cuda_available:
            state = HuggingFaceLocalState.CUDA_UNAVAILABLE
            message = HUGGING_FACE_LOCAL_CUDA_UNAVAILABLE_MESSAGE
        elif is_loaded:
            state = self._status_state
            message = self._status_message
        elif downloaded:
            state = HuggingFaceLocalState.READY
            message = "Cached locally. Click Load to initialize on CUDA."

        return HuggingFaceLocalModelStatus(
            id=model_spec.id,
            label=model_spec.label,
            revision=model_spec.revision,
            resolved_revision=resolved_revision,
            status=state,
            status_message=message,
            downloaded=downloaded,
            loaded=is_loaded,
            recommended=model_spec.id == recommended_model_id,
        )

    def _ensure_runtime_dependencies(self) -> None:
        missing_dependencies = collect_huggingface_local_diagnostics().missing_dependencies
        filtered = [item for item in missing_dependencies if item != "huggingface_hub"]
        if filtered:
            self._set_status(
                HuggingFaceLocalState.ERROR,
                HUGGING_FACE_LOCAL_MISSING_DEPENDENCY_MESSAGE,
            )
            raise LLMScopeError(
                code="HF_LOCAL_DEPENDENCY_MISSING",
                message=HUGGING_FACE_LOCAL_MISSING_DEPENDENCY_MESSAGE,
                status_code=503,
            )

    def _ensure_cuda_available(self) -> None:
        if torch is None or not torch.cuda.is_available():
            self._set_status(
                HuggingFaceLocalState.CUDA_UNAVAILABLE,
                HUGGING_FACE_LOCAL_CUDA_UNAVAILABLE_MESSAGE,
            )
            raise LLMScopeError(
                code="HF_LOCAL_CUDA_UNAVAILABLE",
                message=HUGGING_FACE_LOCAL_CUDA_UNAVAILABLE_MESSAGE,
                status_code=503,
            )

    def _resolve_model(self, model_id: str) -> SupportedLocalModel:
        for model in self._supported_models:
            if model.id == model_id:
                return model

        raise LLMScopeError(
            code="HF_LOCAL_MODEL_UNSUPPORTED",
            message=(
                "Unsupported Hugging Face Local model. Use one of: "
                "Qwen/Qwen2.5-3B-Instruct or Qwen/Qwen2.5-1.5B-Instruct."
            ),
            status_code=400,
        )

    def _require_loaded_runtime(self, model_id: str) -> LoadedLocalRuntime:
        runtime = self._runtime

        if runtime is None or runtime.model_id != model_id:
            raise LLMScopeError(
                code="HF_LOCAL_MODEL_NOT_READY",
                message=(
                    "The selected Hugging Face Local model is not loaded. "
                    "Select the model and click Load before generating."
                ),
                status_code=503,
            )

        return runtime

    def _normalize_revision(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def _normalize_dtype_name(self, dtype: Any) -> str:
        value = str(dtype)
        return value.replace("torch.", "")

    def _config_int(self, config: Any, field_name: str) -> int | None:
        value = getattr(config, field_name, None)
        return int(value) if isinstance(value, int) and value > 0 else None

    def _attention_implementation_name(self, model: Any) -> str | None:
        config = getattr(model, "config", None)
        value = getattr(config, "_attn_implementation", None)
        return value if isinstance(value, str) and value else None

    def _tokenizer_identity(self, tokenizer: Any, fallback_model_id: str) -> str | None:
        value = getattr(tokenizer, "name_or_path", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return fallback_model_id

    def _tokenizer_revision(self, tokenizer: Any, runtime: LoadedLocalRuntime) -> str | None:
        init_kwargs = getattr(tokenizer, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            commit_hash = self._normalize_revision(init_kwargs.get("_commit_hash"))
            if commit_hash:
                return commit_hash
        return runtime.resolved_revision or runtime.revision

    def _set_status(self, state: HuggingFaceLocalState, message: str | None) -> None:
        self._status_state = state
        self._status_message = message

    def _is_model_downloaded(self, model_spec: SupportedLocalModel) -> bool:
        if snapshot_download is None:
            return False

        try:
            snapshot_download(
                model_spec.id,
                revision=model_spec.revision,
                token=self._hf_token or None,
                local_files_only=True,
            )
            return True
        except LocalEntryNotFoundError:
            return False
        except Exception:
            return False

    def _load_transformers_model(self, model_spec: SupportedLocalModel) -> Any:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_spec.id,
                revision=model_spec.revision,
                token=self._hf_token or None,
                trust_remote_code=False,
                dtype=torch.float16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                model_spec.id,
                revision=model_spec.revision,
                token=self._hf_token or None,
                trust_remote_code=False,
                torch_dtype=torch.float16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
            )

    def _raise_load_exception(self, exc: Exception, model_spec: SupportedLocalModel) -> None:
        if isinstance(exc, LLMScopeError):
            raise exc

        if self._looks_like_oom(exc):
            self._cleanup_cuda_after_oom()
            self._set_status(
                HuggingFaceLocalState.OOM,
                "CUDA ran out of memory while loading the local model.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_OOM",
                message=(
                    "CUDA ran out of memory while loading the local model. "
                    "Try the 1.5B model or close other GPU-heavy applications."
                ),
                status_code=503,
            ) from exc

        if isinstance(exc, GatedRepoError):
            self._set_status(
                HuggingFaceLocalState.ERROR,
                "This model is gated and requires Hugging Face authentication.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_AUTH_REQUIRED",
                message="This model is gated and requires Hugging Face authentication.",
                status_code=403,
            ) from exc

        if isinstance(exc, RepositoryNotFoundError):
            self._set_status(
                HuggingFaceLocalState.ERROR,
                "The configured Hugging Face model mapping could not be found.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_MODEL_NOT_FOUND",
                message="The configured Hugging Face model mapping could not be found.",
                status_code=404,
            ) from exc

        if isinstance(exc, HfHubHTTPError):
            self._set_status(
                HuggingFaceLocalState.ERROR,
                f"Unable to download {model_spec.label} from Hugging Face.",
            )
            raise LLMScopeError(
                code="HF_LOCAL_DOWNLOAD_FAILED",
                message=f"Unable to download {model_spec.label} from Hugging Face.",
                status_code=502,
            ) from exc

        self._set_status(
            HuggingFaceLocalState.ERROR,
            f"Unable to initialize {model_spec.label} for local analysis.",
        )
        raise LLMScopeError(
            code="HF_LOCAL_INIT_FAILED",
            message=f"Unable to initialize {model_spec.label} for local analysis.",
            status_code=500,
        ) from exc

    def _cleanup_runtime(self) -> None:
        runtime = self._runtime
        self._runtime = None
        with self._attention_cache_lock:
            self._attention_cache.clear()
        if runtime is not None:
            del runtime.model
            del runtime.tokenizer
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cleanup_cuda_after_oom(self) -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _recommended_model_id(
        self,
        free_vram_gb: float | None,
        total_vram_gb: float | None,
    ) -> str | None:
        if free_vram_gb is not None and free_vram_gb >= RECOMMENDED_3B_FREE_VRAM_GB:
            return "Qwen/Qwen2.5-3B-Instruct"
        if total_vram_gb is not None and total_vram_gb >= RECOMMENDED_3B_FREE_VRAM_GB:
            return "Qwen/Qwen2.5-3B-Instruct"
        if self._default_model in self.supported_model_ids:
            return self._default_model
        return "Qwen/Qwen2.5-1.5B-Instruct"

    def _resolve_generation_prefix(
        self,
        *,
        runtime: LoadedLocalRuntime,
        prompt: str,
        assistant_prefix: str,
        canonical_prefix_token_ids: list[int] | None,
        prompt_token_ids: list[int] | None,
    ) -> tuple[
        list[int],
        list[int] | None,
        str,
        list[CanonicalPromptToken] | None,
        str | None,
        str | None,
    ]:
        if canonical_prefix_token_ids is not None:
            prefix_ids = list(canonical_prefix_token_ids)
            return prefix_ids, prompt_token_ids, assistant_prefix, None, None, None

        if assistant_prefix:
            raise LLMScopeError(
                code="HF_LOCAL_PREFIX_IDS_REQUIRED",
                message=(
                    "Exact local continuation requires canonical token IDs for the selected branch prefix."
                ),
                status_code=400,
            )

        prompt_build = self._build_canonical_prompt_tokens(
            runtime.tokenizer,
            prompt=prompt,
        )
        return (
            prompt_build.prompt_ids,
            prompt_build.prompt_ids,
            "",
            prompt_build.prompt_tokens,
            prompt_build.raw_context_text,
            prompt_build.system_prompt,
        )

    def _run_generation_loop(
        self,
        *,
        runtime: LoadedLocalRuntime,
        branch_id: str,
        parent_node_id: str,
        full_prefix_ids: list[int],
        generated_text_prefix: str,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        repetition_penalty: float,
        max_candidates: int,
    ) -> tuple[list[TokenTrace], str | None]:
        assert torch is not None

        device = next(runtime.model.parameters()).device
        prefix_tensor = torch.tensor([full_prefix_ids], dtype=torch.long, device=device)
        current_input_ids = prefix_tensor
        generated_ids: list[int] = []
        all_prefix_ids = list(full_prefix_ids)
        traces: list[TokenTrace] = []
        context_before = generated_text_prefix
        cumulative_probability = 1.0
        cumulative_log_probability = 0.0
        finish_reason: str | None = None
        eos_token_ids = self._normalize_eos_ids(runtime.tokenizer, runtime.model)

        with torch.inference_mode():
            past_key_values = None

            for step_index in range(max_output_tokens):
                outputs = runtime.model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :].squeeze(0)
                processed_logits = self._apply_logits_processors(
                    logits=next_token_logits,
                    generated_ids=generated_ids,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                log_probs = torch.log_softmax(processed_logits, dim=-1)
                probabilities = torch.softmax(processed_logits, dim=-1)
                chosen_token_id = self._select_token_id(probabilities, temperature)
                chosen_probability = float(probabilities[chosen_token_id].item())
                chosen_log_probability = float(log_probs[chosen_token_id].item())
                chosen_rank = int((probabilities > probabilities[chosen_token_id]).sum().item()) + 1

                candidate_ids = self._select_candidate_ids(
                    probabilities=probabilities,
                    chosen_token_id=chosen_token_id,
                    max_candidates=max_candidates,
                )
                candidate_probabilities = [float(probabilities[token_id].item()) for token_id in candidate_ids]
                normalized_displayed_probabilities = self._normalize_displayed_probabilities(
                    candidate_probabilities
                )
                entropy = self._calculate_entropy(probabilities)

                decoded_contribution = self._decode_token(runtime.tokenizer, chosen_token_id)
                raw_token = self._raw_token(runtime.tokenizer, chosen_token_id)
                all_prefix_ids.append(chosen_token_id)
                generated_ids.append(chosen_token_id)
                cumulative_log_probability = round(
                    cumulative_log_probability + chosen_log_probability,
                    6,
                )
                cumulative_probability = round(
                    max(MIN_PROBABILITY, min(1.0, cumulative_probability * chosen_probability)),
                    6,
                )
                context_after = self._decode_generated_ids(runtime.tokenizer, generated_ids)
                step_parent_id = parent_node_id if step_index == 0 else traces[step_index - 1].id
                step_id = self._make_step_id(
                    branch_id=branch_id,
                    token_index=step_index,
                    token_id=chosen_token_id,
                    raw_token=raw_token,
                )
                token_bytes = self._token_bytes(decoded_contribution, raw_token)
                metadata = {
                    "branch_id": branch_id,
                    "parent_node_id": step_parent_id,
                    "provider": ModelProvider.HUGGING_FACE.value,
                    "device": runtime.device,
                    "precision": runtime.dtype,
                    "resolved_revision": runtime.resolved_revision,
                    "requested_revision": runtime.revision,
                    "tokenizer_identity": self._tokenizer_identity(runtime.tokenizer, runtime.model_id),
                    "tokenizer_revision": self._tokenizer_revision(runtime.tokenizer, runtime),
                    "local_exact": True,
                }
                alternatives: list[AlternativeCandidate] = []

                for candidate_index, candidate_id in enumerate(candidate_ids):
                    if candidate_id == chosen_token_id:
                        continue

                    candidate_probability = candidate_probabilities[candidate_index]
                    candidate_log_probability = float(log_probs[candidate_id].item())
                    candidate_decoded = self._decode_token(runtime.tokenizer, candidate_id)
                    candidate_raw_token = self._raw_token(runtime.tokenizer, candidate_id)
                    candidate_token_bytes = self._token_bytes(candidate_decoded, candidate_raw_token)
                    candidate_rank = int((probabilities > probabilities[candidate_id]).sum().item()) + 1
                    candidate_generated_ids = [*generated_ids[:-1], candidate_id]
                    candidate_context_after = self._decode_generated_ids(
                        runtime.tokenizer,
                        candidate_generated_ids,
                    )
                    alternatives.append(
                        AlternativeCandidate(
                            node_id=self._make_step_id(
                                branch_id=step_parent_id,
                                token_index=step_index,
                                token_id=candidate_id,
                                raw_token=candidate_raw_token,
                            ),
                            token=candidate_raw_token,
                            display_token=self._display_token(candidate_decoded),
                            token_bytes=candidate_token_bytes,
                            decoded_contribution=candidate_decoded,
                            cumulative_decoded_text=candidate_context_after,
                            cumulative_token_ids=[*all_prefix_ids[:-1], candidate_id],
                            cumulative_log_probability=round(
                                cumulative_log_probability
                                - chosen_log_probability
                                + candidate_log_probability,
                                6,
                            ),
                            probability=candidate_probability,
                            raw_probability=candidate_probability,
                            normalized_displayed_probability=normalized_displayed_probabilities[
                                candidate_index
                            ],
                            log_probability=round(candidate_log_probability, 6),
                            entropy=entropy,
                            latency_ms=0,
                            token_id=candidate_id,
                            tokenizer_id=candidate_id,
                            rank=candidate_rank,
                            text_preview=candidate_context_after,
                            context_before=context_before,
                            context_after=candidate_context_after,
                            generation_step=step_index,
                            metadata=metadata,
                        )
                    )

                traces.append(
                    TokenTrace(
                        id=step_id,
                        branch_id=branch_id,
                        parent_node_id=step_parent_id,
                        model=runtime.model_id,
                        source="hugging_face",
                        index=step_index,
                        position=step_index,
                        token=raw_token,
                        display_token=self._display_token(decoded_contribution),
                        token_bytes=token_bytes,
                        decoded_contribution=decoded_contribution,
                        cumulative_decoded_text=context_after,
                        cumulative_token_ids=list(all_prefix_ids),
                        cumulative_log_probability=cumulative_log_probability,
                        token_id=chosen_token_id,
                        tokenizer_id=chosen_token_id,
                        probability=chosen_probability,
                        raw_probability=chosen_probability,
                        normalized_displayed_probability=normalized_displayed_probabilities[
                            candidate_ids.index(chosen_token_id)
                        ],
                        log_probability=round(chosen_log_probability, 6),
                        entropy=entropy,
                        cumulative_probability=cumulative_probability,
                        latency_ms=0,
                        text_preview=context_after,
                        context_before=context_before,
                        context_after=context_after,
                        finish_reason=None,
                        alternatives=alternatives,
                        generation_step=step_index,
                        metadata=metadata,
                    )
                )

                context_before = context_after
                current_input_ids = torch.tensor(
                    [[chosen_token_id]],
                    dtype=torch.long,
                    device=device,
                )

                if chosen_token_id in eos_token_ids:
                    finish_reason = "eos"
                    break

            if finish_reason is None:
                finish_reason = "length"

        for trace in traces:
            if trace.token_id in eos_token_ids:
                trace.finish_reason = "eos"
            elif trace is traces[-1] and finish_reason == "length":
                trace.finish_reason = "length"

        return traces, finish_reason

    def _normalize_eos_ids(self, tokenizer: Any, model: Any) -> set[int]:
        candidate_ids: set[int] = set()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            candidate_ids.add(eos_token_id)
        generation_config_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(generation_config_eos, int):
            candidate_ids.add(generation_config_eos)
        elif isinstance(generation_config_eos, (list, tuple)):
            candidate_ids.update(int(value) for value in generation_config_eos if isinstance(value, int))
        return candidate_ids

    def _apply_logits_processors(
        self,
        *,
        logits: Any,
        generated_ids: list[int],
        repetition_penalty: float,
        temperature: float,
        top_k: int | None,
        top_p: float,
    ) -> Any:
        assert torch is not None
        processed = logits.clone()
        if repetition_penalty != 1.0 and generated_ids:
            unique_ids = torch.unique(
                torch.tensor(generated_ids, dtype=torch.long, device=processed.device)
            )
            repeated_scores = processed[unique_ids]
            processed[unique_ids] = torch.where(
                repeated_scores < 0,
                repeated_scores * repetition_penalty,
                repeated_scores / repetition_penalty,
            )

        if temperature > 0:
            processed = processed / max(temperature, 1e-6)
            if top_k is not None and 0 < top_k < processed.shape[-1]:
                cutoff = torch.topk(processed, top_k).values[..., -1]
                processed = torch.where(
                    processed < cutoff,
                    torch.full_like(processed, float("-inf")),
                    processed,
                )
            if 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(processed, descending=True)
                sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
                sorted_indices_to_remove = cumulative_probabilities > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(processed, dtype=torch.bool)
                indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
                processed = processed.masked_fill(indices_to_remove, float("-inf"))

        return processed

    def _select_token_id(self, probabilities: Any, temperature: float) -> int:
        assert torch is not None
        if temperature <= 0:
            return int(torch.argmax(probabilities).item())
        return int(torch.multinomial(probabilities, num_samples=1).item())

    def _select_candidate_ids(
        self,
        *,
        probabilities: Any,
        chosen_token_id: int,
        max_candidates: int,
    ) -> list[int]:
        assert torch is not None
        top_count = max(1, min(max_candidates, probabilities.shape[-1]))
        top_ids = [int(value) for value in torch.topk(probabilities, top_count).indices.tolist()]
        if chosen_token_id not in top_ids:
            top_ids = [chosen_token_id, *top_ids[: max(0, top_count - 1)]]
        return top_ids[:top_count]

    def _normalize_displayed_probabilities(self, probabilities: list[float]) -> list[float]:
        total = sum(max(value, 0.0) for value in probabilities)
        if total <= 0:
            return [0.0 for _ in probabilities]
        return [round(max(value, 0.0) / total, 6) for value in probabilities]

    def _calculate_entropy(self, probabilities: Any) -> float:
        assert torch is not None
        entropy = float(
            -torch.sum(
                torch.where(
                    probabilities > 0,
                    probabilities * torch.log(probabilities.clamp_min(MIN_PROBABILITY)),
                    torch.zeros_like(probabilities),
                )
            ).item()
        )
        return round(entropy, 6)

    def _display_token(self, decoded_contribution: str) -> str:
        visible = decoded_contribution.replace("\t", "\u21E5").replace("\n", "\u21B5\n")
        if visible.startswith(" "):
            visible = visible.replace(" ", "\u2420", 1)
        return visible or "\u2205"

    def _decode_token(self, tokenizer: Any, token_id: int) -> str:
        return tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

    def _decode_generated_ids(self, tokenizer: Any, token_ids: list[int]) -> str:
        return tokenizer.decode(
            token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )

    def _raw_token(self, tokenizer: Any, token_id: int) -> str:
        return tokenizer.convert_ids_to_tokens([token_id])[0]

    def _token_bytes(self, decoded_contribution: str, raw_token: str) -> list[int]:
        source = decoded_contribution if decoded_contribution else raw_token
        return list(source.encode("utf-8"))

    def _make_step_id(
        self,
        *,
        branch_id: str,
        token_index: int,
        token_id: int,
        raw_token: str,
    ) -> str:
        checksum = zlib.adler32(f"{token_id}:{raw_token}".encode("utf-8")) & 0xFFFFFFFF
        return f"{branch_id}:{token_index}:{checksum:08x}"

    def _looks_like_oom(self, exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()
