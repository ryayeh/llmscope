from __future__ import annotations

import gc
import math
import platform
import shutil
import sys
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from app.core.errors import LLMScopeError
from app.models.provider import ModelProvider
from app.providers.base import (
    DiscoveredModel,
    LLMProvider,
    ProviderCapabilities,
    ProviderDiscoveryResult,
)
from app.schemas.generation import AlternativeCandidate, TokenTrace
from app.schemas.huggingface_local import (
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
    supports_attention=False,
    supports_streaming=False,
    supports_branching=True,
    supports_continuation=True,
)
HUGGING_FACE_LOCAL_CAPABILITIES_DETAIL = ProviderCapabilitiesDetail(
    supports_logprobs=True,
    supports_entropy=True,
    supports_attention=False,
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


@dataclass(frozen=True)
class HuggingFaceGenerationResult:
    completion: str
    prompt_token_ids: list[int] | None
    tokens: list[TokenTrace]
    prompt_tokens: int
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
                tokens=traces,
                prompt_tokens=prompt_token_count,
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
    ) -> tuple[list[int], list[int] | None, str]:
        if canonical_prefix_token_ids is not None:
            prefix_ids = list(canonical_prefix_token_ids)
            return prefix_ids, prompt_token_ids, assistant_prefix

        if assistant_prefix:
            raise LLMScopeError(
                code="HF_LOCAL_PREFIX_IDS_REQUIRED",
                message=(
                    "Exact local continuation requires canonical token IDs for the selected branch prefix."
                ),
                status_code=400,
            )

        prompt_ids_tensor = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        prompt_ids = [int(value) for value in prompt_ids_tensor[0].tolist()]
        return prompt_ids, prompt_ids, ""

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
