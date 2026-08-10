import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.providers.huggingface_provider as huggingface_provider_module
from app.core.errors import LLMScopeError
from app.schemas.generation import CanonicalPromptToken, CanonicalTokenSourceCategory
from app.schemas.huggingface_local import (
    HuggingFaceAttentionAnalysisMode,
    HuggingFaceAttentionAggregationMode,
    HuggingFaceAttentionRequest,
    HuggingFaceLocalDiagnosticsResponse,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 4
        self.eos_token = "<eos>"
        self.name_or_path = "Qwen/Qwen2.5-3B-Instruct"
        self.init_kwargs = {"_commit_hash": "resolved-tokenizer-sha"}
        self._raw_tokens = {
            1: "Ä time",
            2: "Ä split",
            3: "Ä race",
            4: "<eos>",
            11: "<|user|>",
            12: "Question:",
            13: "A",
            14: "Ġgood",
            15: ",",
            16: "Ġpace",
        }
        self._decoded_tokens = {
            1: " time",
            2: " split",
            3: " race",
            4: "",
            11: "<|user|>",
            12: "Question:",
            13: "A",
            14: " good",
            15: ",",
            16: " pace",
        }

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_tensors):
        del messages, add_generation_prompt, tokenize, return_tensors
        return torch.tensor([[11, 12]], dtype=torch.long)

    def decode(self, token_ids, clean_up_tokenization_spaces, skip_special_tokens):
        del clean_up_tokenization_spaces, skip_special_tokens
        return "".join(self._decoded_tokens.get(token_id, f"<{token_id}>") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return [self._raw_tokens.get(token_id, f"tok-{token_id}") for token_id in token_ids]


class TemplateAwareFakeTokenizer:
    SYSTEM_TEXT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    USER_TEXT = "Question: A"
    FORMATTED_PROMPT = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Question: A<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(self) -> None:
        self.all_special_ids = [901, 903, 905]
        self._raw_tokens = {
            201: "You ",
            202: "are ",
            203: "Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            301: "Question: ",
            302: "A",
            901: "<|im_start|>system\n",
            903: "<|im_end|>\n<|im_start|>user\n",
            905: "<|im_end|>\n<|im_start|>assistant\n",
        }

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
        return_tensors=None,
    ):
        del messages, add_generation_prompt, return_tensors
        if tokenize:
            return torch.tensor([[901, 201, 202, 203, 903, 301, 302, 905]], dtype=torch.long)
        return self.FORMATTED_PROMPT

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == self.SYSTEM_TEXT:
            return {"input_ids": [201, 202, 203]}
        if text == self.USER_TEXT:
            return {"input_ids": [301, 302]}
        return {"input_ids": []}

    def decode(self, token_ids, clean_up_tokenization_spaces, skip_special_tokens):
        del clean_up_tokenization_spaces, skip_special_tokens
        return "".join(self._raw_tokens.get(token_id, f"<{token_id}>") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return [self._raw_tokens.get(token_id, f"tok-{token_id}") for token_id in token_ids]


class FakeModel:
    def __init__(self, logits_per_step):
        self._logits_per_step = logits_per_step
        self._call_index = 0
        self._parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        self.config = type(
            "Config",
            (),
            {
                "_commit_hash": "resolved-model-sha",
                "_attn_implementation": "sdpa",
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
            },
        )()
        self.generation_config = type("GenerationConfig", (), {"eos_token_id": 4})()

    def eval(self):
        return self

    def parameters(self):
        yield self._parameter

    def __call__(self, input_ids, past_key_values, use_cache, return_dict):
        del input_ids, past_key_values, use_cache, return_dict
        logits = self._logits_per_step[self._call_index]
        self._call_index += 1
        return type(
            "FakeOutput",
            (),
            {
                "logits": logits.unsqueeze(0).unsqueeze(0),
                "past_key_values": f"pkv-{self._call_index}",
            },
        )()


class FakeAttentionModel:
    def __init__(self, attentions=None, *, raises_oom: bool = False):
        self._parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        self._attentions = attentions or ()
        self._raises_oom = raises_oom
        self._call_count = 0
        num_layers = len(self._attentions) if self._attentions else 2
        num_heads = int(self._attentions[0].shape[1]) if self._attentions else 2
        self.config = type(
            "Config",
            (),
            {
                "_commit_hash": "resolved-model-sha",
                "_attn_implementation": "sdpa",
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
            },
        )()
        self.generation_config = type("GenerationConfig", (), {"eos_token_id": 4})()

    @property
    def call_count(self) -> int:
        return self._call_count

    def eval(self):
        return self

    def parameters(self):
        yield self._parameter

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        del attention_mask, use_cache, output_attentions, output_hidden_states, return_dict
        if self._raises_oom:
            raise RuntimeError("CUDA out of memory")
        self._call_count += 1
        seq_len = int(input_ids.shape[1])
        logits = torch.zeros((1, seq_len, 32), dtype=torch.float32)
        return type(
            "FakeOutput",
            (),
            {
                "logits": logits,
                "past_key_values": None,
                "attentions": self._attentions,
            },
        )()


def make_attention_tensor(
    head_rows: list[list[float]],
    *,
    query_index: int | None = None,
) -> tuple[torch.Tensor, ...]:
    seq_len = len(head_rows[0])
    head_count = len(head_rows)
    tensor = torch.zeros((1, head_count, seq_len, seq_len), dtype=torch.float32)
    effective_query_index = seq_len - 1 if query_index is None else query_index
    for head_index, row in enumerate(head_rows):
        tensor[0, head_index, effective_query_index, : len(row)] = torch.tensor(row, dtype=torch.float32)
    return (tensor.clone(), tensor)


def make_diagnostics(
    *,
    cuda_available=True,
    gpu_free_vram_gb=10.5,
    gpu_total_vram_gb=12.0,
    missing_dependencies=None,
):
    return HuggingFaceLocalDiagnosticsResponse(
        cuda_available=cuda_available,
        selected_device="cuda:0" if cuda_available else None,
        selected_dtype="float16" if cuda_available else None,
        torch_version="2.12.1+cu126",
        transformers_version="4.57.6",
        torch_cuda_runtime="12.6",
        gpu_name="NVIDIA GeForce RTX 3060" if cuda_available else None,
        gpu_total_vram_gb=gpu_total_vram_gb if cuda_available else None,
        gpu_free_vram_gb=gpu_free_vram_gb if cuda_available else None,
        python_version="3.14.3",
        platform="Windows-11",
        disk_free_gb=26.5,
        missing_dependencies=missing_dependencies or [],
    )


def make_runtime(model):
    return huggingface_provider_module.LoadedLocalRuntime(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        label="Qwen2.5 3B Instruct",
        revision="rev-3b",
        resolved_revision="resolved-model-sha",
        tokenizer=FakeTokenizer(),
        model=model,
        device="cuda:0",
        dtype="float16",
        num_hidden_layers=2,
        num_attention_heads=2,
        attention_implementation="sdpa",
    )


def make_prompt_tokens(*token_ids: int) -> list[CanonicalPromptToken]:
    tokenizer = FakeTokenizer()
    prompt_tokens: list[CanonicalPromptToken] = []
    for full_position, token_id in enumerate(token_ids):
        raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
        prompt_tokens.append(
            CanonicalPromptToken(
                token_id=token_id,
                raw_token=raw_token,
                display_token=decoded.replace(" ", "\u2420", 1) if decoded.startswith(" ") else decoded,
                decoded_contribution=decoded,
                token_bytes=list(decoded.encode("utf-8")) if decoded else list(raw_token.encode("utf-8")),
                full_position=full_position,
                source_category=(
                    CanonicalTokenSourceCategory.TEMPLATE
                    if token_id == 11
                    else CanonicalTokenSourceCategory.USER_PROMPT
                ),
                source_label="Template / control" if token_id == 11 else "User prompt",
                special_token=(token_id == 11),
            )
        )
    return prompt_tokens


class HuggingFaceLocalProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = huggingface_provider_module.HuggingFaceLocalProvider(
            default_model="Qwen/Qwen2.5-3B-Instruct",
            model_revisions={
                "Qwen/Qwen2.5-1.5B-Instruct": "rev-15b",
                "Qwen/Qwen2.5-3B-Instruct": "rev-3b",
            },
            hf_token="",
            context_limit=2048,
            default_output_tokens=128,
            max_output_tokens=512,
        )

    def test_discovery_reports_allowlisted_models_with_per_model_status(self) -> None:
        with (
            patch.object(
                huggingface_provider_module,
                "collect_huggingface_local_diagnostics",
                return_value=make_diagnostics(),
            ),
            patch.object(
                self.provider,
                "_is_model_downloaded",
                side_effect=lambda model_spec: model_spec.id == "Qwen/Qwen2.5-3B-Instruct",
            ),
        ):
            discovery = self.provider.discover_models()

        self.assertEqual(discovery.provider_name.value, "hugging_face")
        self.assertTrue(discovery.capabilities.supports_native_continuation)
        self.assertTrue(discovery.capabilities.supports_attention)
        self.assertEqual(
            [model.id for model in discovery.models],
            [
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ],
        )
        self.assertEqual(discovery.models[0].status, "ready")
        self.assertEqual(discovery.models[1].status, "not_downloaded")

    def test_load_model_is_lazy_cached_and_concurrency_safe(self) -> None:
        tokenizer_calls = 0
        model_calls = 0

        def make_tokenizer(*args, **kwargs):
            del args, kwargs
            nonlocal tokenizer_calls
            tokenizer_calls += 1
            time.sleep(0.05)
            return FakeTokenizer()

        def make_model(model_spec):
            del model_spec
            nonlocal model_calls
            model_calls += 1
            time.sleep(0.05)
            return FakeModel([torch.tensor([0.0, 1.0, 0.5, 0.2, -1.0], dtype=torch.float32)])

        results = []

        with (
            patch.object(
                huggingface_provider_module,
                "collect_huggingface_local_diagnostics",
                return_value=make_diagnostics(),
            ),
            patch.object(self.provider, "_is_model_downloaded", return_value=True),
            patch.object(self.provider, "_load_transformers_model", side_effect=make_model),
            patch.object(
                huggingface_provider_module,
                "AutoTokenizer",
                type("AutoTokenizer", (), {"from_pretrained": staticmethod(make_tokenizer)}),
            ),
        ):
            threads = [
                threading.Thread(
                    target=lambda: results.append(
                        self.provider.load_model("Qwen/Qwen2.5-3B-Instruct").active_model_id
                    )
                )
                for _ in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(tokenizer_calls, 1)
        self.assertEqual(model_calls, 1)
        self.assertEqual(results, ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"])

    def test_generate_uses_exact_token_ids_and_preserves_raw_tokens(self) -> None:
        logits_per_step = [
            torch.tensor([0.0, 3.0, 2.0, 1.0, -1.0], dtype=torch.float32),
            torch.tensor([0.0, -1.0, 2.5, 1.0, -2.0], dtype=torch.float32),
        ]
        self.provider._runtime = make_runtime(FakeModel(logits_per_step))

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            result = self.provider.generate(
                model="Qwen/Qwen2.5-3B-Instruct",
                prompt="How fast is a good 400m?",
                assistant_prefix="",
                branch_id="main",
                parent_node_id="root",
                max_output_tokens=2,
                temperature=0,
                top_p=1.0,
                max_candidates=3,
            )

        self.assertEqual(result.prompt_token_ids, [11, 12])
        self.assertEqual(result.completion, " time split")
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.tokens[0].token, "Ä time")
        self.assertEqual(result.tokens[0].decoded_contribution, " time")
        self.assertEqual(result.tokens[0].token_id, 1)
        self.assertEqual(result.tokens[0].cumulative_token_ids, [11, 12, 1])
        self.assertEqual(result.tokens[0].alternatives[0].token, "Ä split")
        self.assertEqual(result.tokens[0].alternatives[0].decoded_contribution, " split")
        self.assertGreater(
            result.tokens[0].probability or 0,
            result.tokens[0].alternatives[0].probability or 0,
        )
        self.assertEqual(result.tokens[1].context_before, " time")
        self.assertEqual(result.tokens[1].context_after, " time split")
        self.assertEqual(result.tokens[1].cumulative_token_ids, [11, 12, 1, 2])

    def test_generate_can_continue_from_canonical_prefix_token_ids(self) -> None:
        self.provider._runtime = make_runtime(
            FakeModel([torch.tensor([0.0, -1.0, 2.2, 1.1, -3.0], dtype=torch.float32)])
        )

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            result = self.provider.generate(
                model="Qwen/Qwen2.5-3B-Instruct",
                prompt="ignored once canonical ids are present",
                assistant_prefix=" time",
                branch_id="node-1",
                parent_node_id="node-1",
                max_output_tokens=1,
                temperature=0,
                top_p=1.0,
                max_candidates=3,
                canonical_prefix_token_ids=[11, 12, 1],
                prompt_token_ids=[11, 12],
            )

        self.assertEqual(result.tokens[0].cumulative_token_ids, [11, 12, 1, 2])
        self.assertEqual(result.tokens[0].context_before, " time")
        self.assertEqual(result.tokens[0].token, "Ä split")

    def test_build_canonical_prompt_tokens_separates_system_user_and_template_roles(self) -> None:
        build_result = self.provider._build_canonical_prompt_tokens(
            TemplateAwareFakeTokenizer(),
            prompt=TemplateAwareFakeTokenizer.USER_TEXT,
        )

        self.assertEqual(
            build_result.prompt_ids,
            [901, 201, 202, 203, 903, 301, 302, 905],
        )
        self.assertEqual(
            build_result.system_prompt,
            TemplateAwareFakeTokenizer.SYSTEM_TEXT,
        )
        self.assertEqual(
            build_result.raw_context_text,
            TemplateAwareFakeTokenizer.FORMATTED_PROMPT,
        )
        self.assertEqual(
            [token.source_category for token in build_result.prompt_tokens],
            [
                CanonicalTokenSourceCategory.TEMPLATE,
                CanonicalTokenSourceCategory.SYSTEM,
                CanonicalTokenSourceCategory.SYSTEM,
                CanonicalTokenSourceCategory.SYSTEM,
                CanonicalTokenSourceCategory.TEMPLATE,
                CanonicalTokenSourceCategory.USER_PROMPT,
                CanonicalTokenSourceCategory.USER_PROMPT,
                CanonicalTokenSourceCategory.TEMPLATE,
            ],
        )

    def test_generate_returns_busy_error_when_runtime_is_locked(self) -> None:
        self.provider._runtime = make_runtime(
            FakeModel([torch.tensor([0.0, 1.0, 0.5, 0.2, -1.0], dtype=torch.float32)])
        )
        self.provider._busy_lock.acquire()
        try:
            with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
                with self.assertRaises(LLMScopeError) as caught:
                    self.provider.generate(
                        model="Qwen/Qwen2.5-3B-Instruct",
                        prompt="Prompt",
                        assistant_prefix="",
                        branch_id="main",
                        parent_node_id="root",
                        max_output_tokens=1,
                        temperature=0,
                        top_p=1.0,
                        max_candidates=3,
                    )
        finally:
            self.provider._busy_lock.release()

        self.assertEqual(caught.exception.code, "HF_LOCAL_BUSY")

    def test_analyze_attention_maps_prompt_and_generated_positions_and_orders_sources(self) -> None:
        attentions = make_attention_tensor(
            [
                [0.10, 0.20, 0.30, 0.40, 0.00],
                [0.20, 0.10, 0.20, 0.50, 0.00],
            ],
            query_index=3,
        )
        fake_model = FakeAttentionModel(attentions)
        self.provider._runtime = make_runtime(fake_model)

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            response = self.provider.analyze_attention(
                HuggingFaceAttentionRequest(
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    model_revision="resolved-model-sha",
                    tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
                    tokenizer_revision="resolved-tokenizer-sha",
                    prompt_token_ids=[11, 12],
                    prompt_tokens=make_prompt_tokens(11, 12),
                    generated_token_ids=[1, 2, 3],
                    selected_generated_token_index=2,
                    selected_layer=1,
                    analysis_mode=HuggingFaceAttentionAnalysisMode.PREDICTION,
                    aggregation_mode=HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
                    max_connections=3,
                    max_context_tokens=256,
                )
            )

        self.assertEqual(fake_model.call_count, 1)
        self.assertEqual(fake_model.config._attn_implementation, "sdpa")
        self.assertEqual(response.analysis_mode, HuggingFaceAttentionAnalysisMode.PREDICTION)
        self.assertEqual(response.selected_token.full_position, 4)
        self.assertEqual(response.selected_token.generated_token_index, 2)
        self.assertTrue(response.selected_token.is_selected_token)
        self.assertEqual(response.query_token.full_position, 3)
        self.assertEqual(response.selected_token.raw_token, "Ä race")
        self.assertEqual(response.num_layers, 2)
        self.assertEqual(response.num_query_heads, 2)
        self.assertFalse(response.truncated_context)
        self.assertFalse(response.context_truncated)
        self.assertEqual(response.original_full_context_length, 5)
        self.assertEqual(response.analyzed_context_length, 5)
        self.assertEqual(response.selected_token_position, 4)
        self.assertEqual(response.query_position, 3)
        self.assertEqual(response.source_positions, [0, 1, 2, 3])
        self.assertAlmostEqual(response.attention_mass_sum, 1.0, places=4)
        self.assertEqual(
            [token.sequence_scope.value for token in response.analyzed_tokens],
            ["prompt", "prompt", "generated", "generated", "generated"],
        )
        self.assertEqual(
            [source.full_position for source in response.sources],
            [3, 2, 0],
        )
        self.assertEqual(
            [round(source.attention_weight, 3) for source in response.sources],
            [0.45, 0.25, 0.15],
        )

    def test_analyze_attention_single_head_and_top_n_filtering(self) -> None:
        attentions = make_attention_tensor(
            [
                [0.10, 0.20, 0.30, 0.40, 0.00],
                [0.15, 0.35, 0.25, 0.25, 0.00],
            ],
            query_index=3,
        )
        self.provider._runtime = make_runtime(FakeAttentionModel(attentions))

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            response = self.provider.analyze_attention(
                HuggingFaceAttentionRequest(
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    prompt_token_ids=[11, 12],
                    prompt_tokens=make_prompt_tokens(11, 12),
                    generated_token_ids=[1, 2, 3],
                    selected_generated_token_index=2,
                    selected_layer=0,
                    selected_head=1,
                    analysis_mode=HuggingFaceAttentionAnalysisMode.PREDICTION,
                    aggregation_mode=HuggingFaceAttentionAggregationMode.SINGLE_HEAD,
                    max_connections=2,
                    max_context_tokens=256,
                )
            )

        self.assertEqual(response.selected_head, 1)
        self.assertEqual(len(response.sources), 2)
        self.assertEqual([source.full_position for source in response.sources], [1, 2])
        self.assertAlmostEqual(sum(token.attention_weight or 0 for token in response.analyzed_tokens), 1.0, places=4)
        self.assertEqual(response.query_position, 3)
        self.assertEqual(response.selected_token_position, 4)
        self.assertIsNone(response.selected_token.attention_weight)

    def test_analyze_attention_truncates_suffix_and_preserves_original_positions(self) -> None:
        attentions = make_attention_tensor(
            [
                [0.30, 0.70, 0.00],
                [0.40, 0.60, 0.00],
            ],
            query_index=1,
        )
        self.provider._runtime = make_runtime(FakeAttentionModel(attentions))

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            response = self.provider.analyze_attention(
                HuggingFaceAttentionRequest(
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    prompt_token_ids=[11, 12],
                    prompt_tokens=make_prompt_tokens(11, 12),
                    generated_token_ids=[13, 14, 15, 16],
                    selected_generated_token_index=3,
                    selected_layer=0,
                    analysis_mode=HuggingFaceAttentionAnalysisMode.PREDICTION,
                    aggregation_mode=HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
                    max_connections=3,
                    max_context_tokens=3,
                    allow_truncated_recompute=True,
                )
            )

        self.assertTrue(response.truncated_context)
        self.assertTrue(response.context_truncated)
        self.assertEqual(response.original_full_context_length, 6)
        self.assertEqual(response.analyzed_context_length, 3)
        self.assertEqual(
            [token.full_position for token in response.analyzed_tokens],
            [3, 4, 5],
        )
        self.assertEqual(response.query_position, 4)
        self.assertEqual(response.source_positions, [3, 4])

    def test_analyze_attention_rejects_model_revision_mismatch(self) -> None:
        self.provider._runtime = make_runtime(FakeAttentionModel(make_attention_tensor([[0.5], [0.5]])))

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            with self.assertRaises(LLMScopeError) as caught:
                self.provider.analyze_attention(
                    HuggingFaceAttentionRequest(
                        model_id="Qwen/Qwen2.5-3B-Instruct",
                        model_revision="different-revision",
                        prompt_token_ids=[11, 12],
                        prompt_tokens=make_prompt_tokens(11, 12),
                        generated_token_ids=[1],
                        selected_generated_token_index=0,
                        selected_layer=0,
                        analysis_mode=HuggingFaceAttentionAnalysisMode.PREDICTION,
                        aggregation_mode=HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
                        max_connections=1,
                        max_context_tokens=256,
                    )
                )

        self.assertEqual(caught.exception.code, "HF_ATTENTION_MODEL_REVISION_MISMATCH")

    def test_analyze_attention_rejects_materially_invalid_prompt_truncation(self) -> None:
        self.provider._runtime = make_runtime(FakeAttentionModel(make_attention_tensor([[1.0], [1.0]])))

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            with self.assertRaises(LLMScopeError) as caught:
                self.provider.analyze_attention(
                    HuggingFaceAttentionRequest(
                        model_id="Qwen/Qwen2.5-3B-Instruct",
                        prompt_token_ids=list(range(300)),
                        prompt_tokens=[
                            CanonicalPromptToken(
                                token_id=token_id,
                                raw_token=f"tok-{token_id}",
                                display_token=f"tok-{token_id}",
                                decoded_contribution=f"tok-{token_id}",
                                token_bytes=list(f"tok-{token_id}".encode("utf-8")),
                                full_position=index,
                                source_category=CanonicalTokenSourceCategory.TEMPLATE,
                                source_label="Template / control",
                                special_token=False,
                            )
                            for index, token_id in enumerate(range(300))
                        ],
                        generated_token_ids=[1],
                        selected_generated_token_index=0,
                        selected_layer=0,
                        analysis_mode=HuggingFaceAttentionAnalysisMode.PREDICTION,
                        aggregation_mode=HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
                        max_connections=1,
                        max_context_tokens=256,
                    )
                )

        self.assertEqual(caught.exception.code, "HF_ATTENTION_CONTEXT_TOO_LONG")

    def test_analyze_attention_converts_cuda_oom_to_structured_error(self) -> None:
        self.provider._runtime = make_runtime(FakeAttentionModel(raises_oom=True))

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            with self.assertRaises(LLMScopeError) as caught:
                self.provider.analyze_attention(
                    HuggingFaceAttentionRequest(
                        model_id="Qwen/Qwen2.5-3B-Instruct",
                        prompt_token_ids=[11, 12],
                        prompt_tokens=make_prompt_tokens(11, 12),
                        generated_token_ids=[1],
                        selected_generated_token_index=0,
                        selected_layer=0,
                        analysis_mode=HuggingFaceAttentionAnalysisMode.PREDICTION,
                        aggregation_mode=HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
                        max_connections=1,
                        max_context_tokens=256,
                    )
                )

        self.assertEqual(caught.exception.code, "HF_ATTENTION_OOM")

    def test_analyze_attention_uses_bounded_cache(self) -> None:
        attentions = make_attention_tensor(
            [
                [0.10, 0.20, 0.30, 0.40, 0.00],
                [0.20, 0.10, 0.20, 0.50, 0.00],
            ],
            query_index=3,
        )
        fake_model = FakeAttentionModel(attentions)
        self.provider._runtime = make_runtime(fake_model)
        request = HuggingFaceAttentionRequest(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            prompt_token_ids=[11, 12],
            prompt_tokens=make_prompt_tokens(11, 12),
            generated_token_ids=[1, 2, 3],
            selected_generated_token_index=2,
            selected_layer=1,
            analysis_mode=HuggingFaceAttentionAnalysisMode.PREDICTION,
            aggregation_mode=HuggingFaceAttentionAggregationMode.AVERAGE_HEADS,
            max_connections=3,
            max_context_tokens=256,
        )

        with patch.object(self.provider, "_ensure_cuda_available", return_value=None):
            first = self.provider.analyze_attention(request)
            second = self.provider.analyze_attention(request)

        self.assertEqual(fake_model.call_count, 1)
        self.assertEqual(first.sources[0].full_position, second.sources[0].full_position)

    def test_unload_clears_runtime(self) -> None:
        self.provider._runtime = make_runtime(
            FakeModel([torch.tensor([0.0, 1.0, 0.5, 0.2, -1.0], dtype=torch.float32)])
        )

        with patch.object(
            huggingface_provider_module,
            "collect_huggingface_local_diagnostics",
            return_value=make_diagnostics(),
        ):
            status = self.provider.unload_model(raise_if_busy=False)

        self.assertIsNone(self.provider._runtime)
        self.assertIsNone(status.active_model_id)


if __name__ == "__main__":
    unittest.main()
