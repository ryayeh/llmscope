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
from app.schemas.huggingface_local import HuggingFaceLocalDiagnosticsResponse


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 4
        self.eos_token = "<eos>"
        self.name_or_path = "Qwen/Qwen2.5-3B-Instruct"
        self.init_kwargs = {"_commit_hash": "resolved-tokenizer-sha"}
        self._raw_tokens = {
            1: "Ġtime",
            2: "Ġsplit",
            3: "Ġrace",
            4: "<eos>",
        }
        self._decoded_tokens = {
            1: " time",
            2: " split",
            3: " race",
            4: "",
        }

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_tensors):
        del messages, add_generation_prompt, tokenize, return_tensors
        return torch.tensor([[11, 12]], dtype=torch.long)

    def decode(self, token_ids, clean_up_tokenization_spaces, skip_special_tokens):
        del clean_up_tokenization_spaces, skip_special_tokens
        return "".join(self._decoded_tokens.get(token_id, f"<{token_id}>") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return [self._raw_tokens.get(token_id, f"tok-{token_id}") for token_id in token_ids]


class FakeModel:
    def __init__(self, logits_per_step):
        self._logits_per_step = logits_per_step
        self._call_index = 0
        self._parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        self.config = type("Config", (), {"_commit_hash": "resolved-model-sha"})()
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
        transformers_version="4.56.0",
        torch_cuda_runtime="12.6",
        gpu_name="NVIDIA GeForce RTX 3060" if cuda_available else None,
        gpu_total_vram_gb=gpu_total_vram_gb if cuda_available else None,
        gpu_free_vram_gb=gpu_free_vram_gb if cuda_available else None,
        python_version="3.14.3",
        platform="Windows-11",
        disk_free_gb=26.5,
        missing_dependencies=missing_dependencies or [],
    )


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
        self.provider._runtime = huggingface_provider_module.LoadedLocalRuntime(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            label="Qwen2.5 3B Instruct",
            revision="rev-3b",
            resolved_revision="resolved-model-sha",
            tokenizer=FakeTokenizer(),
            model=FakeModel(logits_per_step),
            device="cuda:0",
            dtype="float16",
        )

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
        self.assertEqual(result.tokens[0].token, "Ġtime")
        self.assertEqual(result.tokens[0].decoded_contribution, " time")
        self.assertEqual(result.tokens[0].token_id, 1)
        self.assertEqual(result.tokens[0].cumulative_token_ids, [11, 12, 1])
        self.assertEqual(result.tokens[0].alternatives[0].token, "Ġsplit")
        self.assertEqual(result.tokens[0].alternatives[0].decoded_contribution, " split")
        self.assertGreater(result.tokens[0].probability or 0, result.tokens[0].alternatives[0].probability or 0)
        self.assertEqual(result.tokens[1].context_before, " time")
        self.assertEqual(result.tokens[1].context_after, " time split")
        self.assertEqual(result.tokens[1].cumulative_token_ids, [11, 12, 1, 2])

    def test_generate_can_continue_from_canonical_prefix_token_ids(self) -> None:
        self.provider._runtime = huggingface_provider_module.LoadedLocalRuntime(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            label="Qwen2.5 3B Instruct",
            revision="rev-3b",
            resolved_revision="resolved-model-sha",
            tokenizer=FakeTokenizer(),
            model=FakeModel([torch.tensor([0.0, -1.0, 2.2, 1.1, -3.0], dtype=torch.float32)]),
            device="cuda:0",
            dtype="float16",
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
        self.assertEqual(result.tokens[0].token, "Ġsplit")

    def test_generate_returns_busy_error_when_runtime_is_locked(self) -> None:
        self.provider._runtime = huggingface_provider_module.LoadedLocalRuntime(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            label="Qwen2.5 3B Instruct",
            revision="rev-3b",
            resolved_revision="resolved-model-sha",
            tokenizer=FakeTokenizer(),
            model=FakeModel([torch.tensor([0.0, 1.0, 0.5, 0.2, -1.0], dtype=torch.float32)]),
            device="cuda:0",
            dtype="float16",
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

    def test_unload_clears_runtime(self) -> None:
        self.provider._runtime = huggingface_provider_module.LoadedLocalRuntime(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            label="Qwen2.5 3B Instruct",
            revision="rev-3b",
            resolved_revision="resolved-model-sha",
            tokenizer=FakeTokenizer(),
            model=FakeModel([torch.tensor([0.0, 1.0, 0.5, 0.2, -1.0], dtype=torch.float32)]),
            device="cuda:0",
            dtype="float16",
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
