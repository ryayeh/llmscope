import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.errors import LLMScopeError
from app.models.provider import ModelProvider
from app.schemas.generation import (
    AlternativeCandidate,
    ContinueGenerationRequest,
    ContinuationMode,
    GenerationRequest,
    NodeExpansionRequest,
    TokenTrace,
)
from app.services.generation_service import GenerationService, ProviderCapabilities


class GenerationServiceCanonicalStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = GenerationService()

    def _make_logprob_entry(
        self,
        token: str,
        probability: float,
        *,
        top_alternatives: list[tuple[str, float]] | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            token=token,
            logprob=math.log(probability),
            bytes=list(token.encode("utf-8")),
            token_id=None,
            tokenizer_id=None,
            top_logprobs=[
                SimpleNamespace(
                    token=alt_token,
                    logprob=math.log(alt_probability),
                    bytes=list(alt_token.encode("utf-8")),
                    token_id=None,
                    tokenizer_id=None,
                )
                for alt_token, alt_probability in (top_alternatives or [])
            ],
        )

    def _make_response(self, *entries: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text",
                            text="".join(entry.token for entry in entries),
                            logprobs=list(entries),
                        )
                    ],
                )
            ],
            usage=None,
            status="completed",
        )

    def test_ranked_candidates_preserve_cumulative_decoded_text_and_token_history(self) -> None:
        ranked, _ = self.service._build_ranked_candidates(
            chosen_token=" For",
            chosen_logprob=math.log(0.62),
            chosen_bytes=list(" For".encode("utf-8")),
            chosen_token_id=11,
            chosen_tokenizer_id=99,
            top_logprobs=[
                SimpleNamespace(
                    token=" Against",
                    logprob=math.log(0.21),
                    bytes=list(" Against".encode("utf-8")),
                    token_id=12,
                    tokenizer_id=99,
                )
            ],
            context_before="A good 400m time",
            generation_step=5,
            latency_ms=12,
            parent_cumulative_log_probability=math.log(0.5),
            parent_cumulative_token_ids=[1, 2, 3],
        )

        self.assertEqual(ranked[0]["decoded_contribution"], " For")
        self.assertEqual(ranked[0]["cumulative_decoded_text"], "A good 400m time For")
        self.assertEqual(ranked[0]["cumulative_token_ids"], [1, 2, 3, 11])
        self.assertEqual(
            ranked[0]["cumulative_log_probability"],
            round(math.log(0.5) + math.log(0.62), 6),
        )

    def test_live_traces_chain_canonical_prefix_step_by_step(self) -> None:
        traces = self.service._build_live_token_traces(
            branch_id="main",
            model="gpt-4.1-mini",
            source="openai",
            parent_node_id="root",
            context_prefix="A good ",
            logprob_entries=[
                SimpleNamespace(
                    token="400",
                    logprob=math.log(0.7),
                    bytes=list(b"400"),
                    token_id=21,
                    tokenizer_id=7,
                    top_logprobs=[
                        SimpleNamespace(
                            token="300",
                            logprob=math.log(0.12),
                            bytes=list(b"300"),
                            token_id=22,
                            tokenizer_id=7,
                        )
                    ],
                ),
                SimpleNamespace(
                    token="m",
                    logprob=math.log(0.65),
                    bytes=list(b"m"),
                    token_id=23,
                    tokenizer_id=7,
                    top_logprobs=[
                        SimpleNamespace(
                            token=" meters",
                            logprob=math.log(0.15),
                            bytes=list(" meters".encode("utf-8")),
                            token_id=24,
                            tokenizer_id=7,
                        )
                    ],
                ),
            ],
            latency_ms=24,
        )

        self.assertEqual(traces[0].context_before, "A good ")
        self.assertEqual(traces[0].cumulative_decoded_text, "A good 400")
        self.assertEqual(traces[1].context_before, "A good 400")
        self.assertEqual(traces[1].cumulative_decoded_text, "A good 400m")
        self.assertEqual(
            traces[1].alternatives[0].cumulative_decoded_text,
            "A good 400 meters",
        )

    def test_expansion_children_inherit_selected_parent_output_step(self) -> None:
        request = NodeExpansionRequest(
            root_prompt="Prompt",
            model="gpt-4.1-mini",
            preset="general",
            temperature=0.7,
            top_p=1.0,
            parent_node_id="branch-parent",
            parent_token="m",
            assistant_prefix="A good 400m",
            depth=3,
            cumulative_probability=0.41,
            variation=0,
            max_children=4,
            demo_mode=False,
        )
        trace = TokenTrace(
            id="trace",
            branch_id="branch-parent",
            parent_node_id="branch-parent",
            model="gpt-4.1-mini",
            source="openai",
            index=0,
            position=0,
            token=" time",
            display_token="time",
            token_bytes=list(" time".encode("utf-8")),
            decoded_contribution=" time",
            cumulative_decoded_text="A good 400m time",
            cumulative_token_ids=[11, 12, 13, 14],
            cumulative_log_probability=math.log(0.2),
            token_id=14,
            tokenizer_id=7,
            probability=0.64,
            raw_probability=0.64,
            normalized_displayed_probability=0.8,
            log_probability=math.log(0.64),
            entropy=0.42,
            cumulative_probability=0.2,
            latency_ms=18,
            text_preview="A good 400m time",
            context_before="A good 400m",
            context_after="A good 400m time",
            finish_reason=None,
            alternatives=[
                AlternativeCandidate(
                    token=" split",
                    display_token="split",
                    token_bytes=list(" split".encode("utf-8")),
                    decoded_contribution=" split",
                    cumulative_decoded_text="A good 400m split",
                    cumulative_token_ids=[11, 12, 13, 15],
                    cumulative_log_probability=math.log(0.05),
                    probability=0.08,
                    raw_probability=0.08,
                    normalized_displayed_probability=0.1,
                    log_probability=math.log(0.08),
                    entropy=0.42,
                    latency_ms=18,
                    token_id=15,
                    tokenizer_id=7,
                    rank=2,
                    text_preview="A good 400m split",
                    context_before="A good 400m",
                    context_after="A good 400m split",
                    generation_step=0,
                )
            ],
            generation_step=0,
        )

        children = self.service._build_expansion_children_from_trace(
            request=request,
            trace=trace,
            latency_ms=18,
        )

        self.assertTrue(all(child.generation_step == 3 for child in children))
        self.assertTrue(all(child.depth == 4 for child in children))
        self.assertEqual(children[0].cumulative_token_ids, [11, 12, 13, 14])
        self.assertEqual(children[1].cumulative_token_ids, [11, 12, 13, 15])

    def test_ranked_candidates_keep_raw_model_probability_separate_from_normalized_probability(self) -> None:
        ranked, _ = self.service._build_ranked_candidates(
            chosen_token="m",
            chosen_logprob=math.log(0.648),
            chosen_bytes=list(b"m"),
            chosen_token_id=31,
            chosen_tokenizer_id=7,
            top_logprobs=[
                SimpleNamespace(
                    token=" meters",
                    logprob=math.log(0.125),
                    bytes=list(" meters".encode("utf-8")),
                    token_id=32,
                    tokenizer_id=7,
                ),
                SimpleNamespace(
                    token=" for",
                    logprob=math.log(0.082),
                    bytes=list(" for".encode("utf-8")),
                    token_id=33,
                    tokenizer_id=7,
                ),
            ],
            context_before="A good 400",
            generation_step=1,
            latency_ms=12,
            parent_cumulative_log_probability=math.log(0.7),
            parent_cumulative_token_ids=[21],
        )

        chosen = ranked[0]
        raw_sum = sum(candidate["raw_probability"] for candidate in ranked)

        self.assertAlmostEqual(chosen["raw_probability"], 0.648, places=6)
        self.assertLess(chosen["normalized_displayed_probability"], 1.0)
        self.assertAlmostEqual(raw_sum, 0.855, places=6)
        self.assertAlmostEqual(
            chosen["normalized_displayed_probability"],
            round(0.648 / 0.855, 6),
            places=6,
        )

    def test_continuation_context_rejects_display_markers(self) -> None:
        request = NodeExpansionRequest(
            root_prompt="Prompt",
            model="gpt-4.1-mini",
            preset="general",
            temperature=0.7,
            top_p=1.0,
            parent_node_id="node-1",
            parent_token=" For",
            assistant_prefix="A good\u2420For",
            depth=1,
            cumulative_probability=0.5,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with self.assertRaises(LLMScopeError) as caught:
            self.service._build_continuation_context(request)

        self.assertEqual(caught.exception.code, "CONTINUATION_CONTEXT_MISMATCH")
        self.assertIn("display-only whitespace markers", caught.exception.message)

    def test_huggingface_continuation_context_uses_token_ids_for_exact_validation(self) -> None:
        request = NodeExpansionRequest(
            root_prompt="Prompt",
            provider=ModelProvider.HUGGING_FACE,
            model="Qwen/Qwen2.5-3B-Instruct",
            preset="general",
            temperature=0.0,
            top_p=1.0,
            parent_node_id="hf-node-2",
            parent_token="Ä time",
            assistant_prefix="The time",
            prompt_token_ids=[11, 12],
            canonical_prefix_token_ids=[11, 12, 101, 102],
            generated_prefix_token_ids=[101, 102],
            reconstructed_prompt="PromptThe time",
            expected_prompt_length=len("PromptThe time"),
            expected_utf8_length=len("PromptThe time".encode("utf-8")),
            expected_assistant_prefix_length=len("The time"),
            expected_assistant_prefix_utf8_length=len("The time".encode("utf-8")),
            expected_token_count=2,
            selected_token_id=102,
            selected_tokenizer_id=102,
            model_revision="resolved-model-sha",
            tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision="resolved-tokenizer-sha",
            depth=2,
            cumulative_probability=0.42,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with patch.object(
            self.service._huggingface_provider,
            "get_runtime_identity",
            return_value=SimpleNamespace(
                model_id="Qwen/Qwen2.5-3B-Instruct",
                model_revision="resolved-model-sha",
                tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
                tokenizer_revision="resolved-tokenizer-sha",
            ),
        ):
            context = self.service._build_continuation_context(request)

        self.assertEqual(context.prompt_token_ids, [11, 12])
        self.assertEqual(context.canonical_prefix_token_ids, [11, 12, 101, 102])
        self.assertEqual(context.generated_prefix_token_ids, [101, 102])
        self.assertEqual(context.assistant_prefix, "The time")
        self.assertEqual(context.token_count, 2)
        self.assertEqual(context.assistant_character_length, 8)
        self.assertEqual(context.assistant_utf8_length, 8)

    def test_huggingface_continuation_context_rejects_generated_prefix_id_mismatch(self) -> None:
        request = NodeExpansionRequest(
            root_prompt="Prompt",
            provider=ModelProvider.HUGGING_FACE,
            model="Qwen/Qwen2.5-3B-Instruct",
            preset="general",
            temperature=0.0,
            top_p=1.0,
            parent_node_id="hf-node-2",
            parent_token="Ä time",
            assistant_prefix="The time",
            prompt_token_ids=[11, 12],
            canonical_prefix_token_ids=[11, 12, 101, 102],
            generated_prefix_token_ids=[101, 999],
            expected_assistant_prefix_length=len("The time"),
            expected_assistant_prefix_utf8_length=len("The time".encode("utf-8")),
            expected_token_count=2,
            selected_token_id=102,
            selected_tokenizer_id=102,
            model_revision="resolved-model-sha",
            tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision="resolved-tokenizer-sha",
            depth=2,
            cumulative_probability=0.42,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with patch.object(
            self.service._huggingface_provider,
            "get_runtime_identity",
            return_value=SimpleNamespace(
                model_id="Qwen/Qwen2.5-3B-Instruct",
                model_revision="resolved-model-sha",
                tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
                tokenizer_revision="resolved-tokenizer-sha",
            ),
        ):
            with self.assertRaises(LLMScopeError) as caught:
                self.service._build_continuation_context(request)

        self.assertEqual(caught.exception.code, "HF_LOCAL_GENERATED_PREFIX_IDS_MISMATCH")

    def test_huggingface_expansion_children_accept_full_canonical_token_history(self) -> None:
        request = ContinueGenerationRequest(
            root_prompt="Prompt",
            provider=ModelProvider.HUGGING_FACE,
            model="Qwen/Qwen2.5-3B-Instruct",
            preset="general",
            temperature=0.0,
            top_p=1.0,
            parent_node_id="hf-node-2",
            parent_token="Ä time",
            assistant_prefix="The time",
            prompt_token_ids=[11, 12],
            canonical_prefix_token_ids=[11, 12, 101, 102],
            generated_prefix_token_ids=[101, 102],
            expected_assistant_prefix_length=len("The time"),
            expected_assistant_prefix_utf8_length=len("The time".encode("utf-8")),
            expected_token_count=2,
            selected_token_id=102,
            selected_tokenizer_id=102,
            model_revision="resolved-model-sha",
            tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision="resolved-tokenizer-sha",
            depth=2,
            cumulative_probability=0.42,
            variation=0,
            max_children=4,
            demo_mode=False,
        )
        trace = TokenTrace(
            id="hf-trace",
            branch_id="hf-node-2",
            parent_node_id="hf-node-2",
            model="Qwen/Qwen2.5-3B-Instruct",
            source="hugging_face",
            index=0,
            position=0,
            token="Ä split",
            display_token="split",
            token_bytes=list(" split".encode("utf-8")),
            decoded_contribution=" split",
            cumulative_decoded_text="The time split",
            cumulative_token_ids=[11, 12, 101, 102, 103],
            cumulative_log_probability=math.log(0.2),
            token_id=103,
            tokenizer_id=103,
            probability=0.61,
            raw_probability=0.61,
            normalized_displayed_probability=0.74,
            log_probability=math.log(0.61),
            entropy=0.31,
            cumulative_probability=0.2,
            latency_ms=15,
            text_preview="The time split",
            context_before="The time",
            context_after="The time split",
            finish_reason=None,
            alternatives=[
                AlternativeCandidate(
                    token="Ä race",
                    display_token="race",
                    token_bytes=list(" race".encode("utf-8")),
                    decoded_contribution=" race",
                    cumulative_decoded_text="The time race",
                    cumulative_token_ids=[11, 12, 101, 102, 104],
                    cumulative_log_probability=math.log(0.1),
                    probability=0.18,
                    raw_probability=0.18,
                    normalized_displayed_probability=0.22,
                    log_probability=math.log(0.18),
                    entropy=0.31,
                    latency_ms=15,
                    token_id=104,
                    tokenizer_id=104,
                    rank=2,
                    text_preview="The time race",
                    context_before="The time",
                    context_after="The time race",
                    generation_step=0,
                )
            ],
            generation_step=0,
        )

        children = self.service._build_expansion_children_from_trace(
            request=request,
            trace=trace,
            latency_ms=15,
        )

        self.assertEqual(children[0].parent_node_id, "hf-node-2")
        self.assertEqual(children[0].cumulative_token_ids, [11, 12, 101, 102, 103])
        self.assertEqual(children[1].cumulative_token_ids, [11, 12, 101, 102, 104])
        self.assertEqual(children[0].generation_step, 2)
        self.assertEqual(children[0].depth, 3)

    def test_continue_generation_with_huggingface_exact_prefix_attaches_to_selected_parent(self) -> None:
        captured_generate: dict[str, object] = {}

        request = ContinueGenerationRequest(
            root_prompt="Prompt",
            provider=ModelProvider.HUGGING_FACE,
            model="Qwen/Qwen2.5-3B-Instruct",
            preset="general",
            temperature=0.0,
            top_p=1.0,
            parent_node_id="hf-node-2",
            parent_token="Ä time",
            assistant_prefix="The time",
            prompt_token_ids=[11, 12],
            canonical_prefix_token_ids=[11, 12, 101, 102],
            generated_prefix_token_ids=[101, 102],
            reconstructed_prompt="PromptThe time",
            expected_prompt_length=len("PromptThe time"),
            expected_utf8_length=len("PromptThe time".encode("utf-8")),
            expected_assistant_prefix_length=len("The time"),
            expected_assistant_prefix_utf8_length=len("The time".encode("utf-8")),
            expected_token_count=2,
            selected_token_id=102,
            selected_tokenizer_id=102,
            model_revision="resolved-model-sha",
            tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision="resolved-tokenizer-sha",
            depth=2,
            cumulative_probability=0.42,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        def fake_generate(**kwargs):
            captured_generate.update(kwargs)
            return SimpleNamespace(
                latency_ms=15,
                tokens=[
                    TokenTrace(
                        id="hf-trace",
                        branch_id="hf-node-2",
                        parent_node_id="hf-node-2",
                        model="Qwen/Qwen2.5-3B-Instruct",
                        source="hugging_face",
                        index=0,
                        position=0,
                        token="Ä split",
                        display_token="split",
                        token_bytes=list(" split".encode("utf-8")),
                        decoded_contribution=" split",
                        cumulative_decoded_text="The time split",
                        cumulative_token_ids=[11, 12, 101, 102, 103],
                        cumulative_log_probability=math.log(0.2),
                        token_id=103,
                        tokenizer_id=103,
                        probability=0.61,
                        raw_probability=0.61,
                        normalized_displayed_probability=0.74,
                        log_probability=math.log(0.61),
                        entropy=0.31,
                        cumulative_probability=0.2,
                        latency_ms=15,
                        text_preview="The time split",
                        context_before="The time",
                        context_after="The time split",
                        finish_reason=None,
                        alternatives=[
                            AlternativeCandidate(
                                token="Ä race",
                                display_token="race",
                                token_bytes=list(" race".encode("utf-8")),
                                decoded_contribution=" race",
                                cumulative_decoded_text="The time race",
                                cumulative_token_ids=[11, 12, 101, 102, 104],
                                cumulative_log_probability=math.log(0.1),
                                probability=0.18,
                                raw_probability=0.18,
                                normalized_displayed_probability=0.22,
                                log_probability=math.log(0.18),
                                entropy=0.31,
                                latency_ms=15,
                                token_id=104,
                                tokenizer_id=104,
                                rank=2,
                                text_preview="The time race",
                                context_before="The time",
                                context_after="The time race",
                                generation_step=0,
                            )
                        ],
                        generation_step=0,
                    )
                ],
            )

        with (
            patch.object(
                self.service._huggingface_provider,
                "get_runtime_identity",
                return_value=SimpleNamespace(
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    model_revision="resolved-model-sha",
                    tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
                    tokenizer_revision="resolved-tokenizer-sha",
                ),
            ),
            patch.object(
                self.service._huggingface_provider,
                "generate",
                side_effect=fake_generate,
            ) as mocked_generate,
            patch.object(
                self.service._huggingface_provider,
                "load_model",
                side_effect=AssertionError("Continuation should not reload the model"),
            ),
            patch.object(
                self.service._huggingface_provider,
                "unload_model",
                side_effect=AssertionError("Continuation should not unload the model"),
            ),
        ):
            response = self.service.continue_node(request)

        self.assertEqual(mocked_generate.call_count, 1)
        self.assertEqual(captured_generate["canonical_prefix_token_ids"], [11, 12, 101, 102])
        self.assertEqual(captured_generate["prompt_token_ids"], [11, 12])
        self.assertEqual(response.action, "new_provider_segment")
        self.assertEqual(response.continuation_mode, ContinuationMode.EXACT)
        self.assertEqual(response.parent_node_id, "hf-node-2")
        self.assertEqual(response.children[0].parent_node_id, "hf-node-2")
        self.assertEqual(response.children[0].token_id, 103)
        self.assertEqual(response.children[1].token_id, 104)
        self.assertEqual(response.children[0].cumulative_token_ids, [11, 12, 101, 102, 103])
        self.assertEqual(response.children[1].cumulative_token_ids, [11, 12, 101, 102, 104])
        self.assertEqual(response.children[0].probability, 0.61)
        self.assertEqual(response.children[1].raw_probability, 0.18)

    def test_continue_generation_with_huggingface_selected_alternative_token_id_uses_branch_prefix(self) -> None:
        captured_generate: dict[str, object] = {}

        request = ContinueGenerationRequest(
            root_prompt="Prompt",
            provider=ModelProvider.HUGGING_FACE,
            model="Qwen/Qwen2.5-3B-Instruct",
            preset="general",
            temperature=0.0,
            top_p=1.0,
            parent_node_id="hf-alt-2",
            parent_token="Ä race",
            assistant_prefix="The race",
            prompt_token_ids=[11, 12],
            canonical_prefix_token_ids=[11, 12, 101, 104],
            generated_prefix_token_ids=[101, 104],
            expected_assistant_prefix_length=len("The race"),
            expected_assistant_prefix_utf8_length=len("The race".encode("utf-8")),
            expected_token_count=2,
            selected_token_id=104,
            selected_tokenizer_id=104,
            model_revision="resolved-model-sha",
            tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision="resolved-tokenizer-sha",
            depth=2,
            cumulative_probability=0.18,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        def fake_generate(**kwargs):
            captured_generate.update(kwargs)
            return SimpleNamespace(
                latency_ms=11,
                tokens=[
                    TokenTrace(
                        id="hf-branch-trace",
                        branch_id="hf-alt-2",
                        parent_node_id="hf-alt-2",
                        model="Qwen/Qwen2.5-3B-Instruct",
                        source="hugging_face",
                        index=0,
                        position=0,
                        token="Ä pace",
                        display_token="pace",
                        token_bytes=list(" pace".encode("utf-8")),
                        decoded_contribution=" pace",
                        cumulative_decoded_text="The race pace",
                        cumulative_token_ids=[11, 12, 101, 104, 105],
                        cumulative_log_probability=math.log(0.07),
                        token_id=105,
                        tokenizer_id=105,
                        probability=0.39,
                        raw_probability=0.39,
                        normalized_displayed_probability=0.6,
                        log_probability=math.log(0.39),
                        entropy=0.41,
                        cumulative_probability=0.07,
                        latency_ms=11,
                        text_preview="The race pace",
                        context_before="The race",
                        context_after="The race pace",
                        finish_reason=None,
                        alternatives=[],
                        generation_step=0,
                    )
                ],
            )

        with (
            patch.object(
                self.service._huggingface_provider,
                "get_runtime_identity",
                return_value=SimpleNamespace(
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    model_revision="resolved-model-sha",
                    tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
                    tokenizer_revision="resolved-tokenizer-sha",
                ),
            ),
            patch.object(
                self.service._huggingface_provider,
                "generate",
                side_effect=fake_generate,
            ),
        ):
            response = self.service.continue_node(request)

        self.assertEqual(captured_generate["canonical_prefix_token_ids"], [11, 12, 101, 104])
        self.assertEqual(response.children[0].parent_node_id, "hf-alt-2")
        self.assertEqual(response.children[0].cumulative_token_ids, [11, 12, 101, 104, 105])

    def test_continue_generation_with_huggingface_model_not_ready_returns_503(self) -> None:
        request = ContinueGenerationRequest(
            root_prompt="Prompt",
            provider=ModelProvider.HUGGING_FACE,
            model="Qwen/Qwen2.5-3B-Instruct",
            preset="general",
            temperature=0.0,
            top_p=1.0,
            parent_node_id="hf-node-2",
            parent_token="Ä time",
            assistant_prefix="The time",
            prompt_token_ids=[11, 12],
            canonical_prefix_token_ids=[11, 12, 101, 102],
            generated_prefix_token_ids=[101, 102],
            expected_assistant_prefix_length=len("The time"),
            expected_assistant_prefix_utf8_length=len("The time".encode("utf-8")),
            expected_token_count=2,
            selected_token_id=102,
            selected_tokenizer_id=102,
            model_revision="resolved-model-sha",
            tokenizer_identity="Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision="resolved-tokenizer-sha",
            depth=2,
            cumulative_probability=0.42,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with patch.object(
            self.service._huggingface_provider,
            "get_runtime_identity",
            side_effect=LLMScopeError(
                code="HF_LOCAL_MODEL_NOT_READY",
                message="The selected Hugging Face Local model is not loaded. Select the model and click Load before generating.",
                status_code=503,
            ),
        ):
            with self.assertRaises(LLMScopeError) as caught:
                self.service.continue_node(request)

        self.assertEqual(caught.exception.code, "HF_LOCAL_MODEL_NOT_READY")
        self.assertEqual(caught.exception.status_code, 503)

    def test_unavailable_logprobs_raise_explicit_error_code(self) -> None:
        with self.assertRaises(LLMScopeError) as caught:
            self.service._raise_logprobs_unavailable()

        self.assertEqual(caught.exception.code, "TOP_LOGPROBS_UNAVAILABLE")
        self.assertIn("did not include token alternatives", caught.exception.message)

    def test_initial_generation_tokens_are_tagged_exact(self) -> None:
        response = self.service.build_response(
            GenerationRequest(
                prompt="What is a good 400m time for a 16-year-old?",
                demo_mode=True,
            )
        )

        self.assertGreater(len(response.tokens), 0)
        segment_ids = {token.segment_id for token in response.tokens}
        self.assertEqual(len(segment_ids), 1)
        self.assertNotIn(None, segment_ids)
        shared_segment_id = next(iter(segment_ids))

        for token in response.tokens:
            self.assertEqual(token.continuation_mode, ContinuationMode.EXACT)
            self.assertEqual(token.segment_id, shared_segment_id)
            self.assertEqual(token.metadata.get("continuation_mode"), ContinuationMode.EXACT.value)
            for alternative in token.alternatives:
                self.assertEqual(alternative.continuation_mode, ContinuationMode.EXACT)
                self.assertEqual(alternative.segment_id, shared_segment_id)

    def test_continue_generation_requests_use_exact_continuation_payload_and_cache_remaining_tokens(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return self_outer._make_response(
                    self_outer._make_logprob_entry(
                        " old",
                        0.41,
                        top_alternatives=[(" A", 0.19), (" between", 0.12)],
                    ),
                    self_outer._make_logprob_entry(
                        " athlete",
                        0.27,
                        top_alternatives=[(" runner", 0.12)],
                    ),
                )

        self_outer = self
        request = ContinueGenerationRequest(
            root_prompt="What is a good 400m time for a 16-year-old?",
            model="gpt-4.1-mini",
            preset="general",
            temperature=0.7,
            top_p=1.0,
            parent_node_id="node-1",
            parent_token="ranges",
            assistant_prefix="A good 400m time for a 16-year-old typically ranges",
            depth=7,
            cumulative_probability=0.5,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with (
            patch.object(
                self.service,
                "_get_client",
                return_value=SimpleNamespace(responses=FakeResponses()),
            ),
            patch.object(
                self.service,
                "_provider_capabilities_for_model",
                return_value=ProviderCapabilities(
                    supports_native_continuation=True,
                    supports_token_logprobs=True,
                    minimum_output_tokens=16,
                    supports_continuation=True,
                ),
            ),
        ):
            response = self.service.continue_node(request)

        self.assertEqual(captured["max_output_tokens"], 16)
        self.assertEqual(
            captured["input"],
            [
                {
                    "role": "user",
                    "content": "What is a good 400m time for a 16-year-old?",
                },
                {
                    "role": "assistant",
                    "content": "A good 400m time for a 16-year-old typically ranges",
                },
            ],
        )
        instructions = str(captured["instructions"])
        self.assertIn(
            "Treat the supplied assistant prefix as the active assistant turn and continue from its exact endpoint.",
            instructions,
        )
        self.assertIn(
            "Emit only the immediate next assistant text that follows that exact prefix.",
            instructions,
        )
        self.assertNotIn("Answer directly", instructions)
        self.assertNotIn("Be accurate, direct, and useful.", instructions)
        self.assertNotIn("If the user asks for a benchmark", instructions)
        self.assertEqual(response.action, "new_provider_segment")
        self.assertEqual(response.continuation_mode, ContinuationMode.EXACT)
        self.assertEqual(response.children[0].token, " old")
        self.assertEqual(response.cached_token_count, 2)
        self.assertEqual(response.remaining_cached_tokens, 1)
        self.assertIsNotNone(response.segment_id)
        self.assertEqual(response.children[0].continuation_mode, ContinuationMode.EXACT)
        self.assertEqual(response.children[0].segment_id, response.segment_id)
        self.assertEqual(
            response.children[0].metadata.get("next_cached_token_index"),
            1,
        )
        self.assertEqual(
            response.children[0].metadata.get("cached_segment_id"),
            response.segment_id,
        )
        self.assertEqual(
            response.children[0].context_before,
            "A good 400m time for a 16-year-old typically ranges",
        )
        self.assertEqual(response.children[0].token, " old")
        self.assertTrue(any(child.token == " A" for child in response.children[1:]))

    def test_continue_generation_reveals_cached_tokens_without_extra_provider_calls(self) -> None:
        provider_call_count = 0
        self_outer = self

        class FakeResponses:
            def __init__(self) -> None:
                self._responses = [
                    self_outer._make_response(
                        self_outer._make_logprob_entry(
                            " A",
                            0.52,
                            top_alternatives=[(" The", 0.18), (" One", 0.12)],
                        ),
                        self_outer._make_logprob_entry(
                            " good",
                            0.49,
                            top_alternatives=[(" fast", 0.14)],
                        ),
                        self_outer._make_logprob_entry(
                            " time",
                            0.43,
                            top_alternatives=[(" mark", 0.11)],
                        ),
                    ),
                    self_outer._make_response(
                        self_outer._make_logprob_entry(
                            " alternative",
                            0.47,
                            top_alternatives=[(" branch", 0.15)],
                        ),
                    ),
                ]

            def create(self, **kwargs):
                nonlocal provider_call_count
                provider_call_count += 1
                return self._responses.pop(0)

        request = ContinueGenerationRequest(
            root_prompt="Prompt",
            model="gpt-4.1-mini",
            preset="general",
            temperature=0.7,
            top_p=1.0,
            parent_node_id="node-1",
            parent_token="time",
            assistant_prefix="A good 400m time",
            depth=4,
            cumulative_probability=0.5,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with (
            patch.object(
                self.service,
                "_get_client",
                return_value=SimpleNamespace(responses=FakeResponses()),
            ),
            patch.object(
                self.service,
                "_provider_capabilities_for_model",
                return_value=ProviderCapabilities(
                    supports_native_continuation=True,
                    supports_token_logprobs=True,
                    minimum_output_tokens=16,
                    supports_continuation=True,
                ),
            ),
        ):
            request_defaults = request.model_dump(
                exclude={
                    "parent_node_id",
                    "parent_token",
                    "assistant_prefix",
                    "depth",
                    "cumulative_probability",
                    "cached_segment_id",
                    "cached_token_index",
                }
            )
            first = self.service.continue_node(request)
            self.assertEqual(provider_call_count, 1)
            chosen_first = first.children[0]
            second = self.service.continue_node(
                ContinueGenerationRequest(
                    **request_defaults,
                    parent_node_id=chosen_first.id,
                    parent_token=chosen_first.token,
                    assistant_prefix=chosen_first.context_after,
                    depth=chosen_first.depth,
                    cumulative_probability=chosen_first.cumulative_probability,
                    cached_segment_id=str(chosen_first.metadata.get("cached_segment_id")),
                    cached_token_index=int(chosen_first.metadata.get("next_cached_token_index")),
                )
            )
            self.assertEqual(provider_call_count, 1)
            self.assertEqual(second.action, "reveal_cached")
            self.assertEqual(second.continuation_mode, ContinuationMode.EXACT)
            self.assertEqual(second.children[0].token, " good")
            self.assertEqual(second.children[0].continuation_mode, ContinuationMode.EXACT)
            self.assertEqual(second.children[0].segment_id, first.segment_id)
            chosen_second = second.children[0]
            third = self.service.continue_node(
                ContinueGenerationRequest(
                    **request_defaults,
                    parent_node_id=chosen_second.id,
                    parent_token=chosen_second.token,
                    assistant_prefix=chosen_second.context_after,
                    depth=chosen_second.depth,
                    cumulative_probability=chosen_second.cumulative_probability,
                    cached_segment_id=str(chosen_second.metadata.get("cached_segment_id")),
                    cached_token_index=int(chosen_second.metadata.get("next_cached_token_index")),
                )
            )
            self.assertEqual(provider_call_count, 1)
            self.assertEqual(third.continuation_mode, ContinuationMode.EXACT)
            self.assertEqual(third.children[0].token, " time")
            alternative_first = first.children[1]
            alternative_request = ContinueGenerationRequest(
                **request_defaults,
                parent_node_id=alternative_first.id,
                parent_token=alternative_first.token,
                assistant_prefix=alternative_first.context_after,
                depth=alternative_first.depth,
                cumulative_probability=alternative_first.cumulative_probability,
            )
            branch_response = self.service.continue_node(alternative_request)

        self.assertEqual(provider_call_count, 2)
        self.assertEqual(branch_response.action, "new_provider_segment")
        self.assertEqual(branch_response.continuation_mode, ContinuationMode.EXACT)
        self.assertEqual(branch_response.children[0].token, " alternative")

    def test_continue_generation_falls_back_to_approximate_and_recaches_after_exhaustion(self) -> None:
        provider_call_count = 0
        captured_inputs: list[object] = []
        self_outer = self

        class FakeResponses:
            def __init__(self) -> None:
                self._responses = [
                    self_outer._make_response(
                        self_outer._make_logprob_entry(
                            " is",
                            0.44,
                            top_alternatives=[(" would", 0.16), (" A", 0.08)],
                        ),
                        self_outer._make_logprob_entry(
                            " around",
                            0.39,
                            top_alternatives=[(" about", 0.14)],
                        ),
                        self_outer._make_logprob_entry(
                            " 54",
                            0.35,
                            top_alternatives=[(" 53", 0.12)],
                        ),
                    ),
                    self_outer._make_response(
                        self_outer._make_logprob_entry(
                            " seconds",
                            0.41,
                            top_alternatives=[(" flat", 0.15)],
                        ),
                        self_outer._make_logprob_entry(
                            " for",
                            0.34,
                            top_alternatives=[(" in", 0.11)],
                        ),
                    ),
                    self_outer._make_response(
                        self_outer._make_logprob_entry(
                            " branch",
                            0.38,
                            top_alternatives=[(" path", 0.13)],
                        ),
                    ),
                ]

            def create(self, **kwargs):
                nonlocal provider_call_count
                provider_call_count += 1
                captured_inputs.append(kwargs["input"])
                return self._responses.pop(0)

        request = ContinueGenerationRequest(
            root_prompt="Prompt",
            model="gpt-4.1-mini",
            preset="general",
            temperature=0.7,
            top_p=1.0,
            parent_node_id="node-1",
            parent_token="boy",
            assistant_prefix="A good 400m time for a 16-year-old boy",
            depth=8,
            cumulative_probability=0.5,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with (
            patch.object(
                self.service,
                "_get_client",
                return_value=SimpleNamespace(responses=FakeResponses()),
            ),
            patch.object(
                self.service,
                "_provider_capabilities_for_model",
                return_value=ProviderCapabilities(
                    supports_native_continuation=False,
                    supports_token_logprobs=True,
                    minimum_output_tokens=16,
                    supports_continuation=True,
                ),
            ),
        ):
            request_defaults = request.model_dump(
                exclude={
                    "parent_node_id",
                    "parent_token",
                    "assistant_prefix",
                    "depth",
                    "cumulative_probability",
                    "cached_segment_id",
                    "cached_token_index",
                }
            )
            first = self.service.continue_node(request)
            self.assertEqual(provider_call_count, 1)
            self.assertEqual(first.continuation_mode, ContinuationMode.APPROXIMATE)
            self.assertEqual(first.children[0].token, " is")
            self.assertEqual(first.cached_token_count, 3)
            self.assertEqual(first.remaining_cached_tokens, 2)
            self.assertEqual(first.children[0].continuation_mode, ContinuationMode.APPROXIMATE)
            self.assertEqual(first.children[0].segment_id, first.segment_id)
            _, expected_approximate_request = self.service._build_approximate_continuation_request(
                prompt="Prompt",
                assistant_prefix="A good 400m time for a 16-year-old boy",
            )
            self.assertEqual(
                captured_inputs[0],
                expected_approximate_request.input_items,
            )
            self.assertEqual(
                first.children[0].metadata.get("continuation_mode_label"),
                "Approximate",
            )
            self.assertTrue(first.children[0].metadata.get("continuation_mode_tooltip"))

            chosen_first = first.children[0]
            second = self.service.continue_node(
                ContinueGenerationRequest(
                    **request_defaults,
                    parent_node_id=chosen_first.id,
                    parent_token=chosen_first.token,
                    assistant_prefix=chosen_first.context_after,
                    depth=chosen_first.depth,
                    cumulative_probability=chosen_first.cumulative_probability,
                    cached_segment_id=str(chosen_first.metadata.get("cached_segment_id")),
                    cached_token_index=int(chosen_first.metadata.get("next_cached_token_index")),
                )
            )
            self.assertEqual(provider_call_count, 1)
            self.assertEqual(second.continuation_mode, ContinuationMode.APPROXIMATE)
            self.assertEqual(second.children[0].token, " around")

            chosen_second = second.children[0]
            third = self.service.continue_node(
                ContinueGenerationRequest(
                    **request_defaults,
                    parent_node_id=chosen_second.id,
                    parent_token=chosen_second.token,
                    assistant_prefix=chosen_second.context_after,
                    depth=chosen_second.depth,
                    cumulative_probability=chosen_second.cumulative_probability,
                    cached_segment_id=str(chosen_second.metadata.get("cached_segment_id")),
                    cached_token_index=int(chosen_second.metadata.get("next_cached_token_index")),
                )
            )
            self.assertEqual(provider_call_count, 1)
            self.assertEqual(third.children[0].token, " 54")

            chosen_third = third.children[0]
            self.assertIsNone(chosen_third.metadata.get("next_cached_token_index"))
            fourth = self.service.continue_node(
                ContinueGenerationRequest(
                    **request_defaults,
                    parent_node_id=chosen_third.id,
                    parent_token=chosen_third.token,
                    assistant_prefix=chosen_third.context_after,
                    depth=chosen_third.depth,
                    cumulative_probability=chosen_third.cumulative_probability,
                    cached_segment_id=str(chosen_third.metadata.get("cached_segment_id")),
                )
            )
            self.assertEqual(provider_call_count, 2)
            self.assertEqual(fourth.continuation_mode, ContinuationMode.APPROXIMATE)
            self.assertEqual(fourth.children[0].token, " seconds")

            alternative_first = first.children[1]
            branch_response = self.service.continue_node(
                ContinueGenerationRequest(
                    **request_defaults,
                    parent_node_id=alternative_first.id,
                    parent_token=alternative_first.token,
                    assistant_prefix=alternative_first.context_after,
                    depth=alternative_first.depth,
                    cumulative_probability=alternative_first.cumulative_probability,
                )
            )

        self.assertEqual(provider_call_count, 3)
        self.assertEqual(branch_response.continuation_mode, ContinuationMode.APPROXIMATE)
        self.assertEqual(branch_response.children[0].token, " branch")

    def test_model_catalog_surfaces_offline_ollama_provider_state(self) -> None:
        with patch.object(
            self.service._ollama_provider,
            "discover_models",
            return_value=SimpleNamespace(
                provider_name=ModelProvider.OLLAMA,
                provider_label="Ollama",
                status="offline",
                status_message=(
                    "Ollama is not running.\n\nInstall from https://ollama.com/\n\nThen run:\n\nollama serve"
                ),
                recommended_models=["qwen2.5:3b", "phi3", "gemma3", "llama3.2"],
                capabilities=ProviderCapabilities(
                    supports_native_continuation=False,
                    supports_token_logprobs=False,
                    minimum_output_tokens=1,
                    supports_entropy=False,
                    supports_attention=False,
                    supports_streaming=True,
                    supports_branching=False,
                    supports_continuation=False,
                ),
                models=[],
            ),
        ):
            catalog = self.service.list_models(force_refresh=True)

        ollama_provider = next(
            provider for provider in catalog.providers if provider.id == ModelProvider.OLLAMA
        )
        self.assertEqual(ollama_provider.status, "offline")
        self.assertIn("ollama serve", ollama_provider.status_message or "")
        self.assertIn("qwen2.5:3b", ollama_provider.recommended_models)

    def test_ollama_generation_marks_tokens_approximate_and_probabilities_unavailable(self) -> None:
        with patch.object(
            self.service,
            "_resolve_model_option",
            return_value=SimpleNamespace(provider=ModelProvider.OLLAMA),
        ), patch.object(
            self.service._ollama_provider,
            "generate",
            return_value=SimpleNamespace(
                completion="A solid time",
                prompt_tokens=6,
                completion_tokens=3,
                total_tokens=9,
                latency_ms=42,
                finish_reason="stop",
            ),
        ):
            response = self.service.build_response(
                GenerationRequest(
                    prompt="What is a good 400m time?",
                    provider=ModelProvider.OLLAMA,
                    model="phi3",
                    demo_mode=False,
                )
            )

        self.assertEqual(response.request.provider, ModelProvider.OLLAMA)
        self.assertEqual(response.stats.provider, ModelProvider.OLLAMA)
        self.assertFalse(response.provider_capabilities.supports_logprobs)
        self.assertGreater(len(response.tokens), 0)
        for token in response.tokens:
            self.assertEqual(token.continuation_mode, ContinuationMode.APPROXIMATE)
            self.assertIsNone(token.probability)
            self.assertIsNone(token.raw_probability)
            self.assertIsNone(token.log_probability)
            self.assertEqual(token.metadata.get("provider"), ModelProvider.OLLAMA.value)

    def test_build_generation_context_messages_for_hugging_face_uses_saved_system_and_raw_context(self) -> None:
        messages, raw_context_text = self.service._build_generation_context_messages(
            provider=ModelProvider.HUGGING_FACE,
            prompt="Return only OCEAN.",
            preset="general",
            intent="retrieval",
            variation=0,
            local_result=SimpleNamespace(
                raw_context_text=(
                    "<|im_start|>system\nYou are Qwen.\n<|im_end|>\n"
                    "<|im_start|>user\nReturn only OCEAN.\n<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
                system_prompt="You are Qwen.",
            ),
        )

        self.assertEqual(
            raw_context_text,
            (
                "<|im_start|>system\nYou are Qwen.\n<|im_end|>\n"
                "<|im_start|>user\nReturn only OCEAN.\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        )
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, "You are Qwen.")
        self.assertEqual(messages[0].source, "provider_default")
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[1].content, "Return only OCEAN.")

    def test_build_generation_context_messages_for_openai_uses_composed_system_instructions(self) -> None:
        messages, raw_context_text = self.service._build_generation_context_messages(
            provider=ModelProvider.OPENAI,
            prompt="Summarize this.",
            preset="general",
            intent="summary",
            variation=0,
            local_result=None,
        )

        self.assertIsNone(raw_context_text)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].source, "composed_instructions")
        self.assertIn(
            "You are answering inside a model-inspection UI.",
            messages[0].content,
        )
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[1].content, "Summarize this.")


if __name__ == "__main__":
    unittest.main()
