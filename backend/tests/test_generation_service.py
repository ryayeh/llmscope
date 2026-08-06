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


if __name__ == "__main__":
    unittest.main()
