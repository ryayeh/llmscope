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
from app.schemas.generation import AlternativeCandidate, NodeExpansionRequest, TokenTrace
from app.services.generation_service import GenerationService


class GenerationServiceCanonicalStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = GenerationService()

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

    def test_continuation_requests_respect_provider_minimum_and_keep_first_token(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[
                                SimpleNamespace(
                                    type="output_text",
                                    text=" old",
                                    logprobs=[
                                        SimpleNamespace(
                                            token=" old",
                                            logprob=math.log(0.41),
                                            bytes=list(" old".encode("utf-8")),
                                            token_id=None,
                                            tokenizer_id=None,
                                            top_logprobs=[
                                                SimpleNamespace(
                                                    token=" A",
                                                    logprob=math.log(0.19),
                                                    bytes=list(" A".encode("utf-8")),
                                                    token_id=None,
                                                    tokenizer_id=None,
                                                )
                                            ],
                                        ),
                                        SimpleNamespace(
                                            token=" athlete",
                                            logprob=math.log(0.27),
                                            bytes=list(" athlete".encode("utf-8")),
                                            token_id=None,
                                            tokenizer_id=None,
                                            top_logprobs=[
                                                SimpleNamespace(
                                                    token=" runner",
                                                    logprob=math.log(0.12),
                                                    bytes=list(" runner".encode("utf-8")),
                                                    token_id=None,
                                                    tokenizer_id=None,
                                                )
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                    usage=None,
                    status="completed",
                )

        request = NodeExpansionRequest(
            root_prompt="Prompt",
            model="gpt-4.1-mini",
            preset="general",
            temperature=0.7,
            top_p=1.0,
            parent_node_id="node-1",
            parent_token="year",
            assistant_prefix="A good 400m time for a 16 year",
            depth=7,
            cumulative_probability=0.5,
            variation=0,
            max_children=4,
            demo_mode=False,
        )

        with patch.object(self.service, "_get_client", return_value=SimpleNamespace(responses=FakeResponses())):
            steps, _, _ = self.service._request_live_steps(
                request=request,
                prompt="Prompt",
                preset="general",
                intent="benchmark",
                assistant_prefix=request.assistant_prefix,
                branch_id=request.parent_node_id,
                parent_node_id=request.parent_node_id,
                max_output_tokens=1,
                top_logprobs=4,
            )
            response = self.service.expand_node(request)

        self.assertEqual(captured["max_output_tokens"], 16)
        self.assertIn(
            "Continue the existing assistant response from the provided prefix exactly where it stops.",
            str(captured["instructions"]),
        )
        self.assertEqual(steps[0].token, " old")
        self.assertEqual(steps[1].token, " athlete")
        self.assertEqual(steps[0].context_before, "A good 400m time for a 16 year")
        self.assertEqual(steps[0].context_after, "A good 400m time for a 16 year old")

        self.assertEqual(response.children[0].token, " old")
        self.assertTrue(all(child.token != " athlete" for child in response.children))


if __name__ == "__main__":
    unittest.main()
