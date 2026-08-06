from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.generation import ContinueGenerationRequest, GenerationRequest, TokenTrace
from app.services.generation_service import (
    DEFAULT_TOP_LOGPROBS,
    DEFAULT_APPROXIMATE_CONTINUATION_TEMPLATE_ID,
    GenerationService,
)

PROMPT_SUITE = [
    "What is a good 400m time for a 16-year-old?",
    "Explain closures in JavaScript in one short paragraph.",
    "Write a short email asking to reschedule tomorrow's meeting.",
    "Give me a three-step plan to improve my 5k time.",
    "Write a Python function that checks whether a string is a palindrome.",
]


@dataclass(frozen=True)
class DistributionSnapshot:
    chosen_token: str
    entropy: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    prompt: str
    prefix: str
    generation_step: int
    exact: DistributionSnapshot


@dataclass(frozen=True)
class TemplateMeasurement:
    template_id: str
    label: str
    case_id: str
    prompt: str
    generation_step: int
    exact_token: str
    approximate_token: str
    top1_match: bool
    exact_token_probability: float
    approximate_probability_on_exact_token: float
    approximate_entropy: float
    exact_entropy: float
    entropy_delta_abs: float
    js_divergence: float


def snapshot_from_trace(trace: TokenTrace) -> DistributionSnapshot:
    probabilities: dict[str, float] = {trace.token: trace.raw_probability}
    for alternative in trace.alternatives:
        probability = alternative.raw_probability or alternative.probability
        probabilities.setdefault(alternative.token, probability)

    return DistributionSnapshot(
        chosen_token=trace.token,
        entropy=trace.entropy,
        probabilities=probabilities,
    )


def aligned_probability_vectors(
    left: DistributionSnapshot,
    right: DistributionSnapshot,
) -> tuple[list[str], list[float], list[float]]:
    tokens = sorted(set(left.probabilities) | set(right.probabilities))
    left_values = [left.probabilities.get(token, 0.0) for token in tokens]
    right_values = [right.probabilities.get(token, 0.0) for token in tokens]
    left_other = max(0.0, 1.0 - sum(left_values))
    right_other = max(0.0, 1.0 - sum(right_values))
    return [*tokens, "__OTHER__"], [*left_values, left_other], [*right_values, right_other]


def kl_divergence(left: list[float], right: list[float], epsilon: float = 1e-12) -> float:
    total = 0.0
    for left_value, right_value in zip(left, right, strict=False):
        if left_value <= 0:
            continue
        total += left_value * math.log(left_value / max(right_value, epsilon), 2)
    return total


def js_divergence(left: list[float], right: list[float]) -> float:
    midpoint = [(left_value + right_value) / 2 for left_value, right_value in zip(left, right, strict=False)]
    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)


def build_exact_cases(
    service: GenerationService,
    *,
    model: str,
    max_prefix_steps: int,
    temperature: float,
    top_p: float,
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []

    for prompt_index, prompt in enumerate(PROMPT_SUITE, start=1):
        response = service.build_response(
            GenerationRequest(
                prompt=prompt,
                model=model,
                preset="general",
                max_tokens=16,
                temperature=temperature,
                top_p=top_p,
                demo_mode=False,
            )
        )
        non_empty_prefix_traces = [trace for trace in response.tokens if trace.context_before]
        for trace in non_empty_prefix_traces[:max_prefix_steps]:
            cases.append(
                EvaluationCase(
                    case_id=f"prompt-{prompt_index}-step-{trace.generation_step}",
                    prompt=prompt,
                    prefix=trace.context_before,
                    generation_step=trace.generation_step,
                    exact=snapshot_from_trace(trace),
                )
            )

    return cases


def benchmark_template(
    service: GenerationService,
    *,
    case: EvaluationCase,
    model: str,
    template_id: str,
    temperature: float,
    top_p: float,
) -> TemplateMeasurement:
    request = ContinueGenerationRequest(
        root_prompt=case.prompt,
        model=model,
        preset="general",
        temperature=temperature,
        top_p=top_p,
        parent_node_id=f"benchmark:{case.case_id}",
        parent_token=case.exact.chosen_token,
        assistant_prefix=case.prefix,
        depth=case.generation_step,
        cumulative_probability=1.0,
        variation=0,
        max_children=DEFAULT_TOP_LOGPROBS,
        demo_mode=False,
    )
    steps, _, _ = service._request_approximate_steps(
        request=request,
        prompt=case.prompt,
        assistant_prefix=case.prefix,
        branch_id=request.parent_node_id,
        parent_node_id=request.parent_node_id,
        max_output_tokens=1,
        top_logprobs=DEFAULT_TOP_LOGPROBS,
        template_id=template_id,
    )

    if not steps:
        raise RuntimeError(f"No approximate continuation tokens returned for {case.case_id}.")

    approximate = snapshot_from_trace(steps[0])
    _, exact_vector, approximate_vector = aligned_probability_vectors(case.exact, approximate)
    template_label = dict(service.list_approximate_prompt_templates())[template_id]
    exact_token_probability = case.exact.probabilities.get(case.exact.chosen_token, 0.0)
    approximate_probability_on_exact_token = approximate.probabilities.get(case.exact.chosen_token, 0.0)

    return TemplateMeasurement(
        template_id=template_id,
        label=template_label,
        case_id=case.case_id,
        prompt=case.prompt,
        generation_step=case.generation_step,
        exact_token=case.exact.chosen_token,
        approximate_token=approximate.chosen_token,
        top1_match=approximate.chosen_token == case.exact.chosen_token,
        exact_token_probability=exact_token_probability,
        approximate_probability_on_exact_token=approximate_probability_on_exact_token,
        approximate_entropy=approximate.entropy,
        exact_entropy=case.exact.entropy,
        entropy_delta_abs=abs(approximate.entropy - case.exact.entropy),
        js_divergence=js_divergence(exact_vector, approximate_vector),
    )


def summarize_measurements(measurements: list[TemplateMeasurement]) -> list[dict[str, object]]:
    grouped: dict[str, list[TemplateMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.template_id].append(measurement)

    summaries: list[dict[str, object]] = []
    for template_id, template_measurements in grouped.items():
        summaries.append(
            {
                "template_id": template_id,
                "label": template_measurements[0].label,
                "cases": len(template_measurements),
                "top1_match_rate": mean(
                    1.0 if measurement.top1_match else 0.0
                    for measurement in template_measurements
                ),
                "mean_probability_on_exact_token": mean(
                    measurement.approximate_probability_on_exact_token
                    for measurement in template_measurements
                ),
                "mean_js_divergence": mean(
                    measurement.js_divergence for measurement in template_measurements
                ),
                "mean_absolute_entropy_delta": mean(
                    measurement.entropy_delta_abs for measurement in template_measurements
                ),
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            float(item["mean_js_divergence"]),
            -float(item["top1_match_rate"]),
            -float(item["mean_probability_on_exact_token"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark approximate continuation prompt templates.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-prefix-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--template", action="append", dest="templates")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = GenerationService()
    available_templates = service.list_approximate_prompt_templates()
    template_ids = args.templates or [template_id for template_id, _ in available_templates]

    print(
        f"Benchmarking {len(template_ids)} approximate templates on {len(PROMPT_SUITE)} prompts "
        f"with up to {args.max_prefix_steps} non-empty prefixes each."
    )
    print(f"Current default template: {DEFAULT_APPROXIMATE_CONTINUATION_TEMPLATE_ID}")

    cases = build_exact_cases(
        service,
        model=args.model,
        max_prefix_steps=args.max_prefix_steps,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"Collected {len(cases)} exact benchmark cases.")

    measurements: list[TemplateMeasurement] = []
    total_runs = len(template_ids) * len(cases)
    current_run = 0
    for template_id in template_ids:
        template_label = dict(available_templates)[template_id]
        print(f"\n[{template_id}] {template_label}")
        for case in cases:
            current_run += 1
            print(
                f"  {current_run:>3}/{total_runs} | {case.case_id} | prefix chars={len(case.prefix)}",
                flush=True,
            )
            measurements.append(
                benchmark_template(
                    service,
                    case=case,
                    model=args.model,
                    template_id=template_id,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            )

    summaries = summarize_measurements(measurements)

    if args.as_json:
        print(
            json.dumps(
                {
                    "default_template": DEFAULT_APPROXIMATE_CONTINUATION_TEMPLATE_ID,
                    "summaries": summaries,
                    "measurements": [asdict(measurement) for measurement in measurements],
                },
                indent=2,
            )
        )
        return 0

    print("\nTemplate Summary")
    print(
        "template_id".ljust(40)
        + "match".rjust(8)
        + "  p(exact)".rjust(12)
        + "  JS".rjust(10)
        + "  |dH|".rjust(10)
    )
    for summary in summaries:
        print(
            str(summary["template_id"]).ljust(40)
            + f"{float(summary['top1_match_rate']) * 100:7.1f}%"
            + f"{float(summary['mean_probability_on_exact_token']) * 100:11.1f}%"
            + f"{float(summary['mean_js_divergence']):10.4f}"
            + f"{float(summary['mean_absolute_entropy_delta']):10.4f}"
        )

    if summaries:
        best = summaries[0]
        print(
            "\nBest template: "
            f"{best['template_id']} ({best['label']}) | "
            f"match={float(best['top1_match_rate']) * 100:.1f}% | "
            f"p(exact)={float(best['mean_probability_on_exact_token']) * 100:.1f}% | "
            f"JS={float(best['mean_js_divergence']):.4f} | "
            f"|dH|={float(best['mean_absolute_entropy_delta']):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
