from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.providers.huggingface_provider import HuggingFaceLocalProvider


def build_provider() -> HuggingFaceLocalProvider:
    settings = get_settings()
    return HuggingFaceLocalProvider(
        default_model=settings.hugging_face_default_model,
        model_revisions={
            "Qwen/Qwen2.5-1.5B-Instruct": settings.hugging_face_qwen_1_5b_revision,
            "Qwen/Qwen2.5-3B-Instruct": settings.hugging_face_qwen_3b_revision,
        },
        hf_token=settings.hf_token,
        context_limit=settings.hugging_face_context_limit,
        default_output_tokens=settings.hugging_face_default_output_tokens,
        max_output_tokens=settings.hugging_face_max_output_tokens,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Load a supported Hugging Face Local model and run one LLMScope smoke test.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Supported model id to load.",
    )
    parser.add_argument(
        "--prompt",
        default="What is a good 400m time for a 16-year-old?",
        help="Prompt to generate from.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling cutoff.",
    )
    args = parser.parse_args()

    provider = build_provider()
    status = provider.load_model(args.model)
    print(f"Loaded: {status.active_model_label} on {status.device} ({status.precision})")
    if status.active_model_resolved_revision:
        print(f"Resolved revision: {status.active_model_resolved_revision}")

    result = provider.generate(
        model=args.model,
        prompt=args.prompt,
        assistant_prefix="",
        branch_id="main",
        parent_node_id="root",
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_candidates=6,
    )

    print("\nCompletion:\n")
    print(result.completion)
    print("\nTokens:\n")
    for trace in result.tokens:
        print(
            f"{trace.generation_step:03d} id={trace.token_id} raw={trace.token!r} "
            f"decoded={trace.decoded_contribution!r} p={trace.probability:.4f} "
            f"entropy={trace.entropy:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
