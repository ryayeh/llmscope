from __future__ import annotations

from datetime import datetime, timezone
import math
from time import perf_counter
from typing import Any
import re
import zlib

from openai import OpenAI

from app.core.config import get_settings
from app.models.provider import ModelProvider
from app.schemas.generation import (
    AlternativeCandidate,
    GenerationRequest,
    GenerationResponse,
    GenerationStats,
    NodeExpansionCandidate,
    NodeExpansionRequest,
    NodeExpansionResponse,
    PromptInsights,
    RequestEcho,
    TokenTrace,
    TokenTreeNode,
    TreeSummary,
)
from app.schemas.model_catalog import ModelCatalogResponse, ModelOption, PresetOption

TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")

IMPORTANT_SHORT_WORDS = {"ai", "api", "app", "bug", "db", "ml", "sdk", "ui", "ux"}

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
    "brief",
    "briefly",
    "compare",
    "create",
    "debug",
    "english",
    "explain",
}

ALTERNATIVE_MAP: dict[str, list[tuple[str, str]]] = {
    "benchmark": [
        ("range", "Frames the answer as a band instead of one number."),
        ("mark", "Uses a performance-oriented term."),
        ("target", "Leans toward a goal-setting framing."),
    ],
    "difference": [
        ("contrast", "A sharper comparison word."),
        ("distinction", "A more formal comparison term."),
        ("gap", "A shorter, more direct contrast term."),
    ],
    "debug": [
        ("trace", "Shifts toward a step-by-step investigation."),
        ("inspect", "Focuses on looking closely at the failure."),
        ("diagnose", "Frames the issue as something to isolate."),
    ],
    "explain": [
        ("describe", "A broader framing for the same idea."),
        ("clarify", "Leans toward removing ambiguity."),
        ("outline", "Suggests a structured walkthrough."),
    ],
    "fast": [
        ("quick", "Keeps the performance framing direct."),
        ("strong", "Frames the mark as competitive."),
        ("sharp", "Suggests better execution."),
    ],
    "good": [
        ("solid", "Keeps the answer practical."),
        ("strong", "Raises the performance bar slightly."),
        ("competitive", "Frames it relative to other athletes."),
    ],
    "latency": [
        ("speed", "A simpler performance term."),
        ("timing", "Focuses on elapsed execution behavior."),
        ("delay", "Emphasizes waiting time."),
    ],
    "prompt": [
        ("request", "A more general input term."),
        ("instruction", "Frames the input as a directive."),
        ("message", "Uses a chat-oriented label."),
    ],
    "quality": [
        ("accuracy", "Focuses on correctness."),
        ("consistency", "Focuses on repeatable behavior."),
        ("fidelity", "Focuses on staying close to the goal."),
    ],
    "time": [
        ("mark", "Uses a performance label instead of a clock word."),
        ("range", "Suggests a realistic band."),
        ("benchmark", "Frames it as a comparison point."),
    ],
    "token": [
        ("candidate", "Focuses on the decoding choice."),
        ("piece", "Uses a looser text fragment term."),
        ("symbol", "Moves toward a lower-level representation."),
    ],
}

INTENT_FALLBACKS: dict[str, list[tuple[str, str]]] = {
    "comparison": [
        ("trade-off", "Emphasizes the decision boundary."),
        ("control", "Highlights predictability."),
        ("flexibility", "Highlights adaptability."),
        ("baseline", "Keeps the branch grounded in a default."),
    ],
    "debugging": [
        ("trace", "Points at the data to inspect."),
        ("failure", "Keeps the problem concrete."),
        ("guardrail", "Suggests a protective follow-up."),
        ("test", "Points toward verification."),
    ],
    "planning": [
        ("baseline", "Starts from the simplest viable path."),
        ("measure", "Pushes toward evidence before tuning."),
        ("iterate", "Keeps the process incremental."),
        ("checkpoint", "Adds a clear control point."),
    ],
    "enumeration": [
        ("option", "Keeps the branch list-oriented."),
        ("path", "Frames the choice as a route."),
        ("default", "Suggests the safest first choice."),
        ("variant", "Keeps the branch practical."),
    ],
    "summary": [
        ("core", "Pulls toward the main idea."),
        ("signal", "Emphasizes what matters most."),
        ("theme", "Frames the response more broadly."),
        ("focus", "Keeps attention on the center of the prompt."),
    ],
    "explanation": [
        ("context", "Adds a clearer setup."),
        ("example", "Makes the point more concrete."),
        ("result", "Focuses on the practical outcome."),
        ("reason", "Keeps the answer tied to cause and effect."),
    ],
}

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


class GenerationService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: OpenAI | None = None

    def list_models(self) -> ModelCatalogResponse:
        return ModelCatalogResponse(
            default_model="gpt-4.1-mini",
            default_preset="general",
            models=MODEL_OPTIONS,
            presets=PRESET_OPTIONS,
        )

    def build_response(self, request: GenerationRequest) -> GenerationResponse:
        prompt = request.prompt.strip()
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

        completion, response_mode, usage, latency_ms = self._generate_completion(
            request=request,
            prompt=prompt,
            keywords=keywords,
            intent=intent,
            preset=preset,
        )
        completion = self._truncate_completion(completion, request.max_tokens)

        tokens = self._build_token_traces(
            prompt=prompt,
            completion=completion,
            keywords=keywords,
            intent=intent,
            temperature=request.temperature,
        )
        tree, tree_summary = self._build_tree(tokens)

        prompt_tokens = usage.input_tokens if usage else self._estimate_prompt_tokens(prompt, keywords)
        completion_tokens = len(tokens)
        total_tokens = prompt_tokens + completion_tokens
        effective_latency_ms = latency_ms or (
            sum(token.latency_ms for token in tokens) + 150 + (len(keywords) * 10)
        )

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
            response_strategy=PRESET_INSTRUCTIONS[preset] + f" {strategy}",
            suggested_follow_ups=self._build_follow_ups(keywords, intent),
        )

        notes = (
            "Live response with an inferred token trace."
            if response_mode == "live"
            else "Local fallback response with an inferred token trace."
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
                variation=request.variation,
            ),
            insights=insights,
            tokens=tokens,
            tree=tree,
            tree_summary=tree_summary,
            stats=stats,
        )

    def expand_node(self, request: NodeExpansionRequest) -> NodeExpansionResponse:
        parent_preview = request.parent_text_preview.strip()
        context_keywords = self._extract_keywords(
            " ".join(part for part in (request.prompt, parent_preview) if part),
            limit=8,
        )
        intent, _ = self._detect_intent(request.prompt.lower())
        source_trace: TokenTrace | None = None
        response_mode = "fallback"
        response_notes = (
            "Next-token distribution inferred locally from the prompt and current branch context."
        )

        try:
            generation = self.build_response(
                GenerationRequest(
                    prompt=request.prompt,
                    model=request.model,
                    preset=request.preset,
                    max_tokens=max(request.depth + request.max_children + 12, 48),
                    temperature=request.temperature,
                    variation=request.variation,
                )
            )

            if request.depth < len(generation.tokens):
                source_trace = generation.tokens[request.depth]
                response_mode = generation.mode or "inferred"
                response_notes = (
                    "Next-token distribution derived from the current generation path."
                )
        except Exception:
            source_trace = None

        if source_trace is not None:
            candidate_specs = self._candidate_specs_from_source_trace(source_trace)
        else:
            candidate_specs = self._candidate_specs_from_context(
                prompt=request.prompt,
                parent_preview=parent_preview,
                parent_token=request.parent_token,
                context_keywords=context_keywords,
                intent=intent,
                depth=request.depth,
            )

        deduped_specs: list[tuple[str, float, int, str | None]] = []
        seen_tokens: set[str] = set()

        for token, probability, latency_ms, rationale in candidate_specs:
            normalized = token.lower()
            if normalized in seen_tokens:
                continue
            seen_tokens.add(normalized)
            deduped_specs.append((token, probability, latency_ms, rationale))

            if len(deduped_specs) == request.max_children:
                break

        if not deduped_specs:
            fallback_token = request.parent_token.strip() or (
                context_keywords[0] if context_keywords else "next"
            )
            deduped_specs = [
                (
                    fallback_token,
                    1.0,
                    36 + (request.depth * 3),
                    "Fallback continuation inferred from the available branch context.",
                )
            ]

        normalized_probabilities = self._normalize_probabilities(
            [probability for _, probability, _, _ in deduped_specs]
        )
        entropy = self._calculate_entropy(normalized_probabilities)
        children: list[NodeExpansionCandidate] = []

        for index, ((token, _, latency_ms, rationale), probability) in enumerate(
            zip(deduped_specs, normalized_probabilities, strict=False),
            start=1,
        ):
            text_preview = self._join_tokens(
                [*self._tokenize(parent_preview), token]
                if parent_preview
                else [token]
            )
            cumulative_probability = round(
                self._clamp(request.cumulative_probability * probability, 0.000001, 1.0),
                6,
            )
            token_id = self._make_token_id(
                token=token,
                depth=request.depth + 1,
                rank=index,
                preview=text_preview,
            )

            children.append(
                NodeExpansionCandidate(
                    id=f"{request.parent_node_id}::{request.depth + 1}:{index}:{token_id}",
                    parent_node_id=request.parent_node_id,
                    token=token,
                    token_id=token_id,
                    probability=probability,
                    log_probability=self._safe_log(probability),
                    entropy=entropy,
                    cumulative_probability=cumulative_probability,
                    latency_ms=latency_ms,
                    depth=request.depth + 1,
                    rank=index,
                    text_preview=text_preview,
                    rationale=rationale,
                )
            )

        return NodeExpansionResponse(
            mode=response_mode,
            parent_node_id=request.parent_node_id,
            children=children,
            entropy=entropy,
            expanded_at=datetime.now(timezone.utc),
            notes=response_notes,
        )

    def _candidate_specs_from_source_trace(
        self,
        source_trace: TokenTrace,
    ) -> list[tuple[str, float, int, str | None]]:
        candidate_specs: list[tuple[str, float, int, str | None]] = [
            (
                source_trace.token,
                source_trace.probability,
                source_trace.latency_ms,
                "Main continuation from the current generation path.",
            )
        ]
        candidate_specs.extend(
            (
                candidate.token,
                candidate.probability,
                candidate.latency_ms or max(source_trace.latency_ms - 6, 18),
                candidate.rationale,
            )
            for candidate in source_trace.alternatives
        )
        return candidate_specs

    def _candidate_specs_from_context(
        self,
        *,
        prompt: str,
        parent_preview: str,
        parent_token: str,
        context_keywords: list[str],
        intent: str,
        depth: int,
    ) -> list[tuple[str, float, int, str | None]]:
        fallback_token = parent_token.strip() or (
            context_keywords[0] if context_keywords else "next"
        )
        base_probability = 0.62
        candidate_specs: list[tuple[str, float, int, str | None]] = [
            (
                fallback_token,
                base_probability,
                36 + (depth * 3),
                "Fallback continuation inferred from the current prompt context.",
            )
        ]
        candidate_specs.extend(
            (
                candidate.token,
                candidate.probability,
                candidate.latency_ms or 34 + (rank * 4),
                candidate.rationale,
            )
            for rank, candidate in enumerate(
                self._build_alternatives(
                    token=fallback_token,
                    actual_probability=base_probability,
                    keywords=context_keywords,
                    intent=intent,
                    position=depth,
                    context_candidates=self._build_context_candidates(
                        prompt=prompt,
                        completion=parent_preview or fallback_token,
                        keywords=context_keywords,
                    ),
                ),
                start=1,
            )
        )
        return candidate_specs

    def _generate_completion(
        self,
        *,
        request: GenerationRequest,
        prompt: str,
        keywords: list[str],
        intent: str,
        preset: str,
    ) -> tuple[str, str, Any | None, int]:
        client = self._get_client()

        if client and request.model in MODEL_OPTION_MAP:
            try:
                instructions = self._build_live_instructions(
                    preset=preset,
                    intent=intent,
                    variation=request.variation,
                )
                start_time = perf_counter()
                response = client.responses.create(
                    model=request.model,
                    input=prompt,
                    instructions=instructions,
                    max_output_tokens=max(16, min(request.max_tokens, 1024)),
                    temperature=request.temperature,
                )
                elapsed_ms = max(1, int((perf_counter() - start_time) * 1000))
                completion = response.output_text.strip()

                if completion:
                    return completion, "live", response.usage, elapsed_ms
            except Exception:
                pass

        completion = self._build_fallback_completion(
            prompt=prompt,
            keywords=keywords,
            intent=intent,
            max_tokens=request.max_tokens,
            preset=preset,
            variation=request.variation,
        )
        return completion, "fallback", None, 0

    def _build_live_instructions(self, *, preset: str, intent: str, variation: int) -> str:
        style_hint = RESPONSE_STYLE_HINTS[variation % len(RESPONSE_STYLE_HINTS)]

        return " ".join(
            [
                "You are answering inside a minimal prompt-testing UI.",
                "Be accurate, direct, and useful.",
                "Avoid filler, vague generic language, and meta commentary.",
                "If the user asks for a benchmark, range, definition, or recommendation, answer that concrete question first.",
                PRESET_INSTRUCTIONS[preset],
                style_hint,
                f"Detected intent: {intent}.",
            ]
        )

    def _build_fallback_completion(
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
                    "For a 16-year-old in the 400m, a solid time is often around 55 to 60 seconds, "
                    "strong high-school competition is usually closer to 50 to 54, and breaking 50 "
                    "is elite in many settings."
                ),
                (
                    "A good 400m time for a 16-year-old is usually a range, not one number: roughly "
                    "55 to 60 seconds is solid, low-50s is strong, and sub-50 is outstanding for most "
                    "high-school runners."
                ),
                (
                    "If you mean competitive high-school track, many 16-year-olds would see about 55 "
                    "to 60 seconds as a good 400m mark, around 50 to 54 as very strong, and anything "
                    "under 50 as elite."
                ),
            ]
            closing = (
                " The right benchmark still depends a lot on sex, training age, and whether the goal is "
                "general fitness or serious competition."
            )
            completion = options[variation % len(options)] + closing
            return self._truncate_completion(completion, max_tokens)

        topic = self._topic_phrase(prompt, keywords)
        first_focus = keywords[0] if keywords else "the main constraint"
        second_focus = keywords[1] if len(keywords) > 1 else "the outcome"

        if intent == "comparison":
            first_sentence = (
                f"For {topic}, the real difference is which option gives you more control versus more "
                f"speed, and the better default depends on how much {first_focus} matters."
            )
            second_sentence = (
                f"If you are choosing between them, compare latency, reliability, and how each one "
                f"affects {second_focus}."
            )
        elif intent == "debugging":
            first_sentence = (
                f"To debug {topic}, reproduce the smallest failing case first and trace the exact step "
                f"where {first_focus} stops matching the expectation."
            )
            second_sentence = (
                "Once you can see the break clearly, add a focused check or test so the same issue is "
                "easy to catch next time."
            )
        elif intent == "planning":
            first_sentence = (
                f"For {topic}, start with the thinnest version that works, measure {first_focus}, and "
                "add complexity only where it clearly changes the result."
            )
            second_sentence = (
                f"That keeps the plan grounded while still giving you room to improve {second_focus}."
            )
        else:
            opening_variants = [
                f"For {topic}, the best answer depends on the benchmark, context, and what counts as success around {first_focus}.",
                f"The clearest way to think about {topic} is to anchor on a realistic range, then adjust for {first_focus} and context.",
                f"A solid answer for {topic} starts with the direct benchmark, then adds the context that changes {first_focus}.",
            ]
            first_sentence = opening_variants[variation % len(opening_variants)]
            if preset == "coach":
                second_sentence = (
                    f"Start with a simple baseline, then tighten the target as you get more data about {second_focus}."
                )
            elif preset == "coding":
                second_sentence = (
                    f"If this is for implementation, keep the default explicit and make the parts that affect {second_focus} easy to inspect."
                )
            else:
                second_sentence = (
                    f"That keeps the answer practical while still leaving room to refine {second_focus}."
                )

        single_sentence = any(
            marker in prompt_lower
            for marker in ("one sentence", "single sentence", "briefly", "short answer")
        )
        completion = first_sentence if single_sentence else f"{first_sentence} {second_sentence}"
        return self._truncate_completion(completion, max_tokens)

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

    def _build_token_traces(
        self,
        *,
        prompt: str,
        completion: str,
        keywords: list[str],
        intent: str,
        temperature: float,
    ) -> list[TokenTrace]:
        raw_tokens = self._tokenize(completion)
        context_candidates = self._build_context_candidates(
            prompt=prompt,
            completion=completion,
            keywords=keywords,
        )
        traces: list[TokenTrace] = []
        cumulative_probability = 1.0

        for position, token in enumerate(raw_tokens):
            probability = self._score_token(
                token=token,
                position=position,
                total_tokens=len(raw_tokens),
                temperature=temperature,
            )
            cumulative_probability = round(
                self._clamp(cumulative_probability * probability, 0.000001, 1.0),
                6,
            )
            latency_ms = 26 + (len(token) * 4) + (position * 3) + int(temperature * 5)
            alternatives = self._build_alternatives(
                token=token,
                actual_probability=probability,
                keywords=keywords,
                intent=intent,
                position=position,
                context_candidates=context_candidates,
            )
            entropy = self._calculate_entropy(
                self._normalize_probabilities(
                    [probability, *[candidate.probability for candidate in alternatives]]
                )
            )
            text_preview = self._join_tokens(raw_tokens[: position + 1])
            token_id = self._make_token_id(
                token=token,
                depth=position + 1,
                rank=1,
                preview=text_preview,
            )

            traces.append(
                TokenTrace(
                    id=f"token-{position + 1}",
                    token=token,
                    token_id=token_id,
                    probability=probability,
                    log_probability=self._safe_log(probability),
                    entropy=entropy,
                    cumulative_probability=cumulative_probability,
                    latency_ms=latency_ms,
                    position=position,
                    text_preview=text_preview,
                    alternatives=alternatives,
                )
            )

        return traces

    def _build_context_candidates(
        self,
        *,
        prompt: str,
        completion: str,
        keywords: list[str],
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        prompt_focus_templates = [
            "Returns to the prompt's focus on {word}.",
            "Keeps the branch centered on {word}.",
            "Re-anchors the choice around {word}.",
        ]
        answer_context_templates = [
            "Keeps the answer anchored on {word}.",
            "Stays aligned with the answer's emphasis on {word}.",
            "Preserves the wording direction around {word}.",
        ]
        prompt_wording_templates = [
            "Reuses the prompt wording around {word}.",
            "Echoes the original phrasing built around {word}.",
            "Pulls the branch back toward the prompt's wording on {word}.",
        ]

        for index, keyword in enumerate(keywords):
            normalized = keyword.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            template = prompt_focus_templates[index % len(prompt_focus_templates)]
            candidates.append((keyword, template.format(word=keyword)))

        for index, word in enumerate(self._extract_keywords(completion, limit=8)):
            normalized = word.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            template = answer_context_templates[index % len(answer_context_templates)]
            candidates.append((word, template.format(word=word)))

        first_prompt_words = self._extract_keywords(prompt, limit=10)
        for index, word in enumerate(first_prompt_words):
            normalized = word.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            template = prompt_wording_templates[index % len(prompt_wording_templates)]
            candidates.append((word, template.format(word=word)))

        return candidates

    def _build_alternatives(
        self,
        *,
        token: str,
        actual_probability: float,
        keywords: list[str],
        intent: str,
        position: int,
        context_candidates: list[tuple[str, str]],
    ) -> list[AlternativeCandidate]:
        lower_token = token.lower()
        candidate_pool: list[tuple[str, str]] = []

        if re.fullmatch(r"[.,!?;:]", token):
            candidate_pool.extend(
                [
                    (";", "Keeps the sentence open for another clause."),
                    (",", "Softens the pause without ending the thought."),
                    (".", "Ends the sentence earlier."),
                ]
            )
        else:
            candidate_pool.extend(ALTERNATIVE_MAP.get(lower_token, []))
            candidate_pool.extend(context_candidates)
            candidate_pool.extend(INTENT_FALLBACKS[intent])
            if position == 0:
                candidate_pool.extend(
                    [
                        ("For", "Starts with a direct setup."),
                        ("A", "Starts from a more general opening."),
                        ("In", "Starts with a contextual frame."),
                    ]
                )

        alternatives: list[AlternativeCandidate] = []
        seen_tokens = {lower_token}
        base_probability = actual_probability - 0.12

        for index, (candidate_token, rationale) in enumerate(candidate_pool):
            normalized = candidate_token.lower()
            if normalized in seen_tokens:
                continue

            seen_tokens.add(normalized)
            formatted_token = self._match_case(token, candidate_token)
            probability = round(
                self._clamp(
                    base_probability - (index * 0.04),
                    0.04,
                    max(actual_probability - 0.02, 0.05),
                ),
                4,
            )
            alternatives.append(
                AlternativeCandidate(
                    token=formatted_token,
                    probability=probability,
                    log_probability=self._safe_log(probability),
                    latency_ms=max(18, 22 + (len(formatted_token) * 3) + (index * 5)),
                    token_id=self._make_token_id(
                        token=formatted_token,
                        depth=position + 1,
                        rank=index + 2,
                        preview=formatted_token,
                    ),
                    text_preview=formatted_token,
                    rationale=rationale,
                )
            )

            if len(alternatives) == 3:
                break

        entropy = self._calculate_entropy(
            self._normalize_probabilities(
                [actual_probability, *[candidate.probability for candidate in alternatives]]
            )
        )

        for candidate in alternatives:
            candidate.entropy = entropy

        return alternatives

    def _build_tree(self, tokens: list[TokenTrace]) -> tuple[TokenTreeNode, TreeSummary]:
        max_depth = min(6, len(tokens))
        branch_width = 1 + min(
            3,
            max((len(token.alternatives) for token in tokens[:max_depth]), default=0),
        )
        root = TokenTreeNode(
            id="root",
            token="<start>",
            token_id=self._make_token_id(token="<start>", depth=0, rank=1, preview=""),
            probability=1.0,
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
                prefix_tokens=[],
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
        prefix_tokens: list[str],
        selected_path: bool,
        node_prefix: str,
        max_depth: int,
        branch_width: int,
    ) -> list[TokenTreeNode]:
        if position >= max_depth:
            return []

        source = tokens[position]
        candidates: list[tuple[str, float, int, bool]] = [
            (source.token, source.probability, source.latency_ms, True),
        ]
        candidates.extend(
            (
                alternative.token,
                alternative.probability,
                max(source.latency_ms - 5, 18) + (rank * 4),
                False,
            )
            for rank, alternative in enumerate(source.alternatives[: branch_width - 1], start=1)
        )

        children: list[TokenTreeNode] = []

        for rank, (token, probability, latency_ms, is_main_branch) in enumerate(
            candidates,
            start=1,
        ):
            preview_tokens = [*prefix_tokens, token]
            node_id = f"{node_prefix}.{position + 1}.{rank}"
            cumulative_probability = round(
                self._clamp(parent_cumulative * probability, 0.000001, 1.0),
                6,
            )
            on_selected_path = selected_path and is_main_branch

            children.append(
                TokenTreeNode(
                    id=node_id,
                    token=token,
                    token_id=self._make_token_id(
                        token=token,
                        depth=position + 1,
                        rank=rank,
                        preview=self._join_tokens(preview_tokens),
                    ),
                    probability=probability,
                    log_probability=self._safe_log(probability),
                    entropy=source.entropy,
                    cumulative_probability=cumulative_probability,
                    latency_ms=latency_ms,
                    depth=position + 1,
                    rank=rank,
                    text_preview=self._join_tokens(preview_tokens),
                    is_selected_path=on_selected_path,
                    children=self._build_tree_children(
                        tokens=tokens,
                        position=position + 1,
                        parent_cumulative=cumulative_probability,
                        prefix_tokens=preview_tokens,
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
        return min(max(len(self._tokenize(prompt)) + len(keywords), 14), 256)

    def _estimate_cost(self, model: str, *, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING_USD_PER_1K.get(
            model,
            {"input": 0.0004, "output": 0.0016},
        )
        estimated = (
            (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])
        ) / 1000
        return round(estimated, 6)

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

    def _safe_log(self, probability: float) -> float:
        return round(math.log(max(probability, 0.000001)), 6)

    def _make_token_id(self, *, token: str, depth: int, rank: int, preview: str) -> int:
        payload = f"{token}|{depth}|{rank}|{preview}".encode("utf-8")
        return zlib.adler32(payload) & 0xFFFFFFFF

    def _looks_like_track_benchmark(self, prompt_lower: str) -> bool:
        return (
            ("good time" in prompt_lower or "fast time" in prompt_lower)
            and any(event in prompt_lower for event in ("100m", "200m", "400m", "800m", "1600", "mile", "5k"))
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

    def _score_token(
        self,
        *,
        token: str,
        position: int,
        total_tokens: int,
        temperature: float,
    ) -> float:
        base_probability = 0.9
        base_probability -= position * 0.016
        base_probability -= max(temperature - 0.7, 0) * 0.08
        base_probability += min(len(token), 10) * 0.006

        if token[:1].isupper():
            base_probability += 0.03
        if re.fullmatch(r"[.,!?;:]", token):
            base_probability += 0.05
        if position == total_tokens - 1:
            base_probability += 0.02

        return round(self._clamp(base_probability, 0.42, 0.97), 4)

    def _truncate_completion(self, completion: str, max_tokens: int) -> str:
        completion_tokens = self._tokenize(completion)

        if len(completion_tokens) <= max_tokens:
            return completion.strip()

        return self._join_tokens(completion_tokens[:max_tokens]).rstrip(",;:-") + "."

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    def _join_tokens(self, tokens: list[str]) -> str:
        text = ""

        for token in tokens:
            if not text:
                text = token
                continue

            if re.fullmatch(r"[.,!?;:)\]]", token):
                text = f"{text}{token}"
                continue

            if token in {"'s", "'re", "'ve", "'ll"}:
                text = f"{text}{token}"
                continue

            if token in {"(", "[", "{"}:
                text = f"{text} {token}"
                continue

            text = f"{text} {token}"

        return text

    def _match_case(self, source: str, candidate: str) -> str:
        if source[:1].isupper():
            return candidate[:1].upper() + candidate[1:]
        return candidate

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))

    def _get_client(self) -> OpenAI | None:
        if not self._settings.openai_api_key:
            return None

        if self._client is None:
            self._client = OpenAI(api_key=self._settings.openai_api_key)

        return self._client


generation_service = GenerationService()
