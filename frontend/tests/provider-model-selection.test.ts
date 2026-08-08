import assert from "node:assert/strict";
import test from "node:test";

import {
  findCompatibleModelId,
  reconcileProviderModelSelection,
} from "../lib/provider-selection";
import type { ModelOption, ProviderCapabilitiesDetail } from "../types/api";

const OPENAI_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: true,
  supports_entropy: true,
  supports_attention: false,
  supports_exact_continuation: false,
  supports_streaming: false,
  supports_branching: true,
  supports_continuation: true,
  minimum_output_tokens: 16,
};

const OLLAMA_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: false,
  supports_entropy: false,
  supports_attention: false,
  supports_exact_continuation: false,
  supports_streaming: true,
  supports_branching: false,
  supports_continuation: false,
  minimum_output_tokens: 1,
};

const MODELS: ModelOption[] = [
  {
    id: "gpt-4.1-mini",
    label: "GPT-4.1 mini",
    provider: "openai",
    group: "OpenAI",
    status: "ready",
    capabilities: OPENAI_CAPABILITIES,
  },
  {
    id: "gpt-4o-mini",
    label: "GPT-4o mini",
    provider: "openai",
    group: "OpenAI",
    status: "ready",
    capabilities: OPENAI_CAPABILITIES,
  },
  {
    id: "phi3",
    label: "phi3",
    provider: "ollama",
    group: "Ollama",
    status: "ready",
    capabilities: OLLAMA_CAPABILITIES,
  },
  {
    id: "qwen2.5:3b",
    label: "qwen2.5:3b",
    provider: "ollama",
    group: "Ollama",
    status: "ready",
    capabilities: OLLAMA_CAPABILITIES,
  },
];

test("reconciles OpenAI -> Ollama -> OpenAI provider switches to a stable compatible model", () => {
  const openAiState = {
    selectedProvider: "openai" as const,
    model: "gpt-4.1-mini",
  };

  const ollamaTransition = reconcileProviderModelSelection(
    {
      selectedProvider: "ollama",
      model: openAiState.model,
    },
    MODELS,
  );

  assert.deepEqual(ollamaTransition, {
    selectedProvider: "ollama",
    model: "phi3",
  });
  assert.equal(
    reconcileProviderModelSelection(ollamaTransition, MODELS),
    ollamaTransition,
    "Ollama selection should already be stable after one reconciliation step.",
  );

  const openAiTransition = reconcileProviderModelSelection(
    {
      selectedProvider: "openai",
      model: ollamaTransition.model,
    },
    MODELS,
  );

  assert.deepEqual(openAiTransition, {
    selectedProvider: "openai",
    model: "gpt-4.1-mini",
  });
  assert.equal(
    reconcileProviderModelSelection(openAiTransition, MODELS),
    openAiTransition,
    "OpenAI selection should already be stable after one reconciliation step.",
  );
});

test("does not request a model update when the current model already matches the provider", () => {
  const stableState = {
    selectedProvider: "ollama" as const,
    model: "qwen2.5:3b",
  };

  assert.equal(findCompatibleModelId(MODELS, "ollama", stableState.model), stableState.model);
  assert.equal(reconcileProviderModelSelection(stableState, MODELS), stableState);
});

test("leaves the current model alone when the provider has no available models", () => {
  const noOllamaModels = MODELS.filter((model) => model.provider !== "ollama");
  const state = {
    selectedProvider: "ollama" as const,
    model: "gpt-4.1-mini",
  };

  assert.equal(findCompatibleModelId(noOllamaModels, "ollama", state.model), null);
  assert.equal(reconcileProviderModelSelection(state, noOllamaModels), state);
});
