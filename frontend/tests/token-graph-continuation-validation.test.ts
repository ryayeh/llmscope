import assert from "node:assert/strict";
import test from "node:test";

import type { TokenGraphNodeRecord, TokenGraphState } from "../lib/token-graph";
import { buildContinuationValidation } from "../lib/token-graph";
import type { ProviderCapabilitiesDetail } from "../types/api";

const HUGGING_FACE_PROVIDER_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: true,
  supports_entropy: true,
  supports_attention: false,
  supports_exact_continuation: true,
  supports_streaming: false,
  supports_branching: true,
  supports_continuation: true,
  minimum_output_tokens: 1,
};

function makeNode(
  overrides: Partial<TokenGraphNodeRecord> & Pick<TokenGraphNodeRecord, "id" | "kind" | "rawToken">,
): TokenGraphNodeRecord {
  return {
    id: overrides.id,
    kind: overrides.kind,
    parentId: overrides.parentId ?? null,
    childIds: overrides.childIds ?? [],
    generationId: overrides.generationId ?? "gen-hf",
    branchId: overrides.branchId ?? overrides.id,
    continuationMode: overrides.continuationMode ?? "exact",
    segmentId: overrides.segmentId ?? "segment-hf",
    rawToken: overrides.rawToken,
    displayToken: overrides.displayToken ?? overrides.rawToken,
    decodedContribution: overrides.decodedContribution ?? "",
    cumulativeDecodedText: overrides.cumulativeDecodedText ?? "",
    cumulativeRawText: overrides.cumulativeRawText ?? "",
    cumulativeTokenIds: overrides.cumulativeTokenIds ?? null,
    cumulativeLogProbability: overrides.cumulativeLogProbability ?? 0,
    tokenBytes: overrides.tokenBytes ?? [],
    tokenId: overrides.tokenId ?? null,
    tokenizerId: overrides.tokenizerId ?? null,
    logProbability: overrides.logProbability ?? 0,
    probability: overrides.probability ?? 1,
    rawProbability: overrides.rawProbability ?? 1,
    normalizedDisplayedProbability: overrides.normalizedDisplayedProbability ?? 1,
    rank: overrides.rank ?? 1,
    entropy: overrides.entropy ?? 0,
    latencyMs: overrides.latencyMs ?? 0,
    cumulativeProbability: overrides.cumulativeProbability ?? 1,
    branchProbability: overrides.branchProbability ?? 1,
    finishReason: overrides.finishReason ?? null,
    generationDepth: overrides.generationDepth ?? 0,
    generationStep: overrides.generationStep ?? -1,
    contextBefore: overrides.contextBefore ?? "",
    contextAfter: overrides.contextAfter ?? "",
    requestPrompt: overrides.requestPrompt ?? "Prompt",
    requestModel: overrides.requestModel ?? "Qwen/Qwen2.5-3B-Instruct",
    requestPreset: overrides.requestPreset ?? "general",
    requestTemperature: overrides.requestTemperature ?? 0,
    requestTopP: overrides.requestTopP ?? 1,
    requestVariation: overrides.requestVariation ?? 0,
    requestDemoMode: overrides.requestDemoMode ?? false,
    responseMode: overrides.responseMode ?? "live",
    sourceNotes: overrides.sourceNotes ?? "",
    reasoningIntent: overrides.reasoningIntent ?? "",
    reasoningStrategy: overrides.reasoningStrategy ?? "",
    reasoningFocusTerms: overrides.reasoningFocusTerms ?? [],
    branchRationale: overrides.branchRationale ?? null,
    metadata: overrides.metadata ?? {},
    providerCapabilities:
      overrides.providerCapabilities ?? HUGGING_FACE_PROVIDER_CAPABILITIES,
    sourceAlternatives: overrides.sourceAlternatives ?? [],
    alternativesExpanded: overrides.alternativesExpanded ?? false,
    distributionRequested: overrides.distributionRequested ?? false,
    distributionMessage: overrides.distributionMessage ?? null,
  };
}

function createHuggingFaceGraph(
  overrides?: {
    node2ContextAfter?: string;
    node2CumulativeTokenIds?: number[] | null;
    node2TokenId?: number | null;
    node2DecodedText?: string;
  },
): TokenGraphState {
  const root = makeNode({
    id: "root",
    kind: "prompt",
    rawToken: "Prompt",
    childIds: ["hf-0"],
    cumulativeTokenIds: [11, 12],
    requestPrompt: "Prompt",
    metadata: { provider: "hugging_face" },
  });
  const node1 = makeNode({
    id: "hf-0",
    kind: "token",
    rawToken: "The",
    parentId: "root",
    childIds: ["hf-1"],
    decodedContribution: "The",
    cumulativeDecodedText: "The",
    cumulativeRawText: "The",
    cumulativeTokenIds: [11, 12, 101],
    tokenId: 101,
    tokenizerId: 101,
    generationDepth: 1,
    generationStep: 0,
    contextBefore: "",
    contextAfter: "The",
    metadata: {
      provider: "hugging_face",
      resolved_revision: "resolved-model-sha",
      requested_revision: "rev-3b",
      tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
      tokenizer_revision: "resolved-tokenizer-sha",
    },
  });
  const node2 = makeNode({
    id: "hf-1",
    kind: "token",
    rawToken: "\u0120time",
    parentId: "hf-0",
    decodedContribution: " time",
    cumulativeDecodedText: overrides?.node2DecodedText ?? "The time",
    cumulativeRawText: `The${"\u0120"}time`,
    cumulativeTokenIds: overrides?.node2CumulativeTokenIds ?? [11, 12, 101, 102],
    tokenId: overrides?.node2TokenId ?? 102,
    tokenizerId: overrides?.node2TokenId ?? 102,
    generationDepth: 2,
    generationStep: 1,
    contextBefore: "The",
    contextAfter: overrides?.node2ContextAfter ?? `The${"\u0120"}time`,
    metadata: {
      provider: "hugging_face",
      resolved_revision: "resolved-model-sha",
      requested_revision: "rev-3b",
      tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
      tokenizer_revision: "resolved-tokenizer-sha",
    },
  });

  return {
    rootNodeId: "root",
    rootPrompt: "Prompt",
    nodesById: {
      root,
      "hf-0": node1,
      "hf-1": node2,
    },
    generationOrder: [],
    generationsById: {},
  };
}

function createDeepHuggingFaceGraphWithStalePreview(): TokenGraphState {
  const decodedContributions = [
    "The",
    " time",
    " for",
    " a",
    " 16",
    "-",
    "year",
    "-",
    "old",
    " athlete",
    " varies",
    " by",
    " experience",
    " and",
    " level",
    " with",
    " training",
    ",",
    " but",
    " strong",
    " runners",
    " can",
    " improve",
    " with",
    " pacing",
    ",",
  ];
  const rawTokens = [
    "The",
    "\u0120time",
    "\u0120for",
    "\u0120a",
    "\u012016",
    "-",
    "year",
    "-",
    "old",
    "\u0120athlete",
    "\u0120varies",
    "\u0120by",
    "\u0120experience",
    "\u0120and",
    "\u0120level",
    "\u0120with",
    "\u0120training",
    ",",
    "\u0120but",
    "\u0120strong",
    "\u0120runners",
    "\u0120can",
    "\u0120improve",
    "\u0120with",
    "\u0120pacing",
    ",",
  ];
  const promptTokenIds = [11, 12];
  const root = makeNode({
    id: "root",
    kind: "prompt",
    rawToken: "Prompt",
    childIds: ["hf-0"],
    cumulativeTokenIds: promptTokenIds,
    requestPrompt: "Prompt",
    metadata: { provider: "hugging_face" },
  });
  const nodesById: TokenGraphState["nodesById"] = { root };
  let cumulativeDecodedText = "";
  let previousId = "root";

  for (let index = 0; index < decodedContributions.length; index += 1) {
    const currentId = `hf-${index}`;
    const decodedContribution = decodedContributions[index];
    const rawToken = rawTokens[index];
    const isSelectedNode = index === decodedContributions.length - 1;

    cumulativeDecodedText += decodedContribution;
    nodesById[currentId] = makeNode({
      id: currentId,
      kind: "token",
      rawToken,
      parentId: previousId,
      childIds: isSelectedNode ? [] : [`hf-${index + 1}`],
      decodedContribution,
      cumulativeDecodedText: isSelectedNode ? "," : cumulativeDecodedText,
      cumulativeRawText: rawTokens.slice(0, index + 1).join(""),
      cumulativeTokenIds: [...promptTokenIds, ...Array.from({ length: index + 1 }, (_, step) => 101 + step)],
      tokenId: 101 + index,
      tokenizerId: 101 + index,
      generationDepth: index + 1,
      generationStep: index,
      contextBefore: index === 0 ? "" : decodedContributions.slice(0, index).join(""),
      contextAfter: isSelectedNode ? "," : cumulativeDecodedText,
      metadata: isSelectedNode
        ? {
            resolved_revision: "resolved-model-sha",
            requested_revision: "rev-3b",
            tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision: "resolved-tokenizer-sha",
          }
        : {
            provider: "hugging_face",
            resolved_revision: "resolved-model-sha",
            requested_revision: "rev-3b",
            tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
            tokenizer_revision: "resolved-tokenizer-sha",
          },
    });
    previousId = currentId;
  }

  return {
    rootNodeId: "root",
    rootPrompt: "Prompt",
    nodesById,
    generationOrder: [],
    generationsById: {},
  };
}

test("buildContinuationValidation keeps Hugging Face canonical token IDs authoritative", () => {
  const graph = createHuggingFaceGraph();
  const validation = buildContinuationValidation(graph, "hf-1");

  assert.equal(validation.isValid, true);
  assert.equal(validation.validationMode, "token_ids");
  assert.deepEqual(validation.promptTokenIds, [11, 12]);
  assert.deepEqual(validation.canonicalPrefixTokenIds, [11, 12, 101, 102]);
  assert.deepEqual(validation.generatedPrefixTokenIds, [101, 102]);
  assert.equal(validation.selectedTokenId, 102);
  assert.equal(validation.assistantPrefix, "The time");
  assert.equal(validation.tokenCount, 2);
  assert.equal(validation.assistantCharacterLength, 8);
  assert.equal(validation.assistantUtf8Length, 8);
  assert.equal(validation.modelRevision, "resolved-model-sha");
  assert.equal(validation.tokenizerIdentity, "Qwen/Qwen2.5-3B-Instruct");
  assert.equal(validation.tokenizerRevision, "resolved-tokenizer-sha");
  assert.deepEqual(validation.warnings, []);
});

test("buildContinuationValidation does not reject Hugging Face raw-token markers like space-prefixed time", () => {
  const graph = createHuggingFaceGraph({
    node2ContextAfter: `The${"\u0120"}time`,
  });
  const validation = buildContinuationValidation(graph, "hf-1");

  assert.equal(validation.isValid, true);
  assert.deepEqual(validation.warnings, []);
});

test("buildContinuationValidation rejects real Hugging Face token-ID mismatches", () => {
  const graph = createHuggingFaceGraph({
    node2CumulativeTokenIds: [11, 12, 101, 999],
    node2TokenId: 102,
  });
  const validation = buildContinuationValidation(graph, "hf-1");

  assert.equal(validation.isValid, false);
  assert.ok(
    validation.warnings.includes(
      "The selected Hugging Face Local token ID does not match the canonical token-ID prefix.",
    ),
  );
});

test("buildContinuationValidation counts characters and bytes from decoded assistant text", () => {
  const root = makeNode({
    id: "root",
    kind: "prompt",
    rawToken: "Prompt",
    childIds: ["hf-0"],
    cumulativeTokenIds: [11, 12],
    requestPrompt: "Prompt",
    metadata: { provider: "hugging_face" },
  });
  const node1 = makeNode({
    id: "hf-0",
    kind: "token",
    rawToken: "Hi",
    parentId: "root",
    childIds: ["hf-1"],
    decodedContribution: "Hi",
    cumulativeDecodedText: "Hi",
    cumulativeRawText: "Hi",
    cumulativeTokenIds: [11, 12, 101],
    tokenId: 101,
    tokenizerId: 101,
    generationDepth: 1,
    generationStep: 0,
    contextBefore: "",
    contextAfter: "Hi",
    metadata: {
      provider: "hugging_face",
      resolved_revision: "resolved-model-sha",
      requested_revision: "rev-3b",
      tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
      tokenizer_revision: "resolved-tokenizer-sha",
    },
  });
  const node2 = makeNode({
    id: "hf-1",
    kind: "token",
    rawToken: "\u0120\u{1F600}",
    parentId: "hf-0",
    decodedContribution: " \u{1F600}",
    cumulativeDecodedText: "Hi \u{1F600}",
    cumulativeRawText: `Hi${"\u0120"}\u{1F600}`,
    cumulativeTokenIds: [11, 12, 101, 102],
    tokenId: 102,
    tokenizerId: 102,
    generationDepth: 2,
    generationStep: 1,
    contextBefore: "Hi",
    contextAfter: `Hi${"\u0120"}\u{1F600}`,
    metadata: {
      provider: "hugging_face",
      resolved_revision: "resolved-model-sha",
      requested_revision: "rev-3b",
      tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
      tokenizer_revision: "resolved-tokenizer-sha",
    },
  });
  const graph: TokenGraphState = {
    rootNodeId: "root",
    rootPrompt: "Prompt",
    nodesById: {
      root,
      "hf-0": node1,
      "hf-1": node2,
    },
    generationOrder: [],
    generationsById: {},
  };
  const validation = buildContinuationValidation(graph, "hf-1");

  assert.equal(validation.assistantPrefix, "Hi \u{1F600}");
  assert.equal(validation.assistantCharacterLength, 4);
  assert.equal(validation.assistantUtf8Length, 7);
});

test("buildContinuationValidation rebuilds a deep Hugging Face preview from lineage when cached text is stale", () => {
  const graph = createDeepHuggingFaceGraphWithStalePreview();
  const validation = buildContinuationValidation(graph, "hf-25");

  assert.equal(validation.isValid, true);
  assert.equal(validation.validationMode, "token_ids");
  assert.equal(
    validation.assistantPrefix,
    "The time for a 16-year-old athlete varies by experience and level with training, but strong runners can improve with pacing,",
  );
  assert.equal(validation.expectedAssistantPrefix, validation.assistantPrefix);
  assert.equal(validation.tokenCount, 26);
  assert.ok(validation.assistantCharacterLength > 1);
  assert.equal(validation.assistantUtf8Length, validation.assistantPrefix.length);
  assert.deepEqual(
    validation.generatedPrefixTokenIds,
    Array.from({ length: 26 }, (_, index) => 101 + index),
  );
  assert.deepEqual(validation.warnings, []);
});
