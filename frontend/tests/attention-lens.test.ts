import assert from "node:assert/strict";
import test from "node:test";

import type { TokenFlowNode } from "../components/canvas/types";
import {
  buildDeterministicFocusViewport,
  buildAttentionHeadLabel,
  buildAttentionFocusNodeIds,
  buildAttentionOverlayEdges,
  buildAttentionRequestPayload,
  buildAttentionTokenId,
  buildPromptTokenNodeId,
  buildAttentionStripTokens,
  canUseAttentionLens,
  canMutateGraphTokenNode,
  isPromptTokenNodeId,
  layoutPromptTokenLane,
  resolvePromptAnchorNode,
  summarizePromptDisplayNodes,
} from "../lib/attention-lens";
import type {
  CanonicalPromptToken,
  HuggingFaceAttentionAggregationMode,
  HuggingFaceAttentionAnalysisMode,
  HuggingFaceAttentionResponse,
  ProviderCapabilitiesDetail,
} from "../types/api";

const HUGGING_FACE_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: true,
  supports_entropy: true,
  supports_attention: true,
  supports_exact_continuation: true,
  supports_streaming: false,
  supports_branching: true,
  supports_continuation: true,
  minimum_output_tokens: 1,
};

const PROMPT_TOKENS: CanonicalPromptToken[] = [
  {
    token_id: 11,
    raw_token: "<|user|>",
    display_token: "<|user|>",
    decoded_contribution: "<|user|>",
    token_bytes: [60, 124, 117, 115, 101, 114, 124, 62],
    full_position: 0,
    source_category: "template",
    source_label: "Template / control",
    special_token: true,
  },
  {
    token_id: 12,
    raw_token: "Question:",
    display_token: "Question:",
    decoded_contribution: "Question:",
    token_bytes: [81, 117, 101, 115, 116, 105, 111, 110, 58],
    full_position: 1,
    source_category: "user_prompt",
    source_label: "User prompt",
    special_token: false,
  },
];

function makeAttentionTokenInfo(
  overrides: Partial<HuggingFaceAttentionResponse["selected_token"]> = {},
) {
  return {
    token_id: 103,
    raw_token: " time",
    display_token: "time",
    decoded_contribution: " time",
    token_bytes: [32, 116, 105, 109, 101],
    full_position: 4,
    analyzed_position: 4,
    sequence_scope: "generated" as const,
    source_category: "generated_output" as const,
    source_label: "Earlier output",
    special_token: false,
    generated_token_index: 2,
    attention_weight: 0.14,
    is_query: false,
    is_selected_token: false,
    ...overrides,
  };
}

function makeAttentionSource(
  overrides: Partial<HuggingFaceAttentionResponse["sources"][number]> = {},
) {
  return {
    token_id: 102,
    raw_token: " good",
    display_token: "good",
    decoded_contribution: " good",
    token_bytes: [32, 103, 111, 111, 100],
    full_position: 3,
    analyzed_position: 3,
    sequence_scope: "generated" as const,
    source_category: "generated_output" as const,
    source_label: "Earlier output",
    special_token: false,
    generated_token_index: 1,
    attention_weight: 0.31,
    rank: 1,
    ...overrides,
  };
}

function makeTokenNode(overrides?: Partial<TokenFlowNode["data"]>): TokenFlowNode {
  return {
    id: "node-3",
    type: "tokenCard",
    position: { x: 0, y: 0 },
    data: {
      kind: "token",
      generationId: "gen-1",
      predictionId: "pred-3",
      segmentId: "segment-1",
      continuationMode: "exact",
      tokenIndex: 2,
      branchId: "branch-1",
      tokenText: " time",
      displayTokenText: "time",
      decodedContribution: " time",
      cumulativeDecodedText: "A good time",
      cumulativeRawText: "A good time",
      cumulativeTokenIds: [101, 102, 103],
      cumulativeLogProbability: -0.42,
      tokenBytes: [32, 116, 105, 109, 101],
      utf8Length: 5,
      characterLength: 5,
      leadingWhitespaceCount: 1,
      trailingWhitespaceCount: 0,
      probability: 0.64,
      rawProbability: 0.64,
      normalizedDisplayedProbability: 0.64,
      displayProbability: 0.64,
      probabilityCoverage: 0.91,
      remainingProbabilityMass: 0.09,
      probabilityMode: "normalized",
      logProbability: -0.42,
      entropy: 0.7,
      latency: 18,
      tokenId: 103,
      tokenizerId: 103,
      generationStep: 2,
      textPreview: "A good time",
      contextBefore: "A good",
      contextAfter: "A good time",
      cumulativeProbability: 0.21,
      branchProbability: 0.64,
      depth: 3,
      rank: 1,
      finishReason: null,
      parentId: "node-2",
      isMainPath: true,
      isCollapsed: false,
      alternativesExpanded: false,
      distributionRequested: true,
      childCount: 0,
      status: "ready",
      requestPrompt: "How fast is a good 400m time?",
      requestModel: "Qwen/Qwen2.5-3B-Instruct",
      requestPreset: "general",
      requestTemperature: 0.7,
      requestTopP: 1,
      requestVariation: 1,
      requestDemoMode: false,
      responseMode: "live",
      sourceNotes: "",
      reasoningIntent: "",
      reasoningStrategy: "",
      reasoningFocusTerms: [],
      branchRationale: null,
      metadata: {
        provider: "hugging_face",
      },
      providerCapabilities: HUGGING_FACE_CAPABILITIES,
      rawLogits: null,
      topAlternatives: [],
      sourceAlternatives: [],
      distributionMessage: null,
      isSearchMatch: false,
      isSearchFocused: false,
      isDimmed: false,
      isActiveReality: true,
      isPinned: false,
      ...overrides,
    },
  } as TokenFlowNode;
}

function makeAttentionResponse(
  aggregationMode: HuggingFaceAttentionAggregationMode = "average_heads",
  analysisMode: HuggingFaceAttentionAnalysisMode = "prediction",
): HuggingFaceAttentionResponse {
  return {
    provider: "hugging_face",
    model_id: "Qwen/Qwen2.5-3B-Instruct",
    model_revision: "rev-3b",
    tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
    tokenizer_revision: "tok-rev",
    analysis_mode: analysisMode,
    selected_token: makeAttentionTokenInfo({
      is_selected_token: true,
      attention_weight: null,
      is_query: false,
    }),
    query_token: makeAttentionTokenInfo({
      token_id: 102,
      raw_token: " good",
      display_token: "good",
      decoded_contribution: " good",
      token_bytes: [32, 103, 111, 111, 100],
      full_position: 3,
      analyzed_position: 3,
      generated_token_index: 1,
      attention_weight: 0.31,
      is_query: true,
      is_selected_token: false,
    }),
    analyzed_tokens: [
      makeAttentionTokenInfo({
        token_id: 11,
        raw_token: "<|user|>",
        display_token: "<|user|>",
        decoded_contribution: "<|user|>",
        token_bytes: [60, 124, 117, 115, 101, 114, 124, 62],
        full_position: 0,
        analyzed_position: 0,
        sequence_scope: "prompt",
        source_category: "template",
        source_label: "Template / control",
        special_token: true,
        generated_token_index: null,
        attention_weight: 0.12,
      }),
      makeAttentionTokenInfo({
        token_id: 101,
        raw_token: "A",
        display_token: "A",
        decoded_contribution: "A",
        token_bytes: [65],
        full_position: 2,
        analyzed_position: 2,
        generated_token_index: 0,
        attention_weight: 0.22,
      }),
      makeAttentionTokenInfo({
        token_id: 102,
        raw_token: " good",
        display_token: "good",
        decoded_contribution: " good",
        token_bytes: [32, 103, 111, 111, 100],
        full_position: 3,
        analyzed_position: 3,
        generated_token_index: 1,
        attention_weight: 0.31,
        is_query: true,
      }),
      makeAttentionTokenInfo({
        is_selected_token: true,
        attention_weight: analysisMode === "representation" ? 0.14 : null,
      }),
    ],
    sources: [
      makeAttentionSource(),
      makeAttentionSource({
        token_id: 11,
        raw_token: "<|user|>",
        display_token: "<|user|>",
        decoded_contribution: "<|user|>",
        token_bytes: [60, 124, 117, 115, 101, 114, 124, 62],
        full_position: 0,
        analyzed_position: 0,
        sequence_scope: "prompt",
        source_category: "template",
        source_label: "Template / control",
        special_token: true,
        generated_token_index: null,
        attention_weight: 0.12,
        rank: 2,
      }),
    ],
    selected_layer: 23,
    selected_head: aggregationMode === "single_head" ? 5 : null,
    aggregation_mode: aggregationMode,
    attention_implementation_used: "eager",
    num_layers: 24,
    num_query_heads: 16,
    selected_token_position: 4,
    query_position: analysisMode === "prediction" ? 3 : 4,
    selected_token_id: 103,
    query_token_id: analysisMode === "prediction" ? 102 : 103,
    prompt_token_count: 2,
    generated_token_index: 2,
    sequence_length: 5,
    layer_index: 23,
    head_index: aggregationMode === "single_head" ? 5 : null,
    average_heads: aggregationMode === "average_heads",
    source_positions: [3, 0],
    attention_weights: [0.31, 0.12],
    attention_mass_sum: 1,
    top_n_coverage: 0.43,
    truncated_context: false,
    context_truncated: false,
    original_full_context_length: 5,
    analyzed_context_length: 5,
  };
}

test("buildAttentionRequestPayload uses canonical ids and the selected generated occurrence", () => {
  const request = buildAttentionRequestPayload({
    aggregationMode: "average_heads",
    allowTruncatedRecompute: false,
    analysisMode: "prediction",
    generatedTokenIds: [101, 102, 103],
    maxConnections: 8,
    maxContextTokens: 256,
    modelId: "Qwen/Qwen2.5-3B-Instruct",
    modelRevision: "rev-3b",
    promptTokenIds: [11, 12],
    promptTokens: PROMPT_TOKENS,
    selectedHead: null,
    selectedLayer: 23,
    tokenizerIdentity: "Qwen/Qwen2.5-3B-Instruct",
    tokenizerRevision: "tok-rev",
  });

  assert.deepEqual(request.prompt_token_ids, [11, 12]);
  assert.deepEqual(request.generated_token_ids, [101, 102, 103]);
  assert.equal(request.selected_generated_token_index, 2);
  assert.equal(request.selected_layer, 23);
  assert.equal(request.selected_head, null);
  assert.equal(request.analysis_mode, "prediction");
  assert.equal(request.allow_truncated_recompute, false);
  assert.deepEqual(request.prompt_tokens, PROMPT_TOKENS);
  assert.equal(request.max_connections, 8);
  assert.equal(request.max_context_tokens, 256);
});

test("buildAttentionOverlayEdges keeps the overlay separate and omits prompt sources that are unavailable on canvas", () => {
  const analysis = makeAttentionResponse();
  const lineageNodeIds = ["root", "node-1", "node-2", "node-3"];
  const pinnedSourceTokenIds = new Set<string>(["attention:generated:3"]);

  const edges = buildAttentionOverlayEdges({
    analysis,
    lineageNodeIds,
    pinnedSourceTokenIds,
    selectedNodeId: "node-3",
  });

  assert.equal(edges.length, 1);
  const firstEdge = edges[0];

  assert.ok(firstEdge);
  assert.ok(firstEdge.data);
  assert.equal(firstEdge.type, "attentionEdge");
  assert.equal(firstEdge.source, "node-3");
  assert.equal(firstEdge.target, "node-2");
  assert.equal(firstEdge.data.weight, 0.31);
  assert.equal(firstEdge.data.isPinned, true);
  assert.equal(firstEdge.data.analysisMode, "prediction");
  assert.equal(firstEdge.data.queryPosition, 3);
});

test("buildAttentionStripTokens preserves query identity and graph-node mapping", () => {
  const analysis = makeAttentionResponse("single_head");
  const tokens = buildAttentionStripTokens({
    analysis,
    lineageNodeIds: ["root", "node-1", "node-2", "node-3"],
    pinnedSourceTokenIds: new Set<string>(["attention:prompt:0"]),
  });

  assert.equal(tokens.length, 4);
  const promptToken = tokens[0];
  const generatedSourceToken = tokens[2];
  const queryToken = tokens[3];

  assert.ok(promptToken);
  assert.ok(generatedSourceToken);
  assert.ok(queryToken);
  assert.equal(promptToken.sequenceScope, "prompt");
  assert.equal(promptToken.graphTokenId, null);
  assert.equal(promptToken.isPinned, true);
  assert.equal(promptToken.sourceCategory, "template");
  assert.equal(generatedSourceToken.graphTokenId, "node-2");
  assert.equal(queryToken.isSelectedToken, true);
  assert.equal(queryToken.tokenId, 103);
});

test("buildAttentionOverlayEdges maps explicit prompt source nodes when they are visible", () => {
  const analysis = makeAttentionResponse();
  const promptNodeIdByPosition = new Map<number, string>([[0, "prompt-token-0"]]);

  const edges = buildAttentionOverlayEdges({
    analysis,
    lineageNodeIds: ["root", "node-1", "node-2", "node-3"],
    pinnedSourceTokenIds: new Set<string>([buildAttentionTokenId("prompt", 0)]),
    promptNodeIdByPosition,
    selectedNodeId: "node-3",
  });

  const promptEdge = edges.find((edge) => edge.data?.sourceScope === "prompt");
  assert.ok(promptEdge);
  assert.equal(promptEdge.target, "prompt-token-0");
  assert.equal(promptEdge.data?.isPinned, true);
});

test("canUseAttentionLens only allows exact Hugging Face token nodes with attention support", () => {
  const eligibleNode = makeTokenNode();
  const approximateNode = makeTokenNode({
    continuationMode: "approximate",
  });
  const openAiNode = makeTokenNode({
    metadata: { provider: "openai" },
    providerCapabilities: {
      ...HUGGING_FACE_CAPABILITIES,
      supports_attention: false,
    },
  });
  const noCanonicalIdsNode = makeTokenNode({
    cumulativeTokenIds: null,
  });

  assert.equal(canUseAttentionLens(eligibleNode), true);
  assert.equal(canUseAttentionLens(approximateNode), false);
  assert.equal(canUseAttentionLens(openAiNode), false);
  assert.equal(canUseAttentionLens(noCanonicalIdsNode), false);
  assert.equal(canUseAttentionLens(null), false);
});

test("buildAttentionHeadLabel reflects the selected aggregation mode", () => {
  assert.equal(buildAttentionHeadLabel("single_head", 5), "Head 5");
  assert.equal(buildAttentionHeadLabel("average_heads", null), "Average heads");
  assert.equal(buildAttentionHeadLabel("max_heads", null), "Max heads");
});

test("prompt token node ids are stable and distinguish canonical positions", () => {
  const promptNodeId = buildPromptTokenNodeId(25);

  assert.equal(promptNodeId, "prompt-token-25");
  assert.equal(isPromptTokenNodeId(promptNodeId), true);
  assert.equal(isPromptTokenNodeId("node-25"), false);
});

test("layoutPromptTokenLane keeps one node per canonical prompt token and wraps long prompt lanes", () => {
  const placements = layoutPromptTokenLane({
    laneAnchorHeight: 124,
    laneAnchorX: 680,
    laneAnchorY: 0,
    maxRowWidth: 420,
    tokens: Array.from({ length: 7 }, (_, index) => ({
      fullPosition: index,
      height: 94,
      sourceCategory: index < 2 ? "template" : index < 5 ? "user_prompt" : "assistant_prefix",
      width: 188,
    })),
  });

  assert.equal(placements.length, 7);
  assert.deepEqual(
    placements.map((placement) => placement.fullPosition),
    [0, 1, 2, 3, 4, 5, 6],
  );
  assert.ok(placements.some((placement) => placement.row > 0));
});

test("summarizePromptDisplayNodes enforces collapsed and expanded prompt-node invariants", () => {
  const summaryOnly = {
    ...makeTokenNode({
      kind: "prompt",
      predictionId: "prompt-summary",
      tokenText: "Prompt summary",
      displayTokenText: "Prompt summary",
      decodedContribution: "",
      metadata: { provider: "hugging_face" },
      sourceCategory: "template",
      sourceLabel: "Prompt summary",
      specialToken: true,
    }),
    id: "root",
  } as TokenFlowNode;
  const promptA = {
    ...makeTokenNode({
      kind: "prompt",
      predictionId: "prompt-0",
      tokenIndex: 0,
      tokenText: "<|user|>",
      displayTokenText: "<|user|>",
      decodedContribution: "<|user|>",
      metadata: { provider: "hugging_face" },
      sourceCategory: "template",
      sourceLabel: "Template / control",
      specialToken: true,
    }),
    id: "prompt-token-0",
  } as TokenFlowNode;
  const promptB = {
    ...makeTokenNode({
      kind: "prompt",
      predictionId: "prompt-1",
      tokenIndex: 1,
      tokenText: "Question:",
      displayTokenText: "Question:",
      decodedContribution: "Question:",
      metadata: { provider: "hugging_face" },
      sourceCategory: "user_prompt",
      sourceLabel: "User prompt",
      specialToken: false,
    }),
    id: "prompt-token-1",
  } as TokenFlowNode;

  assert.deepEqual(summarizePromptDisplayNodes([summaryOnly, makeTokenNode()]), {
    promptNodeCount: 0,
    promptNodeIds: [],
    promptSummaryCount: 1,
  });
  assert.deepEqual(summarizePromptDisplayNodes([promptA, promptB, makeTokenNode()]), {
    promptNodeCount: 2,
    promptNodeIds: ["prompt-token-0", "prompt-token-1"],
    promptSummaryCount: 0,
  });
});

test("resolvePromptAnchorNode prefers the active-path first token over higher visible root siblings", () => {
  const selectedFirstToken = {
    ...makeTokenNode({
      parentId: "root",
      predictionId: "pred-o",
      tokenIndex: 0,
      tokenText: "O",
      displayTokenText: "O",
      decodedContribution: "O",
      rank: 1,
    }),
    id: "main:0",
    hidden: false,
    position: { x: 120, y: 24 },
  } as TokenFlowNode;
  const higherAlternative = {
    ...makeTokenNode({
      parentId: "root",
      predictionId: "pred-alt",
      tokenIndex: 0,
      tokenText: "BLUE",
      displayTokenText: "BLUE",
      decodedContribution: "BLUE",
      rank: 2,
    }),
    id: "root:0:alt",
    hidden: false,
    position: { x: 120, y: -960 },
  } as TokenFlowNode;
  const laterActiveToken = {
    ...makeTokenNode({
      parentId: "main:0",
      predictionId: "pred-ce",
      tokenIndex: 1,
      tokenText: "CE",
      displayTokenText: "CE",
      decodedContribution: "CE",
      rank: 1,
    }),
    id: "main:1",
    hidden: false,
    position: { x: 320, y: 24 },
  } as TokenFlowNode;

  const anchor = resolvePromptAnchorNode({
    activePathIds: ["root", "main:0", "main:1"],
    displayNodes: [higherAlternative, selectedFirstToken, laterActiveToken],
  });

  assert.equal(anchor?.id, "main:0");
});

test("buildAttentionFocusNodeIds excludes hidden and stale nodes from attention fitting", () => {
  const visibleSelectedNode = makeTokenNode({ predictionId: "pred-visible", tokenIndex: 1 });
  const visiblePromptNode = {
    ...makeTokenNode({
      kind: "prompt",
      predictionId: "prompt-2",
      tokenIndex: 2,
      tokenText: "<|user|>",
      displayTokenText: "<|user|>",
      decodedContribution: "<|user|>",
      metadata: { provider: "hugging_face" },
      parentId: null,
      probability: 1,
      rawProbability: 1,
      displayProbability: 1,
      probabilityCoverage: 1,
      remainingProbabilityMass: 0,
      sourceCategory: "template",
      sourceLabel: "Template / control",
      specialToken: true,
    }),
    id: "prompt-token-2",
    hidden: false,
  } as TokenFlowNode;
  const hiddenNode = {
    ...makeTokenNode({ predictionId: "pred-hidden", tokenIndex: 9 }),
    id: "node-hidden",
    hidden: true,
  } as TokenFlowNode;

  const focusIds = buildAttentionFocusNodeIds({
    displayNodes: [visibleSelectedNode, visiblePromptNode, hiddenNode],
    selectedNodeId: visibleSelectedNode.id,
    sourceGraphNodeIds: [visiblePromptNode.id, hiddenNode.id, "stale-node"],
  });

  assert.deepEqual(focusIds, [visibleSelectedNode.id, visiblePromptNode.id]);
});

test("buildDeterministicFocusViewport keeps only valid live nodes and computes a bounded viewport", () => {
  const result = buildDeterministicFocusViewport({
    containerHeight: 900,
    containerWidth: 1200,
    maxZoom: 1.15,
    minZoom: 0.65,
    nodes: [
      {
        id: "selected",
        hidden: false,
        inDom: true,
        width: 120,
        height: 72,
        x: 400,
        y: 200,
      },
      {
        id: "prompt-token-0",
        hidden: false,
        inDom: true,
        width: 120,
        height: 72,
        x: 180,
        y: 180,
      },
      {
        id: "stale",
        hidden: false,
        inDom: false,
        width: 120,
        height: 72,
        x: 9999,
        y: 9999,
      },
    ],
    padding: 0.18,
    selectedNodeId: "selected",
    sourceNodeIds: ["prompt-token-0", "stale"],
  });

  assert.deepEqual(result.includedIds, ["selected", "prompt-token-0"]);
  assert.ok(result.bounds);
  assert.ok(result.viewport);
  assert.ok(result.viewport!.zoom >= 0.65);
  assert.ok(result.viewport!.zoom <= 1.15);
});

test("buildDeterministicFocusViewport respects occupied viewport insets", () => {
  const noInset = buildDeterministicFocusViewport({
    containerHeight: 900,
    containerWidth: 1200,
    maxZoom: 1.15,
    minZoom: 0.62,
    nodes: [
      {
        id: "selected",
        hidden: false,
        inDom: true,
        width: 160,
        height: 72,
        x: 900,
        y: 240,
      },
      {
        id: "prompt-token-2",
        hidden: false,
        inDom: true,
        width: 120,
        height: 72,
        x: 80,
        y: 120,
      },
    ],
    padding: 0.08,
    selectedNodeId: "selected",
    sourceNodeIds: ["prompt-token-2"],
  });
  const inset = buildDeterministicFocusViewport({
    containerHeight: 900,
    containerWidth: 1200,
    maxZoom: 1.15,
    minZoom: 0.62,
    nodes: [
      {
        id: "selected",
        hidden: false,
        inDom: true,
        width: 160,
        height: 72,
        x: 900,
        y: 240,
      },
      {
        id: "prompt-token-2",
        hidden: false,
        inDom: true,
        width: 120,
        height: 72,
        x: 80,
        y: 120,
      },
    ],
    padding: 0.08,
    selectedNodeId: "selected",
    sourceNodeIds: ["prompt-token-2"],
    viewportInset: {
      right: 280,
    },
  });

  assert.ok(noInset.viewport);
  assert.ok(inset.viewport);
  const selectedRightWithoutInset = noInset.viewport!.x + (900 + 160) * noInset.viewport!.zoom;
  const selectedRightWithInset = inset.viewport!.x + (900 + 160) * inset.viewport!.zoom;

  assert.ok(selectedRightWithInset < selectedRightWithoutInset);
  assert.ok(selectedRightWithInset <= 1200 - 280 + 1);
});

test("buildDeterministicFocusViewport preserves the current viewport when the target is not renderable", () => {
  const result = buildDeterministicFocusViewport({
    containerHeight: 900,
    containerWidth: 1200,
    maxZoom: 1.15,
    minZoom: 0.65,
    nodes: [
      {
        id: "selected",
        hidden: false,
        inDom: false,
        width: 120,
        height: 72,
        x: 400,
        y: 200,
      },
    ],
    padding: 0.18,
    selectedNodeId: "selected",
    sourceNodeIds: [],
  });

  assert.equal(result.viewport, null);
  assert.equal(result.bounds, null);
});

test("prompt nodes remain inspectable but are not mutable graph tokens", () => {
  const promptNode = makeTokenNode({
    kind: "prompt",
    predictionId: "prompt-7",
    tokenIndex: 7,
    tokenText: "Question:",
    displayTokenText: "Question:",
    decodedContribution: "Question:",
    metadata: { provider: "hugging_face" },
    parentId: null,
    sourceCategory: "user_prompt",
    sourceLabel: "User prompt",
    specialToken: false,
  });

  assert.equal(canMutateGraphTokenNode(makeTokenNode()), true);
  assert.equal(canMutateGraphTokenNode(promptNode), false);
});
