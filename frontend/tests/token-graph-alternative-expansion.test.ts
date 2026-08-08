import assert from "node:assert/strict";
import test from "node:test";

import type {
  ApplyExpansionOptions,
  TokenGraphAlternativeRecord,
  TokenGraphNodeRecord,
  TokenGraphState,
} from "../lib/token-graph";
import { applyExpansionToTokenGraph, materializeSourceAlternativesForNode } from "../lib/token-graph";
import type { ContinuationMode, NodeExpansionResponse, ProviderCapabilitiesDetail } from "../types/api";

const encoder = new TextEncoder();
const expansionOptions: ApplyExpansionOptions = {
  requestPrompt: "Prompt",
  model: "demo-model",
  preset: "general",
  temperature: 0.7,
  topP: 1,
  variation: 1,
  demoMode: true,
};

const DEFAULT_PROVIDER_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: true,
  supports_entropy: true,
  supports_attention: false,
  supports_exact_continuation: false,
  supports_streaming: false,
  supports_branching: true,
  supports_continuation: true,
  minimum_output_tokens: 16,
};

function encodeBytes(token: string) {
  return Array.from(encoder.encode(token));
}

function makeAlternative(args: {
  nodeId: string;
  branchId: string;
  rawToken: string;
  parentRawText: string;
  parentDecodedText: string;
  generationStep: number;
  rank: number;
  probability: number;
}): TokenGraphAlternativeRecord {
  return {
    nodeId: args.nodeId,
    branchId: args.branchId,
    continuationMode: "exact",
    segmentId: "segment-main",
    rawToken: args.rawToken,
    displayToken: args.rawToken.trimStart(),
    decodedContribution: args.rawToken,
    cumulativeDecodedText: `${args.parentDecodedText}${args.rawToken}`,
    cumulativeRawText: `${args.parentRawText}${args.rawToken}`,
    cumulativeTokenIds: null,
    cumulativeLogProbability: Math.log(args.probability),
    generationStep: args.generationStep,
    tokenBytes: encodeBytes(args.rawToken),
    tokenId: null,
    tokenizerId: null,
    probability: args.probability,
    rawProbability: args.probability,
    normalizedDisplayedProbability: args.probability,
    logProbability: Math.log(args.probability),
    entropy: 0.2,
    latencyMs: 12,
    rank: args.rank,
    contextBefore: args.parentDecodedText,
    contextAfter: `${args.parentDecodedText}${args.rawToken}`,
    finishReason: null,
    rationale: null,
    metadata: {},
  };
}

function makeNode(args: {
  id: string;
  kind?: "prompt" | "token";
  parentId: string | null;
  childIds?: string[];
  rawToken: string;
  displayToken?: string;
  cumulativeDecodedText: string;
  cumulativeRawText: string;
  generationStep: number;
  rank?: number;
  probability?: number;
  sourceAlternatives?: TokenGraphAlternativeRecord[];
  branchId?: string;
}): TokenGraphNodeRecord {
  const probability = args.probability ?? 0.5;

  return {
    id: args.id,
    kind: args.kind ?? "token",
    parentId: args.parentId,
    childIds: args.childIds ?? [],
    generationId: "gen-main",
    branchId: args.branchId ?? args.id,
    continuationMode: "exact",
    segmentId: "segment-main",
    rawToken: args.rawToken,
    displayToken: args.displayToken ?? args.rawToken.trimStart(),
    decodedContribution: args.rawToken,
    cumulativeDecodedText: args.cumulativeDecodedText,
    cumulativeRawText: args.cumulativeRawText,
    cumulativeTokenIds: null,
    cumulativeLogProbability: Math.log(probability),
    tokenBytes: encodeBytes(args.rawToken),
    tokenId: null,
    tokenizerId: null,
    logProbability: Math.log(probability),
    probability,
    rawProbability: probability,
    normalizedDisplayedProbability: probability,
    rank: args.rank ?? 1,
    entropy: 0.2,
    latencyMs: 12,
    cumulativeProbability: probability,
    branchProbability: probability,
    finishReason: null,
    generationDepth: Math.max(args.generationStep + 1, 0),
    generationStep: args.generationStep,
    contextBefore: args.parentId ? args.cumulativeDecodedText.slice(0, -args.rawToken.length) : "",
    contextAfter: args.cumulativeDecodedText,
    requestPrompt: "Prompt",
    requestModel: "demo-model",
    requestPreset: "general",
    requestTemperature: 0.7,
    requestTopP: 1,
    requestVariation: 1,
    requestDemoMode: true,
    responseMode: "demo",
    sourceNotes: "",
    reasoningIntent: "",
    reasoningStrategy: "",
    reasoningFocusTerms: [],
    branchRationale: null,
    metadata: {},
    providerCapabilities: DEFAULT_PROVIDER_CAPABILITIES,
    sourceAlternatives: args.sourceAlternatives ?? [],
    alternativesExpanded: false,
    distributionRequested: false,
    distributionMessage: null,
  };
}

function createGraph(): TokenGraphState {
  const root = makeNode({
    id: "root",
    kind: "prompt",
    parentId: null,
    childIds: ["n0"],
    rawToken: "",
    displayToken: "Prompt",
    cumulativeDecodedText: "",
    cumulativeRawText: "",
    generationStep: -1,
    branchId: "root",
  });
  const n0 = makeNode({
    id: "n0",
    parentId: "root",
    childIds: ["n1"],
    rawToken: "A",
    cumulativeDecodedText: "A",
    cumulativeRawText: "A",
    generationStep: 0,
    sourceAlternatives: [
      makeAlternative({
        nodeId: "n0-alt-good",
        branchId: "n0-alt-good",
        rawToken: "Good",
        parentRawText: "",
        parentDecodedText: "",
        generationStep: 0,
        rank: 2,
        probability: 0.18,
      }),
      makeAlternative({
        nodeId: "n0-alt-for",
        branchId: "n0-alt-for",
        rawToken: "For",
        parentRawText: "",
        parentDecodedText: "",
        generationStep: 0,
        rank: 3,
        probability: 0.07,
      }),
    ],
  });
  const n1 = makeNode({
    id: "n1",
    parentId: "n0",
    childIds: ["n2"],
    rawToken: " good",
    cumulativeDecodedText: "A good",
    cumulativeRawText: "A good",
    generationStep: 1,
  });
  const n2 = makeNode({
    id: "n2",
    parentId: "n1",
    childIds: ["n3"],
    rawToken: " time",
    cumulativeDecodedText: "A good time",
    cumulativeRawText: "A good time",
    generationStep: 2,
    sourceAlternatives: [
      makeAlternative({
        nodeId: "n2-alt-sprint",
        branchId: "n2-alt-sprint",
        rawToken: " sprint",
        parentRawText: "A good",
        parentDecodedText: "A good",
        generationStep: 2,
        rank: 2,
        probability: 0.16,
      }),
      makeAlternative({
        nodeId: "n2-alt-running",
        branchId: "n2-alt-running",
        rawToken: " running",
        parentRawText: "A good",
        parentDecodedText: "A good",
        generationStep: 2,
        rank: 3,
        probability: 0.11,
      }),
    ],
  });
  const n3 = makeNode({
    id: "n3",
    parentId: "n2",
    childIds: ["n4"],
    rawToken: " for",
    cumulativeDecodedText: "A good time for",
    cumulativeRawText: "A good time for",
    generationStep: 3,
  });
  const n4 = makeNode({
    id: "n4",
    parentId: "n3",
    rawToken: " time",
    cumulativeDecodedText: "A good time for time",
    cumulativeRawText: "A good time for time",
    generationStep: 4,
    sourceAlternatives: [
      makeAlternative({
        nodeId: "n4-alt-race",
        branchId: "n4-alt-race",
        rawToken: " race",
        parentRawText: "A good time for",
        parentDecodedText: "A good time for",
        generationStep: 4,
        rank: 2,
        probability: 0.19,
      }),
      makeAlternative({
        nodeId: "n4-alt-track",
        branchId: "n4-alt-track",
        rawToken: " track",
        parentRawText: "A good time for",
        parentDecodedText: "A good time for",
        generationStep: 4,
        rank: 3,
        probability: 0.14,
      }),
    ],
  });

  return {
    rootNodeId: "root",
    rootPrompt: "Prompt",
    nodesById: {
      root,
      n0,
      n1,
      n2,
      n3,
      n4,
    },
    generationOrder: ["gen-main"],
    generationsById: {},
  };
}

function makeExpansionResponse(
  graph: TokenGraphState,
  parentId: string,
  expandedAt: string,
  children: Array<{
    id: string;
    branchId: string;
    continuationMode?: ContinuationMode;
    segmentId?: string | null;
    rawToken: string;
    generationStep: number;
    rank: number;
    probability: number;
  }>,
): NodeExpansionResponse {
  const parent = graph.nodesById[parentId];

  if (!parent) {
    throw new Error(`Unknown parent node ${parentId}`);
  }

  return {
    mode: "demo",
    parent_node_id: parentId,
    entropy: 0.2,
    expanded_at: expandedAt,
    notes: "continuation test",
    children: children.map((child) => ({
      id: child.id,
      segment_id: child.segmentId ?? "segment-main",
      branch_id: child.branchId,
      parent_node_id: parentId,
      model: "demo-model",
      source: "demo",
      token: child.rawToken,
      display_token: child.rawToken.trimStart(),
      token_bytes: encodeBytes(child.rawToken),
      decoded_contribution: child.rawToken,
      cumulative_decoded_text: `${parent.cumulativeDecodedText}${child.rawToken}`,
      cumulative_token_ids: null,
      cumulative_log_probability: Number(
        (parent.cumulativeLogProbability + Math.log(child.probability)).toFixed(6),
      ),
      token_id: null,
      tokenizer_id: null,
      probability: child.probability,
      raw_probability: child.probability,
      normalized_displayed_probability: child.probability,
      log_probability: Number(Math.log(child.probability).toFixed(6)),
      entropy: 0.2,
      cumulative_probability: Number((parent.cumulativeProbability * child.probability).toFixed(6)),
      latency_ms: 12,
      depth: child.generationStep + 1,
      rank: child.rank,
      text_preview: `${parent.cumulativeDecodedText}${child.rawToken}`,
      context_before: parent.cumulativeRawText,
      context_after: `${parent.cumulativeDecodedText}${child.rawToken}`,
      finish_reason: null,
      rationale: null,
      generation_step: child.generationStep,
      continuation_mode: child.continuationMode ?? "exact",
      metadata: {
        parent_node_id: parentId,
        source: "demo",
      },
    })),
  };
}

test("materializes first-token alternatives under the clicked occurrence without duplicates", () => {
  const graph = createGraph();
  const firstExpansion = materializeSourceAlternativesForNode(graph, "n0");

  assert.deepEqual(firstExpansion.nodesById.root.childIds, ["n0", "n0-alt-good", "n0-alt-for"]);
  assert.equal(firstExpansion.nodesById["n0-alt-good"]?.parentId, "root");
  assert.equal(firstExpansion.nodesById["n0-alt-good"]?.cumulativeRawText, "Good");
  assert.equal(firstExpansion.nodesById.n0.alternativesExpanded, true);

  const secondExpansion = materializeSourceAlternativesForNode(firstExpansion, "n0");

  assert.deepEqual(secondExpansion.nodesById.root.childIds, ["n0", "n0-alt-good", "n0-alt-for"]);
  assert.equal(
    Object.keys(secondExpansion.nodesById).length,
    Object.keys(firstExpansion.nodesById).length,
  );
});

test("materializes a later token's own alternatives instead of the first token's alternatives", () => {
  const graph = createGraph();
  const expanded = materializeSourceAlternativesForNode(graph, "n2");

  assert.deepEqual(expanded.nodesById.n1.childIds, ["n2", "n2-alt-sprint", "n2-alt-running"]);
  assert.equal(expanded.nodesById["n2-alt-sprint"]?.parentId, "n1");
  assert.equal(expanded.nodesById["n2-alt-sprint"]?.cumulativeRawText, "A good sprint");
  assert.equal(expanded.nodesById["n2-alt-running"]?.cumulativeRawText, "A good running");
  assert.deepEqual(
    expanded.nodesById.n2.sourceAlternatives.map((alternative) => alternative.nodeId),
    ["n2-alt-sprint", "n2-alt-running"],
  );
  assert.ok(!expanded.nodesById.n1.childIds.includes("n0-alt-good"));
  assert.ok(!expanded.nodesById.n1.childIds.includes("n0-alt-for"));
});

test("keeps repeated token text tied to its own position and parent branch", () => {
  const graph = createGraph();
  const expanded = materializeSourceAlternativesForNode(graph, "n4");

  assert.deepEqual(expanded.nodesById.n3.childIds, ["n4", "n4-alt-race", "n4-alt-track"]);
  assert.equal(expanded.nodesById["n4-alt-race"]?.parentId, "n3");
  assert.equal(expanded.nodesById["n4-alt-race"]?.cumulativeRawText, "A good time for race");
  assert.equal(expanded.nodesById["n4-alt-track"]?.cumulativeRawText, "A good time for track");
  assert.deepEqual(
    expanded.nodesById.n4.sourceAlternatives.map((alternative) => alternative.nodeId),
    ["n4-alt-race", "n4-alt-track"],
  );
  assert.ok(!expanded.nodesById.n3.childIds.includes("n2-alt-sprint"));
  assert.ok(!expanded.nodesById.n3.childIds.includes("n2-alt-running"));
});

test("continuation-created nodes keep their own step-local alternatives", () => {
  const graph = createGraph();
  const continued = applyExpansionToTokenGraph(
    graph,
    "n2",
    makeExpansionResponse(graph, "n2", "2026-08-06T10:00:00Z", [
      {
        id: "n2:3:competitive",
        branchId: "n2:3:competitive",
        rawToken: " competitive",
        generationStep: 3,
        rank: 1,
        probability: 0.61,
      },
      {
        id: "n2:3:sprint",
        branchId: "n2:3:sprint",
        rawToken: " sprint",
        generationStep: 3,
        rank: 2,
        probability: 0.22,
      },
      {
        id: "n2:3:dash",
        branchId: "n2:3:dash",
        rawToken: " dash",
        generationStep: 3,
        rank: 3,
        probability: 0.12,
      },
    ]),
    expansionOptions,
  );

  assert.deepEqual(continued.nodesById.n2.childIds, ["n2:3:competitive", "n2:3:sprint", "n2:3:dash"]);
  assert.deepEqual(
    continued.nodesById["n2:3:competitive"].sourceAlternatives.map((alternative) => alternative.nodeId),
    ["n2:3:sprint", "n2:3:dash"],
  );
  assert.deepEqual(
    continued.nodesById["n2:3:sprint"].sourceAlternatives.map((alternative) => alternative.nodeId),
    ["n2:3:competitive", "n2:3:dash"],
  );
});

test("continued branches stay isolated even when token text repeats", () => {
  const graph = createGraph();
  const firstContinuation = applyExpansionToTokenGraph(
    graph,
    "n2",
    makeExpansionResponse(graph, "n2", "2026-08-06T10:00:00Z", [
      {
        id: "n2:3:competitive",
        branchId: "n2:3:competitive",
        rawToken: " competitive",
        generationStep: 3,
        rank: 1,
        probability: 0.61,
      },
      {
        id: "n2:3:steady",
        branchId: "n2:3:steady",
        rawToken: " steady",
        generationStep: 3,
        rank: 2,
        probability: 0.18,
      },
    ]),
    expansionOptions,
  );
  const secondContinuation = applyExpansionToTokenGraph(
    firstContinuation,
    "n2:3:competitive",
    makeExpansionResponse(firstContinuation, "n2:3:competitive", "2026-08-06T10:00:01Z", [
      {
        id: "n2:3:competitive:4:A",
        branchId: "n2:3:competitive:4:A",
        rawToken: " A",
        generationStep: 4,
        rank: 1,
        probability: 0.57,
      },
      {
        id: "n2:3:competitive:4:runner",
        branchId: "n2:3:competitive:4:runner",
        rawToken: " runner",
        generationStep: 4,
        rank: 2,
        probability: 0.21,
      },
      {
        id: "n2:3:competitive:4:race",
        branchId: "n2:3:competitive:4:race",
        rawToken: " race",
        generationStep: 4,
        rank: 3,
        probability: 0.11,
      },
    ]),
    expansionOptions,
  );
  const materialized = materializeSourceAlternativesForNode(
    secondContinuation,
    "n2:3:competitive:4:A",
  );

  assert.ok(secondContinuation.nodesById.n0);
  assert.ok(secondContinuation.nodesById["n2:3:competitive:4:A"]);
  assert.notEqual(secondContinuation.nodesById.n0.id, secondContinuation.nodesById["n2:3:competitive:4:A"].id);
  assert.deepEqual(
    secondContinuation.nodesById["n2:3:competitive:4:A"].sourceAlternatives.map(
      (alternative) => alternative.nodeId,
    ),
    ["n2:3:competitive:4:runner", "n2:3:competitive:4:race"],
  );
  assert.deepEqual(
    materialized.nodesById["n2:3:competitive"].childIds,
    [
      "n2:3:competitive:4:A",
      "n2:3:competitive:4:runner",
      "n2:3:competitive:4:race",
    ],
  );
  assert.ok(
    !materialized.nodesById["n2:3:competitive"].childIds.includes("n2:3:steady"),
  );
});

test("continuation mode and segment id propagate through expansion records", () => {
  const graph = createGraph();
  const expanded = applyExpansionToTokenGraph(
    graph,
    "n2",
    makeExpansionResponse(graph, "n2", "2026-08-06T10:00:02Z", [
      {
        id: "n2:3:approx",
        branchId: "n2:3:approx",
        segmentId: "segment-approx-1",
        continuationMode: "approximate",
        rawToken: " approx",
        generationStep: 3,
        rank: 1,
        probability: 0.58,
      },
      {
        id: "n2:3:exact-alt",
        branchId: "n2:3:exact-alt",
        segmentId: "segment-approx-1",
        continuationMode: "approximate",
        rawToken: " branch",
        generationStep: 3,
        rank: 2,
        probability: 0.19,
      },
    ]),
    expansionOptions,
  );

  assert.equal(expanded.nodesById["n2:3:approx"].continuationMode, "approximate");
  assert.equal(expanded.nodesById["n2:3:approx"].segmentId, "segment-approx-1");
  assert.equal(
    expanded.nodesById["n2:3:approx"].sourceAlternatives[0]?.continuationMode,
    "approximate",
  );
  assert.equal(
    expanded.generationsById["2026-08-06T10:00:02Z:n2"]?.continuationMode,
    "approximate",
  );
});
