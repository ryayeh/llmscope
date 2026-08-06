import type {
  AlternativeCandidate,
  GenerationResponse,
  NodeExpansionCandidate,
  NodeExpansionResponse,
  TokenTrace,
} from "@/types/api";

export interface TokenGraphAlternativeRecord {
  nodeId: string | null;
  branchId: string | null;
  rawToken: string;
  displayToken: string;
  decodedContribution: string;
  cumulativeDecodedText: string;
  cumulativeRawText: string;
  cumulativeTokenIds: number[] | null;
  cumulativeLogProbability: number | null;
  generationStep: number | null;
  tokenBytes: number[];
  tokenId: number | null;
  tokenizerId: number | null;
  probability: number;
  rawProbability: number;
  normalizedDisplayedProbability: number | null;
  logProbability: number | null;
  entropy: number | null;
  latencyMs: number | null;
  rank: number | null;
  contextBefore: string;
  contextAfter: string;
  finishReason: string | null;
  rationale: string | null;
  metadata: Record<string, string | number | boolean | null>;
}

export interface TokenGraphNodeRecord {
  id: string;
  kind: "prompt" | "token";
  parentId: string | null;
  childIds: string[];
  generationId: string;
  branchId: string;
  rawToken: string;
  displayToken: string;
  decodedContribution: string;
  cumulativeDecodedText: string;
  cumulativeRawText: string;
  cumulativeTokenIds: number[] | null;
  cumulativeLogProbability: number;
  tokenBytes: number[];
  tokenId: number | null;
  tokenizerId: number | null;
  logProbability: number;
  probability: number;
  rawProbability: number;
  normalizedDisplayedProbability: number;
  rank: number;
  entropy: number;
  latencyMs: number;
  cumulativeProbability: number;
  branchProbability: number;
  finishReason: string | null;
  generationDepth: number;
  generationStep: number;
  contextBefore: string;
  contextAfter: string;
  requestPrompt: string;
  requestModel: string;
  requestPreset: string;
  requestTemperature: number;
  requestTopP: number;
  requestVariation: number;
  requestDemoMode: boolean;
  responseMode: string;
  sourceNotes: string;
  reasoningIntent: string;
  reasoningStrategy: string;
  reasoningFocusTerms: string[];
  branchRationale: string | null;
  metadata: Record<string, string | number | boolean | null>;
  sourceAlternatives: TokenGraphAlternativeRecord[];
  distributionRequested: boolean;
  distributionMessage: string | null;
}

export interface TokenGraphGenerationRecord {
  id: string;
  parentGenerationId: string | null;
  parentNodeId: string | null;
  requestPrompt: string;
  assistantPrefix: string;
  model: string;
  preset: string;
  temperature: number;
  topP: number;
  variation: number;
  demoMode: boolean;
  seed: number | null;
  timestamp: string;
  mode: string;
  source: "initial" | "continuation";
  nodeIds: string[];
  finishReason: string | null;
}

export interface TokenGraphState {
  rootNodeId: string | null;
  rootPrompt: string;
  nodesById: Record<string, TokenGraphNodeRecord>;
  generationOrder: string[];
  generationsById: Record<string, TokenGraphGenerationRecord>;
}

export interface ContinuationValidation {
  isValid: boolean;
  warnings: string[];
  lineageNodeIds: string[];
  lineageRawTokens: string[];
  rootPrompt: string;
  assistantPrefix: string;
  cumulativeRawText: string;
  backendRequestText: string;
  reconstructedPrompt: string;
  expectedAssistantPrefix: string | null;
  expectedPromptLength: number;
  characterLength: number;
  utf8Length: number;
  tokenCount: number;
  cumulativeTokenIds: number[] | null;
  generationStep: number;
}

export interface TokenGraphNodeMetrics {
  tokenBytes: number[];
  utf8Length: number;
  characterLength: number;
  leadingWhitespaceCount: number;
  trailingWhitespaceCount: number;
}

export interface ApplyExpansionOptions {
  requestPrompt: string;
  model: string;
  preset: string;
  temperature: number;
  topP: number;
  variation: number;
  demoMode: boolean;
}

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
const MIN_GRAPH_PROBABILITY = 0.000001;

function encodeTokenId(timestamp: string, suffix: string) {
  return `${timestamp}:${suffix}`;
}

function normalizeTokenBytes(rawToken: string, tokenBytes?: number[] | null) {
  if (tokenBytes && tokenBytes.length > 0) {
    return [...tokenBytes];
  }

  return Array.from(textEncoder.encode(rawToken));
}

function normalizeDecodedContribution(rawToken: string, tokenBytes?: number[] | null) {
  const normalizedTokenBytes = normalizeTokenBytes(rawToken, tokenBytes);

  try {
    const decoded = textDecoder.decode(Uint8Array.from(normalizedTokenBytes));
    return decoded.length > 0 ? decoded : rawToken;
  } catch {
    return rawToken;
  }
}

function normalizeDisplayToken(rawToken: string, displayToken?: string | null) {
  if (displayToken && displayToken.length > 0) {
    return displayToken;
  }

  return rawToken
    .replace(/\t/g, "⇥")
    .replace(/\n/g, "↵\n")
    .replace(/^ /, "␠");
}

function buildCanonicalDisplayToken(
  decodedContribution: string,
  displayToken?: string | null,
) {
  if (displayToken && displayToken.length > 0) {
    return displayToken;
  }

  return decodedContribution
    .replace(/\t/g, "\u21E5")
    .replace(/\n/g, "\u21B5\n")
    .replace(/^ /, "\u2420");
}

function appendTokenIdHistory(parentTokenIds: number[] | null, tokenId: number | null) {
  if (parentTokenIds === null || tokenId === null) {
    return null;
  }

  return [...parentTokenIds, tokenId];
}

function buildCanonicalDecodedContribution(rawToken: string, tokenBytes?: number[] | null) {
  const normalizedTokenBytes = normalizeTokenBytes(rawToken, tokenBytes);

  try {
    const decoded = textDecoder.decode(Uint8Array.from(normalizedTokenBytes));
    return decoded.length > 0 ? decoded : rawToken;
  } catch {
    return rawToken;
  }
}

void normalizeDecodedContribution;
void normalizeDisplayToken;

function buildCanonicalCumulativeDecodedText(
  parentNode: TokenGraphNodeRecord,
  decodedContribution: string,
  cumulativeDecodedText?: string | null,
) {
  return cumulativeDecodedText ?? `${parentNode.cumulativeDecodedText}${decodedContribution}`;
}

function buildCanonicalCumulativeRawText(
  parentNode: TokenGraphNodeRecord,
  rawToken: string,
) {
  return `${parentNode.cumulativeRawText}${rawToken}`;
}

function resolveCumulativeLogProbability(
  parentNode: TokenGraphNodeRecord,
  logProbability: number | null,
  cumulativeLogProbability?: number | null,
) {
  if (typeof cumulativeLogProbability === "number") {
    return cumulativeLogProbability;
  }

  return parentNode.cumulativeLogProbability + (logProbability ?? 0);
}

function normalizeMetadata(
  metadata?: Record<string, string | number | boolean | null> | null,
) {
  return metadata ? { ...metadata } : {};
}

function mapAlternativeCandidate(
  candidate: AlternativeCandidate,
  parentNode: TokenGraphNodeRecord,
): TokenGraphAlternativeRecord {
  const tokenBytes = normalizeTokenBytes(candidate.token, candidate.token_bytes ?? null);
  const decodedContribution =
    candidate.decoded_contribution ??
    buildCanonicalDecodedContribution(candidate.token, tokenBytes);
  const rawProbability = candidate.raw_probability ?? candidate.probability;
  const logProbability =
    candidate.log_probability ?? Math.log(Math.max(rawProbability, MIN_GRAPH_PROBABILITY));
  const cumulativeDecodedText = buildCanonicalCumulativeDecodedText(
    parentNode,
    decodedContribution,
    candidate.cumulative_decoded_text ?? candidate.context_after ?? candidate.text_preview ?? null,
  );
  const cumulativeRawText = buildCanonicalCumulativeRawText(parentNode, candidate.token);
  const contextBefore = candidate.context_before ?? parentNode.cumulativeRawText;

  return {
    nodeId: candidate.node_id ?? null,
    branchId:
      typeof candidate.metadata?.branch_id === "string" ? candidate.metadata.branch_id : null,
    rawToken: candidate.token,
    displayToken: buildCanonicalDisplayToken(decodedContribution, candidate.display_token ?? null),
    decodedContribution,
    cumulativeDecodedText,
    cumulativeRawText,
    cumulativeTokenIds:
      candidate.cumulative_token_ids ??
      appendTokenIdHistory(parentNode.cumulativeTokenIds, candidate.token_id ?? null),
    cumulativeLogProbability: resolveCumulativeLogProbability(
      parentNode,
      logProbability,
      candidate.cumulative_log_probability ?? null,
    ),
    generationStep: candidate.generation_step ?? null,
    tokenBytes,
    tokenId: candidate.token_id ?? null,
    tokenizerId: candidate.tokenizer_id ?? null,
    probability: candidate.probability,
    rawProbability,
    normalizedDisplayedProbability:
      candidate.normalized_displayed_probability ?? null,
    logProbability,
    entropy: candidate.entropy ?? null,
    latencyMs: candidate.latency_ms ?? null,
    rank: candidate.rank ?? null,
    contextBefore,
    contextAfter: cumulativeRawText,
    finishReason: candidate.finish_reason ?? null,
    rationale: candidate.rationale ?? null,
    metadata: normalizeMetadata(candidate.metadata ?? null),
  };
}

function buildInitialTokenNode(
  trace: TokenTrace,
  parentNode: TokenGraphNodeRecord,
  generationId: string,
  payload: GenerationResponse,
): TokenGraphNodeRecord {
  const tokenBytes = normalizeTokenBytes(trace.token, trace.token_bytes ?? null);
  const decodedContribution =
    trace.decoded_contribution ??
    buildCanonicalDecodedContribution(trace.token, tokenBytes);
  const logProbability = trace.log_probability;
  const cumulativeDecodedText = buildCanonicalCumulativeDecodedText(
    parentNode,
    decodedContribution,
    trace.cumulative_decoded_text ?? trace.context_after ?? null,
  );
  const cumulativeRawText = buildCanonicalCumulativeRawText(parentNode, trace.token);

  return {
    id: trace.id,
    kind: "token",
    parentId: parentNode.id,
    childIds: [],
    generationId,
    branchId: trace.branch_id,
    rawToken: trace.token,
    displayToken: buildCanonicalDisplayToken(decodedContribution, trace.display_token),
    decodedContribution,
    cumulativeDecodedText,
    cumulativeRawText,
    cumulativeTokenIds:
      trace.cumulative_token_ids ??
      appendTokenIdHistory(parentNode.cumulativeTokenIds, trace.token_id ?? null),
    cumulativeLogProbability: resolveCumulativeLogProbability(
      parentNode,
      logProbability,
      trace.cumulative_log_probability ?? null,
    ),
    tokenBytes,
    tokenId: trace.token_id ?? null,
    tokenizerId: trace.tokenizer_id ?? null,
    logProbability,
    probability: trace.probability,
    rawProbability: trace.raw_probability,
    normalizedDisplayedProbability: trace.normalized_displayed_probability,
    rank: 1,
    entropy: trace.entropy,
    latencyMs: trace.latency_ms,
    cumulativeProbability: trace.cumulative_probability,
    branchProbability: trace.raw_probability,
    finishReason: trace.finish_reason ?? null,
    generationDepth: trace.position + 1,
    generationStep: trace.generation_step ?? trace.position,
    contextBefore: parentNode.cumulativeRawText,
    contextAfter: cumulativeRawText,
    requestPrompt: payload.prompt_used,
    requestModel: payload.request.model,
    requestPreset: payload.request.preset,
    requestTemperature: payload.request.temperature,
    requestTopP: payload.request.top_p,
    requestVariation: payload.request.variation,
    requestDemoMode: payload.request.demo_mode,
    responseMode: payload.mode,
    sourceNotes: payload.notes,
    reasoningIntent: payload.insights.detected_intent,
    reasoningStrategy: payload.insights.response_strategy,
    reasoningFocusTerms: payload.insights.focus_terms,
    branchRationale: null,
    metadata: normalizeMetadata(trace.metadata ?? null),
    sourceAlternatives: trace.alternatives.map((candidate) =>
      mapAlternativeCandidate(candidate, parentNode),
    ),
    distributionRequested: false,
    distributionMessage: null,
  };
}

function buildExpansionNode(
  candidate: NodeExpansionCandidate,
  parentNode: TokenGraphNodeRecord,
  generationId: string,
  payload: NodeExpansionResponse,
  options: ApplyExpansionOptions,
): TokenGraphNodeRecord {
  const tokenBytes = normalizeTokenBytes(candidate.token, candidate.token_bytes ?? null);
  const decodedContribution =
    candidate.decoded_contribution ??
    buildCanonicalDecodedContribution(candidate.token, tokenBytes);
  const logProbability = candidate.log_probability;
  const cumulativeDecodedText = buildCanonicalCumulativeDecodedText(
    parentNode,
    decodedContribution,
    candidate.cumulative_decoded_text ?? candidate.context_after ?? null,
  );
  const cumulativeRawText = buildCanonicalCumulativeRawText(parentNode, candidate.token);

  return {
    id: candidate.id,
    kind: "token",
    parentId: parentNode.id,
    childIds: [],
    generationId,
    branchId: candidate.branch_id,
    rawToken: candidate.token,
    displayToken: buildCanonicalDisplayToken(decodedContribution, candidate.display_token),
    decodedContribution,
    cumulativeDecodedText,
    cumulativeRawText,
    cumulativeTokenIds:
      candidate.cumulative_token_ids ??
      appendTokenIdHistory(parentNode.cumulativeTokenIds, candidate.token_id ?? null),
    cumulativeLogProbability: resolveCumulativeLogProbability(
      parentNode,
      logProbability,
      candidate.cumulative_log_probability ?? null,
    ),
    tokenBytes,
    tokenId: candidate.token_id ?? null,
    tokenizerId: candidate.tokenizer_id ?? null,
    logProbability,
    probability: candidate.probability,
    rawProbability: candidate.raw_probability,
    normalizedDisplayedProbability: candidate.normalized_displayed_probability,
    rank: candidate.rank,
    entropy: candidate.entropy,
    latencyMs: candidate.latency_ms,
    cumulativeProbability: candidate.cumulative_probability,
    branchProbability: candidate.raw_probability,
    finishReason: candidate.finish_reason ?? null,
    generationDepth: candidate.depth,
    generationStep: candidate.generation_step ?? Math.max(candidate.depth - 1, 0),
    contextBefore: parentNode.cumulativeRawText,
    contextAfter: cumulativeRawText,
    requestPrompt: options.requestPrompt,
    requestModel: options.model,
    requestPreset: options.preset,
    requestTemperature: options.temperature,
    requestTopP: options.topP,
    requestVariation: options.variation,
    requestDemoMode: options.demoMode,
    responseMode: payload.mode,
    sourceNotes: payload.notes,
    reasoningIntent: parentNode.reasoningIntent,
    reasoningStrategy: parentNode.reasoningStrategy,
    reasoningFocusTerms: parentNode.reasoningFocusTerms,
    branchRationale: candidate.rationale ?? null,
    metadata: normalizeMetadata(candidate.metadata ?? null),
    sourceAlternatives: [],
    distributionRequested: false,
    distributionMessage: null,
  };
}

export function createEmptyTokenGraph(): TokenGraphState {
  return {
    rootNodeId: null,
    rootPrompt: "",
    nodesById: {},
    generationOrder: [],
    generationsById: {},
  };
}

export function createTokenGraphFromGeneration(payload: GenerationResponse): TokenGraphState {
  const generationId = encodeTokenId(payload.stats.generated_at, "initial");
  const rootNode: TokenGraphNodeRecord = {
    id: "root",
    kind: "prompt",
    parentId: null,
    childIds: [],
    generationId,
    branchId: "root",
    rawToken: payload.prompt_used,
    displayToken: payload.prompt_used,
    decodedContribution: "",
    cumulativeDecodedText: "",
    cumulativeRawText: "",
    cumulativeTokenIds: [],
    cumulativeLogProbability: 0,
    tokenBytes: normalizeTokenBytes(payload.prompt_used),
    tokenId: null,
    tokenizerId: null,
    logProbability: 0,
    probability: 1,
    rawProbability: 1,
    normalizedDisplayedProbability: 1,
    rank: 1,
    entropy: 0,
    latencyMs: 0,
    cumulativeProbability: 1,
    branchProbability: 1,
    finishReason: null,
    generationDepth: 0,
    generationStep: -1,
    contextBefore: "",
    contextAfter: "",
    requestPrompt: payload.prompt_used,
    requestModel: payload.request.model,
    requestPreset: payload.request.preset,
    requestTemperature: payload.request.temperature,
    requestTopP: payload.request.top_p,
    requestVariation: payload.request.variation,
    requestDemoMode: payload.request.demo_mode,
    responseMode: payload.mode,
    sourceNotes: payload.notes,
    reasoningIntent: payload.insights.detected_intent,
    reasoningStrategy: payload.insights.response_strategy,
    reasoningFocusTerms: payload.insights.focus_terms,
    branchRationale: null,
    metadata: {
      provider: payload.stats.provider,
      source: payload.mode,
      branch_id: "root",
    },
    sourceAlternatives: [],
    distributionRequested: false,
    distributionMessage: null,
  };
  const nodesById: Record<string, TokenGraphNodeRecord> = {
    [rootNode.id]: rootNode,
  };
  const nodeIds: string[] = [];
  let parentNode = rootNode;

  for (const trace of payload.tokens) {
    const node = buildInitialTokenNode(trace, parentNode, generationId, payload);
    nodesById[node.id] = node;
    nodesById[parentNode.id] = {
      ...nodesById[parentNode.id],
      childIds: [...nodesById[parentNode.id].childIds, node.id],
    };
    nodeIds.push(node.id);
    parentNode = node;
  }

  const assistantPrefix =
    nodeIds.length > 0
      ? nodesById[nodeIds[nodeIds.length - 1]]?.cumulativeDecodedText ?? payload.completion
      : payload.completion;

  return {
    rootNodeId: rootNode.id,
    rootPrompt: payload.prompt_used,
    nodesById,
    generationOrder: [generationId],
    generationsById: {
      [generationId]: {
        id: generationId,
        parentGenerationId: null,
        parentNodeId: rootNode.id,
        requestPrompt: payload.prompt_used,
        assistantPrefix,
        model: payload.request.model,
        preset: payload.request.preset,
        temperature: payload.request.temperature,
        topP: payload.request.top_p,
        variation: payload.request.variation,
        demoMode: payload.request.demo_mode,
        seed: null,
        timestamp: payload.stats.generated_at,
        mode: payload.mode,
        source: "initial",
        nodeIds,
        finishReason:
          nodeIds.length > 0 ? nodesById[nodeIds[nodeIds.length - 1]].finishReason : null,
      },
    },
  };
}

export function applyExpansionToTokenGraph(
  graph: TokenGraphState,
  parentNodeId: string,
  payload: NodeExpansionResponse,
  options: ApplyExpansionOptions,
): TokenGraphState {
  const parentNode = graph.nodesById[parentNodeId];

  if (!parentNode) {
    return graph;
  }

  const generationId = encodeTokenId(payload.expanded_at, parentNodeId);
  const nodesById: Record<string, TokenGraphNodeRecord> = {
    ...graph.nodesById,
  };
  const parentGenerationId = parentNode.generationId;
  const addedNodeIds: string[] = [];
  const nextChildIds = payload.children.map((candidate) => candidate.id);

  nodesById[parentNodeId] = {
    ...parentNode,
    childIds: nextChildIds,
    distributionRequested: true,
    distributionMessage: null,
  };

  for (const candidate of payload.children) {
    const existing = nodesById[candidate.id];
    const nextNode = buildExpansionNode(candidate, nodesById[parentNodeId], generationId, payload, options);

    if (existing) {
      nodesById[candidate.id] = {
        ...existing,
        ...nextNode,
        childIds: existing.childIds,
        generationId: existing.generationId,
      };
    } else {
      nodesById[candidate.id] = nextNode;
      addedNodeIds.push(candidate.id);
    }
  }

  return {
    ...graph,
    nodesById,
    generationOrder: [...graph.generationOrder, generationId],
    generationsById: {
      ...graph.generationsById,
      [generationId]: {
        id: generationId,
        parentGenerationId,
        parentNodeId,
        requestPrompt: options.requestPrompt,
        assistantPrefix: reconstructAssistantPrefix({ ...graph, nodesById }, parentNodeId),
        model: options.model,
        preset: options.preset,
        temperature: options.temperature,
        topP: options.topP,
        variation: options.variation,
        demoMode: options.demoMode,
        seed: null,
        timestamp: payload.expanded_at,
        mode: payload.mode,
        source: "continuation",
        nodeIds: addedNodeIds,
        finishReason:
          payload.children[0]?.finish_reason ??
          nodesById[payload.children[0]?.id ?? ""]?.finishReason ??
          null,
      },
    },
  };
}

export function markTokenGraphNodeDistributionMessage(
  graph: TokenGraphState,
  nodeId: string,
  message: string,
): TokenGraphState {
  const node = graph.nodesById[nodeId];

  if (!node) {
    return graph;
  }

  return {
    ...graph,
    nodesById: {
      ...graph.nodesById,
      [nodeId]: {
        ...node,
        distributionRequested: true,
        sourceAlternatives: [],
        distributionMessage: message,
      },
    },
  };
}

export function clearTokenGraph(): TokenGraphState {
  return createEmptyTokenGraph();
}

export function getTokenGraphNode(graph: TokenGraphState, nodeId: string | null) {
  if (!nodeId) {
    return null;
  }

  return graph.nodesById[nodeId] ?? null;
}

export function getLineageNodeIds(graph: TokenGraphState, nodeId: string): string[] {
  const lineage: string[] = [];
  let currentId: string | null = nodeId;

  while (currentId) {
    const currentNode: TokenGraphNodeRecord | undefined = graph.nodesById[currentId];

    if (!currentNode) {
      break;
    }

    lineage.unshift(currentId);
    currentId = currentNode.parentId;
  }

  return lineage;
}

export function reconstructRawPath(graph: TokenGraphState, nodeId: string) {
  const node = graph.nodesById[nodeId];
  if (!node) {
    return graph.rootPrompt;
  }

  return `${graph.rootPrompt}${node.kind === "token" ? node.cumulativeDecodedText : ""}`;
}

export function reconstructAssistantPrefix(graph: TokenGraphState, nodeId: string) {
  const node = graph.nodesById[nodeId];
  if (!node || node.kind !== "token") {
    return "";
  }

  return node.cumulativeDecodedText;
}

export function buildContinuationValidation(
  graph: TokenGraphState,
  nodeId: string,
): ContinuationValidation {
  const node = graph.nodesById[nodeId];
  const warnings: string[] = [];

  if (!node) {
    return {
      isValid: false,
      warnings: ["The selected node does not exist in the token graph."],
      lineageNodeIds: [],
      lineageRawTokens: [],
      rootPrompt: graph.rootPrompt,
      assistantPrefix: "",
      cumulativeRawText: "",
      backendRequestText: "",
      reconstructedPrompt: graph.rootPrompt,
      expectedAssistantPrefix: null,
      expectedPromptLength: graph.rootPrompt.length,
      characterLength: graph.rootPrompt.length,
      utf8Length: textEncoder.encode(graph.rootPrompt).length,
      tokenCount: 0,
      cumulativeTokenIds: null,
      generationStep: -1,
    };
  }

  const lineage = getLineageNodeIds(graph, nodeId);
  const rootNode = graph.rootNodeId ? graph.nodesById[graph.rootNodeId] ?? null : null;
  const rootPrompt = rootNode?.rawToken ?? graph.rootPrompt;
  const lineageTokens = lineage
    .slice(1)
    .map((currentId) => graph.nodesById[currentId]?.decodedContribution ?? "");
  const lineagePrefix = lineageTokens.join("");
  const assistantPrefix = node.kind === "token" ? node.cumulativeDecodedText : "";
  const backendRequestText = assistantPrefix;
  const reconstructedPrompt = `${rootPrompt}${assistantPrefix}`;
  const expectedAssistantPrefix = node.kind === "token" ? node.cumulativeDecodedText : "";

  if (!rootNode) {
    warnings.push("The root prompt node is missing.");
  }

  if (lineage[0] !== graph.rootNodeId) {
    warnings.push("The selected node is not connected to the root prompt.");
  }

  if (rootPrompt !== graph.rootPrompt) {
    warnings.push("The reconstructed root prompt no longer matches the original prompt.");
  }

  if (lineagePrefix !== node.cumulativeDecodedText) {
    warnings.push("The cached cumulative decoded text no longer matches the decoded node lineage.");
  }

  if (node.kind === "token" && node.parentId) {
    const expectedContextBefore =
      graph.nodesById[node.parentId]?.kind === "token"
        ? graph.nodesById[node.parentId]?.cumulativeDecodedText ?? ""
        : "";

    if (node.contextBefore !== expectedContextBefore) {
      warnings.push("The stored prefix before this token no longer matches the graph lineage.");
    }
  }

  if (node.kind === "token" && node.contextAfter !== expectedAssistantPrefix) {
    warnings.push("The stored context through this token no longer matches the canonical cumulative raw text.");
  }

  if (assistantPrefix.length !== expectedAssistantPrefix.length) {
    warnings.push("The reconstructed token length does not match the original continuation length.");
  }

  if (node.kind === "token" && node.generationDepth !== Math.max(lineage.length - 1, 0)) {
    warnings.push("The stored token depth does not match the decoded token lineage depth.");
  }

  if (node.kind === "token" && node.generationStep !== Math.max(lineage.length - 2, 0)) {
    warnings.push("The stored generation step does not match the decoded lineage position.");
  }

  const utf8Length = textEncoder.encode(reconstructedPrompt).length;
  return {
    isValid: warnings.length === 0,
    warnings,
    lineageNodeIds: lineage,
    lineageRawTokens: lineageTokens,
    rootPrompt,
    assistantPrefix,
    cumulativeRawText: assistantPrefix,
    backendRequestText,
    reconstructedPrompt,
    expectedAssistantPrefix,
    expectedPromptLength: rootPrompt.length + expectedAssistantPrefix.length,
    characterLength: reconstructedPrompt.length,
    utf8Length,
    tokenCount: Math.max(lineage.length - 1, 0),
    cumulativeTokenIds: node.kind === "token" ? node.cumulativeTokenIds : [],
    generationStep: node.kind === "token" ? node.generationStep : -1,
  };
}

export function getNodeMetrics(rawToken: string, tokenBytes?: number[] | null): TokenGraphNodeMetrics {
  const normalizedTokenBytes = normalizeTokenBytes(rawToken, tokenBytes);
  const leadingWhitespaceCount = (rawToken.match(/^\s+/)?.[0].length ?? 0);
  const trailingWhitespaceCount = (rawToken.match(/\s+$/)?.[0].length ?? 0);

  return {
    tokenBytes: normalizedTokenBytes,
    utf8Length: normalizedTokenBytes.length,
    characterLength: rawToken.length,
    leadingWhitespaceCount,
    trailingWhitespaceCount,
  };
}

export function getGenerationLineage(graph: TokenGraphState, generationId: string | null) {
  const lineage: TokenGraphGenerationRecord[] = [];
  let currentId = generationId;

  while (currentId) {
    const current = graph.generationsById[currentId];

    if (!current) {
      break;
    }

    lineage.unshift(current);
    currentId = current.parentGenerationId;
  }

  return lineage;
}
