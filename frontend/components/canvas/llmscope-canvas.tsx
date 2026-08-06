"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
  getBezierPath,
  type EdgeChange,
  type NodeChange,
  type NodeMouseHandler,
  type ReactFlowInstance,
} from "@xyflow/react";
import {
  Camera,
  ChevronLeft,
  ChevronRight,
  LoaderCircle,
  Pause,
  Play,
  Redo2,
  RefreshCcw,
  Search,
  Sparkles,
  Undo2,
} from "lucide-react";

import {
  CurrentRealityPanel,
  type CurrentRealityStats,
  type CurrentRealityTokenItem,
} from "@/components/canvas/current-reality-panel";
import { ProbabilityEdge } from "@/components/canvas/probability-edge";
import { TokenContextMenu } from "@/components/canvas/token-context-menu";
import { TokenNode } from "@/components/canvas/token-node";
import type {
  InspectorAlternative,
  ProbabilityFlowEdge,
  ProbabilityViewMode,
  TokenFlowNode,
  TokenNodeData,
} from "@/components/canvas/types";
import {
  applyExpansionToTokenGraph,
  buildContinuationValidation,
  clearTokenGraph,
  createEmptyTokenGraph,
  createTokenGraphFromGeneration,
  getGenerationLineage,
  getNodeMetrics,
  markTokenGraphNodeDistributionMessage,
  materializeSourceAlternativesForNode,
  reconstructAssistantPrefix,
  type TokenGraphAlternativeRecord,
  type TokenGraphNodeRecord,
  type TokenGraphState,
} from "@/lib/token-graph";
import type {
  AlternativeCandidate,
  ContinueGenerationResponse,
  GenerationResponse,
  ModelCatalogResponse,
  ModelOption,
  NodeExpansionResponse,
  PresetOption,
} from "@/types/api";

const FALLBACK_MODEL_CATALOG: ModelCatalogResponse = {
  default_model: "gpt-4.1-mini",
  default_preset: "general",
  models: [
    {
      id: "gpt-4o-mini",
      label: "GPT-4o mini",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
    },
    {
      id: "gpt-4.1-mini",
      label: "GPT-4.1 mini",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
    },
    {
      id: "gpt-4o",
      label: "GPT-4o",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
    },
    {
      id: "gpt-4.1",
      label: "GPT-4.1",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
    },
  ],
  presets: [
    { id: "general", label: "General" },
    { id: "reasoning", label: "Reasoning" },
    { id: "coding", label: "Code" },
    { id: "coach", label: "Coach" },
  ],
};

const INITIAL_PROMPT = "What is a good time for a 16 year old kid in the 400m?";
const INITIAL_TEMPERATURE = 0.7;
const INITIAL_TOP_P = 1;
const INITIAL_MAX_TOKENS = 256;
const HORIZONTAL_GAP = 320;
const VERTICAL_GAP = 168;
const MAX_BRANCH_CHILDREN = 4;
const DEFAULT_PLAYBACK_SPEED = 1;
const FRAME_BUDGET_MS = 16;
const SHOULD_LOG_PERF = process.env.NODE_ENV !== "production";
const SHOULD_LOG_CONTINUATION = process.env.NODE_ENV !== "production";

type BackendState = "checking" | "online" | "offline";
type SurfaceTheme = "midnight" | "graphite";

type DecoratedInspectorAlternative = InspectorAlternative & {
  difference: number;
  displayProbability: number;
  isChosen: boolean;
};

interface ContextMenuState {
  nodeId: string;
  title: string;
  x: number;
  y: number;
}

interface GraphSnapshot {
  branchChoices: Record<string, string>;
  compareLeftId: string | null;
  compareRightId: string | null;
  edges: ProbabilityFlowEdge[];
  nodes: TokenFlowNode[];
  pinnedNodeIds: string[];
  selectedNodeId: string | null;
}

interface ReasoningBundle {
  focusTerms: string[];
  intent: string;
  notes: string;
  strategy: string;
}

interface ParsedApiError {
  code: string | null;
  message: string;
}

interface ContinuationPreviewState {
  nodeId: string;
  steps: number;
  validation: ReturnType<typeof buildContinuationValidation>;
}

interface DragPerformanceStats {
  sampleCount: number;
  totalMs: number;
}

const nodeTypes = {
  tokenCard: TokenNode,
};

const edgeTypes = {
  probabilityEdge: ProbabilityEdge,
};

function logCanvasPerformance(metric: string, payload: Record<string, unknown>) {
  if (!SHOULD_LOG_PERF) {
    return;
  }

  console.debug(`[llmscope-perf] ${metric}`, payload);
}

function logContinuationDebug(metric: string, payload: Record<string, unknown>) {
  if (!SHOULD_LOG_CONTINUATION) {
    return;
  }

  console.debug(`[llmscope-continuation] ${metric}`, payload);
}

function wait(durationMs: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatProbability(value: number) {
  return value.toFixed(4);
}

function formatNumber(value: number) {
  return value.toFixed(4);
}

function formatSignedPercent(value: number) {
  const rounded = (value * 100).toFixed(1);
  return `${value >= 0 ? "+" : ""}${rounded}%`;
}

function readMetadataString(
  metadata: Record<string, string | number | boolean | null> | null | undefined,
  key: string,
) {
  return typeof metadata?.[key] === "string" ? String(metadata[key]) : null;
}

function readMetadataNumber(
  metadata: Record<string, string | number | boolean | null> | null | undefined,
  key: string,
) {
  return typeof metadata?.[key] === "number" ? Number(metadata[key]) : null;
}

function readMetadataBoolean(
  metadata: Record<string, string | number | boolean | null> | null | undefined,
  key: string,
) {
  return typeof metadata?.[key] === "boolean" ? Boolean(metadata[key]) : null;
}

function getContinuationModePresentation(
  metadata: Record<string, string | number | boolean | null> | null | undefined,
) {
  const explicitLabel = readMetadataString(metadata, "continuation_mode_label");
  const explicitMode = readMetadataString(metadata, "continuation_mode");
  const isExact =
    readMetadataBoolean(metadata, "continuation_mode_is_exact") ??
    (explicitMode ? explicitMode !== "approximate" : true);

  return {
    label: explicitLabel ?? (isExact ? "Exact" : "Approximate"),
    mode: explicitMode ?? (isExact ? "cached_exact" : "approximate"),
    title: readMetadataString(metadata, "continuation_mode_tooltip"),
    tone: isExact ? ("exact" as const) : ("approximate" as const),
  };
}

function getProbabilityModeLabel(mode: ProbabilityViewMode) {
  return mode === "normalized" ? "Normalized Top-K" : "Raw";
}

function clampProbability(value: number) {
  return Math.max(0, Math.min(1, value));
}

function getRawProbabilityValue(item: { rawProbability?: number | null; probability: number }) {
  return clampProbability(item.rawProbability ?? item.probability);
}

function normalizeDisplayedProbabilities(probabilities: number[]) {
  const total = probabilities.reduce((sum, value) => sum + Math.max(value, 0), 0);

  if (total <= 0) {
    return probabilities.map(() => 0);
  }

  const scaled = probabilities.map((value) => (Math.max(value, 0) / total) * 1000);
  const floored = scaled.map((value) => Math.floor(value));
  let remainder = 1000 - floored.reduce((sum, value) => sum + value, 0);
  const rankedRemainders = scaled
    .map((value, index) => ({
      fraction: value - Math.floor(value),
      index,
    }))
    .sort((left, right) => right.fraction - left.fraction);

  for (let index = 0; index < rankedRemainders.length && remainder > 0; index += 1) {
    floored[rankedRemainders[index].index] += 1;
    remainder -= 1;
  }

  return floored.map((value) => clampProbability(value / 1000));
}

function buildProbabilityPresentation(
  items: Array<{ rawProbability?: number | null; probability: number }>,
  mode: ProbabilityViewMode,
) {
  const rawProbabilities = items.map(getRawProbabilityValue);
  const coverage = clampProbability(rawProbabilities.reduce((sum, value) => sum + value, 0));
  const displayedProbabilities =
    mode === "normalized" ? normalizeDisplayedProbabilities(rawProbabilities) : rawProbabilities;

  return {
    coverage,
    displayedProbabilities,
    rawProbabilities,
    remainingProbabilityMass: mode === "raw" ? clampProbability(1 - coverage) : 0,
  };
}

function equalNumberLists(left: number[], right: number[]) {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function splitPreview(text: string) {
  return text.split(/\s+/).filter(Boolean);
}

async function parseApiError(response: Response): Promise<ParsedApiError> {
  try {
    const payload = (await response.json()) as
      | { detail?: { code?: string; message?: string } | string | Array<{ msg?: string }> }
      | undefined;

    if (
      payload?.detail &&
      typeof payload.detail === "object" &&
      !Array.isArray(payload.detail)
    ) {
      return {
        code: payload.detail.code ?? null,
        message: payload.detail.message ?? `Request failed with status ${response.status}.`,
      };
    }

    if (typeof payload?.detail === "string") {
      return {
        code: null,
        message: payload.detail,
      };
    }

    if (Array.isArray(payload?.detail)) {
      const messages = payload.detail
        .map((item) => item.msg)
        .filter((message): message is string => Boolean(message));

      if (messages.length > 0) {
        return {
          code: null,
          message: messages.join(", "),
        };
      }
    }
  } catch {
    // Fall through to the generic message below.
  }

  return {
    code: null,
    message: `Request failed with status ${response.status}.`,
  };
}

async function throwApiError(response: Response): Promise<never> {
  const parsed = await parseApiError(response);
  const error = new Error(parsed.message) as Error & { code?: string | null };
  error.code = parsed.code;
  throw error;
}

function getErrorCode(error: unknown) {
  if (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    typeof (error as { code?: unknown }).code === "string"
  ) {
    return (error as { code: string }).code;
  }

  return null;
}

function findModelOption(models: ModelOption[], modelId: string): ModelOption {
  return (
    models.find((item) => item.id === modelId) ?? {
      id: modelId,
      label: modelId,
      provider: "openai",
      group: "Custom",
      status: "ready",
    }
  );
}

function findPresetOption(presets: PresetOption[], presetId: string): PresetOption {
  return (
    presets.find((item) => item.id === presetId) ?? {
      id: presetId,
      label: presetId,
    }
  );
}

function reasoningBundleFromGeneration(payload?: GenerationResponse | null): ReasoningBundle {
  return {
    notes: payload?.notes ?? "",
    intent: payload?.insights.detected_intent ?? "",
    strategy: payload?.insights.response_strategy ?? "",
    focusTerms: payload?.insights.focus_terms ?? [],
  };
}

function mapTraceAlternativesToInspector(alternatives: AlternativeCandidate[]) {
  return alternatives.map<InspectorAlternative>((candidate) => ({
    branchId:
      typeof candidate.metadata?.branch_id === "string" ? candidate.metadata.branch_id : null,
    predictionId: candidate.node_id ?? null,
    tokenIndex: candidate.generation_step ?? null,
    token: candidate.token,
    displayToken: candidate.display_token ?? null,
    tokenBytes: candidate.token_bytes ?? [],
    decodedContribution: candidate.decoded_contribution ?? candidate.token,
    cumulativeDecodedText:
      candidate.cumulative_decoded_text ??
      candidate.context_after ??
      candidate.text_preview ??
      candidate.token,
    cumulativeRawText:
      candidate.context_after ??
      candidate.text_preview ??
      candidate.token,
    cumulativeTokenIds: candidate.cumulative_token_ids ?? null,
    cumulativeLogProbability: candidate.cumulative_log_probability ?? null,
    probability: candidate.probability,
    rawProbability: candidate.raw_probability ?? candidate.probability,
    normalizedDisplayedProbability:
      candidate.normalized_displayed_probability ?? null,
    logProbability: candidate.log_probability ?? null,
    rank: candidate.rank ?? null,
    contextBefore: candidate.context_before ?? null,
    contextAfter: candidate.context_after ?? null,
    finishReason: candidate.finish_reason ?? null,
    rationale: candidate.rationale ?? null,
    tokenId: candidate.token_id ?? null,
    textPreview: candidate.text_preview ?? null,
    nodeId: candidate.node_id ?? null,
    generationStep: candidate.generation_step ?? null,
    metadata: candidate.metadata ?? null,
  }));
}

function mapGraphAlternativeToInspector(
  alternative: TokenGraphAlternativeRecord,
): InspectorAlternative {
  return {
    branchId: alternative.branchId,
    predictionId: alternative.nodeId,
    tokenIndex: alternative.generationStep,
    token: alternative.rawToken,
    displayToken: alternative.displayToken,
    tokenBytes: alternative.tokenBytes,
    decodedContribution: alternative.decodedContribution,
    cumulativeDecodedText: alternative.cumulativeDecodedText,
    cumulativeRawText: alternative.cumulativeRawText,
    cumulativeTokenIds: alternative.cumulativeTokenIds,
    cumulativeLogProbability: alternative.cumulativeLogProbability,
    probability: alternative.probability,
    rawProbability: alternative.rawProbability,
    normalizedDisplayedProbability: alternative.normalizedDisplayedProbability,
    logProbability: alternative.logProbability,
    rank: alternative.rank,
    contextBefore: alternative.contextBefore,
    contextAfter: alternative.contextAfter,
    finishReason: alternative.finishReason,
    rationale: alternative.rationale,
    tokenId: alternative.tokenId,
    textPreview: alternative.contextAfter,
    nodeId: alternative.nodeId,
    generationStep: alternative.generationStep,
    metadata: alternative.metadata,
  };
}

function applyTokenGraphRecordToFlowNode(
  node: TokenFlowNode,
  record: TokenGraphNodeRecord,
): TokenFlowNode {
  const metrics = getNodeMetrics(record.rawToken, record.tokenBytes);

  return {
    ...node,
    data: {
      ...node.data,
      kind: record.kind,
      generationId: record.generationId,
      predictionId: record.id,
      tokenIndex: record.generationStep,
      branchId: record.branchId,
      tokenText: record.rawToken,
      displayTokenText: record.displayToken,
      decodedContribution: record.decodedContribution,
      cumulativeDecodedText: record.cumulativeDecodedText,
      cumulativeRawText: record.cumulativeRawText,
      cumulativeTokenIds: record.cumulativeTokenIds,
      cumulativeLogProbability: record.cumulativeLogProbability,
      tokenBytes: metrics.tokenBytes,
      utf8Length: metrics.utf8Length,
      characterLength: metrics.characterLength,
      leadingWhitespaceCount: metrics.leadingWhitespaceCount,
      trailingWhitespaceCount: metrics.trailingWhitespaceCount,
      probability: record.probability,
      rawProbability: record.rawProbability,
      normalizedDisplayedProbability: record.normalizedDisplayedProbability,
      logProbability: record.logProbability,
      entropy: record.entropy,
      latency: record.latencyMs,
      tokenId: record.tokenId,
      tokenizerId: record.tokenizerId,
      generationStep: record.generationStep,
      textPreview: record.contextAfter,
      contextBefore: record.contextBefore,
      contextAfter: record.contextAfter,
      cumulativeProbability: record.cumulativeProbability,
      branchProbability: record.branchProbability,
      depth: record.generationDepth,
      rank: record.rank,
      finishReason: record.finishReason,
      parentId: record.parentId,
      childCount: record.childIds.length,
      requestPrompt: record.requestPrompt,
      requestModel: record.requestModel,
      requestPreset: record.requestPreset,
      requestTemperature: record.requestTemperature,
      requestTopP: record.requestTopP,
      requestVariation: record.requestVariation,
      requestDemoMode: record.requestDemoMode,
      responseMode: record.responseMode,
      sourceNotes: record.sourceNotes,
      reasoningIntent: record.reasoningIntent,
      reasoningStrategy: record.reasoningStrategy,
      reasoningFocusTerms: record.reasoningFocusTerms,
      branchRationale: record.branchRationale,
      metadata: record.metadata,
      topAlternatives: record.sourceAlternatives.map(mapGraphAlternativeToInspector),
      sourceAlternatives: record.sourceAlternatives.map(mapGraphAlternativeToInspector),
      alternativesExpanded: record.alternativesExpanded,
      distributionRequested: record.distributionRequested,
      distributionMessage: record.distributionMessage,
    },
  };
}

function syncFlowNodesWithTokenGraph(
  nodes: TokenFlowNode[],
  graph: TokenGraphState,
): TokenFlowNode[] {
  return nodes.map((node) => {
    const record = graph.nodesById[node.id];
    return record ? applyTokenGraphRecordToFlowNode(node, record) : node;
  });
}

function buildPromptNode({
  prompt,
  model,
  preset,
  reasoning,
  responseMode,
  status,
  temperature,
  topP,
  variation,
  demoMode,
}: {
  model: string;
  preset: string;
  prompt: string;
  reasoning: ReasoningBundle;
  responseMode: string;
  status: TokenNodeData["status"];
  temperature: number;
  topP: number;
  variation: number;
  demoMode: boolean;
}): TokenFlowNode {
  const metrics = getNodeMetrics(prompt);
  return {
    id: "root",
    type: "tokenCard",
    position: {
      x: 0,
      y: 0,
    },
    data: {
      kind: "prompt",
      generationId: "root",
      predictionId: "root",
      tokenIndex: -1,
      branchId: "root",
      tokenText: prompt,
      displayTokenText: prompt,
      decodedContribution: "",
      cumulativeDecodedText: "",
      cumulativeRawText: "",
      cumulativeTokenIds: [],
      cumulativeLogProbability: 0,
      tokenBytes: metrics.tokenBytes,
      utf8Length: metrics.utf8Length,
      characterLength: metrics.characterLength,
      leadingWhitespaceCount: metrics.leadingWhitespaceCount,
      trailingWhitespaceCount: metrics.trailingWhitespaceCount,
      probability: 1,
      rawProbability: 1,
      normalizedDisplayedProbability: 1,
      displayProbability: 1,
      probabilityCoverage: 1,
      remainingProbabilityMass: 0,
      probabilityMode: "normalized",
      logProbability: 0,
      entropy: 0,
      latency: 0,
      tokenId: null,
      tokenizerId: null,
      generationStep: -1,
      textPreview: prompt,
      contextBefore: "",
      contextAfter: "",
      cumulativeProbability: 1,
      branchProbability: 1,
      depth: 0,
      rank: 1,
      finishReason: null,
      parentId: null,
      isMainPath: true,
      isCollapsed: false,
      alternativesExpanded: false,
      distributionRequested: false,
      childCount: 0,
      status,
      requestPrompt: prompt,
      requestModel: model,
      requestPreset: preset,
      requestTemperature: temperature,
      requestTopP: topP,
      requestVariation: variation,
      requestDemoMode: demoMode,
      responseMode,
      sourceNotes: reasoning.notes,
      reasoningIntent: reasoning.intent,
      reasoningStrategy: reasoning.strategy,
      reasoningFocusTerms: reasoning.focusTerms,
      branchRationale: null,
      metadata: {},
      rawLogits: null,
      topAlternatives: [],
      sourceAlternatives: [],
      distributionMessage: null,
      isSearchMatch: false,
      isSearchFocused: false,
      isDimmed: false,
      isActiveReality: true,
      isPinned: false,
    },
    draggable: true,
  };
}

function buildTokenNode({
  branchRationale,
  cumulativeProbability,
  depth,
  entropy,
  id,
  isMainPath,
  latency,
  logProbability,
  model,
  parentId,
  position,
  preset,
  probability,
  rawProbability,
  normalizedDisplayedProbability,
  prompt,
  rank,
  rawLogits,
  reasoning,
  responseMode,
  topAlternatives,
  sourceAlternatives,
  status,
  temperature,
  topP,
  textPreview,
  token,
  displayToken,
  decodedContribution,
  cumulativeDecodedText,
  cumulativeRawText,
  cumulativeTokenIds,
  cumulativeLogProbability,
  contextBefore,
  contextAfter,
  generationStep,
  tokenId,
  tokenizerId,
  variation,
  demoMode,
  metadata,
}: {
  branchRationale: string | null;
  cumulativeProbability: number;
  depth: number;
  entropy: number;
  id: string;
  isMainPath: boolean;
  latency: number;
  logProbability: number;
  model: string;
  parentId: string;
  position: { x: number; y: number };
  preset: string;
  probability: number;
  rawProbability: number;
  normalizedDisplayedProbability: number;
  prompt: string;
  rank: number;
  rawLogits: number[] | null;
  reasoning: ReasoningBundle;
  responseMode: string;
  topAlternatives?: InspectorAlternative[];
  sourceAlternatives: InspectorAlternative[];
  status: TokenNodeData["status"];
  temperature: number;
  topP: number;
  textPreview: string;
  token: string;
  displayToken: string;
  decodedContribution?: string;
  cumulativeDecodedText?: string;
  cumulativeRawText?: string;
  cumulativeTokenIds?: number[] | null;
  cumulativeLogProbability?: number;
  contextBefore?: string;
  contextAfter?: string;
  generationStep?: number;
  tokenId: number | null;
  tokenizerId: number | null;
  variation: number;
  demoMode: boolean;
  metadata?: Record<string, string | number | boolean | null>;
}): TokenFlowNode {
  const metrics = getNodeMetrics(token);
  const resolvedContextAfter = contextAfter ?? cumulativeDecodedText ?? textPreview;
  const resolvedCumulativeRawText = cumulativeRawText ?? resolvedContextAfter;
  const resolvedDecodedContribution = decodedContribution ?? token;
  const resolvedContextBefore =
    contextBefore ??
    (parentId === "root"
      ? ""
      : resolvedCumulativeRawText.slice(
          0,
          Math.max(resolvedCumulativeRawText.length - token.length, 0),
        ));
  return {
    id,
    type: "tokenCard",
    position,
    data: {
      kind: "token",
      generationId: `${model}:${variation}:${id}`,
      predictionId: id,
      tokenIndex: generationStep ?? Math.max(depth - 1, 0),
      branchId: String(metadata?.branch_id ?? id),
      tokenText: token,
      displayTokenText: displayToken,
      decodedContribution: resolvedDecodedContribution,
      cumulativeDecodedText: cumulativeDecodedText ?? resolvedContextAfter,
      cumulativeRawText: resolvedCumulativeRawText,
      cumulativeTokenIds: cumulativeTokenIds ?? null,
      cumulativeLogProbability: cumulativeLogProbability ?? logProbability,
      tokenBytes: metrics.tokenBytes,
      utf8Length: metrics.utf8Length,
      characterLength: metrics.characterLength,
      leadingWhitespaceCount: metrics.leadingWhitespaceCount,
      trailingWhitespaceCount: metrics.trailingWhitespaceCount,
      probability,
      rawProbability,
      normalizedDisplayedProbability,
      displayProbability: normalizedDisplayedProbability,
      probabilityCoverage: rawProbability,
      remainingProbabilityMass: Math.max(0, 1 - rawProbability),
      probabilityMode: "normalized",
      logProbability,
      entropy,
      latency,
      tokenId,
      tokenizerId,
      generationStep: generationStep ?? Math.max(depth - 1, 0),
      textPreview,
      contextBefore: resolvedContextBefore,
      contextAfter: resolvedContextAfter,
      cumulativeProbability,
      branchProbability: rawProbability,
      depth,
      rank,
      finishReason: null,
      parentId,
      isMainPath,
      isCollapsed: false,
      alternativesExpanded: false,
      distributionRequested: false,
      childCount: 0,
      status,
      requestPrompt: prompt,
      requestModel: model,
      requestPreset: preset,
      requestTemperature: temperature,
      requestTopP: topP,
      requestVariation: variation,
      requestDemoMode: demoMode,
      responseMode,
      sourceNotes: reasoning.notes,
      reasoningIntent: reasoning.intent,
      reasoningStrategy: reasoning.strategy,
      reasoningFocusTerms: reasoning.focusTerms,
      branchRationale,
      metadata: metadata ?? {},
      rawLogits,
      topAlternatives: topAlternatives ?? sourceAlternatives,
      sourceAlternatives,
      distributionMessage: null,
      isSearchMatch: false,
      isSearchFocused: false,
      isDimmed: false,
      isActiveReality: false,
      isPinned: false,
    },
    draggable: true,
  };
}

function buildEdge(
  source: string,
  target: string,
  probability: number,
  isMainPath: boolean,
): ProbabilityFlowEdge {
  return {
    id: `edge:${source}:${target}`,
    source,
    target,
    type: "probabilityEdge",
    animated: true,
    selectable: false,
    data: {
      probability,
      rawProbability: probability,
      probabilityCoverage: probability,
      remainingProbabilityMass: Math.max(0, 1 - probability),
      probabilityMode: "normalized",
      isMainPath,
      isActiveReality: false,
      isDimmed: false,
      isFocused: false,
    },
  };
}

function buildFlowNodeFromRecord(
  record: TokenGraphNodeRecord,
  parentNode: TokenFlowNode | null,
): TokenFlowNode {
  const reasoning: ReasoningBundle = {
    notes: record.sourceNotes,
    intent: record.reasoningIntent,
    strategy: record.reasoningStrategy,
    focusTerms: record.reasoningFocusTerms,
  };
  const parentPosition = parentNode?.position ?? { x: 0, y: 0 };
  const initialPosition = {
    x: parentPosition.x + HORIZONTAL_GAP,
    y: parentPosition.y + branchOffset(record.rank) * VERTICAL_GAP,
  };

  return buildTokenNode({
    id: record.id,
    token: record.rawToken,
    displayToken: record.displayToken,
    decodedContribution: record.decodedContribution,
    cumulativeDecodedText: record.cumulativeDecodedText,
    cumulativeRawText: record.cumulativeRawText,
    cumulativeTokenIds: record.cumulativeTokenIds,
    cumulativeLogProbability: record.cumulativeLogProbability,
    contextBefore: record.contextBefore,
    contextAfter: record.contextAfter,
    generationStep: record.generationStep,
    tokenId: record.tokenId,
    tokenizerId: record.tokenizerId,
    probability: record.probability,
    rawProbability: record.rawProbability,
    normalizedDisplayedProbability: record.normalizedDisplayedProbability,
    logProbability: record.logProbability,
    entropy: record.entropy,
    cumulativeProbability: record.cumulativeProbability,
    latency: record.latencyMs,
    depth: record.generationDepth,
    rank: record.rank,
    parentId: record.parentId ?? "root",
    position: initialPosition,
    prompt: record.requestPrompt,
    model: record.requestModel,
    preset: record.requestPreset,
    temperature: record.requestTemperature,
    topP: record.requestTopP,
    variation: record.requestVariation,
    demoMode: record.requestDemoMode,
    responseMode: record.responseMode,
    textPreview: record.contextAfter,
    isMainPath: record.rank === 1,
    status: "idle",
    reasoning,
    branchRationale: record.branchRationale,
    metadata: record.metadata,
    rawLogits: null,
    topAlternatives: record.sourceAlternatives.map(mapGraphAlternativeToInspector),
    sourceAlternatives: record.sourceAlternatives.map(mapGraphAlternativeToInspector),
  });
}

function ensureProbabilityEdgeData(data?: ProbabilityFlowEdge["data"]) {
  return {
    probability: data?.probability ?? 0.5,
    rawProbability: data?.rawProbability ?? data?.probability ?? 0.5,
    probabilityCoverage: data?.probabilityCoverage ?? data?.probability ?? 0.5,
    remainingProbabilityMass:
      data?.remainingProbabilityMass ?? Math.max(0, 1 - (data?.probability ?? 0.5)),
    probabilityMode: data?.probabilityMode ?? "normalized",
    isMainPath: data?.isMainPath ?? false,
    isActiveReality: data?.isActiveReality ?? false,
    isDimmed: data?.isDimmed ?? false,
    isFocused: data?.isFocused ?? false,
  };
}

function branchOffset(rank: number) {
  if (rank <= 1) {
    return 0;
  }

  const level = Math.ceil((rank - 1) / 2);
  return (rank % 2 === 0 ? -1 : 1) * level;
}

interface TreeLayoutPlacement {
  centerY: number;
  child: TokenFlowNode;
}

interface TreeLayoutSpan {
  top: number;
  bottom: number;
}

function getPreferredChildId(
  parentId: string,
  children: TokenFlowNode[],
  branchChoices: Record<string, string>,
) {
  const explicitChoice = branchChoices[parentId];

  if (explicitChoice && children.some((child) => child.id === explicitChoice)) {
    return explicitChoice;
  }

  return (
    children.find((child) => child.data.isMainPath)?.id ??
    [...children].sort((left, right) => {
      const probabilityDelta = getRawProbabilityValue(right.data) - getRawProbabilityValue(left.data);

      if (Math.abs(probabilityDelta) > 0.000001) {
        return probabilityDelta;
      }

      return left.data.rank - right.data.rank;
    })[0]?.id ??
    null
  );
}

function buildHorizontalTimelineLayout(
  nodes: TokenFlowNode[],
  edges: ProbabilityFlowEdge[],
  branchChoices: Record<string, string>,
) {
  const visibleNodeMap = new Map(
    nodes.filter((node) => !node.hidden).map((node) => [node.id, node]),
  );
  const childMap = new Map<string, TokenFlowNode[]>();

  for (const edge of edges) {
    if (edge.hidden) {
      continue;
    }

    const childNode = visibleNodeMap.get(edge.target);

    if (!childNode) {
      continue;
    }

    const children = childMap.get(edge.source) ?? [];
    children.push(childNode);
    childMap.set(edge.source, children);
  }

  const orderedChildCache = new Map<
    string,
    {
      centered: TokenFlowNode;
      left: TokenFlowNode[];
      right: TokenFlowNode[];
    }
  >();
  const placementCache = new Map<string, TreeLayoutPlacement[]>();
  const spanCache = new Map<string, TreeLayoutSpan>();

  function getOrderedChildren(nodeId: string) {
    const cached = orderedChildCache.get(nodeId);

    if (cached) {
      return cached;
    }

    const children = [...(childMap.get(nodeId) ?? [])].sort((left, right) => {
      const leftStrength = getRawProbabilityValue(left.data);
      const rightStrength = getRawProbabilityValue(right.data);
      const strengthDelta = rightStrength - leftStrength;

      if (Math.abs(strengthDelta) > 0.000001) {
        return strengthDelta;
      }

      const branchDelta = Math.abs(branchOffset(left.data.rank)) - Math.abs(branchOffset(right.data.rank));

      if (branchDelta !== 0) {
        return branchDelta;
      }

      return left.data.rank - right.data.rank;
    });

    const preferredChildId = getPreferredChildId(nodeId, children, branchChoices) ?? children[0]?.id;
    const centered = children.find((child) => child.id === preferredChildId) ?? children[0];
    const others = children.filter((child) => child.id !== centered.id);
    const left: TokenFlowNode[] = [];
    const right: TokenFlowNode[] = [];

    others.forEach((child, index) => {
      if (index % 2 === 0) {
        left.push(child);
        return;
      }

      right.push(child);
    });

    const ordered = { centered, left, right };
    orderedChildCache.set(nodeId, ordered);
    return ordered;
  }

  function measureSubtree(nodeId: string): TreeLayoutSpan {
    const cached = spanCache.get(nodeId);

    if (cached) {
      return cached;
    }

    const node = visibleNodeMap.get(nodeId);

    if (!node) {
      return {
        top: 0,
        bottom: 0,
      };
    }

    const frame = getNodeFrame(node);
    const children = childMap.get(nodeId) ?? [];

    if (children.length === 0) {
      const span = {
        top: -frame.height / 2,
        bottom: frame.height / 2,
      };
      spanCache.set(nodeId, span);
      return span;
    }

    if (children.length === 1) {
      const childSpan = measureSubtree(children[0].id);
      const span = {
        top: Math.min(-frame.height / 2, childSpan.top),
        bottom: Math.max(frame.height / 2, childSpan.bottom),
      };
      placementCache.set(nodeId, [
        {
          child: children[0],
          centerY: 0,
        },
      ]);
      spanCache.set(nodeId, span);
      return span;
    }

    const ordered = getOrderedChildren(nodeId);
    const centeredSpan = measureSubtree(ordered.centered.id);
    const placements: TreeLayoutPlacement[] = [
      {
        child: ordered.centered,
        centerY: 0,
      },
    ];
    let topBoundary = centeredSpan.top;
    let bottomBoundary = centeredSpan.bottom;

    for (const child of ordered.left) {
      const childSpan = measureSubtree(child.id);
      const centerY = topBoundary - VERTICAL_GAP - childSpan.bottom;

      placements.push({
        child,
        centerY,
      });
      topBoundary = centerY + childSpan.top;
    }

    for (const child of ordered.right) {
      const childSpan = measureSubtree(child.id);
      const centerY = bottomBoundary + VERTICAL_GAP - childSpan.top;

      placements.push({
        child,
        centerY,
      });
      bottomBoundary = centerY + childSpan.bottom;
    }

    placementCache.set(nodeId, placements);
    const span = {
      top: Math.min(-frame.height / 2, topBoundary),
      bottom: Math.max(frame.height / 2, bottomBoundary),
    };
    spanCache.set(nodeId, span);
    return span;
  }

  function getPlacements(nodeId: string) {
    if (!placementCache.has(nodeId)) {
      measureSubtree(nodeId);
    }

    return placementCache.get(nodeId) ?? [];
  }

  const nextPositions = new Map<string, { x: number; y: number }>();

  function assignSubtree(nodeId: string, leftX: number, centerY: number) {
    const node = visibleNodeMap.get(nodeId);

    if (!node) {
      return;
    }

    const frame = getNodeFrame(node);
    nextPositions.set(nodeId, {
      x: leftX,
      y: centerY - frame.height / 2,
    });

    const children = childMap.get(nodeId) ?? [];

    if (children.length === 0) {
      return;
    }

    const nextLeftX = leftX + frame.width + HORIZONTAL_GAP;

    if (children.length === 1) {
      assignSubtree(children[0].id, nextLeftX, centerY);
      return;
    }

    for (const placement of getPlacements(nodeId)) {
      assignSubtree(placement.child.id, nextLeftX, centerY + placement.centerY);
    }
  }

  if (visibleNodeMap.has("root")) {
    assignSubtree("root", 0, 0);
  }

  return nodes.map((node) => {
    const nextPosition = nextPositions.get(node.id);

    if (!nextPosition) {
      return node;
    }

    return {
      ...node,
      position: nextPosition,
    };
  });
}

function normalizeGraph(
  nodes: TokenFlowNode[],
  edges: ProbabilityFlowEdge[],
  branchChoices: Record<string, string>,
  options?: { skipLayout?: boolean },
) {
  const childCounts = new Map<string, number>();

  for (const edge of edges) {
    childCounts.set(edge.source, (childCounts.get(edge.source) ?? 0) + 1);
  }

  const nodeMap = new Map(nodes.map((node) => [node.id, node]));

  const nextNodes = nodes.map((node) => {
    let hidden = false;
    let parentId = node.data.parentId;

    while (parentId) {
      const parent = nodeMap.get(parentId);

      if (!parent) {
        break;
      }

      if (parent.data.isCollapsed) {
        hidden = true;
        break;
      }

      parentId = parent.data.parentId;
    }

    return {
      ...node,
      hidden,
      data: {
        ...node.data,
        childCount: childCounts.get(node.id) ?? 0,
      },
    };
  });

  const hiddenIds = new Set(nextNodes.filter((node) => node.hidden).map((node) => node.id));

  const nextEdges = edges.map((edge) => ({
    ...edge,
    hidden: hiddenIds.has(edge.source) || hiddenIds.has(edge.target),
  }));

  const layoutNodes = options?.skipLayout
    ? nextNodes
    : buildHorizontalTimelineLayout(nextNodes, nextEdges, branchChoices);

  return {
    nodes: layoutNodes,
    edges: nextEdges,
  };
}

function buildNodeProbabilityView(
  nodes: TokenFlowNode[],
  mode: ProbabilityViewMode,
) {
  const groups = new Map<string, TokenFlowNode[]>();

  for (const node of nodes) {
    if (node.data.kind !== "token" || !node.data.parentId) {
      continue;
    }

    const group = groups.get(node.data.parentId) ?? [];
    group.push(node);
    groups.set(node.data.parentId, group);
  }

  const views = new Map<
    string,
    {
      displayProbability: number;
      probabilityCoverage: number;
      remainingProbabilityMass: number;
    }
  >();

  for (const group of groups.values()) {
    const fallbackProbabilityView = buildProbabilityPresentation(
      group.map((groupNode) => ({
        probability: groupNode.data.probability,
        rawProbability: groupNode.data.rawProbability,
      })),
      mode,
    );

    group.forEach((node, index) => {
      const explicitCandidateProbabilities = [
        node.data.rawProbability,
        ...node.data.sourceAlternatives.map(
          (alternative) => alternative.rawProbability ?? alternative.probability,
        ),
      ].filter((probability) => Number.isFinite(probability) && probability >= 0);
      const probabilityCoverage =
        explicitCandidateProbabilities.length > 1
          ? clampProbability(
              explicitCandidateProbabilities.reduce((sum, probability) => sum + probability, 0),
            )
          : fallbackProbabilityView.coverage;
      const normalizedProbability =
        typeof node.data.normalizedDisplayedProbability === "number"
          ? clampProbability(node.data.normalizedDisplayedProbability)
          : fallbackProbabilityView.displayedProbabilities[index] ?? node.data.rawProbability;
      const rawProbability = clampProbability(node.data.rawProbability);

      views.set(node.id, {
        displayProbability: mode === "raw" ? rawProbability : normalizedProbability,
        probabilityCoverage,
        remainingProbabilityMass:
          mode === "raw" ? clampProbability(1 - probabilityCoverage) : 0,
      });
    });
  }

  return views;
}

function getMiniMapColor(node: TokenFlowNode) {
  if (node.data.kind === "prompt" || node.data.isMainPath) {
    return "#38bdf8";
  }

  if (node.data.isSearchFocused) {
    return "#f59e0b";
  }

  if (node.data.displayProbability < 0.36) {
    return "#8b5cf6";
  }

  return "#64748b";
}

function collectLineageIds(nodeId: string, nodes: TokenFlowNode[]) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const lineage: string[] = [];
  let currentId: string | null = nodeId;

  while (currentId) {
    lineage.unshift(currentId);
    const currentNode = nodeMap.get(currentId);
    currentId = currentNode?.data.parentId ?? null;
  }

  return lineage;
}

function collectDescendantIds(nodeId: string, edges: ProbabilityFlowEdge[]) {
  const descendants = new Set<string>();
  const queue = [nodeId];

  while (queue.length > 0) {
    const currentId = queue.shift();

    if (!currentId) {
      continue;
    }

    for (const edge of edges) {
      if (edge.source !== currentId || descendants.has(edge.target)) {
        continue;
      }

      descendants.add(edge.target);
      queue.push(edge.target);
    }
  }

  return descendants;
}

function matchesInspectorAlternativeNode(node: TokenFlowNode, alternative: InspectorAlternative) {
  if (alternative.nodeId && node.id === alternative.nodeId) {
    return true;
  }

  if (alternative.cumulativeTokenIds && node.data.cumulativeTokenIds) {
    return equalNumberLists(alternative.cumulativeTokenIds, node.data.cumulativeTokenIds);
  }

  return (
    node.data.tokenText === alternative.token &&
    node.data.tokenId === (alternative.tokenId ?? null) &&
    node.data.generationStep === (alternative.generationStep ?? node.data.generationStep) &&
    node.data.contextAfter === (alternative.contextAfter ?? alternative.textPreview ?? node.data.contextAfter)
  );
}

function collectHoverContextIds(nodeId: string, nodes: TokenFlowNode[], edges: ProbabilityFlowEdge[]) {
  const lineage = collectLineageIds(nodeId, nodes);
  const descendants = collectDescendantIds(nodeId, edges);
  const related = new Set<string>(lineage);

  for (const descendant of descendants) {
    related.add(descendant);
  }

  return related;
}

function buildRealityChoicesForNode(
  nodeId: string,
  nodes: TokenFlowNode[],
  currentChoices: Record<string, string>,
) {
  const nextChoices = { ...currentChoices };
  const lineageIds = collectLineageIds(nodeId, nodes);

  for (let index = 1; index < lineageIds.length; index += 1) {
    nextChoices[lineageIds[index - 1]] = lineageIds[index];
  }

  return nextChoices;
}

function buildActivePathIds(choices: Record<string, string>, nodes: TokenFlowNode[]) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const pathIds = ["root"];
  let currentId = "root";

  while (choices[currentId] && nodeMap.has(choices[currentId])) {
    const nextId = choices[currentId];
    pathIds.push(nextId);
    currentId = nextId;
  }

  return pathIds;
}

function buildActivePathEdgeIds(pathIds: string[]) {
  const edgeIds = new Set<string>();

  for (let index = 1; index < pathIds.length; index += 1) {
    edgeIds.add(`edge:${pathIds[index - 1]}:${pathIds[index]}`);
  }

  return edgeIds;
}

function sanitizeBranchChoices(choices: Record<string, string>, nodes: TokenFlowNode[]) {
  const validIds = new Set(nodes.map((node) => node.id));
  const nextChoices: Record<string, string> = {};

  for (const [parentId, childId] of Object.entries(choices)) {
    if (validIds.has(parentId) && validIds.has(childId)) {
      nextChoices[parentId] = childId;
    }
  }

  return nextChoices;
}

function buildTransitionSnapshot(
  current: GraphSnapshot,
  next: Partial<GraphSnapshot>,
  normalizedNodes: TokenFlowNode[],
  normalizedEdges: ProbabilityFlowEdge[],
) {
  const selectedNodeId =
    next.selectedNodeId && normalizedNodes.some((node) => node.id === next.selectedNodeId)
      ? next.selectedNodeId
      : current.selectedNodeId && normalizedNodes.some((node) => node.id === current.selectedNodeId)
        ? current.selectedNodeId
        : normalizedNodes[0]?.id ?? null;
  const branchChoices = sanitizeBranchChoices(
    next.branchChoices ?? current.branchChoices,
    normalizedNodes,
  );
  const pinnedNodeIds = (next.pinnedNodeIds ?? current.pinnedNodeIds).filter((nodeId) =>
    normalizedNodes.some((node) => node.id === nodeId),
  );
  const compareLeftId = (next.compareLeftId ?? current.compareLeftId) && normalizedNodes.some((node) => node.id === (next.compareLeftId ?? current.compareLeftId))
    ? (next.compareLeftId ?? current.compareLeftId)
    : null;
  const compareRightId = (next.compareRightId ?? current.compareRightId) && normalizedNodes.some((node) => node.id === (next.compareRightId ?? current.compareRightId))
    ? (next.compareRightId ?? current.compareRightId)
    : null;

  return {
    nodes: normalizedNodes,
    edges: normalizedEdges,
    branchChoices,
    pinnedNodeIds,
    compareLeftId,
    compareRightId,
    selectedNodeId,
  };
}

function buildInspectorAlternatives(
  graph: TokenGraphState,
  selectedNodeId: string | null,
  mode: ProbabilityViewMode,
) {
  if (!selectedNodeId) {
    return {
      coverage: 0,
      items: [] as DecoratedInspectorAlternative[],
      remainingProbabilityMass: 0,
    };
  }

  const node = graph.nodesById[selectedNodeId];

  if (!node || node.kind !== "token" || !node.parentId) {
    return {
      coverage: 0,
      items: [] as DecoratedInspectorAlternative[],
      remainingProbabilityMass: 0,
    };
  }

  const alternatives = [
    {
      branchId: node.branchId,
      predictionId: node.id,
      tokenIndex: node.generationStep,
      token: node.rawToken,
      displayToken: node.displayToken,
      tokenBytes: node.tokenBytes,
      decodedContribution: node.decodedContribution,
      cumulativeDecodedText: node.cumulativeDecodedText,
      cumulativeRawText: node.cumulativeRawText,
      cumulativeTokenIds: node.cumulativeTokenIds,
      cumulativeLogProbability: node.cumulativeLogProbability,
      probability: node.probability,
      rawProbability: node.rawProbability,
      normalizedDisplayedProbability: node.normalizedDisplayedProbability,
      logProbability: node.logProbability,
      rank: node.rank,
      contextBefore: node.contextBefore,
      contextAfter: node.contextAfter,
      finishReason: node.finishReason,
      rationale: node.branchRationale,
      tokenId: node.tokenId,
      textPreview: node.contextAfter,
      nodeId: node.id,
      generationStep: node.generationStep,
      metadata: node.metadata,
    },
    ...node.sourceAlternatives.map(mapGraphAlternativeToInspector),
  ].slice(0, 10);

  if (alternatives.length === 0) {
    return {
      coverage: 0,
      items: [] as DecoratedInspectorAlternative[],
      remainingProbabilityMass: 0,
    };
  }

  const items = alternatives.sort(
    (left, right) => (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER),
  );
  const probabilityView = buildProbabilityPresentation(items, mode);
  const chosenIndex = items.findIndex((item) => item.nodeId === node.id);
  const chosenDisplayProbability =
    probabilityView.displayedProbabilities[chosenIndex >= 0 ? chosenIndex : 0] ?? 0;

  return {
    coverage: probabilityView.coverage,
    remainingProbabilityMass: probabilityView.remainingProbabilityMass,
    items: items.map((item, index) => ({
      ...item,
      displayProbability: probabilityView.displayedProbabilities[index] ?? 0,
      difference:
        chosenDisplayProbability - (probabilityView.displayedProbabilities[index] ?? 0),
      isChosen: item.nodeId === node.id,
    })),
  };
}

function buildNaturalLanguageReason(
  node: TokenFlowNode | null,
  inspectorAlternatives: DecoratedInspectorAlternative[],
  mode: ProbabilityViewMode,
) {
  if (!node) {
    return "Select a token to inspect the returned distribution for that context.";
  }

  if (node.data.kind === "prompt") {
    return "The prompt is the root context. Expanding it requests the next-token distribution from that point.";
  }

  const chosenContinuation =
    inspectorAlternatives.find((alternative) => alternative.isChosen) ?? null;

  const strongestAlternative =
    inspectorAlternatives.find((alternative) => !alternative.isChosen) ?? null;

  if (chosenContinuation && strongestAlternative) {
    const comparedProbability = strongestAlternative.displayProbability;
    const contextLabel =
      mode === "normalized" ? "among the displayed top-k candidates" : "among the returned candidates";

    if (chosenContinuation.displayProbability >= comparedProbability) {
      return `After "${node.data.displayTokenText}", "${chosenContinuation.displayToken ?? chosenContinuation.token}" had the highest returned probability ${contextLabel}, ahead of "${strongestAlternative.displayToken ?? strongestAlternative.token}" by ${formatSignedPercent(chosenContinuation.displayProbability - comparedProbability)}.`;
    }

    return `After "${node.data.displayTokenText}", "${chosenContinuation.displayToken ?? chosenContinuation.token}" trails "${strongestAlternative.displayToken ?? strongestAlternative.token}" by ${formatSignedPercent(comparedProbability - chosenContinuation.displayProbability)} ${contextLabel}.`;
  }

  return `"${node.data.displayTokenText}" is the current branch prefix. Expand this node to request the exact next-token distribution for this branch context.`;
}

function getBranchSummaries(nodes: TokenFlowNode[], edges: ProbabilityFlowEdge[]) {
  const tokenNodes = nodes.filter((node) => node.data.kind === "token");

  if (tokenNodes.length === 0) {
    return {
      averageConfidence: 0,
      highestBranchingFactor: 0,
      longestBranch: 0,
      mostUncertainToken: null as TokenFlowNode | null,
    };
  }

  const averageConfidence =
    tokenNodes.reduce((total, node) => total + node.data.displayProbability, 0) /
    tokenNodes.length;
  const childCounts = new Map<string, number>();

  for (const edge of edges) {
    if (edge.hidden) {
      continue;
    }

    childCounts.set(edge.source, (childCounts.get(edge.source) ?? 0) + 1);
  }

  const highestBranchingFactor = Math.max(...childCounts.values(), 0);
  const longestBranch = Math.max(...tokenNodes.map((node) => node.data.depth), 0);
  const mostUncertainToken = tokenNodes.reduce((lowest, node) =>
    node.data.displayProbability < lowest.data.displayProbability ? node : lowest,
  );

  return {
    averageConfidence,
    highestBranchingFactor,
    longestBranch,
    mostUncertainToken,
  };
}

void getBranchSummaries;

function getNodeFrame(node: TokenFlowNode) {
  return {
    height: node.data.kind === "prompt" ? 124 : 112,
    width: node.data.kind === "prompt" ? 292 : 208,
  };
}

function downloadFile(name: string, blob: Blob) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = name;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 500);
}

function buildSvgExport(
  nodes: TokenFlowNode[],
  edges: ProbabilityFlowEdge[],
  activeEdgeIds: Set<string>,
) {
  const visibleNodes = nodes.filter((node) => !node.hidden);
  const visibleEdges = edges.filter((edge) => !edge.hidden);

  if (visibleNodes.length === 0) {
    return null;
  }

  const frames = visibleNodes.map((node) => ({
    ...getNodeFrame(node),
    node,
  }));
  const minX = Math.min(...frames.map(({ node }) => node.position.x)) - 120;
  const minY = Math.min(...frames.map(({ node }) => node.position.y)) - 120;
  const maxX = Math.max(
    ...frames.map(({ node, width }) => node.position.x + width),
  ) + 120;
  const maxY = Math.max(
    ...frames.map(({ node, height }) => node.position.y + height),
  ) + 120;
  const width = maxX - minX;
  const height = maxY - minY;
  const nodeMap = new Map(visibleNodes.map((node) => [node.id, node]));

  const edgePaths = visibleEdges
    .map((edge) => {
      const sourceNode = nodeMap.get(edge.source);
      const targetNode = nodeMap.get(edge.target);

      if (!sourceNode || !targetNode) {
        return null;
      }

      const sourceFrame = getNodeFrame(sourceNode);
      const targetFrame = getNodeFrame(targetNode);
      const [path] = getBezierPath({
        sourceX: sourceNode.position.x + sourceFrame.width,
        sourceY: sourceNode.position.y + sourceFrame.height / 2,
        sourcePosition: Position.Right,
        targetX: targetNode.position.x,
        targetY: targetNode.position.y + targetFrame.height / 2,
        targetPosition: Position.Left,
        curvature: 0.28,
      });

      const edgeData = ensureProbabilityEdgeData(edge.data);
      const stroke = edgeData.isMainPath
        ? "#38bdf8"
        : edgeData.probability < 0.36
          ? "#8b5cf6"
          : "#64748b";
      const opacity = edgeData.isDimmed
        ? 0.24
        : activeEdgeIds.has(edge.id)
          ? 1
          : Math.max(0.26, edgeData.probability);
      const strokeWidth = 1.2 + edgeData.probability * 4.4;

      return `<path d="${path}" fill="none" stroke="${stroke}" stroke-width="${strokeWidth.toFixed(2)}" opacity="${opacity.toFixed(2)}" stroke-linecap="round" />`;
    })
    .filter(Boolean)
    .join("");

  const nodeBlocks = visibleNodes
    .map((node) => {
      const frame = getNodeFrame(node);
      const isSelected = node.data.isSearchFocused;
      const stroke = isSelected
        ? "#f59e0b"
        : node.data.isMainPath
          ? "#38bdf8"
          : node.data.displayProbability < 0.36
            ? "#8b5cf6"
            : "#64748b";
      const opacity = node.data.isDimmed ? 0.34 : 1;
      const probabilityWidth = Math.max(
        node.data.displayProbability * (frame.width - 32),
        node.data.kind === "prompt" ? frame.width - 32 : 20,
      );
      const label = node.data.displayTokenText
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      return `
        <g opacity="${opacity.toFixed(2)}">
          <rect x="${node.position.x}" y="${node.position.y}" rx="24" ry="24" width="${frame.width}" height="${frame.height}" fill="#0b1220" stroke="${stroke}" stroke-width="${isSelected ? 2 : 1.2}" />
          <text x="${node.position.x + 16}" y="${node.position.y + 40}" fill="#f8fafc" font-family="Arial, sans-serif" font-size="${node.data.kind === "prompt" ? 16 : 24}" font-weight="600">${label}</text>
          <text x="${node.position.x + frame.width - 52}" y="${node.position.y + 40}" fill="${stroke}" font-family="Arial, sans-serif" font-size="12" font-weight="700">${formatPercent(node.data.displayProbability)}</text>
          <rect x="${node.position.x + 16}" y="${node.position.y + frame.height - 24}" rx="999" ry="999" width="${frame.width - 32}" height="8" fill="rgba(51,65,85,0.72)" />
          <rect x="${node.position.x + 16}" y="${node.position.y + frame.height - 24}" rx="999" ry="999" width="${probabilityWidth}" height="8" fill="${stroke}" />
        </g>
      `;
    })
    .join("");

  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${minX} ${minY} ${width} ${height}">
      <rect x="${minX}" y="${minY}" width="${width}" height="${height}" fill="#050914" />
      ${edgePaths}
      ${nodeBlocks}
    </svg>
  `;
}

function Workspace() {
  const [graphNodes, setGraphNodes] = useState<TokenFlowNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<ProbabilityFlowEdge[]>([]);
  const [tokenGraph, setTokenGraph] = useState<TokenGraphState>(createEmptyTokenGraph());
  const [prompt, setPrompt] = useState(INITIAL_PROMPT);
  const [temperature, setTemperature] = useState(INITIAL_TEMPERATURE);
  const [topP, setTopP] = useState(INITIAL_TOP_P);
  const [maxTokens, setMaxTokens] = useState(INITIAL_MAX_TOKENS);
  const [demoMode, setDemoMode] = useState(false);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogResponse>(
    FALLBACK_MODEL_CATALOG,
  );
  const [model, setModel] = useState(FALLBACK_MODEL_CATALOG.default_model);
  const [preset, setPreset] = useState(FALLBACK_MODEL_CATALOG.default_preset);
  const [generation, setGeneration] = useState<GenerationResponse | null>(null);
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [typedCompletion, setTypedCompletion] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [branchChoices, setBranchChoices] = useState<Record<string, string>>({});
  const [pinnedNodeIds, setPinnedNodeIds] = useState<string[]>([]);
  const [compareLeftId, setCompareLeftId] = useState<string | null>(null);
  const [compareRightId, setCompareRightId] = useState<string | null>(null);
  const [viewport, setViewport] = useState({
    x: 0,
    y: 0,
    zoom: 1,
  });
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [isGraphInteracting, setIsGraphInteracting] = useState(false);
  const [isSentenceBarExpanded, setIsSentenceBarExpanded] = useState(false);
  const [isDockCollapsed, setIsDockCollapsed] = useState(false);
  const [surfaceTheme, setSurfaceTheme] = useState<SurfaceTheme>("midnight");
  const [probabilityViewMode, setProbabilityViewMode] =
    useState<ProbabilityViewMode>("raw");
  const [playbackSpeed, setPlaybackSpeed] = useState(DEFAULT_PLAYBACK_SPEED);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResultIndex, setSearchResultIndex] = useState(0);
  const [history, setHistory] = useState<GraphSnapshot[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [requestVariation, setRequestVariation] = useState(0);
  const [changedTokenIndexes, setChangedTokenIndexes] = useState<number[]>([]);
  const [continuationPreview, setContinuationPreview] = useState<ContinuationPreviewState | null>(null);
  const flowRef = useRef<ReactFlowInstance<TokenFlowNode, ProbabilityFlowEdge> | null>(null);
  const nodesRef = useRef<TokenFlowNode[]>([]);
  const edgesRef = useRef<ProbabilityFlowEdge[]>([]);
  const tokenGraphRef = useRef<TokenGraphState>(createEmptyTokenGraph());
  const branchChoicesRef = useRef<Record<string, string>>({});
  const pinnedNodeIdsRef = useRef<string[]>([]);
  const compareLeftIdRef = useRef<string | null>(null);
  const compareRightIdRef = useRef<string | null>(null);
  const selectedNodeIdRef = useRef<string | null>(null);
  const animationRunRef = useRef(0);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const previousSentenceTokensRef = useRef<string[]>([]);
  const undoHistoryRef = useRef<() => void>(() => undefined);
  const redoHistoryRef = useRef<() => void>(() => undefined);
  const applyTransitionRef = useRef<
    (next: Partial<GraphSnapshot>, options?: { pushHistory?: boolean; skipLayout?: boolean }) => void
  >(() => undefined);
  const centerNodeRef = useRef<(nodeId: string) => void>(() => undefined);
  const activateRealityRef = useRef<
    (nodeId: string, options?: { pushHistory?: boolean }) => void
  >(() => undefined);
  const expandNodeRef = useRef<
    (nodeId: string, options?: { pushHistory?: boolean }) => Promise<boolean>
  >(async () => false);
  const collapseSubtreeRef = useRef<(nodeId: string) => void>(() => undefined);
  const isGraphInteractingRef = useRef(false);
  const dragPerformanceRef = useRef<DragPerformanceStats>({
    sampleCount: 0,
    totalMs: 0,
  });

  const models = modelCatalog.models;
  const presets = modelCatalog.presets;
  const activePathIds = useMemo(
    () => buildActivePathIds(branchChoices, graphNodes),
    [branchChoices, graphNodes],
  );
  const activePathIdSet = useMemo(() => new Set(activePathIds), [activePathIds]);
  const activeEdgeIdSet = useMemo(
    () => buildActivePathEdgeIds(activePathIds),
    [activePathIds],
  );
  const hoveredRelatedIdSet = useMemo(
    () =>
      hoveredNodeId ? collectHoverContextIds(hoveredNodeId, graphNodes, graphEdges) : null,
    [graphEdges, graphNodes, hoveredNodeId],
  );
  const searchMatches = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    if (!query) {
      return [];
    }

    return graphNodes.filter(
      (node) =>
        node.data.tokenText.toLowerCase().includes(query) ||
        node.data.textPreview.toLowerCase().includes(query),
    );
  }, [graphNodes, searchQuery]);
  const focusedSearchNodeId = useMemo(
    () =>
      searchMatches.length > 0
        ? searchMatches[
            ((searchResultIndex % searchMatches.length) + searchMatches.length) %
              searchMatches.length
          ].id
        : null,
    [searchMatches, searchResultIndex],
  );
  const pinnedSet = useMemo(() => new Set(pinnedNodeIds), [pinnedNodeIds]);
  const nodeProbabilityView = useMemo(
    () => buildNodeProbabilityView(graphNodes, probabilityViewMode),
    [graphNodes, probabilityViewMode],
  );

  const displayNodes = useMemo(
    () =>
      graphNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          displayProbability:
            nodeProbabilityView.get(node.id)?.displayProbability ?? node.data.displayProbability,
          probabilityCoverage:
            nodeProbabilityView.get(node.id)?.probabilityCoverage ??
            node.data.probabilityCoverage,
          remainingProbabilityMass:
            nodeProbabilityView.get(node.id)?.remainingProbabilityMass ??
            node.data.remainingProbabilityMass,
          probabilityMode: probabilityViewMode,
          isSearchMatch: searchMatches.some((match) => match.id === node.id),
          isSearchFocused: focusedSearchNodeId === node.id,
          isDimmed: hoveredRelatedIdSet ? !hoveredRelatedIdSet.has(node.id) : false,
          isActiveReality: activePathIdSet.has(node.id),
          isPinned: pinnedSet.has(node.id),
        },
      })),
    [
      activePathIdSet,
      focusedSearchNodeId,
      graphNodes,
      hoveredRelatedIdSet,
      nodeProbabilityView,
      pinnedSet,
      probabilityViewMode,
      searchMatches,
    ],
  );
  const displayNodeMap = useMemo(
    () => new Map(displayNodes.map((node) => [node.id, node])),
    [displayNodes],
  );
  const displayEdges: ProbabilityFlowEdge[] = useMemo(
    () =>
      graphEdges.map((edge) => {
        const sourceNode = displayNodeMap.get(edge.source);
        const targetNode = displayNodeMap.get(edge.target);
        const sourceDimmed = sourceNode?.data.isDimmed ?? false;
        const targetDimmed = targetNode?.data.isDimmed ?? false;
        const edgeData = ensureProbabilityEdgeData(edge.data);

        return {
          ...edge,
          data: {
            ...edgeData,
            probability: targetNode?.data.displayProbability ?? edgeData.probability,
            rawProbability: targetNode?.data.rawProbability ?? edgeData.rawProbability,
            probabilityCoverage:
              targetNode?.data.probabilityCoverage ?? edgeData.probabilityCoverage,
            remainingProbabilityMass:
              targetNode?.data.remainingProbabilityMass ?? edgeData.remainingProbabilityMass,
            probabilityMode: targetNode?.data.probabilityMode ?? probabilityViewMode,
            isActiveReality: activeEdgeIdSet.has(edge.id),
            isDimmed: sourceDimmed || targetDimmed,
            isFocused: hoveredNodeId
              ? activeEdgeIdSet.has(edge.id) ||
                edge.source === hoveredNodeId ||
                edge.target === hoveredNodeId
              : false,
            },
        };
      }),
    [activeEdgeIdSet, displayNodeMap, graphEdges, hoveredNodeId, probabilityViewMode],
  );

  const selectedNode =
    (selectedNodeId ? displayNodeMap.get(selectedNodeId) ?? null : null) ??
    (activePathIds[activePathIds.length - 1]
      ? displayNodeMap.get(activePathIds[activePathIds.length - 1]) ?? null
      : null) ??
    null;
  const activeLeafNodeId = activePathIds[activePathIds.length - 1] ?? null;
  const selectedRecord =
    (selectedNodeId ? tokenGraph.nodesById[selectedNodeId] ?? null : null) ??
    (activeLeafNodeId ? tokenGraph.nodesById[activeLeafNodeId] ?? null : null);
  const inspectorAlternativeView = useMemo(
    () =>
      buildInspectorAlternatives(
        tokenGraph,
        selectedNode?.id ?? activeLeafNodeId,
        probabilityViewMode,
      ),
    [activeLeafNodeId, probabilityViewMode, selectedNode?.id, tokenGraph],
  );
  const inspectorAlternatives = inspectorAlternativeView.items;
  const generatedModelOption = findModelOption(models, generation?.request.model ?? model);
  const generatedPresetOption = findPresetOption(presets, generation?.request.preset ?? preset);
  const activeSentenceNodes = activePathIds
    .slice(1)
    .map((nodeId) => displayNodeMap.get(nodeId))
    .filter((node): node is TokenFlowNode => Boolean(node));
  const activeSentenceTokens = activeSentenceNodes.map((node) => node.data.tokenText);
  const activeSentenceTokenKey = activeSentenceTokens.join("\u0001");
  const currentSentenceText =
    activeLeafNodeId && activeLeafNodeId !== "root"
      ? reconstructAssistantPrefix(tokenGraph, activeLeafNodeId)
      : typedCompletion || generation?.completion || "";
  const hasSentenceContent = activeSentenceNodes.length > 0 || Boolean(currentSentenceText);
  const isSentenceBarCollapsed =
    hasSentenceContent && !isGenerating && !isReplaying && !isSentenceBarExpanded;
  const naturalReason = buildNaturalLanguageReason(
    selectedNode,
    inspectorAlternatives,
    probabilityViewMode,
  );
  const selectedNodeMetrics =
    selectedNode && selectedRecord
      ? getNodeMetrics(selectedRecord.decodedContribution, selectedRecord.tokenBytes)
      : null;
  const inspectorCoverage = inspectorAlternativeView.coverage;
  const inspectorRemainingProbabilityMass =
    inspectorAlternativeView.remainingProbabilityMass;
  const changedTokenIndexSet = useMemo(
    () => new Set(changedTokenIndexes),
    [changedTokenIndexes],
  );
  const currentRealityTokens = useMemo<CurrentRealityTokenItem[]>(
    () =>
      activeSentenceNodes.map((node, index) => ({
        decodedContribution: node.data.decodedContribution,
        displayProbability: node.data.displayProbability,
        displayToken: node.data.displayTokenText,
        id: node.id,
        isChanged: changedTokenIndexSet.has(index),
        rank: node.data.rank,
        rawProbability: node.data.rawProbability,
        rawToken: node.data.tokenText,
        step: node.data.generationStep,
      })),
    [activeSentenceNodes, changedTokenIndexSet],
  );
  const currentRealityStats = useMemo<CurrentRealityStats>(
    () => ({
      branchDepth: activeSentenceNodes[activeSentenceNodes.length - 1]?.data.depth ?? 0,
      displayProbability:
        activeSentenceNodes[activeSentenceNodes.length - 1]?.data.displayProbability ?? 0,
      entropy: activeSentenceNodes[activeSentenceNodes.length - 1]?.data.entropy ?? 0,
      latency: activeSentenceNodes[activeSentenceNodes.length - 1]?.data.latency ?? 0,
      rawProbability:
        activeSentenceNodes[activeSentenceNodes.length - 1]?.data.rawProbability ?? 0,
      tokenCount: activeSentenceNodes.length,
    }),
    [activeSentenceNodes],
  );
  const currentRealityContinuationMode = useMemo(
    () =>
      getContinuationModePresentation(
        activeSentenceNodes[activeSentenceNodes.length - 1]?.data.metadata ?? selectedNode?.data.metadata,
      ),
    [activeSentenceNodes, selectedNode?.data.metadata],
  );
  const inspectorContinuationMode = useMemo(
    () => getContinuationModePresentation(selectedNode?.data.metadata ?? null),
    [selectedNode?.data.metadata],
  );
  const compareNodes = [
    compareLeftId ? displayNodeMap.get(compareLeftId) ?? null : null,
    compareRightId ? displayNodeMap.get(compareRightId) ?? null : null,
  ] as const;
  const compareSentences = compareNodes.map((node) =>
    node ? reconstructAssistantPrefix(tokenGraph, node.id) : "",
  );
  const compareSentenceTokens = compareSentences.map((sentence) => splitPreview(sentence));
  const compareDivergenceIndex = (() => {
    if (!compareNodes[0] || !compareNodes[1]) {
      return -1;
    }

    const longest = Math.max(compareSentenceTokens[0].length, compareSentenceTokens[1].length);

    for (let index = 0; index < longest; index += 1) {
      if (compareSentenceTokens[0][index] !== compareSentenceTokens[1][index]) {
        return index;
      }
    }

    return -1;
  })();

  useEffect(() => {
    nodesRef.current = graphNodes;
  }, [graphNodes]);

  useEffect(() => {
    edgesRef.current = graphEdges;
  }, [graphEdges]);

  useEffect(() => {
    logCanvasPerformance("graph-size", {
      nodeCount: graphNodes.length,
      edgeCount: graphEdges.length,
    });
  }, [graphEdges.length, graphNodes.length]);

  useEffect(() => {
    tokenGraphRef.current = tokenGraph;
  }, [tokenGraph]);

  useEffect(() => {
    branchChoicesRef.current = branchChoices;
  }, [branchChoices]);

  useEffect(() => {
    pinnedNodeIdsRef.current = pinnedNodeIds;
  }, [pinnedNodeIds]);

  useEffect(() => {
    compareLeftIdRef.current = compareLeftId;
  }, [compareLeftId]);

  useEffect(() => {
    compareRightIdRef.current = compareRightId;
  }, [compareRightId]);

  useEffect(() => {
    selectedNodeIdRef.current = selectedNodeId;
  }, [selectedNodeId]);

  useEffect(() => {
    document.documentElement.dataset.surfaceTheme = surfaceTheme;
  }, [surfaceTheme]);

  useEffect(() => {
    return () => {
      animationRunRef.current += 1;
    };
  }, []);

  useEffect(() => {
    const nextTokens = activeSentenceTokenKey
      ? activeSentenceTokenKey.split("\u0001")
      : [];
    const previousTokens = previousSentenceTokensRef.current;
    const changed = new Set<number>();
    const longest = Math.max(previousTokens.length, nextTokens.length);

    for (let index = 0; index < longest; index += 1) {
      if (previousTokens[index] !== nextTokens[index]) {
        changed.add(index);
      }
    }

    previousSentenceTokensRef.current = nextTokens;
    const nextChangedIndexes = [...changed];

    setChangedTokenIndexes((currentValue) =>
      equalNumberLists(currentValue, nextChangedIndexes) ? currentValue : nextChangedIndexes,
    );

    if (changed.size === 0) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setChangedTokenIndexes((currentValue) =>
        currentValue.length === 0 ? currentValue : [],
      );
    }, 1000);

    return () => window.clearTimeout(timeoutId);
  }, [activeSentenceTokenKey]);

  useEffect(() => {
    setSearchResultIndex(0);
  }, [searchQuery]);

  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "f") {
        event.preventDefault();
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
        return;
      }

      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();

        if (event.shiftKey) {
          redoHistoryRef.current();
        } else {
          undoHistoryRef.current();
        }

        return;
      }

      if (event.key === "Escape") {
        setContextMenu(null);
      }
    };

    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  async function refreshHealth() {
    setIsCheckingHealth(true);

    try {
      const response = await fetch("/api/health", {
        cache: "no-store",
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      await response.json();
      setBackendState("online");
    } catch {
      setBackendState("offline");
    } finally {
      setIsCheckingHealth(false);
    }
  }

  useEffect(() => {
    let isActive = true;

    void (async () => {
      setIsCheckingHealth(true);

      try {
        const response = await fetch("/api/health", {
          cache: "no-store",
        });

        if (isActive && response.ok) {
          setBackendState("online");
        } else if (isActive) {
          setBackendState("offline");
        }
      } catch {
        if (isActive) {
          setBackendState("offline");
        }
      } finally {
        if (isActive) {
          setIsCheckingHealth(false);
        }
      }

      setIsLoadingModels(true);

      try {
        const response = await fetch("/api/models", {
          cache: "no-store",
        });

        if (!response.ok) {
          await throwApiError(response);
        }

        const payload = (await response.json()) as ModelCatalogResponse;

        if (!isActive) {
          return;
        }

        setModelCatalog(payload);
        setModel((currentModel) =>
          payload.models.some((item) => item.id === currentModel)
            ? currentModel
            : payload.default_model,
        );
        setPreset((currentPreset) =>
          payload.presets.some((item) => item.id === currentPreset)
            ? currentPreset
            : payload.default_preset,
        );
      } catch {
        if (isActive) {
          setModelCatalog(FALLBACK_MODEL_CATALOG);
        }
      } finally {
        if (isActive) {
          setIsLoadingModels(false);
        }
      }
    })();

    return () => {
      isActive = false;
    };
  }, []);

  function getCurrentSnapshot(): GraphSnapshot {
    return {
      nodes: nodesRef.current,
      edges: edgesRef.current,
      branchChoices: branchChoicesRef.current,
      pinnedNodeIds: pinnedNodeIdsRef.current,
      compareLeftId: compareLeftIdRef.current,
      compareRightId: compareRightIdRef.current,
      selectedNodeId: selectedNodeIdRef.current,
    };
  }

  function pushHistorySnapshot(snapshot: GraphSnapshot) {
    const clone = structuredClone(snapshot);

    setHistory((currentHistory) => {
      const nextHistory = currentHistory.slice(0, historyIndex + 1);
      nextHistory.push(clone);
      return nextHistory;
    });
    setHistoryIndex((currentIndex) => currentIndex + 1);
  }

  function applySnapshot(snapshot: GraphSnapshot) {
    setGraphNodes(snapshot.nodes);
    setGraphEdges(snapshot.edges);
    setBranchChoices(snapshot.branchChoices);
    setPinnedNodeIds(snapshot.pinnedNodeIds);
    setCompareLeftId(snapshot.compareLeftId);
    setCompareRightId(snapshot.compareRightId);
    setSelectedNodeId(snapshot.selectedNodeId);
    setContextMenu(null);
    nodesRef.current = snapshot.nodes;
    edgesRef.current = snapshot.edges;
    branchChoicesRef.current = snapshot.branchChoices;
    pinnedNodeIdsRef.current = snapshot.pinnedNodeIds;
    compareLeftIdRef.current = snapshot.compareLeftId;
    compareRightIdRef.current = snapshot.compareRightId;
    selectedNodeIdRef.current = snapshot.selectedNodeId;
  }

  function undoHistory() {
    if (historyIndex <= 0) {
      return;
    }

    const snapshot = history[historyIndex - 1];

    if (!snapshot) {
      return;
    }

    setHistoryIndex((currentIndex) => currentIndex - 1);
    applySnapshot(structuredClone(snapshot));
  }

  function redoHistory() {
    if (historyIndex >= history.length - 1) {
      return;
    }

    const snapshot = history[historyIndex + 1];

    if (!snapshot) {
      return;
    }

    setHistoryIndex((currentIndex) => currentIndex + 1);
    applySnapshot(structuredClone(snapshot));
  }

  undoHistoryRef.current = undoHistory;
  redoHistoryRef.current = redoHistory;

  const setGraphInteractionActive = useCallback((active: boolean) => {
    if (isGraphInteractingRef.current === active) {
      return;
    }

    isGraphInteractingRef.current = active;
    setIsGraphInteracting(active);
  }, []);

  const commitTransientNodes = useCallback((nextNodes: TokenFlowNode[]) => {
    setGraphNodes(nextNodes);
    nodesRef.current = nextNodes;
  }, []);

  const commitTransientEdges = useCallback((nextEdges: ProbabilityFlowEdge[]) => {
    setGraphEdges(nextEdges);
    edgesRef.current = nextEdges;
  }, []);

  function applyTransition(
    next: Partial<GraphSnapshot>,
    options?: { pushHistory?: boolean; skipLayout?: boolean },
  ) {
    const transitionStart = performance.now();
    const current = getCurrentSnapshot();
    const branchChoices = sanitizeBranchChoices(
      next.branchChoices ?? current.branchChoices,
      next.nodes ?? current.nodes,
    );
    const normalizeStart = performance.now();
    const normalized = normalizeGraph(
      next.nodes ?? current.nodes,
      next.edges ?? current.edges,
      branchChoices,
      { skipLayout: options?.skipLayout },
    );
    const normalizeDuration = performance.now() - normalizeStart;
    const snapshot = buildTransitionSnapshot(current, next, normalized.nodes, normalized.edges);

    setGraphNodes(snapshot.nodes);
    setGraphEdges(snapshot.edges);
    setBranchChoices(snapshot.branchChoices);
    setPinnedNodeIds(snapshot.pinnedNodeIds);
    setCompareLeftId(snapshot.compareLeftId);
    setCompareRightId(snapshot.compareRightId);
    setSelectedNodeId(snapshot.selectedNodeId);
    nodesRef.current = snapshot.nodes;
    edgesRef.current = snapshot.edges;
    branchChoicesRef.current = snapshot.branchChoices;
    pinnedNodeIdsRef.current = snapshot.pinnedNodeIds;
    compareLeftIdRef.current = snapshot.compareLeftId;
    compareRightIdRef.current = snapshot.compareRightId;
    selectedNodeIdRef.current = snapshot.selectedNodeId;

    if (options?.pushHistory !== false) {
      pushHistorySnapshot(snapshot);
    }

    const totalDuration = performance.now() - transitionStart;
    if (totalDuration > FRAME_BUDGET_MS) {
      logCanvasPerformance("graph-update", {
        durationMs: Number(totalDuration.toFixed(2)),
        layoutDurationMs: Number(normalizeDuration.toFixed(2)),
        nodeCount: snapshot.nodes.length,
        edgeCount: snapshot.edges.length,
        skipLayout: Boolean(options?.skipLayout),
      });
    }
  }

  applyTransitionRef.current = applyTransition;

  function centerNode(nodeId: string) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!node || !flowRef.current) {
      return;
    }

    const frame = getNodeFrame(node);

    void flowRef.current.setCenter(node.position.x + frame.width / 2, node.position.y + frame.height / 2, {
      duration: 520,
      zoom: Math.max(viewport.zoom, 0.9),
    });
  }

  centerNodeRef.current = centerNode;

  function activateReality(nodeId: string, options?: { pushHistory?: boolean }) {
    const nextChoices = buildRealityChoicesForNode(nodeId, nodesRef.current, branchChoicesRef.current);

    applyTransition(
      {
        branchChoices: nextChoices,
        selectedNodeId: nodeId,
      },
      options,
    );
  }

  activateRealityRef.current = activateReality;

  async function playMainPath(payload: GenerationResponse) {
    const runId = ++animationRunRef.current;
    const reasoning = reasoningBundleFromGeneration(payload);
    const nextTokenGraph = createTokenGraphFromGeneration(payload);
    setIsSentenceBarExpanded(true);
    setTokenGraph(nextTokenGraph);
    tokenGraphRef.current = nextTokenGraph;
    const rootNode = buildPromptNode({
      prompt: payload.prompt_used,
      model: payload.request.model,
      preset: payload.request.preset,
      reasoning,
      responseMode: payload.mode,
      status: "ready",
      temperature: payload.request.temperature,
      topP: payload.request.top_p,
      variation: payload.request.variation,
      demoMode: payload.request.demo_mode,
    });
    const rootRecord =
      nextTokenGraph.rootNodeId ? nextTokenGraph.nodesById[nextTokenGraph.rootNodeId] ?? null : null;
    const replayChoices: Record<string, string> = {};

    applyTransition(
      {
        nodes: rootRecord
          ? [applyTokenGraphRecordToFlowNode(rootNode, rootRecord)]
          : [rootNode],
        edges: [],
        branchChoices: replayChoices,
        selectedNodeId: "root",
      },
      { pushHistory: false },
    );
    setTypedCompletion("");
    setIsReplaying(true);

    for (let index = 0; index < payload.tokens.length; index += 1) {
      if (animationRunRef.current !== runId) {
        return;
      }

      const appendStart = performance.now();
      const trace = payload.tokens[index];
      const parentId = index === 0 ? "root" : payload.tokens[index - 1].id;
      const parentNode = nodesRef.current.find((currentNode) => currentNode.id === parentId);
      const nextNode = buildTokenNode({
        id: trace.id,
        token: trace.token,
        tokenId: trace.token_id ?? null,
        tokenizerId: trace.tokenizer_id ?? null,
        displayToken: trace.display_token,
        decodedContribution: trace.decoded_contribution ?? trace.token,
        cumulativeDecodedText:
          trace.cumulative_decoded_text ?? trace.context_after ?? trace.text_preview,
        cumulativeTokenIds: trace.cumulative_token_ids ?? null,
        cumulativeLogProbability: trace.cumulative_log_probability ?? trace.log_probability,
        contextBefore: trace.context_before,
        contextAfter: trace.context_after,
        generationStep: trace.generation_step ?? trace.position,
        probability: trace.probability,
        rawProbability: trace.raw_probability,
        normalizedDisplayedProbability: trace.normalized_displayed_probability,
        logProbability: trace.log_probability,
        entropy: trace.entropy,
        cumulativeProbability: trace.cumulative_probability,
        latency: trace.latency_ms,
        depth: trace.position + 1,
        rank: 1,
        parentId,
        position: parentNode
          ? { x: parentNode.position.x + HORIZONTAL_GAP, y: parentNode.position.y }
          : { x: 0, y: 0 },
        prompt: payload.prompt_used,
        model: payload.request.model,
        preset: payload.request.preset,
        temperature: payload.request.temperature,
        topP: payload.request.top_p,
        variation: payload.request.variation,
        demoMode: payload.request.demo_mode,
        responseMode: payload.mode,
        textPreview: trace.text_preview,
        isMainPath: true,
        status: "idle",
        reasoning,
        branchRationale: null,
        metadata: trace.metadata ?? {},
        rawLogits: null,
        sourceAlternatives: mapTraceAlternativesToInspector(trace.alternatives),
      });
      const canonicalNode = nextTokenGraph.nodesById[trace.id];
      const syncedNextNode = canonicalNode
        ? applyTokenGraphRecordToFlowNode(nextNode, canonicalNode)
        : nextNode;

      replayChoices[parentId] = syncedNextNode.id;
      const nextNodes = syncFlowNodesWithTokenGraph(
        [...nodesRef.current, syncedNextNode],
        nextTokenGraph,
      );
      const nextEdges = [...edgesRef.current, buildEdge(parentId, syncedNextNode.id, trace.probability, true)];
      applyTransition(
        {
          nodes: nextNodes,
          edges: nextEdges,
          branchChoices: { ...replayChoices },
          selectedNodeId: syncedNextNode.id,
        },
        { pushHistory: false },
      );
      setTypedCompletion(trace.text_preview);

      const appendDuration = performance.now() - appendStart;
      if (appendDuration > FRAME_BUDGET_MS) {
        logCanvasPerformance("generation-append", {
          durationMs: Number(appendDuration.toFixed(2)),
          tokenIndex: index,
          nodeCount: nextNodes.length,
          edgeCount: nextEdges.length,
        });
      }
      await wait((index < 4 ? 110 : index < 16 ? 72 : 40) / playbackSpeed);
    }

    if (animationRunRef.current === runId) {
      setIsReplaying(false);
      setIsSentenceBarExpanded(false);
      applyTransition(
        {
          nodes: syncFlowNodesWithTokenGraph(nodesRef.current, nextTokenGraph),
          edges: edgesRef.current,
          branchChoices: { ...replayChoices },
          selectedNodeId: payload.tokens[payload.tokens.length - 1]?.id ?? "root",
        },
        { pushHistory: false },
      );
      pushHistorySnapshot(getCurrentSnapshot());
      window.setTimeout(() => {
        void flowRef.current?.fitView({
          duration: 640,
          padding: 0.18,
        });
      }, 40);
    }
  }

  function getPreferredContinuationTarget(nodeId: string) {
    const currentChoice = branchChoicesRef.current[nodeId];
    const currentEdges = edgesRef.current.filter((edge) => edge.source === nodeId);

    if (currentChoice && currentEdges.some((edge) => edge.target === currentChoice)) {
      return currentChoice;
    }

    return currentEdges.sort(
      (left, right) =>
        ensureProbabilityEdgeData(right.data).probability -
        ensureProbabilityEdgeData(left.data).probability,
    )[0]?.target ?? null;
  }

  function applyExpansionPayloadToCanvas(
    nodeId: string,
    parentNode: TokenFlowNode,
    validation: ReturnType<typeof buildContinuationValidation>,
    payload: NodeExpansionResponse | ContinueGenerationResponse,
    options?: { pushHistory?: boolean },
  ) {
    const nextTokenGraph = applyExpansionToTokenGraph(tokenGraphRef.current, nodeId, payload, {
      requestPrompt: validation.rootPrompt,
      model: parentNode.data.requestModel,
      preset: parentNode.data.requestPreset,
      temperature: parentNode.data.requestTemperature,
      topP: parentNode.data.requestTopP,
      variation: parentNode.data.requestVariation,
      demoMode: parentNode.data.requestDemoMode,
    });
    setTokenGraph(nextTokenGraph);
    tokenGraphRef.current = nextTokenGraph;

    const reasoning: ReasoningBundle = {
      notes: payload.notes || parentNode.data.sourceNotes,
      intent: parentNode.data.reasoningIntent,
      strategy: parentNode.data.reasoningStrategy,
      focusTerms: parentNode.data.reasoningFocusTerms,
    };
    const currentChildren = edgesRef.current
      .filter((edge) => edge.source === nodeId)
      .map((edge) => nodesRef.current.find((item) => item.id === edge.target))
      .filter((item): item is TokenFlowNode => Boolean(item));
    const responseChildIds = new Set(payload.children.map((candidate) => candidate.id));
    const staleNodeIds = new Set<string>();

    for (const child of currentChildren) {
      if (responseChildIds.has(child.id)) {
        continue;
      }

      staleNodeIds.add(child.id);
      for (const descendantId of collectDescendantIds(child.id, edgesRef.current)) {
        staleNodeIds.add(descendantId);
      }
    }

    const nextNodes = nodesRef.current.filter((currentNode) => !staleNodeIds.has(currentNode.id));
    const nextEdges = edgesRef.current.filter(
      (edge) =>
        !staleNodeIds.has(edge.source) &&
        !staleNodeIds.has(edge.target) &&
        (edge.source !== nodeId || responseChildIds.has(edge.target)),
    );
    let graphStructureChanged = staleNodeIds.size > 0;

    for (const candidate of payload.children) {
      const matchingChild = currentChildren.find((child) => child.id === candidate.id);

      if (matchingChild) {
        const nextIndex = nextNodes.findIndex((item) => item.id === matchingChild.id);

        nextNodes[nextIndex] = {
          ...matchingChild,
          data: {
            ...matchingChild.data,
            probability: candidate.probability,
            rawProbability: candidate.raw_probability,
            normalizedDisplayedProbability: candidate.normalized_displayed_probability,
            logProbability: candidate.log_probability,
            entropy: candidate.entropy,
            latency: candidate.latency_ms,
            tokenId: candidate.token_id ?? null,
            tokenizerId: candidate.tokenizer_id ?? null,
            displayTokenText: candidate.display_token,
            cumulativeProbability: candidate.cumulative_probability,
            rank: candidate.rank,
            isMainPath: matchingChild.data.isMainPath || candidate.rank === 1,
            responseMode: payload.mode,
            sourceNotes: payload.notes || matchingChild.data.sourceNotes,
            branchRationale: candidate.rationale ?? matchingChild.data.branchRationale,
            metadata: candidate.metadata ?? matchingChild.data.metadata,
            sourceAlternatives: [],
            distributionMessage: null,
          },
        };
      } else {
        nextNodes.push(
          buildTokenNode({
            id: candidate.id,
            token: candidate.token,
            tokenId: candidate.token_id ?? null,
            tokenizerId: candidate.tokenizer_id ?? null,
            displayToken: candidate.display_token,
            decodedContribution: candidate.decoded_contribution ?? candidate.token,
            cumulativeDecodedText:
              candidate.cumulative_decoded_text ?? candidate.context_after ?? candidate.text_preview,
            cumulativeRawText:
              candidate.context_after ??
              candidate.cumulative_decoded_text ??
              candidate.text_preview ??
              candidate.token,
            cumulativeTokenIds: candidate.cumulative_token_ids ?? null,
            cumulativeLogProbability:
              candidate.cumulative_log_probability ?? candidate.log_probability,
            contextBefore: candidate.context_before,
            contextAfter: candidate.context_after,
            generationStep: candidate.generation_step ?? Math.max(candidate.depth - 1, 0),
            probability: candidate.probability,
            rawProbability: candidate.raw_probability,
            normalizedDisplayedProbability: candidate.normalized_displayed_probability,
            logProbability: candidate.log_probability,
            entropy: candidate.entropy,
            cumulativeProbability: candidate.cumulative_probability,
            latency: candidate.latency_ms,
            depth: candidate.depth,
            rank: candidate.rank,
            parentId: nodeId,
            position: {
              x: parentNode.position.x + HORIZONTAL_GAP,
              y: parentNode.position.y,
            },
            prompt: parentNode.data.requestPrompt,
            model: parentNode.data.requestModel,
            preset: parentNode.data.requestPreset,
            temperature: parentNode.data.requestTemperature,
            topP: parentNode.data.requestTopP,
            variation: parentNode.data.requestVariation,
            demoMode: parentNode.data.requestDemoMode,
            responseMode: payload.mode,
            textPreview: candidate.text_preview,
            isMainPath: candidate.rank === 1,
            status: "idle",
            reasoning,
            branchRationale: candidate.rationale ?? null,
            metadata: candidate.metadata ?? {},
            rawLogits: null,
            sourceAlternatives: [],
          }),
        );
        graphStructureChanged = true;
      }

      const nextEdge = buildEdge(nodeId, candidate.id, candidate.probability, candidate.rank === 1);
      const existingEdgeIndex = nextEdges.findIndex((edge) => edge.id === nextEdge.id);

      if (existingEdgeIndex >= 0) {
        nextEdges[existingEdgeIndex] = {
          ...nextEdges[existingEdgeIndex],
          data: nextEdge.data,
        };
      } else {
        nextEdges.push(nextEdge);
      }
    }

    const parentIndex = nextNodes.findIndex((item) => item.id === nodeId);

    nextNodes[parentIndex] = {
      ...nextNodes[parentIndex],
      data: {
        ...nextNodes[parentIndex].data,
        distributionRequested: true,
        isCollapsed: false,
        status: "ready",
        responseMode: payload.mode,
        sourceNotes: payload.notes || nextNodes[parentIndex].data.sourceNotes,
        distributionMessage: null,
      },
    };

    const syncedNodes = syncFlowNodesWithTokenGraph(nextNodes, nextTokenGraph);

    if (SHOULD_LOG_CONTINUATION) {
      const syncedNodeMap = new Map(syncedNodes.map((currentNode) => [currentNode.id, currentNode]));
      logContinuationDebug("flow-nodes", {
        parentNodeId: nodeId,
        action: "action" in payload ? payload.action : "expand",
        createdNodes: payload.children.map((candidate) => {
          const flowNode = syncedNodeMap.get(candidate.id);

          return {
            rawApiToken: candidate.token,
            rawLogprob: candidate.log_probability,
            parsedGeneratedToken: candidate.token,
            parsedAlternatives: payload.children
              .filter((otherCandidate) => otherCandidate.id !== candidate.id)
              .map((otherCandidate) => ({
                token: otherCandidate.token,
                logprob: otherCandidate.log_probability,
                probability: otherCandidate.probability,
                rank: otherCandidate.rank,
              })),
            contextBefore: candidate.context_before,
            contextThrough: candidate.context_after,
            flowNode: flowNode
              ? {
                  id: flowNode.id,
                  token: flowNode.data.tokenText,
                  predictionId: flowNode.data.predictionId,
                  cachedSegmentId: readMetadataString(flowNode.data.metadata, "cached_segment_id"),
                  nextCachedTokenIndex: readMetadataNumber(
                    flowNode.data.metadata,
                    "next_cached_token_index",
                  ),
                  topAlternatives: flowNode.data.topAlternatives.map((alternative) => ({
                    token: alternative.token,
                    probability: alternative.rawProbability ?? alternative.probability,
                    rank: alternative.rank,
                  })),
                  contextBefore: flowNode.data.contextBefore,
                  contextThrough: flowNode.data.contextAfter,
                }
              : null,
          };
        }),
      });
    }

    applyTransition(
      {
        nodes: syncedNodes,
        edges: nextEdges,
        selectedNodeId: nodeId,
      },
      graphStructureChanged ? { pushHistory: false } : options,
    );

    return true;
  }

  async function expandContinuationNode(nodeId: string, options?: { pushHistory?: boolean }) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);
    const validation = buildContinuationValidation(tokenGraphRef.current, nodeId);

    if (!node || node.data.status === "loading") {
      return false;
    }

    if (!validation.isValid) {
      setErrorMessage(validation.warnings.join(" "));
      return false;
    }

    setSelectedNodeId(nodeId);
    setContextMenu(null);

    if (node.data.isCollapsed) {
      applyTransition(
        {
          nodes: nodesRef.current.map((currentNode) =>
            currentNode.id === nodeId
              ? {
                  ...currentNode,
                  data: {
                    ...currentNode.data,
                    isCollapsed: false,
                  },
                }
              : currentNode,
          ),
        },
        options,
      );
      return true;
    }

    if (node.data.distributionRequested) {
      return true;
    }

    applyTransition(
      {
        nodes: nodesRef.current.map((currentNode) =>
          currentNode.id === nodeId
            ? {
                ...currentNode,
                data: {
                  ...currentNode.data,
                  status: "loading",
                },
              }
            : currentNode,
        ),
        selectedNodeId: nodeId,
      },
      { pushHistory: false },
    );

    try {
      const response = await fetch("/api/expand-node", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          root_prompt: validation.rootPrompt,
          model: node.data.requestModel,
          preset: node.data.requestPreset,
          temperature: node.data.requestTemperature,
          top_p: node.data.requestTopP,
          parent_node_id: nodeId,
          parent_token: node.data.tokenText,
          assistant_prefix: validation.assistantPrefix,
          reconstructed_prompt: validation.reconstructedPrompt,
          expected_prompt_length: validation.characterLength,
          expected_utf8_length: validation.utf8Length,
          expected_token_count: validation.tokenCount,
          depth: node.data.depth,
          cumulative_probability: node.data.cumulativeProbability,
          variation: node.data.requestVariation,
          max_children: MAX_BRANCH_CHILDREN,
          demo_mode: node.data.requestDemoMode,
        }),
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as NodeExpansionResponse;
      const parentNode = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

      if (!parentNode) {
        return false;
      }
      applyExpansionPayloadToCanvas(nodeId, parentNode, validation, payload, options);
      setBackendState("online");
      return true;
    } catch (error) {
      const errorCode = getErrorCode(error);
      const alternativesUnavailable =
        errorCode === "LOGPROBS_UNAVAILABLE" || errorCode === "TOP_LOGPROBS_UNAVAILABLE";
      setBackendState(errorCode === "PROVIDER_REQUEST_FAILED" ? "offline" : "online");
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to expand the selected node.",
      );
      if (alternativesUnavailable && error instanceof Error) {
        const nextTokenGraph = markTokenGraphNodeDistributionMessage(
          tokenGraphRef.current,
          nodeId,
          error.message,
        );
        setTokenGraph(nextTokenGraph);
        tokenGraphRef.current = nextTokenGraph;
      }
      applyTransition(
        {
          nodes: syncFlowNodesWithTokenGraph(
            nodesRef.current.map((currentNode) =>
              currentNode.id === nodeId
                ? {
                    ...currentNode,
                    data: {
                      ...currentNode.data,
                      status: "ready",
                      distributionRequested: alternativesUnavailable
                        ? true
                        : currentNode.data.distributionRequested,
                      sourceAlternatives:
                        alternativesUnavailable ? [] : currentNode.data.sourceAlternatives,
                      distributionMessage:
                        alternativesUnavailable && error instanceof Error
                          ? error.message
                          : currentNode.data.distributionMessage,
                    },
                  }
                : currentNode,
            ),
            alternativesUnavailable ? tokenGraphRef.current : createEmptyTokenGraph(),
          ),
        },
        { pushHistory: false },
      );
      return false;
    }
  }

  async function expandStoredAlternatives(
    nodeId: string,
    options?: { pushHistory?: boolean },
  ) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (
      !node ||
      node.data.kind !== "token" ||
      !node.data.parentId ||
      node.data.topAlternatives.length === 0
    ) {
      return false;
    }

    setSelectedNodeId(nodeId);
    setContextMenu(null);

    const nextTokenGraph = materializeSourceAlternativesForNode(tokenGraphRef.current, nodeId);
    const parentNodeId = node.data.parentId;
    const parentNode = nodesRef.current.find((currentNode) => currentNode.id === parentNodeId);
    const parentRecord = nextTokenGraph.nodesById[parentNodeId];

    if (!parentNode || !parentRecord) {
      return false;
    }

    setTokenGraph(nextTokenGraph);
    tokenGraphRef.current = nextTokenGraph;

    const responseChildIds = new Set(parentRecord.childIds);
    const currentChildren = edgesRef.current
      .filter((edge) => edge.source === parentNodeId)
      .map((edge) => nodesRef.current.find((item) => item.id === edge.target))
      .filter((item): item is TokenFlowNode => Boolean(item));
    const staleNodeIds = new Set<string>();

    for (const child of currentChildren) {
      if (responseChildIds.has(child.id)) {
        continue;
      }

      staleNodeIds.add(child.id);
      for (const descendantId of collectDescendantIds(child.id, edgesRef.current)) {
        staleNodeIds.add(descendantId);
      }
    }

    const nextNodes = nodesRef.current.filter((currentNode) => !staleNodeIds.has(currentNode.id));
    const nextEdges = edgesRef.current.filter(
      (edge) =>
        !staleNodeIds.has(edge.source) &&
        !staleNodeIds.has(edge.target) &&
        (edge.source !== parentNodeId || responseChildIds.has(edge.target)),
    );
    const existingNodeIndexById = new Map(
      nextNodes.map((currentNode, index) => [currentNode.id, index]),
    );
    const materializedChildren = parentRecord.childIds
      .map((childId) => nextTokenGraph.nodesById[childId])
      .filter(
        (candidate): candidate is TokenGraphNodeRecord =>
          Boolean(candidate) && candidate.kind === "token" && candidate.parentId === parentNodeId,
      );

    for (const childRecord of materializedChildren) {
      const existingIndex = existingNodeIndexById.get(childRecord.id);

      if (typeof existingIndex === "number") {
        nextNodes[existingIndex] = applyTokenGraphRecordToFlowNode(nextNodes[existingIndex], childRecord);
      } else {
        nextNodes.push(
          applyTokenGraphRecordToFlowNode(
            buildFlowNodeFromRecord(childRecord, parentNode),
            childRecord,
          ),
        );
      }

      const nextEdge = buildEdge(
        parentNodeId,
        childRecord.id,
        childRecord.probability,
        childRecord.rank === 1,
      );
      const existingEdgeIndex = nextEdges.findIndex((edge) => edge.id === nextEdge.id);

      if (existingEdgeIndex >= 0) {
        nextEdges[existingEdgeIndex] = {
          ...nextEdges[existingEdgeIndex],
          data: nextEdge.data,
        };
      } else {
        nextEdges.push(nextEdge);
      }
    }

    applyTransition(
      {
        nodes: syncFlowNodesWithTokenGraph(nextNodes, nextTokenGraph),
        edges: nextEdges,
        selectedNodeId: nodeId,
      },
      options,
    );

    return true;
  }

  async function expandTokenOccurrence(
    nodeId: string,
    options?: { pushHistory?: boolean },
  ) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!node) {
      return false;
    }

    if (node.data.kind === "token" && node.data.topAlternatives.length > 0) {
      return expandStoredAlternatives(nodeId, options);
    }

    return expandContinuationNode(nodeId, options);
  }

  expandNodeRef.current = expandTokenOccurrence;

  async function requestContinuationStep(
    nodeId: string,
    validation: ReturnType<typeof buildContinuationValidation>,
  ) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!node || node.data.status === "loading") {
      return false;
    }

    applyTransition(
      {
        nodes: nodesRef.current.map((currentNode) =>
          currentNode.id === nodeId
            ? {
                ...currentNode,
                data: {
                  ...currentNode.data,
                  status: "loading",
                },
              }
            : currentNode,
        ),
        selectedNodeId: nodeId,
      },
      { pushHistory: false },
    );

    try {
      const response = await fetch("/api/continue-node", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          root_prompt: validation.rootPrompt,
          model: node.data.requestModel,
          preset: node.data.requestPreset,
          temperature: node.data.requestTemperature,
          top_p: node.data.requestTopP,
          parent_node_id: nodeId,
          parent_token: node.data.tokenText,
          assistant_prefix: validation.assistantPrefix,
          reconstructed_prompt: validation.reconstructedPrompt,
          expected_prompt_length: validation.characterLength,
          expected_utf8_length: validation.utf8Length,
          expected_token_count: validation.tokenCount,
          depth: node.data.depth,
          cumulative_probability: node.data.cumulativeProbability,
          variation: node.data.requestVariation,
          max_children: MAX_BRANCH_CHILDREN,
          demo_mode: node.data.requestDemoMode,
          cached_segment_id: readMetadataString(node.data.metadata, "cached_segment_id"),
          cached_token_index: readMetadataNumber(node.data.metadata, "next_cached_token_index"),
        }),
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as ContinueGenerationResponse;
      const parentNode = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

      if (!parentNode) {
        return false;
      }

      applyExpansionPayloadToCanvas(nodeId, parentNode, validation, payload, { pushHistory: false });
      setBackendState("online");
      return true;
    } catch (error) {
      const errorCode = getErrorCode(error);
      setBackendState(errorCode === "PROVIDER_REQUEST_FAILED" ? "offline" : "online");
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to continue the selected branch.",
      );
      applyTransition(
        {
          nodes: nodesRef.current.map((currentNode) =>
            currentNode.id === nodeId
              ? {
                  ...currentNode,
                  data: {
                    ...currentNode.data,
                    status: "ready",
                  },
                }
              : currentNode,
          ),
        },
        { pushHistory: false },
      );
      return false;
    }
  }

  async function continueGenerationFrom(nodeId: string, steps: number) {
    let currentNodeId = nodeId;

    for (let step = 0; step < steps; step += 1) {
      const existingTarget = getPreferredContinuationTarget(currentNodeId);

      if (existingTarget) {
        currentNodeId = existingTarget;
        activateReality(currentNodeId, { pushHistory: false });
        continue;
      }

      const validation = buildContinuationValidation(tokenGraphRef.current, currentNodeId);

      if (!validation.isValid) {
        setErrorMessage(validation.warnings.join(" "));
        setContinuationPreview({
          nodeId: currentNodeId,
          steps: Math.max(1, steps - step),
          validation,
        });
        break;
      }

      const expanded = await requestContinuationStep(currentNodeId, validation);

      if (!expanded) {
        break;
      }

      const nextTarget = getPreferredContinuationTarget(currentNodeId);

      if (!nextTarget) {
        break;
      }

      currentNodeId = nextTarget;
      activateReality(currentNodeId, { pushHistory: false });
    }

    pushHistorySnapshot(getCurrentSnapshot());
  }

  function togglePin(nodeId: string) {
    const nextPinned = pinnedNodeIdsRef.current.includes(nodeId)
      ? pinnedNodeIdsRef.current.filter((candidateId) => candidateId !== nodeId)
      : [...pinnedNodeIdsRef.current, nodeId];

    applyTransition(
      {
        pinnedNodeIds: nextPinned,
        selectedNodeId: nodeId,
      },
      { pushHistory: true },
    );
  }

  function deleteBranch(nodeId: string) {
    if (nodeId === "root") {
      return;
    }

    const removedIds = new Set([nodeId, ...collectDescendantIds(nodeId, edgesRef.current)]);
    const nextNodes = nodesRef.current.filter((node) => !removedIds.has(node.id));
    const nextEdges = edgesRef.current.filter(
      (edge) => !removedIds.has(edge.source) && !removedIds.has(edge.target),
    );
    const nextChoices = Object.fromEntries(
      Object.entries(branchChoicesRef.current).filter(
        ([parentId, childId]) => !removedIds.has(parentId) && !removedIds.has(childId),
      ),
    );

    applyTransition(
      {
        nodes: nextNodes,
        edges: nextEdges,
        branchChoices: nextChoices,
        selectedNodeId: nextNodes[0]?.id ?? null,
      },
      { pushHistory: true },
    );
  }

  function copyToken(nodeId: string) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!node) {
      return;
    }

    void navigator.clipboard.writeText(node.data.tokenText);
  }

  function copyBranch(nodeId: string) {
    const preview =
      nodeId === "root"
        ? tokenGraphRef.current.rootPrompt
        : reconstructAssistantPrefix(tokenGraphRef.current, nodeId);
    void navigator.clipboard.writeText(preview);
  }

  function openContinuationPreview(nodeId: string, steps: number) {
    const validation = buildContinuationValidation(tokenGraphRef.current, nodeId);
    setSelectedNodeId(nodeId);
    setContinuationPreview({
      nodeId,
      steps,
      validation,
    });
  }

  function startCompare(nodeId: string) {
    if (!compareLeftIdRef.current || (compareLeftIdRef.current && compareRightIdRef.current)) {
      applyTransition(
        {
          compareLeftId: nodeId,
          compareRightId: null,
          selectedNodeId: nodeId,
        },
        { pushHistory: false },
      );
      return;
    }

    if (compareLeftIdRef.current === nodeId) {
      return;
    }

    applyTransition(
      {
        compareLeftId: compareLeftIdRef.current,
        compareRightId: nodeId,
        selectedNodeId: nodeId,
      },
      { pushHistory: false },
    );
  }

  function focusBranch(nodeId: string) {
    centerNode(nodeId);
  }

  function collapseSubtree(nodeId: string) {
    applyTransition(
      {
        nodes: nodesRef.current.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  isCollapsed: true,
                },
              }
            : node,
        ),
        selectedNodeId: nodeId,
      },
      { pushHistory: true },
    );
  }

  collapseSubtreeRef.current = collapseSubtree;

  async function handleSubmit() {
    const promptValue = prompt;

    if (!promptValue.trim()) {
      setErrorMessage("Enter a prompt.");
      return;
    }

    animationRunRef.current += 1;
    const nextVariation = requestVariation + 1;
    const loadingRoot = buildPromptNode({
      prompt: promptValue,
      model,
      preset,
      reasoning: {
        notes: "",
        intent: "",
        strategy: "",
        focusTerms: [],
      },
      responseMode: "pending",
      status: "loading",
      temperature,
      topP,
      variation: nextVariation,
      demoMode,
    });

    setGraphNodes([loadingRoot]);
    setGraphEdges([]);
    setTokenGraph(clearTokenGraph());
    setBranchChoices({});
    setPinnedNodeIds([]);
    setCompareLeftId(null);
    setCompareRightId(null);
    setSelectedNodeId("root");
    nodesRef.current = [loadingRoot];
    edgesRef.current = [];
    tokenGraphRef.current = createEmptyTokenGraph();
    branchChoicesRef.current = {};
    pinnedNodeIdsRef.current = [];
    compareLeftIdRef.current = null;
    compareRightIdRef.current = null;
    selectedNodeIdRef.current = "root";
    setTypedCompletion("");
    setErrorMessage(null);
    setIsGenerating(true);
    setIsSentenceBarExpanded(true);
    setRequestVariation(nextVariation);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: promptValue,
          model,
          preset,
          temperature,
          top_p: topP,
          max_tokens: maxTokens,
          variation: nextVariation,
          demo_mode: demoMode,
        }),
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as GenerationResponse;
      setGeneration(payload);
      setBackendState("online");
      setIsDockCollapsed(true);
      await playMainPath(payload);
    } catch (error) {
      const errorCode = getErrorCode(error);
      setBackendState(errorCode === "PROVIDER_REQUEST_FAILED" ? "offline" : "online");
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to generate a response.",
      );
      setHistory([]);
      setHistoryIndex(-1);
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleReplay() {
    if (!generation) {
      return;
    }

    await playMainPath(generation);
  }

  async function handleExpandAll() {
    const visibleExpandableNodes = displayNodes.filter(
      (node) =>
        !node.hidden &&
        node.id !== "root" &&
        (node.data.topAlternatives.length > 0
          ? !node.data.alternativesExpanded
          : !node.data.distributionRequested),
    );

    for (const node of visibleExpandableNodes) {
      await expandTokenOccurrence(node.id, { pushHistory: false });
    }

    pushHistorySnapshot(getCurrentSnapshot());
  }

  function handleCollapseAll() {
    applyTransition(
      {
        nodes: nodesRef.current.map((node) =>
          node.id === "root"
            ? node
            : {
                ...node,
                data: {
                  ...node.data,
                  isCollapsed: node.data.childCount > 0,
                },
              },
        ),
      },
      { pushHistory: true },
    );
  }

  function handleResetView() {
    void flowRef.current?.fitView({
      duration: 520,
      padding: 0.18,
    });
  }

  function handleExportJson() {
    downloadFile(
      `llmscope-${Date.now()}.json`,
      new Blob(
        [
          JSON.stringify(
            {
              prompt,
              generation,
              graph: {
                nodes: graphNodes,
                edges: graphEdges,
                branchChoices,
                pinnedNodeIds,
                compareLeftId,
                compareRightId,
              },
            },
            null,
            2,
          ),
        ],
        { type: "application/json" },
      ),
    );
  }

  function handleExportSvg() {
    const svgMarkup = buildSvgExport(displayNodes, displayEdges, activeEdgeIdSet);

    if (!svgMarkup) {
      return;
    }

    downloadFile(
      `llmscope-${Date.now()}.svg`,
      new Blob([svgMarkup], { type: "image/svg+xml;charset=utf-8" }),
    );
  }

  async function handleScreenshot() {
    const svgMarkup = buildSvgExport(displayNodes, displayEdges, activeEdgeIdSet);

    if (!svgMarkup) {
      return;
    }

    const svgBlob = new Blob([svgMarkup], { type: "image/svg+xml;charset=utf-8" });
    const objectUrl = URL.createObjectURL(svgBlob);
    const image = new Image();

    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.width * 2;
      canvas.height = image.height * 2;
      const context = canvas.getContext("2d");

      if (!context) {
        URL.revokeObjectURL(objectUrl);
        return;
      }

      context.scale(2, 2);
      context.drawImage(image, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          downloadFile(`llmscope-${Date.now()}.png`, blob);
        }

        URL.revokeObjectURL(objectUrl);
      });
    };

    image.src = objectUrl;
  }

  function jumpToSearchResult(direction: 1 | -1) {
    if (searchMatches.length === 0) {
      return;
    }

    const nextIndex =
      ((searchResultIndex + direction) % searchMatches.length + searchMatches.length) %
      searchMatches.length;
    const targetNodeId = searchMatches[nextIndex].id;

    setSearchResultIndex(nextIndex);
    centerNode(targetNodeId);
    setSelectedNodeId(targetNodeId);
  }

  async function ensureInspectorAlternativeNode(
    sourceNodeId: string,
    alternative: InspectorAlternative,
  ) {
    if (
      alternative.nodeId &&
      nodesRef.current.some((node) => node.id === alternative.nodeId)
    ) {
      return alternative.nodeId;
    }

    const sourceNode = nodesRef.current.find((node) => node.id === sourceNodeId);

    if (!sourceNode || sourceNode.data.kind !== "token" || !sourceNode.data.parentId) {
      return null;
    }

    const parentNodeId = sourceNode.data.parentId;
    const expanded = await expandTokenOccurrence(sourceNodeId, { pushHistory: false });

    if (!expanded) {
      return null;
    }

    const parentChildren = edgesRef.current
      .filter((edge) => edge.source === parentNodeId)
      .map((edge) => nodesRef.current.find((candidate) => candidate.id === edge.target))
      .filter((candidate): candidate is TokenFlowNode => Boolean(candidate));

    return (
      parentChildren.find((candidate) => matchesInspectorAlternativeNode(candidate, alternative))
        ?.id ?? null
    );
  }

  async function handleInspectorAlternativeSelect(
    sourceNodeId: string,
    alternative: InspectorAlternative,
  ) {
    const targetNodeId = await ensureInspectorAlternativeNode(sourceNodeId, alternative);

    if (!targetNodeId) {
      return;
    }

    activateReality(targetNodeId, { pushHistory: true });
    centerNode(targetNodeId);
  }

  const handleNodeContextMenu = useCallback((event: React.MouseEvent, node: TokenFlowNode) => {
    event.preventDefault();
    setSelectedNodeId(node.id);
    setContextMenu({
      nodeId: node.id,
      title: node.data.kind === "prompt" ? "Prompt" : node.data.displayTokenText,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const handleNodeClick = useCallback<NodeMouseHandler<TokenFlowNode>>((event, node) => {
    setContextMenu(null);

    if (event.shiftKey && node.data.childCount > 0) {
      event.preventDefault();
      collapseSubtreeRef.current(node.id);
      return;
    }

    activateRealityRef.current(node.id, { pushHistory: true });
    centerNodeRef.current(node.id);
  }, []);

  const handleNodeDoubleClick = useCallback<NodeMouseHandler<TokenFlowNode>>((event, node) => {
    event.preventDefault();
    setContextMenu(null);
    void expandNodeRef.current(node.id, { pushHistory: true });
  }, []);

  const handleEdgesChange = useCallback((changes: EdgeChange<ProbabilityFlowEdge>[]) => {
    const nextEdges = applyEdgeChanges<ProbabilityFlowEdge>(changes, edgesRef.current);
    commitTransientEdges(nextEdges);
  }, [commitTransientEdges]);

  const handleViewportMoveEnd = useCallback(
    (_event: unknown, nextViewport: { x: number; y: number; zoom: number }) => {
      setViewport((currentViewport) =>
        currentViewport.x === nextViewport.x &&
        currentViewport.y === nextViewport.y &&
        currentViewport.zoom === nextViewport.zoom
          ? currentViewport
          : nextViewport,
      );
    },
    [],
  );

  const handleNodeDragStart = useCallback(() => {
    dragPerformanceRef.current = {
      sampleCount: 0,
      totalMs: 0,
    };
    setGraphInteractionActive(true);
  }, [setGraphInteractionActive]);

  const handleNodeDragStop = useCallback(
    (_event: MouseEvent | TouchEvent, node: TokenFlowNode) => {
      setGraphInteractionActive(false);
      const dragMetrics = dragPerformanceRef.current;

      if (dragMetrics.sampleCount > 0) {
        logCanvasPerformance("drag-summary", {
          averageFrameMs: Number((dragMetrics.totalMs / dragMetrics.sampleCount).toFixed(2)),
          sampleCount: dragMetrics.sampleCount,
        });
      }

      applyTransitionRef.current(
        {
          nodes: nodesRef.current.map((currentNode) =>
            currentNode.id === node.id
              ? {
                  ...currentNode,
                  position: node.position,
                }
              : currentNode,
          ),
        },
        { pushHistory: true, skipLayout: true },
      );
    },
    [setGraphInteractionActive],
  );

  const handleNodeMouseEnter = useCallback((_event: React.MouseEvent, node: TokenFlowNode) => {
    setHoveredNodeId(node.id);
  }, []);

  const handleNodeMouseLeave = useCallback(() => {
    setHoveredNodeId(null);
  }, []);

  const handleNodesChange = useCallback((changes: NodeChange<TokenFlowNode>[]) => {
    const changeStart = performance.now();
    const nextNodes = applyNodeChanges<TokenFlowNode>(changes, nodesRef.current);
    commitTransientNodes(nextNodes);

    const isDragging = changes.some(
      (change) => change.type === "position" && "dragging" in change && Boolean(change.dragging),
    );

    if (!isDragging) {
      return;
    }

    const duration = performance.now() - changeStart;
    dragPerformanceRef.current = {
      sampleCount: dragPerformanceRef.current.sampleCount + 1,
      totalMs: dragPerformanceRef.current.totalMs + duration,
    };

    if (duration > FRAME_BUDGET_MS) {
      logCanvasPerformance("drag-frame", {
        durationMs: Number(duration.toFixed(2)),
        nodeCount: nextNodes.length,
      });
    }
  }, [commitTransientNodes]);

  const handlePaneClick = useCallback(() => {
    setContextMenu(null);
  }, []);

  const currentStatus = isGenerating
    ? "Generating"
    : isReplaying
      ? "Replaying"
      : selectedNode
        ? "Exploring"
        : "Ready";
  const continuationPreviewNode = continuationPreview
    ? tokenGraph.nodesById[continuationPreview.nodeId] ?? null
    : null;
  const continuationGenerationLineage =
    continuationPreviewNode
      ? getGenerationLineage(tokenGraph, continuationPreviewNode.generationId)
      : [];
  const backgroundOpacity = Math.max(0.035, 0.16 - Math.max(viewport.zoom - 0.8, 0) * 0.06);
  const contextMenuActions = contextMenu
    ? [
        {
          label: "Copy token",
          onSelect: () => {
            copyToken(contextMenu.nodeId);
            setContextMenu(null);
          },
        },
        {
          label: "Copy branch",
          onSelect: () => {
            copyBranch(contextMenu.nodeId);
            setContextMenu(null);
          },
        },
        {
          label: "Continue generation",
          onSelect: () => {
            openContinuationPreview(contextMenu.nodeId, 1);
            setContextMenu(null);
          },
        },
        {
          label: "Generate deeper",
          onSelect: () => {
            openContinuationPreview(contextMenu.nodeId, 4);
            setContextMenu(null);
          },
        },
        {
          label: "Compare branches",
          onSelect: () => {
            startCompare(contextMenu.nodeId);
            setContextMenu(null);
          },
        },
        {
          label: pinnedSet.has(contextMenu.nodeId) ? "Unpin branch" : "Pin branch",
          onSelect: () => {
            togglePin(contextMenu.nodeId);
            setContextMenu(null);
          },
        },
        {
          label: "Focus branch",
          onSelect: () => {
            focusBranch(contextMenu.nodeId);
            setContextMenu(null);
          },
        },
        {
          label: "Expand futures",
          onSelect: () => {
            void expandTokenOccurrence(contextMenu.nodeId, { pushHistory: true });
            setContextMenu(null);
          },
        },
        {
          label: "Collapse subtree",
          onSelect: () => {
            collapseSubtree(contextMenu.nodeId);
            setContextMenu(null);
          },
          disabled: !displayNodes.find((node) => node.id === contextMenu.nodeId)?.data.childCount,
        },
        {
          label: "Delete branch",
          destructive: true,
          disabled: contextMenu.nodeId === "root",
          onSelect: () => {
            deleteBranch(contextMenu.nodeId);
            setContextMenu(null);
          },
        },
      ]
    : [];

  return (
    <div className={`llmscope-app${isDockCollapsed ? " llmscope-app--dock-collapsed" : ""}`}>
      <div className="llmscope-status">
        <span
          className={`llmscope-status__dot${
            backendState === "online"
              ? " llmscope-status__dot--online"
              : backendState === "offline"
                ? " llmscope-status__dot--offline"
                : ""
          }`}
        />
        {currentStatus}
      </div>

      {errorMessage ? <div className="llmscope-error">{errorMessage}</div> : null}

      {continuationPreview ? (
        <div className="continuation-preview">
          <div className="continuation-preview__header">
            <div>
              <p className="continuation-preview__eyebrow">Continuation check</p>
              <h2 className="continuation-preview__title">
                {continuationPreview.steps === 1 ? "Continue branch" : `Continue ${continuationPreview.steps} steps`}
              </h2>
            </div>
            <button
              className="icon-button"
              onClick={() => setContinuationPreview(null)}
              type="button"
            >
              Close
            </button>
          </div>

          <div className="continuation-preview__panel">
            <p className="continuation-preview__label">Original prompt</p>
            <pre className="continuation-preview__code continuation-preview__code--prompt">
              {continuationPreview.validation.rootPrompt}
            </pre>

            <p className="continuation-preview__label">Reconstructed continuation prefix</p>
            <pre className="continuation-preview__code continuation-preview__code--continuation">
              {continuationPreview.validation.assistantPrefix || "<empty>"}
            </pre>

            <div className="continuation-preview__stats">
              <span>{`${continuationPreview.validation.tokenCount} tokens`}</span>
              <span>{`${continuationPreview.validation.characterLength} chars`}</span>
              <span>{`${continuationPreview.validation.utf8Length} bytes`}</span>
            </div>

            {continuationPreview.validation.expectedAssistantPrefix !== null ? (
              <>
                <p className="continuation-preview__label">Expected assistant prefix</p>
                <pre className="continuation-preview__code">
                  {continuationPreview.validation.expectedAssistantPrefix || "<empty>"}
                </pre>
              </>
            ) : null}

            {continuationPreview.validation.warnings.length > 0 ? (
              <div className="continuation-preview__warnings">
                {continuationPreview.validation.warnings.map((warning) => (
                  <p key={warning}>{warning}</p>
                ))}
              </div>
            ) : (
              <div className="continuation-preview__ok">
                Raw-token reconstruction matches the stored graph context.
              </div>
            )}

            {continuationPreviewNode ? (
              <div className="continuation-preview__meta">
                <span>{continuationPreviewNode.requestModel}</span>
                <span>{`temp ${continuationPreviewNode.requestTemperature}`}</span>
                <span>{`top_p ${continuationPreviewNode.requestTopP}`}</span>
                <span>{continuationPreviewNode.requestDemoMode ? "demo" : "live"}</span>
                {continuationPreviewNode.finishReason ? (
                  <span>{continuationPreviewNode.finishReason}</span>
                ) : null}
              </div>
            ) : null}

            {continuationGenerationLineage.length > 0 ? (
              <div className="continuation-preview__history">
                <p className="continuation-preview__label">Generation history</p>
                <div className="continuation-preview__history-list">
                  {continuationGenerationLineage.map((item) => (
                    <span key={item.id}>
                      {`${item.model} · ${item.timestamp}`}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="continuation-preview__actions">
            <button
              className="explorer-button explorer-button--ghost"
              onClick={() => setContinuationPreview(null)}
              type="button"
            >
              Cancel
            </button>
            <button
              className="explorer-button explorer-button--primary"
              disabled={!continuationPreview.validation.isValid}
              onClick={() => {
                const pending = continuationPreview;
                setContinuationPreview(null);
                void continueGenerationFrom(pending.nodeId, pending.steps);
              }}
              type="button"
            >
              Continue
            </button>
          </div>
        </div>
      ) : null}

      <div className="top-toolbar">
        <div className="top-toolbar__search">
          <Search className="h-4 w-4" />
          <input
            ref={searchInputRef}
            className="top-toolbar__search-input"
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search tokens"
            value={searchQuery}
          />
          {searchMatches.length > 0 ? (
            <span className="top-toolbar__search-count">
              {searchResultIndex + 1}/{searchMatches.length}
            </span>
          ) : null}
        </div>

        <div className="top-toolbar__actions">
          <button className="tool-button" onClick={() => jumpToSearchResult(-1)} type="button">
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button className="tool-button" onClick={() => jumpToSearchResult(1)} type="button">
            <ChevronRight className="h-4 w-4" />
          </button>
          <button className="tool-button" disabled={historyIndex <= 0} onClick={undoHistory} type="button">
            <Undo2 className="h-4 w-4" />
          </button>
          <button
            className="tool-button"
            disabled={historyIndex >= history.length - 1}
            onClick={redoHistory}
            type="button"
          >
            <Redo2 className="h-4 w-4" />
          </button>
          <button className="tool-button" onClick={handleResetView} type="button">
            Center
          </button>
          <button className="tool-button" onClick={() => void handleExpandAll()} type="button">
            Expand all
          </button>
          <button className="tool-button" onClick={handleCollapseAll} type="button">
            Collapse all
          </button>
          <button className="tool-button" onClick={handleScreenshot} type="button">
            <Camera className="h-4 w-4" />
          </button>
          <button className="tool-button" onClick={handleExportJson} type="button">
            JSON
          </button>
          <button className="tool-button" onClick={handleExportSvg} type="button">
            SVG
          </button>
          <button className="tool-button" onClick={() => void handleReplay()} type="button">
            {isReplaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>
          <select
            className="tool-select"
            onChange={(event) => setPlaybackSpeed(Number(event.target.value))}
            value={playbackSpeed}
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
          </select>
          <details className="tool-popover">
            <summary className="tool-button">Settings</summary>
            <div className="tool-popover__panel">
              <div className="tool-popover__section">
                <p className="tool-popover__label">Probability display</p>
                <button
                  className={`tool-popover__option${
                    probabilityViewMode === "normalized" ? " tool-popover__option--active" : ""
                  }`}
                  onClick={() => setProbabilityViewMode("normalized")}
                  type="button"
                >
                  <span>Normalized</span>
                  <small>Displayed siblings always sum to 100%.</small>
                </button>
                <button
                  className={`tool-popover__option${
                    probabilityViewMode === "raw" ? " tool-popover__option--active" : ""
                  }`}
                  onClick={() => setProbabilityViewMode("raw")}
                  type="button"
                >
                  <span>Raw</span>
                  <small>Show true model mass and the remaining probability.</small>
                </button>
              </div>
            </div>
          </details>
          <button
            className={`tool-button${demoMode ? " tool-button--active" : ""}`}
            onClick={() => setDemoMode((currentValue) => !currentValue)}
            type="button"
          >
            Demo
          </button>
          <button
            className="tool-button"
            onClick={() =>
              setSurfaceTheme((currentTheme) =>
                currentTheme === "midnight" ? "graphite" : "midnight",
              )
            }
            type="button"
          >
            Theme
          </button>
        </div>
      </div>

      <CurrentRealityPanel
        collapsed={isSentenceBarCollapsed}
        continuationModeLabel={currentRealityContinuationMode.label}
        continuationModeTitle={currentRealityContinuationMode.title}
        continuationModeTone={currentRealityContinuationMode.tone}
        hasContent={hasSentenceContent}
        onSelectToken={(nodeId) => {
          activateReality(nodeId, { pushHistory: true });
          centerNode(nodeId);
        }}
        onToggleCollapse={() => setIsSentenceBarExpanded((currentValue) => !currentValue)}
        probabilityMode={probabilityViewMode}
        remainingProbabilityMass={
          activeSentenceNodes[activeSentenceNodes.length - 1]?.data.remainingProbabilityMass ?? 0
        }
        selectedTokenId={
          selectedNode && activeSentenceNodes.some((node) => node.id === selectedNode.id)
            ? selectedNode.id
            : null
        }
        stats={currentRealityStats}
        summary={
          selectedNode?.data.kind === "prompt"
            ? "Expand the prompt to see the first decision."
            : naturalReason
        }
        text={currentSentenceText}
        tokens={currentRealityTokens}
        topKCoverage={
          activeSentenceNodes[activeSentenceNodes.length - 1]?.data.probabilityCoverage ?? 0
        }
      />

      {compareNodes[0] && compareNodes[1] ? (
        <div className="compare-strip">
          {[compareNodes[0], compareNodes[1]].map((node, index) => (
            <div key={node?.id ?? `compare-${index}`} className="compare-strip__panel">
              <p className="compare-strip__label">{index === 0 ? "A" : "B"}</p>
              <div className="compare-strip__tokens">
                {splitPreview(compareSentences[index] ?? "").map((token, tokenIndex) => (
                  <span
                    key={`${node?.id ?? "compare"}-${tokenIndex}`}
                    className={`compare-strip__token${
                      compareDivergenceIndex >= 0 && tokenIndex >= compareDivergenceIndex
                        ? " compare-strip__token--changed"
                        : ""
                    }`}
                  >
                    {token}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      <aside className={`explorer-dock${isDockCollapsed ? " explorer-dock--collapsed" : ""}`}>
        <div className="explorer-dock__header">
          <div>
            <p className="explorer-dock__eyebrow">Generation</p>
            <h1 className="explorer-dock__title">LLMScope</h1>
            {!isDockCollapsed ? (
              <p className="explorer-dock__subtitle">
                Ask a prompt, then shrink this panel and explore why each token won.
              </p>
            ) : null}
          </div>

          <div className="explorer-dock__header-actions">
            <button
              className="icon-button"
              onClick={() => void refreshHealth()}
              type="button"
            >
              <RefreshCcw className={isCheckingHealth ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
            </button>
            <button
              className="icon-button"
              onClick={() => setIsDockCollapsed((currentValue) => !currentValue)}
              type="button"
            >
              {isDockCollapsed ? "+" : "-"}
            </button>
          </div>
        </div>

        {isDockCollapsed ? (
          <div className="explorer-dock__collapsed-actions">
            <button className="explorer-button explorer-button--primary" onClick={() => setIsDockCollapsed(false)} type="button">
              Open
            </button>
            <button className="explorer-button explorer-button--ghost" onClick={() => void handleReplay()} type="button">
              Replay
            </button>
            <button className="explorer-button explorer-button--ghost" onClick={handleResetView} type="button">
              Fit
            </button>
          </div>
        ) : (
          <>
            <textarea
              aria-label="Prompt"
              className="explorer-textarea"
              onChange={(event) => {
                setPrompt(event.target.value);
                setErrorMessage(null);
              }}
              placeholder="Ask something"
              value={prompt}
            />

            <div className="explorer-grid">
              <select
                aria-label="Model"
                className="explorer-input"
                disabled={isLoadingModels}
                onChange={(event) => {
                  setModel(event.target.value);
                  setErrorMessage(null);
                }}
                value={model}
              >
                {models.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>

              <select
                aria-label="Mode"
                className="explorer-input"
                disabled={isLoadingModels}
                onChange={(event) => {
                  setPreset(event.target.value);
                  setErrorMessage(null);
                }}
                value={preset}
              >
                {presets.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="explorer-grid explorer-grid--compact">
              <input
                aria-label="Temperature"
                className="explorer-input"
                max={2}
                min={0}
                onChange={(event) => setTemperature(Number(event.target.value))}
                step={0.1}
                type="number"
                value={temperature}
              />

              <input
                aria-label="Top p"
                className="explorer-input"
                max={1}
                min={0}
                onChange={(event) => setTopP(Number(event.target.value))}
                step={0.05}
                type="number"
                value={topP}
              />
            </div>

            <div className="explorer-grid explorer-grid--compact">
              <input
                aria-label="Max tokens"
                className="explorer-input"
                max={4096}
                min={1}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
                type="number"
                value={maxTokens}
              />

              <button
                className="explorer-button explorer-button--primary"
                disabled={isGenerating || isLoadingModels}
                onClick={() => void handleSubmit()}
                type="button"
              >
                {isGenerating ? (
                  <>
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : isLoadingModels ? (
                  <>
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    Loading...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Generate
                  </>
                )}
              </button>
            </div>

            <div className="explorer-chip-row">
              <span className="explorer-chip">{generatedModelOption.label}</span>
              <span className="explorer-chip">{generatedPresetOption.label}</span>
              {generation?.request.demo_mode ? (
                <span className="explorer-chip">Demo data</span>
              ) : null}
              <span className="explorer-chip">
                {backendState === "online"
                  ? "Online"
                  : backendState === "offline"
                    ? "Offline"
                    : "Checking"}
              </span>
            </div>
          </>
        )}
      </aside>

      <aside className="inspector-panel">
        <div className="inspector-panel__header">
          <div>
            <p className="inspector-panel__eyebrow">Inspector</p>
            <h2 className="inspector-panel__title">
              {selectedNode?.data.kind === "prompt"
                ? "Prompt"
                : selectedNode?.data.displayTokenText ?? "Select a token"}
            </h2>
          </div>
        </div>

        {selectedNode && selectedRecord ? (
          <div className="inspector-panel__content">
            <div className="inspector-section">
              <p className="inspector-section__label">Basic</p>
              <div className="inspector-grid-data">
                <div>
                  <dt>Chosen token</dt>
                  <dd>{selectedRecord.displayToken}</dd>
                </div>
                <div>
                  <dt>Probability</dt>
                  <dd>{formatPercent(selectedNode.data.displayProbability)}</dd>
                </div>
                <div>
                  <dt>Mode</dt>
                  <dd>{getProbabilityModeLabel(probabilityViewMode)}</dd>
                </div>
                <div>
                  <dt>Continuation</dt>
                  <dd>
                    <span
                      className={`inspector-inline-badge inspector-inline-badge--${inspectorContinuationMode.tone}`}
                      title={inspectorContinuationMode.title ?? undefined}
                    >
                      {`Mode: ${inspectorContinuationMode.label}`}
                    </span>
                  </dd>
                </div>
                <div>
                  <dt>Rank</dt>
                  <dd>{selectedRecord.rank}</dd>
                </div>
                <div>
                  <dt>Entropy</dt>
                  <dd>{formatNumber(selectedRecord.entropy)}</dd>
                </div>
                <div>
                  <dt>Latency</dt>
                  <dd>{`${selectedRecord.latencyMs} ms`}</dd>
                </div>
                <div>
                  <dt>Top-K coverage</dt>
                  <dd>{inspectorAlternatives.length > 0 ? formatPercent(inspectorCoverage) : "-"}</dd>
                </div>
              </div>
            </div>

            <div className="inspector-section">
              <p className="inspector-section__label">Context</p>
              <div className="inspector-block inspector-block--stack">
                <p>
                  <strong>Before token</strong>
                  <span>{selectedRecord.contextBefore || "<empty>"}</span>
                </p>
                <p>
                  <strong>Through token</strong>
                  <span>{selectedRecord.cumulativeDecodedText || "<empty>"}</span>
                </p>
                <p>
                  <strong>Interpretation</strong>
                  <span>{naturalReason}</span>
                </p>
              </div>
            </div>

            <div className="inspector-section">
              <div className="inspector-section__heading">
                <p className="inspector-section__label">Top alternatives</p>
                <div className="inspector-section__meta">
                  <span className="inspector-inline-badge">
                    {getProbabilityModeLabel(probabilityViewMode)}
                  </span>
                  {probabilityViewMode === "raw" ? (
                    <span className="inspector-inline-badge">
                      Other tokens {formatPercent(inspectorRemainingProbabilityMass)}
                    </span>
                  ) : null}
                </div>
              </div>
              {inspectorAlternatives.length > 0 ? (
                <div className="alternatives-list">
                  {inspectorAlternatives.map((alternative) => (
                    <button
                      key={`${selectedNode.id}-${alternative.predictionId ?? alternative.nodeId ?? alternative.tokenIndex ?? alternative.token}`}
                      className={`alternative-row${alternative.isChosen ? " alternative-row--chosen" : ""}`}
                      onClick={() => handleInspectorAlternativeSelect(selectedNode.id, alternative)}
                      type="button"
                    >
                      <span className="alternative-row__token">
                        {alternative.displayToken ?? alternative.token}
                      </span>
                      <span className="alternative-row__probability">
                        {formatPercent(alternative.displayProbability)}
                      </span>
                      <span className="alternative-row__difference">
                        {alternative.isChosen ? "Chosen" : formatSignedPercent(alternative.difference)}
                      </span>
                    </button>
                  ))}
                </div>
              ) : selectedNode.data.distributionMessage ? (
                <div className="inspector-empty">No alternatives were returned for this token.</div>
              ) : (
                <div className="inspector-empty">Expand this branch to inspect competing futures.</div>
              )}
            </div>

            <details className="inspector-details" open={false}>
              <summary>Advanced</summary>
              <div className="inspector-section">
                <div className="inspector-grid-data">
                  <div>
                    <dt>Raw token</dt>
                    <dd>{selectedRecord.rawToken || "∅"}</dd>
                  </div>
                  <div>
                    <dt>Decoded token</dt>
                    <dd>{selectedRecord.decodedContribution || "∅"}</dd>
                  </div>
                  <div>
                    <dt>Token id</dt>
                    <dd>{selectedRecord.tokenId ?? "-"}</dd>
                  </div>
                  <div>
                    <dt>Tokenizer id</dt>
                    <dd>{selectedRecord.tokenizerId ?? "-"}</dd>
                  </div>
                  <div>
                    <dt>Raw probability</dt>
                    <dd>{formatProbability(selectedRecord.rawProbability)}</dd>
                  </div>
                  <div>
                    <dt>Log probability</dt>
                    <dd>{formatNumber(selectedRecord.logProbability)}</dd>
                  </div>
                  <div>
                    <dt>Cumulative probability</dt>
                    <dd>{formatProbability(selectedRecord.cumulativeProbability)}</dd>
                  </div>
                  <div>
                    <dt>Cumulative logprob</dt>
                    <dd>{formatNumber(selectedRecord.cumulativeLogProbability)}</dd>
                  </div>
                  <div>
                    <dt>Branch depth</dt>
                    <dd>{selectedRecord.generationDepth}</dd>
                  </div>
                  <div>
                    <dt>Parent id</dt>
                    <dd>{selectedRecord.parentId ?? "-"}</dd>
                  </div>
                  <div>
                    <dt>Generation step</dt>
                    <dd>{selectedRecord.generationStep}</dd>
                  </div>
                  <div>
                    <dt>UTF-8 length</dt>
                    <dd>{selectedNodeMetrics?.utf8Length ?? 0}</dd>
                  </div>
                  <div>
                    <dt>Leading whitespace</dt>
                    <dd>{selectedNodeMetrics?.leadingWhitespaceCount ?? 0}</dd>
                  </div>
                  <div>
                    <dt>Trailing whitespace</dt>
                    <dd>{selectedNodeMetrics?.trailingWhitespaceCount ?? 0}</dd>
                  </div>
                </div>
                <div className="inspector-block inspector-block--stack">
                  <p>
                    <strong>Byte representation</strong>
                    <span>{selectedRecord.tokenBytes.length > 0 ? selectedRecord.tokenBytes.join(" ") : "Unavailable"}</span>
                  </p>
                  <p>
                    <strong>Metadata</strong>
                    <span>
                      {Object.entries(selectedRecord.metadata).length > 0
                        ? Object.entries(selectedRecord.metadata)
                            .map(([key, value]) => `${key}: ${String(value)}`)
                            .join(" · ")
                        : "Unavailable"}
                    </span>
                  </p>
                  <p>
                    <strong>Notes</strong>
                    <span>{selectedNode.data.sourceNotes || "Unavailable"}</span>
                  </p>
                </div>
              </div>
            </details>

            {pinnedNodeIds.length > 0 ? (
              <div className="inspector-section">
                <p className="inspector-section__label">Pinned branches</p>
                <div className="alternatives-list">
                  {pinnedNodeIds.map((nodeId) => {
                    const pinnedNode = displayNodeMap.get(nodeId);

                    if (!pinnedNode) {
                      return null;
                    }

                    return (
                      <button
                        key={nodeId}
                        className="alternative-row"
                        onClick={() => {
                          activateReality(nodeId, { pushHistory: true });
                          centerNode(nodeId);
                        }}
                        type="button"
                      >
                        <span className="alternative-row__token">
                          {pinnedNode.data.displayTokenText}
                        </span>
                        <span className="alternative-row__probability">
                          {formatPercent(pinnedNode.data.displayProbability)}
                        </span>
                        <span className="alternative-row__difference">Jump</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="inspector-empty">Select a token to inspect why it won.</div>
        )}
      </aside>

      <div className="canvas-hint">
        click to switch reality • double-click to expand • shift+click to collapse • cmd/ctrl+f to search • drag the canvas to pan
      </div>

      <ReactFlow<TokenFlowNode, ProbabilityFlowEdge>
        className={isGraphInteracting ? "llmscope-flow llmscope-flow--interacting" : "llmscope-flow"}
        colorMode="dark"
        defaultEdgeOptions={{
          animated: true,
          type: "probabilityEdge",
        }}
        edgeTypes={edgeTypes}
        edges={displayEdges}
        fitView
        fitViewOptions={{
          padding: 0.18,
        }}
        maxZoom={1.9}
        minZoom={0.16}
        nodeClickDistance={8}
        nodeTypes={nodeTypes}
        nodes={displayNodes}
        onEdgesChange={handleEdgesChange}
        onInit={(instance) => {
          flowRef.current = instance;
        }}
        onMoveEnd={handleViewportMoveEnd}
        onNodeClick={handleNodeClick}
        onNodeContextMenu={handleNodeContextMenu}
        onNodeDoubleClick={handleNodeDoubleClick}
        onNodeDragStart={handleNodeDragStart}
        onNodeDragStop={handleNodeDragStop}
        onNodeMouseEnter={handleNodeMouseEnter}
        onNodeMouseLeave={handleNodeMouseLeave}
        onNodesChange={handleNodesChange}
        onPaneClick={handlePaneClick}
        panOnDrag
        selectNodesOnDrag={false}
        selectionOnDrag={false}
        zoomOnDoubleClick={false}
      >
        <Background
          color={`rgba(148, 163, 184, ${backgroundOpacity})`}
          gap={26}
          size={1}
          variant={BackgroundVariant.Lines}
        />
        <MiniMap<TokenFlowNode> nodeColor={getMiniMapColor} pannable zoomable />
        <Controls showInteractive={false} />
      </ReactFlow>

      {contextMenu ? (
        <TokenContextMenu
          actions={contextMenuActions}
          onClose={() => {
            setContextMenu(null);
          }}
          title={contextMenu.title}
          x={contextMenu.x}
          y={contextMenu.y}
        />
      ) : null}
    </div>
  );
}

export function LlmScopeCanvas() {
  return (
    <ReactFlowProvider>
      <Workspace />
    </ReactFlowProvider>
  );
}
