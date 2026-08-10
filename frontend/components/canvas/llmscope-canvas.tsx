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
  Search,
  Undo2,
} from "lucide-react";

import {
  CurrentRealityPanel,
  type CurrentRealityAttentionTokenItem,
  type CurrentRealityConversationSection,
  type CurrentRealityFormattingSelection,
  type CurrentRealityGroupedTokenItem,
  type CurrentRealityStats,
  type CurrentRealitySummaryItem,
  type CurrentRealityTokenItem,
  type CurrentRealityTokenGroup,
} from "@/components/canvas/current-reality-panel";
import {
  GenerationPanel,
  type GenerationPanelSystemPromptState,
} from "@/components/canvas/generation-panel";
import { AttentionEdge } from "@/components/canvas/attention-edge";
import { ProbabilityEdge } from "@/components/canvas/probability-edge";
import { TokenContextMenu } from "@/components/canvas/token-context-menu";
import { TokenNode } from "@/components/canvas/token-node";
import type {
  AttentionFlowEdge,
  CanvasFlowEdge,
  InspectorAlternative,
  ProbabilityFlowEdge,
  ProbabilityViewMode,
  TokenFlowNode,
  TokenNodeData,
} from "@/components/canvas/types";
import {
  buildAttentionCacheKey,
  buildDeterministicFocusViewport,
  buildAttentionHeadLabel,
  buildAttentionOverlayEdges,
  buildAttentionRequestPayload,
  buildAttentionStripTokens,
  buildAttentionTokenId,
  buildPromptTokenNodeId,
  canUseAttentionLens,
  canMutateGraphTokenNode,
  isPromptTokenNodeId,
  layoutPromptTokenLane,
  resolvePromptAnchorNode,
  summarizePromptDisplayNodes,
} from "@/lib/attention-lens";
import { shouldReuseContinuationTarget } from "@/lib/continuation-flow";
import {
  buildBranchBreadcrumb,
  buildConversationSections,
  buildCurrentRealityRawContext,
  buildCurrentRealityTokenIdList,
  buildFormattingSelectionSummary,
  groupPromptTokens,
  type RealityAssistantTokenItem,
  type RealityPromptTokenItem,
  type RealityTokenGroup,
} from "@/lib/current-reality";
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
import {
  findCompatibleModelId,
  getModelsForProvider,
  type ProviderSelectionId,
} from "@/lib/provider-selection";
import {
  getContinuationModePresentation,
  isApproximateBoundary,
} from "@/lib/continuation-mode";
import type {
  AlternativeCandidate,
  CanonicalPromptToken,
  CanonicalTokenSourceCategory,
  ContinueGenerationResponse,
  ContinuationMode,
  GenerationResponse,
  HuggingFaceAttentionAnalysisMode,
  HuggingFaceAttentionAggregationMode,
  HuggingFaceAttentionResponse,
  HuggingFaceLocalStatusResponse,
  ModelCatalogResponse,
  ModelOption,
  NodeExpansionResponse,
  ProviderCapabilitiesDetail,
  ProviderOption,
} from "@/types/api";

const OPENAI_PROVIDER_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: true,
  supports_entropy: true,
  supports_attention: false,
  supports_exact_continuation: false,
  supports_streaming: false,
  supports_branching: true,
  supports_continuation: true,
  minimum_output_tokens: 16,
};

const OLLAMA_PROVIDER_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: false,
  supports_entropy: false,
  supports_attention: false,
  supports_exact_continuation: false,
  supports_streaming: true,
  supports_branching: false,
  supports_continuation: false,
  minimum_output_tokens: 1,
};

const HUGGING_FACE_PROVIDER_CAPABILITIES: ProviderCapabilitiesDetail = {
  supports_logprobs: true,
  supports_entropy: true,
  supports_attention: true,
  supports_exact_continuation: true,
  supports_streaming: false,
  supports_branching: true,
  supports_continuation: true,
  minimum_output_tokens: 1,
};

const FALLBACK_MODEL_CATALOG: ModelCatalogResponse = {
  default_provider: "openai",
  default_model: "gpt-4.1-mini",
  default_preset: "general",
  providers: [
    {
      id: "openai",
      label: "OpenAI",
      status: "ready",
      recommended_models: [],
      capabilities: OPENAI_PROVIDER_CAPABILITIES,
    },
    {
      id: "hugging_face",
      label: "Hugging Face Local",
      status: "ready",
      status_message: "Select a supported model and click Load to initialize local analysis.",
      recommended_models: ["Qwen/Qwen2.5-3B-Instruct"],
      capabilities: HUGGING_FACE_PROVIDER_CAPABILITIES,
    },
    {
      id: "ollama",
      label: "Ollama",
      status: "offline",
      status_message:
        "Ollama is not running.\n\nInstall from https://ollama.com/\n\nThen run:\n\nollama serve",
      recommended_models: ["qwen2.5:3b", "phi3", "gemma3", "llama3.2"],
      capabilities: OLLAMA_PROVIDER_CAPABILITIES,
    },
  ],
  models: [
    {
      id: "Qwen/Qwen2.5-3B-Instruct",
      label: "Qwen2.5 3B Instruct",
      provider: "hugging_face",
      group: "Hugging Face Local",
      status: "ready",
      capabilities: HUGGING_FACE_PROVIDER_CAPABILITIES,
    },
    {
      id: "Qwen/Qwen2.5-1.5B-Instruct",
      label: "Qwen2.5 1.5B Instruct",
      provider: "hugging_face",
      group: "Hugging Face Local",
      status: "ready",
      capabilities: HUGGING_FACE_PROVIDER_CAPABILITIES,
    },
    {
      id: "gpt-4o-mini",
      label: "GPT-4o mini",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
      capabilities: OPENAI_PROVIDER_CAPABILITIES,
    },
    {
      id: "gpt-4.1-mini",
      label: "GPT-4.1 mini",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
      capabilities: OPENAI_PROVIDER_CAPABILITIES,
    },
    {
      id: "gpt-4o",
      label: "GPT-4o",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
      capabilities: OPENAI_PROVIDER_CAPABILITIES,
    },
    {
      id: "gpt-4.1",
      label: "GPT-4.1",
      provider: "openai",
      group: "OpenAI",
      status: "ready",
      capabilities: OPENAI_PROVIDER_CAPABILITIES,
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
const HORIZONTAL_GAP = 170;
const VERTICAL_GAP = 168;
const MAX_BRANCH_CHILDREN = 4;
const DEFAULT_PLAYBACK_SPEED = 1;
const FRAME_BUDGET_MS = 16;
const SHOULD_LOG_PERF = process.env.NODE_ENV !== "production";
const SHOULD_LOG_CONTINUATION = process.env.NODE_ENV !== "production";
const ATTENTION_LENS_TOOLTIP =
  "Attention shows how this layer and head distributed internal focus across earlier tokens. It is a model signal, not proof of reasoning or causation.";
const PROMPT_SECTION_GAP = 36;
const EMPTY_PROMPT_TOKENS: CanonicalPromptToken[] = [];
const PROMPT_NODE_WIDTH = 120;
const PROMPT_NODE_HEIGHT = 72;
const PROMPT_NODE_GAP = 10;
const PROMPT_ROW_GAP = 14;
const PROMPT_MAX_ROW_WIDTH = 1180;
const PROMPT_OUTPUT_GAP = 28;
const ATTENTION_FOCUS_PADDING = 0.08;
const ATTENTION_FOCUS_MIN_ZOOM = 0.7;
const ATTENTION_FOCUS_MAX_ZOOM = 1.15;
const ATTENTION_FOCUS_DURATION_MS = 360;
const PROMPT_MAX_DISTANCE_FROM_OUTPUT = 2200;
const ATTENTION_FOCUS_OCCLUSION_MARGIN = 168;

type BackendState = "checking" | "online" | "offline";
type SurfaceTheme = "midnight" | "graphite";
type ProviderId = ProviderSelectionId;
type TokenDisplayMode = "decoded" | "raw" | "token_id";

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

interface ContinuationStepResult {
  errorMessage: string | null;
  requested: boolean;
  success: boolean;
}

interface ContinueGenerationOptions {
  forceRequestFirstStep?: boolean;
  source?: "graph" | "preview-modal";
}

interface ContinueGenerationResult {
  errorMessage: string | null;
  finalNodeId: string;
  requested: boolean;
  success: boolean;
}

interface DragPerformanceStats {
  sampleCount: number;
  totalMs: number;
}

const nodeTypes = {
  tokenCard: TokenNode,
};

const edgeTypes = {
  attentionEdge: AttentionEdge,
  probabilityEdge: ProbabilityEdge,
};

const DEFAULT_EDGE_OPTIONS = {
  animated: true,
  type: "probabilityEdge",
} as const;

const FIT_VIEW_OPTIONS = {
  padding: 0.18,
} as const;

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

function waitForAnimationFrames(frameCount = 2) {
  return new Promise<void>((resolve) => {
    const step = (remaining: number) => {
      if (remaining <= 0) {
        resolve();
        return;
      }

      window.requestAnimationFrame(() => step(remaining - 1));
    };

    step(frameCount);
  });
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatProbability(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return value.toFixed(4);
}

function formatNumber(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

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
      capabilities: OPENAI_PROVIDER_CAPABILITIES,
    }
  );
}

function getDisplayLabelForTokenMode(
  node: Pick<TokenNodeData, "kind" | "displayTokenText" | "decodedContribution" | "tokenText" | "tokenId">,
  mode: TokenDisplayMode,
) {
  if (node.kind === "prompt") {
    if (mode === "raw") {
      return node.tokenText || node.displayTokenText;
    }

    if (mode === "token_id") {
      return node.tokenId !== null ? `#${node.tokenId}` : node.displayTokenText;
    }

    return node.displayTokenText || node.decodedContribution || node.tokenText;
  }

  if (mode === "raw") {
    return node.tokenText || node.displayTokenText;
  }

  if (mode === "token_id") {
    return node.tokenId !== null ? `#${node.tokenId}` : node.displayTokenText;
  }

  return node.displayTokenText || node.decodedContribution || node.tokenText;
}

function getPromptCategoryLabel(sourceCategory: CanonicalTokenSourceCategory) {
  switch (sourceCategory) {
    case "system":
      return "System";
    case "user_prompt":
      return "User prompt";
    case "assistant_prefix":
      return "Assistant prefix";
    case "template":
      return "Template / control";
    default:
      return "Generated output";
  }
}

function buildPromptTokenNode(
  promptToken: CanonicalPromptToken,
  options: {
    isDimmed: boolean;
    isPinned: boolean;
    isSelected: boolean;
    position: { x: number; y: number };
    providerCapabilities: ProviderCapabilitiesDetail;
    tokenDisplayMode: TokenDisplayMode;
  },
): TokenFlowNode {
  const metrics = getNodeMetrics(promptToken.raw_token, promptToken.token_bytes);
  const nodeData: TokenNodeData = {
    kind: "prompt",
    generationId: `prompt:${promptToken.full_position}`,
    predictionId: buildPromptTokenNodeId(promptToken.full_position),
    segmentId: null,
    continuationMode: "exact",
    tokenIndex: promptToken.full_position,
    branchId: "root",
    tokenText: promptToken.raw_token,
    displayTokenText: promptToken.display_token,
    decodedContribution: promptToken.decoded_contribution,
    cumulativeDecodedText: "",
    cumulativeRawText: "",
    cumulativeTokenIds: null,
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
    probabilityMode: "raw",
    logProbability: 0,
    entropy: 0,
    latency: 0,
    tokenId: promptToken.token_id,
    tokenizerId: promptToken.token_id,
    generationStep: promptToken.full_position,
    textPreview: promptToken.decoded_contribution,
    contextBefore: "",
    contextAfter: "",
    cumulativeProbability: 1,
    branchProbability: 1,
    depth: 0,
    rank: 0,
    finishReason: null,
    parentId: null,
    isMainPath: false,
    isCollapsed: false,
    alternativesExpanded: false,
    distributionRequested: false,
    childCount: 0,
    status: "ready",
    requestPrompt: "",
    requestModel: "",
    requestPreset: "",
    requestTemperature: 0,
    requestTopP: 1,
    requestVariation: 0,
    requestDemoMode: false,
    responseMode: "prompt",
    sourceNotes: "",
    reasoningIntent: "",
    reasoningStrategy: "",
    reasoningFocusTerms: [],
    branchRationale: null,
    metadata: {
      prompt_position: promptToken.full_position,
      provider: "hugging_face",
    },
    sourceCategory: promptToken.source_category,
    sourceLabel: promptToken.source_label || getPromptCategoryLabel(promptToken.source_category),
    specialToken: promptToken.special_token,
    providerCapabilities: options.providerCapabilities,
    rawLogits: null,
    topAlternatives: [],
    sourceAlternatives: [],
    distributionMessage: null,
    isSearchMatch: false,
    isSearchFocused: false,
    isDimmed: options.isDimmed,
    isActiveReality: options.isSelected,
    isPinned: options.isPinned,
  };

  nodeData.displayTokenText = getDisplayLabelForTokenMode(nodeData, options.tokenDisplayMode);

  return {
    id: buildPromptTokenNodeId(promptToken.full_position),
    type: "tokenCard",
    position: options.position,
    style: {
      height: PROMPT_NODE_HEIGHT,
      width: PROMPT_NODE_WIDTH,
    },
    width: PROMPT_NODE_WIDTH,
    height: PROMPT_NODE_HEIGHT,
    data: nodeData,
    draggable: false,
    selectable: true,
  };
}

function isPromptTokenDisplayNode(node: TokenFlowNode | null | undefined) {
  return Boolean(node && node.data.kind === "prompt" && node.id !== "root");
}

function findProviderOption(providers: ProviderOption[], providerId: string): ProviderOption {
  const fallbackCapabilities =
    providerId === "ollama"
      ? OLLAMA_PROVIDER_CAPABILITIES
      : providerId === "hugging_face"
        ? HUGGING_FACE_PROVIDER_CAPABILITIES
        : OPENAI_PROVIDER_CAPABILITIES;

  return (
    providers.find((item) => item.id === providerId) ?? {
      id: providerId,
      label: providerId,
      status: "ready",
      recommended_models: [],
      capabilities: fallbackCapabilities,
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
    segmentId: candidate.segment_id ?? null,
    continuationMode: candidate.continuation_mode ?? "exact",
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
    probability: candidate.probability ?? 0,
    rawProbability: candidate.raw_probability ?? candidate.probability ?? null,
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
    segmentId: alternative.segmentId,
    continuationMode: alternative.continuationMode,
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
      segmentId: record.segmentId,
      continuationMode: record.continuationMode,
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
      sourceCategory: "generated_output",
      sourceLabel: "Generated output",
      specialToken: false,
      providerCapabilities: record.providerCapabilities,
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
  providerCapabilities,
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
  providerCapabilities: ProviderCapabilitiesDetail;
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
      segmentId: null,
      continuationMode: "exact",
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
      sourceCategory: "template",
      sourceLabel: "Prompt summary",
      specialToken: false,
      providerCapabilities,
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
  continuationMode,
  cumulativeProbability,
  depth,
  entropy,
  id,
  isMainPath,
  latency,
  logProbability,
  model,
  segmentId,
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
  providerCapabilities,
}: {
  branchRationale: string | null;
  continuationMode: ContinuationMode;
  cumulativeProbability: number;
  depth: number;
  entropy: number;
  id: string;
  isMainPath: boolean;
  latency: number;
  logProbability: number;
  model: string;
  segmentId: string | null;
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
  providerCapabilities: ProviderCapabilitiesDetail;
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
      segmentId,
      continuationMode,
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
      sourceCategory: "generated_output",
      sourceLabel: "Generated output",
      specialToken: false,
      providerCapabilities,
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
      continuationMode: "exact",
      isModeBoundary: false,
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
    continuationMode: record.continuationMode,
    segmentId: record.segmentId,
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
    providerCapabilities: record.providerCapabilities,
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
    continuationMode: data?.continuationMode ?? "exact",
    isModeBoundary: data?.isModeBoundary ?? false,
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
  selectedNodeId: string | null,
  options?: { skipLayout?: boolean },
) {
  const childCounts = new Map<string, number>();

  for (const edge of edges) {
    childCounts.set(edge.source, (childCounts.get(edge.source) ?? 0) + 1);
  }

  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const activePathIdSet = new Set(buildActivePathIds(branchChoices, nodes));
  const visibleAlternativeParentId =
    selectedNodeId && nodeMap.get(selectedNodeId)?.data.kind === "token"
      ? nodeMap.get(selectedNodeId)?.data.parentId ?? null
      : null;

  const nextNodes = nodes.map((node) => {
    let hidden = false;
    let currentId: string | null = node.id;
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

      if (
        currentId !== null &&
        !activePathIdSet.has(currentId) &&
        parentId !== visibleAlternativeParentId
      ) {
        hidden = true;
        break;
      }

      currentId = parentId;
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
    if (!group[0]?.data.providerCapabilities.supports_logprobs) {
      group.forEach((node) => {
        views.set(node.id, {
          displayProbability: 0,
          probabilityCoverage: 0,
          remainingProbabilityMass: 0,
        });
      });
      continue;
    }

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

  if (!node.data.providerCapabilities.supports_logprobs) {
    return "#64748b";
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

  if (!node.providerCapabilities.supports_branching) {
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
      segmentId: node.segmentId,
      continuationMode: node.continuationMode,
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

  if (!node.data.providerCapabilities.supports_logprobs) {
    return `This provider does not expose token-level probabilities for "${node.data.displayTokenText}".`;
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
  const tokenNodes = nodes.filter(
    (node) => node.data.kind === "token" && node.data.providerCapabilities.supports_logprobs,
  );

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
  const isPromptToken = node.data.kind === "prompt" && node.id !== "root";
  return {
    height: node.data.kind === "prompt" ? (isPromptToken ? 94 : 124) : 112,
    width: node.data.kind === "prompt" ? (isPromptToken ? 188 : 292) : 208,
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
  const [selectedProvider, setSelectedProvider] = useState<ProviderId>(
    FALLBACK_MODEL_CATALOG.default_provider as ProviderId,
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
  const [isLoadingHuggingFaceStatus, setIsLoadingHuggingFaceStatus] = useState(false);
  const [isSubmittingHuggingFaceAction, setIsSubmittingHuggingFaceAction] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [isGraphInteracting, setIsGraphInteracting] = useState(false);
  const [isSentenceBarExpanded, setIsSentenceBarExpanded] = useState(false);
  const [isDockCollapsed, setIsDockCollapsed] = useState(false);
  const [surfaceTheme, setSurfaceTheme] = useState<SurfaceTheme>("midnight");
  const [probabilityViewMode, setProbabilityViewMode] =
    useState<ProbabilityViewMode>("raw");
  const [tokenDisplayMode, setTokenDisplayMode] = useState<TokenDisplayMode>("decoded");
  const [playbackSpeed, setPlaybackSpeed] = useState(DEFAULT_PLAYBACK_SPEED);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResultIndex, setSearchResultIndex] = useState(0);
  const [history, setHistory] = useState<GraphSnapshot[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [requestVariation, setRequestVariation] = useState(0);
  const [changedTokenIndexes, setChangedTokenIndexes] = useState<number[]>([]);
  const [continuationPreview, setContinuationPreview] = useState<ContinuationPreviewState | null>(null);
  const [continuationPreviewError, setContinuationPreviewError] = useState<string | null>(null);
  const [isSubmittingContinuationPreview, setIsSubmittingContinuationPreview] = useState(false);
  const [huggingFaceLocalStatus, setHuggingFaceLocalStatus] =
    useState<HuggingFaceLocalStatusResponse | null>(null);
  const [attentionLensEnabled, setAttentionLensEnabled] = useState(false);
  const [attentionAnalysisMode, setAttentionAnalysisMode] =
    useState<HuggingFaceAttentionAnalysisMode>("prediction");
  const [attentionLayer, setAttentionLayer] = useState<number | null>(null);
  const [attentionAggregationMode, setAttentionAggregationMode] =
    useState<HuggingFaceAttentionAggregationMode>("average_heads");
  const [attentionAdvancedOpen, setAttentionAdvancedOpen] = useState(false);
  const [attentionHeadIndex, setAttentionHeadIndex] = useState(0);
  const [attentionTopN, setAttentionTopN] = useState(8);
  const [showAllAttentionTokens, setShowAllAttentionTokens] = useState(false);
  const [showPromptTokens, setShowPromptTokens] = useState(false);
  const [attentionAnalysis, setAttentionAnalysis] = useState<HuggingFaceAttentionResponse | null>(null);
  const [attentionError, setAttentionError] = useState<string | null>(null);
  const [attentionLoading, setAttentionLoading] = useState(false);
  const [pinnedAttentionSourceIds, setPinnedAttentionSourceIds] = useState<string[]>([]);
  const flowRef = useRef<ReactFlowInstance<TokenFlowNode, CanvasFlowEdge> | null>(null);
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
  const attentionAbortRef = useRef<AbortController | null>(null);
  const attentionCacheRef = useRef<Map<string, HuggingFaceAttentionResponse>>(new Map());
  const attentionViewportRef = useRef<{ x: number; y: number; zoom: number } | null>(null);
  const previousPromptTokensVisibleRef = useRef(false);
  const lastLoggedAttentionAnalysisRef = useRef<HuggingFaceAttentionResponse | null>(null);
  const isGraphInteractingRef = useRef(false);
  const dragPerformanceRef = useRef<DragPerformanceStats>({
    sampleCount: 0,
    totalMs: 0,
  });

  const models = modelCatalog.models;
  const providers = modelCatalog.providers;
  const presets = modelCatalog.presets;
  const selectedProviderOption = findProviderOption(providers, selectedProvider);
  const filteredModels = useMemo(
    () => getModelsForProvider(models, selectedProvider),
    [models, selectedProvider],
  );
  const selectedModelOption = findModelOption(models, model);
  const selectedCapabilities =
    filteredModels.find((option) => option.id === model)?.capabilities ??
    selectedProviderOption.capabilities ??
    selectedModelOption.capabilities;
  const isHuggingFaceProvider = selectedProvider === "hugging_face";
  const generationProviderId =
    generation?.request.provider ??
    (generation ? findModelOption(models, generation.request.model).provider : null);
  const activeCapabilities =
    generation && generationProviderId === selectedProvider
      ? generation.provider_capabilities
      : selectedCapabilities;
  const selectedProviderStatusMessage = selectedProviderOption.status_message ?? null;
  const selectedProviderRecommendations = selectedProviderOption.recommended_models ?? [];
  const selectedProviderReady = selectedProviderOption.status === "ready";
  const generationContextMessages = useMemo(
    () => generation?.context_messages ?? [],
    [generation?.context_messages],
  );
  const activeSystemContextMessage =
    generation && generationProviderId === selectedProvider
      ? generationContextMessages.find((message) => message.role === "system") ?? null
      : null;
  const selectedHuggingFaceModelStatus =
    huggingFaceLocalStatus?.models.find((entry) => entry.id === model) ?? null;
  const selectedHuggingFaceStatusMessage =
    selectedHuggingFaceModelStatus?.status_message ??
    huggingFaceLocalStatus?.status_message ??
    null;
  const huggingFaceModelLoaded =
    huggingFaceLocalStatus?.active_model_id === model &&
    huggingFaceLocalStatus?.status === "ready";
  const showHuggingFaceControls = selectedProvider === "hugging_face";
  const canGenerate =
    demoMode ||
    (selectedProviderReady &&
      filteredModels.length > 0 &&
      (!isHuggingFaceProvider || huggingFaceModelLoaded));
  const systemPromptState = useMemo<GenerationPanelSystemPromptState>(() => {
    if (activeSystemContextMessage) {
      return {
        content: activeSystemContextMessage.content,
        editable: generation?.system_prompt_editable ?? false,
        helper:
          activeSystemContextMessage.source === "provider_default"
            ? "This model injects a provider-default system message through its chat template."
            : "These instructions were composed by the backend for the last generation.",
        sourceLabel:
          activeSystemContextMessage.source === "provider_default"
            ? "Provider default"
            : "Backend-composed",
        title: "The exact system instructions used for the current graph.",
      };
    }

    if (selectedProvider === "openai") {
      return {
        content: null,
        editable: false,
        helper:
          "OpenAI instructions are composed on the backend from the selected preset and detected intent. Generate once to inspect the exact prompt stack.",
        sourceLabel: "Composed at request time",
        title: "The exact instructions appear after a generation finishes.",
      };
    }

    if (selectedProvider === "hugging_face") {
      return {
        content: null,
        editable: false,
        helper:
          "Hugging Face Local uses the model's chat template. Generate once to inspect the exact system message and formatting tokens returned by the tokenizer.",
        sourceLabel: "Model chat template",
        title: "The exact template context appears after a generation finishes.",
      };
    }

    return {
      content: null,
      editable: false,
      helper:
        "This provider does not currently expose a separate system prompt in the saved generation context.",
      sourceLabel: "Provider managed",
      title: "A separate system message is not available for this provider.",
    };
  }, [
    activeSystemContextMessage,
    generation?.system_prompt_editable,
    selectedProvider,
  ]);
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
  const generationPromptTokens = generation?.prompt_tokens ?? EMPTY_PROMPT_TOKENS;
  const promptTokensAvailable =
    generationProviderId === "hugging_face" && generationPromptTokens.length > 0;
  const promptNodeIdByPosition = useMemo(
    () =>
      new Map<number, string>(
        generationPromptTokens.map((promptToken) => [
          promptToken.full_position,
          buildPromptTokenNodeId(promptToken.full_position),
        ]),
      ),
    [generationPromptTokens],
  );
  const selectedPromptTokenNodeId = isPromptTokenNodeId(selectedNodeId) ? selectedNodeId : null;
  const graphNodeMap = useMemo(
    () => new Map<string, TokenFlowNode>(graphNodes.map((node) => [node.id, node])),
    [graphNodes],
  );
  const activeLeafNodeId = activePathIds[activePathIds.length - 1] ?? null;
  const selectedGraphNode =
    selectedPromptTokenNodeId
      ? null
      : (selectedNodeId ? graphNodeMap.get(selectedNodeId) ?? null : null) ??
        (activeLeafNodeId ? graphNodeMap.get(activeLeafNodeId) ?? null : null) ??
        null;
  const selectedRecord =
    selectedPromptTokenNodeId
      ? null
      : (selectedNodeId ? tokenGraph.nodesById[selectedNodeId] ?? null : null) ??
        (activeLeafNodeId ? tokenGraph.nodesById[activeLeafNodeId] ?? null : null);
  const attentionValidation = useMemo(
    () =>
      selectedGraphNode?.id && canUseAttentionLens(selectedGraphNode)
        ? buildContinuationValidation(tokenGraph, selectedGraphNode.id)
        : null,
    [selectedGraphNode, tokenGraph],
  );
  const attentionAvailable = useMemo(
    () =>
      Boolean(
        selectedGraphNode &&
          canUseAttentionLens(selectedGraphNode) &&
          attentionValidation?.validationMode === "token_ids" &&
          attentionValidation.promptTokenIds &&
          promptTokensAvailable &&
          attentionValidation.generatedPrefixTokenIds &&
          attentionValidation.generatedPrefixTokenIds.length > 0,
      ),
    [attentionValidation, promptTokensAvailable, selectedGraphNode],
  );
  const attentionDefaultLayer = useMemo(
    () => Math.max((huggingFaceLocalStatus?.active_model_num_hidden_layers ?? 1) - 1, 0),
    [huggingFaceLocalStatus?.active_model_num_hidden_layers],
  );
  const effectiveAttentionLayer = attentionLayer ?? attentionDefaultLayer;
  const effectiveAttentionHead = attentionAggregationMode === "single_head" ? attentionHeadIndex : null;
  const canShowAllAttentionTokens = useMemo(
    () => (attentionAnalysis?.analyzed_context_length ?? 0) <= 40,
    [attentionAnalysis?.analyzed_context_length],
  );
  const attentionMaxConnections = useMemo(
    () =>
      showAllAttentionTokens && canShowAllAttentionTokens && attentionAnalysis
        ? Math.max(attentionAnalysis.analyzed_context_length - 1, 1)
        : attentionTopN,
    [
      attentionAnalysis,
      attentionTopN,
      canShowAllAttentionTokens,
      showAllAttentionTokens,
    ],
  );
  const attentionRequest = useMemo(() => {
    if (
      !attentionLensEnabled ||
      !attentionAvailable ||
      !selectedGraphNode ||
      !attentionValidation?.promptTokenIds ||
      !attentionValidation.generatedPrefixTokenIds ||
      generationPromptTokens.length === 0
    ) {
      return null;
    }

    const request = buildAttentionRequestPayload({
      allowTruncatedRecompute: false,
      analysisMode: attentionAnalysisMode,
      aggregationMode: attentionAggregationMode,
      generatedTokenIds: attentionValidation.generatedPrefixTokenIds,
      maxConnections: attentionMaxConnections,
      maxContextTokens: 256,
      modelId: selectedGraphNode.data.requestModel,
      modelRevision: attentionValidation.modelRevision,
      promptTokenIds: attentionValidation.promptTokenIds,
      promptTokens: generationPromptTokens,
      selectedHead: effectiveAttentionHead,
      selectedLayer: effectiveAttentionLayer,
      tokenizerIdentity: attentionValidation.tokenizerIdentity,
      tokenizerRevision: attentionValidation.tokenizerRevision,
    });

    return {
      cacheKey: buildAttentionCacheKey(request),
      request,
    };
  }, [
    attentionAnalysisMode,
    attentionAggregationMode,
    attentionAvailable,
    attentionLensEnabled,
    attentionMaxConnections,
    attentionValidation,
    effectiveAttentionHead,
    effectiveAttentionLayer,
    generationPromptTokens,
    selectedGraphNode,
  ]);
  const promptTokensVisible = promptTokensAvailable && showPromptTokens;
  const promptTokensUnavailableReason = promptTokensAvailable
    ? null
    : "Canonical prompt-token metadata is unavailable for this graph.";
  const attentionHeadLabel = useMemo(
    () => buildAttentionHeadLabel(attentionAggregationMode, effectiveAttentionHead),
    [attentionAggregationMode, effectiveAttentionHead],
  );
  const attentionHeadCount = attentionAnalysis?.num_query_heads ??
    huggingFaceLocalStatus?.active_model_num_attention_heads ??
    0;
  const attentionLayerCount = attentionAnalysis?.num_layers ??
    huggingFaceLocalStatus?.active_model_num_hidden_layers ??
    0;
  const inspectorAlternativeView = useMemo(
    () =>
      buildInspectorAlternatives(
        tokenGraph,
        selectedGraphNode?.id ?? activeLeafNodeId,
        probabilityViewMode,
      ),
    [activeLeafNodeId, probabilityViewMode, selectedGraphNode?.id, tokenGraph],
  );
  const inspectorAlternatives = inspectorAlternativeView.items;
  const inspectorCoverage = inspectorAlternativeView.coverage;
  const inspectorRemainingProbabilityMass =
    inspectorAlternativeView.remainingProbabilityMass;
  const changedTokenIndexSet = useMemo(
    () => new Set(changedTokenIndexes),
    [changedTokenIndexes],
  );
  const pinnedAttentionSourceIdSet = useMemo(
    () => new Set<string>(pinnedAttentionSourceIds),
    [pinnedAttentionSourceIds],
  );
  const attentionOverlayEdges = useMemo<AttentionFlowEdge[]>(
    () =>
      attentionLensEnabled &&
      selectedGraphNode &&
      attentionAnalysis &&
      attentionValidation
        ? buildAttentionOverlayEdges({
            analysis: attentionAnalysis,
            lineageNodeIds: attentionValidation.lineageNodeIds,
            pinnedSourceTokenIds: pinnedAttentionSourceIdSet,
            promptNodeIdByPosition: promptTokensVisible ? promptNodeIdByPosition : undefined,
            selectedNodeId: selectedGraphNode.id,
          })
        : [],
    [
      attentionAnalysis,
      attentionLensEnabled,
      attentionValidation,
      pinnedAttentionSourceIdSet,
      promptNodeIdByPosition,
      promptTokensVisible,
      selectedGraphNode,
    ],
  );
  const attentionRelatedNodeIdSet = useMemo<Set<string> | null>(() => {
    if (!attentionLensEnabled || !selectedGraphNode || !attentionAnalysis || !attentionValidation) {
      return null;
    }

    const related = new Set<string>([selectedGraphNode.id]);
    for (const source of attentionAnalysis.sources) {
      if (source.sequence_scope === "generated") {
        const graphNodeId =
          attentionValidation.lineageNodeIds[(source.generated_token_index ?? -1) + 1] ?? null;
        if (graphNodeId) {
          related.add(graphNodeId);
        }
      } else {
        related.add(
          promptTokensVisible
            ? (promptNodeIdByPosition.get(source.full_position) ?? "root")
            : "root",
        );
      }
    }

    return related;
  }, [
    attentionAnalysis,
    attentionLensEnabled,
    attentionValidation,
    promptNodeIdByPosition,
    promptTokensVisible,
    selectedGraphNode,
  ]);
  const attentionMassBreakdown = useMemo(
    () => {
      const breakdown = {
        assistantPrefix: 0,
        generatedOutput: 0,
        prompt: 0,
        system: 0,
        template: 0,
        userPrompt: 0,
      };

      if (!attentionAnalysis) {
        return breakdown;
      }

      for (const token of attentionAnalysis.analyzed_tokens) {
        const weight = typeof token.attention_weight === "number" ? token.attention_weight : 0;

        if (token.sequence_scope === "prompt") {
          breakdown.prompt += weight;
        } else {
          breakdown.generatedOutput += weight;
        }

        switch (token.source_category) {
          case "assistant_prefix":
            breakdown.assistantPrefix += weight;
            break;
          case "system":
            breakdown.system += weight;
            break;
          case "template":
            breakdown.template += weight;
            break;
          case "user_prompt":
            breakdown.userPrompt += weight;
            break;
          default:
            break;
        }
      }

      return breakdown;
    },
    [attentionAnalysis],
  );
  const displayGraphNodes = useMemo<TokenFlowNode[]>(
    () =>
      graphNodes.map((node) => {
        const isRootSummary = node.id === "root" && promptTokensAvailable;
        const promptSummaryLabel =
          isRootSummary && !promptTokensVisible && attentionMassBreakdown.prompt > 0
            ? `Prompt summary · ${formatPercent(attentionMassBreakdown.prompt)}`
            : node.data.sourceLabel;

        return {
          ...node,
          data: {
            ...node.data,
            displayTokenText:
              isRootSummary ? "Prompt summary" : getDisplayLabelForTokenMode(node.data, tokenDisplayMode),
            displayProbability:
              nodeProbabilityView.get(node.id)?.displayProbability ?? node.data.displayProbability,
            probabilityCoverage:
              nodeProbabilityView.get(node.id)?.probabilityCoverage ??
              node.data.probabilityCoverage,
            remainingProbabilityMass:
              nodeProbabilityView.get(node.id)?.remainingProbabilityMass ??
              node.data.remainingProbabilityMass,
            probabilityMode: probabilityViewMode,
            sourceLabel: promptSummaryLabel,
            isSearchMatch: searchMatches.some((match) => match.id === node.id),
            isSearchFocused: focusedSearchNodeId === node.id,
            isDimmed:
              (hoveredRelatedIdSet ? !hoveredRelatedIdSet.has(node.id) : false) ||
              (attentionRelatedNodeIdSet ? !attentionRelatedNodeIdSet.has(node.id) : false),
            isActiveReality: activePathIdSet.has(node.id),
            isPinned: pinnedSet.has(node.id),
          },
        };
      }),
    [
      activePathIdSet,
      attentionMassBreakdown.prompt,
      attentionRelatedNodeIdSet,
      focusedSearchNodeId,
      graphNodes,
      hoveredRelatedIdSet,
      nodeProbabilityView,
      pinnedSet,
      probabilityViewMode,
      promptTokensAvailable,
      promptTokensVisible,
      searchMatches,
      tokenDisplayMode,
    ],
  );
  const attentionFocusedGraphNodes = useMemo<TokenFlowNode[]>(
    () =>
      attentionLensEnabled
        ? displayGraphNodes.filter((node) => {
            if (node.id === "root" || node.data.kind !== "token") {
              return true;
            }

            return activePathIdSet.has(node.id);
          })
        : displayGraphNodes,
    [activePathIdSet, attentionLensEnabled, displayGraphNodes],
  );
  const promptDisplayNodes = useMemo<TokenFlowNode[]>(() => {
    if (!promptTokensVisible || generationPromptTokens.length === 0) {
      return [];
    }

    const rootNode = displayGraphNodes.find((node) => node.id === "root") ?? null;
    const promptAnchorNode = resolvePromptAnchorNode({
      activePathIds,
      displayNodes: displayGraphNodes,
    });

    if (!rootNode || !promptAnchorNode) {
      return [];
    }

    const anchorFrame = getNodeFrame(promptAnchorNode);
    const rawPromptPlacements = layoutPromptTokenLane({
      laneAnchorHeight: anchorFrame.height,
      laneAnchorX: promptAnchorNode.position.x - PROMPT_OUTPUT_GAP,
      laneAnchorY: promptAnchorNode.position.y + (anchorFrame.height - PROMPT_NODE_HEIGHT) / 2,
      maxRowWidth: PROMPT_MAX_ROW_WIDTH,
      nodeGap: PROMPT_NODE_GAP,
      rowGap: PROMPT_ROW_GAP,
      sectionGap: PROMPT_SECTION_GAP,
      tokens: generationPromptTokens.map((promptToken) => ({
        fullPosition: promptToken.full_position,
        height: PROMPT_NODE_HEIGHT,
        sourceCategory: promptToken.source_category,
        width: PROMPT_NODE_WIDTH,
      })),
    });
    const maxRowIndex = Math.max(...rawPromptPlacements.map((placement) => placement.row), 0);
    const verticalCenterShift =
      (maxRowIndex * (PROMPT_NODE_HEIGHT + PROMPT_ROW_GAP)) / 2;
    const promptPlacements = rawPromptPlacements.map((placement) => ({
      ...placement,
      y: placement.y + verticalCenterShift,
    }));
    const promptPlacementByPosition = new Map(
      promptPlacements.map((placement) => [placement.fullPosition, placement]),
    );
    const nextNodes: TokenFlowNode[] = [];

    for (const promptToken of generationPromptTokens) {
      const nodeId = buildPromptTokenNodeId(promptToken.full_position);
      const promptPlacement = promptPlacementByPosition.get(promptToken.full_position);

      if (!promptPlacement) {
        continue;
      }

      const nextNode = buildPromptTokenNode(promptToken, {
        isDimmed:
          (hoveredRelatedIdSet ? !hoveredRelatedIdSet.has(nodeId) : false) ||
          (attentionRelatedNodeIdSet ? !attentionRelatedNodeIdSet.has(nodeId) : false),
        isPinned: pinnedSet.has(nodeId),
        isSelected: selectedNodeId === nodeId,
        position: { x: promptPlacement.x, y: promptPlacement.y },
        providerCapabilities: rootNode.data.providerCapabilities,
        tokenDisplayMode,
      });

      nextNodes.push({
        ...nextNode,
        data: {
          ...nextNode.data,
          requestPrompt: rootNode.data.requestPrompt,
          requestModel: rootNode.data.requestModel,
          requestPreset: rootNode.data.requestPreset,
          requestTemperature: rootNode.data.requestTemperature,
          requestTopP: rootNode.data.requestTopP,
          requestVariation: rootNode.data.requestVariation,
          requestDemoMode: rootNode.data.requestDemoMode,
          metadata: {
            ...nextNode.data.metadata,
            provider: rootNode.data.metadata.provider ?? "hugging_face",
          },
        },
      });
    }

    if (process.env.NODE_ENV !== "production") {
      for (const node of nextNodes) {
        const deltaX = node.position.x - promptAnchorNode.position.x;
        const deltaY = node.position.y - promptAnchorNode.position.y;
        const distance = Math.hypot(deltaX, deltaY);

        if (
          !Number.isFinite(node.position.x) ||
          !Number.isFinite(node.position.y) ||
          distance > PROMPT_MAX_DISTANCE_FROM_OUTPUT
        ) {
          console.warn("[llmscope-attention] prompt-node-layout-outlier", {
            anchorNodeId: promptAnchorNode.id,
            nodeId: node.id,
            position: node.position,
            distance,
          });
        }
      }
    }

    return nextNodes;
  }, [
    attentionRelatedNodeIdSet,
    displayGraphNodes,
      generationPromptTokens,
      hoveredRelatedIdSet,
      pinnedSet,
      promptTokensVisible,
      activePathIds,
      selectedNodeId,
      tokenDisplayMode,
  ]);
  const displayNodes = useMemo<TokenFlowNode[]>(
    () => [
      ...promptDisplayNodes,
      ...attentionFocusedGraphNodes.filter(
        (node) => !(promptTokensVisible && node.id === "root"),
      ),
    ],
    [attentionFocusedGraphNodes, promptDisplayNodes, promptTokensVisible],
  );
  const promptDisplaySummary = useMemo(
    () => summarizePromptDisplayNodes(displayNodes),
    [displayNodes],
  );
  const displayNodeMap = useMemo(
    () => new Map<string, TokenFlowNode>(displayNodes.map((node) => [node.id, node])),
    [displayNodes],
  );
  const selectedNode =
    (selectedNodeId ? displayNodeMap.get(selectedNodeId) ?? null : null) ??
    (selectedGraphNode ? displayNodeMap.get(selectedGraphNode.id) ?? null : null) ??
    null;
  const selectedProviderIdForInspector =
    readMetadataString(selectedNode?.data.metadata ?? null, "provider") ??
    generation?.request.provider ??
    selectedProvider;
  const selectedProviderLabelForInspector = findProviderOption(
    providers,
    selectedProviderIdForInspector,
  ).label;
  const displayEdges: ProbabilityFlowEdge[] = useMemo(
    () => {
      const promptOutputAnchorId =
        promptTokensVisible && promptDisplayNodes.length > 0
          ? promptDisplayNodes[promptDisplayNodes.length - 1]?.id ?? "root"
          : "root";

      return graphEdges.flatMap((edge) => {
        const remappedSource = edge.source === "root" ? promptOutputAnchorId : edge.source;
        const remappedTarget =
          edge.target === "root" && promptTokensVisible && promptDisplayNodes.length > 0
            ? promptDisplayNodes[0]?.id ?? "root"
            : edge.target;
        const sourceNode = displayNodeMap.get(remappedSource);
        const targetNode = displayNodeMap.get(remappedTarget);

        if (!sourceNode || !targetNode) {
          return [];
        }

        const sourceDimmed = sourceNode?.data.isDimmed ?? false;
        const targetDimmed = targetNode?.data.isDimmed ?? false;
        const edgeData = ensureProbabilityEdgeData(edge.data);

        return [{
          ...edge,
          id:
            remappedSource === edge.source && remappedTarget === edge.target
              ? edge.id
              : `${edge.id}:display`,
          source: remappedSource,
          target: remappedTarget,
          data: {
            ...edgeData,
            probability: targetNode?.data.displayProbability ?? edgeData.probability,
            rawProbability: targetNode?.data.rawProbability ?? edgeData.rawProbability,
            probabilityCoverage:
              targetNode?.data.probabilityCoverage ?? edgeData.probabilityCoverage,
            remainingProbabilityMass:
              targetNode?.data.remainingProbabilityMass ?? edgeData.remainingProbabilityMass,
            probabilityMode: targetNode?.data.probabilityMode ?? probabilityViewMode,
            continuationMode: targetNode?.data.continuationMode ?? edgeData.continuationMode,
            isModeBoundary: isApproximateBoundary(
              sourceNode?.data.continuationMode,
              targetNode?.data.continuationMode,
            ),
            isActiveReality: activeEdgeIdSet.has(edge.id),
            isDimmed: sourceDimmed || targetDimmed,
            isFocused: hoveredNodeId
              ? activeEdgeIdSet.has(edge.id) ||
                remappedSource === hoveredNodeId ||
                remappedTarget === hoveredNodeId
              : false,
          },
        }];
      });
    },
    [
      activeEdgeIdSet,
      displayNodeMap,
      graphEdges,
      hoveredNodeId,
      probabilityViewMode,
      promptDisplayNodes,
      promptTokensVisible,
    ],
  );
  const promptChainEdges = useMemo<ProbabilityFlowEdge[]>(() => {
    if (!promptTokensVisible || promptDisplayNodes.length === 0) {
      return [];
    }

    const derivedEdges: ProbabilityFlowEdge[] = [];

    for (let index = 0; index < promptDisplayNodes.length; index += 1) {
      const sourceNode = promptDisplayNodes[index];
      const targetNode = promptDisplayNodes[index + 1] ?? null;

      if (!sourceNode || !targetNode) {
        continue;
      }

      derivedEdges.push({
        id: `prompt-chain:${sourceNode.id}:${targetNode.id}`,
        source: sourceNode.id,
        target: targetNode.id,
        type: "probabilityEdge",
        selectable: false,
        data: {
          probability: 0.52,
          rawProbability: 0.52,
          probabilityCoverage: 1,
          remainingProbabilityMass: 0,
          probabilityMode: "raw",
          continuationMode: "exact",
          isModeBoundary: false,
          isMainPath: false,
          isActiveReality: false,
          isDimmed:
            (sourceNode.data.isDimmed ?? false) || (targetNode.data.isDimmed ?? false),
          isFocused: false,
        },
      });
    }

    return derivedEdges;
  }, [promptDisplayNodes, promptTokensVisible]);
  const displayAttentionEdges = useMemo<AttentionFlowEdge[]>(
    () =>
      attentionOverlayEdges.map((edge) => {
        const edgeData = edge.data as AttentionFlowEdge["data"];

        return {
          ...edge,
          data: {
            ...edgeData,
            isDimmed:
              (displayNodeMap.get(edge.source)?.data.isDimmed ?? false) ||
              (displayNodeMap.get(edge.target)?.data.isDimmed ?? false),
          },
        } as AttentionFlowEdge;
      }),
    [attentionOverlayEdges, displayNodeMap],
  );
  const renderEdges = useMemo<CanvasFlowEdge[]>(
    () => [...promptChainEdges, ...displayEdges, ...displayAttentionEdges],
    [displayAttentionEdges, displayEdges, promptChainEdges],
  );
  const naturalReason = buildNaturalLanguageReason(
    selectedNode,
    inspectorAlternatives,
    probabilityViewMode,
  );
  const selectedNodeMetrics =
    selectedNode
      ? getNodeMetrics(selectedNode.data.tokenText || selectedNode.data.decodedContribution, selectedNode.data.tokenBytes)
      : null;
  const isSelectedPromptToken = isPromptTokenDisplayNode(selectedNode);
  const selectedNodeDisplayLabel = selectedNode
    ? getDisplayLabelForTokenMode(selectedNode.data, tokenDisplayMode)
    : "Select a token";
  const attentionSourceEntries = useMemo(() => {
    if (!attentionAnalysis || !attentionValidation) {
      return [];
    }

    return attentionAnalysis.sources.map((source) => {
      const sourceId = buildAttentionTokenId(source.sequence_scope, source.full_position);
      const graphNodeId =
        source.sequence_scope === "generated"
          ? attentionValidation.lineageNodeIds[(source.generated_token_index ?? -1) + 1] ?? null
          : promptTokensVisible
            ? (promptNodeIdByPosition.get(source.full_position) ?? null)
            : "root";
      const sourceLabel =
        source.sequence_scope === "prompt" && !promptTokensVisible
          ? "Prompt summary"
          : source.source_label;

      return {
        graphNodeId,
        isCanvasNodeUnavailable: graphNodeId === null,
        source: {
          ...source,
          source_label: sourceLabel,
        },
        sourceId,
      };
    });
  }, [
    attentionAnalysis,
    attentionValidation,
    promptNodeIdByPosition,
    promptTokensVisible,
  ]);
  const attentionSourceEntryById = useMemo(
    () => new Map(attentionSourceEntries.map((entry) => [entry.sourceId, entry])),
    [attentionSourceEntries],
  );
  const visibleAttentionSourceCount = useMemo(
    () =>
      attentionSourceEntries.filter(
        (entry) => entry.graphNodeId !== null && displayNodeMap.has(entry.graphNodeId),
      ).length,
    [attentionSourceEntries, displayNodeMap],
  );
  const canFocusAttention =
    attentionLensEnabled &&
    Boolean(selectedGraphNode?.id) &&
    attentionSourceEntries.some((entry) => entry.graphNodeId !== null);
  const showAttentionInspectorSection =
    Boolean(
      selectedNode &&
        selectedNode.data.providerCapabilities.supports_attention &&
        (
          readMetadataString(selectedNode.data.metadata ?? null, "provider") ??
          generationProviderId ??
          null
        ) === "hugging_face",
    );
  const attentionHeadline =
    attentionAnalysisMode === "representation"
      ? `How "${selectedNodeDisplayLabel}" attended backward after entering the sequence`
      : `What did the model attend to while predicting "${selectedNodeDisplayLabel}"?`;
  const attentionModeDescription =
    attentionAnalysisMode === "representation"
      ? `How "${selectedNodeDisplayLabel}" attended backward after entering the sequence`
      : `Attention used while predicting "${selectedNodeDisplayLabel}"`;
  const attentionUnavailableMessage =
    showAttentionInspectorSection && !attentionAvailable
      ? promptTokensAvailable
        ? isSelectedPromptToken
          ? 'Select a generated token to inspect prediction attention, or select a prompt token to inspect how that token represented earlier context. Prompt-token target analysis is not yet available in this view, but prompt tokens remain exact input-context nodes and attention sources.'
          : "Select a generated token to inspect prediction attention, or select a prompt token to inspect how that token represented earlier context."
        : "Attention analysis is unavailable for this graph because canonical prompt-token metadata was not saved with it."
      : null;
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
        supportsLogprobs: node.data.providerCapabilities.supports_logprobs,
      })),
    [activeSentenceNodes, changedTokenIndexSet],
  );
  const currentRealityAssistantContextTokens = useMemo<RealityAssistantTokenItem[]>(
    () =>
      activeSentenceNodes.map((node) => ({
        decodedContribution: node.data.decodedContribution,
        displayToken: node.data.displayTokenText,
        graphTokenId: node.id,
        id: node.id,
        rank: node.data.rank,
        rawToken: node.data.tokenText,
        step: node.data.generationStep,
        tokenId: node.data.tokenId,
      })),
    [activeSentenceNodes],
  );
  const currentRealityPromptGroups = useMemo<RealityTokenGroup[]>(
    () => groupPromptTokens(generationPromptTokens, promptNodeIdByPosition),
    [generationPromptTokens, promptNodeIdByPosition],
  );
  const currentRealityPromptDisplayGroups = useMemo<CurrentRealityTokenGroup[]>(
    () =>
      currentRealityPromptGroups.map((group) => ({
        category: group.category,
        id: group.id,
        label: group.label,
        tokens: group.tokens.map((token): CurrentRealityGroupedTokenItem => {
          const promptToken = token as RealityPromptTokenItem;

          return {
          canonicalPosition: promptToken.canonicalPosition,
          decodedContribution: promptToken.decodedContribution,
          displayProbability: null,
          displayToken: promptToken.displayToken,
          graphTokenId: promptToken.graphTokenId,
          id: promptToken.id,
          kind: "prompt",
          rank: null,
          rawProbability: null,
          rawToken: promptToken.rawToken,
          sourceCategory: promptToken.sourceCategory,
          sourceLabel: promptToken.sourceLabel,
          specialToken: promptToken.specialToken,
          step: promptToken.canonicalPosition,
          supportsLogprobs: false,
          tokenId: promptToken.tokenId,
        };
        }),
      })),
    [currentRealityPromptGroups],
  );
  const currentRealityConversationSections = useMemo<CurrentRealityConversationSection[]>(
    () =>
      buildConversationSections({
        assistantTokens: currentRealityAssistantContextTokens,
        contextMessages: generationContextMessages,
        promptGroups: currentRealityPromptGroups,
      }).map((section) => ({
        id: section.id,
        label: section.label,
        role: section.role,
        text: section.text,
        tokenIds: section.tokenIds,
      })),
    [
      currentRealityAssistantContextTokens,
      currentRealityPromptGroups,
      generationContextMessages,
    ],
  );
  const currentRealityFormattingSelection = useMemo<CurrentRealityFormattingSelection | null>(
    () => {
      const summaryRecord = buildFormattingSelectionSummary(
        currentRealityPromptGroups,
        selectedNodeId,
      );

      if (!summaryRecord) {
        return null;
      }

      return {
        description:
          summaryRecord.category === "assistant_prefix"
            ? "Selected assistant-prefix token"
            : "Selected formatting token hidden in Conversation mode",
        label: summaryRecord.label,
        token: summaryRecord.token.rawToken,
      };
    },
    [currentRealityPromptGroups, selectedNodeId],
  );
  const currentRealityRawContextText = useMemo(
    () =>
      buildCurrentRealityRawContext({
        assistantTokens: currentRealityAssistantContextTokens,
        promptGroups: currentRealityPromptGroups,
        rawContextText: generation?.raw_context_text ?? null,
      }),
    [
      currentRealityAssistantContextTokens,
      currentRealityPromptGroups,
      generation?.raw_context_text,
    ],
  );
  const currentRealityConversationCopyText = useMemo(
    () =>
      currentRealityConversationSections
        .map((section) => `${section.label.toUpperCase()}\n${section.text}`.trim())
        .filter(Boolean)
        .join("\n\n"),
    [currentRealityConversationSections],
  );
  const currentRealityUserPromptCopyText = useMemo(
    () =>
      currentRealityConversationSections.find((section) => section.role === "user")?.text ?? "",
    [currentRealityConversationSections],
  );
  const currentRealityTokenIdCopyText = useMemo(
    () =>
      buildCurrentRealityTokenIdList({
        assistantTokens: currentRealityAssistantContextTokens,
        promptGroups: currentRealityPromptGroups,
      }),
    [currentRealityAssistantContextTokens, currentRealityPromptGroups],
  );
  const currentRealityBranchBreadcrumb = useMemo(
    () => buildBranchBreadcrumb(currentRealityAssistantContextTokens),
    [currentRealityAssistantContextTokens],
  );
  const currentRealityAttentionTokens = useMemo<CurrentRealityAttentionTokenItem[]>(
    () =>
      attentionLensEnabled && attentionAnalysis && attentionValidation
        ? buildAttentionStripTokens({
            analysis: attentionAnalysis,
            lineageNodeIds: attentionValidation.lineageNodeIds,
            pinnedSourceTokenIds: pinnedAttentionSourceIdSet,
            promptNodeIdByPosition: promptTokensVisible ? promptNodeIdByPosition : undefined,
          }).map((token): CurrentRealityAttentionTokenItem => ({
            attentionWeight: token.attentionWeight,
            decodedContribution: token.decodedContribution,
            displayToken:
              tokenDisplayMode === "raw"
                ? token.rawToken || token.displayToken
                : tokenDisplayMode === "token_id"
                  ? `#${token.tokenId}`
                  : token.displayToken || token.decodedContribution || token.rawToken,
            fullPosition: token.fullPosition,
            graphTokenId: token.graphTokenId,
            id: token.id,
            isPinned: token.isPinned,
            isQuery: token.isQuery,
            rawToken: token.rawToken,
            sequenceScope: token.sequenceScope,
            tokenId: token.tokenId,
          }))
        : [],
    [
      attentionAnalysis,
      attentionLensEnabled,
      attentionValidation,
      pinnedAttentionSourceIdSet,
      promptNodeIdByPosition,
      promptTokensVisible,
      tokenDisplayMode,
    ],
  );
  const currentRealityStats = useMemo<CurrentRealityStats>(
    () => ({
      branchDepth: activeSentenceNodes[activeSentenceNodes.length - 1]?.data.depth ?? 0,
      displayProbability:
        activeCapabilities.supports_logprobs
          ? (activeSentenceNodes[activeSentenceNodes.length - 1]?.data.displayProbability ?? 0)
          : null,
      entropy: activeCapabilities.supports_entropy
        ? (activeSentenceNodes[activeSentenceNodes.length - 1]?.data.entropy ?? 0)
        : null,
      latency: activeSentenceNodes[activeSentenceNodes.length - 1]?.data.latency ?? 0,
      rawProbability:
        activeCapabilities.supports_logprobs
          ? (activeSentenceNodes[activeSentenceNodes.length - 1]?.data.rawProbability ?? 0)
          : null,
      supportsEntropy: activeCapabilities.supports_entropy,
      supportsLogprobs: activeCapabilities.supports_logprobs,
      tokenCount: activeSentenceNodes.length,
    }),
    [activeCapabilities.supports_entropy, activeCapabilities.supports_logprobs, activeSentenceNodes],
  );
  const currentRealityContinuationMode = useMemo(
    () =>
      getContinuationModePresentation({
        continuationMode:
          activeSentenceNodes[activeSentenceNodes.length - 1]?.data.continuationMode ??
          selectedNode?.data.continuationMode,
        metadata:
          activeSentenceNodes[activeSentenceNodes.length - 1]?.data.metadata ??
          selectedNode?.data.metadata,
      }),
    [activeSentenceNodes, selectedNode?.data.continuationMode, selectedNode?.data.metadata],
  );
  const currentRealitySummaryItems = useMemo<CurrentRealitySummaryItem[]>(() => {
    const contextWindowLimit =
      generationProviderId === "hugging_face"
        ? (huggingFaceLocalStatus?.limits.context_window_tokens ?? null)
        : null;
    const promptCount = generation?.stats.prompt_tokens ?? generationPromptTokens.length;
    const outputCount = currentRealityTokens.length;
    const contextUsageValue =
      typeof contextWindowLimit === "number" && contextWindowLimit > 0
        ? `${promptCount + outputCount} / ${contextWindowLimit}`
        : `${promptCount + outputCount} used`;

    return [
      {
        label: "Provider",
        tone: "muted",
        value: selectedProviderOption.label,
      },
      {
        label: "Model",
        tone: "muted",
        value: findModelOption(models, generation?.request.model ?? model).label,
      },
      {
        label: "Prompt",
        value: `${promptCount} tokens`,
      },
      {
        label: "Context",
        value: contextUsageValue,
      },
    ];
  }, [
    generation?.request.model,
    generation?.stats.prompt_tokens,
    generationPromptTokens.length,
    generationProviderId,
    huggingFaceLocalStatus?.limits.context_window_tokens,
    currentRealityTokens.length,
    model,
    models,
    selectedProviderOption.label,
  ]);
  const currentRealityDetailItems = useMemo<CurrentRealitySummaryItem[]>(
    () => [
      { label: "Temperature", value: String(generation?.request.temperature ?? temperature) },
      { label: "Top-p", value: String(generation?.request.top_p ?? topP) },
      {
        label: "Selected branch",
        value: currentRealityBranchBreadcrumb,
      },
      {
        label: "Output tokens",
        value: String(currentRealityTokens.length),
      },
    ],
    [
      currentRealityBranchBreadcrumb,
      currentRealityTokens.length,
      generation?.request.temperature,
      generation?.request.top_p,
      temperature,
      topP,
    ],
  );
  const inspectorContinuationMode = useMemo(
    () =>
      getContinuationModePresentation({
        continuationMode: selectedNode?.data.continuationMode,
        metadata: selectedNode?.data.metadata ?? null,
      }),
    [selectedNode?.data.continuationMode, selectedNode?.data.metadata],
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
  const attentionGraphVersion = useMemo(
    () =>
      `${tokenGraph.rootPrompt}\u0001${tokenGraph.generationOrder.join("|")}\u0001${
        Object.keys(tokenGraph.nodesById).length
      }`,
    [tokenGraph.generationOrder, tokenGraph.nodesById, tokenGraph.rootPrompt],
  );

  const restoreAttentionViewport = useCallback((overrideViewport?: {
    x: number;
    y: number;
    zoom: number;
  } | null) => {
    const savedViewport = overrideViewport ?? attentionViewportRef.current;

    if (!savedViewport || !flowRef.current) {
      if (!overrideViewport) {
        attentionViewportRef.current = null;
      }
      return;
    }

    if (!overrideViewport) {
      attentionViewportRef.current = null;
    }
    setViewport((currentViewport) =>
      currentViewport.x === savedViewport.x &&
      currentViewport.y === savedViewport.y &&
      currentViewport.zoom === savedViewport.zoom
        ? currentViewport
        : savedViewport,
    );
    void flowRef.current.setViewport(savedViewport, {
      duration: 420,
    });
  }, []);

  const logAttentionDiagnostics = useCallback(
    (
      reason: string,
      options?: {
        bounds?: ReturnType<typeof buildDeterministicFocusViewport>["bounds"];
        focusCandidateIds?: string[];
        includedFocusIds?: string[];
        targetViewport?: { x: number; y: number; zoom: number } | null;
      },
    ) => {
      if (process.env.NODE_ENV === "production") {
        return null;
      }

      const instance = flowRef.current;
      const liveNodes =
        ((instance?.getNodes() as Array<
          TokenFlowNode & {
            height?: number;
            measured?: { height?: number; width?: number };
            width?: number;
          }
        > | undefined) ?? []);
      const liveNodeMap = new Map(liveNodes.map((node) => [node.id, node]));
      const internalNodeGetter = (
        instance as unknown as {
          getInternalNode?: (nodeId: string) => {
            measured?: { height?: number; width?: number };
            internals?: { positionAbsolute?: { x: number; y: number } };
          } | null;
        } | null
      )?.getInternalNode;
      const containerElement = document.querySelector(".react-flow");
      const containerRect = containerElement?.getBoundingClientRect() ?? null;
      const currentZoom = instance?.getViewport().zoom ?? viewport.zoom ?? 1;
      const relevantNodeIds = new Set<string>([
        ...promptDisplaySummary.promptNodeIds,
        ...(selectedGraphNode?.id ? [selectedGraphNode.id] : []),
        ...attentionSourceEntries
          .map((entry) => entry.graphNodeId)
          .filter((nodeId): nodeId is string => Boolean(nodeId)),
        ...(options?.focusCandidateIds ?? []),
        ...(options?.includedFocusIds ?? []),
      ]);
      const nodeSnapshots = [...relevantNodeIds].map((nodeId) => {
        const displayNode = displayNodeMap.get(nodeId) ?? null;
        const liveNode = liveNodeMap.get(nodeId) ?? null;
        const internalNode = internalNodeGetter?.(nodeId) ?? null;
        const domElement = document.querySelector(`[data-id="${CSS.escape(nodeId)}"]`);
        const domRect = domElement?.getBoundingClientRect() ?? null;
        const measuredWidth =
          internalNode?.measured?.width ??
          liveNode?.measured?.width ??
          liveNode?.width ??
          (domRect && currentZoom > 0 ? domRect.width / currentZoom : null);
        const measuredHeight =
          internalNode?.measured?.height ??
          liveNode?.measured?.height ??
          liveNode?.height ??
          (domRect && currentZoom > 0 ? domRect.height / currentZoom : null);
        const isInsideViewport = Boolean(
          domRect &&
            containerRect &&
            domRect.width > 0 &&
            domRect.height > 0 &&
            domRect.right >= containerRect.left &&
            domRect.left <= containerRect.right &&
            domRect.bottom >= containerRect.top &&
            domRect.top <= containerRect.bottom,
        );

        return {
          canonicalPosition:
            displayNode?.data.kind === "prompt"
              ? readMetadataNumber(displayNode.data.metadata, "prompt_position")
              : displayNode?.data.tokenIndex ?? null,
          displayNodeHidden: displayNode?.hidden ?? null,
          domRect:
            domRect
              ? {
                  bottom: domRect.bottom,
                  height: domRect.height,
                  left: domRect.left,
                  right: domRect.right,
                  top: domRect.top,
                  width: domRect.width,
                  x: domRect.x,
                  y: domRect.y,
                }
              : null,
          existsInDom: Boolean(domElement),
          existsInFlowGetNodes: liveNodeMap.has(nodeId),
          id: nodeId,
          includedInFocus: options?.includedFocusIds?.includes(nodeId) ?? false,
          insideVisibleViewport: isInsideViewport,
          internalPositionAbsolute:
            internalNode?.internals?.positionAbsolute
              ? {
                  x: internalNode.internals.positionAbsolute.x,
                  y: internalNode.internals.positionAbsolute.y,
                }
              : null,
          measured:
            liveNode || domRect
              ? {
                  height: measuredHeight,
                  width: measuredWidth,
                }
              : null,
          position:
            liveNode
              ? {
                  x: liveNode.position.x,
                  y: liveNode.position.y,
                }
              : null,
          sourceCategory: displayNode?.data.sourceCategory ?? null,
          type: displayNode?.data.kind ?? null,
        };
      });
      const snapshot = {
        bounds: options?.bounds ?? null,
        currentViewport: instance?.getViewport() ?? viewport,
        displayNodeCount: displayNodes.length,
        domNodeCount: document.querySelectorAll(".react-flow__node").length,
        domPromptNodeCount: document.querySelectorAll('[data-id^="prompt-token-"]').length,
        focusCandidateIds: options?.focusCandidateIds ?? [],
        includedFocusIds: options?.includedFocusIds ?? [],
        liveNodeCount: liveNodes.length,
        nodes: nodeSnapshots,
        promptExpansionState: {
          canonicalPromptTokenCount: generationPromptTokens.length,
          constructedPromptNodeCount: promptDisplayNodes.length,
          invariant: promptDisplaySummary,
          showPromptTokens,
        },
        reactFlowContainerRect:
          containerRect
            ? {
                bottom: containerRect.bottom,
                height: containerRect.height,
                left: containerRect.left,
                right: containerRect.right,
                top: containerRect.top,
                width: containerRect.width,
                x: containerRect.x,
                y: containerRect.y,
              }
            : null,
        reason,
        targetViewport: options?.targetViewport ?? null,
      };

      (window as typeof window & { __LLMSCOPE_LAST_DIAGNOSTIC__?: unknown }).__LLMSCOPE_LAST_DIAGNOSTIC__ =
        snapshot;
      console.debug("[llmscope-attention]", snapshot);
      return snapshot;
    },
    [
      attentionSourceEntries,
      displayNodeMap,
      displayNodes,
      generationPromptTokens.length,
      promptDisplayNodes.length,
      promptDisplaySummary,
      selectedGraphNode?.id,
      showPromptTokens,
      viewport,
    ],
  );

  const computeAttentionFocusViewport = useCallback(() => {
    const instance = flowRef.current;
    const containerElement = document.querySelector(".react-flow");

    if (!instance || !containerElement || !selectedGraphNode?.id) {
      logAttentionDiagnostics("focus-attention:missing-prerequisites");
      return null;
    }

    const containerRect = containerElement.getBoundingClientRect();
    const inspectorRect = document.querySelector(".inspector-panel")?.getBoundingClientRect() ?? null;
    const focusViewportInset = {
      right:
        inspectorRect && inspectorRect.right > containerRect.left
          ? Math.max(
              containerRect.right - inspectorRect.left + ATTENTION_FOCUS_OCCLUSION_MARGIN,
              0,
            )
          : 0,
    };
    const currentZoom = instance.getViewport().zoom || 1;
    const liveNodes = (instance.getNodes() as Array<
      TokenFlowNode & {
        height?: number;
        measured?: { height?: number; width?: number };
        width?: number;
      }
    >) ?? [];
    const internalNodeGetter = (
      instance as unknown as {
        getInternalNode?: (nodeId: string) => {
          measured?: { height?: number; width?: number };
          internals?: { positionAbsolute?: { x: number; y: number } };
        } | null;
      }
    ).getInternalNode;
    const sourceNodeIds = attentionSourceEntries
      .map((entry) => entry.graphNodeId)
      .filter((nodeId): nodeId is string => Boolean(nodeId));
    const focusResult = buildDeterministicFocusViewport({
      containerHeight: containerRect.height,
      containerWidth: containerRect.width,
      maxZoom: ATTENTION_FOCUS_MAX_ZOOM,
      minZoom: ATTENTION_FOCUS_MIN_ZOOM,
      nodes: liveNodes.map((node) => {
        const internalNode = internalNodeGetter?.(node.id) ?? null;
        const positionAbsolute = internalNode?.internals?.positionAbsolute ?? node.position;
        const domElement = document.querySelector(`[data-id="${CSS.escape(node.id)}"]`);
        const domRect = domElement?.getBoundingClientRect() ?? null;
        const width =
          internalNode?.measured?.width ??
          node.measured?.width ??
          node.width ??
          (domRect && currentZoom > 0 ? domRect.width / currentZoom : null);
        const height =
          internalNode?.measured?.height ??
          node.measured?.height ??
          node.height ??
          (domRect && currentZoom > 0 ? domRect.height / currentZoom : null);

        return {
          height,
          hidden: Boolean(node.hidden),
          id: node.id,
          inDom: Boolean(domElement),
          width,
          x: positionAbsolute?.x ?? null,
          y: positionAbsolute?.y ?? null,
        };
      }),
      padding: ATTENTION_FOCUS_PADDING,
      selectedNodeId: selectedGraphNode.id,
      sourceNodeIds,
      viewportInset: focusViewportInset,
    });

    logAttentionDiagnostics("focus-attention:computed", {
      bounds: focusResult.bounds,
      focusCandidateIds: [selectedGraphNode.id, ...sourceNodeIds],
      includedFocusIds: focusResult.includedIds,
      targetViewport: focusResult.viewport,
    });

    return focusResult;
  }, [attentionSourceEntries, logAttentionDiagnostics, selectedGraphNode?.id]);

  const focusAttentionNeighborhood = useCallback(async () => {
    if (!attentionLensEnabled || !selectedGraphNode?.id) {
      logAttentionDiagnostics("focus-attention:skipped");
      return false;
    }

    const hasPromptSources = attentionSourceEntries.some(
      (entry) => entry.source.sequence_scope === "prompt",
    );

    if (hasPromptSources && !promptTokensVisible) {
      setShowPromptTokens(true);
      await waitForAnimationFrames(2);
      await waitForAnimationFrames(2);
    }

    const focusResult = computeAttentionFocusViewport();

    if (!focusResult?.viewport || !flowRef.current) {
      console.warn("[llmscope-attention] focus-attention-unavailable");
      return false;
    }

    if (!attentionViewportRef.current) {
      attentionViewportRef.current = flowRef.current.getViewport();
    }

    setViewport((currentViewport) =>
      currentViewport.x === focusResult.viewport!.x &&
      currentViewport.y === focusResult.viewport!.y &&
      currentViewport.zoom === focusResult.viewport!.zoom
        ? currentViewport
        : focusResult.viewport!,
    );
    void flowRef.current.setViewport(focusResult.viewport, {
      duration: ATTENTION_FOCUS_DURATION_MS,
    });
    return true;
  }, [
    attentionLensEnabled,
    attentionSourceEntries,
    computeAttentionFocusViewport,
    logAttentionDiagnostics,
    promptTokensVisible,
    selectedGraphNode?.id,
  ]);

  const clearAttentionLens = useCallback((options?: { restoreViewport?: boolean }) => {
    const shouldRestoreViewport = options?.restoreViewport ?? true;
    attentionAbortRef.current?.abort();
    attentionAbortRef.current = null;
    const savedViewport = attentionViewportRef.current;
    setAttentionLensEnabled(false);
    setAttentionAnalysis((currentValue) => (currentValue === null ? currentValue : null));
    setPinnedAttentionSourceIds((currentValue) => (currentValue.length === 0 ? currentValue : []));
    setAttentionError((currentValue) => (currentValue === null ? currentValue : null));
    setAttentionLoading(false);
    attentionViewportRef.current = null;

    if (shouldRestoreViewport && savedViewport) {
      restoreAttentionViewport(savedViewport);
    }
  }, [restoreAttentionViewport]);

  const toggleAttentionLens = useCallback(() => {
    if (attentionLensEnabled) {
      clearAttentionLens({ restoreViewport: true });
      return;
    }

    setPinnedAttentionSourceIds([]);
    setAttentionError(null);
    setAttentionLensEnabled(true);
  }, [attentionLensEnabled, clearAttentionLens]);

  const togglePromptTokens = useCallback(() => {
    if (!promptTokensAvailable) {
      return;
    }

    setShowPromptTokens((currentValue) => !currentValue);
  }, [promptTokensAvailable]);

  const handleAttentionSourceFocus = useCallback((sourceId: string, graphNodeId: string | null) => {
    setPinnedAttentionSourceIds((currentValue) =>
      currentValue.length === 1 && currentValue[0] === sourceId ? [] : [sourceId],
    );

    if (graphNodeId && displayNodeMap.has(graphNodeId)) {
      centerNodeRef.current(graphNodeId);
      return;
    }

    console.warn("[llmscope-attention] source-node-unavailable", {
      graphNodeId,
      sourceId,
    });
  }, [displayNodeMap]);

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
    attentionAbortRef.current?.abort();
    attentionAbortRef.current = null;
    attentionCacheRef.current.clear();
    setAttentionAnalysis((currentValue) => (currentValue === null ? currentValue : null));
    setPinnedAttentionSourceIds((currentValue) => (currentValue.length === 0 ? currentValue : []));
    setAttentionAdvancedOpen(false);
    setShowAllAttentionTokens(false);
    setShowPromptTokens(false);
    setAttentionError((currentValue) => (currentValue === null ? currentValue : null));
    setAttentionLoading(false);
    attentionViewportRef.current = null;
  }, [attentionGraphVersion, selectedProvider]);

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
    if (process.env.NODE_ENV === "production" || !promptTokensAvailable) {
      return;
    }

    const expectedPromptNodeCount = promptTokensVisible ? generationPromptTokens.length : 0;
    const expectedPromptSummaryCount = promptTokensVisible ? 0 : 1;
    const hasDuplicatePromptIds =
      new Set(promptDisplaySummary.promptNodeIds).size !== promptDisplaySummary.promptNodeCount;

    if (
      promptDisplaySummary.promptNodeCount !== expectedPromptNodeCount ||
      promptDisplaySummary.promptSummaryCount !== expectedPromptSummaryCount ||
      hasDuplicatePromptIds
    ) {
      console.warn("[llmscope-attention] prompt-display-invariant-failed", {
        expectedPromptNodeCount,
        expectedPromptSummaryCount,
        promptDisplaySummary,
      });
    }
  }, [
    generationPromptTokens.length,
    promptDisplaySummary,
    promptTokensAvailable,
    promptTokensVisible,
  ]);

  useEffect(() => {
    if (attentionLensEnabled && !attentionAvailable && !isSelectedPromptToken) {
      clearAttentionLens({ restoreViewport: false });
    }
  }, [attentionAvailable, attentionLensEnabled, clearAttentionLens, isSelectedPromptToken]);

  useEffect(() => {
    if (!attentionLensEnabled || !selectedGraphNode?.id) {
      return;
    }

    setPinnedAttentionSourceIds((currentValue) => (currentValue.length === 0 ? currentValue : []));
    setAttentionAnalysis((currentValue) => (currentValue === null ? currentValue : null));
  }, [attentionLensEnabled, selectedGraphNode?.id]);

  useEffect(() => {
    if (!attentionLensEnabled || !attentionRequest) {
      attentionAbortRef.current?.abort();
      attentionAbortRef.current = null;
      setAttentionLoading(false);
      setAttentionError((currentValue) => (currentValue === null ? currentValue : null));
      if (!attentionLensEnabled) {
        setAttentionAnalysis((currentValue) => (currentValue === null ? currentValue : null));
        setPinnedAttentionSourceIds((currentValue) => (currentValue.length === 0 ? currentValue : []));
      }
      return;
    }

    const cached = attentionCacheRef.current.get(attentionRequest.cacheKey) ?? null;
    if (cached) {
      setAttentionAnalysis((currentValue) =>
        JSON.stringify(currentValue) === JSON.stringify(cached) ? currentValue : cached,
      );
      setAttentionLoading(false);
      setAttentionError((currentValue) => (currentValue === null ? currentValue : null));
      return;
    }

    const controller = new AbortController();
    attentionAbortRef.current?.abort();
    attentionAbortRef.current = controller;
    setAttentionLoading(true);
    setAttentionError((currentValue) => (currentValue === null ? currentValue : null));

    void (async () => {
      try {
        const response = await fetch("/api/providers/hugging-face-local/attention", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(attentionRequest.request),
          signal: controller.signal,
        });

        if (!response.ok) {
          await throwApiError(response);
        }

        const payload = (await response.json()) as HuggingFaceAttentionResponse;
        if (controller.signal.aborted) {
          return;
        }

        const nextCache = attentionCacheRef.current;
        nextCache.set(attentionRequest.cacheKey, payload);
        while (nextCache.size > 32) {
          const oldestKey = nextCache.keys().next().value;
          if (!oldestKey) {
            break;
          }
          nextCache.delete(oldestKey);
        }
        setAttentionAnalysis((currentValue) =>
          JSON.stringify(currentValue) === JSON.stringify(payload) ? currentValue : payload,
        );
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setAttentionAnalysis(null);
        setPinnedAttentionSourceIds((currentValue) => (currentValue.length === 0 ? currentValue : []));
        setAttentionError(
          error instanceof Error ? error.message : "Unable to compute attention for this token.",
        );
      } finally {
        if (!controller.signal.aborted) {
          setAttentionLoading(false);
        }
      }
    })();

    return () => {
      controller.abort();
    };
  }, [attentionLensEnabled, attentionRequest]);

  useEffect(() => {
    if (!promptTokensVisible || previousPromptTokensVisibleRef.current) {
      previousPromptTokensVisibleRef.current = promptTokensVisible;
      return;
    }

    previousPromptTokensVisibleRef.current = promptTokensVisible;
    void waitForAnimationFrames(2).then(() => {
      logAttentionDiagnostics("prompt-tokens-shown");
    });
  }, [logAttentionDiagnostics, promptTokensVisible]);

  useEffect(() => {
    if (!attentionLensEnabled || !attentionAnalysis) {
      lastLoggedAttentionAnalysisRef.current = attentionAnalysis;
      return;
    }

    if (lastLoggedAttentionAnalysisRef.current === attentionAnalysis) {
      return;
    }

    lastLoggedAttentionAnalysisRef.current = attentionAnalysis;
    void waitForAnimationFrames(2).then(() => {
      logAttentionDiagnostics("attention-analysis-complete");
    });
  }, [attentionAnalysis, attentionLensEnabled, logAttentionDiagnostics]);

  useEffect(() => {
    const nextModel = findCompatibleModelId(models, selectedProvider, model);

    if (!nextModel || nextModel === model) {
      return;
    }

    setModel(nextModel);
  }, [model, models, selectedProvider]);

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
        if (attentionLensEnabled) {
          clearAttentionLens({ restoreViewport: true });
        }
        setContextMenu(null);
      }
    };

    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [attentionLensEnabled, clearAttentionLens]);

  async function loadModelCatalog(forceRefresh = false) {
    setIsLoadingModels(true);

    try {
      const response = await fetch(forceRefresh ? "/api/models?refresh=1" : "/api/models", {
        cache: "no-store",
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as ModelCatalogResponse;
      setModelCatalog(payload);
      setSelectedProvider((currentProvider) =>
        payload.providers.some((item) => item.id === currentProvider)
          ? (currentProvider as ProviderId)
          : (payload.default_provider as ProviderId),
      );
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
      setModelCatalog(FALLBACK_MODEL_CATALOG);
      setSelectedProvider(FALLBACK_MODEL_CATALOG.default_provider as ProviderId);
    } finally {
      setIsLoadingModels(false);
    }
  }

  const loadHuggingFaceLocalStatus = useCallback(async () => {
    setIsLoadingHuggingFaceStatus(true);

    try {
      const response = await fetch("/api/providers/hugging-face-local", {
        cache: "no-store",
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as HuggingFaceLocalStatusResponse;
      setHuggingFaceLocalStatus((currentValue) =>
        JSON.stringify(currentValue) === JSON.stringify(payload) ? currentValue : payload,
      );
    } catch (error) {
      void error;
      setHuggingFaceLocalStatus(null);
    } finally {
      setIsLoadingHuggingFaceStatus(false);
    }
  }, []);

  async function loadSelectedHuggingFaceModel() {
    if (selectedProvider !== "hugging_face") {
      return;
    }

    setIsSubmittingHuggingFaceAction(true);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/providers/hugging-face-local/load", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_id: model }),
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as HuggingFaceLocalStatusResponse;
      setHuggingFaceLocalStatus(payload);
      await loadModelCatalog(true);
      setBackendState("online");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to load the selected local model.",
      );
    } finally {
      setIsSubmittingHuggingFaceAction(false);
    }
  }

  async function unloadHuggingFaceModel() {
    if (selectedProvider !== "hugging_face") {
      return;
    }

    setIsSubmittingHuggingFaceAction(true);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/providers/hugging-face-local/unload", {
        method: "POST",
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as HuggingFaceLocalStatusResponse;
      setHuggingFaceLocalStatus(payload);
      await loadModelCatalog(true);
      setBackendState("online");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to unload the local model.",
      );
    } finally {
      setIsSubmittingHuggingFaceAction(false);
    }
  }

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

    await loadModelCatalog(true);
    await loadHuggingFaceLocalStatus();
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

      if (isActive) {
        await loadModelCatalog(false);
        await loadHuggingFaceLocalStatus();
      }
    })();

    return () => {
      isActive = false;
    };
  }, [loadHuggingFaceLocalStatus]);

  const handleProviderChange = useCallback(
    (nextProvider: ProviderId) => {
      setErrorMessage(null);
      setSelectedProvider((currentProvider) =>
        currentProvider === nextProvider ? currentProvider : nextProvider,
      );
      setModel((currentModel) => {
        const nextModel = findCompatibleModelId(models, nextProvider, currentModel);

        return nextModel && nextModel !== currentModel ? nextModel : currentModel;
      });
    },
    [models],
  );

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
    const selectionCandidate =
      next.selectedNodeId && (next.nodes ?? current.nodes).some((node) => node.id === next.selectedNodeId)
        ? next.selectedNodeId
        : current.selectedNodeId &&
            (next.nodes ?? current.nodes).some((node) => node.id === current.selectedNodeId)
          ? current.selectedNodeId
          : null;
    const normalizeStart = performance.now();
    const normalized = normalizeGraph(
      next.nodes ?? current.nodes,
      next.edges ?? current.edges,
      branchChoices,
      selectionCandidate,
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
    const node =
      displayNodeMap.get(nodeId) ??
      nodesRef.current.find((currentNode) => currentNode.id === nodeId) ??
      null;

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

  function materializeStoredAlternativesForSelection(nodeId: string) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (
      !node ||
      node.data.kind !== "token" ||
      !node.data.parentId ||
      node.data.topAlternatives.length === 0
    ) {
      return null;
    }

    const nextTokenGraph = materializeSourceAlternativesForNode(tokenGraphRef.current, nodeId);
    const parentNodeId = node.data.parentId;
    const parentNode = nodesRef.current.find((currentNode) => currentNode.id === parentNodeId);
    const parentRecord = nextTokenGraph.nodesById[parentNodeId];

    if (!parentNode || !parentRecord) {
      return null;
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

    return {
      nodes: syncFlowNodesWithTokenGraph(nextNodes, nextTokenGraph),
      edges: nextEdges,
    };
  }

  function activateReality(nodeId: string, options?: { pushHistory?: boolean }) {
    const nextChoices = buildRealityChoicesForNode(nodeId, nodesRef.current, branchChoicesRef.current);
    const materialized = materializeStoredAlternativesForSelection(nodeId);

    applyTransition(
      {
        ...(materialized ?? {}),
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
    setIsSentenceBarExpanded(false);
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
      providerCapabilities: payload.provider_capabilities,
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
        segmentId: trace.segment_id ?? null,
        continuationMode: trace.continuation_mode ?? "exact",
        decodedContribution: trace.decoded_contribution ?? trace.token,
        cumulativeDecodedText:
          trace.cumulative_decoded_text ?? trace.context_after ?? trace.text_preview,
        cumulativeTokenIds: trace.cumulative_token_ids ?? null,
        cumulativeLogProbability: trace.cumulative_log_probability ?? trace.log_probability ?? undefined,
        contextBefore: trace.context_before,
        contextAfter: trace.context_after,
        generationStep: trace.generation_step ?? trace.position,
        probability: trace.probability ?? 0,
        rawProbability: trace.raw_probability ?? 0,
        normalizedDisplayedProbability: trace.normalized_displayed_probability ?? 0,
        logProbability: trace.log_probability ?? 0,
        entropy: trace.entropy ?? 0,
        cumulativeProbability: trace.cumulative_probability ?? 0,
        latency: trace.latency_ms,
        depth: trace.position + 1,
        rank: 1,
        parentId,
        providerCapabilities: payload.provider_capabilities,
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
      const nextEdges = [
        ...edgesRef.current,
        buildEdge(parentId, syncedNextNode.id, trace.probability ?? trace.raw_probability ?? 0, true),
      ];
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
    const payloadProviderCapabilities =
      "provider_capabilities" in payload
        ? payload.provider_capabilities
        : parentNode.data.providerCapabilities;

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
            segmentId: candidate.segment_id ?? matchingChild.data.segmentId,
            continuationMode: candidate.continuation_mode ?? matchingChild.data.continuationMode,
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
            segmentId: candidate.segment_id ?? null,
            continuationMode: candidate.continuation_mode ?? "exact",
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
            providerCapabilities: payloadProviderCapabilities,
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
          provider: node.data.metadata.provider ?? generation?.request.provider ?? selectedProvider,
          model: node.data.requestModel,
          preset: node.data.requestPreset,
          temperature: node.data.requestTemperature,
          top_p: node.data.requestTopP,
          parent_node_id: nodeId,
          parent_token: node.data.tokenText,
          assistant_prefix: validation.assistantPrefix,
          prompt_token_ids: validation.promptTokenIds,
          canonical_prefix_token_ids: validation.canonicalPrefixTokenIds,
          generated_prefix_token_ids: validation.generatedPrefixTokenIds,
          reconstructed_prompt: validation.reconstructedPrompt,
          expected_prompt_length: validation.characterLength,
          expected_utf8_length: validation.utf8Length,
          expected_assistant_prefix_length: validation.assistantCharacterLength,
          expected_assistant_prefix_utf8_length: validation.assistantUtf8Length,
          expected_token_count: validation.tokenCount,
          selected_token_id: validation.selectedTokenId,
          selected_tokenizer_id: validation.selectedTokenizerId,
          model_revision: validation.modelRevision,
          tokenizer_identity: validation.tokenizerIdentity,
          tokenizer_revision: validation.tokenizerRevision,
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
    const materialized = materializeStoredAlternativesForSelection(nodeId);

    if (!materialized) {
      return false;
    }

    applyTransition(
      {
        nodes: materialized.nodes,
        edges: materialized.edges,
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

    if (!canMutateGraphTokenNode(node)) {
      setErrorMessage("Prompt tokens are fixed input context and cannot branch into alternatives.");
      return false;
    }

    if (!node.data.providerCapabilities.supports_branching) {
      setErrorMessage("This provider does not expose token-level alternatives for branching.");
      return false;
    }

    if (node.data.topAlternatives.length > 0) {
      return expandStoredAlternatives(nodeId, options);
    }

    return expandContinuationNode(nodeId, options);
  }

  expandNodeRef.current = expandTokenOccurrence;

  async function requestContinuationStep(
    nodeId: string,
    validation: ReturnType<typeof buildContinuationValidation>,
  ): Promise<ContinuationStepResult> {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!canMutateGraphTokenNode(node)) {
      const message = "Prompt tokens are fixed input context and cannot continue generation.";
      setErrorMessage(message);
      return {
        errorMessage: message,
        requested: false,
        success: false,
      };
    }

    if (node.data.status === "loading") {
      return {
        errorMessage: null,
        requested: false,
        success: false,
      };
    }

    if (!node.data.providerCapabilities.supports_continuation) {
      const message = "This provider does not support graph continuation from the selected node.";
      setErrorMessage(message);
      return {
        errorMessage: message,
        requested: false,
        success: false,
      };
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

    const requestBody = {
      root_prompt: validation.rootPrompt,
      provider: node.data.metadata.provider ?? generation?.request.provider ?? selectedProvider,
      model: node.data.requestModel,
      preset: node.data.requestPreset,
      temperature: node.data.requestTemperature,
      top_p: node.data.requestTopP,
      parent_node_id: nodeId,
      parent_token: node.data.tokenText,
      assistant_prefix: validation.assistantPrefix,
      prompt_token_ids: validation.promptTokenIds,
      canonical_prefix_token_ids: validation.canonicalPrefixTokenIds,
      generated_prefix_token_ids: validation.generatedPrefixTokenIds,
      reconstructed_prompt: validation.reconstructedPrompt,
      expected_prompt_length: validation.characterLength,
      expected_utf8_length: validation.utf8Length,
      expected_assistant_prefix_length: validation.assistantCharacterLength,
      expected_assistant_prefix_utf8_length: validation.assistantUtf8Length,
      expected_token_count: validation.tokenCount,
      selected_token_id: validation.selectedTokenId,
      selected_tokenizer_id: validation.selectedTokenizerId,
      model_revision: validation.modelRevision,
      tokenizer_identity: validation.tokenizerIdentity,
      tokenizer_revision: validation.tokenizerRevision,
      depth: node.data.depth,
      cumulative_probability: node.data.cumulativeProbability,
      variation: node.data.requestVariation,
      max_children: MAX_BRANCH_CHILDREN,
      demo_mode: node.data.requestDemoMode,
      cached_segment_id: readMetadataString(node.data.metadata, "cached_segment_id"),
      cached_token_index: readMetadataNumber(node.data.metadata, "next_cached_token_index"),
    };
    logContinuationDebug("continue-handler:request-built", {
      apiUrl: "/api/continue-node",
      nodeId,
      requestBody,
    });

    try {
      logContinuationDebug("continue-handler:fetch-started", {
        apiUrl: "/api/continue-node",
        nodeId,
      });
      const response = await fetch("/api/continue-node", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        await throwApiError(response);
      }

      const payload = (await response.json()) as ContinueGenerationResponse;
      logContinuationDebug("continue-handler:response", {
        action: payload.action,
        childIds: payload.children.map((candidate) => candidate.id),
        continuationMode: payload.continuation_mode,
        nodeId,
        remainingCachedTokens: payload.remaining_cached_tokens,
      });
      const parentNode = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

      if (!parentNode) {
        return {
          errorMessage:
            "The continuation parent node disappeared before the response could be merged.",
          requested: true,
          success: false,
        };
      }

      applyExpansionPayloadToCanvas(nodeId, parentNode, validation, payload, { pushHistory: false });
      setBackendState("online");
      return {
        errorMessage: null,
        requested: true,
        success: true,
      };
    } catch (error) {
      const errorCode = getErrorCode(error);
      const message =
        error instanceof Error ? error.message : "Unable to continue the selected branch.";
      setBackendState(errorCode === "PROVIDER_REQUEST_FAILED" ? "offline" : "online");
      setErrorMessage(message);
      logContinuationDebug("continue-handler:error", {
        errorCode,
        message,
        nodeId,
      });
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
      return {
        errorMessage: message,
        requested: true,
        success: false,
      };
    }
  }

  async function continueGenerationFrom(
    nodeId: string,
    steps: number,
    options?: ContinueGenerationOptions,
  ): Promise<ContinueGenerationResult> {
    let currentNodeId = nodeId;
    let requested = false;

    for (let step = 0; step < steps; step += 1) {
      const existingTarget = getPreferredContinuationTarget(currentNodeId);

      if (
        shouldReuseContinuationTarget({
          forceRequestFirstStep: options?.forceRequestFirstStep,
          hasExistingTarget: Boolean(existingTarget),
          stepIndex: step,
        }) &&
        existingTarget
      ) {
        logContinuationDebug("continue-handler:reused-existing-target", {
          currentNodeId,
          existingTarget,
          source: options?.source ?? "graph",
          step,
        });
        currentNodeId = existingTarget;
        activateReality(currentNodeId, { pushHistory: false });
        continue;
      }

      const validation = buildContinuationValidation(tokenGraphRef.current, currentNodeId);

      if (!validation.isValid) {
        const message = validation.warnings.join(" ");
        setErrorMessage(message);
        setContinuationPreview({
          nodeId: currentNodeId,
          steps: Math.max(1, steps - step),
          validation,
        });
        return {
          errorMessage: message,
          finalNodeId: currentNodeId,
          requested,
          success: false,
        };
      }

      const expanded = await requestContinuationStep(currentNodeId, validation);
      requested = requested || expanded.requested;

      if (!expanded.success) {
        return {
          errorMessage: expanded.errorMessage,
          finalNodeId: currentNodeId,
          requested,
          success: false,
        };
      }

      const nextTarget = getPreferredContinuationTarget(currentNodeId);

      if (!nextTarget) {
        pushHistorySnapshot(getCurrentSnapshot());
        return {
          errorMessage: null,
          finalNodeId: currentNodeId,
          requested,
          success: true,
        };
      }

      currentNodeId = nextTarget;
      activateReality(currentNodeId, { pushHistory: false });
    }

    pushHistorySnapshot(getCurrentSnapshot());
    return {
      errorMessage: null,
      finalNodeId: currentNodeId,
      requested,
      success: true,
    };
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
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!canMutateGraphTokenNode(node)) {
      setErrorMessage("Prompt tokens are fixed input context and cannot continue generation.");
      return;
    }

    const validation = buildContinuationValidation(tokenGraphRef.current, nodeId);
    setSelectedNodeId(nodeId);
    setContinuationPreviewError(null);
    setIsSubmittingContinuationPreview(false);
    setContinuationPreview({
      nodeId,
      steps,
      validation,
    });
  }

  async function handleSubmitContinuationPreview() {
    logContinuationDebug("continue-button:click", {
      nodeId: continuationPreview?.nodeId ?? null,
      pending: isSubmittingContinuationPreview,
      steps: continuationPreview?.steps ?? 0,
    });
    const pending = continuationPreview;

    if (!pending) {
      return;
    }

    const previewNode = tokenGraphRef.current.nodesById[pending.nodeId] ?? null;
    logContinuationDebug("continue-handler:entered", {
      attentionLoading,
      model: previewNode?.requestModel ?? null,
      nodeId: pending.nodeId,
      pending: isSubmittingContinuationPreview,
      provider: readMetadataString(previewNode?.metadata ?? null, "provider"),
      steps: pending.steps,
      validationMode: pending.validation.validationMode,
    });

    if (isSubmittingContinuationPreview) {
      return;
    }

    if (!pending.validation.isValid) {
      const message =
        pending.validation.warnings.join(" ") ||
        "Continuation is unavailable because the selected prefix failed validation.";
      setContinuationPreviewError(message);
      logContinuationDebug("continue-handler:validation", {
        isValid: false,
        message,
        nodeId: pending.nodeId,
      });
      return;
    }

    setContinuationPreviewError(null);
    setErrorMessage(null);
    setIsSubmittingContinuationPreview(true);
    logContinuationDebug("continue-handler:validation", {
      assistantPrefix: pending.validation.assistantPrefix,
      isValid: true,
      nodeId: pending.nodeId,
      tokenCount: pending.validation.tokenCount,
    });

    const result = await continueGenerationFrom(pending.nodeId, pending.steps, {
      forceRequestFirstStep: true,
      source: "preview-modal",
    });

    logContinuationDebug("continue-handler:finished", {
      errorMessage: result.errorMessage,
      finalNodeId: result.finalNodeId,
      requested: result.requested,
      success: result.success,
    });
    setIsSubmittingContinuationPreview(false);

    if (result.success) {
      setContinuationPreview(null);
      return;
    }

    setContinuationPreviewError(
      result.errorMessage ?? "Unable to continue the selected branch from this node.",
    );
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
      providerCapabilities: selectedCapabilities,
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
    setIsSentenceBarExpanded(false);
    setRequestVariation(nextVariation);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: promptValue,
          provider: selectedProvider,
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
        node.data.kind === "token" &&
        node.data.providerCapabilities.supports_branching &&
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
    if (node.data.kind === "prompt") {
      setContextMenu(null);
      setSelectedNodeId(node.id);
      centerNodeRef.current(node.id);
      return;
    }
    setSelectedNodeId(node.id);
    setContextMenu({
      nodeId: node.id,
      title: node.data.displayTokenText,
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const handleNodeClick = useCallback<NodeMouseHandler<TokenFlowNode>>((event, node) => {
    setContextMenu(null);

    if (node.data.kind === "prompt") {
      setSelectedNodeId(node.id);
      centerNodeRef.current(node.id);
      return;
    }

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
    if (node.data.kind === "prompt" || !node.data.providerCapabilities.supports_branching) {
      return;
    }
    void expandNodeRef.current(node.id, { pushHistory: true });
  }, []);

  const handleEdgesChange = useCallback((changes: EdgeChange<CanvasFlowEdge>[]) => {
    const probabilityChanges = changes.filter(
      (change) => change.type !== "add" || change.item.type !== "attentionEdge",
    ) as EdgeChange<ProbabilityFlowEdge>[];

    if (probabilityChanges.length === 0) {
      return;
    }

    const nextEdges = applyEdgeChanges<ProbabilityFlowEdge>(probabilityChanges, edgesRef.current);
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

  const handleNodeMouseEnter = useCallback<NodeMouseHandler<TokenFlowNode>>((_event, node) => {
    setHoveredNodeId(node.id);
  }, []);

  const handleNodeMouseLeave = useCallback<NodeMouseHandler<TokenFlowNode>>(() => {
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
  const contextMenuNode =
    contextMenu ? displayNodes.find((node) => node.id === contextMenu.nodeId) ?? null : null;
  const mutableContextMenuNode = canMutateGraphTokenNode(contextMenuNode) ? contextMenuNode : null;
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
          disabled:
            !mutableContextMenuNode ||
            !mutableContextMenuNode.data.providerCapabilities.supports_continuation,
          onSelect: () => {
            openContinuationPreview(contextMenu.nodeId, 1);
            setContextMenu(null);
          },
        },
        {
          label: "Generate deeper",
          disabled:
            !mutableContextMenuNode ||
            !mutableContextMenuNode.data.providerCapabilities.supports_continuation,
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
          disabled:
            !mutableContextMenuNode ||
            !mutableContextMenuNode.data.providerCapabilities.supports_branching,
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
          disabled: !contextMenuNode?.data.childCount,
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
              disabled={isSubmittingContinuationPreview}
              onClick={() => {
                setContinuationPreview(null);
                setContinuationPreviewError(null);
              }}
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
              <span>{`${continuationPreview.validation.assistantCharacterLength} chars`}</span>
              <span>{`${continuationPreview.validation.assistantUtf8Length} bytes`}</span>
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

            {continuationPreviewError ? (
              <div className="continuation-preview__warnings">
                <p>{continuationPreviewError}</p>
              </div>
            ) : null}

            {isSubmittingContinuationPreview ? (
              <div className="continuation-preview__ok">
                {`Continuing from "${continuationPreview.validation.assistantPrefix || "<empty>"}"...`}
              </div>
            ) : null}
          </div>

          <div className="continuation-preview__actions">
            <button
              className="explorer-button explorer-button--ghost"
              disabled={isSubmittingContinuationPreview}
              onClick={() => {
                setContinuationPreview(null);
                setContinuationPreviewError(null);
              }}
              type="button"
            >
              Cancel
            </button>
            <button
              className="explorer-button explorer-button--primary"
              disabled={!continuationPreview.validation.isValid || isSubmittingContinuationPreview}
              onClick={() => {
                void handleSubmitContinuationPreview();
              }}
              onPointerDown={() => {
                logContinuationDebug("continue-button:pointer-down", {
                  nodeId: continuationPreview.nodeId,
                  steps: continuationPreview.steps,
                });
              }}
              type="button"
            >
              {isSubmittingContinuationPreview ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Continuing...
                </>
              ) : (
                "Continue"
              )}
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
          <button
            className="tool-button"
            disabled={!activeCapabilities.supports_branching}
            onClick={() => void handleExpandAll()}
            type="button"
          >
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
              {activeCapabilities.supports_logprobs ? (
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
              ) : (
                <div className="tool-popover__section">
                  <p className="tool-popover__label">Provider capabilities</p>
                  <p className="inspector-empty">
                    This provider does not expose token-level probabilities.
                  </p>
                </div>
              )}
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
        attentionEnabled={attentionLensEnabled}
        attentionHint={ATTENTION_LENS_TOOLTIP}
        attentionTokens={currentRealityAttentionTokens}
        branchBreadcrumb={currentRealityBranchBreadcrumb}
        collapsed={isSentenceBarCollapsed}
        continuationModeLabel={currentRealityContinuationMode.label}
        continuationModeTitle={currentRealityContinuationMode.title}
        continuationModeTone={currentRealityContinuationMode.tone}
        conversationSections={currentRealityConversationSections}
        copyConversationText={currentRealityConversationCopyText}
        copyRawContextText={currentRealityRawContextText}
        copyTokenIdsText={currentRealityTokenIdCopyText}
        copyUserPromptText={currentRealityUserPromptCopyText}
        detailItems={currentRealityDetailItems}
        formattingSelection={currentRealityFormattingSelection}
        hasContent={hasSentenceContent}
        onSelectToken={(nodeId) => {
          activateReality(nodeId, { pushHistory: true });
          centerNode(nodeId);
        }}
        onToggleCollapse={() => setIsSentenceBarExpanded((currentValue) => !currentValue)}
        onToggleAttentionPin={(tokenId) =>
          handleAttentionSourceFocus(tokenId, attentionSourceEntryById.get(tokenId)?.graphNodeId ?? null)
        }
        probabilityMode={probabilityViewMode}
        remainingProbabilityMass={
          activeSentenceNodes[activeSentenceNodes.length - 1]?.data.remainingProbabilityMass ?? 0
        }
        promptTokenGroups={currentRealityPromptDisplayGroups}
        rawContextText={currentRealityRawContextText}
        selectedTokenId={selectedNode?.id ?? null}
        summaryItems={currentRealitySummaryItems}
        stats={currentRealityStats}
        supportsEntropy={activeCapabilities.supports_entropy}
        supportsLogprobs={activeCapabilities.supports_logprobs}
        summary={
          !hasSentenceContent
            ? "System, user, and assistant context appear here after generation."
            : selectedNode?.data.kind === "prompt"
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

      <GenerationPanel
        backendState={backendState}
        canGenerate={canGenerate}
        collapsed={isDockCollapsed}
        demoMode={demoMode}
        filteredModels={filteredModels}
        huggingFaceLocalStatus={huggingFaceLocalStatus}
        isCheckingHealth={isCheckingHealth}
        isGenerating={isGenerating}
        isLoadingHuggingFaceStatus={isLoadingHuggingFaceStatus}
        isLoadingModels={isLoadingModels}
        isSubmittingHuggingFaceAction={isSubmittingHuggingFaceAction}
        maxTokens={maxTokens}
        model={model}
        onApplyExample={(nextPrompt) => {
          setPrompt(nextPrompt);
          setErrorMessage(null);
        }}
        onClearPrompt={() => {
          setPrompt("");
          setErrorMessage(null);
        }}
        onGenerate={() => void handleSubmit()}
        onLoadSelectedHuggingFaceModel={() => void loadSelectedHuggingFaceModel()}
        onMaxTokensChange={(value) => {
          setMaxTokens(value);
          setErrorMessage(null);
        }}
        onModelChange={(value) => {
          setModel(value);
          setErrorMessage(null);
        }}
        onPresetChange={(value) => {
          setPreset(value);
          setErrorMessage(null);
        }}
        onPromptChange={(value) => {
          setPrompt(value);
          setErrorMessage(null);
        }}
        onProviderChange={(value) => {
          handleProviderChange(value);
        }}
        onRefreshHealth={() => void refreshHealth()}
        onSetDemoMode={(value) => setDemoMode(value)}
        onTemperatureChange={(value) => setTemperature(value)}
        onToggleCollapsed={() => setIsDockCollapsed((currentValue) => !currentValue)}
        onTopPChange={(value) => setTopP(value)}
        onTokenDisplayModeChange={(value) => setTokenDisplayMode(value)}
        onUnloadHuggingFaceModel={() => void unloadHuggingFaceModel()}
        preset={preset}
        presets={presets}
        prompt={prompt}
        providerStatusMessage={
          showHuggingFaceControls
            ? selectedHuggingFaceStatusMessage
            : selectedProviderStatusMessage
        }
        providerRecommendations={selectedProviderRecommendations}
        providers={providers}
        selectedCapabilities={selectedCapabilities}
        selectedHuggingFaceModelStatus={selectedHuggingFaceModelStatus}
        selectedProvider={selectedProvider}
        showHuggingFaceControls={showHuggingFaceControls}
        systemPromptState={systemPromptState}
        temperature={temperature}
        tokenDisplayMode={tokenDisplayMode}
        topP={topP}
      />

      <aside className="inspector-panel">
        <div className="inspector-panel__header">
          <div>
            <p className="inspector-panel__eyebrow">Inspector</p>
            <h2 className="inspector-panel__title">
              {selectedNode?.data.kind === "prompt"
                ? "Prompt"
                : selectedNodeDisplayLabel}
            </h2>
          </div>
        </div>

        {selectedNode ? (
          <div className="inspector-panel__content">
            <div className="inspector-section">
              <p className="inspector-section__label">
                {isSelectedPromptToken ? "Prompt token" : "Basic"}
              </p>
              <div className="inspector-grid-data">
                <div>
                  <dt>Chosen token</dt>
                  <dd>{selectedNodeDisplayLabel}</dd>
                </div>
                <div>
                  <dt>Probability</dt>
                  <dd>
                    {isSelectedPromptToken
                      ? "Input token"
                      : selectedNode.data.providerCapabilities.supports_logprobs
                      ? formatPercent(selectedNode.data.displayProbability)
                      : "Unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>{isSelectedPromptToken ? "Category" : "Mode"}</dt>
                  <dd>
                    {isSelectedPromptToken
                      ? selectedNode.data.sourceLabel
                      : selectedNode.data.providerCapabilities.supports_logprobs
                      ? getProbabilityModeLabel(probabilityViewMode)
                      : "Unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>Continuation</dt>
                  <dd>
                    {isSelectedPromptToken ? (
                      <span className="inspector-inline-badge">Input context</span>
                    ) : (
                      <span
                        className={`inspector-inline-badge inspector-inline-badge--${inspectorContinuationMode.tone}`}
                        title={inspectorContinuationMode.title ?? undefined}
                      >
                        {`Mode: ${inspectorContinuationMode.label}`}
                      </span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt>{isSelectedPromptToken ? "Position" : "Rank"}</dt>
                  <dd>{isSelectedPromptToken ? selectedNode.data.tokenIndex : (selectedRecord?.rank ?? "-")}</dd>
                </div>
                <div>
                  <dt>{isSelectedPromptToken ? "Special token" : "Entropy"}</dt>
                  <dd>
                    {isSelectedPromptToken
                      ? selectedNode.data.specialToken
                        ? "Yes"
                        : "No"
                      : selectedNode.data.providerCapabilities.supports_entropy
                      ? formatNumber(selectedRecord?.entropy)
                      : "Unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>{isSelectedPromptToken ? "Token id" : "Latency"}</dt>
                  <dd>{isSelectedPromptToken ? (selectedNode.data.tokenId ?? "-") : `${selectedRecord?.latencyMs ?? 0} ms`}</dd>
                </div>
                <div>
                  <dt>{isSelectedPromptToken ? "Scope" : "Top-K coverage"}</dt>
                  <dd>
                    {isSelectedPromptToken
                      ? getPromptCategoryLabel(selectedNode.data.sourceCategory)
                      : selectedNode.data.providerCapabilities.supports_branching
                      ? inspectorAlternatives.length > 0
                        ? formatPercent(inspectorCoverage)
                        : "-"
                      : "Unavailable"}
                  </dd>
                </div>
                <div>
                  <dt>Provider</dt>
                  <dd>{selectedProviderLabelForInspector}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{selectedNode.data.requestModel || generation?.request.model || "Unavailable"}</dd>
                </div>
              </div>
            </div>

            <div className="inspector-section">
              <p className="inspector-section__label">
                {isSelectedPromptToken ? "Token metadata" : "Context"}
              </p>
              <div className="inspector-block inspector-block--stack">
                {isSelectedPromptToken ? (
                  <>
                    <p>
                      <strong>Raw token</strong>
                      <span>{selectedNode.data.tokenText || "<empty>"}</span>
                    </p>
                    <p>
                      <strong>Decoded contribution</strong>
                      <span>{selectedNode.data.decodedContribution || "<empty>"}</span>
                    </p>
                    <p>
                      <strong>Source</strong>
                      <span>{selectedNode.data.sourceLabel}</span>
                    </p>
                    <p>
                      <strong>Canonical position</strong>
                      <span>{selectedNode.data.tokenIndex}</span>
                    </p>
                  </>
                ) : (
                  <>
                    <p>
                      <strong>Before token</strong>
                      <span>{selectedRecord?.contextBefore || "<empty>"}</span>
                    </p>
                    <p>
                      <strong>Through token</strong>
                      <span>{selectedRecord?.cumulativeDecodedText || "<empty>"}</span>
                    </p>
                    <p>
                      <strong>Interpretation</strong>
                      <span>{naturalReason}</span>
                    </p>
                  </>
                )}
              </div>
            </div>

            {showAttentionInspectorSection ? (
              <div className="inspector-section">
                <div className="inspector-section__heading">
                  <p className="inspector-section__label">Attention Lens</p>
                  <div className="inspector-section__meta">
                    <span className="inspector-inline-badge" title={ATTENTION_LENS_TOOLTIP}>
                      {attentionAnalysisMode === "representation" ? "Representation" : "Prediction"}
                    </span>
                    <span className="inspector-inline-badge">
                      Layer {effectiveAttentionLayer}
                    </span>
                  </div>
                </div>
                <div className="inspector-block inspector-block--stack">
                  <div className="attention-controls__row">
                    <button
                      className={`explorer-button${
                        attentionLensEnabled ? " explorer-button--primary" : " explorer-button--ghost"
                      }`}
                      disabled={!attentionAvailable}
                      onClick={() => toggleAttentionLens()}
                      type="button"
                    >
                      {attentionLensEnabled ? "Clear attention" : "Enable Attention Lens"}
                    </button>
                    <button
                      className="explorer-button explorer-button--ghost"
                      disabled={!promptTokensAvailable}
                      onClick={() => togglePromptTokens()}
                      title={promptTokensUnavailableReason ?? undefined}
                      type="button"
                    >
                      {promptTokensVisible ? "Hide prompt tokens" : "Show prompt tokens"}
                    </button>
                    {attentionLensEnabled ? (
                      <button
                        className="explorer-button explorer-button--ghost"
                        disabled={!canFocusAttention}
                        onClick={() => {
                          void focusAttentionNeighborhood();
                        }}
                        type="button"
                      >
                        Focus attention
                      </button>
                    ) : null}
                    {attentionLoading ? (
                      <button
                        className="explorer-button explorer-button--ghost"
                        onClick={() => attentionAbortRef.current?.abort()}
                        type="button"
                      >
                        Cancel
                      </button>
                    ) : null}
                  </div>

                  {!attentionLensEnabled && attentionUnavailableMessage ? (
                    <div className="inspector-empty">{attentionUnavailableMessage}</div>
                  ) : null}

                  {attentionLensEnabled ? (
                    <>
                      <div className="attention-summary">
                        <p className="attention-summary__title">{attentionHeadline}</p>
                        <p className="attention-controls__hint" title={ATTENTION_LENS_TOOLTIP}>
                          {ATTENTION_LENS_TOOLTIP}
                        </p>
                      </div>
                      <p className="attention-controls__hint attention-controls__hint--muted">
                        Temporary eager attention may run more slowly on local Qwen models.
                      </p>

                      {attentionLoading ? (
                        <div className="inspector-empty">Computing attention...</div>
                      ) : null}
                      {attentionError ? (
                        <div className="inspector-empty">{attentionError}</div>
                      ) : null}
                      {!attentionAnalysis && attentionUnavailableMessage ? (
                        <div className="inspector-empty">{attentionUnavailableMessage}</div>
                      ) : null}
                      {attentionAnalysis ? (
                        <>
                          <div className="attention-legend">
                            <span className="attention-legend__item">
                              <span className="attention-legend__swatch attention-legend__swatch--arc" />
                              {attentionHeadLabel}
                            </span>
                            <span className="attention-legend__item">{attentionModeDescription}</span>
                            <span className="attention-legend__item">
                              {attentionAnalysis.truncated_context
                                ? `Truncated ${attentionAnalysis.analyzed_context_length}/${attentionAnalysis.original_full_context_length}`
                                : `${attentionAnalysis.analyzed_context_length} analyzed tokens`}
                            </span>
                          </div>
                          <div className="attention-breakdown">
                            <span className="inspector-inline-badge">
                              Prompt attention {formatPercent(attentionMassBreakdown.prompt)}
                            </span>
                            <span className="inspector-inline-badge">
                              Earlier output {formatPercent(attentionMassBreakdown.generatedOutput)}
                            </span>
                            <span className="inspector-inline-badge">
                              Template/control {formatPercent(attentionMassBreakdown.template)}
                            </span>
                            <span className="inspector-inline-badge">
                              Top N coverage {formatPercent(attentionAnalysis.top_n_coverage)}
                            </span>
                          </div>
                          <p className="attention-source-list__summary">
                            {`${attentionSourceEntries.length} ranked sources · ${visibleAttentionSourceCount} visible`}
                          </p>
                          <div className="alternatives-list attention-source-list">
                            {attentionSourceEntries.map(({ graphNodeId, isCanvasNodeUnavailable, source, sourceId }) => (
                              <button
                                key={sourceId}
                                className={`alternative-row attention-source-row${
                                  pinnedAttentionSourceIdSet.has(sourceId)
                                    ? " attention-source-row--active"
                                    : ""
                                }`}
                                onClick={() => handleAttentionSourceFocus(sourceId, graphNodeId)}
                                type="button"
                              >
                                <span className="alternative-row__token">
                                  {`${source.rank}. ${source.display_token || source.raw_token}`}
                                </span>
                                <span className="alternative-row__probability">
                                  {formatPercent(source.attention_weight)}
                                </span>
                                <span className="attention-source-row__meta">
                                  {isCanvasNodeUnavailable
                                    ? "Source node unavailable on canvas"
                                    : `${source.source_label} · pos ${source.full_position}`}
                                </span>
                                <span className="attention-source-row__bar">
                                  <span
                                    className="attention-source-row__bar-fill"
                                    style={{ width: `${Math.max(source.attention_weight * 100, 4)}%` }}
                                  />
                                </span>
                              </button>
                            ))}
                          </div>
                        </>
                      ) : null}

                      <details
                        className="inspector-details"
                        onToggle={(event) =>
                          setAttentionAdvancedOpen((event.currentTarget as HTMLDetailsElement).open)
                        }
                        open={attentionAdvancedOpen}
                      >
                        <summary>Advanced</summary>
                        <div className="attention-controls__grid">
                          <label className="attention-controls__label">
                            <span>Mode</span>
                            <select
                              className="explorer-select"
                              disabled={attentionLoading}
                              onChange={(event) =>
                                setAttentionAnalysisMode(
                                  event.target.value as HuggingFaceAttentionAnalysisMode,
                                )
                              }
                              value={attentionAnalysisMode}
                            >
                              <option value="prediction">Prediction attention</option>
                              <option value="representation">Representation attention</option>
                            </select>
                          </label>
                          <label className="attention-controls__label">
                            <span>Top sources</span>
                            <select
                              className="explorer-select"
                              disabled={attentionLoading}
                              onChange={(event) => {
                                const nextValue = event.target.value;
                                if (nextValue === "all") {
                                  setShowAllAttentionTokens(true);
                                } else {
                                  setShowAllAttentionTokens(false);
                                  setAttentionTopN(Number(nextValue));
                                }
                              }}
                              value={showAllAttentionTokens ? "all" : String(attentionTopN)}
                            >
                              <option value="5">Top 5</option>
                              <option value="8">Top 8</option>
                              <option value="12">Top 12</option>
                              <option value="20">Top 20</option>
                              {canShowAllAttentionTokens ? <option value="all">All analyzed</option> : null}
                            </select>
                          </label>
                        </div>
                        <div className="attention-controls__row attention-controls__row--dense">
                          <button
                            className="icon-button"
                            disabled={attentionLoading || effectiveAttentionLayer <= 0}
                            onClick={() =>
                              setAttentionLayer((currentValue) =>
                                Math.max((currentValue ?? attentionDefaultLayer) - 1, 0),
                              )
                            }
                            type="button"
                          >
                            <ChevronLeft className="h-4 w-4" />
                          </button>
                          <label className="attention-controls__label">
                            <span>Layer</span>
                            <select
                              className="explorer-select"
                              disabled={attentionLoading || attentionLayerCount <= 0}
                              onChange={(event) => setAttentionLayer(Number(event.target.value))}
                              value={effectiveAttentionLayer}
                            >
                              {Array.from({ length: Math.max(attentionLayerCount, 1) }, (_, index) => (
                                <option key={index} value={index}>
                                  {index}
                                </option>
                              ))}
                            </select>
                          </label>
                          <button
                            className="icon-button"
                            disabled={
                              attentionLoading ||
                              attentionLayerCount <= 0 ||
                              effectiveAttentionLayer >= Math.max(attentionLayerCount - 1, 0)
                            }
                            onClick={() =>
                              setAttentionLayer((currentValue) =>
                                Math.min(
                                  (currentValue ?? attentionDefaultLayer) + 1,
                                  Math.max(attentionLayerCount - 1, 0),
                                ),
                              )
                            }
                            type="button"
                          >
                            <ChevronRight className="h-4 w-4" />
                          </button>
                        </div>
                        <div className="attention-controls__grid">
                          <label className="attention-controls__label">
                            <span>Head mode</span>
                            <select
                              className="explorer-select"
                              disabled={attentionLoading}
                              onChange={(event) => {
                                const nextMode = event.target.value as HuggingFaceAttentionAggregationMode;
                                setAttentionAggregationMode(nextMode);
                                if (nextMode !== "single_head") {
                                  setAttentionHeadIndex(0);
                                }
                              }}
                              value={attentionAggregationMode}
                            >
                              <option value="average_heads">Average heads</option>
                              <option value="max_heads">Max heads</option>
                              <option value="single_head">Single head</option>
                            </select>
                          </label>
                          <label className="attention-controls__label">
                            <span>Implementation</span>
                            <span className="attention-controls__static">
                              {huggingFaceLocalStatus?.active_model_attention_implementation ?? "sdpa -> eager"}
                            </span>
                          </label>
                        </div>
                        {attentionAggregationMode === "single_head" ? (
                          <div className="attention-controls__row attention-controls__row--dense">
                            <button
                              className="icon-button"
                              disabled={attentionLoading || attentionHeadIndex <= 0}
                              onClick={() => setAttentionHeadIndex((currentValue) => Math.max(currentValue - 1, 0))}
                              type="button"
                            >
                              <ChevronLeft className="h-4 w-4" />
                            </button>
                            <label className="attention-controls__label">
                              <span>Head</span>
                              <select
                                className="explorer-select"
                                disabled={attentionLoading || attentionHeadCount <= 0}
                                onChange={(event) => setAttentionHeadIndex(Number(event.target.value))}
                                value={attentionHeadIndex}
                              >
                                {Array.from({ length: Math.max(attentionHeadCount, 1) }, (_, index) => (
                                  <option key={index} value={index}>
                                    {index}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <button
                              className="icon-button"
                              disabled={
                                attentionLoading ||
                                attentionHeadCount <= 0 ||
                                attentionHeadIndex >= Math.max(attentionHeadCount - 1, 0)
                              }
                              onClick={() =>
                                setAttentionHeadIndex((currentValue) =>
                                  Math.min(currentValue + 1, Math.max(attentionHeadCount - 1, 0)),
                                )
                              }
                              type="button"
                            >
                              <ChevronRight className="h-4 w-4" />
                            </button>
                          </div>
                        ) : null}
                      </details>
                    </>
                  ) : attentionUnavailableMessage ? (
                    <div className="inspector-empty">{attentionUnavailableMessage}</div>
                  ) : null}
                </div>
              </div>
            ) : null}

            <div className="inspector-section">
              <div className="inspector-section__heading">
                <p className="inspector-section__label">Top alternatives</p>
                <div className="inspector-section__meta">
                  {!isSelectedPromptToken && selectedNode.data.providerCapabilities.supports_logprobs ? (
                    <span className="inspector-inline-badge">
                      {getProbabilityModeLabel(probabilityViewMode)}
                    </span>
                  ) : null}
                  {!isSelectedPromptToken &&
                  selectedNode.data.providerCapabilities.supports_logprobs &&
                  probabilityViewMode === "raw" ? (
                    <span className="inspector-inline-badge">
                      Other tokens {formatPercent(inspectorRemainingProbabilityMass)}
                    </span>
                  ) : null}
                </div>
              </div>
              {isSelectedPromptToken ? (
                <div className="inspector-empty">
                  Prompt tokens are fixed input context. They do not expose sampled alternatives.
                </div>
              ) : inspectorAlternatives.length > 0 ? (
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
              ) : !selectedNode.data.providerCapabilities.supports_branching ? (
                <div className="inspector-empty">
                  This provider does not expose token-level alternatives.
                </div>
              ) : selectedNode.data.distributionMessage ? (
                <div className="inspector-empty">No alternatives were returned for this token.</div>
              ) : (
                <div className="inspector-empty">Expand this branch to inspect competing futures.</div>
              )}
            </div>

            {isSelectedPromptToken ? (
              <details className="inspector-details" open={false}>
                <summary>Advanced</summary>
                <div className="inspector-section">
                  <div className="inspector-grid-data">
                    <div>
                      <dt>Raw token</dt>
                      <dd>{selectedNode.data.tokenText || "∅"}</dd>
                    </div>
                    <div>
                      <dt>Decoded token</dt>
                      <dd>{selectedNode.data.decodedContribution || "∅"}</dd>
                    </div>
                    <div>
                      <dt>Token id</dt>
                      <dd>{selectedNode.data.tokenId ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>Tokenizer id</dt>
                      <dd>{selectedNode.data.tokenizerId ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>Canonical position</dt>
                      <dd>{selectedNode.data.tokenIndex}</dd>
                    </div>
                    <div>
                      <dt>Source category</dt>
                      <dd>{getPromptCategoryLabel(selectedNode.data.sourceCategory)}</dd>
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
                      <span>{selectedNode.data.tokenBytes.length > 0 ? selectedNode.data.tokenBytes.join(" ") : "Unavailable"}</span>
                    </p>
                    <p>
                      <strong>Metadata</strong>
                      <span>
                        {Object.entries(selectedNode.data.metadata).length > 0
                          ? Object.entries(selectedNode.data.metadata)
                              .map(([key, value]) => `${key}: ${String(value)}`)
                              .join(" · ")
                          : "Unavailable"}
                      </span>
                    </p>
                  </div>
                </div>
              </details>
            ) : (
              <details className="inspector-details" open={false}>
                <summary>Advanced</summary>
                <div className="inspector-section">
                  <div className="inspector-grid-data">
                    <div>
                      <dt>Raw token</dt>
                      <dd>{selectedRecord?.rawToken || "∅"}</dd>
                    </div>
                    <div>
                      <dt>Decoded token</dt>
                      <dd>{selectedRecord?.decodedContribution || "∅"}</dd>
                    </div>
                    <div>
                      <dt>Token id</dt>
                      <dd>{selectedRecord?.tokenId ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>Tokenizer id</dt>
                      <dd>{selectedRecord?.tokenizerId ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>Raw probability</dt>
                      <dd>
                        {selectedNode.data.providerCapabilities.supports_logprobs
                          ? formatProbability(selectedRecord?.rawProbability)
                          : "Unavailable"}
                      </dd>
                    </div>
                    <div>
                      <dt>Log probability</dt>
                      <dd>
                        {selectedNode.data.providerCapabilities.supports_logprobs
                          ? formatNumber(selectedRecord?.logProbability)
                          : "Unavailable"}
                      </dd>
                    </div>
                    <div>
                      <dt>Cumulative probability</dt>
                      <dd>
                        {selectedNode.data.providerCapabilities.supports_logprobs
                          ? formatProbability(selectedRecord?.cumulativeProbability)
                          : "Unavailable"}
                      </dd>
                    </div>
                    <div>
                      <dt>Cumulative logprob</dt>
                      <dd>
                        {selectedNode.data.providerCapabilities.supports_logprobs
                          ? formatNumber(selectedRecord?.cumulativeLogProbability)
                          : "Unavailable"}
                      </dd>
                    </div>
                    <div>
                      <dt>Branch depth</dt>
                      <dd>{selectedRecord?.generationDepth ?? selectedNode.data.depth}</dd>
                    </div>
                    <div>
                      <dt>Parent id</dt>
                      <dd>{selectedRecord?.parentId ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>Generation step</dt>
                      <dd>{selectedRecord?.generationStep ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>Segment id</dt>
                      <dd>{selectedRecord?.segmentId ?? "-"}</dd>
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
                      <span>{selectedRecord?.tokenBytes.length ? selectedRecord.tokenBytes.join(" ") : "Unavailable"}</span>
                    </p>
                    <p>
                      <strong>Metadata</strong>
                      <span>
                        {selectedRecord && Object.entries(selectedRecord.metadata).length > 0
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
            )}

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

      <ReactFlow<TokenFlowNode, CanvasFlowEdge>
        className={isGraphInteracting ? "llmscope-flow llmscope-flow--interacting" : "llmscope-flow"}
        colorMode="dark"
        defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
        edgeTypes={edgeTypes}
        edges={renderEdges}
        fitView
        fitViewOptions={FIT_VIEW_OPTIONS}
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
