import type {
  AttentionFlowEdge,
  TokenFlowNode,
} from "@/components/canvas/types";
import type {
  CanonicalPromptToken,
  HuggingFaceAttentionAnalysisMode,
  HuggingFaceAttentionAggregationMode,
  HuggingFaceAttentionRequest,
  HuggingFaceAttentionResponse,
} from "@/types/api";

const PROMPT_TOKEN_NODE_ID_PREFIX = "prompt-token-";

export interface AttentionStripToken {
  attentionWeight: number | null;
  decodedContribution: string;
  displayToken: string;
  fullPosition: number;
  graphTokenId: string | null;
  id: string;
  isPinned: boolean;
  isQuery: boolean;
  isSelectedToken: boolean;
  rawToken: string;
  sequenceScope: "prompt" | "generated";
  sourceCategory: string;
  sourceLabel: string;
  specialToken: boolean;
  tokenId: number;
}

interface BuildAttentionRequestParams {
  allowTruncatedRecompute: boolean;
  analysisMode: HuggingFaceAttentionAnalysisMode;
  generatedTokenIds: number[];
  maxConnections: number;
  maxContextTokens: number;
  modelId: string;
  modelRevision?: string | null;
  promptTokenIds: number[];
  promptTokens: CanonicalPromptToken[];
  selectedHead: number | null;
  selectedLayer: number;
  tokenizerIdentity?: string | null;
  tokenizerRevision?: string | null;
  aggregationMode: HuggingFaceAttentionAggregationMode;
}

interface BuildAttentionOverlayEdgesParams {
  analysis: HuggingFaceAttentionResponse;
  lineageNodeIds: string[];
  pinnedSourceTokenIds: Set<string>;
  promptNodeIdByPosition?: Map<number, string>;
  selectedNodeId: string;
}

interface BuildAttentionStripTokensParams {
  analysis: HuggingFaceAttentionResponse;
  lineageNodeIds: string[];
  pinnedSourceTokenIds: Set<string>;
  promptNodeIdByPosition?: Map<number, string>;
}

interface PromptLaneLayoutToken {
  fullPosition: number;
  height: number;
  sourceCategory: CanonicalPromptToken["source_category"];
  width: number;
}

interface PromptLaneLayoutParams {
  laneAnchorX: number;
  laneAnchorY: number;
  laneAnchorHeight: number;
  maxRowWidth?: number;
  nodeGap?: number;
  rowGap?: number;
  sectionGap?: number;
  tokens: PromptLaneLayoutToken[];
}

export interface PromptLaneLayoutItem {
  fullPosition: number;
  row: number;
  x: number;
  y: number;
}

interface BuildAttentionFocusNodeIdsParams {
  displayNodes: TokenFlowNode[];
  selectedNodeId: string | null;
  sourceGraphNodeIds: Array<string | null>;
}

export interface PromptDisplayNodeSummary {
  promptNodeCount: number;
  promptNodeIds: string[];
  promptSummaryCount: number;
}

export interface LiveCanvasFocusNode {
  height: number | null;
  hidden: boolean;
  id: string;
  inDom: boolean;
  width: number | null;
  x: number | null;
  y: number | null;
}

export interface DeterministicFocusViewport {
  bounds: {
    height: number;
    maxX: number;
    maxY: number;
    minX: number;
    minY: number;
    width: number;
  } | null;
  includedIds: string[];
  viewport: {
    x: number;
    y: number;
    zoom: number;
  } | null;
}

export function buildAttentionTokenId(scope: "prompt" | "generated", fullPosition: number) {
  return `attention:${scope}:${fullPosition}`;
}

export function buildPromptTokenNodeId(fullPosition: number) {
  return `${PROMPT_TOKEN_NODE_ID_PREFIX}${fullPosition}`;
}

export function isPromptTokenNodeId(nodeId: string | null) {
  return typeof nodeId === "string" && nodeId.startsWith(PROMPT_TOKEN_NODE_ID_PREFIX);
}

export function canMutateGraphTokenNode(
  node: TokenFlowNode | null | undefined,
): node is TokenFlowNode & { data: TokenFlowNode["data"] & { kind: "token" } } {
  return Boolean(node && node.data.kind === "token");
}

export function summarizePromptDisplayNodes(
  displayNodes: TokenFlowNode[],
): PromptDisplayNodeSummary {
  const promptNodeIds = displayNodes
    .map((node) => node.id)
    .filter((nodeId) => isPromptTokenNodeId(nodeId));

  return {
    promptNodeCount: promptNodeIds.length,
    promptNodeIds,
    promptSummaryCount: displayNodes.filter((node) => node.id === "root").length,
  };
}

export function layoutPromptTokenLane(
  params: PromptLaneLayoutParams,
): PromptLaneLayoutItem[] {
  const maxRowWidth = params.maxRowWidth ?? 760;
  const nodeGap = params.nodeGap ?? 12;
  const rowGap = params.rowGap ?? 18;
  const sectionGap = params.sectionGap ?? 36;
  const placements: PromptLaneLayoutItem[] = [];
  const baseY = params.laneAnchorY;
  const laneRightX = params.laneAnchorX;
  const defaultRowHeight = params.tokens.reduce(
    (highest, token) => Math.max(highest, token.height),
    params.laneAnchorHeight,
  );

  let rowIndex = 0;
  let rowWidth = 0;
  let cursorRight = laneRightX;

  for (let index = params.tokens.length - 1; index >= 0; index -= 1) {
    const token = params.tokens[index];
    const nextToken = params.tokens[index + 1] ?? null;
    const gapBeforeToken =
      rowWidth === 0
        ? 0
        : nextToken && nextToken.sourceCategory !== token.sourceCategory
          ? sectionGap
          : nodeGap;

    if (rowWidth > 0 && rowWidth + gapBeforeToken + token.width > maxRowWidth) {
      rowIndex += 1;
      rowWidth = 0;
      cursorRight = laneRightX;
    }

    const appliedGap =
      rowWidth === 0
        ? 0
        : nextToken && nextToken.sourceCategory !== token.sourceCategory
          ? sectionGap
          : nodeGap;

    cursorRight -= appliedGap;
    rowWidth += appliedGap;
    cursorRight -= token.width;
    rowWidth += token.width;

    placements.unshift({
      fullPosition: token.fullPosition,
      row: rowIndex,
      x: cursorRight,
      y: baseY - rowIndex * (defaultRowHeight + rowGap),
    });
  }

  return placements;
}

export function buildAttentionFocusNodeIds(
  params: BuildAttentionFocusNodeIdsParams,
) {
  if (!params.selectedNodeId) {
    return [];
  }

  const visibleNodeIds = new Set(
    params.displayNodes
      .filter(
        (node) =>
          !node.hidden &&
          Number.isFinite(node.position.x) &&
          Number.isFinite(node.position.y),
      )
      .map((node) => node.id),
  );
  const dedupedIds = new Set<string>();

  const includeNodeId = (nodeId: string | null) => {
    if (!nodeId || !visibleNodeIds.has(nodeId)) {
      return;
    }

    dedupedIds.add(nodeId);
  };

  includeNodeId(params.selectedNodeId);

  for (const sourceGraphNodeId of params.sourceGraphNodeIds) {
    includeNodeId(sourceGraphNodeId);
  }

  return [...dedupedIds];
}

export function buildDeterministicFocusViewport(params: {
  containerHeight: number;
  containerWidth: number;
  maxZoom: number;
  minZoom: number;
  nodes: LiveCanvasFocusNode[];
  padding: number;
  selectedNodeId: string | null;
  sourceNodeIds: Array<string | null>;
  viewportInset?: {
    bottom?: number;
    left?: number;
    right?: number;
    top?: number;
  };
}): DeterministicFocusViewport {
  if (
    !params.selectedNodeId ||
    !Number.isFinite(params.containerWidth) ||
    !Number.isFinite(params.containerHeight) ||
    params.containerWidth <= 0 ||
    params.containerHeight <= 0
  ) {
    return {
      bounds: null,
      includedIds: [],
      viewport: null,
    };
  }

  const nodeById = new Map(params.nodes.map((node) => [node.id, node]));
  const insetLeft = Math.max(params.viewportInset?.left ?? 0, 0);
  const insetRight = Math.max(params.viewportInset?.right ?? 0, 0);
  const insetTop = Math.max(params.viewportInset?.top ?? 0, 0);
  const insetBottom = Math.max(params.viewportInset?.bottom ?? 0, 0);
  const availableWidth = params.containerWidth - insetLeft - insetRight;
  const availableHeight = params.containerHeight - insetTop - insetBottom;

  if (availableWidth <= 0 || availableHeight <= 0) {
    return {
      bounds: null,
      includedIds: [],
      viewport: null,
    };
  }

  const orderedIds = [params.selectedNodeId, ...params.sourceNodeIds].filter(
    (nodeId): nodeId is string => Boolean(nodeId),
  );
  const includedIds: string[] = [];
  const includedNodes: LiveCanvasFocusNode[] = [];
  const seenIds = new Set<string>();

  for (const nodeId of orderedIds) {
    if (seenIds.has(nodeId)) {
      continue;
    }

    seenIds.add(nodeId);
    const node = nodeById.get(nodeId);

    if (
      !node ||
      node.hidden ||
      !node.inDom ||
      !Number.isFinite(node.x) ||
      !Number.isFinite(node.y) ||
      !Number.isFinite(node.width) ||
      !Number.isFinite(node.height) ||
      (node.width ?? 0) <= 0 ||
      (node.height ?? 0) <= 0
    ) {
      continue;
    }

    includedIds.push(nodeId);
    includedNodes.push(node);
  }

  if (includedIds[0] !== params.selectedNodeId || includedNodes.length === 0) {
    return {
      bounds: null,
      includedIds,
      viewport: null,
    };
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const node of includedNodes) {
    const nodeRight = (node.x ?? 0) + (node.width ?? 0);
    const nodeBottom = (node.y ?? 0) + (node.height ?? 0);
    minX = Math.min(minX, node.x ?? 0);
    minY = Math.min(minY, node.y ?? 0);
    maxX = Math.max(maxX, nodeRight);
    maxY = Math.max(maxY, nodeBottom);
  }

  const boundsWidth = maxX - minX;
  const boundsHeight = maxY - minY;

  if (
    !Number.isFinite(minX) ||
    !Number.isFinite(minY) ||
    !Number.isFinite(maxX) ||
    !Number.isFinite(maxY) ||
    boundsWidth <= 0 ||
    boundsHeight <= 0 ||
    boundsWidth > 100_000 ||
    boundsHeight > 100_000
  ) {
    return {
      bounds: null,
      includedIds,
      viewport: null,
    };
  }

  const paddedWidth = boundsWidth * (1 + params.padding * 2);
  const paddedHeight = boundsHeight * (1 + params.padding * 2);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const unclampedZoom = Math.min(
    availableWidth / Math.max(paddedWidth, 1),
    availableHeight / Math.max(paddedHeight, 1),
  );
  const zoom = Math.min(params.maxZoom, Math.max(params.minZoom, unclampedZoom));

  if (!Number.isFinite(zoom) || zoom <= 0) {
    return {
      bounds: null,
      includedIds,
      viewport: null,
    };
  }

  return {
    bounds: {
      height: boundsHeight,
      maxX,
      maxY,
      minX,
      minY,
      width: boundsWidth,
    },
    includedIds,
    viewport: {
      x: insetLeft + availableWidth / 2 - centerX * zoom,
      y: insetTop + availableHeight / 2 - centerY * zoom,
      zoom,
    },
  };
}

function getGraphNodeIdForAttentionToken(
  lineageNodeIds: string[],
  generatedTokenIndex: number | null | undefined,
) {
  if (generatedTokenIndex === null || generatedTokenIndex === undefined) {
    return null;
  }

  return lineageNodeIds[generatedTokenIndex + 1] ?? null;
}

export function canUseAttentionLens(node: TokenFlowNode | null) {
  if (!node || node.data.kind !== "token") {
    return false;
  }

  return (
    node.data.providerCapabilities.supports_attention &&
    node.data.continuationMode === "exact" &&
    (node.data.metadata.provider ?? null) === "hugging_face" &&
    Array.isArray(node.data.cumulativeTokenIds) &&
    node.data.cumulativeTokenIds.length > 0
  );
}

export function buildAttentionRequestPayload(
  params: BuildAttentionRequestParams,
): HuggingFaceAttentionRequest {
  return {
    model_id: params.modelId,
    model_revision: params.modelRevision ?? null,
    tokenizer_identity: params.tokenizerIdentity ?? null,
    tokenizer_revision: params.tokenizerRevision ?? null,
    prompt_token_ids: [...params.promptTokenIds],
    prompt_tokens: params.promptTokens.map((token) => ({
      ...token,
      token_bytes: [...token.token_bytes],
    })),
    generated_token_ids: [...params.generatedTokenIds],
    selected_generated_token_index: Math.max(params.generatedTokenIds.length - 1, 0),
    selected_layer: params.selectedLayer,
    selected_head: params.selectedHead,
    analysis_mode: params.analysisMode,
    aggregation_mode: params.aggregationMode,
    max_connections: params.maxConnections,
    max_context_tokens: params.maxContextTokens,
    allow_truncated_recompute: params.allowTruncatedRecompute,
  };
}

export function buildAttentionCacheKey(request: HuggingFaceAttentionRequest) {
  return JSON.stringify(request);
}

export function buildAttentionHeadLabel(
  aggregationMode: HuggingFaceAttentionAggregationMode,
  selectedHead: number | null | undefined,
) {
  if (aggregationMode === "single_head") {
    return `Head ${selectedHead ?? 0}`;
  }

  return aggregationMode === "max_heads" ? "Max heads" : "Average heads";
}

export function buildAttentionOverlayEdges(
  params: BuildAttentionOverlayEdgesParams,
): AttentionFlowEdge[] {
  const headLabel = buildAttentionHeadLabel(
    params.analysis.aggregation_mode,
    params.analysis.selected_head,
  );
  const edgesById = new Map<string, AttentionFlowEdge>();

  for (const source of params.analysis.sources) {
    const sourceGraphNodeId =
      source.sequence_scope === "generated"
        ? getGraphNodeIdForAttentionToken(params.lineageNodeIds, source.generated_token_index)
        : params.promptNodeIdByPosition?.get(source.full_position) ?? null;

    if (!sourceGraphNodeId) {
      continue;
    }

    const tokenId = buildAttentionTokenId(source.sequence_scope, source.full_position);
    const edgeId =
      `attention:${params.selectedNodeId}:${source.sequence_scope}:${source.full_position}:${params.analysis.selected_layer}:${params.analysis.selected_head ?? params.analysis.aggregation_mode}`;
    const existing = edgesById.get(edgeId);

    if (existing) {
      const existingData = existing.data;
      if (!existingData) {
        continue;
      }
      edgesById.set(edgeId, {
        ...existing,
        data: {
          ...existingData,
          weight: existingData.weight + source.attention_weight,
          isPinned: existingData.isPinned || params.pinnedSourceTokenIds.has(tokenId),
        },
      });
      continue;
    }

    edgesById.set(edgeId, {
      id: edgeId,
      source: params.selectedNodeId,
      target: sourceGraphNodeId,
      type: "attentionEdge",
      selectable: false,
      animated: false,
      data: {
        analysisMode: params.analysis.analysis_mode,
        queryPosition: params.analysis.query_position,
        weight: source.attention_weight,
        rank: source.rank,
        layer: params.analysis.selected_layer,
        headLabel,
        sourceDisplayToken: source.display_token,
        targetDisplayToken: params.analysis.selected_token.display_token,
        sourceScope: source.sequence_scope,
        sourceCategory: source.source_category,
        sourceLabel: source.source_label,
        sourceFullPosition: source.full_position,
        targetFullPosition: params.analysis.selected_token.full_position,
        isPinned: params.pinnedSourceTokenIds.has(tokenId),
        isDimmed: false,
      },
    });
  }

  return [...edgesById.values()];
}

export function buildAttentionStripTokens(
  params: BuildAttentionStripTokensParams,
): AttentionStripToken[] {
  return params.analysis.analyzed_tokens.map((token) => {
    const tokenId = buildAttentionTokenId(token.sequence_scope, token.full_position);
    return {
      attentionWeight: token.attention_weight ?? null,
      decodedContribution: token.decoded_contribution,
      displayToken: token.display_token,
      fullPosition: token.full_position,
      graphTokenId:
        token.sequence_scope === "generated"
          ? getGraphNodeIdForAttentionToken(params.lineageNodeIds, token.generated_token_index)
          : params.promptNodeIdByPosition?.get(token.full_position) ?? null,
      id: tokenId,
      isPinned: params.pinnedSourceTokenIds.has(tokenId),
      isQuery: token.is_query,
      isSelectedToken: token.is_selected_token,
      rawToken: token.raw_token,
      sequenceScope: token.sequence_scope,
      sourceCategory: token.source_category,
      sourceLabel: token.source_label,
      specialToken: token.special_token,
      tokenId: token.token_id,
    };
  });
}
