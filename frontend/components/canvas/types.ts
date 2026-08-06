import type { Edge, Node } from "@xyflow/react";

export type ProbabilityViewMode = "normalized" | "raw";

export interface InspectorAlternative {
  branchId: string | null;
  token: string;
  displayToken: string | null;
  tokenBytes: number[];
  decodedContribution: string | null;
  cumulativeDecodedText: string | null;
  cumulativeRawText: string | null;
  cumulativeTokenIds: number[] | null;
  cumulativeLogProbability: number | null;
  probability: number;
  rawProbability: number | null;
  normalizedDisplayedProbability: number | null;
  logProbability: number | null;
  rank: number | null;
  contextBefore: string | null;
  contextAfter: string | null;
  finishReason: string | null;
  rationale: string | null;
  tokenId: number | null;
  textPreview: string | null;
  nodeId: string | null;
  generationStep: number | null;
  metadata: Record<string, string | number | boolean | null> | null;
}

export interface TokenNodeData extends Record<string, unknown> {
  kind: "prompt" | "token";
  generationId: string;
  branchId: string;
  tokenText: string;
  displayTokenText: string;
  decodedContribution: string;
  cumulativeDecodedText: string;
  cumulativeRawText: string;
  cumulativeTokenIds: number[] | null;
  cumulativeLogProbability: number;
  tokenBytes: number[];
  utf8Length: number;
  characterLength: number;
  leadingWhitespaceCount: number;
  trailingWhitespaceCount: number;
  probability: number;
  rawProbability: number;
  normalizedDisplayedProbability: number;
  displayProbability: number;
  probabilityCoverage: number;
  remainingProbabilityMass: number;
  probabilityMode: ProbabilityViewMode;
  logProbability: number;
  entropy: number;
  latency: number;
  tokenId: number | null;
  tokenizerId: number | null;
  generationStep: number;
  textPreview: string;
  contextBefore: string;
  contextAfter: string;
  cumulativeProbability: number;
  branchProbability: number;
  depth: number;
  rank: number;
  finishReason: string | null;
  parentId: string | null;
  isMainPath: boolean;
  isCollapsed: boolean;
  distributionRequested: boolean;
  childCount: number;
  status: "idle" | "loading" | "ready";
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
  rawLogits: number[] | null;
  sourceAlternatives: InspectorAlternative[];
  distributionMessage: string | null;
  isSearchMatch: boolean;
  isSearchFocused: boolean;
  isDimmed: boolean;
  isActiveReality: boolean;
  isPinned: boolean;
}

export type TokenFlowNode = Node<TokenNodeData, "tokenCard">;

export interface ProbabilityEdgeData extends Record<string, unknown> {
  probability: number;
  rawProbability: number;
  probabilityCoverage: number;
  remainingProbabilityMass: number;
  probabilityMode: ProbabilityViewMode;
  isMainPath: boolean;
  isActiveReality: boolean;
  isDimmed: boolean;
  isFocused: boolean;
}

export type ProbabilityFlowEdge = Edge<ProbabilityEdgeData, "probabilityEdge">;
