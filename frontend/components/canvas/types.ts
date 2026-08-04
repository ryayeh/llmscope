import type { Edge, Node } from "@xyflow/react";

export interface TokenNodeData extends Record<string, unknown> {
  kind: "prompt" | "token";
  tokenText: string;
  probability: number;
  logProbability: number;
  entropy: number;
  latency: number;
  tokenId: number;
  tokenizerId: number;
  textPreview: string;
  cumulativeProbability: number;
  depth: number;
  rank: number;
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
  requestVariation: number;
  responseMode: string;
  sourceNotes: string;
  reasoningIntent: string;
  reasoningStrategy: string;
  reasoningFocusTerms: string[];
  branchRationale: string | null;
  rawLogits: number[] | null;
}

export type TokenFlowNode = Node<TokenNodeData, "tokenCard">;

export interface ProbabilityEdgeData extends Record<string, unknown> {
  probability: number;
  isMainPath: boolean;
}

export type ProbabilityFlowEdge = Edge<ProbabilityEdgeData, "probabilityEdge">;
