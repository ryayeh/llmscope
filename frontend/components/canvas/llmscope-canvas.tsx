"use client";

import { useEffect, useRef, useState } from "react";

import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  applyEdgeChanges,
  applyNodeChanges,
  type EdgeChange,
  type NodeChange,
  type NodeMouseHandler,
  type ReactFlowInstance,
} from "@xyflow/react";
import { LoaderCircle, Orbit, Play, RefreshCcw, Sparkles } from "lucide-react";

import { ProbabilityEdge } from "@/components/canvas/probability-edge";
import { TokenContextMenu } from "@/components/canvas/token-context-menu";
import { TokenNode } from "@/components/canvas/token-node";
import type {
  ProbabilityFlowEdge,
  TokenFlowNode,
  TokenNodeData,
} from "@/components/canvas/types";
import type {
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
const INITIAL_MAX_TOKENS = 256;
const HORIZONTAL_GAP = 320;
const VERTICAL_GAP = 168;
const X_COLLISION_THRESHOLD = 220;
const Y_COLLISION_THRESHOLD = 142;
const MAX_BRANCH_CHILDREN = 4;

type BackendState = "checking" | "online" | "offline";

interface ContextMenuState {
  canCollapse: boolean;
  canExpand: boolean;
  nodeId: string;
  title: string;
  x: number;
  y: number;
}

interface ReasoningBundle {
  focusTerms: string[];
  intent: string;
  notes: string;
  strategy: string;
}

const nodeTypes = {
  tokenCard: TokenNode,
};

const edgeTypes = {
  probabilityEdge: ProbabilityEdge,
};

function wait(durationMs: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function formatProbability(value: number) {
  return value.toFixed(4);
}

function formatNumber(value: number) {
  return value.toFixed(4);
}

function joinTokenText(tokens: string[]) {
  let text = "";

  for (const token of tokens) {
    if (!text) {
      text = token;
      continue;
    }

    if (/^[.,!?;:)\]]$/.test(token)) {
      text = `${text}${token}`;
      continue;
    }

    if (["'s", "'re", "'ve", "'ll"].includes(token)) {
      text = `${text}${token}`;
      continue;
    }

    if (["(", "[", "{"].includes(token)) {
      text = `${text} ${token}`;
      continue;
    }

    text = `${text} ${token}`;
  }

  return text;
}

async function parseApiError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as
      | { detail?: string | Array<{ msg?: string }> }
      | undefined;

    if (typeof payload?.detail === "string") {
      return payload.detail;
    }

    if (Array.isArray(payload?.detail)) {
      const messages = payload.detail
        .map((item) => item.msg)
        .filter((message): message is string => Boolean(message));

      if (messages.length > 0) {
        return messages.join(", ");
      }
    }
  } catch {
    // Fall through to the generic message below.
  }

  return `Request failed with status ${response.status}.`;
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

function getNodePosition(node: TokenFlowNode) {
  return node.position;
}

function reasoningBundleFromGeneration(payload?: GenerationResponse | null): ReasoningBundle {
  return {
    notes: payload?.notes ?? "",
    intent: payload?.insights.detected_intent ?? "",
    strategy: payload?.insights.response_strategy ?? "",
    focusTerms: payload?.insights.focus_terms ?? [],
  };
}

function buildPromptNode({
  prompt,
  model,
  preset,
  reasoning,
  responseMode,
  status,
  temperature,
  variation,
}: {
  model: string;
  preset: string;
  prompt: string;
  reasoning: ReasoningBundle;
  responseMode: string;
  status: TokenNodeData["status"];
  temperature: number;
  variation: number;
}): TokenFlowNode {
  return {
    id: "root",
    type: "tokenCard",
    position: {
      x: 0,
      y: 0,
    },
    data: {
      kind: "prompt",
      tokenText: prompt,
      probability: 1,
      logProbability: 0,
      entropy: 0,
      latency: 0,
      tokenId: 0,
      tokenizerId: 0,
      textPreview: prompt,
      cumulativeProbability: 1,
      depth: 0,
      rank: 1,
      parentId: null,
      isMainPath: true,
      isCollapsed: false,
      distributionRequested: false,
      childCount: 0,
      status,
      requestPrompt: prompt,
      requestModel: model,
      requestPreset: preset,
      requestTemperature: temperature,
      requestVariation: variation,
      responseMode,
      sourceNotes: reasoning.notes,
      reasoningIntent: reasoning.intent,
      reasoningStrategy: reasoning.strategy,
      reasoningFocusTerms: reasoning.focusTerms,
      branchRationale: null,
      rawLogits: null,
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
  prompt,
  rank,
  rawLogits,
  reasoning,
  responseMode,
  status,
  temperature,
  textPreview,
  token,
  tokenId,
  variation,
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
  prompt: string;
  rank: number;
  rawLogits: number[] | null;
  reasoning: ReasoningBundle;
  responseMode: string;
  status: TokenNodeData["status"];
  temperature: number;
  textPreview: string;
  token: string;
  tokenId: number;
  variation: number;
}): TokenFlowNode {
  return {
    id,
    type: "tokenCard",
    position,
    data: {
      kind: "token",
      tokenText: token,
      probability,
      logProbability,
      entropy,
      latency,
      tokenId,
      tokenizerId: tokenId,
      textPreview,
      cumulativeProbability,
      depth,
      rank,
      parentId,
      isMainPath,
      isCollapsed: false,
      distributionRequested: false,
      childCount: 0,
      status,
      requestPrompt: prompt,
      requestModel: model,
      requestPreset: preset,
      requestTemperature: temperature,
      requestVariation: variation,
      responseMode,
      sourceNotes: reasoning.notes,
      reasoningIntent: reasoning.intent,
      reasoningStrategy: reasoning.strategy,
      reasoningFocusTerms: reasoning.focusTerms,
      branchRationale,
      rawLogits,
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
      isMainPath,
    },
  };
}

function branchOffset(rank: number) {
  if (rank <= 1) {
    return 0;
  }

  const level = Math.ceil((rank - 1) / 2);
  return (rank % 2 === 0 ? -1 : 1) * level;
}

function normalizeGraph(nodes: TokenFlowNode[], edges: ProbabilityFlowEdge[]) {
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

  return {
    nodes: nextNodes,
    edges: nextEdges,
  };
}

function findAvailableY(
  x: number,
  targetY: number,
  nodes: TokenFlowNode[],
  ignoreIds: Set<string>,
) {
  let candidateY = targetY;
  let attempt = 0;

  while (attempt < 40) {
    const collides = nodes.some((node) => {
      if (node.hidden || ignoreIds.has(node.id)) {
        return false;
      }

      const position = getNodePosition(node);

      return (
        Math.abs(position.x - x) < X_COLLISION_THRESHOLD &&
        Math.abs(position.y - candidateY) < Y_COLLISION_THRESHOLD
      );
    });

    if (!collides) {
      return candidateY;
    }

    attempt += 1;
    const step = Math.ceil(attempt / 2) * VERTICAL_GAP;
    candidateY = targetY + step * (attempt % 2 === 0 ? -1 : 1);
  }

  return candidateY;
}

function getMiniMapColor(node: TokenFlowNode) {
  if (node.data.kind === "prompt" || node.data.isMainPath) {
    return "#38bdf8";
  }

  if (node.data.probability < 0.36) {
    return "#8b5cf6";
  }

  return "#64748b";
}

function Workspace() {
  const [prompt, setPrompt] = useState(INITIAL_PROMPT);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogResponse>(
    FALLBACK_MODEL_CATALOG,
  );
  const [model, setModel] = useState(FALLBACK_MODEL_CATALOG.default_model);
  const [preset, setPreset] = useState(FALLBACK_MODEL_CATALOG.default_preset);
  const [temperature, setTemperature] = useState(INITIAL_TEMPERATURE);
  const [maxTokens, setMaxTokens] = useState(INITIAL_MAX_TOKENS);
  const [nodes, setNodes] = useState<TokenFlowNode[]>([]);
  const [edges, setEdges] = useState<ProbabilityFlowEdge[]>([]);
  const [generation, setGeneration] = useState<GenerationResponse | null>(null);
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [typedCompletion, setTypedCompletion] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [viewport, setViewport] = useState({
    x: 0,
    y: 0,
    zoom: 1,
  });
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [requestVariation, setRequestVariation] = useState(0);
  const flowRef = useRef<ReactFlowInstance<TokenFlowNode, ProbabilityFlowEdge> | null>(null);
  const nodesRef = useRef<TokenFlowNode[]>([]);
  const edgesRef = useRef<ProbabilityFlowEdge[]>([]);
  const animationRunRef = useRef(0);

  const models = modelCatalog.models;
  const presets = modelCatalog.presets;
  const selectedNode = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const generatedModelOption = findModelOption(models, generation?.request.model ?? model);
  const generatedPresetOption = findPresetOption(presets, generation?.request.preset ?? preset);

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    edgesRef.current = edges;
  }, [edges]);

  useEffect(() => {
    return () => {
      animationRunRef.current += 1;
    };
  }, []);

  function replaceGraph(nextNodes: TokenFlowNode[], nextEdges: ProbabilityFlowEdge[]) {
    const normalized = normalizeGraph(nextNodes, nextEdges);
    nodesRef.current = normalized.nodes;
    edgesRef.current = normalized.edges;
    setNodes(normalized.nodes);
    setEdges(normalized.edges);
  }

  function updateGraph(
    updater: (
      currentNodes: TokenFlowNode[],
      currentEdges: ProbabilityFlowEdge[],
    ) => {
      edges: ProbabilityFlowEdge[];
      nodes: TokenFlowNode[];
    },
  ) {
    const nextGraph = updater(nodesRef.current, edgesRef.current);
    replaceGraph(nextGraph.nodes, nextGraph.edges);
  }

  async function refreshHealth() {
    setIsCheckingHealth(true);

    try {
      const response = await fetch("/api/health", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(await parseApiError(response));
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
          throw new Error(await parseApiError(response));
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

  function setNodeStatus(nodeId: string, status: TokenNodeData["status"]) {
    updateGraph((currentNodes, currentEdges) => ({
      nodes: currentNodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                status,
              },
            }
          : node,
      ),
      edges: currentEdges,
    }));
  }

  function toggleCollapsed(nodeId: string, isCollapsed: boolean) {
    updateGraph((currentNodes, currentEdges) => ({
      nodes: currentNodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                isCollapsed,
              },
            }
          : node,
      ),
      edges: currentEdges,
    }));
  }

  function centerNode(nodeId: string) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!node || !flowRef.current) {
      return;
    }

    const position = getNodePosition(node);

    void flowRef.current.setCenter(position.x + 120, position.y + 44, {
      duration: 520,
      zoom: Math.max(viewport.zoom, 0.9),
    });
  }

  async function playMainPath(payload: GenerationResponse) {
    const runId = ++animationRunRef.current;
    const reasoning = reasoningBundleFromGeneration(payload);
    const rootNode = buildPromptNode({
      prompt: payload.prompt_used,
      model: payload.request.model,
      preset: payload.request.preset,
      reasoning,
      responseMode: payload.mode,
      status: "ready",
      temperature: payload.request.temperature,
      variation: payload.request.variation,
    });

    replaceGraph([rootNode], []);
    setTypedCompletion("");
    setSelectedNodeId("root");
    setIsReplaying(true);

    for (let index = 0; index < payload.tokens.length; index += 1) {
      if (animationRunRef.current !== runId) {
        return;
      }

      const trace = payload.tokens[index];
      const parentId = index === 0 ? "root" : payload.tokens[index - 1].id;
      const nextNode = buildTokenNode({
        id: trace.id,
        token: trace.token,
        tokenId: trace.token_id,
        probability: trace.probability,
        logProbability: trace.log_probability,
        entropy: trace.entropy,
        cumulativeProbability: trace.cumulative_probability,
        latency: trace.latency_ms,
        depth: trace.position + 1,
        rank: 1,
        parentId,
        position: {
          x: (index + 1) * HORIZONTAL_GAP,
          y: 0,
        },
        prompt: payload.prompt_used,
        model: payload.request.model,
        preset: payload.request.preset,
        temperature: payload.request.temperature,
        variation: payload.request.variation,
        responseMode: payload.mode,
        textPreview: trace.text_preview,
        isMainPath: true,
        status: "idle",
        reasoning,
        branchRationale: null,
        rawLogits: null,
      });

      replaceGraph(
        [...nodesRef.current, nextNode],
        [...edgesRef.current, buildEdge(parentId, nextNode.id, trace.probability, true)],
      );
      setTypedCompletion(
        joinTokenText(payload.tokens.slice(0, index + 1).map((token) => token.token)),
      );

      await wait(index < 4 ? 110 : index < 16 ? 72 : 40);
    }

    if (animationRunRef.current === runId) {
      setIsReplaying(false);
      window.setTimeout(() => {
        void flowRef.current?.fitView({
          duration: 640,
          padding: 0.18,
        });
      }, 40);
    }
  }

  async function expandNode(nodeId: string) {
    const node = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

    if (!node || node.data.status === "loading") {
      return;
    }

    setSelectedNodeId(nodeId);
    setContextMenu(null);

    if (node.data.isCollapsed) {
      toggleCollapsed(nodeId, false);
      return;
    }

    if (node.data.distributionRequested) {
      return;
    }

    setNodeStatus(nodeId, "loading");

    try {
      const response = await fetch("/api/expand-node", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: node.data.requestPrompt,
          model: node.data.requestModel,
          preset: node.data.requestPreset,
          temperature: node.data.requestTemperature,
          parent_node_id: nodeId,
          parent_token: node.data.tokenText,
          parent_text_preview: nodeId === "root" ? "" : node.data.textPreview,
          depth: node.data.depth,
          cumulative_probability: node.data.cumulativeProbability,
          variation: node.data.requestVariation,
          max_children: MAX_BRANCH_CHILDREN,
        }),
      });

      if (!response.ok) {
        throw new Error(await parseApiError(response));
      }

      const payload = (await response.json()) as NodeExpansionResponse;
      const parentNode = nodesRef.current.find((currentNode) => currentNode.id === nodeId);

      if (!parentNode) {
        return;
      }

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
      const nextNodes = [...nodesRef.current];
      const nextEdges = [...edgesRef.current];
      const animatedTargets: Array<{ id: string; x: number; y: number }> = [];
      const parentPosition = getNodePosition(parentNode);
      const targetX = parentPosition.x + HORIZONTAL_GAP;

      for (const candidate of payload.children) {
        const matchingChild = currentChildren.find(
          (child) =>
            child.data.depth === candidate.depth &&
            child.data.tokenText === candidate.token &&
            child.data.textPreview === candidate.text_preview,
        );

        const targetNodeId = matchingChild?.id ?? candidate.id;

        if (matchingChild) {
          const nextIndex = nextNodes.findIndex((item) => item.id === matchingChild.id);

          nextNodes[nextIndex] = {
            ...matchingChild,
            data: {
              ...matchingChild.data,
              probability: candidate.probability,
              logProbability: candidate.log_probability,
              entropy: candidate.entropy,
              latency: candidate.latency_ms,
              tokenId: candidate.token_id,
              tokenizerId: candidate.token_id,
              cumulativeProbability: candidate.cumulative_probability,
              rank: candidate.rank,
              isMainPath: matchingChild.data.isMainPath || candidate.rank === 1,
              responseMode: payload.mode,
              sourceNotes: payload.notes || matchingChild.data.sourceNotes,
              branchRationale: candidate.rationale ?? matchingChild.data.branchRationale,
            },
          };
        } else {
          const preferredY = parentPosition.y + branchOffset(candidate.rank) * VERTICAL_GAP;
          const targetY = findAvailableY(targetX, preferredY, nextNodes, new Set([nodeId]));
          const startPosition = {
            x: parentPosition.x + 42,
            y: parentPosition.y,
          };

          nextNodes.push(
            buildTokenNode({
              id: candidate.id,
              token: candidate.token,
              tokenId: candidate.token_id,
              probability: candidate.probability,
              logProbability: candidate.log_probability,
              entropy: candidate.entropy,
              cumulativeProbability: candidate.cumulative_probability,
              latency: candidate.latency_ms,
              depth: candidate.depth,
              rank: candidate.rank,
              parentId: nodeId,
              position: startPosition,
              prompt: node.data.requestPrompt,
              model: node.data.requestModel,
              preset: node.data.requestPreset,
              temperature: node.data.requestTemperature,
              variation: node.data.requestVariation,
              responseMode: payload.mode,
              textPreview: candidate.text_preview,
              isMainPath: candidate.rank === 1,
              status: "idle",
              reasoning,
              branchRationale: candidate.rationale ?? null,
              rawLogits: null,
            }),
          );
          animatedTargets.push({
            id: candidate.id,
            x: targetX,
            y: targetY,
          });
        }

        const nextEdge = buildEdge(
          nodeId,
          targetNodeId,
          candidate.probability,
          candidate.rank === 1,
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
        },
      };

      replaceGraph(nextNodes, nextEdges);

      if (animatedTargets.length > 0) {
        window.requestAnimationFrame(() => {
          updateGraph((currentNodes, currentEdges) => ({
            nodes: currentNodes.map((currentNode) => {
              const target = animatedTargets.find((item) => item.id === currentNode.id);

              if (!target || currentNode.data.status === "loading") {
                return currentNode;
              }

              return {
                ...currentNode,
                position: {
                  x: target.x,
                  y: target.y,
                },
              };
            }),
            edges: currentEdges,
          }));
        });
      }

      setBackendState("online");
    } catch (error) {
      setBackendState("offline");
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to expand the selected node.",
      );
    } finally {
      setNodeStatus(nodeId, "ready");
    }
  }

  async function handleSubmit() {
    const trimmedPrompt = prompt.trim();

    if (!trimmedPrompt) {
      setErrorMessage("Enter a prompt.");
      return;
    }

    animationRunRef.current += 1;
    const nextVariation = requestVariation + 1;
    const loadingRoot = buildPromptNode({
      prompt: trimmedPrompt,
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
      variation: nextVariation,
    });

    replaceGraph([loadingRoot], []);
    setSelectedNodeId("root");
    setTypedCompletion("");
    setErrorMessage(null);
    setIsGenerating(true);
    setRequestVariation(nextVariation);

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: trimmedPrompt,
          model,
          preset,
          temperature,
          max_tokens: maxTokens,
          variation: nextVariation,
        }),
      });

      if (!response.ok) {
        throw new Error(await parseApiError(response));
      }

      const payload = (await response.json()) as GenerationResponse;
      setGeneration(payload);
      setBackendState("online");
      await playMainPath(payload);
    } catch (error) {
      setBackendState("offline");
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to generate a response.",
      );
      replaceGraph(
        [
          buildPromptNode({
            prompt: trimmedPrompt,
            model,
            preset,
            reasoning: {
              notes: "",
              intent: "",
              strategy: "",
              focusTerms: [],
            },
            responseMode: "error",
            status: "idle",
            temperature,
            variation: nextVariation,
          }),
        ],
        [],
      );
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

  function handleResetView() {
    void flowRef.current?.fitView({
      duration: 520,
      padding: 0.18,
    });
  }

  function handleNodeContextMenu(event: React.MouseEvent, node: TokenFlowNode) {
    event.preventDefault();
    setSelectedNodeId(node.id);
    setContextMenu({
      nodeId: node.id,
      title: node.data.kind === "prompt" ? "Prompt" : node.data.tokenText,
      x: event.clientX,
      y: event.clientY,
      canExpand: node.data.isCollapsed || !node.data.distributionRequested,
      canCollapse: node.data.childCount > 0 && !node.data.isCollapsed,
    });
  }

  const handleNodeClick: NodeMouseHandler<TokenFlowNode> = (event, node) => {
    setContextMenu(null);
    setSelectedNodeId(node.id);

    if (event.shiftKey && node.data.childCount > 0) {
      event.preventDefault();
      toggleCollapsed(node.id, true);
    }
  };

  const handleNodeDoubleClick: NodeMouseHandler<TokenFlowNode> = (event, node) => {
    event.preventDefault();
    setContextMenu(null);
    setSelectedNodeId(node.id);
    void expandNode(node.id);
  };

  const currentStatus = isGenerating
    ? "Generating"
    : isReplaying
      ? "Replaying"
      : selectedNode
        ? "Exploring"
        : "Ready";
  const currentPreview =
    selectedNode?.data.textPreview ||
    typedCompletion ||
    generation?.completion ||
    "Double-click any token to reveal what the model almost chose next.";

  return (
    <div className="llmscope-app">
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

      <aside className="explorer-dock">
        <div className="explorer-dock__header">
          <div>
            <p className="explorer-dock__eyebrow">LLMScope</p>
            <h1 className="explorer-dock__title">Explore the tree</h1>
            <p className="explorer-dock__subtitle">
              Double-click a token to reveal alternate futures.
            </p>
          </div>

          <button
            className="icon-button"
            onClick={() => void refreshHealth()}
            type="button"
          >
            <RefreshCcw className={isCheckingHealth ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
          </button>
        </div>

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
            aria-label="Max tokens"
            className="explorer-input"
            max={4096}
            min={1}
            onChange={(event) => setMaxTokens(Number(event.target.value))}
            type="number"
            value={maxTokens}
          />
        </div>

        <div className="explorer-actions">
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

          <button
            className="explorer-button explorer-button--ghost"
            disabled={!generation || isGenerating}
            onClick={() => void handleReplay()}
            type="button"
          >
            <Play className="h-4 w-4" />
            Replay
          </button>

          <button className="explorer-button explorer-button--ghost" onClick={handleResetView} type="button">
            <Orbit className="h-4 w-4" />
            Fit
          </button>
        </div>

        <div className="explorer-chip-row">
          <span className="explorer-chip">{generatedModelOption.label}</span>
          <span className="explorer-chip">{generatedPresetOption.label}</span>
          <span className="explorer-chip">
            {backendState === "online"
              ? "Online"
              : backendState === "offline"
                ? "Offline"
                : "Checking"}
          </span>
        </div>

        <div className="path-card">
          <p className="path-card__label">Current path</p>
          <p className="path-card__text">{currentPreview}</p>
        </div>
      </aside>

      <aside className="inspector-panel">
        <div className="inspector-panel__header">
          <p className="inspector-panel__eyebrow">Inspector</p>
          <h2 className="inspector-panel__title">
            {selectedNode?.data.kind === "prompt"
              ? "Prompt"
              : selectedNode?.data.tokenText ?? "Select a node"}
          </h2>
        </div>

        {selectedNode ? (
          <div className="inspector-panel__content">
            <div className="inspector-section">
              <p className="inspector-section__label">Path</p>
              <div className="inspector-block">
                <p>{selectedNode.data.textPreview}</p>
              </div>
            </div>

            <div className="inspector-section">
              <p className="inspector-section__label">Metadata</p>
              <dl className="inspector-grid-data">
                <div>
                  <dt>Token</dt>
                  <dd>{selectedNode.data.tokenText}</dd>
                </div>
                <div>
                  <dt>Probability</dt>
                  <dd>{formatProbability(selectedNode.data.probability)}</dd>
                </div>
                <div>
                  <dt>Log probability</dt>
                  <dd>{formatNumber(selectedNode.data.logProbability)}</dd>
                </div>
                <div>
                  <dt>Entropy</dt>
                  <dd>{formatNumber(selectedNode.data.entropy)}</dd>
                </div>
                <div>
                  <dt>Latency</dt>
                  <dd>{`${selectedNode.data.latency} ms`}</dd>
                </div>
                <div>
                  <dt>Token id</dt>
                  <dd>{selectedNode.data.tokenId}</dd>
                </div>
                <div>
                  <dt>Tokenizer id</dt>
                  <dd>{selectedNode.data.tokenizerId}</dd>
                </div>
                <div>
                  <dt>Cumulative probability</dt>
                  <dd>{formatProbability(selectedNode.data.cumulativeProbability)}</dd>
                </div>
                <div>
                  <dt>Model</dt>
                  <dd>{selectedNode.data.requestModel}</dd>
                </div>
                <div>
                  <dt>Mode</dt>
                  <dd>{selectedNode.data.responseMode}</dd>
                </div>
              </dl>
            </div>

            <div className="inspector-section">
              <p className="inspector-section__label">Reasoning metadata</p>
              <div className="inspector-block inspector-block--stack">
                <p>
                  <strong>Intent</strong>
                  <span>{selectedNode.data.reasoningIntent || "Unavailable"}</span>
                </p>
                <p>
                  <strong>Strategy</strong>
                  <span>{selectedNode.data.reasoningStrategy || "Unavailable"}</span>
                </p>
                <p>
                  <strong>Focus terms</strong>
                  <span>
                    {selectedNode.data.reasoningFocusTerms.length > 0
                      ? selectedNode.data.reasoningFocusTerms.join(", ")
                      : "Unavailable"}
                  </span>
                </p>
                <p>
                  <strong>Branch rationale</strong>
                  <span>{selectedNode.data.branchRationale ?? "Unavailable"}</span>
                </p>
                <p>
                  <strong>Notes</strong>
                  <span>{selectedNode.data.sourceNotes || "Unavailable"}</span>
                </p>
              </div>
            </div>

            <div className="inspector-section">
              <p className="inspector-section__label">Raw logits</p>
              <div className="inspector-block">
                <p>
                  {selectedNode.data.rawLogits && selectedNode.data.rawLogits.length > 0
                    ? selectedNode.data.rawLogits.join(", ")
                    : "Not exposed by the current backend."}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="inspector-empty">
            Select a node, then inspect the branch here.
          </div>
        )}
      </aside>

      <div className="canvas-hint">
        click to select • double-click to expand • shift+click to collapse • space to pan • scroll to zoom
      </div>

      <ReactFlow<TokenFlowNode, ProbabilityFlowEdge>
        colorMode="dark"
        defaultEdgeOptions={{
          animated: true,
          type: "probabilityEdge",
        }}
        edgeTypes={edgeTypes}
        edges={edges}
        fitView
        fitViewOptions={{
          padding: 0.18,
        }}
        maxZoom={1.9}
        minZoom={0.16}
        nodeClickDistance={8}
        nodeTypes={nodeTypes}
        nodes={nodes}
        onEdgesChange={(changes: EdgeChange<ProbabilityFlowEdge>[]) => {
          replaceGraph(
            nodesRef.current,
            applyEdgeChanges<ProbabilityFlowEdge>(changes, edgesRef.current),
          );
        }}
        onInit={(instance) => {
          flowRef.current = instance;
        }}
        onMoveEnd={(_event, nextViewport) => {
          setViewport(nextViewport);
        }}
        onNodeClick={handleNodeClick}
        onNodeContextMenu={handleNodeContextMenu}
        onNodeDoubleClick={handleNodeDoubleClick}
        onNodeDragStop={(_event, node) => {
          updateGraph((currentNodes, currentEdges) => ({
            nodes: currentNodes.map((currentNode) =>
              currentNode.id === node.id
                ? {
                    ...currentNode,
                    position: node.position,
                  }
                : currentNode,
            ),
            edges: currentEdges,
          }));
        }}
        onNodesChange={(changes: NodeChange<TokenFlowNode>[]) => {
          replaceGraph(
            applyNodeChanges<TokenFlowNode>(changes, nodesRef.current),
            edgesRef.current,
          );
        }}
        onPaneClick={() => {
          setContextMenu(null);
        }}
        panActivationKeyCode="Space"
        panOnDrag={false}
        selectNodesOnDrag={false}
        selectionOnDrag={false}
        zoomOnDoubleClick={false}
      >
        <Background
          color="rgba(148, 163, 184, 0.15)"
          gap={26}
          size={1}
          variant={BackgroundVariant.Lines}
        />
        <MiniMap<TokenFlowNode> nodeColor={getMiniMapColor} pannable zoomable />
        <Controls showInteractive={false} />
      </ReactFlow>

      {contextMenu ? (
        <TokenContextMenu
          canCollapse={contextMenu.canCollapse}
          canExpand={contextMenu.canExpand}
          onCenter={() => {
            centerNode(contextMenu.nodeId);
            setContextMenu(null);
          }}
          onClose={() => {
            setContextMenu(null);
          }}
          onCollapse={() => {
            toggleCollapsed(contextMenu.nodeId, true);
            setContextMenu(null);
          }}
          onExpand={() => {
            void expandNode(contextMenu.nodeId);
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
