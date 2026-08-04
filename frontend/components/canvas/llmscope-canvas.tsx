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
import { Bot, LoaderCircle, Orbit, Play, RefreshCcw, ScanSearch, Sparkles } from "lucide-react";

import { ProbabilityEdge } from "@/components/canvas/probability-edge";
import { TokenContextMenu } from "@/components/canvas/token-context-menu";
import { TokenNode } from "@/components/canvas/token-node";
import type {
  CanvasTab,
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
const HORIZONTAL_GAP = 340;
const VERTICAL_GAP = 190;
const X_COLLISION_THRESHOLD = 240;
const Y_COLLISION_THRESHOLD = 150;
const MAX_BRANCH_CHILDREN = 5;

type BackendState = "checking" | "online" | "offline";

interface ContextMenuState {
  canCollapse: boolean;
  canExpand: boolean;
  nodeId: string;
  title: string;
  x: number;
  y: number;
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

function formatCompact(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`;
  }

  return `${value}`;
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

function buildPromptNode({
  prompt,
  model,
  preset,
  responseMode,
  status,
  temperature,
  variation,
}: {
  model: string;
  preset: string;
  prompt: string;
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
      displayText: prompt,
      probability: 1,
      logProbability: 0,
      entropy: 0,
      latency: 0,
      tokenId: 0,
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
    },
    draggable: true,
  };
}

function buildTokenNode({
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
  responseMode,
  status,
  temperature,
  textPreview,
  token,
  tokenId,
  variation,
}: {
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
      displayText: token,
      probability,
      logProbability,
      entropy,
      latency,
      tokenId,
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
  const [activeTab, setActiveTab] = useState<CanvasTab>("prompt");
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
  const generatedModelOption = generation
    ? findModelOption(models, generation.request.model)
    : findModelOption(models, model);
  const generatedPresetOption = generation
    ? findPresetOption(presets, generation.request.preset)
    : findPresetOption(presets, preset);

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

    void flowRef.current.setCenter(position.x + 120, position.y + 64, {
      duration: 520,
      zoom: Math.max(viewport.zoom, 0.9),
    });
  }

  async function playMainPath(payload: GenerationResponse) {
    const runId = ++animationRunRef.current;
    const rootNode = buildPromptNode({
      prompt: payload.prompt_used,
      model: payload.request.model,
      preset: payload.request.preset,
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
      });

      replaceGraph(
        [...nodesRef.current, nextNode],
        [...edgesRef.current, buildEdge(parentId, nextNode.id, trace.probability, true)],
      );
      setTypedCompletion(
        joinTokenText(payload.tokens.slice(0, index + 1).map((token) => token.token)),
      );

      await wait(index < 4 ? 110 : index < 16 ? 70 : 38);
    }

    if (animationRunRef.current === runId) {
      setIsReplaying(false);
      window.setTimeout(() => {
        void flowRef.current?.fitView({
          duration: 650,
          padding: 0.16,
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
              cumulativeProbability: candidate.cumulative_probability,
              rank: candidate.rank,
              isMainPath: matchingChild.data.isMainPath || candidate.rank === 1,
              responseMode: payload.mode,
            },
          };
        } else {
          const preferredY = parentPosition.y + branchOffset(candidate.rank) * VERTICAL_GAP;
          const targetY = findAvailableY(targetX, preferredY, nextNodes, new Set([nodeId]));
          const startPosition = {
            x: parentPosition.x + 56,
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
    setActiveTab("prompt");

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

    setActiveTab("replay");
    await playMainPath(generation);
  }

  function handleResetView() {
    void flowRef.current?.fitView({
      duration: 520,
      padding: 0.16,
    });
  }

  function handleNodeContextMenu(event: React.MouseEvent, node: TokenFlowNode) {
    event.preventDefault();
    setSelectedNodeId(node.id);
    setActiveTab("graph");
    setContextMenu({
      nodeId: node.id,
      title: node.data.kind === "prompt" ? "Prompt" : node.data.tokenText,
      x: event.clientX,
      y: event.clientY,
      canExpand: node.data.isCollapsed || !node.data.distributionRequested,
      canCollapse: node.data.childCount > 0 && !node.data.isCollapsed,
    });
  }

  const handleNodeClick: NodeMouseHandler<TokenFlowNode> = (_event, node) => {
    setSelectedNodeId(node.id);
    setActiveTab("graph");

    if (node.data.isCollapsed || !node.data.distributionRequested) {
      void expandNode(node.id);
    }
  };

  const currentStatus = isGenerating
    ? "Generating"
    : isReplaying
      ? "Replaying"
      : selectedNode
        ? "Inspecting"
        : "Ready";

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

      <aside className="floating-panel floating-panel--left">
        <div className="floating-panel__header">
          <div>
            <p className="floating-panel__eyebrow">LLMScope</p>
            <h1 className="floating-panel__title">Canvas</h1>
          </div>

          <button
            className="floating-icon-button"
            onClick={() => void refreshHealth()}
            type="button"
          >
            <RefreshCcw className={isCheckingHealth ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
          </button>
        </div>

        <div className="dock-tabs">
          {[
            { id: "prompt", label: "Prompt", icon: Sparkles },
            { id: "replay", label: "Replay", icon: Play },
            { id: "graph", label: "Graph", icon: ScanSearch },
          ].map((tab) => {
            const Icon = tab.icon;

            return (
              <button
                key={tab.id}
                className={`dock-tab${activeTab === tab.id ? " dock-tab--active" : ""}`}
                onClick={() => setActiveTab(tab.id as CanvasTab)}
                type="button"
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {activeTab === "prompt" ? (
          <div className="floating-panel__content">
            <textarea
              aria-label="Prompt"
              className="floating-textarea"
              onChange={(event) => {
                setPrompt(event.target.value);
                setErrorMessage(null);
              }}
              placeholder="Ask something"
              value={prompt}
            />

            <div className="floating-grid">
              <select
                aria-label="Model"
                className="floating-input"
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
                className="floating-input"
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

            <div className="floating-grid floating-grid--tight">
              <label className="mini-field">
                <span>Temp</span>
                <input
                  className="floating-input"
                  max={2}
                  min={0}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                  step={0.1}
                  type="number"
                  value={temperature}
                />
              </label>

              <label className="mini-field">
                <span>Max</span>
                <input
                  className="floating-input"
                  max={4096}
                  min={1}
                  onChange={(event) => setMaxTokens(Number(event.target.value))}
                  type="number"
                  value={maxTokens}
                />
              </label>
            </div>

            <button
              className="floating-submit"
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

            <div className="inline-stats">
              <span className="inline-stats__chip">{generatedModelOption.label}</span>
              <span className="inline-stats__chip">{generatedPresetOption.label}</span>
              <span className="inline-stats__chip">
                {backendState === "online" ? "Online" : backendState === "offline" ? "Offline" : "Checking"}
              </span>
            </div>

            <div className="response-preview">
              <div className="response-preview__label">
                <Bot className="h-4 w-4" />
                Live
              </div>
              <p className="response-preview__text">
                {typedCompletion || " "}
                {isGenerating || isReplaying ? (
                  <span className="response-preview__caret" />
                ) : null}
              </p>
            </div>
          </div>
        ) : null}

        {activeTab === "replay" ? (
          <div className="floating-panel__content">
            <button
              className="floating-submit floating-submit--secondary"
              disabled={!generation || isGenerating}
              onClick={() => void handleReplay()}
              type="button"
            >
              <Play className="h-4 w-4" />
              Replay path
            </button>

            <button
              className="floating-submit floating-submit--ghost"
              onClick={handleResetView}
              type="button"
            >
              <Orbit className="h-4 w-4" />
              Fit view
            </button>

            <div className="metric-stack">
              <div className="metric-card">
                <span>Tokens</span>
                <strong>{generation?.tokens.length ?? 0}</strong>
              </div>
              <div className="metric-card">
                <span>Latency</span>
                <strong>{generation ? `${generation.stats.latency_ms}ms` : "-"}</strong>
              </div>
              <div className="metric-card">
                <span>Mode</span>
                <strong>{generation?.mode ?? "-"}</strong>
              </div>
            </div>
          </div>
        ) : null}

        {activeTab === "graph" ? (
          <div className="floating-panel__content">
            <div className="metric-stack">
              <div className="metric-card">
                <span>Nodes</span>
                <strong>{formatCompact(nodes.filter((node) => !node.hidden).length)}</strong>
              </div>
              <div className="metric-card">
                <span>Edges</span>
                <strong>{formatCompact(edges.filter((edge) => !edge.hidden).length)}</strong>
              </div>
              <div className="metric-card">
                <span>Zoom</span>
                <strong>{`${Math.round(viewport.zoom * 100)}%`}</strong>
              </div>
            </div>

            <div className="legend-bar">
              <span className="legend-bar__item legend-bar__item--high">High</span>
              <span className="legend-bar__item legend-bar__item--mid">Mid</span>
              <span className="legend-bar__item legend-bar__item--low">Low</span>
            </div>

            <p className="floating-note">
              Click to expand. Right-click for branch actions. Drag nodes to tune layout.
            </p>
          </div>
        ) : null}
      </aside>

      <aside className="floating-panel floating-panel--right">
        <div className="floating-panel__header">
          <div>
            <p className="floating-panel__eyebrow">Inspect</p>
            <h2 className="floating-panel__title floating-panel__title--small">
              {selectedNode?.data.kind === "prompt"
                ? "Prompt"
                : selectedNode?.data.tokenText ?? "Node"}
            </h2>
          </div>
        </div>

        <div className="floating-panel__content">
          {selectedNode ? (
            <>
              <div className="inspect-card">
                <p className="inspect-card__text">{selectedNode.data.textPreview}</p>
              </div>

              <div className="metric-stack metric-stack--two">
                <div className="metric-card">
                  <span>p</span>
                  <strong>{formatPercent(selectedNode.data.probability)}</strong>
                </div>
                <div className="metric-card">
                  <span>logP</span>
                  <strong>{selectedNode.data.logProbability.toFixed(3)}</strong>
                </div>
                <div className="metric-card">
                  <span>H</span>
                  <strong>{selectedNode.data.entropy.toFixed(3)}</strong>
                </div>
                <div className="metric-card">
                  <span>ms</span>
                  <strong>{selectedNode.data.latency}</strong>
                </div>
                <div className="metric-card">
                  <span>id</span>
                  <strong>{selectedNode.data.tokenId}</strong>
                </div>
                <div className="metric-card">
                  <span>path</span>
                  <strong>{formatPercent(selectedNode.data.cumulativeProbability)}</strong>
                </div>
              </div>
            </>
          ) : (
            <div className="inspect-card">
              <p className="inspect-card__text">Select a node.</p>
            </div>
          )}

          {generation ? (
            <div className="session-strip">
              <span>{generation.stats.provider}</span>
              <span>{generation.stats.model}</span>
              <span>{generation.stats.total_tokens}</span>
            </div>
          ) : null}
        </div>
      </aside>

      <ReactFlow<TokenFlowNode, ProbabilityFlowEdge>
        colorMode="dark"
        defaultEdgeOptions={{
          animated: true,
          type: "probabilityEdge",
        }}
        edgeTypes={edgeTypes}
        edges={edges}
        fitView
        maxZoom={1.8}
        minZoom={0.18}
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
        panOnDrag
        selectionOnDrag={false}
      >
        <Background color="rgba(148, 163, 184, 0.18)" gap={28} size={1.1} variant={BackgroundVariant.Lines} />
        <MiniMap<TokenFlowNode>
          nodeColor={(node) => {
            if (node.data.kind === "prompt") {
              return "#c084fc";
            }

            if (node.data.probability >= 0.72) {
              return "#38bdf8";
            }

            if (node.data.probability >= 0.45) {
              return "#f59e0b";
            }

            return "#fb7185";
          }}
          pannable
          zoomable
        />
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
