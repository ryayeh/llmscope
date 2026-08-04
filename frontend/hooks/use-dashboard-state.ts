"use client";

import { useCallback, useEffect, useState } from "react";

import type {
  GenerationResponse,
  HealthResponse,
  ModelCatalogResponse,
  ModelOption,
  PresetOption,
  TokenTreeNode,
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

type BackendState = "checking" | "online" | "offline";
type CopyState = "idle" | "copied" | "error";
type ActiveTab = "create" | "answer" | "tokens" | "tree";

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
    // Fall back to a generic message if the response isn't JSON.
  }

  return `Request failed with status ${response.status}.`;
}

function findTreeNodeById(node: TokenTreeNode, nodeId: string): TokenTreeNode | null {
  if (node.id === nodeId) {
    return node;
  }

  for (const child of node.children) {
    const match = findTreeNodeById(child, nodeId);
    if (match) {
      return match;
    }
  }

  return null;
}

function collectTreeNodeIds(node: TokenTreeNode): string[] {
  return [node.id, ...node.children.flatMap((child) => collectTreeNodeIds(child))];
}

function collectSelectedPathNodeIds(node: TokenTreeNode): string[] {
  const ids = [node.id];
  const selectedChild = node.children.find((child) => child.is_selected_path);

  if (!selectedChild) {
    return ids;
  }

  return [...ids, ...collectSelectedPathNodeIds(selectedChild)];
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

export function useDashboardState() {
  const [prompt, setPrompt] = useState(INITIAL_PROMPT);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogResponse>(
    FALLBACK_MODEL_CATALOG,
  );
  const [model, setModel] = useState(FALLBACK_MODEL_CATALOG.default_model);
  const [preset, setPreset] = useState(FALLBACK_MODEL_CATALOG.default_preset);
  const [temperature, setTemperature] = useState(INITIAL_TEMPERATURE);
  const [maxTokens, setMaxTokens] = useState(INITIAL_MAX_TOKENS);
  const [generation, setGeneration] = useState<GenerationResponse | null>(null);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState(0);
  const [selectedTreeNodeId, setSelectedTreeNodeId] = useState<string | null>(null);
  const [expandedTreeNodeIds, setExpandedTreeNodeIds] = useState<string[]>([]);
  const [tokenSearchQuery, setTokenSearchQuery] = useState("");
  const [treeSearchQuery, setTreeSearchQuery] = useState("");
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const [activeTab, setActiveTab] = useState<ActiveTab>("create");
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [requestVariation, setRequestVariation] = useState(0);
  const [displayedCompletion, setDisplayedCompletion] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [isTypingResponse, setIsTypingResponse] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const selectedToken = generation?.tokens[selectedTokenIndex] ?? null;
  const selectedTreeNode =
    generation && selectedTreeNodeId
      ? findTreeNodeById(generation.tree, selectedTreeNodeId)
      : null;
  const models = modelCatalog.models;
  const presets = modelCatalog.presets;
  const currentModelOption = findModelOption(models, model);
  const currentPresetOption = findPresetOption(presets, preset);
  const generatedModelOption = generation
    ? findModelOption(models, generation.request.model)
    : null;
  const generatedPresetOption = generation
    ? findPresetOption(presets, generation.request.preset)
    : null;

  const refreshHealth = useCallback(async () => {
    setIsCheckingHealth(true);

    try {
      const response = await fetch("/api/health", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(await parseApiError(response));
      }

      const payload = (await response.json()) as HealthResponse;
      setHealth(payload);
      setBackendState("online");
    } catch {
      setBackendState("offline");
    } finally {
      setIsCheckingHealth(false);
    }
  }, []);

  const refreshModels = useCallback(async () => {
    setIsLoadingModels(true);

    try {
      const response = await fetch("/api/models", {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(await parseApiError(response));
      }

      const payload = (await response.json()) as ModelCatalogResponse;
      setModelCatalog(payload);

      if (!payload.models.some((item) => item.id === model)) {
        setModel(payload.default_model);
      }

      if (!payload.presets.some((item) => item.id === preset)) {
        setPreset(payload.default_preset);
      }
    } catch {
      setModelCatalog(FALLBACK_MODEL_CATALOG);
    } finally {
      setIsLoadingModels(false);
    }
  }, [model, preset]);

  const updatePrompt = (value: string) => {
    setPrompt(value);
    setErrorMessage(null);
  };

  const updateModel = (value: string) => {
    setModel(value);
    setErrorMessage(null);
  };

  const updatePreset = (value: string) => {
    setPreset(value);
    setErrorMessage(null);
  };

  const updateTemperature = (value: number) => {
    setTemperature(value);
    setErrorMessage(null);
  };

  const updateMaxTokens = (value: number) => {
    setMaxTokens(value);
    setErrorMessage(null);
  };

  const handleSubmit = async () => {
    const trimmedPrompt = prompt.trim();

    if (!trimmedPrompt) {
      setErrorMessage("Enter a prompt.");
      return;
    }

    const nextVariation = requestVariation + 1;

    setRequestVariation(nextVariation);
    setIsLoading(true);
    setIsTypingResponse(false);
    setDisplayedCompletion("");
    setPendingPrompt(trimmedPrompt);
    setErrorMessage(null);
    setActiveTab("answer");

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
      const selectedPathNodeIds = collectSelectedPathNodeIds(payload.tree);

      setGeneration(payload);
      setSelectedTokenIndex(0);
      setSelectedTreeNodeId(selectedPathNodeIds[1] ?? payload.tree.id);
      setExpandedTreeNodeIds(selectedPathNodeIds);
      setTokenSearchQuery("");
      setTreeSearchQuery("");
      setBackendState("online");
    } catch (error) {
      setBackendState("offline");
      setErrorMessage(
        error instanceof Error ? error.message : "Unable to generate a response.",
      );
    } finally {
      setPendingPrompt(null);
      setIsLoading(false);
    }
  };

  const handleTokenSelection = (index: number) => {
    setSelectedTokenIndex(index);

    if (!generation) {
      return;
    }

    const selectedPathNodeIds = collectSelectedPathNodeIds(generation.tree);
    const matchingNodeId = selectedPathNodeIds[index + 1];

    if (matchingNodeId) {
      setSelectedTreeNodeId(matchingNodeId);
    }
  };

  const handleTreeNodeSelection = (nodeId: string) => {
    if (!generation) {
      return;
    }

    const node = findTreeNodeById(generation.tree, nodeId);

    if (!node) {
      return;
    }

    setSelectedTreeNodeId(nodeId);

    if (node.depth > 0) {
      setSelectedTokenIndex(Math.min(node.depth - 1, generation.tokens.length - 1));
    }
  };

  const toggleTreeNode = (nodeId: string) => {
    setExpandedTreeNodeIds((current) =>
      current.includes(nodeId)
        ? current.filter((id) => id !== nodeId)
        : [...current, nodeId],
    );
  };

  const expandAllTreeNodes = () => {
    if (!generation) {
      return;
    }

    setExpandedTreeNodeIds(collectTreeNodeIds(generation.tree));
  };

  const collapseTreeToSelectedPath = () => {
    if (!generation) {
      return;
    }

    setExpandedTreeNodeIds(collectSelectedPathNodeIds(generation.tree));
  };

  const copyCompletion = async () => {
    if (!generation) {
      return;
    }

    try {
      await navigator.clipboard.writeText(generation.completion);
      setCopyState("copied");

      window.setTimeout(() => {
        setCopyState("idle");
      }, 1800);
    } catch {
      setCopyState("error");
    }
  };

  useEffect(() => {
    void refreshHealth();
    void refreshModels();
  }, [refreshHealth, refreshModels]);

  useEffect(() => {
    if (!generation || !treeSearchQuery.trim()) {
      return;
    }

    setExpandedTreeNodeIds(collectTreeNodeIds(generation.tree));
  }, [generation, treeSearchQuery]);

  useEffect(() => {
    if (!generation) {
      setDisplayedCompletion("");
      setIsTypingResponse(false);
      return;
    }

    const responseTokens = generation.tokens.map((token) => token.token);

    if (responseTokens.length === 0) {
      setDisplayedCompletion(generation.completion);
      setIsTypingResponse(false);
      return;
    }

    let nextIndex = 0;
    let timeoutId: number | null = null;

    setDisplayedCompletion("");
    setIsTypingResponse(true);

    const revealNextToken = () => {
      nextIndex += 1;
      setDisplayedCompletion(joinTokenText(responseTokens.slice(0, nextIndex)));

      if (nextIndex >= responseTokens.length) {
        setIsTypingResponse(false);
        return;
      }

      const delay = nextIndex < 10 ? 48 : nextIndex < 28 ? 32 : 22;
      timeoutId = window.setTimeout(revealNextToken, delay);
    };

    timeoutId = window.setTimeout(revealNextToken, 80);

    return () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [generation]);

  return {
    activeTab,
    backendState,
    collapseTreeToSelectedPath,
    copyCompletion,
    copyState,
    currentModelOption,
    currentPresetOption,
    displayedCompletion,
    errorMessage,
    expandAllTreeNodes,
    expandedTreeNodeIds,
    generatedModelOption,
    generatedPresetOption,
    generation,
    handleSubmit,
    handleTokenSelection,
    handleTreeNodeSelection,
    health,
    isCheckingHealth,
    isLoading,
    isLoadingModels,
    isTypingResponse,
    maxTokens,
    model,
    models,
    pendingPrompt,
    preset,
    presets,
    prompt,
    refreshHealth,
    selectedToken,
    selectedTreeNode,
    setActiveTab,
    setTokenSearchQuery,
    setTreeSearchQuery,
    temperature,
    tokenSearchQuery,
    toggleTreeNode,
    treeSearchQuery,
    updateMaxTokens,
    updateModel,
    updatePreset,
    updatePrompt,
    updateTemperature,
  };
}
