"use client";

import {
  ChevronLeft,
  Cpu,
  LoaderCircle,
  RefreshCcw,
  Sparkles,
  Wand2,
  X,
} from "lucide-react";

import type { ProviderSelectionId } from "@/lib/provider-selection";
import type {
  HuggingFaceLocalStatusResponse,
  ModelOption,
  PresetOption,
  ProviderCapabilitiesDetail,
  ProviderOption,
} from "@/types/api";

type BackendState = "checking" | "online" | "offline";
type TokenDisplayMode = "decoded" | "raw" | "token_id";

export interface GenerationPanelSystemPromptState {
  content: string | null;
  editable: boolean;
  helper: string;
  sourceLabel: string | null;
  title: string;
}

interface GenerationExamplePrompt {
  description: string;
  id: string;
  label: string;
  prompt: string;
}

interface GenerationPanelProps {
  backendState: BackendState;
  canGenerate: boolean;
  collapsed: boolean;
  demoMode: boolean;
  filteredModels: ModelOption[];
  huggingFaceLocalStatus: HuggingFaceLocalStatusResponse | null;
  isCheckingHealth: boolean;
  isGenerating: boolean;
  isLoadingHuggingFaceStatus: boolean;
  isLoadingModels: boolean;
  isSubmittingHuggingFaceAction: boolean;
  maxTokens: number;
  model: string;
  onApplyExample: (prompt: string) => void;
  onClearPrompt: () => void;
  onGenerate: () => void;
  onLoadSelectedHuggingFaceModel: () => void;
  onMaxTokensChange: (value: number) => void;
  onModelChange: (model: string) => void;
  onPresetChange: (preset: string) => void;
  onPromptChange: (prompt: string) => void;
  onProviderChange: (provider: ProviderSelectionId) => void;
  onRefreshHealth: () => void;
  onSetDemoMode: (demoMode: boolean) => void;
  onTemperatureChange: (value: number) => void;
  onToggleCollapsed: () => void;
  onTopPChange: (value: number) => void;
  onTokenDisplayModeChange: (mode: TokenDisplayMode) => void;
  onUnloadHuggingFaceModel: () => void;
  preset: string;
  presets: PresetOption[];
  prompt: string;
  providerStatusMessage: string | null;
  providerRecommendations: string[];
  providers: ProviderOption[];
  selectedCapabilities: ProviderCapabilitiesDetail;
  selectedHuggingFaceModelStatus:
    | HuggingFaceLocalStatusResponse["models"][number]
    | null;
  selectedProvider: ProviderSelectionId;
  showHuggingFaceControls: boolean;
  systemPromptState: GenerationPanelSystemPromptState;
  temperature: number;
  tokenDisplayMode: TokenDisplayMode;
  topP: number;
}

const EXAMPLE_PROMPTS: GenerationExamplePrompt[] = [
  {
    id: "key-retrieval",
    label: "Key retrieval",
    description: "Inspect exact retrieval and attention around short answers.",
    prompt:
      "Red key: APPLE\nBlue key: OCEAN\nGreen key: FOREST\n\nReturn only the value of the blue key.",
  },
  {
    id: "name-binding",
    label: "Name binding",
    description: "Test whether the model keeps entity bindings straight.",
    prompt:
      "Ava owns the violin.\nNoah owns the telescope.\nLiam owns the bicycle.\n\nWhat does Noah own? Answer with one word.",
  },
  {
    id: "long-distance",
    label: "Long-distance recall",
    description: "A useful prompt for later-token recall and continuation checks.",
    prompt:
      "First note: copper.\nSecond note: glacier.\nThird note: lantern.\nFourth note: apricot.\n\nReply with only the third note.",
  },
  {
    id: "translation",
    label: "Translation",
    description: "A compact multilingual prompt with structured output.",
    prompt:
      "Translate this into French and keep the tone formal: The report is ready for review.",
  },
  {
    id: "creative",
    label: "Creative generation",
    description: "Good for branch exploration and lower-confidence continuations.",
    prompt: "Write a short opening line for a sci-fi story set in a flooded city.",
  },
];

function formatVram(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return `${value.toFixed(1)} GB`;
}

function buildCapabilityBadges(capabilities: ProviderCapabilitiesDetail, isLocal: boolean) {
  return [
    capabilities.supports_logprobs ? "Exact probabilities" : null,
    capabilities.supports_attention ? "Attention supported" : null,
    capabilities.supports_exact_continuation ? "Exact continuation" : null,
    isLocal ? "Runs locally" : null,
    capabilities.supports_streaming ? "Streaming" : null,
  ].filter((badge): badge is string => Boolean(badge));
}

export function GenerationPanel({
  backendState,
  canGenerate,
  collapsed,
  demoMode,
  filteredModels,
  huggingFaceLocalStatus,
  isCheckingHealth,
  isGenerating,
  isLoadingHuggingFaceStatus,
  isLoadingModels,
  isSubmittingHuggingFaceAction,
  maxTokens,
  model,
  onApplyExample,
  onClearPrompt,
  onGenerate,
  onLoadSelectedHuggingFaceModel,
  onMaxTokensChange,
  onModelChange,
  onPresetChange,
  onPromptChange,
  onProviderChange,
  onRefreshHealth,
  onSetDemoMode,
  onTemperatureChange,
  onToggleCollapsed,
  onTopPChange,
  onTokenDisplayModeChange,
  onUnloadHuggingFaceModel,
  preset,
  presets,
  prompt,
  providerStatusMessage,
  providerRecommendations,
  providers,
  selectedCapabilities,
  selectedHuggingFaceModelStatus,
  selectedProvider,
  showHuggingFaceControls,
  systemPromptState,
  temperature,
  tokenDisplayMode,
  topP,
}: GenerationPanelProps) {
  const promptCharacterCount = prompt.length;
  const promptTokenEstimate = Math.max(1, Math.ceil(promptCharacterCount / 4));
  const capabilityBadges = buildCapabilityBadges(
    selectedCapabilities,
    selectedProvider === "hugging_face" || selectedProvider === "ollama",
  );
  const selectedModelLabel =
    filteredModels.find((option) => option.id === model)?.label ?? model;
  const activeModelLoaded =
    huggingFaceLocalStatus?.active_model_id === model &&
    huggingFaceLocalStatus?.status === "ready";

  return (
    <aside className={`generation-panel${collapsed ? " generation-panel--collapsed" : ""}`}>
      <div className="generation-panel__header">
        <div>
          <p className="generation-panel__eyebrow">LLMScope</p>
          {!collapsed ? (
            <>
              <h1 className="generation-panel__title">Explore how a model chooses each token</h1>
              <p className="generation-panel__subtitle">
                Prompt, inspect, branch, continue, and compare without losing the underlying context.
              </p>
            </>
          ) : null}
        </div>
        <div className="generation-panel__header-actions">
          <button className="icon-button" onClick={onRefreshHealth} type="button">
            <RefreshCcw className={isCheckingHealth ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
          </button>
          <button className="icon-button" onClick={onToggleCollapsed} type="button">
            <ChevronLeft className={`h-4 w-4${collapsed ? " rotate-180" : ""}`} />
          </button>
        </div>
      </div>

      {collapsed ? (
        <div className="generation-panel__collapsed-actions">
          <button
            className="explorer-button explorer-button--primary"
            onClick={onToggleCollapsed}
            type="button"
          >
            Open
          </button>
          <button
            className="explorer-button explorer-button--ghost"
            disabled={!canGenerate || isGenerating}
            onClick={onGenerate}
            type="button"
          >
            Generate
          </button>
        </div>
      ) : (
        <>
          <div className="generation-panel__body">
          <section className="generation-panel__section">
            <div className="generation-panel__section-heading">
              <p className="generation-panel__section-step">1. Prompt</p>
              <div className="generation-panel__inline-actions">
                <button
                  className="generation-panel__mini-action"
                  onClick={onClearPrompt}
                  type="button"
                >
                  <X className="h-3.5 w-3.5" />
                  Clear
                </button>
                <details className="generation-panel__examples">
                  <summary className="generation-panel__mini-action">
                    <Wand2 className="h-3.5 w-3.5" />
                    Try an example
                  </summary>
                  <div className="generation-panel__examples-menu">
                    {EXAMPLE_PROMPTS.map((example) => (
                      <button
                        key={example.id}
                        className="generation-panel__example"
                        onClick={() => onApplyExample(example.prompt)}
                        type="button"
                      >
                        <span>{example.label}</span>
                        <small>{example.description}</small>
                      </button>
                    ))}
                  </div>
                </details>
              </div>
            </div>
            <p className="generation-panel__section-helper">
              Ask something, provide a fact to retrieve, or test how the model follows instructions.
            </p>
            <textarea
              aria-label="Prompt"
              className="generation-panel__textarea"
              onChange={(event) => onPromptChange(event.target.value)}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !isGenerating) {
                  event.preventDefault();
                  onGenerate();
                }
              }}
              placeholder="Ask something, provide a fact to retrieve, or test how the model follows instructions..."
              value={prompt}
            />
            <div className="generation-panel__prompt-meta">
              <span>{`${promptCharacterCount} characters`}</span>
              <span>{`~${promptTokenEstimate} tokens`}</span>
              <span>Ctrl+Enter to generate</span>
            </div>
          </section>

          <details className="generation-panel__section" open={Boolean(systemPromptState.content)}>
            <summary className="generation-panel__section-heading generation-panel__section-heading--summary">
              <div>
                <p className="generation-panel__section-step">2. System instructions</p>
                <p className="generation-panel__section-helper">{systemPromptState.title}</p>
              </div>
              <span className="generation-panel__section-pill">
                {systemPromptState.editable ? "Editable" : "Read only"}
              </span>
            </summary>
            <div className="generation-panel__system-card">
              {systemPromptState.sourceLabel ? (
                <span className="generation-panel__system-source">
                  {systemPromptState.sourceLabel}
                </span>
              ) : null}
              <textarea
                aria-label="System instructions"
                className="generation-panel__system-textarea"
                readOnly
                value={systemPromptState.content ?? systemPromptState.helper}
              />
              <p className="generation-panel__system-helper">{systemPromptState.helper}</p>
            </div>
          </details>

          <section className="generation-panel__section">
            <div className="generation-panel__section-heading">
              <p className="generation-panel__section-step">3. Model</p>
            </div>
            <div className="generation-panel__field-grid">
              <label className="generation-panel__field">
                <span>Provider</span>
                <select
                  className="explorer-input"
                  disabled={isLoadingModels}
                  onChange={(event) =>
                    onProviderChange(event.target.value as ProviderSelectionId)
                  }
                  value={selectedProvider}
                >
                  {providers.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="generation-panel__field">
                <span>Model</span>
                <select
                  className="explorer-input"
                  disabled={isLoadingModels || filteredModels.length === 0}
                  onChange={(event) => onModelChange(event.target.value)}
                  value={filteredModels.some((option) => option.id === model) ? model : ""}
                >
                  {filteredModels.length === 0 ? (
                    <option value="">No models available</option>
                  ) : null}
                  {filteredModels.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="generation-panel__field">
                <span>Preset</span>
                <select
                  className="explorer-input"
                  disabled={isLoadingModels}
                  onChange={(event) => onPresetChange(event.target.value)}
                  value={preset}
                >
                  {presets.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="generation-panel__badge-row">
              {capabilityBadges.map((badge) => (
                <span key={badge} className="generation-panel__capability">
                  {badge}
                </span>
              ))}
            </div>
            {providerStatusMessage ? (
              <div className="generation-panel__status-note">
                <p>{providerStatusMessage}</p>
                {providerRecommendations.length > 0 ? (
                  <p>{`Recommended models: ${providerRecommendations.join(", ")}`}</p>
                ) : null}
              </div>
            ) : null}
          </section>

          <section className="generation-panel__section">
            <div className="generation-panel__section-heading">
              <p className="generation-panel__section-step">4. Generation settings</p>
            </div>
            <label className="generation-panel__slider-field">
              <span>Temperature</span>
              <div className="generation-panel__slider-row">
                <input
                  max={1.5}
                  min={0}
                  onChange={(event) => onTemperatureChange(Number(event.target.value))}
                  step={0.05}
                  type="range"
                  value={temperature}
                />
                <input
                  className="explorer-input"
                  max={1.5}
                  min={0}
                  onChange={(event) => onTemperatureChange(Number(event.target.value))}
                  step={0.05}
                  type="number"
                  value={temperature}
                />
              </div>
              <small>0.0 Precise {"->"} 1.5 Creative</small>
            </label>
            <div className="generation-panel__field-grid generation-panel__field-grid--compact">
              <label className="generation-panel__field">
                <span>Maximum output tokens</span>
                <input
                  className="explorer-input"
                  max={4096}
                  min={1}
                  onChange={(event) => onMaxTokensChange(Number(event.target.value))}
                  type="number"
                  value={maxTokens}
                />
              </label>
              <label className="generation-panel__field">
                <span>Top-p</span>
                <input
                  className="explorer-input"
                  max={1}
                  min={0}
                  onChange={(event) => onTopPChange(Number(event.target.value))}
                  step={0.05}
                  type="number"
                  value={topP}
                />
              </label>
            </div>
            <details className="generation-panel__advanced">
              <summary>Advanced</summary>
              <div className="generation-panel__field-grid generation-panel__field-grid--compact">
                <label className="generation-panel__field">
                  <span>Token display</span>
                  <select
                    className="explorer-input"
                    onChange={(event) =>
                      onTokenDisplayModeChange(event.target.value as TokenDisplayMode)
                    }
                    value={tokenDisplayMode}
                  >
                    <option value="decoded">Decoded tokens</option>
                    <option value="raw">Raw tokens</option>
                    <option value="token_id">Token IDs</option>
                  </select>
                </label>
                <label className="generation-panel__field generation-panel__field--toggle">
                  <span>Demo data</span>
                  <button
                    className={`generation-panel__toggle${
                      demoMode ? " generation-panel__toggle--active" : ""
                    }`}
                    onClick={() => onSetDemoMode(!demoMode)}
                    type="button"
                  >
                    {demoMode ? "Enabled" : "Disabled"}
                  </button>
                </label>
              </div>
            </details>
          </section>

          {showHuggingFaceControls ? (
            <section className="generation-panel__section">
              <div className="generation-panel__section-heading">
                <p className="generation-panel__section-step">5. Local model status</p>
                <span className="generation-panel__section-pill">
                  {selectedHuggingFaceModelStatus?.status ??
                    huggingFaceLocalStatus?.status ??
                    "checking"}
                </span>
              </div>
              <div className="generation-panel__status-card">
                <div className="generation-panel__status-grid">
                  <div>
                    <dt>Model</dt>
                    <dd>{selectedModelLabel}</dd>
                  </div>
                  <div>
                    <dt>Device</dt>
                    <dd>{huggingFaceLocalStatus?.device ?? "Unavailable"}</dd>
                  </div>
                  <div>
                    <dt>Precision</dt>
                    <dd>{huggingFaceLocalStatus?.precision ?? "Unavailable"}</dd>
                  </div>
                  <div>
                    <dt>VRAM free</dt>
                    <dd>{formatVram(huggingFaceLocalStatus?.gpu_free_vram_gb)}</dd>
                  </div>
                </div>
                {selectedHuggingFaceModelStatus?.status_message ? (
                  <p className="generation-panel__status-copy">
                    {selectedHuggingFaceModelStatus.status_message}
                  </p>
                ) : null}
                <div className="generation-panel__status-actions">
                  <button
                    className="explorer-button explorer-button--primary"
                    disabled={
                      isLoadingModels ||
                      isLoadingHuggingFaceStatus ||
                      isSubmittingHuggingFaceAction ||
                      activeModelLoaded
                    }
                    onClick={onLoadSelectedHuggingFaceModel}
                    type="button"
                  >
                    {isSubmittingHuggingFaceAction && !activeModelLoaded ? (
                      <>
                        <LoaderCircle className="h-4 w-4 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      <>
                        <Cpu className="h-4 w-4" />
                        Load model
                      </>
                    )}
                  </button>
                  <button
                    className="explorer-button explorer-button--ghost"
                    disabled={
                      isLoadingModels ||
                      isLoadingHuggingFaceStatus ||
                      isSubmittingHuggingFaceAction ||
                      !huggingFaceLocalStatus?.active_model_id
                    }
                    onClick={onUnloadHuggingFaceModel}
                    type="button"
                  >
                    Unload model
                  </button>
                </div>
              </div>
            </section>
          ) : null}
        </div>

          <section className="generation-panel__section generation-panel__section--generate">
            <div className="generation-panel__section-heading">
              <p className="generation-panel__section-step">
                {showHuggingFaceControls ? "6. Generate" : "5. Generate"}
              </p>
            </div>
            <div className="generation-panel__status-row">
              <span
                className={`generation-panel__status-pill generation-panel__status-pill--${backendState}`}
              >
                {backendState === "online"
                  ? "Backend connected"
                  : backendState === "offline"
                    ? "Backend unavailable"
                    : "Checking backend"}
              </span>
              <span className="generation-panel__status-pill">
                {canGenerate ? "Provider ready" : "Provider not ready"}
              </span>
            </div>
            <button
              className="generation-panel__generate"
              disabled={isGenerating || isLoadingModels || !canGenerate}
              onClick={onGenerate}
              type="button"
            >
              {isGenerating ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Generate
                </>
              )}
            </button>
            <p className="generation-panel__section-helper">
              The prompt remains intact while you inspect, branch, replay, and compare the resulting graph.
            </p>
          </section>
        </>
      )}
    </aside>
  );
}
