"use client";

import {
  Activity,
  Blocks,
  Bot,
  Check,
  Clock3,
  Coins,
  Copy,
  GitBranch,
  LoaderCircle,
  RefreshCcw,
  Search,
  SendHorizontal,
  Server,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";

import { TokenBranchTree } from "@/components/generation/token-branch-tree";
import { Panel } from "@/components/ui/panel";
import { useDashboardState } from "@/hooks/use-dashboard-state";
import { cn } from "@/lib/utils";
import type { ModelOption } from "@/types/api";

function formatPercent(value: number) {
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  }).format(value);
}

function getAlternativeLabel(rationale?: string | null) {
  const normalized = rationale?.toLowerCase() ?? "";

  if (
    normalized.includes("prompt's focus") ||
    normalized.includes("branch centered") ||
    normalized.includes("choice around")
  ) {
    return "Prompt focus";
  }

  if (
    normalized.includes("prompt wording") ||
    normalized.includes("original phrasing") ||
    normalized.includes("wording on")
  ) {
    return "Prompt wording";
  }

  if (
    normalized.includes("answer anchored") ||
    normalized.includes("answer's emphasis") ||
    normalized.includes("wording direction")
  ) {
    return "Answer context";
  }

  if (normalized.includes("sentence")) {
    return "Structure";
  }

  return "Variant";
}

function getAlternativeBadgeClasses(label: string) {
  if (label === "Prompt focus") {
    return "bg-sky-100 text-sky-700";
  }

  if (label === "Prompt wording") {
    return "bg-indigo-100 text-indigo-700";
  }

  if (label === "Answer context") {
    return "bg-emerald-100 text-emerald-700";
  }

  return "bg-slate-100 text-slate-700";
}

function getBackendBadgeClasses(state: "checking" | "online" | "offline") {
  if (state === "online") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }

  if (state === "offline") {
    return "border-rose-200 bg-rose-50 text-rose-700";
  }

  return "border-amber-200 bg-amber-50 text-amber-700";
}

function groupModels(models: ModelOption[]) {
  return models.reduce<Record<string, ModelOption[]>>((groups, model) => {
    groups[model.group] = [...(groups[model.group] ?? []), model];
    return groups;
  }, {});
}

function EmptyState({
  actionLabel,
  onAction,
  text,
}: {
  actionLabel: string;
  onAction: () => void;
  text: string;
}) {
  return (
    <div className="flex min-h-[420px] flex-col items-center justify-center gap-4 rounded-3xl border border-dashed border-slate-300 bg-slate-50 px-6 text-center">
      <p className="max-w-md text-sm text-slate-500">{text}</p>
      <button
        className="inline-flex h-11 items-center justify-center rounded-full bg-slate-950 px-5 text-sm font-semibold text-white hover:bg-slate-800"
        onClick={onAction}
        type="button"
      >
        {actionLabel}
      </button>
    </div>
  );
}

export function AppShell() {
  const state = useDashboardState();
  const groupedModels = groupModels(state.models);
  const tokenQuery = state.tokenSearchQuery.trim().toLowerCase();
  const filteredTokens =
    state.generation?.tokens.filter((token) => {
      if (!tokenQuery) {
        return true;
      }

      return (
        token.token.toLowerCase().includes(tokenQuery) ||
        token.text_preview.toLowerCase().includes(tokenQuery)
      );
    }) ?? [];
  const selectedTreeChildren = state.selectedTreeNode?.children ?? [];
  const isBusy = state.isLoading || state.isTypingResponse;
  const isRegenerating = state.isLoading && Boolean(state.generation);
  const submitLabel = state.isLoading
    ? isRegenerating
      ? "Regenerating..."
      : "Generating..."
    : state.isTypingResponse
      ? "Generating..."
      : state.generation
        ? "Regenerate"
        : "Generate";
  const generatedModelLabel =
    state.generatedModelOption?.label ?? state.generation?.request.model ?? state.currentModelOption.label;
  const generatedPresetLabel =
    state.generatedPresetOption?.label ?? state.generation?.request.preset ?? state.currentPresetOption.label;
  const completionText = state.generation
    ? state.isTypingResponse
      ? state.displayedCompletion
      : state.displayedCompletion || state.generation.completion
    : "";

  return (
    <main className="min-h-screen">
      <div className="mx-auto flex w-full max-w-[96vw] flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <h1 className="font-display text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl">
            LLMScope
          </h1>

          <div className="flex flex-wrap items-center gap-3">
            <span
              className={cn(
                "inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm font-medium",
                getBackendBadgeClasses(state.backendState),
              )}
            >
              <Server className="h-4 w-4" />
              {state.backendState === "online"
                ? "Online"
                : state.backendState === "offline"
                  ? "Offline"
                  : "Checking"}
            </span>

            <button
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              onClick={() => void state.refreshHealth()}
              type="button"
            >
              <RefreshCcw
                className={cn("h-4 w-4", state.isCheckingHealth ? "animate-spin" : "")}
              />
              Refresh
            </button>
          </div>
        </header>

        <div className="flex flex-wrap gap-3">
          {[
            { id: "create", label: "Create", icon: SlidersHorizontal },
            { id: "answer", label: "Answer", icon: Sparkles },
            { id: "tokens", label: "Tokens", icon: Blocks },
            { id: "tree", label: "Tree", icon: GitBranch },
          ].map((tab) => {
            const Icon = tab.icon;

            return (
              <button
                key={tab.id}
                className={cn(
                  "inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium",
                  state.activeTab === tab.id
                    ? "border-slate-950 bg-slate-950 text-white"
                    : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50",
                )}
                onClick={() => state.setActiveTab(tab.id as typeof state.activeTab)}
                type="button"
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {state.errorMessage ? (
          <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm font-medium text-rose-700">
            {state.errorMessage}
          </div>
        ) : null}

        {state.activeTab === "create" ? (
          <Panel title="Create">
            <form
              className="space-y-5"
              onSubmit={(event) => {
                event.preventDefault();
                void state.handleSubmit();
              }}
            >
              <textarea
                className="min-h-[260px] w-full rounded-3xl border border-slate-200 bg-slate-50 px-4 py-4 text-sm leading-6 text-slate-900 outline-none focus:border-sky-300 focus:bg-white focus:ring-4 focus:ring-sky-100"
                onChange={(event) => state.updatePrompt(event.target.value)}
                placeholder="Ask something"
                value={state.prompt}
              />

              <div className="grid gap-4 xl:grid-cols-[minmax(220px,1fr)_minmax(180px,0.8fr)_minmax(180px,0.8fr)_minmax(140px,0.55fr)_auto]">
                <label className="block space-y-2">
                  <span className="text-sm font-medium text-slate-700">Model</span>
                  <select
                    className="h-12 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 text-sm text-slate-900 outline-none focus:border-sky-300 focus:bg-white focus:ring-4 focus:ring-sky-100"
                    disabled={state.isLoadingModels}
                    onChange={(event) => state.updateModel(event.target.value)}
                    value={state.model}
                  >
                    {Object.entries(groupedModels).map(([group, options]) => (
                      <optgroup key={group} label={group}>
                        {options.map((option) => (
                          <option key={option.id} value={option.id}>
                            {option.label}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                  </select>
                </label>

                <label className="block space-y-2">
                  <span className="text-sm font-medium text-slate-700">Mode</span>
                  <select
                    className="h-12 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 text-sm text-slate-900 outline-none focus:border-sky-300 focus:bg-white focus:ring-4 focus:ring-sky-100"
                    disabled={state.isLoadingModels}
                    onChange={(event) => state.updatePreset(event.target.value)}
                    value={state.preset}
                  >
                    {state.presets.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="block space-y-2">
                  <span className="flex items-center justify-between text-sm font-medium text-slate-700">
                    <span>Temperature</span>
                    <span>{state.temperature.toFixed(1)}</span>
                  </span>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-4">
                    <input
                      className="w-full accent-sky-600"
                      max={2}
                      min={0}
                      onChange={(event) => state.updateTemperature(Number(event.target.value))}
                      step={0.1}
                      type="range"
                      value={state.temperature}
                    />
                  </div>
                </label>

                <label className="block space-y-2">
                  <span className="text-sm font-medium text-slate-700">Max</span>
                  <input
                    className="h-12 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 text-sm text-slate-900 outline-none focus:border-sky-300 focus:bg-white focus:ring-4 focus:ring-sky-100"
                    max={4096}
                    min={1}
                    onChange={(event) => state.updateMaxTokens(Number(event.target.value))}
                    type="number"
                    value={state.maxTokens}
                  />
                </label>

                <button
                  className="inline-flex h-12 items-center justify-center gap-2 rounded-full bg-slate-950 px-5 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                  disabled={isBusy || state.isLoadingModels}
                  type="submit"
                >
                  {isBusy ? (
                    <>
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                      {submitLabel}
                    </>
                  ) : state.isLoadingModels ? (
                    <>
                      <LoaderCircle className="h-4 w-4 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <SendHorizontal className="h-4 w-4" />
                      {submitLabel}
                    </>
                  )}
                </button>
              </div>

              {state.generation ? (
                <div className="flex flex-wrap gap-3">
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
                    {generatedModelLabel}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
                    {generatedPresetLabel}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
                    {state.generation.mode === "live" ? "Live" : "Fallback"}
                  </span>
                </div>
              ) : null}
            </form>
          </Panel>
        ) : null}

        {state.activeTab === "answer" ? (
          <Panel
            action={
              state.generation ? (
                <button
                  className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                  onClick={() => void state.copyCompletion()}
                  type="button"
                >
                  {state.copyState === "copied" ? (
                    <>
                      <Check className="h-4 w-4 text-emerald-600" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4" />
                      Copy
                    </>
                  )}
                </button>
              ) : null
            }
            title="Answer"
          >
            {state.isLoading ? (
              <div className="flex min-h-[420px] flex-col items-center justify-center gap-3 rounded-3xl border border-slate-200 bg-slate-50 text-center">
                <LoaderCircle className="h-8 w-8 animate-spin text-sky-600" />
                <p className="text-base font-semibold text-slate-900">{submitLabel}</p>
                <p className="max-w-2xl text-sm text-slate-500">{state.pendingPrompt}</p>
              </div>
            ) : state.generation ? (
              <div className="space-y-4">
                <div className="flex flex-wrap gap-3">
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
                    {generatedModelLabel}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
                    {generatedPresetLabel}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
                    {state.generation.mode === "live" ? "Live" : "Fallback"}
                  </span>
                  {state.isTypingResponse ? (
                    <span className="rounded-full border border-sky-200 bg-sky-50 px-3 py-2 text-sm font-medium text-sky-700">
                      Typing live
                    </span>
                  ) : null}
                </div>

                <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
                  <p className="text-sm leading-7 text-slate-700">{state.generation.prompt_used}</p>
                </div>

                <div className="rounded-3xl border border-slate-200 bg-white p-5">
                  <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                    <Bot className="h-4 w-4 text-sky-600" />
                    Output
                  </div>
                  <p className="mt-4 whitespace-pre-wrap text-lg leading-8 text-slate-900">
                    {completionText}
                    {state.isTypingResponse ? (
                      <span className="ml-1 inline-block h-6 w-0.5 animate-pulse rounded-full bg-sky-500 align-[-0.2em]" />
                    ) : null}
                  </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                      <Activity className="h-4 w-4 text-sky-600" />
                      Tokens
                    </div>
                    <p className="mt-3 text-2xl font-semibold text-slate-950">
                      {state.generation.stats.total_tokens}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                      <Clock3 className="h-4 w-4 text-sky-600" />
                      Latency
                    </div>
                    <p className="mt-3 text-2xl font-semibold text-slate-950">
                      {state.generation.stats.latency_ms} ms
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                      <Coins className="h-4 w-4 text-sky-600" />
                      Cost
                    </div>
                    <p className="mt-3 text-2xl font-semibold text-slate-950">
                      {formatCurrency(state.generation.stats.estimated_cost_usd)}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-white p-4">
                    <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                      <Server className="h-4 w-4 text-sky-600" />
                      Provider
                    </div>
                    <p className="mt-3 text-2xl font-semibold capitalize text-slate-950">
                      {state.generation.stats.provider.replace("_", " ")}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <EmptyState
                actionLabel="Create"
                onAction={() => state.setActiveTab("create")}
                text="Run a prompt to see the answer here."
              />
            )}
          </Panel>
        ) : null}

        {state.activeTab === "tokens" ? (
          <Panel title="Tokens">
            {state.isLoading ? (
              <div className="flex min-h-[520px] flex-col items-center justify-center gap-3 rounded-3xl border border-slate-200 bg-slate-50 text-center">
                <LoaderCircle className="h-8 w-8 animate-spin text-sky-600" />
                <p className="text-base font-semibold text-slate-900">{submitLabel}</p>
                <p className="max-w-2xl text-sm text-slate-500">{state.pendingPrompt}</p>
              </div>
            ) : state.generation ? (
              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_320px]">
                <div className="space-y-4">
                  <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
                    <Search className="h-4 w-4 text-slate-400" />
                    <input
                      className="w-full bg-transparent text-sm text-slate-900 outline-none placeholder:text-slate-400"
                      onChange={(event) => state.setTokenSearchQuery(event.target.value)}
                      placeholder="Search token"
                      value={state.tokenSearchQuery}
                    />
                  </div>

                  <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white">
                    <div className="grid grid-cols-[minmax(0,1fr)_90px_90px_90px] gap-3 border-b border-slate-200 px-4 py-3 text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
                      <span>Token</span>
                      <span>Local</span>
                      <span>Path</span>
                      <span>Ms</span>
                    </div>
                    <div className="divide-y divide-slate-200">
                      {filteredTokens.map((token) => (
                        <button
                          key={token.id}
                          className={cn(
                            "grid w-full grid-cols-[minmax(0,1fr)_90px_90px_90px] gap-3 px-4 py-3 text-left text-sm",
                            state.selectedToken?.id === token.id
                              ? "bg-sky-50"
                              : "bg-white hover:bg-slate-50",
                          )}
                          onClick={() => state.handleTokenSelection(token.position)}
                          type="button"
                        >
                          <div className="min-w-0">
                            <span className="truncate font-medium text-slate-900">
                              {token.token}
                            </span>
                            <p className="mt-1 truncate text-xs text-slate-500">
                              {token.text_preview}
                            </p>
                          </div>
                          <span className="text-slate-600">
                            {formatPercent(token.probability)}
                          </span>
                          <span className="text-slate-600">
                            {formatPercent(token.cumulative_probability)}
                          </span>
                          <span className="text-slate-600">{token.latency_ms}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  {state.selectedToken ? (
                    <>
                      <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
                        <p className="text-3xl font-semibold text-slate-950">
                          {state.selectedToken.token}
                        </p>
                        <p className="mt-3 text-sm leading-6 text-slate-600">
                          {state.selectedToken.text_preview}
                        </p>
                      </div>

                      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                        <div className="rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-xs text-slate-500">Local</p>
                          <p className="mt-2 text-lg font-semibold text-slate-950">
                            {formatPercent(state.selectedToken.probability)}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-xs text-slate-500">Path</p>
                          <p className="mt-2 text-lg font-semibold text-slate-950">
                            {formatPercent(state.selectedToken.cumulative_probability)}
                          </p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        {state.selectedToken.alternatives.map((candidate) => (
                          <div
                            key={`${state.selectedToken?.id ?? "selected"}-${candidate.token}`}
                            className="rounded-2xl border border-slate-200 bg-white p-4"
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="text-base font-semibold text-slate-900">
                                {candidate.token}
                              </span>
                              <span className="text-sm font-medium text-slate-500">
                                {formatPercent(candidate.probability)}
                              </span>
                            </div>

                            <div className="mt-3 flex flex-wrap items-center gap-2">
                              <span
                                className={cn(
                                  "rounded-full px-2.5 py-1 text-xs font-medium",
                                  getAlternativeBadgeClasses(
                                    getAlternativeLabel(candidate.rationale),
                                  ),
                                )}
                              >
                                {getAlternativeLabel(candidate.rationale)}
                              </span>
                            </div>

                            {candidate.rationale ? (
                              <p className="mt-3 text-sm leading-6 text-slate-500">
                                {candidate.rationale}
                              </p>
                            ) : null}
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 px-4 py-6 text-sm text-slate-500">
                      Select a token.
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <EmptyState
                actionLabel="Create"
                onAction={() => state.setActiveTab("create")}
                text="Run a prompt to inspect token paths."
              />
            )}
          </Panel>
        ) : null}

        {state.activeTab === "tree" ? (
          <Panel
            action={
              state.generation ? (
                <div className="flex flex-wrap gap-2">
                  <button
                    className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                    onClick={() => state.expandAllTreeNodes()}
                    type="button"
                  >
                    Expand
                  </button>
                  <button
                    className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                    onClick={() => state.collapseTreeToSelectedPath()}
                    type="button"
                  >
                    Main
                  </button>
                </div>
              ) : null
            }
            title="Tree"
          >
            {state.isLoading ? (
              <div className="flex min-h-[720px] flex-col items-center justify-center gap-3 rounded-3xl border border-slate-200 bg-slate-50 text-center">
                <LoaderCircle className="h-8 w-8 animate-spin text-sky-600" />
                <p className="text-base font-semibold text-slate-900">{submitLabel}</p>
                <p className="max-w-2xl text-sm text-slate-500">{state.pendingPrompt}</p>
              </div>
            ) : state.generation ? (
              <div className="space-y-5">
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex min-w-[260px] flex-1 items-center gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2">
                    <Search className="h-4 w-4 text-slate-400" />
                    <input
                      className="w-full bg-transparent text-sm text-slate-900 outline-none placeholder:text-slate-400"
                      onChange={(event) => state.setTreeSearchQuery(event.target.value)}
                      placeholder="Search tree"
                      value={state.treeSearchQuery}
                    />
                  </div>
                  <span className="rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700">
                    {state.generation.tree_summary.total_nodes} nodes
                  </span>
                  <span className="rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700">
                    depth {state.generation.tree_summary.max_depth}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700">
                    width {state.generation.tree_summary.branch_width}
                  </span>
                </div>

                <TokenBranchTree
                  expandedNodeIds={state.expandedTreeNodeIds}
                  onSelectNode={state.handleTreeNodeSelection}
                  onToggleNode={state.toggleTreeNode}
                  root={state.generation.tree}
                  searchQuery={state.treeSearchQuery}
                  selectedNodeId={state.selectedTreeNode?.id ?? null}
                />

                {state.selectedTreeNode ? (
                  <div className="space-y-4 rounded-[2rem] border border-slate-200 bg-slate-50 p-4">
                    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_360px]">
                      <div className="rounded-3xl border border-slate-200 bg-white p-5">
                        <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                          <GitBranch className="h-4 w-4 text-sky-600" />
                          Selected
                        </div>
                        <p className="mt-3 text-4xl font-semibold text-slate-950">
                          {state.selectedTreeNode.token}
                        </p>
                        <p className="mt-4 text-base leading-8 text-slate-700">
                          {state.selectedTreeNode.text_preview}
                        </p>
                      </div>

                      <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
                        <div className="rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-xs text-slate-500">Local</p>
                          <p className="mt-2 text-xl font-semibold text-slate-950">
                            {formatPercent(state.selectedTreeNode.probability)}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-xs text-slate-500">Path</p>
                          <p className="mt-2 text-xl font-semibold text-slate-950">
                            {formatPercent(state.selectedTreeNode.cumulative_probability)}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-slate-200 bg-white p-4">
                          <p className="text-xs text-slate-500">Latency</p>
                          <p className="mt-2 text-xl font-semibold text-slate-950">
                            {state.selectedTreeNode.latency_ms} ms
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <h4 className="text-sm font-semibold uppercase tracking-[0.08em] text-slate-500">
                          Next Branches
                        </h4>
                        <span className="rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700">
                          {selectedTreeChildren.length}
                        </span>
                      </div>

                      {selectedTreeChildren.length > 0 ? (
                        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                          {selectedTreeChildren.map((child) => (
                            <button
                              key={child.id}
                              className="rounded-2xl border border-slate-200 bg-white p-4 text-left hover:border-slate-300 hover:bg-slate-50"
                              onClick={() => state.handleTreeNodeSelection(child.id)}
                              type="button"
                            >
                              <div className="flex items-center justify-between gap-3">
                                <span className="text-base font-semibold text-slate-900">
                                  {child.token}
                                </span>
                                <span className="text-sm text-slate-500">
                                  {formatPercent(child.probability)}
                                </span>
                              </div>
                              <p className="mt-3 text-sm leading-6 text-slate-600">
                                {child.text_preview}
                              </p>
                            </button>
                          ))}
                        </div>
                      ) : (
                        <div className="rounded-2xl border border-slate-200 bg-white px-4 py-4 text-sm text-slate-500">
                          Leaf
                        </div>
                      )}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : (
              <EmptyState
                actionLabel="Create"
                onAction={() => state.setActiveTab("create")}
                text="Run a prompt to open the branch tree."
              />
            )}
          </Panel>
        ) : null}
      </div>
    </main>
  );
}
