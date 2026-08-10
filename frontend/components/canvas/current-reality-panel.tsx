"use client";

import {
  memo,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

import {
  Bot,
  Braces,
  Check,
  Copy,
  Settings2,
  UserRound,
} from "lucide-react";

import type { CanonicalTokenSourceCategory } from "@/types/api";
import type { ProbabilityViewMode } from "@/components/canvas/types";

type CurrentRealityTab = "conversation" | "tokens" | "raw" | "statistics" | "markdown";

export interface CurrentRealityTokenItem {
  decodedContribution: string;
  displayProbability: number;
  displayToken: string;
  id: string;
  isChanged: boolean;
  rawProbability: number;
  rawToken: string;
  rank: number;
  step: number;
  supportsLogprobs: boolean;
}

export interface CurrentRealityAttentionTokenItem {
  attentionWeight: number | null;
  decodedContribution: string;
  displayToken: string;
  fullPosition: number;
  graphTokenId: string | null;
  id: string;
  isPinned: boolean;
  isQuery: boolean;
  rawToken: string;
  sequenceScope: "prompt" | "generated";
  tokenId: number;
}

export interface CurrentRealityConversationSection {
  id: string;
  label: string;
  role: "system" | "user" | "assistant";
  text: string;
  tokenIds: string[];
}

export interface CurrentRealityGroupedTokenItem {
  canonicalPosition: number | null;
  decodedContribution: string;
  displayProbability: number | null;
  displayToken: string;
  graphTokenId: string | null;
  id: string;
  kind: "assistant" | "prompt";
  rank: number | null;
  rawProbability: number | null;
  rawToken: string;
  sourceCategory: CanonicalTokenSourceCategory | "generated_output";
  sourceLabel: string;
  specialToken: boolean;
  step: number | null;
  supportsLogprobs: boolean;
  tokenId: number | null;
}

export interface CurrentRealityTokenGroup {
  category: CanonicalTokenSourceCategory | "generated_output";
  id: string;
  label: string;
  tokens: CurrentRealityGroupedTokenItem[];
}

export interface CurrentRealitySummaryItem {
  label: string;
  tone?: "accent" | "approximate" | "exact" | "muted";
  value: string;
}

export interface CurrentRealityFormattingSelection {
  description: string;
  label: string;
  token: string;
}

export interface CurrentRealityStats {
  branchDepth: number;
  displayProbability: number | null;
  entropy: number | null;
  latency: number;
  rawProbability: number | null;
  supportsEntropy: boolean;
  supportsLogprobs: boolean;
  tokenCount: number;
}

interface CurrentRealityPanelProps {
  attentionEnabled?: boolean;
  attentionHint?: string | null;
  attentionTokens?: CurrentRealityAttentionTokenItem[];
  branchBreadcrumb?: string | null;
  collapsed: boolean;
  continuationModeLabel: string;
  continuationModeTitle: string | null;
  continuationModeTone: "exact" | "approximate";
  conversationSections?: CurrentRealityConversationSection[];
  copyConversationText?: string;
  copyRawContextText?: string;
  copyTokenIdsText?: string;
  copyUserPromptText?: string;
  detailItems?: CurrentRealitySummaryItem[];
  formattingSelection?: CurrentRealityFormattingSelection | null;
  hasContent: boolean;
  probabilityMode: ProbabilityViewMode;
  promptTokenGroups?: CurrentRealityTokenGroup[];
  rawContextText?: string;
  remainingProbabilityMass: number;
  selectedTokenId: string | null;
  summaryItems?: CurrentRealitySummaryItem[];
  stats: CurrentRealityStats;
  supportsEntropy: boolean;
  supportsLogprobs: boolean;
  summary: string;
  text: string;
  tokens: CurrentRealityTokenItem[];
  topKCoverage: number;
  onToggleAttentionPin?: (tokenId: string) => void;
  onSelectToken: (tokenId: string) => void;
  onToggleCollapse: () => void;
}

interface InlineMarkdownMatch {
  content: string;
  href?: string;
  type: "bold" | "code" | "italic" | "link";
}

type MarkdownBlock =
  | { content: string; type: "paragraph" }
  | { content: string; level: number; type: "heading" }
  | { content: string[]; type: "quote" }
  | { content: string[]; language: string; type: "code" }
  | { items: string[]; ordered: boolean; type: "list" };

const TAB_LABELS: Record<CurrentRealityTab, string> = {
  conversation: "Conversation",
  markdown: "Markdown",
  raw: "Raw context",
  statistics: "Statistics",
  tokens: "Tokens",
};

function formatPercent(value: number) {
  return `${(Math.max(0, value) * 100).toFixed(1)}%`;
}

function formatMetricPercent(value: number | null, available: boolean) {
  if (!available || typeof value !== "number") {
    return "Unavailable";
  }

  return formatPercent(value);
}

function formatNumber(value: number | null, available = true) {
  if (!available || typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return value.toFixed(4);
}

function formatLatency(value: number) {
  return `${Math.round(value)} ms`;
}

function getProbabilityModeLabel(mode: ProbabilityViewMode) {
  return mode === "normalized" ? "Normalized Top-K" : "Raw";
}

export function normalizeTokenChipLabel(token: CurrentRealityTokenItem) {
  const candidate = token.displayToken || token.decodedContribution || token.rawToken;

  if (candidate.trim()) {
    return candidate;
  }

  return candidate
    .replace(/ /g, "␠")
    .replace(/\n/g, "↵\n")
    .replace(/\t/g, "⇥");
}

function formatGroupedTokenChipLabel(token: CurrentRealityGroupedTokenItem) {
  const candidate = token.displayToken || token.decodedContribution || token.rawToken;

  if (candidate.trim()) {
    return candidate;
  }

  return candidate.replace(/ /g, "\u2420").replace(/\n/g, "\u21B5\n").replace(/\t/g, "\u21E5");
}

function getConversationRoleIcon(role: CurrentRealityConversationSection["role"]) {
  switch (role) {
    case "system":
      return Settings2;
    case "user":
      return UserRound;
    default:
      return Bot;
  }
}

function getSummaryToneClass(item: CurrentRealitySummaryItem) {
  if (item.tone === "exact") {
    return " sentence-bar__badge--exact";
  }

  if (item.tone === "approximate") {
    return " sentence-bar__badge--approximate";
  }

  if (item.tone === "accent") {
    return " sentence-bar__badge--accent";
  }

  if (item.tone === "muted") {
    return " sentence-bar__badge--muted";
  }

  return "";
}

function renderInlineMarkdown(text: string) {
  const segments: ReactNode[] = [];
  const pattern =
    /(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null = pattern.exec(text);

  while (match) {
    if (match.index > lastIndex) {
      segments.push(text.slice(lastIndex, match.index));
    }

    const value = match[0];
    const inlineMatch: InlineMarkdownMatch | null = match[2]
      ? { content: match[2], href: match[3], type: "link" }
      : match[4]
        ? { content: match[4], type: "code" }
        : match[5]
          ? { content: match[5], type: "bold" }
          : match[6]
            ? { content: match[6], type: "italic" }
            : null;

    if (!inlineMatch) {
      segments.push(value);
    } else if (inlineMatch.type === "link" && inlineMatch.href) {
      segments.push(
        <a
          key={`${inlineMatch.type}-${match.index}`}
          className="markdown-preview__link"
          href={inlineMatch.href}
          rel="noreferrer"
          target="_blank"
        >
          {inlineMatch.content}
        </a>,
      );
    } else if (inlineMatch.type === "code") {
      segments.push(
        <code key={`${inlineMatch.type}-${match.index}`} className="markdown-preview__inline-code">
          {inlineMatch.content}
        </code>,
      );
    } else if (inlineMatch.type === "bold") {
      segments.push(
        <strong key={`${inlineMatch.type}-${match.index}`}>{inlineMatch.content}</strong>,
      );
    } else {
      segments.push(<em key={`${inlineMatch.type}-${match.index}`}>{inlineMatch.content}</em>);
    }

    lastIndex = match.index + value.length;
    match = pattern.exec(text);
  }

  if (lastIndex < text.length) {
    segments.push(text.slice(lastIndex));
  }

  return segments;
}

function highlightCodeLine(line: string) {
  const segments: ReactNode[] = [];
  const pattern =
    /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`|\/\/.*$|#.*$|\b\d+(?:\.\d+)?\b|\b(?:async|await|break|case|catch|class|const|continue|def|default|else|export|extends|false|finally|for|from|function|if|import|in|interface|let|new|null|pass|return|switch|throw|true|try|type|undefined|var|while)\b)/gm;
  let lastIndex = 0;
  let match: RegExpExecArray | null = pattern.exec(line);

  while (match) {
    if (match.index > lastIndex) {
      segments.push(line.slice(lastIndex, match.index));
    }

    const token = match[0];
    const className = /^["'`]/.test(token)
      ? "markdown-code__token markdown-code__token--string"
      : /^(\/\/|#)/.test(token)
        ? "markdown-code__token markdown-code__token--comment"
        : /^\d/.test(token)
          ? "markdown-code__token markdown-code__token--number"
          : "markdown-code__token markdown-code__token--keyword";

    segments.push(
      <span key={`${token}-${match.index}`} className={className}>
        {token}
      </span>,
    );

    lastIndex = match.index + token.length;
    match = pattern.exec(line);
  }

  if (lastIndex < line.length) {
    segments.push(line.slice(lastIndex));
  }

  return segments;
}

function parseMarkdownBlocks(markdown: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = [];
  const lines = markdown.split(/\r?\n/);
  let paragraph: string[] = [];
  let quote: string[] = [];
  let listItems: string[] = [];
  let listOrdered = false;
  let codeLanguage = "";
  let codeLines: string[] = [];

  function flushParagraph() {
    if (paragraph.length > 0) {
      blocks.push({ content: paragraph.join(" "), type: "paragraph" });
      paragraph = [];
    }
  }

  function flushQuote() {
    if (quote.length > 0) {
      blocks.push({ content: [...quote], type: "quote" });
      quote = [];
    }
  }

  function flushList() {
    if (listItems.length > 0) {
      blocks.push({ items: [...listItems], ordered: listOrdered, type: "list" });
      listItems = [];
    }
  }

  function flushCode() {
    if (codeLines.length > 0) {
      blocks.push({ content: [...codeLines], language: codeLanguage, type: "code" });
      codeLines = [];
      codeLanguage = "";
    }
  }

  for (const line of lines) {
    const codeFence = line.match(/^```([\w-]+)?\s*$/);

    if (codeFence) {
      flushParagraph();
      flushQuote();
      flushList();

      if (codeLanguage || codeLines.length > 0) {
        flushCode();
      } else {
        codeLanguage = codeFence[1] ?? "";
      }

      continue;
    }

    if (codeLanguage || codeLines.length > 0) {
      codeLines.push(line);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const quoteLine = line.match(/^>\s?(.*)$/);
    const orderedItem = line.match(/^\d+\.\s+(.*)$/);
    const unorderedItem = line.match(/^[-*+]\s+(.*)$/);

    if (!line.trim()) {
      flushParagraph();
      flushQuote();
      flushList();
      continue;
    }

    if (heading) {
      flushParagraph();
      flushQuote();
      flushList();
      blocks.push({
        content: heading[2],
        level: heading[1].length,
        type: "heading",
      });
      continue;
    }

    if (quoteLine) {
      flushParagraph();
      flushList();
      quote.push(quoteLine[1]);
      continue;
    }

    if (orderedItem || unorderedItem) {
      flushParagraph();
      flushQuote();
      const nextItem = orderedItem?.[1] ?? unorderedItem?.[1] ?? "";
      const nextOrdered = Boolean(orderedItem);

      if (listItems.length === 0) {
        listOrdered = nextOrdered;
      }

      if (listOrdered !== nextOrdered) {
        flushList();
        listOrdered = nextOrdered;
      }

      listItems.push(nextItem);
      continue;
    }

    paragraph.push(line.trim());
  }

  flushParagraph();
  flushQuote();
  flushList();
  flushCode();

  return blocks;
}

function MarkdownPreview({ text }: { text: string }) {
  const blocks = useMemo(() => parseMarkdownBlocks(text), [text]);

  if (!text.trim()) {
    return <div className="reality-workspace__empty">Nothing to render yet.</div>;
  }

  return (
    <div className="markdown-preview">
      {blocks.map((block, index) => {
        if (block.type === "heading") {
          const content = renderInlineMarkdown(block.content);

          switch (Math.min(block.level + 1, 6)) {
            case 1:
              return <h1 key={`heading-${index}`}>{content}</h1>;
            case 2:
              return <h2 key={`heading-${index}`}>{content}</h2>;
            case 3:
              return <h3 key={`heading-${index}`}>{content}</h3>;
            case 4:
              return <h4 key={`heading-${index}`}>{content}</h4>;
            case 5:
              return <h5 key={`heading-${index}`}>{content}</h5>;
            default:
              return <h6 key={`heading-${index}`}>{content}</h6>;
          }
        }

        if (block.type === "quote") {
          return (
            <blockquote key={`quote-${index}`}>
              {block.content.map((line, lineIndex) => (
                <p key={`quote-line-${lineIndex}`}>{renderInlineMarkdown(line)}</p>
              ))}
            </blockquote>
          );
        }

        if (block.type === "list") {
          const ListTag = block.ordered ? "ol" : "ul";
          return (
            <ListTag key={`list-${index}`}>
              {block.items.map((item, itemIndex) => (
                <li key={`list-item-${itemIndex}`}>{renderInlineMarkdown(item)}</li>
              ))}
            </ListTag>
          );
        }

        if (block.type === "code") {
          return (
            <div key={`code-${index}`} className="markdown-code">
              <div className="markdown-code__header">
                <span>{block.language || "plain text"}</span>
              </div>
              <pre className="markdown-code__body">
                {block.content.map((line, lineIndex) => (
                  <span key={`code-line-${lineIndex}`} className="markdown-code__line">
                    {highlightCodeLine(line)}
                  </span>
                ))}
              </pre>
            </div>
          );
        }

        return <p key={`paragraph-${index}`}>{renderInlineMarkdown(block.content)}</p>;
      })}
    </div>
  );
}

export const CurrentRealityPanel = memo(function CurrentRealityPanel({
  attentionEnabled = false,
  attentionHint = null,
  attentionTokens = [],
  branchBreadcrumb = null,
  collapsed,
  continuationModeLabel,
  continuationModeTitle,
  continuationModeTone,
  conversationSections = [],
  copyConversationText = "",
  copyRawContextText = "",
  copyTokenIdsText = "",
  copyUserPromptText = "",
  detailItems = [],
  formattingSelection = null,
  hasContent,
  probabilityMode,
  promptTokenGroups = [],
  rawContextText = "",
  remainingProbabilityMass,
  selectedTokenId,
  summaryItems = [],
  stats,
  supportsEntropy,
  supportsLogprobs,
  summary,
  text,
  tokens,
  topKCoverage,
  onToggleAttentionPin,
  onSelectToken,
  onToggleCollapse,
}: CurrentRealityPanelProps) {
  const [activeTab, setActiveTab] = useState<CurrentRealityTab>("conversation");
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [hoveredTokenId, setHoveredTokenId] = useState<string | null>(null);
  const chipRefs = useRef(new Map<string, HTMLButtonElement>());
  const textRefs = useRef(new Map<string, HTMLSpanElement>());

  const assistantGroupTokens = useMemo<CurrentRealityGroupedTokenItem[]>(
    () =>
      tokens.map((token) => ({
        canonicalPosition: null,
        decodedContribution: token.decodedContribution,
        displayProbability: token.displayProbability,
        displayToken: token.displayToken,
        graphTokenId: token.id,
        id: token.id,
        kind: "assistant",
        rank: token.rank,
        rawProbability: token.rawProbability,
        rawToken: token.rawToken,
        sourceCategory: "generated_output",
        sourceLabel: "Generated output",
        specialToken: false,
        step: token.step,
        supportsLogprobs: token.supportsLogprobs,
        tokenId: null,
      })),
    [tokens],
  );
  const tokenGroups = useMemo<CurrentRealityTokenGroup[]>(
    () => [
      ...promptTokenGroups,
      ...(assistantGroupTokens.length > 0
        ? [
            {
              category: "generated_output" as const,
              id: "generated-output",
              label: "Generated output",
              tokens: assistantGroupTokens,
            },
          ]
        : []),
    ],
    [assistantGroupTokens, promptTokenGroups],
  );
  const groupedTokenMap = useMemo(
    () =>
      new Map(
        tokenGroups.flatMap((group) => group.tokens.map((token) => [token.id, token] as const)),
      ),
    [tokenGroups],
  );
  const activeTokenId =
    hoveredTokenId ?? selectedTokenId ?? tokens[tokens.length - 1]?.id ?? null;
  const activeToken = activeTokenId ? groupedTokenMap.get(activeTokenId) ?? null : null;
  const modeLabel = getProbabilityModeLabel(probabilityMode);
  const stripUsesAttention = attentionEnabled && attentionTokens.length > 0;
  const showExpandedWorkspace = !collapsed && hasContent;

  useEffect(() => {
    if (!copiedKey) {
      return;
    }

    const timeoutId = window.setTimeout(() => setCopiedKey(null), 1200);
    return () => window.clearTimeout(timeoutId);
  }, [copiedKey]);

  useEffect(() => {
    if (collapsed) {
      setHoveredTokenId(null);
    }
  }, [collapsed]);

  function focusToken(tokenId: string) {
    setHoveredTokenId(tokenId);
    textRefs.current.get(tokenId)?.scrollIntoView({
      block: "center",
      inline: "nearest",
      behavior: "smooth",
    });
    chipRefs.current.get(tokenId)?.scrollIntoView({
      block: "nearest",
      inline: "center",
      behavior: "smooth",
    });
    onSelectToken(tokenId);
  }

  async function handleCopy(copyKey: "assistant" | "conversation" | "raw" | "tokenIds" | "user") {
    const payload =
      copyKey === "conversation"
        ? copyConversationText
        : copyKey === "raw"
          ? copyRawContextText
          : copyKey === "tokenIds"
            ? copyTokenIdsText
            : copyKey === "user"
              ? copyUserPromptText
              : text;

    await navigator.clipboard.writeText(payload);
    setCopiedKey(copyKey);
  }

  function renderInteractiveToken(
    token: CurrentRealityGroupedTokenItem,
    options?: {
      chipRef?: boolean;
      className?: string;
      style?: CSSProperties;
    },
  ) {
    const isActive = activeTokenId === token.id;
    const tokenId = token.graphTokenId ?? token.id;

    return (
      <button
        key={token.id}
        ref={(element) => {
          if (!options?.chipRef) {
            return;
          }

          if (element) {
            chipRefs.current.set(token.id, element);
          } else {
            chipRefs.current.delete(token.id);
          }
        }}
        className={`${options?.className ?? "sentence-token"}${
          isActive ? " sentence-token--active" : ""
        }`}
        onClick={() => focusToken(tokenId)}
        onMouseEnter={() => setHoveredTokenId(token.id)}
        onMouseLeave={() => setHoveredTokenId((current) => (current === token.id ? null : current))}
        style={options?.style}
        type="button"
      >
        {formatGroupedTokenChipLabel(token)}
      </button>
    );
  }

  return (
    <div
      aria-expanded={!collapsed}
      className={`sentence-bar${collapsed ? " sentence-bar--collapsed" : ""}${
        !hasContent ? " sentence-bar--empty" : ""
      }`}
    >
      <div className="sentence-bar__header">
        <div className="sentence-bar__header-copy">
          <div className="sentence-bar__header-topline">
            <p className="sentence-bar__eyebrow">Current reality</p>
            <div className="sentence-bar__badges">
              <span
                className={`sentence-bar__badge sentence-bar__badge--${continuationModeTone}`}
                title={continuationModeTitle ?? undefined}
              >
                {`Mode: ${continuationModeLabel}`}
              </span>
              {supportsLogprobs ? <span className="sentence-bar__badge">{modeLabel}</span> : null}
              <span className="sentence-bar__badge">{stats.tokenCount} output tokens</span>
              {summaryItems.map((item) => (
                <span
                  key={`${item.label}:${item.value}`}
                  className={`sentence-bar__badge${getSummaryToneClass(item)}`}
                >
                  {`${item.label}: ${item.value}`}
                </span>
              ))}
              {supportsLogprobs && probabilityMode === "raw" && remainingProbabilityMass > 0 ? (
                <span className="sentence-bar__badge">
                  Other tokens {formatPercent(remainingProbabilityMass)}
                </span>
              ) : null}
            </div>
          </div>
          {branchBreadcrumb ? <p className="sentence-bar__breadcrumb">{branchBreadcrumb}</p> : null}
          <p className="sentence-bar__summary">{summary}</p>
          {!collapsed && detailItems.length > 0 ? (
            <details className="sentence-bar__details">
              <summary>Details</summary>
              <div className="reality-workspace__detail-grid reality-workspace__detail-grid--summary">
                {detailItems.map((item) => (
                  <div key={`${item.label}:${item.value}`}>
                    <dt>{item.label}</dt>
                    <dd>{item.value}</dd>
                  </div>
                ))}
              </div>
            </details>
          ) : null}
        </div>
        {hasContent ? (
          <button
            aria-label={collapsed ? "Expand current reality panel" : "Collapse current reality panel"}
            className="sentence-bar__toggle"
            onClick={onToggleCollapse}
            type="button"
          >
            {collapsed ? "Expand" : "Collapse"}
          </button>
        ) : null}
      </div>

      <div className="sentence-bar__workspace">
        {hasContent ? (
          <pre className="sentence-bar__line">{text}</pre>
        ) : (
          <div className="sentence-bar__empty-state">
            <p className="reality-workspace__panel-label">Start with a prompt</p>
            <p className="sentence-bar__summary">
              Generate a response to inspect the system prompt, user prompt, assistant output,
              probabilities, branches, and attention from one place.
            </p>
          </div>
        )}

        {showExpandedWorkspace ? (
          <>
            <div className="reality-workspace__tabs">
              {(
                ["conversation", "tokens", "raw", "statistics", "markdown"] as CurrentRealityTab[]
              ).map((tab) => (
                <button
                  key={tab}
                  className={`reality-workspace__tab${
                    activeTab === tab ? " reality-workspace__tab--active" : ""
                  }`}
                  onClick={() => setActiveTab(tab)}
                  type="button"
                >
                  {TAB_LABELS[tab]}
                </button>
              ))}
            </div>

            <div className="reality-workspace__actions">
              <button className="reality-workspace__copy" onClick={() => void handleCopy("conversation")} type="button">
                {copiedKey === "conversation" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copiedKey === "conversation" ? "Copied conversation" : "Copy conversation"}
              </button>
              <button className="reality-workspace__copy" onClick={() => void handleCopy("user")} type="button">
                {copiedKey === "user" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copiedKey === "user" ? "Copied prompt" : "Copy user prompt"}
              </button>
              <button className="reality-workspace__copy" onClick={() => void handleCopy("raw")} type="button">
                {copiedKey === "raw" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copiedKey === "raw" ? "Copied raw context" : "Copy raw context"}
              </button>
              <button
                className="reality-workspace__copy"
                onClick={() => void handleCopy("tokenIds")}
                type="button"
              >
                {copiedKey === "tokenIds" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copiedKey === "tokenIds" ? "Copied token IDs" : "Copy token IDs"}
              </button>
            </div>

            <div className="reality-workspace__panel">
              {activeTab === "conversation" ? (
                <div className="reality-workspace__scroller">
                  <div className="reality-conversation">
                    {conversationSections.map((section) => {
                      const Icon = getConversationRoleIcon(section.role);
                      const sectionTokens = section.tokenIds
                        .map((tokenId) => groupedTokenMap.get(tokenId) ?? null)
                        .filter(
                          (token): token is CurrentRealityGroupedTokenItem => Boolean(token),
                        );

                      return (
                        <section
                          key={section.id}
                          className={`reality-conversation__card reality-conversation__card--${section.role}`}
                        >
                          <div className="reality-conversation__header">
                            <span className="reality-conversation__icon">
                              <Icon className="h-4 w-4" />
                            </span>
                            <p className="reality-conversation__label">{section.label}</p>
                          </div>
                          <div className="reality-conversation__body">
                            {sectionTokens.length > 0 ? (
                              sectionTokens.map((token) => {
                                const isActive = activeTokenId === token.id;
                                const tokenId = token.graphTokenId ?? token.id;

                                return (
                                  <span
                                    key={token.id}
                                    ref={(element) => {
                                      if (element) {
                                        textRefs.current.set(token.id, element);
                                      } else {
                                        textRefs.current.delete(token.id);
                                      }
                                    }}
                                    className={`reality-text-token${
                                      isActive ? " reality-text-token--active" : ""
                                    }`}
                                    onClick={() => focusToken(tokenId)}
                                    onMouseEnter={() => setHoveredTokenId(token.id)}
                                    onMouseLeave={() =>
                                      setHoveredTokenId((current) =>
                                        current === token.id ? null : current,
                                      )
                                    }
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter" || event.key === " ") {
                                        event.preventDefault();
                                        focusToken(tokenId);
                                      }
                                    }}
                                    role="button"
                                    tabIndex={0}
                                  >
                                    {token.decodedContribution}
                                  </span>
                                );
                              })
                            ) : (
                              <pre className="reality-conversation__text">{section.text || "None"}</pre>
                            )}
                          </div>
                        </section>
                      );
                    })}

                    {formattingSelection ? (
                      <div className="reality-conversation__formatting">
                        <span className="reality-conversation__icon">
                          <Braces className="h-4 w-4" />
                        </span>
                        <div>
                          <p className="reality-conversation__label">{formattingSelection.label}</p>
                          <p className="reality-conversation__formatting-copy">
                            {`${formattingSelection.description}: ${formattingSelection.token}`}
                          </p>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {activeTab === "tokens" ? (
                <div className="reality-workspace__tokens-panel">
                  {tokenGroups.length > 0 ? (
                    tokenGroups.map((group) => (
                      <section key={group.id} className="reality-token-group">
                        <div className="reality-workspace__panel-header">
                          <p className="reality-workspace__panel-label">{group.label}</p>
                          <span className="sentence-bar__badge">
                            {`${group.tokens.length} token${group.tokens.length === 1 ? "" : "s"}`}
                          </span>
                        </div>
                        <div className="reality-workspace__timeline">
                          {group.tokens.map((token) =>
                            renderInteractiveToken(token, {
                              className: `sentence-token${
                                token.kind === "prompt" ? " sentence-token--prompt-scope" : ""
                              }`,
                              style:
                                token.kind === "assistant"
                                  ? ({
                                      "--sentence-strength": `${
                                        token.supportsLogprobs
                                          ? Math.max(token.displayProbability ?? 0, 0.08)
                                          : 0.38
                                      }`,
                                    } as CSSProperties)
                                  : undefined,
                            }),
                          )}
                        </div>
                      </section>
                    ))
                  ) : (
                    <div className="reality-workspace__empty">No canonical tokens available yet.</div>
                  )}

                  {activeToken ? (
                    <div className="reality-workspace__detail-grid">
                      <div>
                        <dt>Display</dt>
                        <dd>{activeToken.displayToken || formatGroupedTokenChipLabel(activeToken)}</dd>
                      </div>
                      <div>
                        <dt>Decoded</dt>
                        <dd>{activeToken.decodedContribution || "<empty>"}</dd>
                      </div>
                      <div>
                        <dt>Raw token</dt>
                        <dd>{activeToken.rawToken || "<empty>"}</dd>
                      </div>
                      <div>
                        <dt>Token id</dt>
                        <dd>{activeToken.tokenId ?? "Unavailable"}</dd>
                      </div>
                      <div>
                        <dt>Source</dt>
                        <dd>{activeToken.sourceLabel}</dd>
                      </div>
                      <div>
                        <dt>{activeToken.kind === "prompt" ? "Position" : "Step"}</dt>
                        <dd>{activeToken.kind === "prompt" ? activeToken.canonicalPosition : activeToken.step}</dd>
                      </div>
                      <div>
                        <dt>Displayed probability</dt>
                        <dd>
                          {formatMetricPercent(
                            activeToken.displayProbability,
                            activeToken.supportsLogprobs,
                          )}
                        </dd>
                      </div>
                      <div>
                        <dt>Raw probability</dt>
                        <dd>
                          {formatMetricPercent(
                            activeToken.rawProbability,
                            activeToken.supportsLogprobs,
                          )}
                        </dd>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}

              {activeTab === "raw" ? (
                <div className="reality-workspace__scroller">
                  <div className="reality-workspace__panel-header">
                    <p className="reality-workspace__panel-label">Exact raw context</p>
                    <button className="reality-workspace__copy" onClick={() => void handleCopy("raw")} type="button">
                      {copiedKey === "raw" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      {copiedKey === "raw" ? "Copied" : "Copy"}
                    </button>
                  </div>
                  <pre className="reality-workspace__raw">
                    {rawContextText || "Raw context is unavailable for this provider response."}
                  </pre>
                </div>
              ) : null}

              {activeTab === "statistics" ? (
                <div className="reality-workspace__detail-grid">
                  <div>
                    <dt>Probability</dt>
                    <dd>{formatMetricPercent(stats.displayProbability, supportsLogprobs)}</dd>
                  </div>
                  <div>
                    <dt>Raw probability</dt>
                    <dd>{formatMetricPercent(stats.rawProbability, supportsLogprobs)}</dd>
                  </div>
                  <div>
                    <dt>Entropy</dt>
                    <dd>{formatNumber(stats.entropy, supportsEntropy)}</dd>
                  </div>
                  <div>
                    <dt>Latency</dt>
                    <dd>{formatLatency(stats.latency)}</dd>
                  </div>
                  <div>
                    <dt>Output token count</dt>
                    <dd>{stats.tokenCount}</dd>
                  </div>
                  <div>
                    <dt>Branch depth</dt>
                    <dd>{stats.branchDepth}</dd>
                  </div>
                  <div>
                    <dt>Top-K coverage</dt>
                    <dd>{supportsLogprobs ? formatPercent(topKCoverage) : "Unavailable"}</dd>
                  </div>
                  <div>
                    <dt>Other tokens</dt>
                    <dd>{supportsLogprobs ? formatPercent(remainingProbabilityMass) : "Unavailable"}</dd>
                  </div>
                </div>
              ) : null}

              {activeTab === "markdown" ? (
                <div className="reality-workspace__scroller">
                  <div className="reality-workspace__panel-header">
                    <p className="reality-workspace__panel-label">Assistant markdown</p>
                    <button
                      className="reality-workspace__copy"
                      onClick={() => void handleCopy("assistant")}
                      type="button"
                    >
                      {copiedKey === "assistant" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      {copiedKey === "assistant" ? "Copied" : "Copy assistant"}
                    </button>
                  </div>
                  <MarkdownPreview text={text} />
                </div>
              ) : null}
            </div>

            <div className="reality-workspace__strip">
              <div className="reality-workspace__panel-header">
                <p className="reality-workspace__panel-label">
                  {stripUsesAttention ? "Attention strip" : "Pinned token strip"}
                </p>
                {stripUsesAttention && attentionHint ? (
                  <span className="reality-workspace__attention-hint" title={attentionHint}>
                    Attention Lens
                  </span>
                ) : null}
              </div>
              <div className="reality-workspace__strip-track">
                {stripUsesAttention ? (
                  attentionTokens.map((token) => (
                    <button
                      key={token.id}
                      className={`sentence-token sentence-token--attention${
                        token.sequenceScope === "prompt" ? " sentence-token--prompt-scope" : ""
                      }${token.isQuery ? " sentence-token--query" : ""}${
                        token.isPinned ? " sentence-token--attention-pinned" : ""
                      }`}
                      onClick={() => onToggleAttentionPin?.(token.id)}
                      style={
                        {
                          "--attention-weight": `${Math.max(token.attentionWeight ?? 0, 0)}`,
                        } as CSSProperties
                      }
                      title={`${token.displayToken || token.decodedContribution || token.rawToken}
Weight: ${
                        typeof token.attentionWeight === "number"
                          ? formatPercent(token.attentionWeight)
                          : "Unavailable"
                      }
Position: ${token.fullPosition}`}
                      type="button"
                    >
                      {token.displayToken || token.decodedContribution || token.rawToken}
                    </button>
                  ))
                ) : assistantGroupTokens.length > 0 ? (
                  assistantGroupTokens.map((token) =>
                    renderInteractiveToken(token, {
                      chipRef: true,
                      className: "sentence-token",
                      style: {
                        "--sentence-strength": `${
                          token.supportsLogprobs
                            ? Math.max(token.displayProbability ?? 0, 0.08)
                            : 0.38
                        }`,
                      } as CSSProperties,
                    }),
                  )
                ) : (
                  <span className="sentence-bar__placeholder">
                    Generate a response, then click around the graph to switch realities.
                  </span>
                )}
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
});
