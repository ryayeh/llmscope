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

import { Check, Copy } from "lucide-react";

import type { ProbabilityViewMode } from "@/components/canvas/types";

type CurrentRealityTab = "text" | "tokens" | "statistics" | "markdown";

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
  collapsed: boolean;
  continuationModeLabel: string;
  continuationModeTitle: string | null;
  continuationModeTone: "exact" | "approximate";
  hasContent: boolean;
  probabilityMode: ProbabilityViewMode;
  remainingProbabilityMass: number;
  selectedTokenId: string | null;
  stats: CurrentRealityStats;
  supportsEntropy: boolean;
  supportsLogprobs: boolean;
  summary: string;
  text: string;
  tokens: CurrentRealityTokenItem[];
  topKCoverage: number;
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
  markdown: "Markdown",
  statistics: "Statistics",
  text: "Text",
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

function formatTokenChipLabel(token: CurrentRealityTokenItem) {
  const candidate = token.displayToken || token.decodedContribution || token.rawToken;

  if (candidate.trim()) {
    return candidate;
  }

  return candidate.replace(/ /g, "\u2420").replace(/\n/g, "\u21B5\n").replace(/\t/g, "\u21E5");
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
  collapsed,
  continuationModeLabel,
  continuationModeTitle,
  continuationModeTone,
  hasContent,
  probabilityMode,
  remainingProbabilityMass,
  selectedTokenId,
  stats,
  supportsEntropy,
  supportsLogprobs,
  summary,
  text,
  tokens,
  topKCoverage,
  onSelectToken,
  onToggleCollapse,
}: CurrentRealityPanelProps) {
  const [activeTab, setActiveTab] = useState<CurrentRealityTab>("text");
  const [copied, setCopied] = useState(false);
  const [hoveredTokenId, setHoveredTokenId] = useState<string | null>(null);
  const chipRefs = useRef(new Map<string, HTMLButtonElement>());
  const textRefs = useRef(new Map<string, HTMLSpanElement>());

  const activeTokenId = hoveredTokenId ?? selectedTokenId ?? tokens[tokens.length - 1]?.id ?? null;
  const activeToken = tokens.find((token) => token.id === activeTokenId) ?? null;
  const modeLabel = getProbabilityModeLabel(probabilityMode);

  useEffect(() => {
    if (!copied) {
      return;
    }

    const timeoutId = window.setTimeout(() => setCopied(false), 1200);
    return () => window.clearTimeout(timeoutId);
  }, [copied]);

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

  async function handleCopy() {
    await navigator.clipboard.writeText(text);
    setCopied(true);
  }

  return (
    <div
      aria-expanded={!collapsed}
      className={`sentence-bar${collapsed ? " sentence-bar--collapsed" : ""}`}
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
              <span className="sentence-bar__badge">{stats.tokenCount} tokens</span>
              {supportsLogprobs ? (
                <span className="sentence-bar__badge">
                  {formatMetricPercent(stats.displayProbability, supportsLogprobs)}
                </span>
              ) : null}
              {supportsLogprobs && probabilityMode === "raw" && remainingProbabilityMass > 0 ? (
                <span className="sentence-bar__badge">
                  Other tokens {formatPercent(remainingProbabilityMass)}
                </span>
              ) : null}
            </div>
          </div>
          <p className="sentence-bar__summary">{summary}</p>
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
        <pre className="sentence-bar__line">
          {text || "Generate a response to start exploring the graph."}
        </pre>

        {!collapsed ? (
          <>
            <div className="reality-workspace__tabs">
              {(
                ["text", "tokens", "statistics", "markdown"] as CurrentRealityTab[]
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

            <div className="reality-workspace__panel">
              {(activeTab === "text" || activeTab === "markdown") && (
                <div className="reality-workspace__panel-header">
                  <p className="reality-workspace__panel-label">
                    {activeTab === "text" ? "Generated text" : "Markdown view"}
                  </p>
                  <button className="reality-workspace__copy" onClick={() => void handleCopy()} type="button">
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
              )}

              {activeTab === "text" ? (
                <div className="reality-workspace__scroller">
                  <div className="reality-workspace__text">
                    {tokens.length > 0 ? (
                      tokens.map((token) => {
                        const isActive = activeTokenId === token.id;
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
                            className={`reality-text-token${isActive ? " reality-text-token--active" : ""}`}
                            onClick={() => focusToken(token.id)}
                            onMouseEnter={() => setHoveredTokenId(token.id)}
                            onMouseLeave={() => setHoveredTokenId((current) => (current === token.id ? null : current))}
                            role="button"
                            tabIndex={0}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                focusToken(token.id);
                              }
                            }}
                          >
                            {token.decodedContribution}
                          </span>
                        );
                      })
                    ) : (
                      <span>{text || "Generate a response to start exploring the graph."}</span>
                    )}
                  </div>
                </div>
              ) : null}

              {activeTab === "tokens" ? (
                <div className="reality-workspace__tokens-panel">
                  <div className="reality-workspace__timeline">
                    {tokens.length > 0 ? (
                      tokens.map((token) => {
                        const isActive = activeTokenId === token.id;
                        return (
                          <button
                            key={token.id}
                            className={`sentence-token${token.isChanged ? " sentence-token--changed" : ""}${
                              isActive ? " sentence-token--active" : ""
                            }`}
                            onClick={() => focusToken(token.id)}
                            onMouseEnter={() => setHoveredTokenId(token.id)}
                            onMouseLeave={() => setHoveredTokenId((current) => (current === token.id ? null : current))}
                            style={
                              {
                                "--sentence-strength": `${
                                  token.supportsLogprobs
                                    ? Math.max(token.displayProbability, 0.08)
                                    : 0.38
                                }`,
                              } as CSSProperties
                            }
                            type="button"
                          >
                            {formatTokenChipLabel(token)}
                          </button>
                        );
                      })
                    ) : (
                      <div className="reality-workspace__empty">No generated tokens yet.</div>
                    )}
                  </div>

                  {activeToken ? (
                    <div className="reality-workspace__detail-grid">
                      <div>
                        <dt>Display</dt>
                        <dd>{activeToken.displayToken || formatTokenChipLabel(activeToken)}</dd>
                      </div>
                      <div>
                        <dt>Decoded</dt>
                        <dd>{activeToken.decodedContribution || "<empty>"}</dd>
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
                      <div>
                        <dt>Rank</dt>
                        <dd>{activeToken.rank}</dd>
                      </div>
                      <div>
                        <dt>Step</dt>
                        <dd>{activeToken.step}</dd>
                      </div>
                    </div>
                  ) : null}
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
                    <dt>Token count</dt>
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
                    <dd>
                      {supportsLogprobs ? formatPercent(remainingProbabilityMass) : "Unavailable"}
                    </dd>
                  </div>
                </div>
              ) : null}

              {activeTab === "markdown" ? (
                <div className="reality-workspace__scroller">
                  <MarkdownPreview text={text} />
                </div>
              ) : null}
            </div>

            <div className="reality-workspace__strip">
              <p className="reality-workspace__panel-label">Pinned token strip</p>
              <div className="reality-workspace__strip-track">
                {tokens.length > 0 ? (
                  tokens.map((token) => {
                    const isActive = activeTokenId === token.id;
                    return (
                      <button
                        key={token.id}
                        ref={(element) => {
                          if (element) {
                            chipRefs.current.set(token.id, element);
                          } else {
                            chipRefs.current.delete(token.id);
                          }
                        }}
                        className={`sentence-token${token.isChanged ? " sentence-token--changed" : ""}${
                          isActive ? " sentence-token--active" : ""
                        }`}
                        onClick={() => focusToken(token.id)}
                        onMouseEnter={() => setHoveredTokenId(token.id)}
                        onMouseLeave={() => setHoveredTokenId((current) => (current === token.id ? null : current))}
                        style={
                          {
                            "--sentence-strength": `${
                              token.supportsLogprobs
                                ? Math.max(token.displayProbability, 0.08)
                                : 0.38
                            }`,
                          } as CSSProperties
                        }
                        type="button"
                      >
                        {formatTokenChipLabel(token)}
                      </button>
                    );
                  })
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
