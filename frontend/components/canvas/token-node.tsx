"use client";

import type { CSSProperties } from "react";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { LoaderCircle } from "lucide-react";

import type { TokenFlowNode } from "@/components/canvas/types";

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function getNodeTone({
  isActiveReality,
  probability,
  selected,
  supportsLogprobs,
}: {
  isActiveReality: boolean;
  probability: number;
  selected: boolean;
  supportsLogprobs: boolean;
}) {
  if (selected) {
    return {
      accent: "#f59e0b",
      border: "rgba(251, 191, 36, 0.46)",
      glow: "rgba(245, 158, 11, 0.28)",
      rail: "linear-gradient(90deg, rgba(251, 191, 36, 1), rgba(249, 115, 22, 0.86))",
    };
  }

  if (isActiveReality) {
    return {
      accent: "#38bdf8",
      border: "rgba(56, 189, 248, 0.4)",
      glow: "rgba(56, 189, 248, 0.28)",
      rail: "linear-gradient(90deg, rgba(56, 189, 248, 1), rgba(14, 165, 233, 0.82))",
    };
  }

  if (!supportsLogprobs) {
    return {
      accent: "#94a3b8",
      border: "rgba(148, 163, 184, 0.26)",
      glow: "rgba(148, 163, 184, 0.16)",
      rail: "linear-gradient(90deg, rgba(148, 163, 184, 0.86), rgba(100, 116, 139, 0.72))",
    };
  }

  if (probability < 0.36) {
    return {
      accent: "#a78bfa",
      border: "rgba(167, 139, 250, 0.34)",
      glow: "rgba(167, 139, 250, 0.22)",
      rail: "linear-gradient(90deg, rgba(167, 139, 250, 0.94), rgba(139, 92, 246, 0.76))",
    };
  }

  return {
    accent: "#94a3b8",
    border: "rgba(148, 163, 184, 0.22)",
    glow: "rgba(148, 163, 184, 0.16)",
    rail: "linear-gradient(90deg, rgba(148, 163, 184, 0.88), rgba(100, 116, 139, 0.68))",
  };
}

export function TokenNode({ data, selected }: NodeProps<TokenFlowNode>) {
  const supportsLogprobs = data.providerCapabilities.supports_logprobs;
  const supportsBranching = data.providerCapabilities.supports_branching;
  const tone = getNodeTone({
    isActiveReality: data.isActiveReality || data.kind === "prompt",
    probability: data.displayProbability,
    selected,
    supportsLogprobs,
  });
  const style = {
    "--node-accent": tone.accent,
    "--node-border": tone.border,
    "--node-glow": tone.glow,
    "--node-rail": tone.rail,
  } as CSSProperties;

  return (
    <div
      className={`token-node${selected ? " token-node--selected" : ""}${
        data.kind === "prompt" ? " token-node--prompt" : ""
      }${data.status === "loading" ? " token-node--loading" : ""}${
        data.isDimmed ? " token-node--dimmed" : ""
      }${data.isSearchMatch ? " token-node--search-match" : ""}${
        data.isSearchFocused ? " token-node--search-focused" : ""
      }${data.isActiveReality ? " token-node--active-reality" : ""}${
        data.isPinned ? " token-node--pinned" : ""
      }`}
      style={style}
    >
      <Handle className="token-node__handle" position={Position.Left} type="target" />
      <Handle className="token-node__handle" position={Position.Right} type="source" />

      {data.status === "loading" ? (
        <div className="token-node__loading">
          <LoaderCircle className="h-4 w-4 animate-spin" />
        </div>
      ) : null}

      <div className="token-node__body">
        <p className="token-node__title">{data.displayTokenText}</p>
        {data.kind === "token" && data.isActiveReality ? (
          <span className="token-node__status-badge">Selected</span>
        ) : null}
      </div>

      <div className="token-node__probability">
        {supportsLogprobs ? formatPercent(data.displayProbability) : "Unavailable"}
      </div>

      <div className="token-node__rail">
        <div
          className="token-node__rail-fill"
          style={{
            width: `${
              supportsLogprobs
                ? Math.max(data.displayProbability * 100, data.kind === "prompt" ? 100 : 10)
                : 100
            }%`,
          }}
        />
      </div>

      <div className="token-node__hover">
        <p className="token-node__hover-label">
          {supportsLogprobs
            ? `${formatPercent(data.displayProbability)} | ${
                data.probabilityMode === "normalized" ? "Normalized Top-K" : "Raw"
              }`
            : "Probability unavailable"}
        </p>
        <div className="token-node__hover-grid">
          <p>
            <strong>Raw</strong>
            <span className="token-node__hover-code">{data.tokenText || "..."}</span>
          </p>
          <p>
            <strong>Shown</strong>
            <span className="token-node__hover-code">{data.displayTokenText || "..."}</span>
          </p>
          <p>
            <strong>Shown p</strong>
            <span>{supportsLogprobs ? formatPercent(data.displayProbability) : "Unavailable"}</span>
          </p>
          <p>
            <strong>Raw p</strong>
            <span>{supportsLogprobs ? formatPercent(data.rawProbability) : "Unavailable"}</span>
          </p>
          <p>
            <strong>Bytes</strong>
            <span className="token-node__hover-code">{data.tokenBytes.join(" ") || "..."}</span>
          </p>
          <p>
            <strong>UTF-8</strong>
            <span>{data.utf8Length}</span>
          </p>
          <p>
            <strong>Chars</strong>
            <span>{data.characterLength}</span>
          </p>
          <p>
            <strong>Lead ws</strong>
            <span>{data.leadingWhitespaceCount}</span>
          </p>
          <p>
            <strong>Trail ws</strong>
            <span>{data.trailingWhitespaceCount}</span>
          </p>
          <p>
            <strong>Tokenizer</strong>
            <span>{data.tokenizerId ?? "-"}</span>
          </p>
        </div>
        <p className="token-node__hover-preview">{data.contextAfter || data.textPreview}</p>
        <p className="token-node__hover-hint">
          {supportsBranching
            ? data.kind === "prompt"
              ? "Double-click to open futures"
              : "Double-click to expand alternatives"
            : "Selection only"}
        </p>
      </div>
    </div>
  );
}
