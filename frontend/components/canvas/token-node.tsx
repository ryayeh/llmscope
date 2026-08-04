"use client";

import type { CSSProperties } from "react";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { LoaderCircle } from "lucide-react";

import type { TokenFlowNode } from "@/components/canvas/types";

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function getNodeTone({
  isMainPath,
  probability,
  selected,
}: {
  isMainPath: boolean;
  probability: number;
  selected: boolean;
}) {
  if (selected) {
    return {
      accent: "#f59e0b",
      border: "rgba(251, 191, 36, 0.46)",
      glow: "rgba(245, 158, 11, 0.28)",
      rail: "linear-gradient(90deg, rgba(251, 191, 36, 1), rgba(249, 115, 22, 0.86))",
    };
  }

  if (isMainPath) {
    return {
      accent: "#38bdf8",
      border: "rgba(56, 189, 248, 0.4)",
      glow: "rgba(56, 189, 248, 0.28)",
      rail: "linear-gradient(90deg, rgba(56, 189, 248, 1), rgba(14, 165, 233, 0.82))",
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
  const tone = getNodeTone({
    isMainPath: data.isMainPath || data.kind === "prompt",
    probability: data.probability,
    selected,
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
      }${data.status === "loading" ? " token-node--loading" : ""}`}
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
        <p className="token-node__title">{data.tokenText}</p>
      </div>

      <div className="token-node__rail">
        <div
          className="token-node__rail-fill"
          style={{ width: `${Math.max(data.probability * 100, data.kind === "prompt" ? 100 : 10)}%` }}
        />
      </div>

      <div className="token-node__hover">
        <p className="token-node__hover-label">{formatPercent(data.probability)}</p>
        <p className="token-node__hover-preview">{data.textPreview}</p>
        <p className="token-node__hover-hint">
          {data.kind === "prompt" ? "Double-click to open futures" : "Double-click to continue"}
        </p>
      </div>
    </div>
  );
}
