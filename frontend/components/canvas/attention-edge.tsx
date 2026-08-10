"use client";

import { BaseEdge, getBezierPath, type EdgeProps } from "@xyflow/react";

import type { AttentionFlowEdge } from "@/components/canvas/types";

function getAttentionTone({
  isDimmed,
  isPinned,
  weight,
}: {
  isDimmed: boolean;
  isPinned: boolean;
  weight: number;
}) {
  const width = 1.4 + weight * 9;

  if (isPinned) {
    return {
      stroke: "#c084fc",
      glow: "rgba(192, 132, 252, 0.22)",
      width: width + 0.8,
      opacity: isDimmed ? 0.44 : 0.98,
    };
  }

  return {
    stroke: "#8b5cf6",
    glow: "rgba(139, 92, 246, 0.18)",
    width,
    opacity: isDimmed ? 0.16 : Math.max(0.28, Math.min(0.9, 0.2 + weight * 1.2)),
  };
}

export function AttentionEdge({
  id,
  data,
  sourceX,
  sourceY,
  sourcePosition,
  targetX,
  targetY,
  targetPosition,
}: EdgeProps<AttentionFlowEdge>) {
  const weight = data?.weight ?? 0;
  const tone = getAttentionTone({
    weight,
    isPinned: data?.isPinned ?? false,
    isDimmed: data?.isDimmed ?? false,
  });
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    curvature: 0.42,
  });
  const title = [
    `${data?.sourceDisplayToken ?? "Source"} -> ${data?.targetDisplayToken ?? "Target"}`,
    `Mode: ${data?.analysisMode === "representation" ? "Representation attention" : "Prediction attention"}`,
    `Attention weight: ${((data?.weight ?? 0) * 100).toFixed(2)}%`,
    `Rank: ${data?.rank ?? 0}`,
    `Layer: ${data?.layer ?? 0}`,
    `Head: ${data?.headLabel ?? "Average heads"}`,
    `Query position: ${data?.queryPosition ?? 0}`,
    `Source: ${data?.sourceLabel ?? "Earlier token"} (${data?.sourceCategory ?? "generated_output"})`,
    `Positions: ${data?.sourceFullPosition ?? 0} -> ${data?.targetFullPosition ?? 0}`,
  ].join("\n");

  return (
    <>
      <BaseEdge
        id={`${id}-glow`}
        className="attention-edge attention-edge--glow"
        path={path}
        style={{
          stroke: tone.glow,
          strokeWidth: tone.width + 10,
          opacity: tone.opacity,
        }}
      />
      <BaseEdge
        id={id}
        className={`attention-edge${data?.isPinned ? " attention-edge--pinned" : ""}`}
        path={path}
        style={{
          stroke: tone.stroke,
          strokeWidth: tone.width,
          opacity: tone.opacity,
        }}
      />
      <title>{title}</title>
    </>
  );
}
