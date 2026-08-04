"use client";

import { BaseEdge, getBezierPath, type EdgeProps } from "@xyflow/react";

import type { ProbabilityFlowEdge } from "@/components/canvas/types";

function getEdgeTone(probability: number, isMainPath: boolean) {
  const width = 1.2 + probability * 4.4;

  if (isMainPath) {
    return {
      stroke: "#38bdf8",
      glow: "rgba(56, 189, 248, 0.18)",
      width: width + 0.3,
      opacity: 0.95,
    };
  }

  if (probability < 0.36) {
    return {
      stroke: "#8b5cf6",
      glow: "rgba(139, 92, 246, 0.16)",
      width,
      opacity: 0.72,
    };
  }

  return {
    stroke: "#64748b",
    glow: "rgba(100, 116, 139, 0.12)",
    width,
    opacity: Math.max(0.26, probability),
  };
}

export function ProbabilityEdge({
  id,
  data,
  sourceX,
  sourceY,
  sourcePosition,
  targetX,
  targetY,
  targetPosition,
}: EdgeProps<ProbabilityFlowEdge>) {
  const probability = data?.probability ?? 0.5;
  const tone = getEdgeTone(probability, data?.isMainPath ?? false);
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    curvature: 0.28,
  });

  return (
    <>
      <BaseEdge
        id={`${id}-glow`}
        className="probability-edge probability-edge--glow"
        path={path}
        style={{
          stroke: tone.glow,
          strokeWidth: tone.width + 8,
          opacity: tone.opacity,
        }}
      />
      <BaseEdge
        id={id}
        className={`probability-edge${data?.isMainPath ? " probability-edge--main" : ""}`}
        path={path}
        style={{
          stroke: tone.stroke,
          strokeWidth: tone.width,
          opacity: tone.opacity,
        }}
      />
    </>
  );
}
