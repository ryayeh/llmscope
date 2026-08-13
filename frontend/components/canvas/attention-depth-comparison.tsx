"use client";

import {
  getAttentionDepthSummary,
  getAttentionSourceCategoryLabel,
} from "@/lib/attention-guided";
import type { HuggingFaceAttentionLayerSummary } from "@/types/api";

function formatPercent(value: number) {
  return `${(Math.max(value, 0) * 100).toFixed(1)}%`;
}

interface AttentionDepthComparisonProps {
  layerCount: number;
  loading: boolean;
  selectedLayer: number;
  summaries: HuggingFaceAttentionLayerSummary[];
  onSelectLayer: (layerIndex: number) => void;
}

export function AttentionDepthComparison({
  layerCount,
  loading,
  selectedLayer,
  summaries,
  onSelectLayer,
}: AttentionDepthComparisonProps) {
  if (loading && summaries.length === 0) {
    return <div className="inspector-empty">Comparing 3 depths...</div>;
  }

  if (summaries.length === 0) {
    return <div className="inspector-empty">Depth comparison will appear after attention analysis completes.</div>;
  }

  return (
    <div className="attention-compare">
      {summaries.map((summary) => {
        const depthSummary = getAttentionDepthSummary(summary.layer_index, layerCount);
        const topSource = summary.top_meaningful_source;
        const topLabel = topSource
          ? topSource.display_token || topSource.raw_token
          : "No readable non-template source surfaced";
        const topMeta = topSource ? getAttentionSourceCategoryLabel(topSource) : "Grouped context";

        return (
          <button
            key={summary.layer_index}
            aria-pressed={selectedLayer === summary.layer_index}
            className={`attention-compare__row${
              selectedLayer === summary.layer_index ? " attention-compare__row--active" : ""
            }`}
            onClick={() => onSelectLayer(summary.layer_index)}
            type="button"
          >
            <div className="attention-compare__row-header">
              <span className="attention-compare__label">{depthSummary.label}</span>
              <span className="attention-compare__meta">Layer {summary.layer_index}</span>
            </div>
            <p className="attention-compare__source">{topLabel}</p>
            <p className="attention-compare__submeta">{topMeta}</p>
            <div className="attention-compare__stats">
              <span>{formatPercent(topSource?.attention_weight ?? 0)}</span>
              <span>Input context {formatPercent(summary.category_breakdown.input_context)}</span>
              <span>Earlier output {formatPercent(summary.category_breakdown.earlier_output)}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
