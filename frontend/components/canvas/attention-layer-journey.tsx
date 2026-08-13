"use client";

import type { CSSProperties } from "react";

import type { HuggingFaceAttentionLayerJourney } from "@/types/api";

function formatPercent(value: number) {
  return `${(Math.max(value, 0) * 100).toFixed(1)}%`;
}

interface AttentionLayerJourneyProps {
  headLabel: string;
  journey: HuggingFaceAttentionLayerJourney | null | undefined;
  loading: boolean;
  selectedLayer: number;
  onSelectLayer: (layerIndex: number) => void;
}

export function AttentionLayerJourney({
  headLabel,
  journey,
  loading,
  selectedLayer,
  onSelectLayer,
}: AttentionLayerJourneyProps) {
  if (loading && !journey) {
    return <div className="inspector-empty">Loading layer journey...</div>;
  }

  if (!journey || journey.layers.length === 0 || journey.rows.length === 0) {
    return <div className="inspector-empty">Layer journey becomes available once attention analysis completes.</div>;
  }

  const scale = journey.scale_max > 0 ? journey.scale_max : 1;

  return (
    <div className="attention-journey">
      <div className="attention-journey__header">
        <p className="attention-journey__scale">
          Shared scale max {formatPercent(scale)}
        </p>
        {journey.sampled ? (
          <span className="inspector-inline-badge">Sampled layers</span>
        ) : null}
      </div>
      <div className="attention-journey__grid">
        <div className="attention-journey__corner">Source</div>
        {journey.layers.map((layerIndex) => (
          <button
            key={`layer-${layerIndex}`}
            className={`attention-journey__layer${
              selectedLayer === layerIndex ? " attention-journey__layer--active" : ""
            }`}
            onClick={() => onSelectLayer(layerIndex)}
            type="button"
          >
            {layerIndex}
          </button>
        ))}
        {journey.rows.map((row) => (
          <div className="attention-journey__row" key={row.row_id}>
            <div className="attention-journey__label" title={row.source?.raw_token ?? row.label}>
              {row.label}
            </div>
            {row.weights.map((weight, index) => {
              const layerIndex = journey.layers[index] ?? index;
              const strength = Math.max(weight / scale, 0);
              const title = row.source
                ? `${row.label} · pos ${row.source.full_position} · ${row.source.source_label} · layer ${layerIndex} · ${headLabel} · ${formatPercent(weight)}`
                : `${row.label} · layer ${layerIndex} · ${headLabel} · ${formatPercent(weight)}`;

              return (
                <button
                  key={`${row.row_id}:${layerIndex}`}
                  aria-label={title}
                  className="attention-journey__cell"
                  onClick={() => onSelectLayer(layerIndex)}
                  style={{ "--attention-strength": String(strength) } as CSSProperties}
                  title={title}
                  type="button"
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
