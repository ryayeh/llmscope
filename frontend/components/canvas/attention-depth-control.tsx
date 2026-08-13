"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

import {
  getAttentionDepthSummary,
  getRepresentativeAttentionLayers,
} from "@/lib/attention-guided";

interface AttentionDepthControlProps {
  comparePressed: boolean;
  disabled: boolean;
  layerCount: number;
  selectedLayer: number;
  onLayerChange: (layerIndex: number) => void;
  onOpenCompare: () => void;
}

const DEPTH_TOOLTIP =
  "Earlier, middle, and later are location labels, not guaranteed cognitive stages. Attention patterns can change substantially across depth.";

export function AttentionDepthControl({
  comparePressed,
  disabled,
  layerCount,
  selectedLayer,
  onLayerChange,
  onOpenCompare,
}: AttentionDepthControlProps) {
  const safeLayerCount = Math.max(layerCount, 1);
  const summary = getAttentionDepthSummary(selectedLayer, safeLayerCount);
  const presets = getRepresentativeAttentionLayers(safeLayerCount);
  const previousDisabled = disabled || selectedLayer <= 0;
  const nextDisabled = disabled || selectedLayer >= Math.max(safeLayerCount - 1, 0);

  return (
    <section className="attention-panel attention-panel--depth" aria-labelledby="attention-depth">
      <div className="attention-panel__header">
        <div>
          <p className="attention-panel__eyebrow" id="attention-depth">
            Model depth
          </p>
          <p className="attention-panel__title">{summary.label}</p>
        </div>
        <span className="inspector-inline-badge" title={DEPTH_TOOLTIP}>
          {summary.percentThrough}% through model
        </span>
      </div>

      <div className="attention-depth__axis" aria-hidden="true">
        <span>Earlier</span>
        <div className="attention-depth__axis-line" />
        <span>Later</span>
      </div>

      <div className="attention-depth__controls">
        <button
          className="icon-button"
          disabled={previousDisabled}
          onClick={() => onLayerChange(Math.max(selectedLayer - 1, 0))}
          type="button"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <label className="attention-depth__slider">
          <span className="sr-only">
            Select model depth layer. Currently layer {selectedLayer} of {safeLayerCount}.
          </span>
          <input
            aria-label={`Model depth layer ${selectedLayer} of ${safeLayerCount}`}
            disabled={disabled}
            max={Math.max(safeLayerCount - 1, 0)}
            min={0}
            onChange={(event) => onLayerChange(Number(event.target.value))}
            type="range"
            value={selectedLayer}
          />
        </label>
        <button
          className="icon-button"
          disabled={nextDisabled}
          onClick={() =>
            onLayerChange(Math.min(selectedLayer + 1, Math.max(safeLayerCount - 1, 0)))
          }
          type="button"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      <div className="attention-depth__meta">
        <span>Layer {selectedLayer} of {safeLayerCount}</span>
        <span>{summary.preset === "custom" ? "Custom" : summary.preset}</span>
      </div>

      <div className="attention-depth__presets" role="toolbar" aria-label="Model depth presets">
        <button
          aria-pressed={summary.preset === "earlier"}
          className="explorer-button explorer-button--ghost"
          disabled={disabled}
          onClick={() => onLayerChange(presets.earlier)}
          type="button"
        >
          Earlier
        </button>
        <button
          aria-pressed={summary.preset === "middle"}
          className="explorer-button explorer-button--ghost"
          disabled={disabled}
          onClick={() => onLayerChange(presets.middle)}
          type="button"
        >
          Middle
        </button>
        <button
          aria-pressed={summary.preset === "later"}
          className="explorer-button explorer-button--ghost"
          disabled={disabled}
          onClick={() => onLayerChange(presets.later)}
          type="button"
        >
          Later
        </button>
        <button
          aria-pressed={comparePressed}
          className="explorer-button explorer-button--ghost"
          disabled={disabled}
          onClick={onOpenCompare}
          type="button"
        >
          Compare
        </button>
      </div>

      <p className="attention-controls__hint">
        Each layer is another transformation of the model&apos;s internal representation. Attention
        patterns can change substantially across depth.
      </p>
    </section>
  );
}
