import type {
  HuggingFaceAttentionCategoryBreakdown,
  HuggingFaceAttentionLayerJourney,
  HuggingFaceAttentionLayerSummary,
  HuggingFaceAttentionResponse,
  HuggingFaceAttentionSource,
} from "@/types/api";

export type AttentionDepthBand = "earlier" | "middle" | "later";
export type AttentionDepthPresetId = AttentionDepthBand | "custom";
export type AttentionSourceView = "grouped" | "all_tokens";
export type AttentionSourceSort = "weight" | "position";

export interface AttentionDepthPresets {
  earlier: number;
  middle: number;
  later: number;
}

export interface AttentionDepthSummary {
  band: AttentionDepthBand;
  label: string;
  layerIndex: number;
  layerCount: number;
  percentThrough: number;
  preset: AttentionDepthPresetId;
}

export interface GroupedAttentionSourceRow {
  id: string;
  label: string;
  members: HuggingFaceAttentionSource[];
  meta: string;
  rowKind: "group" | "source";
  source: HuggingFaceAttentionSource | null;
  weight: number;
}

export interface AttentionNarrative {
  bullets: string[];
  depthLine: string;
  headline: string;
  note: string;
  preface: string;
}

const ATTENTION_SINK_THRESHOLD = 0.35;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function formatPercent(value: number) {
  return `${(Math.max(value, 0) * 100).toFixed(1)}%`;
}

function getDisplayToken(source: { display_token: string; raw_token: string }) {
  return source.display_token || source.raw_token;
}

export function getAttentionSourceCategoryLabel(source: HuggingFaceAttentionSource) {
  switch (source.source_category) {
    case "system":
      return "system message";
    case "user_prompt":
      return "user prompt";
    case "assistant_prefix":
      return "assistant prefix";
    case "generated_output":
      return "earlier output";
    default:
      return "chat formatting";
  }
}

export function getRepresentativeAttentionLayers(layerCount: number): AttentionDepthPresets {
  if (layerCount <= 1) {
    return { earlier: 0, middle: 0, later: 0 };
  }

  const lastLayer = layerCount - 1;
  return {
    earlier: clamp(Math.floor(lastLayer * 0.25), 0, lastLayer),
    middle: clamp(Math.round(lastLayer * 0.5), 0, lastLayer),
    later: clamp(Math.ceil(lastLayer * 0.85), 0, lastLayer),
  };
}

export function getAttentionDepthBand(layerIndex: number, layerCount: number): AttentionDepthBand {
  if (layerCount <= 1) {
    return "middle";
  }

  const ratio = layerIndex / Math.max(layerCount - 1, 1);
  if (ratio >= 0.67) {
    return "later";
  }
  if (ratio <= 0.33) {
    return "earlier";
  }
  return "middle";
}

export function getAttentionPresetForLayer(
  layerIndex: number,
  layerCount: number,
): AttentionDepthPresetId {
  const presets = getRepresentativeAttentionLayers(layerCount);
  if (layerIndex === presets.earlier) {
    return "earlier";
  }
  if (layerIndex === presets.middle) {
    return "middle";
  }
  if (layerIndex === presets.later) {
    return "later";
  }
  return "custom";
}

export function getAttentionDepthSummary(
  layerIndex: number,
  layerCount: number,
): AttentionDepthSummary {
  const band = getAttentionDepthBand(layerIndex, layerCount);
  const percentThrough =
    layerCount <= 1 ? 100 : Math.round((layerIndex / Math.max(layerCount - 1, 1)) * 100);

  return {
    band,
    label: band === "earlier" ? "Earlier depth" : band === "later" ? "Later depth" : "Middle depth",
    layerIndex,
    layerCount,
    percentThrough,
    preset: getAttentionPresetForLayer(layerIndex, layerCount),
  };
}

export function getAttentionJourneyLayers(layerCount: number) {
  return Array.from({ length: Math.max(layerCount, 0) }, (_, index) => index);
}

export function buildGroupedAttentionSourceRows(
  analysis: HuggingFaceAttentionResponse,
  topMeaningfulCount: number,
): GroupedAttentionSourceRow[] {
  const meaningfulSources = analysis.all_sources.filter(
    (source) => source.source_category !== "template",
  );
  const templateSources = analysis.all_sources.filter(
    (source) => source.source_category === "template",
  );
  const rows: GroupedAttentionSourceRow[] = meaningfulSources
    .slice(0, Math.max(topMeaningfulCount, 1))
    .map((source) => ({
      id: `source:${source.full_position}`,
      label: getDisplayToken(source),
      members: [source],
      meta: `${getAttentionSourceCategoryLabel(source)} · pos ${source.full_position}`,
      rowKind: "source",
      source,
      weight: source.attention_weight,
    }));

  if (templateSources.length > 0) {
    const templateMass = templateSources.reduce(
      (total, source) => total + source.attention_weight,
      0,
    );
    rows.push({
      id: "template-group",
      label: "Chat formatting",
      members: templateSources,
      meta: `${templateSources.length} template/control tokens`,
      rowKind: "group",
      source: null,
      weight: templateMass,
    });
  }

  rows.sort((left, right) => right.weight - left.weight || left.label.localeCompare(right.label));
  return rows;
}

export function buildAllAttentionSourceRows(
  analysis: HuggingFaceAttentionResponse,
  sortMode: AttentionSourceSort,
) {
  const rows = [...analysis.all_sources];
  if (sortMode === "position") {
    rows.sort((left, right) => left.full_position - right.full_position);
    return rows;
  }

  rows.sort(
    (left, right) =>
      right.attention_weight - left.attention_weight ||
      left.full_position - right.full_position,
  );
  return rows;
}

export function buildAttentionCategoryRows(breakdown: HuggingFaceAttentionCategoryBreakdown) {
  return [
    { label: "Input context", value: breakdown.input_context },
    { label: "Earlier output", value: breakdown.earlier_output },
    { label: "System message", value: breakdown.system_message },
    { label: "User prompt", value: breakdown.user_prompt },
    { label: "Assistant prefix", value: breakdown.assistant_prefix },
    { label: "Template/control", value: breakdown.template_control },
  ];
}

export function shouldShowAttentionSinkNote(
  breakdown: HuggingFaceAttentionCategoryBreakdown,
) {
  return breakdown.template_control >= ATTENTION_SINK_THRESHOLD;
}

export function buildAttentionNarrative(
  analysis: HuggingFaceAttentionResponse,
  depthSummary: AttentionDepthSummary,
  groupedRows: GroupedAttentionSourceRow[],
): AttentionNarrative {
  const selectedLabel =
    analysis.selected_token.display_token || analysis.selected_token.raw_token || "token";
  const headline =
    analysis.analysis_mode === "representation"
      ? `Representing "${selectedLabel}"`
      : `Predicting "${selectedLabel}"`;
  const preface =
    analysis.analysis_mode === "representation"
      ? "Representation mode inspects the selected token after it is present in the sequence. It differs from the attention used to predict it."
      : analysis.generated_token_index === 0
        ? "Because this is the first output token, prediction attention uses the final prompt/chat-template position."
        : `Prediction attention uses the position immediately before this token. The previous token here is "${analysis.query_token.display_token || analysis.query_token.raw_token}".`;

  const bullets = groupedRows.slice(0, 3).map((row) => {
    if (row.rowKind === "group") {
      return `${row.label.toLowerCase()} received ${formatPercent(row.weight)}.`;
    }

    return `"${row.label}" in the ${getAttentionSourceCategoryLabel(row.source!)} received ${formatPercent(row.weight)}.`;
  });

  return {
    bullets,
    depthLine: `${depthSummary.label} · Layer ${depthSummary.layerIndex} of ${depthSummary.layerCount}`,
    headline,
    note:
      "Attention shows where this layer gathered information. It is a model signal, not a complete explanation or proof of reasoning.",
    preface,
  };
}

export function buildAttentionCopyPayload(
  analysis: HuggingFaceAttentionResponse,
  metadata: {
    layerSummary: AttentionDepthSummary;
    sourceView: AttentionSourceView;
    sourceSort: AttentionSourceSort;
  },
) {
  return JSON.stringify(
    {
      analysis,
      metadata: {
        layer_summary: metadata.layerSummary,
        source_sort: metadata.sourceSort,
        source_view: metadata.sourceView,
      },
    },
    null,
    2,
  );
}

export function getComparisonLayerMap(summaries: HuggingFaceAttentionLayerSummary[]) {
  return new Map(summaries.map((summary) => [summary.layer_index, summary]));
}

export function getJourneyScale(journey: HuggingFaceAttentionLayerJourney | null | undefined) {
  return journey?.scale_max ?? 0;
}
