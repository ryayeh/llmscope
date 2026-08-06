import type { ContinuationMode } from "@/types/api";

type ContinuationMetadata = Record<string, string | number | boolean | null> | null | undefined;

export interface ContinuationPresentation {
  label: string;
  mode: ContinuationMode;
  title: string;
  tone: "exact" | "approximate";
}

const EXACT_TOOLTIP = "Probabilities come directly from one uninterrupted provider generation.";
const APPROXIMATE_TOOLTIP =
  "Probabilities come from a regenerated continuation because this provider cannot resume an unfinished assistant generation.";

function readMetadataString(metadata: ContinuationMetadata, key: string) {
  return typeof metadata?.[key] === "string" ? String(metadata[key]) : null;
}

export function resolveContinuationMode(
  continuationMode: ContinuationMode | null | undefined,
  metadata?: ContinuationMetadata,
): ContinuationMode {
  if (continuationMode) {
    return continuationMode;
  }

  const explicitMode = readMetadataString(metadata, "continuation_mode");
  return explicitMode === "approximate" ? "approximate" : "exact";
}

export function getContinuationModePresentation(args: {
  continuationMode: ContinuationMode | null | undefined;
  metadata?: ContinuationMetadata;
}): ContinuationPresentation {
  const mode = resolveContinuationMode(args.continuationMode, args.metadata);
  const explicitLabel = readMetadataString(args.metadata, "continuation_mode_label");
  const explicitTitle = readMetadataString(args.metadata, "continuation_mode_tooltip");

  return {
    label: explicitLabel ?? (mode === "exact" ? "Exact" : "Approximate"),
    mode,
    title: explicitTitle ?? (mode === "exact" ? EXACT_TOOLTIP : APPROXIMATE_TOOLTIP),
    tone: mode === "exact" ? "exact" : "approximate",
  };
}

export function isApproximateBoundary(
  sourceMode: ContinuationMode | null | undefined,
  targetMode: ContinuationMode | null | undefined,
) {
  return resolveContinuationMode(sourceMode) !== resolveContinuationMode(targetMode) &&
    resolveContinuationMode(targetMode) === "approximate";
}
