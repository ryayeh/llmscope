import type { ModelOption } from "@/types/api";

export type ProviderSelectionId = ModelOption["provider"];

export interface ProviderModelSelectionState {
  model: string;
  selectedProvider: ProviderSelectionId;
}

export function getModelsForProvider(
  models: ModelOption[],
  provider: ProviderSelectionId,
): ModelOption[] {
  return models.filter((option) => option.provider === provider);
}

export function findCompatibleModelId(
  models: ModelOption[],
  provider: ProviderSelectionId,
  currentModel: string,
): string | null {
  if (models.some((option) => option.provider === provider && option.id === currentModel)) {
    return currentModel;
  }

  return models.find((option) => option.provider === provider)?.id ?? null;
}

export function reconcileProviderModelSelection(
  state: ProviderModelSelectionState,
  models: ModelOption[],
): ProviderModelSelectionState {
  const nextModel = findCompatibleModelId(models, state.selectedProvider, state.model);

  if (!nextModel || nextModel === state.model) {
    return state;
  }

  return {
    ...state,
    model: nextModel,
  };
}
