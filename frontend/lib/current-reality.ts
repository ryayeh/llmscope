import type {
  CanonicalPromptToken,
  CanonicalTokenSourceCategory,
  GenerationContextMessage,
} from "@/types/api";

export interface RealityPromptTokenItem {
  canonicalPosition: number;
  decodedContribution: string;
  displayToken: string;
  graphTokenId: string | null;
  id: string;
  rawToken: string;
  sourceCategory: CanonicalTokenSourceCategory;
  sourceLabel: string;
  specialToken: boolean;
  tokenId: number;
}

export interface RealityAssistantTokenItem {
  decodedContribution: string;
  displayToken: string;
  graphTokenId: string;
  id: string;
  rank: number;
  rawToken: string;
  step: number;
  tokenId: number | null;
}

export interface RealityTokenGroup {
  category: CanonicalTokenSourceCategory | "generated_output";
  id: string;
  label: string;
  tokens: Array<RealityPromptTokenItem | RealityAssistantTokenItem>;
}

export interface RealityConversationSection {
  id: string;
  label: string;
  role: "system" | "user" | "assistant";
  text: string;
  tokenIds: string[];
}

function fallbackLabel(category: CanonicalTokenSourceCategory) {
  switch (category) {
    case "system":
      return "System";
    case "user_prompt":
      return "User prompt";
    case "assistant_prefix":
      return "Assistant prefix";
    case "template":
      return "Template / control";
    default:
      return "Generated output";
  }
}

function flattenText(
  tokens: Array<Pick<RealityPromptTokenItem, "decodedContribution"> | Pick<RealityAssistantTokenItem, "decodedContribution">>,
) {
  return tokens.map((token) => token.decodedContribution).join("");
}

export function groupPromptTokens(
  promptTokens: CanonicalPromptToken[],
  promptNodeIdByPosition?: Map<number, string>,
) {
  const groups: RealityTokenGroup[] = [];

  for (const promptToken of promptTokens) {
    const lastGroup = groups[groups.length - 1] ?? null;
    const item: RealityPromptTokenItem = {
      canonicalPosition: promptToken.full_position,
      decodedContribution: promptToken.decoded_contribution,
      displayToken: promptToken.display_token,
      graphTokenId: promptNodeIdByPosition?.get(promptToken.full_position) ?? "root",
      id:
        promptNodeIdByPosition?.get(promptToken.full_position) ??
        `prompt:${promptToken.full_position}`,
      rawToken: promptToken.raw_token,
      sourceCategory: promptToken.source_category,
      sourceLabel: promptToken.source_label || fallbackLabel(promptToken.source_category),
      specialToken: promptToken.special_token,
      tokenId: promptToken.token_id,
    };

    if (lastGroup && lastGroup.category === promptToken.source_category) {
      lastGroup.tokens.push(item);
      continue;
    }

    groups.push({
      category: promptToken.source_category,
      id: `prompt-group:${promptToken.full_position}:${promptToken.source_category}`,
      label: promptToken.source_label || fallbackLabel(promptToken.source_category),
      tokens: [item],
    });
  }

  return groups;
}

export function buildConversationSections(params: {
  assistantTokens: RealityAssistantTokenItem[];
  contextMessages: GenerationContextMessage[];
  promptGroups: RealityTokenGroup[];
}) {
  const systemTokens = params.promptGroups
    .filter((group) => group.category === "system")
    .flatMap((group) => group.tokens as RealityPromptTokenItem[]);
  const userTokens = params.promptGroups
    .filter((group) => group.category === "user_prompt")
    .flatMap((group) => group.tokens as RealityPromptTokenItem[]);
  const assistantText = flattenText(params.assistantTokens);
  const systemMessage = params.contextMessages.find((message) => message.role === "system") ?? null;
  const userMessage = params.contextMessages.find((message) => message.role === "user") ?? null;
  const sections: RealityConversationSection[] = [];

  if (systemMessage || systemTokens.length > 0) {
    sections.push({
      id: "conversation:system",
      label: systemMessage?.label ?? "System",
      role: "system",
      text: systemTokens.length > 0 ? flattenText(systemTokens) : (systemMessage?.content ?? ""),
      tokenIds: systemTokens.map((token) => token.id),
    });
  }

  sections.push({
    id: "conversation:user",
    label: userMessage?.label ?? "User",
    role: "user",
    text: userTokens.length > 0 ? flattenText(userTokens) : (userMessage?.content ?? ""),
    tokenIds: userTokens.map((token) => token.id),
  });

  sections.push({
    id: "conversation:assistant",
    label: "Assistant",
    role: "assistant",
    text: assistantText,
    tokenIds: params.assistantTokens.map((token) => token.id),
  });

  return sections;
}

export function buildFormattingSelectionSummary(
  promptGroups: RealityTokenGroup[],
  selectedTokenId: string | null,
) {
  if (!selectedTokenId) {
    return null;
  }

  for (const group of promptGroups) {
    if (group.category !== "template" && group.category !== "assistant_prefix") {
      continue;
    }

    const selectedToken = group.tokens.find((token) => token.id === selectedTokenId);
    if (!selectedToken) {
      continue;
    }

    return {
      category: group.category,
      label: group.label,
      token: selectedToken,
    };
  }

  return null;
}

export function buildCurrentRealityRawContext(params: {
  assistantTokens: RealityAssistantTokenItem[];
  promptGroups: RealityTokenGroup[];
  rawContextText?: string | null;
}) {
  const promptRawText =
    params.rawContextText ??
    params.promptGroups
      .flatMap((group) => group.tokens as RealityPromptTokenItem[])
      .map((token) => token.rawToken)
      .join("");
  const assistantRawText = params.assistantTokens.map((token) => token.rawToken).join("");
  return `${promptRawText}${assistantRawText}`;
}

export function buildCurrentRealityTokenGroups(params: {
  assistantTokens: RealityAssistantTokenItem[];
  promptGroups: RealityTokenGroup[];
}) {
  if (params.assistantTokens.length === 0) {
    return params.promptGroups;
  }

  return [
    ...params.promptGroups,
    {
      category: "generated_output" as const,
      id: "generated-output",
      label: "Generated output",
      tokens: params.assistantTokens,
    },
  ];
}

export function buildCurrentRealityTokenIdList(params: {
  assistantTokens: RealityAssistantTokenItem[];
  promptGroups: RealityTokenGroup[];
}) {
  return [
    ...params.promptGroups.flatMap((group) =>
      (group.tokens as RealityPromptTokenItem[]).map((token) => String(token.tokenId)),
    ),
    ...params.assistantTokens
      .map((token) => token.tokenId)
      .filter((tokenId): tokenId is number => typeof tokenId === "number")
      .map((tokenId) => String(tokenId)),
  ].join(" ");
}

export function buildBranchBreadcrumb(
  assistantTokens: Array<Pick<RealityAssistantTokenItem, "displayToken" | "rank">>,
) {
  if (assistantTokens.length === 0) {
    return "Awaiting generation";
  }

  const firstAlternativeIndex = assistantTokens.findIndex((token) => token.rank > 1);
  if (firstAlternativeIndex < 0) {
    return "Main response";
  }

  const selectedAlternative = assistantTokens[firstAlternativeIndex];
  const continuedCount = Math.max(assistantTokens.length - firstAlternativeIndex - 1, 0);
  const label = selectedAlternative.displayToken || "alternative";

  if (continuedCount <= 0) {
    return `Main response > alternative "${label}"`;
  }

  return `Main response > alternative "${label}" > continued ${continuedCount} token${
    continuedCount === 1 ? "" : "s"
  }`;
}
