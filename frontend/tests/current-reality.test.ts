import assert from "node:assert/strict";
import test from "node:test";

import {
  buildBranchBreadcrumb,
  buildConversationSections,
  buildCurrentRealityRawContext,
  buildFormattingSelectionSummary,
  buildCurrentRealityTokenIdList,
  groupPromptTokens,
  type RealityAssistantTokenItem,
} from "../lib/current-reality";
import type { CanonicalPromptToken, GenerationContextMessage } from "../types/api";

const PROMPT_TOKENS: CanonicalPromptToken[] = [
  {
    token_id: 1,
    raw_token: "<|im_start|>",
    display_token: "<|im_start|>",
    decoded_contribution: "<|im_start|>",
    token_bytes: [60],
    full_position: 0,
    source_category: "template",
    source_label: "Template / control",
    special_token: true,
  },
  {
    token_id: 2,
    raw_token: "You are helpful.",
    display_token: "You are helpful.",
    decoded_contribution: "You are helpful.",
    token_bytes: [89],
    full_position: 1,
    source_category: "system",
    source_label: "System message",
    special_token: false,
  },
  {
    token_id: 3,
    raw_token: "\n",
    display_token: "\\n",
    decoded_contribution: "\n",
    token_bytes: [10],
    full_position: 2,
    source_category: "template",
    source_label: "Template / control",
    special_token: true,
  },
  {
    token_id: 4,
    raw_token: "Return only OCEAN.",
    display_token: "Return only OCEAN.",
    decoded_contribution: "Return only OCEAN.",
    token_bytes: [82],
    full_position: 3,
    source_category: "user_prompt",
    source_label: "User prompt",
    special_token: false,
  },
  {
    token_id: 5,
    raw_token: "<|assistant|>",
    display_token: "<|assistant|>",
    decoded_contribution: "<|assistant|>",
    token_bytes: [60],
    full_position: 4,
    source_category: "assistant_prefix",
    source_label: "Assistant prefix",
    special_token: true,
  },
];

const CONTEXT_MESSAGES: GenerationContextMessage[] = [
  {
    role: "system",
    label: "System",
    content: "You are helpful.",
    source: "provider_default",
    editable: false,
  },
  {
    role: "user",
    label: "User",
    content: "Return only OCEAN.",
    source: "user_prompt",
    editable: false,
  },
];

const ASSISTANT_TOKENS: RealityAssistantTokenItem[] = [
  {
    decodedContribution: "O",
    displayToken: "O",
    graphTokenId: "token-1",
    id: "token-1",
    rank: 1,
    rawToken: "O",
    step: 0,
    tokenId: 10,
  },
  {
    decodedContribution: "CEAN",
    displayToken: "CEAN",
    graphTokenId: "token-2",
    id: "token-2",
    rank: 2,
    rawToken: "CEAN",
    step: 1,
    tokenId: 11,
  },
  {
    decodedContribution: "!",
    displayToken: "!",
    graphTokenId: "token-3",
    id: "token-3",
    rank: 1,
    rawToken: "!",
    step: 2,
    tokenId: 12,
  },
];

test("groupPromptTokens preserves canonical grouping and visible-node fallbacks", () => {
  const groups = groupPromptTokens(PROMPT_TOKENS);

  assert.equal(groups.length, 5);
  assert.equal(groups[0]?.category, "template");
  assert.equal(groups[1]?.category, "system");
  assert.equal(groups[3]?.category, "user_prompt");
  assert.equal(groups[4]?.tokens[0]?.graphTokenId, "root");
  assert.equal(groups[4]?.tokens[0]?.id, "prompt:4");
});

test("buildConversationSections reconstructs system, user, and assistant views from canonical sources", () => {
  const promptGroups = groupPromptTokens(PROMPT_TOKENS, new Map([[1, "prompt-1"], [3, "prompt-3"]]));
  const sections = buildConversationSections({
    assistantTokens: ASSISTANT_TOKENS,
    contextMessages: CONTEXT_MESSAGES,
    promptGroups,
  });

  assert.deepEqual(
    sections.map((section) => section.role),
    ["system", "user", "assistant"],
  );
  assert.equal(sections[0]?.text, "You are helpful.");
  assert.equal(sections[1]?.text, "Return only OCEAN.");
  assert.equal(sections[2]?.text, "OCEAN!");
});

test("raw context and token ids stay canonical", () => {
  const promptGroups = groupPromptTokens(PROMPT_TOKENS);
  const rawContext = buildCurrentRealityRawContext({
    assistantTokens: ASSISTANT_TOKENS,
    promptGroups,
    rawContextText: "<|im_start|>You are helpful.\nReturn only OCEAN.<|assistant|>",
  });
  const tokenIds = buildCurrentRealityTokenIdList({
    assistantTokens: ASSISTANT_TOKENS,
    promptGroups,
  });

  assert.equal(
    rawContext,
    "<|im_start|>You are helpful.\nReturn only OCEAN.<|assistant|>OCEAN!",
  );
  assert.equal(tokenIds, "1 2 3 4 5 10 11 12");
});

test("formatting selection and branch breadcrumb stay informative", () => {
  const promptGroups = groupPromptTokens(PROMPT_TOKENS);
  const assistantPrefixSelection = buildFormattingSelectionSummary(promptGroups, "prompt:4");
  const templateSelection = buildFormattingSelectionSummary(promptGroups, "prompt:0");
  const breadcrumb = buildBranchBreadcrumb(ASSISTANT_TOKENS);

  assert.equal(assistantPrefixSelection?.label, "Assistant prefix");
  assert.equal(templateSelection?.label, "Template / control");
  assert.equal(breadcrumb, 'Main response > alternative "CEAN" > continued 1 token');
});
