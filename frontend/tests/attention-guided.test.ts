import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAllAttentionSourceRows,
  buildAttentionNarrative,
  buildGroupedAttentionSourceRows,
  getAttentionDepthBand,
  getAttentionDepthSummary,
  getRepresentativeAttentionLayers,
  shouldShowAttentionSinkNote,
} from "../lib/attention-guided";
import type { HuggingFaceAttentionResponse } from "../types/api";

function makeResponse(): HuggingFaceAttentionResponse {
  return {
    provider: "hugging_face",
    model_id: "Qwen/Qwen2.5-3B-Instruct",
    model_revision: "rev-3b",
    tokenizer_identity: "Qwen/Qwen2.5-3B-Instruct",
    tokenizer_revision: "tok-rev",
    analysis_mode: "prediction",
    selected_token: {
      token_id: 103,
      raw_token: " telescope",
      display_token: "telescope",
      decoded_contribution: " telescope",
      token_bytes: [32, 116, 101, 108, 101, 115, 99, 111, 112, 101],
      full_position: 4,
      analyzed_position: 4,
      sequence_scope: "generated",
      source_category: "generated_output",
      source_label: "Earlier output",
      special_token: false,
      generated_token_index: 2,
      attention_weight: null,
      is_query: false,
      is_selected_token: true,
    },
    query_token: {
      token_id: 102,
      raw_token: " te",
      display_token: "te",
      decoded_contribution: " te",
      token_bytes: [32, 116, 101],
      full_position: 3,
      analyzed_position: 3,
      sequence_scope: "generated",
      source_category: "generated_output",
      source_label: "Earlier output",
      special_token: false,
      generated_token_index: 1,
      attention_weight: 0.28,
      is_query: true,
      is_selected_token: false,
    },
    analyzed_tokens: [],
    sources: [],
    all_sources: [
      {
        token_id: 201,
        raw_token: " telescope",
        display_token: "telescope",
        decoded_contribution: " telescope",
        token_bytes: [32, 116, 101, 108, 101, 115, 99, 111, 112, 101],
        full_position: 1,
        analyzed_position: 1,
        sequence_scope: "prompt",
        source_category: "user_prompt",
        source_label: "User prompt",
        special_token: false,
        generated_token_index: null,
        attention_weight: 0.18,
        rank: 1,
      },
      {
        token_id: 11,
        raw_token: "<|im_start|>",
        display_token: "<|im_start|>",
        decoded_contribution: "<|im_start|>",
        token_bytes: [60, 124],
        full_position: 0,
        analyzed_position: 0,
        sequence_scope: "prompt",
        source_category: "template",
        source_label: "Template / control",
        special_token: true,
        generated_token_index: null,
        attention_weight: 0.42,
        rank: 2,
      },
      {
        token_id: 102,
        raw_token: " te",
        display_token: "te",
        decoded_contribution: " te",
        token_bytes: [32, 116, 101],
        full_position: 3,
        analyzed_position: 3,
        sequence_scope: "generated",
        source_category: "generated_output",
        source_label: "Earlier output",
        special_token: false,
        generated_token_index: 1,
        attention_weight: 0.11,
        rank: 3,
      },
      {
        token_id: 301,
        raw_token: " answer",
        display_token: "answer",
        decoded_contribution: " answer",
        token_bytes: [32, 97, 110, 115, 119, 101, 114],
        full_position: 2,
        analyzed_position: 2,
        sequence_scope: "prompt",
        source_category: "assistant_prefix",
        source_label: "Assistant prefix",
        special_token: false,
        generated_token_index: null,
        attention_weight: 0.09,
        rank: 4,
      },
    ],
    selected_layer: 18,
    selected_head: null,
    aggregation_mode: "average_heads",
    attention_implementation_used: "eager",
    num_layers: 36,
    num_query_heads: 16,
    selected_token_position: 4,
    query_position: 3,
    selected_token_id: 103,
    query_token_id: 102,
    prompt_token_count: 2,
    generated_token_index: 2,
    sequence_length: 5,
    layer_index: 18,
    head_index: null,
    average_heads: true,
    source_positions: [1, 0, 3, 2],
    attention_weights: [0.18, 0.42, 0.11, 0.09],
    attention_mass_sum: 0.8,
    top_n_coverage: 0.38,
    truncated_context: false,
    context_truncated: false,
    original_full_context_length: 5,
    analyzed_context_length: 5,
    category_breakdown: {
      input_context: 0.69,
      earlier_output: 0.11,
      system_message: 0,
      user_prompt: 0.18,
      assistant_prefix: 0.09,
      template_control: 0.42,
      exclusive_total: 0.8,
    },
    comparison_layers: [],
    layer_journey: null,
  };
}

test("representative attention layers adapt to the model depth", () => {
  assert.deepEqual(getRepresentativeAttentionLayers(1), {
    earlier: 0,
    middle: 0,
    later: 0,
  });
  assert.deepEqual(getRepresentativeAttentionLayers(3), {
    earlier: 0,
    middle: 1,
    later: 2,
  });
  assert.deepEqual(getRepresentativeAttentionLayers(36), {
    earlier: 8,
    middle: 18,
    later: 30,
  });
});

test("depth bands are navigational and derived from layer position", () => {
  assert.equal(getAttentionDepthBand(0, 36), "earlier");
  assert.equal(getAttentionDepthBand(18, 36), "middle");
  assert.equal(getAttentionDepthBand(33, 36), "later");

  const summary = getAttentionDepthSummary(18, 36);
  assert.equal(summary.label, "Middle depth");
  assert.equal(summary.percentThrough, 51);
});

test("grouped source rows preserve template mass without renormalizing", () => {
  const analysis = makeResponse();
  const rows = buildGroupedAttentionSourceRows(analysis, 3);
  const templateRow = rows.find((row) => row.rowKind === "group");
  const total = rows.reduce((sum, row) => sum + row.weight, 0);

  assert.ok(templateRow);
  assert.equal(templateRow?.label, "Chat formatting");
  assert.equal(templateRow?.members.length, 1);
  assert.equal(templateRow?.weight, 0.42);
  assert.equal(rows[0]?.rowKind, "group");
  assert.ok(Math.abs(total - 0.8) < 1e-9);
});

test("all-token source rows remain exact and sortable by canonical position", () => {
  const analysis = makeResponse();
  const byWeight = buildAllAttentionSourceRows(analysis, "weight");
  const byPosition = buildAllAttentionSourceRows(analysis, "position");

  assert.deepEqual(
    byWeight.map((source) => source.full_position),
    [0, 1, 3, 2],
  );
  assert.deepEqual(
    byPosition.map((source) => source.full_position),
    [0, 1, 2, 3],
  );
});

test("attention sink note appears only when template mass is materially high", () => {
  const analysis = makeResponse();

  assert.equal(shouldShowAttentionSinkNote(analysis.category_breakdown), true);
  assert.equal(
    shouldShowAttentionSinkNote({
      ...analysis.category_breakdown,
      template_control: 0.12,
    }),
    false,
  );
});

test("guided narrative is deterministic and grounded in exact values", () => {
  const analysis = makeResponse();
  const rows = buildGroupedAttentionSourceRows(analysis, 3);
  const narrative = buildAttentionNarrative(
    analysis,
    getAttentionDepthSummary(18, 36),
    rows,
  );

  assert.equal(narrative.headline, 'Predicting "telescope"');
  assert.match(narrative.preface, /previous token here is "te"/);
  assert.equal(narrative.depthLine, "Middle depth · Layer 18 of 36");
  assert.equal(narrative.bullets.length, 3);
  assert.match(narrative.note, /model signal/);
});
