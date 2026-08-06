import assert from "node:assert/strict";
import test from "node:test";

import {
  getContinuationModePresentation,
  isApproximateBoundary,
  resolveContinuationMode,
} from "../lib/continuation-mode";

test("Exact continuation presents the exact badge and tooltip", () => {
  const presentation = getContinuationModePresentation({
    continuationMode: "exact",
    metadata: null,
  });

  assert.equal(presentation.label, "Exact");
  assert.equal(presentation.mode, "exact");
  assert.equal(presentation.tone, "exact");
  assert.match(presentation.title, /uninterrupted provider generation/i);
});

test("Approximate continuation presents the approximate badge and tooltip", () => {
  const presentation = getContinuationModePresentation({
    continuationMode: "approximate",
    metadata: null,
  });

  assert.equal(presentation.label, "Approximate");
  assert.equal(presentation.mode, "approximate");
  assert.equal(presentation.tone, "approximate");
  assert.match(presentation.title, /regenerated continuation/i);
});

test("Approximate boundaries are only flagged when the graph crosses from exact to approximate", () => {
  assert.equal(isApproximateBoundary("exact", "approximate"), true);
  assert.equal(isApproximateBoundary("approximate", "approximate"), false);
  assert.equal(isApproximateBoundary("approximate", "exact"), false);
  assert.equal(resolveContinuationMode(undefined, { continuation_mode: "approximate" }), "approximate");
});
