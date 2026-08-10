import assert from "node:assert/strict";
import test from "node:test";

import { shouldReuseContinuationTarget } from "../lib/continuation-flow";

test("shouldReuseContinuationTarget forces a real first-step request for modal continuation", () => {
  assert.equal(
    shouldReuseContinuationTarget({
      forceRequestFirstStep: true,
      hasExistingTarget: true,
      stepIndex: 0,
    }),
    false,
  );
});

test("shouldReuseContinuationTarget allows later-step reuse after the first request", () => {
  assert.equal(
    shouldReuseContinuationTarget({
      forceRequestFirstStep: true,
      hasExistingTarget: true,
      stepIndex: 1,
    }),
    true,
  );
});

test("shouldReuseContinuationTarget still reuses existing targets for normal graph traversal", () => {
  assert.equal(
    shouldReuseContinuationTarget({
      forceRequestFirstStep: false,
      hasExistingTarget: true,
      stepIndex: 0,
    }),
    true,
  );
  assert.equal(
    shouldReuseContinuationTarget({
      forceRequestFirstStep: false,
      hasExistingTarget: false,
      stepIndex: 0,
    }),
    false,
  );
});
