export function shouldReuseContinuationTarget(params: {
  forceRequestFirstStep?: boolean;
  hasExistingTarget: boolean;
  stepIndex: number;
}) {
  if (!params.hasExistingTarget) {
    return false;
  }

  if (params.forceRequestFirstStep && params.stepIndex === 0) {
    return false;
  }

  return true;
}
