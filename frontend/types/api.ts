export interface ApiErrorDetail {
  code: string;
  message: string;
}

export type ContinuationMode = "exact" | "approximate";

export interface ProviderCapabilitiesDetail {
  supports_logprobs: boolean;
  supports_entropy: boolean;
  supports_attention: boolean;
  supports_exact_continuation: boolean;
  supports_streaming: boolean;
  supports_branching: boolean;
  supports_continuation: boolean;
  minimum_output_tokens: number;
}

export interface AlternativeCandidate {
  node_id?: string | null;
  segment_id?: string | null;
  token: string;
  display_token?: string | null;
  token_bytes?: number[] | null;
  decoded_contribution?: string | null;
  cumulative_decoded_text?: string | null;
  cumulative_token_ids?: number[] | null;
  cumulative_log_probability?: number | null;
  probability?: number | null;
  raw_probability?: number | null;
  normalized_displayed_probability?: number | null;
  log_probability?: number | null;
  entropy?: number | null;
  latency_ms?: number | null;
  token_id?: number | null;
  tokenizer_id?: number | null;
  rank?: number | null;
  text_preview?: string | null;
  context_before?: string | null;
  context_after?: string | null;
  finish_reason?: string | null;
  rationale?: string | null;
  generation_step?: number | null;
  continuation_mode?: ContinuationMode | null;
  metadata?: Record<string, string | number | boolean | null> | null;
}

export interface ModelOption {
  id: string;
  label: string;
  provider: string;
  group: string;
  status: string;
  capabilities: ProviderCapabilitiesDetail;
}

export interface ProviderOption {
  id: string;
  label: string;
  status: string;
  status_message?: string | null;
  recommended_models: string[];
  capabilities: ProviderCapabilitiesDetail;
}

export interface PresetOption {
  id: string;
  label: string;
}

export interface ModelCatalogResponse {
  default_provider: string;
  default_model: string;
  default_preset: string;
  providers: ProviderOption[];
  models: ModelOption[];
  presets: PresetOption[];
}

export interface TokenTrace {
  id: string;
  segment_id?: string | null;
  branch_id: string;
  parent_node_id?: string | null;
  model: string;
  source: "openai" | "ollama" | "hugging_face" | "demo";
  index: number;
  position: number;
  token: string;
  display_token: string;
  token_bytes?: number[] | null;
  decoded_contribution?: string | null;
  cumulative_decoded_text?: string | null;
  cumulative_token_ids?: number[] | null;
  cumulative_log_probability?: number | null;
  token_id?: number | null;
  tokenizer_id?: number | null;
  probability?: number | null;
  raw_probability?: number | null;
  normalized_displayed_probability?: number | null;
  log_probability?: number | null;
  entropy?: number | null;
  cumulative_probability?: number | null;
  latency_ms: number;
  text_preview: string;
  context_before: string;
  context_after: string;
  finish_reason?: string | null;
  alternatives: AlternativeCandidate[];
  generation_step?: number | null;
  continuation_mode?: ContinuationMode | null;
  metadata?: Record<string, string | number | boolean | null> | null;
}

export interface TokenTreeNode {
  id: string;
  token: string;
  display_token?: string | null;
  token_id?: number | null;
  tokenizer_id?: number | null;
  probability?: number | null;
  raw_probability?: number | null;
  normalized_displayed_probability?: number | null;
  log_probability?: number | null;
  entropy?: number | null;
  cumulative_probability?: number | null;
  latency_ms: number;
  depth: number;
  rank: number;
  text_preview: string;
  is_selected_path: boolean;
  children: TokenTreeNode[];
}

export interface TreeSummary {
  max_depth: number;
  branch_width: number;
  total_nodes: number;
  explored_paths: number;
  selected_path_depth: number;
}

export interface PromptInsights {
  detected_intent: string;
  focus_terms: string[];
  response_strategy: string;
  suggested_follow_ups: string[];
}

export interface RequestEcho {
  prompt: string;
  provider?: string | null;
  model: string;
  preset: string;
  max_tokens: number;
  temperature: number;
  top_p: number;
  variation: number;
  demo_mode: boolean;
}

export interface GenerationStats {
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms: number;
  estimated_cost_usd: number;
  generated_at: string;
}

export interface GenerationResponse {
  mode: string;
  prompt_used: string;
  prompt_token_ids?: number[] | null;
  completion: string;
  notes: string;
  request: RequestEcho;
  insights: PromptInsights;
  tokens: TokenTrace[];
  tree: TokenTreeNode;
  tree_summary: TreeSummary;
  stats: GenerationStats;
  provider_capabilities: ProviderCapabilitiesDetail;
}

export interface NodeExpansionRequest {
  root_prompt: string;
  provider?: string | null;
  model: string;
  preset: string;
  temperature: number;
  top_p: number;
  parent_node_id: string;
  parent_token: string;
  assistant_prefix: string;
  prompt_token_ids?: number[] | null;
  canonical_prefix_token_ids?: number[] | null;
  generated_prefix_token_ids?: number[] | null;
  reconstructed_prompt: string;
  expected_prompt_length?: number | null;
  expected_utf8_length?: number | null;
  expected_assistant_prefix_length?: number | null;
  expected_assistant_prefix_utf8_length?: number | null;
  expected_token_count?: number | null;
  selected_token_id?: number | null;
  selected_tokenizer_id?: number | null;
  model_revision?: string | null;
  tokenizer_identity?: string | null;
  tokenizer_revision?: string | null;
  depth: number;
  cumulative_probability: number;
  variation: number;
  max_children: number;
  demo_mode: boolean;
}

export interface ContinueGenerationRequest extends NodeExpansionRequest {
  cached_segment_id?: string | null;
  cached_token_index?: number | null;
}

export interface NodeExpansionCandidate {
  id: string;
  segment_id?: string | null;
  branch_id: string;
  parent_node_id: string;
  model: string;
  source: "openai" | "ollama" | "hugging_face" | "demo";
  token: string;
  display_token: string;
  token_bytes?: number[] | null;
  decoded_contribution?: string | null;
  cumulative_decoded_text?: string | null;
  cumulative_token_ids?: number[] | null;
  cumulative_log_probability?: number | null;
  token_id?: number | null;
  tokenizer_id?: number | null;
  probability: number;
  raw_probability: number;
  normalized_displayed_probability: number;
  log_probability: number;
  entropy: number;
  cumulative_probability: number;
  latency_ms: number;
  depth: number;
  rank: number;
  text_preview: string;
  context_before: string;
  context_after: string;
  finish_reason?: string | null;
  rationale?: string | null;
  generation_step?: number | null;
  continuation_mode?: ContinuationMode | null;
  metadata?: Record<string, string | number | boolean | null> | null;
}

export interface NodeExpansionResponse {
  mode: string;
  parent_node_id: string;
  children: NodeExpansionCandidate[];
  entropy: number;
  expanded_at: string;
  notes: string;
}

export interface ContinueGenerationResponse extends NodeExpansionResponse {
  action: "reveal_cached" | "new_provider_segment";
  continuation_mode: ContinuationMode;
  provider_capabilities: ProviderCapabilitiesDetail;
  segment_id?: string | null;
  revealed_count: number;
  cached_token_count: number;
  remaining_cached_tokens: number;
}

export type HuggingFaceLocalState =
  | "not_downloaded"
  | "downloading"
  | "loading"
  | "ready"
  | "busy"
  | "cuda_unavailable"
  | "oom"
  | "error";

export interface HuggingFaceLocalModelStatus {
  id: string;
  label: string;
  revision?: string | null;
  resolved_revision?: string | null;
  status: HuggingFaceLocalState;
  status_message?: string | null;
  downloaded: boolean;
  loaded: boolean;
  recommended: boolean;
}

export interface HuggingFaceLocalLimits {
  context_window_tokens: number;
  default_output_tokens: number;
  max_output_tokens: number;
  stored_top_alternatives: number;
}

export interface HuggingFaceLocalStatusResponse {
  provider: string;
  label: string;
  status: HuggingFaceLocalState;
  status_message?: string | null;
  capabilities: ProviderCapabilitiesDetail;
  cuda_available: boolean;
  busy: boolean;
  device?: string | null;
  precision?: string | null;
  torch_version?: string | null;
  transformers_version?: string | null;
  gpu_name?: string | null;
  gpu_total_vram_gb?: number | null;
  gpu_free_vram_gb?: number | null;
  active_model_id?: string | null;
  active_model_label?: string | null;
  active_model_revision?: string | null;
  active_model_resolved_revision?: string | null;
  recommended_model_id?: string | null;
  missing_dependencies: string[];
  limits: HuggingFaceLocalLimits;
  models: HuggingFaceLocalModelStatus[];
}

export interface HuggingFaceLocalLoadRequest {
  model_id: string;
}

export interface HuggingFaceLocalDiagnosticsResponse {
  cuda_available: boolean;
  selected_device?: string | null;
  selected_dtype?: string | null;
  torch_version?: string | null;
  transformers_version?: string | null;
  torch_cuda_runtime?: string | null;
  gpu_name?: string | null;
  gpu_total_vram_gb?: number | null;
  gpu_free_vram_gb?: number | null;
  python_version: string;
  platform: string;
  disk_free_gb?: number | null;
  missing_dependencies: string[];
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
}
