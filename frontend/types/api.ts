export interface ApiErrorDetail {
  code: string;
  message: string;
}

export type ContinuationMode = "exact" | "approximate";
export type CanonicalTokenSourceCategory =
  | "system"
  | "user_prompt"
  | "template"
  | "assistant_prefix"
  | "generated_output";
export type HuggingFaceAttentionAnalysisMode = "prediction" | "representation";
export type HuggingFaceAttentionAggregationMode = "single_head" | "average_heads" | "max_heads";
export type HuggingFaceAttentionSequenceScope = "prompt" | "generated";

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

export interface CanonicalPromptToken {
  token_id: number;
  raw_token: string;
  display_token: string;
  decoded_contribution: string;
  token_bytes: number[];
  full_position: number;
  source_category: CanonicalTokenSourceCategory;
  source_label: string;
  special_token: boolean;
}

export interface GenerationContextMessage {
  role: "system" | "user" | "assistant";
  label: string;
  content: string;
  source:
    | "provider_default"
    | "composed_instructions"
    | "user_prompt"
    | "assistant_prefix";
  editable: boolean;
}

export interface GenerationResponse {
  mode: string;
  prompt_used: string;
  prompt_token_ids?: number[] | null;
  prompt_tokens?: CanonicalPromptToken[] | null;
  context_messages?: GenerationContextMessage[];
  raw_context_text?: string | null;
  system_prompt_editable?: boolean;
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

export interface HuggingFaceAttentionRequest {
  model_id: string;
  model_revision?: string | null;
  tokenizer_identity?: string | null;
  tokenizer_revision?: string | null;
  prompt_token_ids: number[];
  prompt_tokens: CanonicalPromptToken[];
  generated_token_ids: number[];
  selected_generated_token_index: number;
  selected_layer: number;
  selected_head?: number | null;
  analysis_mode: HuggingFaceAttentionAnalysisMode;
  aggregation_mode: HuggingFaceAttentionAggregationMode;
  max_connections: number;
  max_context_tokens: number;
  allow_truncated_recompute: boolean;
}

export interface HuggingFaceAttentionTokenInfo {
  token_id: number;
  raw_token: string;
  display_token: string;
  decoded_contribution: string;
  token_bytes: number[];
  full_position: number;
  analyzed_position: number;
  sequence_scope: HuggingFaceAttentionSequenceScope;
  source_category: CanonicalTokenSourceCategory;
  source_label: string;
  special_token: boolean;
  generated_token_index?: number | null;
  attention_weight?: number | null;
  is_query: boolean;
  is_selected_token: boolean;
}

export interface HuggingFaceAttentionSource {
  token_id: number;
  raw_token: string;
  display_token: string;
  decoded_contribution: string;
  token_bytes: number[];
  full_position: number;
  analyzed_position: number;
  sequence_scope: HuggingFaceAttentionSequenceScope;
  source_category: CanonicalTokenSourceCategory;
  source_label: string;
  special_token: boolean;
  generated_token_index?: number | null;
  attention_weight: number;
  rank: number;
}

export interface HuggingFaceAttentionResponse {
  provider: "hugging_face";
  model_id: string;
  model_revision?: string | null;
  tokenizer_identity?: string | null;
  tokenizer_revision?: string | null;
  analysis_mode: HuggingFaceAttentionAnalysisMode;
  selected_token: HuggingFaceAttentionTokenInfo;
  query_token: HuggingFaceAttentionTokenInfo;
  analyzed_tokens: HuggingFaceAttentionTokenInfo[];
  sources: HuggingFaceAttentionSource[];
  selected_layer: number;
  selected_head?: number | null;
  aggregation_mode: HuggingFaceAttentionAggregationMode;
  attention_implementation_used: string;
  num_layers: number;
  num_query_heads: number;
  selected_token_position: number;
  query_position: number;
  selected_token_id: number;
  query_token_id: number;
  prompt_token_count: number;
  generated_token_index: number;
  sequence_length: number;
  layer_index: number;
  head_index?: number | null;
  average_heads: boolean;
  source_positions: number[];
  attention_weights: number[];
  attention_mass_sum: number;
  top_n_coverage: number;
  truncated_context: boolean;
  context_truncated: boolean;
  original_full_context_length: number;
  analyzed_context_length: number;
}

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
  active_model_num_hidden_layers?: number | null;
  active_model_num_attention_heads?: number | null;
  active_model_attention_implementation?: string | null;
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
