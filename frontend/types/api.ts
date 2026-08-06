export interface ApiErrorDetail {
  code: string;
  message: string;
}

export interface AlternativeCandidate {
  node_id?: string | null;
  token: string;
  display_token?: string | null;
  token_bytes?: number[] | null;
  decoded_contribution?: string | null;
  cumulative_decoded_text?: string | null;
  cumulative_token_ids?: number[] | null;
  cumulative_log_probability?: number | null;
  probability: number;
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
  metadata?: Record<string, string | number | boolean | null> | null;
}

export interface ModelOption {
  id: string;
  label: string;
  provider: string;
  group: string;
  status: string;
}

export interface PresetOption {
  id: string;
  label: string;
}

export interface ModelCatalogResponse {
  default_model: string;
  default_preset: string;
  models: ModelOption[];
  presets: PresetOption[];
}

export interface TokenTrace {
  id: string;
  branch_id: string;
  parent_node_id?: string | null;
  model: string;
  source: "openai" | "demo";
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
  probability: number;
  raw_probability: number;
  normalized_displayed_probability: number;
  log_probability: number;
  entropy: number;
  cumulative_probability: number;
  latency_ms: number;
  text_preview: string;
  context_before: string;
  context_after: string;
  finish_reason?: string | null;
  alternatives: AlternativeCandidate[];
  generation_step?: number | null;
  metadata?: Record<string, string | number | boolean | null> | null;
}

export interface TokenTreeNode {
  id: string;
  token: string;
  display_token?: string | null;
  token_id?: number | null;
  tokenizer_id?: number | null;
  probability: number;
  raw_probability?: number | null;
  normalized_displayed_probability?: number | null;
  log_probability: number;
  entropy: number;
  cumulative_probability: number;
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
  completion: string;
  notes: string;
  request: RequestEcho;
  insights: PromptInsights;
  tokens: TokenTrace[];
  tree: TokenTreeNode;
  tree_summary: TreeSummary;
  stats: GenerationStats;
}

export interface NodeExpansionRequest {
  root_prompt: string;
  model: string;
  preset: string;
  temperature: number;
  top_p: number;
  parent_node_id: string;
  parent_token: string;
  assistant_prefix: string;
  reconstructed_prompt: string;
  expected_prompt_length?: number | null;
  expected_utf8_length?: number | null;
  expected_token_count?: number | null;
  depth: number;
  cumulative_probability: number;
  variation: number;
  max_children: number;
  demo_mode: boolean;
}

export interface NodeExpansionCandidate {
  id: string;
  branch_id: string;
  parent_node_id: string;
  model: string;
  source: "openai" | "demo";
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

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  timestamp: string;
}
