export interface AlternativeCandidate {
  token: string;
  probability: number;
  log_probability?: number | null;
  entropy?: number | null;
  latency_ms?: number | null;
  token_id?: number | null;
  text_preview?: string | null;
  rationale?: string | null;
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
  token: string;
  token_id: number;
  probability: number;
  log_probability: number;
  entropy: number;
  cumulative_probability: number;
  latency_ms: number;
  position: number;
  text_preview: string;
  alternatives: AlternativeCandidate[];
}

export interface TokenTreeNode {
  id: string;
  token: string;
  token_id: number;
  probability: number;
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
  variation: number;
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
  prompt: string;
  model: string;
  preset: string;
  temperature: number;
  parent_node_id: string;
  parent_token: string;
  parent_text_preview: string;
  depth: number;
  cumulative_probability: number;
  variation: number;
  max_children: number;
}

export interface NodeExpansionCandidate {
  id: string;
  parent_node_id: string;
  token: string;
  token_id: number;
  probability: number;
  log_probability: number;
  entropy: number;
  cumulative_probability: number;
  latency_ms: number;
  depth: number;
  rank: number;
  text_preview: string;
  rationale?: string | null;
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
