/**
 * TypeScript Data Contracts for Adaptive Threat Intelligence NIDS.
 * Exactly mirrors backend Pydantic schemas in backend/app/schemas/flow.py.
 */

export interface NetworkFlowRecord {
  // Categorical features (3)
  proto: string;
  service: string;
  state: string;

  // Numerical flow statistics (39)
  dur: number;
  spkts: number;
  dpkts: number;
  sbytes: number;
  dbytes: number;
  rate: number;
  sttl?: number;
  dttl: number;
  sload: number;
  dload: number;
  sloss: number;
  dloss: number;
  sinpkt: number;
  dinpkt: number;
  sjit: number;
  djit: number;
  swin: number;
  stcpb: number;
  dtcpb: number;
  dwin: number;
  tcprtt: number;
  synack: number;
  ackdat: number;
  smean: number;
  dmean: number;
  trans_depth: number;
  response_body_len: number;
  ct_srv_src: number;
  ct_state_ttl: number;
  ct_dst_ltm: number;
  ct_src_dport_ltm: number;
  ct_dst_sport_ltm: number;
  ct_dst_src_ltm: number;
  is_ftp_login: number;
  ct_ftp_cmd: number;
  ct_flw_http_mthd: number;
  ct_src_ltm: number;
  ct_srv_dst: number;
  is_sm_ips_ports: number;
}

// ============================================================================
// Binary Prediction
// ============================================================================

export type BinaryThresholdPreset =
  | "default"
  | "f1_optimal"
  | "cost_optimal"
  | "low_false_alarm"
  | "high_recall";

export interface BinaryPredictionRequest {
  flows: NetworkFlowRecord[];
  threshold_preset?: BinaryThresholdPreset | string;
}

export interface BinaryPredictionItem {
  index: number;
  label: 0 | 1;
  label_name: "Normal" | "Attack" | string;
  attack_probability: number;
  threshold_applied: number;
}

export interface BinaryPredictionResponse {
  model_id: string;
  model_name: string;
  threshold_preset: string;
  threshold_applied: number;
  predictions: BinaryPredictionItem[];
  flow_count: number;
  processing_time_ms: number;
}

// ============================================================================
// Multiclass Classification
// ============================================================================

export type MulticlassModelId = "c8" | "c10";

export type AttackFamily =
  | "Normal"
  | "Generic"
  | "Exploits"
  | "Fuzzers"
  | "DoS"
  | "Reconnaissance"
  | "Analysis"
  | "Backdoor"
  | "Shellcode"
  | "Worms";

export interface MulticlassPredictionRequest {
  flows: NetworkFlowRecord[];
  model_id?: MulticlassModelId | string;
}

export interface MulticlassPredictionItem {
  index: number;
  predicted_class_id: number;
  predicted_class_name: AttackFamily | string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface MulticlassPredictionResponse {
  model_id: string;
  model_name: string;
  schema_used: "primary" | "alternative_no_sttl" | string;
  predictions: MulticlassPredictionItem[];
  flow_count: number;
  processing_time_ms: number;
}

// ============================================================================
// Model Comparison (C8 vs C10)
// ============================================================================

export interface ModelComparisonRequest {
  flows: NetworkFlowRecord[];
}

export interface ModelComparisonItem {
  index: number;
  c8_predicted_class: string;
  c8_confidence: number;
  c10_predicted_class: string;
  c10_confidence: number;
  agreement: boolean;
  l1_distribution_divergence: number;
}

export interface ModelComparisonResponse {
  total_flows: number;
  agreement_count: number;
  disagreement_count: number;
  agreement_rate: number;
  disagreement_rate: number;
  comparison_items: ModelComparisonItem[];
  c8_model_name: string;
  c10_model_name: string;
  processing_time_ms: number;
}

// ============================================================================
// Drift Monitoring
// ============================================================================

export interface DriftMonitoringRequest {
  flows: Record<string, unknown>[];
  batch_id?: string;
}

export interface DriftDecisionItem {
  action: string;
  priority: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  trigger_reason: string;
  evidence_metrics: Record<string, unknown>;
  operational_guidance: string;
}

export interface NumericalFeatureDrift {
  psi: number;
  ks_statistic: number;
  ks_pvalue: number;
  drift_level: "STABLE" | "MODERATE_DRIFT" | "CRITICAL_DRIFT";
  missing_rate: number;
  non_finite_rate: number;
}

export interface CategoricalFeatureDrift {
  psi: number;
  drift_level: "STABLE" | "MODERATE_DRIFT" | "CRITICAL_DRIFT";
  unknown_category_count: number;
  unknown_category_rate: number;
  novel_categories: string[];
}

export interface PredictionDriftSummary {
  model_id: string;
  prediction_psi: number;
  mean_confidence: number;
  min_confidence: number;
  low_confidence_rate: number;
  mean_entropy: number;
  drift_level: string;
  predicted_distribution: Record<string, number>;
  c8_c10_disagreement_rate?: number;
}

export interface DriftMonitoringResponse {
  batch_id: string;
  batch_size: number;
  is_valid_size: boolean;
  overall_drift_status: "STABLE" | "LOW_DRIFT" | "MODERATE_DRIFT" | "CRITICAL_DRIFT";
  drifting_features_count: number;
  critical_features_count: number;
  primary_recommendation: string;
  decisions: DriftDecisionItem[];
  requires_soc_escalation: boolean;
  requires_pipeline_investigation: boolean;
  feature_drift_summary: {
    numerical: Record<string, NumericalFeatureDrift>;
    categorical: Record<string, CategoricalFeatureDrift>;
  };
  prediction_drift_summary?: PredictionDriftSummary;
  warnings: string[];
  reference_dataset_info: {
    source: string;
    sample_count: number;
    sealed_test_partition_status: string;
  };
  processing_time_ms: number;
}

// ============================================================================
// Explainability (SHAP)
// ============================================================================

export interface ExplainRequest {
  flows: NetworkFlowRecord[];
  model_id?: MulticlassModelId | string;
  top_k?: number;
}

export interface FeatureAttributionItem {
  feature_name: string;
  attribution_margin: number;
  direction: "push_towards_class" | "push_away_from_class";
}

export interface FlowExplanationItem {
  index: number;
  predicted_class: string;
  output_space: string;
  base_margin: number;
  top_attributions: FeatureAttributionItem[];
}

export interface ExplainResponse {
  model_id: string;
  model_name: string;
  output_space_description: string;
  explanations: FlowExplanationItem[];
  flow_count: number;
  processing_time_ms: number;
}

// ============================================================================
// Metadata, Health & Status
// ============================================================================

export interface HealthResponse {
  status: "healthy" | string;
  service: string;
  version: string;
  models_ready: boolean;
  reference_data_ready: boolean;
}

export interface SystemStatusResponse {
  status: "operational" | "initializing" | string;
  uptime_seconds: number;
  models_loaded: string[];
  reference_data_loaded: boolean;
  reference_rows: number;
  environment: string;
  memory_rss_mb: number;
  active_binary_threshold: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  task_type: string;
  schema_name: string;
  feature_count: number;
  description: string;
  is_default: boolean;
  threshold_info?: Record<string, number>;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  count: number;
}

export interface ApiErrorDetail {
  field?: string;
  message: string;
}

export interface ApiErrorResponse {
  error_type: string;
  message: string;
  detail?: ApiErrorDetail[] | unknown;
}
