"""Pydantic schemas for network flow telemetry, prediction requests, and responses."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class NetworkFlowRecord(BaseModel):
    """Network flow record matching the 42 UNSW-NB15 predictor features."""

    # Categorical features
    proto: str = Field(..., description="Transaction protocol (e.g. 'tcp', 'udp', 'arp')")
    service: str = Field(..., description="Application service (e.g. 'http', 'dns', '-')")
    state: str = Field(..., description="Flow connection state (e.g. 'FIN', 'CON', 'INT')")

    # Numerical features
    dur: float = Field(..., ge=0.0, description="Flow duration in seconds")
    spkts: float = Field(..., ge=0.0, description="Source to destination packet count")
    dpkts: float = Field(..., ge=0.0, description="Destination to source packet count")
    sbytes: float = Field(..., ge=0.0, description="Source to destination transaction bytes")
    dbytes: float = Field(..., ge=0.0, description="Destination to source transaction bytes")
    rate: float = Field(..., ge=0.0, description="Flow transmission rate")
    sttl: Optional[float] = Field(64.0, ge=0.0, description="Source time to live")
    dttl: float = Field(..., ge=0.0, description="Destination time to live")
    sload: float = Field(..., ge=0.0, description="Source bits per second")
    dload: float = Field(..., ge=0.0, description="Destination bits per second")
    sloss: float = Field(..., ge=0.0, description="Source packets retransmitted or dropped")
    dloss: float = Field(..., ge=0.0, description="Destination packets retransmitted or dropped")
    sinpkt: float = Field(..., ge=0.0, description="Source interpacket arrival time (msec)")
    dinpkt: float = Field(..., ge=0.0, description="Destination interpacket arrival time (msec)")
    sjit: float = Field(..., ge=0.0, description="Source jitter (msec)")
    djit: float = Field(..., ge=0.0, description="Destination jitter (msec)")
    swin: float = Field(..., ge=0.0, description="Source TCP window advertisement")
    stcpb: float = Field(..., ge=0.0, description="Source TCP sequence number")
    dtcpb: float = Field(..., ge=0.0, description="Destination TCP sequence number")
    dwin: float = Field(..., ge=0.0, description="Destination TCP window advertisement")
    tcprtt: float = Field(..., ge=0.0, description="TCP round trip time (msec)")
    synack: float = Field(..., ge=0.0, description="TCP SYN to SYN-ACK time (msec)")
    ackdat: float = Field(..., ge=0.0, description="TCP SYN-ACK to ACK time (msec)")
    smean: float = Field(..., ge=0.0, description="Mean packet size transmitted by source")
    dmean: float = Field(..., ge=0.0, description="Mean packet size transmitted by destination")
    trans_depth: float = Field(..., ge=0.0, description="Pipelined HTTP request count")
    response_body_len: float = Field(..., ge=0.0, description="Actual uncompressed HTTP body length")
    ct_srv_src: float = Field(..., ge=0.0, description="Connections to same service from source IP")
    ct_state_ttl: float = Field(..., ge=0.0, description="Connections with identical state and TTL combination")
    ct_dst_ltm: float = Field(..., ge=0.0, description="Connections to same destination address")
    ct_src_dport_ltm: float = Field(..., ge=0.0, description="Connections from same source IP to destination port")
    ct_dst_sport_ltm: float = Field(..., ge=0.0, description="Connections to same destination from source port")
    ct_dst_src_ltm: float = Field(..., ge=0.0, description="Connections between source and destination pair")
    is_ftp_login: float = Field(..., ge=0.0, description="Binary flag: 1 if FTP session logged in")
    ct_ftp_cmd: float = Field(..., ge=0.0, description="Count of FTP commands in session")
    ct_flw_http_mthd: float = Field(..., ge=0.0, description="Count of HTTP GET/POST methods in session")
    ct_src_ltm: float = Field(..., ge=0.0, description="Connections from same source address")
    ct_srv_dst: float = Field(..., ge=0.0, description="Connections to same service and destination IP")
    is_sm_ips_ports: float = Field(..., ge=0.0, description="Binary flag: 1 if source/dest IP and port match")

    @field_validator("*", mode="before")
    @classmethod
    def validate_finite_floats(cls, v: Any) -> Any:
        """Reject non-finite floats (NaN, Inf, -Inf) during ingestion."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Field must contain a finite numeric value; NaN and Inf are rejected.")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "proto": "tcp",
                "service": "http",
                "state": "FIN",
                "dur": 0.1214,
                "spkts": 10,
                "dpkts": 8,
                "sbytes": 2520,
                "dbytes": 1040,
                "rate": 140.0,
                "sttl": 62.0,
                "dttl": 252.0,
                "sload": 150000.0,
                "dload": 65000.0,
                "sloss": 0.0,
                "dloss": 0.0,
                "sinpkt": 12.14,
                "dinpkt": 14.28,
                "sjit": 1.5,
                "djit": 1.2,
                "swin": 255.0,
                "stcpb": 624100.0,
                "dtcpb": 812400.0,
                "dwin": 255.0,
                "tcprtt": 0.0012,
                "synack": 0.0006,
                "ackdat": 0.0006,
                "smean": 252.0,
                "dmean": 130.0,
                "trans_depth": 1.0,
                "response_body_len": 512.0,
                "ct_srv_src": 2.0,
                "ct_state_ttl": 0.0,
                "ct_dst_ltm": 3.0,
                "ct_src_dport_ltm": 1.0,
                "ct_dst_sport_ltm": 1.0,
                "ct_dst_src_ltm": 2.0,
                "is_ftp_login": 0.0,
                "ct_ftp_cmd": 0.0,
                "ct_flw_http_mthd": 1.0,
                "ct_src_ltm": 2.0,
                "ct_srv_dst": 2.0,
                "is_sm_ips_ports": 0.0,
            }
        }
    )


# ============================================================================
# Request / Response Schemas: Binary Prediction
# ============================================================================

class BinaryPredictionRequest(BaseModel):
    """Batch of network flows for binary intrusion classification."""

    flows: List[NetworkFlowRecord] = Field(..., min_length=1, max_length=1000)
    threshold_preset: Optional[str] = Field(
        "default",
        description="Operating threshold preset: 'default', 'f1_optimal', 'cost_optimal', 'low_false_alarm', 'high_recall'",
    )


class BinaryPredictionItem(BaseModel):
    index: int
    label: int
    label_name: str
    attack_probability: float
    threshold_applied: float


class BinaryPredictionResponse(BaseModel):
    model_id: str
    model_name: str
    threshold_preset: str
    threshold_applied: float
    predictions: List[BinaryPredictionItem]
    flow_count: int
    processing_time_ms: float


# ============================================================================
# Request / Response Schemas: Multiclass Classification
# ============================================================================

class MulticlassPredictionRequest(BaseModel):
    """Batch of network flows for multiclass attack family classification."""

    flows: List[NetworkFlowRecord] = Field(..., min_length=1, max_length=1000)
    model_id: Optional[str] = Field(
        "c8",
        description="Explicit model selection: 'c8' (Primary w/ STTL) or 'c10' (Alternative No-STTL). Never silently switched.",
    )


class MulticlassPredictionItem(BaseModel):
    index: int
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    probabilities: Dict[str, float]


class MulticlassPredictionResponse(BaseModel):
    model_id: str
    model_name: str
    schema_used: str
    predictions: List[MulticlassPredictionItem]
    flow_count: int
    processing_time_ms: float


# ============================================================================
# Request / Response Schemas: Model Comparison (C8 vs C10)
# ============================================================================

class ModelComparisonRequest(BaseModel):
    """Batch of network flows for side-by-side comparison between C8 and C10."""

    flows: List[NetworkFlowRecord] = Field(..., min_length=1, max_length=1000)


class ModelComparisonItem(BaseModel):
    index: int
    c8_predicted_class: str
    c8_confidence: float
    c10_predicted_class: str
    c10_confidence: float
    agreement: bool
    l1_distribution_divergence: float


class ModelComparisonResponse(BaseModel):
    total_flows: int
    agreement_count: int
    disagreement_count: int
    agreement_rate: float
    disagreement_rate: float
    comparison_items: List[ModelComparisonItem]
    c8_model_name: str
    c10_model_name: str
    processing_time_ms: float


# ============================================================================
# Request / Response Schemas: Drift Monitoring
# ============================================================================

class DriftMonitoringRequest(BaseModel):
    """Batch of network telemetry flows for statistical drift & data quality evaluation."""

    flows: List[Dict[str, Any]] = Field(..., min_length=1, max_length=5000)
    batch_id: Optional[str] = Field("api_batch_001", description="Identifier for telemetry batch")


class DriftDecisionItem(BaseModel):
    action: str
    priority: str
    trigger_reason: str
    evidence_metrics: Dict[str, Any]
    operational_guidance: str


class DriftMonitoringResponse(BaseModel):
    batch_id: str
    batch_size: int
    is_valid_size: bool
    overall_drift_status: str
    drifting_features_count: int
    critical_features_count: int
    primary_recommendation: str
    decisions: List[DriftDecisionItem]
    requires_soc_escalation: bool
    requires_pipeline_investigation: bool
    feature_drift_summary: Dict[str, Any]
    prediction_drift_summary: Optional[Dict[str, Any]] = None
    warnings: List[str]
    reference_dataset_info: Dict[str, Any]
    processing_time_ms: float


# ============================================================================
# Request / Response Schemas: Explainability
# ============================================================================

class ExplainRequest(BaseModel):
    """Small bounded batch of flows for SHAP feature attribution."""

    flows: List[NetworkFlowRecord] = Field(..., min_length=1, max_length=10)
    model_id: Optional[str] = Field("c8", description="Model to explain: 'c8' or 'c10'")
    top_k: Optional[int] = Field(10, ge=1, le=42, description="Number of top features to report")


class FeatureAttributionItem(BaseModel):
    feature_name: str
    attribution_margin: float
    direction: str  # 'push_towards_class' or 'push_away_from_class'


class FlowExplanationItem(BaseModel):
    index: int
    predicted_class: str
    output_space: str
    base_margin: float
    top_attributions: List[FeatureAttributionItem]


class ExplainResponse(BaseModel):
    model_id: str
    model_name: str
    output_space_description: str
    explanations: List[FlowExplanationItem]
    flow_count: int
    processing_time_ms: float


# ============================================================================
# Metadata / Health / Error Schemas
# ============================================================================

class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    task_type: str
    schema_name: str
    feature_count: int
    description: str
    is_default: bool
    threshold_info: Optional[Dict[str, Any]] = None


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    count: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_ready: bool
    reference_data_ready: bool


class SystemStatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    models_loaded: List[str]
    reference_data_loaded: bool
    reference_rows: int
    environment: str
    memory_rss_mb: float
    active_binary_threshold: float


class ErrorResponse(BaseModel):
    error_type: str
    message: str
    timestamp: str
    detail: Optional[Any] = None
