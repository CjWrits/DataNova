"""Feature and target schema definitions for the UNSW-NB15 dataset.

Defines feature boundaries, target mappings, predictor feature groupings,
and primary vs. alternative (without sttl) feature schemas.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

# Identifiers and targets that must NEVER enter predictor feature matrices
IDENTIFIER_COLUMNS: List[str] = ["id"]
TARGET_BINARY: str = "label"
TARGET_MULTICLASS: str = "attack_cat"
TARGET_COLUMNS: List[str] = [TARGET_BINARY, TARGET_MULTICLASS]

# Categorical predictor features
CATEGORICAL_FEATURES: List[str] = [
    "proto",
    "service",
    "state",
]

# Continuous and discrete numerical predictor flow features
NUMERICAL_FEATURES: List[str] = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

# Primary feature schema (all 42 predictor attributes: 3 categorical + 39 numerical)
PRIMARY_FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Alternative feature schema excluding synthetic testbed artifact 'sttl' (41 predictor attributes)
ALTERNATIVE_FEATURE_COLUMNS_NO_STTL: List[str] = [
    col for col in PRIMARY_FEATURE_COLUMNS if col != "sttl"
]

# Target class taxonomy and mappings
BINARY_LABEL_MAPPING: Dict[int, str] = {
    0: "Normal",
    1: "Attack",
}

MULTICLASS_LABEL_MAPPING: Dict[str, int] = {
    "Normal": 0,
    "Generic": 1,
    "Exploits": 2,
    "Fuzzers": 3,
    "DoS": 4,
    "Reconnaissance": 5,
    "Analysis": 6,
    "Backdoor": 7,
    "Shellcode": 8,
    "Worms": 9,
}

MULTICLASS_ID_TO_LABEL: Dict[int, str] = {v: k for k, v in MULTICLASS_LABEL_MAPPING.items()}


def validate_feature_matrix_columns(columns: List[str], schema: str = "primary") -> Tuple[bool, List[str]]:
    """Validate that feature columns contain no forbidden identifiers or target columns."""
    violations: List[str] = []
    forbidden: Set[str] = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS)

    for col in columns:
        if col in forbidden:
            violations.append(f"Forbidden column detected in feature matrix: '{col}'")

    expected_cols = PRIMARY_FEATURE_COLUMNS if schema == "primary" else ALTERNATIVE_FEATURE_COLUMNS_NO_STTL
    missing = set(expected_cols) - set(columns)
    if missing:
        violations.append(f"Missing expected features for schema '{schema}': {sorted(list(missing))}")

    return len(violations) == 0, violations
