"""Data preprocessing, partition splitters, duplicate auditing, and transformation pipelines."""

from ml.preprocessing.duplicate_audit import (
    audit_duplicates_and_conflicts,
    compute_feature_hashes,
    save_quarantine_manifests,
)
from ml.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    prepare_train_val_test_matrices,
)
from ml.preprocessing.schema import (
    ALTERNATIVE_FEATURE_COLUMNS_NO_STTL,
    BINARY_LABEL_MAPPING,
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    MULTICLASS_ID_TO_LABEL,
    MULTICLASS_LABEL_MAPPING,
    NUMERICAL_FEATURES,
    PRIMARY_FEATURE_COLUMNS,
    TARGET_BINARY,
    TARGET_COLUMNS,
    TARGET_MULTICLASS,
    validate_feature_matrix_columns,
)
from ml.preprocessing.split import (
    create_leakage_resistant_split,
    export_evaluation_splits,
    partition_groups_stratified,
)
from ml.preprocessing.transformers import (
    FeatureSchemaSelector,
    InfClipper,
    build_categorical_encoder,
    build_numerical_scaler,
)

__all__ = [
    "IDENTIFIER_COLUMNS",
    "TARGET_BINARY",
    "TARGET_MULTICLASS",
    "TARGET_COLUMNS",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "PRIMARY_FEATURE_COLUMNS",
    "ALTERNATIVE_FEATURE_COLUMNS_NO_STTL",
    "BINARY_LABEL_MAPPING",
    "MULTICLASS_LABEL_MAPPING",
    "MULTICLASS_ID_TO_LABEL",
    "validate_feature_matrix_columns",
    "compute_feature_hashes",
    "audit_duplicates_and_conflicts",
    "save_quarantine_manifests",
    "partition_groups_stratified",
    "create_leakage_resistant_split",
    "export_evaluation_splits",
    "InfClipper",
    "FeatureSchemaSelector",
    "build_categorical_encoder",
    "build_numerical_scaler",
    "create_preprocessing_pipeline",
    "prepare_train_val_test_matrices",
]
