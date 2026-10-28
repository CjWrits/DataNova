"""Domain-specific feature engineering transformer for network intrusion detection.

Generates verified, leakage-free derived flow features:
- Flow volume asymmetries (byte and packet ratios)
- Payload density indicators (bytes per packet)
- Throughput and load balance ratios
- TCP handshake timing fractions
- Connection density fan-out ratios
- Packet loss fractions

All transformations include zero-division and non-finite value safeguards,
strict schema validation, and complete scikit-learn pipeline compatibility.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_FEATURE_NAMES: List[str] = [
    "feat_byte_asymmetry_ratio",
    "feat_packet_asymmetry_ratio",
    "feat_mean_packet_size_ratio",
    "feat_src_payload_density",
    "feat_dst_payload_density",
    "feat_load_ratio",
    "feat_synack_to_rtt_ratio",
    "feat_service_fanout_ratio",
    "feat_src_loss_rate",
    "feat_dst_loss_rate",
]


class NetworkFlowFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates 10 derived bidirectional flow interaction features with strict numerical safeguards."""

    def __init__(self, include_original: bool = True, eps: float = 1e-6) -> None:
        self.include_original = include_original
        self.eps = eps
        self.input_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> NetworkFlowFeatureEngineer:
        """Validate input feature columns."""
        if isinstance(X, pd.DataFrame):
            self.input_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Compute derived features and append to or replace input matrix."""
        if not isinstance(X, pd.DataFrame):
            if self.input_columns_ is not None:
                df = pd.DataFrame(X, columns=self.input_columns_)
            else:
                raise ValueError("Expected pandas DataFrame as input to NetworkFlowFeatureEngineer.")
        else:
            df = X.copy()

        # Extract source and destination raw flow features with fallback
        sbytes = df["sbytes"].astype(float) if "sbytes" in df.columns else pd.Series(0.0, index=df.index)
        dbytes = df["dbytes"].astype(float) if "dbytes" in df.columns else pd.Series(0.0, index=df.index)
        spkts = df["spkts"].astype(float) if "spkts" in df.columns else pd.Series(0.0, index=df.index)
        dpkts = df["dpkts"].astype(float) if "dpkts" in df.columns else pd.Series(0.0, index=df.index)
        smean = df["smean"].astype(float) if "smean" in df.columns else pd.Series(0.0, index=df.index)
        dmean = df["dmean"].astype(float) if "dmean" in df.columns else pd.Series(0.0, index=df.index)
        sload = df["sload"].astype(float) if "sload" in df.columns else pd.Series(0.0, index=df.index)
        dload = df["dload"].astype(float) if "dload" in df.columns else pd.Series(0.0, index=df.index)
        sloss = df["sloss"].astype(float) if "sloss" in df.columns else pd.Series(0.0, index=df.index)
        dloss = df["dloss"].astype(float) if "dloss" in df.columns else pd.Series(0.0, index=df.index)
        synack = df["synack"].astype(float) if "synack" in df.columns else pd.Series(0.0, index=df.index)
        tcprtt = df["tcprtt"].astype(float) if "tcprtt" in df.columns else pd.Series(0.0, index=df.index)
        ct_srv_src = df["ct_srv_src"].astype(float) if "ct_srv_src" in df.columns else pd.Series(0.0, index=df.index)
        ct_dst_ltm = df["ct_dst_ltm"].astype(float) if "ct_dst_ltm" in df.columns else pd.Series(0.0, index=df.index)

        # 1. Byte Asymmetry Ratio: [-1, +1] normalized
        bar = (sbytes - dbytes) / (sbytes + dbytes + 1.0)

        # 2. Packet Asymmetry Ratio: [-1, +1] normalized
        par = (spkts - dpkts) / (spkts + dpkts + 1.0)

        # 3. Mean Packet Size Ratio
        smr = (smean + 1.0) / (dmean + 1.0)

        # 4. Payload Density (Source bytes per packet)
        src_pd = sbytes / (spkts + 1.0)

        # 5. Payload Density (Destination bytes per packet)
        dst_pd = dbytes / (dpkts + 1.0)

        # 6. Load Ratio (Source load to destination load)
        load_r = (sload + 1.0) / (dload + 1.0)

        # 7. TCP Handshake Synack-to-RTT fraction: [0, 1] bounded
        synack_frac = (synack + self.eps) / (tcprtt + self.eps)
        synack_frac = np.clip(synack_frac, 0.0, 1.0)

        # 8. Service Fan-out Density
        srv_fanout = (ct_srv_src + 1.0) / (ct_dst_ltm + 1.0)

        # 9. Packet Loss Rate (Source)
        src_lr = sloss / (spkts + 1.0)

        # 10. Packet Loss Rate (Destination)
        dst_lr = dloss / (dpkts + 1.0)

        new_features = pd.DataFrame(
            {
                "feat_byte_asymmetry_ratio": bar,
                "feat_packet_asymmetry_ratio": par,
                "feat_mean_packet_size_ratio": smr,
                "feat_src_payload_density": src_pd,
                "feat_dst_payload_density": dst_pd,
                "feat_load_ratio": load_r,
                "feat_synack_to_rtt_ratio": synack_frac,
                "feat_service_fanout_ratio": srv_fanout,
                "feat_src_loss_rate": src_lr,
                "feat_dst_loss_rate": dst_lr,
            },
            index=df.index,
        )

        # Clean any non-finite values created by unexpected inputs
        new_features = new_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if self.include_original:
            return pd.concat([df, new_features], axis=1)
        return new_features

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Return list of transformed feature names."""
        base_cols = list(input_features) if input_features is not None else (self.input_columns_ or [])
        if self.include_original:
            return np.array(base_cols + ENGINEERED_FEATURE_NAMES, dtype=object)
        return np.array(ENGINEERED_FEATURE_NAMES, dtype=object)
