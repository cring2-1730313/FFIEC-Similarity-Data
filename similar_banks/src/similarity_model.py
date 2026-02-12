from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from .feature_engineering import FeatureSet
    from .geographic_similarity import GeographicSimilarityEngine
except ImportError:
    from feature_engineering import FeatureSet
    from geographic_similarity import GeographicSimilarityEngine


def _fmt_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000_000:
        return f"${value/1_000_000_000_000:.1f}T"
    if abs_val >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    return f"${value:,.0f}"


def _safe_pct_diff(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.inf
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


@dataclass
class GutCheckResult:
    passed: bool
    details: str


class SimilarBankRecommender:
    def __init__(self, weights_config: Dict[str, Any]) -> None:
        self.top_n = int(weights_config["model"]["top_n"])
        self.candidate_k = int(weights_config["model"]["candidate_k"])
        self.weights = weights_config["weights"]
        self.categorical_weights = weights_config["categorical_weights"]
        self.driver_thresholds = weights_config["driver_thresholds"]
        self.numeric_feature_weights = weights_config.get("numeric_feature_weights", {})

    def _categorical_similarity(self, subject: Dict[str, Any], peer: Dict[str, Any]) -> float:
        total_weight = 0.0
        score = 0.0

        charter_weight = float(self.categorical_weights.get("charter_type", 0.0))
        total_weight += charter_weight
        score += charter_weight * float(subject["charter_type"] == peer["charter_type"])

        holding_weight = float(self.categorical_weights.get("holding_company", 0.0))
        total_weight += holding_weight
        score += holding_weight * float(
            int(subject["has_holding_company"]) == int(peer["has_holding_company"])
        )

        specialty_weight = float(self.categorical_weights.get("specialty_group", 0.0))
        total_weight += specialty_weight
        score += specialty_weight * float(subject["specialty_group"] == peer["specialty_group"])

        if total_weight <= 0:
            return 0.0
        return score / total_weight

    def _build_similarity_drivers(
        self,
        subject: Dict[str, Any],
        peer: Dict[str, Any],
        geo_diag: Dict[str, float],
    ) -> str:
        reasons: List[str] = []
        size_threshold = float(self.driver_thresholds["size_pct_diff_max"])
        ldr_threshold = float(self.driver_thresholds["loan_to_deposit_abs_diff_max"])
        growth_threshold = float(self.driver_thresholds["growth_abs_diff_max"])
        overlap_threshold = float(self.driver_thresholds["overlap_for_strong_geo"])

        size_diff = _safe_pct_diff(subject["total_assets"], peer["total_assets"])
        if size_diff <= size_threshold:
            reasons.append(
                f"Comparable asset size ({_fmt_currency(subject['total_assets'])} vs {_fmt_currency(peer['total_assets'])})"
            )

        ldr_a = subject.get("loan_to_deposit_ratio")
        ldr_b = peer.get("loan_to_deposit_ratio")
        if pd.notna(ldr_a) and pd.notna(ldr_b) and abs(float(ldr_a) - float(ldr_b)) <= ldr_threshold:
            reasons.append(
                f"Similar loan-to-deposit ratio ({float(ldr_a):.2f} vs {float(ldr_b):.2f})"
            )

        growth_a = subject.get("avg_3y_assets_yoy_growth")
        growth_b = peer.get("avg_3y_assets_yoy_growth")
        if (
            pd.notna(growth_a)
            and pd.notna(growth_b)
            and abs(float(growth_a) - float(growth_b)) <= growth_threshold
        ):
            reasons.append(
                f"Aligned 3Y asset growth ({float(growth_a)*100:.1f}% vs {float(growth_b)*100:.1f}%)"
            )

        if geo_diag["market_overlap"] >= overlap_threshold:
            reasons.append(
                f"Strong market overlap ({int(geo_diag['shared_markets'])} shared CBSAs)"
            )
        elif geo_diag["markets_a"] >= 20 and geo_diag["markets_b"] >= 20:
            reasons.append(
                f"Both broad-footprint banks ({int(geo_diag['markets_a'])} vs {int(geo_diag['markets_b'])} CBSAs)"
            )

        if subject["charter_type"] == peer["charter_type"]:
            reasons.append(f"Same charter type ({subject['charter_type']})")

        if int(subject["has_holding_company"]) == int(peer["has_holding_company"]):
            if int(subject["has_holding_company"]) == 1:
                reasons.append("Both in holding-company structures")
            else:
                reasons.append("Both stand-alone institutions")

        if len(reasons) < 3:
            reasons.append("Comparable balance-sheet structure")
        if len(reasons) < 3:
            reasons.append("Similar operating profile")
        return " | ".join(reasons[:3])

    def compute(
        self,
        feature_set: FeatureSet,
        geo_engine: GeographicSimilarityEngine,
        computed_date: date,
    ) -> pd.DataFrame:
        features = feature_set.frame.reset_index(drop=True).copy()
        n_banks = len(features)
        if n_banks == 0:
            return pd.DataFrame(
                columns=[
                    "subject_idrssd",
                    "subject_name",
                    "similar_rank",
                    "similar_idrssd",
                    "similar_name",
                    "similarity_score",
                    "similarity_drivers",
                    "computed_date",
                ]
            )

        numeric_cols = feature_set.numeric_features
        if not numeric_cols:
            raise ValueError("No numeric features available for similarity model.")

        x = features[numeric_cols].to_numpy(dtype=float)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        feature_weight_vector = np.array(
            [float(self.numeric_feature_weights.get(col, 1.0)) for col in numeric_cols],
            dtype=float,
        )
        x_scaled = x_scaled * feature_weight_vector

        max_neighbors = min(self.candidate_k + 1, n_banks)
        nn = NearestNeighbors(n_neighbors=max_neighbors, metric="euclidean")
        nn.fit(x_scaled)
        distances, indices = nn.kneighbors(x_scaled)

        records = features.to_dict(orient="records")
        out_rows: List[Dict[str, Any]] = []
        for i, subject in enumerate(records):
            candidates: List[Tuple[float, int, str]] = []
            for dist, j in zip(distances[i], indices[i]):
                if i == j:
                    continue
                peer = records[int(j)]
                fin_sim = 1.0 / (1.0 + float(max(dist, 0.0)))
                geo_diag = geo_engine.similarity(int(subject["bank_id"]), int(peer["bank_id"]))
                cat_sim = self._categorical_similarity(subject, peer)

                score = (
                    float(self.weights["financial"]) * fin_sim
                    + float(self.weights["geographic"]) * float(geo_diag["geo_score"])
                    + float(self.weights["categorical"]) * cat_sim
                )
                drivers = self._build_similarity_drivers(subject, peer, geo_diag)
                candidates.append((float(np.clip(score, 0.0, 1.0)), int(j), drivers))

            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[: self.top_n]
            for rank, (score, peer_idx, drivers) in enumerate(top, start=1):
                peer = records[peer_idx]
                out_rows.append(
                    {
                        "subject_idrssd": int(subject["bank_id"]),
                        "subject_name": str(subject["subject_name"]),
                        "similar_rank": rank,
                        "similar_idrssd": int(peer["bank_id"]),
                        "similar_name": str(peer["subject_name"]),
                        "similarity_score": float(score),
                        "similarity_drivers": drivers,
                        "computed_date": pd.to_datetime(computed_date),
                    }
                )

        return pd.DataFrame(out_rows).sort_values(
            ["subject_idrssd", "similar_rank"], kind="stable"
        )


def spot_check(similar_df: pd.DataFrame, query: str | int) -> pd.DataFrame:
    if similar_df.empty:
        return similar_df

    if isinstance(query, int) or (isinstance(query, str) and query.strip().isdigit()):
        bank_id = int(query)
        return similar_df.loc[similar_df["subject_idrssd"] == bank_id].copy()

    q = str(query).strip().upper()
    return similar_df.loc[similar_df["subject_name"].str.upper().str.contains(q, na=False)].copy()


def gut_check_wells(similar_df: pd.DataFrame) -> GutCheckResult:
    wells = similar_df.loc[
        similar_df["subject_name"]
        .str.upper()
        .str.contains("WELLS FARGO BANK, NATIONAL ASSOCIATION", na=False)
    ]
    if wells.empty:
        return GutCheckResult(
            passed=False,
            details="Wells Fargo Bank, National Association not found in output subjects.",
        )

    peers = set(wells["similar_name"].str.upper().tolist())
    has_jpm = any("JPMORGAN CHASE BANK" in p for p in peers)
    has_bofa = any("BANK OF AMERICA" in p for p in peers)
    passed = has_jpm and has_bofa
    details = f"JPM present={has_jpm}, BofA present={has_bofa}"
    return GutCheckResult(passed=passed, details=details)
