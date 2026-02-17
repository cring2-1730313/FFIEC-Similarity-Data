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


def _fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


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
        reason_candidates: List[Tuple[float, str]] = []
        seen_reasons: set[str] = set()

        def add_reason(priority: float, text: str) -> None:
            if not text or text in seen_reasons:
                return
            seen_reasons.add(text)
            reason_candidates.append((float(priority), text))

        size_threshold = float(self.driver_thresholds["size_pct_diff_max"])
        ldr_threshold = float(self.driver_thresholds["loan_to_deposit_abs_diff_max"])
        growth_threshold = float(self.driver_thresholds["growth_abs_diff_max"])
        overlap_threshold = float(self.driver_thresholds["overlap_for_strong_geo"])

        size_diff = _safe_pct_diff(subject["total_assets"], peer["total_assets"])
        if np.isfinite(size_diff):
            size_priority = max(0.1, 1.0 - min(size_diff, 1.0))
            add_reason(
                size_priority,
                f"Comparable asset size ({_fmt_currency(subject['total_assets'])} vs {_fmt_currency(peer['total_assets'])})",
            )

        ldr_a = subject.get("loan_to_deposit_ratio")
        ldr_b = peer.get("loan_to_deposit_ratio")
        if pd.notna(ldr_a) and pd.notna(ldr_b):
            ldr_diff = abs(float(ldr_a) - float(ldr_b))
            add_reason(
                max(0.1, 1.0 - min(ldr_diff / max(ldr_threshold * 2, 0.01), 1.0)),
                f"Similar loan-to-deposit ratio ({_fmt_pct(float(ldr_a))} vs {_fmt_pct(float(ldr_b))})",
            )

        growth_a = subject.get("avg_3y_assets_yoy_growth")
        growth_b = peer.get("avg_3y_assets_yoy_growth")
        if pd.notna(growth_a) and pd.notna(growth_b):
            growth_diff = abs(float(growth_a) - float(growth_b))
            add_reason(
                max(0.05, 1.0 - min(growth_diff / max(growth_threshold * 3, 0.01), 1.0)),
                f"Aligned 3Y asset growth ({_fmt_pct(float(growth_a))} vs {_fmt_pct(float(growth_b))})",
            )

        lending_mix_features = [
            ("loan_mix_cre_pct", "CRE"),
            ("loan_mix_ci_pct", "C&I"),
            ("loan_mix_residential_mortgage_pct", "Residential mortgage"),
            ("loan_mix_consumer_pct", "Consumer"),
            ("loan_mix_agricultural_pct", "Agricultural"),
            ("loan_mix_construction_pct", "Construction/development"),
        ]
        lending_diffs: List[Tuple[float, str]] = []
        for feat, label in lending_mix_features:
            a = subject.get(feat)
            b = peer.get(feat)
            if pd.notna(a) and pd.notna(b):
                diff = abs(float(a) - float(b))
                avg_share = (abs(float(a)) + abs(float(b))) / 2.0
                if avg_share < 0.02:
                    continue
                significance = min(1.0, max(avg_share / 0.08, 0.15))
                lending_diffs.append(
                    (
                        diff / significance,
                        f"Similar {label} lending mix ({_fmt_pct(float(a))} vs {_fmt_pct(float(b))})",
                    )
                )
        lending_diffs.sort(key=lambda x: x[0])
        for diff, text in lending_diffs[:2]:
            add_reason(max(0.1, 1.0 - min(diff / 0.30, 1.0)), text)

        for label, feat in [("Retail deposits", "deposit_mix_retail_pct"), ("Business deposits", "deposit_mix_business_pct")]:
            a = subject.get(feat)
            b = peer.get(feat)
            if pd.notna(a) and pd.notna(b):
                diff = abs(float(a) - float(b))
                avg_share = (abs(float(a)) + abs(float(b))) / 2.0
                significance = min(1.0, max(avg_share / 0.20, 0.25))
                add_reason(
                    max(0.1, 1.0 - min((diff / significance) / 0.25, 1.0)),
                    f"Similar {label.lower()} mix ({_fmt_pct(float(a))} vs {_fmt_pct(float(b))})",
                )

        if geo_diag["market_overlap"] >= overlap_threshold:
            add_reason(
                0.95,
                f"Strong market overlap ({int(geo_diag['shared_markets'])} shared CBSAs)",
            )
        elif geo_diag["markets_a"] >= 20 and geo_diag["markets_b"] >= 20:
            add_reason(
                0.6,
                f"Both broad-footprint banks ({int(geo_diag['markets_a'])} vs {int(geo_diag['markets_b'])} CBSAs)",
            )

        if subject["charter_type"] == peer["charter_type"]:
            add_reason(0.45, f"Same charter type ({subject['charter_type']})")

        if int(subject["has_holding_company"]) == int(peer["has_holding_company"]):
            if int(subject["has_holding_company"]) == 1:
                add_reason(0.30, "Both in holding-company structures")
            else:
                add_reason(0.30, "Both stand-alone institutions")

        if subject.get("specialty_group") == peer.get("specialty_group") and str(subject.get("specialty_group")) != "UNKNOWN":
            add_reason(0.28, f"Same specialty group ({subject.get('specialty_group')})")

        loans_to_assets_a = subject.get("loans_to_assets")
        loans_to_assets_b = peer.get("loans_to_assets")
        if pd.notna(loans_to_assets_a) and pd.notna(loans_to_assets_b):
            add_reason(
                0.22,
                f"Comparable loans-to-assets mix ({_fmt_pct(float(loans_to_assets_a))} vs {_fmt_pct(float(loans_to_assets_b))})",
            )

        deposits_to_assets_a = subject.get("deposits_to_assets")
        deposits_to_assets_b = peer.get("deposits_to_assets")
        if pd.notna(deposits_to_assets_a) and pd.notna(deposits_to_assets_b):
            add_reason(
                0.20,
                f"Comparable deposits-to-assets mix ({_fmt_pct(float(deposits_to_assets_a))} vs {_fmt_pct(float(deposits_to_assets_b))})",
            )

        growth_loans_a = subject.get("avg_3y_loans_yoy_growth")
        growth_loans_b = peer.get("avg_3y_loans_yoy_growth")
        if pd.notna(growth_loans_a) and pd.notna(growth_loans_b):
            add_reason(
                0.18,
                f"Aligned 3Y loan growth ({_fmt_pct(float(growth_loans_a))} vs {_fmt_pct(float(growth_loans_b))})",
            )

        reason_candidates.sort(key=lambda x: x[0], reverse=True)
        top_reasons = [text for _, text in reason_candidates[:3]]
        if not top_reasons:
            top_reasons = ["Asset and lending metrics are numerically similar"]
        return " | ".join(top_reasons)

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
