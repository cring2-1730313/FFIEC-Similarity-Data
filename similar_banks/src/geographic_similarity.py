from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _as_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class GeographicSimilarityEngine:
    def __init__(
        self,
        locations_df: pd.DataFrame,
        bank_features_df: pd.DataFrame,
        mappings: Dict[str, Any],
        geography_config: Dict[str, float],
    ) -> None:
        ids = mappings["identifiers"]
        geo = mappings["geography"]

        cert_col = ids["cert"]
        market_col = geo["market_code"]
        self.overlap_weight = float(geography_config["market_overlap_weight"])
        self.concentration_weight = float(geography_config["concentration_weight"])
        self.both_missing_score = float(geography_config["both_missing_score"])
        self.one_missing_score = float(geography_config["one_missing_score"])

        self.bank_to_cert: Dict[int, Optional[int]] = {
            int(row.bank_id): _as_int(row.cert)
            for row in bank_features_df[["bank_id", "cert"]].itertuples(index=False)
        }

        loc = locations_df[[cert_col, market_col]].dropna().copy()
        loc[cert_col] = pd.to_numeric(loc[cert_col], errors="coerce").astype("Int64")
        loc[market_col] = loc[market_col].astype(str)
        loc = loc.dropna(subset=[cert_col])

        market_counts = (
            loc.groupby([cert_col, market_col], as_index=False)
            .size()
            .rename(columns={"size": "branch_count"})
        )

        self.cert_market_shares: Dict[int, Dict[str, float]] = {}
        self.cert_hhi: Dict[int, float] = {}
        for cert, group in market_counts.groupby(cert_col):
            cert_int = _as_int(cert)
            if cert_int is None:
                continue
            total = float(group["branch_count"].sum())
            if total <= 0:
                continue
            shares = {
                str(row[market_col]): float(row["branch_count"] / total)
                for _, row in group.iterrows()
            }
            self.cert_market_shares[cert_int] = shares
            self.cert_hhi[cert_int] = sum(v * v for v in shares.values())

        self._pair_cache: Dict[Tuple[int, int], Dict[str, float]] = {}

    @staticmethod
    def _weighted_jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        min_sum = 0.0
        max_sum = 0.0
        for k in keys:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            min_sum += min(av, bv)
            max_sum += max(av, bv)
        if max_sum <= 0:
            return 0.0
        return min_sum / max_sum

    def similarity(self, bank_a_id: int, bank_b_id: int) -> Dict[str, float]:
        key = (bank_a_id, bank_b_id) if bank_a_id < bank_b_id else (bank_b_id, bank_a_id)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached

        cert_a = self.bank_to_cert.get(bank_a_id)
        cert_b = self.bank_to_cert.get(bank_b_id)
        shares_a = self.cert_market_shares.get(cert_a or -1)
        shares_b = self.cert_market_shares.get(cert_b or -1)

        if shares_a is None and shares_b is None:
            result = {
                "geo_score": self.both_missing_score,
                "market_overlap": self.both_missing_score,
                "concentration_similarity": self.both_missing_score,
                "shared_markets": 0.0,
                "markets_a": 0.0,
                "markets_b": 0.0,
            }
        elif shares_a is None or shares_b is None:
            markets_a = float(len(shares_a) if shares_a is not None else 0)
            markets_b = float(len(shares_b) if shares_b is not None else 0)
            result = {
                "geo_score": self.one_missing_score,
                "market_overlap": self.one_missing_score,
                "concentration_similarity": self.one_missing_score,
                "shared_markets": 0.0,
                "markets_a": markets_a,
                "markets_b": markets_b,
            }
        else:
            overlap = self._weighted_jaccard(shares_a, shares_b)
            hhi_a = self.cert_hhi.get(cert_a or -1, 1.0)
            hhi_b = self.cert_hhi.get(cert_b or -1, 1.0)
            concentration_similarity = max(0.0, 1.0 - abs(hhi_a - hhi_b))
            geo_score = (self.overlap_weight * overlap) + (
                self.concentration_weight * concentration_similarity
            )
            shared = len(set(shares_a).intersection(shares_b))
            result = {
                "geo_score": geo_score,
                "market_overlap": overlap,
                "concentration_similarity": concentration_similarity,
                "shared_markets": float(shared),
                "markets_a": float(len(shares_a)),
                "markets_b": float(len(shares_b)),
            }

        self._pair_cache[key] = result
        return result
