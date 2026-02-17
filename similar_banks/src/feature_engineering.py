from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    numeric_features: List[str]
    categorical_features: List[str]
    missing_critical: pd.DataFrame
    data_gaps: List[str]


def _first_existing_column(df: pd.DataFrame, candidates: Any) -> Optional[str]:
    if isinstance(candidates, str):
        if candidates in df.columns:
            return candidates
        for suffix in ("_land", "_inst"):
            candidate = f"{candidates}{suffix}"
            if candidate in df.columns:
                return candidate
        return None
    if isinstance(candidates, list):
        for col in candidates:
            resolved = _first_existing_column(df, col)
            if resolved is not None:
                return resolved
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_mixed_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    yyyymmdd_mask = numeric.notna() & numeric.between(19000101, 21991231)
    if yyyymmdd_mask.any():
        as_text = numeric[yyyymmdd_mask].round().astype("Int64").astype(str)
        parsed.loc[yyyymmdd_mask] = pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
    return parsed


def _resolve_column(
    df: pd.DataFrame, column: Optional[str], prefer_suffixes: tuple[str, ...] = ()
) -> Optional[str]:
    if not column:
        return None
    if column in df.columns:
        return column
    for suffix in prefer_suffixes:
        candidate = f"{column}_{suffix}"
        if candidate in df.columns:
            return candidate
    return None


def _series_or_default(
    df: pd.DataFrame,
    column: Optional[str],
    default: Any,
    prefer_suffixes: tuple[str, ...] = (),
) -> pd.Series:
    resolved = _resolve_column(df, column, prefer_suffixes=prefer_suffixes)
    if resolved:
        return df[resolved]
    return pd.Series(default, index=df.index)


def compute_growth_features(
    land_history: pd.DataFrame, mappings: Dict[str, Any], logger: logging.Logger
) -> pd.DataFrame:
    ids = mappings["identifiers"]
    dates = mappings["dates"]
    financial = mappings["financial"]

    bank_id_col = ids["bank_id"]
    report_date_col = dates["land_report_date"]
    growth_metrics = {
        "assets": financial["total_assets"],
        "deposits": financial["total_deposits"],
        "loans": financial["total_loans"],
    }

    needed_cols = [bank_id_col, report_date_col] + list(growth_metrics.values())
    hist = land_history[needed_cols].copy()
    hist[report_date_col] = _parse_mixed_date(hist[report_date_col])
    hist[bank_id_col] = pd.to_numeric(hist[bank_id_col], errors="coerce").astype("Int64")

    for col in growth_metrics.values():
        hist[col] = _to_numeric(hist[col])

    hist = hist.dropna(subset=[bank_id_col, report_date_col]).sort_values(
        [bank_id_col, report_date_col]
    )
    for label, source_col in growth_metrics.items():
        lag = hist.groupby(bank_id_col)[source_col].shift(4)
        hist[f"yoy_{label}"] = np.where(
            (lag > 0) & hist[source_col].notna(),
            (hist[source_col] / lag) - 1.0,
            np.nan,
        )

    latest_date = hist[report_date_col].max()
    if pd.isna(latest_date):
        logger.warning("Growth feature computation skipped because land report date is missing.")
        return pd.DataFrame(columns=[bank_id_col])

    window_start = latest_date - pd.DateOffset(years=3)
    rolling_window = hist.loc[
        (hist[report_date_col] > window_start) & (hist[report_date_col] <= latest_date)
    ]
    growth_cols = ["yoy_assets", "yoy_deposits", "yoy_loans"]
    growth_agg = rolling_window.groupby(bank_id_col, as_index=False)[growth_cols].mean()
    growth_agg = growth_agg.rename(
        columns={
            "yoy_assets": "avg_3y_assets_yoy_growth",
            "yoy_deposits": "avg_3y_deposits_yoy_growth",
            "yoy_loans": "avg_3y_loans_yoy_growth",
        }
    )
    return growth_agg


def engineer_bank_features(
    bank_base: pd.DataFrame,
    land_history: pd.DataFrame,
    mappings: Dict[str, Any],
    logger: logging.Logger,
) -> FeatureSet:
    ids = mappings["identifiers"]
    names = mappings["names"]
    financial = mappings["financial"]
    lending = mappings["lending_profile"]
    deposit_mix = mappings["deposit_mix"]
    institutional = mappings["institutional"]

    bank_id_col = ids["bank_id"]
    cert_col = ids["cert"]

    land_name_col = names["land_name"]
    institution_name_col = names["institution_name"]

    out = pd.DataFrame()
    out["bank_id"] = pd.to_numeric(bank_base[bank_id_col], errors="coerce").astype("Int64")
    institution_name = _series_or_default(
        bank_base, institution_name_col, np.nan, prefer_suffixes=("inst", "land")
    )
    land_name = _series_or_default(
        bank_base, land_name_col, np.nan, prefer_suffixes=("land", "inst")
    )
    out["subject_name"] = (
        institution_name.where(institution_name.notna(), land_name).fillna("UNKNOWN").astype(str)
    )
    out["cert"] = pd.to_numeric(
        _series_or_default(bank_base, cert_col, np.nan, prefer_suffixes=("inst", "land")),
        errors="coerce",
    ).astype("Int64")

    total_assets_col = financial["total_assets"]
    total_deposits_col = financial["total_deposits"]
    total_loans_col = financial["total_loans"]
    loan_to_deposit_col = financial["loan_to_deposit_ratio"]
    asset_growth_cagr_col = financial["asset_growth_cagr_5y"]
    core_deposits_per_fte_col = financial["core_deposits_per_fte"]
    asset_size_bucket_col = financial["asset_size_bucket"]

    out["total_assets"] = _to_numeric(
        _series_or_default(bank_base, total_assets_col, np.nan, prefer_suffixes=("land", "inst"))
    )
    out["total_deposits"] = _to_numeric(
        _series_or_default(bank_base, total_deposits_col, np.nan, prefer_suffixes=("land", "inst"))
    )
    out["total_loans"] = _to_numeric(
        _series_or_default(bank_base, total_loans_col, np.nan, prefer_suffixes=("land", "inst"))
    )

    loan_to_deposit_series = _to_numeric(
        _series_or_default(bank_base, loan_to_deposit_col, np.nan, prefer_suffixes=("land", "inst"))
    )
    ratio_p95 = float(loan_to_deposit_series.quantile(0.95)) if loan_to_deposit_series.notna().any() else np.nan
    if pd.notna(ratio_p95) and ratio_p95 > 2.0:
        loan_to_deposit_series = loan_to_deposit_series / 100.0
    derived_ratio = np.where(
        out["total_deposits"] > 0,
        out["total_loans"] / out["total_deposits"],
        np.nan,
    )
    out["loan_to_deposit_ratio"] = loan_to_deposit_series.fillna(pd.Series(derived_ratio, index=bank_base.index))

    out["asset_growth_cagr_5y"] = _to_numeric(
        _series_or_default(bank_base, asset_growth_cagr_col, np.nan, prefer_suffixes=("land", "inst"))
    )

    core_dep_raw = _to_numeric(
        _series_or_default(
            bank_base, core_deposits_per_fte_col, np.nan, prefer_suffixes=("land", "inst")
        )
    )
    core_col_name = (core_deposits_per_fte_col or "").upper()
    if "COREDEP" in core_col_name and "FTE" not in core_col_name:
        out["core_deposits_per_fte_ttm"] = np.where(
            out["total_deposits"] > 0, core_dep_raw / out["total_deposits"], np.nan
        )
    else:
        out["core_deposits_per_fte_ttm"] = core_dep_raw

    resolved_bucket_col = _resolve_column(
        bank_base, asset_size_bucket_col, prefer_suffixes=("land", "inst")
    )
    if resolved_bucket_col in bank_base.columns:
        out["asset_size_bucket"] = bank_base[resolved_bucket_col].fillna("UNKNOWN").astype(str)
    else:
        bins = [-np.inf, 100_000_000, 1_000_000_000, 10_000_000_000, 100_000_000_000, np.inf]
        labels = ["<=100M", "100M-1B", "1B-10B", "10B-100B", "100B+"]
        out["asset_size_bucket"] = pd.cut(out["total_assets"], bins=bins, labels=labels).astype(str)
        out["asset_size_bucket"] = out["asset_size_bucket"].replace("nan", "UNKNOWN")

    out["log_total_assets"] = np.log1p(out["total_assets"].clip(lower=0))
    out["log_total_deposits"] = np.log1p(out["total_deposits"].clip(lower=0))
    out["log_total_loans"] = np.log1p(out["total_loans"].clip(lower=0))
    out["loans_to_assets"] = np.where(
        out["total_assets"] > 0, out["total_loans"] / out["total_assets"], np.nan
    )
    out["deposits_to_assets"] = np.where(
        out["total_assets"] > 0, out["total_deposits"] / out["total_assets"], np.nan
    )

    data_gaps: List[str] = []
    for label, candidate_cols in lending.items():
        feature_name = f"loan_mix_{label}_pct"
        source_col = _first_existing_column(bank_base, candidate_cols)
        if source_col is None:
            data_gaps.append(f"Missing lending profile column for {label}")
            out[feature_name] = np.nan
            continue
        source = _to_numeric(bank_base[source_col])
        source_lower = source_col.lower()
        if "pct" in source_lower or "percent" in source_lower:
            source_p95 = float(source.quantile(0.95)) if source.notna().any() else np.nan
            if pd.notna(source_p95) and source_p95 > 1.5:
                source = source / 100.0
            out[feature_name] = source
        else:
            out[feature_name] = np.where(out["total_loans"] > 0, source / out["total_loans"], np.nan)

    for label, candidate_cols in deposit_mix.items():
        feature_name = f"deposit_mix_{label}_pct"
        source_col = _first_existing_column(bank_base, candidate_cols)
        if source_col is None:
            data_gaps.append(f"Missing deposit mix column for {label}")
            out[feature_name] = np.nan
            continue
        source = _to_numeric(bank_base[source_col])
        source_lower = source_col.lower()
        if "pct" in source_lower or "percent" in source_lower:
            source_p95 = float(source.quantile(0.95)) if source.notna().any() else np.nan
            if pd.notna(source_p95) and source_p95 > 1.5:
                source = source / 100.0
            out[feature_name] = source
        else:
            out[feature_name] = np.where(
                out["total_deposits"] > 0, source / out["total_deposits"], np.nan
            )

    if "deposit_mix_business_pct" in out.columns and out["deposit_mix_business_pct"].isna().all():
        if "deposit_mix_retail_pct" in out.columns and out["deposit_mix_retail_pct"].notna().any():
            out["deposit_mix_business_pct"] = 1.0 - out["deposit_mix_retail_pct"]

    if "deposit_mix_retail_pct" in out.columns and out["deposit_mix_retail_pct"].isna().all():
        if "deposit_mix_business_pct" in out.columns and out["deposit_mix_business_pct"].notna().any():
            out["deposit_mix_retail_pct"] = 1.0 - out["deposit_mix_business_pct"]

    charter_col = institutional["charter_type"]
    holding_col = institutional["holding_company_rssd"]
    specialty_col = institutional["specialty_group"]

    out["charter_type"] = _series_or_default(
        bank_base, charter_col, "UNKNOWN", prefer_suffixes=("inst", "land")
    ).fillna("UNKNOWN").astype(str)
    out["holding_company_rssd"] = pd.to_numeric(
        _series_or_default(bank_base, holding_col, np.nan, prefer_suffixes=("inst", "land")),
        errors="coerce",
    )
    out["has_holding_company"] = (
        out["holding_company_rssd"].notna() & (out["holding_company_rssd"] > 0)
    ).astype(int)
    out["specialty_group"] = _series_or_default(
        bank_base, specialty_col, "UNKNOWN", prefer_suffixes=("inst", "land")
    ).fillna("UNKNOWN").astype(str)

    growth = compute_growth_features(land_history=land_history, mappings=mappings, logger=logger)
    growth = growth.rename(columns={ids["bank_id"]: "bank_id"})
    out = out.merge(growth, how="left", on="bank_id")

    if out["avg_3y_assets_yoy_growth"].notna().sum() == 0:
        data_gaps.append("Unable to compute 3-year average YoY growth from historical data.")

    critical_raw = ["total_assets", "total_deposits", "total_loans"]
    missing_critical = out.loc[out[critical_raw].isna().any(axis=1), ["bank_id", "subject_name"] + critical_raw]

    candidate_numeric = [
        "log_total_assets",
        "log_total_deposits",
        "log_total_loans",
        "loans_to_assets",
        "deposits_to_assets",
        "loan_to_deposit_ratio",
        "asset_growth_cagr_5y",
        "core_deposits_per_fte_ttm",
        "avg_3y_assets_yoy_growth",
        "avg_3y_deposits_yoy_growth",
        "avg_3y_loans_yoy_growth",
    ] + [c for c in out.columns if c.startswith("loan_mix_") or c.startswith("deposit_mix_")]

    numeric_features = [c for c in candidate_numeric if c in out.columns and not out[c].isna().all()]
    for col in numeric_features:
        if out[col].isna().any():
            group_med = out.groupby("asset_size_bucket")[col].transform("median")
            out[col] = out[col].fillna(group_med)
            out[col] = out[col].fillna(out[col].median())

    out = out.dropna(subset=["bank_id"]).drop_duplicates(subset=["bank_id"], keep="last").copy()
    out["bank_id"] = out["bank_id"].astype("int64")
    out["cert"] = out["cert"].astype("Int64")

    categorical_features = ["charter_type", "has_holding_company", "specialty_group"]
    logger.info("Engineered %d numeric features and %d categorical features", len(numeric_features), len(categorical_features))
    if data_gaps:
        logger.warning("Detected %d data gaps: %s", len(data_gaps), "; ".join(data_gaps))

    return FeatureSet(
        frame=out,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        missing_critical=missing_critical,
        data_gaps=data_gaps,
    )
