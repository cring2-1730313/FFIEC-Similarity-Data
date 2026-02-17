from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class DataBundle:
    land_history: pd.DataFrame
    land_latest: pd.DataFrame
    institutions_latest: pd.DataFrame
    locations: pd.DataFrame
    bank_base: pd.DataFrame
    latest_land_date: pd.Timestamp
    data_profile: pd.DataFrame


def _normalize_numeric_id(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    raw = series.copy()
    parsed = pd.to_datetime(raw, errors="coerce")
    numeric = pd.to_numeric(raw, errors="coerce")
    yyyymmdd_mask = numeric.notna() & numeric.between(19000101, 21991231)
    if yyyymmdd_mask.any():
        yyyymmdd = numeric[yyyymmdd_mask].round().astype("Int64").astype(str)
        parsed.loc[yyyymmdd_mask] = pd.to_datetime(
            yyyymmdd, format="%Y%m%d", errors="coerce"
        )
    return parsed


def _normalize_datetime(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = _parse_mixed_datetime(df[column])


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file extension for {path}")


def load_source_data(data_dir: Path, mappings: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    files = mappings["files"]
    financial_filename = files.get("financial_history", files.get("land"))
    if not financial_filename:
        raise KeyError("column_mappings.yaml must define files.financial_history or files.land")
    paths = {
        "land": data_dir / financial_filename,
        "locations": data_dir / files["locations"],
        "institutions": data_dir / files["institutions"],
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source file for {label}: {path}")

    return {
        "land": _read_table(paths["land"]),
        "locations": _read_table(paths["locations"]),
        "institutions": _read_table(paths["institutions"]),
    }


def create_data_profile(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name, df in datasets.items():
        total = len(df)
        for col in df.columns:
            series = df[col]
            non_null = int(series.notna().sum())
            rows.append(
                {
                    "dataset": dataset_name,
                    "column_name": col,
                    "dtype": str(series.dtype),
                    "row_count": total,
                    "non_null_count": non_null,
                    "non_null_pct": round((non_null / total) * 100, 4) if total else 0.0,
                    "n_unique": int(series.nunique(dropna=True)),
                    "sample_value": str(series.dropna().iloc[0]) if non_null else "",
                }
            )
    return pd.DataFrame(rows)


def prepare_snapshots(
    raw_data: Dict[str, pd.DataFrame], mappings: Dict[str, Any], logger: logging.Logger
) -> DataBundle:
    land = raw_data["land"].copy()
    locations = raw_data["locations"].copy()
    institutions = raw_data["institutions"].copy()

    ids = mappings["identifiers"]
    dates = mappings["dates"]

    bank_id_col = ids["bank_id"]
    institution_rssd_col = ids["institution_rssd"]
    cert_col = ids["cert"]

    land_date_col = dates["land_report_date"]
    institution_date_col = dates["institution_report_date"]
    institution_run_date_col = dates["institution_run_date"]
    location_run_date_col = dates["location_run_date"]

    _normalize_datetime(land, land_date_col)
    _normalize_datetime(institutions, institution_date_col)
    _normalize_datetime(institutions, institution_run_date_col)
    _normalize_datetime(locations, location_run_date_col)
    _normalize_numeric_id(land, bank_id_col)
    _normalize_numeric_id(institutions, institution_rssd_col)
    _normalize_numeric_id(institutions, cert_col)
    _normalize_numeric_id(locations, cert_col)

    latest_land_date = land[land_date_col].max()
    land_latest = (
        land.loc[land[land_date_col] == latest_land_date]
        .dropna(subset=[bank_id_col])
        .sort_values(land_date_col)
        .drop_duplicates(subset=[bank_id_col], keep="last")
        .copy()
    )

    institutions_latest = (
        institutions.dropna(subset=[institution_rssd_col])
        .sort_values(
            [
                c
                for c in [institution_rssd_col, institution_date_col, institution_run_date_col]
                if c in institutions.columns
            ]
        )
        .groupby(institution_rssd_col, as_index=False)
        .tail(1)
        .copy()
    )

    bank_base = land_latest.merge(
        institutions_latest,
        how="left",
        left_on=bank_id_col,
        right_on=institution_rssd_col,
        suffixes=("_land", "_inst"),
    )

    if cert_col not in bank_base.columns:
        cert_candidates = [f"{cert_col}_inst", f"{cert_col}_land", cert_col]
        chosen_cert = next((c for c in cert_candidates if c in bank_base.columns), None)
        if chosen_cert is not None:
            bank_base[cert_col] = bank_base[chosen_cert]

    locations = locations.dropna(subset=[cert_col]).copy()
    location_cert_set = set(locations[cert_col].dropna().astype("int64"))

    join_match_count = int(bank_base[institution_rssd_col].notna().sum())
    location_match_count = int(
        bank_base[cert_col].dropna().astype("int64").isin(location_cert_set).sum()
    )
    latest_str = latest_land_date.date() if pd.notna(latest_land_date) else "UNKNOWN"
    logger.info("Latest financial report date: %s", latest_str)
    logger.info("Latest-quarter financial institutions: %d", len(land_latest))
    logger.info(
        "IDRSSD -> FED_RSSD matched: %d/%d (%.2f%%)",
        join_match_count,
        len(land_latest),
        (join_match_count / max(len(land_latest), 1)) * 100,
    )
    logger.info(
        "Financial institutions with location coverage via CERT: %d/%d (%.2f%%)",
        location_match_count,
        len(land_latest),
        (location_match_count / max(len(land_latest), 1)) * 100,
    )

    data_profile = create_data_profile(raw_data)
    return DataBundle(
        land_history=land,
        land_latest=land_latest,
        institutions_latest=institutions_latest,
        locations=locations,
        bank_base=bank_base,
        latest_land_date=latest_land_date,
        data_profile=data_profile,
    )
