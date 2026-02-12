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


def _normalize_datetime(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")


def load_source_data(data_dir: Path, mappings: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    files = mappings["files"]
    paths = {
        "land": data_dir / files["land"],
        "locations": data_dir / files["locations"],
        "institutions": data_dir / files["institutions"],
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source file for {label}: {path}")

    return {
        "land": pd.read_csv(paths["land"], low_memory=False),
        "locations": pd.read_csv(paths["locations"], low_memory=False),
        "institutions": pd.read_csv(paths["institutions"], low_memory=False),
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
        .sort_values([institution_rssd_col, institution_date_col, institution_run_date_col])
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

    locations = locations.dropna(subset=[cert_col]).copy()
    location_cert_set = set(locations[cert_col].dropna().astype("int64"))

    join_match_count = int(bank_base[institution_rssd_col].notna().sum())
    location_match_count = int(
        bank_base[cert_col].dropna().astype("int64").isin(location_cert_set).sum()
    )
    logger.info("Latest land report date: %s", latest_land_date.date())
    logger.info("Land latest banks: %d", len(land_latest))
    logger.info(
        "IDRSSD -> FED_RSSD matched: %d/%d (%.2f%%)",
        join_match_count,
        len(land_latest),
        (join_match_count / max(len(land_latest), 1)) * 100,
    )
    logger.info(
        "Land banks with location coverage via CERT: %d/%d (%.2f%%)",
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
