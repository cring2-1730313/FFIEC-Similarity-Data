from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict

import yaml

from data_loader import load_source_data, prepare_snapshots
from feature_engineering import engineer_bank_features
from geographic_similarity import GeographicSimilarityEngine
from similarity_model import SimilarBankRecommender, gut_check_wells, spot_check


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compute similar banks lookup table")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing source CSV files",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=script_root / "config",
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_root / "output",
        help="Directory to write parquet and profiling outputs",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default="",
        help="Optional bank name substring or IDRSSD for spot-check output",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("similar_banks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    column_map_path = args.config_dir / "column_mappings.yaml"
    weight_path = args.config_dir / "feature_weights.yaml"
    mappings = _load_yaml(column_map_path)
    weights = _load_yaml(weight_path)

    logger.info("Loading source data from %s", args.data_dir)
    raw_data = load_source_data(args.data_dir, mappings)
    bundle = prepare_snapshots(raw_data, mappings, logger)

    profile_path = args.output_dir / "data_dictionary.csv"
    bundle.data_profile.to_csv(profile_path, index=False)
    logger.info("Wrote data dictionary: %s", profile_path)

    feature_set = engineer_bank_features(
        bank_base=bundle.bank_base,
        land_history=bundle.land_history,
        mappings=mappings,
        logger=logger,
    )

    missing_path = args.output_dir / "missing_critical_features.csv"
    feature_set.missing_critical.to_csv(missing_path, index=False)
    logger.info("Wrote missing-feature report: %s", missing_path)

    data_gap_path = args.output_dir / "feature_data_gaps.json"
    with data_gap_path.open("w", encoding="utf-8") as f:
        json.dump({"gaps": feature_set.data_gaps}, f, indent=2)
    logger.info("Wrote data-gap report: %s", data_gap_path)

    ids = mappings["identifiers"]
    bank_id_col = ids["bank_id"]
    institution_rssd_col = ids["institution_rssd"]
    cert_col = ids["cert"]
    bank_base = bundle.bank_base.copy()
    location_certs = set(
        bundle.locations[cert_col].dropna().astype("int64").tolist()
    )
    join_match = int(bank_base[institution_rssd_col].notna().sum())
    location_match = int(
        bank_base[cert_col].dropna().astype("int64").isin(location_certs).sum()
    )
    summary_lines = [
        "# Data Quality Summary",
        "",
        f"- Latest land report date: {bundle.latest_land_date.date()}",
        f"- Land latest institutions: {len(bundle.land_latest):,}",
        f"- Join key used (financial to metadata): `{bank_id_col}` -> `{institution_rssd_col}`",
        f"- Join coverage (`{bank_id_col}` matched): {join_match:,}/{len(bundle.land_latest):,} ({join_match / max(len(bundle.land_latest), 1) * 100:.2f}%)",
        f"- Join key used (metadata to branches): `{cert_col}` -> `{cert_col}`",
        f"- Location coverage (banks with branch records): {location_match:,}/{len(bundle.land_latest):,} ({location_match / max(len(bundle.land_latest), 1) * 100:.2f}%)",
        f"- Missing critical feature rows: {len(feature_set.missing_critical):,}",
        f"- Data gaps detected: {len(feature_set.data_gaps):,}",
    ]
    if feature_set.data_gaps:
        summary_lines.append("- Gap details:")
        for gap in feature_set.data_gaps:
            summary_lines.append(f"  - {gap}")
    summary_path = args.output_dir / "data_quality_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Wrote data quality summary: %s", summary_path)

    geo_engine = GeographicSimilarityEngine(
        locations_df=bundle.locations,
        bank_features_df=feature_set.frame,
        mappings=mappings,
        geography_config=weights["geography"],
    )
    recommender = SimilarBankRecommender(weights_config=weights)
    similar_df = recommender.compute(
        feature_set=feature_set,
        geo_engine=geo_engine,
        computed_date=date.today(),
    )

    output_path = args.output_dir / "similar_banks.parquet"
    similar_df.to_parquet(output_path, index=False)
    logger.info("Wrote similar banks parquet: %s", output_path)
    logger.info("Output rows: %d", len(similar_df))

    gut_check = gut_check_wells(similar_df)
    if gut_check.passed:
        logger.info("Gut check passed: %s", gut_check.details)
    else:
        logger.warning("Gut check failed: %s", gut_check.details)

    if args.validate:
        check_df = spot_check(similar_df, args.validate).sort_values("similar_rank")
        if check_df.empty:
            logger.warning("No subject bank matched validation query: %s", args.validate)
        else:
            logger.info("Validation sample for query '%s':", args.validate)
            logger.info(
                "\n%s",
                check_df[
                    [
                        "subject_idrssd",
                        "subject_name",
                        "similar_rank",
                        "similar_idrssd",
                        "similar_name",
                        "similarity_score",
                    ]
                ].head(15).to_string(index=False),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
