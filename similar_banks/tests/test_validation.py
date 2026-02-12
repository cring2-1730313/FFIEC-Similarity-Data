from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from similarity_model import gut_check_wells, spot_check


def test_spot_check_by_id_and_name() -> None:
    df = pd.DataFrame(
        [
            {
                "subject_idrssd": 1,
                "subject_name": "Wells Fargo Bank, National Association",
                "similar_rank": 1,
                "similar_idrssd": 2,
                "similar_name": "JPMorgan Chase Bank, National Association",
                "similarity_score": 0.9,
            },
            {
                "subject_idrssd": 3,
                "subject_name": "Community National Bank",
                "similar_rank": 1,
                "similar_idrssd": 4,
                "similar_name": "Regional Bank",
                "similarity_score": 0.8,
            },
        ]
    )
    by_id = spot_check(df, 1)
    assert len(by_id) == 1
    assert by_id.iloc[0]["subject_name"] == "Wells Fargo Bank, National Association"

    by_name = spot_check(df, "community")
    assert len(by_name) == 1
    assert by_name.iloc[0]["subject_idrssd"] == 3


def test_gut_check_helper_with_synthetic_sample() -> None:
    df = pd.DataFrame(
        [
            {
                "subject_idrssd": 451965,
                "subject_name": "Wells Fargo Bank, National Association",
                "similar_rank": 1,
                "similar_idrssd": 852218,
                "similar_name": "JPMorgan Chase Bank, National Association",
                "similarity_score": 0.95,
            },
            {
                "subject_idrssd": 451965,
                "subject_name": "Wells Fargo Bank, National Association",
                "similar_rank": 2,
                "similar_idrssd": 480228,
                "similar_name": "Bank of America, National Association",
                "similarity_score": 0.93,
            },
        ]
    )
    result = gut_check_wells(df)
    assert result.passed


def test_wells_gut_check_on_output_if_present() -> None:
    parquet_path = Path(__file__).resolve().parents[1] / "output" / "similar_banks.parquet"
    if not parquet_path.exists():
        pytest.skip("Output parquet not generated yet.")

    df = pd.read_parquet(parquet_path)
    result = gut_check_wells(df)
    assert result.passed, result.details
