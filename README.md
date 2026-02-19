# Banking Industry Machine Learning

## Executive Summary

This project builds a **precomputed Similar Banks engine** for U.S. FDIC-insured institutions, designed for quarterly refresh and downstream BI consumption.

From a business (strategy consulting) lens, the tool solves a common decision bottleneck: teams often compare banks using inconsistent “peer sets” built manually. This model creates a **repeatable, explainable, and scalable peer-definition layer** so product, strategy, risk, and relationship teams are working from the same benchmark universe.

The output is a ranked, explainable top-10 peer list for every bank, with drivers that describe *why* banks are similar.

## Why This Is Valuable

### 1) Better Decisions, Faster
- Replaces ad hoc peer selection with systematic recommendations.
- Reduces analyst cycle time for peer benchmarking and board-ready comparisons.

### 2) More Consistent Storytelling
- Standardizes “apples-to-apples” comparisons in executive, investor, and client conversations.
- Produces traceable similarity drivers (size, lending mix, deposit mix, growth, footprint overlap).

### 3) Stronger Commercial and Risk Use Cases
- Supports territory planning, competitive intelligence, and cross-sell prioritization.
- Enables more context-aware risk and performance benchmarking across comparable institutions.

### 4) Operationally Scalable
- Precomputed parquet output is compatible with Databricks/Power BI.
- Quarterly recompute aligns with call report cadence and supports controlled model governance.

## What the Tool Produces

`similar_banks/output/similar_banks.parquet` with:
- `subject_idrssd`
- `subject_name`
- `similar_rank` (1-10)
- `similar_idrssd`
- `similar_name`
- `similarity_score`
- `similarity_drivers`
- `computed_date`

Peer benchmarking output (standalone from similarity scoring):
- `banksuite_financials_last_3y_full_PeerBenchmarks.parquet`
- Adds peer-comparison columns with `pb_` prefix for each metric, including:
  - `pb_<metric>_value`
  - `pb_<metric>_p25`
  - `pb_<metric>_median`
  - `pb_<metric>_p75`
  - `pb_<metric>_delta_to_median`
  - `pb_<metric>_status`
  - `pb_<metric>_lag_flag`
  - `pb_<metric>_peer_percentile_rank_pct`
  - `pb_<metric>_effective_peer_count`

## Current Data Inputs

- Financial history (12 quarters): `banksuite_financials_last_3y_full.parquet`
- Branch geography: `Locations__Bank_Suite_.csv`
- Institution metadata: `Institutions__Bank_Suite_.csv`

Primary joins:
- Financial -> institution metadata: `RSSDID` -> `FED_RSSD`
- Institution metadata -> branch geography: `CERT` -> `CERT`

## Methodology (Current)

### Feature Groups
- Size and balance sheet structure (assets, deposits, loans, ratios)
- Lending profile mix (CRE, C&I, residential, consumer, agricultural, construction)
- Deposit mix (retail/business)
- Growth trajectory (3Y YoY averages + CAGR-derived signals where available)
- Institutional attributes (charter/holding company/specialty)
- Geographic footprint similarity (CBSA overlap + concentration + footprint scale)

### Similarity Scoring

Composite score combines:
- Financial similarity
- Geographic similarity
- Categorical similarity

Weights are configured in `similar_banks/config/feature_weights.yaml`.

## Peer Benchmarking Extension

`similar_banks/src/compute_peer_percentiles.py` computes metric-level peer benchmarks
using the precomputed peer groups in `similar_banks/output/similar_banks.parquet`.

Design choices:
- Uses latest-quarter snapshot for benchmark calculations.
- Leaves non-latest quarters blank by default (`--all-dates` can override).
- Applies direction-aware lag logic:
  - `higher_better`
  - `lower_better`
  - `target_range` (for loan-to-deposit ratio)
- Computes `peer_percentile_rank_pct` as a direction-adjusted rank from 0 to 100
  (higher percentile means stronger standing for that metric).
- Uses normalized FFIEC `EEFFR` for efficiency ratio benchmarking.
- Keeps mapped peer count as `pb_peer_count`; also stores metric-valid sample count as
  `pb_<metric>_effective_peer_count`.
- Amount-based `pb_` value/percentile/delta fields are scaled from FFIEC thousands to full dollars.

Current metric set includes:
- Efficiency Ratio
- Net Interest Margin
- Loan to Deposit Ratio
- Loan Loss Provision
- Net Operating Revenue
- Return on Assets
- Return on Equity
- Total Deposits
- Total Loans and Leases
- Deposits per FTE
- Revenue per FTE
- Assets
- Liabilities
- Earning Assets
- Net Income
- Net Interest Income
- Net Interest Expense
- Premises Fixed Assets / Assets
- Non-Interest Expense
- Non-Interest Expense % of Total Expenses

## Recent Enhancements

### 1) Asset-Size Candidate Band (Option 1)

Added a pre-ranking size filter so peers are selected from a size-comparable universe before final scoring.

Config:
- `asset_size_filter.enabled: true`
- `asset_size_filter.min_ratio: 0.33`
- `asset_size_filter.max_ratio: 3.0`
- `asset_size_filter.backfill_out_of_band: true`

This prevents mismatches where a very large bank is paired with a materially smaller bank unless the candidate pool requires controlled backfill.

### 2) Currency Unit Correction in Drivers

Similarity drivers now format FFIEC-style values correctly as dollar amounts (thousands scaled to full dollars), so size statements appear correctly (for example, `$96.5B` instead of `$96.5M`).

### 3) Improved CBSA Footprint Logic

Geographic score now blends:
- weighted market overlap (share-based Jaccard)
- concentration similarity (HHI-based)
- market-count similarity (`min(cbsa_a, cbsa_b) / max(cbsa_a, cbsa_b)`)

Broad-footprint language is only shown when counts are reasonably comparable.

## Project Structure

- `similar_banks/src`: loaders, feature engineering, geo similarity, model, orchestration
- `similar_banks/config`: column mapping and weighting configuration
- `similar_banks/tests`: validation helpers
- `similar_banks/notebooks`: schema/data exploration notebook
- `similar_banks/output`: generated artifacts

## Run

From repo root:

```powershell
python similar_banks/src/compute_similar_banks.py --data-dir . --config-dir similar_banks/config --output-dir similar_banks/output
```

Spot-check by IDRSSD:

```powershell
python similar_banks/src/compute_similar_banks.py --data-dir . --config-dir similar_banks/config --output-dir similar_banks/output --validate 63069
```

Inspect similarity drivers for a bank:

```powershell
python -c "import pandas as pd; df=pd.read_parquet('similar_banks/output/similar_banks.parquet'); out=df[df.subject_idrssd==63069].sort_values('similar_rank')[['similar_rank','similar_idrssd','similar_name','similarity_score','similarity_drivers']]; print(out.to_string(index=False))"
```

Generate peer benchmark columns (keeps source parquet unchanged):

```powershell
python similar_banks/src/compute_peer_percentiles.py --base-file banksuite_financials_last_3y_full.parquet --similarity-file similar_banks/output/similar_banks.parquet --output-file banksuite_financials_last_3y_full_PeerBenchmarks.parquet
```

## Output and Quality Artifacts

- `similar_banks/output/similar_banks.parquet`: final recommendation table
- `similar_banks/output/data_dictionary.csv`: profiled schema dictionary
- `similar_banks/output/data_quality_summary.md`: join coverage and key diagnostics
- `similar_banks/output/missing_critical_features.csv`: rows with missing critical inputs
- `similar_banks/output/feature_data_gaps.json`: unresolved feature gaps
- `banksuite_financials_last_3y_full_PeerBenchmarks.parquet`: source financials plus `pb_` peer benchmark columns

## Validation and Governance

- Built-in gut check for major bank relationships (for example, Wells/JPM/BofA pattern check).
- Config-driven weighting and threshold logic supports controlled tuning over time.
- Quarterly refresh cadence supports stable reporting and auditability.
