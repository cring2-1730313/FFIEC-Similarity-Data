# Banking Industry Machine Learning

Similar-bank recommendation system for FDIC-insured institutions, built for quarterly refreshes and Power BI integration.

## Current Implementation

- Financial source uses `banksuite_financials_last_3y_full.parquet` (12 quarters of history).
- Geography and institution metadata use:
  - `Locations__Bank_Suite_.csv`
  - `Institutions__Bank_Suite_.csv`
- Primary joins:
  - Financial -> institution metadata: `RSSDID` -> `FED_RSSD`
  - Institution metadata -> branch geography: `CERT` -> `CERT`

## Major Updates

- Added native parquet support in loader.
- Added robust mixed date parsing (including `YYYYMMDD` integer-style report dates).
- Mapped feature engineering to Banksuite financial columns and lending profile fields.
- Enabled 3-year YoY growth feature computation from history.
- Increased emphasis on lending composition in similarity distance via configurable `loan_mix_*` weights.
- Lowered holding-company categorical weight (`holding_company: 0.25`).
- Improved `similarity_drivers` text to use specific numeric comparisons (lending mix, deposit mix, ratios, growth, geographic overlap), replacing generic fallback phrases.

## Project Structure

- `similar_banks/src`: data loading, feature engineering, geographic similarity, recommender model, orchestration script
- `similar_banks/config`: column mappings and tunable weighting config
- `similar_banks/tests`: validation helpers and gut-check tests
- `similar_banks/notebooks`: schema/data exploration notebook
- `similar_banks/output`: generated parquet and profiling artifacts

## Run

From repo root:

```powershell
python similar_banks/src/compute_similar_banks.py --data-dir . --config-dir similar_banks/config --output-dir similar_banks/output
```

Spot-check by IDRSSD:

```powershell
python similar_banks/src/compute_similar_banks.py --data-dir . --config-dir similar_banks/config --output-dir similar_banks/output --validate 63069
```

## Inspect Similarity Drivers

After computing output:

```powershell
python -c "import pandas as pd; df=pd.read_parquet('similar_banks/output/similar_banks.parquet'); out=df[df.subject_idrssd==63069].sort_values('similar_rank')[['similar_rank','similar_idrssd','similar_name','similarity_score','similarity_drivers']]; print(out.to_string(index=False))"
```

## Output Files

- `similar_banks/output/similar_banks.parquet`: final top-7 similar bank table
- `similar_banks/output/data_dictionary.csv`: profiled schema data dictionary
- `similar_banks/output/data_quality_summary.md`: join coverage and quality summary
- `similar_banks/output/missing_critical_features.csv`: missing critical feature rows
- `similar_banks/output/feature_data_gaps.json`: unresolved feature-gap report
