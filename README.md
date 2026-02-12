# Banking Industry Machine Learning

Similar-bank recommendation system for FDIC-insured institutions, built for quarterly refreshes and Power BI integration.

## Project Structure

- `similar_banks/src`: data loading, feature engineering, geographic similarity, recommender model, orchestration script
- `similar_banks/config`: column mappings and tunable weighting config
- `similar_banks/tests`: validation helpers and gut-check tests
- `similar_banks/notebooks`: schema/data exploration notebook

## Run

From repo root:

```powershell
python similar_banks/src/compute_similar_banks.py --data-dir . --config-dir similar_banks/config --output-dir similar_banks/output
```

Spot-check a bank:

```powershell
python similar_banks/src/compute_similar_banks.py --data-dir . --config-dir similar_banks/config --output-dir similar_banks/output --validate 75633
```

## Notes

- Latest implementation uses `IDRSSD -> FED_RSSD` and `CERT -> CERT` joins.
- Some requested features (detailed lending mix and retail/business deposit mix) are logged as data gaps when not present in source columns.
