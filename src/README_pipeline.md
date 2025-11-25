# pipeline.py — Credit Risk Project (v3)

This module provides a reusable sklearn pipeline for the credit risk model.  
It supports loading a trained pipeline, running predictions on new data, and reading metadata.

---

## Features
- Load or save a serialized sklearn pipeline (`.pkl`)
- Run predictions on `.parquet` or `.csv` input files
- Include ID columns in prediction output
- Read metadata from `07_pipeline_summary.json`
- Command-line interface (CLI) for batch inference

---

## Example (CLI)

```bash
python src/pipeline.py \
  --input data/new_clients.parquet \
  --output reports/predictions_new.csv \
  --pipeline artifacts/final_model_pipeline_v3.pkl
```

## Example (Python import)

```python
from pipeline import load_pipeline, predict
import pandas as pd

pipe = load_pipeline("artifacts/final_model_pipeline_v3.pkl")
df = pd.read_parquet("data/sample.parquet")
preds = predict(pipe, df, id_cols=["id", "rn"])
print(preds.head())
```

## Output

- `predictions.csv` — predictions with probabilities or labels  
- `07_pipeline_summary.json` — metadata with model details and metrics
