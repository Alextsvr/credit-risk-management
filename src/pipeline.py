# -*- coding: utf-8 -*-
"""
pipeline.py
Reusable sklearn pipeline for the Credit Risk project (v3).

Features:
- Load/save trained pipeline (.pkl)
- Run predictions on new data (.parquet or .csv)
- Read metadata (07_pipeline_summary.json)
- CLI for batch inference
"""

# ====================================================
# Imports
# ====================================================
import argparse
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from sklearn.pipeline import Pipeline


# ====================================================
# 1. Load / Save Utilities
# ====================================================

def load_pipeline(path: Union[str, Path]) -> Pipeline:
    """
    Load a serialized sklearn pipeline.

    Parameters
    ----------
    path : str | Path
        Path to .pkl file.

    Returns
    -------
    Pipeline
        Loaded sklearn pipeline object.
    """
    path = Path(path)
    assert path.exists(), f"Pipeline file not found: {path}"
    pipeline = joblib.load(path)
    print(f"Loaded pipeline → {path.name}")
    return pipeline


def save_pipeline(pipeline: Pipeline, path: Union[str, Path]) -> None:
    """
    Save a sklearn pipeline to pickle.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline.
    path : str | Path
        Destination .pkl path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved pipeline → {path.name}")


# ====================================================
# 2. Prediction Interface
# ====================================================

def predict(
    pipeline: Pipeline,
    data: pd.DataFrame,
    proba: bool = True,
    id_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate predictions using the loaded pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Trained sklearn pipeline.
    data : pd.DataFrame
        Input data for inference.
    proba : bool, default=True
        If True, return probability of class=1.
    id_cols : list[str] | None, default=None
        Optional ID columns to include in output (e.g. ['id', 'rn']).

    Returns
    -------
    pd.DataFrame
        Predictions DataFrame (with id columns if provided).
    """
    preds = (
        pipeline.predict_proba(data)[:, 1] if proba else pipeline.predict(data)
    )
    out = pd.DataFrame({"pred_proba" if proba else "pred_label": preds})

    if id_cols:
        for col in id_cols:
            if col in data.columns:
                out[col] = data[col].values
        cols = id_cols + [c for c in out.columns if c not in id_cols]
        out = out[cols]

    return out


# ====================================================
# 3. Metadata Utility
# ====================================================

def read_metadata(json_path: Union[str, Path]) -> dict:
    """
    Read pipeline metadata from JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to metadata JSON file.

    Returns
    -------
    dict
        Metadata dictionary.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(
        f"Metadata loaded | trained_at={meta.get('timestamp')} "
        f"| ROC-AUC self-check={meta.get('roc_auc_selfcheck', 0):.3f}"
    )
    return meta


# ====================================================
# 4. Inference on Parquet/CSV
# ====================================================

def run_inference_on_parquet(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    pipeline_path: Union[str, Path],
    id_cols: Optional[list[str]] = None,
    proba: bool = True,
) -> Path:
    """
    Run inference using a trained pipeline on a parquet or csv file.

    Parameters
    ----------
    input_path : str | Path
        Path to input .parquet or .csv file.
    output_path : str | Path
        Destination path for predictions (.csv).
    pipeline_path : str | Path
        Path to serialized sklearn pipeline (.pkl).
    id_cols : list[str] | None
        ID columns to include in predictions.
    proba : bool, default=True
        Whether to output probability instead of class labels.

    Returns
    -------
    Path
        Path to saved predictions file.
    """
    input_path = Path(input_path)
    assert input_path.exists(), f"Input file not found: {input_path}"

    # Load data
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported input format. Use .parquet or .csv.")

    print(f"Loaded input data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Load pipeline
    pipeline = load_pipeline(pipeline_path)

    # Run predictions
    preds = predict(pipeline, df, proba=proba, id_cols=id_cols)

    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return output_path


# ====================================================
# 5. CLI Interface
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using a trained sklearn pipeline."
    )
    parser.add_argument("--input", required=False, help="Path to input .parquet or .csv file")
    parser.add_argument("--output", required=False, help="Path to save predictions (.csv)")
    parser.add_argument("--pipeline", default="artifacts/final_model_pipeline_v3.pkl", help="Path to trained pipeline .pkl")
    parser.add_argument("--id-cols", nargs="+", default=["id", "rn"], help="ID columns to include in predictions")
    parser.add_argument("--label", action="store_true", help="Return class labels instead of probabilities")

    args = parser.parse_args()

    # Example usage when executed directly
    if args.input and args.output:
        run_inference_on_parquet(
            input_path=args.input,
            output_path=args.output,
            pipeline_path=args.pipeline,
            id_cols=args.id_cols,
            proba=not args.label
        )
    else:
        # Default quick test (load metadata and preview)
        ARTIFACTS_DIR = Path("artifacts")
        PIPE_PATH     = ARTIFACTS_DIR / "final_model_pipeline_v3.pkl"
        META_PATH     = Path("reports") / "07_pipeline_summary.json"

        pipe = load_pipeline(PIPE_PATH)
        meta = read_metadata(META_PATH)

        PRED_PATH = ARTIFACTS_DIR / "pipeline_predictions_v3.csv"
        if PRED_PATH.exists():
            df = pd.read_csv(PRED_PATH)
            print(f"Loaded saved predictions ({df.shape[0]} rows, {df.shape[1]} cols).")
        else:
            print("No saved prediction CSV found for quick test.")