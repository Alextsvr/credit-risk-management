# src/utils/read_parquet_safe.py
from __future__ import annotations

import warnings
from typing import Optional, List, Union
from pathlib import Path

import pandas as pd

# Try importing pyarrow; if unavailable, fall back to pandas-only mode
try:
    import pyarrow as pa  # noqa: F401  # kept for type consistency / future use
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except Exception:
    _HAVE_PYARROW = False


PathLike = Union[str, Path, bytes]


def read_parquet_safe(
    path: PathLike,
    columns: Optional[List[str]] = None,
    sample_n: Optional[int] = None,
    prefer_arrow_dtypes: bool = True,
) -> pd.DataFrame:
    """
    Safe and convenient Parquet reader for pandas.

    Parameters
    ----------
    path : str | Path | bytes
        Path to a .parquet file (local or remote, e.g., s3://, gs:// if fsspec is available).
    columns : list[str] | None
        Optional list of column names to read selectively.
    sample_n : int | None
        If provided, returns the first `sample_n` rows while avoiding reading the entire file
        when possible. With pyarrow, attempts to read only the first row group.
    prefer_arrow_dtypes : bool
        If True, converts to pandas using Arrow-backed dtypes
        (pandas>=2.0: dtype_backend="pyarrow"), which provides nullable dtypes
        and better memory efficiency.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with safe dtypes.

    Notes
    -----
    - With pyarrow installed, reading is typically faster and more memory-friendly.
    - When pyarrow is not available, pandas will read the full file; `sample_n` is then applied via head().
    """
    path = str(path)

    # Fast path via pyarrow (recommended for large files)
    if _HAVE_PYARROW:
        pqf = None
        # ParquetFile may fail for partitioned directory datasets — that's OK, we'll skip sampling then.
        try:
            pqf = pq.ParquetFile(path)
        except Exception:
            pqf = None  # fall back to read_table directly

        # Lightweight sampling via the first row group (only when ParquetFile succeeded)
        if pqf is not None and sample_n and sample_n > 0:
            try:
                if pqf.metadata and pqf.metadata.num_row_groups > 0:
                    tbl = pqf.read_row_group(0, columns=columns)
                    if prefer_arrow_dtypes:
                        try:
                            return tbl.to_pandas(types_mapper=pd.ArrowDtype)[:sample_n].copy()
                        except TypeError:
                            # Older pandas versions w/o types_mapper support
                            return tbl.to_pandas()[:sample_n].copy()
            except Exception as e:
                warnings.warn(
                    f"Row-group sampling failed ({type(e).__name__}: {e}). "
                    "Falling back to full read + head()."
                )

        # Full/partial read with pyarrow (works for files and directory datasets)
        tbl = pq.read_table(path, columns=columns)
        if prefer_arrow_dtypes:
            try:
                return tbl.to_pandas(types_mapper=pd.ArrowDtype)
            except TypeError:
                # Older pandas fallback
                return tbl.to_pandas()
        return tbl.to_pandas()

    # Fallback: pandas-only engine (slower for large files)
    if sample_n and sample_n > 0:
        warnings.warn(
            "pyarrow not available; reading full file with pandas and slicing head(sample_n)."
        )
    try:
        if prefer_arrow_dtypes:
            # pandas >= 2.0 supports dtype_backend="pyarrow"
            try:
                df = pd.read_parquet(path, columns=columns, dtype_backend="pyarrow")
                return df.head(sample_n) if sample_n else df
            except TypeError:
                # Older pandas — no dtype_backend argument
                pass

        df = pd.read_parquet(path, columns=columns)
        return df.head(sample_n) if sample_n else df

    except Exception as e:
        raise RuntimeError(
            "Failed to read parquet. Install pyarrow (preferred) "
            "or ensure pandas has a supported engine.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e


if __name__ == "__main__":
    # Minimal CLI for quick manual checks:
    #   python src/utils/read_parquet_safe.py path/to/file.parquet --n 5 --no-arrow
    import argparse

    parser = argparse.ArgumentParser(description="Quick test for read_parquet_safe")
    parser.add_argument("path", type=str, help="Path to .parquet file")
    parser.add_argument("--cols", type=str, nargs="*", default=None, help="Columns to read")
    parser.add_argument("--n", type=int, default=None, help="Sample first N rows")
    parser.add_argument(
        "--no-arrow",
        action="store_true",
        help="Disable Arrow-backed dtypes (prefer_arrow_dtypes=False)",
    )
    args = parser.parse_args()

    df_ = read_parquet_safe(
        args.path,
        columns=args.cols,
        sample_n=args.n,
        prefer_arrow_dtypes=not args.no_arrow,
    )
    print(df_.head())
    print("\nShape:", df_.shape)
    print("\nDtypes:\n", df_.dtypes)
