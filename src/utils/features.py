from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """
    Configuration for feature building.

    Parameters
    ----------
    cat_cols : list[str]
        Categorical columns in X to one-hot encode.
    ohe_prefix : str
        Prefix for OHE columns.
    paym_cols : list[str] | None
        Payment status columns in `paym` ordered left->right as time goes **backwards**
        (i.e., enc_paym_0 is the most recent). If None, they will be auto-detected as
        columns starting with 'enc_paym_'.
    verbose : bool
        If True, prints short progress messages.
    """
    cat_cols: List[str] = field(default_factory=list)
    ohe_prefix: str = "OHE"
    paym_cols: Optional[List[str]] = None
    verbose: bool = False


class FeatureBuilder:
    """
    Builds engineered features and keeps human-readable justifications.

    Usage
    -----
    cfg = FeatureConfig(cat_cols=["gender", "region"], ohe_prefix="OHE", verbose=True)
    fb = FeatureBuilder(cfg)
    X_out = fb.transform(X, paym)
    notes = fb.get_justification()
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.feature_justification: Dict[str, str] = {}

    # ---------- public API ----------

    def transform(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Base feature frame (will be copied).
        paym : pd.DataFrame
            Payment statuses aligned by rows with X. Must contain 'enc_paym_0'
            and other enc_paym_* columns if streaks are desired.

        Returns
        -------
        pd.DataFrame
            New DataFrame with engineered features appended.
        """
        X = X.copy()

        # 1) Most recent payment status
        if "enc_paym_0" in paym.columns:
            X["paym_last_status"] = paym["enc_paym_0"].astype("float32")
            self._just(
                "paym_last_status",
                "Most recent payment status: captures current discipline.",
            )

        # 2) Length of the last on-time streak (count leading zeros)
        ordered_cols = self._order_paym_cols(paym)
        if ordered_cols:
            X["paym_last_clean_streak"] = (
                paym[ordered_cols].apply(self._last_clean_streak, axis=1).astype("int16")
            )
            self._just(
                "paym_last_clean_streak",
                "Length of last on-time streak: momentum of discipline.",
            )

        # 3) One-Hot encoding for configured categorical columns
        for col in self.cfg.cat_cols:
            if col in X.columns:
                # keep ints as Int32 nullable; otherwise use original dtype
                series = (
                    X[col].astype("Int32", errors="ignore")
                    if pd.api.types.is_integer_dtype(X[col])
                    else X[col]
                )
                dummies = pd.get_dummies(
                    series,
                    prefix=f"{self.cfg.ohe_prefix}_{col}",
                    drop_first=False,
                    dtype="uint8",
                )
                X = pd.concat([X, dummies], axis=1)
                self._just(
                    f"OHE[{col}]",
                    "One-Hot encoding for tree/linear models that benefit from explicit categories.",
                )

        # 4) Technical cleanups / casting
        self._cleanup_inplace(X)

        if self.cfg.verbose:
            print(f"[FeatureBuilder] Done. Shape: {X.shape}")

        return X

    def get_justification(self) -> Dict[str, str]:
        """Return dict of feature_name -> human-readable justification."""
        return dict(self.feature_justification)

    # ---------- helpers ----------

    def _just(self, name: str, why: str) -> None:
        self.feature_justification[name] = why
        if self.cfg.verbose:
            print(f"[just] {name}: {why}")

    def _order_paym_cols(self, paym: pd.DataFrame) -> List[str]:
        """
        Put 'enc_paym_0' first, then the rest of enc_paym_* that are present in paym.
        If cfg.paym_cols is provided, it is respected and filtered by actual columns.
        """
        if self.cfg.paym_cols is not None:
            base = [c for c in self.cfg.paym_cols if c in paym.columns]
        else:
            base = [c for c in paym.columns if c.startswith("enc_paym_")]

        if not base:
            return []

        ordered = [c for c in base if c == "enc_paym_0"] + [c for c in base if c != "enc_paym_0"]
        return [c for c in ordered if c in paym.columns]

    @staticmethod
    def _last_clean_streak(row: pd.Series) -> int:
        """
        Count leading zeros in a row (0 = on-time), stopping at the first non-zero.
        """
        cnt = 0
        for v in row.values:
            if v == 0:
                cnt += 1
            else:
                break
        return cnt

    @staticmethod
    def _cleanup_inplace(X: pd.DataFrame) -> None:
        """
        Replace inf with NaN and cast certain columns to float32 for memory efficiency.
        """
        # Ratios that might explode to inf
        for c in ["debt_to_limit", "overdue_to_limit", "maxoverdue_to_limit", "loan_term_ratio"]:
            if c in X.columns:
                X[c] = X[c].replace([np.inf, -np.inf], np.nan).astype("float32")

        # Other numeric signals we want as float32
        for c in [
            "since_ratio",
            "till_close_gap",
            "total_delays",
            "serious_delay_ratio",
            "paym_good_count",
            "paym_bad_count",
            "paym_last_status",
            "paym_last_clean_streak",
        ]:
            if c in X.columns:
                X[c] = X[c].astype("float32")


# ---- Backward-compatible alias so notebooks can use FeatureGenerator ----
FeatureGenerator = FeatureBuilder

__all__ = ["FeatureConfig", "FeatureBuilder", "FeatureGenerator"]
