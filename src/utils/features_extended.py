# features_extended.py
# -*- coding: utf-8 -*-
"""
Extended feature generation utilities for the credit risk project.

Design goals:
- Keep API compatible with existing features.py (FeatureConfig/FeatureBuilder).
- Add sequence-based payment behavior, robust ratios, severity, and interactions.
- Be safe on large datasets: vectorize where possible, avoid heavy Python loops.
- Provide human-readable justifications via _just(), like in the base builder.

Usage
-----
from features import FeatureConfig, FeatureGenerator  # existing
from features_extended import FeatureConfigExtended, FeatureGeneratorExtended

cfg_ext = FeatureConfigExtended(
    use_payment_seq=True,
    use_ratios=True,
    use_bucket_severity=True,
    use_interactions=True,
    windows=[3,6,12,24],
    paym_ok_values=(0, 1),
    paym_late_values=(2, 3, 4, 5, 6, 7, 8, 9),
    verbose=True,
)
fg_ext = FeatureGeneratorExtended(cfg_ext)

X_ext = fg_ext.transform(X_base, paym)  # paym aligned by rows with X_base
notes = fg_ext.get_justification()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple
import numpy as np
import pandas as pd

# Import base classes from your existing module
# Assumes this file is placed next to features.py or importable from it.
try:
    from .features import FeatureConfig, FeatureBuilder
except ImportError:
    from features import FeatureConfig, FeatureBuilder


# -------------------------------
# Config
# -------------------------------

@dataclass
class FeatureConfigExtended(FeatureConfig):
    """
    Extended configuration for feature building.

    Parameters
    ----------
    use_payment_seq : bool
        Generate sequence-based payment features from enc_paym_0..N.
    use_ratios : bool
        Generate robust ratios normalized by credit limit (with capping).
    use_bucket_severity : bool
        Collapse delinquency buckets into severity scores and recent flags.
    use_interactions : bool
        Generate key interaction features (e.g., util * overdue_to_limit).
    windows : list[int]
        Rolling windows for sequence summaries (e.g., [3,6,12,24]).
    cap_outliers : bool
        Apply winsorization (percentile capping) to sensitive ratios.
    cap_bounds : dict[str, tuple[float,float]]
        Per-column percentile cap bounds; "_default" used when column not specified.
        Example: {"_default": (1,99), "overdue_to_limit": (1, 98)}
    eps : float
        Small epsilon to avoid division by zero.
    paym_ok_values : iterable[int]
        Encoded payment values to treat as "OK/on-time".
    paym_late_values : iterable[int]
        Encoded payment values to treat as "late".
    """

    use_payment_seq: bool = True
    use_ratios: bool = True
    use_bucket_severity: bool = True
    use_interactions: bool = True

    windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])

    cap_outliers: bool = True
    cap_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"_default": (1, 99)})
    eps: float = 1e-4

    paym_ok_values: Iterable[int] = field(default_factory=lambda: (0, 1))
    paym_late_values: Iterable[int] = field(default_factory=lambda: (2, 3, 4, 5, 6, 7, 8, 9))


# -------------------------------
# Generator
# -------------------------------

class FeatureGeneratorExtended(FeatureBuilder):
    """
    Extended generator that builds on top of FeatureBuilder.

    Public API mirrors FeatureBuilder:
      - transform(X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame
      - get_justification() -> Dict[str, str]
    """

    def __init__(self, cfg: FeatureConfigExtended) -> None:
        super().__init__(cfg)  # keeps feature_justification, verbose, etc.
        # typed self.cfg for editors
        self.cfg: FeatureConfigExtended = cfg  # type: ignore[assignment]

    # ---------- public API ----------

    def transform(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Base feature frame (will be copied). Must include pre_* columns if ratios are desired.
        paym : pd.DataFrame
            Payment statuses aligned by rows with X. Must contain enc_paym_* columns if
            use_payment_seq=True.

        Returns
        -------
        pd.DataFrame
            New DataFrame with extended features appended.
        """
        X = X.copy()

        # Payment sequence features
        if self.cfg.use_payment_seq:
            try:
                X = self._payment_sequence_features(X, paym)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[FeatureGeneratorExtended] payment_seq failed: {e}")

        # Robust ratios (normalized by credit limit)
        if self.cfg.use_ratios:
            try:
                X = self._ratio_features(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[FeatureGeneratorExtended] ratios failed: {e}")

        # Bucket severity / recent flags
        if self.cfg.use_bucket_severity:
            try:
                X = self._bucket_severity(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[FeatureGeneratorExtended] bucket_severity failed: {e}")

        # Interactions
        if self.cfg.use_interactions:
            try:
                X = self._interactions(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[FeatureGeneratorExtended] interactions failed: {e}")

        # Final cleanup/casting like base builder does (extends base list)
        self._cleanup_extended_inplace(X)

        if self.cfg.verbose:
            print(f"[FeatureGeneratorExtended] Done. Shape: {X.shape}")

        return X

    # ---------- helpers: payment sequences ----------

    def _order_paym_cols_from_df(self, df: pd.DataFrame) -> List[str]:
        """Return enc_paym_* columns ordered with enc_paym_0 first, then ascending."""
        if self.cfg.paym_cols is not None:
            cols = [c for c in self.cfg.paym_cols if c in df.columns]
        else:
            cols = sorted([c for c in df.columns if c.startswith("enc_paym_")],
                          key=lambda s: int(s.split("_")[-1]))
        if not cols:
            return []
        # Move enc_paym_0 to front explicitly if present
        cols = [c for c in cols if c != "enc_paym_0"]
        cols.insert(0, "enc_paym_0") if "enc_paym_0" in df.columns else None
        return [c for c in cols if c in df.columns]

    def _payment_sequence_features(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        """
        Add behavior features built from enc_paym_0..N:
          - shares of OK/late over windows
          - longest streaks (OK, late) over horizon 24 if available
          - recency of last OK/late
          - simple trend of OK share over last 6 months
        """
        cols = self._order_paym_cols_from_df(paym)
        if not cols:
            if self.cfg.verbose:
                print("[payment_seq] No enc_paym_* columns found; skipping.")
            return X

        mat = paym[cols].to_numpy(copy=False)
        n_rows, n_cols = mat.shape

        # Build OK and LATE binaries
        ok_vals = set(self.cfg.paym_ok_values)
        late_vals = set(self.cfg.paym_late_values)

        ok_bin = np.isin(mat, list(ok_vals)).astype(np.uint8)
        late_bin = np.isin(mat, list(late_vals)).astype(np.uint8)

        # Shares over windows
        for k in self.cfg.windows:
            k_eff = min(k, n_cols)
            if k_eff <= 0:
                continue
            # Note: by spec enc_paym_0 is most recent -> take first k columns
            ok_share = ok_bin[:, :k_eff].mean(axis=1).astype(np.float32)
            late_share = late_bin[:, :k_eff].mean(axis=1).astype(np.float32)

            X[f"paym_ok_share_{k_eff}"] = ok_share
            self._just(f"paym_ok_share_{k_eff}",
                       f"Share of OK payments within last {k_eff} periods.")
            X[f"paym_late_share_{k_eff}"] = late_share
            self._just(f"paym_late_share_{k_eff}",
                       f"Share of late payments within last {k_eff} periods.")

        # Longest streaks across up to 24 (or available)
        horizon = min(24, n_cols)
        if horizon > 0:
            ok_h = ok_bin[:, :horizon]
            late_h = late_bin[:, :horizon]

            X["paym_longest_ok_streak_24"] = self._longest_streak_batch(ok_h, want=1).astype(np.int16)
            self._just("paym_longest_ok_streak_24", "Longest consecutive OK streak within ~24 periods.")

            X["paym_longest_late_streak_24"] = self._longest_streak_batch(late_h, want=1).astype(np.int16)
            self._just("paym_longest_late_streak_24", "Longest consecutive LATE streak within ~24 periods.")

        # Recency: distance to most recent 1 from the left (col 0 is most recent)
        # We compute from left-to-right: recency = index of first 1; if none -> NaN.
        X["paym_last_late_recency"] = self._recency_from_left(late_bin).astype(np.float32)
        self._just("paym_last_late_recency", "Recency of the last late event (0=now, 1=prev, NaN=never).")

        X["paym_last_ok_recency"] = self._recency_from_left(ok_bin).astype(np.float32)
        self._just("paym_last_ok_recency", "Recency of the last OK event (0=now, 1=prev, NaN=never).")

        # Trend: OK share slope over last 6 months (or available)
        k_trend = min(6, n_cols)
        if k_trend >= 2:
            ok_k = ok_bin[:, :k_trend].astype(np.float32)
            X["paym_ok_trend_6"] = self._slope_last_k(ok_k).astype(np.float32)
            self._just("paym_ok_trend_6", f"Slope of OK share over last {k_trend} periods (trend of discipline).")

        return X

    @staticmethod
    def _longest_streak_batch(bin_mat: np.ndarray, want: int = 1) -> np.ndarray:
        """
        Compute longest consecutive streak of 'want' for each row of a {0,1} matrix.
        Vectorized over columns using cumulative tricks.

        bin_mat : shape (n_rows, n_cols) of {0,1}
        """
        # Convert to consecutive run lengths: reset where value != want
        m = (bin_mat == want).astype(np.int32)
        # We scan per row; vectorization trick:
        # When m[i,j]==1, streak = 1 + streak at previous col; else 0.
        # We'll do it with a simple loop over columns (fast in NumPy, O(n_rows*n_cols))
        streaks = np.zeros_like(m, dtype=np.int32)
        if m.shape[1] == 0:
            return m.sum(axis=1)  # zeros

        streaks[:, 0] = m[:, 0]
        for j in range(1, m.shape[1]):
            streaks[:, j] = (streaks[:, j-1] + 1) * m[:, j]

        return streaks.max(axis=1)

    @staticmethod
    def _recency_from_left(bin_mat: np.ndarray) -> np.ndarray:
        """
        For each row, return index of the left-most 1 (i.e., most recent event).
        If a row has no 1s, return NaN.
        """
        n_rows, n_cols = bin_mat.shape
        # mask rows that have at least one 1
        has_one = bin_mat.any(axis=1)
        rec = np.full(n_rows, np.nan, dtype=np.float32)
        if not has_one.any():
            return rec
        # argmax of reversed cumulative? Easier: find first 1 per row with argmax on a mask where 1s remain 1s.
        # We'll use np.argmax on a boolean array where True is 1, but we must ensure there's at least one True.
        # Because np.argmax returns 0 when all False; guard with has_one.
        first_one_idx = (bin_mat.argmax(axis=1)).astype(np.int32)
        rec[has_one] = first_one_idx[has_one].astype(np.float32)
        # If the first column isn't 1 but there is a 1 later, argmax returns that later index (correct).
        return rec

    @staticmethod
    def _slope_last_k(ok_mat: np.ndarray) -> np.ndarray:
        """
        Compute slope of OK values over columns (0..k-1) for each row:
            slope = cov(x,y)/var(x),  x = [0,1,...,k-1], y in {0,1}, per-row.
        ok_mat : shape (n_rows, k)
        """
        n_rows, k = ok_mat.shape
        x = np.arange(k, dtype=np.float32)
        x_mean = x.mean()
        vx = (x - x_mean)
        denom = (vx ** 2).sum()
        if denom <= 0:
            return np.zeros(n_rows, dtype=np.float32)
        y_mean = ok_mat.mean(axis=1)
        # cov(x,y) = E[(x-xm)(y-ym)] = sum((x-xm)*(y-ym))/k
        # We'll compute numerator row-wise
        num = (ok_mat - y_mean[:, None]) * vx[None, :]
        num = num.sum(axis=1)
        slope = num / denom
        return slope.astype(np.float32)

    # ---------- helpers: ratios ----------

    def _ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ratios normalized by credit limit; winsorize heavy tails; cast to float32.
        """
        def _cap_series(s: pd.Series, name: str) -> pd.Series:
            if not self.cfg.cap_outliers:
                return s
            p_lo, p_hi = self.cfg.cap_bounds.get(name, self.cfg.cap_bounds.get("_default", (1, 99)))
            lo, hi = np.nanpercentile(s.astype("float64"), [p_lo, p_hi])
            return s.clip(lo, hi)

        eps = self.cfg.eps
        cols_needed = {
            "pre_loans_total_overdue": "overdue_to_limit",
            "pre_loans_max_overdue_sum": "maxover_to_limit",
            "pre_loans_outstanding": "outstanding_to_limit",
            "pre_loans_next_pay_summ": "nextpay_to_limit",
        }

        if "pre_loans_credit_limit" not in X.columns:
            if self.cfg.verbose:
                print("[ratios] pre_loans_credit_limit not found; skipping ratios.")
            return X

        denom = (X["pre_loans_credit_limit"].astype("float64").abs() + eps)

        for src, dst in cols_needed.items():
            if src not in X.columns:
                if self.cfg.verbose:
                    print(f"[ratios] {src} not found; skipping {dst}.")
                continue
            val = X[src].astype("float64") / denom
            val = _cap_series(val, dst).astype("float32")
            X[dst] = val
            self._just(dst, f"{src} normalized by credit limit (capped).")

        return X

    # ---------- helpers: buckets / severity ----------

    def _bucket_severity(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted collapse of delinquency buckets; simple recent flag.
        """
        weights = {
            "pre_loans5": 1,
            "pre_loans530": 2,
            "pre_loans3060": 3,
            "pre_loans6090": 4,
            "pre_loans90": 5,
        }

        present = [c for c in weights.keys() if c in X.columns]
        if not present:
            if self.cfg.verbose:
                print("[bucket_severity] no pre_loans* bucket columns found; skipping.")
            return X

        # severity score
        total = np.zeros(len(X), dtype=np.float64)
        for c in present:
            total += X[c].astype("float64") * weights[c]
        X["bucket_severity_score"] = total.astype("float32")
        self._just("bucket_severity_score", "Weighted sum of delinquency buckets (severity proxy).")

        # simple recent 90+ proxy (if column exists)
        if "pre_loans90" in X.columns:
            X["has_recent_90p"] = (X["pre_loans90"].astype("float64") > 0).astype("int8")
            self._just("has_recent_90p", "Indicator of 90+ delinquency presence (recent risk proxy).")

        return X

    # ---------- helpers: interactions ----------

    def _interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Key interactions to expose non-linearity for simple models.
        """
        if "pre_util" in X.columns and "overdue_to_limit" in X.columns:
            X["util_x_overdue"] = (X["pre_util"].astype("float32") * X["overdue_to_limit"].astype("float32"))
            self._just("util_x_overdue", "Interaction: utilization × overdue_to_limit.")

        if "pre_util" in X.columns and "outstanding_to_limit" in X.columns:
            X["util_x_outstanding"] = (X["pre_util"].astype("float32") * X["outstanding_to_limit"].astype("float32"))
            self._just("util_x_outstanding", "Interaction: utilization × outstanding_to_limit.")

        return X

    # ---------- cleanup / casting ----------

    def _cleanup_extended_inplace(self, X: pd.DataFrame) -> None:
        """
        Extend base cleanup to include newly created columns.
        """
        # Base cleanup handles some names already; we extend the list
        # Replace inf with NaN
        for c in X.columns:
            if pd.api.types.is_float_dtype(X[c]) or pd.api.types.is_integer_dtype(X[c]):
                for c in X.select_dtypes(include=[np.number]).columns:
                    X[c] = X[c].replace([np.inf, -np.inf], np.nan)

        # Cast types for extended features (memory-aware)
        float_cols = [
            # ratios
            "overdue_to_limit", "maxover_to_limit", "outstanding_to_limit", "nextpay_to_limit",
            # payment shares/trend/recency
            *[f"paym_ok_share_{k}" for k in self.cfg.windows],
            *[f"paym_late_share_{k}" for k in self.cfg.windows],
            "paym_ok_trend_6",
            "paym_last_late_recency", "paym_last_ok_recency",
            # severity
            "bucket_severity_score",
            # interactions
            "util_x_overdue", "util_x_outstanding",
        ]
        int_cols = [
            "paym_longest_ok_streak_24",
            "paym_longest_late_streak_24",
        ]
        small_int_cols = [
            "has_recent_90p",
        ]

        for c in float_cols:
            if c in X.columns:
                X[c] = X[c].astype("float32")
        for c in int_cols:
            if c in X.columns:
                X[c] = X[c].astype("int16")
        for c in small_int_cols:
            if c in X.columns:
                X[c] = X[c].astype("int8")


__all__ = ["FeatureConfigExtended", "FeatureGeneratorExtended"]
