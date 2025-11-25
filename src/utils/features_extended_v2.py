# features_extended_v2.py
# -*- coding: utf-8 -*-
"""
Extended v2 feature generation utilities for the credit risk project.

Goals
-----
- Zero-conflicts: standalone file (does not modify features_extended.py).
- API-compatible with your base builder: FeatureConfig* + FeatureGenerator*.transform(X, paym).
- Vectorized, memory-aware, safe on large frames.
- Adds transitions/volatility/decay, extra ratios, logs, zero-aggregates, deltas.
- Plus: momentum, multi-decay, cross-ratios, behavioral flags, age/exposure diffs, compact interactions, optional util-band.

Usage
-----
from features import FeatureConfig, FeatureBuilder
from features_extended_v2 import (
    FeatureConfigExtendedV2, FeatureGeneratorExtendedV2
)

cfg = FeatureConfigExtendedV2(
    use_payment_seq=True,
    use_payment_transitions=True,
    use_ratios=True,
    use_outstanding_ratios=True,
    use_bucket_severity=True,
    use_interactions=True,
    use_zero_aggregates=True,
    use_logs=True,
    windows=[3,6,12,24],
    time_decay=0.88,
    verbose=True,
)
fe = FeatureGeneratorExtendedV2(cfg)
X_new = fe.transform(X_base, paym_df)   # paym_df must contain enc_paym_*
notes = fe.get_justification()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype

# --- base classes (keeps compatibility no matter how you import the module) ---
try:
    from .features import FeatureConfig, FeatureBuilder
except Exception:
    from features import FeatureConfig, FeatureBuilder  # type: ignore


# ===============================
# Config V2
# ===============================
@dataclass
class FeatureConfigExtendedV2(FeatureConfig):
    """
    Extra configuration switches and params for the v2 generator.
    """
    # NEW toggles
    use_momentum: bool = True              # late share momentum
    use_cross_ratios: bool = True          # extra ratios beyond *_to_limit
    use_behavioral_flags: bool = True      # compact business rules as flags
    use_age_exposure: bool = True          # age/exposure diffs
    use_interaction_grid: bool = True      # a couple of hand-picked interactions

    # NEW params
    momentum_windows: List[int] = field(default_factory=lambda: [6, 12, 24])
    time_decays: List[float] = field(default_factory=lambda: [0.88, 0.95, 0.70])  # multi-decay
    risk_band_bins: int = 0   # 0 = off; e.g. 10 → decile band for util proxy (leakage-aware)
    
    # Base toggles (some mirror v1 to keep familiarity)
    use_payment_seq: bool = True                  # ok/late shares, longest streaks, recency, trend
    use_ratios: bool = True                       # to-credit-limit ratios (robust, capped)
    use_bucket_severity: bool = True              # weighted delinquency buckets
    use_interactions: bool = True                 # key interactions

    # New toggles (v2)
    use_payment_transitions: bool = True          # good↔bad transitions, volatility, any-late flags, deltas
    use_outstanding_ratios: bool = True           # overdue_ratio / nextpay_ratio vs OUTSTANDING
    use_zero_aggregates: bool = True              # aggregates over is_zero_*
    use_logs: bool = True                         # log1p transforms
    use_time_decay: bool = True                   # recency-weighted "late" score

    # Params
    windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    cap_outliers: bool = True
    cap_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"_default": (1, 99)})
    eps: float = 1e-4
    time_decay: float = 0.85                      # 0<decay<1; weight recency for "late" events

    # Payment code mapping
    paym_ok_values: Iterable[int] = field(default_factory=lambda: (0, 1))
    paym_late_values: Iterable[int] = field(default_factory=lambda: (2, 3, 4, 5, 6, 7, 8, 9))

    # Optional column lists to override auto-detection
    paym_cols: Optional[List[str]] = None


# ===============================
# Generator V2
# ===============================
class FeatureGeneratorExtendedV2(FeatureBuilder):
    """
    V2 generator that builds on top of your FeatureBuilder.

    Public API mirrors FeatureBuilder:
      - transform(X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame
      - get_justification() -> Dict[str, str]
    """

    def __init__(self, cfg: FeatureConfigExtendedV2) -> None:
        super().__init__(cfg)
        self.cfg: FeatureConfigExtendedV2 = cfg  # type: ignore[assignment]

    # ---------- public API ----------
    def transform(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Base feature frame (will be copied). Must include pre_* columns if ratios are desired.
        paym : pd.DataFrame
            Payment statuses aligned by rows with X. Must contain enc_paym_* columns if
            use_payment_seq or use_payment_transitions are True.

        Returns
        -------
        pd.DataFrame
            DataFrame with extended v2 features appended.
        """
        X = X.copy()

        # 1) Sequence-derived features
        if self.cfg.use_payment_seq:
            try:
                X = self._payment_sequence_features(X, paym)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] payment_seq failed: {e}")

        if self.cfg.use_payment_transitions:
            try:
                X = self._payment_transitions_and_volatility(X, paym)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] paym_transitions failed: {e}")

        # 2) Ratios to credit limit
        if self.cfg.use_ratios:
            try:
                X = self._ratio_features_to_limit(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] ratios_to_limit failed: {e}")

        # 3) Ratios to outstanding
        if self.cfg.use_outstanding_ratios:
            try:
                X = self._ratio_features_to_outstanding(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] ratios_to_outstanding failed: {e}")

        # 4) Severity collapse
        if self.cfg.use_bucket_severity:
            try:
                X = self._bucket_severity(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] bucket_severity failed: {e}")

        # 5) Interactions
        if self.cfg.use_interactions:
            try:
                X = self._interactions(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] interactions failed: {e}")

        # 6) Zero-aggregates
        if self.cfg.use_zero_aggregates:
            try:
                X = self._zero_flag_aggregates(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] zero_aggregates failed: {e}")

        # 7) Log transforms
        if self.cfg.use_logs:
            try:
                X = self._log_transforms(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] log_transforms failed: {e}")

        # 8) NEW: payment momentum (share_K - share_K/2)
        if self.cfg.use_momentum:
            try:
                X = self._payment_momentum(X, paym)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] momentum failed: {e}")

        # 9) NEW: multi-decay recency scores
        if self.cfg.use_time_decay:
            try:
                X = self._time_decay_multi(X, paym)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] time_decay_multi failed: {e}")

        # 10) NEW: extra cross-ratios
        if self.cfg.use_cross_ratios:
            try:
                X = self._cross_ratios(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] cross_ratios failed: {e}")

        # 11) NEW: behavioral flags
        if self.cfg.use_behavioral_flags:
            try:
                X = self._behavioral_flags(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] behavioral_flags failed: {e}")

        # 12) NEW: age/exposure diffs
        if self.cfg.use_age_exposure:
            try:
                X = self._age_exposure(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] age_exposure failed: {e}")

        # 13) NEW: compact interaction grid
        if self.cfg.use_interaction_grid:
            try:
                X = self._interaction_grid(X)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] interaction_grid failed: {e}")

        # 14) NEW: optional unsupervised util band
        if self.cfg.risk_band_bins and self.cfg.risk_band_bins > 1:
            try:
                X = self._util_band(X, self.cfg.risk_band_bins)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[V2] util_band failed: {e}")

        # Final cleanup/casting
        self._cleanup_v2_inplace(X)
        if self.cfg.verbose:
            print(f"[FeatureGeneratorExtendedV2] Done. Shape: {X.shape}")
        return X

    # ---------- helpers: utils ----------
    def _order_paym_cols(self, paym: pd.DataFrame) -> List[str]:
        """Return enc_paym_* columns ordered with enc_paym_0 first, then ascending."""
        if self.cfg.paym_cols:
            cols = [c for c in self.cfg.paym_cols if c in paym.columns]
        else:
            cols = [c for c in paym.columns if c.startswith("enc_paym_")]
        if not cols:
            return []
        cols = sorted(cols, key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 10**9)
        if "enc_paym_0" in cols:
            cols.remove("enc_paym_0")
            cols.insert(0, "enc_paym_0")
        return cols

    # ---------- helpers: payment sequences ----------
    def _payment_sequence_features(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        """
        Build:
          - ok/late shares over windows
          - longest streaks (OK, LATE) within up to 24 periods
          - recency of last OK/LATE
          - short-term trend of OK share (6 or available)
        """
        cols = self._order_paym_cols(paym)
        if not cols:
            if self.cfg.verbose:
                print("[V2/payment_seq] No enc_paym_* columns; skipping.")
            return X

        mat = paym[cols].to_numpy(copy=False)
        n_rows, n_cols = mat.shape

        ok_vals = set(self.cfg.paym_ok_values)
        late_vals = set(self.cfg.paym_late_values)
        ok_bin = np.isin(mat, list(ok_vals)).astype(np.uint8)
        late_bin = np.isin(mat, list(late_vals)).astype(np.uint8)

        # shares
        for k in self.cfg.windows:
            k_eff = min(k, n_cols)
            if k_eff <= 0: 
                continue
            X[f"paym_ok_share_{k_eff}"] = ok_bin[:, :k_eff].mean(axis=1).astype(np.float32)
            self._just(f"paym_ok_share_{k_eff}", f"Share of OK payments within last {k_eff} periods.")
            X[f"paym_late_share_{k_eff}"] = late_bin[:, :k_eff].mean(axis=1).astype(np.float32)
            self._just(f"paym_late_share_{k_eff}", f"Share of late payments within last {k_eff} periods.")

        # longest streaks up to 24
        horizon = min(24, n_cols)
        if horizon > 0:
            ok_h = ok_bin[:, :horizon]
            late_h = late_bin[:, :horizon]
            X["paym_longest_ok_streak_24"] = self._longest_streak_batch(ok_h, want=1).astype(np.int16)
            self._just("paym_longest_ok_streak_24", "Longest consecutive OK streak within ~24 periods.")
            X["paym_longest_late_streak_24"] = self._longest_streak_batch(late_h, want=1).astype(np.int16)
            self._just("paym_longest_late_streak_24", "Longest consecutive LATE streak within ~24 periods.")

        # recency indexes (0=now)
        X["paym_last_late_recency"] = self._recency_from_left(late_bin).astype(np.float32)
        self._just("paym_last_late_recency", "Recency of the last late event (0=now, 1=prev, NaN=never).")
        X["paym_last_ok_recency"] = self._recency_from_left(ok_bin).astype(np.float32)
        self._just("paym_last_ok_recency", "Recency of the last OK event (0=now, 1=prev, NaN=never).")

        # trend of OK over last 6 (or available)
        k_trend = min(6, n_cols)
        if k_trend >= 2:
            ok_k = ok_bin[:, :k_trend].astype(np.float32)
            X["paym_ok_trend_6"] = self._slope_last_k(ok_k).astype(np.float32)
            self._just("paym_ok_trend_6", f"Slope of OK share over last {k_trend} periods.")

        return X

    @staticmethod
    def _longest_streak_batch(bin_mat: np.ndarray, want: int = 1) -> np.ndarray:
        m = (bin_mat == want).astype(np.int32)
        if m.shape[1] == 0:
            return np.zeros(m.shape[0], dtype=np.int32)
        streaks = np.zeros_like(m, dtype=np.int32)
        streaks[:, 0] = m[:, 0]
        for j in range(1, m.shape[1]):
            streaks[:, j] = (streaks[:, j-1] + 1) * m[:, j]
        return streaks.max(axis=1)

    @staticmethod
    def _recency_from_left(bin_mat: np.ndarray) -> np.ndarray:
        n_rows = bin_mat.shape[0]
        has_one = bin_mat.any(axis=1)
        rec = np.full(n_rows, np.nan, dtype=np.float32)
        if not has_one.any():
            return rec
        idx = bin_mat.argmax(axis=1).astype(np.int32)  # first 1 from the left
        rec[has_one] = idx[has_one].astype(np.float32)
        return rec

    @staticmethod
    def _slope_last_k(ok_mat: np.ndarray) -> np.ndarray:
        n_rows, k = ok_mat.shape
        x = np.arange(k, dtype=np.float32)
        vx = x - x.mean()
        denom = (vx * vx).sum()
        if denom <= 0:
            return np.zeros(n_rows, dtype=np.float32)
        y_mean = ok_mat.mean(axis=1)
        num = ((ok_mat - y_mean[:, None]) * vx[None, :]).sum(axis=1)
        return (num / denom).astype(np.float32)

    # ---------- helpers: transitions/volatility/decay/deltas ----------
    def _payment_transitions_and_volatility(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        cols = self._order_paym_cols(paym)
        if not cols:
            if self.cfg.verbose:
                print("[V2/paym_transitions] No enc_paym_*; skipping.")
            return X

        mat = paym[cols].to_numpy(copy=False)
        late = np.isin(mat, list(self.cfg.paym_late_values)).astype(np.uint8)

        for k in self.cfg.windows:
            k_eff = min(k, late.shape[1])
            if k_eff < 2:
                continue
            window = late[:, :k_eff]
            # transitions good↔bad
            d = np.abs(np.diff(window, axis=1))
            X[f"paym_transitions_{k_eff}"] = d.sum(axis=1).astype(np.int16)
            self._just(f"paym_transitions_{k_eff}", f"Number of good↔bad switches over last {k_eff} periods.")
            # volatility (std)
            X[f"paym_late_std_{k_eff}"] = window.astype(np.float32).std(axis=1).astype(np.float32)
            self._just(f"paym_late_std_{k_eff}", f"Volatility of late indicator over last {k_eff} periods.")
            # any late flag
            X[f"paym_any_late_{k_eff}"] = (window.sum(axis=1) > 0).astype(np.int8)
            self._just(f"paym_any_late_{k_eff}", f"Any late present within last {k_eff} periods.")

        # single-decay legacy (keep original name)
        if self.cfg.use_time_decay and late.shape[1] > 0:
            T = late.shape[1]
            weights = (self.cfg.time_decay ** np.arange(T, dtype=np.float32))[None, :]
            X["paym_late_time_decay"] = (late.astype(np.float32) * weights).sum(axis=1).astype(np.float32)
            self._just("paym_late_time_decay", f"Recency-weighted sum of late events (decay={self.cfg.time_decay}).")

        # deltas between OK shares (if produced earlier)
        def _delta(a: str, b: str, out: str):
            if a in X.columns and b in X.columns:
                X[out] = (X[a] - X[b]).astype(np.float32)
                self._just(out, f"Delta of OK share: {a} - {b} (recent improvement/deterioration).")

        _delta("paym_ok_share_3", "paym_ok_share_12", "paym_ok_share_delta_3_12")
        _delta("paym_ok_share_6", "paym_ok_share_12", "paym_ok_share_delta_6_12")

        return X

    # ---------- NEW: momentum ----------
    def _payment_momentum(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        cols = self._order_paym_cols(paym)
        if not cols:
            return X

        def _ensure_share(k: int) -> str:
            col = f"paym_late_share_{k}"
            if col in X.columns:
                return col
            # fallback — локальный расчёт
            k_eff = min(k, len(cols))
            if k_eff <= 0:
                return ""
            late_vals = set(self.cfg.paym_late_values)
            late_bin = np.isin(paym[cols[:k_eff]].to_numpy(copy=False), list(late_vals)).astype(np.uint8)
            X[col] = late_bin.mean(axis=1).astype(np.float32)
            self._just(col, f"Share of late payments within last {k_eff} periods (on-demand).")
            return col

        for k in self.cfg.momentum_windows:
            k2 = max(2, k // 2)
            c1 = _ensure_share(k)
            c0 = _ensure_share(k2)
            if c1 and c0:
                name = f"paym_late_momentum_{k}"
                X[name] = (X[c1].astype("float32") - X[c0].astype("float32"))
                self._just(name, f"Momentum of late share: share_{k} - share_{k2}.")
        return X

    # ---------- NEW: multi-decay ----------
    def _time_decay_multi(self, X: pd.DataFrame, paym: pd.DataFrame) -> pd.DataFrame:
        cols = self._order_paym_cols(paym)
        if not cols:
            return X
        late_vals = set(self.cfg.paym_late_values)
        late = np.isin(paym[cols].to_numpy(copy=False), list(late_vals)).astype(np.float32)
        n = late.shape[1]
        idx = np.arange(n, dtype=np.float32)  # 0 = most recent
        for d in self.cfg.time_decays:
            w = d ** idx
            s = (late * w[None, :]).sum(axis=1) / (w.sum() + self.cfg.eps)
            name = f"paym_late_time_decay_{str(d).replace('.','')}"
            X[name] = s.astype(np.float32)
            self._just(name, f"Recency-weighted late sum with decay={d}.")
        return X

    # ---------- helpers: ratios ----------
    def _cap_series(self, s: pd.Series, name: str) -> pd.Series:
        if not self.cfg.cap_outliers:
            return s
        p_lo, p_hi = self.cfg.cap_bounds.get(name, self.cfg.cap_bounds.get("_default", (1, 99)))
        lo, hi = np.nanpercentile(s.astype("float64"), [p_lo, p_hi])
        return s.clip(lo, hi)

    def _ratio_features_to_limit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize by credit limit; winsorize; cast to float32.
        """
        if "pre_loans_credit_limit" not in X.columns:
            if self.cfg.verbose:
                print("[V2/ratios_to_limit] pre_loans_credit_limit missing; skipping.")
            return X

        eps = self.cfg.eps
        denom = (X["pre_loans_credit_limit"].astype("float64").abs() + eps)

        mapping = {
            "pre_loans_total_overdue": "overdue_to_limit",
            "pre_loans_max_overdue_sum": "maxover_to_limit",
            "pre_loans_outstanding": "outstanding_to_limit",
            "pre_loans_next_pay_summ": "nextpay_to_limit",
        }
        for src, dst in mapping.items():
            if src not in X.columns:
                if self.cfg.verbose:
                    print(f"[V2/ratios_to_limit] {src} not found; skipping {dst}.")
                continue
            val = (X[src].astype("float64") / denom)
            X[dst] = self._cap_series(val, dst).astype(np.float32)
            self._just(dst, f"{src} normalized by credit limit (winsorized).")

        return X

    def _ratio_features_to_outstanding(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize by outstanding balance; cast to float32.
        """
        if "pre_loans_outstanding" not in X.columns:
            if self.cfg.verbose:
                print("[V2/ratios_to_outstanding] pre_loans_outstanding missing; skipping.")
            return X

        eps = self.cfg.eps
        denom = (X["pre_loans_outstanding"].astype("float64").abs() + eps)

        if "pre_loans_total_overdue" in X.columns:
            X["overdue_ratio"] = (X["pre_loans_total_overdue"].astype("float64") / denom).astype(np.float32)
            self._just("overdue_ratio", "Total overdue normalized by outstanding balance.")

        if "pre_loans_next_pay_summ" in X.columns:
            X["nextpay_ratio"] = (X["pre_loans_next_pay_summ"].astype("float64") / denom).astype(np.float32)
            self._just("nextpay_ratio", "Next payment normalized by outstanding balance.")

        return X

    # ---------- NEW: extra cross-ratios ----------
    def _cross_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        eps = self.cfg.eps
        if {"pre_loans_total_overdue","pre_loans_outstanding"}.issubset(X.columns):
            X["overdue_to_outstanding"] = (
                X["pre_loans_total_overdue"].astype("float64") /
                (X["pre_loans_outstanding"].astype("float64").abs() + eps)
            ).astype(np.float32)
            self._just("overdue_to_outstanding", "Total overdue normalized by outstanding.")

        if {"pre_loans_max_overdue_sum","pre_loans_outstanding"}.issubset(X.columns):
            X["maxover_to_outstanding"] = (
                X["pre_loans_max_overdue_sum"].astype("float64") /
                (X["pre_loans_outstanding"].astype("float64").abs() + eps)
            ).astype(np.float32)
            self._just("maxover_to_outstanding", "Max overdue sum normalized by outstanding.")

        if "outstanding_to_limit" in X.columns:
            X["credit_headroom"] = (1.0 - X["outstanding_to_limit"].astype("float32"))
            self._just("credit_headroom", "1 - outstanding_to_limit (remaining headroom).")
        return X

    # ---------- helpers: util proxy ----------
    def _util_proxy(self, X: pd.DataFrame) -> Optional[pd.Series]:
        """
        Return utilization series:
            - prefer 'pre_util' if present,
            - else use 'outstanding_to_limit' as a proxy,
            - else None.
        """
        if "pre_util" in X.columns:
            return X["pre_util"].astype("float32")
        if "outstanding_to_limit" in X.columns:
            return X["outstanding_to_limit"].astype("float32")  # proxy
        return None
    
    # ---------- helpers: buckets / severity ----------
    def _bucket_severity(self, X: pd.DataFrame) -> pd.DataFrame:
        weights = {
            "pre_loans5": 1,
            "pre_loans530": 2,
            "pre_loans3060": 3,
            "pre_loans6090": 4,
            "pre_loans90": 5,
        }
        present = [c for c in weights if c in X.columns]
        if not present:
            if self.cfg.verbose:
                print("[V2/severity] no pre_loans* bucket columns; skipping.")
            return X

        total = np.zeros(len(X), dtype=np.float64)
        for c in present:
            total += X[c].astype("float64") * weights[c]
        X["bucket_severity_score"] = total.astype(np.float32)
        self._just("bucket_severity_score", "Weighted sum of delinquency buckets (severity proxy).")

        if "pre_loans90" in X.columns:
            X["has_recent_90p"] = (X["pre_loans90"].astype("float64") > 0).astype(np.int8)
            self._just("has_recent_90p", "Indicator of 90+ delinquency presence (recent risk proxy).")
        return X

    # ---------- helpers: interactions ----------
    def _interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        u = self._util_proxy(X)
        if u is None:
            return X

        if "overdue_to_limit" in X.columns:
            X["util_x_overdue"] = u * X["overdue_to_limit"].astype("float32")
            self._just("util_x_overdue",
                        "Interaction: utilization (or proxy) × overdue_to_limit.")

        if "outstanding_to_limit" in X.columns:
            X["util_x_outstanding"] = u * X["outstanding_to_limit"].astype("float32")
            self._just("util_x_outstanding",
                       "Interaction: utilization (or proxy) × outstanding_to_limit.")
        return X

    # ---------- NEW: compact interaction grid ----------
    def _interaction_grid(self, X: pd.DataFrame) -> pd.DataFrame:
        if {"paym_last_late_recency","pre_loans_total_overdue"}.issubset(X.columns):
            X["recency_x_total_overdue"] = (
                X["paym_last_late_recency"].astype("float32") *
                X["pre_loans_total_overdue"].astype("float32")
            )
            self._just("recency_x_total_overdue", "Interaction: last late recency × total overdue.")

        if {"paym_late_share_6","outstanding_to_limit"}.issubset(X.columns):
            X["late6_x_util"] = (
                X["paym_late_share_6"].astype("float32") *
                X["outstanding_to_limit"].astype("float32")
            )
            self._just("late6_x_util", "Interaction: late share 6 × outstanding_to_limit.")
        return X

    # ---------- helpers: zero_* aggregates ----------
    def _zero_flag_aggregates(self, X: pd.DataFrame) -> pd.DataFrame:
        zero_cols = [c for c in X.columns if c.startswith("is_zero_")]
        if not zero_cols:
            return X
        z = X[zero_cols].astype("int8")
        X["zero_flags_sum"] = z.sum(axis=1).astype(np.int16)
        self._just("zero_flags_sum", "Sum of is_zero_* flags (broad 'no-issue' indicator).")

        cols_ge30 = [c for c in zero_cols if c in ("is_zero_loans3060","is_zero_loans6090","is_zero_loans90")]
        if cols_ge30:
            X["zero_no_30plus_overdue"] = (X[cols_ge30].sum(axis=1) == len(cols_ge30)).astype(np.int8)
            self._just("zero_no_30plus_overdue", "No 30+ day delinquency flags set.")
        return X

    # ---------- helpers: logs ----------
    def _log_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        def add_log1p(src: str, dst: str):
            if src in X.columns:
                X[dst] = np.log1p(X[src].astype("float64")).astype(np.float32)
                self._just(dst, f"log1p transform of {src} (stabilizes heavy tails).")

        u = self._util_proxy(X)
        if u is not None:
            X["util_log1p"] = np.log1p(u.astype("float64")).astype(np.float32)
            self._just("util_log1p", "log1p of utilization (or proxy) to stabilize tails.")
        add_log1p("pre_loans_outstanding", "outstanding_log1p")
        add_log1p("pre_loans_total_overdue", "total_overdue_log1p")
        add_log1p("pre_loans_next_pay_summ", "nextpay_log1p")
        return X

    # ---------- NEW: behavioral flags ----------
    def _behavioral_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        if "paym_last_late_recency" in X.columns:
            X["flag_recently_late"] = (X["paym_last_late_recency"].astype("float32") < 3).astype("int8")
            self._just("flag_recently_late", "1 if last late recency < 3.")

        if {"outstanding_to_limit","paym_late_share_6"}.issubset(X.columns):
            X["flag_high_util_recently"] = (
                (X["outstanding_to_limit"].astype("float32") > 0.90) &
                (X["paym_late_share_6"].astype("float32") > 0.20)
            ).astype("int8")
            self._just("flag_high_util_recently", "High utilization & recent late share > 0.2.")

        if {"outstanding_to_limit","paym_late_share_24"}.issubset(X.columns):
            X["flag_clean_but_high_util"] = (
                (X["paym_late_share_24"].astype("float32") == 0.0) &
                (X["outstanding_to_limit"].astype("float32") > 0.80)
            ).astype("int8")
            self._just("flag_clean_but_high_util", "No late in 24 but utilization > 0.8.")
        return X

    # ---------- NEW: age/exposure diffs ----------
    def _age_exposure(self, X: pd.DataFrame) -> pd.DataFrame:
        if {"pre_since_opened","pre_since_confirmed"}.issubset(X.columns):
            X["age_since_opened_minus_confirmed"] = (
                X["pre_since_opened"].astype("float32") - X["pre_since_confirmed"].astype("float32")
            )
            self._just("age_since_opened_minus_confirmed",
                       "Difference between since_opened and since_confirmed.")
        return X

    # ---------- NEW: util band (unsupervised, optional) ----------
    def _util_band(self, X: pd.DataFrame, bins: int) -> pd.DataFrame:
        u = self._util_proxy(X)
        if u is None:
            return X
        try:
            q = pd.qcut(u.clip(lower=0, upper=5), q=bins, labels=False, duplicates="drop").astype("int16")
            X[f"util_band_q{bins}"] = q
            self._just(f"util_band_q{bins}", f"Quantile band of utilization proxy (q={bins}).")
        except Exception:
            pass
        return X

    # ---------- cleanup / casting ----------
    def _cleanup_v2_inplace(self, X: pd.DataFrame) -> None:
        # Replace infs with NaNs for numeric columns
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

        float_cols = [
            # ratios to limit
            "overdue_to_limit","maxover_to_limit","outstanding_to_limit","nextpay_to_limit",
            # shares/trend/recency
            *[f"paym_ok_share_{k}" for k in self.cfg.windows],
            *[f"paym_late_share_{k}" for k in self.cfg.windows],
            "paym_ok_trend_6","paym_last_late_recency","paym_last_ok_recency",
            # severity
            "bucket_severity_score",
            # interactions
            "util_x_overdue","util_x_outstanding",
            # v2 extras
            *[f"paym_late_std_{k}" for k in self.cfg.windows],
            "paym_late_time_decay",
            "overdue_ratio","nextpay_ratio",
            "util_log1p","outstanding_log1p","total_overdue_log1p","nextpay_log1p",
            "paym_ok_share_delta_3_12","paym_ok_share_delta_6_12",
            # NEW: cross-ratios & momentum & interaction grid
            "overdue_to_outstanding","maxover_to_outstanding","credit_headroom",
            *[f"paym_late_momentum_{k}" for k in self.cfg.momentum_windows],
            "recency_x_total_overdue","late6_x_util",
            # NEW: multi-decay names
            *[f"paym_late_time_decay_{str(d).replace('.','')}" for d in self.cfg.time_decays],
            # NEW: age diff
            "age_since_opened_minus_confirmed",
        ]
        int_cols = [
            "paym_longest_ok_streak_24","paym_longest_late_streak_24",
            *[f"paym_transitions_{k}" for k in self.cfg.windows],
        ]
        small_int_cols = [
            "has_recent_90p",
            *[f"paym_any_late_{k}" for k in self.cfg.windows],
            "zero_no_30plus_overdue",
            # NEW flags
            "flag_recently_late","flag_high_util_recently","flag_clean_but_high_util",
        ]
        # optional band
        if self.cfg.risk_band_bins and self.cfg.risk_band_bins > 1:
            int_cols += [f"util_band_q{self.cfg.risk_band_bins}"]

        for c in float_cols:
            if c in X.columns:
                X[c] = X[c].astype("float32")
        for c in int_cols:
            if c in X.columns:
                X[c] = X[c].astype("int16")
        for c in small_int_cols:
            if c in X.columns:
                X[c] = X[c].astype("int8")


# ---------- utils: dtype sanitization & NaN flags ----------
def sanitize_dtypes(
    df: pd.DataFrame,
    *,
    bool_to_int8: bool = True,
    arrow_to_numpy: bool = True,
    downcast_int: bool = True,
    downcast_float: bool = True,
    category_to_codes: bool = False,
) -> pd.DataFrame:
    """
    Bring df dtypes to safe, model-friendly forms.
    - bool -> int8 (optional)
    - pyarrow-backed -> numpy
    - downcast ints/floats
    - category -> codes (optional)
    """
    out = df.copy()

    for c in out.columns:
        s = out[c]

        # 1) pyarrow extension -> numpy
        if arrow_to_numpy and str(s.dtype).endswith("[pyarrow]"):
            if pd.api.types.is_integer_dtype(s):
                out[c] = s.astype("int32")
            elif pd.api.types.is_float_dtype(s):
                out[c] = s.astype("float32")
            elif pd.api.types.is_bool_dtype(s):
                out[c] = s.astype("int8")
            else:
                # fallback: string-like → object (or keep as-is)
                out[c] = s.astype("float32", errors="ignore")

        # 2) bool -> int8
        elif bool_to_int8 and pd.api.types.is_bool_dtype(s):
            out[c] = s.astype("int8")

        elif category_to_codes and is_categorical_dtype(s):
            out[c] = s.cat.codes.astype("int32")

        # 4) numeric downcast (after conversions above)
        s2 = out[c]
        if downcast_int and pd.api.types.is_integer_dtype(s2):
            out[c] = pd.to_numeric(s2, downcast="integer")
        elif downcast_float and pd.api.types.is_float_dtype(s2):
            out[c] = pd.to_numeric(s2, downcast="float")

    return out


def add_nan_flags(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    *,
    suffix: str = "_isnan",
    dtype: str = "int8",
) -> pd.DataFrame:
    """
    For selected columns (or all numeric by default), append NaN indicator flags.
    """
    out = df.copy()
    if cols is None:
        cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in cols:
        out[f"{c}{suffix}"] = out[c].isna().astype(dtype, copy=False)
    return out


# ---------- tiny sanity test ----------
if __name__ == "__main__":
    # minimal synthetic check to ensure nothing crashes
    rng = np.random.RandomState(42)
    n = 8
    # mock base frame
    Xb = pd.DataFrame({
        "pre_loans_credit_limit": rng.randint(1_000, 10_000, size=n).astype("int32"),
        "pre_loans_outstanding": rng.randint(0, 9_000, size=n).astype("int32"),
        "pre_loans_total_overdue": rng.randint(0, 2_000, size=n).astype("int32"),
        "pre_loans_next_pay_summ": rng.randint(0, 3_000, size=n).astype("int32"),
        "pre_util": rng.rand(n).astype("float32"),
        "pre_loans90": rng.randint(0, 2, size=n).astype("int8"),
        "is_zero_loans3060": rng.randint(0, 2, size=n).astype("int8"),
        "is_zero_loans6090": rng.randint(0, 2, size=n).astype("int8"),
        "is_zero_loans90":  rng.randint(0, 2, size=n).astype("int8"),
    })
    # mock payment sequence (enc_paym_0 most recent)
    paym = pd.DataFrame({
        **{f"enc_paym_{i}": rng.randint(0, 5, size=n).astype("int8") for i in range(10)}
    })

    # sanitize
    Xb = sanitize_dtypes(Xb)

    # run generator
    cfg = FeatureConfigExtendedV2(
        windows=[3,6,10],
        time_decays=[0.9, 0.8],
        time_decay=0.9,
        verbose=True
    )
    gen = FeatureGeneratorExtendedV2(cfg)
    Xv2 = gen.transform(Xb, paym)
    print("Sanity OK. Shape:", Xv2.shape)


__all__ = ["FeatureConfigExtendedV2", "FeatureGeneratorExtendedV2"]
