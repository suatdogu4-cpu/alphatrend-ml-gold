"""AlphaTrend ML GOLD v1.2 — research and backtesting ONLY; no live orders.

Original AlphaTrend algorithm © KivancOzbilgic, MPL-2.0:
https://mozilla.org/MPL/2.0/
Developer of this research adaptation: Suat DOĞU.

Reproduces the original AlphaTrend recursive core and its Pine BUY/SELL cross
logic. Exact TradingView parity still requires cross-platform OHLCV checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, log_loss, precision_score, recall_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASS_VALUES = np.array([-1, 0, 1], dtype=int)
CLASS_NAMES = {-1: "ASAGI", 0: "BELIRSIZ", 1: "YUKARI"}
FEATURES = [
    # v1.0 technical features (preserved)
    "rsi", "mfi", "atr_pct", "width_pct", "width_ratio", "bb_width",
    "donchian_width", "alpha_slope", "center_distance_atr", "return_1",
    "return_3", "volume_ratio", "candle_body_atr", "upper_wick_atr",
    "lower_wick_atr", "range_10_atr", "hour_sin", "hour_cos",
    # v1.1: Original AlphaTrend center line, bands, Pine BUY/SELL logic
    "at_above", "at_below", "at_side", "at_distance_step", "at_trend_up",
    "at_trend_down", "at_slope_2", "at_slope_5", "at_flat_bars",
    "at_buy_raw", "at_sell_raw", "at_buy_valid", "at_sell_valid",
    "at_buy_confirmed", "at_sell_confirmed", "at_last_signal",
    "at_signal_age", "at_buy_count_10", "at_sell_count_10",
    "at_price_cross_up", "at_price_cross_down", "at_candle_above",
    "at_candle_below", "at_low_cross_up", "at_high_cross_down",
    "at_zone", "at_upper1_break", "at_lower1_break", "at_center_slope_sign",
    "squeeze_duration", "pre_narrow_streak", "width_change_3", "range_5_atr", "return_5",
    "rsi_delta_3", "volume_trend_3", "close_location_range",
    "mtf1_above", "mtf1_trend_up", "mtf1_last_signal", "mtf1_slope",
    "mtf2_above", "mtf2_trend_up", "mtf2_last_signal", "mtf2_slope",
    "mtf_alignment",
]


def rsi_wilder(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)

    def rma(series):
        x = series.to_numpy(dtype=float)
        y = np.full(len(x), np.nan)
        if len(x) <= length:
            return pd.Series(y, index=series.index)
        y[length] = np.nanmean(x[1:length + 1])
        for t in range(length + 1, len(x)):
            y[t] = (y[t - 1] * (length - 1) + x[t]) / length
        return pd.Series(y, index=series.index)

    ru, rd = rma(up), rma(down)
    out = 100 - 100 / (1 + ru / rd.replace(0, np.nan))
    return out.mask((rd == 0) & (ru > 0), 100).mask((rd == 0) & (ru == 0), 50)


def money_flow_index(df: pd.DataFrame, length: int) -> pd.Series:
    typical = (df.high + df.low + df.close) / 3
    flow = typical * df.volume
    pos = flow.where(typical > typical.shift(), 0.0).rolling(length, min_periods=length).sum()
    neg = flow.where(typical < typical.shift(), 0.0).rolling(length, min_periods=length).sum()
    out = 100 * pos / (pos + neg).replace(0, np.nan)
    return out.mask((neg == 0) & (pos > 0), 100).fillna(50)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {"datetime": "time", "date": "time", "timestamp": "time", "open time": "time", "vol": "volume"}
    df.rename(columns={c: rename.get(c, c) for c in df.columns}, inplace=True)
    required = {"time", "open", "high", "low", "close", "volume"}
    if len(df.columns) != len(set(df.columns)):
        raise ValueError("CSV'de yinelenen sütun isimleri var.")
    if not required.issubset(df.columns):
        raise ValueError("CSV gerekli sütunlar: time,open,high,low,close,volume")
    col = df.time
    if pd.api.types.is_numeric_dtype(col):
        nonnull = col.dropna()
        if nonnull.empty:
            raise ValueError("CSV zaman sütunu boş.")
        unit = "ms" if nonnull.abs().median() > 1e11 else "s"
        df["time"] = pd.to_datetime(col, unit=unit, utc=True, errors="coerce")
    else:
        df["time"] = pd.to_datetime(col, utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(required)).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(df) < 300:
        raise ValueError(f"En az 300 kapanmış mum gerekli; gelen {len(df)}.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Fiyatlar pozitif olmalı.")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
            (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Tutarsız OHLC fiyatları var.")
    if (df.volume < 0).any():
        raise ValueError("Negatif hacim var.")
    return df


def bars_since(flag: pd.Series) -> pd.Series:
    """Equivalent to Pine ta.barssince: na before the first true."""
    out = np.full(len(flag), np.nan)
    last = None
    for i, v in enumerate(flag.fillna(False).to_numpy(dtype=bool)):
        if v:
            last = i
        if last is not None:
            out[i] = i - last
    return pd.Series(out, index=flag.index)


def running_true_length(flag: pd.Series) -> pd.Series:
    out = np.zeros(len(flag), dtype=int)
    for i, val in enumerate(flag.fillna(False).to_numpy(dtype=bool)):
        out[i] = (out[i - 1] + 1 if i else 1) if val else 0
    return pd.Series(out, index=flag.index)


def make_features(df: pd.DataFrame, period=14, multiplier=1.0, band_mult=0.5,
                  sqz_len=20, lookback=30, sqz_factor=0.85, use_rsi=False):
    d = df.copy()
    prev = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - prev).abs(), (d.low - prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(period, min_periods=period).mean()  # original SMA(true range)
    d["rsi"] = rsi_wilder(d.close, period)
    d["mfi"] = money_flow_index(d, period)
    if (d.volume == 0).all() and not use_rsi:
        raise ValueError("Hacim tamamen sıfır; orijinal AlphaTrend no-volume seçeneği için --use-rsi verin.")
    gate = d.rsi if use_rsi else d.mfi

    # Original AlphaTrend recursive stop logic. Pine initialization: 0.0.
    alpha = np.zeros(len(d))
    lo, hi = d.low.to_numpy(), d.high.to_numpy()
    atr_np, gate_np = d.atr.to_numpy(), gate.to_numpy()
    for i in range(len(d)):
        old = alpha[i - 1] if i else 0.0
        if not np.isfinite(atr_np[i]):
            alpha[i] = old
        elif gate_np[i] >= 50:
            alpha[i] = max(old, lo[i] - atr_np[i] * multiplier)
        else:
            alpha[i] = min(old, hi[i] + atr_np[i] * multiplier)
    d["alpha"] = alpha
    d["step_size"] = d.atr * band_mult
    for n in (1, 2, 3, 4):
        d[f"upper{n}"] = d.alpha + d.step_size * n
        d[f"lower{n}"] = d.alpha - d.step_size * n
    d["width_pct"] = (d.upper4 - d.lower4) / d.alpha.abs().replace(0, np.nan) * 100
    width_ma = d.width_pct.rolling(sqz_len, min_periods=sqz_len).mean()
    d["width_ratio"] = d.width_pct / width_ma.replace(0, np.nan)
    narrowest = d.width_pct.rolling(lookback, min_periods=lookback).min()
    d["squeeze"] = ((d.width_pct < width_ma * sqz_factor) & (d.width_pct <= narrowest * 1.05)).fillna(False)
    d["event"] = d["squeeze"] & ~d["squeeze"].shift(fill_value=False)
    d["squeeze_duration"] = running_true_length(d["squeeze"])
    # Known at time t; unlike future squeeze length, this causes no target leakage.
    d["pre_narrow_streak"] = running_true_length(d.width_ratio < 1.0).clip(upper=100)
    d["width_change_3"] = (d.width_pct - d.width_pct.shift(3)) / d.width_pct.shift(3).replace(0, np.nan)

    d["atr_pct"] = d.atr / d.close * 100
    d["alpha_slope"] = (d.alpha - d.alpha.shift(3)) / d.atr.replace(0, np.nan)
    d["center_distance_atr"] = (d.close - d.alpha) / d.atr.replace(0, np.nan)
    for k in (1, 3, 5):
        d[f"return_{k}"] = d.close.pct_change(k, fill_method=None) * 100
    std20 = d.close.rolling(20, min_periods=20).std()
    d["bb_width"] = 4 * std20 / d.close.rolling(20, min_periods=20).mean() * 100
    d["donchian_width"] = (d.high.rolling(20).max() - d.low.rolling(20).min()) / d.close * 100
    d["volume_ratio"] = d.volume / d.volume.rolling(20).mean().replace(0, np.nan)
    d["volume_trend_3"] = d.volume / d.volume.shift(3).replace(0, np.nan)
    if use_rsi and (d.volume == 0).all():
        # When source has no volume, neutral placeholders avoid dropping every event.
        d["volume_ratio"] = 1.0
        d["volume_trend_3"] = 1.0
    d["candle_body_atr"] = (d.close - d.open) / d.atr.replace(0, np.nan)
    d["upper_wick_atr"] = (d.high - d[["open", "close"]].max(axis=1)) / d.atr.replace(0, np.nan)
    d["lower_wick_atr"] = (d[["open", "close"]].min(axis=1) - d.low) / d.atr.replace(0, np.nan)
    d["range_10_atr"] = (d.high.rolling(10).max() - d.low.rolling(10).min()) / d.atr.replace(0, np.nan)
    d["range_5_atr"] = (d.high.rolling(5).max() - d.low.rolling(5).min()) / d.atr.replace(0, np.nan)
    d["rsi_delta_3"] = d.rsi - d.rsi.shift(3)
    d["close_location_range"] = (d.close - d.low) / (d.high - d.low).replace(0, np.nan)
    hour = d.time.dt.hour + d.time.dt.minute / 60
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # AlphaTrend line status, distance, four-band location and transitions.
    d["at_above"] = (d.close > d.alpha).astype(int)
    d["at_below"] = (d.close < d.alpha).astype(int)
    d["at_side"] = d.at_above - d.at_below
    d["at_distance_step"] = (d.close - d.alpha) / d.step_size.replace(0, np.nan)
    d["at_zone"] = d.at_distance_step.clip(lower=-5, upper=5)
    d["at_trend_up"] = (d.alpha > d.alpha.shift(2)).astype(int)
    d["at_trend_down"] = (d.alpha < d.alpha.shift(2)).astype(int)
    d["at_slope_2"] = (d.alpha - d.alpha.shift(2)) / d.atr.replace(0, np.nan)
    d["at_slope_5"] = (d.alpha - d.alpha.shift(5)) / d.atr.replace(0, np.nan)
    d["at_center_slope_sign"] = np.sign(d.at_slope_2)
    d["at_flat_bars"] = bars_since(d.alpha.diff().fillna(0).ne(0)).clip(upper=100).fillna(100)

    # Pine: ta.crossover(AlphaTrend, AlphaTrend[2]) / ta.crossunder(...)
    d["at_buy_raw"] = (d.alpha > d.alpha.shift(2)) & (d.alpha.shift(1) <= d.alpha.shift(3))
    d["at_sell_raw"] = (d.alpha < d.alpha.shift(2)) & (d.alpha.shift(1) >= d.alpha.shift(3))
    buy_age, sell_age = bars_since(d.at_buy_raw), bars_since(d.at_sell_raw)
    prior_buy_age = bars_since(d.at_buy_raw.shift(1, fill_value=False))
    prior_sell_age = bars_since(d.at_sell_raw.shift(1, fill_value=False))
    # Original BUY gate: O1 > K2 ; original SELL gate: O2 > K1
    d["at_buy_valid"] = d.at_buy_raw & (prior_buy_age > sell_age)
    d["at_sell_valid"] = d.at_sell_raw & (prior_sell_age > buy_age)
    # Original confirmed alerts: buySignal[1] and O1[1] > K2 (similarly sell).
    d["at_buy_confirmed"] = d.at_buy_raw.shift(1, fill_value=False) & (prior_buy_age.shift(1) > sell_age)
    d["at_sell_confirmed"] = d.at_sell_raw.shift(1, fill_value=False) & (prior_sell_age.shift(1) > buy_age)
    d["at_buy_count_10"] = d.at_buy_valid.astype(int).rolling(10, min_periods=1).sum()
    d["at_sell_count_10"] = d.at_sell_valid.astype(int).rolling(10, min_periods=1).sum()
    # Last *valid* signal direction and age, including the event's current close.
    last_side, last_idx = 0, None
    last_sides, last_ages = [], []
    for i, (buy, sell) in enumerate(zip(d.at_buy_valid, d.at_sell_valid)):
        if buy:
            last_side, last_idx = 1, i
        elif sell:
            last_side, last_idx = -1, i
        last_sides.append(last_side)
        last_ages.append(min(i - last_idx, 200) if last_idx is not None else 200)
    d["at_last_signal"] = last_sides
    d["at_signal_age"] = last_ages

    def cross_up(a, b):
        return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)

    def cross_down(a, b):
        return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False)

    d["at_price_cross_up"] = cross_up(d.close, d.alpha)
    d["at_price_cross_down"] = cross_down(d.close, d.alpha)
    d["at_candle_above"] = (d.low > d.alpha).astype(int)
    d["at_candle_below"] = (d.high < d.alpha).astype(int)
    d["at_low_cross_up"] = cross_up(d.low, d.alpha)
    d["at_high_cross_down"] = cross_down(d.high, d.alpha)
    d["at_upper1_break"] = cross_up(d.close, d.upper1)
    d["at_lower1_break"] = cross_down(d.close, d.lower1)
    for column in ("at_buy_raw", "at_sell_raw", "at_buy_valid", "at_sell_valid",
                   "at_buy_confirmed", "at_sell_confirmed", "at_price_cross_up",
                   "at_price_cross_down", "at_low_cross_up", "at_high_cross_down",
                   "at_upper1_break", "at_lower1_break"):
        d[column] = d[column].astype(int)
    return d


def augment_mtf_features(d: pd.DataFrame, tf="5m", period=14, multiplier=1.0,
                         band_mult=0.5, sqz_len=20, lookback=30, sqz_factor=0.85,
                         use_rsi=False):
    """Causal HTF context from fully closed higher-timeframe candles only.

    CSV timestamps represent the *opening* time of each closed input candle.
    A 5m candle opened at 10:10 closes at 10:15, when its 15m aggregate
    10:00..10:15 is first known. Partial 15m/60m groups are discarded.
    No 1m or 3m candles are hallucinated from 5m inputs.
    """
    if not tf.lower().endswith("m") or not tf[:-1].isdigit():
        raise ValueError("Dakika cinsinden mum periyodu gerekli (1m,3m,5m,15m).")
    base = int(tf[:-1])
    mapping = {1: (5, 15), 3: (15, 60), 5: (15, 60), 15: (60, 240)}
    if base not in mapping:
        raise ValueError("MTF için 1m, 3m, 5m, 15m destekleniyor.")
    d = d.copy()
    src = d.set_index("time")[["open", "high", "low", "close", "volume"]]
    d["known_time"] = d.time + pd.Timedelta(minutes=base)
    targets = mapping[base]
    for k, minutes in enumerate(targets, start=1):
        group = src.resample(f"{minutes}min", label="right", closed="left", origin="start_day")
        higher = group.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        higher["bar_count"] = group.size()
        higher = higher.loc[higher.bar_count == minutes // base].drop(columns="bar_count")
        higher = higher.dropna().reset_index().rename(columns={"time": "known_time"})
        if len(higher) < 40:
            raise ValueError(f"{minutes}m üst zaman için yeterli tam kapanmış mum yok.")
        high_d = higher.rename(columns={"known_time": "time"})
        high_d = make_features(high_d, period=period, multiplier=multiplier, band_mult=band_mult,
                               sqz_len=sqz_len, lookback=lookback, sqz_factor=sqz_factor,
                               use_rsi=use_rsi)
        subset = high_d[["time", "at_above", "at_trend_up", "at_last_signal", "alpha_slope"]].rename(
            columns={"time": f"htf{k}_known_time", "at_above": f"mtf{k}_above",
                     "at_trend_up": f"mtf{k}_trend_up", "at_last_signal": f"mtf{k}_last_signal",
                     "alpha_slope": f"mtf{k}_slope"})
        # Tolerance prevents using Friday's stale HTF snapshot after a weekend.
        d = pd.merge_asof(d.sort_values("known_time"), subset.sort_values(f"htf{k}_known_time"),
                          left_on="known_time", right_on=f"htf{k}_known_time",
                          direction="backward", tolerance=pd.Timedelta(minutes=minutes * 2))
        d = d.drop(columns=[f"htf{k}_known_time"])
    d["mtf_alignment"] = (d.at_side * (d.mtf1_above * 2 - 1) +
                           d.at_side * (d.mtf2_above * 2 - 1)) / 2
    return d.drop(columns=["known_time"]).sort_values("time").reset_index(drop=True), targets


def label_events(d: pd.DataFrame, horizon=12, barrier_atr=1.5, min_events=35, tf="5m"):
    valid = []
    unit = tf.lower().strip()
    if not unit.endswith("m") or not unit[:-1].isdigit() or int(unit[:-1]) < 1:
        raise ValueError("Bu sürümde veri periyodu 1m, 3m, 5m veya 15m gibi dakika cinsinden olmalı.")
    expected = pd.Timedelta(minutes=int(unit[:-1]))
    for ix, row in d.loc[d.event].iterrows():
        if ix + horizon >= len(d) or not np.isfinite(row.atr) or row.atr <= 0:
            continue
        # A 12-bar horizon may otherwise jump overnight/weekends due to data gaps.
        stamps = d.time.iloc[ix:ix + horizon + 1]
        if not stamps.diff().iloc[1:].eq(expected).all():
            continue
        if not np.isfinite(row[FEATURES].to_numpy(dtype=float)).all():
            continue
        upper = float(row.close + row.atr * barrier_atr)
        lower = float(row.close - row.atr * barrier_atr)
        outcome, hit_bars, ambiguous = 0, horizon, False
        for offset in range(1, horizon + 1):
            j = ix + offset
            u = d.high.iat[j] >= upper
            l = d.low.iat[j] <= lower
            if u and l:
                ambiguous = True  # Intrabar order unknown, never guess.
                hit_bars = offset
                break
            if u or l:
                outcome = 1 if u else -1
                hit_bars = offset
                break
        valid.append((ix, outcome, hit_bars, ambiguous))
    if len(valid) < min_events:
        raise ValueError(f"Yeterli daralma yok: {len(valid)} olay, en az {min_events} gerekir. Daha uzun gerçek veri kullanın.")
    indices = [v[0] for v in valid]
    # Deduplication prevents sklearn/narwhals DuplicateError (v1.0 bug).
    cols = list(dict.fromkeys(["time", "close", "alpha", "atr", "step_size", "width_pct",
                               "upper1", "upper2", "upper3", "upper4", "lower1",
                               "lower2", "lower3", "lower4"] + FEATURES))
    events = d.loc[indices, cols].copy()
    events["label"] = [v[1] for v in valid]
    events["bars_to_outcome"] = [v[2] for v in valid]
    events["ambiguous_bar"] = [v[3] for v in valid]
    events["bar_index"] = indices
    assert events.columns.is_unique
    return events.reset_index(drop=True)


def purged_splits(events: pd.DataFrame, horizon: int, min_events: int):
    """Three expanding train/validation windows inside first 80%; last 20% untouched."""
    n = len(events)
    cut = int(n * 0.80)
    test = events.iloc[cut:].copy()
    dev = events.iloc[:cut].copy()
    folds = []
    for start, end in ((0.45, 0.60), (0.60, 0.75), (0.75, 0.90)):
        a, b = int(len(dev) * start), int(len(dev) * end)
        fit = dev.iloc[:a].copy()
        validation = dev.iloc[a:b].copy()
        if fit.empty or validation.empty:
            continue
        # Past event's future observation horizon must end BEFORE validation begins.
        fit = fit.loc[fit.bar_index + horizon < validation.bar_index.min()]
        if len(fit) >= 40 and len(validation) >= 10 and fit.label.nunique() >= 2:
            folds.append((fit, validation))
    final_train = dev.loc[dev.bar_index + horizon < test.bar_index.min()].copy()
    if not folds:
        raise ValueError("Walk-forward katları için yeterli olay yok. Daha uzun veri gerekli.")
    if len(test) < 10 or len(final_train) < min_events or final_train.label.nunique() < 2:
        raise ValueError("Bağımsız test veya eğitim veri sayısı yetersiz.")
    return folds, final_train, test


class CatBoostCompat:
    """Avoid CatBoost/scikit-learn Pipeline tag incompatibility across versions."""
    def __init__(self, estimator):
        self.imputer = SimpleImputer(strategy="median")
        self.estimator = estimator
        self.classes_ = np.array([0, 1, 2])

    def fit(self, x, y):
        clean = self.imputer.fit_transform(x)
        self.estimator.fit(clean, y)
        self.classes_ = np.asarray(self.estimator.classes_, dtype=int)
        return self

    def predict_proba(self, x):
        return self.estimator.predict_proba(self.imputer.transform(x))


def candidates():
    """Optional external libraries are skipped gracefully, not silently reported as fitted."""
    models = {
        "reference_prior": lambda: DummyClassifier(strategy="prior"),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7, l2_regularization=3, learning_rate=0.04, random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=130, max_depth=5, min_samples_leaf=8, max_features="sqrt", n_jobs=2, random_state=42),
        "logistic_regression": lambda: LogisticRegression(max_iter=600, C=0.25, random_state=42),
    }
    unavailable = {}
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = lambda: LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=7, min_child_samples=18, learning_rate=0.04, reg_lambda=4, verbosity=-1, n_jobs=2, random_state=42)
    except ImportError as exc:
        unavailable["lightgbm"] = str(exc)
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = lambda: XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, reg_lambda=4, tree_method="hist", n_jobs=2, random_state=42)
    except ImportError as exc:
        unavailable["xgboost"] = str(exc)
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = lambda: CatBoostClassifier(iterations=120, depth=4, learning_rate=0.04, l2_leaf_reg=5, loss_function="MultiClass", verbose=False, thread_count=2, random_seed=42)
    except ImportError as exc:
        unavailable["catboost"] = str(exc)
    return models, unavailable


def build_model(name, constructor):
    if name == "catboost":
        return CatBoostCompat(constructor())
    if name == "logistic_regression":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), constructor())
    return make_pipeline(SimpleImputer(strategy="median"), constructor())


def fit_model(name, constructor, x, y):
    model = build_model(name, constructor)
    # XGBoost requires contiguous y class IDs, and CatBoost can be treated
    # consistently using the same -1,0,1 -> 0,1,2 mapping. No target leakage.
    if name in ("xgboost", "catboost"):
        if set(y.astype(int)) != {-1, 0, 1}:
            raise ValueError("XGBoost/CatBoost için eğitimde üç etiket de olmalı.")
        model.fit(x, y.to_numpy(dtype=int) + 1)
        return model, True
    model.fit(x, y)
    return model, False


def aligned_proba(model, encoded, x):
    src = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    if encoded:
        classes = classes - 1
    out = np.zeros((len(src), len(CLASS_VALUES)), dtype=float)
    for k, label in enumerate(CLASS_VALUES):
        found = np.where(classes == label)[0]
        if len(found):
            out[:, k] = src[:, found[0]]
    out = np.maximum(out, 1e-9)
    out /= out.sum(axis=1, keepdims=True)
    return out


def temperature_scale(probas, temperature):
    # Monotone calibration: does not change the predicted class.
    logits = np.log(np.maximum(probas, 1e-9)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def metrics(y_true, probabilities):
    yt = np.asarray(y_true, dtype=int)
    predicted = CLASS_VALUES[np.argmax(probabilities, axis=1)]
    counts = {CLASS_NAMES[int(i)]: int((yt == i).sum()) for i in CLASS_VALUES}
    precisions = precision_score(yt, predicted, labels=CLASS_VALUES, average=None, zero_division=0)
    recalls = recall_score(yt, predicted, labels=CLASS_VALUES, average=None, zero_division=0)
    brier = float(np.mean(np.sum((probabilities - (yt[:, None] == CLASS_VALUES[None, :]).astype(float)) ** 2, axis=1)))
    return {
        "n": len(yt), "actual_classes": counts,
        "accuracy": round(float(accuracy_score(yt, predicted)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, predicted)), 5),
        "log_loss": round(float(log_loss(yt, probabilities, labels=CLASS_VALUES)), 5),
        "brier_multiclass": round(brier, 5),
        "confusion_labels": ["ASAGI", "BELIRSIZ", "YUKARI"],
        "confusion_matrix": confusion_matrix(yt, predicted, labels=CLASS_VALUES).tolist(),
        "class_precision": {CLASS_NAMES[int(k)]: round(float(v), 5) for k, v in zip(CLASS_VALUES, precisions)},
        "class_recall": {CLASS_NAMES[int(k)]: round(float(v), 5) for k, v in zip(CLASS_VALUES, recalls)},
        "prediction_distribution": {CLASS_NAMES[int(i)]: int((predicted == i).sum()) for i in CLASS_VALUES}
    }


def threshold_audit(y, probabilities, threshold=0.60, gap=0.08):
    """Signal diagnostics, NOT realized P&L. Only directionally confident calls."""
    chosen = CLASS_VALUES[np.argmax(probabilities, axis=1)]
    sorted_proba = np.sort(probabilities, axis=1)
    selected = ((chosen != 0) & (sorted_proba[:, -1] >= threshold) &
                ((sorted_proba[:, -1] - sorted_proba[:, -2]) >= gap))
    ground = np.asarray(y, dtype=int)
    def stats(side):
        m = selected & (chosen == side)
        return {"signals": int(m.sum()), "correct": int(np.sum(ground[m] == side)),
                "precision": round(float(np.mean(ground[m] == side)), 5) if np.any(m) else None}
    return {"min_probability": threshold, "min_margin": gap,
            "long": stats(1), "short": stats(-1), "abstentions": int((~selected).sum()),
            "warning": "Confidence filter was fixed in advance, not optimized on holdout; not a P&L backtest."}


def alpha_segment_audit(events, probabilities):
    """Test by status of AlphaTrend center and last valid Pine signal."""
    y = events.label.to_numpy(dtype=int)
    pred = CLASS_VALUES[np.argmax(probabilities, axis=1)]
    result = {}
    for name, mask in {
        "price_above_alpha": (events.at_above.to_numpy() == 1),
        "price_below_alpha": (events.at_below.to_numpy() == 1),
        "last_valid_buy": (events.at_last_signal.to_numpy() == 1),
        "last_valid_sell": (events.at_last_signal.to_numpy() == -1),
        "buy_on_event": (events.at_buy_valid.to_numpy() == 1),
        "sell_on_event": (events.at_sell_valid.to_numpy() == 1),
    }.items():
        count = int(np.sum(mask))
        result[name] = {"n": count, "accuracy": round(float(np.mean(pred[mask] == y[mask])), 5) if count else None,
                        "actual_up": int(np.sum(y[mask] == 1)), "actual_down": int(np.sum(y[mask] == -1))}
    return result


def make_report(args):
    df = load_csv(args.csv)
    d = make_features(df, period=args.period, multiplier=args.multiplier, band_mult=args.band_mult,
                      sqz_len=args.sqz_len, lookback=args.lookback, sqz_factor=args.sqz_factor,
                      use_rsi=args.use_rsi)
    d, mtf_targets = augment_mtf_features(d, tf=args.tf, period=args.period, multiplier=args.multiplier,
                                           band_mult=args.band_mult, sqz_len=args.sqz_len,
                                           lookback=args.lookback, sqz_factor=args.sqz_factor,
                                           use_rsi=args.use_rsi)
    events = label_events(d, horizon=args.horizon, barrier_atr=args.barrier_atr,
                          min_events=args.min_events, tf=args.tf)
    folds, final_train, test = purged_splits(events, args.horizon, args.min_events)
    models, unavailable = candidates()
    validation = {}
    errors = {}
    successful = {}
    # Validation folds are chronological; folds have disjoint evaluation blocks.
    for name, constructor in models.items():
        pred_all, y_all, fold_details = [], [], []
        try:
            for fit, val in folds:
                model, encoded = fit_model(name, constructor, fit[FEATURES], fit.label)
                p = aligned_proba(model, encoded, val[FEATURES])
                pred_all.append(p)
                y_all.append(val.label.to_numpy(dtype=int))
                fold_details.append(metrics(val.label, p))
            merged = np.concatenate(pred_all)
            observed = np.concatenate(y_all)
            validation[name] = {"aggregate": metrics(observed, merged),
                                "folds": fold_details,
                                "fold_count": len(fold_details)}
            successful[name] = constructor
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    model_names = [k for k in successful if k != "reference_prior"]
    if not model_names:
        raise ValueError("Hiçbir gerçek ML modeli eğitilemedi. Hatalar: " + str(errors))
    selected = min(model_names, key=lambda name: validation[name]["aggregate"]["log_loss"])
    ref_loss = validation["reference_prior"]["aggregate"]["log_loss"]
    best_loss = validation[selected]["aggregate"]["log_loss"]
    # Guardrail: ML must outperform the baseline on chronological validation.
    baseline_beaten = bool(best_loss + args.min_logloss_gain < ref_loss)

    # Out-of-fold probabilities for temperature calibration; NEVER use test labels.
    stacked_oof, stacked_targets = [], []
    for train_fold, val_fold in folds:
        model, encoded = fit_model(selected, successful[selected], train_fold[FEATURES], train_fold.label)
        stacked_oof.append(aligned_proba(model, encoded, val_fold[FEATURES]))
        stacked_targets.append(val_fold.label.to_numpy(dtype=int))
    p_val = np.concatenate(stacked_oof)
    y_val = np.concatenate(stacked_targets)
    def objective(temp):
        return log_loss(y_val, temperature_scale(p_val, temp), labels=CLASS_VALUES)
    opt = minimize_scalar(objective, bounds=(0.7, 3.0), method="bounded", options={"xatol": 0.01})
    temp = float(opt.x) if opt.success and len(y_val) >= 60 else 1.0
    # Calibration on OOF if demonstrably better; limited evidence means no change.
    if temp != 1.0 and objective(temp) >= objective(1.0):
        temp = 1.0
    calibrated_val = metrics(y_val, temperature_scale(p_val, temp))

    # Fit only on pre-test history after purge. No leakage of test observations.
    final, encoded = fit_model(selected, successful[selected], final_train[FEATURES], final_train.label)
    model_test_p = aligned_proba(final, encoded, test[FEATURES])
    test_p = temperature_scale(model_test_p, temp)
    baseline, baseline_enc = fit_model("reference_prior", models["reference_prior"], final_train[FEATURES], final_train.label)
    baseline_test_p = aligned_proba(baseline, baseline_enc, test[FEATURES])
    model_test = metrics(test.label, test_p)
    base_test = metrics(test.label, baseline_test_p)
    # No readiness claim from test-driven tuning; conservative flag requires
    # minimum samples and baseline advantage in validation and holdout.
    ready = (baseline_beaten and len(test) >= 100 and model_test["balanced_accuracy"] >
             base_test["balanced_accuracy"] + 0.03 and
             model_test["log_loss"] < base_test["log_loss"] - 0.02)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "events.csv", index=False)
    hold = test[["time", "close", "alpha", "at_above", "at_below", "at_buy_raw",
                 "at_sell_raw", "at_buy_valid", "at_sell_valid", "at_last_signal",
                 "at_signal_age", "label"]].copy()
    hold["prediction"] = CLASS_VALUES[np.argmax(test_p, axis=1)]
    hold["reference_prediction"] = CLASS_VALUES[np.argmax(baseline_test_p, axis=1)]
    for i, label in enumerate(CLASS_VALUES):
        hold[f"prob_{CLASS_NAMES[int(label)].lower()}"] = test_p[:, i]
    hold.to_csv(output / "test_predictions.csv", index=False)

    # Safe-to-read, candle-close research snapshot; no real-time execution.
    last = d.iloc[-1]
    proba_latest = temperature_scale(aligned_proba(final, encoded, d.iloc[[-1]][FEATURES]), temp)[0] if np.isfinite(last[FEATURES].to_numpy(dtype=float)).all() else None
    latest = {
        "data_time_utc": str(last.time), "symbol": args.symbol, "timeframe": args.tf,
        "last_close": float(last.close), "alpha_center": float(last.alpha),
        "price_above_alpha": bool(last.at_above), "price_below_alpha": bool(last.at_below),
        "trend_up": bool(last.at_trend_up), "trend_down": bool(last.at_trend_down),
        "buy_raw": bool(last.at_buy_raw), "sell_raw": bool(last.at_sell_raw),
        "buy_valid": bool(last.at_buy_valid), "sell_valid": bool(last.at_sell_valid),
        "last_valid_signal": int(last.at_last_signal), "bars_since_valid_signal": int(last.at_signal_age),
        "squeeze_active": bool(last["squeeze"]),
        "mtf_timeframes_minutes": list(mtf_targets),
        "mtf_higher_agreement": float(last.mtf_alignment) if np.isfinite(last.mtf_alignment) else None,
        "bands": {key: float(last[key]) for key in ["upper1", "upper2", "upper3", "upper4", "lower1", "lower2", "lower3", "lower4"]},
        "research_probabilities": {CLASS_NAMES[int(label)]: float(proba_latest[i]) for i, label in enumerate(CLASS_VALUES)} if proba_latest is not None else None,
        "decision": "BEKLE / ANALIZ (NOT VALIDATED)" if not ready else "KARAR YALNIZCA ARASTIRMA; CANLI EMIR YOK",
        "not_live": "Last historical CSV bar only. Not a live market quote or trading recommendation."
    }
    (output / "latest_analysis.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "version": "1.1", "symbol": args.symbol, "timeframe": args.tf, "data_source": args.csv,
        "first_bar_utc": str(df.time.iloc[0]), "last_bar_utc": str(df.time.iloc[-1]),
        "candles": len(df), "squeeze_events": len(events),
        "labels_all": {CLASS_NAMES[int(i)]: int((events.label == i).sum()) for i in CLASS_VALUES},
        "test_events": len(test), "training_events": len(final_train),
        "walk_forward_folds": len(folds), "total_oof_events": len(y_val),
        "features_total": len(FEATURES), "feature_names": FEATURES,
        "higher_timeframes_minutes": list(mtf_targets),
        "selected_ml_model": selected,
        "optional_libraries_missing": unavailable, "failed_candidate_models": errors,
        "validation_comparison": validation,
        "baseline_validation_log_loss": ref_loss,
        "selected_validation_log_loss": best_loss,
        "required_validation_logloss_improvement": args.min_logloss_gain,
        "baseline_beaten_on_validation": baseline_beaten,
        "temperature_calibration": {"temperature": round(temp, 4), "oof_result": calibrated_val},
        "test_selected_model": model_test,
        "test_reference_model": base_test,
        "test_alpha_segments": alpha_segment_audit(test, test_p),
        "test_signal_filter": threshold_audit(test.label, test_p, args.signal_threshold, args.signal_margin),
        "signal_ready_for_deployment": ready,
        "caveats": [
            "NOT a trading or profit certification; historical backtest only.",
            "GC=F futures is not XAUUSD spot or XAUUSDT.",
            "Same-candle dual barrier hits are AMBIGUOUS, not guessed.",
            "Original Pine logic adapted to Python; exact candle-by-candle parity must be checked.",
            "Higher timeframe data uses only fully closed aggregate candles; missing gaps excluded.",
            "Overlapping squeeze outcomes and multiple comparisons still limit certainty.",
            "Model selection uses only out-of-fold validation. Holdout never tunes thresholds.",
            "Live orders disabled. Commission, slippage and executable-entry P&L not modeled."
        ]
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": final, "classes_encoded": encoded, "features": FEATURES,
                 "temperature": temp, "model_name": selected, "parameters": vars(args)}, output / "model.joblib")
    summary = {
        "version": "1.1", "selected_ml_model": selected,
        "candles": len(df), "squeeze_events": len(events), "test_events": len(test),
        "baseline_validation_log_loss": ref_loss, "ml_validation_log_loss": best_loss,
        "ml_test": {"accuracy": model_test["accuracy"], "balanced_accuracy": model_test["balanced_accuracy"], "log_loss": model_test["log_loss"]},
        "baseline_test": {"accuracy": base_test["accuracy"], "balanced_accuracy": base_test["balanced_accuracy"], "log_loss": base_test["log_loss"]},
        "signal_ready_for_deployment": ready,
        "output": str(output.resolve()),
    }
    # Small, phone-friendly flat comparison tables alongside the detailed JSON.
    rows = []
    for key, comparison in (
        ('A AlphaTrend ML', audit_a), ('A Simple baseline', audit_a_ref),
        ('B Squeeze ML', audit_b), ('B Simple baseline', audit_b_ref),
        ('A on B same holdout', matched['A_alpha_on_squeeze_test']),
        ('Fusion 50/50 on B same holdout', matched['fixed_50_50_blend_on_same_test']),
    ):
        rows.append({'model': key, 'samples': comparison['n'],
                     'accuracy': comparison['accuracy'],
                     'balanced_accuracy': comparison['balanced_accuracy'],
                     'log_loss': comparison['log_loss']})
    pd.DataFrame(rows).to_csv(output / 'model_comparison.csv', index=False)
    ab_rows = []
    for engine_name, group_info in [('A', ablate_a), ('B', ablate_b)]:
        for group_name, stats in group_info.items():
            if 'features' in stats:
                ab_rows.append({'engine': engine_name, 'feature_set': group_name,
                                'features': stats['features'],
                                'validation_log_loss': stats['log_loss'],
                                'validation_accuracy': stats['accuracy']})
    pd.DataFrame(ab_rows).to_csv(output / 'feature_ablation.csv', index=False)
    summary_text = (
        'AlphaTrend ML GOLD v1.2 - EGITIM OZETI\n'
        '------------------------------------\n'
        f'Sembol: {args.symbol} / {args.tf}\n'
        f'Kapanmis mum: {len(df)}\n'
        f'AlphaTrend egitim olaylari: {len(engine_a)}\n'
        f'Daralma egitim olaylari: {len(engine_b)}\n'
        f'Test A (seyreltilmis): {len(a_test)} olay\n'
        f'Test B: {len(b_test)} olay\n'
        f'A modeli: {fit_a["selected"]} / dogruluk %{audit_a["accuracy"]*100:.2f}\n'
        f'A referansi: %{audit_a_ref["accuracy"]*100:.2f}\n'
        f'B modeli: {fit_b["selected"]} / dogruluk %{audit_b["accuracy"]*100:.2f}\n'
        f'B referansi: %{audit_b_ref["accuracy"]*100:.2f}\n'
        f'Ayni B olaylarinda sabit 50/50: %{matched["fixed_50_50_blend_on_same_test"]["accuracy"]*100:.2f}\n'
        'CANLI ISLEM: HAYIR. GERCEK ALTIN ICIN AYRICA TEST GEREKLI.\n'
        'Etiket dogrulugu kar-zarar veya islem isabeti demek degildir.\n'
    )
    (output / 'OZET_TR.txt').write_text(summary_text, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="AlphaTrend 4B squeeze ML v1.1 — research only")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", default="GC=F")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--output", default="result")
    parser.add_argument("--period", type=int, default=14)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--band-mult", type=float, default=0.5)
    parser.add_argument("--sqz-len", type=int, default=20)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--sqz-factor", type=float, default=0.85)
    parser.add_argument("--use-rsi", action="store_true")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--barrier-atr", type=float, default=1.5)
    parser.add_argument("--min-events", type=int, default=35)
    parser.add_argument("--min-logloss-gain", type=float, default=0.02)
    parser.add_argument("--signal-threshold", type=float, default=0.60)
    parser.add_argument("--signal-margin", type=float, default=0.08)
    return parser


# =============================================================
# v1.2 — DUAL ENGINE RESEARCH (preserves original AlphaTrend core)
# A: Price / AlphaTrend / BUY-SELL behavior on every eligible closed bar
# B: Squeeze beginnings only (legacy v1.1 task)
# Predictions are compared on separate chronological holdouts.
# =============================================================

BASE_FEATURES = list(FEATURES[:18])
AT_FEATURES = [f for f in FEATURES if f.startswith('at_')]
MTF_FEATURES = [f for f in FEATURES if f.startswith('mtf')]
SQUEEZE_FEATURES = [f for f in FEATURES if f not in BASE_FEATURES + AT_FEATURES + MTF_FEATURES]

ALPHA_GROUPS = {
    'teknik': BASE_FEATURES,
    'teknik_alpha': BASE_FEATURES + AT_FEATURES,
    'teknik_alpha_mtf': BASE_FEATURES + AT_FEATURES + MTF_FEATURES,
}
SQUEEZE_GROUPS = {
    'daralma_teknik': BASE_FEATURES + SQUEEZE_FEATURES,
    'daralma_alpha': BASE_FEATURES + SQUEEZE_FEATURES + AT_FEATURES,
    'daralma_alpha_mtf': list(FEATURES),
}


def label_every_closed_bar(d: pd.DataFrame, horizon: int, barrier_atr: float, tf: str) -> pd.DataFrame:
    """All eligible candle-close decisions; labels from *subsequent* candles.

    A single future observation window is used for both engines. Events crossing
    overnight / weekend gaps are excluded. The sample is densely overlapping,
    so validation and test scores are measured on horizon-spaced observations.
    """
    unit = tf.lower().strip()
    if not unit.endswith('m') or not unit[:-1].isdigit() or int(unit[:-1]) not in (1, 3, 5, 15):
        raise ValueError('Girdi 1m, 3m, 5m veya 15m mum olmalı.')
    if horizon < 1 or barrier_atr <= 0:
        raise ValueError('horizon pozitif ve barrier_atr 0 uzeri olmalı.')
    step = pd.Timedelta(minutes=int(unit[:-1]))
    bad = d.time.diff().iloc[1:].ne(step).to_numpy(dtype=int)
    gap_prefix = np.concatenate(([0], np.cumsum(bad)))
    all_features = d[FEATURES].to_numpy(dtype=float)
    good = np.isfinite(all_features).all(axis=1)
    atr, close = d.atr.to_numpy(), d.close.to_numpy()
    high, low = d.high.to_numpy(), d.low.to_numpy()
    rows = []
    for ix in range(len(d) - horizon):
        if not good[ix] or not np.isfinite(atr[ix]) or atr[ix] <= 0:
            continue
        # Gap entries are transition indices from ix+1 through ix+horizon.
        if gap_prefix[ix + horizon] != gap_prefix[ix]:
            continue
        upper, lower = close[ix] + atr[ix] * barrier_atr, close[ix] - atr[ix] * barrier_atr
        result, bars, both_hit = 0, horizon, False
        for j in range(ix + 1, ix + horizon + 1):
            up, down = high[j] >= upper, low[j] <= lower
            if up and down:
                both_hit, bars = True, j - ix
                break
            if up or down:
                result, bars = (1 if up else -1), j - ix
                break
        rows.append((ix, result, bars, both_hit))
    if len(rows) < 300:
        raise ValueError(f'AlphaTrend motoru icin yeterli uygun mum yok: {len(rows)}')
    indices = [r[0] for r in rows]
    cols = list(dict.fromkeys([
        'time','close','alpha','atr','step_size','width_pct','squeeze','event',
        'at_above','at_below','at_buy_raw','at_sell_raw','at_buy_valid',
        'at_sell_valid','at_buy_confirmed','at_sell_confirmed',
        'at_last_signal','at_signal_age'
    ] + FEATURES))
    out = d.loc[indices, cols].copy()
    out['label'] = [r[1] for r in rows]
    out['bars_to_outcome'] = [r[2] for r in rows]
    out['ambiguous_bar'] = [r[3] for r in rows]
    out['bar_index'] = indices
    assert out.columns.is_unique
    return out.reset_index(drop=True)


def spaced_audit(frame: pd.DataFrame, horizon: int, start_index=None) -> pd.DataFrame:
    """Evaluation subset with nonoverlapping future barrier horizons.

    Sampling is precommitted by absolute source bar_index, independent of labels.
    """
    anchor = 0 if start_index is None else int(start_index)
    return frame.loc[(frame.bar_index - anchor) % horizon == 0].copy()


def align_alpha_folds(all_bars: pd.DataFrame, squeeze_folds, horizon: int):
    folds = []
    for _, squeeze_val in squeeze_folds:
        first = int(squeeze_val.bar_index.min())
        last = int(squeeze_val.bar_index.max())
        # Purge training rows whose future labeling extends into validation.
        alpha_fit = all_bars.loc[all_bars.bar_index + horizon < first].copy()
        alpha_val = all_bars.loc[(all_bars.bar_index >= first) &
                                 (all_bars.bar_index <= last)].copy()
        alpha_val = spaced_audit(alpha_val, horizon, first)
        if len(alpha_fit) < 100 or len(alpha_val) < 10 or alpha_fit.label.nunique() < 2:
            raise ValueError('AlphaTrend walk-forward verisi yetersiz; daha uzun OHLCV gecmisi gerekli.')
        folds.append((alpha_fit, alpha_val))
    return folds


def compare_ablation(groups: dict, folds, estimator_name='hist_gradient_boosting'):
    """Fixed-family A/B feature ablation on the *same* chronology.

    The feature group comparisons are diagnostics; no holdout data is used
    to select input features.
    """
    constructors, _ = candidates()
    results = {}
    for feature_group, cols in groups.items():
        y_parts, p_parts = [], []
        try:
            for fit, val in folds:
                model, enc = fit_model(estimator_name, constructors[estimator_name],
                                       fit[cols], fit.label)
                y_parts.append(val.label.to_numpy(dtype=int))
                p_parts.append(aligned_proba(model, enc, val[cols]))
            result = metrics(np.concatenate(y_parts), np.concatenate(p_parts))
            results[feature_group] = {'features': len(cols), 'log_loss': result['log_loss'],
                                      'balanced_accuracy': result['balanced_accuracy'],
                                      'accuracy': result['accuracy'], 'n': result['n']}
        except Exception as exc:
            results[feature_group] = {'error': f'{type(exc).__name__}: {exc}'}
    keys = list(groups)
    for name, before, after in (('alpha_features_effect', keys[0], keys[1]),
                                ('mtf_features_effect', keys[1], keys[2])):
        if 'log_loss' in results[before] and 'log_loss' in results[after]:
            diff = round(results[after]['log_loss'] - results[before]['log_loss'], 5)
            results[name] = {
                'before': before, 'after': after,
                'delta_log_loss': diff, 'helped_on_validation': diff < 0,
                'note': 'Negative delta improves log loss for the fixed reference ML family; not proof of profit.'
            }
    return results


def cross_validate_engine(folds, cols, engine_name: str, min_gain: float):
    """Select a model family using validation only; the test remains untouched."""
    constructors, missing = candidates()
    comparison, failed = {}, {}
    successful = {}
    oof = {}
    for name, factory in constructors.items():
        y_chunks, p_chunks, fold_scores = [], [], []
        try:
            for fit, val in folds:
                estimator, enc = fit_model(name, factory, fit[cols], fit.label)
                prediction = aligned_proba(estimator, enc, val[cols])
                y_chunks.append(val.label.to_numpy(dtype=int))
                p_chunks.append(prediction)
                fold_scores.append(metrics(val.label, prediction))
            yy, pp = np.concatenate(y_chunks), np.vstack(p_chunks)
            comparison[name] = {
                'pooled': metrics(yy, pp),
                'folds': fold_scores,
                'fold_count': len(fold_scores),
            }
            successful[name] = factory
            oof[name] = (yy, pp)
        except Exception as exc:
            failed[name] = f'{type(exc).__name__}: {exc}'
    if 'reference_prior' not in successful:
        raise ValueError(f'{engine_name}: referans modeli egitilemedi: {failed}')
    contenders = [name for name in successful if name != 'reference_prior']
    if not contenders:
        raise ValueError(f'{engine_name}: hicbir ML algoritmasi egitilemedi: {failed}')
    selected = min(contenders, key=lambda name: comparison[name]['pooled']['log_loss'])
    baseline_loss = comparison['reference_prior']['pooled']['log_loss']
    ml_loss = comparison[selected]['pooled']['log_loss']
    validation_gain = round(baseline_loss - ml_loss, 5)
    yy, pp = oof[selected]
    temperature = 1.0
    # Temperature calibration on *only* the chronological out-of-fold scores.
    if len(yy) >= 60:
        def loss(temp):
            return log_loss(yy, temperature_scale(pp, temp), labels=CLASS_VALUES)
        optimized = minimize_scalar(loss, bounds=(0.7, 3.0), method='bounded')
        if optimized.success and loss(float(optimized.x)) + 1e-5 < loss(1.0):
            temperature = float(optimized.x)
    return {
        'name': engine_name,
        'selected': selected,
        'constructors': successful,
        'features': cols,
        'temperature': temperature,
        'comparison': comparison,
        'missing_libraries': missing,
        'errors': failed,
        'validation_baseline_log_loss': baseline_loss,
        'validation_selected_log_loss': ml_loss,
        'validation_gain': validation_gain,
        'validated_better_than_baseline': validation_gain >= min_gain,
        'selected_calibrated_oof': metrics(yy, temperature_scale(pp, temperature)),
    }


def train_final_engine(result: dict, fit: pd.DataFrame, test: pd.DataFrame):
    cols = result['features']
    name = result['selected']
    mdl, encoded = fit_model(name, result['constructors'][name], fit[cols], fit.label)
    ref, ref_encoded = fit_model('reference_prior', result['constructors']['reference_prior'],
                                 fit[cols], fit.label)
    predictions = temperature_scale(aligned_proba(mdl, encoded, test[cols]), result['temperature'])
    baselines = aligned_proba(ref, ref_encoded, test[cols])
    return mdl, encoded, predictions, baselines


def save_predictions(frame, probas: dict, path: Path, extra_cols=None):
    cols = ['time', 'bar_index', 'close', 'alpha', 'at_above', 'at_below',
            'at_buy_valid', 'at_sell_valid', 'at_last_signal', 'at_signal_age', 'label']
    cols += [k for k in (extra_cols or []) if k in frame.columns and k not in cols]
    out = frame[cols].copy()
    for name, p in probas.items():
        out[f'{name}_pred'] = CLASS_VALUES[np.argmax(p, axis=1)]
        for i, klass in enumerate(CLASS_VALUES):
            out[f'{name}_prob_{CLASS_NAMES[int(klass)].lower()}'] = p[:, i]
    out.to_csv(path, index=False)


def native_signal_outcome_audit(events):
    """What happened AFTER original raw/filtered/confirmed BUY and SELL bars?

    Event's own close is treated as a reference for a future-outcome label;
    these are not executable fill-based return statistics.
    """
    result = {}
    for column, target in (
        ('at_buy_raw', 1), ('at_sell_raw', -1),
        ('at_buy_valid', 1), ('at_sell_valid', -1),
        ('at_buy_confirmed', 1), ('at_sell_confirmed', -1)
    ):
        mask = events[column].to_numpy(dtype=bool)
        n = int(mask.sum())
        correct = int((events.loc[mask, 'label'] == target).sum())
        result[column] = {
            'n': n, 'correct_direction': correct,
            'direction_hit_rate': round(correct / n, 5) if n else None,
            'ambiguous': int((events.loc[mask, 'label'] == 0).sum()),
            'warning': 'Sample count may be too small; this is an event outcome, not trading P&L.'
        }
    return result


def make_dual_report(args):
    if args.min_events < 35:
        raise ValueError('Araştırma icin en az 35 daralma olayi gerekir.')
    df = load_csv(args.csv)
    d = make_features(df, period=args.period, multiplier=args.multiplier,
                      band_mult=args.band_mult, sqz_len=args.sqz_len, lookback=args.lookback,
                      sqz_factor=args.sqz_factor, use_rsi=args.use_rsi)
    d, mtf_targets = augment_mtf_features(d, tf=args.tf, period=args.period,
                         multiplier=args.multiplier, band_mult=args.band_mult,
                         sqz_len=args.sqz_len, lookback=args.lookback,
                         sqz_factor=args.sqz_factor, use_rsi=args.use_rsi)
    # The two tasks share future-outcome rules and a single test date boundary.
    engine_b = label_events(d, horizon=args.horizon, barrier_atr=args.barrier_atr,
                            min_events=args.min_events, tf=args.tf)
    engine_a = label_every_closed_bar(d, horizon=args.horizon,
                                      barrier_atr=args.barrier_atr, tf=args.tf)
    b_folds, b_final, b_test = purged_splits(engine_b, args.horizon, args.min_events)
    a_folds = align_alpha_folds(engine_a, b_folds, args.horizon)
    test_start = int(b_test.bar_index.min())
    a_final = engine_a.loc[engine_a.bar_index + args.horizon < test_start].copy()
    a_test_window = engine_a.loc[(engine_a.bar_index >= test_start) &
                                 (engine_a.bar_index <= int(b_test.bar_index.max()))].copy()
    a_test = spaced_audit(a_test_window, args.horizon, test_start)
    if len(a_test) < 20 or a_final.label.nunique() < 2:
        raise ValueError('Motor A bagimsiz testi icin yeterli zaman verisi yok.')
    # Same future labels for both engines at their shared squeeze indices.
    pairs = engine_a.set_index('bar_index')['label'].reindex(engine_b.bar_index)
    if pairs.isna().any() or not np.array_equal(pairs.to_numpy(dtype=int), engine_b.label.to_numpy(dtype=int)):
        raise AssertionError('Motor A/B olay etiketleri ayni zaman damgasinda uyusmuyor.')

    # Ablation explicitly checks whether AT core / BUY-SELL / MTF add value.
    ablate_a = compare_ablation(ALPHA_GROUPS, a_folds)
    ablate_b = compare_ablation(SQUEEZE_GROUPS, b_folds)

    fit_a = cross_validate_engine(a_folds, ALPHA_GROUPS['teknik_alpha_mtf'],
                                   'alpha_all_closed_bars', args.min_logloss_gain)
    fit_b = cross_validate_engine(b_folds, SQUEEZE_GROUPS['daralma_alpha_mtf'],
                                   'squeeze_start_events', args.min_logloss_gain)
    model_a, encoded_a, probs_a, prior_a = train_final_engine(fit_a, a_final, a_test)
    model_b, encoded_b, probs_b, prior_b = train_final_engine(fit_b, b_final, b_test)
    # Evaluate A at the EXACT same B test events: fair matched comparison.
    a_at_b = temperature_scale(aligned_proba(model_a, encoded_a,
                        b_test[fit_a['features']]), fit_a['temperature'])
    # Fixed weights declared before observing holdout; never grid-search weights on holdout.
    blend = (a_at_b + probs_b) / 2.0
    blend /= blend.sum(axis=1, keepdims=True)

    audit_a = metrics(a_test.label, probs_a)
    audit_a_ref = metrics(a_test.label, prior_a)
    audit_b = metrics(b_test.label, probs_b)
    audit_b_ref = metrics(b_test.label, prior_b)
    matched = {
        'A_alpha_on_squeeze_test': metrics(b_test.label, a_at_b),
        'B_squeeze_on_same_test': audit_b,
        'fixed_50_50_blend_on_same_test': metrics(b_test.label, blend),
        'B_reference_same_test': audit_b_ref,
    }
    ready = False  # Never use retrospective classification alone as live approval.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep the historical squeeze table and deliver compact all-bar labels.
    engine_b.to_csv(output / 'events_squeeze.csv', index=False)
    engine_a[['time','bar_index','close','alpha','at_above','at_below',
              'at_buy_raw','at_sell_raw','at_buy_valid','at_sell_valid',
              'at_buy_confirmed','at_sell_confirmed', 'at_last_signal','at_signal_age',
              'squeeze','label','bars_to_outcome','ambiguous_bar']].to_csv(
                  output / 'alpha_all_candles_labels.csv', index=False)
    save_predictions(a_test, {'alpha_ml': probs_a, 'alpha_reference': prior_a},
                     output / 'test_A_predictions.csv')
    save_predictions(b_test, {'alpha_ml': a_at_b, 'squeeze_ml': probs_b,
                              'fusion_50_50': blend, 'reference': prior_b},
                     output / 'test_B_fusion_predictions.csv', extra_cols=['width_pct','at_buy_raw','at_sell_raw'])

    last = d.iloc[-1]
    last_a = None
    if np.isfinite(last[fit_a['features']].to_numpy(dtype=float)).all():
        last_a = temperature_scale(aligned_proba(model_a, encoded_a,
                            d.iloc[[-1]][fit_a['features']]), fit_a['temperature'])[0]
    last_b = None
    if bool(last.event) and np.isfinite(last[fit_b['features']].to_numpy(dtype=float)).all():
        last_b = temperature_scale(aligned_proba(model_b, encoded_b,
                            d.iloc[[-1]][fit_b['features']]), fit_b['temperature'])[0]
    latest = {
        'data_time_utc': str(last.time), 'symbol': args.symbol, 'tf': args.tf,
        'data_is_live': False, 'last_close': float(last.close),
        'alpha': float(last.alpha), 'price_above_alpha': bool(last.at_above),
        'price_below_alpha': bool(last.at_below),
        'raw_buy': bool(last.at_buy_raw), 'raw_sell': bool(last.at_sell_raw),
        'valid_buy': bool(last.at_buy_valid), 'valid_sell': bool(last.at_sell_valid),
        'confirmed_buy': bool(last.at_buy_confirmed), 'confirmed_sell': bool(last.at_sell_confirmed),
        'last_valid_signal': int(last.at_last_signal), 'bars_since_valid_signal': int(last.at_signal_age),
        'squeeze_active': bool(last.squeeze), 'new_squeeze_event': bool(last.event),
        'research_alpha_probabilities': {CLASS_NAMES[int(v)]: float(last_a[i]) for i,v in enumerate(CLASS_VALUES)} if last_a is not None else None,
        'research_squeeze_event_probabilities': {CLASS_NAMES[int(v)]: float(last_b[i]) for i,v in enumerate(CLASS_VALUES)} if last_b is not None else None,
        'decision': 'BEKLE — OOS AVANTAJ DOGRULANMADI',
        'note': 'Historical dataset last candle, not a live signal. No real orders.'
    }
    (output / 'latest_analysis.json').write_text(json.dumps(latest, indent=2, ensure_ascii=False), encoding='utf-8')

    def display_engine(result):
        return {
            'feature_count': len(result['features']),
            'feature_names': result['features'],
            'selected_model': result['selected'],
            'validation_models': {name: {'pooled': v['pooled'], 'fold_count': v['fold_count']}
                                  for name, v in result['comparison'].items()},
            'validation_baseline_log_loss': result['validation_baseline_log_loss'],
            'validation_ml_log_loss': result['validation_selected_log_loss'],
            'validation_gain': result['validation_gain'],
            'beats_baseline_validation_guard': result['validated_better_than_baseline'],
            'temperature_from_validation': round(result['temperature'], 4),
            'calibrated_oof': result['selected_calibrated_oof'],
            'optional_model_errors': result['errors'],
            'missing_libraries': result['missing_libraries'],
        }
    report = {
        'version': '1.2', 'dataset_symbol': args.symbol, 'data_source': str(args.csv),
        'timeframe': args.tf, 'first_bar_utc': str(df.time.iloc[0]),
        'last_bar_utc': str(df.time.iloc[-1]), 'candles': len(df),
        'horizon_bars': args.horizon, 'barrier_atr': args.barrier_atr,
        'mtf_minutes': list(mtf_targets), 'data_type': 'historical_closed_candles',
        'alpha_all_bars_labeled': len(engine_a),
        'alpha_train_rows': len(a_final), 'alpha_test_independent_horizon_spaced': len(a_test),
        'alpha_buy_valid_count_all': int(engine_a.at_buy_valid.sum()),
        'alpha_sell_valid_count_all': int(engine_a.at_sell_valid.sum()),
        'alpha_buy_confirmed_count_all': int(engine_a.at_buy_confirmed.sum()),
        'alpha_sell_confirmed_count_all': int(engine_a.at_sell_confirmed.sum()),
        'squeeze_events': len(engine_b), 'squeeze_train_events': len(b_final),
        'squeeze_test_events': len(b_test),
        'squeeze_labels_all': {CLASS_NAMES[int(i)]: int((engine_b.label == i).sum()) for i in CLASS_VALUES},
        'shared_test_start_utc': str(b_test.time.iloc[0]),
        'feature_ablation_alpha_motor': ablate_a,
        'feature_ablation_squeeze_motor': ablate_b,
        'motor_A': display_engine(fit_a), 'motor_B': display_engine(fit_b),
        'test_motor_A_nonoverlap': audit_a,
        'test_motor_A_reference_nonoverlap': audit_a_ref,
        'test_motor_B': audit_b,
        'test_motor_B_reference': audit_b_ref,
        'test_same_squeeze_events': matched,
        'test_alpha_line_and_buy_sell_segments': alpha_segment_audit(a_test, probs_a),
        'test_squeeze_line_and_buy_sell_segments': alpha_segment_audit(b_test, probs_b),
        'native_buy_sell_outcomes_motor_A_test': native_signal_outcome_audit(a_test_window),
        'native_buy_sell_outcomes_motor_B_test': native_signal_outcome_audit(b_test),
        'test_alpha_signal_threshold': threshold_audit(a_test.label, probs_a, args.signal_threshold, args.signal_margin),
        'test_squeeze_signal_threshold': threshold_audit(b_test.label, probs_b, args.signal_threshold, args.signal_margin),
        'deployment_ready': ready,
        'evidence_warning': 'Two research models and a fixed fusion are compared on historical outcomes; this is NOT a validated trading advantage.',
        'limitations': [
            'GC=F altin vadelidir; XAUUSD/XAUUSDT degildir.',
            'Yalnizca kapanmis mumlar ve gecmiste bilinen indikatorler kullanilir.',
            'A motoru tum uygun mumlarda egitilir; komsu hedefler ortusebilir. Bagimsiz test A icin horizon aralikli seyreltildi.',
            'B motoru sadece daralma baslangiclarinda egitilir; veri sayisi daha az olabilir.',
            'A/B modeli ayni daralma test olaylarinda ayrica karsilastirildi; 50/50 agirlik testte ayarlanmadi.',
            'Etiketler gelecekteki mumlarda hedefe ilk ulasilmasina gore olusturulur; ayni mumda iki hedefe dokunulursa belirsiz.',
            'Baska piyasa donemlerinde daha uzun, gercek XAUUSD/XAUUSDT OHLCV ve gercek fiyat yayilimi maliyetleri test edilmelidir.',
            'Model secimi walk-forward validation ile yapilir; holdout yalnizca raporlanir.',
            'Komisyon, spread, slippage, gecikme ve gercek emir yurutme getirisi hesaplanmaz.',
            'Olasilik kalibrasyonu zaman disi farkli piyasalarda dogrulanmadi. Canli emirler kapali.',
        ]
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    joblib.dump({'model': model_a, 'encoded': encoded_a, 'features': fit_a['features'],
                 'temperature': fit_a['temperature'], 'tf': args.tf, 'name': fit_a['selected']}, output / 'engine_A.joblib')
    joblib.dump({'model': model_b, 'encoded': encoded_b, 'features': fit_b['features'],
                 'temperature': fit_b['temperature'], 'tf': args.tf, 'name': fit_b['selected']}, output / 'engine_B.joblib')
    summary = {
        'version': '1.2', 'candles': len(df), 'alpha_training_bars': len(engine_a),
        'squeeze_events': len(engine_b), 'test_A_independent': len(a_test),
        'test_B': len(b_test), 'model_A': fit_a['selected'], 'model_B': fit_b['selected'],
        'A_accuracy': audit_a['accuracy'], 'A_reference_accuracy': audit_a_ref['accuracy'],
        'B_accuracy': audit_b['accuracy'], 'B_reference_accuracy': audit_b_ref['accuracy'],
        'fusion_accuracy_on_B': matched['fixed_50_50_blend_on_same_test']['accuracy'],
        'deployment_ready': False,
        'output_directory': str(output.resolve()),
    }
    # Small, phone-friendly flat comparison tables alongside the detailed JSON.
    rows = []
    for key, comparison in (
        ('A AlphaTrend ML', audit_a), ('A Simple baseline', audit_a_ref),
        ('B Squeeze ML', audit_b), ('B Simple baseline', audit_b_ref),
        ('A on B same holdout', matched['A_alpha_on_squeeze_test']),
        ('Fusion 50/50 on B same holdout', matched['fixed_50_50_blend_on_same_test']),
    ):
        rows.append({'model': key, 'samples': comparison['n'],
                     'accuracy': comparison['accuracy'],
                     'balanced_accuracy': comparison['balanced_accuracy'],
                     'log_loss': comparison['log_loss']})
    pd.DataFrame(rows).to_csv(output / 'model_comparison.csv', index=False)
    ab_rows = []
    for engine_name, group_info in [('A', ablate_a), ('B', ablate_b)]:
        for group_name, stats in group_info.items():
            if 'features' in stats:
                ab_rows.append({'engine': engine_name, 'feature_set': group_name,
                                'features': stats['features'],
                                'validation_log_loss': stats['log_loss'],
                                'validation_accuracy': stats['accuracy']})
    pd.DataFrame(ab_rows).to_csv(output / 'feature_ablation.csv', index=False)
    summary_text = (
        'AlphaTrend ML GOLD v1.2 - EGITIM OZETI\n'
        '------------------------------------\n'
        f'Sembol: {args.symbol} / {args.tf}\n'
        f'Kapanmis mum: {len(df)}\n'
        f'AlphaTrend egitim olaylari: {len(engine_a)}\n'
        f'Daralma egitim olaylari: {len(engine_b)}\n'
        f'Test A (seyreltilmis): {len(a_test)} olay\n'
        f'Test B: {len(b_test)} olay\n'
        f'A modeli: {fit_a["selected"]} / dogruluk %{audit_a["accuracy"]*100:.2f}\n'
        f'A referansi: %{audit_a_ref["accuracy"]*100:.2f}\n'
        f'B modeli: {fit_b["selected"]} / dogruluk %{audit_b["accuracy"]*100:.2f}\n'
        f'B referansi: %{audit_b_ref["accuracy"]*100:.2f}\n'
        f'Ayni B olaylarinda sabit 50/50: %{matched["fixed_50_50_blend_on_same_test"]["accuracy"]*100:.2f}\n'
        'CANLI ISLEM: HAYIR. GERCEK ALTIN ICIN AYRICA TEST GEREKLI.\n'
        'Etiket dogrulugu kar-zarar veya islem isabeti demek degildir.\n'
    )
    (output / 'OZET_TR.txt').write_text(summary_text, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    make_dual_report(build_parser().parse_args())
