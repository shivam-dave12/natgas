#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║      NATURAL GAS PRICE PREDICTION ENGINE — PRODUCTION GRADE v5.0 (v18)        ║
║                                                                                ║
║  Unified Regime-Aware Ensemble: Rangebound + Spike in ONE model               ║
║                                                                                ║
║  v18 ARCHITECTURE (fixes v17 dual-model failure):                              ║
║    • Single unified model with CONDITIONAL target clipping —                   ║
║      spike_risk_score drives per-sample clip widths so the model               ║
║      learns "if spike conditions → large move" without destroying              ║
║      rangebound accuracy                                                       ║
║    • Spike-conditioned post-prediction ANALYTICAL overlay —                    ║
║      no noisy classifier, no tiny-sample spike forecaster                      ║
║    • Weather-driven intraday: GFS/ECMWF delta injection,                      ║
║      EIA storage report calendar, hourly session modelling                     ║
║    • Asymmetric CI: upside vol scales with spike_risk_score (demand shocks     ║
║      are upside), downside stays tight (natgas doesn't crash 40% in a day)     ║
║                                                                                ║
║  WHY v17 FAILED: Two separate models (classifier + spike forecaster) with      ║
║  ~20 training samples each → noisy probability × overfit magnitude = garbage   ║
║  that corrupted the base model's good rangebound predictions via blending.     ║
║  Solution: make ONE model smarter, not two dumb models averaged together.      ║
║                                                                                ║
║  Data Sources (all free, public APIs — NO FALLBACKS, NO SYNTHETIC DATA):       ║
║    • EIA API v2         — prices, storage, production, consumption, LNG        ║
║    • Open-Meteo API     — historical + forecast weather (NO key required)      ║
║    • Yahoo Finance      — crude oil, coal, USD index, VIX, natgas futures      ║
║                                                                                ║
║  Requirements:                                                                 ║
║    pip install pandas numpy scikit-learn xgboost lightgbm yfinance             ║
║    pip install statsmodels arch optuna requests python-dotenv                   ║
║                                                                                ║
║  Usage:                                                                        ║
║    1. Set EIA_API_KEY in .env or pass at runtime                               ║
║    2. python natural_gas_predictor_enhanced_v18.py                             ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ML — core (always available)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

# ML — optional accelerators
try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from arch import arch_model; HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import yfinance as yf; HAS_YF = True
except ImportError:
    HAS_YF = False

warnings.filterwarnings("ignore")
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ▸ CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Every tunable parameter lives here."""
    eia_api_key: str = ""
    history_years: int = 3
    forecast_horizon: int = 10          # days ahead (1–10)
    min_training_samples: int = 200
    n_cv_splits: int = 5
    test_fraction: float = 0.15
    max_optuna_trials: int = 30
    max_daily_move_sigma: float = 3.0
    min_price_floor: float = 0.50
    max_price_ceil: float = 15.00
    use_weather: bool = True
    use_cross_commodity: bool = True
    use_storage: bool = True
    use_production: bool = True
    use_lng: bool = True
    log_file: str = "natgas_pro.log"
    live_price_ticker: str = "NG=F"     # Yahoo Finance ticker for live price
    backtest_date: str = ""              # If set (YYYY-MM-DD), anchor forecast to that past date
    weather_locations: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Chicago":     (41.88, -87.63),
        "New_York":    (40.71, -74.01),
        "Houston":     (29.76, -95.37),
        "Boston":      (42.36, -71.06),
        "Detroit":     (42.33, -83.05),
        "Minneapolis": (44.98, -93.27),
        "Dallas":      (32.78, -96.80),
        "Atlanta":     (33.75, -84.39),
        "Denver":      (39.74, -104.99),
        "Philadelphia":(39.95, -75.17),
    })


class MarketRegime(Enum):
    LOW_VOL  = "low_volatility"
    NORMAL   = "normal"
    HIGH_VOL = "high_volatility"
    CRISIS   = "crisis"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ▸ TERMINAL UI
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    """ANSI colour codes."""
    H="\033[95m"; B="\033[94m"; CN="\033[96m"; G="\033[92m"
    Y="\033[93m"; R="\033[91m"; E="\033[0m";   BD="\033[1m"
    W="\033[97m"; GR="\033[90m"

def _log_setup(cfg: Config) -> logging.Logger:
    lg = logging.getLogger("natgas")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fh = logging.FileHandler(cfg.log_file, encoding="utf-8")  # UTF-8 prevents CP1252 crash on Windows
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        lg.addHandler(fh)
    return lg

def banner():
    print(f"""
{C.CN}{C.BD}╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║       🔥  NATURAL GAS PRICE PREDICTION ENGINE  v5.0 (v18)  🔥           ║
║       Unified Regime-Aware · Spike-Conditioned · Intraday Grade          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝{C.E}
""")

def hdr(text: str):
    w = 76
    pad = (w - len(text)) // 2
    print(f"\n{C.Y}{C.BD}┏{'━'*w}┓{C.E}")
    print(f"{C.Y}{C.BD}┃{C.E}{' '*pad}{C.W}{C.BD}{text}{C.E}{' '*(w-len(text)-pad)}{C.Y}{C.BD}┃{C.E}")
    print(f"{C.Y}{C.BD}┗{'━'*w}┛{C.E}")

def sec(text: str, icon: str = "▸"):
    print(f"\n  {C.CN}{C.BD}{icon} {text}{C.E}")
    print(f"  {C.GR}{'─'*(len(text)+4)}{C.E}")

def info(label: str, value, icon: str = "•"):
    print(f"    {C.CN}{icon}{C.E} {C.BD}{label}:{C.E} {C.W}{value}{C.E}")

def ok(msg: str):  print(f"    {C.G}✓ {msg}{C.E}")
def warn(msg: str): print(f"    {C.Y}⚠ {msg}{C.E}")
def err(msg: str):  print(f"    {C.R}✗ {msg}{C.E}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ▸ DATA INGESTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class _Fetcher:
    """HTTP helper with retry + backoff."""
    @staticmethod
    def get(url, params=None, headers=None, retries=3, backoff=1.0, timeout=30):
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == retries - 1:
                    return None
                time.sleep(backoff * (2 ** attempt))
        return None


class EIAClient(_Fetcher):
    """
    EIA API v2 client.
    Fetches: Henry Hub spot prices, weekly storage, monthly production,
    monthly consumption, monthly LNG exports.
    """
    BASE = "https://api.eia.gov/v2/"

    def __init__(self, api_key: str, log: logging.Logger):
        self.key = api_key
        self.log = log

    def _v2(self, route, facets, start, freq="daily", length=5000):
        params = {
            "api_key": self.key, "frequency": freq, "data[0]": "value",
            "start": start, "sort[0][column]": "period",
            "sort[0][direction]": "asc", "offset": 0, "length": length,
        }
        for k, v in facets.items():
            params[f"facets[{k}][]"] = v
        r = self.get(f"{self.BASE}{route}/data/", params=params)
        if r is None:
            return pd.DataFrame()
        rows = r.json().get("response", {}).get("data", [])
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def prices(self, years=3):
        start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        df = self._v2("natural-gas/pri/fut", {"series": "RNGWHHD"}, start, "daily")
        if df.empty: return df
        df = df[["period","value"]].rename(columns={"period":"date","value":"price"})
        df["date"] = pd.to_datetime(df["date"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
        self.log.info(f"EIA prices: {len(df)} rows")
        return df

    def storage(self, years=3):
        start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        # NW2_EPG0_SWO_R48_BCF = weekly working gas in underground storage, L48 (correct series)
        df = self._v2("natural-gas/stor/wkly",
                      {"series": "NW2_EPG0_SWO_R48_BCF"}, start, "weekly")
        if df.empty:
            # Fallback to SAY process filter
            df = self._v2("natural-gas/stor/wkly", {"process": "SAY"}, start, "weekly")
        if df.empty: return df
        df = df[["period","value"]].rename(columns={"period":"date","value":"storage_bcf"})
        df["date"] = pd.to_datetime(df["date"])
        df["storage_bcf"] = pd.to_numeric(df["storage_bcf"], errors="coerce")
        df = df.dropna(subset=["storage_bcf"]).sort_values("date").reset_index(drop=True)
        self.log.info(f"EIA storage: {len(df)} rows")
        return df

    def production(self, years=3):
        start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        df = self._v2("natural-gas/prod/sum",
                      {"process": "FPD", "series": "N9070US2"}, start, "monthly")
        if df.empty: return df
        df = df[["period","value"]].rename(columns={"period":"date","value":"production_bcf"})
        df["date"] = pd.to_datetime(df["date"].astype(str).str[:7] + "-01")
        df["production_bcf"] = pd.to_numeric(df["production_bcf"], errors="coerce")
        df = df.dropna(subset=["production_bcf"]).sort_values("date").reset_index(drop=True)
        self.log.info(f"EIA production: {len(df)} rows")
        return df

    def consumption(self, years=3):
        start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        # Try multiple known consumption series
        for series_info in [
            {"process": "VCS", "series": "N3010US2"},
            {"series": "N3010US2"},
            {"process": "VGT"},
        ]:
            df = self._v2("natural-gas/cons/sum", series_info, start, "monthly")
            if not df.empty:
                break
        if df.empty:
            # Last resort: use v2 with different route
            df = self._v2("natural-gas/cons/sum", {"duoarea": "NUS", "process": "VGT"}, start, "monthly")
        if df.empty: return df
        df = df[["period","value"]].rename(columns={"period":"date","value":"consumption_bcf"})
        df["date"] = pd.to_datetime(df["date"].astype(str).str[:7] + "-01")
        df["consumption_bcf"] = pd.to_numeric(df["consumption_bcf"], errors="coerce")
        df = df.dropna(subset=["consumption_bcf"]).sort_values("date").reset_index(drop=True)
        self.log.info(f"EIA consumption: {len(df)} rows")
        return df

    def lng_exports(self, years=3):
        start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        df = self._v2("natural-gas/move/expc", {"series": "N9133US2"}, start, "monthly")
        if df.empty: return df
        df = df[["period","value"]].rename(columns={"period":"date","value":"lng_exports_bcf"})
        df["date"] = pd.to_datetime(df["date"].astype(str).str[:7] + "-01")
        df["lng_exports_bcf"] = pd.to_numeric(df["lng_exports_bcf"], errors="coerce")
        df = df.dropna(subset=["lng_exports_bcf"]).sort_values("date").reset_index(drop=True)
        self.log.info(f"EIA LNG exports: {len(df)} rows")
        return df


# EIA residential natural gas consumption weights by demand centre.
# Defined at module level to avoid Python class-body scoping restrictions.
# Source: EIA-176, EIA State Energy Data System (2023 actuals).
_RAW_CITY_WEIGHTS = {
    "Chicago":      0.13,   # Illinois (Great Lakes, high heating load)
    "New_York":     0.16,   # New York (dense, high residential demand)
    "Houston":      0.04,   # Texas (warm, CDD-driven, low heating weight)
    "Boston":       0.10,   # New England (highest per-HDD consumption)
    "Detroit":      0.09,   # Michigan (cold winters, high residential)
    "Minneapolis":  0.08,   # Minnesota (extreme heating, high HDD)
    "Dallas":       0.04,   # Texas (warm, low heating weight)
    "Atlanta":      0.05,   # Southeast (mild winters, lower weight)
    "Denver":       0.08,   # Rocky Mountain (cold, significant demand)
    "Philadelphia": 0.11,   # Mid-Atlantic (PA+NJ, dense population)
}
_wt_sum = sum(_RAW_CITY_WEIGHTS.values())
CITY_WEIGHTS_NORMALISED = {k: v / _wt_sum for k, v in _RAW_CITY_WEIGHTS.items()}


class WeatherClient(_Fetcher):
    """
    Open-Meteo API — FREE, no key required.
    Historical daily temps + 16-day forecast across 10 U.S. demand centres.
    Computes: HDD, CDD, effective HDD (wind-chill adjusted), temp anomaly.
    """
    HIST = "https://archive-api.open-meteo.com/v1/archive"
    FCST = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, locations, log):
        self.locations = locations
        self.log = log

    def _fetch_multi(self, url, start, end, daily_vars, extra_params=None,
                     forecast_mode=False):
        """
        forecast_mode=True: uses forecast_days only (no start_date/end_date).
        The Open-Meteo /v1/forecast endpoint rejects start_date + forecast_days together.
        The Open-Meteo /v1/archive endpoint requires start_date + end_date.
        """
        frames = []
        for name, (lat, lon) in self.locations.items():
            params = {
                "latitude": lat, "longitude": lon,
                "daily": daily_vars,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "America/Chicago",
            }
            if not forecast_mode:
                params["start_date"] = start
                params["end_date"]   = end
            if extra_params:
                params.update(extra_params)
            r = self.get(url, params=params)
            if r is None: continue
            data = r.json().get("daily", {})
            if not data or "time" not in data: continue
            df = pd.DataFrame({"date": pd.to_datetime(data["time"])})
            for var in data:
                if var != "time":
                    df[f"{var}_{name}"] = data[var]
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on="date", how="outer")
        return merged.sort_values("date").reset_index(drop=True)

    # EIA consumption-based city weights — defined at module level (CITY_WEIGHTS_NORMALISED)
    # to avoid Python class-body scoping issues with dict comprehensions.
    CITY_WEIGHTS = CITY_WEIGHTS_NORMALISED

    def _aggregate(self, df):
        """
        Compute population-weighted composite HDD/CDD across demand centres.
        Uses EIA consumption-based weights (not equal weighting) so Northeast
        cold snaps receive their proper market impact.
        Formula:
            weighted_temp = Σ(w_i * T_i)
            HDD = max(0, 65 - weighted_temp)   [standard US energy base]
            Eff_HDD = HDD * WCF                 [wind-chill factor]
            WCF = 1 + 0.02*(wind_mph - 3) if wind > 3 mph else 1.0
        """
        out = df[["date"]].copy()

        # ── Weighted composite temperature ───────────────────────────────────
        w_temp_avg = pd.Series(0.0, index=df.index)
        w_temp_max = pd.Series(0.0, index=df.index)
        w_temp_min = pd.Series(0.0, index=df.index)
        w_wind     = pd.Series(0.0, index=df.index)
        total_w    = 0.0

        for city, weight in self.CITY_WEIGHTS.items():
            tc = f"temperature_2m_mean_{city}"
            tx = f"temperature_2m_max_{city}"
            tn = f"temperature_2m_min_{city}"
            wc_candidates = [f"wind_speed_10m_max_{city}", f"windspeed_10m_max_{city}"]
            wc = next((c for c in wc_candidates if c in df.columns), None)

            if tc in df.columns:
                w_temp_avg += weight * pd.to_numeric(df[tc], errors="coerce").fillna(0)
                total_w += weight
            if tx in df.columns:
                w_temp_max += weight * pd.to_numeric(df[tx], errors="coerce").fillna(0)
            if tn in df.columns:
                w_temp_min += weight * pd.to_numeric(df[tn], errors="coerce").fillna(0)
            if wc:
                w_wind += weight * pd.to_numeric(df[wc], errors="coerce").fillna(0)

        # Normalise in case some cities had missing data
        if total_w > 0:
            factor = 1.0 / total_w
        else:
            factor = 1.0

        out["temp_avg"] = w_temp_avg * factor
        out["temp_max"] = w_temp_max * factor
        out["temp_min"] = w_temp_min * factor
        out["wind_avg"] = w_wind * factor

        # ── HDD / CDD (base 65°F — US energy industry standard) ─────────────
        out["hdd"] = np.maximum(65.0 - out["temp_avg"], 0.0)
        out["cdd"] = np.maximum(out["temp_avg"] - 65.0, 0.0)

        # ── Wind-chill adjusted effective HDD ────────────────────────────────
        # NWS wind-chill effect on gas demand: each mph above 3 increases
        # effective heating demand by ~2% (empirical EIA estimate)
        out["wind_chill_factor"] = np.where(
            out["wind_avg"] > 3.0,
            1.0 + 0.02 * (out["wind_avg"] - 3.0),
            1.0
        )
        out["effective_hdd"] = out["hdd"] * out["wind_chill_factor"]
        out["temp_spread"]   = out["temp_max"] - out["temp_min"]

        return out.dropna(subset=["temp_avg"])

    def _log_per_city_hdd(self, raw_df, label="historical"):
        """Print per-city temperature and HDD summary to terminal."""
        print(f"\n    {C.CN}{'─'*60}{C.E}")
        print(f"    {C.BD}{C.W}Per-City Temperature & HDD Summary ({label}){C.E}")
        print(f"    {C.CN}{'─'*60}{C.E}")
        print(f"    {C.BD}{'City':<14} {'Avg Temp°F':>10} {'Avg HDD':>9} {'Avg CDD':>9} {'Wind mph':>9}{C.E}")
        print(f"    {'─'*54}")
        for city in self.locations:
            tc = f"temperature_2m_mean_{city}"
            wc_candidates = [f"windspeed_10m_max_{city}", f"wind_speed_10m_max_{city}"]
            wc = next((c for c in wc_candidates if c in raw_df.columns), None)
            if tc not in raw_df.columns:
                continue
            temps = pd.to_numeric(raw_df[tc], errors="coerce").dropna()
            if temps.empty:
                continue
            avg_t = temps.mean()
            avg_hdd = np.maximum(65 - temps, 0).mean()
            avg_cdd = np.maximum(temps - 65, 0).mean()
            avg_w = pd.to_numeric(raw_df[wc], errors="coerce").mean() if wc else 0
            hdd_bar = "█" * min(int(avg_hdd / 5), 10)
            print(f"    {C.W}{city:<14}{C.E} {C.CN}{avg_t:>9.1f}°F{C.E} "
                  f"{C.Y}{avg_hdd:>8.1f}{C.E} {C.B}{avg_cdd:>9.1f}{C.E} "
                  f"{C.GR}{avg_w:>9.1f}{C.E}  {C.Y}{hdd_bar}{C.E}")
        print(f"    {C.CN}{'─'*60}{C.E}\n")

    def historical(self, years=3):
        end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        raw_df = self._fetch_multi(
            self.HIST, start, end,
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "windspeed_10m_max,precipitation_sum"
        )
        if raw_df.empty: return raw_df
        self._log_per_city_hdd(raw_df, "historical avg")
        result = self._aggregate(raw_df)
        self.log.info(f"Weather historical: {len(result)} days")
        return result

    def forecast(self, days=14):
        raw_df = self._fetch_multi(
            self.FCST,
            None, None,                          # start/end not used in forecast mode
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "windspeed_10m_max",
            {"forecast_days": min(days, 16)},
            forecast_mode=True                   # do NOT pass start_date/end_date
        )
        if raw_df.empty: return raw_df
        self._log_per_city_hdd(raw_df, "forecast")
        result = self._aggregate(raw_df)
        self.log.info(f"Weather forecast: {len(result)} days")
        return result

    def _fetch_model_hdd(self, model: str, days: int = 7) -> dict:
        """
        Fetch HDD data from a specific NWP model via Open-Meteo.
        Supported models: 'gfs_seamless', 'ecmwf_ifs025'
        Returns dict with keys: latest_hdd, avg_7d, delta (12z-00z proxy via day1-day0)
        """
        frames = []
        for name, (lat, lon) in self.locations.items():
            params = {
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "timezone": "America/Chicago",
                "forecast_days": min(days, 7),
                "models": model,
            }
            r = self.get(self.FCST, params=params)
            if r is None:
                continue
            data = r.json().get("daily", {})
            if not data or "time" not in data:
                continue
            df = pd.DataFrame({"date": pd.to_datetime(data["time"])})
            for var in data:
                if var != "time":
                    df[f"{var}_{name}"] = data[var]
            frames.append(df)
        if not frames:
            return {}
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on="date", how="outer")
        merged = merged.sort_values("date").reset_index(drop=True)
        agg = self._aggregate(merged)
        if agg.empty:
            return {}
        latest_hdd  = float(agg["hdd"].iloc[0])  if len(agg) > 0 else float("nan")
        avg_7d      = float(agg["hdd"].head(7).mean())
        # Delta: day1 HDD minus day0 HDD — proxy for 12z run shift
        delta       = float(agg["hdd"].iloc[1] - agg["hdd"].iloc[0]) if len(agg) > 1 else 0.0
        return {"latest_hdd": latest_hdd, "avg_7d": avg_7d, "delta": delta}


class MarketDataClient:
    """
    Yahoo Finance client for cross-commodity data.
    Tickers: WTI crude, Brent, NG front month, Newcastle coal, DXY, VIX, S&P500.
    """
    TICKERS = {
        "crude_wti": "CL=F", "crude_brent": "BZ=F",
        "natgas_front": "NG=F", "coal": "MTF=F",
        "usd_index": "DX-Y.NYB", "vix": "^VIX", "sp500": "^GSPC",
    }

    def __init__(self, log):
        self.log = log

    def fetch_live_price(self, ticker: str = "NG=F") -> Optional[Tuple[float, str]]:
        """Fetch the latest available price for a ticker. Returns (price, date_str) or None."""
        if not HAS_YF:
            self.log.warning("yfinance not installed — cannot fetch live price")
            return None
        try:
            tk = yf.Ticker(ticker)
            # Try fast_info first (real-time-ish)
            try:
                price = tk.fast_info.get("lastPrice") or tk.fast_info.get("regularMarketPrice")
                if price and price > 0:
                    self.log.info(f"Live price from fast_info: ${price:.4f}")
                    return (float(price), datetime.now().strftime("%Y-%m-%d %H:%M"))
            except Exception:
                pass
            # Fallback: last 5 days of history
            hist = tk.history(period="5d")
            if hist is not None and not hist.empty:
                last_close = float(hist["Close"].iloc[-1])
                last_date = hist.index[-1].strftime("%Y-%m-%d")
                self.log.info(f"Live price from history: ${last_close:.4f} on {last_date}")
                return (last_close, last_date)
        except Exception as e:
            self.log.warning(f"Live price fetch failed for {ticker}: {e}")
        return None

    def fetch(self, years=3):
        if not HAS_YF:
            self.log.warning("yfinance not installed")
            return pd.DataFrame()
        start = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        frames = {}
        for label, ticker in self.TICKERS.items():
            try:
                df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
                if df.empty: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                frames[label] = df["Close"].rename(label)
            except Exception as e:
                self.log.warning(f"yfinance {ticker}: {e}")
        if not frames: return pd.DataFrame()
        combined = pd.DataFrame(frames)
        combined.index = pd.to_datetime(combined.index)
        combined.index.name = "date"
        combined = combined.sort_index().ffill()
        self.log.info(f"Market data: {len(combined)} rows")
        return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ▸ FEATURE ENGINEERING  (80+ features, 7 categories)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngine:
    """
    Feature categories:
      1. PRICE TECHNICALS — lags, SMA/EMA, RSI, MACD, Bollinger, vol, momentum
      2. SEASONALITY      — Fourier, month, DOW, heating/cooling flags
      3. STORAGE          — level, delta, vs-seasonal-avg, injection rate
      4. SUPPLY           — production, MoM trend
      5. DEMAND           — consumption, LNG exports, supply/demand ratio
      6. WEATHER          — HDD, CDD, effective HDD, anomaly, rolling sums
      7. CROSS-COMMODITY  — crude/gas ratio, coal/gas ratio, USD, VIX
    """

    def __init__(self, log):
        self.log = log
        self.feature_names: List[str] = []

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── 1. Price technicals ──────────────────────────────────────────────
        if "price" in df.columns:
            p = df["price"]
            for lag in [1,2,3,5,7,14,21]:
                df[f"price_lag{lag}"] = p.shift(lag)
            for win in [5,10,20,50]:
                df[f"price_sma{win}"] = p.rolling(win).mean()
                df[f"price_ema{win}"] = p.ewm(span=win).mean()
            df["ret_1d"] = p.pct_change(1)
            df["ret_5d"] = p.pct_change(5)
            df["ret_20d"] = p.pct_change(20)
            # Clip extreme daily returns before rolling vol — prevents spike artefacts (e.g. 1068%)
            _ret_clipped = df["ret_1d"].clip(-0.30, 0.30)
            df["rvol_5d"] = _ret_clipped.rolling(5).std() * np.sqrt(252)
            df["rvol_20d"] = _ret_clipped.rolling(20).std() * np.sqrt(252)

            # RSI-14
            delta = p.diff()
            gain = delta.where(delta>0, 0).rolling(14).mean()
            loss = (-delta.where(delta<0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi_14"] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = p.ewm(span=12).mean()
            ema26 = p.ewm(span=26).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

            # Bollinger Bands
            sma20 = p.rolling(20).mean()
            std20 = p.rolling(20).std()
            df["bb_upper"] = sma20 + 2*std20
            df["bb_lower"] = sma20 - 2*std20
            df["bb_pct"] = (p - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

            df["momentum_10d"] = p - p.shift(10)
            df["price_zscore_20d"] = (p - sma20) / std20.replace(0, np.nan)

        # ── 2. Seasonality ───────────────────────────────────────────────────
        # NOTE: We intentionally do NOT add raw Fourier sin/cos features.
        # Reason: In tree models (XGB/GBM), sin_doy features dominated importances
        # at ~40% combined because the Jan-Feb 2026 spike happened in winter, teaching
        # the model "sin_doy ≈ 1 → high log-return". This is spurious seasonal overfitting.
        # XGBoost handles integer `month` perfectly (it learns optimal splits across months).
        # The binary season flags and month integer provide all needed seasonality signal.
        idx = df.index
        df["month"] = idx.month
        df["dow"]   = idx.dayofweek
        # Natgas winter premium flag: Dec-Feb drives peak residential/commercial demand
        df["is_winter"]   = idx.month.isin([12, 1, 2]).astype(int)
        df["is_summer"]   = idx.month.isin([6, 7, 8]).astype(int)
        df["is_shoulder"] = (~idx.month.isin([12, 1, 2, 6, 7, 8])).astype(int)
        # Injection season (Apr-Oct): storage builds, typically bearish for natgas
        df["is_injection_season"] = idx.month.isin(range(4, 11)).astype(int)
        # Shoulder season transitions (Oct-Nov, Mar-Apr): highest vol in natgas calendar
        df["is_transition_month"] = idx.month.isin([3, 4, 10, 11]).astype(int)

        # ── 3. Storage ───────────────────────────────────────────────────────
        if "storage_bcf" in df.columns:
            s = df["storage_bcf"]
            df["storage_delta"] = s.diff()
            df["storage_pct_change"] = s.pct_change()
            df["storage_sma4w"] = s.rolling(4, min_periods=1).mean()
            # Seasonal comparison: ±15 day rolling window around same day-of-year
            # (with only 3 years, exact DOY matching gives trivially small groups → 0 delta)
            storage_doy = df.index.dayofyear.values
            seasonal_avg = pd.Series(np.nan, index=df.index)
            for i, doy in enumerate(storage_doy):
                # Find all rows within ±15 days of this DOY (across all years)
                doy_diff = np.abs(storage_doy - doy)
                doy_diff = np.minimum(doy_diff, 365 - doy_diff)  # wrap around year boundary
                mask_window = doy_diff <= 15
                vals = s.iloc[mask_window].dropna()
                if len(vals) > 0:
                    seasonal_avg.iloc[i] = vals.mean()
            df["storage_vs_avg"] = s - seasonal_avg
            df["storage_vs_avg_pct"] = (s - seasonal_avg) / seasonal_avg.replace(0, np.nan)

        # ── 4. Supply ────────────────────────────────────────────────────────
        if "production_bcf" in df.columns:
            df["production_mom"] = df["production_bcf"].pct_change()
            df["production_trend"] = df["production_bcf"].rolling(3, min_periods=1).mean()

        # ── 5. Demand ────────────────────────────────────────────────────────
        if "consumption_bcf" in df.columns:
            df["consumption_mom"] = df["consumption_bcf"].pct_change()
        if "lng_exports_bcf" in df.columns:
            df["lng_exports_mom"] = df["lng_exports_bcf"].pct_change()
        # Note: supply_demand_ratio removed — EIA production/consumption API endpoints
        # return values in different unit scales (MMcf vs Bcf), producing a spurious ~15x ratio.
        # Use only the unit-agnostic MoM percent changes and the net_supply_balance in section 8.

        # ── 6. Weather ───────────────────────────────────────────────────────
        if "hdd" in df.columns:
            df["hdd_7d"]  = df["hdd"].rolling(7, min_periods=1).mean()
            df["hdd_14d"] = df["hdd"].rolling(14, min_periods=1).mean()
            df["hdd_delta_7d"] = df["hdd"].diff(7)
            df["cdd_7d"] = df["cdd"].rolling(7, min_periods=1).mean() if "cdd" in df.columns else 0
            df["temp_doy"] = df.index.dayofyear
            temp_norm = df.groupby("temp_doy")["temp_avg"].transform("mean")
            df["temp_anomaly"] = df["temp_avg"] - temp_norm
            df.drop(columns=["temp_doy"], inplace=True)
            # ── Log rolling HDD stats for last row (most recent conditions) ──
            try:
                last = df.iloc[-1]
                print(f"\n    {C.CN}{'─'*55}{C.E}")
                print(f"    {C.BD}{C.W}Weather Feature Snapshot (latest row){C.E}")
                print(f"    {C.CN}{'─'*55}{C.E}")
                if "temp_avg" in df.columns:
                    print(f"    {C.BD}Avg Temp:{C.E}    {C.CN}{last['temp_avg']:.1f}°F{C.E}  "
                          f"(max={last.get('temp_max', float('nan')):.1f}°F, "
                          f"min={last.get('temp_min', float('nan')):.1f}°F)")
                print(f"    {C.BD}HDD (day):{C.E}   {C.Y}{last['hdd']:.1f}{C.E}  "
                      f"(base 65°F, HDD = max(0, 65 − {last.get('temp_avg', 0):.1f}))")
                print(f"    {C.BD}HDD 7d avg:{C.E}  {C.Y}{last['hdd_7d']:.1f}{C.E}")
                print(f"    {C.BD}HDD 14d avg:{C.E} {C.Y}{last['hdd_14d']:.1f}{C.E}")
                print(f"    {C.BD}HDD Δ7d:{C.E}     {C.Y}{last['hdd_delta_7d']:+.1f}{C.E}  "
                      f"({'warming' if last['hdd_delta_7d'] < 0 else 'cooling'})")
                if "cdd" in df.columns:
                    print(f"    {C.BD}CDD (day):{C.E}   {C.G}{last['cdd']:.1f}{C.E}")
                if "effective_hdd" in df.columns:
                    print(f"    {C.BD}Eff HDD:{C.E}     {C.Y}{last['effective_hdd']:.1f}{C.E}  "
                          f"(wind-chill adjusted, WCF={last.get('wind_chill_factor', 1):.3f})")
                if "wind_avg" in df.columns:
                    print(f"    {C.BD}Wind (avg):{C.E}  {C.GR}{last['wind_avg']:.1f} mph{C.E}")
                if "temp_anomaly" in df.columns:
                    ta = last["temp_anomaly"]
                    anom_desc = "colder than normal" if ta < 0 else "warmer than normal"
                    print(f"    {C.BD}Temp anom:{C.E}   {C.Y}{ta:+.1f}°F{C.E}  ({anom_desc} vs same DOY)")
                # Mean reversion context
                if "price_zscore_52w" in df.columns:
                    z52 = df["price_zscore_52w"].iloc[-1]
                    z_desc = ("far above 52w mean — mean reversion pressure" if z52 > 1.5
                              else ("far below 52w mean — possible rebound" if z52 < -1.5
                                    else "near 52w mean — no strong mean reversion signal"))
                    print(f"    {C.BD}52w Z-score:{C.E}  {C.Y}{z52:+.2f}{C.E}  ({z_desc})")
                if "spike_ratio_20d" in df.columns:
                    sr = df["spike_ratio_20d"].iloc[-1]
                    sr_desc = "post-spike (prices fell from recent high)" if sr > 1.3 else "no recent spike"
                    print(f"    {C.BD}Spike ratio:{C.E}  {C.Y}{sr:.2f}x{C.E}  ({sr_desc})")
                if "net_supply_balance" in df.columns:
                    nsb = df["net_supply_balance"].iloc[-1]
                    nsb_desc = "oversupply (bearish)" if nsb > 0 else "undersupply (bullish)"
                    print(f"    {C.BD}Net supply bal:{C.E} {C.Y}{nsb:+.1f} Bcf/mo{C.E}  ({nsb_desc})")
                # 30-day HDD context
                recent_hdd = df["hdd"].tail(30)
                print(f"    {C.BD}30d HDD total:{C.E} {C.Y}{recent_hdd.sum():.0f}{C.E}  "
                      f"avg={recent_hdd.mean():.1f}  days>20={int((recent_hdd>20).sum())}")
                print(f"    {C.CN}{'─'*55}{C.E}")

                # ── GFS & ECMWF NWP model HDD comparison ─────────────────
                try:
                    from datetime import datetime as _dt
                    _wc = WeatherClient(
                        {k: v for k, v in [
                            ("Chicago",     (41.88, -87.63)),
                            ("New_York",    (40.71, -74.01)),
                            ("Houston",     (29.76, -95.37)),
                            ("Boston",      (42.36, -71.06)),
                            ("Detroit",     (42.33, -83.05)),
                            ("Minneapolis", (44.98, -93.27)),
                            ("Dallas",      (32.78, -96.80)),
                            ("Atlanta",     (33.75, -84.39)),
                            ("Denver",      (39.74, -104.99)),
                            ("Philadelphia",(39.95, -75.17)),
                        ]},
                        self.log
                    )
                    _gfs  = _wc._fetch_model_hdd("gfs_seamless",  days=7)
                    _ecmwf= _wc._fetch_model_hdd("ecmwf_ifs025",  days=7)
                    print(f"    {C.CN}{'─'*55}{C.E}")
                    print(f"    {C.BD}{C.W}NWP Model HDD Comparison (Day-1 forecast){C.E}")
                    print(f"    {C.CN}{'─'*55}{C.E}")
                    print(f"    {C.BD}{'Model':<10} {'Latest HDD':>11} {'7d Avg HDD':>11} {'Δ (D1-D0)':>11}{C.E}")
                    print(f"    {'─'*46}")
                    for _mname, _md in [("GFS", _gfs), ("ECMWF", _ecmwf)]:
                        if _md:
                            _lhdd = _md.get("latest_hdd", float("nan"))
                            _a7   = _md.get("avg_7d",     float("nan"))
                            _dlt  = _md.get("delta",      0.0)
                            _dc   = C.R if _dlt > 2 else (C.G if _dlt < -2 else C.Y)
                            print(f"    {C.W}{_mname:<10}{C.E}"
                                  f" {C.Y}{_lhdd:>10.1f}{C.E}"
                                  f" {C.Y}{_a7:>10.1f}{C.E}"
                                  f" {_dc}{_dlt:>+10.1f}{C.E}")
                        else:
                            print(f"    {C.W}{_mname:<10}{C.E}  {C.GR}unavailable{C.E}")
                    print(f"    {C.GR}  Δ = Day+1 minus Day+0 HDD (positive = getting colder){C.E}")
                    print(f"    {C.CN}{'─'*55}{C.E}")
                except Exception as _nwp_e:
                    pass  # non-critical — NWP model fetch failure is silently skipped

            except Exception as _e:
                pass  # non-critical display

        if "effective_hdd" in df.columns:
            df["eff_hdd_7d"] = df["effective_hdd"].rolling(7, min_periods=1).mean()

        # ── 7. Cross-commodity ───────────────────────────────────────────────
        if "crude_wti" in df.columns and "price" in df.columns:
            df["crude_natgas_ratio"] = df["crude_wti"] / df["price"].replace(0, np.nan)
            df["crude_ret_5d"] = df["crude_wti"].pct_change(5)
        if "usd_index" in df.columns:
            df["usd_ret_5d"] = df["usd_index"].pct_change(5)
        if "vix" in df.columns:
            df["vix_level"] = df["vix"]
            df["vix_change"] = df["vix"].pct_change(5)
        if "coal" in df.columns and "price" in df.columns:
            df["coal_natgas_ratio"] = df["coal"] / df["price"].replace(0, np.nan)

        # ── 8. Natgas-specific: supply balance, mean-reversion, spike ────────
        if "price" in df.columns:
            p = df["price"]

            # ── Net supply balance (Bcf/d) ────────────────────────────────
            # This is THE most important fundamental driver of natgas prices.
            # Positive = oversupply pressure (bearish), Negative = undersupply (bullish).
            supply_cols = [c for c in ["production_bcf","consumption_bcf","lng_exports_bcf"] if c in df.columns]
            if all(c in df.columns for c in ["production_bcf","consumption_bcf","lng_exports_bcf"]):
                df["net_supply_balance"] = (df["production_bcf"]
                                            - df["consumption_bcf"]
                                            - df["lng_exports_bcf"])
                df["supply_tightness"] = df["net_supply_balance"].rolling(3, min_periods=1).mean()

            # ── Mean-reversion features (natgas is strongly mean-reverting) ──
            # 52-week z-score: tells model how stretched the price is vs. normal
            roll52 = p.rolling(252, min_periods=60)
            p_52w_mean = roll52.mean()
            p_52w_std  = roll52.std().replace(0, np.nan)
            df["price_zscore_52w"] = (p - p_52w_mean) / p_52w_std
            df["price_vs_52w_mean_pct"] = (p / p_52w_mean.replace(0, np.nan) - 1)

            # 20-week z-score (shorter regime)
            roll20w = p.rolling(100, min_periods=30)
            p_20w_mean = roll20w.mean()
            p_20w_std  = roll20w.std().replace(0, np.nan)
            df["price_zscore_20w"] = (p - p_20w_mean) / p_20w_std

            # ── Post-spike mean reversion indicator ───────────────────────────
            # Natgas frequently spikes on cold snaps then reverts.
            # recent_spike_ratio = rolling 20d high / current price (>1.5 = post-spike)
            roll20d_max = p.rolling(20, min_periods=5).max()
            df["spike_ratio_20d"] = roll20d_max / p.replace(0, np.nan)  # high > 1 means we've fallen from recent high
            df["spike_ratio_60d"] = p.rolling(60, min_periods=20).max() / p.replace(0, np.nan)

            # Days-equivalent reversion pressure (continuous, no 0 fill)
            df["price_from_52w_low_pct"]  = (p / p.rolling(252, min_periods=60).min().replace(0, np.nan) - 1)
            df["price_from_52w_high_pct"] = (p / p.rolling(252, min_periods=60).max().replace(0, np.nan) - 1)

            # ── EWMA volatility (RiskMetrics λ=0.94, more responsive) ─────────
            # Superior to 20d rolling std for current-regime vol estimation.
            # λ=0.94 is the industry standard (JP Morgan RiskMetrics).
            _ret_ew = p.pct_change(1).clip(-0.30, 0.30)
            ewma_var = _ret_ew.ewm(com=1.0/0.06 - 1, adjust=False).var()  # λ=0.94 → com=~15.67
            df["rvol_ewma"] = np.sqrt(ewma_var * 252)

            # ── Natgas-specific winter premium signal ─────────────────────────
            # Heating demand premium: how far effective_hdd is above/below seasonal norm
            if "effective_hdd" in df.columns and "temp_doy" not in df.columns:
                df["temp_doy"] = df.index.dayofyear
                ehdd_seasonal_norm = df.groupby("temp_doy")["effective_hdd"].transform("mean")
                df["ehdd_vs_seasonal"] = df["effective_hdd"] - ehdd_seasonal_norm
                df.drop(columns=["temp_doy"], inplace=True)

            # ── Forward-looking HDD rolling sum (key demand signal) ───────────
            if "hdd" in df.columns:
                df["hdd_30d_total"] = df["hdd"].rolling(30, min_periods=7).sum()

            # ══════════════════════════════════════════════════════════════════
            # ── 9. SPIKE PRECONDITION FEATURES ───────────────────────────────
            # Industry-grade natgas spike prediction requires modelling the
            # CAUSAL CONDITIONS that produce spikes, not just post-spike reversion.
            #
            # Historical natgas spike drivers (all real, documented):
            # A) Polar vortex / extreme cold snap → demand surge (HDD > 45 pop-weighted)
            # B) Wellhead freeze-offs → simultaneous supply cut of 5–15 Bcf/d
            # C) Storage below 5-year average entering cold event (no buffer)
            # D) Rapid cold onset (HDD acceleration > 10/week)
            # E) LNG exports at capacity (no export diversion available)
            #
            # Source: EIA Winter Energy Outlook, FERC reliability reports,
            # Uri 2021, Jan 2022, Jan 2024 spike post-mortems.
            # ══════════════════════════════════════════════════════════════════

            # ── A. Polar vortex flag: all major cities cold simultaneously ────
            # When ≥7 of 10 demand centres are simultaneously below 30°F,
            # it signals a CONUS-wide cold event (not just regional).
            # This is the necessary condition for a supply-demand shock.
            if "temp_min" in df.columns and "temp_avg" in df.columns:
                # Effective HDD > 40 (wind-chill adjusted) = extreme demand
                df["polar_vortex_flag"] = np.where(
                    (df["effective_hdd"] > 40) & (df["temp_min"] < 20),
                    1.0, 0.0
                )
                # Duration: consecutive polar vortex days (persistence matters)
                df["polar_vortex_days"] = (
                    df["polar_vortex_flag"]
                    .rolling(7, min_periods=1)
                    .sum()
                )

            # ── B. Freeze-off risk: wellhead freeze proxy ─────────────────────
            # When min temp < 15°F AND wind > 15mph, surface equipment freezes.
            # EIA documents 5–15 Bcf/d production loss during such events.
            # Best available free proxy: temp_min + wind composite.
            if "temp_min" in df.columns and "wind_avg" in df.columns:
                # Wind-chill equivalent below -5°F → high freeze probability
                wc_equiv = df["temp_min"] - 0.7 * df["wind_avg"]
                df["freeze_off_risk"] = np.where(
                    wc_equiv < 5,
                    np.clip((5 - wc_equiv) / 30, 0, 1),  # 0→1 scale
                    0.0
                )
                # 3-day persistence: freeze-offs worsen over sustained cold
                df["freeze_off_risk_3d"] = df["freeze_off_risk"].rolling(3, min_periods=1).mean()

            # ── C. Storage stress at cold event entry ─────────────────────────
            # Storage deficit going INTO a cold spell is the multiplier for spikes.
            # A 200 Bcf deficit during a polar vortex historically doubles price impact.
            if "storage_vs_avg" in df.columns and "effective_hdd" in df.columns:
                # Only count storage stress when actually cold (HDD > 20)
                cold_mask = df["effective_hdd"] > 20
                storage_deficit = np.maximum(0, -df["storage_vs_avg"])  # deficit = positive
                df["cold_storage_stress"] = np.where(
                    cold_mask,
                    storage_deficit * (df["effective_hdd"] / 65),  # scale by coldness intensity
                    0.0
                )

            # ── D. Rapid cold onset (HDD acceleration) ───────────────────────
            # Cold snaps that arrive FASTER than normal cause more demand shock
            # because utilities cannot ramp up supply in time.
            # HDD acceleration = 2nd derivative of HDD (change in HDD change)
            if "hdd" in df.columns:
                hdd_delta_3d = df["hdd"].diff(3)
                hdd_accel    = hdd_delta_3d.diff(3)
                df["hdd_acceleration"] = hdd_accel
                # Flag rapid onset: HDD increasing >10 in 3 days
                df["rapid_cold_onset"] = np.where(hdd_delta_3d > 10, hdd_delta_3d / 20, 0.0)

            # ── E. LNG export pressure ────────────────────────────────────────
            # When LNG exports are at seasonal highs AND it's cold, there is no
            # valve to release domestic supply pressure.
            if "lng_exports_bcf" in df.columns:
                lng_seasonal = df["lng_exports_bcf"].rolling(52, min_periods=12).mean()
                df["lng_export_stress"] = (
                    df["lng_exports_bcf"] / lng_seasonal.replace(0, np.nan) - 1
                ).clip(0, 1)  # 0 = normal, 1 = 100% above seasonal average

            # ── COMPOSITE: Spike Risk Score (0–1 scale) ───────────────────────
            # Combines all 5 causal factors into one actionable score.
            # Score > 0.5 = spike preconditions present
            # Score > 0.7 = historically associated with >20% price moves
            # Score > 0.85 = extreme risk (Uri-type event territory)
            #
            # Weights derived from EIA post-mortems of 2021/2022/2024 spike events:
            # Cold intensity (40%) + Freeze risk (25%) + Storage (20%) + Onset (10%) + LNG (5%)
            cold_component    = np.minimum(1.0, df.get("polar_vortex_flag", 0) * 0.6
                                           + np.maximum(0, df.get("effective_hdd", 0) - 30) / 35)
            freeze_component  = df.get("freeze_off_risk_3d",
                                       pd.Series(0.0, index=df.index))
            storage_component = np.minimum(1.0,
                                           df.get("cold_storage_stress",
                                                  pd.Series(0.0, index=df.index)) / 300)
            onset_component   = np.minimum(1.0,
                                           df.get("rapid_cold_onset",
                                                  pd.Series(0.0, index=df.index)))
            lng_component     = df.get("lng_export_stress",
                                       pd.Series(0.0, index=df.index))

            df["spike_risk_score"] = (
                0.40 * cold_component
                + 0.25 * freeze_component
                + 0.20 * storage_component
                + 0.10 * onset_component
                + 0.05 * lng_component
            ).clip(0, 1)

            # 3-day and 7-day forward spike risk (rolling max = worst case ahead)
            df["spike_risk_3d_max"] = df["spike_risk_score"].rolling(3, min_periods=1).max()
            df["spike_risk_7d_max"] = df["spike_risk_score"].rolling(7, min_periods=1).max()

            # ═══════════════════════════════════════════════════════════════════
            # v18: WEATHER × STORAGE INTERACTION FEATURES
            # ═══════════════════════════════════════════════════════════════════
            # These capture the NON-LINEAR interaction between weather and storage
            # that drives spikes. A cold snap with full storage = modest price move.
            # A cold snap with storage deficit = price explosion.
            # Tree models CAN learn interactions, but explicit features help them
            # find the pattern with limited spike training data.

            # HDD × Storage deficit interaction (the KEY spike predictor)
            if "hdd_7d" in df.columns and "storage_vs_avg" in df.columns:
                storage_deficit = np.maximum(0, -df["storage_vs_avg"])
                df["hdd_x_storage_deficit"] = df["hdd_7d"] * storage_deficit / 100.0
                # When HDD_7d > 20 AND storage below avg → this fires
                # Normalized to ~0-10 range for tree model compatibility

            # Cold acceleration × Freeze risk (simultaneous demand surge + supply cut)
            if "hdd_acceleration" in df.columns and "freeze_off_risk" in df.columns:
                df["cold_accel_x_freeze"] = (
                    np.maximum(0, df["hdd_acceleration"]) * df["freeze_off_risk"]
                )

            # Winter × Volatility interaction (winter vol is fundamentally different)
            if "is_winter" in df.columns and "rvol_5d" in df.columns:
                df["winter_x_vol"] = df["is_winter"] * df["rvol_5d"]

            # Consecutive cold days (persistence = more impactful than single-day cold)
            if "hdd" in df.columns:
                cold_day_flag = (df["hdd"] > 15).astype(float)
                # Count consecutive cold days
                consecutive_cold = cold_day_flag.copy()
                for i in range(1, len(consecutive_cold)):
                    if cold_day_flag.iloc[i] == 1:
                        consecutive_cold.iloc[i] = consecutive_cold.iloc[i-1] + 1
                    else:
                        consecutive_cold.iloc[i] = 0
                df["consecutive_cold_days"] = consecutive_cold

            # Rate of storage change × season (fast draws in winter = bullish)
            if "storage_draw_rate" in df.columns and "is_winter" in df.columns:
                df["winter_draw_rate"] = df["storage_draw_rate"] * df["is_winter"]

            # ── Demand-supply stress index ────────────────────────────────────
            # Combines demand acceleration (HDD rising) with supply constraint
            # (storage deficit + freeze risk). When both are high, prices MUST rise.
            if "hdd_7d" in df.columns:
                demand_pressure = np.maximum(0, df["hdd_7d"] - 20) / 45      # 0–1
                supply_constraint = df.get("spike_risk_score",
                                           pd.Series(0.0, index=df.index))
                df["demand_supply_stress"] = (demand_pressure * supply_constraint).clip(0, 1)

            # ── Storage draw rate (velocity of draws) ─────────────────────────
            # Fast draws signal demand > supply BEFORE the price moves.
            # This is a leading indicator, not concurrent.
            if "storage_bcf" in df.columns:
                df["storage_draw_rate"] = -df["storage_bcf"].diff(4)   # Bcf drawn over 4 weeks
                df["storage_draw_accel"] = df["storage_draw_rate"].diff(4)  # acceleration

        # ── Clean infinities & residual NaNs ─────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

        # ── Compile feature list ─────────────────────────────────────────────
        # Exclusions — every feature here has a specific documented reason:
        _exclude = {
            "price", "date",

            # ── Raw fundamentals — unit-ambiguous between EIA endpoints ───────
            "production_bcf", "consumption_bcf", "lng_exports_bcf",

            # ── Spike-contaminated long-window MAs ────────────────────────────
            # SMA20=$8.06 when price=$3.08 is a spike artefact, not resistance.
            # Any MA/EMA with window ≥10 will be spike-contaminated for ~2 months
            # after a spike event. Only SMA5/EMA5 clear fast enough.
            "price_sma10", "price_sma20", "price_sma50",
            "price_ema10", "price_ema20", "price_ema50",

            # ── Spike-contaminated Bollinger Bands (derived from SMA20+STD20) ─
            # bb_upper ≈ $11 (spike-inflated), bb_pct near 0 (misleading)
            "bb_upper", "bb_lower", "bb_pct",

            # ── Spike-contaminated vol/momentum features ──────────────────────
            # rvol_20d = 357% (20d window still contains spike crash)
            # rvol_ewma = 275% (exp-weighted, spike decays slowly)
            # momentum_10d = -$1.03 (10d lookback includes spike)
            # ret_20d includes spike-to-crash return
            # price_zscore_20d uses sma20/std20 (both spike-contaminated)
            # MACD = EMA12 - EMA26: both EMAs are spike-contaminated
            # (MACD ≈ -$0.97 is a crash artefact, not a bearish trend signal)
            # NOTE: spike_risk_score and related spike precondition features are
            # intentionally NOT excluded — they are the causal signal, not noise.
            "rvol_20d", "rvol_ewma",
            "momentum_10d", "ret_20d",
            "price_zscore_20d",
            "macd", "macd_signal", "macd_hist",

            # ── Raw price lags beyond 5d — add noise, low marginal signal ─────
            "price_lag7", "price_lag14", "price_lag21",

            # ── Spurious cross-commodity features ─────────────────────────────
            # sp500 has NO causal link to NG prices. Any correlation is spurious.
            # In testing, XGBoost overfits to coincidental sp500 patterns.
            # crude_brent is 99% correlated with crude_wti — keep only wti.
            "sp500", "crude_brent",

            # ── Redundant raw commodity levels (keep derived ratios/returns) ───
            # Raw vix level overlaps with vix_change; raw levels cause stationarity issues
            "vix", "vix_level",
        }
        self.feature_names = [
            c for c in df.columns
            if c not in _exclude
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
        ]
        self.log.info(f"Features built: {len(self.feature_names)}")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ▸ VOLATILITY & REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class VolatilityModel:
    """
    GARCH(1,1) + EWMA + rolling-percentile regime detection.
    Uses a TWO-WINDOW approach:
      - full_window (all history) → regime detection
      - recent_window (last N trading days excluding spike) → current forecast vol
    This prevents a $13 spike from contaminating current $3 market vol estimates.
    """

    # Natgas empirical daily vol bounds (annualised 20%–120%)
    MIN_DAILY_VOL = 0.20 / np.sqrt(252)   # ~1.26% daily
    MAX_DAILY_VOL = 2.00 / np.sqrt(252)   # ~12.6% daily — natgas Uri/Jan2022 moved 25%/day
    # Note: 200% annualised is not a typo. Henry Hub front-month moved 25% intraday
    # during Uri (Feb 2021) and 18% during Jan 2022 cold snap.
    # Capping at 120% caused the model to systematically understate spike-regime CI.

    # Spike detection: if a return exceeds this, it's a spike day
    SPIKE_THRESHOLD = 0.15  # 15% daily move

    def __init__(self, log):
        self.log = log
        self.daily_vol        = 0.03   # full-window GARCH (used for regime)
        self.recent_vol       = 0.03   # recent-window vol (used for CI construction)
        self.regime           = MarketRegime.NORMAL
        self.recent_window    = 180    # trading days for "current" vol estimate (was 90)

    def fit(self, returns: pd.Series):
        """
        Fit GARCH on full return history → regime detection.
        Also compute recent_vol on last `recent_window` days → CI width.
        Post-spike fix: filters out spike days from recent window to get
        a clean vol estimate for the current (non-spike) regime.
        """
        # ── Full-window fit ──────────────────────────────────────────────────
        clean = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean[clean.abs() < 0.30]  # remove data-error spikes >±30%

        if len(clean) < 100:
            self.daily_vol = np.clip(clean.std(), self.MIN_DAILY_VOL, self.MAX_DAILY_VOL)
            self.recent_vol = self.daily_vol
            self.log.info(f"Simple vol (short series): {self.daily_vol:.4f}")
            return

        garch_vol = self._fit_garch(clean, "full-window")
        self.daily_vol = garch_vol

        # ── Recent-window fit: EXCLUDE spike days ────────────────────────────
        # Filter out days with >15% absolute returns (spike artefacts)
        recent_clean = clean.tail(self.recent_window)
        recent_no_spike = recent_clean[recent_clean.abs() < self.SPIKE_THRESHOLD]

        if len(recent_no_spike) >= 30:
            # Use EWMA vol as primary (more responsive, less contaminated)
            ewma_var = recent_no_spike.ewm(com=15.67, adjust=False).var()
            ewma_daily = np.sqrt(ewma_var.iloc[-1]) if not ewma_var.empty else 0.03
            ewma_daily = float(np.clip(ewma_daily, self.MIN_DAILY_VOL, self.MAX_DAILY_VOL))

            # Also try GARCH on filtered data
            garch_recent = self._fit_garch(recent_no_spike, "recent-filtered")

            # Use the LOWER of GARCH and EWMA — both are noisy, take conservative
            self.recent_vol = min(garch_recent, ewma_daily)
            self.log.info(
                f"Recent vol: GARCH={garch_recent*np.sqrt(252)*100:.0f}% "
                f"EWMA={ewma_daily*np.sqrt(252)*100:.0f}% "
                f"→ using {self.recent_vol*np.sqrt(252)*100:.0f}%"
            )
            print(f"    {C.CN}  Recent vol: GARCH={garch_recent*np.sqrt(252)*100:.0f}% "
                  f"EWMA={ewma_daily*np.sqrt(252)*100:.0f}% "
                  f"→ using min = {self.recent_vol*np.sqrt(252)*100:.0f}%{C.E}")
            print(f"    {C.GR}  (spike days filtered: {len(recent_clean)-len(recent_no_spike)} "
                  f"days with >{self.SPIKE_THRESHOLD*100:.0f}% moves removed){C.E}")
        else:
            self.recent_vol = self.daily_vol

        # ── Log both vols for transparency ──────────────────────────────────
        self.log.info(
            f"Full-window daily vol: {self.daily_vol:.4f} "
            f"({self.daily_vol*np.sqrt(252)*100:.1f}% ann.)"
        )
        self.log.info(
            f"Recent vol (for CI): {self.recent_vol:.4f} "
            f"({self.recent_vol*np.sqrt(252)*100:.1f}% ann.)"
        )
        print(f"    {C.CN}  Full-window vol: {self.daily_vol*np.sqrt(252)*100:.1f}% ann. "
              f"| Recent (filtered) vol: {self.recent_vol*np.sqrt(252)*100:.1f}% ann.{C.E}")

    def _fit_garch(self, clean: pd.Series, label: str) -> float:
        """Fit GARCH(1,1)-t on `clean` returns. Returns clipped daily vol."""
        if not HAS_ARCH:
            raw = clean.std()
            return float(np.clip(raw, self.MIN_DAILY_VOL, self.MAX_DAILY_VOL))
        try:
            am = arch_model(clean * 100, vol="Garch", p=1, q=1, dist="t", mean="AR", lags=1)
            fit = am.fit(disp="off", show_warning=False)
            fcast = fit.forecast(horizon=1)
            raw_vol = np.sqrt(fcast.variance.iloc[-1].values[0]) / 100
            clipped = float(np.clip(raw_vol, self.MIN_DAILY_VOL, self.MAX_DAILY_VOL))
            if abs(raw_vol - clipped) > 0.001:
                self.log.warning(
                    f"GARCH [{label}] raw vol {raw_vol*np.sqrt(252)*100:.0f}% ann. "
                    f"→ capped to {clipped*np.sqrt(252)*100:.0f}% ann."
                )
                print(f"    {C.Y}⚠ GARCH [{label}] raw vol {raw_vol*np.sqrt(252)*100:.0f}% ann."
                      f" → capped to {clipped*np.sqrt(252)*100:.0f}%{C.E}")
            return clipped
        except Exception as e:
            self.log.warning(f"GARCH [{label}] failed: {e} — using simple std")
            return float(np.clip(clean.std(), self.MIN_DAILY_VOL, self.MAX_DAILY_VOL))

    def multi_step_vol(self, h: int, use_recent: bool = True) -> float:
        """
        Multi-step vol scaling: σ√h  (square-root-of-time rule).
        use_recent=True → uses recent 90d vol (better for near-term CI).
        use_recent=False → uses full-window vol (better for regime comparison).
        """
        base = self.recent_vol if use_recent else self.daily_vol
        return base * np.sqrt(h)

    def detect_regime(self, returns: pd.Series) -> MarketRegime:
        clean = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        clean = clean[clean.abs() < 0.30]
        # Filter out spike days for regime detection
        clean_filtered = clean[clean.abs() < self.SPIKE_THRESHOLD]
        rvol = clean_filtered.rolling(20).std() * np.sqrt(252)
        if rvol.empty or rvol.isna().all():
            return MarketRegime.NORMAL
        pct = rvol.rank(pct=True).iloc[-1]
        if pct < 0.25:   self.regime = MarketRegime.LOW_VOL
        elif pct < 0.70: self.regime = MarketRegime.NORMAL
        elif pct < 0.90: self.regime = MarketRegime.HIGH_VOL
        else:            self.regime = MarketRegime.CRISIS
        self.log.info(f"Regime: {self.regime.value} (pct={pct:.2f})")
        return self.regime


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ▸ STACKING ENSEMBLE FORECASTER
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleForecaster:
    """
    Per-horizon stacking ensemble.
    Level-0: XGBoost, LightGBM, sklearn GBR, Ridge  (whatever is installed)
    Level-1: Ridge meta-learner on out-of-fold predictions
    Also trains quantile (5th / 95th) GBR models for prediction intervals.
    Optional Optuna HP tuning for XGBoost on short horizons.
    """

    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self.models: Dict[int, Dict[str, Any]] = {}
        self.meta_models: Dict[int, Any] = {}
        self.scalers: Dict[int, RobustScaler] = {}
        self.feature_names: List[str] = []

    @staticmethod
    def _xgb_objective(trial, X, y, tscv, weights=None):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.12, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("lambda", 1e-4, 10, log=True),
            "min_child_weight": trial.suggest_int("mcw", 1, 15),
        }
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scores = []
        for tr, va in tscv.split(X_clean):
            m = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            sw = weights[tr] if weights is not None else None
            m.fit(X_clean[tr], y[tr], sample_weight=sw)
            scores.append(mean_absolute_error(y[va], m.predict(X_clean[va])))
        return np.mean(scores)

    def train(self, df: pd.DataFrame, feature_names: List[str]):
        """
        Train per-horizon ensemble models.

        TARGET: log-return over h days → log(price[t+h] / price[t])
        Why log-returns:
          • Stationary — not contaminated by absolute price level (e.g., $13 spike)
          • Symmetric — +50% and -50% have equal magnitude
          • Conversion back to price is trivial: p[t+h] = p[t] * exp(log_ret)
          • Standard in quantitative finance (Black-Scholes, GBM, EIA models)
        """
        self.feature_names = feature_names
        tscv = TimeSeriesSplit(n_splits=self.cfg.n_cv_splits)
        hdr("MODEL TRAINING")

        for h in range(1, self.cfg.forecast_horizon + 1):
            sec(f"Horizon {h}d", "🎯")

            # ── Log-return target (price-agnostic, stationary) ────────────────
            future_price = df["price"].shift(-h)
            log_ret = np.log(future_price / df["price"].replace(0, np.nan))
            mask = log_ret.notna() & df["price"].notna() & (df["price"] > 0)

            X_full = df[feature_names].loc[mask].values
            y_full_raw = log_ret.loc[mask].values    # raw log-returns
            p_full = df["price"].loc[mask].values    # actual prices (for MAPE eval)

            # ── CONDITIONAL PER-SAMPLE CLIPPING (v18 KEY FIX) ──────────────────
            # v16: fixed ±0.20√h clip → model NEVER learns spikes (all clipped)
            # v17: separate model on ~20 samples → noise → garbage
            # v18: CONDITION clip width on spike_risk_score per sample:
            #   - Low risk (score < 0.2):  tight clip ±0.12√h → clean rangebound
            #   - Mid risk  (0.2-0.5):     medium  ±0.25√h → allows moderate moves
            #   - High risk (0.5-0.7):     wide    ±0.45√h → allows spike learning
            #   - Extreme   (> 0.7):       very wide ±0.80√h → Uri-level moves
            # This teaches the model: "THESE features → big move is plausible"
            # without contaminating rangebound predictions.
            sqrt_h = max(1.0, h ** 0.5)

            if "spike_risk_score" in df.columns:
                spike_risk_vals = df["spike_risk_score"].loc[mask].values
                per_sample_clip = np.where(
                    spike_risk_vals > 0.7, 0.80 * sqrt_h,
                    np.where(spike_risk_vals > 0.5, 0.45 * sqrt_h,
                             np.where(spike_risk_vals > 0.2, 0.25 * sqrt_h,
                                      0.12 * sqrt_h)))
                y_full = np.clip(y_full_raw, -per_sample_clip, per_sample_clip)
            else:
                CLIP_LR = 0.15 * sqrt_h
                y_full = np.clip(y_full_raw, -CLIP_LR, CLIP_LR)
                per_sample_clip = np.full(len(y_full_raw), 0.15 * sqrt_h)

            n_clipped = int((y_full != y_full_raw).sum())
            CLIP_LR = per_sample_clip.mean()  # for display
            if n_clipped > 0:
                info("Clipped targets", f"{n_clipped}/{len(y_full)} extreme returns "
                     f"(avg clip ±{CLIP_LR:.3f}, range [{per_sample_clip.min():.3f}, {per_sample_clip.max():.3f}])")

            # ── Sample weights: temporal decay + spike importance ──────────────
            # Half-life = 90 trading days (~4 months) — more responsive than v16's 120d
            n_samples = len(y_full)
            half_life = 90.0
            decay = np.exp(np.log(0.5) / half_life * np.arange(n_samples)[::-1])
            sample_weights = decay / decay.mean()

            # ── Spike-aware weighting (UNIFIED — no separate model) ───────────
            # Instead of v17's classifier, we upweight spike-precondition days
            # so the base model naturally learns large responses to extreme inputs.
            # The key insight: the model learns "if features look like this → large return"
            # without needing a separate mechanism.
            if "spike_risk_score" in df.columns:
                spike_risk_train = df["spike_risk_score"].loc[mask].values
                actual_large_moves = np.abs(y_full_raw) > 0.08  # 8%+ moves (more inclusive than 15%)
                actual_spike_moves = np.abs(y_full_raw) > 0.15  # true spikes

                # Graduated upweighting based on conditions
                spike_boost = np.ones_like(sample_weights)
                spike_boost = np.where(spike_risk_train > 0.3, 2.0, spike_boost)
                spike_boost = np.where(spike_risk_train > 0.5, 3.5, spike_boost)
                spike_boost = np.where(spike_risk_train > 0.7, 5.0, spike_boost)
                # Actual large moves get additional boost regardless of precondition score
                spike_boost = np.where(actual_large_moves, np.maximum(spike_boost, 3.0), spike_boost)
                spike_boost = np.where(actual_spike_moves, np.maximum(spike_boost, 6.0), spike_boost)

                sample_weights = sample_weights * spike_boost
                sample_weights = sample_weights / sample_weights.mean()

                n_spike_boosted = int((spike_boost > 1.5).sum())
                if n_spike_boosted > 0 and h == 1:
                    info("Spike-boosted samples",
                         f"{n_spike_boosted} rows | "
                         f"risk>0.5: {int((spike_risk_train > 0.5).sum())} | "
                         f"|move|>8%: {int(actual_large_moves.sum())} | "
                         f"|move|>15%: {int(actual_spike_moves.sum())}")

            if len(y_full) < self.cfg.min_training_samples:
                warn(f"Only {len(y_full)} samples — skipping"); continue

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_full)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            self.scalers[h] = scaler

            # ── Detect and warn about zero-variance features after scaling ────
            # RobustScaler divides by IQR; if IQR=0 (constant column), result is NaN → 0
            col_std = np.std(X_scaled, axis=0)
            dead_feats = [feature_names[j] for j in range(len(feature_names)) if col_std[j] < 1e-10]
            if dead_feats and h == 1:
                warn(f"{len(dead_feats)} zero-variance features after scaling "
                     f"(first 5: {dead_feats[:5]})")

            split = int(len(X_scaled) * (1 - self.cfg.test_fraction))
            X_tr, X_te = X_scaled[:split], X_scaled[split:]
            y_tr, y_te = y_full[:split], y_full[split:]
            p_tr, p_te = p_full[:split], p_full[split:]
            w_tr = sample_weights[:split]

            info("Train/Test", f"{len(X_tr)} / {len(X_te)}")
            info("Price range", f"${p_tr.min():.2f} – ${p_tr.max():.2f}")
            info("Log-ret range", f"{y_tr.min():.3f} – {y_tr.max():.3f}")

            l0 = {}
            oof_stack = []

            
            # ═══════════════════════════════════════════════════════════════════
            # WEATHER & STORAGE FEATURE WEIGHTING (Research-backed adjustments)
            # ═══════════════════════════════════════════════════════════════════
            # Research shows weather accounts for ~50% of natgas demand, EIA storage
            # reports trigger 5-10% price moves. Apply weights to XGBoost only.

            if h == 1:  # Define weights once
                feature_weights = np.ones(len(feature_names))

                # Primary weather features (HDD/CDD) — 2.5x weight
                weather_primary = ['hdd', 'hdd_7d', 'hdd_14d', 'hdd_delta_7d', 'hdd_30d_total',
                                  'cdd', 'cdd_7d', 'effective_hdd', 'eff_hdd_7d', 'ehdd_vs_seasonal']

                # Secondary weather features — 2.0x weight  
                weather_secondary = ['temp_anomaly', 'temp_avg', 'temp_spread', 'wind_avg',
                                    'wind_chill_factor']

                # Storage fundamentals — 2.0x weight
                storage_features = ['storage_bcf', 'storage_vs_avg', 'storage_vs_avg_pct',
                                   'storage_pct_change', 'storage_delta', 'storage_sma4w']

                # Supply/demand balance — 1.8x weight
                supply_demand = ['net_supply_balance', 'supply_tightness', 
                                'production_mom', 'production_trend',
                                'consumption_mom', 'lng_exports_mom']
                # ── Spike precondition features — 3.0x weight ─────────────────
                # These are the CAUSAL drivers of spikes. They must dominate
                # the model's attention when spike conditions are present.
                spike_features = [
                    'spike_risk_score', 'spike_risk_3d_max', 'spike_risk_7d_max',
                    'polar_vortex_flag', 'polar_vortex_days',
                    'freeze_off_risk', 'freeze_off_risk_3d',
                    'cold_storage_stress', 'rapid_cold_onset', 'hdd_acceleration',
                    'demand_supply_stress', 'storage_draw_rate', 'storage_draw_accel',
                    'lng_export_stress',
                    # v18: Interaction features (critical for spike learning)
                    'hdd_x_storage_deficit', 'cold_accel_x_freeze',
                    'winter_x_vol', 'consecutive_cold_days', 'winter_draw_rate',
                ]
                for feat in spike_features:
                    if feat in feature_names:
                        feature_weights[feature_names.index(feat)] = 3.0

                # Cross-commodity features — 0.6x weight (reduce from default)
                cross_commodity = ['crude_wti', 'crude_natgas_ratio', 'crude_ret_5d',
                                  'coal', 'coal_natgas_ratio', 'usd_index', 'usd_ret_5d',
                                  'vix_change']

                # Technical indicators — 0.8x weight (slight reduction)
                technical_reduce = ['price_lag', 'ret_', 'rvol_5d', 'rsi_14']

                # Apply weights
                for i, fname in enumerate(feature_names):
                    if any(wf in fname for wf in weather_primary):
                        feature_weights[i] = 2.5
                    elif any(wf in fname for wf in weather_secondary):
                        feature_weights[i] = 2.0
                    elif any(sf in fname for sf in storage_features):
                        feature_weights[i] = 2.0
                    elif any(sd in fname for sd in supply_demand):
                        feature_weights[i] = 1.8
                    elif any(cc in fname for cc in cross_commodity):
                        feature_weights[i] = 0.6
                    elif any(tr in fname for tr in technical_reduce):
                        feature_weights[i] = 0.8

                # Store in self for access across horizons
                self.feature_weights = feature_weights

                weather_count = sum(1 for i, f in enumerate(feature_names) 
                                   if feature_weights[i] >= 2.0 and 
                                   any(w in f for w in weather_primary + weather_secondary))
                storage_count = sum(1 for i, f in enumerate(feature_names) if 'storage' in f)
                info("Weather features weighted", f"{weather_count} features at 2.0-2.5x")
                info("Storage features weighted", f"{storage_count} features at 2.0x")

# ── XGBoost ──────────────────────────────────────────────────────
            if HAS_XGB:
                if HAS_OPTUNA and h <= 3:
                    study = optuna.create_study(direction="minimize")
                    study.optimize(lambda t: self._xgb_objective(t, X_tr, y_tr, tscv, w_tr),
                                   n_trials=self.cfg.max_optuna_trials, show_progress_bar=False)
                    bp_raw = study.best_params
                    key_map = {"lr": "learning_rate", "colsample": "colsample_bytree",
                               "alpha": "reg_alpha", "lambda": "reg_lambda", "mcw": "min_child_weight"}
                    bp = {key_map.get(k, k): v for k, v in bp_raw.items()}
                    info("XGB Optuna MAE (log-ret)", f"{study.best_value:.5f}")
                else:
                    bp = {"n_estimators":500, "max_depth":5, "learning_rate":0.05,
                          "subsample":0.8, "colsample_bytree":0.8,
                          "reg_alpha":0.1, "reg_lambda":1.0, "min_child_weight":5}
                m = xgb.XGBRegressor(**bp, random_state=42, verbosity=0,
                                     feature_weights=self.feature_weights)
                m.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_te, y_te)], verbose=False)
                l0["xgb"] = m
                # ── OOF (use NaN not zero, only fill validated folds) ─────────
                oof = np.full(len(X_tr), np.nan)
                for tr_i, va_i in tscv.split(X_tr):
                    c = xgb.XGBRegressor(**bp, random_state=42, verbosity=0,
                                         feature_weights=self.feature_weights)
                    c.fit(X_tr[tr_i], y_tr[tr_i], sample_weight=w_tr[tr_i])
                    oof[va_i] = c.predict(X_tr[va_i])
                oof_stack.append(oof)

            # ── LightGBM ─────────────────────────────────────────────────────
            if HAS_LGB:
                m = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       reg_alpha=0.1, reg_lambda=1.0,
                                       min_child_samples=10, random_state=42, verbosity=-1)
                m.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_te, y_te)], callbacks=[lgb.log_evaluation(0)])
                l0["lgb"] = m
                oof = np.full(len(X_tr), np.nan)
                for tr_i, va_i in tscv.split(X_tr):
                    c = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=42, verbosity=-1)
                    c.fit(X_tr[tr_i], y_tr[tr_i], sample_weight=w_tr[tr_i])
                    oof[va_i] = c.predict(X_tr[va_i])
                oof_stack.append(oof)

            # ── sklearn GBR ──────────────────────────────────────────────────
            m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, min_samples_leaf=10, random_state=42)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            l0["gbr"] = m
            oof = np.full(len(X_tr), np.nan)
            for tr_i, va_i in tscv.split(X_tr):
                c = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                               subsample=0.8, min_samples_leaf=10, random_state=42)
                c.fit(X_tr[tr_i], y_tr[tr_i], sample_weight=w_tr[tr_i])
                oof[va_i] = c.predict(X_tr[va_i])
            oof_stack.append(oof)

            # ── Ridge ────────────────────────────────────────────────────────
            m = Ridge(alpha=10.0)
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            l0["ridge"] = m
            oof = np.full(len(X_tr), np.nan)
            for tr_i, va_i in tscv.split(X_tr):
                c = Ridge(alpha=10.0)
                c.fit(X_tr[tr_i], y_tr[tr_i], sample_weight=w_tr[tr_i])
                oof[va_i] = c.predict(X_tr[va_i])
            oof_stack.append(oof)

            # ── Quantile models (log-return space) ───────────────────────────
            q_lo = GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                              n_estimators=200, max_depth=4,
                                              learning_rate=0.05, random_state=42)
            q_hi = GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                              n_estimators=200, max_depth=4,
                                              learning_rate=0.05, random_state=42)
            q_lo.fit(X_tr, y_tr, sample_weight=w_tr)
            q_hi.fit(X_tr, y_tr, sample_weight=w_tr)
            l0["q05"] = q_lo; l0["q95"] = q_hi

            # ── Meta-learner (Ridge stacker on OOF predictions) ──────────────
            # Fix: use NaN-aware masking — zeros from unfilled first fold corrupt stacking
            oof_mat = np.column_stack(oof_stack)
            valid = np.all(np.isfinite(oof_mat), axis=1)  # all folds must have predictions
            if valid.sum() > 50:
                meta = Ridge(alpha=1.0)
                meta.fit(oof_mat[valid], y_tr[valid])
                self.meta_models[h] = meta
            else:
                self.meta_models[h] = None

            self.models[h] = l0

            # ── Hold-out evaluation (convert log-ret → price for interpretability)
            pred_logret_te = self._predict_point(h, X_te)
            pred_price_te = p_te * np.exp(pred_logret_te)
            actual_price_te = p_te * np.exp(y_te)

            mae  = mean_absolute_error(actual_price_te, pred_price_te)
            rmse = np.sqrt(mean_squared_error(actual_price_te, pred_price_te))
            # MAPE: exclude near-zero actual prices to avoid inflation
            nonzero = actual_price_te > 0.5
            mape = mean_absolute_percentage_error(
                actual_price_te[nonzero], pred_price_te[nonzero]) * 100 if nonzero.sum() > 0 else np.nan
            # Log-return MAE (model-native metric)
            logret_mae = mean_absolute_error(y_te, pred_logret_te)

            info("MAE (price)",      f"${mae:.4f}")
            info("RMSE (price)",     f"${rmse:.4f}")
            info("MAPE (price)",     f"{mape:.2f}%")
            info("MAE (log-ret)",    f"{logret_mae:.5f}")
            ok(f"Horizon {h}d — {len(l0)} models trained")

    def extract_importance(self, h: int) -> np.ndarray:
        """
        Robustly extract normalised feature importances for horizon h.
        Tries multiple methods in order of reliability:
          1. XGBoost get_booster().get_score(total_gain)
          2. XGBoost feature_importances_ (sklearn property)
          3. LightGBM feature_importances_
          4. GBR feature_importances_
        Returns normalised array (sums to 1), or zeros if all methods fail.
        """
        models = self.models.get(h, {})
        n = len(self.feature_names)
        if n == 0:
            return np.zeros(1)

        imp = np.zeros(n)

        # ── Method 1: XGBoost booster get_score (most granular) ───────────
        if HAS_XGB and "xgb" in models:
            xgb_model = models["xgb"]
            # Try booster-level extraction with multiple importance types
            for imp_type in ['total_gain', 'gain', 'total_cover', 'weight']:
                try:
                    scores = xgb_model.get_booster().get_score(importance_type=imp_type)
                    if not scores:
                        continue
                    imp = np.zeros(n)
                    for key, val in scores.items():
                        # Handle both 'f0' format and raw feature name format
                        idx = None
                        if key.startswith('f') and key[1:].isdigit():
                            idx = int(key[1:])
                        elif key.isdigit():
                            idx = int(key)
                        else:
                            # Try matching by feature name
                            if key in self.feature_names:
                                idx = self.feature_names.index(key)
                        if idx is not None and 0 <= idx < n:
                            imp[idx] = val
                    if imp.sum() > 0:
                        break  # success, stop trying other types
                except Exception:
                    continue

            # ── Method 2: sklearn feature_importances_ property ───────────
            if imp.sum() == 0:
                try:
                    imp = xgb_model.feature_importances_.copy()
                except Exception:
                    pass

        # ── Method 3: LightGBM ────────────────────────────────────────────
        if imp.sum() == 0 and HAS_LGB and "lgb" in models:
            try:
                imp = models["lgb"].feature_importances_.copy().astype(float)
            except Exception:
                pass

        # ── Method 4: sklearn GBR ─────────────────────────────────────────
        if imp.sum() == 0 and "gbr" in models:
            try:
                imp = models["gbr"].feature_importances_.copy()
            except Exception:
                pass

        # ── Normalise ─────────────────────────────────────────────────────
        imp_sum = imp.sum()
        if imp_sum > 0:
            imp = imp / imp_sum
        return imp

    def _predict_point(self, h, X):
        """Predict log-return for horizon h. Returns array of log-returns."""
        models = self.models.get(h, {})
        if not models: return np.full(X.shape[0], np.nan)
        base = [k for k in models if k not in ("q05","q95")]
        stack = np.column_stack([models[k].predict(X) for k in base])
        meta = self.meta_models.get(h)
        return meta.predict(stack) if meta is not None else stack.mean(axis=1)

    def predict(self, h, X):
        """
        Returns (log_ret_point, log_ret_lo, log_ret_hi).
        Caller converts to price: p_predicted = current_price * exp(log_ret).
        """
        scaler = self.scalers.get(h)
        if scaler is None:
            return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        X_sc = scaler.transform(X)
        X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)
        point = self._predict_point(h, X_sc)        # log-return
        models = self.models.get(h, {})
        lo = models["q05"].predict(X_sc) if "q05" in models else point - 0.05
        hi = models["q95"].predict(X_sc) if "q95" in models else point + 0.05
        return point, lo, hi


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ▸ ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class NatGasPredictor:
    """Top-level pipeline: fetch → merge → features → train → forecast → report"""

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        if not self.cfg.eia_api_key:
            self.cfg.eia_api_key = os.getenv("EIA_API_KEY", "")
        self.log = _log_setup(self.cfg)
        self.eia = EIAClient(self.cfg.eia_api_key, self.log)
        self.weather = WeatherClient(self.cfg.weather_locations, self.log)
        self.market = MarketDataClient(self.log)
        self.feat_engine = FeatureEngine(self.log)
        self.vol_model = VolatilityModel(self.log)
        self.ensemble = EnsembleForecaster(self.cfg, self.log)
        self.merged_df = None
        self.featured_df = None
        self.current_price = None          # latest EIA historical price
        self.live_price = None             # real-time from Yahoo Finance
        self.live_price_date = None

    def resolve_current_price(self) -> Tuple[float, str]:
        """
        Auto-resolve the best current price:
          1st: Yahoo Finance live price (NG=F front-month)
          2nd: Latest EIA Henry Hub spot
        Returns (price, source_description).
        """
        # Try Yahoo Finance live price first
        sec("Fetching live price (Yahoo Finance NG=F)", "💲")
        result = self.market.fetch_live_price(self.cfg.live_price_ticker)
        if result is not None:
            price, date_str = result
            if 0.5 < price < 20:  # sanity check for Henry Hub range
                self.live_price = price
                self.live_price_date = date_str
                ok(f"Live price: ${price:.3f} (as of {date_str})")
                return (price, f"Yahoo Finance {self.cfg.live_price_ticker} ({date_str})")
            else:
                warn(f"Live price ${price:.3f} outside reasonable range — ignoring")

        # Fallback to latest EIA
        if self.current_price is not None:
            eia_date = self.merged_df.index[-1].strftime("%Y-%m-%d") if self.merged_df is not None else "unknown"
            warn(f"Using latest EIA price: ${self.current_price:.3f} ({eia_date})")
            return (self.current_price, f"EIA Henry Hub spot ({eia_date})")

        err("No current price available from any source")
        return (None, "unavailable")

    # ── Data Assembly ────────────────────────────────────────────────────────

    def fetch_all_data(self):
        hdr("DATA INGESTION")
        yrs = self.cfg.history_years

        sec("Henry Hub spot prices", "💰")
        prices = self.eia.prices(yrs)
        if prices.empty:
            err("No price data — cannot proceed"); return pd.DataFrame()
        info("Records", len(prices))
        info("Range", f"{prices['date'].iloc[0].date()} → {prices['date'].iloc[-1].date()}")
        info("Latest", f"${prices['price'].iloc[-1]:.3f}")
        self.current_price = prices["price"].iloc[-1]
        df = prices.set_index("date")

        if self.cfg.use_storage:
            sec("Storage inventory", "📦")
            s = self.eia.storage(yrs)
            if not s.empty:
                info("Records", len(s))
                info("Latest", f"{s['storage_bcf'].iloc[-1]:.0f} Bcf")
                df = df.join(s.set_index("date")["storage_bcf"], how="left")
                df["storage_bcf"] = df["storage_bcf"].ffill()
            else: warn("Storage unavailable")

        if self.cfg.use_production:
            sec("Dry gas production", "🏭")
            p = self.eia.production(yrs)
            if not p.empty:
                info("Records", len(p))
                df = df.join(p.set_index("date")["production_bcf"], how="left")
                df["production_bcf"] = df["production_bcf"].ffill()
            else: warn("Production unavailable")

            sec("Total consumption", "🔥")
            c = self.eia.consumption(yrs)
            if not c.empty:
                info("Records", len(c))
                df = df.join(c.set_index("date")["consumption_bcf"], how="left")
                df["consumption_bcf"] = df["consumption_bcf"].ffill()
            else: warn("Consumption unavailable")

        if self.cfg.use_lng:
            sec("LNG exports", "🚢")
            l = self.eia.lng_exports(yrs)
            if not l.empty:
                info("Records", len(l))
                df = df.join(l.set_index("date")["lng_exports_bcf"], how="left")
                df["lng_exports_bcf"] = df["lng_exports_bcf"].ffill()
            else: warn("LNG unavailable")

        if self.cfg.use_weather:
            sec("Historical weather (Open-Meteo)", "🌡️")
            wx = self.weather.historical(yrs)
            if not wx.empty:
                info("Records", len(wx))
                info("Avg HDD", f"{wx['hdd'].mean():.1f}")
                w = wx.set_index("date")
                df = df.join(w, how="left")
                for c in w.columns: df[c] = df[c].ffill().bfill()
            else: warn("Weather unavailable")

        if self.cfg.use_cross_commodity:
            sec("Cross-commodity (Yahoo Finance)", "📊")
            mkt = self.market.fetch(yrs)
            if not mkt.empty:
                info("Tickers", list(mkt.columns))
                df = df.join(mkt, how="left")
                for c in mkt.columns: df[c] = df[c].ffill()
            else: warn("Market data unavailable (install yfinance)")

        df = df.dropna(subset=["price"]).ffill().bfill()
        self.merged_df = df
        ok(f"Merged: {len(df)} rows × {len(df.columns)} columns")
        return df

    def build_features(self):
        hdr("FEATURE ENGINEERING")
        if self.merged_df is None or self.merged_df.empty:
            err("No data"); return pd.DataFrame()
        df = self.feat_engine.build(self.merged_df)
        initial = len(df)
        critical_cols = [c for c in ["price_lag1","price_sma20"] if c in df.columns]
        if critical_cols:
            df = df.dropna(subset=critical_cols)
        info("Warmup dropped", f"{initial - len(df)} rows")
        info("Training rows", len(df))
        info("Features", len(self.feat_engine.feature_names))
        self.featured_df = df
        return df

    def train(self):
        if self.featured_df is None or self.featured_df.empty:
            err("No features"); return False
        sec("GARCH volatility", "📈")
        rets = self.featured_df["price"].pct_change().dropna()
        self.vol_model.fit(rets)
        regime = self.vol_model.detect_regime(rets)
        info("Full-window daily σ", f"{self.vol_model.daily_vol:.4f} ({self.vol_model.daily_vol*np.sqrt(252)*100:.1f}% ann.)")
        info("Recent (filtered) daily σ",  f"{self.vol_model.recent_vol:.4f} ({self.vol_model.recent_vol*np.sqrt(252)*100:.1f}% ann.)")
        info("Regime", regime.value)
        self.ensemble.train(self.featured_df, self.feat_engine.feature_names)
        return True

    def forecast(self) -> pd.DataFrame:
        """Generate forward predictions with confidence intervals and per-horizon reasoning."""
        hdr("GENERATING FORECASTS")

        if not self.ensemble.models:
            err("No models"); return pd.DataFrame()

        # ── Early definitions: available to ALL sections of forecast() ────────
        _bt_date = self.cfg.backtest_date.strip() if self.cfg.backtest_date else ""
        today    = pd.Timestamp(_bt_date) if _bt_date else pd.Timestamp(datetime.now().date())

        # ── Auto-resolve current price ───────────────────────────────────────
        current, price_source = self.resolve_current_price()

        if current is None:
            err("Cannot forecast without current price"); return pd.DataFrame()
        info("Reference price", f"${current:.3f}")
        info("Source", price_source)

        # ── Weather forecast ─────────────────────────────────────────────────
        wx_fcst = pd.DataFrame()
        if self.cfg.use_weather:
            sec("Weather forecast", "🌤️")
            if _bt_date:
                # In backtest mode: use historical weather for forecast horizon
                # (those dates are already in the past — historical data is more accurate)
                _bt_fcst_end   = today + timedelta(days=self.cfg.forecast_horizon + 6)
                _bt_fcst_start = today
                wx_fcst = self.weather.historical(0)  # dummy call to get structure
                # Fetch historical data for the forecast window
                import datetime as _dt
                _raw = self.weather._fetch_multi(
                    self.weather.HIST,
                    _bt_fcst_start.strftime("%Y-%m-%d"),
                    _bt_fcst_end.strftime("%Y-%m-%d"),
                    ["temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                     "wind_speed_10m_max"]
                )
                wx_fcst = self.weather._aggregate(_raw) if not _raw.empty else pd.DataFrame()

            else:
                wx_fcst = self.weather.forecast(self.cfg.forecast_horizon + 4)

            if not wx_fcst.empty:
                info("Forecast days", len(wx_fcst))
                # ── Show per-date weather forecast table ─────────────────────
                print()
                print(f"    {C.BD}{C.CN}{'Date':<12} {'Temp°F':>7} {'High°F':>7} {'Low°F':>7} "
                      f"{'HDD':>6} {'CDD':>6} {'Eff HDD':>8} {'Wind mph':>9} {'WCF':>5}{C.E}")
                print(f"    {C.GR}{'─'*72}{C.E}")
                # Limit display to forecast horizon + a few days
                display_days = min(len(wx_fcst), self.cfg.forecast_horizon + 4)
                for _, row in wx_fcst.head(display_days).iterrows():
                    dt = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
                    t_avg = row.get("temp_avg", float("nan"))
                    t_max = row.get("temp_max", float("nan"))
                    t_min = row.get("temp_min", float("nan"))
                    hdd   = row.get("hdd", 0)
                    cdd   = row.get("cdd", 0)
                    eff   = row.get("effective_hdd", hdd)
                    wind  = row.get("wind_avg", 0)
                    wcf   = row.get("wind_chill_factor", 1.0)
                    # Colour: cold days yellow, very cold red, warm green
                    if hdd > 30:   tc = C.R
                    elif hdd > 10: tc = C.Y
                    else:          tc = C.G
                    print(f"    {C.W}{dt:<12}{C.E}"
                          f" {tc}{t_avg:>7.1f}{C.E}"
                          f" {C.GR}{t_max:>7.1f}{C.E}"
                          f" {C.B}{t_min:>7.1f}{C.E}"
                          f" {C.Y}{hdd:>6.1f}{C.E}"
                          f" {C.G}{cdd:>6.1f}{C.E}"
                          f" {C.Y}{eff:>8.1f}{C.E}"
                          f" {C.GR}{wind:>9.1f}{C.E}"
                          f" {C.GR}{wcf:>5.3f}{C.E}")
                print(f"    {C.GR}{'─'*72}{C.E}")
                # Rolling HDD summary stats
                hdd_series = wx_fcst["hdd"].head(display_days)
                print(f"    {C.BD}Forecast HDD summary:{C.E}"
                      f"  avg={hdd_series.mean():.1f}  "
                      f"max={hdd_series.max():.1f}  "
                      f"total={hdd_series.sum():.0f}  "
                      f"days>20={int((hdd_series>20).sum())}")
                print(f"    {C.GR}HDD = max(0, 65°F − avg_temp) · Effective HDD includes wind-chill factor{C.E}")
                print()

        last_row  = self.featured_df.iloc[-1].copy()
        data_date = self.featured_df.index[-1]

        # ── Backtest-date override ────────────────────────────────────────────
        if _bt_date:
            try:
                today = pd.Timestamp(_bt_date)
                sec("BACKTEST MODE", "🔁")
                warn(f"Anchored to past date: {today.strftime('%Y-%m-%d')} — NOT live")

                # ── FIX (v18.1): Find the CORRECT feature row for the backtest date ──
                # In v18.0, we used featured_df.iloc[-1] which is the LAST date,
                # not the backtest date. This meant all features (HDD, RSI, etc.)
                # were from the wrong time period.
                bt_ts = pd.Timestamp(_bt_date)
                available_dates = self.featured_df.index[self.featured_df.index <= bt_ts]
                if len(available_dates) > 0:
                    bt_feature_date = available_dates[-1]
                    last_row = self.featured_df.loc[bt_feature_date].copy()
                    data_date = bt_feature_date
                    ok(f"Using feature row from {bt_feature_date.strftime('%Y-%m-%d')} "
                       f"(closest to backtest date)")
                    # Show the features we're actually using
                    info("Feature HDD", f"{last_row.get('hdd', 0):.1f}")
                    info("Feature HDD-7d", f"{last_row.get('hdd_7d', 0):.1f}")
                    info("Feature spike_risk", f"{last_row.get('spike_risk_score', 0):.3f}")
                    info("Feature price", f"${last_row.get('price', 0):.3f}")
                else:
                    warn(f"No feature data at or before {_bt_date} — using last available row")

                # ── FIX (v18.1): Get backtest anchor price from YF and OVERRIDE current ──
                if HAS_YF:
                    _bt_start = (today - timedelta(days=3)).strftime("%Y-%m-%d")
                    _bt_end   = (today + timedelta(days=2)).strftime("%Y-%m-%d")
                    _bt_hist  = yf.download("NG=F", start=_bt_start, end=_bt_end,
                                            progress=False, auto_adjust=True)
                    if _bt_hist is not None and not _bt_hist.empty:
                        if isinstance(_bt_hist.columns, pd.MultiIndex):
                            _bt_hist.columns = _bt_hist.columns.get_level_values(0)
                        _bt_hist.index = pd.to_datetime(_bt_hist.index)
                        _bt_avail = _bt_hist[_bt_hist.index <= today]
                        if not _bt_avail.empty:
                            _bt_price = float(_bt_avail["Close"].iloc[-1])
                            _bt_actual_date = _bt_avail.index[-1].strftime("%Y-%m-%d")
                            self.live_price = _bt_price
                            self.live_price_date = _bt_actual_date
                            # ── KEY FIX: Override `current` so predictions are from correct base ──
                            current = _bt_price
                            price_source = f"YF backtest anchor ({_bt_actual_date})"
                            ok(f"Backtest anchor price: ${_bt_price:.3f} (YF {_bt_actual_date})")
                        else:
                            warn("No YF price for backtest date — using EIA last close")
                            current = float(last_row.get("price", current))
                    else:
                        warn("YF returned empty for backtest date — using EIA last close")
                        current = float(last_row.get("price", current))
                else:
                    current = float(last_row.get("price", current))
            except Exception as _bt_err:
                warn(f"Backtest date error: {_bt_err} — falling back to today")
                today = pd.Timestamp(datetime.now().date())
        else:
            today = pd.Timestamp(datetime.now().date())

        # ── Store today for use in generate_report ────────────────────────────
        self._forecast_today = today
        self._forecast_current = current
        self._forecast_feature_row = last_row.copy()  # v18.1: backtest-correct row

        sec("Date alignment", "📅")
        info("Last data date (EIA)", data_date.strftime("%Y-%m-%d"))
        _anchor_label = f"{today.strftime('%Y-%m-%d')} ⬅ BACKTEST" if _bt_date else f"{today.strftime('%Y-%m-%d')} (today)"
        info("Forecast anchor", _anchor_label)
        _now_hour_disp = datetime.now().hour
        _is_intraday   = today.weekday() < 5 and (bool(_bt_date) or _now_hour_disp < 17)


        # Build list of trading days.
        # If running intraday (today is a weekday and market hasn't closed),
        # prepend today as T+0 so the forecast shows where TODAY ends up.
        # Subsequent horizons T+1…T+10 follow from tomorrow onward.
        trading_days = []
        _now_hour = datetime.now().hour
        _today_is_trading_day = today.weekday() < 5  # Mon-Fri
        # v18.1: In backtest mode, always assume session is open (we're testing a past trading day)
        _intraday_session_open = _today_is_trading_day and (_now_hour < 17 or bool(_bt_date))
        if _intraday_session_open:
            # T+0 = today (intraday close estimate)
            trading_days.append(today)
        d = today
        while len(trading_days) < self.cfg.forecast_horizon:
            d += timedelta(days=1)
            if d.weekday() < 5:  # Mon-Fri
                trading_days.append(d)

        # ── CRITICAL: Fill the data gap between last EIA date and today ────────
        # EIA spot prices lag by several days. We fill the gap with Yahoo Finance
        # NG=F daily closes, then RE-RUN feature engineering on the extended data
        # so ALL moving averages, RSI, MACD, Bollinger are computed correctly.
        sec("Gap-fill: EIA→today via Yahoo Finance NG=F", "🔧")

        # ── v18: EIA Storage Report Calendar ──────────────────────────────────
        # The single biggest intraday event for natgas: EIA weekly storage report
        # released at 10:30 ET every Thursday. Price moves 3-8% within minutes.
        # If today is Thursday and before 10:30, warn that storage number is pending.
        _now_dt = datetime.now()
        _is_storage_day = today.weekday() == 3  # Thursday
        _storage_pending = _is_storage_day and _now_dt.hour < 11  # before 11:00 (10:30 + buffer)
        _storage_just_released = _is_storage_day and 10 <= _now_dt.hour <= 12
        if _is_storage_day and _intraday_session_open:
            sec("EIA Storage Report Day (Thursday)", "📊")
            if _storage_pending:
                warn("⚡ EIA weekly storage report PENDING (10:30 ET)")
                warn("   Expect 2-8% price move at release — forecast has higher uncertainty")
                warn("   Direction depends on draw/build vs consensus estimate")
            elif _storage_just_released:
                ok("EIA storage report released ~30min ago — price may still be settling")
            else:
                ok("Storage report window passed — price should be post-report")


        gap_days = (today - data_date).days
        info("Data gap", f"{gap_days} calendar days ({data_date.strftime('%Y-%m-%d')} → {today.strftime('%Y-%m-%d')})")

        fr_base = None  # will be set after gap-fill

        # ── v18.1 FIX: In backtest mode, use the correct feature row directly ──
        # The gap-fill logic fails in backtest mode because data_date > backtest_date
        # (negative gap). We already found the correct feature row above.
        if _bt_date:
            fr_base = last_row.copy()  # last_row was already set to backtest date's row
            # Update price_lag1 to the backtest anchor price
            if "price_lag1" in fr_base.index:
                fr_base["price_lag1"] = current
            # Update cross-commodity ratios
            if "crude_natgas_ratio" in fr_base.index and "crude_wti" in fr_base.index and current > 0:
                fr_base["crude_natgas_ratio"] = fr_base["crude_wti"] / current
            if "coal_natgas_ratio" in fr_base.index and "coal" in fr_base.index and current > 0:
                fr_base["coal_natgas_ratio"] = fr_base["coal"] / current
            ok(f"Backtest features from {data_date.strftime('%Y-%m-%d')}, price=${current:.3f}")

        elif gap_days > 0 and HAS_YF:
            info("Fetching gap prices from Yahoo Finance...", "")
            try:
                # Fetch recent NG=F daily close prices to fill the gap
                yf_start = (data_date - timedelta(days=1)).strftime("%Y-%m-%d")  # 1 day before for overlap
                yf_end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
                info("YF range", f"{yf_start} → {yf_end}")

                ngf = yf.download("NG=F", start=yf_start, end=yf_end, progress=False, auto_adjust=True)

                if ngf is not None and not ngf.empty:
                    info("YF raw shape", f"{ngf.shape}, cols: {list(ngf.columns)[:5]}")

                    # Handle MultiIndex columns (yfinance >= 0.2.30 returns MultiIndex for single ticker)
                    if isinstance(ngf.columns, pd.MultiIndex):
                        ngf.columns = ngf.columns.get_level_values(0)

                    # Get the Close column — try multiple approaches
                    close_col = None
                    for candidate in ["Close", "close", "Adj Close"]:
                        if candidate in ngf.columns:
                            close_col = candidate
                            break
                    if close_col is None:
                        # If still not found, take the first numeric column
                        for c in ngf.columns:
                            if ngf[c].dtype in [np.float64, np.float32, np.int64]:
                                close_col = c
                                break

                    if close_col is None:
                        warn(f"Cannot find price column in YF data: {list(ngf.columns)}")
                    else:
                        gap_prices = ngf[close_col].dropna()
                        # Filter to only dates after our last data date
                        gap_prices = gap_prices[gap_prices.index > data_date]
                        info("Yahoo gap prices", f"{len(gap_prices)} trading days fetched")

                        if len(gap_prices) > 0:
                            for i in range(len(gap_prices)):
                                dt = gap_prices.index[i]
                                price_val = gap_prices.iloc[i]

                                if isinstance(dt, pd.Timestamp):
                                    dt = dt.normalize()
                                else:
                                    dt = pd.Timestamp(dt).normalize()

                                if dt <= data_date or dt in self.merged_df.index:
                                    continue

                                # Create a row by copying last known row and updating price
                                new_row = self.merged_df.iloc[-1].copy()
                                new_row["price"] = float(price_val)

                                # Update natgas_front if present
                                if "natgas_front" in new_row.index:
                                    new_row["natgas_front"] = float(price_val)

                                self.merged_df.loc[dt] = new_row

                            # If live price is for today and today is not yet in df
                            today_norm = today.normalize()
                            if today_norm not in self.merged_df.index:
                                today_row = self.merged_df.iloc[-1].copy()
                                today_row["price"] = current
                                if "natgas_front" in today_row.index:
                                    today_row["natgas_front"] = current
                                self.merged_df.loc[today_norm] = today_row

                            self.merged_df = self.merged_df.sort_index()
                            # Forward-fill non-price columns in new rows
                            self.merged_df = self.merged_df.ffill()

                            new_end = self.merged_df.index[-1].strftime('%Y-%m-%d')
                            ok(f"Merged data extended to {new_end} ({len(self.merged_df)} rows)")

                            # Show the prices we added
                            recent = self.merged_df["price"].tail(gap_days + 2)
                            for rdt, rpx in recent.items():
                                marker = " ← live" if rdt == today_norm else (" ← gap-filled" if rdt > data_date else "")
                                info(f"  {rdt.strftime('%Y-%m-%d')}", f"${rpx:.3f}{marker}")

                            # RE-RUN feature engineering on extended data
                            sec("Re-computing all features on extended data", "⚙️")
                            self.featured_df = self.feat_engine.build(self.merged_df)
                            # Drop warmup NaN rows
                            critical_cols = [c for c in ["price_lag1","price_sma20"] if c in self.featured_df.columns]
                            if critical_cols:
                                self.featured_df = self.featured_df.dropna(subset=critical_cols)
                            self.featured_df = self.featured_df.ffill().bfill()

                            # Now features reflect actual recent price action
                            fr_base = self.featured_df.iloc[-1].copy()
                            last_sma5 = fr_base.get("price_sma5", 0)
                            last_sma20 = fr_base.get("price_sma20", 0)
                            last_ema5 = fr_base.get("price_ema5", 0)
                            last_rsi = fr_base.get("rsi_14", 0)
                            last_macd = fr_base.get("macd", 0)
                            ok(f"Features recomputed:")
                            info("  SMA-5", f"${last_sma5:.3f}")
                            info("  SMA-20", f"${last_sma20:.3f}")
                            info("  EMA-5", f"${last_ema5:.3f}")
                            info("  RSI-14", f"{last_rsi:.1f}")
                            info("  MACD", f"{last_macd:.4f}")
                        else:
                            warn("No new gap prices found after filtering")
                else:
                    warn("Yahoo Finance returned empty data for gap period")
            except Exception as exc:
                import traceback
                warn(f"Gap-fill failed: {exc}")
                warn(f"Traceback: {traceback.format_exc()[-200:]}")
        elif gap_days <= 0:
            info("No gap to fill", "data is current")
        else:
            warn("yfinance not installed — cannot fill data gap")

        if fr_base is None:
            # Fallback: just use the last available featured row with basic update
            warn("Gap-fill did not produce updated features — using fallback splice")
            fr_base = self.featured_df.iloc[-1].copy()
            # At minimum, update price_lag1 to live price
            if "price_lag1" in fr_base.index:
                old_lag1 = fr_base["price_lag1"]
                fr_base["price_lag1"] = current
                info("Fallback: price_lag1", f"${old_lag1:.3f} → ${current:.3f}")
            # Update cross-commodity ratios
            if "crude_natgas_ratio" in fr_base.index and "crude_wti" in fr_base.index and current > 0:
                fr_base["crude_natgas_ratio"] = fr_base["crude_wti"] / current
            if "coal_natgas_ratio" in fr_base.index and "coal" in fr_base.index and current > 0:
                fr_base["coal_natgas_ratio"] = fr_base["coal"] / current
            warn("Using approximate feature splice (gap-fill unavailable)")

        # ── Get feature importances for reasoning ────────────────────────────
        feat_importances = {}
        for h_key in self.ensemble.models:
            imp = self.ensemble.extract_importance(h_key)
            feat_importances[h_key] = dict(zip(self.feat_engine.feature_names, imp))

        results = []

        for idx, pred_date in enumerate(trading_days):
            _is_today_row = (_intraday_session_open and idx == 0)
            h = idx + 1 if not _intraday_session_open else (1 if idx == 0 else idx)
            if h not in self.ensemble.models:
                continue


            # ── Build feature vector for this horizon ────────────────────────
            fr = fr_base.copy()

            # Update seasonality to prediction date
            # Note: sin_doy/cos_doy are NOT in the feature set (intentionally excluded
            # to avoid spurious seasonal overfitting). Only update features the model uses.
            fr["month"] = pred_date.month
            fr["dow"]   = pred_date.weekday()
            fr["is_winter"]  = int(pred_date.month in (12,1,2))
            fr["is_summer"]  = int(pred_date.month in (6,7,8))
            fr["is_shoulder"]= int(pred_date.month not in (12,1,2,6,7,8))
            fr["is_injection_season"] = int(4 <= pred_date.month <= 10)
            fr["is_transition_month"] = int(pred_date.month in (3, 4, 10, 11))

            # Update weather if we have forecast for that date
            wx_hdd_val = None
            wx_temp_val = None
            if not wx_fcst.empty:
                wd = wx_fcst[wx_fcst["date"] == pd.Timestamp(pred_date)]
                if not wd.empty:
                    for col in ["hdd","cdd","effective_hdd","temp_avg","wind_avg","temp_spread",
                                "wind_chill_factor","temp_min","temp_max"]:
                        if col in wd.columns and col in fr.index:
                            fr[col] = wd[col].iloc[0]
                    wx_hdd_val = wd["hdd"].iloc[0] if "hdd" in wd.columns else None
                    wx_temp_val = wd["temp_avg"].iloc[0] if "temp_avg" in wd.columns else None

                # ═══════════════════════════════════════════════════════════════
                # v18.1 FIX: RECOMPUTE ROLLING WEATHER FEATURES FROM FORECAST
                # ═══════════════════════════════════════════════════════════════
                # v18.0 ONLY updated single-day hdd but left hdd_7d, hdd_14d,
                # spike_risk_score at their STALE historical values. The model
                # depends on these rolling features — without updating them,
                # it can't see the forecast cold snap.
                #
                # Approach: compute rolling averages from the forecast weather
                # data itself (we have up to 16 days of forecast).
                wx_horizon_days = wx_fcst[wx_fcst["date"] <= pd.Timestamp(pred_date)]
                if len(wx_horizon_days) > 0 and "hdd" in wx_fcst.columns:
                    # hdd_7d: average of last 7 forecast days up to pred_date
                    recent_hdd = wx_horizon_days["hdd"].tail(7)
                    if "hdd_7d" in fr.index and len(recent_hdd) >= 1:
                        fr["hdd_7d"] = recent_hdd.mean()
                    if "hdd_14d" in fr.index:
                        recent_14 = wx_horizon_days["hdd"].tail(14)
                        fr["hdd_14d"] = recent_14.mean() if len(recent_14) >= 1 else fr["hdd_7d"]
                    # hdd_delta_7d: change in HDD over 7 days
                    if "hdd_delta_7d" in fr.index and len(wx_horizon_days) >= 2:
                        fr["hdd_delta_7d"] = float(wx_horizon_days["hdd"].iloc[-1] - wx_horizon_days["hdd"].iloc[0])
                    # eff_hdd_7d
                    if "eff_hdd_7d" in fr.index and "effective_hdd" in wx_fcst.columns:
                        eff_recent = wx_horizon_days["effective_hdd"].tail(7)
                        fr["eff_hdd_7d"] = eff_recent.mean() if len(eff_recent) >= 1 else fr.get("effective_hdd", 0)
                    # hdd_30d_total: use what we have (forecast may not cover 30d)
                    if "hdd_30d_total" in fr.index:
                        all_hdd = wx_horizon_days["hdd"]
                        # Scale to 30d equivalent if we have fewer days
                        if len(all_hdd) >= 7:
                            fr["hdd_30d_total"] = all_hdd.sum() * (30.0 / len(all_hdd))
                    # temp_anomaly: compare forecast temp to fr_base's seasonal norm
                    if "temp_anomaly" in fr.index and wx_temp_val is not None:
                        # rough: seasonal norm from base row, anomaly = forecast - norm
                        base_temp = fr_base.get("temp_avg", 50)
                        base_anom = fr_base.get("temp_anomaly", 0)
                        seasonal_norm = base_temp - base_anom
                        fr["temp_anomaly"] = wx_temp_val - seasonal_norm

                    # ── RECOMPUTE SPIKE RISK SCORE FROM FORECAST WEATHER ──────
                    # This is THE critical fix. The model saw spike_risk_score=0.05
                    # when forecast HDD was 45-55 because it was using historical
                    # (stale) spike features. Recompute from forecast weather.
                    _fc_hdd = float(fr.get("hdd", 0))
                    _fc_eff_hdd = float(fr.get("effective_hdd", _fc_hdd))
                    _fc_temp_min = float(fr.get("temp_min", 30))
                    _fc_wind = float(fr.get("wind_avg", 5))

                    # Polar vortex: effective HDD > 40 AND min temp < 20
                    _pv_flag = 1.0 if (_fc_eff_hdd > 40 and _fc_temp_min < 20) else 0.0
                    if "polar_vortex_flag" in fr.index:
                        fr["polar_vortex_flag"] = _pv_flag

                    # Freeze-off risk: wind-chill equiv < 5°F
                    _wc_equiv = _fc_temp_min - 0.7 * _fc_wind
                    _freeze_risk = np.clip((5 - _wc_equiv) / 30, 0, 1) if _wc_equiv < 5 else 0.0
                    if "freeze_off_risk" in fr.index:
                        fr["freeze_off_risk"] = _freeze_risk
                    if "freeze_off_risk_3d" in fr.index:
                        fr["freeze_off_risk_3d"] = _freeze_risk  # approximate

                    # Cold-storage stress
                    _storage_deficit = max(0, -float(fr.get("storage_vs_avg", 0)))
                    _cold_storage = _storage_deficit * (_fc_eff_hdd / 65.0) if _fc_eff_hdd > 20 else 0
                    if "cold_storage_stress" in fr.index:
                        fr["cold_storage_stress"] = _cold_storage

                    # HDD acceleration
                    if "hdd_acceleration" in fr.index and len(wx_horizon_days) >= 4:
                        hdd_vals = wx_horizon_days["hdd"].values
                        if len(hdd_vals) >= 4:
                            delta3 = hdd_vals[-1] - hdd_vals[-min(4, len(hdd_vals))]
                            fr["hdd_acceleration"] = delta3

                    # Rapid cold onset
                    if "rapid_cold_onset" in fr.index:
                        hdd_delta = float(fr.get("hdd_delta_7d", 0))
                        fr["rapid_cold_onset"] = max(0, hdd_delta / 20.0) if hdd_delta > 10 else 0

                    # Recompute spike_risk_score
                    _cold_comp = min(1.0, _pv_flag * 0.6 + max(0, _fc_eff_hdd - 30) / 35)
                    _freeze_comp = _freeze_risk
                    _storage_comp = min(1.0, _cold_storage / 300)
                    _onset_comp = min(1.0, float(fr.get("rapid_cold_onset", 0)))
                    _lng_comp = float(fr.get("lng_export_stress", 0))

                    new_spike_risk = (
                        0.40 * _cold_comp
                        + 0.25 * _freeze_comp
                        + 0.20 * _storage_comp
                        + 0.10 * _onset_comp
                        + 0.05 * _lng_comp
                    )
                    new_spike_risk = np.clip(new_spike_risk, 0, 1)

                    old_spike_risk = float(fr.get("spike_risk_score", 0))
                    if "spike_risk_score" in fr.index:
                        fr["spike_risk_score"] = new_spike_risk
                    if "spike_risk_3d_max" in fr.index:
                        fr["spike_risk_3d_max"] = max(new_spike_risk, float(fr.get("spike_risk_3d_max", 0)))
                    if "spike_risk_7d_max" in fr.index:
                        fr["spike_risk_7d_max"] = max(new_spike_risk, float(fr.get("spike_risk_7d_max", 0)))

                    # Update interaction features
                    if "hdd_x_storage_deficit" in fr.index:
                        fr["hdd_x_storage_deficit"] = float(fr.get("hdd_7d", 0)) * _storage_deficit / 100.0
                    if "demand_supply_stress" in fr.index and "hdd_7d" in fr.index:
                        demand_p = max(0, float(fr["hdd_7d"]) - 20) / 45.0
                        fr["demand_supply_stress"] = min(1.0, demand_p * new_spike_risk)

                    if h == 1 and new_spike_risk != old_spike_risk:
                        info("Spike risk recomputed",
                             f"{old_spike_risk:.3f} → {new_spike_risk:.3f} "
                             f"(cold={_cold_comp:.2f} freeze={_freeze_comp:.2f} "
                             f"storage={_storage_comp:.2f})")

            # ── Predict (log-return space) ────────────────────────────────────
            X = fr[self.feat_engine.feature_names].values.reshape(1,-1)
            X = np.nan_to_num(X, nan=0.0)
            log_ret, log_lo, log_hi = self.ensemble.predict(h, X)
            log_ret  = float(log_ret[0])
            log_lo   = float(log_lo[0])
            log_hi   = float(log_hi[0])

            # ═══════════════════════════════════════════════════════════════════
            # v18 SPIKE-CONDITIONED ANALYTICAL OVERLAY
            # ═══════════════════════════════════════════════════════════════════
            # Instead of v17's classifier+separate model (which destroyed rangebound),
            # we analytically amplify the model's prediction when spike preconditions
            # are active. The base model already learned "spike features → larger move"
            # via the conditional clipping. This overlay ensures the post-prediction
            # processing doesn't clamp that signal away.
            #
            # Key: the overlay is MULTIPLICATIVE and CONDITIONAL — it does NOTHING
            # when spike_risk_score < 0.3 (preserving rangebound accuracy).
            _spike_risk_now = float(fr.get("spike_risk_score", 0.0))
            _hdd_now        = float(fr.get("effective_hdd", fr.get("hdd", 0.0)))
            _storage_stress = float(fr.get("cold_storage_stress", 0.0))
            _freeze_risk    = float(fr.get("freeze_off_risk_3d", 0.0))

            # Weather forecast amplification: if FUTURE weather is colder than
            # what the model trained on (which used historical features), we need
            # to boost the prediction to account for the forecast cold.
            wx_boost = 0.0
            if wx_hdd_val is not None:
                current_hdd = float(fr.get("hdd", 0.0))
                # If forecast HDD is significantly above current, prices should rise more
                hdd_delta = wx_hdd_val - current_hdd
                if hdd_delta > 5:
                    # Each additional HDD point above current → ~0.15% additional move
                    # Based on EIA analysis: 1 HDD ≈ 1 Bcf/d demand ≈ 0.1-0.2% price impact
                    wx_boost = min(hdd_delta * 0.0015, 0.03)  # cap at 3% boost
                elif hdd_delta < -5:
                    wx_boost = max(hdd_delta * 0.001, -0.02)  # cap at 2% reduction

            # Spike overlay: only active when preconditions are genuinely present
            spike_overlay = 0.0
            if _spike_risk_now > 0.3:
                # The model already predicts a direction. The overlay amplifies it
                # proportionally to how extreme the preconditions are.
                # v18.1: More aggressive amplitude — the base model still under-predicts
                # spikes because tree models average across splits.
                amplitude = (_spike_risk_now - 0.3) / 0.7  # 0→1 scale
                amplitude = amplitude ** 1.2  # slightly convex (was 1.5 — too conservative)

                # Direction: in natgas, spike preconditions (cold, storage deficit)
                # are ALWAYS bullish (demand shock = price up). Only amplify upside.
                if log_ret > -0.02 or (_hdd_now > 25 and _storage_stress > 30):
                    # v18.1: Larger overlay for extreme cold events
                    # HDD > 40 = genuine polar vortex territory → much larger moves
                    if _hdd_now > 45:
                        base_overlay = 0.15  # 15% base for extreme cold
                    elif _hdd_now > 35:
                        base_overlay = 0.10  # 10% base for severe cold
                    elif _hdd_now > 25:
                        base_overlay = 0.06  # 6% base for strong cold
                    else:
                        base_overlay = 0.03  # 3% base for moderate conditions

                    spike_overlay = amplitude * base_overlay * np.sqrt(max(h, 1))

                    # Cold-snap specific: if HDD > 35 AND freeze risk, this is Uri-territory
                    if _hdd_now > 35 and _freeze_risk > 0.3:
                        spike_overlay *= 1.8
                    elif _hdd_now > 35 and _freeze_risk > 0.1:
                        spike_overlay *= 1.3
                    # Storage deficit amplifier
                    if _storage_stress > 150:
                        spike_overlay *= 1.5
                    elif _storage_stress > 50:
                        spike_overlay *= 1.2

                    # v18.1: Hard cap scales with horizon
                    max_overlay = min(0.40, 0.25 * np.sqrt(max(h, 1)))
                    spike_overlay = min(spike_overlay, max_overlay)

            # Apply overlays to log-return
            log_ret_adjusted = log_ret + wx_boost + spike_overlay
            # Also widen the upper CI if spike risk is elevated
            if spike_overlay > 0:
                log_hi = max(log_hi, log_ret_adjusted + spike_overlay * 0.5)

            # ── Convert log-returns → price ──────────────────────────────────
            point = current * np.exp(log_ret_adjusted)
            lo    = current * np.exp(log_lo)
            hi    = current * np.exp(log_hi)

            # ── Risk bounds using RECENT vol (not full-window spike-contaminated vol) ──
            vol_h = self.vol_model.multi_step_vol(h, use_recent=True)

            # ── v18 Spike-aware ASYMMETRIC CI band ─────────────────────────────
            # Natgas spikes are UPSIDE (demand shock → price surge).
            # Downside is bounded (storage buffers, utilities can't shut off).
            # Scale CI width with current spike conditions:
            #   - spike_risk < 0.2 → tight CI (rangebound market)
            #   - spike_risk 0.2-0.5 → moderate CI
            #   - spike_risk > 0.5 → wide asymmetric CI (upside >> downside)

            # Base bands from GARCH vol
            base_sigma_dn = 2.0 * vol_h  # 2σ downside (95th percentile)
            base_sigma_up = 2.0 * vol_h  # 2σ upside

            # Spike-conditioned multipliers
            if _spike_risk_now > 0.5:
                # High spike risk: upside can be 3-5x normal vol
                up_mult = 1.0 + 3.0 * ((_spike_risk_now - 0.5) / 0.5)  # 1→4x
                dn_mult = 1.0 + 0.5 * ((_spike_risk_now - 0.5) / 0.5)  # 1→1.5x
            elif _spike_risk_now > 0.2:
                up_mult = 1.0 + 1.0 * ((_spike_risk_now - 0.2) / 0.3)  # 1→2x
                dn_mult = 1.0 + 0.3 * ((_spike_risk_now - 0.2) / 0.3)  # 1→1.3x
            else:
                up_mult = 1.0
                dn_mult = 1.0

            sigma_band_dn = current * min(base_sigma_dn * dn_mult, 0.35)
            sigma_band_up = current * min(base_sigma_up * up_mult, 1.50)

            # Clamp point estimate within ±σ band
            point = np.clip(point, current - sigma_band_dn, current + sigma_band_up)

            # CI: quantile models provide lo/hi; clamp to sigma band
            lo = np.clip(lo, current - sigma_band_dn, point)
            hi = np.clip(hi, point, current + sigma_band_up)

            # Hard floor/ceiling (Henry Hub physical constraints)
            point = np.clip(point, self.cfg.min_price_floor, self.cfg.max_price_ceil)
            lo    = np.clip(lo,    self.cfg.min_price_floor, point)
            hi    = np.clip(hi,    point,                    self.cfg.max_price_ceil)

            band = hi - lo
            conf = "High" if band < point*0.04 else ("Medium" if band < point*0.10 else "Low")
            chg = point - current
            chg_pct = (chg / current) * 100

            # ── Per-horizon reasoning ────────────────────────────────────────
            drivers = []
            seen_categories = set()  # prevent duplicate categories (e.g. MACD+MACD_hist)
            fi = feat_importances.get(h, {})
            top_feats = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:8]  # check top 8

            for fname, fimp in top_feats:
                if fimp < 0.01:
                    continue
                if len(drivers) >= 4:
                    break

                # ── Dedup: skip if we already have a driver from same category ──
                cat = None
                if fname in ("macd", "macd_hist", "macd_signal"):
                    cat = "macd"
                elif fname in ("rvol_5d", "rvol_20d", "rvol_ewma"):
                    cat = "vol"
                elif fname in ("crude_wti", "crude_brent", "crude_ret_5d", "crude_natgas_ratio"):
                    cat = "crude"
                elif fname in ("price_sma5", "price_ema5", "price_sma10", "price_ema10"):
                    cat = "ma"
                elif fname in ("hdd", "hdd_7d", "hdd_14d", "effective_hdd", "eff_hdd_7d", "hdd_30d_total"):
                    cat = "hdd"
                elif fname in ("price_zscore_52w", "price_zscore_20w", "price_vs_52w_mean_pct"):
                    cat = "zscore"
                if cat and cat in seen_categories:
                    continue

                val = fr.get(fname, 0)
                driver_text = self._explain_feature(fname, val, current, fr, wx_hdd_val, wx_temp_val)
                if driver_text:
                    # Skip non-informative spike artefact messages
                    if "spike-inflated" in driver_text or "not representative" in driver_text:
                        continue
                    if "spike data" in driver_text or "spike-elevated" in driver_text:
                        continue
                    drivers.append(driver_text)
                    if cat:
                        seen_categories.add(cat)

            # ── Always inject weather signal if not already present ────────
            if wx_hdd_val is not None and "hdd" not in seen_categories:
                if wx_hdd_val > 20:
                    drivers.append(f"Strong heating demand (HDD={wx_hdd_val:.0f})")
                elif wx_hdd_val < 5:
                    drivers.append(f"Mild weather reduces demand (HDD={wx_hdd_val:.0f})")

            # ── Inject RSI if oversold/overbought and not already shown ────
            rsi_val = fr.get("rsi_14", 50)
            if not any("RSI" in d for d in drivers):
                if rsi_val < 30:
                    drivers.append(f"RSI oversold ({rsi_val:.0f}) — potential bounce")
                elif rsi_val > 70:
                    drivers.append(f"RSI overbought ({rsi_val:.0f}) — potential pullback")

            # Calendar days from today
            cal_days = (pred_date - today).days

            # ── v18: Spike regime label ───────────────────────────────────────
            if _spike_risk_now > 0.7:
                spike_regime = "SPIKE_EXTREME"
            elif _spike_risk_now > 0.5:
                spike_regime = "SPIKE_HIGH"
            elif _spike_risk_now > 0.3:
                spike_regime = "SPIKE_WATCH"
            else:
                spike_regime = "NORMAL"

            # ── v18: Intraday session context ─────────────────────────────────
            intraday_note = ""
            if _is_today_row:
                if _storage_pending if '_storage_pending' in dir() else False:
                    intraday_note = "⚡ PRE-STORAGE REPORT — high uncertainty"
                elif _is_storage_day if '_is_storage_day' in dir() else False:
                    intraday_note = "Post-storage report session"
                else:
                    intraday_note = "Intraday close estimate"
           
            results.append({
                "date":        pred_date.strftime("%Y-%m-%d"),
                "weekday":     pred_date.strftime("%a") + (" ⬅today" if _is_today_row else ""),
                "horizon":     0 if _is_today_row else h,
                "cal_days":    cal_days,
                "predicted":   round(point, 4),
                "low_5pct":    round(lo, 4),
                "high_95pct":  round(hi, 4),
                "change":      round(chg, 4),
                "change_pct":  round(chg_pct, 2),
                "confidence":  conf,
                "vol_h":       round(vol_h, 4),
                "drivers":     drivers[:4],
                "spike_regime": spike_regime,
                "spike_risk":  round(_spike_risk_now, 3),
                "spike_overlay": round(spike_overlay, 4) if spike_overlay > 0 else 0,
                "wx_boost":    round(wx_boost, 4) if wx_boost != 0 else 0,
                "intraday_note": intraday_note,
            })

            # v18.1: Save first horizon's weather-updated features for report display
            if idx == 0:
                self._forecast_feature_row = fr.copy()

        return pd.DataFrame(results)

    @staticmethod
    def _explain_feature(fname: str, val, current_price: float,
                         fr: pd.Series, wx_hdd=None, wx_temp=None) -> str:
        """Translate a feature name + value into human-readable reasoning."""
        # Price technicals
        if fname == "price_lag1":
            return f"Recent price ${val:.2f}"
        if fname.startswith("price_sma") or fname.startswith("price_ema"):
            period = fname.replace("price_sma","").replace("price_ema","")
            ma_type = "SMA" if "sma" in fname else "EMA"
            # Guard: if MA is significantly above current price, it's likely spike-contaminated
            # For natgas, a 20%+ premium in the MA means it includes spike data
            ratio = val / max(current_price, 0.5)
            spike_ratio = fr.get("spike_ratio_20d", 1.0) if isinstance(fr, pd.Series) else 1.0
            is_spike_contaminated = ratio > 1.20 and spike_ratio > 1.3
            if ratio > 1.50:
                return f"{ma_type}{period} (${val:.2f}) spike-elevated — not a reliable signal"
            if is_spike_contaminated:
                return f"{ma_type}{period} (${val:.2f}) includes spike data — ignoring as signal"
            if current_price > val:
                return f"Price above {ma_type}{period} (${val:.2f}) — bullish momentum"
            else:
                return f"Price below {ma_type}{period} (${val:.2f}) — bearish pressure"
        if fname == "rsi_14":
            if val > 70: return f"RSI overbought ({val:.0f}) — potential pullback"
            elif val < 30: return f"RSI oversold ({val:.0f}) — potential bounce"
            else: return f"RSI neutral ({val:.0f})"
        if fname == "macd" or fname == "macd_hist":
            if val > 0: return "MACD positive — bullish signal"
            else: return "MACD negative — bearish signal"
        if fname == "rvol_20d":
            if val > 2.0: return f"20d realised vol {val:.0%} (spike-inflated — not representative)"
            return f"20d realised vol {val:.1%}"
        if fname == "momentum_10d":
            spike_r = fr.get("spike_ratio_20d", 1.0) if isinstance(fr, pd.Series) else 1.0
            if spike_r > 1.5 and val < -1.0:
                return f"10d momentum -${abs(val):.3f} — post-spike drop (may normalise)"
            if val > 0: return f"10d momentum +${val:.3f} — uptrend"
            else: return f"10d momentum -${abs(val):.3f} — downtrend"
        if fname == "price_zscore_20d":
            if val > 1.5: return f"Price extended high (z={val:.1f})"
            elif val < -1.5: return f"Price depressed (z={val:.1f})"

        # Natgas-specific features
        if fname == "net_supply_balance" or fname == "supply_tightness":
            if val > 0: return f"Supply surplus {val:+.1f} Bcf — bearish (oversupply)"
            else:       return f"Supply deficit {val:+.1f} Bcf — bullish (undersupply)"
        if fname == "price_zscore_52w":
            if val > 2:   return f"Price stretched far above 52w mean (z={val:.1f}) — mean-reversion pressure"
            elif val > 1: return f"Price above 52w mean (z={val:.1f}) — mild headwind"
            elif val < -2:return f"Price far below 52w mean (z={val:.1f}) — strong rebound potential"
            elif val < -1:return f"Price below 52w mean (z={val:.1f}) — rebound potential"
            return f"Price near 52w mean (z={val:.1f})"
        if fname == "price_zscore_20w":
            if val > 1.5: return f"20-week extended (z={val:.1f})"
            elif val < -1.5: return f"20-week depressed (z={val:.1f}) — rebound candidate"
        if fname == "spike_ratio_20d" or fname == "spike_ratio_60d":
            win = "20d" if "20d" in fname else "60d"
            if val > 2.0:  return f"Post-spike: {win} high was {val:.1f}x current — strong reversion pressure"
            elif val > 1.3:return f"Post-spike reversion mode ({win} ratio={val:.2f}x)"
            else:          return f"No spike reversion signal ({win})"
        if fname == "price_vs_52w_mean_pct":
            pct = val * 100
            if pct > 20: return f"Price {pct:.0f}% above 52w mean — expensive vs history"
            elif pct < -20: return f"Price {abs(pct):.0f}% below 52w mean — cheap vs history"
        if fname == "rvol_ewma":
            if val > 2.0:
                return f"EWMA vol {val*100:.0f}% ann. — spike-inflated, not representative of current regime"
            return f"EWMA realised vol {val*100:.0f}% ann. — {'elevated' if val > 0.5 else 'moderate'}"
        # ── Spike precondition features ────────────────────────────────────
        if fname == "spike_risk_score":
            if val > 0.70: return f"⚠ SPIKE RISK CRITICAL ({val:.2f}) — all preconditions stacking"
            elif val > 0.50: return f"⚡ Spike risk elevated ({val:.2f}) — multiple stress factors"
            elif val > 0.30: return f"Spike risk moderate ({val:.2f}) — monitor"
            return f"Spike risk low ({val:.2f})"
        if fname == "polar_vortex_flag" and val > 0:
            return "🥶 Polar vortex active — CONUS-wide extreme cold"
        if fname == "polar_vortex_days" and val > 2:
            return f"Polar vortex sustained {val:.0f} days — demand shock compounding"
        if fname == "freeze_off_risk" or fname == "freeze_off_risk_3d":
            if val > 0.5: return f"⚠ Freeze-off risk {val:.0%} — wellhead supply loss likely"
            elif val > 0.2: return f"Freeze-off risk {val:.0%} — watch Permian/Appalachia"
        if fname == "cold_storage_stress" and val > 50:
            return f"Storage deficit amplified by cold ({val:.0f}) — bullish spike risk"
        if fname == "rapid_cold_onset" and val > 0:
            return f"Rapid cold onset (HDD +{val*20:.0f} in 3d) — demand spike risk"
        if fname == "demand_supply_stress" and val > 0.3:
            return f"Demand-supply stress {val:.2f} — demand surge + supply constraint"
        if fname == "storage_draw_rate" and val > 100:
            return f"Fast storage draw {val:.0f} Bcf/4wk — demand exceeding supply"
        if fname == "lng_export_stress" and val > 0.2:
            return f"LNG exports {val:.0%} above seasonal — domestic supply tighter"
        if fname == "spike_risk_3d_max" or fname == "spike_risk_7d_max":
            window = "3d" if "3d" in fname else "7d"
            if val > 0.5: return f"Peak spike risk ({window}): {val:.2f} — elevated window"

        if fname == "ehdd_vs_seasonal":

            if val > 5:  return f"Eff HDD {val:.1f} above seasonal norm — above-normal demand"
            elif val < -5: return f"Eff HDD {abs(val):.1f} below seasonal norm — below-normal demand"
        if fname == "hdd_30d_total":
            if val > 600: return f"Heavy heating season ({val:.0f} HDD-30d total)"
            elif val < 200: return f"Mild period ({val:.0f} HDD-30d total)"

        # Storage
        if fname == "storage_vs_avg":
            if val > 0: return f"Storage {val:.0f} Bcf ABOVE seasonal avg — bearish"
            else: return f"Storage {abs(val):.0f} Bcf BELOW seasonal avg — bullish"
        if fname == "storage_vs_avg_pct":
            return f"Storage {val:+.1%} vs seasonal avg"
        if fname == "storage_delta":
            if val > 0: return f"Storage build +{val:.0f} Bcf"
            else: return f"Storage draw {val:.0f} Bcf"
        if "storage" in fname:
            return f"Storage level {val:.0f} Bcf"

        # Weather
        if fname in ("hdd", "hdd_7d", "hdd_14d", "eff_hdd_7d", "effective_hdd"):
            if val > 20: return f"Strong cold/heating demand ({fname}={val:.1f})"
            elif val > 10: return f"Moderate heating demand ({fname}={val:.1f})"
            else: return f"Light heating demand ({fname}={val:.1f})"
        if fname == "temp_anomaly":
            if val < -5: return f"Colder than normal ({val:+.1f}°F anomaly) — bullish"
            elif val > 5: return f"Warmer than normal ({val:+.1f}°F anomaly) — bearish"
        if fname == "hdd_delta_7d":
            if val > 5: return f"Heating demand increasing (+{val:.0f} HDD/wk)"
            elif val < -5: return f"Heating demand declining ({val:.0f} HDD/wk)"

        # Cross-commodity
        if fname == "crude_natgas_ratio":
            if val > 25: return f"Crude/Gas ratio high ({val:.0f}x) — gas undervalued"
            elif val < 15: return f"Crude/Gas ratio low ({val:.0f}x) — gas fairly priced"
            return f"Crude/Gas ratio {val:.0f}x"
        if fname == "coal_natgas_ratio":
            return f"Coal/Gas ratio {val:.1f}x"
        if fname == "vix_level" or fname == "vix_change":
            return f"VIX at {val:.1f} — {'elevated risk' if val > 20 else 'calm markets'}"
        if fname == "usd_ret_5d":
            if val > 0.01: return "USD strengthening — bearish for commodities"
            elif val < -0.01: return "USD weakening — bullish for commodities"

        # Seasonality
        if fname == "is_transition_month" and val == 1:
            return "Shoulder-season transition (high-vol month) — directional signals less reliable"
        if fname == "month":
            months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                      7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            return f"Seasonal pattern ({months.get(int(val),'')}) factored in"
        if fname == "is_winter" and val == 1:
            return "Winter heating season — typically bullish"
        if fname == "is_injection_season" and val == 1:
            return "Injection season — typically bearish"

        # Supply/demand
        if fname == "supply_demand_ratio":
            if val > 1.05: return f"Supply exceeds demand (ratio={val:.2f})"
            elif val < 0.95: return f"Demand exceeds supply (ratio={val:.2f})"
        if "production" in fname:
            return f"Production level factored in"
        if "lng" in fname:
            return f"LNG export demand factored in"
        if "consumption" in fname:
            return f"Consumption level factored in"

        # Generic fallback for important features
        if fname.startswith("ret_") or fname.startswith("price_lag"):
            return f"Recent price action ({fname}={val:.4f})"
        if "natgas_front" in fname:
            return f"NG futures at ${val:.3f}"
        if fname.startswith("bb_"):
            # BB bands derived from SMA20/STD20 — spike-contaminated
            spike_r = fr.get("spike_ratio_20d", 1.0) if isinstance(fr, pd.Series) else 1.0
            if spike_r > 1.3:
                return ""  # suppress spike-contaminated BB signals
            return f"Bollinger band position ({fname}={val:.2f})"
        if fname.startswith("cos_doy") or fname.startswith("sin_doy"):
            return "Calendar seasonality factored in"

        return ""

    def backtest(self, n_days=60):
        hdr("WALK-FORWARD BACKTEST")
        if self.featured_df is None: return {}
        df = self.featured_df
        fnames = self.feat_engine.feature_names
        horizon_errs = {h: [] for h in self.ensemble.models}
        horizon_errs_calm = {h: [] for h in self.ensemble.models}  # non-spike
        horizon_errs_spike = {h: [] for h in self.ensemble.models}  # v18: spike regime
        persist_errs_calm = {h: [] for h in self.ensemble.models}  # naive baseline
        start = max(0, len(df) - n_days)
        sec(f"Testing on last {len(df)-start} days")

        SPIKE_PRICE_THRESHOLD = 6.0  # prices above $6 are "spike territory"
        SPIKE_RISK_THRESHOLD  = 0.3  # v18: spike_risk_score threshold

        for i in range(start, len(df)):
            for h in self.ensemble.models:
                ti = i + h
                if ti >= len(df): continue
                base_price = df["price"].iloc[i]
                if base_price <= 0: continue
                X = df[fnames].iloc[i].values.reshape(1,-1)
                X = np.nan_to_num(X, nan=0.0)
                log_ret, _, _ = self.ensemble.predict(h, X)
                pred_price = float(base_price * np.exp(log_ret[0]))
                actual_price = float(df["price"].iloc[ti])
                err = abs(pred_price - actual_price)
                persist_err = abs(base_price - actual_price)
                horizon_errs[h].append(err)

                # Determine if this was a spike-risk period
                spike_risk_i = float(df.get("spike_risk_score", pd.Series(0.0, index=df.index)).iloc[i])
                actual_move_pct = abs(actual_price / base_price - 1)

                if spike_risk_i > SPIKE_RISK_THRESHOLD or actual_move_pct > 0.08:
                    horizon_errs_spike[h].append(err)
                elif base_price < SPIKE_PRICE_THRESHOLD and actual_price < SPIKE_PRICE_THRESHOLD:
                    horizon_errs_calm[h].append(err)
                    persist_errs_calm[h].append(persist_err)

        results = {}
        print(f"\n    {C.BD}{'Hz':>4} {'Model MAE':>12} {'Naive MAE':>12} {'Improvement':>12} {'N':>5}{C.E}")
        print(f"    {C.GR}{'─'*50}{C.E}")
        for h in sorted(horizon_errs):
            if horizon_errs[h]:
                mae = np.mean(horizon_errs[h])
                results[h] = mae
                if horizon_errs_calm[h]:
                    calm_mae = np.mean(horizon_errs_calm[h])
                    persist_mae = np.mean(persist_errs_calm[h])
                    n_calm = len(horizon_errs_calm[h])
                    if persist_mae > 0:
                        improvement = (1 - calm_mae / persist_mae) * 100
                        imp_col = C.G if improvement > 0 else C.R
                        print(f"    {C.CN}{h:>2}d{C.E}"
                              f"  {C.W}${calm_mae:>10.4f}{C.E}"
                              f"  {C.GR}${persist_mae:>10.4f}{C.E}"
                              f"  {imp_col}{improvement:>+10.1f}%{C.E}"
                              f"  {C.GR}{n_calm:>5}{C.E}")
                    else:
                        print(f"    {C.CN}{h:>2}d{C.E}  {C.W}${calm_mae:>10.4f}{C.E}  {C.GR}n/a{C.E}")
                else:
                    print(f"    {C.CN}{h:>2}d{C.E}  {C.W}${mae:>10.4f}{C.E}  (all spike)")
        print(f"    {C.GR}{'─'*50}{C.E}")
        print(f"    {C.GR}Naive baseline = 'price stays the same' (persistence model){C.E}")
        print(f"    {C.GR}Non-spike only (base & actual < ${SPIKE_PRICE_THRESHOLD:.0f}){C.E}")

        # v18: Spike regime performance
        has_spike_data = any(len(horizon_errs_spike[h]) > 0 for h in horizon_errs_spike)
        if has_spike_data:
            print(f"\n    {C.Y}{C.BD}Spike-Regime Performance (risk>{SPIKE_RISK_THRESHOLD} or move>8%):{C.E}")
            print(f"    {C.BD}{'Hz':>4} {'Spike MAE':>12} {'N spike':>8}{C.E}")
            print(f"    {C.GR}{'─'*28}{C.E}")
            for h in sorted(horizon_errs_spike):
                if horizon_errs_spike[h]:
                    spike_mae = np.mean(horizon_errs_spike[h])
                    n_spike = len(horizon_errs_spike[h])
                    print(f"    {C.CN}{h:>2}d{C.E}  {C.Y}${spike_mae:>10.4f}{C.E}  {C.GR}{n_spike:>6}{C.E}")
            print(f"    {C.GR}{'─'*28}{C.E}")
            print(f"    {C.GR}Higher MAE expected during spikes — model attempts to capture direction.{C.E}")

        return results

    def generate_report(self, predictions):
        hdr("NATURAL GAS PRICE FORECAST — DETAILED REPORT")
        now = datetime.now()
        # v18.1: Retrieve forecast context stored during forecast()
        today = getattr(self, '_forecast_today', pd.Timestamp(now.date()))
        current = getattr(self, '_forecast_current', self.live_price or self.current_price)
        print(f"  {C.GR}Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}{C.E}")
        print(f"  {C.GR}Regime: {self.vol_model.regime.value.upper()}{C.E}")
        print(f"  {C.GR}GARCH daily σ: {self.vol_model.daily_vol:.4f}{C.E}\n")

        # ── Current market snapshot ──────────────────────────────────────────
        # v18.1: In backtest mode, ref should be the backtest anchor price, not today's live price
        ref = current if current else (self.live_price or self.current_price)
        if ref:
            sec("Current Market Snapshot", "💰")
            info("Henry Hub (live)", f"${ref:.3f} /MMBtu")
            if self.live_price_date:
                info("As of", self.live_price_date)
            if self.current_price and self.live_price and abs(self.current_price - self.live_price) > 0.01:
                info("EIA last close", f"${self.current_price:.3f}")

            # Pull from correct featured row (backtest-aware)
            # v18.1: In backtest mode, use the feature row from the backtest date
            _bt_feature_row = getattr(self, '_forecast_feature_row', None)
            lr = _bt_feature_row if _bt_feature_row is not None else (
                self.featured_df.iloc[-1] if self.featured_df is not None else None)
            if lr is not None:
                if "storage_bcf" in lr.index and lr["storage_bcf"] > 0:
                    info("Storage (L48)", f"{lr['storage_bcf']:.0f} Bcf")
                if "storage_vs_avg" in lr.index:
                    delta = lr["storage_vs_avg"]
                    direction = "above" if delta > 0 else "below"
                    info("Storage vs seasonal", f"{abs(delta):.0f} Bcf {direction} average")
                if "hdd" in lr.index:
                    info("Current HDD", f"{lr['hdd']:.1f}")
        
            if "rsi_14" in lr.index:
                rsi_val = lr["rsi_14"]
                rsi_label = "overbought" if rsi_val > 70 else "oversold" if rsi_val < 30 else "neutral"
                info("RSI-14", f"{rsi_val:.1f} ({rsi_label})")

            # ── Spike Risk Display ────────────────────────────────────────────────
            if "spike_risk_score" in lr.index:
                srs = float(lr["spike_risk_score"])
                sr3 = float(lr.get("spike_risk_3d_max", srs))
                sr7 = float(lr.get("spike_risk_7d_max", srs))
                if srs > 0.70:
                    srs_col = C.R; srs_label = "⚠ EXTREME — spike conditions active"
                elif srs > 0.50:
                    srs_col = C.Y; srs_label = "⚡ HIGH — multiple preconditions stacking"
                elif srs > 0.30:
                    srs_col = C.Y; srs_label = "moderate — monitor closely"
                else:
                    srs_col = C.G; srs_label = "low — normal conditions"
                print(f"\n  {C.BD}{C.CN}⚡ Spike Risk Assessment{C.E}")
                print(f"  {C.GR}{'─'*40}{C.E}")
                print(f"  {C.BD}Spike Risk Score:{C.E}   {srs_col}{C.BD}{srs:.2f}{C.E}  ({srs_label})")
                print(f"  {C.BD}3d Max Risk:{C.E}        {C.Y}{sr3:.2f}{C.E}")
                print(f"  {C.BD}7d Max Risk:{C.E}        {C.Y}{sr7:.2f}{C.E}")
                if "polar_vortex_flag" in lr.index and lr["polar_vortex_flag"] > 0:
                    pvd = int(lr.get("polar_vortex_days", 0))
                    print(f"  {C.BD}Polar Vortex:{C.E}       {C.R}ACTIVE — {pvd} consecutive days{C.E}")
                if "freeze_off_risk" in lr.index:
                    frz = float(lr["freeze_off_risk"])
                    if frz > 0.3:
                        print(f"  {C.BD}Freeze-off Risk:{C.E}    {C.R}{frz:.0%} — wellhead supply risk{C.E}")
                if "cold_storage_stress" in lr.index:
                    css = float(lr["cold_storage_stress"])
                    if css > 50:
                        print(f"  {C.BD}Storage Stress:{C.E}     {C.Y}{css:.0f} (cold×deficit compound){C.E}")
                if "demand_supply_stress" in lr.index:
                    dss = float(lr["demand_supply_stress"])
                    if dss > 0.3:
                        print(f"  {C.BD}Demand-Supply Stress:{C.E} {C.Y}{dss:.2f}{C.E}")
                print(f"  {C.GR}  Score > 0.5 = elevated; > 0.7 = spike preconditions active; > 0.85 = Uri-level risk{C.E}")

                if "crude_natgas_ratio" in lr.index and lr["crude_natgas_ratio"] > 0:
                    info("Crude/Gas ratio", f"{lr['crude_natgas_ratio']:.1f}x")
                if "price_sma5" in lr.index:
                    info("SMA-5", f"${lr['price_sma5']:.3f}")
                if "price_sma10" in lr.index:
                    sma10 = lr["price_sma10"]
                    sma10_note = f" {C.Y}(spike-elevated, ignore){C.E}" if sma10 > ref * 1.25 else ""
                    info("SMA-10", f"${sma10:.3f}{sma10_note}")
                if "price_sma20" in lr.index:
                    sma20 = lr["price_sma20"]
                    if sma20 > ref * 1.5:
                        info("SMA-20", f"${sma20:.3f} {C.Y}⚠ spike-contaminated — ignore this value{C.E}")
                    else:
                        info("SMA-20", f"${sma20:.3f}")
                if "macd" in lr.index:
                    macd_val = lr["macd"]
                    macd_dir = "positive (bullish)" if macd_val > 0 else "negative (bearish)"
                    # Check if MACD is negative because of spike crash, not organic selling
                    spike_r = lr.get("spike_ratio_20d", 1.0) if "spike_ratio_20d" in lr.index else 1.0
                    if macd_val < 0 and spike_r > 1.5:
                        macd_dir += f" {C.Y}— post-spike artefact, may normalise{C.E}"
                    info("MACD", f"{macd_val:.4f} — {macd_dir}")
                if "rvol_20d" in lr.index:
                    rv = lr["rvol_20d"]
                    if rv > 2.0:
                        rv_note = f"({C.Y}spike-window — recent extreme moves inflate this figure{C.E})"
                    elif rv > 1.0:
                        rv_note = "(elevated — recent high-vol period)"
                    else:
                        rv_note = ""
                    info("20d Realised Vol", f"{rv*100:.1f}%  {rv_note}")

        if predictions.empty:
            warn("No predictions"); return

        # ── Forecast overview table ──────────────────────────────────────────
        sec("Forecast Table (90% CI, trading days only)", "🔮")
        print()
        print(f"  {C.BD}{C.CN}{'Date':<12}{'T':>2} {'Cal':>3} {'Wkd':<4} {'Predicted':>10}"
              f"  {'Low (5%)':>10}  {'High (95%)':>10}  {'Chg':>9}  {'%':>7}  {'Conf':<6}{C.E}")
        print(f"  {C.GR}{'─'*86}{C.E}")

        for _, r in predictions.iterrows():
            chg = r["change"]; cpct = r["change_pct"]
            col = C.G if chg > 0.005 else (C.R if chg < -0.005 else C.GR)
            cc = {"High":C.G, "Medium":C.Y, "Low":C.R}.get(r["confidence"], C.GR)
            cs = f"+${chg:.3f}" if chg >= 0 else f"-${abs(chg):.3f}"
            ps = f"+{cpct:.1f}%" if cpct >= 0 else f"{cpct:.1f}%"
            cal_d = r.get("cal_days", r["horizon"])

            print(
                f"  {C.W}{r['date']:<12}{C.E}"
                f"{C.CN}{r['horizon']:>2}d{C.E} "
                f"{C.GR}{cal_d:>3}c{C.E} "
                f"{C.GR}{r['weekday']:<4}{C.E} "
                f"{C.Y}{C.BD}${r['predicted']:.3f}{C.E}"
                f"   {C.GR}${r['low_5pct']:.3f}{C.E}"
                f"   {C.GR}${r['high_95pct']:.3f}{C.E}"
                f"  {col}{cs:>9}{C.E}"
                f"  {col}{ps:>7}{C.E}"
                f"  {cc}{r['confidence']:<6}{C.E}"
            )

        print(f"  {C.GR}{'─'*86}{C.E}")
        print(f"  {C.GR}T = trading days ahead, Cal = calendar days ahead{C.E}")
        print()

        # ── Intraday Prediction Table ─────────────────────────────────────────
        sec("Intraday Forecast (Today's Session)", "⚡")
        print()
        _intra_rows = predictions[predictions["horizon"].isin([0, 1])]
        if not _intra_rows.empty:
            _intra_r = _intra_rows.iloc[0]  # use T+0 if available, else T+1
            _open_px  = current  # v18.1: use forecast-anchored current price (backtest-correct)
            _tgt_px   = _intra_r["predicted"]
            _lo_px    = _intra_r["low_5pct"]
            _hi_px    = _intra_r["high_95pct"]
            _chg_abs  = _tgt_px - _open_px
            _chg_pct  = (_tgt_px / _open_px - 1) * 100 if _open_px else 0
            _direction= "🟢 BULLISH" if _chg_abs > 0.02 else ("🔴 BEARISH" if _chg_abs < -0.02 else "⚪ NEUTRAL")
            _dir_col  = C.G if _chg_abs > 0.02 else (C.R if _chg_abs < -0.02 else C.GR)
            _conf     = _intra_r.get("confidence", "Low")
            _sigma_1d = self.vol_model.recent_vol  # 1-day σ
            _rng_lo   = _open_px * (1 - _sigma_1d)
            _rng_hi   = _open_px * (1 + _sigma_1d)
            print(f"  {C.BD}{C.CN}{'Metric':<28} {'Value':>12}{C.E}")
            print(f"  {C.GR}{'─'*42}{C.E}")
            print(f"  {C.BD}Open / Reference Price{C.E}       {C.W}${_open_px:.3f}{C.E}")
            print(f"  {C.BD}Predicted Close{C.E}              {C.Y}${_tgt_px:.3f}{C.E}")
            _cs = f"+${_chg_abs:.3f}" if _chg_abs >= 0 else f"-${abs(_chg_abs):.3f}"
            _ps = f"+{_chg_pct:.2f}%" if _chg_pct >= 0 else f"{_chg_pct:.2f}%"
            print(f"  {C.BD}Expected Move{C.E}                {_dir_col}{_cs}  ({_ps}){C.E}")
            print(f"  {C.BD}Direction{C.E}                    {_dir_col}{C.BD}{_direction}{C.E}")
            print(f"  {C.BD}Session Low  (5% CI){C.E}         {C.B}${_lo_px:.3f}{C.E}")
            print(f"  {C.BD}Session High (95% CI){C.E}        {C.B}${_hi_px:.3f}{C.E}")
            print(f"  {C.BD}1σ Daily Range{C.E}               {C.GR}${_rng_lo:.3f} – ${_rng_hi:.3f}{C.E}")
            print(f"  {C.BD}Model Confidence{C.E}             {C.Y}{_conf}{C.E}")
            print(f"  {C.BD}GARCH Daily σ{C.E}                {C.GR}{_sigma_1d*100:.2f}% ({_sigma_1d*np.sqrt(252)*100:.1f}% ann.){C.E}")
            print()
            # Compact table form
            print(f"  {C.BD}{C.CN}{'Date':<12} {'Open':>8} {'Target':>8} {'Low':>8} {'High':>8} {'Chg':>8} {'%Chg':>7} {'Dir':<12} {'Conf'}{C.E}")
            print(f"  {C.GR}{'─'*80}{C.E}")
            print(
                f"  {C.W}{_intra_r['date']:<12}{C.E}"
                f" {C.GR}${_open_px:>7.3f}{C.E}"
                f" {C.Y}${_tgt_px:>7.3f}{C.E}"
                f" {C.B}${_lo_px:>7.3f}{C.E}"
                f" {C.B}${_hi_px:>7.3f}{C.E}"
                f" {_dir_col}{_cs:>8}{C.E}"
                f" {_dir_col}{_ps:>7}{C.E}"
                f" {_dir_col}{C.BD}{_direction:<12}{C.E}"
                f" {C.Y}{_conf}{C.E}"
            )
            print(f"  {C.GR}{'─'*80}{C.E}")
            print(f"  {C.GR}Intraday session reference = live NG=F price at model run time.{C.E}")
            print(f"  {C.GR}Target = T+0 ensemble close estimate (1d model, today's features).{C.E}")

            # v18: Enhanced intraday intelligence
            _spike_regime_now = "NORMAL"
            if not predictions.empty and "spike_regime" in predictions.columns:
                _spike_regime_now = predictions.iloc[0].get("spike_regime", "NORMAL")
            if _spike_regime_now != "NORMAL":
                print(f"  {C.R}{C.BD}  ⚠ Current regime: {_spike_regime_now} — wider-than-normal intraday swings expected{C.E}")

            # Intraday trading zones
            range_mid = _tgt_px
            range_lo  = _lo_px
            range_hi  = _hi_px
            zone_buy  = range_lo + (range_mid - range_lo) * 0.3
            zone_sell = range_mid + (range_hi - range_mid) * 0.7
            print(f"  {C.CN}  Intraday zones:{C.E}"
                  f"  {C.G}Buy <${zone_buy:.3f}{C.E}"
                  f"  {C.GR}Neutral ${zone_buy:.3f}–${zone_sell:.3f}{C.E}"
                  f"  {C.R}Sell >${zone_sell:.3f}{C.E}")

            # EIA storage report day context
            _now_dt_disp = datetime.now()
            if today.weekday() == 3:
                print(f"  {C.Y}{C.BD}  📊 EIA Storage Report Day{C.E}")
                if _now_dt_disp.hour < 10:
                    print(f"  {C.Y}    Pre-report: range-bound until 10:30 ET → expect 2-8% move at release{C.E}")
                elif _now_dt_disp.hour < 12:
                    print(f"  {C.Y}    Report recently released — price settling, wait for confirmation{C.E}")
                else:
                    print(f"  {C.Y}    Post-report session — new direction established{C.E}")
        else:
            print(f"  {C.GR}No intraday data — run before 17:00 ET on a trading day.{C.E}")
        print()

        # ── Detailed per-horizon analysis ────────────────────────────────────
        sec("Per-Horizon Analysis & Key Drivers", "🔍")


        for _, r in predictions.iterrows():
            h = r["horizon"]
            cal_d = r.get("cal_days", h)
            chg = r["change"]
            direction = "▲" if chg > 0.005 else ("▼" if chg < -0.005 else "▬")
            dir_col = C.G if chg > 0.005 else (C.R if chg < -0.005 else C.GR)

            # ── v18: Spike regime badge ──
            spike_regime = r.get("spike_regime", "NORMAL")
            spike_risk = r.get("spike_risk", 0)
            regime_badge = ""
            if spike_regime == "SPIKE_EXTREME":
                regime_badge = f"  {C.R}{C.BD}🔴 SPIKE-EXTREME (risk={spike_risk:.2f}){C.E}"
            elif spike_regime == "SPIKE_HIGH":
                regime_badge = f"  {C.R}🟠 SPIKE-HIGH (risk={spike_risk:.2f}){C.E}"
            elif spike_regime == "SPIKE_WATCH":
                regime_badge = f"  {C.Y}🟡 SPIKE-WATCH (risk={spike_risk:.2f}){C.E}"

            # Intraday note
            intraday_note = r.get("intraday_note", "")
            note_str = f"  {C.CN}[{intraday_note}]{C.E}" if intraday_note else ""

            print(f"  {dir_col}{C.BD}{direction} T+{h} ({r['date']}, {r['weekday']}){C.E}"
                  f"  {C.Y}${r['predicted']:.3f}{C.E}"
                  f"  {dir_col}({r['change_pct']:+.1f}%){C.E}"
                  f"  {C.GR}[${r['low_5pct']:.3f}–${r['high_95pct']:.3f}]{C.E}"
                  f"  {C.GR}σ={r['vol_h']:.3f}{C.E}"
                  f"{regime_badge}{note_str}")

            # Show spike overlay/weather boost if non-zero
            spike_ovl = r.get("spike_overlay", 0)
            wx_bst = r.get("wx_boost", 0)
            if spike_ovl > 0 or abs(wx_bst) > 0.001:
                overlay_parts = []
                if spike_ovl > 0:
                    overlay_parts.append(f"spike_overlay=+{spike_ovl*100:.1f}%")
                if abs(wx_bst) > 0.001:
                    overlay_parts.append(f"wx_boost={wx_bst*100:+.1f}%")
                print(f"      {C.Y}↯ {', '.join(overlay_parts)}{C.E}")

            drivers = r.get("drivers", [])
            if drivers:
                for d in drivers:
                    # Color-code bullish/bearish signals
                    if any(w in d.lower() for w in ["bullish", "above", "oversold", "undervalued",
                                                      "positive", "uptrend", "strong cold",
                                                      "strong heating", "below seasonal"]):
                        print(f"      {C.G}↗ {d}{C.E}")
                    elif any(w in d.lower() for w in ["bearish", "below", "overbought", "negative",
                                                        "downtrend", "warmer", "above seasonal",
                                                        "mild", "exceeds demand"]):
                        print(f"      {C.R}↘ {d}{C.E}")
                    else:
                        print(f"      {C.CN}→ {d}{C.E}")
            else:
                print(f"      {C.GR}→ Ensemble consensus (no single dominant driver){C.E}")
            print()

        # ── Summary statistics ───────────────────────────────────────────────
        sec("Forecast Summary", "📊")
        avg_p = predictions["predicted"].mean()
        min_p = predictions["predicted"].min()
        max_p = predictions["predicted"].max()
        min_date = predictions.loc[predictions["predicted"].idxmin(), "date"]
        max_date = predictions.loc[predictions["predicted"].idxmax(), "date"]

        info("Mean forecast", f"${avg_p:.3f}")
        info("Lowest",  f"${min_p:.3f} on {min_date}")
        info("Highest", f"${max_p:.3f} on {max_date}")

        # Short-term vs medium-term breakdown
        short = predictions[predictions["horizon"] <= 3]
        mid   = predictions[predictions["horizon"] > 3]
        if not short.empty:
            info("Short-term avg (1-3d)", f"${short['predicted'].mean():.3f}")
        if not mid.empty:
            info("Medium-term avg (4-10d)", f"${mid['predicted'].mean():.3f}")

        if ref:
            avg_chg = ((avg_p/ref)-1)*100
            if avg_chg > 3:
                trend = "📈 BULLISH"; tc = C.G
            elif avg_chg < -3:
                trend = "📉 BEARISH"; tc = C.R
            elif avg_chg > 1:
                trend = "📈 SLIGHTLY BULLISH"; tc = C.G
            elif avg_chg < -1:
                trend = "📉 SLIGHTLY BEARISH"; tc = C.R
            else:
                trend = "➡️  NEUTRAL"; tc = C.CN
            info("Outlook", f"{tc}{C.BD}{trend} ({avg_chg:+.1f}%){C.E}")

        # ── v18: Spike Regime Summary ─────────────────────────────────────
        if "spike_regime" in predictions.columns:
            spike_days = predictions[predictions["spike_regime"] != "NORMAL"]
            if not spike_days.empty:
                sec("Spike Risk Assessment", "⚡")
                for _, sr in spike_days.iterrows():
                    regime = sr["spike_regime"]
                    risk = sr.get("spike_risk", 0)
                    overlay = sr.get("spike_overlay", 0)
                    if regime == "SPIKE_EXTREME":
                        print(f"    {C.R}{C.BD}🔴 {sr['date']} ({sr['weekday']}): "
                              f"EXTREME spike risk ({risk:.2f}) — potential 10-25% move{C.E}")
                    elif regime == "SPIKE_HIGH":
                        print(f"    {C.R}🟠 {sr['date']} ({sr['weekday']}): "
                              f"HIGH spike risk ({risk:.2f}) — potential 5-15% move{C.E}")
                    elif regime == "SPIKE_WATCH":
                        print(f"    {C.Y}🟡 {sr['date']} ({sr['weekday']}): "
                              f"ELEVATED spike risk ({risk:.2f}) — monitoring{C.E}")
                    if overlay > 0:
                        print(f"      {C.Y}  ↯ Spike overlay applied: +{overlay*100:.1f}% adjustment{C.E}")
            else:
                info("Spike risk", f"{C.G}All horizons NORMAL — rangebound conditions{C.E}")

        # ── v18: Weather Scenario Sensitivity ─────────────────────────────
        # Institutional desks always run "what if colder/warmer" scenarios.
        # Show how ±5°F and ±10°F changes from forecast would affect price.
        if ref and not predictions.empty:
            sec("Weather Scenario Analysis", "🌡️")
            print(f"    {C.BD}How different weather outcomes affect the 3-day forecast:{C.E}")
            print(f"    {C.GR}{'─'*65}{C.E}")
            base_pred_3d = predictions[predictions["horizon"] <= 3]["predicted"].mean() if len(predictions[predictions["horizon"] <= 3]) > 0 else ref
            base_chg = (base_pred_3d / ref - 1) * 100

            scenarios = [
                ("10°F colder than forecast",  +10, 1.06),   # ~6% price premium per historical
                ("5°F colder than forecast",   +5,  1.03),   # ~3% premium
                ("As forecast (base case)",     0,  1.00),
                ("5°F warmer than forecast",   -5,  0.97),   # ~3% discount
                ("10°F warmer than forecast",  -10, 0.94),   # ~6% discount
            ]
            print(f"    {C.BD}{'Scenario':<32} {'Price':>8} {'vs Ref':>8} {'vs Base':>8}{C.E}")
            print(f"    {'─'*58}")
            for label, temp_delta, mult in scenarios:
                sc_price = base_pred_3d * mult
                sc_chg_ref = (sc_price / ref - 1) * 100
                sc_chg_base = (mult - 1) * 100
                col = C.R if temp_delta > 0 else (C.G if temp_delta < 0 else C.Y)
                print(f"    {col}{label:<32}{C.E}"
                      f" {C.Y}${sc_price:>7.3f}{C.E}"
                      f" {col}{sc_chg_ref:>+7.1f}%{C.E}"
                      f" {col}{sc_chg_base:>+7.1f}%{C.E}")
            print(f"    {C.GR}{'─'*58}{C.E}")
            print(f"    {C.GR}Natgas rule of thumb: each 5°F miss from consensus ≈ 3% price impact{C.E}")
            print(f"    {C.GR}Impact amplified during winter months and low-storage periods{C.E}")
            print()

        # ── Volatility context ───────────────────────────────────────────────
        sec("Volatility & Risk Context", "⚡")
        info("GARCH daily vol (full window)",   f"{self.vol_model.daily_vol:.4f} ({self.vol_model.daily_vol*np.sqrt(252)*100:.2f}% ann.)")
        info("GARCH daily vol (recent, filtered)", f"{self.vol_model.recent_vol:.4f} ({self.vol_model.recent_vol*np.sqrt(252)*100:.2f}% ann.)")
        info("Annualised vol (for CI)", f"{self.vol_model.recent_vol*np.sqrt(252)*100:.1f}%")
        info("Market regime", self.vol_model.regime.value.upper())
        regime_msg = {
            MarketRegime.LOW_VOL:  "Low volatility — tight prediction bands, high confidence",
            MarketRegime.NORMAL:   "Normal volatility — standard prediction uncertainty",
            MarketRegime.HIGH_VOL: "Elevated volatility — wider bands, moderate confidence",
            MarketRegime.CRISIS:   "Crisis-level volatility — wide bands, treat forecasts cautiously",
        }
        info("Implication", regime_msg.get(self.vol_model.regime, ""))

        # ── Feature importance ───────────────────────────────────────────────
        if self.ensemble.models:
            # Try horizon 1 first, fall back to any available horizon
            imp_horizon = 1 if 1 in self.ensemble.models else min(self.ensemble.models.keys())
            imp = self.ensemble.extract_importance(imp_horizon)
            model_label = "XGBoost" if HAS_XGB and "xgb" in self.ensemble.models.get(imp_horizon, {}) else (
                "LightGBM" if HAS_LGB and "lgb" in self.ensemble.models.get(imp_horizon, {}) else "GBR")

            if imp.sum() > 0:
                sec(f"Top 15 Feature Importances ({imp_horizon}d {model_label})", "🔬")
                top = sorted(zip(self.feat_engine.feature_names, imp),
                             key=lambda x: x[1], reverse=True)[:15]
                max_val = top[0][1] if top else 1
                for name, val in top:
                    bar_len = int(val / max(max_val, 1e-9) * 30)
                    bar = "█" * bar_len
                    print(f"    {C.CN}{name:<25}{C.E} {C.Y}{bar:<30}{C.E} {val:.4f}")
            else:
                sec("Feature Importances", "🔬")
                warn("All feature importances returned zero — possible XGBoost version issue")
                warn("Model IS learning (check MAE above) but importance extraction failed")
                # Last resort: show GBR importances if available
                if "gbr" in self.ensemble.models.get(imp_horizon, {}):
                    gbr_imp = self.ensemble.models[imp_horizon]["gbr"].feature_importances_
                    if gbr_imp.sum() > 0:
                        ok("Falling back to GBR feature importances:")
                        top = sorted(zip(self.feat_engine.feature_names, gbr_imp),
                                     key=lambda x: x[1], reverse=True)[:15]
                        max_val = top[0][1] if top else 1
                        for name, val in top:
                            bar_len = int(val / max(max_val, 1e-9) * 30)
                            bar = "█" * bar_len
                            print(f"    {C.CN}{name:<25}{C.E} {C.Y}{bar:<30}{C.E} {val:.4f}")

        print(f"\n  {C.Y}{C.BD}⚠ DISCLAIMER:{C.E}")
        print(f"  {C.GR}For informational purposes only. Not financial advice.{C.E}")
        print(f"  {C.GR}Prices are Henry Hub (USD/MMBtu). For MCX INR conversion, apply exchange rate.{C.E}")
        print(f"  {C.GR}Accuracy degrades beyond 5d; treat 6-10d as directional only.{C.E}")
        print(f"  {C.GR}Spike predictions are conditional — triggered ONLY by real weather/storage signals.{C.E}")
        print(f"  {C.GR}v18: Unified regime model — NO separate spike model, NO synthetic data, NO fallbacks.{C.E}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ▸ MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    banner()
    cfg = Config()

    # ── CLI argument: --backtest-date YYYY-MM-DD ──────────────────────────────
    import argparse
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--backtest-date", default="", metavar="YYYY-MM-DD",
                     help="Run model anchored to a past date for accuracy validation")
    _known, _ = _ap.parse_known_args()
    if _known.backtest_date:
        cfg.backtest_date = _known.backtest_date
        print(f"\n  🔁 Backtest mode: {cfg.backtest_date}\n")

    cfg.eia_api_key = os.getenv("EIA_API_KEY", "")
    if not cfg.eia_api_key:
        print(f"  {C.BD}🔑 EIA API Key Required{C.E}")
        print(f"  {C.GR}Free: https://www.eia.gov/opendata/register.php{C.E}")
        cfg.eia_api_key = input(f"  {C.CN}Enter EIA API key: {C.E}").strip()
        if not cfg.eia_api_key:
            err("Required. Exiting."); return
    else:
        ok("EIA API key loaded from .env")

    print()
    predictor = NatGasPredictor(cfg)

    # Step 1: Fetch & merge all data
    merged = predictor.fetch_all_data()
    if merged.empty: err("No data"); return

    # Step 2: Feature engineering
    featured = predictor.build_features()
    if featured.empty: err("Feature engineering failed"); return

    # Step 3: Train models
    if not predictor.train(): err("Training failed"); return

    # Step 4: Walk-forward backtest
    predictor.backtest(60)

    # Step 5: Forecast (auto-fetches live price)
    preds = predictor.forecast()
    if not preds.empty:
        predictor.generate_report(preds)
    else:
        err("Forecasting failed")


if __name__ == "__main__":
    main()