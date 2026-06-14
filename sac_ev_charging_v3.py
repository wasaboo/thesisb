"""
=============================================================================
SAC EV Charging / V2G Controller — Sydney/NSW Electricity Prices
Thesis B Prototype  (v3 — realistic commute schedule + home charger)
=============================================================================

Changes vs v2
-------------
  * Replaced generic 11 kW charger with a typical Australian home EVSE:
    7.2 kW AC (Type 2, single-phase 32 A) — most common residential install.
    V2G discharge capped at the same 7.2 kW (same hardware path).
  * Added realistic commute schedule:
      Weekday (Mon–Fri):
        - Morning departure  ~07:00 → return ~09:00 (2 hr commute, ~30 km).
        - Afternoon departure ~16:00 → return ~18:00 (2 hr return, ~30 km).
        - EV is unplugged during both windows; charging/V2G forced to 0 kW.
        - SoC reduced by drive energy (0.20 kWh/km × 30 km = 6 kWh each leg).
      Weekend (Sat–Sun):
        - Random single departure between 09:00–13:00.
        - Random return between 21:00–22:00.
        - Drive distance randomly drawn from U[20, 80] km each way.
  * Day type (weekday / weekend) randomly sampled each episode.
  * State vector extended: plugged_in flag + is_weekday flag (11 dims total).
  * Charging availability mask enforced in step(): power = 0 when unplugged.
  * Departure SoC check applies only on weekdays; weekend check at episode end.
  * Changes vs v1
  ---------------
  * AEMO LEGACY CSV parser rewritten to handle multi-section dispatch files.
  * Time step changed to 15 min → 96 steps per episode.
  * Price resolution stays at 30 min; each 30-min price is applied to the
    two consecutive 15-min steps that fall within it.
  * Fixed IndexError: _get_obs() clamps step index to N_STEPS-1.
  * Added guidance on downloading a full-day price file from AEMO NEMWeb.

Download real 30-min prices
-----------------------------
  Option A — TradingIS (recommended for 30-min):
    https://nemweb.com.au/Reports/CURRENT/TradingIS_Reports/
    File prefix: PUBLIC_TRADINGIS_*
    Contains table TRADING, REGIONSUM with SETTLEMENTDATE and RRP (30-min).

  Option B — Daily aggregated dispatch (full trading day, 5-min → 30-min):
    https://nemweb.com.au/Reports/CURRENT/Daily_Reports/
    File prefix: PUBLIC_DAILY_*
    Contains DREGION table with 5-min dispatch intervals for the full AEMO
    trading day (04:05 day D through 04:00 day D+1 = 288 intervals).
    Pass with --csv; auto-detected by filename prefix.

  Option C — Aggregate many 5-min DispatchIS files for one full day:
    https://nemweb.com.au/Reports/CURRENT/DispatchIS_Reports/
    or the archive at:
    https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/

  The supplied file (PUBLIC_DISPATCH_*_LEGACY.CSV) is a single 5-min dispatch
  interval snapshot — it is parsed correctly but only gives ONE price point.
  For a full 24-hour training day, either:
    (a) pass a directory path (--csv-dir) containing all 5-min dispatch CSVs
        for the day, or
    (b) pass a single TradingIS CSV (--csv).

Usage
-----
  python sac_ev_charging.py                             # synthetic prices
  python sac_ev_charging.py --csv TRADINGIS.csv         # 30-min TradingIS
  python sac_ev_charging.py --csv-dir ./dispatch_5min/  # dir of 5-min files

Dependencies
------------
  pip install gymnasium stable-baselines3 pandas numpy matplotlib torch
=============================================================================
"""

import argparse
import glob
import os
import warnings

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_TIMESTEPS: int   = 500_000
MODEL_SAVE_PATH: str   = "sac_ev_model"
RANDOM_SEED:     int   = 42

# Battery
BAT_CAPACITY_KWH:  float = 60.0
SOC_INIT_LOW:      float = 0.20
SOC_INIT_HIGH:     float = 0.60
SOC_TARGET:        float = 0.85
SOC_MIN:           float = 0.20
SOC_MAX:           float = 0.90
ETA_CHARGE:        float = 0.95
ETA_DISCHARGE:     float = 0.95

# Time grid — 15-min steps, 30-min prices
TIME_STEP_H:  float = 0.25    # 15 min = 0.25 h
N_STEPS:      int   = 96      # 24 h / 0.25 h  (96 steps per episode)
N_PRICE_BINS: int   = 48      # 30-min price slots; each covers 2 steps

# ── Home charger / EVSE ───────────────────────────────────────────────────────
# Typical Australian residential EVSE: Type 2 AC, single-phase 32 A → 7.2 kW.
# This is the most common home charging hardware in NSW/Victoria.
# V2G discharge uses the same hardware path, so also capped at 7.2 kW.
MAX_CHARGE_KW:    float = 7.2
MAX_DISCHARGE_KW: float = 7.2

# Feeder limit
FEEDER_LIMIT_KW: float = 20.0

# ── Commute / usage schedule ──────────────────────────────────────────────────
# Steps are 15-min intervals (step 0 = 00:00, step 28 = 07:00, step 64 = 16:00)
# Weekday: morning departure 07:00 → return 09:00; afternoon 16:00 → 18:00
WD_DEPART_AM:   int   = 28   # 07:00
WD_RETURN_AM:   int   = 36   # 09:00
WD_DEPART_PM:   int   = 64   # 16:00
WD_RETURN_PM:   int   = 72   # 18:00
WD_DRIVE_KM:    float = 30.0  # one-way commute distance (km)
# Weekend: random departure (09:00-13:00), return late (21:00-22:00)
WE_DEPART_LOW:  int   = 36   # 09:00
WE_DEPART_HIGH: int   = 52   # 13:00
WE_RETURN_LOW:  int   = 84   # 21:00
WE_RETURN_HIGH: int   = 88   # 22:00
WE_DRIVE_KM_LOW:  float = 20.0
WE_DRIVE_KM_HIGH: float = 80.0
# EV energy consumption (kWh per km) — representative mid-size BEV sedan
KWH_PER_KM: float = 0.20

# Reward weights
LAMBDA_PEAK:       float = 0.05
LAMBDA_DEGRADATION:float = 1.00
LAMBDA_SOC:        float = 10.00
LAMBDA_CONSTRAINT: float = 0.50

# V2G export revenue factor.
# Retail export/feed-in value is usually lower than import price, so V2G
# revenue is scaled down to prevent unrealistic arbitrage cycling.
EXPORT_MULTIPLIER: float = 0.40

# Battery health
C_RATE_THRESHOLD:       float = 0.12
SOC_HIGH_THRESHOLD:     float = 0.80
SOC_LOW_THRESHOLD:      float = 0.25
TEMP_AMBIENT:           float = 25.0
TEMP_PENALTY_THRESHOLD: float = 35.0
TEMP_RISE_PER_KW:       float = 0.20
TEMP_COOLING_RATE:      float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 1.  AEMO PRICE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_legacy_dispatch_csv(filepath: str) -> pd.DataFrame:
    """
    Parse an AEMO multi-section CSV (TradingIS or LEGACY dispatch format).

    Handles two section types for NSW1 prices:
      - TRADING / PRICE  (TradingIS files: PUBLIC_TRADINGIS_*.CSV)
          D, TRADING, PRICE, 3, SETTLEMENTDATE, RUNNO, REGIONID, PERIODID, RRP, ...
      - DREGION           (LEGACY dispatch files: PUBLIC_DISPATCH_*_LEGACY.CSV)
          D, DREGION, <blank>, 3, SETTLEMENTDATE, RUNNO, REGIONID, INTERVENTION, RRP, ...

    Returns a DataFrame with columns: SETTLEMENTDATE, RRP
    """
    records = []
    with open(filepath, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9 or parts[0] != "D":
                continue

            section = parts[1].strip('"')
            subsect = parts[2].strip('"')

            # TradingIS: D, TRADING, PRICE, 3, SETTLEMENTDATE, RUNNO, REGIONID, PERIODID, RRP
            if section == "TRADING" and subsect == "PRICE":
                if parts[6].strip('"') != "NSW1":
                    continue
                date_str = parts[4].strip('"')
                rrp_str  = parts[8].strip('"')

            # Legacy dispatch: D, DREGION, <blank>, 3, SETTLEMENTDATE, RUNNO, REGIONID, INTERVENTION, RRP
            elif section == "DREGION":
                if parts[6].strip('"') != "NSW1":
                    continue
                date_str = parts[4].strip('"')
                rrp_str  = parts[8].strip('"')

            else:
                continue

            try:
                records.append({
                    "SETTLEMENTDATE": pd.to_datetime(date_str),
                    "RRP": float(rrp_str),
                })
            except (ValueError, IndexError):
                pass

    return pd.DataFrame(records)


def load_aemo_prices_tradingis(csv_path: str, date: str | None = None) -> np.ndarray:
    """
    Load 30-min NSW1 prices from a TradingIS report CSV.

    TradingIS files (PUBLIC_TRADINGIS_*.CSV) contain a TRADING/REGIONSUM
    table with columns including SETTLEMENTDATE and RRP at 30-min resolution.
    This parser also falls back to the LEGACY DREGION format.

    Returns np.ndarray of shape (48,) in $/kWh.
    """
    # ── Try reading as a standard flat CSV first (TradingIS format) ──────
    try:
        df = pd.read_csv(csv_path, skiprows=1, low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        if {"SETTLEMENTDATE", "REGIONID", "RRP"}.issubset(df.columns):
            df = df[df["REGIONID"].str.strip() == "NSW1"].copy()
            df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
            df = df.sort_values("SETTLEMENTDATE")
            if date is None:
                date = str(df["SETTLEMENTDATE"].dt.date.iloc[0])
            mask = df["SETTLEMENTDATE"].dt.date == pd.to_datetime(date).date()
            day_df = df[mask].set_index("SETTLEMENTDATE")["RRP"]
            day_df = day_df.resample("30min").mean()
            if len(day_df) >= 48:
                prices = day_df.values[:48].astype(float) / 1000.0
                return prices
    except Exception:
        pass

    # ── Fall back to LEGACY multi-section format ──────────────────────────
    df = _parse_legacy_dispatch_csv(csv_path)
    if df.empty:
        raise ValueError("No NSW1 DREGION rows found in file.")

    df = df.sort_values("SETTLEMENTDATE")
    if date is None:
        date = str(df["SETTLEMENTDATE"].dt.date.iloc[0])

    mask = df["SETTLEMENTDATE"].dt.date == pd.to_datetime(date).date()
    day_df = df[mask].set_index("SETTLEMENTDATE")["RRP"]

    if day_df.empty:
        raise ValueError(f"No NSW1 data for date {date}.")

    # Resample to 30-min means
    day_df = day_df.resample("30min").mean()

    n_found = day_df.notna().sum()
    if n_found < 48:
        print(f"[INFO] Only {n_found} price intervals found in this LEGACY file "
              "(single dispatch snapshot covers ~1 interval).")
        print("[INFO] A full-day price series requires either:")
        print("       (a) a TradingIS CSV  (nemweb.com.au → TradingIS_Reports)")
        print("       (b) a directory of all 5-min dispatch files for the day "
              "(--csv-dir).")
        print("[INFO] Padding missing intervals with the available price and "
              "synthetic noise for demonstration.")
        # Pad to 48 with the known price ± noise
        known_price = float(day_df.mean())
        prices_mwh = _pad_prices(day_df, known_price)
    else:
        prices_mwh = day_df.values[:48].astype(float)

    return prices_mwh / 1000.0


def load_aemo_prices_from_dir(csv_dir: str, date: str | None = None) -> np.ndarray:
    """
    Aggregate multiple 5-min AEMO LEGACY dispatch CSVs from a directory
    into a 30-min price series.

    Parameters
    ----------
    csv_dir : str  Path containing PUBLIC_DISPATCH_*.CSV files for one day.
    date    : str  'YYYY-MM-DD'.  If None, uses the date in the first file.

    Returns np.ndarray shape (48,) $/kWh.
    """
    pattern = os.path.join(csv_dir, "*.CSV")
    files = sorted(glob.glob(pattern)) + sorted(glob.glob(pattern.replace(".CSV", ".csv")))
    if not files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    frames = []
    for fp in files:
        try:
            frames.append(_parse_legacy_dispatch_csv(fp))
        except Exception:
            pass

    if not frames:
        raise ValueError("Could not parse any AEMO LEGACY files in directory.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates("SETTLEMENTDATE")
    df = df.sort_values("SETTLEMENTDATE")

    if date is None:
        date = str(df["SETTLEMENTDATE"].dt.date.iloc[0])

    mask = df["SETTLEMENTDATE"].dt.date == pd.to_datetime(date).date()
    day_df = df[mask].set_index("SETTLEMENTDATE")["RRP"].resample("30min").mean()

    idx = pd.date_range(date, periods=48, freq="30min")
    day_df = day_df.reindex(idx).ffill().bfill()

    return day_df.values[:48].astype(float) / 1000.0




def load_aemo_prices_daily(csv_path: str) -> np.ndarray:
    """
    Load NSW1 5-min dispatch prices from an AEMO PUBLIC_DAILY_*.CSV file and
    aggregate to 48 half-hourly slots.

    AEMO daily files cover one *trading day* (04:05 on day D through 04:00 on
    day D+1, i.e. 288 five-minute intervals = 576 rows for 5 regions).  This
    loader:
      1. Parses DREGION rows for NSW1 (version-2 or version-3 layout).
      2. Resamples to 30-min means.
      3. Takes the first 48 contiguous half-hour slots (04:00 D → 03:30 D+1),
         giving a complete 24-hour price series aligned to the AEMO trading day.

    Returns np.ndarray shape (48,) in $/kWh.
    """
    df = _parse_legacy_dispatch_csv(csv_path)
    if df.empty:
        raise ValueError(
            f"No NSW1 price rows found in daily file: {csv_path}"
        )

    df = df.sort_values("SETTLEMENTDATE")

    # Resample all rows to 30-min — do NOT filter by calendar date because
    # the trading day spans two calendar dates (e.g. 2026-04-11 04:05 →
    # 2026-04-12 04:00).
    day_df = df.set_index("SETTLEMENTDATE")["RRP"].resample("30min").mean().dropna()

    n = len(day_df)
    if n < 48:
        raise ValueError(
            f"Daily file only yielded {n} half-hour slots (expected ≥48). "
            "Check the file covers a full AEMO trading day."
        )

    prices_mwh = day_df.values[:48].astype(float)
    start_ts   = day_df.index[0]
    end_ts     = day_df.index[47]
    print(f"[DAILY] NSW1 prices: {start_ts} → {end_ts}  "
          f"(min ${prices_mwh.min():.2f}, max ${prices_mwh.max():.2f}, "
          f"mean ${prices_mwh.mean():.2f} $/MWh)")
    return prices_mwh / 1000.0

def _pad_prices(series: pd.Series, anchor: float) -> np.ndarray:
    """Fill a sparse price series to 48 slots using synthetic NSW profile."""
    rng = np.random.default_rng(0)
    synthetic = synthetic_nsw_prices(seed=0) * 1000.0   # back to $/MWh scale
    # Scale synthetic to anchor around the known price
    scale = anchor / (synthetic.mean() + 1e-9)
    padded = synthetic * scale
    # Overwrite with known values
    for ts, val in series.dropna().items():
        slot = int((ts.hour * 60 + ts.minute) // 30)
        if 0 <= slot < 48:
            padded[slot] = val
    return padded


def synthetic_nsw_prices(seed: int = 0) -> np.ndarray:
    """
    Fallback synthetic NSW TOU price profile — 48 half-hourly $/kWh values.

    NOTE: Used when no AEMO CSV is provided or parseable.
    Mimics a typical NSW summer weekday: low overnight, shoulder morning,
    pronounced evening peak (~17:00–20:00).
    """
    rng = np.random.default_rng(seed)
    # fmt: off
    base = np.array([
        0.04, 0.04, 0.04, 0.04,  # 00:00–01:30
        0.04, 0.04, 0.04, 0.04,  # 02:00–03:30
        0.04, 0.04, 0.05, 0.06,  # 04:00–05:30
        0.08, 0.10, 0.12, 0.13,  # 06:00–07:30
        0.13, 0.12, 0.10, 0.08,  # 08:00–09:30
        0.07, 0.06, 0.05, 0.05,  # 10:00–11:30
        0.05, 0.05, 0.05, 0.06,  # 12:00–13:30
        0.06, 0.07, 0.08, 0.10,  # 14:00–15:30
        0.14, 0.20, 0.28, 0.32,  # 16:00–17:30
        0.35, 0.38, 0.36, 0.30,  # 18:00–19:30
        0.24, 0.18, 0.13, 0.10,  # 20:00–21:30
        0.07, 0.06, 0.05, 0.04,  # 22:00–23:30
    ], dtype=float)
    # fmt: on
    noise = rng.normal(0, 0.005, size=48)
    return np.clip(base + noise, 0.01, None)


def prices_30min_to_15min(prices_30: np.ndarray) -> np.ndarray:
    """
    Expand a 48-element (30-min) price array to 96 elements (15-min)
    by repeating each value twice.
    """
    assert len(prices_30) == 48
    return np.repeat(prices_30, 2).astype(np.float32)


def get_prices(csv_path: str = "", csv_dir: str = "", seed: int = 0
               ) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load 30-min prices (48 elements) and expand to 15-min (96 elements).
    Returns (prices_15min, prices_30min, source_label).
    """
    p30 = None

    if csv_dir and os.path.isdir(csv_dir):
        try:
            p30 = load_aemo_prices_from_dir(csv_dir)
            source = f"AEMO NEMWeb 5-min files aggregated from {csv_dir}"
        except Exception as exc:
            print(f"[WARNING] Could not load from directory: {exc}")

    if p30 is None and csv_path and os.path.isfile(csv_path):
        basename = os.path.basename(csv_path).upper()
        # Auto-detect AEMO daily file by filename prefix
        if basename.startswith("PUBLIC_DAILY_"):
            try:
                p30 = load_aemo_prices_daily(csv_path)
                source = f"AEMO Daily dispatch — {os.path.basename(csv_path)}"
            except Exception as exc:
                print(f"[WARNING] Could not load daily CSV: {exc}")
                print("[WARNING] Falling back to synthetic NSW price profile.")
        else:
            try:
                p30 = load_aemo_prices_tradingis(csv_path)
                source = f"AEMO NEMWeb — {os.path.basename(csv_path)}"
            except Exception as exc:
                print(f"[WARNING] Could not load CSV: {exc}")
                print("[WARNING] Falling back to synthetic NSW price profile.")

    if p30 is None:
        print("[INFO] Using SYNTHETIC NSW time-of-use price profile "
              "(no valid AEMO data available).")
        p30 = synthetic_nsw_prices(seed=seed)
        source = "Synthetic NSW TOU (fallback)"

    p15 = prices_30min_to_15min(p30)
    return p15, p30, source


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GYMNASIUM ENVIRONMENT  (15-min steps, 96 steps/episode)
# ─────────────────────────────────────────────────────────────────────────────

class SydneyEVChargingEnv(gym.Env):
    """
    Custom Gymnasium environment for EV charging / V2G optimisation.

    Time resolution : 15-min steps, 96 steps per 24-hour episode.
    Price resolution: 30-min (each price covers 2 consecutive steps).

    Commute schedule (sampled each episode reset)
    ---------------------------------------------
    Weekday (5/7 probability):
      Morning commute : 07:00–09:00 (steps 28–35, EV unplugged)
      Afternoon commute: 16:00–18:00 (steps 64–71, EV unplugged)
      Drive energy: 0.20 kWh/km × 30 km = 6 kWh deducted per leg.
    Weekend (2/7 probability):
      Single outing: random departure 09:00–13:00, return 21:00–22:00.
      Drive distance: uniform 20–80 km each way.
    While unplugged, the agent action is forced to 0 kW.

    State vector (11 dimensions)
    ----------------------------
    [0]  sin(2pi*step/96)       cyclic time encoding
    [1]  cos(2pi*step/96)
    [2]  soc                   current SoC in [0, 1]
    [3]  price_now             current 30-min price ($/kWh), normalised
    [4]  price_next_slot       next 30-min price slot, normalised
    [5]  steps_remaining       (96 - step) / 96
    [6]  battery_temp_norm     normalised temperature
    [7]  cum_throughput_norm   cumulative |energy| / (2 × capacity)
    [8]  soc_deficit           max(0, target - soc) / target
    [9]  plugged_in            1.0 if at home charger, 0.0 if driving
    [10] is_weekday            1.0 if weekday episode, 0.0 if weekend

    Action
    ------
    Continuous scalar in [-MAX_DISCHARGE_KW, +MAX_CHARGE_KW] (7.2 kW home EVSE).
    Positive = grid→battery (charging); Negative = battery→grid (V2G).
    Action is automatically zeroed when EV is not plugged in.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices_15min: np.ndarray,          # shape (96,) $/kWh at 15-min
        prices_30min: np.ndarray,          # shape (48,) $/kWh at 30-min
        local_demand_proxy: np.ndarray | None = None,
        feeder_limit_kw: float = FEEDER_LIMIT_KW,
        seed: int = RANDOM_SEED,
    ):
        super().__init__()

        assert len(prices_15min) == N_STEPS,   f"Need {N_STEPS} 15-min prices"
        assert len(prices_30min) == N_PRICE_BINS, f"Need {N_PRICE_BINS} 30-min prices"

        self.prices_15   = prices_15min.astype(np.float32)
        self.prices_30   = prices_30min.astype(np.float32)
        self.price_max   = float(np.max(np.abs(prices_15min)) + 1e-6)
        self.feeder_limit = feeder_limit_kw
        self._rng = np.random.default_rng(seed)

        # Background demand profile (kW per 15-min step)
        if local_demand_proxy is None:
            t = np.linspace(0, 2 * np.pi, N_STEPS, endpoint=False)
            self.local_demand = (3.0 + 2.0 * np.sin(t - np.pi / 3)).astype(np.float32)
        else:
            self.local_demand = local_demand_proxy.astype(np.float32)

        # Observation space — all dims in ~[-1, 1] or [0, 1]
        # 11 dims: [sin_t, cos_t, soc, price_now, price_next, steps_rem,
        #           temp_norm, thr_norm, soc_deficit, plugged_in, is_weekday]
        obs_low  = np.array([-1., -1., 0., -1., -1., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
        obs_high = np.array([ 1.,  1., 1.,  2.,  2., 1., 1., 1., 1., 1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action space: charging power in kW
        self.action_space = spaces.Box(
            low =np.array([-MAX_DISCHARGE_KW], dtype=np.float32),
            high=np.array([ MAX_CHARGE_KW],    dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self.step_idx:            int   = 0
        self.soc:                 float = 0.4
        self.battery_temp:        float = TEMP_AMBIENT
        self.cum_throughput:      float = 0.0
        self.constraint_violations: int = 0
        self.total_degradation:   float = 0.0
        # Commute schedule (set in reset())
        self.is_weekday:          bool  = True
        self.plugged_in_mask: np.ndarray = np.ones(N_STEPS, dtype=bool)
        self.drive_soc_drops: dict       = {}   # {step: soc_drop}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _price_slot(self, step: int) -> int:
        """Map a 15-min step index to its 30-min price slot."""
        return min(step // 2, N_PRICE_BINS - 1)

    def _build_commute_schedule(self) -> None:
        """
        Build the plugged_in mask and drive_soc_drops dict for one episode.

        Weekday (is_weekday=True):
          Two absence windows:  [WD_DEPART_AM, WD_RETURN_AM)  (07:00-09:00)
                                [WD_DEPART_PM, WD_RETURN_PM)  (16:00-18:00)
          SoC drop applied at first step of each return window (energy used
          to drive WD_DRIVE_KM km at KWH_PER_KM kWh/km).

        Weekend (is_weekday=False):
          Single absence window: [depart, return) with random bounds.
          SoC drop proportional to random drive distance both ways.
        """
        mask  = np.ones(N_STEPS, dtype=bool)   # True = plugged in at home
        drops = {}                              # step -> soc_drop (positive)

        if self.is_weekday:
            # Morning absence
            for s in range(WD_DEPART_AM, WD_RETURN_AM):
                mask[s] = False
            drive_soc = (WD_DRIVE_KM * KWH_PER_KM) / BAT_CAPACITY_KWH
            drops[WD_RETURN_AM] = drive_soc          # deduct on return
            # Afternoon absence
            for s in range(WD_DEPART_PM, WD_RETURN_PM):
                mask[s] = False
            drops[WD_RETURN_PM] = drive_soc
        else:
            # Weekend outing
            depart = int(self._rng.integers(WE_DEPART_LOW, WE_DEPART_HIGH + 1))
            ret    = int(self._rng.integers(WE_RETURN_LOW,  WE_RETURN_HIGH + 1))
            dist_km = float(self._rng.uniform(WE_DRIVE_KM_LOW, WE_DRIVE_KM_HIGH))
            for s in range(depart, min(ret, N_STEPS)):
                mask[s] = False
            drive_soc = (dist_km * 2 * KWH_PER_KM) / BAT_CAPACITY_KWH  # round trip
            if ret < N_STEPS:
                drops[ret] = drive_soc

        self.plugged_in_mask = mask
        self.drive_soc_drops = drops

    def _get_obs(self) -> np.ndarray:
        # Clamp so we never go out of bounds on the final step
        i     = min(self.step_idx, N_STEPS - 1)
        slot  = self._price_slot(i)
        nslot = min(slot + 1, N_PRICE_BINS - 1)

        t_sin       = np.sin(2 * np.pi * i / N_STEPS)
        t_cos       = np.cos(2 * np.pi * i / N_STEPS)
        price_now   = self.prices_30[slot]  / self.price_max
        price_next  = self.prices_30[nslot] / self.price_max
        steps_rem   = (N_STEPS - i) / N_STEPS
        temp_norm   = np.clip((self.battery_temp - TEMP_AMBIENT) / 30.0, 0., 1.)
        thr_norm    = np.clip(self.cum_throughput / (BAT_CAPACITY_KWH * 2), 0., 1.)
        soc_deficit = max(0., SOC_TARGET - self.soc) / SOC_TARGET
        plugged     = float(self.plugged_in_mask[i])
        is_wd       = float(self.is_weekday)

        return np.array(
            [t_sin, t_cos, self.soc,
             price_now, price_next,
             steps_rem, temp_norm, thr_norm, soc_deficit,
             plugged, is_wd],
            dtype=np.float32,
        )

    def _degradation_penalty(self, power_kw: float) -> float:
        penalty = 0.0

        # 1. Smooth C-rate penalty.
        # The old threshold-based penalty rarely activated because a 7.2 kW
        # charger on a 60 kWh battery is only about 0.12 C. This version
        # penalises higher power continuously so SAC does not learn excessive
        # charge/discharge cycling just for price arbitrage.
        c_rate = abs(power_kw) / BAT_CAPACITY_KWH
        penalty += 5.0 * c_rate ** 2

        # 2. High SoC exposure
        if self.soc > SOC_HIGH_THRESHOLD:
            penalty += (self.soc - SOC_HIGH_THRESHOLD) ** 2 * 10

        # 3. Low SoC exposure
        if self.soc < SOC_LOW_THRESHOLD:
            penalty += (SOC_LOW_THRESHOLD - self.soc) ** 2 * 10

        # 4. Throughput. Increased from 0.001 to 0.02 to discourage
        # unnecessary V2G cycling and reduce battery wear proxy.
        penalty += 0.02 * abs(power_kw) * TIME_STEP_H

        # 5. Temperature exposure
        if self.battery_temp > TEMP_PENALTY_THRESHOLD:
            penalty += (self.battery_temp - TEMP_PENALTY_THRESHOLD) ** 2 * 0.01

        return penalty

    def _update_temperature(self, power_kw: float):
        heat    = TEMP_RISE_PER_KW * abs(power_kw)
        cooling = TEMP_COOLING_RATE * (self.battery_temp - TEMP_AMBIENT)
        self.battery_temp = self.battery_temp + heat - cooling

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_idx              = 0
        self.soc                   = float(self._rng.uniform(SOC_INIT_LOW, SOC_INIT_HIGH))
        self.battery_temp          = TEMP_AMBIENT
        self.cum_throughput        = 0.0
        self.constraint_violations = 0
        self.total_degradation     = 0.0
        # Sample day type: weekday ~5/7, weekend ~2/7
        self.is_weekday = bool(self._rng.random() < 5 / 7)
        self._build_commute_schedule()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        i        = min(self.step_idx, N_STEPS - 1)
        price    = self.prices_15[i]

        # ── Apply drive SoC loss if EV just returned (before charging) ────
        if self.step_idx in self.drive_soc_drops:
            drop = self.drive_soc_drops[self.step_idx]
            self.soc = float(np.clip(self.soc - drop, SOC_MIN, SOC_MAX))

        # ── Force power = 0 when unplugged (EV on the road) ──────────────
        if not self.plugged_in_mask[i]:
            power_kw = 0.0
        else:
            power_kw = float(np.clip(action[0], -MAX_DISCHARGE_KW, MAX_CHARGE_KW))

        # ── Feeder limit (only when plugged in and charging) ──────────────
        constraint_penalty = 0.0
        if power_kw > 0:
            total_load = self.local_demand[i] + power_kw
            if total_load > self.feeder_limit:
                excess = total_load - self.feeder_limit
                constraint_penalty += excess
                power_kw = max(0., self.feeder_limit - self.local_demand[i])
                self.constraint_violations += 1

        # ── Prospective SoC ───────────────────────────────────────────────
        if power_kw >= 0:
            delta_soc = (power_kw * ETA_CHARGE    * TIME_STEP_H) / BAT_CAPACITY_KWH
        else:
            delta_soc = (power_kw * TIME_STEP_H) / (ETA_DISCHARGE * BAT_CAPACITY_KWH)
        new_soc = self.soc + delta_soc

        # Penalise SoC violations before clipping
        if new_soc > SOC_MAX:
            constraint_penalty += (new_soc - SOC_MAX) * 10
            self.constraint_violations += 1
        if new_soc < SOC_MIN:
            constraint_penalty += (SOC_MIN - new_soc) * 10
            self.constraint_violations += 1
        new_soc = float(np.clip(new_soc, SOC_MIN, SOC_MAX))

        # Recompute actual power after clipping
        actual_dsoc = new_soc - self.soc
        if actual_dsoc >= 0:
            actual_power = (actual_dsoc * BAT_CAPACITY_KWH) / (ETA_CHARGE * TIME_STEP_H)
        else:
            actual_power = (actual_dsoc * BAT_CAPACITY_KWH * ETA_DISCHARGE) / TIME_STEP_H

        # ── Cost, penalties, reward ───────────────────────────────────────
        energy_kwh = actual_power * TIME_STEP_H

        # Charging imports energy at the full price. V2G export receives only
        # a fraction of the price to represent lower export/feed-in value and
        # avoid unrealistic same-price buy/sell arbitrage.
        if actual_power >= 0:
            energy_cost = price * energy_kwh
        else:
            energy_cost = price * EXPORT_MULTIPLIER * energy_kwh

        peak_demand_penalty = max(0., actual_power) * price
        degradation_penalty = self._degradation_penalty(actual_power)
        self.total_degradation += degradation_penalty

        # Update state
        self.soc             = new_soc
        self.cum_throughput += abs(energy_kwh)
        self._update_temperature(actual_power)
        self.step_idx       += 1
        terminated           = self.step_idx >= N_STEPS

        # Departure SoC penalty — only at final step
        departure_soc_penalty = 0.0
        if terminated:
            deficit = max(0., SOC_TARGET - self.soc)
            departure_soc_penalty = deficit ** 2 * 500

        reward = (
            - energy_cost
            - LAMBDA_PEAK        * peak_demand_penalty
            - LAMBDA_DEGRADATION * degradation_penalty
            - LAMBDA_SOC         * departure_soc_penalty
            - LAMBDA_CONSTRAINT  * constraint_penalty
        )

        info = {
            "energy_cost":           energy_cost,
            "peak_demand_penalty":   peak_demand_penalty,
            "degradation_penalty":   degradation_penalty,
            "departure_soc_penalty": departure_soc_penalty,
            "constraint_penalty":    constraint_penalty,
            "soc":                   self.soc,
            "actual_power_kw":       actual_power,
            "battery_temp":          self.battery_temp,
            "plugged_in":            bool(self.plugged_in_mask[i]),
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BASELINE CONTROLLERS  (all use 15-min steps / 96 steps)
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_loop(
    prices_15: np.ndarray,
    power_fn,
    init_soc: float,
    plugged_in_mask: np.ndarray | None = None,
    drive_soc_drops: dict | None = None,
) -> dict:
    """
    Generic 96-step rollout.

    power_fn(step, soc, prices_15) -> requested power kW.

    plugged_in_mask : bool array shape (96,).  If None, EV is always home.
    drive_soc_drops : {step: soc_drop} applied before that step.
    """
    if plugged_in_mask is None:
        plugged_in_mask = np.ones(N_STEPS, dtype=bool)
    if drive_soc_drops is None:
        drive_soc_drops = {}

    soc  = init_soc
    temp = TEMP_AMBIENT
    powers, socs, temps, costs = [], [], [], []
    cum_cost = 0.0
    degradation = 0.0

    for i in range(N_STEPS):
        # Apply drive energy loss on return
        if i in drive_soc_drops:
            soc = float(np.clip(soc - drive_soc_drops[i], SOC_MIN, SOC_MAX))

        # Only charge/discharge when plugged in at home charger
        if not plugged_in_mask[i]:
            power = 0.0
        else:
            power = power_fn(i, soc, prices_15)

        delta_soc = (power * ETA_CHARGE * TIME_STEP_H) / BAT_CAPACITY_KWH
        new_soc   = float(np.clip(soc + delta_soc, SOC_MIN, SOC_MAX))
        actual_pw = (new_soc - soc) * BAT_CAPACITY_KWH / (ETA_CHARGE * TIME_STEP_H)

        energy = actual_pw * TIME_STEP_H
        if actual_pw >= 0:
            cum_cost += prices_15[i] * energy
        else:
            cum_cost += prices_15[i] * EXPORT_MULTIPLIER * energy

        # Use the same battery-health proxy structure as the SAC environment
        # so baseline degradation values are comparable.
        c_rate = abs(actual_pw) / BAT_CAPACITY_KWH
        deg = 5.0 * c_rate ** 2
        if soc > SOC_HIGH_THRESHOLD:
            deg += (soc - SOC_HIGH_THRESHOLD) ** 2 * 10
        if soc < SOC_LOW_THRESHOLD:
            deg += (SOC_LOW_THRESHOLD - soc) ** 2 * 10
        deg += 0.02 * abs(actual_pw) * TIME_STEP_H
        if temp > TEMP_PENALTY_THRESHOLD:
            deg += (temp - TEMP_PENALTY_THRESHOLD) ** 2 * 0.01
        degradation += deg

        heat    = TEMP_RISE_PER_KW * abs(actual_pw)
        cooling = TEMP_COOLING_RATE * (temp - TEMP_AMBIENT)
        temp    = temp + heat - cooling

        soc = new_soc
        powers.append(actual_pw)
        socs.append(soc)
        temps.append(temp)
        costs.append(cum_cost)

    return dict(
        powers      = np.array(powers),
        socs        = np.array(socs),
        temps       = np.array(temps),
        cum_costs   = np.array(costs),
        total_cost  = cum_cost,
        final_soc   = soc,
        violations  = 0,
        degradation = degradation,
    )


def _weekday_commute_schedule(seed: int = 0):
    """Return (plugged_in_mask, drive_soc_drops) for a standard weekday commute."""
    mask  = np.ones(N_STEPS, dtype=bool)
    drops = {}
    drive_soc = (WD_DRIVE_KM * KWH_PER_KM) / BAT_CAPACITY_KWH
    for s in range(WD_DEPART_AM, WD_RETURN_AM):
        mask[s] = False
    drops[WD_RETURN_AM] = drive_soc
    for s in range(WD_DEPART_PM, WD_RETURN_PM):
        mask[s] = False
    drops[WD_RETURN_PM] = drive_soc
    return mask, drops


def run_immediate_charging(p15: np.ndarray, init_soc: float = 0.3) -> dict:
    """Baseline 1 — charge immediately at max rate until target SoC (weekday schedule)."""
    mask, drops = _weekday_commute_schedule()
    def fn(i, soc, p):
        return MAX_CHARGE_KW if soc < SOC_TARGET else 0.
    return _baseline_loop(p15, fn, init_soc, mask, drops)


def run_offpeak_charging(p15: np.ndarray, init_soc: float = 0.3) -> dict:
    """Baseline 2 — charge only below median price (weekday schedule)."""
    threshold = float(np.median(p15))
    mask, drops = _weekday_commute_schedule()
    def fn(i, soc, p):
        return MAX_CHARGE_KW if (p[i] <= threshold and soc < SOC_TARGET) else 0.
    return _baseline_loop(p15, fn, init_soc, mask, drops)


def run_greedy_no_v2g(p15: np.ndarray, init_soc: float = 0.3) -> dict:
    """Baseline 3 — greedy inverse-price-weighted charging, no V2G (weekday schedule)."""
    p_norm = (p15 - p15.min()) / (p15.max() - p15.min() + 1e-9)
    mask, drops = _weekday_commute_schedule()
    def fn(i, soc, p):
        return 0. if soc >= SOC_TARGET else (1.0 - p_norm[i]) * MAX_CHARGE_KW
    return _baseline_loop(p15, fn, init_soc, mask, drops)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_sac(env: gym.Env, timesteps: int = TRAIN_TIMESTEPS) -> SAC:
    model = SAC(
        policy          = "MlpPolicy",
        env             = Monitor(env),
        learning_rate   = 3e-4,
        batch_size      = 256,
        gamma           = 0.99,
        buffer_size     = 200_000,
        tau             = 0.005,
        ent_coef        = "auto",
        target_update_interval=1,
        gradient_steps  = 1,
        verbose         = 1,
        seed            = RANDOM_SEED,
        device          = "auto",
    )
    print(f"\n[TRAINING] SAC for {timesteps:,} time steps …")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(MODEL_SAVE_PATH)
    print(f"[TRAINING] Model saved → {MODEL_SAVE_PATH}.zip")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(model: SAC, env: SydneyEVChargingEnv) -> dict:
    """
    Run one deterministic 24-hour episode on a WEEKDAY for comparison.
    The environment reset is seeded so is_weekday=True is forced via
    the RNG producing a value < 5/7.  We override to ensure a weekday.
    """
    obs, _ = env.reset(seed=RANDOM_SEED + 99)
    # Force a weekday episode for fair comparison with baselines
    env.is_weekday = True
    env._build_commute_schedule()

    # Fair comparison with baselines: all controllers start from the same
    # state for evaluation. Training still uses random initial SoC.
    env.soc = 0.30
    env.battery_temp = TEMP_AMBIENT
    env.cum_throughput = 0.0
    env.constraint_violations = 0
    env.total_degradation = 0.0

    obs = env._get_obs()

    powers, socs, temps, costs, plugged = [], [], [], [], []
    cum_cost = 0.
    done     = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cum_cost += info["energy_cost"]
        powers.append(info["actual_power_kw"])
        socs.append(info["soc"])
        temps.append(info["battery_temp"])
        costs.append(cum_cost)
        plugged.append(float(info["plugged_in"]))
    return dict(
        powers      = np.array(powers),
        socs        = np.array(socs),
        temps       = np.array(temps),
        cum_costs   = np.array(costs),
        total_cost  = cum_cost,
        final_soc   = env.soc,
        violations  = env.constraint_violations,
        degradation = env.total_degradation,
        plugged_in  = np.array(plugged),
    )


def print_metrics(label: str, r: dict):
    print(f"\n{'─'*52}")
    print(f"  {label}")
    print(f"{'─'*52}")
    print(f"  Total electricity cost   : ${r['total_cost']:.4f}")
    print(f"  Final SoC                : {r['final_soc']:.3f}  (target {SOC_TARGET:.2f})")
    print(f"  Peak EV power (kW)       : {np.max(np.abs(r['powers'])):.2f}")
    print(f"  Total energy throughput  : {np.sum(np.abs(r['powers'])) * TIME_STEP_H:.2f} kWh")
    print(f"  Constraint violations    : {r['violations']}")
    print(f"  Battery health penalty   : {r['degradation']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(prices_15, prices_30, sac_r, b1, b2, b3, source):
    time_15 = np.linspace(0, 24, N_STEPS,      endpoint=False)
    time_30 = np.linspace(0, 24, N_PRICE_BINS, endpoint=False)

    # Commute window hour ranges for weekday shading
    commute_windows = [
        (WD_DEPART_AM * TIME_STEP_H, WD_RETURN_AM * TIME_STEP_H, "Morning commute\n07:00-09:00"),
        (WD_DEPART_PM * TIME_STEP_H, WD_RETURN_PM * TIME_STEP_H, "Afternoon commute\n16:00-18:00"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "SAC EV Charging/V2G Controller — Sydney/NSW  (Weekday, Typical Commuter)\n"
        f"Home EVSE: 7.2 kW AC Type 2  |  Price data: {source}  |  "
        "15-min steps (96/day), 30-min pricing",
        fontsize=11, fontweight="bold",
    )

    pal = {"SAC": "#1f77b4", "Immediate": "#d62728",
           "Off-Peak": "#2ca02c", "Greedy": "#ff7f0e"}

    def shade_commute(ax, y_frac_top=1.0):
        """Grey vertical bands for the two daily commute windows."""
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        for t0, t1, lbl in commute_windows:
            ax.axvspan(t0, t1, color="lightgrey", alpha=0.55, zorder=0)
            ax.text((t0 + t1) / 2, ymin + span * 0.03, lbl,
                    ha="center", va="bottom", fontsize=6, color="dimgrey",
                    style="italic")

    def add_price_bg(ax):
        ax2 = ax.twinx()
        ax2.fill_between(time_30, prices_30, alpha=0.08, color="gold")
        ax2.set_ylabel("Price ($/kWh)", fontsize=7, color="goldenrod")
        ax2.tick_params(axis="y", labelcolor="goldenrod", labelsize=7)

    # ── 1: Price ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.step(time_30, prices_30, where="post", color="goldenrod", lw=2)
    ax.fill_between(time_30, prices_30, step="post", alpha=0.3, color="gold")
    ax.set_title("NSW Electricity Price (30-min)"); ax.set_xlabel("Hour")
    ax.set_ylabel("$/kWh"); ax.set_xlim(0, 24); ax.grid(alpha=.3)
    shade_commute(ax)

    # ── 2: Power ─────────────────────────────────────────────────────────────
    ax = axes[0, 1]; add_price_bg(ax)
    for lbl, r, ls in [("SAC", sac_r, "-"), ("Immediate", b1, "--"),
                        ("Off-Peak", b2, "-."), ("Greedy", b3, ":")]:
        ax.step(time_15, r["powers"], where="post", label=lbl,
                color=pal[lbl], linestyle=ls, lw=2 if lbl=="SAC" else 1.2)
    ax.axhline(0,              color="grey", lw=.5)
    ax.axhline(MAX_CHARGE_KW,  color="steelblue", lw=.8, ls=":", alpha=.6,
               label=f"EVSE limit +{MAX_CHARGE_KW} kW")
    ax.axhline(-MAX_DISCHARGE_KW, color="steelblue", lw=.8, ls=":", alpha=.6,
               label=f"V2G limit -{MAX_DISCHARGE_KW} kW")
    ax.set_title("EV Charging/Discharging Power — 7.2 kW home EVSE (15-min)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power kW  [+ charge, − V2G]"); ax.set_xlim(0, 24)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    shade_commute(ax)

    # ── 3: SoC ───────────────────────────────────────────────────────────────
    ax = axes[1, 0]; add_price_bg(ax)
    for lbl, r, ls in [("SAC", sac_r, "-"), ("Immediate", b1, "--"),
                        ("Off-Peak", b2, "-."), ("Greedy", b3, ":")]:
        ax.plot(time_15, r["socs"], label=lbl, color=pal[lbl],
                linestyle=ls, lw=2 if lbl=="SAC" else 1.2)
    ax.axhline(SOC_TARGET, color="k",       lw=1.2, ls="--", label=f"Target {SOC_TARGET}")
    ax.axhline(SOC_MIN,    color="red",     lw=.8,  ls=":",  alpha=.7, label=f"Min {SOC_MIN}")
    ax.axhline(SOC_MAX,    color="darkred", lw=.8,  ls=":",  alpha=.7, label=f"Max {SOC_MAX}")
    ax.set_title("Battery SoC  (SoC drops during commute = drive energy used)")
    ax.set_xlabel("Hour"); ax.set_ylabel("SoC")
    ax.set_xlim(0, 24); ax.set_ylim(.1, 1.)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    shade_commute(ax)

    # ── 4: Temperature ───────────────────────────────────────────────────────
    ax = axes[1, 1]; add_price_bg(ax)
    for lbl, r, ls in [("SAC", sac_r, "-"), ("Immediate", b1, "--"),
                        ("Off-Peak", b2, "-."), ("Greedy", b3, ":")]:
        ax.plot(time_15, r["temps"], label=lbl, color=pal[lbl],
                linestyle=ls, lw=2 if lbl=="SAC" else 1.2)
    ax.axhline(TEMP_PENALTY_THRESHOLD, color="red", lw=1, ls="--",
               label=f"Penalty >{TEMP_PENALTY_THRESHOLD}°C")
    ax.set_title("Battery Temperature"); ax.set_xlabel("Hour")
    ax.set_ylabel("°C"); ax.set_xlim(0, 24); ax.legend(fontsize=8); ax.grid(alpha=.3)
    shade_commute(ax)

    # ── 5: Cumulative cost ───────────────────────────────────────────────────
    ax = axes[2, 0]
    for lbl, r, ls in [("SAC", sac_r, "-"), ("Immediate", b1, "--"),
                        ("Off-Peak", b2, "-."), ("Greedy", b3, ":")]:
        ax.plot(time_15, r["cum_costs"], label=lbl, color=pal[lbl],
                linestyle=ls, lw=2 if lbl=="SAC" else 1.2)
    ax.set_title("Cumulative Electricity Cost"); ax.set_xlabel("Hour")
    ax.set_ylabel("Cost ($)"); ax.set_xlim(0, 24); ax.legend(fontsize=8); ax.grid(alpha=.3)
    shade_commute(ax)

    # ── 6: C-rate proxy ──────────────────────────────────────────────────────
    ax = axes[2, 1]; add_price_bg(ax)
    w = 24 / N_STEPS * 0.4
    ax.bar(time_15, np.abs(sac_r["powers"]) / BAT_CAPACITY_KWH,
           width=w, alpha=.7, color=pal["SAC"], label="SAC |C-rate|")
    ax.bar(time_15 + w, np.abs(b1["powers"]) / BAT_CAPACITY_KWH,
           width=w, alpha=.4, color=pal["Immediate"], label="Immediate |C-rate|")
    ax.axhline(C_RATE_THRESHOLD, color="red", lw=1, ls="--",
               label=f"Threshold {C_RATE_THRESHOLD}")
    ax.set_title("C-Rate Degradation Proxy"); ax.set_xlabel("Hour")
    ax.set_ylabel("|C-rate| h\u207b\xb9"); ax.set_xlim(0, 24)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    shade_commute(ax)

    plt.tight_layout()
    out = "sac_ev_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[PLOT] Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAC EV Charging — Sydney/NSW v3 (commute + home EVSE)")
    parser.add_argument("--csv",      type=str, default="",
                        help="Path to AEMO CSV: TradingIS (PUBLIC_TRADINGIS_*), "
                             "daily dispatch aggregate (PUBLIC_DAILY_*), "
                             "or single LEGACY dispatch snapshot.")
    parser.add_argument("--csv-dir",  type=str, default="",
                        help="Directory of 5-min LEGACY dispatch CSVs for one day.")
    parser.add_argument("--timesteps",type=int, default=TRAIN_TIMESTEPS)
    parser.add_argument("--skip-train", action="store_true",
                        help="Load existing model instead of training.")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="Allow the synthetic NSW TOU fallback if AEMO data cannot be loaded.")
    args = parser.parse_args()

    # ── Prices ───────────────────────────────────────────────────────────
    p15, p30, source = get_prices(
        csv_path=args.csv,
        csv_dir =getattr(args, "csv_dir", ""),
        seed    =RANDOM_SEED,
    )
    if "Synthetic" in source and not args.allow_synthetic:
        raise RuntimeError(
            "AEMO data was not loaded and synthetic fallback is disabled. "
            "Check your --csv path or rerun with --allow-synthetic for testing only."
        )

    print(f"\n[PRICES] Source : {source}")
    print(f"[PRICES] 30-min — Min: ${p30.min():.4f}  "
          f"Max: ${p30.max():.4f}  Mean: ${p30.mean():.4f}  $/kWh")
    print(f"[PRICES] 15-min steps: {len(p15)}  |  30-min slots: {len(p30)}")

    # ── Environment ───────────────────────────────────────────────────────
    env = SydneyEVChargingEnv(prices_15min=p15, prices_30min=p30, seed=RANDOM_SEED)

    # ── Train / load ──────────────────────────────────────────────────────
    if args.skip_train and os.path.isfile(f"{MODEL_SAVE_PATH}.zip"):
        print(f"\n[TRAINING] Loading model from {MODEL_SAVE_PATH}.zip")
        model = SAC.load(MODEL_SAVE_PATH, env=env)
    else:
        model = train_sac(env, timesteps=args.timesteps)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n[EVALUATION] Running SAC agent for one 24-hour test day …")
    sac_r = evaluate_agent(model, env)

    INIT_SOC = 0.30
    b1 = run_immediate_charging(p15, INIT_SOC)
    b2 = run_offpeak_charging  (p15, INIT_SOC)
    b3 = run_greedy_no_v2g     (p15, INIT_SOC)

    print("\n" + "═"*52)
    print("  PERFORMANCE METRICS  (24-hour episode, 96 steps)")
    print("═"*52)
    print_metrics("SAC Agent",                   sac_r)
    print_metrics("Baseline 1 — Immediate",       b1)
    print_metrics("Baseline 2 — Off-Peak Only",   b2)
    print_metrics("Baseline 3 — Greedy No-V2G",   b3)

    plot_results(p15, p30, sac_r, b1, b2, b3, source)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  THESIS NOTES
# ─────────────────────────────────────────────────────────────────────────────

THESIS_NOTES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  THESIS NOTES — SAC EV Charging/V2G Controller  (v3)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHY SAC FOR CONTINUOUS EV CHARGING CONTROL
───────────────────────────────────────────
EV charging/discharging is a continuous-action control problem: the agent must
set charging power anywhere in [−MAX_DISCHARGE, +MAX_CHARGE] kW at each
15-minute interval.  Soft Actor-Critic (SAC) is well suited because:

  1. Maximum-entropy framework: SAC adds an entropy bonus to the reward,
     explicitly encouraging exploration.  This helps discover nuanced
     strategies (partial V2G at peak, slow overnight charging) rather than
     collapsing to bang-bang extremes.

  2. Native continuous actions: unlike DQN, no discretisation needed.
     The policy outputs a real-valued kW setpoint directly.

  3. Off-policy + replay buffer: high sample efficiency — critical for a
     simulator-based prototype with limited training time.

  4. Automatic entropy tuning (ent_coef='auto'): removes a sensitive
     hyperparameter and stabilises training.

  5. Strong empirical track record in energy systems (Cao et al. 2020;
     Deuschle et al. 2022; Wan et al. 2023).

AEMO DATA & 15-MIN / 30-MIN ARCHITECTURE
─────────────────────────────────────────
  • Electricity prices are sourced from AEMO NEMWeb at 30-min resolution
    (NSW1 regional spot price, RRP in $/MWh → $/kWh).

  • The agent acts every 15 minutes (96 steps/day) for finer control
    granularity, but the price signal is held constant for both 15-min
    steps within each 30-min interval — reflecting how retail tariffs
    are typically settled in Australia.

  • The state exposes both the current and next 30-min price slot so the
    agent can anticipate an upcoming price change.

  DOWNLOADING REAL DATA
  ---------------------
  TradingIS (easiest — native 30-min):
    https://nemweb.com.au/Reports/CURRENT/TradingIS_Reports/
    Files: PUBLIC_TRADINGIS_*.ZIP → extract CSV, pass with --csv.

  Full-day 5-min dispatch (for a specific date):
    Download all PUBLIC_DISPATCH_YYYYMMDD*.CSV for that date into one
    directory, then pass --csv-dir ./that_directory/
    URL: https://nemweb.com.au/Reports/CURRENT/DispatchIS_Reports/
    Or historical: https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/

HOW THE REWARD BALANCES COST AND BATTERY HEALTH
─────────────────────────────────────────────────
  reward = −energy_cost
           − λ_peak  · peak_demand_penalty     (grid friendliness)
           − λ_deg   · degradation_penalty     (five sub-terms)
           − λ_soc   · departure_soc_penalty   (user requirement, terminal only)
           − λ_con   · constraint_penalty      (hard-limit proxy)

  Five degradation sub-terms:
    · C-rate      → high current accelerates SEI growth (electro-chemistry).
    · High SoC    → lithium plating risk above ~80%.
    · Low SoC     → excessive DoD shortens cycle life.
    · Throughput  → total Ah correlates with calendar ageing.
    · Temperature → Arrhenius; >35 °C roughly doubles ageing rate.

  Lambda weights are tunable.  A full thesis study would sweep them to
  trace the Pareto frontier between cost and battery health.

REALISTIC COMMUTE SCHEDULE & HOME EVSE
───────────────────────────────────────────────────────────────────────────────
  The episode samples a day type each reset:
    • Weekday (71% probability, ~5/7 days):
        Morning departure 07:00 → return 09:00  (30 km, 6 kWh used)
        Afternoon departure 16:00 → return 18:00 (30 km, 6 kWh used)
        Charging only available outside these windows.
    • Weekend (29% probability, ~2/7 days):
        Random departure 09:00–13:00, return 21:00–22:00
        Random trip 20–80 km each way (0.20 kWh/km assumed)
  Home charger: 7.2 kW AC (Type 2, single-phase 32 A) — the most common
  residential EVSE installation in NSW/Victoria.  V2G export capped at the
  same 7.2 kW.  This makes the agent's charging strategy directly relevant
  to typical Australian EV owners.

LIMITATIONS
───────────────────────────────────────────────────────────────────────────────
  1. Single-day episodic training — no cross-day degradation accumulation.
  2. Perfect price lookahead (next slot in state) — real systems must forecast.
  3. Simplified 1st-order thermal model (no cooling system, no cell chemistry).
  4. No voltage or reactive-power network constraints.
  5. Static lambda weights — curriculum or adaptive reward shaping may help.
  6. 100 k steps is a prototype budget; production policies need 1–10 M steps.
  7. Single-environment training; VecEnv would scale linearly with n_envs.
  8. Commute schedule is fixed per episode; a richer model would draw from
     a stochastic mobility distribution across the full week.
  9. Drive energy (0.20 kWh/km) is fixed; real consumption varies with
     speed, HVAC load, terrain, and driving style.
"""

if __name__ == "__main__":
    print(THESIS_NOTES)
    main()
