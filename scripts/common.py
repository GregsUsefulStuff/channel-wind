"""Shared config + helpers for the Channel Islands wind prediction backend."""

import time
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# Paths are relative to this file, so scripts work from any folder.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HIST_DIR = DATA_DIR / "history"

MS_TO_KNOTS = 1.94384
RIDGE_STRENGTH = 0.05
CHUNK_DAYS = 45

MODELS = [
    "gfs_seamless", "ecmwf_ifs025", "icon_seamless",
    "ukmo_seamless", "gem_seamless", "ncep_hrrr_conus",
]

# Which SPEED method won the validated walk-forward test, per buoy/lead.
# Direction always uses the fitted vector blend everywhere -- it was the
# most consistently reliable choice for direction at every buoy tested.
BUOYS = {
    "46053": {
        "lat": 34.262, "lon": -119.879,
        "label": "East Santa Barbara Channel (SB -> Santa Cruz Is. crossing)",
        "speed_method": {1: "fitted_blend", 2: "equal_average"},
    },
    "46054": {
        "lat": 34.265, "lon": -120.477,
        "label": "West Santa Barbara, near Pt. Conception (gap-wind zone)",
        "speed_method": {1: "equal_average", 2: "equal_average"},
    },
    "46069": {
        "lat": 33.674, "lon": -120.212,
        "label": "South Santa Rosa Island (outer-island side)",
        "speed_method": {1: "fitted_blend", 2: "fitted_blend"},
    },
    "46025": {
        "lat": 33.749, "lon": -119.053,
        "label": "Santa Monica Basin (eastern approach toward Anacapa)",
        "speed_method": {1: "fitted_blend", 2: "equal_average"},
    },
}

# Validated direction accuracy (deg, mean absolute angular error), from the
# two-fold walk-forward test. Only used to size the uncertainty cone on the
# site -- not refit automatically (see README for why).
DIR_MAE_DEG = {
    "46053": {1: 34.4, 2: 36.2},
    "46054": {1: 16.4, 2: 18.3},
    "46069": {1: 13.8, 2: 16.6},
    "46025": {1: 34.3, 2: 37.7},
}

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def to_uv(speed, direction_from_deg):
    rad = np.radians(direction_from_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def bearing_from_uv(u, v):
    return (np.degrees(np.arctan2(-u, -v))) % 360


def fit_ridge(X, y, ridge_strength=RIDGE_STRENGTH):
    n = X.shape[0]
    Xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - Xm, y - ym
    lam = ridge_strength * n
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ yc)
    intercept = float(ym - Xm @ w)
    return w, intercept



# ----------------------------------------------------------------------
# Network helpers: retry with backoff (free public APIs are sometimes slow,
# and shared cloud IPs get throttled)
# ----------------------------------------------------------------------

ERDDAP_BASE = "https://erddap.cencoos.org/erddap/tabledap"
GRID_ORIGIN = dt.date(2024, 1, 1)   # start of Open-Meteo's Previous Runs archive
CHUNK_DAYS = 45


class NoRetry(Exception):
    """A permanent error (bad request) -- retrying won't help."""


def _retry(fn, tries=5, base_wait=5, label=""):
    for attempt in range(tries):
        try:
            return fn()
        except NoRetry:
            raise
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait = min(90, base_wait * (2 ** attempt))
            print(f"    [retry {attempt + 1}/{tries - 1}] {label}: {type(e).__name__} -- waiting {wait}s", flush=True)
            time.sleep(wait)


def fetch_json(url, params, label=""):
    def go():
        r = requests.get(url, params=params, timeout=90)
        if 400 <= r.status_code < 500 and r.status_code != 429:
            raise NoRetry(f"HTTP {r.status_code}: {r.text[:150]}")
        r.raise_for_status()
        return r.json()
    return _retry(go, label=label)


def fetch_csv_text(url, label=""):
    def go():
        r = requests.get(url, timeout=90)
        if r.status_code == 404:      # ERDDAP answers 404 when a range simply has no data
            return ""
        if 400 <= r.status_code < 500 and r.status_code != 429:
            raise NoRetry(f"HTTP {r.status_code}: {r.text[:150]}")
        r.raise_for_status()
        return r.text
    return _retry(go, label=label)


def chunk_grid(window_start, window_end):
    """Fixed 45-day windows aligned to GRID_ORIGIN (so chunk identity is stable as the window slides)."""
    k = max(0, (window_start - GRID_ORIGIN).days // CHUNK_DAYS)
    chunks = []
    while True:
        s = GRID_ORIGIN + dt.timedelta(days=k * CHUNK_DAYS)
        if s > window_end:
            break
        chunks.append((s, s + dt.timedelta(days=CHUNK_DAYS - 1)))   # inclusive end date
        k += 1
    return chunks


def fetch_live_forecast(model, lat, lon):
    params = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,wind_direction_10m",
              "models": model, "forecast_days": 3, "wind_speed_unit": "ms", "timezone": "UTC"}
    try:
        data = fetch_json(LIVE_FORECAST_URL, params, label=f"live {model}")
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly or "wind_speed_10m" not in hourly:
            return pd.DataFrame(columns=["time", f"{model}_spd", f"{model}_dir"])
        df = pd.DataFrame(hourly).rename(columns={"wind_speed_10m": f"{model}_spd",
                                                    "wind_direction_10m": f"{model}_dir"})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    except Exception as e:
        print(f"    [warn] live forecast failed for {model}: {e}")
        return pd.DataFrame(columns=["time", f"{model}_spd", f"{model}_dir"])
