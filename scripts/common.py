"""Shared config + helpers for the Channel Islands wind prediction backend."""

import time
import datetime as dt
import numpy as np
import pandas as pd
import requests

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


def fetch_observations_vector(buoy_id, start, end):
    url = f"https://erddap.cencoos.org/erddap/tabledap/wmo_{buoy_id}.csv"
    frames = []
    cur = start
    while cur < end:
        chunk_end = min(cur + dt.timedelta(days=CHUNK_DAYS), end)
        params_str = (f"time,wind_speed,wind_from_direction&time>={cur.isoformat()}T00:00:00Z"
                       f"&time<={chunk_end.isoformat()}T00:00:00Z")
        try:
            r = requests.get(f"{url}?{params_str}", timeout=60)
            r.raise_for_status()
            frames.append(pd.read_csv(pd.io.common.StringIO(r.text), skiprows=[1]))
        except Exception as e:
            print(f"    [warn] obs chunk {cur}..{chunk_end} failed: {e}")
        cur = chunk_end
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame(columns=["time", "obs_speed", "obs_u", "obs_v"])
    obs = pd.concat(frames, ignore_index=True)
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    obs = obs.rename(columns={"wind_speed": "obs_speed", "wind_from_direction": "obs_dir"})
    obs = obs[["time", "obs_speed", "obs_dir"]].dropna()
    obs["time"] = obs["time"].dt.round("h")
    obs = obs.groupby("time", as_index=False).mean()
    obs["obs_u"], obs["obs_v"] = to_uv(obs["obs_speed"].to_numpy(), obs["obs_dir"].to_numpy())
    return obs


def fetch_model_history_vector(model, lat, lon, start, end, lead_day):
    spd_var = f"wind_speed_10m_previous_day{lead_day}"
    dir_var = f"wind_direction_10m_previous_day{lead_day}"
    frames = []
    cur = start
    while cur < end:
        chunk_end = min(cur + dt.timedelta(days=CHUNK_DAYS), end)
        params = {"latitude": lat, "longitude": lon, "start_date": cur.isoformat(),
                   "end_date": chunk_end.isoformat(), "hourly": f"{spd_var},{dir_var}",
                   "models": model, "wind_speed_unit": "ms", "timezone": "UTC"}
        try:
            r = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=60)
            r.raise_for_status()
            hourly = r.json().get("hourly", {})
            if hourly and "time" in hourly and spd_var in hourly and dir_var in hourly:
                frames.append(pd.DataFrame(hourly))
        except Exception as e:
            print(f"    [warn] history chunk {cur}..{chunk_end} failed for {model}: {e}")
        cur = chunk_end
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame(columns=["time", f"{model}_spd", f"{model}_u", f"{model}_v"])
    fc = pd.concat(frames, ignore_index=True)
    fc["time"] = pd.to_datetime(fc["time"], utc=True)
    fc = fc.drop_duplicates(subset="time")
    fc[spd_var] = pd.to_numeric(fc[spd_var], errors="coerce")
    fc[dir_var] = pd.to_numeric(fc[dir_var], errors="coerce")
    u, v = to_uv(fc[spd_var].to_numpy(), fc[dir_var].to_numpy())
    return pd.DataFrame({"time": fc["time"], f"{model}_spd": fc[spd_var], f"{model}_u": u, f"{model}_v": v})


def fetch_live_forecast(model, lat, lon):
    params = {"latitude": lat, "longitude": lon, "hourly": "wind_speed_10m,wind_direction_10m",
              "models": model, "forecast_days": 3, "wind_speed_unit": "ms", "timezone": "UTC"}
    try:
        r = requests.get(LIVE_FORECAST_URL, params=params, timeout=60)
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
        if not hourly or "time" not in hourly or "wind_speed_10m" not in hourly:
            return pd.DataFrame(columns=["time", f"{model}_spd", f"{model}_dir"])
        df = pd.DataFrame(hourly).rename(columns={"wind_speed_10m": f"{model}_spd",
                                                    "wind_direction_10m": f"{model}_dir"})
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df
    except Exception as e:
        print(f"    [warn] live forecast failed for {model}: {e}")
        return pd.DataFrame(columns=["time", f"{model}_spd", f"{model}_dir"])
