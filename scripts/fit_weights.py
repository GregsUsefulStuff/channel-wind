"""
Monthly job: fit speed + direction blend weights on a trailing 2-year window.

This is the SLOW, expensive step (pulls ~2 years of history per model per
buoy), which is exactly why it's separate from the daily job. Run this
roughly monthly (GitHub Actions schedule handles that automatically), or
manually trigger it any time via the Actions tab.

Output: data/model_weights.json -- read by daily_predict.py, which is fast
because it just loads these numbers instead of re-deriving them every day.
"""

import datetime as dt
import json
import numpy as np
import pandas as pd

from common import (BUOYS, MODELS, fit_ridge, fetch_observations_vector,
                     fetch_model_history_vector)

LOOKBACK_DAYS = 730  # ~2 years


def fit_for_buoy_lead(buoy_id, lat, lon, lead_day, start, end):
    obs = fetch_observations_vector(buoy_id, start, end)
    merged = obs.copy()
    usable = []
    for model in MODELS:
        hist = fetch_model_history_vector(model, lat, lon, start, end, lead_day)
        if hist.empty or hist[f"{model}_u"].notna().sum() == 0:
            print(f"    [info] '{model}' has no {lead_day}-day history -- excluding.")
            continue
        merged = pd.merge(merged, hist, on="time", how="left")
        usable.append(model)

    needed = ["obs_speed", "obs_u", "obs_v"] + [f"{m}_spd" for m in usable] \
        + [f"{m}_u" for m in usable] + [f"{m}_v" for m in usable]
    df = merged.dropna(subset=needed)
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)
    print(f"    fitted on {len(df)} hours using {usable}")

    # speed: plain ridge on each model's own speed
    speed_w, speed_b = fit_ridge(df[[f"{m}_spd" for m in usable]].to_numpy(),
                                  df["obs_speed"].to_numpy())
    # direction: separate ridge for u and v components
    u_w, u_b = fit_ridge(df[[f"{m}_u" for m in usable]].to_numpy(), df["obs_u"].to_numpy())
    v_w, v_b = fit_ridge(df[[f"{m}_v" for m in usable]].to_numpy(), df["obs_v"].to_numpy())

    return {
        "models": usable,
        "speed_weights": dict(zip(usable, speed_w.tolist())),
        "speed_intercept": speed_b,
        "dir_weights_u": dict(zip(usable, u_w.tolist())),
        "dir_intercept_u": u_b,
        "dir_weights_v": dict(zip(usable, v_w.tolist())),
        "dir_intercept_v": v_b,
        "n_hours": len(df),
    }


def main():
    end = dt.date.today() - dt.timedelta(days=2)
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    print(f"Fitting on {start} to {end}\n")

    result = {"fitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "lookback_start": str(start), "lookback_end": str(end), "buoys": {}}

    for buoy_id, cfg in BUOYS.items():
        print(f"=== {buoy_id} ===")
        result["buoys"][buoy_id] = {}
        for lead_day in (1, 2):
            print(f"  lead {lead_day}:")
            result["buoys"][buoy_id][str(lead_day)] = fit_for_buoy_lead(
                buoy_id, cfg["lat"], cfg["lon"], lead_day, start, end)

    with open("data/model_weights.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved data/model_weights.json")


if __name__ == "__main__":
    main()
