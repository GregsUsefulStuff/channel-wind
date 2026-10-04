"""
Daily job: load the weights fit_weights.py already computed, apply them to
TODAY's live forecast, publish data/latest.json for the website, and append
a compact entry to data/predictions_log.jsonl so there's a growing record
to check predictions against actual outcomes later (a natural next step,
not built yet -- see README).

This is deliberately fast: no multi-year history fetch here, just today's
live numbers from each model.
"""

import datetime as dt
import json
import numpy as np
import pandas as pd

from common import (BUOYS, MODELS, DIR_MAE_DEG, MS_TO_KNOTS, DATA_DIR,
                     to_uv, bearing_from_uv, fetch_live_forecast)

WEIGHTS_PATH = DATA_DIR / "model_weights.json"


def main():
    with open(WEIGHTS_PATH) as f:
        weights = json.load(f)

    today = dt.datetime.now(dt.timezone.utc).date()
    lead_dates = {1: today + dt.timedelta(days=1), 2: today + dt.timedelta(days=2)}

    output = {}
    log_entries = []

    for buoy_id, cfg in BUOYS.items():
        print(f"=== {buoy_id} ===")
        lat, lon = cfg["lat"], cfg["lon"]

        live = None
        for model in MODELS:
            fc = fetch_live_forecast(model, lat, lon)
            if fc.empty:
                continue
            live = fc if live is None else pd.merge(live, fc, on="time", how="outer")
        if live is None:
            print("  [warn] no live data -- skipping buoy.")
            continue
        live["time"] = pd.to_datetime(live["time"], utc=True)

        output[buoy_id] = {"label": cfg["label"]}

        for lead_day, target_date in lead_dates.items():
            day_rows = live[live["time"].dt.date == target_date].sort_values("time")
            if day_rows.empty:
                continue

            w = weights["buoys"][buoy_id][str(lead_day)]
            usable = w["models"]

            # ---- speed ----
            speed_method = cfg["speed_method"][lead_day]
            present = [m for m in usable if f"{m}_spd" in day_rows.columns and day_rows[f"{m}_spd"].notna().any()]
            if speed_method == "fitted_blend":
                pred_speed = sum(day_rows[f"{m}_spd"].fillna(0) * w["speed_weights"][m] for m in present) \
                    + w["speed_intercept"]
            else:
                pred_speed = day_rows[[f"{m}_spd" for m in present]].mean(axis=1)
            pred_kts = pred_speed * MS_TO_KNOTS

            # ---- direction (always fitted vector blend) ----
            u_pred = np.zeros(len(day_rows))
            v_pred = np.zeros(len(day_rows))
            for m in present:
                if f"{m}_dir" not in day_rows.columns or not day_rows[f"{m}_dir"].notna().any():
                    continue
                u_m, v_m = to_uv(day_rows[f"{m}_spd"].to_numpy(), day_rows[f"{m}_dir"].to_numpy())
                u_pred = u_pred + np.nan_to_num(u_m) * w["dir_weights_u"][m]
                v_pred = v_pred + np.nan_to_num(v_m) * w["dir_weights_v"][m]
            u_pred = u_pred + w["dir_intercept_u"]
            v_pred = v_pred + w["dir_intercept_v"]
            hourly_bearing = bearing_from_uv(u_pred, v_pred)
            day_bearing = float(bearing_from_uv(np.nanmean(u_pred), np.nanmean(v_pred)))

            times_local = [(pd.Timestamp(t) - pd.Timedelta(hours=7)).strftime("%-I%p").lstrip("0")
                           for t in day_rows["time"]]

            output[buoy_id][str(lead_day)] = {
                "date": str(target_date),
                "times": times_local,
                "speed_kts": [round(float(x), 1) for x in pred_kts],
                "dir_from_deg": [round(float(x), 0) for x in hourly_bearing],
                "unc": DIR_MAE_DEG[buoy_id][lead_day],
            }

            log_entries.append({
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "buoy": buoy_id, "lead_day": lead_day, "target_date": str(target_date),
                "avg_kts": round(float(pred_kts.mean()), 1), "dir_deg": round(day_bearing, 0),
            })
            print(f"  lead {lead_day} ({target_date}): {pred_kts.mean():.1f} kts, {day_bearing:.0f} deg")

    with open(DATA_DIR / "latest.json", "w") as f:
        json.dump(output, f, indent=2)

    with open(DATA_DIR / "predictions_log.jsonl", "a") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    print("\nSaved data/latest.json and appended data/predictions_log.jsonl")


if __name__ == "__main__":
    main()
