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


def fill_with_hour_mean(df):
    """df: rows = hours, columns = models. A model with no value for an hour gets the
    average of the OTHER models for that hour -- instead of being treated as zero,
    which would drag that hour's speed down and skew its direction."""
    hour_mean = df.mean(axis=1)                       # averages only the models that have data
    return df.apply(lambda col: col.fillna(hour_mean))


def model_column(day_rows, name):
    return day_rows[name] if name in day_rows.columns else pd.Series(np.nan, index=day_rows.index)


def clean(x, digits):
    """Round for output; anything not a real number becomes null (never an invalid NaN in the JSON)."""
    x = float(x)
    return round(x, digits) if np.isfinite(x) else None


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
            spd = fill_with_hour_mean(pd.DataFrame(
                {m: model_column(day_rows, f"{m}_spd") for m in usable}, index=day_rows.index))
            if speed_method == "fitted_blend":
                pred_speed = sum(spd[m] * w["speed_weights"][m] for m in usable) + w["speed_intercept"]
            else:
                pred_speed = spd.mean(axis=1)
            pred_kts = pred_speed * MS_TO_KNOTS

            # ---- direction (always fitted vector blend) ----
            U = pd.DataFrame(index=day_rows.index)
            V = pd.DataFrame(index=day_rows.index)
            for m in usable:
                u_m, v_m = to_uv(model_column(day_rows, f"{m}_spd").to_numpy(dtype=float),
                                 model_column(day_rows, f"{m}_dir").to_numpy(dtype=float))
                U[m], V[m] = u_m, v_m
            U, V = fill_with_hour_mean(U), fill_with_hour_mean(V)
            u_pred = sum(U[m] * w["dir_weights_u"][m] for m in usable) + w["dir_intercept_u"]
            v_pred = sum(V[m] * w["dir_weights_v"][m] for m in usable) + w["dir_intercept_v"]
            hourly_bearing = bearing_from_uv(u_pred.to_numpy(), v_pred.to_numpy())
            day_bearing = float(bearing_from_uv(np.nanmean(u_pred), np.nanmean(v_pred)))

            # Real Pacific time (handles daylight saving; a fixed -7h offset would be
            # an hour off from November to March).
            local = day_rows["time"].dt.tz_convert("America/Los_Angeles")
            times_local = [t.strftime("%-I%p") for t in local]
            start_label = local.iloc[0].strftime("%a %-I %p")
            end_label = local.iloc[-1].strftime("%a %-I %p")

            output[buoy_id][str(lead_day)] = {
                "date": str(target_date),
                "start_label": start_label,   # e.g. "Sun 5 PM" -- first hour shown, Pacific time
                "end_label": end_label,       # e.g. "Mon 4 PM" -- last hour shown
                "times": times_local,
                "speed_kts": [clean(x, 1) for x in pred_kts],
                "dir_from_deg": [clean(x, 0) for x in hourly_bearing],
                "unc": DIR_MAE_DEG[buoy_id][lead_day],
            }

            log_entries.append({
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "buoy": buoy_id, "lead_day": lead_day, "target_date": str(target_date),
                "avg_kts": clean(np.nanmean(pred_kts), 1), "dir_deg": clean(day_bearing, 0),
            })
            print(f"  lead {lead_day} ({target_date}): {np.nanmean(pred_kts):.1f} kts, {day_bearing:.0f} deg")

    with open(DATA_DIR / "latest.json", "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)

    with open(DATA_DIR / "predictions_log.jsonl", "a") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    print("\nSaved data/latest.json and appended data/predictions_log.jsonl")


if __name__ == "__main__":
    main()
