"""
Fit speed + direction blend weights from the saved history cache.

No network access here at all -- it only reads data/history/*.csv.gz (built by
build_history.py), so it's fast and can't time out.

Safety net: if a buoy/lead has too few usable hours (e.g. because requests
failed while building history), its new fit is REJECTED and the previous
weights are kept (flagged "stale"). A bad fit never overwrites a good one.
If there are no previous weights to fall back on, nothing is written and the
script exits with an error.

Output: data/model_weights.json (read by daily_predict.py)
"""

import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

import common as C

MIN_HOURS = 6000      # ~250 days of fully usable hours; below this, don't trust a fit
WEIGHTS_PATH = C.DATA_DIR / "model_weights.json"


def load_history(buoy_id):
    path = C.HIST_DIR / f"{buoy_id}.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, compression="gzip")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def fit_lead(df, lead):
    n_obs = int(df["obs_speed"].notna().sum()) if "obs_speed" in df else 0
    usable = [m for m in C.MODELS
              if f"{m}_spd{lead}" in df.columns and f"{m}_dir{lead}" in df.columns
              and df[f"{m}_spd{lead}"].notna().sum() >= 0.3 * n_obs]
    if len(usable) < 2:
        print(f"    lead {lead}: fewer than 2 usable models -- skipping")
        return None

    cols = (["obs_speed", "obs_dir"]
            + [f"{m}_spd{lead}" for m in usable] + [f"{m}_dir{lead}" for m in usable])
    d = df.dropna(subset=cols)
    if len(d) < MIN_HOURS:
        print(f"    lead {lead}: only {len(d)} usable hours (need {MIN_HOURS}) -- REJECTED")
        return None

    obs_u, obs_v = C.to_uv(d["obs_speed"].to_numpy(), d["obs_dir"].to_numpy())
    U, V, S = [], [], []
    for m in usable:
        u, v = C.to_uv(d[f"{m}_spd{lead}"].to_numpy(), d[f"{m}_dir{lead}"].to_numpy())
        U.append(u)
        V.append(v)
        S.append(d[f"{m}_spd{lead}"].to_numpy())
    U, V, S = np.column_stack(U), np.column_stack(V), np.column_stack(S)

    sw, sb = C.fit_ridge(S, d["obs_speed"].to_numpy())
    uw, ub = C.fit_ridge(U, obs_u)
    vw, vb = C.fit_ridge(V, obs_v)
    print(f"    lead {lead}: fitted on {len(d)} hours using {usable}")
    return {
        "models": usable,
        "speed_weights": dict(zip(usable, sw.tolist())), "speed_intercept": sb,
        "dir_weights_u": dict(zip(usable, uw.tolist())), "dir_intercept_u": ub,
        "dir_weights_v": dict(zip(usable, vw.tolist())), "dir_intercept_v": vb,
        "n_hours": int(len(d)), "stale": False,
        "history_through": str(d.index.max().date()),
    }


def main():
    previous = json.loads(WEIGHTS_PATH.read_text()) if WEIGHTS_PATH.exists() else {"buoys": {}}
    result = {"fitted_at": dt.datetime.now(dt.timezone.utc).isoformat(), "buoys": {}}
    missing = []

    for buoy_id in C.BUOYS:
        print(f"=== {buoy_id} ===")
        result["buoys"][buoy_id] = {}
        df = load_history(buoy_id)
        for lead in (1, 2):
            fit = fit_lead(df, lead) if df is not None else None
            if fit is None:
                old = previous.get("buoys", {}).get(buoy_id, {}).get(str(lead))
                if old:
                    old["stale"] = True
                    fit = old
                    print(f"    lead {lead}: keeping previous weights (marked stale)")
                else:
                    missing.append(f"{buoy_id} lead {lead}")
            if fit:
                result["buoys"][buoy_id][str(lead)] = fit

    if missing:
        print(f"\nERROR: no usable weights (new or previous) for: {', '.join(missing)}")
        print("Run scripts/build_history.py (again) to fill in history, then re-run this.")
        sys.exit(1)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
