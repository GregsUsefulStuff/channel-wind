"""
Build / top up the local history cache: data/history/<buoy>.csv.gz

For each buoy this stores ~2 years of hourly data in one wide table:
  obs_speed, obs_dir                          (what the buoy measured)
  <model>_spd<lead>, <model>_dir<lead>        (what each model had forecast 1 / 2 days ahead)

It is RESUMABLE and INCREMENTAL. A small sidecar file (<buoy>_coverage.json)
records which 45-day chunks have been fetched successfully, so:
  - the first run does the big backfill (do this on your own computer --
    shared cloud servers get throttled by the free weather APIs),
  - a later run only fetches the chunks that failed before plus the newest
    partial chunk (a minute or two),
  - if it's interrupted or some requests fail, just run it again.

Usage:
    python3 scripts/build_history.py            # all buoys
    python3 scripts/build_history.py 46053      # just one
"""

import datetime as dt
import io
import json
import sys
import time

import pandas as pd

import common as C

LOOKBACK_DAYS = 730
PAUSE = 0.3   # seconds between requests (be polite to free APIs)


def load_state(buoy_id):
    csv_path = C.HIST_DIR / f"{buoy_id}.csv.gz"
    cov_path = C.HIST_DIR / f"{buoy_id}_coverage.json"
    if csv_path.exists():
        df = pd.read_csv(csv_path, compression="gzip")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    else:
        df = pd.DataFrame()
    cov = json.loads(cov_path.read_text()) if cov_path.exists() else {}
    return df, cov


def save_state(buoy_id, df, cov):
    C.HIST_DIR.mkdir(parents=True, exist_ok=True)
    out = df.sort_index().copy()
    out.index.name = "time"
    out.round(3).to_csv(C.HIST_DIR / f"{buoy_id}.csv.gz", compression="gzip")
    (C.HIST_DIR / f"{buoy_id}_coverage.json").write_text(json.dumps(cov, indent=1, sort_keys=True))


def fetch_obs_chunk(buoy_id, s, fetch_end):
    url = (f"{C.ERDDAP_BASE}/wmo_{buoy_id}.csv?time,wind_speed,wind_from_direction"
           f"&time>={s.isoformat()}T00:00:00Z&time<={fetch_end.isoformat()}T23:59:59Z")
    text = C.fetch_csv_text(url, label=f"obs {buoy_id} {s}")
    if not text.strip():
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO(text), skiprows=[1])   # second row is units
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw = raw.rename(columns={"wind_speed": "obs_speed", "wind_from_direction": "obs_dir"})
    raw = raw[["time", "obs_speed", "obs_dir"]].dropna()
    if raw.empty:
        return pd.DataFrame()
    raw["time"] = raw["time"].dt.round("h")
    raw["u"], raw["v"] = C.to_uv(raw["obs_speed"].to_numpy(), raw["obs_dir"].to_numpy())
    g = raw.groupby("time").agg(obs_speed=("obs_speed", "mean"), u=("u", "mean"), v=("v", "mean"))
    g["obs_dir"] = C.bearing_from_uv(g["u"].to_numpy(), g["v"].to_numpy())   # circular-safe average
    return g[["obs_speed", "obs_dir"]]


def fetch_model_chunk(model, lat, lon, lead, s, fetch_end):
    spd_var = f"wind_speed_10m_previous_day{lead}"
    dir_var = f"wind_direction_10m_previous_day{lead}"
    params = {"latitude": lat, "longitude": lon,
              "start_date": s.isoformat(), "end_date": fetch_end.isoformat(),
              "hourly": f"{spd_var},{dir_var}", "models": model,
              "wind_speed_unit": "ms", "timezone": "UTC"}
    data = C.fetch_json(C.PREVIOUS_RUNS_URL, params, label=f"{model} lead{lead} {s}")
    hourly = data.get("hourly") or {}
    if "time" not in hourly:
        raise RuntimeError(f"no hourly block in response: {str(data)[:150]}")
    if spd_var not in hourly or dir_var not in hourly:
        return pd.DataFrame()   # this model doesn't provide this lead (e.g. HRRR at 2 days)
    out = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"], utc=True),
        f"{model}_spd{lead}": pd.to_numeric(pd.Series(hourly[spd_var]), errors="coerce"),
        f"{model}_dir{lead}": pd.to_numeric(pd.Series(hourly[dir_var]), errors="coerce"),
    })
    return out.drop_duplicates("time").set_index("time")


def update_buoy(buoy_id, cfg, window_start, window_end):
    df, cov = load_state(buoy_id)
    sources = ["obs"] + [f"{m}|{lead}" for m in C.MODELS for lead in (1, 2)]
    failed = 0
    fetched = 0

    for (s, e) in C.chunk_grid(window_start, window_end):
        fetch_end = min(e, window_end)
        fully_past = e <= window_end     # a chunk that's still filling gets refetched every run
        for src in sources:
            if fully_past and s.isoformat() in cov.get(src, []):
                continue
            try:
                if src == "obs":
                    piece = fetch_obs_chunk(buoy_id, s, fetch_end)
                else:
                    model, lead = src.split("|")
                    piece = fetch_model_chunk(model, cfg["lat"], cfg["lon"], int(lead), s, fetch_end)
            except Exception as ex:
                failed += 1
                print(f"  [warn] {buoy_id} {src} chunk {s}: {type(ex).__name__}: {str(ex)[:100]}", flush=True)
                time.sleep(PAUSE)
                continue
            fetched += 1
            if not piece.empty:
                df = df.combine_first(piece) if not df.empty else piece
            if fully_past:
                cov.setdefault(src, [])
                if s.isoformat() not in cov[src]:
                    cov[src].append(s.isoformat())
            time.sleep(PAUSE)
        # save after every chunk so an interruption loses almost nothing
        if not df.empty:
            keep = df[df.index >= pd.Timestamp(window_start, tz="UTC")]
            save_state(buoy_id, keep, cov)
            df = keep
        print(f"  {buoy_id}: chunk {s} done ({fetched} fetched, {failed} failed so far)", flush=True)

    return failed


def main():
    end = dt.date.today() - dt.timedelta(days=2)
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    wanted = sys.argv[1:] or list(C.BUOYS)
    print(f"Window: {start} .. {end}", flush=True)

    total_failed = 0
    for buoy_id in wanted:
        print(f"=== {buoy_id} ===", flush=True)
        total_failed += update_buoy(buoy_id, C.BUOYS[buoy_id], start, end)

    if total_failed:
        print(f"\n{total_failed} request(s) failed after retries. Nothing is lost -- run this "
              f"script again and it will fill in just the missing pieces.")
    else:
        print("\nHistory is complete.")


if __name__ == "__main__":
    main()
