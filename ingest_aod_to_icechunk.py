"""
Ingest MODIS MAIAC AOD (MCD19A2CMG) and VIIRS NOAA-20 L3 AOD
(AER_DBDT_D10KM_L3_VIIRS_NOAA20) into icechunk stores on GCS,
month by month for a given date range.

Bbox: 20°E–100°E, 15°S–40°N  (Extended East Africa to India)

GCS stores:
  gs://<GCS_BUCKET>/<MODIS_STORE_PATH>   e.g. gs://cdi_arco/temp_modis_aod
  gs://<GCS_BUCKET>/<VIIRS_STORE_PATH>   e.g. gs://cdi_arco/temp_viirs_aod

.env keys required:
  EARTHDATA_USERNAME, EARTHDATA_PASSWORD
  GCS_BUCKET, GCS_SERVICE_ACCOUNT_FILE
  MODIS_STORE_PATH, VIIRS_STORE_PATH

Usage:
  # Oct 2024 – Feb 2025
  python ingest_aod_to_icechunk.py --start 20241001 --end 20250228

  # Oct 2025 – Feb 2026
  python ingest_aod_to_icechunk.py --start 20251001 --end 20260228

  # Single product
  python ingest_aod_to_icechunk.py --start 20241001 --end 20250228 --product viirs

Each granule file is deleted immediately after it is read and subsetted,
so peak local disk usage is one raw file (~44 MB) at a time.
"""

import argparse
import os
import shutil
import tempfile
import numpy as np
import xarray as xr
import earthaccess
import icechunk
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()

# ── credentials & config ──────────────────────────────────────────────────────
EARTHDATA_USERNAME = os.environ["EARTHDATA_USERNAME"]
EARTHDATA_PASSWORD = os.environ["EARTHDATA_PASSWORD"]
GCS_BUCKET         = os.environ["GCS_BUCKET"]
GCS_SA_FILE        = os.environ["GCS_SERVICE_ACCOUNT_FILE"]
MODIS_STORE_PATH   = os.environ["MODIS_STORE_PATH"]
VIIRS_STORE_PATH   = os.environ["VIIRS_STORE_PATH"]

# ── spatial extent ────────────────────────────────────────────────────────────
WEST, SOUTH, EAST, NORTH = 20, -15, 100, 40
BBOX = (WEST, SOUTH, EAST, NORTH)

# ── zarr chunking ─────────────────────────────────────────────────────────────
TIME_CHUNK = 10
LAT_CHUNK  = 200
LON_CHUNK  = 200


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True,
                   help="Start date YYYYMMDD, e.g. 20241001")
    p.add_argument("--end",   required=True,
                   help="End date YYYYMMDD,   e.g. 20250228")
    p.add_argument("--product", choices=["modis", "viirs", "both"],
                   default="both", help="Which product to ingest (default: both)")
    return p.parse_args()


def parse_date(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def setup_earthdata() -> None:
    netrc = Path.home() / ".netrc"
    netrc.write_text(
        f"machine urs.earthdata.nasa.gov login {EARTHDATA_USERNAME} password {EARTHDATA_PASSWORD}"
    )
    netrc.chmod(0o600)
    earthaccess.login(strategy="environment")


def open_repo(store_path: str) -> icechunk.Repository:
    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=store_path,
        service_account_file=GCS_SA_FILE,
    )
    return icechunk.Repository.open_or_create(storage=storage)


def month_range(start: date, end: date):
    """Yield (month_start, month_end) for every calendar month overlapping [start, end]."""
    cur = start.replace(day=1)
    while cur <= end:
        nxt = cur + relativedelta(months=1)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


# ── MODIS MCD19A2CMG reader ───────────────────────────────────────────────────

def read_modis(filepath) -> xr.Dataset | None:
    """
    Open MCD19A2CMG (HDF-EOS2 / HDF4) with pyhdf.
    Grid: 3600 lat × 7200 lon at 0.05°, descending lat (90→-90).
    Returns bbox-subsetted xr.Dataset or None on error.
    """
    from pyhdf.SD import SD, SDC

    try:
        hdf = SD(os.fspath(filepath), SDC.READ)
        v   = hdf.select("AOD_055")
        raw = v.get().astype(np.float32)
        att = v.attributes()
        hdf.end()

        fill  = att.get("_FillValue", -28672)
        scale = att.get("scale_factor", 0.001)
        raw[raw == fill] = np.nan
        aod = raw * scale

        lats = np.linspace(90 - 0.025, -90 + 0.025, 3600)
        lons = np.linspace(-180 + 0.025, 180 - 0.025, 7200)

        ds = xr.Dataset(
            {"AOD_055": (["lat", "lon"], aod)},
            coords={"lat": lats, "lon": lons},
        )
        # lat is descending so slice high→low
        return ds.sel(lat=slice(NORTH, SOUTH), lon=slice(WEST, EAST))

    except Exception as exc:
        print(f"    [MODIS] read error: {exc}")
        return None


# ── VIIRS AER_DBDT_D10KM_L3_VIIRS_NOAA20 reader ──────────────────────────────

def read_viirs(filepath) -> xr.Dataset | None:
    """
    Open AER_DBDT_D10KM_L3_VIIRS_NOAA20 NetCDF.
    dims: (Time=1, Longitude=3600, Latitude=1800) at 0.1°, ascending lat.
    Returns bbox-subsetted xr.Dataset or None on error.
    """
    try:
        ds = xr.open_dataset(filepath, engine="netcdf4")
        ds = ds[["COMBINE_AOD_550_AVG"]].rename({"COMBINE_AOD_550_AVG": "AOD_055"})
        ds = ds.rename({"Latitude": "lat", "Longitude": "lon"})
        ds = ds.squeeze("Time", drop=True)
        # lat is ascending so slice low→high
        return ds.sel(lat=slice(SOUTH, NORTH), lon=slice(WEST, EAST))

    except Exception as exc:
        print(f"    [VIIRS] read error: {exc}")
        return None


# ── generic monthly ingest ────────────────────────────────────────────────────

def already_stored(repo: icechunk.Repository, obs_date: date) -> bool:
    """Return True if obs_date is already in the store."""
    try:
        session  = repo.readonly_session("main")
        existing = xr.open_zarr(session.store, consolidated=False)
        stored   = existing.time.values.astype("datetime64[D]")
        return np.datetime64(obs_date, "D") in stored
    except Exception:
        return False


def store_is_empty(repo: icechunk.Repository) -> bool:
    try:
        session = repo.readonly_session("main")
        xr.open_zarr(session.store, consolidated=False)
        return False
    except Exception:
        return True


def ingest_product(
    short_name: str,
    version: str | None,
    store_path: str,
    reader,
    label: str,
    start: date,
    end: date,
) -> None:
    print(f"\n{'='*60}")
    print(f"[{label}] {start} → {end}")
    print(f"  Store: gs://{GCS_BUCKET}/{store_path}")
    print(f"{'='*60}")

    repo = open_repo(store_path)

    for month_start, month_end in month_range(start, end):
        print(f"\n[{label}] Month: {month_start} → {month_end}")

        search_kwargs = dict(
            short_name=short_name,
            temporal=(str(month_start), str(month_end)),
            bounding_box=BBOX,
            count=40,
        )
        if version:
            search_kwargs["version"] = version

        granules = earthaccess.search_data(**search_kwargs)
        print(f"  Granules found: {len(granules)}")
        if not granules:
            continue

        daily = []

        # Download and process one file at a time — delete immediately after read
        for g in granules:
            t_str   = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            obs_dt  = date.fromisoformat(t_str[:10])

            # Skip days already in the store
            if already_stored(repo, obs_dt):
                print(f"  skip {obs_dt} (already stored)")
                continue

            tmpdir = tempfile.mkdtemp()
            try:
                files = earthaccess.download([g], local_path=tmpdir)
                if not files:
                    print(f"  {obs_dt}: download failed")
                    continue

                filepath = files[0]
                ds = reader(filepath)

                # Delete raw file immediately — only the small subset is kept
                try:
                    Path(filepath).unlink()
                except Exception:
                    pass

                if ds is None:
                    continue

                ds = ds.expand_dims(time=[np.datetime64(t_str[:10], "ns")])
                daily.append(ds)
                print(f"  {obs_dt}: read ok, subset shape {dict(ds.dims)}")

            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        if not daily:
            print(f"  No new data for {month_start:%Y-%m} — skipping commit")
            continue

        month_ds = xr.concat(daily, dim="time").sortby("time")
        # chunk only spatial axes — time chunking is set once on first write via encoding
        month_ds = month_ds.chunk({"time": -1, "lat": LAT_CHUNK, "lon": LON_CHUNK})

        w_session = repo.writable_session("main")
        store_obj = w_session.store

        if store_is_empty(repo):
            # first write: encode time chunk size explicitly
            encoding = {"AOD_055": {"chunks": [TIME_CHUNK, LAT_CHUNK, LON_CHUNK]}}
            month_ds.to_zarr(
                store_obj, mode="w", consolidated=False,
                zarr_format=3, encoding=encoding,
            )
        else:
            # sequential append — safe_chunks=False because we never write in parallel
            month_ds.to_zarr(
                store_obj, append_dim="time",
                consolidated=False, safe_chunks=False,
            )

        w_session.commit(f"{label} {month_start:%Y-%m}")
        print(f"  ✓ Committed {len(daily)} days → gs://{GCS_BUCKET}/{store_path}")

        # Free memory
        del daily, month_ds


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    start   = parse_date(args.start)
    end     = parse_date(args.end)
    product = args.product

    print(f"Run: {start} → {end}  |  product={product}")
    setup_earthdata()

    if product in ("modis", "both"):
        ingest_product(
            short_name="MCD19A2CMG",
            version="061",
            store_path=MODIS_STORE_PATH,
            reader=read_modis,
            label="MODIS MAIAC AOD",
            start=start,
            end=end,
        )

    if product in ("viirs", "both"):
        ingest_product(
            short_name="AER_DBDT_D10KM_L3_VIIRS_NOAA20",
            version=None,
            store_path=VIIRS_STORE_PATH,
            reader=read_viirs,
            label="VIIRS NOAA-20 L3 AOD",
            start=start,
            end=end,
        )


if __name__ == "__main__":
    main()
