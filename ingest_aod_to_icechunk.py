"""
Ingest MODIS MAIAC AOD (MCD19A2CMG) and VIIRS NOAA-20 L3 AOD
(AER_DBDT_D10KM_L3_VIIRS_NOAA20) into icechunk stores on GCS,
month by month from 2019-01-01 to present.

Bbox: 20°E–100°E, 15°S–40°N  (Extended East Africa to India)

GCS stores:
  gs://<GCS_BUCKET>/<MODIS_STORE_PATH>   e.g. gs://cdi_arco/temp_modis_aod
  gs://<GCS_BUCKET>/<VIIRS_STORE_PATH>   e.g. gs://cdi_arco/temp_viirs_aod

.env keys required:
  EARTHDATA_USERNAME, EARTHDATA_PASSWORD
  GCS_BUCKET, GCS_SERVICE_ACCOUNT_FILE
  MODIS_STORE_PATH, VIIRS_STORE_PATH
"""

import os
import tempfile
import numpy as np
import xarray as xr
import earthaccess
import icechunk
import zarr
from pathlib import Path
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()

# ── credentials & config ──────────────────────────────────────────────────────
EARTHDATA_USERNAME    = os.environ["EARTHDATA_USERNAME"]
EARTHDATA_PASSWORD    = os.environ["EARTHDATA_PASSWORD"]
GCS_BUCKET            = os.environ["GCS_BUCKET"]
GCS_SA_FILE           = os.environ["GCS_SERVICE_ACCOUNT_FILE"]
MODIS_STORE_PATH      = os.environ["MODIS_STORE_PATH"]
VIIRS_STORE_PATH      = os.environ["VIIRS_STORE_PATH"]

# ── spatial extent ────────────────────────────────────────────────────────────
# Extended East Africa → India: 20°E–100°E, 15°S–40°N
WEST, SOUTH, EAST, NORTH = 20, -15, 100, 40
BBOX = (WEST, SOUTH, EAST, NORTH)

# ── time range ────────────────────────────────────────────────────────────────
START = date(2019, 1, 1)
END   = date.today()

# ── zarr chunking ─────────────────────────────────────────────────────────────
TIME_CHUNK = 10   # days per chunk along time
LAT_CHUNK  = 200
LON_CHUNK  = 200


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """Yield (month_start, month_end) for every month between start and end."""
    cur = start.replace(day=1)
    while cur <= end:
        nxt = cur + relativedelta(months=1)
        yield cur, min(nxt - timedelta(days=1), end)
        cur = nxt


# ── MODIS MCD19A2CMG reader ───────────────────────────────────────────────────

def read_modis(filepath) -> xr.Dataset | None:
    """
    Open MCD19A2CMG (HDF-EOS2 / HDF4) with pyhdf.
    Grid: 3600 lat × 7200 lon at 0.05°, upper-left origin 90°N / 180°W.
    Returns xr.Dataset subsetted to BBOX or None on error.
    """
    from pyhdf.SD import SD, SDC

    try:
        hdf = SD(os.fspath(filepath), SDC.READ)
        v   = hdf.select("AOD_055")
        raw = v.get().astype(np.float32)           # shape (3600, 7200) — (lat, lon)
        att = v.attributes()
        hdf.end()

        fill  = att.get("_FillValue", -28672)
        scale = att.get("scale_factor", 0.001)

        raw[raw == fill] = np.nan
        aod = raw * scale

        # CMG grid centres: lat top→bottom, lon left→right
        lats = np.linspace(90 - 0.025, -90 + 0.025, 3600)   # 89.975 … -89.975
        lons = np.linspace(-180 + 0.025, 180 - 0.025, 7200)  # -179.975 … 179.975

        ds = xr.Dataset(
            {"AOD_055": (["lat", "lon"], aod)},
            coords={"lat": lats, "lon": lons},
        )
        # subset to bbox (lat must be sliced high→low since it's descending)
        return ds.sel(lat=slice(NORTH, SOUTH), lon=slice(WEST, EAST))

    except Exception as exc:
        print(f"    [MODIS] read error: {exc}")
        return None


# ── VIIRS AER_DBDT_D10KM_L3_VIIRS_NOAA20 reader ──────────────────────────────

def read_viirs(filepath) -> xr.Dataset | None:
    """
    Open AER_DBDT_D10KM_L3_VIIRS_NOAA20 NetCDF.
    dims: (Time=1, Longitude=3600, Latitude=1800) at 0.1°.
    Returns xr.Dataset subsetted to BBOX or None on error.
    """
    try:
        ds = xr.open_dataset(filepath, engine="netcdf4")

        # Keep only the combined AOD variable
        ds = ds[["COMBINE_AOD_550_AVG"]].rename({"COMBINE_AOD_550_AVG": "AOD_055"})

        # Rename coords to standard names and drop the Time dim (handled externally)
        ds = ds.rename({"Latitude": "lat", "Longitude": "lon"})
        ds = ds.squeeze("Time", drop=True)

        # Subset: Longitude coord is (3600,) -180→180; Latitude is (1800,) -90→90
        return ds.sel(lat=slice(SOUTH, NORTH), lon=slice(WEST, EAST))

    except Exception as exc:
        print(f"    [VIIRS] read error: {exc}")
        return None


# ── generic monthly ingest ────────────────────────────────────────────────────

def ingest_product(
    short_name: str,
    version: str | None,
    store_path: str,
    reader,
    label: str,
) -> None:
    repo  = open_repo(store_path)
    first = True

    # Detect if store already has data so we can resume
    try:
        session   = repo.readonly_session("main")
        existing  = xr.open_zarr(session.store(), consolidated=False)
        last_time = existing.time.values[-1]
        resume_from = (
            (last_time.astype("datetime64[D]").astype(date)) + timedelta(days=1)
        )
        first = False
        print(f"[{label}] Resuming from {resume_from} (last stored: {last_time})")
    except Exception:
        resume_from = START
        print(f"[{label}] Starting fresh from {START}")

    for month_start, month_end in month_range(resume_from, END):
        print(f"\n[{label}] {month_start} → {month_end}")

        search_kwargs = dict(
            short_name=short_name,
            temporal=(str(month_start), str(month_end)),
            bounding_box=BBOX,
            count=40,
        )
        if version:
            search_kwargs["version"] = version

        granules = earthaccess.search_data(**search_kwargs)
        print(f"  Found {len(granules)} granules")
        if not granules:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            files = earthaccess.download(granules, local_path=tmpdir)

            daily = []
            for f, g in zip(sorted(files), granules):
                t_str  = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
                obs_dt = np.datetime64(t_str[:10], "ns")
                ds     = reader(f)
                if ds is None:
                    continue
                ds = ds.expand_dims(time=[obs_dt])
                daily.append(ds)

            if not daily:
                print("  No valid files — skipping month")
                continue

            month_ds = xr.concat(daily, dim="time")
            month_ds = month_ds.chunk(
                {"time": TIME_CHUNK, "lat": LAT_CHUNK, "lon": LON_CHUNK}
            )

            w_session = repo.writable_session("main")
            store     = w_session.store()

            if first:
                month_ds.to_zarr(store, mode="w", consolidated=False, zarr_format=3)
                first = False
            else:
                month_ds.to_zarr(store, append_dim="time", consolidated=False)

            w_session.commit(f"{label} {month_start:%Y-%m}")
            print(f"  Committed {len(daily)} days → gs://{GCS_BUCKET}/{store_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    setup_earthdata()

    ingest_product(
        short_name="MCD19A2CMG",
        version="061",
        store_path=MODIS_STORE_PATH,
        reader=read_modis,
        label="MODIS MAIAC AOD",
    )

    ingest_product(
        short_name="AER_DBDT_D10KM_L3_VIIRS_NOAA20",
        version=None,
        store_path=VIIRS_STORE_PATH,
        reader=read_viirs,
        label="VIIRS NOAA-20 L3 AOD",
    )


if __name__ == "__main__":
    main()
