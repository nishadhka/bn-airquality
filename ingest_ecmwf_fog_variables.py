#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "xarray",
#     "dask",
#     "fsspec",
#     "s3fs",
#     "pyarrow",
#     "icechunk",
#     "gribberish",
#     "cfgrib",
#     "eccodes",
#     "coiled",
#     "distributed",
#     "bokeh>=3.1.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
ECMWF IFS Ensemble — fog Bayesian-network input vars to source.coop Icechunk
============================================================================

Streams ECMWF IFS ensemble fog-relevant variables (51 members, 0.25°)
over the bn-airquality bbox (20°E–100°E, 15°S–40°N), writing a
materialised Icechunk store on source.coop S3. Adapted from
``grib-index-kerchunk/ecmwf/dev-test/ecmwf_ea_tp_source_coop.py``.

Variable set — reconciled to what the published GIK parquet
(``E4DRR/gik-ecmwf-par``) actually contains, mapped to BN nodes:

  BN node                 Original plan      Stored here          Notes
  ─────────────────────── ─────────────────  ───────────────────  ──────────────────────
  Surface temperature     t2m                t2m   (= 2t)         direct
  Surface dewpoint        d2m                d2m   (= 2d)         derive RH via Magnus
  Wind                    u10/v10            u10/v10              speed = √(u²+v²)
  Boundary layer height   blh  ❌ not in pq  t_pl  (= t /pl)      inversion = t_pl − t2m
  Cloud cover             tcc  ❌ not in pq  strd                 thermal-radiation proxy
  Mid-tropo RH            r 925/850/700      r_pl  (= r /pl)      single-level only †
  Synoptic                —                  msl                  Western Disturbance
  Bonus: solar radiation  —                  ssrd                 fog dissipation timing
  Bonus: precipitation    —                  tp                   exclusion variable

  † GIK parquet exposes one ``r/pl`` (and one ``t/pl``) entry per step,
    not a profile. The level value is encoded in the GRIB message itself.
    Run ``probe-levels`` once to decode and print the actual hPa values
    for r_pl and t_pl, then update the metadata block below.

Pipeline:
  HF parquets (E4DRR/gik-ecmwf-par)
    -> Coiled/Dask workers fetch GRIB byte-ranges from s3://ecmwf-forecasts (anon)
    -> Decode with gribberish, subset to bbox
    -> Coordinator writes per-(date) regions to source.coop S3 Icechunk

Credentials:
  Workers     : none (HF public, ECMWF S3 anonymous)
  Coordinator : AWS STS temporary credentials in .env (1-hour lifetime)
                Same credential-timeout guard as the tp example.

Subcommands:
  init          — Create empty template store on source.coop
  fill          — Populate with real data via Dask/Coiled
  verify        — Inspect store contents (anonymous read)
  probe-levels  — Decode one r/pl and t/pl GRIB to print the actual hPa level

Usage:
    # 0. (one-off) confirm which pressure level the parquet exposes for r/t
    uv run ingest_ecmwf_fog_variables.py probe-levels --date 20251001

    # 1. Create .env with fresh STS credentials for source.coop
    cat > .env << 'EOF'
    export AWS_ACCESS_KEY_ID="ASIA..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_SESSION_TOKEN="..."
    export AWS_DEFAULT_REGION="us-west-2"
    EOF

    # 2. One-time template
    uv run ingest_ecmwf_fog_variables.py init  --start-date 20251001 --end-date 20251031

    # 3. Fill (refresh creds between sessions if it times out)
    uv run ingest_ecmwf_fog_variables.py fill  --start-date 20251001 --end-date 20251031 --n-workers 30

    # 4. Verify
    uv run ingest_ecmwf_fog_variables.py verify

Author: ICPAC GIK / bn-airquality
"""

import json
import logging
import os
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ingest_ecmwf_fog_variables.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_dotenv_into_environ():
    """Parse SCRIPT_DIR/.env (shell ``export`` syntax) into os.environ if it
    exists. Runs at import so env-overridable constants (S3_BUCKET, S3_PREFIX)
    pick up .env values. No-op on Coiled workers where .env is absent."""
    env_path = SCRIPT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv_into_environ()

# source.coop S3 target — overridable via env so the same code can target
# different repos on source.coop (or any S3-compatible store) without an edit.
S3_BUCKET = os.environ.get("S3_BUCKET", "us-west-2.opendata.source.coop")
S3_PREFIX = os.environ.get("S3_PREFIX", "e4drr-project/forecasts/ecmwf_fog_ifs_ens")
S3_REGION = os.environ.get("S3_REGION", "us-west-2")

# HuggingFace parquet source
HF_BASE_URL = (
    "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/run_par_ecmwf"
)
HF_COMBINED_URL = (
    "https://huggingface.co/datasets/E4DRR/gik-ecmwf-par/resolve/main/combined"
)

# Forecast lead times (00z run): "next 7 days of timesteps"
# 0–144h every 3h + 150–168h every 6h = 53 steps
LEAD_TIME_HOURS_3H = list(range(0, 145, 3))
LEAD_TIME_HOURS_6H = [150, 156, 162, 168]
LEAD_TIME_HOURS = LEAD_TIME_HOURS_3H + LEAD_TIME_HOURS_6H

# Ensemble: 1 control + 50 ensemble members
MEMBER_IDS = ["control"] + [f"ens_{i:02d}" for i in range(1, 51)]

# ECMWF global grid (0.25°)
ECMWF_GRID_SHAPE = (721, 1440)
ECMWF_LATS = np.linspace(90, -90, 721)
ECMWF_LONS = np.linspace(-180, 179.75, 1440)

# bn-airquality bbox: extended East-Africa-to-India
LAT_MIN, LAT_MAX = -15, 40
LON_MIN, LON_MAX = 20, 100
_lat_mask = (ECMWF_LATS >= LAT_MIN) & (ECMWF_LATS <= LAT_MAX)
_lon_mask = (ECMWF_LONS >= LON_MIN) & (ECMWF_LONS <= LON_MAX)
LAT_INDICES = np.where(_lat_mask)[0]
LON_INDICES = np.where(_lon_mask)[0]
EA_LATS = ECMWF_LATS[LAT_INDICES[0]: LAT_INDICES[-1] + 1]
EA_LONS = ECMWF_LONS[LON_INDICES[0]: LON_INDICES[-1] + 1]
N_LAT = len(EA_LATS)
N_LON = len(EA_LONS)
N_STEPS = len(LEAD_TIME_HOURS)
N_MEMBERS = len(MEMBER_IDS)

# Variables present in the GIK parquet (verified empirically against
# 2025-10-01 control parquet — all listed below were found).
SURFACE_VARS: Dict[str, str] = {
    "2t":   "t2m",
    "2d":   "d2m",
    "10u":  "u10",
    "10v":  "v10",
    "msl":  "msl",
    "strd": "strd",
    "ssrd": "ssrd",
    "tp":   "tp",
}
SURFACE_VAR_ATTRS: Dict[str, Dict[str, str]] = {
    "t2m":  {"long_name": "2 metre temperature", "units": "K"},
    "d2m":  {"long_name": "2 metre dewpoint temperature", "units": "K"},
    "u10":  {"long_name": "10 metre U wind component", "units": "m s-1"},
    "v10":  {"long_name": "10 metre V wind component", "units": "m s-1"},
    "msl":  {"long_name": "Mean sea level pressure", "units": "Pa"},
    "strd": {"long_name": "Surface thermal radiation downwards (cloud proxy)", "units": "J m-2"},
    "ssrd": {"long_name": "Surface solar radiation downwards", "units": "J m-2"},
    "tp":   {"long_name": "Total precipitation", "units": "m"},
}

# Single-level pressure-level vars exposed by the parquet — level value is
# inside the GRIB message; run `probe-levels` to confirm it.
PRESSURE_VARS: Dict[str, str] = {
    "r": "r_pl",
    "t": "t_pl",
}
PRESSURE_VAR_ATTRS: Dict[str, Dict[str, str]] = {
    "r_pl": {
        "long_name": "Relative humidity at pressure level",
        "units": "%",
        "note": "Single level as exposed by GIK parquet; "
                "run `probe-levels` to identify the hPa value.",
    },
    "t_pl": {
        "long_name": "Air temperature at pressure level",
        "units": "K",
        "note": "Use with t2m to compute inversion strength (BLH proxy). "
                "Run `probe-levels` to identify the hPa value.",
    },
}

ALL_OUT_NAMES = list(SURFACE_VARS.values()) + list(PRESSURE_VARS.values())

# Chunk shape: one date, one member, all lead times + spatial
CHUNK_SHAPE = (1, 1, N_STEPS, N_LAT, N_LON)

# Stop ~15 min before the 1-hour STS token expires
CREDENTIAL_TIMEOUT_SECONDS = 45 * 60


# ─── Credential / storage helpers ──────────────────────────────────────────


def load_s3_credentials():
    """Load source.coop S3 credentials from .env (shell ``export`` syntax)."""
    env_path = SCRIPT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise RuntimeError(
            "AWS credentials not found. Create a .env with:\n"
            '  export AWS_ACCESS_KEY_ID="ASIA..."\n'
            '  export AWS_SECRET_ACCESS_KEY="..."\n'
            '  export AWS_SESSION_TOKEN="..."\n'
            '  export AWS_DEFAULT_REGION="us-west-2"'
        )
    return {
        "access_key_id": access_key,
        "secret_access_key": secret_key,
        "session_token": session_token,
    }


def make_s3_storage(creds=None, anonymous=False):
    import icechunk

    if anonymous:
        return icechunk.s3_storage(
            bucket=S3_BUCKET,
            prefix=S3_PREFIX,
            region=S3_REGION,
            anonymous=True,
        )
    if creds is None:
        creds = load_s3_credentials()
    return icechunk.s3_storage(
        bucket=S3_BUCKET,
        prefix=S3_PREFIX,
        region=S3_REGION,
        access_key_id=creds["access_key_id"],
        secret_access_key=creds["secret_access_key"],
        session_token=creds.get("session_token"),
    )


def make_storage(local: str = None, anonymous: bool = False):
    if local:
        import icechunk
        return icechunk.local_filesystem_storage(path=local)
    return make_s3_storage(anonymous=anonymous)


# ─── Date helpers ───────────────────────────────────────────────────────────


def build_date_list(start_date: str, end_date: str) -> List[str]:
    return [d.strftime("%Y%m%d")
            for d in pd.date_range(start_date, end_date, freq="D")]


# ─── Phase 0: probe-levels ─────────────────────────────────────────────────


def probe_levels(args):
    """Decode one r/pl and t/pl GRIB byte-range from the parquet to print
    the actual hPa level. Run once before init to confirm the metadata."""
    import fsspec
    import xarray as xr

    date_str = args.date
    parquet_url = (
        f"{HF_BASE_URL}/{date_str[:4]}/{date_str[4:6]}/{date_str}/00z/"
        f"{date_str}00z-control.parquet"
    )
    logger.info(f"Reading parquet: {parquet_url}")
    df = pd.read_parquet(parquet_url)

    zstore = {}
    for _, row in df.iterrows():
        k, v = row["key"], row["value"]
        if isinstance(v, bytes):
            try:
                d = v.decode("utf-8")
                v = json.loads(d) if d.startswith(("[", "{")) else d
            except Exception:
                pass
        elif isinstance(v, str) and v.startswith(("[", "{")):
            try:
                v = json.loads(v)
            except Exception:
                pass
        zstore[k] = v

    s3 = fsspec.filesystem("s3", anon=True)

    def decode_one(key):
        ref = zstore.get(key)
        if not ref:
            logger.warning(f"  key not found: {key}")
            return
        url, off, length = ref[0], ref[1], ref[2]
        if not url.endswith(".grib2"):
            url = url + ".grib2"
        logger.info(f"  fetching {length} bytes from {url} @ {off}")
        with s3.open(url, "rb") as f:
            f.seek(off)
            grib_bytes = f.read(length)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as t:
            t.write(grib_bytes)
            p = t.name
        try:
            ds = xr.open_dataset(p, engine="cfgrib",
                                 backend_kwargs={"indexpath": ""})
            logger.info(f"  data_vars: {list(ds.data_vars)}")
            for c in ("isobaricInhPa", "level", "pressure"):
                if c in ds.coords:
                    logger.info(f"  level coord '{c}': {ds[c].values}")
            for v in ds.data_vars:
                logger.info(f"  attrs[{v}]: {dict(ds[v].attrs)}")
        finally:
            os.unlink(p)

    for ec_var, out_name in PRESSURE_VARS.items():
        logger.info(f"\n=== {ec_var}/pl @ step_036 (out: {out_name}) ===")
        decode_one(f"step_036/{ec_var}/pl/control/0.0.0")


# ─── Phase 1: init ──────────────────────────────────────────────────────────


def init_store(args):
    """Create empty template store with all variables."""
    import dask.array as da
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("INIT: ECMWF fog-vars Icechunk store on source.coop")
    logger.info("=" * 60)
    start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    n_dates = len(dates)

    logger.info(f"  Dates       : {n_dates} ({dates[0]} -> {dates[-1]})")
    logger.info(f"  Members     : {N_MEMBERS}")
    logger.info(f"  Lead times  : {N_STEPS} (0..168 h)")
    logger.info(f"  Spatial     : {N_LAT} lat x {N_LON} lon "
                f"({LAT_MIN}..{LAT_MAX}N, {LON_MIN}..{LON_MAX}E)")
    logger.info(f"  Surface vars: {list(SURFACE_VARS.values())}")
    logger.info(f"  Pressure vars (single-level): {list(PRESSURE_VARS.values())}")

    init_date = pd.to_datetime(dates).values.astype("datetime64[ns]")
    lead_time = np.array(LEAD_TIME_HOURS, dtype=np.int32)
    member = np.array(MEMBER_IDS, dtype="U10")

    shape = (n_dates, N_MEMBERS, N_STEPS, N_LAT, N_LON)
    size_gb = np.prod(shape) * 4 * len(ALL_OUT_NAMES) / (1024 ** 3)
    logger.info(f"  Per-var shape: {shape}")
    logger.info(f"  Uncompressed total ({len(ALL_OUT_NAMES)} vars): {size_gb:.1f} GiB")

    data_vars = {}
    encoding = {}
    for out_name, attrs in {**{v: SURFACE_VAR_ATTRS[v] for v in SURFACE_VARS.values()},
                            **{v: PRESSURE_VAR_ATTRS[v] for v in PRESSURE_VARS.values()}}.items():
        data_vars[out_name] = (
            ("init_date", "member", "lead_time", "lat", "lon"),
            da.zeros(shape, chunks=shape, dtype=np.float32),
            attrs,
        )
        encoding[out_name] = {"chunks": CHUNK_SHAPE, "fill_value": float("nan")}

    template = xr.Dataset(
        data_vars,
        coords={
            "init_date": ("init_date", init_date),
            "member":    ("member", member),
            "lead_time": ("lead_time", lead_time, {"units": "hours"}),
            "lat":       ("lat", EA_LATS, {"units": "degrees_north"}),
            "lon":       ("lon", EA_LONS, {"units": "degrees_east"}),
        },
        attrs={
            "title": "ECMWF IFS Ensemble — fog Bayesian-network input variables",
            "source": "GIK parquet refs (E4DRR/gik-ecmwf-par) -> S3 GRIB byte-ranges -> gribberish",
            "institution": "ICPAC / bn-airquality",
            "region": "20-100E, 15S-40N (extended East-Africa-to-India)",
            "variables": ",".join(ALL_OUT_NAMES),
            "n_members": str(N_MEMBERS),
            "lead_time_hours": "0..168 (3h to 144h, 6h to 168h)",
            "bn_mapping": (
                "t2m=temperature; d2m=dewpoint(->RH); u10/v10=wind; "
                "t_pl-t2m=inversion(BLH proxy); strd=cloud proxy; "
                "r_pl=mid-tropo RH; msl=synoptic; ssrd=fog-dissipation; "
                "tp=exclusion"
            ),
            "storage": f"s3://{S3_BUCKET}/{S3_PREFIX}/",
        },
    )
    logger.info(f"  Template:\n{template}")

    storage = make_storage(args.local)
    config = icechunk.RepositoryConfig.default()
    try:
        repo = icechunk.Repository.create(storage, config=config)
        logger.info("  Created new repository")
    except Exception:
        repo = icechunk.Repository.open(storage, config=config)
        logger.info("  Opened existing repository (will overwrite)")

    session = repo.writable_session("main")
    template.to_zarr(
        session.store,
        compute=False,
        mode="w",
        encoding=encoding,
        consolidated=False,
    )
    session.commit("initialize ECMWF fog-vars template")

    logger.info("=" * 60)
    logger.info(f"INIT COMPLETE in {time.time() - start:.1f}s")
    logger.info(f"  Target: s3://{S3_BUCKET}/{S3_PREFIX}/")
    logger.info("=" * 60)


# ─── Worker function ────────────────────────────────────────────────────────


def read_member_fog_vars(
    date_str: str,
    member_id: str,
    lead_time_hours: List[int],
    surface_vars: Dict[str, str],
    pressure_vars: Dict[str, str],
    hf_base_url: str,
    grid_shape: Tuple[int, int],
    lat_idx_start: int,
    lat_idx_end: int,
    lon_idx_start: int,
    lon_idx_end: int,
    hf_combined_url: str = None,
):
    """One Dask task = one (date, member). Fetches every variable across all steps."""
    import json
    import os
    import tempfile
    import warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import fsspec
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    try:
        import gribberish
        has_gribberish = True
    except ImportError:
        has_gribberish = False

    n_steps = len(lead_time_hours)
    n_lat = lat_idx_end - lat_idx_start
    n_lon = lon_idx_end - lon_idx_start
    year = date_str[:4]
    month = date_str[4:6]

    df = None
    if hf_combined_url:
        try:
            combined_url = f"{hf_combined_url}/ecmwf_gik_00z.parquet"
            cache_dir = os.path.join(tempfile.gettempdir(), "gik_combined_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "ecmwf_gik_00z.parquet")
            if not os.path.exists(cache_path):
                import urllib.request
                urllib.request.urlretrieve(combined_url, cache_path)
            import pyarrow.parquet as pq
            table = pq.read_table(
                cache_path,
                filters=[("date", "==", date_str), ("member", "==", member_id)],
            )
            df = table.to_pandas()
            del table
        except Exception:
            df = None

    if df is None or df.empty:
        parquet_url = (
            f"{hf_base_url}/{year}/{month}/{date_str}/00z/"
            f"{date_str}00z-{member_id}.parquet"
        )
        df = pd.read_parquet(parquet_url)

    zstore = {}
    for _, row in df.iterrows():
        key = row["key"]
        value = row["value"]
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
                if decoded.startswith("[") or decoded.startswith("{"):
                    value = json.loads(decoded)
                else:
                    value = decoded
            except Exception:
                pass
        elif isinstance(value, str):
            if value.startswith("[") or value.startswith("{"):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
        zstore[key] = value
    del df

    member_key = member_id.replace("_", "")  # 'ens_01' -> 'ens01'

    out_data = {
        out_name: np.full((n_steps, n_lat, n_lon), np.nan, dtype=np.float32)
        for out_name in list(surface_vars.values()) + list(pressure_vars.values())
    }

    def find_ref(patterns):
        for p in patterns:
            if p in zstore:
                v = zstore[p]
                if isinstance(v, list) and len(v) >= 3:
                    return v
        return None

    work: List[Tuple[str, int, list]] = []
    for s_idx, step_h in enumerate(lead_time_hours):
        for ec_var, out_name in surface_vars.items():
            ref = find_ref([
                f"step_{step_h:03d}/{ec_var}/sfc/{member_key}/0.0.0",
                f"step_{step_h:03d}/{ec_var}/sfc/0.0.0",
                f"step_{step_h:03d}/{ec_var}/surface/{member_key}/0.0.0",
            ])
            if ref is not None:
                work.append((out_name, s_idx, ref))

        for ec_var, out_name in pressure_vars.items():
            ref = find_ref([
                f"step_{step_h:03d}/{ec_var}/pl/{member_key}/0.0.0",
                f"step_{step_h:03d}/{ec_var}/pl/0.0.0",
            ])
            if ref is not None:
                work.append((out_name, s_idx, ref))
    del zstore

    if not work:
        return {"date_str": date_str, "member_id": member_id, "data": out_data}

    s3_fs = fsspec.filesystem("s3", anon=True)

    def _fetch_one(out_name, s_idx, ref):
        url, offset, length = ref[0], ref[1], ref[2]
        if not url.endswith(".grib2"):
            url = url + ".grib2"
        with s3_fs.open(url, "rb") as f:
            f.seek(offset)
            grib_bytes = f.read(length)

        arr = None
        if has_gribberish:
            try:
                flat = gribberish.parse_grib_array(grib_bytes, 0)
                arr = flat.reshape(grid_shape)
            except Exception:
                pass
        if arr is None:
            import xarray as xr
            with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
                tmp.write(grib_bytes)
                tmp_path = tmp.name
            try:
                ds = xr.open_dataset(tmp_path, engine="cfgrib",
                                     backend_kwargs={"indexpath": ""})
                arr = ds[list(ds.data_vars)[0]].values.copy()
                ds.close()
            finally:
                os.unlink(tmp_path)

        ea_arr = arr[lat_idx_start:lat_idx_end,
                     lon_idx_start:lon_idx_end].astype(np.float32)
        return out_name, s_idx, ea_arr

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_fetch_one, on, si, ref) for on, si, ref in work]
        for fut in as_completed(futs):
            try:
                out_name, s_idx, ea_arr = fut.result()
                out_data[out_name][s_idx] = ea_arr
            except Exception:
                pass

    return {"date_str": date_str, "member_id": member_id, "data": out_data}


# ─── Phase 2a: local-fill (single-machine smoke test) ──────────────────────


def local_fill(args):
    """Single-machine fill — no Coiled. Smoke-tests one day end-to-end against
    the target store. Members run in a ThreadPoolExecutor; each member's
    inner step fetches use the 8-thread pool inside read_member_fog_vars."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("LOCAL-FILL: ECMWF fog-vars Icechunk (no Coiled)")
    logger.info("=" * 60)
    overall_start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    members = MEMBER_IDS[:args.limit_members] if args.limit_members else MEMBER_IDS
    logger.info(f"  Dates  : {len(dates)} ({dates[0]} -> {dates[-1]})")
    logger.info(f"  Members: {len(members)} ({members[0]} .. {members[-1]})")
    logger.info(f"  Member-parallelism: {args.member_workers}")

    target_storage = make_storage(args.local)
    target_repo = icechunk.Repository.open(
        target_storage, config=icechunk.RepositoryConfig.default()
    )

    session_ro = target_repo.readonly_session("main")
    template_ds = xr.open_zarr(session_ro.store, consolidated=False)
    init_dates = pd.to_datetime(template_ds["init_date"].values)
    date_to_idx = {d.strftime("%Y%m%d"): i for i, d in enumerate(init_dates)}
    template_ds.close()

    lat_idx_start = int(LAT_INDICES[0])
    lat_idx_end = int(LAT_INDICES[-1]) + 1
    lon_idx_start = int(LON_INDICES[0])
    lon_idx_end = int(LON_INDICES[-1]) + 1

    total_written = 0
    for date_str in dates:
        if date_str not in date_to_idx:
            logger.error(f"  date {date_str} not in template init_date axis "
                         f"(min={init_dates[0]}, max={init_dates[-1]}) — skip")
            continue
        date_idx = date_to_idx[date_str]
        t_date = time.time()
        logger.info(f"\n  -- {date_str}  (init_date idx {date_idx}) --")

        member_data: Dict[int, dict] = {}
        n_ok = 0
        n_fail = 0

        with ThreadPoolExecutor(max_workers=args.member_workers) as pool:
            futs = {
                pool.submit(
                    read_member_fog_vars,
                    date_str,
                    member_id,
                    LEAD_TIME_HOURS,
                    SURFACE_VARS,
                    PRESSURE_VARS,
                    HF_BASE_URL,
                    ECMWF_GRID_SHAPE,
                    lat_idx_start, lat_idx_end,
                    lon_idx_start, lon_idx_end,
                    HF_COMBINED_URL,
                ): (m_idx, member_id)
                for m_idx, member_id in enumerate(members)
            }
            done = 0
            for fut in as_completed(futs):
                m_idx, member_id = futs[fut]
                done += 1
                try:
                    member_data[m_idx] = fut.result()
                    n_ok += 1
                    logger.info(f"    [{done:2d}/{len(members):2d}] {member_id} OK")
                except Exception as e:
                    n_fail += 1
                    logger.error(f"    [{done:2d}/{len(members):2d}] {member_id} FAILED: {e}")

        if n_ok == 0:
            logger.error(f"  {date_str}: all members failed — nothing to write")
            continue

        # Only allocate buffers for the requested member slice — full N_MEMBERS
        # × 10 vars × float32 is ~7 GiB and OOMs on an 8 GiB box. Members are
        # contiguous (we always take the first N), so we can write one zarr
        # region covering exactly slice(0, len(members)).
        n_m = len(members)
        arrs = {
            out_name: np.full(
                (n_m, N_STEPS, N_LAT, N_LON), np.nan, dtype=np.float32
            )
            for out_name in ALL_OUT_NAMES
        }
        for m_i, res in member_data.items():
            for out_name in ALL_OUT_NAMES:
                arrs[out_name][m_i] = res["data"][out_name]

        session = target_repo.writable_session("main")
        ds_vars = {
            out_name: (
                ("init_date", "member", "lead_time", "lat", "lon"),
                arrs[out_name][np.newaxis],
            )
            for out_name in ALL_OUT_NAMES
        }
        ds_write = xr.Dataset(ds_vars)
        ds_write.to_zarr(
            session.store,
            region={
                "init_date": slice(date_idx, date_idx + 1),
                "member": slice(0, n_m),
            },
            consolidated=False,
        )
        session.commit(
            f"fill date {date_idx} ({date_str}): {n_ok}/{len(members)} members "
            f"[local-fill]"
        )
        total_written += 1
        logger.info(f"  {date_str}: committed in {time.time() - t_date:.1f}s "
                    f"({n_ok}/{len(members)} members, {n_fail} failed)")

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info(f"LOCAL-FILL COMPLETE: {total_written}/{len(dates)} dates "
                f"in {elapsed/60:.1f} min")
    logger.info("=" * 60)


# ─── Phase 2: fill ──────────────────────────────────────────────────────────


def fill_store(args):
    """Populate store using Dask/Coiled — one task per (date, member)."""
    import coiled
    import distributed
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("FILL: ECMWF fog-vars Icechunk store on source.coop")
    logger.info("=" * 60)
    overall_start = time.time()
    session_start = time.time()

    dates = build_date_list(args.start_date, args.end_date)
    n_dates = len(dates)
    logger.info(f"  Dates: {n_dates} ({dates[0]} -> {dates[-1]})")

    target_storage = make_storage(args.local)
    target_repo = icechunk.Repository.open(
        target_storage, config=icechunk.RepositoryConfig.default()
    )

    completed_indices = set()
    try:
        for commit in target_repo.ancestry(branch="main"):
            msg = commit.message
            if msg.startswith("fill date "):
                try:
                    idx_str = msg.split("fill date ")[1].split(" ")[0]
                    completed_indices.add(int(idx_str))
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    start_idx = max(completed_indices) + 1 if completed_indices else 0
    if start_idx > 0:
        logger.info(f"  Resuming from date index {start_idx} "
                    f"({len(completed_indices)} dates already done)")

    remaining = [(i, d) for i, d in enumerate(dates) if i >= start_idx]
    if not remaining:
        logger.info("  All dates already filled.")
        return
    logger.info(f"  Remaining: {len(remaining)} dates")

    timeout = args.credential_timeout
    logger.info(f"  Credential timeout: {timeout}s ({timeout/60:.0f} min)")

    cluster = coiled.Cluster(
        name=f"ecmwf-fog-{int(time.time()) % 10000}",
        n_workers=args.n_workers,
        worker_vm_types=args.worker_vm_types,
        package_sync=True,
        region=args.coiled_region,
        idle_timeout="30 minutes",
        workspace=args.workspace,
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=min(10, args.n_workers), timeout=600)
    logger.info(f"  Cluster ready: {client.dashboard_link}")

    lat_idx_start = int(LAT_INDICES[0])
    lat_idx_end = int(LAT_INDICES[-1]) + 1
    lon_idx_start = int(LON_INDICES[0])
    lon_idx_end = int(LON_INDICES[-1]) + 1

    total_written = 0
    total_failed = 0
    failed_dates: List[str] = []
    timed_out = False

    # Process one date at a time. As each member future completes, stream-write
    # its slab into a single writable_session (chunk-aligned: one chunk per
    # (date, member)), then commit once per date. This keeps the coordinator
    # memory peak at ~150 MiB instead of the ~14 GiB the previous batch-buffer
    # approach required (51 members × 53 steps × 221 × 321 × float32 × 10 vars
    # — buffered both in date_members and in arrs simultaneously).
    for date_idx, date_str in remaining:
        elapsed_session = time.time() - session_start
        if elapsed_session > timeout:
            logger.warning(
                f"  Credential timeout ({timeout}s) reached after "
                f"{elapsed_session:.0f}s. Stopping to avoid expired token. "
                f"Refresh .env and rerun to resume."
            )
            timed_out = True
            break

        date_t0 = time.time()
        logger.info(
            f"\n  -- date {date_idx} ({date_str}) — submitting {N_MEMBERS} "
            f"member fetches  ({total_written}/{len(remaining)} done, "
            f"{elapsed_session:.0f}s/{timeout}s budget)"
        )

        futures = {}
        for m_idx, member_id in enumerate(MEMBER_IDS):
            future = client.submit(
                read_member_fog_vars,
                date_str,
                member_id,
                LEAD_TIME_HOURS,
                SURFACE_VARS,
                PRESSURE_VARS,
                HF_BASE_URL,
                ECMWF_GRID_SHAPE,
                lat_idx_start, lat_idx_end,
                lon_idx_start, lon_idx_end,
                HF_COMBINED_URL,
                key=f"d{date_idx}-m{m_idx:02d}",
            )
            futures[future] = (m_idx, member_id)

        session = target_repo.writable_session("main")
        n_ok = 0
        n_fail = 0
        write_failed = False
        for future in distributed.as_completed(list(futures.keys())):
            m_idx, member_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                n_fail += 1
                logger.error(f"    member {member_id} (m_idx={m_idx}) FETCH FAILED: {e}")
                continue

            try:
                ds_vars = {
                    out_name: (
                        ("init_date", "member", "lead_time", "lat", "lon"),
                        result["data"][out_name][np.newaxis, np.newaxis],
                    )
                    for out_name in ALL_OUT_NAMES
                }
                xr.Dataset(ds_vars).to_zarr(
                    session.store,
                    region={
                        "init_date": slice(date_idx, date_idx + 1),
                        "member": slice(m_idx, m_idx + 1),
                    },
                    consolidated=False,
                )
                n_ok += 1
                del result, ds_vars
                if n_ok % 10 == 0 or n_ok == N_MEMBERS:
                    logger.info(
                        f"    [{n_ok + n_fail:2d}/{N_MEMBERS}] "
                        f"member {member_id} written "
                        f"(ok={n_ok}, failed={n_fail}, "
                        f"{time.time() - date_t0:.0f}s elapsed)"
                    )
            except Exception as e:
                write_failed = True
                logger.error(f"    member {member_id} WRITE FAILED: {e}")
                break

        if write_failed or n_ok == 0:
            total_failed += 1
            failed_dates.append(date_str)
            logger.error(
                f"    Date {date_idx} ({date_str}) FAILED — discarding session "
                f"(ok={n_ok}, failed={n_fail})"
            )
            continue

        try:
            session.commit(
                f"fill date {date_idx} ({date_str}): {n_ok}/{N_MEMBERS} members"
            )
            total_written += 1
            logger.info(
                f"    Committed date {date_idx} ({date_str}) "
                f"[{n_ok}/{N_MEMBERS} members, {n_fail} failed] "
                f"in {time.time() - date_t0:.0f}s "
                f"({total_written}/{len(remaining)} total)"
            )
        except Exception as e:
            total_failed += 1
            failed_dates.append(date_str)
            logger.error(
                f"    Date {date_idx} ({date_str}) COMMIT FAILED: {e}"
            )

    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("FILL COMPLETE" + (" (timed out — resume with fresh creds)" if timed_out else ""))
    logger.info(f"  Dates written: {total_written}/{n_dates}")
    logger.info(f"  Failed: {total_failed} -- {failed_dates[:20]}")
    logger.info(f"  Time: {elapsed/60:.1f} min")
    logger.info(f"  Store: s3://{S3_BUCKET}/{S3_PREFIX}/")
    logger.info("=" * 60)

    results = {
        "status": "timed_out" if timed_out else ("success" if not failed_dates else "partial"),
        "dates_written": total_written,
        "dates_total": n_dates,
        "failed_dates": failed_dates,
        "timed_out": timed_out,
        "elapsed_min": elapsed / 60,
    }
    out = SCRIPT_DIR / f"ecmwf_fog_vars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  Results: {out}")


# ─── Phase 3: verify ────────────────────────────────────────────────────────


def verify_store(args):
    """Inspect store contents (anonymous read against source.coop)."""
    import icechunk
    import xarray as xr

    logger.info("=" * 60)
    logger.info("VERIFY: ECMWF fog-vars Icechunk store")
    logger.info("=" * 60)

    storage = make_storage(args.local, anonymous=not args.local)
    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    logger.info(f"\nDataset:\n{ds}")
    logger.info(f"\nDimensions: {dict(ds.sizes)}")
    for dim in ["init_date", "member", "lead_time", "lat", "lon"]:
        if dim in ds.dims:
            v = ds[dim].values
            logger.info(f"  {dim}: {ds.sizes[dim]} [{v[0]} .. {v[-1]}]")

    for var in ds.data_vars:
        d = ds[var]
        logger.info(f"\nVariable '{var}': dtype={d.dtype}, shape={d.shape}")

    if args.spot_check:
        logger.info("\nSpot-check: first date, first member...")
        for var in ds.data_vars:
            try:
                sample = ds[var].isel(init_date=0, member=0).load()
                vals = sample.values
                n_valid = int((~np.isnan(vals)).sum())
                pct = 100 * n_valid / vals.size if vals.size else 0
                line = f"  {var}: {n_valid}/{vals.size} valid ({pct:.1f}%)"
                if n_valid > 0:
                    good = vals[~np.isnan(vals)]
                    line += (f"  min={float(good.min()):.4g}"
                             f"  max={float(good.max()):.4g}"
                             f"  mean={float(good.mean()):.4g}")
                logger.info(line)
            except Exception as e:
                logger.error(f"  {var}: spot-check failed: {e}")

    try:
        commits = list(repo.ancestry(branch="main"))
        logger.info(f"\nCommits ({len(commits)}):")
        for c in commits[:10]:
            logger.info(f"  {c.message}")
        if len(commits) > 10:
            logger.info(f"  ... and {len(commits) - 10} more")
    except Exception:
        pass

    logger.info("\nVerification complete.")


# ─── CLI ────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ECMWF IFS ensemble fog-relevant vars to source.coop Icechunk",
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--local", type=str, default=None,
                        help="Local Icechunk path (overrides source.coop S3)")

    p_probe = sub.add_parser("probe-levels",
                             help="Decode one r/pl + t/pl GRIB to print actual hPa level")
    p_probe.add_argument("--date", type=str, default="20251001")

    p_init = sub.add_parser("init", parents=[common],
                            help="Create empty template store")
    p_init.add_argument("--start-date", type=str, default="20251001")
    p_init.add_argument("--end-date",   type=str, default="20251031")

    p_local = sub.add_parser("local-fill", parents=[common],
                             help="Single-machine smoke test (no Coiled) — fills "
                                  "the already-init'd store for the given date(s)")
    p_local.add_argument("--start-date",     type=str, default="20251001")
    p_local.add_argument("--end-date",       type=str, default="20251001")
    p_local.add_argument("--limit-members",  type=int, default=None,
                         help="Only fetch the first N members (smoke test); "
                              "missing members stored as NaN")
    p_local.add_argument("--member-workers", type=int, default=4,
                         help="Concurrent member fetches (default 4)")

    p_fill = sub.add_parser("fill", parents=[common],
                            help="Fill store from HF parquets + S3 GRIB byte-range reads")
    p_fill.add_argument("--start-date",      type=str, default="20251001")
    p_fill.add_argument("--end-date",        type=str, default="20251031")
    p_fill.add_argument("--n-workers",       type=int, default=20)
    p_fill.add_argument("--commit-batch",    type=int, default=10,
                        help="Number of dates per processing batch")
    p_fill.add_argument("--worker-vm-types", type=str, default="e2-standard-4")
    p_fill.add_argument("--coiled-region",   type=str, default="us-east1")
    p_fill.add_argument("--workspace",       type=str, default=None,
                        help="Coiled workspace (default: account default)")
    p_fill.add_argument("--credential-timeout", type=int,
                        default=CREDENTIAL_TIMEOUT_SECONDS,
                        help="Stop before this many seconds to avoid expired "
                             "STS token (default: 2700 = 45 min)")

    p_verify = sub.add_parser("verify", parents=[common],
                              help="Inspect store contents")
    p_verify.add_argument("--spot-check",   dest="spot_check", action="store_true",  default=True)
    p_verify.add_argument("--no-spot-check", dest="spot_check", action="store_false")

    args = parser.parse_args()
    if args.command == "probe-levels":
        probe_levels(args)
    elif args.command == "init":
        init_store(args)
    elif args.command == "local-fill":
        local_fill(args)
    elif args.command == "fill":
        fill_store(args)
    elif args.command == "verify":
        verify_store(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
