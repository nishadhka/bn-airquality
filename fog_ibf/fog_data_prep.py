#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "icechunk",
#     "xarray",
#     "zarr>=3",
#     "numpy",
#     "pandas",
#     "geopandas",
#     "regionmask",
#     "netcdf4",
#     "pyarrow",
#     "scipy",
#     "fsspec",
#     "s3fs",
#     "requests",
#     "shapely",
#     "pyproj",
# ]
# ///
"""
Fog / Air-Quality BN-IBF v1 — per-day zone-level input generator.

Mirrors bn-ibf/flood_ibf/flood_data_prep.py. Reads:
  - ECMWF IFS ensemble fog-vars Icechunk store on source.coop (51 mem, 10 vars)
  - Sentinel-5P-PAL daily L3 AOD / TCWV / NO2 via STAC (optional)
  - Existing yearly S5P L3 AOD NetCDF as climatological fallback
  - ICPAC admin-1 GeoJSON (227 boundaries)

Writes one CSV row per admin-1 holding the evidence vector consumed by
fog_bn_ibf_v1.jl:
    id, name, country,
    antecedent_aerosol_aod, antecedent_aerosol_state,
    antecedent_moisture_kgm2, antecedent_moisture_state,
    ifs_fog_prob, fog_prob_24h, fog_prob_48h, fog_prob_72h,
    spatial_coverage, stagnation_trend, stagnation_slope_ms,
    extreme_fog_tail_p95, ens_max_fog_index_peak,
    target_date

With --soft-evidence, adds Gaussian-binned probability vectors:
    aer_p1..p3, mois_p1..p3, fog_p1..p5, stag_p1..p3, tail_p1..p4
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import icechunk as ic
import numpy as np
import pandas as pd
import regionmask
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─── Constants ────────────────────────────────────────────────────────

# IFS fog-vars store location on source.coop. Override via env if you
# republish to a different prefix.
S3_BUCKET = os.environ.get("FOG_IFS_S3_BUCKET", "us-west-2.opendata.source.coop")
S3_PREFIX = os.environ.get("FOG_IFS_S3_PREFIX", "nishadhka/aq-icechunk-store-ifs")
S3_REGION = os.environ.get("FOG_IFS_S3_REGION", "us-west-2")

# Lead-time windows we'll sample for forecast probabilities (hours)
LEADS_24H = [21, 24, 27]      # ~D+1
LEADS_48H = [45, 48, 51]      # ~D+2
LEADS_72H = [69, 72, 75]      # ~D+3
LEADS_ALL_FORMATION = [9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
                        51, 54, 57, 60, 63, 66, 69, 72]  # ≈ overnight windows

# Stagnation trend window: compare wind speed near-term vs out-3d
LEADS_NEAR = [12, 24]
LEADS_FAR  = [60, 72]
TREND_BAND_MS = 0.5

# Fog-formation thresholds (used to compute per-pixel per-member fog index).
#
# Calibration notes (probed against IFS init 2025-01-01 over the EA-to-India
# bbox, 51 members):
#   - 2 m RH      mean ~59%, p90 ~82%, max ~100% → 92% pivot keeps only
#                 near-saturation pixels.
#   - 10 m WS     mean ~4 m/s, p10 ~1 m/s → 4 m/s pivot retains low-wind tail.
#   - tp          is *cumulative* since init; at lead 72h max ~180 mm. Use 2 mm
#                 as the "wet enough to wash out fog" threshold.
#   - t_pl        in this store is at upper-troposphere (~300 hPa, mean ~238 K),
#                 so t_pl - t2m is structurally negative ~-55 K and is NOT a
#                 useful inversion proxy. Term removed; revisit when a 925/850
#                 hPa level is published (run `probe-levels` to confirm).
FOG_RH_PIVOT_PCT  = 92.0
FOG_RH_SCALE_PCT  = 3.0
FOG_WIND_PIVOT    = 4.0
FOG_WIND_SCALE    = 1.5
FOG_TP_PIVOT_M    = 2e-3
FOG_TP_SCALE_M    = 2e-3
FOG_INDEX_FOG_THR = 0.5

# State threshold cutoffs — must match the Julia categorize_* functions
AER_THRESHOLDS  = (0.3, 0.6)
MOIS_THRESHOLDS = (15.0, 30.0)
FOG_THRESHOLDS  = (0.2, 0.4, 0.6, 0.8)
STAG_THRESHOLDS = (-TREND_BAND_MS, TREND_BAND_MS)
TAIL_THRESHOLDS = (0.5, 0.75, 0.9)

ISO_TO_COUNTRY = {
    "BDI": "Burundi", "DJI": "Djibouti", "ERI": "Eritrea", "ETH": "Ethiopia",
    "KEN": "Kenya", "RWA": "Rwanda", "SOM": "Somalia", "SSD": "South Sudan",
    "SDN": "Sudan", "TZA": "Tanzania", "UGA": "Uganda",
}


# ─── Soft-evidence binning (mirrors flood pattern) ────────────────────

_NODE_EDGES = {
    "aer":  [-np.inf, AER_THRESHOLDS[0], AER_THRESHOLDS[1], np.inf],            # 3 bins
    "mois": [-np.inf, MOIS_THRESHOLDS[0], MOIS_THRESHOLDS[1], np.inf],          # 3 bins
    "fog":  [-np.inf, FOG_THRESHOLDS[0], FOG_THRESHOLDS[1],
             FOG_THRESHOLDS[2], FOG_THRESHOLDS[3], np.inf],                      # 5 bins
    "stag": [-np.inf, STAG_THRESHOLDS[0], STAG_THRESHOLDS[1], np.inf],          # 3 bins
    "tail": [-np.inf, TAIL_THRESHOLDS[0], TAIL_THRESHOLDS[1],
             TAIL_THRESHOLDS[2], np.inf],                                        # 4 bins
}
_NODE_SIGMA_DEFAULT = {
    "aer":  0.05, "mois": 5.0, "fog":  0.05, "stag": 0.3, "tail": 0.05,
}


def soft_bin(x: float, node: str, sigma: Optional[float] = None) -> np.ndarray:
    from scipy import stats as _st
    edges = _NODE_EDGES[node]
    k = len(edges) - 1
    if not np.isfinite(x):
        return np.full(k, 1.0 / k)
    s = _NODE_SIGMA_DEFAULT[node] if sigma is None else sigma
    probs = np.diff(_st.norm.cdf(edges, loc=x, scale=s))
    tot = probs.sum()
    return probs / tot if tot > 0 else np.full(k, 1.0 / k)


def add_soft_columns(df: pd.DataFrame, *,
                     aer: np.ndarray, mois: np.ndarray, fog: np.ndarray,
                     stag_slope: np.ndarray, tail: np.ndarray) -> None:
    """In-place: 3+3+5+3+4 = 18 soft-evidence columns."""
    for node, vals, k in [("aer", aer, 3), ("mois", mois, 3),
                          ("fog", fog, 5), ("stag", stag_slope, 3),
                          ("tail", tail, 4)]:
        probs = np.vstack([soft_bin(float(v), node) for v in vals])
        for i in range(k):
            df[f"{node}_p{i+1}"] = np.round(probs[:, i], 4)


# ─── Hard classifications (mirror Julia categorize_* exactly) ─────────

def classify_aer(aod: float) -> str:
    if not np.isfinite(aod): return "Moderate"
    if aod < AER_THRESHOLDS[0]: return "Low"
    if aod < AER_THRESHOLDS[1]: return "Moderate"
    return "High"


def classify_mois(tcwv: float) -> str:
    if not np.isfinite(tcwv): return "Moderate"
    if tcwv < MOIS_THRESHOLDS[0]: return "Dry"
    if tcwv < MOIS_THRESHOLDS[1]: return "Moderate"
    return "Moist"


def classify_stag(slope: float) -> str:
    if not np.isfinite(slope): return "Stable"
    if slope > STAG_THRESHOLDS[1]: return "Improving"
    if slope < STAG_THRESHOLDS[0]: return "Stagnating"
    return "Stable"


# ─── IFS Icechunk reader ──────────────────────────────────────────────

def open_ifs_icechunk(prefix: Optional[str] = None) -> xr.Dataset:
    """Open the source.coop IFS fog-vars Icechunk store anonymously.

    decode_timedelta=False so lead_time stays as int32 hours; we look up
    leads by integer hour counts in compute_ifs_evidence().
    """
    storage = ic.s3_storage(
        bucket=S3_BUCKET,
        prefix=prefix or S3_PREFIX,
        region=S3_REGION,
        anonymous=True,
    )
    repo = ic.Repository.open(storage)
    return xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        decode_timedelta=False,
    )


# ─── Per-pixel per-member fog-formation index ─────────────────────────

def _sigmoid(x):
    if isinstance(x, xr.DataArray):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def magnus_rh(t2m_K: xr.DataArray, d2m_K: xr.DataArray) -> xr.DataArray:
    """RH (%) from 2-m temperature and dewpoint (K). Magnus / Alduchov-Eskridge."""
    a, b = 17.625, 243.04
    t_C = t2m_K - 273.15
    d_C = d2m_K - 273.15
    es_t = np.exp(a * t_C / (b + t_C))
    es_d = np.exp(a * d_C / (b + d_C))
    rh = 100.0 * (es_d / es_t)
    return rh.clip(0.0, 150.0)


def fog_index_per_member(
    t2m: xr.DataArray, d2m: xr.DataArray,
    u10: xr.DataArray, v10: xr.DataArray,
    t_pl: xr.DataArray, tp: xr.DataArray,
) -> xr.DataArray:
    """Compute fog-formation index F ∈ [0,1] per (member, lead, lat, lon).

    Three fuzzy factors:
      • RH near 92% (sigmoid pivot)         — saturation
      • 10 m WS near 4 m/s (descending)     — calm boundary layer
      • Cumulative tp below 2 mm            — exclude wash-out by precip

    t_pl is intentionally unused: in this store the level is upper-troposphere,
    so t_pl - t2m is structurally negative and not a usable BLH proxy.
    """
    _ = t_pl  # parameter retained for backwards-compat with caller
    rh = magnus_rh(t2m, d2m)
    ws = (u10 * u10 + v10 * v10) ** 0.5
    rh_term = _sigmoid((rh - FOG_RH_PIVOT_PCT) / FOG_RH_SCALE_PCT)
    ws_term = _sigmoid((FOG_WIND_PIVOT - ws) / FOG_WIND_SCALE)
    tp_term = 1.0 - _sigmoid((tp - FOG_TP_PIVOT_M) / FOG_TP_SCALE_M)
    return (rh_term * ws_term * tp_term).astype("float32")


# ─── Zonal helpers (lifted from flood pattern) ────────────────────────

# Boundary-file schemas we know how to consume. Each entry maps the canonical
# names (id / name / group) the rest of the script expects to the property
# columns the source file actually carries. Add new entries here when a new
# admin-boundary GeoJSON shows up.
_SCHEMAS = {
    "icpac_adm1":     {"id": "GID_1",     "name": "NAME_1",   "group": None,    "iso_from_id": True},
    "india_district": {"id": "dist_code", "name": "district", "group": "state", "iso_from_id": False},
}


def detect_schema(gdf: gpd.GeoDataFrame) -> dict:
    cols = set(gdf.columns)
    for tag, schema in _SCHEMAS.items():
        if schema["id"] in cols and schema["name"] in cols:
            schema = dict(schema)
            schema["_tag"] = tag
            return schema
    raise ValueError(f"Boundary file has unknown schema (cols={sorted(cols)}); "
                     f"add an entry to _SCHEMAS.")


def build_mask(gdf: gpd.GeoDataFrame, lat: xr.DataArray, lon: xr.DataArray,
               schema: Optional[dict] = None) -> xr.DataArray:
    s = schema or detect_schema(gdf)
    # overlap=False → each pixel maps to a single region (highest-numbered one
    # in case of overlap). India district polygons have minor topological
    # overlaps; for ICPAC adm1 there are none and the result is identical.
    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        numbers=list(range(len(gdf))),
        names=list(gdf[s["name"]]),
        abbrevs=list(gdf[s["id"]].astype(str)),
        name=s["_tag"],
        overlap=False,
    )
    return regions.mask(lon, lat)


def zonal_reduce(da: xr.DataArray, mask: xr.DataArray, lat: xr.DataArray,
                 n_regions: int, thresh: Optional[float] = None) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))
    w2d = weights.broadcast_like(da)
    src = (da >= thresh).astype("float32") if thresh is not None else da
    valid = (~da.isnull()).astype("float32")
    mask_vals = mask.values
    src_vals = src.values
    w_vals = w2d.values
    v_vals = valid.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        w = w_vals[sel] * v_vals[sel]
        den = w.sum()
        if den <= 0:
            continue
        num = float((src_vals[sel] * w).sum())
        out[r] = num / float(den)
    return out


def zonal_quantile(da: xr.DataArray, mask: xr.DataArray, n_regions: int,
                   q: float = 0.95) -> np.ndarray:
    mask_vals = mask.values
    vals = da.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        v = vals[sel][np.isfinite(vals[sel])]
        if v.size == 0:
            continue
        out[r] = float(np.quantile(v, q))
    return out


def zonal_max(da: xr.DataArray, mask: xr.DataArray, n_regions: int) -> np.ndarray:
    mask_vals = mask.values
    vals = da.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        v = vals[sel][np.isfinite(vals[sel])]
        if v.size == 0:
            continue
        out[r] = float(np.max(v))
    return out


def fill_small_boundaries(values: np.ndarray, da: xr.DataArray,
                          gdf: gpd.GeoDataFrame, thresh: Optional[float] = None) -> np.ndarray:
    out = values.copy()
    missing = np.where(np.isnan(out))[0]
    if len(missing) == 0:
        return out
    cent = gdf.iloc[missing].geometry.centroid
    src = (da >= thresh).astype("float32") if thresh is not None else da
    for i, pt in zip(missing, cent):
        try:
            val = float(src.sel(lat=pt.y, lon=pt.x, method="nearest").values)
        except Exception:
            val = np.nan
        out[i] = val
    return out


# ─── Satellite indicator extraction (S5P daily L3 → zonal aer/mois) ───

def _zonal_da(da: xr.DataArray, gdf: gpd.GeoDataFrame, n_regions: int) -> np.ndarray:
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims:
        da = da.rename({"longitude": "lon"})
    mask = build_mask(gdf, da["lat"], da["lon"])
    return zonal_reduce(da, mask, da["lat"], n_regions)


def fetch_satellite_evidence(
    date_str: str,
    gdf: gpd.GeoDataFrame,
    n_regions: int,
    bbox: list[float],
    yearly_aod_path: Optional[Path] = None,
    skip_satellite: bool = False,
    cache_dir: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """Per-zone arrays for `aer` (AOD), `mois` (TCWV), `no2`. NaN where missing."""
    aer = np.full(n_regions, np.nan)
    mois = np.full(n_regions, np.nan)
    no2 = np.full(n_regions, np.nan)

    if not skip_satellite:
        try:
            import importlib.util
            here = Path(__file__).resolve().parent
            sat_path = here.parent / "fog_satellite_indicators.py"
            if sat_path.exists():
                spec = importlib.util.spec_from_file_location("fog_sat", sat_path)
                fog_sat = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fog_sat)
                pseudo_region = {"lat": (bbox[1], bbox[3]),
                                 "lon": (bbox[0], bbox[2])}
                cache = cache_dir or (here / "satellite_cache")
                cache.mkdir(parents=True, exist_ok=True)

                for var_key, target in [("aod", "aer"), ("tcwv", "mois"), ("no2", "no2")]:
                    try:
                        ds = fog_sat.fetch_s5p_daily(var_key, date_str, pseudo_region, cache)
                    except Exception as e:
                        print(f"[prep]   S5P {var_key} fetch error: {e}")
                        ds = None
                    if ds is None:
                        continue
                    name_hints = {"aod": ("aerosol", "aot"),
                                  "tcwv": ("water", "vapour", "tcwv"),
                                  "no2":  ("no2", "nitrogen")}[var_key]
                    var = next((v for v in ds.data_vars
                                if any(h in v.lower() for h in name_hints)),
                               list(ds.data_vars)[0])
                    da = ds[var]
                    if "time" in da.dims:
                        da = da.isel(time=0)
                    z = _zonal_da(da, gdf, n_regions)
                    if target == "aer":
                        aer = z
                    elif target == "mois":
                        mois = z
                    else:
                        no2 = z
        except Exception as e:
            print(f"[prep]   satellite fetch failed (non-fatal): {e}")

    # Yearly AOD climatology fallback for missing zones
    if yearly_aod_path is not None and yearly_aod_path.exists():
        try:
            ds = xr.open_dataset(yearly_aod_path)
            da = ds["aerosol_optical_depth"]
            if "time" in da.dims:
                da = da.isel(time=0)
            aer_clim = _zonal_da(da, gdf, n_regions)
            mask_missing = ~np.isfinite(aer)
            aer[mask_missing] = aer_clim[mask_missing]
        except Exception as e:
            print(f"[prep]   yearly-AOD fallback failed: {e}")

    return {"aer": aer, "mois": mois, "no2": no2}


# ─── Main IFS aggregation ─────────────────────────────────────────────

def compute_ifs_evidence(
    date_str: str, gdf: gpd.GeoDataFrame, n_regions: int,
    icechunk_prefix: Optional[str] = None,
) -> dict:
    """Read IFS, compute per-pixel fog index per member, reduce to zones."""
    print(f"[prep] opening IFS Icechunk: s3://{S3_BUCKET}/{icechunk_prefix or S3_PREFIX}/")
    ds = open_ifs_icechunk(icechunk_prefix)

    init_dates = pd.to_datetime(ds.init_date.values)
    D = pd.Timestamp(date_str)
    if D not in init_dates:
        raise SystemExit(f"[prep] init_date {D.date()} not in IFS store "
                         f"(range {init_dates.min().date()}..{init_dates.max().date()})")

    sub = ds.sel(init_date=D)
    leads = sub.lead_time.values
    lt_to_idx = {int(h): i for i, h in enumerate(leads)}

    def _idx(h: int) -> Optional[int]:
        return lt_to_idx.get(int(h))

    fmt_idx = sorted({i for h in LEADS_ALL_FORMATION if (i := _idx(h)) is not None})
    near_idx = sorted({i for h in LEADS_NEAR if (i := _idx(h)) is not None})
    far_idx  = sorted({i for h in LEADS_FAR  if (i := _idx(h)) is not None})

    if not fmt_idx:
        raise SystemExit("[prep] no fog-formation lead-times found in store")

    print(f"[prep]   formation leads (h): {[int(leads[i]) for i in fmt_idx]} ({len(fmt_idx)} steps)")
    print(f"[prep]   wind near (h): {[int(leads[i]) for i in near_idx]}")
    print(f"[prep]   wind far  (h): {[int(leads[i]) for i in far_idx]}")

    needed = ["t2m", "d2m", "u10", "v10", "t_pl", "tp"]
    sub_fmt = sub[needed].isel(lead_time=fmt_idx)
    n_mem = sub_fmt.sizes["member"]
    n_lead = sub_fmt.sizes["lead_time"]
    n_lat = sub_fmt.sizes["lat"]
    n_lon = sub_fmt.sizes["lon"]
    print(f"[prep]   streaming formation slab per member: "
          f"members={n_mem}, leads={n_lead}, lat={n_lat}, lon={n_lon}")

    # Stream per member to keep peak memory at ~100 MiB instead of ~4.6 GiB.
    # Accumulators (numpy):
    #   sum_meets   : (lead, lat, lon)  count of members where F ≥ FOG_THR
    #   sum_ws_near : (lat, lon)        sum_m mean over near leads of WS
    #   sum_ws_far  : (lat, lon)        sum_m mean over far  leads of WS
    #   F_ensmax    : (lat, lon)        running max over (m, lead) of F
    sum_meets = np.zeros((n_lead, n_lat, n_lon), dtype=np.float64)
    sum_ws_near = np.zeros((n_lat, n_lon), dtype=np.float64) if near_idx else None
    sum_ws_far  = np.zeros((n_lat, n_lon), dtype=np.float64) if far_idx  else None
    F_ensmax    = np.zeros((n_lat, n_lon), dtype=np.float32)
    n_near = len(near_idx)
    n_far  = len(far_idx)

    sub_near = sub[["u10", "v10"]].isel(lead_time=near_idx) if near_idx else None
    sub_far  = sub[["u10", "v10"]].isel(lead_time=far_idx)  if far_idx  else None

    # Per-pixel valid-member counts for the wind-speed accumulators (so a
    # single NaN cell doesn't poison the whole zone).
    cnt_ws_near = np.zeros((n_lat, n_lon), dtype=np.int32) if near_idx else None
    cnt_ws_far  = np.zeros((n_lat, n_lon), dtype=np.int32) if far_idx  else None

    for m in range(n_mem):
        m_slab = sub_fmt.isel(member=m).load()
        F_m = fog_index_per_member(m_slab.t2m, m_slab.d2m,
                                   m_slab.u10, m_slab.v10,
                                   m_slab.t_pl, m_slab.tp).values  # (lead, lat, lon)
        # NaN-safe accumulators: np.fmax ignores NaN; ≥ comparison treats NaN
        # as False so sum_meets stays clean.
        sum_meets += (F_m >= FOG_INDEX_FOG_THR).astype(np.float64)
        F_max_lead = np.nanmax(F_m, axis=0)
        F_ensmax = np.fmax(F_ensmax, F_max_lead)

        if sub_near is not None:
            ws = ((sub_near.u10.isel(member=m).load().values ** 2 +
                   sub_near.v10.isel(member=m).load().values ** 2) ** 0.5)
            ws_m = np.nanmean(ws, axis=0)
            valid = np.isfinite(ws_m)
            sum_ws_near[valid] += ws_m[valid]
            cnt_ws_near[valid] += 1
        if sub_far is not None:
            ws = ((sub_far.u10.isel(member=m).load().values ** 2 +
                   sub_far.v10.isel(member=m).load().values ** 2) ** 0.5)
            ws_m = np.nanmean(ws, axis=0)
            valid = np.isfinite(ws_m)
            sum_ws_far[valid] += ws_m[valid]
            cnt_ws_far[valid] += 1

        if (m + 1) % 10 == 0 or (m + 1) == n_mem:
            print(f"[prep]     member {m+1:2d}/{n_mem} processed")

    # Per-pixel ensemble fog probability per lead, and overall max-over-leads
    P_pix_per_lead_arr = sum_meets / max(n_mem, 1)             # (lead, lat, lon)
    P_pix_arr = P_pix_per_lead_arr.max(axis=0).astype(np.float32)

    # Wrap as DataArrays so the existing zonal helpers (which expect lat coord
    # access on the input) keep working unchanged.
    coords = {"lat": sub.lat, "lon": sub.lon}
    P_pix = xr.DataArray(P_pix_arr, dims=("lat", "lon"), coords=coords)

    def _window_prob_arr(window_hours: list[int]) -> xr.DataArray:
        idxs = [_idx(h) for h in window_hours]
        idxs = [i for i in idxs if i is not None]
        local = [fmt_idx.index(i) for i in idxs if i in fmt_idx]
        if not local:
            return xr.DataArray(np.full_like(P_pix_arr, np.nan, dtype=np.float32),
                                dims=("lat", "lon"), coords=coords)
        win = P_pix_per_lead_arr[local, :, :].max(axis=0).astype(np.float32)
        return xr.DataArray(win, dims=("lat", "lon"), coords=coords)

    P24 = _window_prob_arr(LEADS_24H)
    P48 = _window_prob_arr(LEADS_48H)
    P72 = _window_prob_arr(LEADS_72H)

    F_ensmax_da = xr.DataArray(F_ensmax, dims=("lat", "lon"), coords=coords)

    # Wind-speed slope = mean(far) - mean(near), ensemble-mean, in m/s.
    # Use the per-pixel valid-member count so NaN cells don't bias the average.
    if sum_ws_near is not None and sum_ws_far is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            ws_near_em = np.where(cnt_ws_near > 0, sum_ws_near / cnt_ws_near, np.nan)
            ws_far_em  = np.where(cnt_ws_far  > 0, sum_ws_far  / cnt_ws_far,  np.nan)
        ws_slope_arr = (ws_far_em - ws_near_em).astype(np.float32)
    else:
        ws_slope_arr = np.zeros_like(P_pix_arr, dtype=np.float32)
    ws_slope = xr.DataArray(ws_slope_arr, dims=("lat", "lon"), coords=coords)
    F_ensmax = F_ensmax_da

    print(f"[prep]   building adm-1 mask on IFS grid ...")
    mask = build_mask(gdf, P_pix.lat, P_pix.lon)

    fog_prob_zone     = zonal_reduce(P_pix, mask, P_pix.lat, n_regions)
    fog_prob_24_zone  = zonal_reduce(P24,   mask, P_pix.lat, n_regions)
    fog_prob_48_zone  = zonal_reduce(P48,   mask, P_pix.lat, n_regions)
    fog_prob_72_zone  = zonal_reduce(P72,   mask, P_pix.lat, n_regions)
    spatial_cov_zone  = zonal_reduce(P_pix, mask, P_pix.lat, n_regions, thresh=0.5)
    tail_p95_zone     = zonal_quantile(F_ensmax, mask, n_regions, q=0.95)
    tail_peak_zone    = zonal_max(F_ensmax, mask, n_regions)
    stag_slope_zone   = zonal_reduce(ws_slope, mask, P_pix.lat, n_regions)

    fog_prob_zone    = fill_small_boundaries(fog_prob_zone,    P_pix, gdf)
    fog_prob_24_zone = fill_small_boundaries(fog_prob_24_zone, P24,   gdf)
    fog_prob_48_zone = fill_small_boundaries(fog_prob_48_zone, P48,   gdf)
    fog_prob_72_zone = fill_small_boundaries(fog_prob_72_zone, P72,   gdf)
    spatial_cov_zone = fill_small_boundaries(spatial_cov_zone, P_pix, gdf, thresh=0.5)
    tail_p95_zone    = fill_small_boundaries(tail_p95_zone,    F_ensmax, gdf)
    tail_peak_zone   = fill_small_boundaries(tail_peak_zone,   F_ensmax, gdf)
    stag_slope_zone  = fill_small_boundaries(stag_slope_zone,  ws_slope, gdf)

    return {
        "fog_prob": fog_prob_zone,
        "fog_prob_24h": fog_prob_24_zone,
        "fog_prob_48h": fog_prob_48_zone,
        "fog_prob_72h": fog_prob_72_zone,
        "spatial_coverage": spatial_cov_zone,
        "tail_p95": tail_p95_zone,
        "tail_peak": tail_peak_zone,
        "stag_slope_ms": stag_slope_zone,
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-day zone-level fog/AQ evidence generator for fog_bn_ibf_v1.jl"
    )
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD (init_date in IFS store)")
    ap.add_argument("--out", required=True, help="Output CSV path (one row per boundary)")
    ap.add_argument("--boundaries",
                    default=str(Path(__file__).resolve().parent / "icpac_adm1v3.geojson"),
                    help="GeoJSON of analysis zones. Schema is auto-detected: ICPAC adm1 "
                         "(GID_1/NAME_1) or India districts (dist_code/district/state). "
                         "Add new schemas to _SCHEMAS in this script.")
    # Back-compat alias
    ap.add_argument("--adm1", dest="boundaries", help=argparse.SUPPRESS)
    ap.add_argument("--ifs-prefix", default=None,
                    help="Override S3_PREFIX of the IFS Icechunk store")
    ap.add_argument("--yearly-aod",
                    default=str(Path(__file__).resolve().parent.parent
                                / "271d5630-56e5-4d0e-85c3-5bef29b0e4e5"),
                    help="Yearly S5P L3 AOD NetCDF for climatology fallback")
    ap.add_argument("--skip-satellite", action="store_true",
                    help="Skip S5P-PAL daily L3 fetch; use yearly AOD only")
    ap.add_argument("--soft-evidence", action="store_true",
                    help="Emit Gaussian-soft-binned probability columns "
                         "{aer,mois,fog,stag,tail}_p{1..K}")
    args = ap.parse_args()

    D = pd.Timestamp(args.date)
    print(f"[prep] D={D.date()}  IFS store=s3://{S3_BUCKET}/{args.ifs_prefix or S3_PREFIX}/")

    gdf = gpd.read_file(args.boundaries)
    schema = detect_schema(gdf)
    raw_n = len(gdf)
    # Drop null geometries; fix invalid ones with buffer(0). The India districts
    # GeoJSON has 6 null geoms + 13 invalid (self-intersecting) features that
    # regionmask refuses; we silently repair so every kept row makes it to the
    # output CSV.
    null_mask = gdf.geometry.isna() | gdf.geometry.is_empty
    if null_mask.any():
        print(f"[prep]   dropping {int(null_mask.sum())} boundaries with null/empty geometry")
        gdf = gdf.loc[~null_mask].copy()
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        print(f"[prep]   repairing {int(invalid_mask.sum())} invalid geometries (buffer(0))")
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
    gdf = gdf.reset_index(drop=True)
    n_adm = len(gdf)
    print(f"[prep] boundaries: {n_adm} of {raw_n}  (schema={schema['_tag']}, "
          f"id={schema['id']}, name={schema['name']})")

    ifs = compute_ifs_evidence(args.date, gdf, n_adm, icechunk_prefix=args.ifs_prefix)

    bbox = [float(gdf.total_bounds[0]), float(gdf.total_bounds[1]),
            float(gdf.total_bounds[2]), float(gdf.total_bounds[3])]
    sat = fetch_satellite_evidence(
        date_str=args.date, gdf=gdf, n_regions=n_adm, bbox=bbox,
        yearly_aod_path=Path(args.yearly_aod) if args.yearly_aod else None,
        skip_satellite=args.skip_satellite,
    )

    aer_state = np.array([classify_aer(v) for v in sat["aer"]])
    mois_state = np.array([classify_mois(v) for v in sat["mois"]])
    stag_state = np.array([classify_stag(v) for v in ifs["stag_slope_ms"]])

    # `country` column holds the higher-level grouping the BN reports use:
    # ISO country for ICPAC adm1 (derived from GID_1 prefix), state name
    # for India districts. Whatever the schema, downstream code only sees a
    # `country` string column and treats it as a label.
    if schema["iso_from_id"]:
        country = (gdf[schema["id"]].astype(str).str.split(".").str[0]
                   .map(ISO_TO_COUNTRY).fillna("Unknown"))
    elif schema["group"] is not None:
        country = gdf[schema["group"]].astype(str).fillna("Unknown")
    else:
        country = pd.Series(["Unknown"] * n_adm, index=gdf.index)

    df = pd.DataFrame({
        "id": gdf[schema["id"]].astype(str),
        "name": gdf[schema["name"]].astype(str),
        "country": country,
        "antecedent_aerosol_aod":     np.round(sat["aer"], 4),
        "antecedent_aerosol_state":   aer_state,
        "antecedent_moisture_kgm2":   np.round(sat["mois"], 2),
        "antecedent_moisture_state":  mois_state,
        "antecedent_no2_molec_cm2":   np.round(sat["no2"], 4),
        "ifs_fog_prob":            np.round(ifs["fog_prob"], 4),
        "fog_prob_24h":            np.round(ifs["fog_prob_24h"], 4),
        "fog_prob_48h":            np.round(ifs["fog_prob_48h"], 4),
        "fog_prob_72h":            np.round(ifs["fog_prob_72h"], 4),
        "spatial_coverage":        np.round(ifs["spatial_coverage"], 4),
        "stagnation_trend":        stag_state,
        "stagnation_slope_ms":     np.round(ifs["stag_slope_ms"], 3),
        "extreme_fog_tail_p95":    np.round(ifs["tail_p95"], 4),
        "ens_max_fog_index_peak":  np.round(ifs["tail_peak"], 4),
        "target_date":             str(D.date()),
    })

    if args.soft_evidence:
        add_soft_columns(df,
                         aer=sat["aer"], mois=sat["mois"], fog=ifs["fog_prob"],
                         stag_slope=ifs["stag_slope_ms"], tail=ifs["tail_p95"])
        print(f"[prep] soft-evidence columns added (18 cols)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    fog_mean = float(np.nanmean(ifs["fog_prob"]))
    aer_mean = float(np.nanmean(sat["aer"]))
    print(f"[prep] wrote {out}  rows={len(df)}  cols={len(df.columns)}  "
          f"fog_prob_mean={fog_mean:.3f}  aod_mean={aer_mean:.3f}")


if __name__ == "__main__":
    main()
