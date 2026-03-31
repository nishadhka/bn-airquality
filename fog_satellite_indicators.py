#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "xarray",
#     "netcdf4",
#     "h5py",
#     "pystac",
#     "pystac-client",
#     "requests",
#     "matplotlib",
#     "cartopy",
# ]
# ///
"""
Fog Satellite Indicators: Merge Sentinel-5P L3 and Sentinel-3 SLSTR for fog precondition assessment.

Sentinel-5P (antecedent atmospheric composition):
  - AOD at 354/388nm → aerosol loading for hygroscopic fog CCN
  - TCWV → column moisture availability
  - NO2 tropospheric → pollution intensity / boundary layer depth proxy
  (Accessed via S5P-PAL STAC API: https://data-portal.s5p-pal.com/api/s5p-l3)

Sentinel-3 SLSTR (surface/thermal conditions):
  - LST → land surface temperature for radiative cooling assessment
  - Cloud mask → fog/low-cloud detection via 3.7-11μm BTD
  (Accessed via EOPF STAC API: https://stac.core.eopf.eodc.eu)

Existing S5P L3 AOD yearly composite (HARP-processed NetCDF) is used as the
reference grid and climatological baseline.

Paper basis:
  - Parde et al. (2022): "chemical composition and hygroscopicity of aerosols
    determine activation properties" — S5P fills this gap
  - Boneh et al. (2015): "moisture availability upstream is a good indicator" — S5P TCWV

Usage:
    uv run fog_satellite_indicators.py
    uv run fog_satellite_indicators.py --date 2025-12-15
    uv run fog_satellite_indicators.py --date 2025-12-15 --region india-igp
    uv run fog_satellite_indicators.py --skip-s3   # skip Sentinel-3 (EOPF may be slow)
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import xarray as xr

warnings.filterwarnings("ignore")

# ── Region definitions ────────────────────────────────────────────────
REGIONS = {
    "india": {"lat": (6, 37), "lon": (68, 98), "label": "India"},
    "india-igp": {"lat": (22, 32), "lon": (72, 90), "label": "Indo-Gangetic Plain"},
    "india-delhi": {"lat": (27, 30), "lon": (76, 78), "label": "Delhi NCR"},
}

# ── S5P-PAL STAC API ─────────────────────────────────────────────────
S5P_L3_STAC = "https://data-portal.s5p-pal.com/api/s5p-l3"
S5P_L2_STAC = "https://data-portal.s5p-pal.com/api/s5p-l2"

# S5P-PAL L3 collection IDs (from STAC catalog)
S5P_COLLECTIONS = {
    "aod": "L3__AER_OT",   # Aerosol Optical Thickness
    "tcwv": "L3__TCWV__",  # Total Column Water Vapour
    "no2": "L3__NO2___",   # Nitrogen Dioxide
}

# ── EOPF STAC for Sentinel-3 ─────────────────────────────────────────
EOPF_STAC = "https://stac.core.eopf.eodc.eu"
S3_COLLECTION = "sentinel-3-slstr-l2-lst"

# ── Local S5P L3 reference file ──────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
S5P_AOD_YEARLY = SCRIPT_DIR / "271d5630-56e5-4d0e-85c3-5bef29b0e4e5"

# ── Output ────────────────────────────────────────────────────────────
OUTPUT_DIR = SCRIPT_DIR / "fog_indicators_output"


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Load existing S5P L3 AOD (HARP-processed yearly composite)
# ═══════════════════════════════════════════════════════════════════════

def load_s5p_aod_yearly(filepath: Path, region: dict) -> Optional[xr.Dataset]:
    """Load the existing S5P L3 AOD yearly NetCDF and subset to region.

    This file was produced by HARP:
        harpmerge -a 'wavelength==494; aerosol_optical_depth_validity>50;
        bin_spatial(8193, -90, 0.02197265625, 16385, -180, 0.02197265625)'

    Grid: 8192 x 16384, ~0.022° resolution, global.
    Variables: aerosol_optical_depth, aerosol_optical_depth_uncertainty
    Period: 2025-01-01 to 2026-01-01 (yearly composite)
    """
    if not filepath.exists():
        print(f"  Warning: S5P AOD yearly file not found: {filepath}")
        return None

    print(f"  Loading S5P L3 AOD yearly: {filepath.name}")
    ds = xr.open_dataset(filepath)

    # Subset to region
    lat_min, lat_max = region["lat"]
    lon_min, lon_max = region["lon"]
    ds_sub = ds.sel(
        latitude=slice(lat_max, lat_min),   # HARP grid is N→S
        longitude=slice(lon_min, lon_max),
    )

    n_valid = int(np.isfinite(ds_sub["aerosol_optical_depth"]).sum())
    print(f"  AOD yearly grid: {ds_sub.sizes} | valid pixels: {n_valid:,}")

    aod = ds_sub["aerosol_optical_depth"].isel(time=0)
    print(f"  AOD stats: min={float(aod.min()):.3f}, "
          f"mean={float(aod.mean()):.3f}, max={float(aod.max()):.3f}")

    return ds_sub


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Query S5P-PAL STAC for daily L3 products
# ═══════════════════════════════════════════════════════════════════════

def search_s5p_stac(
    collection: str,
    date_str: str,
    bbox: list[float],
    catalog_url: str = S5P_L3_STAC,
    max_items: int = 10,
) -> list[dict]:
    """Search S5P-PAL STAC API for items matching date and bbox.

    S5P-PAL implements STAC Item Search with sort and filter fragments.
    API docs: https://data-portal.s5p-pal.com/apidoc

    Args:
        collection: e.g. "L3__AER_OT", "L3__TCWV__", "L3__NO2___"
        date_str: "YYYY-MM-DD" — searches that date
        bbox: [lon_min, lat_min, lon_max, lat_max]
        catalog_url: STAC catalog root
        max_items: max results

    Returns:
        List of STAC item dicts with asset download URLs.
    """
    # Build search endpoint from catalog
    search_url = f"{catalog_url}/search"

    # Date range for the target day
    dt_start = f"{date_str}T00:00:00Z"
    dt_end = f"{date_str}T23:59:59Z"

    payload = {
        "collections": [collection],
        "datetime": f"{dt_start}/{dt_end}",
        "bbox": bbox,
        "limit": max_items,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }

    print(f"  STAC search: {collection} on {date_str} ...")

    try:
        resp = requests.post(search_url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            print(f"  Found {len(features)} items")
            return features
        elif resp.status_code == 405:
            # Try GET-based search
            params = {
                "collections": collection,
                "datetime": f"{dt_start}/{dt_end}",
                "bbox": ",".join(str(b) for b in bbox),
                "limit": max_items,
            }
            resp = requests.get(search_url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                features = data.get("features", [])
                print(f"  Found {len(features)} items (GET)")
                return features
            else:
                print(f"  STAC GET also failed: {resp.status_code}")
                return []
        else:
            print(f"  STAC search failed: {resp.status_code} — {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"  STAC search error: {e}")
        return []


def download_s5p_item(item: dict, output_dir: Path) -> Optional[Path]:
    """Download the NetCDF asset from a STAC item.

    S5P-PAL items have assets with 'data' or 'download' roles.
    """
    assets = item.get("assets", {})

    # Find the data asset (NetCDF)
    download_url = None
    for key, asset in assets.items():
        href = asset.get("href", "")
        media = asset.get("type", "")
        if href.endswith(".nc") or "netcdf" in media.lower():
            download_url = href
            break
        if "download" in key.lower() or "data" in key.lower():
            download_url = href
            break

    if not download_url:
        # Use the first asset href
        if assets:
            download_url = list(assets.values())[0].get("href")

    if not download_url:
        print(f"  No download URL found in item {item.get('id', '?')}")
        return None

    filename = Path(download_url).name
    if not filename.endswith(".nc"):
        filename = f"{item.get('id', 'unknown')}.nc"

    local_path = output_dir / filename
    if local_path.exists():
        print(f"  Already downloaded: {filename}")
        return local_path

    print(f"  Downloading: {filename} ...")
    try:
        resp = requests.get(download_url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = local_path.stat().st_size / 1e6
        print(f"  Downloaded: {size_mb:.1f} MB")
        return local_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def fetch_s5p_daily(
    var_key: str,
    date_str: str,
    region: dict,
    output_dir: Path,
) -> Optional[xr.Dataset]:
    """Fetch a daily S5P L3 product from STAC and load as xarray Dataset.

    Args:
        var_key: "aod", "tcwv", or "no2"
        date_str: "YYYY-MM-DD"
        region: dict with lat/lon bounds
        output_dir: where to cache downloads

    Returns:
        xarray Dataset subset to region, or None if unavailable.
    """
    collection = S5P_COLLECTIONS[var_key]
    lat_min, lat_max = region["lat"]
    lon_min, lon_max = region["lon"]
    bbox = [lon_min, lat_min, lon_max, lat_max]

    items = search_s5p_stac(collection, date_str, bbox)

    if not items:
        print(f"  No {var_key} items found for {date_str}")
        return None

    # Download first (most recent) item
    item = items[0]
    local_path = download_s5p_item(item, output_dir)
    if local_path is None:
        return None

    try:
        ds = xr.open_dataset(local_path)
        # Try to subset (HARP L3 uses latitude/longitude dims)
        if "latitude" in ds.dims and "longitude" in ds.dims:
            lats = ds["latitude"].values
            if lats[0] > lats[-1]:  # N→S
                ds = ds.sel(
                    latitude=slice(lat_max, lat_min),
                    longitude=slice(lon_min, lon_max),
                )
            else:  # S→N
                ds = ds.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max),
                )
        return ds
    except Exception as e:
        print(f"  Failed to open {local_path}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Query EOPF STAC for Sentinel-3 SLSTR LST
# ═══════════════════════════════════════════════════════════════════════

def fetch_s3_slstr_lst(
    date_str: str,
    region: dict,
    output_dir: Path,
) -> Optional[xr.Dataset]:
    """Fetch Sentinel-3 SLSTR L2 LST via EOPF STAC Zarr API.

    EOPF STAC: https://stac.core.eopf.eodc.eu
    Collection: sentinel-3-slstr-l2-lst
    Zarr structure: /measurements/orphan/lst (ungridded swath data)

    LST provides surface temperature for radiative cooling assessment.
    Cloud mask provides fog/low-cloud detection.

    Args:
        date_str: "YYYY-MM-DD"
        region: dict with lat/lon bounds
        output_dir: cache directory

    Returns:
        xarray Dataset with LST and cloud info, or None.
    """
    from pystac_client import Client

    lat_min, lat_max = region["lat"]
    lon_min, lon_max = region["lon"]
    bbox = [lon_min, lat_min, lon_max, lat_max]

    print(f"  Searching EOPF STAC for S3 SLSTR LST on {date_str} ...")

    try:
        catalog = Client.open(EOPF_STAC)
        results = catalog.search(
            collections=[S3_COLLECTION],
            bbox=bbox,
            datetime=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z",
            max_items=5,
        )
        items = list(results.items())
        print(f"  Found {len(items)} S3 SLSTR granules")

        if not items:
            return None

        # Load the first granule's zarr
        item = items[0]
        zarr_asset = None
        for key, asset in item.assets.items():
            if "zarr" in key.lower() or "zarr" in asset.media_type.lower() \
                    if hasattr(asset, "media_type") and asset.media_type else False:
                zarr_asset = asset
                break
            if asset.href and ".zarr" in asset.href:
                zarr_asset = asset
                break

        if zarr_asset is None:
            # Try the first asset
            zarr_asset = list(item.assets.values())[0] if item.assets else None

        if zarr_asset is None:
            print("  No zarr asset found in S3 SLSTR item")
            return None

        zarr_url = zarr_asset.href
        print(f"  Opening zarr: {zarr_url[:100]}...")

        # Open the measurements group for LST
        # EOPF zarr structure: /measurements/orphan/lst
        try:
            ds = xr.open_datatree(zarr_url, engine="zarr")
            # Navigate to LST measurement
            lst_group = ds["measurements/orphan"]
            lst_ds = lst_group.to_dataset()

            # Subset to region using lat/lon
            if "latitude" in lst_ds:
                lat_mask = (lst_ds["latitude"] >= lat_min) & (lst_ds["latitude"] <= lat_max)
                lon_mask = (lst_ds["longitude"] >= lon_min) & (lst_ds["longitude"] <= lon_max)
                region_mask = lat_mask & lon_mask
                lst_ds = lst_ds.where(region_mask, drop=True)

            print(f"  S3 LST loaded: {lst_ds.sizes}")
            if "lst" in lst_ds:
                lst = lst_ds["lst"]
                valid = lst.values[np.isfinite(lst.values)]
                if len(valid) > 0:
                    print(f"  LST stats: min={valid.min():.1f}K, "
                          f"mean={valid.mean():.1f}K, max={valid.max():.1f}K")
            return lst_ds

        except Exception as e:
            print(f"  Failed to open zarr datatree: {e}")
            # Fallback: try direct xr.open_zarr
            try:
                ds = xr.open_zarr(zarr_url)
                print(f"  Fallback zarr loaded: {list(ds.data_vars)}")
                return ds
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
                return None

    except Exception as e:
        print(f"  EOPF STAC error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Compute Fog Indicators
# ═══════════════════════════════════════════════════════════════════════

def compute_fog_indicators(
    aod_yearly: Optional[xr.Dataset],
    aod_daily: Optional[xr.Dataset],
    tcwv_daily: Optional[xr.Dataset],
    no2_daily: Optional[xr.Dataset],
    s3_lst: Optional[xr.Dataset],
    region: dict,
) -> dict:
    """Compute fog precondition indicators from satellite data.

    Indicators follow the Bayesian Network design from the plan:

    1. Pollution Index (BN parent node):
       - AOD > 0.5 → high aerosol loading (hygroscopic fog CCN)
         [Parde §3.2(d): "chemical composition and hygroscopicity"]
       - NO2 > 10e15 molec/cm² → polluted boundary layer
       - Combined: low / moderate / high

    2. Moisture Index:
       - TCWV > 30 kg/m² → moist column
         [Boneh §3a.1: "moisture availability upstream"]

    3. AOD Anomaly:
       - Daily AOD vs yearly mean → enhanced aerosol day
       - Positive anomaly → fog more likely

    4. Surface Cooling Potential (from S3 LST):
       - Afternoon LST - air temp estimate → radiative cooling capacity
         [Boneh §3a.3: "low-level inversion"]

    Returns dict of indicator arrays and summary statistics.
    """
    indicators = {}

    # ── AOD-based Pollution Index ──
    if aod_daily is not None:
        aod_var = None
        for v in aod_daily.data_vars:
            if "aerosol" in v.lower() or "aod" in v.lower() or "aot" in v.lower():
                aod_var = v
                break
        if aod_var is None:
            aod_var = list(aod_daily.data_vars)[0]

        aod = aod_daily[aod_var]
        if aod.ndim > 2:
            aod = aod.isel(time=0) if "time" in aod.dims else aod.squeeze()

        valid_aod = aod.values[np.isfinite(aod.values)]
        if len(valid_aod) > 0:
            indicators["aod_daily_mean"] = float(np.nanmean(valid_aod))
            indicators["aod_daily_max"] = float(np.nanmax(valid_aod))
            indicators["aod_daily_p90"] = float(np.nanpercentile(valid_aod, 90))

            # Pollution states (for BN node)
            frac_high = float(np.mean(valid_aod > 0.8))
            frac_mod = float(np.mean((valid_aod > 0.3) & (valid_aod <= 0.8)))
            frac_low = float(np.mean(valid_aod <= 0.3))
            indicators["aod_frac_high"] = frac_high
            indicators["aod_frac_moderate"] = frac_mod
            indicators["aod_frac_low"] = frac_low

            # Pollution Index state
            if frac_high > 0.3:
                indicators["pollution_index_state"] = "high"
            elif frac_mod > 0.4:
                indicators["pollution_index_state"] = "moderate"
            else:
                indicators["pollution_index_state"] = "low"

            indicators["_aod_data"] = aod
            print(f"  AOD daily: mean={indicators['aod_daily_mean']:.3f}, "
                  f"p90={indicators['aod_daily_p90']:.3f}, "
                  f"pollution={indicators['pollution_index_state']}")

    # ── AOD Anomaly vs yearly climatology ──
    if aod_yearly is not None and "_aod_data" in indicators:
        aod_clim = aod_yearly["aerosol_optical_depth"].isel(time=0)
        clim_mean = float(aod_clim.mean())
        daily_mean = indicators["aod_daily_mean"]
        indicators["aod_anomaly"] = daily_mean - clim_mean
        indicators["aod_clim_mean"] = clim_mean
        print(f"  AOD anomaly: {indicators['aod_anomaly']:+.3f} "
              f"(daily {daily_mean:.3f} vs clim {clim_mean:.3f})")

    # ── TCWV Moisture Index ──
    if tcwv_daily is not None:
        tcwv_var = None
        for v in tcwv_daily.data_vars:
            if "water" in v.lower() or "tcwv" in v.lower() or "vapour" in v.lower():
                tcwv_var = v
                break
        if tcwv_var is None:
            tcwv_var = list(tcwv_daily.data_vars)[0]

        tcwv = tcwv_daily[tcwv_var]
        if tcwv.ndim > 2:
            tcwv = tcwv.isel(time=0) if "time" in tcwv.dims else tcwv.squeeze()

        valid_tcwv = tcwv.values[np.isfinite(tcwv.values)]
        if len(valid_tcwv) > 0:
            indicators["tcwv_mean"] = float(np.nanmean(valid_tcwv))
            indicators["tcwv_max"] = float(np.nanmax(valid_tcwv))

            # Moisture states for BN
            frac_moist = float(np.mean(valid_tcwv > 30))
            frac_dry = float(np.mean(valid_tcwv < 15))
            indicators["tcwv_frac_moist"] = frac_moist
            indicators["tcwv_frac_dry"] = frac_dry

            if frac_moist > 0.5:
                indicators["moisture_state"] = "moist"
            elif frac_dry > 0.5:
                indicators["moisture_state"] = "dry"
            else:
                indicators["moisture_state"] = "moderate"

            print(f"  TCWV: mean={indicators['tcwv_mean']:.1f} kg/m², "
                  f"moisture={indicators['moisture_state']}")

    # ── NO2 Pollution Intensity ──
    if no2_daily is not None:
        no2_var = None
        for v in no2_daily.data_vars:
            if "no2" in v.lower() or "nitrogen" in v.lower():
                no2_var = v
                break
        if no2_var is None:
            no2_var = list(no2_daily.data_vars)[0]

        no2 = no2_daily[no2_var]
        if no2.ndim > 2:
            no2 = no2.isel(time=0) if "time" in no2.dims else no2.squeeze()

        valid_no2 = no2.values[np.isfinite(no2.values)]
        if len(valid_no2) > 0:
            indicators["no2_mean"] = float(np.nanmean(valid_no2))
            indicators["no2_max"] = float(np.nanmax(valid_no2))
            indicators["no2_p90"] = float(np.nanpercentile(valid_no2, 90))

            # NO2 as BL depth proxy: high NO2 → shallow BL → fog favorable
            # Typical: > 5e15 molec/cm² is polluted
            indicators["no2_polluted_frac"] = float(np.mean(valid_no2 > 5e15))
            print(f"  NO2: mean={indicators['no2_mean']:.2e}, "
                  f"polluted_frac={indicators['no2_polluted_frac']:.2f}")

    # ── S3 LST Surface Cooling ──
    if s3_lst is not None and "lst" in s3_lst:
        lst = s3_lst["lst"]
        valid_lst = lst.values[np.isfinite(lst.values)]
        if len(valid_lst) > 0:
            indicators["lst_mean_K"] = float(np.nanmean(valid_lst))
            indicators["lst_min_K"] = float(np.nanmin(valid_lst))
            indicators["lst_max_K"] = float(np.nanmax(valid_lst))
            # Large LST spread → strong radiative cooling potential
            indicators["lst_range_K"] = indicators["lst_max_K"] - indicators["lst_min_K"]
            print(f"  S3 LST: mean={indicators['lst_mean_K']:.1f}K, "
                  f"range={indicators['lst_range_K']:.1f}K")

    # ── Combined Fog Precondition Score ──
    # Simple heuristic combining available indicators
    score_components = []
    if "pollution_index_state" in indicators:
        score_components.append(
            {"high": 0.9, "moderate": 0.5, "low": 0.1}[indicators["pollution_index_state"]]
        )
    if "moisture_state" in indicators:
        score_components.append(
            {"moist": 0.8, "moderate": 0.5, "dry": 0.1}[indicators["moisture_state"]]
        )
    if "aod_anomaly" in indicators:
        # Positive anomaly → more aerosol → more fog
        anom = indicators["aod_anomaly"]
        score_components.append(min(max(0.5 + anom * 2, 0), 1))

    if score_components:
        indicators["fog_precondition_score"] = float(np.mean(score_components))
        print(f"\n  *** Fog Precondition Score: {indicators['fog_precondition_score']:.2f} "
              f"(0=unlikely, 1=very favorable) ***")

    # Remove internal data arrays from summary
    indicators.pop("_aod_data", None)

    return indicators


# ═══════════════════════════════════════════════════════════════════════
# PART 5: Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_fog_indicators(
    aod_yearly: Optional[xr.Dataset],
    aod_daily: Optional[xr.Dataset],
    indicators: dict,
    region: dict,
    date_str: str,
    output_dir: Path,
):
    """Plot AOD maps and fog indicator summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    lat_min, lat_max = region["lat"]
    lon_min, lon_max = region["lon"]

    # Panel 1: Yearly AOD climatology
    if aod_yearly is not None:
        aod_clim = aod_yearly["aerosol_optical_depth"].isel(time=0)
        im1 = axes[0].pcolormesh(
            aod_clim.longitude, aod_clim.latitude, aod_clim.values,
            cmap="YlOrRd", vmin=0, vmax=1, shading="auto"
        )
        axes[0].set_title("S5P L3 AOD Yearly Climatology (2025)", fontweight="bold")
        plt.colorbar(im1, ax=axes[0], label="AOD 494nm", shrink=0.8)
    else:
        axes[0].text(0.5, 0.5, "No yearly AOD data", transform=axes[0].transAxes,
                     ha="center", va="center")
        axes[0].set_title("S5P L3 AOD Yearly (unavailable)")

    # Panel 2: Fog indicator summary
    ax2 = axes[1]
    ax2.axis("off")
    summary_lines = [
        f"Date: {date_str}",
        f"Region: {region['label']}",
        "",
        "── Fog Precondition Indicators ──",
        "",
    ]
    for k, v in sorted(indicators.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            summary_lines.append(f"  {k}: {v:.4f}")
        else:
            summary_lines.append(f"  {k}: {v}")

    ax2.text(0.05, 0.95, "\n".join(summary_lines), transform=ax2.transAxes,
             fontsize=10, fontfamily="monospace", verticalalignment="top")
    ax2.set_title(f"Fog Indicators — {date_str}", fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / f"fog_indicators_{date_str}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute fog precondition indicators from S5P + S3 satellite data"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: yesterday)"
    )
    parser.add_argument(
        "--region", type=str, default="india-igp",
        choices=list(REGIONS.keys()),
        help="Region of interest (default: india-igp)"
    )
    parser.add_argument(
        "--skip-s3", action="store_true",
        help="Skip Sentinel-3 SLSTR fetch (EOPF can be slow)"
    )
    parser.add_argument(
        "--skip-stac", action="store_true",
        help="Skip all STAC queries (use only local yearly AOD)"
    )
    parser.add_argument(
        "--aod-file", type=str, default=str(S5P_AOD_YEARLY),
        help="Path to S5P L3 AOD yearly NetCDF file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory"
    )
    args = parser.parse_args()

    # Resolve date
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = (datetime.utcnow() - timedelta(days=1)).date()
    date_str = target_date.strftime("%Y-%m-%d")

    region = REGIONS[args.region]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FOGWATCH Satellite Indicators")
    print("=" * 70)
    print(f"Date:   {date_str}")
    print(f"Region: {region['label']} (lat {region['lat']}, lon {region['lon']})")
    print(f"Output: {output_dir}")
    print()

    # ── Step 1: Load yearly AOD climatology ──
    print("[Step 1] Loading S5P L3 AOD yearly climatology")
    print("-" * 50)
    aod_yearly = load_s5p_aod_yearly(Path(args.aod_file), region)

    # ── Step 2: Fetch daily S5P L3 from STAC ──
    aod_daily = None
    tcwv_daily = None
    no2_daily = None

    if not args.skip_stac:
        print()
        print("[Step 2] Fetching daily S5P L3 products from STAC")
        print("-" * 50)

        for var_key in ["aod", "tcwv", "no2"]:
            print(f"\n  --- {var_key.upper()} ---")
            ds = fetch_s5p_daily(var_key, date_str, region, output_dir)
            if var_key == "aod":
                aod_daily = ds
            elif var_key == "tcwv":
                tcwv_daily = ds
            elif var_key == "no2":
                no2_daily = ds
    else:
        print("\n[Step 2] Skipped (--skip-stac)")

    # ── Step 3: Fetch Sentinel-3 SLSTR LST ──
    s3_lst = None
    if not args.skip_s3 and not args.skip_stac:
        print()
        print("[Step 3] Fetching Sentinel-3 SLSTR LST from EOPF STAC")
        print("-" * 50)
        s3_lst = fetch_s3_slstr_lst(date_str, region, output_dir)
    else:
        print(f"\n[Step 3] Skipped ({'--skip-s3' if args.skip_s3 else '--skip-stac'})")

    # ── Step 4: Compute fog indicators ──
    print()
    print("[Step 4] Computing fog precondition indicators")
    print("-" * 50)
    indicators = compute_fog_indicators(
        aod_yearly, aod_daily, tcwv_daily, no2_daily, s3_lst, region
    )

    # ── Step 5: Save indicators JSON ──
    indicators_file = output_dir / f"fog_indicators_{date_str}.json"
    with open(indicators_file, "w") as f:
        json.dump(indicators, f, indent=2, default=str)
    print(f"\n  Indicators saved: {indicators_file}")

    # ── Step 6: Plot ──
    print()
    print("[Step 6] Plotting")
    print("-" * 50)
    plot_fog_indicators(aod_yearly, aod_daily, indicators, region, date_str, output_dir)

    # ── Summary ──
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in sorted(indicators.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            print(f"  {k:<30s} {v:.4f}")
        else:
            print(f"  {k:<30s} {v}")
    print()
    print("Next: Feed these indicators as evidence into the Bayesian Network")
    print("      alongside ECMWF IFS ensemble variables (via GIK streaming)")


if __name__ == "__main__":
    main()
