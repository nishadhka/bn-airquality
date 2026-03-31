# bn-aq: Bayesian Network for Air Quality & Fog Risk

District-level fog risk monitoring system for India, fusing **ECMWF IFS ensemble forecasts** (via [GIK streaming](https://github.com/ipcac-igad/grib-index-kerchunk)) with **Sentinel-5P** atmospheric composition and **Sentinel-3** surface observations through a Bayesian Decision Network.

## Quick Start

```bash
# 1. Run fog satellite indicators (S5P + S3 merge) for a winter date
uv run fog_satellite_indicators.py --date 2025-12-15 --region india-igp

# 2. Local-only mode (skip STAC queries, use yearly AOD climatology)
uv run fog_satellite_indicators.py --skip-stac --region india-igp --date 2025-12-15

# 3. Skip Sentinel-3 only (EOPF can be slow)
uv run fog_satellite_indicators.py --skip-s3 --date 2025-12-15
```

## Files

| File | Description |
|---|---|
| `fog_satellite_indicators.py` | PEP 723 `uv run` script — fetches S5P L3 (AOD, TCWV, NO2) from STAC + S3 SLSTR LST from EOPF Zarr, computes fog precondition indicators |


## How It Works

### Step 1 — Satellite Observations (antecedent conditions)

`fog_satellite_indicators.py` gathers the pollution/moisture state from the afternoon before fog night:

```
Sentinel-5P (S5P-PAL STAC)          Sentinel-3 (EOPF Zarr)
├── AOD 354/388nm → aerosol load    ├── LST → surface cooling
├── TCWV → column moisture           └── Cloud mask → fog detection
├── NO2 → pollution / BL depth
└── UVAI → aerosol type
         │                                    │
         └──────────┬─────────────────────────┘
                    ▼
         Fog Precondition Indicators
         (JSON + plot per district)
```

### Step 2 — ECMWF IFS Ensemble (forecast conditions)

Stream 15 surface + 4 pressure-level variables × 51 members via GIK:

```
GIK Parquets (GCS) → S3 byte-range reads → gribberish decode
    │
    ├── 2t + 2d → T-Td (saturation)     ──► Moisture node
    ├── 10u + 10v → wind speed           ──► Wind node
    ├── blh + t(925)-2t → inversion      ──► Stability node
    ├── tp → 24h rainfall                ──► Rain parent node
    ├── msl → pressure gradient          ──► Synoptic typing
    ├── tcc + lcc → cloud cover          ──► Cloud node
    └── str + ssr → radiative cooling    ──► Stability node
```

Variables traced to: Parde et al. (2022) §2.1-§2.3 and Boneh et al. (2015) §3a (see plan §3.4.1).

### Step 3 — Bayesian Network → Traffic Light

```
         Parents                    Children (diagnostic)
    ┌──────────┐
    │  Season  │               ┌─────────┐ ┌──────┐
    └────┬─────┘               │Stability│ │ AOD  │
         │                     └─────────┘ └──────┘
  Rain ──┤── FOG ──────────── Moisture, Wind, S3 Fog
         │  (yes/no)
  Pollution
  Index ─┘

  P(fog) → GREEN / YELLOW / ORANGE / RED per district
```

| Color | P(fog) | Action |
|---|---|---|
| GREEN | < 10% | Monitor |
| YELLOW | 10-30% | Be Aware |
| ORANGE | 30-60% | Be Prepared |
| RED | > 60% | Take Action |

## Data Sources

| Source | Access | Variables |
|---|---|---|
| S5P-PAL L3 | STAC: `https://data-portal.s5p-pal.com/api/s5p-l3` | AOD, TCWV, NO2, SO2, UVAI |
| Sentinel-3 SLSTR | EOPF Zarr: `https://stac.core.eopf.eodc.eu` | LST, cloud mask |
| ECMWF IFS 51-member | GIK: `s3://ecmwf-forecasts/` (anon) | 19 variables, see plan §3.4.2 |
| S5P L3 AOD yearly | Local NetCDF (HARP-processed) | Climatological baseline |

## References

1. Parde et al. (2022) "Operational probabilistic fog prediction based on ensemble forecast system" — *Atmosphere* 13(10):1608
2. Boneh et al. (2015) "Fog forecasting for Melbourne Airport using a Bayesian decision network" — *Weather and Forecasting* 30(5):1218-1233
