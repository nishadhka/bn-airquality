# AOD Daily Dataset Search — VIIRS & MODIS

Investigation into available satellite Aerosol Optical Depth (AOD) daily gridded datasets
for the Extended East Africa → India region (20°E–100°E, 15°S–40°N), from 2019 onwards.

---

## 1. NOAA JPSS NODD (AWS S3) — Search Result

The original script `download_viirs_reprocessed_aod_gridded_v1.py` targets the NOAA JPSS
NODD archive at `s3://noaa-jpss/`, specifically:

| Path | Status |
|---|---|
| `s3://noaa-jpss/SNPP/VIIRS/SNPP_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/` | **Not found** |
| `s3://noaa-jpss/NOAA20/VIIRS/NOAA20_VIIRS_Aerosol_Optical_Depth_Gridded_Reprocessed/` | **Not found** |

The `noaa-jpss` S3 bucket is publicly accessible (anonymous) but contains no AOD/aerosol
gridded data. The reprocessed Level 3 AOD dataset (SNPP: 2012–2020, NOAA-20: 2018–2020)
was originally hosted on a **NOAA/NESDIS/STAR web server** referenced in the
[AMS Python Workshop 2023 notebook](https://github.com/modern-tools-workshop/AMS-python-workshop-2023/blob/main/download_satellite_data.ipynb).
That server is no longer accessible.

### Other NOAA PDS buckets checked

`noaa-nesdis-snpp-pds` and `noaa-nesdis-n21-pds` contain **Level 2 swath** JRR-AOD
data (operational, not reprocessed), but no Level 3 gridded AOD:

| Bucket | Product | Years | Type |
|---|---|---|---|
| `noaa-nesdis-snpp-pds/VIIRS-JRR-AOD/` | JRR-AOD v3r2 | 2022–present | L2 swath granules |
| `noaa-nesdis-n21-pds/VIIRS-JRR-AOD/` | JRR-AOD v3r2 | 2023–present | L2 swath granules |

Gridded products (`GRIDDED_VIIRS_LSA_DLY`, `GRIDDED_VIIRS_LST_D/N`) exist for Land
Surface Albedo and Temperature only — no AOD equivalent.

---

## 2. NASA EarthData (LAADS DAAC / GES DISC) — Selected Datasets

All NASA EarthData products require a free
[NASA EarthData account](https://urs.earthdata.nasa.gov). Authentication is handled via
`earthaccess` using credentials stored in `.env`.

### 2.1 MODIS MAIAC AOD — MCD19A2CMG ✅ Selected

| Attribute | Value |
|---|---|
| **Short name** | `MCD19A2CMG` v061 |
| **Provider** | NASA LPDAAC (EarthData Cloud) |
| **Description** | MODIS Terra+Aqua AOD and Water Vapor from MAIAC, Daily L3 Global |
| **Resolution** | 0.05° × 0.05° (~5.5 km) |
| **Format** | HDF-EOS2 (HDF4) — requires `pyhdf` to read |
| **Coverage** | 2000-02-24 → ongoing |
| **File size** | ~44 MB/day (global), 1 file per day |
| **Key variable** | `AOD_055` — daily average AOD at 550 nm, scale factor 0.001, fill -28672 |
| **Grid** | CMGgrid: 3600 lat × 7200 lon, upper-left origin (90°N, 180°W), descending lat |
| **2019–present global** | ~115 GB |
| **2019–present (bbox, cropped)** | ~7 GB |

**Bbox subset shape:** 1100 lat × 1600 lon (40°N→-15°S descending, 20°E→100°E)

### 2.2 VIIRS NOAA-20 L3 AOD — AER_DBDT_D10KM_L3_VIIRS_NOAA20 ✅ Selected

| Attribute | Value |
|---|---|
| **Short name** | `AER_DBDT_D10KM_L3_VIIRS_NOAA20` v001 |
| **Provider** | NASA GES DISC |
| **Description** | NOAA-20 VIIRS High Resolution Level 3 daily aerosol, 0.1×0.1 degree grid |
| **Resolution** | 0.1° × 0.1° (~11 km) |
| **Format** | NetCDF-4 — readable directly with `xarray` |
| **Coverage** | 2018-02-17 → ongoing |
| **File size** | ~38 MB/day (global), 1 file per day |
| **Key variable** | `COMBINE_AOD_550_AVG` — combined Dark Target + Deep Blue AOD at 550 nm |
| **Other variables** | `DT_AOD_550_AVG`, `DB_AOD_550_AVG`, `DT_DB_AOD_550_AVG`, `DB_DT_AOD_550_AVG` |
| **Grid** | 3600 lon × 1800 lat, ascending lat (-89.95→89.95), lon (-179.95→179.95) |
| **2019–present global** | ~98 GB |
| **2019–present (bbox, cropped)** | ~6 GB |

**Bbox subset shape:** 550 lat × 800 lon (-15°S→40°N ascending, 20°E→100°E)

### 2.3 Other VIIRS AOD products found (not selected)

| Short name | Satellite | Type | Resolution | Format | Coverage |
|---|---|---|---|---|---|
| `AERDB_L2_VIIRS_SNPP` | SNPP | Deep Blue L2 swath | 6 km | netCDF-4 | 2012→present |
| `AERDT_L2_VIIRS_SNPP` | SNPP | Dark Target L2 swath | 6 km | netCDF-4 | 2012→present |
| `AERDT_L2_VIIRS_NOAA20` | NOAA-20 | Dark Target L2 swath | 6 km | netCDF-4 | 2018→present |
| `AER_DBDT_D10KM_L3_VIIRS_SNPP` | SNPP | Deep Blue+DT L3 gridded | 0.1° | NetCDF | 2012→present |

L2 swath products require gridding before use (~375 GB for 2019–present per satellite).
`AER_DBDT_D10KM_L3_VIIRS_SNPP` is the SNPP equivalent of the selected NOAA-20 product
and can be added later to extend the record back to 2012.

---

## 3. Ingestion Pipeline

Script: `ingest_aod_to_icechunk.py`

Downloads data month-by-month, subsets to bbox, and writes to
[icechunk](https://icechunk.io) zarr v3 stores on Google Cloud Storage.

### GCS store layout

```
gs://cdi_arco/
├── temp_modis_aod/    ← MCD19A2CMG, 0.05°, HDF4 read via pyhdf
└── temp_viirs_aod/    ← AER_DBDT_D10KM_L3_VIIRS_NOAA20, 0.1°, NetCDF via xarray
```

### Zarr array dimensions

| Store | Variable | Dimensions | Chunk shape |
|---|---|---|---|
| `temp_modis_aod` | `AOD_055` | `(time, lat, lon)` | `(10, 200, 200)` |
| `temp_viirs_aod` | `AOD_055` | `(time, lat, lon)` | `(10, 200, 200)` |

### Environment variables (`.env`)

```
EARTHDATA_USERNAME=<nasa_earthdata_user>
EARTHDATA_PASSWORD=<nasa_earthdata_password>
GCS_BUCKET=cdi_arco
GCS_SERVICE_ACCOUNT_FILE=<path_to_service_account.json>
MODIS_STORE_PATH=temp_modis_aod
VIIRS_STORE_PATH=temp_viirs_aod
```

### Run

```bash
uv run --with earthaccess --with python-dotenv --with xarray --with netcdf4 \
       --with pyhdf --with icechunk --with python-dateutil \
       python ingest_aod_to_icechunk.py
```

The script resumes from the last committed snapshot on re-run — safe to restart after
interruption.
