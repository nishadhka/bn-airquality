# ECMWF IFS Ensemble — Fog-Variable Ingest Test Run

End-to-end smoke test of `ingest_ecmwf_fog_variables.py` writing one day of
ECMWF IFS ensemble (51 members × 10 variables × 53 lead-times × 221 × 321) to
an Icechunk store on source.coop, using a Coiled Dask cluster for fan-out.

## Run summary

| Field | Value |
|---|---|
| Init date | `2025-01-01` (single day) |
| Members | 51 (`control` + `ens_01..ens_50`) |
| Lead times | 53 (3-hourly to 144h, 6-hourly to 168h) |
| Variables | 10 (8 surface + 2 single-level pressure) |
| Spatial | 221 lat × 321 lon at 0.25°, bbox 20°E–100°E, 15°S–40°N |
| Coiled cluster | 15 × `e2-standard-4`, GCP `us-east1`, workspace `gcp-sewaa-nka` |
| Wall-clock | 21.7 min (3 min cluster start + 18.7 min fetch/write/commit) |
| Success | 51/51 members, 0 failed |
| STS budget used | ~22 / 45 min |
| Store path | `s3://us-west-2.opendata.source.coop/nishadhka/aq-icechunk-store-ifs/` |
| Browse | <https://source.coop/nishadhka/aq-icechunk-store-ifs> |

## Variables

Selected for a Bayesian-network fog model. Mapping is also stored as
`bn_mapping` in the dataset attributes.

### Surface (8)

| Var | GRIB short-name | Long name | Units | BN role |
|---|---|---|---|---|
| `t2m` | `2t` | 2 m temperature | K | temperature |
| `d2m` | `2d` | 2 m dewpoint temperature | K | dewpoint → RH |
| `u10` | `10u` | 10 m U wind | m s⁻¹ | wind |
| `v10` | `10v` | 10 m V wind | m s⁻¹ | wind |
| `msl` | `msl` | Mean sea-level pressure | Pa | synoptic |
| `strd` | `strd` | Surface thermal radiation downwards | J m⁻² | cloud proxy |
| `ssrd` | `ssrd` | Surface solar radiation downwards | J m⁻² | fog-dissipation |
| `tp` | `tp` | Total precipitation | m | exclusion |

### Single-level pressure (2)

The GIK parquet exposes a single hPa level per variable inside the GRIB
message. Run `uv run ingest_ecmwf_fog_variables.py probe-levels --date <YYYYMMDD>`
to print the actual hPa value before treating these as a known level.

| Var | GRIB short-name | Long name | Units | BN role |
|---|---|---|---|---|
| `r_pl` | `r/pl` | Relative humidity at pressure level | % | mid-tropo RH |
| `t_pl` | `t/pl` | Air temperature at pressure level | K | `t_pl − t2m` → inversion (BLH proxy) |

## Storage layout

Icechunk repository on source.coop with five top-level prefixes:

| Prefix | Objects | Bytes |
|---|---:|---:|
| `chunks/` | 510 | 2 741 662 985 |
| `manifests/` | 14 | 14 709 |
| `snapshots/` | 3 | 6 713 |
| `transactions/` | 3 | 2 029 |
| `overwritten/` | 2 | 713 |
| `repo` (file) | 1 | 563 |
| **Total** | **533** | **2 741 687 712** (≈ 2.74 GB) |

Uncompressed the same data is `51 × 53 × 221 × 321 × 4 B × 10 vars ≈ 7.14 GiB`,
so zstd compression is ~38% of raw — typical for ECMWF surface fields.

### Dimensions and chunking

```
dims:    init_date × member × lead_time × lat × lon
shape:   1         × 51     × 53        × 221 × 321
chunks:  1         × 1      × 53        × 221 × 321   (per-variable)
```

One chunk per `(member, variable)` — exactly 510 chunks (51 members × 10
vars × 1 date), confirmed by the S3 object count. Average chunk is
~5.4 MB compressed (~15 MB uncompressed).

### Coordinates

| Coord | Size | Range / values |
|---|---:|---|
| `init_date` | 1 | `2025-01-01` |
| `member` | 51 | `control`, `ens_01` … `ens_50` |
| `lead_time` | 53 | 0,3,6,…,144,150,156,162,168 (hours) |
| `lat` | 221 | 40.0 → -15.0 (0.25° step, north-down) |
| `lon` | 321 | 20.0 → 100.0 (0.25° step) |

## Method

The script does **not** download full GRIB files. Instead it uses the
[`grib-index-kerchunk`](https://github.com/E4DRR/gik-ecmwf-par) parquet
manifests on Hugging Face to look up byte-ranges for each `(step, variable,
member)` GRIB message inside the public ECMWF S3 archive, then issues
ranged S3 GETs per message.

```
GIK parquet (HF)              Public ECMWF S3                   Worker
───────────────              ──────────────                   ──────
key=step_036/2t/sfc/control   url, offset, length    ─►    GET range
                                                             │
                                                             ▼
                                                          GRIB msg ─► cfgrib
                                                                       │
                                                                       ▼
                                                                    np.ndarray (slice to bbox)
```

### Per-member task (`read_member_fog_vars`)

Runs on a Coiled worker:

1. Read parquet for `(date, member)` from HF (single HTTP fetch).
2. Decode parquet rows into a flat zstore dict mapping zarr keys → `(url,
   offset, length)`.
3. For each `(lead_time, variable)` build a fetch list (510 / 51 = 10
   variables × 53 lead-times = ~530 byte-ranges per member).
4. 8-thread `ThreadPoolExecutor` per worker submits ranged S3 GETs (anonymous,
   `fsspec.filesystem("s3", anon=True)`).
5. Each fetched byte-range is decoded with `cfgrib`, sliced to the bbox, and
   stored as `(53, 221, 321) float32`.
6. Returns `{"data": {var_name: np.ndarray, …}}` to the coordinator.

### Coordinator-side write loop (`fill_store`)

One date at a time, streaming per member:

1. Submit 51 member futures to the Coiled cluster.
2. Open one writable Icechunk session.
3. As each future completes, write that single member's slabs with
   `region={"init_date": slice(d, d+1), "member": slice(m, m+1)}` —
   chunk-aligned, one zarr write per `(date, member)`.
4. After all members for the date are done (or fail), commit once with
   message `fill date <idx> (<YYYYMMDD>): N/51 members`.

This keeps coordinator memory peak at ~150 MiB (one member's data in flight)
instead of the ~14 GiB the original full-batch buffer required.

## Credentials

Source.coop publishes 1-hour STS tokens via its console. These are AWS-style
keys and live in `.env` (gitignored):

```
export AWS_ACCESS_KEY_ID="ASIA…"
export AWS_SECRET_ACCESS_KEY="…"
export AWS_SESSION_TOKEN="…"
export AWS_DEFAULT_REGION="us-west-2"
export S3_PREFIX="nishadhka/aq-icechunk-store-ifs"
```

`.env` is loaded at module import (`_load_dotenv_into_environ`) so the
env-overridable constants `S3_BUCKET`, `S3_PREFIX`, `S3_REGION` pick up the
custom prefix without any code edit.

**The credentials never leave the coordinator.** `make_s3_storage()` builds
an `icechunk.s3_storage(...)` handle with the keys and passes it to
`Repository.open()` on the local process. All `to_zarr(session.store, ...)`
and `session.commit(...)` calls run on the coordinator. Coiled workers only
run `read_member_fog_vars`, which uses `fsspec.filesystem("s3", anon=True)`
for the public ECMWF GRIB byte-ranges — they do not need (and never see)
the source.coop write credentials.

The script enforces a `--credential-timeout` (default 2700 s = 45 min) and
will stop at a clean per-date boundary before STS expiration, allowing
resume from the last committed date after a credential refresh.

## Verification

`uv run ingest_ecmwf_fog_variables.py verify` (anonymous read against
source.coop) reports:

```
Variable 't2m': dtype=float32, shape=(1, 51, 53, 221, 321)
Variable 'd2m': dtype=float32, shape=(1, 51, 53, 221, 321)
…
Spot-check: first date, first member (control)
  d2m:   3759873/3759873 valid (100.0%)  min=224.5  max=299.2  mean=284.1
  msl:   3688932/3759873 valid (98.1%)  min=1.001e+05  max=1.057e+05  mean=1.015e+05
  r_pl:  3688932/3759873 valid (98.1%)  min=-1.586  max=121.1  mean=31.54
  ssrd:  3617991/3759873 valid (96.2%)  min=0  max=1.972e+08  mean=6.114e+07
  strd:  3688932/3759873 valid (98.1%)  min=0  max=2.611e+08  mean=9.841e+07
  t2m:   3759873/3759873 valid (100.0%)  min=230.3  max=314.9  mean=291.8
  u10:   3688932/3759873 valid (98.1%)  min=-16.48  max=18.78  mean=-0.5606
  v10:   3759873/3759873 valid (100.0%)  min=-16.41  max=17.73  mean=-1.311
  tp:    3759873/3759873 valid (100.0%)  min=0  max=0.4089  mean=0.009158
  t_pl:  3688932/3759873 valid (98.1%)  min=216.7  max=247.3  mean=238.4

Commits (3):
  fill date 0 (20250101): 51/51 members
  initialize ECMWF fog-vars template
  Repository initialized
```

The 96–98% "valid" rate on `msl`, `r_pl`, `ssrd`, `strd`, `u10`, `t_pl` is
expected: those variables have no analysis-step (lead = 0) value in the GIK
parquet — accumulations and pressure-level fields aren't published at step
0. `(53 - 1) / 53 = 98.1%` matches exactly. `t2m`, `d2m`, `v10`, `tp` are
populated at all 53 lead-times (100%).

## Reproduce / extend

```bash
# 1. Refresh source.coop creds in .env (1-hour STS tokens).
# 2. Init template for the date range you want to cover:
uv run ingest_ecmwf_fog_variables.py init --start-date 20250101 --end-date 20250131

# 3. Fill, with Coiled:
uv run ingest_ecmwf_fog_variables.py fill \
    --start-date 20250101 --end-date 20250131 \
    --n-workers 15 --coiled-region us-east1 --workspace gcp-sewaa-nka

# Resume after a credential timeout: refresh .env and rerun the same fill —
# completed dates are skipped via the commit-message resume probe.
```

For a single-machine smoke test (no Coiled), use `local-fill` instead — it
expects an `init`'d store and supports `--limit-members N` to keep memory
small.

```bash
uv run ingest_ecmwf_fog_variables.py local-fill \
    --start-date 20250101 --end-date 20250101 \
    --limit-members 4 --member-workers 4 --local /tmp/test_store
```

## Throughput notes

- Cluster spin-up (15 × e2-standard-4 in GCP us-east1): ~3 min.
- Per-member fetch (~530 byte-ranges, anonymous S3 GETs from public ECMWF
  bucket → cfgrib decode → bbox slice): ~3.5 min on `e2-standard-4` with the
  worker's 8-thread S3 pool.
- 51 members on 15 workers ≈ 3.4 rounds × 3.5 min ≈ 12 min fetch time.
- Streaming write + final commit: ~6 min.
- Per-date wall-clock observed: 18.7 min.
- For a 31-day backfill: ~10 hours total, fits comfortably in repeated
  ~45-minute credential windows.
