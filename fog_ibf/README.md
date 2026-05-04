# Fog & Air-Quality Impact-Based Forecasting (BN-IBF)

ICPAC pipeline for daily fog / air-quality risk advisories, mirroring the
`bn-ibf/flood_ibf/` architecture. Replaces precipitation-driven evidence
(IMERG observations + GEFS ensemble + CMORPH return periods) with:

- **ECMWF IFS ensemble** (51 members × 10 fog-relevant variables × 53 lead-times,
  0.25°, on source.coop Icechunk — produced by
  `../ingest_ecmwf_fog_variables.py`).
- **Sentinel-5P / Sentinel-3 satellite indicators** (AOD, TCWV, NO₂, LST) from
  `../fog_satellite_indicators.py` — observed antecedent state.

…and feeds them through a 5-parent Bayesian Network (RxInfer / message-passing,
soft-evidence enabled per the April 2026 upgrade) to produce per-admin-1 risk
posteriors and CRMA decision states (`Monitor / Evaluate / Assess /
Actionable_Risk`).

## Why fog *and* air quality

The two phenomena are physically coupled:
- A stable, calm, high-RH boundary layer (the conditions that form fog)
  **also** suppresses pollutant dispersion → AQ degradation.
- High aerosol loading provides hygroscopic CCN that lowers the RH threshold
  for fog formation (Parde et al. 2022) and intensifies visibility loss.

We treat them as a **joint risk** — one hidden node `fog_aq_risk` whose
parents combine forecast fog-formation potential (from IFS) with observed
aerosol/pollution loading (from S5P). Decision logic (CRMA cost-loss
trigger) is identical to the flood pipeline; only the state semantics
change.

## Pipeline stages

```
┌──────────────────────────┐    ┌──────────────────────────┐
│ source.coop IFS Icechunk │    │ S5P-PAL / EOPF STAC      │
│ (51 mem × 10 vars        │    │ (AOD, TCWV, NO2, LST)    │
│  × 53 leads × 0.25°)     │    │  daily L3 + yearly clim  │
└────────────┬─────────────┘    └────────────┬─────────────┘
             │ open_icechunk()                │ fetch_s5p_daily()
             ▼                                ▼
        per-pixel fog-formation       per-zone aerosol /
        index (per member, per         moisture / pollution
        lead-time)                     summaries
             │                                │
             └─────────────┬──────────────────┘
                           ▼
        Stage 1: fog_data_prep.py
        ────────────────────────────
        Zonal reductions over icpac_adm1v3.geojson:
          - antecedent_aerosol  (S5P AOD daily mean)
          - antecedent_moisture (S5P TCWV)
          - ifs_fog_prob        (P(fog conditions met, any lead, mean of ens))
          - spatial_coverage    (fraction of zone with prob > 0.5)
          - stagnation_trend    (10m wind speed slope across leads)
          - extreme_fog_tail    (ensemble-max fog index, p95 over zone)
        + optional soft-evidence columns
        ─────────────────────────────────────────
        Output:  bn_inputs/fog_inputs_YYYY-MM-DD.csv  (one row per adm-1)

                           │
                           ▼
        Stage 2: fog_bn_ibf_v1.jl
        ────────────────────────────
        RxInfer 5-parent message passing with virtual-evidence channels:
          aer_data ─►  aer  ─┐
          mois_data ─► mois ─┤
          fog_data ─►  fog  ─┼──► fog_aq_risk ──► CRMA cost-loss decision
          stag_data ─► stag ─┤        (5 states)         (4 states)
          tail_data ─► tail ─┘
        Domain CPTs encoded as expert rules (see compute_risk_probs).
        ─────────────────────────────────────────
        Output:  output/fog_bn_v1_YYYY-MM-DD.csv  (one row per adm-1)
                 + risk posterior, CRMA state, traffic light
```

## BN structure

| Node | States | Source | Description |
|---|---|---|---|
| `aer` (aerosol load) | Low / Moderate / High (3) | S5P AOD daily | Hygroscopic CCN availability + AQ baseline |
| `mois` (moisture) | Dry / Moderate / Moist (3) | S5P TCWV (or IFS r_pl) | Column moisture available for condensation |
| `fog` (fog formation prob) | Very_Low / Low / Medium / High / Very_High (5) | IFS ensemble | P(fog conditions met) over members & leads |
| `stag` (stagnation) | Improving / Stable / Stagnating (3) | IFS u10/v10 trend | Boundary-layer ventilation outlook |
| `tail` (extreme fog tail) | Nil / Low / Moderate / High (4) | IFS ens-max | Worst-member fog index — captures storyline-tail risk |
| `fog_aq_risk` | Minimal / Low / Moderate / High / Extreme (5) | **hidden, inferred** | Combined fog + AQ joint risk |
| `crma_state` | Monitor / Evaluate / Assess / Actionable_Risk (4) | **deterministic** | Cost-loss trigger on `fog_aq_risk` posterior |

DAG: all five evidence nodes are root parents of `fog_aq_risk`. Each has a
virtual-evidence sibling `*_data` that receives the prep CSV's hard
classification (one-hot) **or** the soft-binned probability vector. CRMA
is computed deterministically from the risk posterior using the same
cost-loss rule the flood pipeline uses (default C/L = 0.2).

## IFS variable → BN evidence mapping

The IFS store carries 10 variables. Per-grid-point per-member, we derive a
**fog-formation index** F ∈ [0,1] from:

| IFS var | Role | Threshold / function |
|---|---|---|
| `t2m`, `d2m` | RH at 2 m via Magnus | RH > 90% strongly favorable |
| `u10`, `v10` | 10 m wind speed | Speed < 3 m s⁻¹ favorable (calm) |
| `t_pl - t2m` | Inversion proxy (BLH proxy) | Positive ΔT → inversion; favorable |
| `tp` | Excludes fog (washout) | tp < 0.5 mm/3 h required |
| `ssrd` | Fog dissipation | High ssrd → fog burns off (used for **next-day persistence**, not formation) |
| `strd` | Cloud / longwave context | Used as soft modifier on radiation-fog cooling |
| `msl` | Synoptic stability | High pressure → calm; soft modifier |
| `r_pl` | Mid-tropo RH | Soft modifier when 2-m RH is borderline |

Per-pixel per-member fog index:
```
F_i,m = clamp(
    sigmoid((RH2m - 85) / 5)              # 0.5 at RH=85, ~1 at RH≥95
  * sigmoid((3 - WindSpeed) / 1)          # ~1 at calm, 0 at WS≥5
  * sigmoid((t_pl - t2m + 2) / 2)         # ~0 at strong instability, ~1 at inversion
  * (1 - sigmoid((tp - 5e-4) / 5e-4))     # 1 at dry, 0 at tp ≥ 1 mm/3h
  , 0, 1)
```
Per-pixel ensemble probability: `P_i = mean over m of (F_i,m ≥ 0.5)`.

Per-zone reductions (area-weighted, cosine-latitude):
- `ifs_fog_prob` = mean P_i over zone, max over selected leads (e.g. 24/48/72 h)
- `spatial_coverage` = fraction of zone where P_i > 0.5
- `extreme_fog_tail` = p95 (over zone) of ens-max F_i,m
- `stagnation_trend` = (mean WindSpeed over leads 48-72 h) − (mean over leads 0-24 h),
   sign + magnitude binned as Improving/Stable/Stagnating

## Satellite → BN evidence mapping

From `fog_satellite_indicators.py` (S5P-PAL daily L3 + S3 SLSTR LST):

| Indicator | BN node | Aggregation |
|---|---|---|
| AOD daily L3 (S5P-PAL `L3__AER_OT`) | `aer` | Zonal mean; thresholds 0.3 / 0.6 → Low/Moderate/High |
| AOD anomaly vs yearly clim | (logged, used for explanation only) | — |
| TCWV daily L3 (`L3__TCWV__`) | `mois` | Zonal mean; 15 / 30 kg m⁻² → Dry/Moderate/Moist |
| NO₂ daily L3 (`L3__NO2___`) | (logged) | Used in CRMA explanation when `aer` = High |
| Sentinel-3 LST (EOPF) | (logged, optional) | Surface cooling potential narrative |

Fallback when satellite data isn't available for the date:
- `aer` ← yearly AOD climatology zonal mean (uniform over date)
- `mois` ← IFS column-RH proxy (mean of `r_pl`, ensemble-mean, lead 0)

## File layout

```
fog_ibf/
├── README.md                      ← this file
├── Project.toml                   ← Julia deps for fog_bn_ibf_v1.jl
├── icpac_adm1v3.geojson           ← ICPAC admin-1 boundaries (227 polygons)
├── fog_data_prep.py               ← Stage 1 (Python)
├── fog_bn_ibf_v1.jl               ← Stage 2 (Julia / RxInfer)
├── bn_inputs/                     ← daily prep CSVs land here
│   └── fog_inputs_YYYY-MM-DD.csv
└── output/                        ← daily BN outputs
    └── fog_bn_v1_YYYY-MM-DD.csv
```

## Boundary schemas

`fog_data_prep.py` auto-detects the boundary GeoJSON schema and exports a
canonical `id, name, country` triple to the prep CSV regardless of source.
Two schemas are wired up; add new ones to `_SCHEMAS` at the top of the
script.

| GeoJSON | Schema tag | id field | name field | "country" field | n |
|---|---|---|---|---|---|
| `icpac_adm1v3.geojson` (East Africa) | `icpac_adm1` | `GID_1` | `NAME_1` | derived from GID_1 prefix → ISO country | 227 |
| `../INDIA_DISTRICTS15.json` (India) | `india_district` | `dist_code` | `district` | `state` (used as `country` column) | 820 → 814 after dropping null/empty geoms |

The Julia BN reads only `id`, `name`, `country` so the inference step is
identical for either domain.

Example (India, single date):

```bash
uv run fog_data_prep.py \
    --date 2025-01-01 \
    --boundaries ../INDIA_DISTRICTS15.json \
    --out bn_inputs/fog_inputs_india_2025-01-01.csv \
    --skip-satellite --soft-evidence
```

Verified output for that run: 814 rows × 36 cols, mean `ifs_fog_prob` 0.29,
top districts Jorhat / Majuli / Bongaigaon (Assam — Brahmaputra valley) and
several Bihar IGP districts at 100%, matching January fog climatology.

## Calibration of `fog_index_per_member`

The per-pixel per-member fog index F ∈ [0,1] is a sigmoid product of three
fuzzy factors. Pivots and scales (`FOG_RH_PIVOT_PCT`, `FOG_WIND_PIVOT`,
`FOG_TP_PIVOT_M`) live at the top of `fog_data_prep.py` and are calibrated
against the IFS init 2025-01-01 over the bbox:

| Factor | Pivot | Scale | Why |
|---|---|---|---|
| 2-m RH (Magnus from t2m, d2m) | 92 % | 3 % | Saturation; mean RH was ~59 % so a 92 % pivot keeps only near-saturation pixels |
| 10-m wind speed | 4 m s⁻¹ | 1.5 m s⁻¹ | Mean WS ~4 m s⁻¹; sigmoid descends — calmer pixels score higher |
| Cumulative `tp` | 2 mm | 2 mm | Cumulative since init; 2 mm cuts off rainy pixels |

The `t_pl` variable is **intentionally not used** in this version. In the
current store the pressure level is upper-tropospheric (~300 hPa, mean t_pl
~238 K), so `t_pl − t2m ≈ -56 K` everywhere and is not a usable BLH
inversion proxy. Run `../ingest_ecmwf_fog_variables.py probe-levels`
to confirm the actual hPa value; if a 925 / 850 hPa level becomes
available, re-enable the inversion term in `fog_index_per_member` and
adjust `FOG_INV_PIVOT` accordingly.

These thresholds are starting points and should be validated against
METAR/SYNOP visibility observations before operational use.

## How to run

### 1. Prep one day

```bash
cd fog_ibf
uv run fog_data_prep.py \
    --date 2025-01-01 \
    --out bn_inputs/fog_inputs_2025-01-01.csv \
    --soft-evidence
```

This:
- Opens the source.coop IFS Icechunk store anonymously.
- Computes per-zone fog-formation evidence (5-min on a laptop for one date).
- Optionally fetches S5P daily L3 indicators (skip with `--skip-satellite`).
- Writes the prep CSV.

### 2. Inference

```bash
julia --project=. fog_bn_ibf_v1.jl \
    --input bn_inputs/fog_inputs_2025-01-01.csv \
    --output output/fog_bn_v1_2025-01-01.csv \
    --include-tail-risk --soft-evidence
```

This:
- Builds the 5-parent risk CPT once (≈ 900 rules, expert-defined).
- Runs RxInfer message passing per zone with hard or soft evidence.
- Computes CRMA state from the posterior (cost-loss C/L = 0.2 by default).
- Writes per-zone risk CSV.

### 3. Range / DBN (deferred)

The flood pipeline's `run_dbn_sequence` (temporal chaining across days)
ports unchanged once we have ≥ 2 days of IFS data ingested. Today's run
is single-day.

## Differences from the flood pipeline

| Aspect | Flood | Fog / AQ |
|---|---|---|
| Hazard | Surface-water flooding | Fog formation + AQ degradation |
| Hidden node | `risk_level` | `fog_aq_risk` (same 5-state schema) |
| Forecast source | GEFS / ECMWF TP | ECMWF IFS (10 vars, fog-derived) |
| Antecedent source | IMERG 7-day rainfall | S5P daily AOD + TCWV |
| Threshold definition | CMORPH return periods (per pixel) | Heuristic fog-conditions thresholds |
| Tail-risk parent | `ens_max / RP` ratio | `ens-max F_i,m` p95 over zone |
| Trend parent | rainfall trend slope | wind-stagnation slope |
| Spatial parent | Heavy-rain coverage | Fog-conditions coverage |
| Decision | CRMA traffic light | Same |
| CPT engine | RxInfer + matmul fallback | RxInfer (5-parent w/ tail) |

## Implementation notes / things deliberately left out v1

- **No DBN temporal chaining** in v1 — needs ≥ 2 days of IFS data.
- **Forecast-agreement node** is omitted (would need a 6-parent CPT, exceeds
  RxInfer's tensor arity). Can be re-added via the matmul path if the
  ensemble spread becomes a primary signal.
- **Per-member storyline sidecar** is left as a `--member-evidence-sidecar`
  hook in the prep script — the Julia side already knows how to consume the
  schema (mirrors flood).
- **Threshold tuning** — the fog-formation index thresholds are heuristic
  starting points (Lawrence approximation for RH; 90 % / 3 m s⁻¹ / inversion
  cutoffs from typical fog-climatology lit). Final values must be validated
  against METAR / SYNOP visibility observations.

## References

- Parde, A. N. et al. (2022). Hygroscopic CCN activation properties for fog.
- Boneh, T. et al. (2015). BN-based fog forecasting; moisture availability
  and inversion strength as primary preconditions.
- Uusitalo, L. (2007); Marcot, B. G. (2012). Discretisation effects in BNs
  — the case for soft evidence.
- ICPAC IBF Team (April 2026). Soft-evidence upgrade design doc.
