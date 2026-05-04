#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "geopandas",
#     "matplotlib",
#     "shapely",
#     "pyproj",
# ]
# ///
"""
Plot a choropleth PNG from a fog_ibf CSV.

Auto-detects whether the input is:
  • A Stage-1 prep CSV   (has `ifs_fog_prob`)            → fog-probability map
  • A Stage-2 BN output  (has `crma_state` or `risk_level`) → 4-panel:
      crma_state, risk_level, P(High+Extreme), antecedent aerosol

Joins to the boundaries GeoJSON via the CSV's `id` column matched against
either `GID_1` (ICPAC adm1) or `dist_code` (India districts).

Usage:
    uv run plot_fog_choropleth.py \
        --csv bn_inputs/fog_inputs_india_2025-01-01.csv \
        --boundaries ../INDIA_DISTRICTS15.json \
        --out output/fog_prob_india_2025-01-01.png

    uv run plot_fog_choropleth.py \
        --csv output/fog_bn_v1_2025-01-01.csv \
        --boundaries ../INDIA_DISTRICTS15.json \
        --out output/fog_bn_v1_india_2025-01-01.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

# Boundary-file schemas (matches fog_data_prep._SCHEMAS)
_SCHEMAS = {
    "icpac_adm1":     {"id": "GID_1",     "name": "NAME_1"},
    "india_district": {"id": "dist_code", "name": "district"},
}


def detect_schema(gdf: gpd.GeoDataFrame) -> dict:
    cols = set(gdf.columns)
    for tag, schema in _SCHEMAS.items():
        if schema["id"] in cols and schema["name"] in cols:
            schema = dict(schema)
            schema["_tag"] = tag
            return schema
    raise ValueError(f"Boundary file schema not recognised. Columns: {sorted(cols)}")


def load_and_join(csv_path: Path, geojson_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    # BN output uses `boundary_id`; prep CSV uses `id`. Normalise.
    if "id" not in df.columns and "boundary_id" in df.columns:
        df = df.rename(columns={"boundary_id": "id"})
    df["id"] = df["id"].astype(str)
    # India districts file has dup ids ("NOT AVAILABLE" × 26, NaN → "nan",
    # and some integer codes shared across polygons). The prep script writes
    # one CSV row per non-null polygon; keep the first occurrence so the
    # merge is 1-to-many polygon-side, not many-to-many.
    n_csv_in = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    if len(df) < n_csv_in:
        print(f"[plot] deduped CSV by id: {n_csv_in} → {len(df)} rows "
              f"(dropped {n_csv_in - len(df)} duplicate ids)")

    gdf = gpd.read_file(geojson_path)
    schema = detect_schema(gdf)
    gdf[schema["id"]] = gdf[schema["id"]].astype(str)
    gdf = gdf.loc[~gdf.geometry.isna() & ~gdf.geometry.is_empty].copy()
    gdf.loc[~gdf.geometry.is_valid, "geometry"] = gdf.loc[~gdf.geometry.is_valid, "geometry"].buffer(0)

    merged = gdf.merge(df, left_on=schema["id"], right_on="id", how="left")
    n_unmatched = int(merged["id"].isna().sum())
    print(f"[plot] polygons={len(gdf)}  matched={len(merged) - n_unmatched}  "
          f"unmatched={n_unmatched}")
    return merged, schema


# ─── Renderers ────────────────────────────────────────────────────────

CRMA_COLORS = {
    "Monitor":         "#1a9641",  # green
    "Evaluate":        "#fdae61",  # yellow-orange
    "Assess":          "#d7301f",  # orange
    "Actionable_Risk": "#7f0000",  # red
}
RISK_COLORS = {
    "Minimal":  "#2c7bb6",
    "Low":      "#abd9e9",
    "Moderate": "#ffffbf",
    "High":     "#fdae61",
    "Extreme":  "#d7191c",
}
AER_COLORS = {
    "Low":      "#fefce6",
    "Moderate": "#fdae61",
    "High":     "#7f0000",
}


def _plot_categorical(ax, merged: gpd.GeoDataFrame, col: str, color_map: dict,
                      title: str):
    from matplotlib.patches import Patch
    handles = []
    for state, color in color_map.items():
        sub = merged[merged[col] == state]
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, color=color, edgecolor="black", linewidth=0.1)
        handles.append(Patch(facecolor=color, edgecolor="black", label=state))
    # Anything not in the colour map (NaN, unknown) → grey
    other = merged[~merged[col].isin(color_map.keys()) | merged[col].isna()]
    if len(other) > 0:
        other.plot(ax=ax, color="#cccccc", edgecolor="black", linewidth=0.1)
        handles.append(Patch(facecolor="#cccccc", edgecolor="black", label="(no data)"))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=8,
                  frameon=True, framealpha=0.9, ncol=1)


def _plot_continuous(ax, merged: gpd.GeoDataFrame, col: str, *,
                     cmap: str, vmin: float, vmax: float, title: str,
                     label: str):
    plot = merged.plot(column=col, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolor="black", linewidth=0.1,
                       missing_kwds={"color": "#cccccc", "edgecolor": "black",
                                     "linewidth": 0.1, "label": "(no data)"})
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(label, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def render_prep(merged: gpd.GeoDataFrame, out: Path, date_str: str, title_suffix: str):
    """Stage-1 prep CSV → 4-panel diagnostic map."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    _plot_continuous(axes[0, 0], merged, "ifs_fog_prob",
                     cmap="YlGnBu", vmin=0, vmax=1,
                     title="Ensemble fog probability (max over leads 9–72 h)",
                     label="P(fog conditions met)")
    _plot_continuous(axes[0, 1], merged, "spatial_coverage",
                     cmap="YlOrBr", vmin=0, vmax=1,
                     title="Spatial coverage (fraction of zone with P > 0.5)",
                     label="coverage fraction")
    _plot_categorical(axes[1, 0], merged, "stagnation_trend",
                      {"Improving": "#1a9641", "Stable": "#fdae61",
                       "Stagnating": "#d7301f"},
                      title="Stagnation trend (10 m wind slope, near vs far leads)")
    _plot_continuous(axes[1, 1], merged, "extreme_fog_tail_p95",
                     cmap="Reds", vmin=0, vmax=1,
                     title="Extreme fog tail (zone p95 of ens-max F)",
                     label="ens-max fog index, p95")

    fig.suptitle(f"Stage-1 fog prep — {title_suffix} ({date_str})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def render_bn(merged: gpd.GeoDataFrame, out: Path, date_str: str, title_suffix: str):
    """Stage-2 BN output CSV → 4-panel risk map."""
    if "p_high_extreme" not in merged.columns:
        for hi, ex in [("risk_high", "risk_extreme")]:
            if hi in merged.columns and ex in merged.columns:
                merged["p_high_extreme"] = merged[hi].fillna(0) + merged[ex].fillna(0)
                break

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    if "crma_state" in merged.columns:
        _plot_categorical(axes[0, 0], merged, "crma_state", CRMA_COLORS,
                          title="CRMA decision (cost-loss trigger, C/L = 0.2)")
    if "risk_level" in merged.columns:
        _plot_categorical(axes[0, 1], merged, "risk_level", RISK_COLORS,
                          title="Argmax risk level")
    if "p_high_extreme" in merged.columns:
        _plot_continuous(axes[1, 0], merged, "p_high_extreme",
                         cmap="Reds", vmin=0, vmax=1,
                         title="P(High) + P(Extreme)  — escalation probability",
                         label="probability")
    if "antecedent_aerosol_state" in merged.columns:
        _plot_categorical(axes[1, 1], merged, "antecedent_aerosol_state", AER_COLORS,
                          title="Antecedent aerosol load (S5P AOD)")
    elif "ifs_fog_prob" in merged.columns:
        _plot_continuous(axes[1, 1], merged, "ifs_fog_prob",
                         cmap="YlGnBu", vmin=0, vmax=1,
                         title="Underlying ensemble fog prob (forecast)",
                         label="P(fog conditions met)")

    fig.suptitle(f"Fog BN-IBF — {title_suffix} ({date_str})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Choropleth PNG of fog_ibf CSV outputs")
    ap.add_argument("--csv", required=True, help="Stage-1 prep CSV or Stage-2 BN output CSV")
    ap.add_argument("--boundaries", required=True, help="GeoJSON of zones (icpac_adm1 or india_district)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title-suffix", default="", help="Extra string appended to suptitle")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    geo_path = Path(args.boundaries)
    out_path = Path(args.out)

    merged, schema = load_and_join(csv_path, geo_path)
    date_str = (str(merged["target_date"].dropna().iloc[0])
                if "target_date" in merged.columns and merged["target_date"].notna().any()
                else "")
    suffix = args.title_suffix or f"{schema['_tag']}"

    is_bn = ("crma_state" in merged.columns) or ("risk_level" in merged.columns)
    if is_bn:
        render_bn(merged, out_path, date_str, suffix)
    else:
        render_prep(merged, out_path, date_str, suffix)


if __name__ == "__main__":
    main()
