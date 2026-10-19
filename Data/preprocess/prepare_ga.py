#!/usr/bin/env python
"""GA -- human germ cell development (Garcia-Alonso et al. 2022).

    CELLXGENE_API=<api-base> python Data/preprocess/prepare_ga.py

The raw data comes from CZ CELLxGENE. The script downloads it itself, so no files need to be
prepared by hand, but the API base URL is cleared in this anonymous submission and must be
supplied through the CELLXGENE_API environment variable:

    collection 661a402a-2a5a-4c71-9b05-b346c57bc451
      -> dataset "Human Germ Cells"   (~11k cells, saved to Data/GA/raw/)

The lookup is by title rather than UUID, so the script keeps working after CELLxGENE re-releases.
`aria2c` is used for a multi-connection download when available (the h5ad is about 200 MB), otherwise
it falls back to urllib.

Pipeline:
    1. download the h5ad
    2. PCW -> obs['timepoint'] / obs['time']
    3. drop cells with a missing PCW (scDiTT requires a known time point for every cell)
    4. PCW -> obs['tp'] (11 time points, 0..10); the raw PCW is kept in obs['pcw_real']
    5. HVG(2000); the X published by CELLxGENE is already normalised + log1p, so normalisation is skipped
    6. clean the nullable columns of obs/var -> Data/GA/garcia_alonso.h5ad
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import scanpy as sc

from _common import (assign_tp, banner, build_argparser, looks_log1p,
                     normalize_log1p, sanitise_for_h5ad, save, select_hvg, step)

OUT_NAME = "garcia_alonso.h5ad"

# Base URL of the CELLxGENE Discover curation API. Cleared for the anonymous
# submission; set it via the CELLXGENE_API environment variable (the public
# endpoint is documented by the CELLxGENE Discover curation API).
CELLXGENE_API = os.environ.get("CELLXGENE_API", "")
COLLECTION_ID = "661a402a-2a5a-4c71-9b05-b346c57bc451"
DATASET_TITLE = "Human Germ Cells"

TIME_COLUMNS = ["PCW", "development_stage", "donor_pcw", "pcw", "age", "stage"]


def resolve_h5ad_url(collection_id: str, title: str) -> str:
    if not CELLXGENE_API:
        raise SystemExit(
            "CELLXGENE_API is empty. Set it to the base URL of the CELLxGENE Discover "
            "curation API, e.g. CELLXGENE_API=<api-base> python Data/preprocess/prepare_ga.py")
    url = f"{CELLXGENE_API}/collections/{collection_id}"
    print(f"[api  ] {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        coll = json.loads(resp.read())
    by_title = {ds.get("title"): ds for ds in coll.get("datasets", [])}
    ds = by_title.get(title)
    if ds is None:
        raise SystemExit(f"no dataset titled {title!r} in the collection; "
                         f"available titles: {sorted(by_title)}")
    print(f"[pick ] {title}  n_cells={ds.get('cell_count')}  id={ds.get('dataset_id')}")
    for asset in ds.get("assets", []):
        if asset.get("filetype", "").lower() == "h5ad":
            return asset["url"]
    raise SystemExit(f"dataset {title!r} has no h5ad asset")


def download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            remote = int(resp.headers.get("Content-Length", "0")) or None
    except Exception:
        remote = None
    if out.exists() and (remote is None or out.stat().st_size == remote):
        print(f"[skip ] already downloaded: {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
        return out

    print(f"[get  ] {url}")
    if shutil.which("aria2c"):
        cmd = ["aria2c", "--max-connection-per-server=8", "--split=8",
               "--min-split-size=5M", "--max-tries=10", "--retry-wait=10",
               "--continue=true", "--allow-overwrite=true",
               "--auto-file-renaming=false", "--console-log-level=warn",
               f"--dir={out.parent}", f"--out={out.name}", url]
        if subprocess.run(cmd).returncode == 0:
            return out
        print("[warn ] aria2c failed, falling back to urllib")
    urllib.request.urlretrieve(url, out)
    return out


def parse_week(value) -> float:
    s = str(value)
    m = re.search(r"^\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*(?:st|nd|rd|th)?\s*week", s, re.I)
    return float(m.group(1)) if m else np.nan


def normalise_time(adata) -> None:
    col = next((c for c in TIME_COLUMNS if c in adata.obs.columns), None)
    if col is None:
        raise SystemExit(f"no time column found in obs, candidates: {TIME_COLUMNS}")
    weeks = adata.obs[col].map(parse_week).astype(float)
    adata.obs["timepoint"] = np.rint(weeks).astype("Int64")
    adata.obs["time"] = adata.obs["timepoint"].astype(str)
    print(f"[time ] using obs['{col}']; PCW values "
          f"{sorted(adata.obs['timepoint'].dropna().unique().tolist())}")


def drop_na_timepoint(adata):
    keep = adata.obs["timepoint"].notna().to_numpy()
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"[drop ] dropping cells with a missing PCW: {adata.n_obs:,} -> {int(keep.sum()):,}")
    return adata[keep].copy()


def main() -> None:
    args = build_argparser("GA", OUT_NAME, __doc__).parse_args()
    banner("GA — garcia_alonso (Garcia-Alonso et al. 2022, CELLxGENE)")

    step("1. download from CELLxGENE")
    raw_path = args.raw_dir / "Human_Germ_Cells.h5ad"
    if args.force and raw_path.exists():
        raw_path.unlink()
    download(resolve_h5ad_url(COLLECTION_ID, DATASET_TITLE), raw_path)
    adata = sc.read_h5ad(raw_path)
    print(f"[load ] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    step("2. parse PCW")
    normalise_time(adata)

    step("3. drop cells with a missing time point")
    adata = drop_na_timepoint(adata)

    step("4. PCW -> obs['tp']")
    adata.obs["pcw_real"] = adata.obs["timepoint"].astype("float32")
    assign_tp(adata, adata.obs["pcw_real"])

    step("5. HVG")
    adata.layers["counts"] = adata.X.copy()
    if looks_log1p(adata):
        print("[norm ] X is already normalised + log1p (as published by CELLxGENE), skipping normalisation")
    else:
        normalize_log1p(adata)
    adata = select_hvg(adata, args.n_top_genes)

    step("6. clean the nullable columns of obs / var")
    sanitise_for_h5ad(adata)

    save(adata, args.out, "GA", compression="gzip")


if __name__ == "__main__":
    main()
