#!/usr/bin/env python
"""SC -- MEF reprogramming time course (Schiebinger et al. 2019, GEO: GSE122662).

    python Data/preprocess/prepare_sc.py

Raw inputs (place in `Data/SC/raw/`):

    GSM3195*_D*_*_C*_gene_bc_mat.h5      126 10x h5 files (the dense time course of GSE122662)
    broadinstitute_metadata/cell_sets.gmt  cell types published by the authors (GMT format)
    broadinstitute_metadata/FLE.txt        force-directed layout coordinates published by the authors

The 10x h5 filenames look like `GSM3195650_D0.5_Dox_C1_gene_bc_mat.h5`, where `D<day>`, `<condition>`
and `C<batch>` are parsed into obs and combined into cell names of the form `D0.5_Dox_C1_<barcode>` --
exactly the ID form used by cell_sets.gmt / FLE.txt.

Pipeline:
    1. merge the 126 10x h5 files into one h5ad (cached in raw/schiebinger_2019_dense_raw.h5ad)
    2. read cell types from the GMT and 2-D coordinates from FLE.txt -> obsm['X_fle']
    3. back up the raw counts to layers['counts'], QC + normalize_total(1e4) + log1p
    4. HVG(2000) + injection of pluripotency / EMT / MET markers
    5. take the subset with integer days (day 0..18) and batch_replicate == '1', day -> obs['tp']
    6. PCA / UMAP -> Data/SC/schiebinger_2019_preprocess_subset.h5ad

The subsetting in step 5 is necessary: the raw data contains sampling points at half-day intervals and
iPSC endpoint samples, whereas scDiTT needs 19 evenly spaced integer time points; keeping a single batch
also avoids batch effects.
"""
from __future__ import annotations

import glob
import os
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from _common import (assign_tp, banner, build_argparser, load_or_build,
                     normalize_log1p, qc_filter, save, select_hvg, step)

OUT_NAME = "schiebinger_2019_preprocess_subset.h5ad"

FILENAME_RE = re.compile(r"D([0-9.]+|iPSC)_([A-Za-z0-9]+)_C(\d+)")
IPSC_DAY = 19.0

KEEP_DAYS = list(range(19))
KEEP_BATCH = "1"

MANUAL_MARKERS: list[str] = [
    "Pou5f1", "Sox2", "Klf4", "Myc", "Nanog", "Esrrb", "Utf1", "Lin28a", "Sall4",
    "Vim", "Fn1", "Zeb1", "Zeb2", "Snai1", "Snai2", "Twist1", "Cdh2",
    "Cdh1", "Epcam", "Ocln", "Krt8", "Krt18",
    "Gata3", "Eomes", "Cdx2", "Elf5",
    "Pax6", "Sox1", "Nes",
]


def build_raw(raw_dir):
    paths = sorted(glob.glob(os.path.join(raw_dir, "GSM3195*.h5")))
    print(f"[load ] found {len(paths)} 10x h5 files")
    parts = []
    for path in paths:
        fname = os.path.basename(path)
        m = FILENAME_RE.search(fname)
        if m is None:
            print(f"[skip ] filename does not match: {fname}")
            continue
        day_str, condition, batch = m.groups()
        a = sc.read_10x_h5(path)
        a.var_names_make_unique()
        prefix = f"D{day_str}_{condition}_C{batch}_"
        a.obs_names = [prefix + bc for bc in a.obs_names]
        a.obs["day"] = IPSC_DAY if day_str == "iPSC" else float(day_str)
        a.obs["condition"] = condition
        a.obs["batch_replicate"] = batch
        a.obs["source_file"] = fname
        parts.append(a)

    adata = ad.concat(parts, join="outer")
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    print(f"[load ] after merge {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def read_gmt(path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) >= 3:
                mapping.update({cid: parts[0] for cid in parts[2:]})
    return mapping


def attach_metadata(adata, meta_dir) -> None:
    cell_type = read_gmt(meta_dir / "cell_sets.gmt")
    print(f"[meta ] GMT provides {len(cell_type):,} cell-type annotations")
    adata.obs["cell_type"] = (pd.Series(adata.obs_names.map(cell_type),
                                        index=adata.obs_names)
                              .fillna("Unknown").astype("category"))
    print(adata.obs["cell_type"].value_counts().to_string())

    fle = pd.read_csv(meta_dir / "FLE.txt", skiprows=[1], index_col=0, header=0)
    fle.columns = [c.strip().upper() for c in fle.columns]
    if not {"X", "Y"} <= set(fle.columns):
        fle = pd.read_csv(meta_dir / "FLE.txt", sep="\t", skiprows=[1],
                          index_col=0, header=0)
        fle.columns = [c.strip().upper() for c in fle.columns]
    coords = fle.reindex(adata.obs_names)[["X", "Y"]].to_numpy()
    adata.obsm["X_fle"] = coords
    n_missing = int(np.isnan(coords[:, 0]).sum())
    print(f"[meta ] FLE coordinates missing for {n_missing:,} / {adata.n_obs:,} cells")


def main() -> None:
    args = build_argparser("SC", OUT_NAME, __doc__).parse_args()
    banner("SC — schiebinger (Schiebinger et al. 2019, GSE122662)")

    step("1. 126 10x h5 files -> h5ad")
    adata = load_or_build(args.raw_dir / "schiebinger_2019_dense_raw.h5ad",
                          lambda: build_raw(args.raw_dir), force=args.force)

    step("2. cell types (GMT) and FLE coordinates")
    attach_metadata(adata, args.raw_dir / "broadinstitute_metadata")

    step("3. QC + normalisation + log1p")
    adata.layers["counts"] = adata.X.copy()
    adata = qc_filter(adata, min_genes=500)
    normalize_log1p(adata)
    adata.raw = adata

    step("4. HVG + marker injection")
    adata = select_hvg(adata, args.n_top_genes, keep_genes=MANUAL_MARKERS)

    step("5. PCA")
    sc.tl.pca(adata, svd_solver="arpack")

    step(f"6. subset: day in {KEEP_DAYS[0]}..{KEEP_DAYS[-1]} and batch_replicate == {KEEP_BATCH!r}")
    n0 = adata.n_obs
    adata = adata[adata.obs["day"].isin(KEEP_DAYS)]
    adata = adata[adata.obs["batch_replicate"] == KEEP_BATCH].copy()
    print(f"[subset] {n0:,} -> {adata.n_obs:,} cells")
    assign_tp(adata, adata.obs["day"].astype(int))

    step("7. UMAP")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)

    save(adata, args.out, "SC")


if __name__ == "__main__":
    main()
