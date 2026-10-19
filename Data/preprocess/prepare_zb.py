#!/usr/bin/env python
"""ZB -- zebrafish embryonic development (Farrell et al. 2018, URD).

    python Data/preprocess/zb.py

Raw inputs (place in `Data/ZB/raw/`; all three csv files are exported from the R object published by URD):

    zebrafish_raw_counts.csv    cell x gene raw counts   (URD `@count.data`, transposed)
    zebrafish_meta.csv          cell x metadata          (URD `@meta`, includes HPF)
    cell_groups_meta.csv        cell x cluster annotation (URD `@group.ids`)

The R code that exports these three csv files -- run once in R, after which this repository needs no R:

    obj <- readRDS("URD_Zebrafish_Object.rds")
    write.csv(as.matrix(t(obj@count.data)), "zebrafish_raw_counts.csv")
    write.csv(obj@meta,                     "zebrafish_meta.csv")
    write.csv(obj@group.ids,                "cell_groups_meta.csv")

Pipeline:
    1. combine the three csv files into an h5ad (cached in raw/zebrafish_raw_counts.h5ad)
    2. lower-case the gene names (zebrafish gene symbol convention)
    3. HPF -> obs['tp'] (12 time points, 0..11)
    4. QC + normalize_total(1e4) + log1p
    5. marker scoring, per-cell annotation of obs['celltype']
    6. HVG(2000) + marker injection
    7. PCA / UMAP -> Data/ZB/zebrafish_hvg_annotated.h5ad
"""
from __future__ import annotations

import pandas as pd
import scanpy as sc

from _common import (annotate, assign_tp, banner, build_argparser, flatten,
                     load_or_build, marker_scores, normalize_log1p, qc_filter,
                     save, select_hvg, step)

OUT_NAME = "zebrafish_hvg_annotated.h5ad"

MARKERS: dict[str, list[str]] = {
    "Blastomeres":   ["pou5f3", "nanog", "sox19b"],
    "Periderm":      ["krt4", "krt18", "tp63"],
    "Axial":         ["ntla", "shha", "col2a1a"],
    "Neural":        ["sox2", "sox3", "otx2"],
    "Muscle":        ["myod1", "mylz2", "actc1b"],
    "Hematopoietic": ["spi1b", "gata1a", "drl"],
    "Endoderm":      ["sox17", "foxa2", "gata5"],
    "PGC":           ["nanos3", "ddx4"],
    "Eye":           ["rx3", "six3b", "vsx2"],
    "Cephalic":      ["tcf21", "twist1a"],
    "Pronephros":    ["pax2a", "cdh17"],
    "Glial":         ["gfap", "slc1a3b"],
}
SCORE_THRESHOLD = 0.05


def build_raw(raw_dir):
    adata = sc.read_csv(raw_dir / "zebrafish_raw_counts.csv")
    meta = pd.read_csv(raw_dir / "zebrafish_meta.csv", index_col=0)
    groups = pd.read_csv(raw_dir / "cell_groups_meta.csv", index_col=0)
    adata.obs = pd.concat([meta, groups.loc[meta.index, :]], axis=1)
    print(f"[load ] {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def main() -> None:
    args = build_argparser("ZB", OUT_NAME, __doc__).parse_args()
    banner("ZB — zebrafish (Farrell et al. 2018, URD)")

    step("1. raw csv -> h5ad")
    adata = load_or_build(args.raw_dir / "zebrafish_raw_counts.h5ad",
                          lambda: build_raw(args.raw_dir), force=args.force)

    step("2. gene name normalisation")
    adata.var_names = adata.var_names.str.lower()
    adata.var_names_make_unique()

    step("3. HPF -> obs['tp']")
    assign_tp(adata, adata.obs["HPF"])

    step("4. QC + normalisation + log1p")
    adata = qc_filter(adata, min_genes=200)
    normalize_log1p(adata)

    step("5. marker scoring -> obs['celltype']")
    adata.obs["celltype"] = annotate(marker_scores(adata, MARKERS),
                                     threshold=SCORE_THRESHOLD)
    print(adata.obs["celltype"].value_counts().to_string())

    step("6. HVG + marker injection")
    adata = select_hvg(adata, args.n_top_genes, keep_genes=flatten(MARKERS))

    step("7. PCA / UMAP")
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)

    save(adata, args.out, "ZB")


if __name__ == "__main__":
    main()
