#!/usr/bin/env python
"""DR -- Drosophila embryonic development (Calderon et al. 2022, GEO: GSE190147).

    python Data/preprocess/dr.py

Raw inputs (place in `Data/DR/raw/`, taken directly from the GSE190147 supplementary):

    GSE190147_scirnaseq_gene_matrix.mtx.gz          gene x cell sparse counts
    GSE190147_scirnaseq_gene_matrix.rows.txt.gz     gene names (no header)
    GSE190147_scirnaseq_gene_matrix.columns.txt.gz  cell metadata (with header, includes time)

Pipeline:
    1. mtx + rows + columns -> h5ad, mapping the 11 time windows to obs['tp'] and their
       midpoints in hours to obs['hrs_real'] (cached in raw/DR_Calderon_2022_raw_tps.h5ad)
    2. QC + normalize_total(1e4) + log1p
    3. HVG(2000) + marker injection
    4. PCA / UMAP / Leiden clustering
    5. annotate obs['celltype'] and obs['celltype_germlayer'] from per-cluster mean marker scores
    6. stratified random sampling of 10% per time point (seed 42), reducing 450k cells to 45k
       -> Data/DR/DR_Calderon_2022_hvg_annotated_subsampled.h5ad
"""
from __future__ import annotations

import pandas as pd
import scanpy as sc

from _common import (annotate, assign_tp, banner, build_argparser, flatten,
                     load_or_build, marker_scores, normalize_log1p, qc_filter,
                     save, select_hvg, step, stratified_subsample)

OUT_NAME = "DR_Calderon_2022_hvg_annotated_subsampled.h5ad"

WINDOW_LIST = ["00_02", "01_03", "02_04", "03_07", "04_08", "06_10",
               "08_12", "10_14", "12_16", "14_18", "16_20"]
TIME_ORDER = [f"hrs_{w}" for w in WINDOW_LIST]

CELLTYPE_MARKERS: dict[str, list[str]] = {
    "Germ cells":      ["vas", "nos", "pgc", "zpg"],
    "Yolk":            ["Yp1", "Yp2", "Yp3"],
    "Amnioserosa":     ["hnt", "pnr", "Kr", "zen"],
    "Epidermis":       ["pdm1", "nub", "Osi23", "grh", "TwdlL", "Osi2"],
    "Tracheal system": ["trh", "vvl", "btl", "kni"],
    "Muscle_Somatic":  ["Mef2", "Mhc", "Mlc2", "twi", "Act57B", "Zasp52"],
    "Fat body":        ["srp", "Cg25C", "Fbp1", "vkg", "ppl"],
    "Salivary gland":  ["CrebA", "fkh", "Scr"],
    "Fore_Hindgut":    ["byn", "cad", "wg", "otp"],
    "Midgut":          ["GATAe", "dve", "lab", "Ser"],
    "Hemocytes":       ["hml", "crq", "Pxn", "Hml", "He"],
    "Crystal cells":   ["lz", "PPO1", "PPO2"],
    "CNS_Brain":       ["elav", "nSyb", "repo", "wor", "ase", "insc"],
    "PNS_Sense":       ["ato", "sens", "ct", "Poxn", "fne"],
    "Mixed ectoderm":  ["neurl", "Dl", "E(spl)m8-HLH", "m8"],
}

GERMLAYER_MARKERS: dict[str, list[str]] = {
    "Blastoderm":    ["slam", "nullo", "bnk", "zld"],
    "ExE embryo":    ["zen", "hnt", "pnr", "Kr"],
    "Mesoderm":      ["twi", "sna", "Mef2", "Mhc", "Poxm"],
    "Endoderm":      ["GATAe", "fkh", "srp", "lab", "dve"],
    "Neuroectoderm": ["elav", "nSyb", "repo", "SoxN", "D", "vnd"],
    "Ectoderm":      ["pdm1", "nub", "grh", "Osi23", "trh", "byn", "cad"],
}
GERMLAYER_THRESHOLD = 0.05
BLASTODERM_BONUS = 0.5
SAMPLE_RATIO = 0.1
SAMPLE_SEED = 42


def build_raw(raw_dir):
    """mtx (gene x cell) + rows + columns -> AnnData (cell x gene) + obs['tp']."""
    prefix = raw_dir / "GSE190147_scirnaseq_gene_matrix"
    adata = sc.read_mtx(f"{prefix}.mtx.gz").T
    genes = pd.read_csv(f"{prefix}.rows.txt.gz", header=None, sep="\t")
    adata.var_names = genes[0].astype(str).to_numpy()
    adata.var_names_make_unique()

    meta = pd.read_csv(f"{prefix}.columns.txt.gz", header=0, sep="\t")
    adata.obs = meta
    adata.obs_names = meta["cell"].to_numpy()
    adata.obs = adata.obs.drop(columns=["cell"])
    print(f"[load ] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    assign_tp(adata, adata.obs["time"], order=TIME_ORDER)

    # Numeric real time alongside the stage index: the midpoint of each
    # collection window, in hours (hrs_00_02 -> 1, hrs_03_07 -> 5, ...). The
    # windows are unevenly spaced, so this is what `real_time_key: hrs_real` in
    # configs/dataset/dr.yaml feeds to the continuous-time encoding.
    mid = {w: (int(w.split("_")[0]) + int(w.split("_")[1])) / 2.0
           for w in WINDOW_LIST}
    adata.obs["hrs_real"] = (adata.obs["time"].astype(str)
                             .str[len("hrs_"):].map(mid).astype("float32"))
    if adata.obs["hrs_real"].isna().any():
        raise ValueError("some obs['time'] values are outside WINDOW_LIST")
    print("[time ] hrs_real (window midpoints): "
          + ", ".join(f"{mid[w]:g}" for w in WINDOW_LIST))
    return adata


def main() -> None:
    args = build_argparser("DR", OUT_NAME, __doc__).parse_args()
    banner("DR — drosophila (Calderon et al. 2022, GSE190147)")

    step("1. raw mtx -> h5ad (with obs['tp'])")
    adata = load_or_build(args.raw_dir / "DR_Calderon_2022_raw_tps.h5ad",
                          lambda: build_raw(args.raw_dir), force=args.force)
    adata.var_names_make_unique()

    step("2. QC + normalisation + log1p")
    adata = qc_filter(adata, min_genes=200)
    normalize_log1p(adata)

    step("3. HVG + marker injection")
    adata = select_hvg(adata, args.n_top_genes, keep_genes=flatten(CELLTYPE_MARKERS))

    step("4. PCA / UMAP / Leiden")
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
    print(f"[leiden] {adata.obs['leiden'].nunique()} clusters")

    step("5a. cluster annotation -> obs['celltype']")
    adata.obs["celltype"] = annotate(marker_scores(adata, CELLTYPE_MARKERS),
                                     groupby=adata.obs["leiden"])
    print(adata.obs["celltype"].value_counts().to_string())

    step("5b. cluster annotation -> obs['celltype_germlayer']")
    gl_scores = marker_scores(adata, GERMLAYER_MARKERS)
    if "Blastoderm" in gl_scores.columns:
        gl_scores.loc[(adata.obs["tp"] == 0).to_numpy(), "Blastoderm"] += BLASTODERM_BONUS
    adata.obs["celltype_germlayer"] = annotate(gl_scores, groupby=adata.obs["leiden"],
                                               threshold=GERMLAYER_THRESHOLD)
    print(adata.obs["celltype_germlayer"].value_counts().to_string())

    step(f"6. stratified sampling by time point {SAMPLE_RATIO:.0%} (seed={SAMPLE_SEED})")
    adata = stratified_subsample(adata, SAMPLE_RATIO, seed=SAMPLE_SEED)

    save(adata, args.out, "DR")


if __name__ == "__main__":
    main()
