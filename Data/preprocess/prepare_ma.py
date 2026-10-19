#!/usr/bin/env python
"""MA -- human embryonic pancreas development (Ma et al. 2023, OMIX: OMIX001616).

    OMIX_BASE=<omix001616-dir-url> python Data/preprocess/prepare_ma.py

The raw data is the public OMIX001616 record; the script downloads it into `Data/MA/raw/` and no
account is required. The download URL is cleared in this anonymous submission: set OMIX_BASE to the
OMIX001616 download directory, or place these six files into `Data/MA/raw/` by hand:

    scRNA_allcell_afterQC_PCW4_6.txt.gz    gene x cell counts (PCW 4-6)
    scRNA_allcell_afterQC_PCW7_11.txt.gz   gene x cell counts (PCW 7-11)
    scRNA_meta_panc_epi_PCW4_11.txt        per-cell metadata (cell type / PCW / ...)
    scRNA_meta_epi_PCW4_5.txt              supplementary epithelial annotation for PCW4-5
    scRNA_meta_endocrine.txt               endocrine cell annotation (with pseudotime)
    scRNA_meta_extra.txt                   spare annotation table

Cell barcodes are written in two ways: the count-matrix header uses `AAACC.1_S37` (R/Seurat replaces the
hyphen with a dot when writing csv) while the metadata uses `AAACC-1_S37`; the script normalises both to
the hyphen form.

Pipeline:
    1. download the 6 OMIX files
    2. read both count matrices as cell x gene and concatenate (cached in raw/ma_counts.h5ad)
    3. merge the four metadata tables by priority (earlier tables win) into obs
    4. PCW -> obs['timepoint'] / obs['time']; mixed-type columns in obs are cast to strings
    5. drop cells with a missing PCW, PCW -> obs['tp'] (8 time points, 0..7),
       the raw PCW is kept in obs['pcw_real']
    6. normalize_total(1e4) + log1p + HVG(2000)
       -> Data/MA/ma_forecast.h5ad

Note: step 2 reads the dense txt tables fully into memory before converting to sparse, so peak memory
demand is high (tens of GB). Run it once on a large-memory machine; every later run reuses the cached
raw/ma_counts.h5ad.
"""
from __future__ import annotations

import gzip
import os
import re
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from _common import (assign_tp, banner, build_argparser, load_or_build,
                     normalize_log1p, sanitise_for_h5ad, save, select_hvg, step)

OUT_NAME = "ma_forecast.h5ad"

# Base URL of the OMIX001616 download directory. Cleared for the anonymous
# submission; set it via the OMIX_BASE environment variable, or download the six
# files listed below by hand into Data/MA/raw/.
OMIX_BASE = os.environ.get("OMIX_BASE", "")
OMIX_FILES = [
    ("OMIX001616-01.txt.gz", "scRNA_allcell_afterQC_PCW7_11.txt.gz"),
    ("OMIX001616-02.txt.gz", "scRNA_allcell_afterQC_PCW4_6.txt.gz"),
    ("OMIX001616-04.txt",    "scRNA_meta_panc_epi_PCW4_11.txt"),
    ("OMIX001616-05.txt",    "scRNA_meta_endocrine.txt"),
    ("OMIX001616-13.txt",    "scRNA_meta_extra.txt"),
    ("OMIX001616-18.txt",    "scRNA_meta_epi_PCW4_5.txt"),
]

META_PRIORITY = [
    "scRNA_meta_panc_epi_PCW4_11.txt",
    "scRNA_meta_epi_PCW4_5.txt",
    "scRNA_meta_endocrine.txt",
    "scRNA_meta_extra.txt",
]

DOT_BARCODE_RE = re.compile(r"\.(\d+_)")     # "AAACC.1_S37" -> "AAACC-1_S37"


def download_all(raw_dir: Path, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [(part, name) for part, name in OMIX_FILES
               if force or not (raw_dir / name).exists() or (raw_dir / name).stat().st_size == 0]
    if missing and not OMIX_BASE:
        raise SystemExit(
            "OMIX_BASE is empty and these files are missing from "
            f"{raw_dir}: {', '.join(n for _, n in missing)}\n"
            "Either set OMIX_BASE to the OMIX001616 download directory, or place the "
            "files there by hand (the OMIX part number of each is listed in OMIX_FILES).")
    for part, name in OMIX_FILES:
        out = raw_dir / name
        if out.exists() and out.stat().st_size > 0 and not force:
            print(f"[skip ] {name} ({out.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"[get  ] {OMIX_BASE + part} -> {name}")
        urllib.request.urlretrieve(OMIX_BASE + part, out)


def read_dense_counts(path: Path):
    print(f"[load ] {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    opener = gzip.open if path.suffix == ".gz" else open
    df = pd.read_csv(opener(path, "rb"), sep="\t", index_col=0, low_memory=False)
    print(f"[load ] {df.shape[0]:,} genes x {df.shape[1]:,} cells")
    X = sparse.csr_matrix(df.T.to_numpy().astype(np.float32))
    cells = [DOT_BARCODE_RE.sub(r"-\1", c) for c in df.columns.astype(str)]
    return ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(cells)),
                      var=pd.DataFrame(index=pd.Index(df.index.astype(str))))


def build_counts(raw_dir: Path):
    parts = []
    for path in sorted(raw_dir.glob("scRNA_allcell_afterQC_*.txt.gz")):
        a = read_dense_counts(path)
        m = re.search(r"PCW(\d+)_(\d+)", path.name)
        if m:
            a.obs["pcw_range"] = f"{m.group(1)}-{m.group(2)}"
        parts.append(a)
    if not parts:
        raise SystemExit(f"{raw_dir} contains no scRNA_allcell_afterQC_*.txt.gz")
    adata = ad.concat(parts, join="outer", merge="unique")
    adata.obs_names_make_unique()
    print(f"[load ] after merge {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in META_PRIORITY:
        path = raw_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path, sep="\t", index_col=0)
        df.index = df.index.astype(str)
        df.columns = [c.strip() for c in df.columns]
        if "sex" in df.columns and "gender" not in df.columns:
            df = df.rename(columns={"sex": "gender"})
        df = df[~df.index.duplicated(keep="first")]
        print(f"[meta ] {name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    index = frames[0].index
    for df in frames[1:]:
        index = index.union(df.index)
    columns: list[str] = []
    for df in frames:
        columns += [c for c in df.columns if c not in columns]

    merged = pd.DataFrame(index=index, columns=columns, dtype=object)
    for df in frames:
        rows = merged.index.intersection(df.index)
        for col in df.columns:
            todo = rows[merged.loc[rows, col].isna().to_numpy()]
            if len(todo):
                merged.loc[todo, col] = df.loc[todo, col].values
    return merged


def annotate_obs(adata, meta: pd.DataFrame) -> None:
    if meta.empty:
        print("[meta ] no usable metadata, skipping")
        return
    common = adata.obs.index.intersection(meta.index)
    print(f"[meta ] {len(common):,} / {adata.n_obs:,} cells matched to metadata")
    for col in meta.columns:
        if col not in adata.obs.columns:
            adata.obs[col] = pd.NA
        adata.obs.loc[common, col] = meta.loc[common, col].to_numpy()


def parse_pcw(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    m = re.search(r"(\d+)", str(value))
    return float(m.group(1)) if m else np.nan


def main() -> None:
    args = build_argparser("MA", OUT_NAME, __doc__).parse_args()
    banner("MA — ma_forecast (Ma et al. 2023, OMIX001616)")

    step("1. download the raw OMIX files")
    download_all(args.raw_dir)

    step("2. count matrices -> h5ad")
    adata = load_or_build(args.raw_dir / "ma_counts.h5ad",
                          lambda: build_counts(args.raw_dir), force=args.force)

    step("3. merge metadata")
    annotate_obs(adata, load_metadata(args.raw_dir))

    step("4. PCW -> obs['timepoint']")
    adata.obs["timepoint"] = adata.obs["PCW"].map(parse_pcw).astype(float)
    adata.obs["time"] = adata.obs["timepoint"].astype(str)
    print(adata.obs["timepoint"].value_counts().sort_index().to_string())
    sanitise_for_h5ad(adata)

    step("5. drop cells with a missing time point, PCW -> obs['tp']")
    keep = adata.obs["timepoint"].notna().to_numpy()
    print(f"[drop ] {adata.n_obs:,} -> {int(keep.sum()):,} cells")
    adata = adata[keep].copy()
    adata.obs["pcw_real"] = adata.obs["timepoint"].astype("float32")
    assign_tp(adata, adata.obs["pcw_real"])

    step("6. normalisation + log1p + HVG")
    adata.layers["counts"] = adata.X.copy()
    normalize_log1p(adata)
    adata = select_hvg(adata, args.n_top_genes)

    save(adata, args.out, "MA", compression="gzip")


if __name__ == "__main__":
    main()
