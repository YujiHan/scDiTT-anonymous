# Data

This directory holds the 5 single-cell time-series datasets used by scDiTT, together
with the preprocessing scripts (`preprocess/`) that build them from the public raw
data. **The data files are not distributed with the repository** (they are large);
download and preprocess them yourself. The repository only tracks the code under
`preprocess/` and this document.

## Directory layout

```
Data/
├── preprocess/                # preprocessing code (shipped with the repository)
│   ├── _common.py             # helpers shared by the five scripts
│   ├── prepare_zb.py          # one dataset = one script, raw files straight through to the final h5ad
│   ├── prepare_dr.py
│   ├── prepare_sc.py
│   ├── prepare_ga.py
│   └── prepare_ma.py
├── ZB/
│   ├── raw/                   # raw downloads (the scripts' input)
│   └── zebrafish_hvg_annotated.h5ad        # the scripts' output (input to training/evaluation)
├── DR/ …
├── SC/ …
├── GA/ …
├── MA/ …
```

The model code locates the data through `dataset.h5ad_path` in
`configs/dataset/{ds}.yaml`; the default `--out` of each preprocessing script already
matches that field one-to-one, so running them with default arguments is enough to
start training right away.

## Dataset overview

| Abbrev. | Dataset | T | Size after preprocessing | Raw data source |
|---|---|---|---|---|
| ZB | zebrafish embryonic development (Farrell et al. 2018) | 12 | 38,731 × 2,005 | the R object `URD_Zebrafish_Object.rds` from Broad SCP **SCP162** (**see the caveat below**) |
| DR | drosophila embryonic development (Calderon et al. 2022) | 11 | 44,931 × 2,040 | GEO **GSE190147** |
| SC | MEF reprogramming (Schiebinger et al. 2019) | 19 | 62,840 × 2,014 | GEO **GSE122662** + Broad's `cell_sets.gmt` / `FLE.txt` |
| GA | human germ cells (Garcia-Alonso et al. 2022) | 11 | 10,895 × 2,000 | CZ CELLxGENE (downloaded by the script; set `CELLXGENE_API`) |
| MA | human embryonic pancreas (Ma et al. 2023) | 8 | 19,224 × 2,000 | OMIX **OMIX001616** (downloaded by the script; set `OMIX_BASE`) |

### ⚠️ The two ZB accessions are not interchangeable

The data of Farrell et al. 2018 is published in two places, and the two hold
**different things** -- only one of them is directly usable:

| Accession | What it contains | Usable |
|---|---|---|
| Broad Single Cell Portal **SCP162** | the processed count matrix and the preprocessed URD R object `URD_Zebrafish_Object.rds` | ✅ this is what the repository uses |
| GEO **GSE106587** | aligned BAM files only, **no expression matrix** | ❌ |

GEO is where most people look first, but downloading `GSE106587` gives BAM files: you
would have to realign and requantify, and would not reproduce the matrix used here.
The SCP162 page reports 38,731 cells / 17,239 genes, matching the input of
`prepare_zb.py` (reduced to 38,731 × 2,005 after preprocessing).

URD is the trajectory-reconstruction R package introduced by that paper, and `.rds` is
its object format. The input of `prepare_zb.py` is three csv files exported from that
object; the export code is in the docstring at the top of the script and **only needs
to be run once in R, after which this repository needs no R at all**:

```r
obj <- readRDS("URD_Zebrafish_Object.rds")
write.csv(as.matrix(t(obj@count.data)), "zebrafish_raw_counts.csv")  # raw counts
write.csv(obj@meta,                     "zebrafish_meta.csv")        # includes HPF
write.csv(obj@group.ids,                "cell_groups_meta.csv")      # cluster annotation
```

Put those three csv files into `Data/ZB/raw/` and `prepare_zb.py` will run.

Biological meaning of `obs['tp']`:

| Abbrev. | `tp` values | Corresponding real time | Real-time column |
|---|---|---|---|
| ZB | 0–11 | HPF (hours post fertilisation), 12 sampling points | `HPF` |
| DR | 0–10 | 11 sampling windows of 2–4 hours (`hrs_00_02` … `hrs_16_20`) | `hrs_real` (window midpoint, hours) |
| SC | 0–18 | reprogramming days 0–18 | `day` |
| GA | 0–10 | the 11 weeks that actually occur between PCW (post-conception weeks) 9 and 21 | `pcw_real` |
| MA | 0–7 | PCW 4–11 | `pcw_real` |

`tp` is a rank transform, so it is always evenly spaced; the last column keeps the original
units and is unevenly spaced on ZB, DR and GA. It is the last column that feeds the model's
continuous-time encoding: `real_time_key` in `configs/dataset/*.yaml` names it for every
dataset. **A dataset therefore has to carry that column**, or training stops with a
`KeyError` — see the "Sampling time" section of the repository-root
[`README.md`](../README.md).

The single source of truth for the dataset constants is
[`src/data/splits.py`](../src/data/splits.py) (`ALL_TPS` / `TIME_KEY` / `SPLITS`).

## Preprocessing

**One script per dataset, in a single step**: it reads the raw files from
`Data/<DS>/raw/` and writes the `.h5ad` used for training directly, with no manual
chaining of intermediate stages.

```bash
# the default --raw-dir / --out of each script already point at the layout above,
# so arguments are usually unnecessary
python Data/preprocess/prepare_zb.py
python Data/preprocess/prepare_dr.py
python Data/preprocess/prepare_sc.py
CELLXGENE_API=<api-base>  python Data/preprocess/prepare_ga.py   # downloads from CELLxGENE
OMIX_BASE=<omix001616-url> python Data/preprocess/prepare_ma.py  # downloads from OMIX

# all five scripts share the same command line
python Data/preprocess/prepare_zb.py --raw-dir /path/to/raw --out /path/to/out.h5ad \
                                     --n-top-genes 2000 --force
```

The raw files each script needs -- and, for ZB, the export code to run once in R --
are documented in the docstring at the top of each script
(`python Data/preprocess/prepare_zb.py --help` shows it too).

**The uniform processing convention** (details in `preprocess/_common.py`):

1. **QC** -- annotate mitochondrial genes, then filter cells with few genes, lowly
   expressed genes, and cells with a mitochondrial fraction > 10%.
2. **Normalisation** -- `normalize_total(target_sum=1e4)` + `log1p`. The raw GA matrix
   is already normalised on the CELLxGENE side; the script detects this and skips the
   step.
3. **Feature selection** -- `highly_variable_genes(n_top_genes=2000, flavor="seurat")`.
   ZB / DR / SC additionally force lineage marker genes into the feature space (hence
   slightly more than 2000 genes), so that genes describing the key cell types are
   guaranteed to be present.
4. **Time points** -- map HPF / day / PCW / time window onto `obs['tp']` (contiguous
   integers `0 … T-1`); GA / MA also keep the original PCW in `obs['pcw_real']`.
5. **Self-check before writing** -- verify that `obs['tp']` really is a contiguous
   integer range `0 … T-1`, and print the shape, the magnitude of `X`, and the number
   of cells per time point.

**Intermediate caching**: the one-off conversion from the raw format to h5ad is slow
(SC merges 126 10x h5 files, DR is a 540k-cell mtx, MA is a dense txt with a peak
memory demand of tens of GB). The output of that step is cached under
`Data/<DS>/raw/` and reused on reruns; `--force` rebuilds it.

**Extra dependencies**: preprocessing needs `leidenalg` (Leiden clustering for DR) and
`umap-learn` (UMAP for ZB / DR / SC). Neither is in `requirements.txt` -- training and
evaluating the model does not need them, only rerunning preprocessing does:

```bash
pip install leidenalg umap-learn
```

## Data format requirements

If you want to plug in your own dataset without going through the scripts above, each
`.h5ad` must satisfy:

1. `adata.X` is an **already preprocessed** expression matrix -- HVG selection
   (G ≈ 2000) + `normalize_total` + `log1p`. The code in this repository performs **no
   further preprocessing** and reads `adata.X` directly (see `run/preprocess_vae.py`).
2. `adata.obs['tp']` holds integer time points, which must be a contiguous integer
   range `0 .. T-1` (all 5 datasets use the lower-case column name `tp`).
3. Sparse or dense matrices are both fine; the code calls `toarray()` as needed.
4. A numeric column holding the real acquisition time, constant within a time point and
   strictly increasing across them. `real_time_key` in `configs/dataset/<ds>.yaml` names it,
   and all five shipped configs do; it is what the continuous-time encoding reads. Set
   `real_time_key: null` if your own dataset has no such column, and the model falls back to
   the `tp` index.

The preprocessing convention is the one the baseline methods are evaluated under, so the
evaluation metrics are directly comparable with them.

## Once the data is in place

```bash
python run/preprocess_vae.py      --dataset ZB --task interpolation --seed 0
python run/preprocess_coupling.py --dataset ZB --task interpolation --seed 0
python run/train_model.py         --dataset zb --task interpolation --seed 0
```

All artefacts go to `results/` (not tracked); see the repository-root
[`README.md`](../README.md) for details.
