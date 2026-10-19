from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "Data"

TARGET_SUM = 1e4
N_TOP_GENES = 2000


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def step(text: str) -> None:
    print(f"\n--- {text} ---", flush=True)


def build_argparser(dataset: str, out_name: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=DATA_ROOT / dataset / "raw",
                   help="raw file directory")
    p.add_argument("--out", type=Path, default=DATA_ROOT / dataset / out_name,
                   help="final h5ad output path (must match h5ad_path in configs/dataset)")
    p.add_argument("--n-top-genes", type=int, default=N_TOP_GENES, help="number of HVGs")
    p.add_argument("--force", action="store_true",
                   help="ignore existing intermediates and rebuild from the raw files")
    return p


def load_or_build(path: Path, build: Callable[[], "sc.AnnData"], *,
                  force: bool = False) -> "sc.AnnData":
    path = Path(path)
    if path.exists() and not force:
        print(f"[cache] reusing existing intermediate: {path}")
        return sc.read_h5ad(path)
    adata = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cache] writing intermediate: {path}")
    adata.write_h5ad(path)
    return adata


def assign_tp(adata, values, *, order: Sequence | None = None, key: str = "tp"):
    s = pd.Series(np.asarray(values), index=adata.obs_names)
    uniq = sorted(s.dropna().unique().tolist()) if order is None else list(order)
    mapping = {v: i for i, v in enumerate(uniq)}
    tp = s.map(mapping)
    if tp.isna().any():
        bad = sorted(s[tp.isna()].unique().tolist())[:10]
        raise ValueError(f"time annotations that cannot be mapped to a tp: {bad}")
    adata.obs[key] = tp.astype("int32").to_numpy()
    print(f"[tp   ] T={len(mapping)}  " +
          ", ".join(f"{v}->{i}" for v, i in mapping.items()))
    return mapping


def qc_filter(adata, *, min_genes: int, min_cells: int = 3,
              mt_prefix: str = "mt-", max_pct_mt: float = 10.0):
    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)
    n0, g0 = adata.n_obs, adata.n_vars
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if adata.obs["pct_counts_mt"].max() > 0:
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt, :].copy()
    print(f"[qc   ] {n0:,} x {g0:,} -> {adata.n_obs:,} x {adata.n_vars:,}")
    return adata


def normalize_log1p(adata, target_sum: float = TARGET_SUM):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def xmax(adata) -> float:
    X = adata.X
    if sparse.issparse(X):
        return float(X.data.max()) if X.nnz > 0 else 0.0
    return float(np.asarray(X).max())


def looks_log1p(adata, threshold: float = 50.0) -> bool:
    return xmax(adata) < threshold


def select_hvg(adata, n_top_genes: int = N_TOP_GENES,
               keep_genes: Iterable[str] = ()):
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars),
                                flavor="seurat")
    if keep_genes:
        present = [g for g in dict.fromkeys(keep_genes) if g in adata.var_names]
        adata.var.loc[present, "highly_variable"] = True
        print(f"[hvg  ] force-keeping {len(present)} marker genes")
    out = adata[:, adata.var["highly_variable"]].copy()
    print(f"[hvg  ] {adata.n_vars:,} -> {out.n_vars:,} genes")
    return out


def stratified_subsample(adata, frac: float, *, key: str = "tp", seed: int = 42):
    rng = np.random.RandomState(seed)
    labels = adata.obs[key]
    idx: list[int] = []
    for v in pd.unique(labels):
        pool = np.where(np.asarray(labels == v))[0]
        n = max(1, int(len(pool) * frac))
        idx.extend(rng.choice(pool, size=n, replace=False))
        print(f"[sub  ] {key}={v}: {len(pool):,} -> {n:,}")
    out = adata[np.asarray(idx)].copy()
    print(f"[sub  ] total {adata.n_obs:,} -> {out.n_obs:,} cells")
    return out


def marker_scores(adata, markers: Mapping[str, Sequence[str]],
                  ctrl_size: int = 50) -> pd.DataFrame:
    scores = pd.DataFrame(index=adata.obs_names)
    for ct, genes in markers.items():
        valid = [g for g in genes if g in adata.var_names]
        if not valid:
            print(f"[score] {ct}: no marker genes matched, skipping")
            continue
        tmp = f"_score_{ct}"
        sc.tl.score_genes(adata, gene_list=valid, score_name=tmp, ctrl_size=ctrl_size)
        scores[ct] = adata.obs.pop(tmp)
        print(f"[score] {ct}: {len(valid)}/{len(genes)} genes ({', '.join(valid)})")
    return scores


def annotate(scores: pd.DataFrame, *, groupby=None,
             threshold: float | None = None, unknown: str = "Unknown") -> pd.Series:
    if groupby is None:
        best_type = scores.idxmax(axis=1)
        best_val = scores.max(axis=1)
        if threshold is not None:
            n_unknown = int((best_val < threshold).sum())
            best_type = best_type.where(best_val >= threshold, unknown)
            print(f"[anno ] {n_unknown:,} cells score < {threshold}, recorded as {unknown}")
        return best_type.astype("category")

    g = np.asarray(groupby)
    mean = scores.groupby(g, observed=False).mean()
    best_type = mean.idxmax(axis=1)
    best_val = mean.max(axis=1)
    if threshold is not None:
        best_type = best_type.where(best_val >= threshold, unknown)
    for c in mean.index:
        print(f"[anno ] cluster {c}: {best_type[c]} (score={best_val[c]:.3f})")
    return pd.Series(g, index=scores.index).map(best_type).astype("category")


def flatten(markers: Mapping[str, Sequence[str]]) -> list[str]:
    return list(dict.fromkeys(g for genes in markers.values() for g in genes))


def sanitise_for_h5ad(adata) -> None:
    def fix(df: pd.DataFrame) -> None:
        for col in list(df.columns):
            s = df[col]
            if isinstance(s.dtype, pd.CategoricalDtype):
                cats = s.cat.categories
                if cats.dtype != object or any(not isinstance(c, str) for c in cats):
                    df[col] = s.astype(str).astype("category")
            elif pd.api.types.is_extension_array_dtype(s) or s.dtype == object:
                df[col] = s.astype(str).to_numpy(dtype=object)

    fix(adata.obs)
    fix(adata.var)
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)


def report(adata, name: str) -> None:
    tps = np.sort(np.unique(adata.obs["tp"].to_numpy()))
    expected = np.arange(len(tps))
    ok_tp = np.array_equal(tps, expected)
    X = adata.X
    m = xmax(adata)

    step(f"{name} self-check")
    print(f"  shape        : {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"  X            : {type(X).__name__}, dtype={X.dtype}, max={m:.3f} "
          f"({'log1p magnitude OK' if m < 50 else '! does not look like log1p'})")
    print(f"  obs['tp']    : T={len(tps)}, values {tps.tolist()} "
          f"({'contiguous 0..T-1 OK' if ok_tp else '! not a contiguous 0..T-1'})")
    print("  cells per time point:")
    for t, n in adata.obs["tp"].value_counts().sort_index().items():
        print(f"    tp={t:<3d} {n:>8,}")
    if not ok_tp:
        raise ValueError(f"{name}: obs['tp'] must be contiguous integers 0..T-1, got {tps.tolist()}")


def save(adata, out: Path, name: str, *, compression: str | None = None) -> None:
    report(adata, name)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    step(f"wrote {out}")
    adata.write_h5ad(out, compression=compression)
    print(f"[done ] {out}  ({out.stat().st_size / 1e6:.1f} MB)")
