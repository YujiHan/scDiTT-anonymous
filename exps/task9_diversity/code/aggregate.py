from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

COLS = ["dataset", "task", "stage", "method", "cfg_weight", "train_seed",
        "K", "n_t", "N_t", "floor_real", "floor_self",
        "fid_raw", "div_raw", "fid_norm", "div_norm_self", "div_norm_real"]

VALUE_COLS = ["fid_norm", "div_norm_self", "div_norm_real",
              "fid_raw", "div_raw", "floor_real", "floor_self"]


def collect_raw() -> pd.DataFrame:
    parts = sorted(C.METRICS_DIR.glob("raw_*.csv"))
    if not parts:
        raise FileNotFoundError(f"no raw_*.csv in {C.METRICS_DIR}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    key = ["dataset", "task", "stage", "method", "cfg_weight", "train_seed"]
    df["_has_div"] = df["div_raw"].notna().astype(int)
    df = (df.sort_values(key + ["_has_div"])
            .drop_duplicates(key, keep="last")
            .drop(columns="_has_div")
            .sort_values(key).reset_index(drop=True))
    return df


def _mean_with_count(df: pd.DataFrame, keys: list[str],
                     count_name: str) -> pd.DataFrame:
    grp = df.groupby(keys, as_index=False, dropna=False)
    out = grp[VALUE_COLS].mean()
    cnt = grp.size().rename(columns={"size": count_name})
    return out.merge(cnt, on=keys, how="left")


def by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return _mean_with_count(
        df, ["dataset", "task", "method", "cfg_weight", "train_seed"], "n_stages")


def overall(dsl: pd.DataFrame) -> pd.DataFrame:
    per_seed = _mean_with_count(
        dsl, ["task", "method", "cfg_weight", "train_seed"], "n_datasets")
    g2 = ["task", "method", "cfg_weight"]
    grp = per_seed.groupby(g2, as_index=False, dropna=False)
    mean = grp[VALUE_COLS].mean()
    std = (grp[VALUE_COLS].std(ddof=1)
           .rename(columns={c: c + "_std" for c in VALUE_COLS}))
    n = grp.size().rename(columns={"size": "n_seeds"})
    return mean.merge(std, on=g2).merge(n, on=g2)


def main() -> int:
    df = collect_raw()
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.METRICS_DIR / "raw.csv", index=False)
    cols = [c for c in COLS if c in df.columns]
    df[cols].to_csv(C.METRICS_DIR / "aggregated.csv", index=False)

    dsl = by_dataset(df)
    dsl.to_csv(C.METRICS_DIR / "by_dataset.csv", index=False)
    ov = overall(dsl)
    ov.to_csv(C.METRICS_DIR / "overall.csv", index=False)

    print(f"[agg] raw.csv          {len(df):5d} rows  "
          f"({df.dataset.nunique()} ds x {df.task.nunique()} task x "
          f"{df.method.nunique()} method)")
    print(f"[agg] aggregated.csv   {len(df):5d} rows")
    print(f"[agg] by_dataset.csv   {len(dsl):5d} rows")
    print(f"[agg] overall.csv      {len(ov):5d} rows\n")
    show = ["task", "method", "cfg_weight", "fid_norm",
            "div_norm_self", "div_norm_real"]
    print(ov[show].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
