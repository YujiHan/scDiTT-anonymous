#!/usr/bin/env python3
"""Read-only inventory scan of all experiment result sources.

Reports, for every (method, dataset, task[, ex_step]), how many seeds have
VALID metrics (status ok + avg.w2_avg present). Does NOT modify anything.
"""
import json
import os
from collections import defaultdict

DATASETS = ["DR", "GA", "MA", "SC", "ZB"]
SEEDS_ALL = [42, 43, 44, 45, 46]
EX_LEVELS = [f"ex{i}" for i in range(7)]

SCDITT_ROOT = ""

# baseline method -> list of candidate roots (searched in order)
BASELINE_ROOTS = {
    "PI-SDE": [""],
    "scIMF": [""],
    "scNODE": [""],
    "MIOFlow": [""],
    "PRESCIENT": [""],
    "scDiffEq": [""],
}


def valid(path):
    """Return w2_avg if metrics file valid, else None."""
    if not os.path.isfile(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if d.get("status") not in ("ok", None):
        return None
    avg = d.get("avg")
    if not isinstance(avg, dict) or "w2_avg" not in avg:
        return None
    return avg["w2_avg"]


def scan_method(name, root):
    """root points at the method dir containing DS/task subdirs."""
    rows = []
    for ds in DATASETS:
        # interpolation / extrapolation
        for task in ["interpolation", "extrapolation"]:
            cnt = 0
            for s in SEEDS_ALL:
                p = os.path.join(root, ds, task, f"seed{s}", "metrics.json")
                if valid(p) is not None:
                    cnt += 1
            rows.append((name, ds, task, "", cnt))
        # extrapolation_steps ex0..ex6, alltp & lasttp
        for ex in EX_LEVELS:
            for kind, fn in [("alltp", "metrics_alltp.json"), ("lasttp", "metrics_lasttp.json")]:
                cnt = 0
                for s in SEEDS_ALL:
                    p = os.path.join(root, ds, "extrapolation_steps", ex, f"seed{s}", fn)
                    if valid(p) is not None:
                        cnt += 1
                rows.append((name, ds, f"steps_{ex}", kind, cnt))
    return rows


def main():
    print("=" * 70)
    print("scDiTT")
    print("=" * 70)
    for r in scan_method("scDiTT", SCDITT_ROOT):
        if r[4] > 0:
            print(f"  {r[1]:3} {r[2]:20} {r[3]:7} seeds={r[4]}")

    for name, roots in BASELINE_ROOTS.items():
        print("=" * 70)
        print(name)
        print("=" * 70)
        for root in roots:
            tag = "A/B/C" if "baseline_standard" not in root else "STANDARD"
            print(f"  [{tag}] {root}")
            for r in scan_method(name, root):
                if r[4] > 0:
                    print(f"      {r[1]:3} {r[2]:20} {r[3]:7} seeds={r[4]}")


if __name__ == "__main__":
    main()
