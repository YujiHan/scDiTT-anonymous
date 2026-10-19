from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--ds", nargs="*", default=C.DATASETS)
    ap.add_argument("--task", nargs="*", default=C.TASKS)
    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--cfg", nargs="*", type=float, default=None)
    ap.add_argument("--script", default="run_scditt_infer.py",
                    help="script each (ds, task) job runs (same CLI convention)")
    ap.add_argument("--tag", default="infer", help="log file prefix")
    args = ap.parse_args()

    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, task) for ds in args.ds for task in args.task]
    slots = [g for g in args.gpus for _ in range(args.per_gpu)]

    running: list[tuple[subprocess.Popen, str, int]] = []
    queue = list(jobs)
    t0 = time.time()
    while queue or running:
        for p, name, gpu in list(running):
            if p.poll() is not None:
                running.remove((p, name, gpu))
                slots.append(gpu)
                print(f"[launch] {name} finished rc={p.returncode} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
        while queue and slots:
            ds, task = queue.pop(0)
            gpu = slots.pop(0)
            name = f"{ds}_{task}_seed{args.train_seed}"
            cmd = [sys.executable, str(HERE / args.script),
                   "--ds", ds, "--task", task, "--gpu", str(gpu),
                   "--train_seed", str(args.train_seed)]
            if args.cfg:
                cmd += ["--cfg"] + [str(w) for w in args.cfg]
            log = C.LOGS_DIR / f"{args.tag}_{name}.log"
            fh = open(log, "w")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((p, name, gpu))
            print(f"[launch] {name} -> gpu{gpu} (log {log.name})", flush=True)
        time.sleep(5)
    print(f"[launch] all done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
