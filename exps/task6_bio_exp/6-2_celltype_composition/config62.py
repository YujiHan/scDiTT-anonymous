from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent
TASK6_DIR = EXP_DIR.parent
RESULTS = TASK6_DIR / "results" / "6-2_celltype_composition"
DIR_CLF = RESULTS / "classifier"
DIR_LABELS = RESULTS / "labels"
DIR_METRICS = RESULTS / "metrics"


def ensure_dirs() -> None:
    for d in (RESULTS, DIR_CLF, DIR_LABELS, DIR_METRICS):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#:
DATASET_SPECS: Dict[str, Tuple[str, str]] = {
    "ZB": ("ZB", "celltype"),
    "DR": ("DR", "celltype"),
    "GA": ("GA", "celltype"),
    "MA": ("MA", "cell_type"),
    "SC": ("SC", "cell_type"),
    "DRgl": ("DR", "celltype_germlayer"),
}
PRIMARY_DSKEYS: List[str] = ["ZB", "DR", "GA", "MA", "SC"]
ALL_DSKEYS: List[str] = PRIMARY_DSKEYS + ["DRgl"]

TASKS: List[str] = ["interpolation", "extrapolation"]
SEEDS: List[int] = [42, 43, 44, 45, 46]

LOW_POWER_DSKEYS = frozenset({"GA"})
LOW_POWER_NOTE = "insufficient power"

UNKNOWN_LABEL = "Unknown"

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
KNN_K = 15
N_PCS = 50
TRAIN_FRAC = 0.70
SPLIT_SEED = 0
PCA_FIT_MAX_CELLS = 20000
OOF_FOLDS = 5

GATE_ACC_MIN = 0.80
GATE_MACRO_F1_MIN = 0.50
GATE_RARE_RECALL_MIN = 0.30
GATE_ABSORB_MAX = 0.50

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
N_EFF_CAP = 2000
BOOTSTRAP_B = 100
CI_ALPHA = 0.05

RARE_P_THRESHOLD = 0.01    # p_ref < 1%
RARE_MIN_N = 20


def n_eff(n_real: int, cap: int = N_EFF_CAP) -> int:
    return int(min(cap, n_real))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#:
#:
#:
NEG_SHUFFLE_SCOPE = "pooled"

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#:
FROZEN_RARE: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {
    # --- ZB -----------------------------------------------------------------
    ("ZB", "interpolation", 4): {"Glial": 0.19, "Cephalic": 0.03, "Eye": 0.09,
                                 "Pronephros": 0.03, "Hematopoietic": 0.51,
                                 "Muscle": 0.26, "Axial": 0.19},
    ("ZB", "interpolation", 6): {"Glial": 0.10, "Pronephros": 0.05, "Eye": 0.61},
    ("ZB", "interpolation", 8): {"Glial": 0.04},
    ("ZB", "extrapolation", 10): {"PGC": 0.50, "Blastomeres": 0.74, "Hematopoietic": 0.37},
    ("ZB", "extrapolation", 11): {"PGC": 0.16, "Blastomeres": 0.36},
    # --- SC -----------------------------------------------------------------
    ("SC", "interpolation", 5): {"Epithelial": 0.05, "Stromal": 0.03, "IPS": 0.00},
    ("SC", "interpolation", 7): {"Stromal": 0.51, "IPS": 0.00},
    ("SC", "interpolation", 9): {"IPS": 0.04, "RG": 0.00, "OtherNeural": 0.00},
    ("SC", "interpolation", 11): {"IPS": 2.94, "OtherNeural": 0.10, "RG": 0.05, "MET": 0.39},
    ("SC", "extrapolation", 15): {"Astrocyte": None, "OPC": None, "Neuron": None, "Trophoblast": None},
    ("SC", "extrapolation", 16): {"Astrocyte": None, "OPC": None, "Neuron": None, "Trophoblast": None},
    ("SC", "extrapolation", 17): {"Astrocyte": None, "OPC": None, "Neuron": None, "Trophoblast": None},
    ("SC", "extrapolation", 18): {"Astrocyte": None, "OPC": None, "Neuron": None, "Trophoblast": None},
    # --- MA -----------------------------------------------------------------
    ("MA", "interpolation", 2): {"Delta": 0.13, "Dorsal MP": 0.50, "Trunk": 0.50},
    ("MA", "interpolation", 6): {"Epsilon": 0.29, "Duct": 1.14},
    ("MA", "extrapolation", 7): {"Epsilon": 1.02, "Delta": 1.73, "Beta": 2.39, "Alpha": 3.14},
    ("GA", "interpolation", 4): {"GC": 0.33, "pre_spermatogonia": 0.66},
    ("GA", "interpolation", 6): {"pre_spermatogonia": 2.02, "GC": 2.65},
    ("GA", "interpolation", 8): {"GC": 1.79, "pre_spermatogonia": 2.79},
    ("GA", "extrapolation", 8): {"GC": 1.79, "pre_spermatogonia": 2.79},
    ("GA", "extrapolation", 9): {"oogonia_meiotic": 0.61, "oocyte": 0.00, "pre_oocyte": 0.00},
    ("GA", "extrapolation", 10): {"GC_mitotic": 0.66},
    # --- DR / DRgl ---------------------------------------------------------
}

FROZEN_RARE_RANGES: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {
    ("SC", "extrapolation"): {
        "Astrocyte": (0.00, 0.32),
        "OPC": (0.05, 0.55),
        "Neuron": (0.37, 1.66),
        "Trophoblast": (0.00, 0.02),
    },
}

FROZEN_LISTED_KEYS = frozenset(k[:2] for k in FROZEN_RARE)

PREF_VERIFY_TOL_PCT = 0.006


def frozen_rare_for(dskey: str, task: str, tp: int) -> Dict[str, Optional[float]]:
    return dict(FROZEN_RARE.get((dskey, task, int(tp)), {}))


def has_frozen_list(dskey: str, task: str) -> bool:
    return (dskey, task) in FROZEN_LISTED_KEYS


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def rare_category(n_k: int, p_k: float) -> str:
    if n_k == 0:
        return "false_positive"
    if n_k < RARE_MIN_N:
        return "detection"
    return "recall"


def rule_implied_rare(counts: Dict[str, int], n_total: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c, n in counts.items():
        p = n / n_total if n_total else 0.0
        if n == 0:
            out[c] = "false_positive"
        elif p < RARE_P_THRESHOLD:
            out[c] = "detection" if n < RARE_MIN_N else "recall"
    return out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
NEG_ARMS = ("neg_mean_noise", "neg_shuffle")
POS_ARM = "pos_split_half"
OUR_ARM = "scDiTT"


#:
PREREG_ARMS: Tuple[str, ...] = (
    "neg_mean_noise", "neg_shuffle",
    "PI-SDE", "scIMF", "scNODE", "MIOFlow", "PRESCIENT", "scDiffEq",
    "scDiTT",
    "pos_split_half",
)


def arm_order() -> List[str]:
    from exps.task6_bio_exp.common import controls
    known = set(controls.ARM_ORDER)
    missing = [a for a in PREREG_ARMS if a not in known]
    if missing:
        raise RuntimeError(f"common/controls.py: ARM_ORDER is missing the 6-2 pre-registered arm {missing}")
    return [a for a in controls.ARM_ORDER if a in set(PREREG_ARMS)]


def labels_npz_path(dskey: str, task: str, seed: int, arm: str) -> Path:
    return DIR_LABELS / f"{dskey}_{task}_seed{seed}_{arm}_labels.npz"


__all__ = [
    "EXP_DIR", "RESULTS", "DIR_CLF", "DIR_LABELS", "DIR_METRICS", "ensure_dirs",
    "DATASET_SPECS", "PRIMARY_DSKEYS", "ALL_DSKEYS", "TASKS", "SEEDS",
    "LOW_POWER_DSKEYS", "LOW_POWER_NOTE", "UNKNOWN_LABEL",
    "KNN_K", "N_PCS", "TRAIN_FRAC", "SPLIT_SEED", "PCA_FIT_MAX_CELLS", "OOF_FOLDS",
    "GATE_ACC_MIN", "GATE_MACRO_F1_MIN", "GATE_RARE_RECALL_MIN", "GATE_ABSORB_MAX",
    "N_EFF_CAP", "BOOTSTRAP_B", "CI_ALPHA", "RARE_P_THRESHOLD", "RARE_MIN_N", "n_eff",
    "NEG_SHUFFLE_SCOPE", "FROZEN_RARE", "FROZEN_RARE_RANGES", "FROZEN_LISTED_KEYS",
    "PREF_VERIFY_TOL_PCT", "frozen_rare_for", "has_frozen_list",
    "rare_category", "rule_implied_rare",
    "NEG_ARMS", "POS_ARM", "OUR_ARM", "PREREG_ARMS", "arm_order", "labels_npz_path",
]
