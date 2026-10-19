"""Verifies the evaluation scoring wrapper scoring.py (eval-tp selection / organisation of the 5 metrics / seed aggregation).

- eval_tps / aggregate are pure python/numpy and always run.
- score needs scipy + scikit-learn (MMD/Pearson/W1/PCA); torch/geomloss is skipped via include_w2=False,
  and the test is skipped automatically when scipy/sklearn are missing.
- If the test environment can import reference_baseline's evaluator, the results are additionally
  cross-checked against it (they should agree bit for bit at EVAL_SEED=0); otherwise it is skipped.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import scoring


def test_eval_tps_interp_extrap():
    assert scoring.eval_tps("interpolation", 12, [4, 6, 8]) == [4, 6, 8]
    assert scoring.eval_tps("extrapolation", 11, [8, 9, 10]) == [8, 9, 10]
    assert scoring.eval_tps("all", 3, [0, 1, 2]) == [0, 1, 2]


def test_eval_tps_extrapolation_steps_schemes():
    assert scoring.eval_tps("extrapolation_steps", 12, [9, 10, 11],
                            scheme="alltp") == list(range(12))
    assert scoring.eval_tps("extrapolation_steps", 12, [9, 10, 11],
                            scheme="lasttp") == [11]
    with pytest.raises(ValueError):
        scoring.eval_tps("extrapolation_steps", 12, [11], scheme=None)
    with pytest.raises(ValueError):
        scoring.eval_tps("bogus_task", 12, [1])


def test_aggregate_mean_std():
    d1 = dict(zip(scoring.METRIC_KEYS, [1.0, 2.0, 3.0, 4.0, 5.0]))
    d2 = dict(zip(scoring.METRIC_KEYS, [3.0, 4.0, 5.0, 6.0, 7.0]))
    agg = scoring.aggregate([d1, d2])
    assert set(agg.keys()) == set(scoring.METRIC_KEYS)
    assert agg["w2_avg"]["mean"] == pytest.approx(2.0)
    assert agg["w2_avg"]["std"] == pytest.approx(1.0)  # population std (nanstd)


def _toy_data(seed=0, G=20, n_real=200, n_pred=150, tps=(0, 1, 2)):
    rng = np.random.default_rng(seed)
    real = {t: rng.normal(size=(n_real, G)).astype(np.float32) for t in tps}
    pred = {t: rng.normal(size=(n_pred, G)).astype(np.float32) for t in tps}
    return pred, real


def test_score_non_w2_metrics():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pred, real = _toy_data()
    res = scoring.score(pred, real, tps=[2], all_tps=[0, 1, 2], include_w2=False)
    assert set(res["avg"].keys()) == set(scoring.METRIC_KEYS)
    assert np.isnan(res["avg"]["w2_avg"])  # include_w2=False -> NaN
    for k in ["mmd_avg", "pearson_mean", "sampled_pearson", "pg_w1_mean"]:
        assert np.isfinite(res["avg"][k]), k
    assert set(res["per_tp"][2].keys()) == set(scoring.PER_TP_KEYS)
    assert np.isnan(res["per_tp"][2]["w2"])


def test_score_reproducible():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pred, real = _toy_data()
    a = scoring.score(pred, real, tps=[1, 2], all_tps=[0, 1, 2], include_w2=False)
    b = scoring.score(pred, real, tps=[1, 2], all_tps=[0, 1, 2], include_w2=False)
    for k in scoring.METRIC_KEYS:
        assert a["avg"][k] == pytest.approx(b["avg"][k], rel=0, abs=0) or \
               np.isnan(a["avg"][k])


def test_score_matches_reference_evaluator():
    """Cross-check against reference_baseline/common/evaluator.evaluate (the 4 non-W2 metrics)."""
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    root = Path(__file__).resolve().parents[2] / "reference_baseline"
    runs = root / "baseline_runs"
    if not runs.exists():
        pytest.skip("reference_baseline not present")
    for p in (str(runs), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from common import evaluator as ref_eval  # noqa: E402
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot import reference evaluator: {e}")

    pred, real = _toy_data(seed=1, G=15, n_real=120, n_pred=90)
    tps, all_tps = [1, 2], [0, 1, 2]
    mine = scoring.score(pred, real, tps, all_tps, include_w2=False)
    theirs = ref_eval.evaluate(pred, real, tps, all_tps, include_w2=False)
    for k in ["mmd_avg", "pearson_mean", "sampled_pearson", "pg_w1_mean"]:
        assert mine["avg"][k] == pytest.approx(theirs["avg"][k], rel=1e-9, abs=1e-9), k
