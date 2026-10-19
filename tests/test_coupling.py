import numpy as np

from src.data.dataset import _sample_categorical_batch
from src.data.coupling import compute_coupling


def test_coupling_is_row_normalized():
    rng = np.random.default_rng(0)
    z_a = rng.normal(size=(30, 8)).astype(np.float32)
    z_b = rng.normal(size=(40, 8)).astype(np.float32)
    res = compute_coupling(z_a, z_b, epsilon=0.1, max_cells=100, rng=rng)
    pi = res["pi"]
    row_sum = pi.sum(axis=1)
    assert np.allclose(row_sum, 1.0, atol=1e-4)


def test_coupling_captures_nearness():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(20, 4)).astype(np.float32)
    z_a = base
    z_b = base + 0.01 * rng.normal(size=base.shape).astype(np.float32)
    res = compute_coupling(z_a, z_b, epsilon=0.01,
                           max_cells=100, normalize_cost=False, rng=rng)
    pi = res["pi"]
    argmax = pi.argmax(axis=1)
    correct = (argmax == np.arange(len(argmax))).sum()
    assert correct / len(argmax) > 0.5


def test_sample_categorical_batch_respects_probabilities():
    rng = np.random.default_rng(0)
    K, N = 10, 5
    probs = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        probs[k, k % N] = 1.0
    sampled = _sample_categorical_batch(probs, rng)
    for k in range(K):
        assert sampled[k] == k % N


def test_sample_categorical_batch_statistical():
    rng = np.random.default_rng(0)
    probs = np.full((1, 4), 0.25, dtype=np.float32)
    counts = np.zeros(4)
    N = 20000
    for _ in range(N):
        s = _sample_categorical_batch(probs, rng)
        counts[s[0]] += 1
    for c in counts:
        assert abs(c / N - 0.25) < 0.02
