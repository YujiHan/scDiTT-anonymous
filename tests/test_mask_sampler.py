import numpy as np

from src.data.mask_sampler import DEFAULT_WEIGHTS, MaskSampler


def test_default_weights_sum_to_one():
    s = sum(DEFAULT_WEIGHTS.values())
    assert abs(s - 1.0) < 1e-9


def test_empirical_weights_match_config():
    train_tps = [0, 1, 2, 3, 5, 7, 9, 10, 11]  # ZB interpolation
    T = 12
    rng = np.random.default_rng(42)
    ms = MaskSampler(train_tps=train_tps, T=T, rng=rng)
    counts = {k: 0 for k in ms.pattern_names}
    N = 10000
    for _ in range(N):
        _, name = ms.sample_with_name()
        counts[name] += 1
    for name in ms.pattern_names:
        emp = counts[name] / N
        target = DEFAULT_WEIGHTS[name]
        assert abs(emp - target) < 0.02, \
            f"pattern {name}: empirical={emp:.3f}, target={target:.3f}"


def test_full_pattern_masks_all_train_tps():
    train_tps = [0, 1, 3, 5, 7, 9]
    T = 12
    rng = np.random.default_rng(0)
    ms = MaskSampler(train_tps=train_tps, T=T, rng=rng)
    mask = ms._sample_full()
    for t in train_tps:
        assert mask[t]
    for t in range(T):
        if t not in train_tps:
            assert not mask[t]


def test_tail_uniform_block_strided_return_right_shape():
    train_tps = [0, 1, 3, 5, 7, 9]
    T = 12
    rng = np.random.default_rng(1)
    ms = MaskSampler(train_tps, T, rng=rng)
    for fn in (ms._sample_uniform, ms._sample_tail,
               ms._sample_block, ms._sample_strided):
        m = fn()
        assert m.shape == (T,)
        assert m.dtype == bool
        for t in range(T):
            if t not in train_tps:
                assert not m[t], f"fn={fn.__name__} wrote to test_tp {t}"


def test_tail_mask_covers_right_half():
    train_tps = [0, 2, 4, 6, 8]
    T = 10
    rng = np.random.default_rng(2)
    ms = MaskSampler(train_tps, T, rng=rng)
    for _ in range(100):
        m = ms._sample_tail()
        idx_in_tps = [i for i, t in enumerate(train_tps) if m[t]]
        if idx_in_tps:
            assert idx_in_tps[-1] == len(train_tps) - 1
            assert idx_in_tps == list(range(idx_in_tps[0], len(train_tps)))
