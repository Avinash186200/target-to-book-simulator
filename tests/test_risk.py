import numpy as np

from targetbook.risk import ex_ante_vol, ledoit_wolf_corr, shrunk_cov, vol_cap


def _sample(T, n=8, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(0.4 * np.ones((n, n)) + 0.6 * np.eye(n))
    return rng.standard_normal((T, n)) @ L.T * np.linspace(0.005, 0.02, n)


def test_ledoit_wolf_intensity_is_in_unit_interval_and_falls_with_more_data():
    deltas = []
    for T in (30, 300, 30000):
        C, d = ledoit_wolf_corr(_sample(T))
        assert 0.0 <= d <= 1.0
        assert np.allclose(np.diag(C), 1.0)
        assert np.linalg.eigvalsh(C).min() > 0
        deltas.append(d)
    assert deltas[0] > deltas[1] > deltas[2]
    assert deltas[2] < 0.01


def test_ledoit_wolf_matches_the_textbook_loop():
    x = _sample(60, n=5, seed=3)
    z = (x - x.mean(0)) / x.std(0)
    S = z.T @ z / len(z)
    d2 = ((S - np.eye(5)) ** 2).sum()
    b2 = sum(((np.outer(r, r) - S) ** 2).sum() for r in z) / len(z) ** 2
    expected = min(b2, d2) / d2
    assert np.isclose(ledoit_wolf_corr(x)[1], expected)


def test_shrunk_cov_uses_only_the_trailing_window():
    x = _sample(400)
    c1, _ = shrunk_cov(x, 299)
    y = x.copy()
    y[300:] *= 5.0  # change the future
    c2, _ = shrunk_cov(y, 299)
    assert np.array_equal(c1, c2)


def test_vol_cap_scales_down_to_the_cap_and_never_up():
    cov = np.diag([0.01, 0.02]) ** 2
    w = np.array([1.0, 1.0])
    v = ex_ante_vol(w, cov)
    capped, hit = vol_cap(w, cov, v / 2)
    assert hit and np.isclose(ex_ante_vol(capped, cov), v / 2)
    same, hit = vol_cap(w, cov, v * 2)
    assert not hit and np.array_equal(same, w)
