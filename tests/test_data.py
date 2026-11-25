import numpy as np

from targetbook.data import WEEK, MarketConfig, make_panel


def test_same_seed_same_panel_and_seeds_differ():
    a, b, c = make_panel(7), make_panel(7), make_panel(8)
    assert np.array_equal(a.returns, b.returns) and np.array_equal(a.signal, b.signal)
    assert not np.allclose(a.returns, c.returns)


def test_shapes_and_rebalance_calendar():
    cfg = MarketConfig()
    p = make_panel(1)
    assert p.returns.shape == (cfg.n_days, cfg.n_assets) == p.signal.shape == p.control.shape
    assert p.rebalance_days[0] == cfg.warmup_days
    assert set(np.diff(p.rebalance_days)) == {WEEK}
    assert p.rebalance_days[-1] < cfg.n_days - 1  # every hand-over has at least one day of returns after it


def test_realised_vols_are_close_to_the_configured_ones():
    cfg = MarketConfig()
    p = make_panel(3)
    realised = p.returns.std(axis=0) * np.sqrt(252)
    target = np.repeat(cfg.group_vols, cfg.per_group)
    assert np.all(np.abs(realised / target - 1) < 0.25)


def test_model_signal_predicts_next_day_and_control_does_not():
    """Pooled correlation of today's signal with tomorrow's vol-scaled return."""
    model, control = [], []
    for seed in (11, 12, 13):
        p = make_panel(seed)
        z = p.returns[1:] / p.sigma
        model.append(np.corrcoef(p.signal[:-1].ravel(), z.ravel())[0, 1])
        control.append(np.corrcoef(p.control[:-1].ravel(), z.ravel())[0, 1])
    assert min(model) > 0.01
    assert max(abs(c) for c in control) < 0.01


def test_alpha_drives_the_next_day_not_the_same_day():
    """The planted drift of day t comes from alpha[t-1]: removing it from returns
    leaves residuals that are uncorrelated with alpha[t-1]."""
    p = make_panel(5)
    mu = np.zeros_like(p.returns)
    mu[1:] = p.sigma * p.alpha[:-1]
    resid = (p.returns - mu)[1:] / p.sigma
    assert abs(np.corrcoef(p.alpha[:-1].ravel(), resid.ravel())[0, 1]) < 0.01
