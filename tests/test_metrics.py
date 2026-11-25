import numpy as np

from targetbook.metrics import max_drawdown, sharpe


def test_sharpe_annualises_with_sqrt_252():
    r = np.array([0.01, -0.01] * 50) + 0.001
    assert np.isclose(sharpe(r), r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert sharpe(np.zeros(10)) == 0.0


def test_max_drawdown_of_a_simple_path():
    r = np.array([0.10, -0.50, 0.20])  # nav 1.1, 0.55, 0.66: peak 1.1, trough 0.55
    assert np.isclose(max_drawdown(r), 0.5)
    assert max_drawdown(np.array([0.01, 0.02])) == 0.0
