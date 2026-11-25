"""Synthetic daily panel: asset returns with a planted, modest edge, a model signal
that sees part of it, and a no-edge control signal with the same dynamics.

Every asset's expected daily return is ``mu[t, i] = sigma_i * a[t, i]``, where
``a`` is a slow AR(1) "true alpha" in units of daily volatility. The model signal
observed at the close of day t is ``a[t] + n[t]`` (n is a faster AR(1) noise), so
it carries information about returns from t + 1 onward. The control signal is an
independent AR(1) with the same persistence and the same noise, so it trades the
same way but knows nothing.

Returns: ``r[t] = sigma_i * a[t - 1] + sigma_i * v[t] * eps[t]``, so the drift
of day t comes from the alpha known at the previous close; ``v`` is a common stochastic-volatility multiplier with E[v^2] = 1 and
``eps`` has a one-market-factor plus one-factor-per-group correlation structure
with Student-t innovations scaled to unit variance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TRADING_DAYS = 252
WEEK = 5


@dataclass(frozen=True)
class MarketConfig:
    n_years: int = 11  # one warm-up year + ten evaluated years
    warmup_days: int = TRADING_DAYS
    group_names: tuple[str, ...] = ("equity", "rates", "commodities", "credit")
    per_group: int = 3
    group_vols: tuple[float, ...] = (0.16, 0.06, 0.20, 0.08)  # annualised
    rho_market: float = 0.15  # correlation from the common factor
    rho_group: float = 0.50  # total correlation within a group
    t_df: float = 5.0
    sv_halflife: float = 60.0  # days, log-vol AR(1)
    sv_sd: float = 0.30  # stationary sd of log vol
    alpha_sd: float = 0.03  # stationary sd of a, in units of daily vol
    alpha_halflife: float = 60.0  # days
    noise_sd: float = 0.04  # stationary sd of the signal noise
    noise_halflife: float = 10.0  # days

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def n_assets(self) -> int:
        return len(self.group_names) * self.per_group

    @property
    def n_days(self) -> int:
        return self.n_years * TRADING_DAYS


@dataclass
class Panel:
    returns: np.ndarray  # (T, N) daily simple returns
    signal: np.ndarray  # (T, N) model signal known at the close of day t
    control: np.ndarray  # (T, N) no-edge control signal, same dynamics
    alpha: np.ndarray  # (T, N) true a[t]; expected return of day t+1 is sigma * a[t]
    groups: np.ndarray  # (N,) group index per asset
    sigma: np.ndarray  # (N,) daily vol per asset (before stochastic vol)
    rebalance_days: np.ndarray  # (K,) day indices of weekly target hand-overs
    config: MarketConfig


def _ar1(rng: np.random.Generator, shape: tuple[int, int], halflife: float, sd: float) -> np.ndarray:
    """Stationary AR(1) along axis 0 with the given half-life (days) and sd."""
    phi = 0.5 ** (1.0 / halflife)
    innov = rng.standard_normal(shape) * sd * np.sqrt(1.0 - phi * phi)
    x = np.empty(shape)
    x[0] = rng.standard_normal(shape[1]) * sd
    for t in range(1, shape[0]):
        x[t] = phi * x[t - 1] + innov[t]
    return x


def _unit_t(rng: np.random.Generator, size, df: float) -> np.ndarray:
    return rng.standard_t(df, size=size) * np.sqrt((df - 2.0) / df)


def make_panel(seed: int, cfg: MarketConfig | None = None) -> Panel:
    cfg = cfg or MarketConfig()
    rng = np.random.default_rng(seed)
    T, N, G = cfg.n_days, cfg.n_assets, len(cfg.group_names)
    groups = np.repeat(np.arange(G), cfg.per_group)
    sigma = np.repeat(np.asarray(cfg.group_vols), cfg.per_group) / np.sqrt(TRADING_DAYS)

    # Draw every random block in a fixed order so that a seed always gives the same panel.
    alpha = _ar1(rng, (T, N), cfg.alpha_halflife, cfg.alpha_sd)
    ctrl_alpha = _ar1(rng, (T, N), cfg.alpha_halflife, cfg.alpha_sd)
    noise = _ar1(rng, (T, N), cfg.noise_halflife, cfg.noise_sd)
    ctrl_noise = _ar1(rng, (T, N), cfg.noise_halflife, cfg.noise_sd)
    logv = _ar1(rng, (T, 1), cfg.sv_halflife, cfg.sv_sd)
    vol_mult = np.exp(logv - cfg.sv_sd**2)  # E[vol_mult^2] = 1
    f_mkt = _unit_t(rng, (T, 1), cfg.t_df)
    f_grp = _unit_t(rng, (T, G), cfg.t_df)
    e_idio = _unit_t(rng, (T, N), cfg.t_df)

    eps = (
        np.sqrt(cfg.rho_market) * f_mkt
        + np.sqrt(cfg.rho_group - cfg.rho_market) * f_grp[:, groups]
        + np.sqrt(1.0 - cfg.rho_group) * e_idio
    )
    mu = np.zeros((T, N))
    mu[1:] = sigma * alpha[:-1]  # alpha known at close t-1 drives day t
    returns = mu + sigma * vol_mult * eps

    start = cfg.warmup_days
    rebalance_days = np.arange(start, T - 1, WEEK)
    return Panel(
        returns=returns,
        signal=alpha + noise,
        control=ctrl_alpha + ctrl_noise,
        alpha=alpha,
        groups=groups,
        sigma=sigma,
        rebalance_days=rebalance_days,
        config=cfg,
    )
