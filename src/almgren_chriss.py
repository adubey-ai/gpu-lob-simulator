"""Almgren-Chriss (2000) optimal execution schedule.

Classic textbook model — minimizes expected cost + λ·variance of execution cost
for a parent order of size X over horizon T, under:

    Permanent impact:  γ * (x_k / τ)
    Temporary impact:  η * (x_k / τ)     where x_k is the shares traded in slice k.

The optimal schedule follows a hyperbolic-cosine trajectory:

    x_k / X = cosh(κ(T - t_k)) / cosh(κT) / n_slices-like scaling,

with κ = sqrt(λ σ² / η̃), η̃ = η − γ·τ/2.

We implement the discrete-time version (Almgren-Chriss §4) which is what
real execution desks use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ACParams:
    total_shares: int       # parent size X
    horizon: float          # execution horizon T (seconds)
    num_slices: int         # N
    sigma: float            # per-step price volatility (ticks)
    eta: float              # temporary impact coefficient
    gamma: float            # permanent impact coefficient
    risk_aversion: float    # λ

    @property
    def tau(self) -> float:
        return self.horizon / self.num_slices

    @property
    def eta_tilde(self) -> float:
        return self.eta - self.gamma * self.tau / 2

    @property
    def kappa_squared(self) -> float:
        if self.eta_tilde <= 0:
            return 1e-8
        return self.risk_aversion * self.sigma ** 2 / self.eta_tilde

    @property
    def kappa(self) -> float:
        return np.sqrt(self.kappa_squared)


def optimal_trajectory(p: ACParams) -> np.ndarray:
    """Return remaining-shares trajectory x_k at slice boundaries k = 0..N.

    x_k / X = sinh(κ(T - t_k)) / sinh(κT)  — monotone decreasing to zero.
    """
    kT = p.kappa * p.horizon
    t = np.linspace(0.0, p.horizon, p.num_slices + 1)
    denom = np.sinh(kT) if kT > 1e-6 else kT
    return p.total_shares * np.sinh(p.kappa * (p.horizon - t)) / denom


def slice_sizes(p: ACParams) -> np.ndarray:
    """Shares to trade in each of the N slices."""
    traj = optimal_trajectory(p)
    return traj[:-1] - traj[1:]


def expected_cost(p: ACParams) -> float:
    """Expected transaction cost in shares × ticks (Almgren-Chriss eq. 15)."""
    traj = optimal_trajectory(p)
    slices = traj[:-1] - traj[1:]
    permanent = 0.5 * p.gamma * (p.total_shares ** 2)
    temporary = p.eta_tilde * np.sum(slices ** 2) / p.tau
    return permanent + temporary


def cost_variance(p: ACParams) -> float:
    """Variance of execution cost (Almgren-Chriss eq. 16)."""
    traj = optimal_trajectory(p)
    return (p.sigma ** 2) * p.tau * np.sum(traj[:-1] ** 2)
