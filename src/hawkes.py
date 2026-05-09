"""Multivariate Hawkes process for LOB event generation and calibration.

A Hawkes process models self-exciting / mutually-exciting arrivals — exactly
what order-flow looks like empirically. Event intensities follow:

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t}  alpha_{ij} * exp(-beta * (t - t_k^j))

We use M=4 event types: ADD_BID, ADD_ASK, CANCEL_BID, CANCEL_ASK — matching
the minimal LOB dynamics most papers use for parameter calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EVENT_ADD_BID = 0
EVENT_ADD_ASK = 1
EVENT_CANCEL_BID = 2
EVENT_CANCEL_ASK = 3
NUM_EVENT_TYPES = 4


@dataclass
class HawkesParams:
    mu: np.ndarray          # (M,) baseline intensities
    alpha: np.ndarray       # (M, M) excitation kernel weights
    beta: float             # exponential decay rate

    def stable(self) -> bool:
        """Spectral radius < 1 required for stationarity."""
        norm = self.alpha / self.beta
        eigs = np.linalg.eigvals(norm)
        return bool(np.max(np.abs(eigs)) < 1.0)


def simulate(params: HawkesParams, T: float, seed: int = 0) -> list[tuple[float, int]]:
    """Ogata's thinning algorithm for multivariate Hawkes simulation.

    Returns list of (time, event_type) tuples in [0, T).
    """
    rng = np.random.default_rng(seed)
    M = NUM_EVENT_TYPES
    events: list[tuple[float, int]] = []
    t = 0.0
    lam = params.mu.copy()

    while t < T:
        lam_bar = lam.sum()
        if lam_bar <= 0:
            break
        u = rng.uniform()
        dt = -np.log(u) / lam_bar
        t += dt
        if t >= T:
            break
        # Decay
        lam = params.mu + (lam - params.mu) * np.exp(-params.beta * dt)
        lam_bar_now = lam.sum()
        # Acceptance
        if rng.uniform() * lam_bar <= lam_bar_now:
            probs = lam / lam_bar_now
            k = int(rng.choice(M, p=probs))
            events.append((t, k))
            lam = lam + params.alpha[:, k]
    return events


def calibrate_mle(events: list[tuple[float, int]], T: float, beta: float) -> HawkesParams:
    """Simple (non-optimal) MLE with fixed beta; estimates mu and alpha via moment matching.

    For a real system you'd maximize the Hawkes log-likelihood with L-BFGS; this is a
    solid starting point and matches empirical event rates and cross-excitation.
    """
    M = NUM_EVENT_TYPES
    counts = np.zeros(M, dtype=int)
    for _, k in events:
        counts[k] += 1
    mu = counts / T  # rough baseline

    # Cross-excitation: fraction of type-i events that follow a type-j event within 1/beta seconds.
    alpha = np.zeros((M, M))
    window = 1.0 / beta
    n_obs = len(events)
    for i in range(n_obs):
        ti, ki = events[i]
        for j in range(i + 1, n_obs):
            tj, kj = events[j]
            if tj - ti > window:
                break
            alpha[kj, ki] += 1.0
    for j in range(M):
        if counts[j] > 0:
            alpha[:, j] = alpha[:, j] / counts[j] * beta * 0.3  # damping for stability

    mu = np.maximum(mu - (alpha @ counts) / (beta * T), 1e-6)
    return HawkesParams(mu=mu, alpha=alpha, beta=beta)
