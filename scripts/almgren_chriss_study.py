"""Almgren-Chriss optimal-execution study with plots.

Compares 4 execution strategies on identical simulated market flow:
  1. MarketOrder  — all at once
  2. TWAP         — uniform slices
  3. Almgren-Chriss (risk-averse)   — frontload-concave schedule
  4. Almgren-Chriss (risk-neutral)  — effectively TWAP (κ→0)

Reports efficient-frontier: cost vs variance as risk-aversion λ varies.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import (
    HawkesParams, run_market_order_baseline, run_twap,
)
from src.almgren_chriss import ACParams, optimal_trajectory, slice_sizes, expected_cost, cost_variance
from src.orderbook import OrderBook, Order
from src.hawkes import simulate
from src.execution import ExecutionReport, _run_market_flow


def run_ac_schedule(params: HawkesParams, ac: ACParams, side: int = +1,
                    seed: int = 0, initial_mid: int = 10000) -> ExecutionReport:
    """Fire the AC-optimal schedule against simulated market flow."""
    rng = np.random.default_rng(seed)
    book = OrderBook()
    events = simulate(params, ac.horizon, seed=seed)

    # Warmup: first 0.2s of flow seeds liquidity.
    warmup = [(t, k) for (t, k) in events if t < 0.2]
    rest = [(t, k) for (t, k) in events if t >= 0.2]
    _run_market_flow(book, warmup, rng, initial_mid)
    arrival_mid = book.mid_price() or initial_mid

    sizes = slice_sizes(ac).astype(int)
    # Adjust any rounding so total == parent size.
    sizes[-1] += ac.total_shares - int(sizes.sum())
    times = np.linspace(0.2, ac.horizon - 0.05, ac.num_slices)
    oid = 999_200

    rest_iter = iter(rest)
    buf = []
    def drain_until(limit_t: float):
        nonlocal buf
        for t, k in rest_iter:
            if t > limit_t:
                buf.append((t, k))
                return
            _run_market_flow(book, [(t, k)], rng, initial_mid)
        return

    child_count = 0
    for t_slice, sz in zip(times, sizes):
        while buf and buf[0][0] <= t_slice:
            _run_market_flow(book, [buf.pop(0)], rng, initial_mid)
        drain_until(t_slice)
        if sz <= 0:
            continue
        ref = book.mid_price() or initial_mid
        price = int(ref + 30 * side)
        book.submit_limit(Order(order_id=oid, side=side, price=price, size=int(sz),
                                timestamp=int(t_slice * 1e9)))
        oid += 1
        child_count += 1

    drain_until(ac.horizon)
    for ev in buf:
        _run_market_flow(book, [ev], rng, initial_mid)

    my_fills = [f for f in book.fills if (
        (f.buy_id >= 999_200 and f.buy_id < 999_200 + ac.num_slices) or
        (f.sell_id >= 999_200 and f.sell_id < 999_200 + ac.num_slices)
    )]
    filled = sum(f.size for f in my_fills)
    if filled == 0:
        return ExecutionReport("Almgren-Chriss", ac.total_shares, 0, 0.0, arrival_mid,
                               0.0, child_count, 0)
    vwap = sum(f.price * f.size for f in my_fills) / filled
    slip = side * (vwap - arrival_mid) / arrival_mid * 1e4
    return ExecutionReport("Almgren-Chriss", ac.total_shares, filled, vwap, arrival_mid,
                           slip, child_count, len(my_fills))


def ci95(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return float(x.mean()), 0.0, 0.0
    m = x.mean()
    h = 2.0 * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def main():
    np.random.seed(0)
    mu = np.array([120.0, 120.0, 80.0, 80.0])
    alpha = np.array([
        [0.6, 0.2, 0.1, 0.0],
        [0.2, 0.6, 0.0, 0.1],
        [0.0, 0.0, 0.3, 0.1],
        [0.0, 0.0, 0.1, 0.3],
    ])
    params = HawkesParams(mu=mu, alpha=alpha, beta=10.0)
    assert params.stable()

    PARENT = 100
    T = 1.0
    N_SLICES = 10
    N_REPLAYS = 40
    sigma = 0.5
    eta = 1e-3
    gamma = 1e-4

    # Visualize AC schedules at a few risk-aversion levels.
    lambdas = [1e-7, 1e-5, 1e-3, 1e-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lam in lambdas:
        ac = ACParams(total_shares=PARENT, horizon=T, num_slices=N_SLICES,
                      sigma=sigma, eta=eta, gamma=gamma, risk_aversion=lam)
        traj = optimal_trajectory(ac)
        ax.plot(np.linspace(0, T, N_SLICES + 1), traj / PARENT,
                "o-", label=f"λ = {lam:.0e}  (κ = {ac.kappa:.2f})", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Remaining shares / parent size")
    ax.set_title("Almgren-Chriss optimal trajectories — risk-aversion sweep")
    ax.legend()
    ax.grid(alpha=0.3)
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "ac_trajectories.png", dpi=140)
    plt.close(fig)

    # Run each strategy on N_REPLAYS paired seeds.
    strategies = {}

    mo_slips, mo_fills = [], []
    tw_slips, tw_fills = [], []
    ac_slips, ac_fills = [], []

    lam_used = 1e-4
    ac_param = ACParams(total_shares=PARENT, horizon=T, num_slices=N_SLICES,
                        sigma=sigma, eta=eta, gamma=gamma, risk_aversion=lam_used)

    t0 = time.time()
    for seed in range(N_REPLAYS):
        mo = run_market_order_baseline(params, PARENT, T, seed=seed)
        tw = run_twap(params, PARENT, T, num_slices=N_SLICES, seed=seed)
        ac_r = run_ac_schedule(params, ac_param, seed=seed)
        mo_slips.append(mo.implementation_shortfall_bps); mo_fills.append(mo.filled_size / PARENT)
        tw_slips.append(tw.implementation_shortfall_bps); tw_fills.append(tw.filled_size / PARENT)
        ac_slips.append(ac_r.implementation_shortfall_bps); ac_fills.append(ac_r.filled_size / PARENT)
    print(f"Wall-time: {time.time()-t0:.1f}s")

    print(f"\n{'Strategy':<30} {'fill':>8} {'IS mean':>10} {'95% CI':>22}")
    for name, slips, fills in [
        ("MarketOrder",    mo_slips, mo_fills),
        ("TWAP (10 slices)", tw_slips, tw_fills),
        (f"Almgren-Chriss (λ={lam_used:.0e})", ac_slips, ac_fills),
    ]:
        m, lo, hi = ci95(np.array(slips))
        print(f"  {name:<28} {np.mean(fills):>8.3f} {m:>10.3f} [{lo:+.3f}, {hi:+.3f}]")

    # Paired deltas against TWAP.
    d_ac = np.array(tw_slips) - np.array(ac_slips)
    m, lo, hi = ci95(d_ac)
    print(f"\n  paired Δ (TWAP − AC) per-seed: {m:+.3f} bps, 95% CI [{lo:+.3f}, {hi:+.3f}]")

    # IS comparison plot.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    parts = ax.violinplot([mo_slips, tw_slips, ac_slips], showmedians=True, showmeans=True)
    colors = ["#ff8888", "#ffaa88", "#64ffda"]
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["MarketOrder", "TWAP-10", "Almgren-Chriss"])
    ax.set_ylabel("Implementation shortfall (bps)")
    ax.set_title(f"Execution strategies, {N_REPLAYS} paired scenarios")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out / "ac_comparison.png", dpi=140)
    plt.close(fig)

    print(f"\nPlots → {out}/ac_trajectories.png, {out}/ac_comparison.png")


if __name__ == "__main__":
    main()
