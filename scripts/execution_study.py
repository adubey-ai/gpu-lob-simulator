"""Execution-strategy study: TWAP vs market order, implementation shortfall with CIs + plots."""
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

from src import HawkesParams, run_market_order_baseline, run_twap


def ci95(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return float(x.mean()), float(x.min()), float(x.max())
    m = x.mean()
    h = 2.0 * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def main():
    mu = np.array([120.0, 120.0, 80.0, 80.0])
    alpha = np.array([
        [0.6, 0.2, 0.1, 0.0],
        [0.2, 0.6, 0.0, 0.1],
        [0.0, 0.0, 0.3, 0.1],
        [0.0, 0.0, 0.1, 0.3],
    ])
    params = HawkesParams(mu=mu, alpha=alpha, beta=10.0)
    assert params.stable()

    N_REPLAYS = 60
    T = 1.0
    PARENT_SIZE = 100

    print(f"Execution study: {N_REPLAYS} replays, T={T}s, parent size={PARENT_SIZE}")

    t0 = time.time()
    mo_slips = []
    twap_slips = []
    mo_fills = []
    twap_fills = []
    for seed in range(N_REPLAYS):
        mo = run_market_order_baseline(params, PARENT_SIZE, T, seed=seed)
        tw = run_twap(params, PARENT_SIZE, T, num_slices=10, seed=seed)
        mo_slips.append(mo.implementation_shortfall_bps)
        twap_slips.append(tw.implementation_shortfall_bps)
        mo_fills.append(mo.filled_size / PARENT_SIZE)
        twap_fills.append(tw.filled_size / PARENT_SIZE)

    print(f"  wall-time: {time.time()-t0:.1f}s\n")

    mo_m, mo_lo, mo_hi = ci95(np.array(mo_slips))
    tw_m, tw_lo, tw_hi = ci95(np.array(twap_slips))
    print(f"  {'Strategy':<20} {'fill rate':>10} {'IS mean (bps)':>15} {'95% CI':>22}")
    print(f"  {'MarketOrder':<20} {np.mean(mo_fills):>10.3f} {mo_m:>15.3f} {f'[{mo_lo:+.3f}, {mo_hi:+.3f}]':>22}")
    print(f"  {'TWAP (10 slices)':<20} {np.mean(twap_fills):>10.3f} {tw_m:>15.3f} {f'[{tw_lo:+.3f}, {tw_hi:+.3f}]':>22}")

    diff = np.array(mo_slips) - np.array(twap_slips)
    paired_m, paired_lo, paired_hi = ci95(diff)
    print(f"\n  paired Δ (MO − TWAP) per-seed: mean={paired_m:+.3f} bps  95% CI=[{paired_lo:+.3f}, {paired_hi:+.3f}]")

    # Plot: violin of IS for both strategies.
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    parts = ax.violinplot([mo_slips, twap_slips], showmedians=True, showmeans=True)
    colors = ["#ff8888", "#64ffda"]
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MarketOrder", "TWAP-10"])
    ax.set_ylabel("Implementation shortfall (bps, side-adjusted)")
    ax.set_title(f"Execution-strategy comparison, {N_REPLAYS} independent scenarios")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out / "implementation_shortfall.png", dpi=140)
    plt.close(fig)

    # Plot: empirical CDF of IS.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for data, label, color in [(mo_slips, "MarketOrder", "#ff8888"),
                                (twap_slips, "TWAP-10", "#64ffda")]:
        arr = np.sort(np.array(data))
        y = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, y, label=label, color=color, linewidth=2)
    ax.set_xlabel("Implementation shortfall (bps)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("IS distribution — TWAP vs MarketOrder")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "is_cdf.png", dpi=140)
    plt.close(fig)

    print(f"\nPlots written to {out}/")


if __name__ == "__main__":
    main()
