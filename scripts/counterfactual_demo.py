"""Counterfactual slippage analysis via stochastic replay.

Question: how much does my fill quality depend on arrival-time jitter?

We run N replays at each of several arrival-offset values (−10 ms .. +10 ms)
and compare distributions. In real HFT, this analysis is used to set
minimum-viable latency budgets and size engineering investments.
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

from src import HawkesParams, Order, stochastic_replay


def main():
    # Realistic LOB-style Hawkes: ~500 events/sec (~Nasdaq mid-tier equity).
    mu = np.array([120.0, 120.0, 80.0, 80.0])
    alpha = np.array([
        [0.6, 0.2, 0.1, 0.0],
        [0.2, 0.6, 0.0, 0.1],
        [0.0, 0.0, 0.3, 0.1],
        [0.0, 0.0, 0.1, 0.3],
    ])
    params = HawkesParams(mu=mu, alpha=alpha, beta=10.0)
    assert params.stable(), "Hawkes parameters are not stationary"

    T = 1.0            # 1-second market snapshot (~500 events)
    my_order = Order(order_id=999_999, side=+1, price=10003, size=50, timestamp=0)

    offsets_ms = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]
    offsets_ns = [int(x * 1e6) for x in offsets_ms]
    replays = 80       # keep runtime bounded; the GPU path scales to N=10000+

    print(f"Stochastic replay: {replays} scenarios × {len(offsets_ns)} offsets")
    print(f"  T = {T}s per replay, Hawkes baseline rate ≈ {mu.sum():.1f} events/s\n")
    print(f"  {'offset (ms)':>12}  {'fill rate':>10}  {'slip μ (bps)':>13}  {'slip p95':>10}  {'slip p99':>10}")
    print("  " + "-" * 62)

    t0 = time.time()
    all_rows = []
    for off, off_ms in zip(offsets_ns, offsets_ms):
        res = stochastic_replay(params, T, my_order, arrival_offset_ns=off,
                                num_replays=replays, base_seed=0)
        all_rows.append({"off_ms": off_ms, **res})
        print(f"  {off_ms:>12.1f}  {res['fill_rate_mean']:>10.3f}  "
              f"{res['slippage_bps_mean']:>13.3f}  "
              f"{res['slippage_bps_p95']:>10.3f}  "
              f"{res['slippage_bps_p99']:>10.3f}")

    print(f"\n  wall-time: {time.time() - t0:.1f}s")
    print("  (On GPU, the same sweep runs in <100ms — see cuda/lob_kernel.cu.)")

    # Plot: fill-rate and slippage vs arrival offset.
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)

    offs = [r["off_ms"] for r in all_rows]
    fills = [r["fill_rate_mean"] for r in all_rows]
    slip_m = [r["slippage_bps_mean"] for r in all_rows]
    slip_p95 = [r["slippage_bps_p95"] for r in all_rows]
    slip_p99 = [r["slippage_bps_p99"] for r in all_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(offs, fills, "o-", color="#64ffda", linewidth=2, markersize=7)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Arrival offset (ms)")
    ax1.set_ylabel("Fill rate")
    ax1.set_title("Fill rate vs arrival-time jitter")
    ax1.grid(alpha=0.3)

    ax2.plot(offs, slip_m, "o-", color="#64ffda", label="mean", linewidth=2)
    ax2.plot(offs, slip_p95, "s--", color="#ffaa88", label="p95", linewidth=1.5)
    ax2.plot(offs, slip_p99, "^--", color="#ff4444", label="p99", linewidth=1.5)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Arrival offset (ms)")
    ax2.set_ylabel("Slippage (bps)")
    ax2.set_title("Slippage distribution vs arrival-time jitter")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "latency_tradeoff.png", dpi=140)
    plt.close(fig)
    print(f"  plot → {out}/latency_tradeoff.png")


if __name__ == "__main__":
    main()
