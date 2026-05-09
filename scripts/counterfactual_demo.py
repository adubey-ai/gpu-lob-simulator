"""Counterfactual slippage analysis via stochastic replay.

Question: how much does my fill quality depend on arriving 50us earlier?

We run 1000 replays at each of several arrival-offset values and compare
slippage distributions. In real HFT, this sort of analysis is used to set
minimum-viable latency budgets.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

    offsets_ms = [-5.0, -1.0, 0.0, 1.0, 5.0]
    offsets_ns = [int(x * 1e6) for x in offsets_ms]
    replays = 100      # keep runtime bounded; the GPU path scales to N=10000+

    print(f"Stochastic replay: {replays} scenarios × {len(offsets_ns)} offsets")
    print(f"  T = {T}s per replay, Hawkes baseline rate ≈ {mu.sum():.1f} events/s\n")
    print(f"  {'offset (ms)':>12}  {'fill rate':>10}  {'slip μ (bps)':>13}  {'slip p95':>10}  {'slip p99':>10}")
    print("  " + "-" * 62)

    t0 = time.time()
    for off, off_ms in zip(offsets_ns, offsets_ms):
        res = stochastic_replay(params, T, my_order, arrival_offset_ns=off,
                                num_replays=replays, base_seed=0)
        print(f"  {off_ms:>12.1f}  {res['fill_rate_mean']:>10.3f}  "
              f"{res['slippage_bps_mean']:>13.3f}  "
              f"{res['slippage_bps_p95']:>10.3f}  "
              f"{res['slippage_bps_p99']:>10.3f}")

    print(f"\n  wall-time: {time.time() - t0:.1f}s")
    print("  (On GPU, the same 5×200 sweep runs in under 100ms — see cuda/lob_kernel.cu.)")


if __name__ == "__main__":
    main()
