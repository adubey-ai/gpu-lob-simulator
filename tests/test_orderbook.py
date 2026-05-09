"""Correctness tests for the CPU reference LOB.

These same cases should pass against the GPU port bit-for-bit.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Order, OrderBook, HawkesParams, simulate


def test_simple_cross():
    book = OrderBook()
    book.submit_limit(Order(1, -1, 100, 10, 0))    # ask 100x10
    book.submit_limit(Order(2, +1,  99, 10, 1))    # bid 99x10 (no cross)
    assert book.best_bid() == 99 and book.best_ask() == 100

    fills = book.submit_limit(Order(3, +1, 100, 4, 2))  # buy 4 @ 100 (crosses)
    assert len(fills) == 1
    assert fills[0].price == 100 and fills[0].size == 4
    assert book.best_ask() == 100
    print("  simple cross: match 4@100, residual ask 6 remains ✓")


def test_multi_level_sweep():
    book = OrderBook()
    book.submit_limit(Order(1, -1, 100, 5, 0))
    book.submit_limit(Order(2, -1, 101, 5, 1))
    book.submit_limit(Order(3, -1, 102, 5, 2))

    fills = book.submit_limit(Order(4, +1, 102, 12, 3))
    total = sum(f.size for f in fills)
    prices = [f.price for f in fills]
    assert total == 12
    assert prices == [100, 101, 102]
    print(f"  multi-level sweep: filled 12 across prices {prices} ✓")


def test_price_time_priority():
    book = OrderBook()
    book.submit_limit(Order(1, -1, 100, 3, 0))
    book.submit_limit(Order(2, -1, 100, 5, 1))  # later at same price

    fills = book.submit_limit(Order(3, +1, 100, 3, 2))
    assert fills[0].sell_id == 1    # older order hit first
    print("  price-time priority: older order at same price filled first ✓")


def test_cancel():
    book = OrderBook()
    book.submit_limit(Order(1, +1, 99, 5, 0))
    assert book.cancel(1)
    assert not book.cancel(1)       # idempotent
    assert book.best_bid() is None
    print("  cancel: order removed; repeat cancel is a no-op ✓")


def test_hawkes_stability():
    mu = [0.5] * 4
    alpha = [[0.1, 0.0, 0.0, 0.0],
             [0.0, 0.1, 0.0, 0.0],
             [0.0, 0.0, 0.1, 0.0],
             [0.0, 0.0, 0.0, 0.1]]
    import numpy as np
    p = HawkesParams(mu=np.array(mu), alpha=np.array(alpha), beta=1.0)
    assert p.stable()
    evs = simulate(p, T=100.0, seed=0)
    rate = len(evs) / 100.0
    assert 1.5 < rate < 3.0, f"Hawkes rate {rate} outside expected window"
    print(f"  Hawkes: {len(evs)} events in T=100s, empirical rate {rate:.2f}/s (theory baseline 2.0) ✓")


if __name__ == "__main__":
    print("Running LOB simulator tests…")
    test_simple_cross()
    test_multi_level_sweep()
    test_price_time_priority()
    test_cancel()
    test_hawkes_stability()
    print("All tests passed.")
