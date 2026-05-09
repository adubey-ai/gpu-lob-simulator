"""Stochastic replay: run N perturbed copies of a Hawkes scenario in parallel.

The point is to estimate *distributions* over execution outcomes rather than
single point values. Given a strategy (e.g. market-order child slicing), we
can ask:

    "What would slippage have been if my order arrived 50us earlier?"
    "What's the 99th-percentile adverse selection on this VWAP schedule?"

These questions are what a real execution desk cares about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hawkes import HawkesParams, simulate, EVENT_ADD_BID, EVENT_ADD_ASK, EVENT_CANCEL_BID, EVENT_CANCEL_ASK
from .orderbook import Order, OrderBook


@dataclass
class ExecutionResult:
    target_size: int
    filled_size: int
    vwap: float
    slippage_bps: float     # vs arrival mid-price
    num_fills: int
    time_to_complete: float


def random_limit_order(rng: np.random.Generator, side: int, ref_price: int,
                       order_id: int, timestamp: int) -> Order:
    spread_ticks = int(rng.integers(-3, 4))
    price = ref_price + (spread_ticks if side > 0 else -spread_ticks)
    size = int(rng.integers(1, 20))
    return Order(order_id, side, price, size, timestamp)


def run_scenario(params: HawkesParams, T: float, my_order: Order,
                 arrival_offset_ns: int, seed: int, initial_mid: int = 10000) -> ExecutionResult:
    """Single replay: simulate order flow, inject my_order at a perturbed time, measure outcome."""
    rng = np.random.default_rng(seed)
    book = OrderBook()
    events = simulate(params, T, seed=seed)

    arrival_mid = None
    oid = 1
    my_injected = False
    my_arrival_t = arrival_offset_ns / 1e9

    for t, k in events:
        ref = book.mid_price() if book.mid_price() else initial_mid

        # Inject our order at its arrival time.
        if not my_injected and t * 1 >= my_arrival_t:
            arrival_mid = book.mid_price() or initial_mid
            my_order_to_send = Order(
                order_id=999_999,
                side=my_order.side,
                price=my_order.price,
                size=my_order.size,
                timestamp=int(t * 1e9),
            )
            book.submit_limit(my_order_to_send)
            my_injected = True

        ts_ns = int(t * 1e9)
        if k == EVENT_ADD_BID:
            book.submit_limit(random_limit_order(rng, +1, int(ref), oid, ts_ns))
        elif k == EVENT_ADD_ASK:
            book.submit_limit(random_limit_order(rng, -1, int(ref), oid, ts_ns))
        elif k == EVENT_CANCEL_BID:
            if book._bids:
                p = next(iter(book._bids))
                if book._bids[p].orders:
                    book.cancel(book._bids[p].orders[-1].order_id)
        elif k == EVENT_CANCEL_ASK:
            if book._asks:
                p = next(iter(book._asks))
                if book._asks[p].orders:
                    book.cancel(book._asks[p].orders[-1].order_id)
        oid += 1

    if not my_injected:
        arrival_mid = book.mid_price() or initial_mid
        book.submit_limit(my_order)
        my_injected = True

    # Aggregate fills of my order.
    my_fills = [f for f in book.fills if f.buy_id == 999_999 or f.sell_id == 999_999]
    total_sz = sum(f.size for f in my_fills)
    if total_sz == 0 or arrival_mid is None:
        return ExecutionResult(my_order.size, 0, 0.0, 0.0, 0, T)

    vwap = sum(f.price * f.size for f in my_fills) / total_sz
    side_sign = my_order.side
    slippage = side_sign * (vwap - arrival_mid) / arrival_mid * 1e4
    ttc = my_fills[-1].timestamp / 1e9 - (my_arrival_t)

    return ExecutionResult(
        target_size=my_order.size,
        filled_size=total_sz,
        vwap=vwap,
        slippage_bps=slippage,
        num_fills=len(my_fills),
        time_to_complete=max(ttc, 0.0),
    )


def stochastic_replay(params: HawkesParams, T: float, my_order: Order,
                      arrival_offset_ns: int, num_replays: int = 1000,
                      base_seed: int = 0) -> dict:
    """Run N replays and return summary statistics."""
    results = [run_scenario(params, T, my_order, arrival_offset_ns, base_seed + i)
               for i in range(num_replays)]

    slipp = np.array([r.slippage_bps for r in results])
    fills = np.array([r.filled_size for r in results])
    ttc = np.array([r.time_to_complete for r in results])

    return {
        "num_replays": num_replays,
        "arrival_offset_ns": arrival_offset_ns,
        "fill_rate_mean": float(fills.mean() / my_order.size),
        "fill_rate_p05": float(np.percentile(fills / my_order.size, 5)),
        "slippage_bps_mean": float(slipp.mean()),
        "slippage_bps_p95": float(np.percentile(slipp, 95)),
        "slippage_bps_p99": float(np.percentile(slipp, 99)),
        "time_to_complete_mean": float(ttc.mean()),
    }
