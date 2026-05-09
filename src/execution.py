"""Execution algorithms running inside the LOB simulator.

Implements two strategies so we can measure implementation shortfall (IS)
against a naive market-order baseline — the standard execution-research
benchmark. Every number produced by these strategies is computed on the
same simulator that ``replay.py`` drives, so comparisons are fair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hawkes import HawkesParams, simulate, EVENT_ADD_BID, EVENT_ADD_ASK, EVENT_CANCEL_BID, EVENT_CANCEL_ASK
from .orderbook import Order, OrderBook
from .replay import random_limit_order


@dataclass
class ExecutionReport:
    name: str
    target_size: int
    filled_size: int
    vwap: float
    arrival_mid: float
    implementation_shortfall_bps: float
    num_child_orders: int
    num_fills: int


def _run_market_flow(book: OrderBook, events, rng, initial_mid: int) -> None:
    """Apply Hawkes-generated order flow to the book (shared utility)."""
    oid = 1
    for t, k in events:
        ref = book.mid_price() if book.mid_price() else initial_mid
        ts_ns = int(t * 1e9)
        if k == EVENT_ADD_BID:
            book.submit_limit(random_limit_order(rng, +1, int(ref), oid, ts_ns))
        elif k == EVENT_ADD_ASK:
            book.submit_limit(random_limit_order(rng, -1, int(ref), oid, ts_ns))
        elif k == EVENT_CANCEL_BID:
            if book._bids:
                for p in list(book._bids):
                    lvl = book._bids.get(p)
                    if lvl and lvl.orders:
                        book.cancel(lvl.orders[-1].order_id)
                        break
        elif k == EVENT_CANCEL_ASK:
            if book._asks:
                for p in list(book._asks):
                    lvl = book._asks.get(p)
                    if lvl and lvl.orders:
                        book.cancel(lvl.orders[-1].order_id)
                        break
        oid += 1


def run_market_order_baseline(params: HawkesParams, total_size: int, T: float,
                              side: int = +1, seed: int = 0,
                              initial_mid: int = 10000) -> ExecutionReport:
    """Fire the whole parent order as a single marketable limit at time 0."""
    rng = np.random.default_rng(seed)
    book = OrderBook()
    events = simulate(params, T, seed=seed)

    # Seed the book with some resting liquidity by running the first 200ms of flow.
    warmup_events = [(t, k) for (t, k) in events if t < 0.2]
    remaining_events = [(t, k) for (t, k) in events if t >= 0.2]
    _run_market_flow(book, warmup_events, rng, initial_mid)

    arrival_mid = book.mid_price() or initial_mid
    price = int(arrival_mid + (50 * side))  # aggressive limit
    parent = Order(order_id=999_000, side=side, price=price, size=total_size, timestamp=int(0.2e9))
    book.submit_limit(parent)

    _run_market_flow(book, remaining_events, rng, initial_mid)

    my_fills = [f for f in book.fills if (f.buy_id == 999_000 or f.sell_id == 999_000)]
    filled = sum(f.size for f in my_fills)
    if filled == 0:
        return ExecutionReport("MarketOrder", total_size, 0, 0.0, arrival_mid, 0.0, 1, 0)
    vwap = sum(f.price * f.size for f in my_fills) / filled
    slip = side * (vwap - arrival_mid) / arrival_mid * 1e4
    return ExecutionReport("MarketOrder", total_size, filled, vwap, arrival_mid, slip,
                           num_child_orders=1, num_fills=len(my_fills))


def run_twap(params: HawkesParams, total_size: int, T: float, num_slices: int = 10,
             side: int = +1, seed: int = 0, initial_mid: int = 10000) -> ExecutionReport:
    """Time-weighted average price: equal-size children at equal intervals, each marketable."""
    rng = np.random.default_rng(seed)
    book = OrderBook()
    events = simulate(params, T, seed=seed)

    slice_size = max(1, total_size // num_slices)
    slice_times = np.linspace(0.2, T - 0.05, num_slices)
    oid = 999_100

    event_iter = iter(events)
    buf = []
    def drain_events_until(limit_t: float):
        nonlocal buf
        for t, k in event_iter:
            if t > limit_t:
                buf.append((t, k))
                return
            _run_market_flow(book, [(t, k)], rng, initial_mid)
        # exhausted
    # Run warmup
    drain_events_until(0.2)

    arrival_mid = book.mid_price() or initial_mid

    child_orders_fired = 0
    for slice_t in slice_times:
        # Drain events up to this slice.
        while buf and buf[0][0] <= slice_t:
            ev = buf.pop(0)
            _run_market_flow(book, [ev], rng, initial_mid)
        drain_events_until(slice_t)

        ref = book.mid_price() or initial_mid
        price = int(ref + 30 * side)  # aggressive limit
        book.submit_limit(Order(order_id=oid, side=side, price=price, size=slice_size,
                                timestamp=int(slice_t * 1e9)))
        oid += 1
        child_orders_fired += 1

    # Drain remaining.
    drain_events_until(T)
    for ev in buf:
        _run_market_flow(book, [ev], rng, initial_mid)

    my_fills = [f for f in book.fills if (
        (f.buy_id >= 999_100 and f.buy_id < 999_100 + num_slices) or
        (f.sell_id >= 999_100 and f.sell_id < 999_100 + num_slices)
    )]
    filled = sum(f.size for f in my_fills)
    if filled == 0:
        return ExecutionReport("TWAP", total_size, 0, 0.0, arrival_mid, 0.0, child_orders_fired, 0)
    vwap = sum(f.price * f.size for f in my_fills) / filled
    slip = side * (vwap - arrival_mid) / arrival_mid * 1e4
    return ExecutionReport("TWAP", total_size, filled, vwap, arrival_mid, slip,
                           num_child_orders=child_orders_fired, num_fills=len(my_fills))
