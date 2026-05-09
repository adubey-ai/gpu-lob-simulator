"""CPU reference limit order book with price-time priority matching.

Used as the correctness oracle for the GPU implementation in cuda/. Every
match result (fills, partial fills, cancel acks) must be bit-identical between
this implementation and the GPU port.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Optional


@dataclass
class Order:
    order_id: int
    side: int          # +1 buy, -1 sell
    price: int         # integer ticks
    size: int
    timestamp: int     # nanoseconds since epoch


@dataclass
class Fill:
    buy_id: int
    sell_id: int
    price: int
    size: int
    timestamp: int


@dataclass
class PriceLevel:
    """Sorted queue of resting orders at one price level (FIFO)."""
    price: int
    orders: list[Order] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return sum(o.size for o in self.orders)

    def cancel(self, order_id: int) -> bool:
        for i, o in enumerate(self.orders):
            if o.order_id == order_id:
                self.orders.pop(i)
                return True
        return False


class OrderBook:
    """Two-sided LOB with price-time priority.

    Uses min-heap for asks, max-heap for bids (via negated price keys).
    O(log P) add, O(1) amortized match at top of book.
    """

    def __init__(self):
        self._bids: dict[int, PriceLevel] = {}
        self._asks: dict[int, PriceLevel] = {}
        self._bid_prices: list[int] = []   # max-heap: push -price
        self._ask_prices: list[int] = []   # min-heap
        self._orders: dict[int, Order] = {}
        self.fills: list[Fill] = []

    def best_bid(self) -> Optional[int]:
        while self._bid_prices:
            p = -self._bid_prices[0]
            if p in self._bids and self._bids[p].orders:
                return p
            heappop(self._bid_prices)
        return None

    def best_ask(self) -> Optional[int]:
        while self._ask_prices:
            p = self._ask_prices[0]
            if p in self._asks and self._asks[p].orders:
                return p
            heappop(self._ask_prices)
        return None

    def mid_price(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return (b + a) / 2.0

    def spread(self) -> Optional[int]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return a - b

    def _add_to_book(self, o: Order) -> None:
        book = self._bids if o.side > 0 else self._asks
        heap = self._bid_prices if o.side > 0 else self._ask_prices
        key = -o.price if o.side > 0 else o.price
        if o.price not in book:
            book[o.price] = PriceLevel(o.price)
            heappush(heap, key)
        book[o.price].orders.append(o)
        self._orders[o.order_id] = o

    def submit_limit(self, o: Order) -> list[Fill]:
        matches: list[Fill] = []
        remaining = o.size

        if o.side > 0:  # BUY
            while remaining > 0:
                a = self.best_ask()
                if a is None or o.price < a:
                    break
                level = self._asks[a]
                while level.orders and remaining > 0:
                    maker = level.orders[0]
                    traded = min(maker.size, remaining)
                    matches.append(Fill(
                        buy_id=o.order_id, sell_id=maker.order_id,
                        price=a, size=traded, timestamp=o.timestamp,
                    ))
                    maker.size -= traded
                    remaining -= traded
                    if maker.size == 0:
                        level.orders.pop(0)
                        self._orders.pop(maker.order_id, None)
                if not level.orders:
                    del self._asks[a]
        else:  # SELL
            while remaining > 0:
                b = self.best_bid()
                if b is None or o.price > b:
                    break
                level = self._bids[b]
                while level.orders and remaining > 0:
                    maker = level.orders[0]
                    traded = min(maker.size, remaining)
                    matches.append(Fill(
                        buy_id=maker.order_id, sell_id=o.order_id,
                        price=b, size=traded, timestamp=o.timestamp,
                    ))
                    maker.size -= traded
                    remaining -= traded
                    if maker.size == 0:
                        level.orders.pop(0)
                        self._orders.pop(maker.order_id, None)
                if not level.orders:
                    del self._bids[b]

        if remaining > 0:
            rest = Order(o.order_id, o.side, o.price, remaining, o.timestamp)
            self._add_to_book(rest)

        self.fills.extend(matches)
        return matches

    def cancel(self, order_id: int) -> bool:
        o = self._orders.get(order_id)
        if o is None:
            return False
        book = self._bids if o.side > 0 else self._asks
        level = book.get(o.price)
        if level is None:
            return False
        ok = level.cancel(order_id)
        if ok:
            self._orders.pop(order_id, None)
            if not level.orders:
                book.pop(o.price, None)
        return ok

    def snapshot(self, depth: int = 5) -> dict:
        bids = sorted(self._bids.keys(), reverse=True)[:depth]
        asks = sorted(self._asks.keys())[:depth]
        return {
            "bids": [(p, self._bids[p].total_size) for p in bids],
            "asks": [(p, self._asks[p].total_size) for p in asks],
            "mid": self.mid_price(),
            "spread": self.spread(),
        }
