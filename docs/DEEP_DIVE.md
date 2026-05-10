# GPU LOB Simulator — Deep Dive

A walk-through with enough depth to **defend this project in a quant/HFT
interview**. Covers the problem, the math, the implementation choices,
the results, and the questions an interviewer may ask.

---

## Table of contents

1. [The core problem](#problem)
2. [Architecture & data flow](#arch)
3. [Matching-engine internals](#matching)
4. [Hawkes process & calibration](#hawkes)
5. [Stochastic replay](#replay)
6. [Execution strategies](#exec)
7. [GPU kernel design](#cuda)
8. [Results & their interpretation](#results)
9. [Interview probes](#probes)

---

<a id="problem"></a>
## 1. The core problem

A quant execution desk has a parent order of, say, **100,000 shares of AAPL**
to execute over the next 10 minutes. The desk has to choose:

- How fast to trade (front-loaded? uniform? back-loaded?)
- How aggressively to cross the spread
- How to split the parent into child orders

Any single backtest on historical data gives *one* answer. But real markets
are stochastic — the liquidity that was there yesterday may not be there
today. What the desk actually needs is the **distribution** of execution
outcomes (slippage, fill rate, IS) under realistic market dynamics, so they
can choose a strategy that's robust across many possible futures.

This project builds the infrastructure to do exactly that:

1. A faithful LOB matching engine (price-time priority, partial fills, cancels).
2. A calibrated **multivariate Hawkes process** for realistic order flow —
   captures self-excitation (trades beget trades) and cross-excitation
   (asks get hit → more bids arrive).
3. A **stochastic replay** driver that runs N independent scenarios in
   parallel, each perturbing arrival times / sizes of events.
4. Execution strategies (MarketOrder, TWAP, Almgren-Chriss) running
   **inside** the simulator.
5. A GPU matching kernel for production scale.

<a id="arch"></a>
## 2. Architecture & data flow

```
                        ┌────────────────────────────┐
                        │  Hawkes Params (μ, α, β)   │
                        │  calibrated from real log  │
                        └──────────────┬─────────────┘
                                       │
                                       ▼
               ┌──────────────────────────────────────────┐
               │   Hawkes simulator (Ogata thinning)      │
               │   generates (t, event_type) list in [0,T)│
               └──────────────┬───────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Replay driver  — for each of N seeds, creates fresh LOB │
   │  and plays the event list through it, injecting my_order │
   │  at a perturbed arrival time                             │
   └─────┬─────────────────┬──────────────────┬───────────────┘
         │                 │                  │
         ▼                 ▼                  ▼
    ┌────────┐        ┌────────┐         ┌────────┐
    │ Book 1 │        │ Book 2 │   ...   │ Book N │
    │ fills  │        │ fills  │         │ fills  │
    └───┬────┘        └───┬────┘         └───┬────┘
        └─────────────────┴──────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Aggregate stats:  │
                │  fill rate, VWAP,  │
                │  slippage (bps),   │
                │  IS distribution   │
                └────────────────────┘
```

**Key insight**: each book is **independent** — perfect for data-parallel
GPU execution. On the GPU path (`cuda/lob_kernel.cu`), each thread block
owns one book; thousands of books run concurrently.

<a id="matching"></a>
## 3. Matching-engine internals

### Price-time priority

The gold-standard matching rule:

1. Incoming order tries to match against resting liquidity.
2. **Best price wins** (best ask for buy, best bid for sell).
3. **Within the same price level**, earlier-arriving orders fill first (FIFO).

### Data structures (`src/orderbook.py`)

```python
class PriceLevel:
    orders: list[Order]        # FIFO queue at one price
    @property total_size: int

class OrderBook:
    _bids: dict[int, PriceLevel]     # price -> level
    _asks: dict[int, PriceLevel]
    _bid_prices: list[int]           # max-heap (negated keys)
    _ask_prices: list[int]           # min-heap
    _orders: dict[int, Order]        # id -> order, for O(1) cancel lookup
```

**Why heaps + dicts?** O(log P) insertion, O(1) peek at top of book, O(1)
cancel by order id. P = number of distinct price levels, typically ≪ N.

### Matching loop (simplified)

```python
def submit_limit(o):
    remaining = o.size
    while remaining > 0:
        best = best_ask() if o.side > 0 else best_bid()
        if best is None or price_doesnt_cross(o, best): break
        maker = level[best].orders[0]          # FIFO head
        traded = min(maker.size, remaining)
        fills.append(Fill(...))
        maker.size -= traded
        remaining -= traded
        if maker.size == 0:
            level[best].orders.pop(0)
    if remaining > 0:
        add_to_book(Order(..., size=remaining))   # rest becomes liquidity
```

### What makes this a correctness oracle

Every test in `tests/test_orderbook.py` is a simple, verifiable scenario
(simple cross, multi-level sweep, price-time priority, cancel idempotency).
The CUDA port must produce **bit-identical fills** for the same inputs.
Same fills → same VWAP → same slippage → same conclusions.

<a id="hawkes"></a>
## 4. Hawkes process & calibration

### Why Hawkes?

Real LOB event arrivals exhibit two empirical facts:

1. **Self-excitation**: one trade tends to trigger more trades (liquidity
   discovery, information cascades).
2. **Cross-excitation**: an event of type A increases the probability of
   event type B (aggressive ask → more bids arrive).

Poisson processes capture neither. Hawkes processes capture both with a
clean parametric form.

### The model

For M event types (we use M=4: ADD_BID, ADD_ASK, CANCEL_BID, CANCEL_ASK),
the intensity of event type i at time t is:

```
λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t}  α_{ij} · exp(−β (t − t_k^j))
```

Baseline rate μ_i plus a kernel-weighted sum of previous events that
decays exponentially with rate β.

### Stationarity condition

The process is stationary iff the spectral radius of `α / β` is < 1. Our
`HawkesParams.stable()` checks this — essential before simulation, because
non-stationary Hawkes processes can explode.

### Ogata thinning simulation (`simulate`)

The standard algorithm for Hawkes sampling:

1. Compute `λ̄ = Σ λ_i(t)` (upper bound on total intensity at time t).
2. Draw `dt ~ Exp(λ̄)`, advance `t += dt`.
3. **Decay**: update intensities via `λ_i(t) = μ_i + (λ_i(t−dt) − μ_i) e^{−β dt}`.
4. **Accept** with probability `λ̄_now / λ̄_prev`. If accepted, draw event
   type k with probability `λ_k / λ̄_now`.
5. **Excite**: `λ += α[:, k]` (adds to each component's intensity).
6. Repeat until `t ≥ T`.

### Calibration (`calibrate_mle`)

A moment-matching calibrator:

- `μ_i` from empirical event rate.
- `α_{ij}` from fraction of type-i events occurring within `1/β` of a
  type-j event, normalized for counts.

**Honest caveat**: NOT full MLE; a production system would maximize the
Hawkes log-likelihood with L-BFGS. Moment matching is a sensible baseline
that's stable enough for this simulator.

<a id="replay"></a>
## 5. Stochastic replay

```python
for seed in range(N_REPLAYS):
    rng = numpy.random.default_rng(seed)
    events = simulate(params, T, seed=seed)
    book = OrderBook()
    for (t, k) in events:
        if t >= my_arrival_time and not my_injected:
            arrival_mid = book.mid_price()
            book.submit_limit(my_order)
            my_injected = True
        book.submit_limit(random_order_from_event(k, rng))
    my_fills = [f for f in book.fills if f involves my_order]
    slippage = compute_slippage(my_fills, arrival_mid)
```

**Counterfactual sweep** (`scripts/counterfactual_demo.py`): repeat the
above for arrival offsets −10 ms, −5 ms, 0, 5 ms, 10 ms; report fill-rate
and slippage distribution at each.

<a id="exec"></a>
## 6. Execution strategies

### MarketOrder (baseline)

Fire the entire parent as one aggressive limit order crossing the spread.

- **Pro**: zero signaling, immediate execution when liquidity is there.
- **Con**: walks through multiple price levels; paid-spread concentrated.

### TWAP (Time-Weighted Average Price)

Split the parent into N equal-size slices at equal time intervals.

- **Pro**: predictable; benchmark-friendly (TWAP is itself a benchmark).
- **Con**: Ignores intraday volatility patterns. Front- or back-loaded
  schedules often beat TWAP when there's regime structure.

### Almgren-Chriss (closed-form optimal)

Minimizes `E[cost] + λ · Var[cost]` under the linear-impact model:

- **Permanent impact**: `γ · x_k / τ` (price moves permanently per share)
- **Temporary impact**: `η · x_k / τ` (extra cost paid on this slice)

Solution (Almgren-Chriss 2000, §4):

```
x(t) / X   = sinh(κ(T − t)) / sinh(κT)            # remaining shares
κ²         = λ σ² / (η − γ τ / 2)
```

Three regimes to understand:

| Regime | κT | Behavior |
|---|---|---|
| Risk-neutral | κT → 0 | `sinh(κ(T−t))/sinh(κT) → (T−t)/T` — pure TWAP |
| Moderate | κT ≈ 1 | Front-loaded concave schedule |
| Very risk-averse | κT → ∞ | Execute almost all at t=0 (MarketOrder) |

**In our results, κT ≈ 0.3** because λσ² ≪ η. So AC collapses toward
TWAP — and the paired comparison (Δ = −0.18 bps, 95% CI crosses 0)
correctly shows statistical equivalence. **An interviewer who asks
"why didn't AC win?" is checking if you understand this limit behavior.**

<a id="cuda"></a>
## 7. GPU kernel design

File: `cuda/lob_kernel.cu`

### Parallelization strategy

```
blockIdx.x = book_id    (one thread block per independent LOB)
threadIdx  = within-block worker threads
```

This works because **different replays don't share state**. Same kernel,
different seeds, different books.

### Memory layout — SoA bucketed price levels

```cpp
struct PriceLevel {
    int order_ids[MAX_ORDERS_LEVEL];   // FIFO ring
    int sizes[MAX_ORDERS_LEVEL];
    int head, tail;                     // ring pointers
    int total_size;                     // O(1) level check
};
struct Book {
    PriceLevel bids[MAX_PRICE_LEVELS];   // bucketed by integer price offset
    PriceLevel asks[MAX_PRICE_LEVELS];
    int best_bid_offset, best_ask_offset;
};
```

**Why bucketed?** O(1) insertion and O(1) top-of-book access at the cost
of memory. Real-world LOBs have 10s of active price levels, so
`MAX_PRICE_LEVELS = 2048` is abundant.

### Warp-divergence mitigation

Matching is **inherently sequential** per book (order-of-events matters).
So thread 0 of each block does the matching; other threads handle
cooperative init (clearing price levels) and cooperative reductions (e.g.,
computing slippage). This keeps the critical path straight and minimizes
divergence.

### Expected throughput

For 10,000 books × 8,000 events each, on an A100:

- Single-book CPU: ~50 μs/event × 8000 = 400 ms per book → 4000 s total.
- GPU with 10,000 books in parallel: ~100 ms total for the whole sweep.

**40,000× speedup at batch size 10,000** is the promised benefit. (I can't
measure this on the current dev host; the number is what the design targets
based on the kernel profile.)

<a id="results"></a>
## 8. Results & their interpretation

### Counterfactual slippage (80 replays × 9 offsets)

```
 offset (ms)   fill rate   slip mean (bps)    slip p99
        -10.0       0.361        1.912            3.000
          0.0       0.361        1.912            3.000
          5.0       0.505        2.080            3.334
         10.0       0.543        1.799            3.013
```

**Reading the table**:

- Symmetric "early arrival" doesn't help — the LOB already has enough
  liquidity at t=0 to fill what it can.
- Late arrival gives more time for liquidity to arrive (+40% fill rate at
  +10 ms), at the cost of slightly worse p99 slippage.

**This is the latency/fill tradeoff** HFT desks quantify. A firm spending
$500k/year on a 1 ms feed upgrade wants numbers like this to justify ROI.

### Execution-strategy comparison (60 paired seeds)

| Strategy | Fill | IS mean (bps) | 95% CI |
|---|---|---|---|
| MarketOrder | 0.60 | +33.06 | [+28.11, +38.02] |
| TWAP-10 | 1.00 | +8.17 | [+6.44, +9.90] |
| Almgren-Chriss (λ=1e−4) | 0.99 | +8.58 | [+6.61, +10.54] |

**Δ MO − TWAP = +24.9 bps, 95% CI [+19.8, +30.0]** — TWAP wins decisively
and significantly.

**Δ TWAP − AC = −0.18 bps, 95% CI [−1.42, +1.05]** — AC and TWAP are
statistically tied at this noise/impact ratio, the expected κT → 0 limit.

<a id="probes"></a>
## 9. Interview probes — be ready for these

**Q: "Why does TWAP beat MarketOrder so much in your study?"**
A: MarketOrder consumes all available liquidity at the best price levels,
then walks up the book, paying wider spreads. TWAP waits for liquidity to
refresh, so it fills more of its size at better prices. Specifically in
my results, MarketOrder filled only 60% of the parent (ran out of resting
liquidity), vs TWAP's 100%.

**Q: "Why didn't Almgren-Chriss beat TWAP?"**
A: AC reduces to TWAP when κT → 0, where κ² = λσ² / (η − γτ/2). In my
parameterization, σ = 0.5 ticks and λ = 1e−4, so λσ² ≈ 2.5e−5, while
η ≈ 1e−3 — impact cost dominates. The AC trajectory becomes nearly
linear, ≈ TWAP. AC wins meaningfully only in high-vol regimes.

**Q: "Can your Hawkes calibrator handle non-stationary real markets?"**
A: As implemented, no — moment matching assumes stationarity. For real
markets I'd use a rolling-window MLE with L-BFGS, and for known
heteroskedasticity (open/close), piecewise-stationary Hawkes models.

**Q: "What if there's adverse selection?"**
A: Adverse selection (toxic flow informed about short-horizon moves) would
show up as negative post-trade drift on the passive side. My current
simulator doesn't model informed flow; a production extension would add a
"pinger" event type that fires before price moves, and measure AS as the
5-second post-trade mid change conditional on our fill.

**Q: "Why a GPU for LOB matching?"**
A: Matching itself is sequential per book, but stochastic replay runs
thousands of independent books. Batch size is what parallelizes. This is
classic SIMT workload: same kernel, different data, no synchronization
between blocks.

## References

- Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions",
  *J. of Risk*.
- Bacry, Mastromatteo, Muzy (2015), "Hawkes processes in finance".
- Cartea, Jaimungal, Penalva (2015), *Algorithmic and High-Frequency
  Trading*, Cambridge Univ. Press — Ch. 7 on execution.
- Hasbrouck (2007), *Empirical Market Microstructure*, OUP.
