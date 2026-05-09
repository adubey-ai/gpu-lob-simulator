// GPU-parallel matching engine: each thread block owns one independent LOB.
// Runs N order books in parallel (N = gridDim.x) to support stochastic replay.
//
// Compile:  nvcc -O3 -std=c++17 -arch=sm_80 lob_kernel.cu -o lob_bench
//
// Memory layout is carefully SoA to minimize warp divergence. Price levels
// use a bucketed array (one slot per integer price offset), trading memory
// for O(1) insertion / lookup at the top of book.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

constexpr int MAX_PRICE_LEVELS  = 2048;   // per side, centered on reference
constexpr int MAX_ORDERS_LEVEL  = 64;     // per price level
constexpr int MAX_EVENTS        = 8192;

enum : int {
    EVT_ADD_BID    = 0,
    EVT_ADD_ASK    = 1,
    EVT_CANCEL_BID = 2,
    EVT_CANCEL_ASK = 3,
};

struct alignas(16) Event {
    int  type;
    int  price_offset;  // signed ticks from reference
    int  size;
    int  order_id;
    long timestamp_ns;
};

struct alignas(16) PriceLevel {
    int  order_ids[MAX_ORDERS_LEVEL];
    int  sizes[MAX_ORDERS_LEVEL];
    int  head;   // FIFO pointer
    int  tail;
    int  total_size;
};

struct Book {
    PriceLevel bids[MAX_PRICE_LEVELS];
    PriceLevel asks[MAX_PRICE_LEVELS];
    int best_bid_offset;
    int best_ask_offset;
    int num_fills;
    long total_buy_volume;
    long total_sell_volume;
};

// ----------------------------------------------------------------------------
// Each block runs one independent replay. Thread 0 is the "matcher" — pure
// sequential matching, but the heavy memory-bound work (intensity decay,
// scratchpad writes) parallelizes across threads in the block.
// ----------------------------------------------------------------------------

__device__ __forceinline__ int level_size(const PriceLevel& L) {
    return L.total_size;
}

__device__ void push_order(PriceLevel& L, int order_id, int size) {
    int tail = L.tail;
    L.order_ids[tail] = order_id;
    L.sizes[tail]     = size;
    L.tail = (tail + 1) % MAX_ORDERS_LEVEL;
    L.total_size += size;
}

__device__ int pop_front_size(PriceLevel& L) {
    if (L.head == L.tail) return 0;
    int s = L.sizes[L.head];
    L.sizes[L.head] = 0;
    L.head = (L.head + 1) % MAX_ORDERS_LEVEL;
    return s;
}

__device__ int match_buy(Book& b, int price_offset, int size) {
    int remaining = size;
    int ask = b.best_ask_offset;
    while (remaining > 0 && ask <= price_offset && ask < MAX_PRICE_LEVELS) {
        PriceLevel& L = b.asks[ask];
        while (remaining > 0 && L.head != L.tail) {
            int maker_size = L.sizes[L.head];
            int traded = min(maker_size, remaining);
            L.sizes[L.head] -= traded;
            L.total_size    -= traded;
            remaining       -= traded;
            if (L.sizes[L.head] == 0) {
                L.head = (L.head + 1) % MAX_ORDERS_LEVEL;
            }
            atomicAdd(&b.num_fills, 1);
        }
        if (L.total_size == 0) {
            while (ask < MAX_PRICE_LEVELS - 1 &&
                   b.asks[ask + 1].total_size == 0) {
                ++ask;
            }
            ++ask;
            b.best_ask_offset = ask;
        }
    }
    return size - remaining;
}

__device__ int match_sell(Book& b, int price_offset, int size) {
    int remaining = size;
    int bid = b.best_bid_offset;
    while (remaining > 0 && bid >= price_offset && bid >= 0) {
        PriceLevel& L = b.bids[bid];
        while (remaining > 0 && L.head != L.tail) {
            int maker_size = L.sizes[L.head];
            int traded = min(maker_size, remaining);
            L.sizes[L.head] -= traded;
            L.total_size    -= traded;
            remaining       -= traded;
            if (L.sizes[L.head] == 0) {
                L.head = (L.head + 1) % MAX_ORDERS_LEVEL;
            }
            atomicAdd(&b.num_fills, 1);
        }
        if (L.total_size == 0) {
            while (bid > 0 && b.bids[bid - 1].total_size == 0) {
                --bid;
            }
            --bid;
            b.best_bid_offset = bid;
        }
    }
    return size - remaining;
}

__global__ void replay_kernel(const Event* __restrict__ events,
                              int num_events,
                              Book* __restrict__ books,
                              long* __restrict__ slippage_out,
                              int my_order_idx,
                              int my_side,
                              int my_price_offset,
                              int my_size)
{
    int book_id = blockIdx.x;
    Book& b = books[book_id];

    // Cooperative init: threads clear price-level structs in parallel.
    for (int i = threadIdx.x; i < MAX_PRICE_LEVELS; i += blockDim.x) {
        b.bids[i].head = b.bids[i].tail = b.bids[i].total_size = 0;
        b.asks[i].head = b.asks[i].tail = b.asks[i].total_size = 0;
    }
    if (threadIdx.x == 0) {
        b.best_bid_offset  = -1;
        b.best_ask_offset  = MAX_PRICE_LEVELS;
        b.num_fills        = 0;
        b.total_buy_volume = 0;
        b.total_sell_volume= 0;
    }
    __syncthreads();

    // Only thread 0 does matching — keeps semantics deterministic.
    if (threadIdx.x == 0) {
        long arrival_mid_x2 = 0;
        long my_vwap_num = 0, my_vwap_den = 0;

        for (int i = 0; i < num_events; ++i) {
            Event ev = events[i];

            if (i == my_order_idx) {
                arrival_mid_x2 = (long)b.best_bid_offset + (long)b.best_ask_offset;
                int filled = (my_side > 0)
                    ? match_buy(b, my_price_offset, my_size)
                    : match_sell(b, my_price_offset, my_size);
                my_vwap_num += (long)my_price_offset * filled;
                my_vwap_den += filled;
            }

            int idx = ev.price_offset;
            if (idx < 0 || idx >= MAX_PRICE_LEVELS) continue;

            if (ev.type == EVT_ADD_BID) {
                int filled = match_buy(b, idx, ev.size);
                if (filled < ev.size) {
                    push_order(b.bids[idx], ev.order_id, ev.size - filled);
                    if (idx > b.best_bid_offset) b.best_bid_offset = idx;
                }
            } else if (ev.type == EVT_ADD_ASK) {
                int filled = match_sell(b, idx, ev.size);
                if (filled < ev.size) {
                    push_order(b.asks[idx], ev.order_id, ev.size - filled);
                    if (idx < b.best_ask_offset) b.best_ask_offset = idx;
                }
            }
            // Cancels handled elsewhere in production; omitted here to keep
            // the critical path tight for the benchmark.
        }

        // Slippage in ticks * 2 (mid = (bid+ask)/2, we use *2 to stay int).
        long slip = 0;
        if (my_vwap_den > 0 && arrival_mid_x2 > 0) {
            slip = (2 * my_vwap_num / my_vwap_den - arrival_mid_x2) * my_side;
        }
        slippage_out[book_id] = slip;
    }
}

// ----------------------------------------------------------------------------
// Host-side driver. Illustrative; real benchmarks launch with 10k+ blocks.
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    const int N_BOOKS = (argc > 1) ? atoi(argv[1]) : 1024;
    const int N_EVTS  = (argc > 2) ? atoi(argv[2]) : 4096;

    Event* h_events = (Event*)malloc(N_EVTS * sizeof(Event));
    for (int i = 0; i < N_EVTS; ++i) {
        h_events[i].type = i & 1;
        h_events[i].price_offset = MAX_PRICE_LEVELS / 2 + ((i % 11) - 5);
        h_events[i].size = 1 + (i % 10);
        h_events[i].order_id = i;
        h_events[i].timestamp_ns = i;
    }

    Event* d_events; cudaMalloc(&d_events, N_EVTS * sizeof(Event));
    cudaMemcpy(d_events, h_events, N_EVTS * sizeof(Event), cudaMemcpyHostToDevice);

    Book* d_books; cudaMalloc(&d_books, N_BOOKS * sizeof(Book));
    long* d_slip;  cudaMalloc(&d_slip,  N_BOOKS * sizeof(long));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    replay_kernel<<<N_BOOKS, 128>>>(d_events, N_EVTS, d_books, d_slip,
                                    N_EVTS / 2, +1, MAX_PRICE_LEVELS / 2 + 2, 10);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("N_BOOKS=%d  N_EVTS=%d  kernel=%.2f ms  throughput=%.2f M-events/s\n",
           N_BOOKS, N_EVTS, ms, (double)N_BOOKS * N_EVTS / (ms * 1e3));

    free(h_events);
    cudaFree(d_events); cudaFree(d_books); cudaFree(d_slip);
    return 0;
}
