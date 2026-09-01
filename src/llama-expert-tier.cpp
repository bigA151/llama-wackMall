#include "llama-expert-tier.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#define NOMINMAX
#include <malloc.h>
#include <intrin.h>
#include <windows.h>
#include <BaseTsd.h> // SSIZE_T
typedef SSIZE_T ssize_t; // POSIX type not provided on Windows
#endif

// expert tiering hooks live in the CPU backend (ggml-cpu.c). Under
// GGML_BACKEND_DL the backend is a runtime-loaded .so, so libllama cannot
// link the setters directly; resolve them once from the backend registry
// (same pattern as llama-context.cpp for ggml_backend_cpu_set_threadpool).
typedef void (*moe_predict_hook_fn)(const ggml_tensor *, const ggml_tensor *);
typedef const char * (*moe_addr_hook_fn)(const void *, int64_t, const char *);
typedef void (*moe_prefetch_use_hook_fn)(const void *, const int64_t *, int64_t, bool);

typedef void (*moe_set_predict_fn)(moe_predict_hook_fn);
typedef void (*moe_set_addr_fn)(moe_addr_hook_fn);
typedef void (*moe_set_prefetch_use_fn)(moe_prefetch_use_hook_fn);
typedef void (*moe_set_route_fn)(FILE *, int);
typedef uint64_t (*moe_timer_fn)(void);
typedef void (*moe_timers_fn)(uint64_t *, uint64_t *, uint64_t *, uint64_t *);

static moe_set_predict_fn g_fn_predict = NULL;
static moe_set_predict_fn g_fn_predict_match = NULL;
static moe_set_addr_fn    g_fn_probe   = NULL;
static moe_set_addr_fn    g_fn_fetch   = NULL;
static moe_set_prefetch_use_fn g_fn_prefetch_use = NULL;
static moe_set_route_fn   g_fn_route   = NULL;
static moe_timer_fn       g_fn_timer   = NULL;
static moe_timers_fn      g_fn_timers  = NULL;

static void tier_resolve_moe_hooks(void) {
    if (g_fn_predict) {
        return;
    }
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return;
    }
    g_fn_predict = (moe_set_predict_fn) ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_predict_hook");
    g_fn_predict_match = (moe_set_predict_fn) ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_predict_match_hook");
    g_fn_probe   = (moe_set_addr_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_probe_hook");
    g_fn_fetch   = (moe_set_addr_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_fetch_hook");
    g_fn_prefetch_use = (moe_set_prefetch_use_fn) ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_prefetch_use_hook");
    g_fn_route   = (moe_set_route_fn)   ggml_backend_reg_get_proc_address(reg, "ggml_set_route_trace");
    g_fn_timer   = (moe_timer_fn)       ggml_backend_reg_get_proc_address(reg, "ggml_moe_cold_timer_us");
    g_fn_timers  = (moe_timers_fn)      ggml_backend_reg_get_proc_address(reg, "ggml_moe_cold_timers_us");
}

#define MOE_PREDICT_HOOK(fn)   do { if (g_fn_predict) g_fn_predict((fn)); } while (0)
#define MOE_PREDICT_MATCH_HOOK(fn) do { if (g_fn_predict_match) g_fn_predict_match((fn)); } while (0)
#define MOE_PROBE_HOOK(fn)     do { if (g_fn_probe)   g_fn_probe((fn));   } while (0)
#define MOE_FETCH_HOOK(fn)     do { if (g_fn_fetch)   g_fn_fetch((fn));   } while (0)
#define MOE_PREFETCH_USE_HOOK(fn) do { if (g_fn_prefetch_use) g_fn_prefetch_use((fn)); } while (0)
#define MOE_ROUTE_HOOK(f, n)   do { if (g_fn_route)   g_fn_route((f), (n)); } while (0)
#define MOE_TIMER_HOOK()       (g_fn_timer ? g_fn_timer() : 0)

struct moe_timing_snapshot {
    uint64_t total = 0;
    uint64_t setup = 0;
    uint64_t gate_up = 0;
    uint64_t activation = 0;
};

static moe_timing_snapshot moe_timing_snapshot_get() {
    moe_timing_snapshot result;
    if (g_fn_timers) {
        g_fn_timers(&result.total, &result.setup, &result.gate_up, &result.activation);
    } else {
        result.total = MOE_TIMER_HOOK();
    }
    return result;
}

// portable atomic access to single i32s inside plain buffers (lut_host must
// stay a plain i32 vector: it is uploaded wholesale into the GPU lut tensor)
static inline int32_t tier_atomic_load_i32(const int32_t * p) {
#if defined(_MSC_VER)
    return _InterlockedExchangeAdd((volatile long *) p, 0);
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

static inline void tier_atomic_store_i32(int32_t * p, int32_t v) {
#if defined(_MSC_VER)
    _InterlockedExchange((volatile long *) p, (long) v);
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

// tier status must print at default verbosity; libllama INFO requires -v
#define TIER_LOG(...) fprintf(stderr, __VA_ARGS__)

// page hints on mmap'd source weights: drop = free the file-backed pages of a
// GPU-resident expert (bytes refault from the model file on next touch).
// never call on malloc'd buffers - DONTNEED would zero anonymous pages.
static void tier_madvise(const void * p, size_t len, bool drop) {
#if !defined(_WIN32)
    static const long page = sysconf(_SC_PAGESIZE);
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) (page - 1);
    const uintptr_t b = ((uintptr_t) p + len + page - 1) & ~(uintptr_t) (page - 1);
    if (b > a) {
        madvise((void *) a, b - a, drop ? MADV_DONTNEED : MADV_WILLNEED);
    }
#else
    (void) p; (void) len; (void) drop;
#endif
}

// pread helper: portable persistent read for demand fetch
// POSIX: pread() on a shared read-only fd (thread-safe)
// Windows: ReadFile+OVERLAPPED (compile-guarded, marked UNTESTED)
static int g_pread_fd = -1;
#if defined(_WIN32)
static HANDLE g_pread_handle = INVALID_HANDLE_VALUE;
#endif
static bool g_pread_disabled = false; // KAT failure or mmap-off disables the path

static bool tier_pread_init(const char * path) {
    if (g_pread_disabled) {
        return false;
    }
#if !defined(_WIN32)
    g_pread_fd = open(path, O_RDONLY);
    if (g_pread_fd < 0) {
        TIER_LOG("%s: pread open failed: %s\n", __func__, strerror(errno));
        g_pread_disabled = true;
        return false;
    }
#else
    g_pread_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g_pread_handle == INVALID_HANDLE_VALUE) {
        TIER_LOG("%s: pread CreateFileA failed\n", __func__);
        g_pread_disabled = true;
        return false;
    }
#endif
    return true;
}

static void tier_pread_close() {
#if !defined(_WIN32)
    if (g_pread_fd >= 0) {
        close(g_pread_fd);
        g_pread_fd = -1;
    }
#else
    if (g_pread_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(g_pread_handle);
        g_pread_handle = INVALID_HANDLE_VALUE;
    }
#endif
}

// pread a list of (offset, size) pairs into dest; returns total bytes read or -1 on error
// coalesces adjacent reads (offset + size == next offset) into a single call
static ssize_t tier_pread_list(void * dest, const std::vector<std::pair<size_t, size_t>> & spans) {
    if (g_pread_disabled || spans.empty()) {
        return -1;
    }
    char * dp = (char *) dest;
    ssize_t total = 0;
    size_t i = 0;
    while (i < spans.size()) {
        size_t off = spans[i].first;
        size_t sz = spans[i].second;
        size_t j = i + 1;
        while (j < spans.size() && spans[j].first == off + sz) {
            sz += spans[j].second;
            j++;
        }
#if !defined(_WIN32)
        ssize_t n = pread(g_pread_fd, dp, sz, off);
        if (n != (ssize_t) sz) {
            return -1;
        }
#else
        OVERLAPPED ov = {};
        ov.Offset = (DWORD) off;
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD n = 0;
        if (!ReadFile(g_pread_handle, dp, (DWORD) sz, &n, &ov)) {
            return -1;
        }
        if (n != (DWORD) sz) {
            return -1;
        }
#endif
        dp += sz;
        total += (ssize_t) sz;
        i = j;
    }
    return total;
}

// KAT: pread the first 4 KiB of a host expert tensor, byte-compare vs the
// mmap. offset 0 is unusable: the loader munmaps the metadata fragment after
// load (llama_model_loader::load_all_data), so the header pages are no
// longer backed by the file mapping. the tensor span also validates the
// (data - mmap_base) offset math the fetch path relies on.
static bool tier_pread_kat(const void * tensor_data, size_t file_off) {
    if (g_pread_disabled || !tensor_data) {
        return false;
    }
    const size_t K = 4096;
    char buf[K];
    ssize_t n = tier_pread_list(buf, {{file_off, K}});
    if (n != (ssize_t) K) {
        TIER_LOG("%s: KAT pread failed\n", __func__);
        g_pread_disabled = true;
        return false;
    }
    if (memcmp(buf, tensor_data, K) != 0) {
        TIER_LOG("%s: KAT byte mismatch\n", __func__);
        g_pread_disabled = true;
        return false;
    }
    return true;
}

namespace llama_expert_tier {

struct prediction_ticket;
struct prefetch_request;

struct store {
    ggml_tensor * w_hot  = nullptr; // [ne0, ne1, n_slots] on GPU, sentinel slot zeroed
    ggml_tensor * lut    = nullptr; // i32 [n_expert] on GPU: expert -> hot slot | sentinel
    ggml_tensor * mask   = nullptr; // i32 [n_expert] on CPU: 1 = cold
    ggml_tensor * counts = nullptr; // i32 [n_expert+1] on CPU: selection stats
    ggml_tensor * ptrs   = nullptr; // i64 [n_expert] on CPU: weight source address (pool slot | mmap)
    int  il      = -1;
    bool is_down = false;
    bool discardable = false; // mmap-backed and not mlocked: pages may be dropped
    bool poolable    = false; // weights_discardable regardless of LLAMA_EXPERT_MADVISE
    size_t pool_off = 0;      // offset of this tensor's slice within a pool slot
};

// per-layer grouping for stats and online repin
struct layer_tier {
    int il       = -1;
    int n_expert = 0;
    int n_slots  = 0; // incl. sentinel slot
    int sentinel = 0;
    store * sd = nullptr; // store holding the down weight (counts source)
    std::vector<std::pair<const ggml_tensor *, store *>> ws;
    std::vector<int32_t> slot_expert; // [n_slots], -1 = empty
    std::vector<int32_t> lut_host;    // [n_expert]
    std::vector<float>   score;       // [n_expert], cumulative (LLAMA_EXPERT_DECAY)
    std::vector<uint64_t> cum;        // [n_expert], exact counts for persistence
    std::vector<int32_t> dwell;       // [n_slots]
    std::unique_ptr<std::atomic<int32_t>[]> hot_pin_count; // [n_slots], sentinel unused
    std::unique_ptr<std::atomic<uint64_t>[]> hot_generation; // [n_slots], changes on repin
    uint64_t cum_cold = 0, cum_total = 0, cum_graphs = 0;
    // RAM pool: slot = one expert across all tiered tensors of this layer
    char * pool = nullptr;
    size_t pool_slot_bytes = 0;
    int    n_pool_slots = 0;
    std::vector<int32_t> pool_slot_expert; // [n_pool_slots], -1 = empty
    std::vector<int32_t> pool_dwell;       // [n_pool_slots]
    // residency contract: 0 free / 1 filling / 2 ready / 3 resident.
    // the update() window owns FREE/READY/RESIDENT transitions; the prefetch
    // worker only claims FREE->FILLING->READY mid-step (atomic CAS).
    std::unique_ptr<std::atomic<int32_t>[]> pool_slot_state; // [n_pool_slots]
    std::unique_ptr<std::atomic<int32_t>[]> pool_lut;        // [n_expert], -1 = not pooled
    // spec pool (prefetch): worker-only cache for predicted experts, probe-served
    char * poolB = nullptr;
    int    n_slotsB = 0;
    std::vector<int32_t> ownersB;                // [n_slotsB], -1 = empty
    std::unique_ptr<std::atomic<int32_t>[]> stateB; // [n_slotsB] 0 free/1 filling/2 ready/3 resident
    std::unique_ptr<std::atomic<int32_t>[]> lutB;   // [n_expert], -1 = not in B
    std::unique_ptr<std::atomic<int64_t>[]> lastB;  // [n_slotsB] last-predicted step
    std::unique_ptr<std::atomic<uint64_t>[]> fillB; // [n_slotsB] successful-fill generation
    std::unique_ptr<std::atomic<uint64_t>[]> checkedB; // [n_slotsB] generation checked at first decode
};

static int  g_S        = 16;
static int  g_tmax     = 16;
static bool g_adapt    = false;
static bool g_madvise  = true; // LLAMA_EXPERT_MADVISE=0 disables page hints
static bool g_hot_only = false; // set by build_moe_cold per layer

static int  g_prefetch_slots = -1; // command-line override; -1 keeps legacy GiB allocation
static int  g_prefetch_max = 3;
static size_t g_prefetch_chunk_bytes = 4*1024*1024;
static bool g_prefetch_calibrate = false;
static float g_prefetch_entropy_threshold = -1.0f;
static float g_prefetch_probability_threshold = -1.0f;

static ggml_context * g_ctx_gpu = nullptr; // owned for process lifetime
static ggml_context * g_ctx_cpu = nullptr;

static std::unordered_map<const ggml_tensor *, store> g_stores;
static std::vector<layer_tier> g_layers;
static uint64_t g_repins = 0; // hot-set changes since init

static size_t   g_pool_bytes = 0; // LLAMA_EXPERT_RAMPOOL budget in bytes (0 = off)
static size_t   g_pool_alloc = 0; // resident pool bytes
static size_t   g_pool_fill_budget = 0; // per-update() fill budget
static uint64_t g_pool_fills = 0, g_pool_evictions = 0;
static uint64_t g_pool_hits = 0, g_pool_cold = 0;

static uint64_t g_fetch_us = 0;   // cumulative wall time in pool_fill() memcpy
static uint64_t g_steps = 0;      // update() calls (= graph computes)
static uint64_t g_timing_interval = 0; // LLAMA_EXPERT_TIMING: updates between logs
static FILE *   g_route_log = nullptr; // actual-routing trace (co-opened with pred log)

static std::atomic<bool> g_perf_trace{false};
static std::atomic<uint64_t> g_trace_ssd_read_us{0};
static std::atomic<uint64_t> g_trace_ssd_read_bytes{0};
static std::atomic<uint64_t> g_trace_ssd_reads{0};
static std::atomic<uint64_t> g_trace_stage_wait_us{0};
static std::atomic<uint64_t> g_trace_stage_waits{0};
static std::atomic<uint64_t> g_trace_pool_fill_us{0};
static std::atomic<uint64_t> g_trace_pool_fill_bytes{0};
static std::atomic<uint64_t> g_trace_pool_fills{0};

// pre-gate predictor: immutable CPU mirrors of each MoE layer's router and
// ffn norm weights. The fused cold op reports its router input x through the
// predict hook; running physical layer L+4's router on layer L's x (with its
// norm reconstructed as w_target * (x / w_cur)) predicts that exact layer's
// experts four layers ahead of demand. No learned parameters: the predictor
// is the model's own gating function applied early.
static constexpr int PREGATE_LOOKAHEAD = 4;

struct pred_layer {
    int il = -1;
    int target_ix = -1;
    int n_routed = 0;
    float historical_cold_rate = 1.0f;
    std::vector<float> norm;
    std::vector<float> gate;
};

static std::vector<pred_layer> g_pred;
static std::unordered_map<const void *, int> g_pred_ix; // counts->data -> g_pred index
static size_t g_pred_pairs = 0; // source layers with an exact physical L+4 target
static bool g_predict = false; // LLAMA_EXPERT_PREDICT=1 enables
static FILE * g_pred_log = nullptr; // LLAMA_EXPERT_PREDICT_LOG (debug)
static uint64_t g_pred_pushes = 0;
static uint64_t g_predicted_experts = 0, g_predicted_experts_used = 0;
static float g_prefetch_min_cold_rate = 0.0f;
static std::atomic<uint64_t> g_prediction_skipped_low_cold{0};

enum prefetch_request_state {
    PREFETCH_QUEUED = 0,
    PREFETCH_FILLING = 1,
    PREFETCH_READY = 2,
    PREFETCH_CANCELED = 3,
    PREFETCH_FAILED = 4,
};

struct hot_pin_ref {
    int il = -1;
    int slot = -1;
    int expert = -1;
    uint64_t generation = 0;
    bool held = false;
};

struct prefetch_request {
    int pi = -1;
    int il = -1;
    int expert = -1;
    int slot = -1;
    uint64_t prediction_id = 0;
    int prediction_rank = -1;
    std::atomic<int32_t> state{PREFETCH_QUEUED};
    std::atomic<bool> cancel{false};
    std::atomic<bool> demand{false};
    std::atomic<bool> demand_deferred{false};
    std::atomic<bool> bytes_accounted{false};
    std::atomic<int32_t> actual_correct{-1}; // -1 unknown, 0 wrong, 1 routed
    std::atomic<uint64_t> bytes_filled{0};
    std::atomic<uint64_t> cancel_requested_us{0};
};

struct prediction_ticket {
    uint64_t id = 0;
    uint64_t seq = 0;
    int source_pi = -1;
    int target_pi = -1;
    int target_layer = -1;
    uint64_t queued_us = 0;
    uint64_t compute_start_us = 0;
    uint64_t compute_end_us = 0;
    std::vector<float> x;
    std::atomic<bool> cancel{false};
    std::mutex mu;
    bool prediction_done = false;
    bool actual_known = false;
    float normalized_entropy = 1.0f;
    std::vector<int32_t> selected;
    std::vector<int32_t> top_experts;
    std::vector<float> top_probabilities;
    std::string selection_reason;
    std::vector<std::string> residency;
    std::vector<int32_t> actual;
    std::vector<hot_pin_ref> hot_pins;
    std::vector<std::shared_ptr<prefetch_request>> requests;
};

static std::mutex g_prediction_mu;
static std::condition_variable g_prediction_cv;
static std::condition_variable g_work_cv;
static std::deque<std::shared_ptr<prediction_ticket>> g_prediction_q;
static std::vector<std::shared_ptr<prediction_ticket>> g_ticket_by_layer;
static std::thread g_prediction_worker;
static std::atomic<uint64_t> g_prediction_id{0};
static std::atomic<uint64_t> g_prediction_late{0};
static std::atomic<uint64_t> g_prediction_canceled{0};
static std::atomic<uint64_t> g_hot_pins{0};
static std::atomic<uint64_t> g_hot_unpins{0};
static std::atomic<uint64_t> g_prefetch_promotions{0};
static std::atomic<uint64_t> g_prefetch_waits{0};
static std::atomic<uint64_t> g_prefetch_wait_us{0};
static std::atomic<uint64_t> g_prefetch_chunk_cancels{0};
static std::atomic<uint64_t> g_prefetch_cancel_requests{0};
static std::atomic<uint64_t> g_prefetch_expired_requests{0};
static std::atomic<uint64_t> g_prefetch_merges{0};
static std::atomic<uint64_t> g_prefetch_demand_deferred{0};
static std::atomic<uint64_t> g_hot_eviction_blocks{0};
static std::atomic<uint64_t> g_prefetch_correct_bytes{0};
static std::atomic<uint64_t> g_prefetch_wrong_bytes{0};
static std::atomic<uint64_t> g_prefetch_copy_us{0};
static std::atomic<uint64_t> g_residency_hot{0};
static std::atomic<uint64_t> g_residency_demand{0};
static std::atomic<uint64_t> g_residency_prefetch_ready{0};
static std::atomic<uint64_t> g_residency_prefetch_filling{0};
static std::atomic<uint64_t> g_residency_enqueued{0};
static std::atomic<uint64_t> g_prediction_tickets_with_io{0};
static constexpr int PREFETCH_RANK_STATS = 8;
static std::atomic<uint64_t> g_rank_selected[PREFETCH_RANK_STATS]{};
static std::atomic<uint64_t> g_rank_used[PREFETCH_RANK_STATS]{};
static std::atomic<uint64_t> g_rank_enqueued[PREFETCH_RANK_STATS]{};
static std::atomic<uint64_t> g_rank_correct_bytes[PREFETCH_RANK_STATS]{};
static std::atomic<uint64_t> g_rank_wrong_bytes[PREFETCH_RANK_STATS]{};
static std::atomic<uint64_t> g_prefetch_duplicate_claims{0};
static std::atomic<uint64_t> g_prefetch_partial_probe_rejects{0};
static std::atomic<uint64_t> g_hot_stale_unpins{0};
static std::atomic<int32_t> g_demand_active{0};
static std::mutex g_hot_mu;
static std::mutex g_metric_mu;
static std::vector<uint64_t> g_wait_samples_us;
static std::vector<uint64_t> g_cancel_response_samples_us;
static std::vector<uint64_t> g_prediction_queue_samples_us;
static std::vector<uint64_t> g_prediction_compute_samples_us;

static void note_demand_start();
static void note_demand_end();

// prefetch probe: lets a cold op read an expert straight from a READY
// (fully memcpy'd, not yet published) pool slot instead of the mmap source.
// keyed by the store's ptrs->data; content is byte-identical either way, so
// a probe miss costs speed, never correctness.
struct probe_ctx {
    const std::atomic<int32_t> * lut;
    const std::atomic<int32_t> * states;
    const int32_t * owners;
    const char * pool;
    int64_t slot_bytes;
    int64_t pool_off;
};
static std::unordered_map<const void *, probe_ctx> g_probe_ix; // ptrs->data -> ctx
static std::atomic<uint64_t> g_probe_hits{0}, g_probe_miss{0};
static std::atomic<uint64_t> g_pred_fill_generation{0};
static std::atomic<uint64_t> g_pred_first_decode_used{0};

static void pregate_prefetch_use(const void * key, const int64_t * row_counts, int64_t n_expert, bool is_decode) {
    if (!is_decode || !row_counts) {
        return;
    }
    auto it = g_probe_ix.find(key);
    if (it == g_probe_ix.end()) {
        return;
    }
    const probe_ctx & c = it->second;
    for (const auto & L : g_layers) {
        if (L.lutB.get() != c.lut || L.n_expert != n_expert) {
            continue;
        }
        for (int k = 0; k < L.n_slotsB; k++) {
            if (L.stateB[k].load(std::memory_order_acquire) < 2) {
                continue;
            }
            const uint64_t generation = L.fillB[k].load(std::memory_order_acquire);
            if (generation == 0 || L.checkedB[k].exchange(generation, std::memory_order_acq_rel) == generation) {
                continue;
            }
            const int32_t e = L.ownersB[k];
            if (e >= 0 && e < n_expert && row_counts[e] > 0) {
                g_pred_first_decode_used.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return;
    }
}

static void pregate_predict_match(const ggml_tensor * counts, const ggml_tensor * ids);

static bool g_dirty = false;
static std::string g_sidecar_path;
static uint64_t g_fingerprint = 0;

static const char * pregate_probe(const void * key, int64_t e, const char * fallback) {
    auto it = g_probe_ix.find(key);
    if (it == g_probe_ix.end()) {
        return fallback;
    }
    const probe_ctx & c = it->second;
    const int32_t k = c.lut[e].load(std::memory_order_acquire);
    if (k >= 0 && c.states[k].load(std::memory_order_acquire) >= 2 && c.owners[k] == (int32_t) e) {
        g_probe_hits.fetch_add(1, std::memory_order_relaxed);
        return c.pool + (size_t) k*c.slot_bytes + c.pool_off;
    }
    if (k >= 0 && c.states[k].load(std::memory_order_acquire) < 2) {
        g_prefetch_partial_probe_rejects.fetch_add(1, std::memory_order_relaxed);
    }
    g_probe_miss.fetch_add(1, std::memory_order_relaxed);
    return fallback;
}

static uint64_t fnv1a_64(const void * data, size_t len, uint64_t h = 14695981039346656037ULL) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv1a_64_str(const char * s, uint64_t h = 14695981039346656037ULL) {
    return fnv1a_64(s, strlen(s), h);
}

static uint64_t fnv1a_64_u64(uint64_t v, uint64_t h = 14695981039346656037ULL) {
    return fnv1a_64(&v, sizeof(v), h);
}

static uint64_t compute_fingerprint(const llama_model & model) {
    uint64_t h = 14695981039346656037ULL;
    const char * arch = llm_arch_name(model.arch);
    h = fnv1a_64_str(arch ? arch : "unknown", h);
    h = fnv1a_64_u64((uint64_t) model.size(), h);
    h = fnv1a_64_u64((uint64_t) model.n_tensors(), h);
    for (const auto & kv : model.tensors_by_name) {
        h = fnv1a_64_str(kv.first.c_str(), h);
        const ggml_tensor * t = kv.second;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            h = fnv1a_64_u64((uint64_t) t->ne[i], h);
        }
    }
    return h;
}

static void save_sidecar();

static std::vector<std::vector<float>> load_sidecar(const llama_model & model) {
    std::vector<std::vector<float>> result;
    if (g_sidecar_path.empty()) {
        return result;
    }
    std::ifstream in(g_sidecar_path, std::ios::binary);
    if (!in) {
        TIER_LOG("%s: no sidecar at %s\n", __func__, g_sidecar_path.c_str());
        return result;
    }
    uint32_t version = 0;
    uint64_t fp = 0;
    in.read((char *) &version, sizeof(version));
    in.read((char *) &fp, sizeof(fp));
    if (!in || version != 1) {
        TIER_LOG("%s: sidecar version mismatch (got %u, want 1), ignoring\n", __func__, version);
        return result;
    }
    if (fp != g_fingerprint) {
        TIER_LOG("%s: sidecar fingerprint mismatch (got %016llx, want %016llx), ignoring\n",
                __func__, (unsigned long long) fp, (unsigned long long) g_fingerprint);
        return result;
    }
    const int n_layer = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;
    result.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        result[il].resize(n_expert);
        in.read((char *) result[il].data(), n_expert * sizeof(float));
        if (!in) {
            TIER_LOG("%s: sidecar truncated at layer %d, ignoring\n", __func__, il);
            result.clear();
            return result;
        }
    }
    TIER_LOG("%s: loaded sidecar %s (fingerprint %016llx)\n", __func__, g_sidecar_path.c_str(), (unsigned long long) fp);
    return result;
}

static void save_sidecar() {
    if (!g_dirty || g_sidecar_path.empty() || g_fingerprint == 0) {
        return;
    }
    const std::string tmp = g_sidecar_path + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
        TIER_LOG("%s: cannot create %s\n", __func__, tmp.c_str());
        return;
    }
    const uint32_t version = 1;
    out.write((const char *) &version, sizeof(version));
    out.write((const char *) &g_fingerprint, sizeof(g_fingerprint));
    for (const auto & L : g_layers) {
        if (L.score.empty()) {
            continue;
        }
        out.write((const char *) L.score.data(), L.score.size() * sizeof(float));
    }
    out.close();
    if (!out) {
        TIER_LOG("%s: write failed for %s\n", __func__, tmp.c_str());
        std::remove(tmp.c_str());
        return;
    }
#if defined(_WIN32)
    const bool ok = MoveFileExA(tmp.c_str(), g_sidecar_path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    const bool ok = std::rename(tmp.c_str(), g_sidecar_path.c_str()) == 0;
#endif
    if (!ok) {
        TIER_LOG("%s: rename failed for %s\n", __func__, g_sidecar_path.c_str());
        std::remove(tmp.c_str());
    } else {
        TIER_LOG("%s: saved sidecar %s (fingerprint %016llx)\n", __func__, g_sidecar_path.c_str(), (unsigned long long) g_fingerprint);
    }
}

// demand-fetch staging ring (LLAMA_EXPERT_PREAD): bounded, page-aligned,
// owned by the cold path, separate from the score-managed demand pool. the
// first compute thread to touch a pool-miss expert claims a slot and preads
// the expert's slices (all tiered tensors of its layer) from the model file;
// threads claiming different experts pread in parallel; a thread needing an
// in-flight expert waits on the slot word. a slot is reclaimable only by a
// claim from a different layer's op: cold ops are serialized by the
// threadpool, so a slot keyed by another layer has no live readers. no
// pool/lut/score metadata is touched here; update() stays the single writer.
// slot word: bits [0:2) state (0 free / 1 claimed / 2 ready), [2:34) expert,
// [34:64) layer - one atomic word so claim+key is a single CAS.
static char * g_stage_base = nullptr;
static size_t g_stage_stride = 0; // page-aligned slot stride
static int    g_stage_n = 0;
static std::unique_ptr<std::atomic<uint64_t>[]> g_stage_word; // [g_stage_n]
static const char * g_stage_mmap = nullptr; // model mmap base (offset math)
static std::atomic<uint64_t> g_stage_fetches{0}, g_stage_bytes{0};
static std::atomic<uint64_t> g_stage_stalls{0}, g_stage_fallbacks{0}, g_stage_fails{0};
static bool g_stage_fail_test = false; // LLAMA_EXPERT_PREAD_FAIL (test-only)

struct stage_ctx {
    int          il;
    int64_t      pool_off; // this tensor's offset within a slot
    const char * w_data;   // mmap base of the tensor (residency check)
    int64_t      slice;    // per-expert bytes of this tensor
};
static std::unordered_map<const void *, stage_ctx> g_stage_ix; // ptrs->data -> ctx

static uint64_t stage_word(int il, int64_t e, uint64_t st) {
    return ((uint64_t) il << 34) | ((uint64_t) e << 2) | st;
}

static const char * stage_fetch(const void * key, int64_t e, const char * fallback) {
    auto it = g_stage_ix.find(key);
    if (it == g_stage_ix.end()) {
        return fallback;
    }
    const stage_ctx & c = it->second;
    if (fallback != c.w_data + e*c.slice) {
        return fallback; // resident: demand pool or a READY spec slot
    }
    const uint64_t mykey = stage_word(c.il, e, 0);
    for (int attempt = 0; attempt < 4; attempt++) {
        int cand = -1;
        uint64_t candw = 0;
        for (int k = 0; k < g_stage_n; k++) {
            const uint64_t w = g_stage_word[k].load(std::memory_order_acquire);
            const uint64_t st = w & 3;
            if (st != 0 && (w & ~3ull) == mykey) {
                if (st == 2) {
                    return g_stage_base + (size_t) k*g_stage_stride + c.pool_off;
                }
                // claimed by another thread: wait for the fill to finish
                g_stage_stalls.fetch_add(1, std::memory_order_relaxed);
                const int64_t wait_start = g_perf_trace.load(std::memory_order_relaxed) ? ggml_time_us() : 0;
                uint64_t v = w;
                int spins = 0;
                while ((v = g_stage_word[k].load(std::memory_order_acquire)) == w) {
                    if ((++spins & 0x3FF) == 0) {
                        std::this_thread::yield();
                    }
                }
                if (wait_start) {
                    const uint64_t wait_us = (uint64_t) (ggml_time_us() - wait_start);
                    g_trace_stage_wait_us.fetch_add(wait_us, std::memory_order_relaxed);
                    g_trace_stage_waits.fetch_add(1, std::memory_order_relaxed);
                    LLAMA_LOG_INFO("[PERF_TRACE][expert_stage_wait] layer=%d expert=%lld wait=%.3f ms\n",
                            c.il, (long long) e, wait_us/1000.0);
                }
                if ((v & ~3ull) == mykey && (v & 3) == 2) {
                    return g_stage_base + (size_t) k*g_stage_stride + c.pool_off;
                }
                g_stage_fallbacks.fetch_add(1, std::memory_order_relaxed);
                return fallback; // fill failed: mmap fallback
            }
            if (cand < 0 && (st == 0 || (st == 2 && (int) (w >> 34) != c.il))) {
                cand  = k;
                candw = w;
            }
        }
        if (cand < 0) {
            break; // ring full of live slots: mmap fallback
        }
        if (!g_stage_word[cand].compare_exchange_strong(candw, mykey | 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue; // lost the claim race: rescan
        }
        char * slot = g_stage_base + (size_t) cand*g_stage_stride;
        size_t total = 0;
        bool ok = !g_stage_fail_test;
        if (ok) {
            const int64_t demand_start = g_perf_trace.load(std::memory_order_relaxed) ? ggml_time_us() : 0;
            note_demand_start();
            const layer_tier & L = g_layers[c.il];
            for (const auto & kv : L.ws) {
                const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
                const size_t off = (size_t) ((const char *) kv.first->data - g_stage_mmap) + (size_t) e*slice;
                const int64_t read_start = g_perf_trace.load(std::memory_order_relaxed) ? ggml_time_us() : 0;
                if (tier_pread_list(slot + kv.second->pool_off, {{off, slice}}) != (ssize_t) slice) {
                    ok = false;
                    break;
                }
                if (read_start) {
                    const uint64_t read_us = (uint64_t) (ggml_time_us() - read_start);
                    g_trace_ssd_read_us.fetch_add(read_us, std::memory_order_relaxed);
                    g_trace_ssd_read_bytes.fetch_add(slice, std::memory_order_relaxed);
                    g_trace_ssd_reads.fetch_add(1, std::memory_order_relaxed);
                    LLAMA_LOG_INFO("[PERF_TRACE][expert_ssd_read] layer=%d expert=%lld tensor=%s bytes=%zu read=%.3f ms\n",
                            c.il, (long long) e, kv.first->name, slice, read_us/1000.0);
                }
                total += slice;
            }
            note_demand_end();
            if (demand_start) {
                LLAMA_LOG_INFO("[PERF_TRACE][expert_demand_fetch] layer=%d expert=%lld bytes=%zu ok=%d total=%.3f ms\n",
                        c.il, (long long) e, total, ok ? 1 : 0, (ggml_time_us() - demand_start)/1000.0);
                LLAMA_LOG_INFO("[PERF_TRACE][expert_demand_block] kind=direct layer=%d expert=%lld total=%.3f ms\n",
                        c.il, (long long) e, (ggml_time_us() - demand_start)/1000.0);
            }
        }
        if (ok) {
            g_stage_word[cand].store(mykey | 2, std::memory_order_release);
            g_stage_fetches.fetch_add(1, std::memory_order_relaxed);
            g_stage_bytes.fetch_add(total, std::memory_order_relaxed);
            return slot + c.pool_off;
        }
        g_stage_word[cand].store(0, std::memory_order_release);
        const uint64_t nf = g_stage_fails.fetch_add(1, std::memory_order_relaxed) + 1;
        if (nf <= 3 || (nf & (nf - 1)) == 0) { // rate-limited
            TIER_LOG("%s: pread failed (%llu total), mmap fallback\n", __func__, (unsigned long long) nf);
        }
        return fallback;
    }
    g_stage_fallbacks.fetch_add(1, std::memory_order_relaxed);
    return fallback;
}

// Prediction and I/O are separate stages. Demand-promoted requests have their
// own queue; speculative workers do not start new copies while demand I/O is
// active, but copies which already started are allowed to finish.
static std::deque<std::shared_ptr<prefetch_request>> g_work_q;
static std::deque<std::shared_ptr<prefetch_request>> g_demand_q;
static std::unordered_map<uint64_t, std::shared_ptr<prefetch_request>> g_request_by_expert;
static std::mutex g_work_mu;
static std::condition_variable g_ready_cv;
static std::atomic<bool> g_work_exit{false};
static std::vector<std::thread> g_workers;
static std::atomic<int64_t> g_work_inflight{0};
static int64_t g_work_budget = 64*1024*1024; // LLAMA_EXPERT_PREFETCH_MB
static size_t g_poolB_bytes = 0;   // LLAMA_EXPERT_PREFETCH_GB (default 0 = off)
static size_t g_poolB_alloc = 0;
static size_t g_hot_alloc = 0;
static size_t g_stage_alloc = 0;
static size_t g_predictor_mirror_bytes = 0;
static size_t g_model_mmap_bytes = 0;
static uint64_t g_init_total_us = 0;
static uint64_t g_init_hot_fill_us = 0;
static uint64_t g_init_demand_pool_fill_us = 0;
static int g_n_workers = 2;        // LLAMA_EXPERT_PREFETCH_THREADS (clamp 1..8)
static std::atomic<uint64_t> g_pred_step{0}; // window tick for lastB timestamps
static uint64_t g_predB_evict = 0; // timestamp evictions by the window
static std::atomic<uint64_t> g_pred_fills{0};
static uint64_t g_pred_published = 0;
static std::atomic<uint64_t> g_pred_drop_hot{0}, g_pred_drop_dup{0}, g_pred_drop_budget{0}, g_pred_drop_full{0};

static void record_metric_sample(std::vector<uint64_t> & samples, uint64_t value) {
    std::lock_guard<std::mutex> lk(g_metric_mu);
    samples.push_back(value);
}

static uint64_t metric_percentile(const std::vector<uint64_t> & source, double probability) {
    std::vector<uint64_t> samples;
    {
        std::lock_guard<std::mutex> lk(g_metric_mu);
        samples = source;
    }
    if (samples.empty()) {
        return 0;
    }
    std::sort(samples.begin(), samples.end());
    const size_t index = std::min(samples.size() - 1,
            (size_t) std::max(0.0, std::ceil(probability*samples.size()) - 1.0));
    return samples[index];
}

static uint64_t metric_sum(const std::vector<uint64_t> & source) {
    std::lock_guard<std::mutex> lk(g_metric_mu);
    uint64_t total = 0;
    for (uint64_t value : source) {
        total += value;
    }
    return total;
}

static void note_demand_start() {
    g_demand_active.fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lk(g_work_mu);
    for (const auto & req : g_work_q) {
        if (!req->demand_deferred.exchange(true, std::memory_order_acq_rel)) {
            g_prefetch_demand_deferred.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

static void note_demand_end() {
    g_demand_active.fetch_sub(1, std::memory_order_acq_rel);
    g_work_cv.notify_all();
}

static void request_prefetch_cancel(const std::shared_ptr<prefetch_request> & req, bool expired) {
    bool expected = false;
    if (req->cancel.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        req->cancel_requested_us.store((uint64_t) ggml_time_us(), std::memory_order_release);
        g_prefetch_cancel_requests.fetch_add(1, std::memory_order_relaxed);
        if (expired) {
            g_prefetch_expired_requests.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

static void account_prefetch_bytes(const std::shared_ptr<prefetch_request> & req) {
    const int32_t correct = req->actual_correct.load(std::memory_order_acquire);
    if (correct < 0 || req->state.load(std::memory_order_acquire) < PREFETCH_READY ||
            req->bytes_accounted.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    const uint64_t bytes = req->bytes_filled.load(std::memory_order_acquire);
    (correct ? g_prefetch_correct_bytes : g_prefetch_wrong_bytes).fetch_add(bytes, std::memory_order_relaxed);
    if (req->prediction_rank >= 0 && req->prediction_rank < PREFETCH_RANK_STATS) {
        (correct ? g_rank_correct_bytes[req->prediction_rank] :
                g_rank_wrong_bytes[req->prediction_rank]).fetch_add(bytes, std::memory_order_relaxed);
    }
}

static uint64_t prefetch_key(int il, int e) {
    return ((uint64_t) (uint32_t) il << 32) | (uint32_t) e;
}

static void release_hot_pin(hot_pin_ref & ref) {
    std::lock_guard<std::mutex> hot_lk(g_hot_mu);
    if (!ref.held || ref.il < 0 || ref.il >= (int) g_layers.size()) {
        return;
    }
    layer_tier & L = g_layers[ref.il];
    if (ref.slot >= 0 && ref.slot < L.n_slots &&
            L.hot_generation[ref.slot].load(std::memory_order_acquire) == ref.generation &&
            L.slot_expert[ref.slot] == ref.expert) {
        L.hot_pin_count[ref.slot].fetch_sub(1, std::memory_order_acq_rel);
        g_hot_unpins.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Generation ownership prevents an old prediction from unlocking a
        // slot that has already been recycled for another expert.
        g_hot_stale_unpins.fetch_add(1, std::memory_order_relaxed);
    }
    ref.held = false;
}

static bool pin_hot_expert(prediction_ticket & ticket, layer_tier & L, int e) {
    std::lock_guard<std::mutex> hot_lk(g_hot_mu);
    const int slot = tier_atomic_load_i32(&L.lut_host[e]);
    if (slot < 0 || slot >= L.n_slots || slot == L.sentinel || L.slot_expert[slot] != e) {
        return false;
    }
    const uint64_t generation = L.hot_generation[slot].load(std::memory_order_acquire);
    L.hot_pin_count[slot].fetch_add(1, std::memory_order_acq_rel);
    if (tier_atomic_load_i32(&L.lut_host[e]) != slot ||
            L.hot_generation[slot].load(std::memory_order_acquire) != generation ||
            L.slot_expert[slot] != e) {
        L.hot_pin_count[slot].fetch_sub(1, std::memory_order_acq_rel);
        return false;
    }
    ticket.hot_pins.push_back({L.il, slot, e, generation, true});
    g_hot_pins.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static const char * classify_prediction_locked(
        const std::shared_ptr<prediction_ticket> & ticket, int target_pi, int e, int rank) {
    layer_tier & L = g_layers[g_pred[target_pi].il];
    if (pin_hot_expert(*ticket, L, e)) {
        g_pred_drop_hot.fetch_add(1, std::memory_order_relaxed);
        g_residency_hot.fetch_add(1, std::memory_order_relaxed);
        return "HOT_LOCKED";
    }
    if (L.pool_lut && L.pool_lut[e].load(std::memory_order_acquire) >= 0) {
        g_pred_drop_dup.fetch_add(1, std::memory_order_relaxed);
        g_residency_demand.fetch_add(1, std::memory_order_relaxed);
        return "DEMAND_POOL";
    }
    if (L.lutB) {
        const int32_t k = L.lutB[e].load(std::memory_order_acquire);
        if (k >= 0) {
            L.lastB[k].store((int64_t) g_pred_step.load(std::memory_order_relaxed), std::memory_order_release);
            if (L.stateB[k].load(std::memory_order_acquire) == 1) {
                std::lock_guard<std::mutex> lk(g_work_mu);
                auto it = g_request_by_expert.find(prefetch_key(L.il, e));
                if (it != g_request_by_expert.end()) {
                    ticket->requests.push_back(it->second);
                    g_prefetch_merges.fetch_add(1, std::memory_order_relaxed);
                    g_residency_prefetch_filling.fetch_add(1, std::memory_order_relaxed);
                }
            }
            g_pred_drop_dup.fetch_add(1, std::memory_order_relaxed);
            if (L.stateB[k].load(std::memory_order_acquire) == 1) {
                return "PREFETCH_FILLING_MERGED";
            }
            g_residency_prefetch_ready.fetch_add(1, std::memory_order_relaxed);
            return "PREFETCH_READY_REFRESHED";
        }
    }
    if (g_prefetch_calibrate || !L.poolB) {
        return "NONE_NO_PHYSICAL_POOL";
    }
    auto req = std::make_shared<prefetch_request>();
    req->pi = target_pi;
    req->il = L.il;
    req->expert = e;
    req->prediction_id = ticket->id;
    req->prediction_rank = rank;
    ticket->requests.push_back(req);
    {
        std::lock_guard<std::mutex> lk(g_work_mu);
        if (g_work_q.size() >= 4096) {
            req->state.store(PREFETCH_FAILED, std::memory_order_release);
            return "NONE_QUEUE_FULL";
        }
        g_request_by_expert[prefetch_key(L.il, e)] = req;
        g_work_q.push_back(req);
    }
    g_work_cv.notify_one();
    g_residency_enqueued.fetch_add(1, std::memory_order_relaxed);
    if (rank >= 0 && rank < PREFETCH_RANK_STATS) {
        g_rank_enqueued[rank].fetch_add(1, std::memory_order_relaxed);
    }
    return "NONE_ENQUEUED";
}

static void finish_prefetch_request(const std::shared_ptr<prefetch_request> & req, int state) {
    req->state.store(state, std::memory_order_release);
    account_prefetch_bytes(req);
    {
        std::lock_guard<std::mutex> lk(g_work_mu);
        auto it = g_request_by_expert.find(prefetch_key(req->il, req->expert));
        if (it != g_request_by_expert.end() && it->second.get() == req.get()) {
            g_request_by_expert.erase(it);
        }
    }
    g_ready_cv.notify_all();
}

static void prefetch_fill(const std::shared_ptr<prefetch_request> & req) {
    layer_tier & L = g_layers[req->il];
    const int e = req->expert;
    if (!L.poolB) {
        finish_prefetch_request(req, PREFETCH_FAILED);
        return;
    }
    if (tier_atomic_load_i32(&L.lut_host[e]) != L.sentinel) {
        g_pred_drop_hot.fetch_add(1, std::memory_order_relaxed);
        finish_prefetch_request(req, PREFETCH_READY);
        return;
    }
    if (L.pool_lut && L.pool_lut[e].load(std::memory_order_acquire) >= 0) {
        g_pred_drop_dup.fetch_add(1, std::memory_order_relaxed);
        finish_prefetch_request(req, PREFETCH_READY);
        return;
    }
    const int64_t step = (int64_t) g_pred_step.load(std::memory_order_relaxed);
    const int32_t k0 = L.lutB[e].load(std::memory_order_acquire);
    if (k0 >= 0) {
        L.lastB[k0].store(step, std::memory_order_release); // refresh on re-prediction
        g_pred_drop_dup.fetch_add(1, std::memory_order_relaxed);
        finish_prefetch_request(req,
                L.stateB[k0].load(std::memory_order_acquire) >= 2 ? PREFETCH_READY : PREFETCH_FAILED);
        return;
    }
    const int64_t before = g_work_inflight.fetch_add((int64_t) L.pool_slot_bytes, std::memory_order_acq_rel);
    if (!req->demand.load(std::memory_order_acquire) && before + (int64_t) L.pool_slot_bytes > g_work_budget) {
        g_work_inflight.fetch_sub((int64_t) L.pool_slot_bytes, std::memory_order_acq_rel);
        g_pred_drop_budget.fetch_add(1, std::memory_order_relaxed);
        finish_prefetch_request(req, PREFETCH_FAILED);
        return;
    }
    // claim a FREE slot; never evict - the window keeps ~25% of slots free
    int k = -1;
    for (int i = 0; i < L.n_slotsB; i++) {
        int32_t free_state = 0;
        if (L.stateB[i].compare_exchange_strong(free_state, 1, std::memory_order_acq_rel)) {
            k = i;
            break;
        }
    }
    if (k < 0) {
        g_work_inflight.fetch_sub((int64_t) L.pool_slot_bytes, std::memory_order_acq_rel);
        g_pred_drop_full.fetch_add(1, std::memory_order_relaxed);
        finish_prefetch_request(req, PREFETCH_FAILED);
        return;
    }
    int32_t expect = -1;
    if (!L.lutB[e].compare_exchange_strong(expect, k, std::memory_order_release)) {
        // another worker won; free the slot, refresh the winner's timestamp
        L.stateB[k].store(0, std::memory_order_release);
        L.lastB[expect].store(step, std::memory_order_release);
        g_work_inflight.fetch_sub((int64_t) L.pool_slot_bytes, std::memory_order_acq_rel);
        g_pred_drop_dup.fetch_add(1, std::memory_order_relaxed);
        g_prefetch_duplicate_claims.fetch_add(1, std::memory_order_relaxed);
        finish_prefetch_request(req, PREFETCH_FAILED);
        return;
    }
    req->slot = k;
    L.ownersB[k] = e;
    L.lastB[k].store(step, std::memory_order_relaxed);
    const int64_t fill_start = ggml_time_us();
    size_t fill_bytes = 0;
    bool canceled = false;
    // Explicitly copy gate/up (or fused gate-up) before down.
    for (int pass = 0; pass < 2 && !canceled; pass++) {
        for (auto & kv : L.ws) {
            store & st = *kv.second;
            if (st.is_down != (pass == 1)) {
                continue;
            }
            const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
            for (size_t off = 0; off < slice; off += g_prefetch_chunk_bytes) {
                if (req->cancel.load(std::memory_order_acquire) && !req->demand.load(std::memory_order_acquire)) {
                    canceled = true;
                    break;
                }
                const size_t n = std::min(g_prefetch_chunk_bytes, slice - off);
                memcpy(L.poolB + (size_t) k*L.pool_slot_bytes + st.pool_off + off,
                        (const char *) kv.first->data + slice*e + off, n);
                fill_bytes += n;
                req->bytes_filled.fetch_add(n, std::memory_order_relaxed);
            }
        }
    }
    if (canceled) {
        g_prefetch_copy_us.fetch_add((uint64_t) (ggml_time_us() - fill_start), std::memory_order_relaxed);
        int32_t expected_slot = k;
        L.lutB[e].compare_exchange_strong(expected_slot, -1, std::memory_order_acq_rel);
        L.ownersB[k] = -1;
        L.stateB[k].store(0, std::memory_order_release);
        g_work_inflight.fetch_sub((int64_t) L.pool_slot_bytes, std::memory_order_acq_rel);
        g_prefetch_chunk_cancels.fetch_add(1, std::memory_order_relaxed);
        const uint64_t requested = req->cancel_requested_us.load(std::memory_order_acquire);
        if (requested > 0) {
            record_metric_sample(g_cancel_response_samples_us,
                    (uint64_t) std::max<int64_t>(0, ggml_time_us() - (int64_t) requested));
        }
        finish_prefetch_request(req, PREFETCH_CANCELED);
        return;
    }
    {
        const uint64_t fill_us = (uint64_t) (ggml_time_us() - fill_start);
        g_prefetch_copy_us.fetch_add(fill_us, std::memory_order_relaxed);
        if (g_perf_trace.load(std::memory_order_relaxed)) {
            g_trace_pool_fill_us.fetch_add(fill_us, std::memory_order_relaxed);
            g_trace_pool_fill_bytes.fetch_add(fill_bytes, std::memory_order_relaxed);
            g_trace_pool_fills.fetch_add(1, std::memory_order_relaxed);
            LLAMA_LOG_INFO("[PERF_TRACE][expert_pool_fill] source=prefetch_mmap layer=%d expert=%d bytes=%zu memcpy=%.3f ms\n",
                    L.il, e, fill_bytes, fill_us/1000.0);
        }
    }
    L.fillB[k].store(g_pred_fill_generation.fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_release);
    L.stateB[k].store(2, std::memory_order_release);
    g_work_inflight.fetch_sub((int64_t) L.pool_slot_bytes, std::memory_order_acq_rel);
    g_pred_fills.fetch_add(1, std::memory_order_relaxed);
    finish_prefetch_request(req, PREFETCH_READY);
}

static void prefetch_worker() {
    std::unique_lock<std::mutex> lk(g_work_mu);
    for (;;) {
        g_work_cv.wait(lk, [] {
            return g_work_exit || !g_demand_q.empty() ||
                    (!g_work_q.empty() && g_demand_active.load(std::memory_order_acquire) == 0);
        });
        if (g_work_exit) {
            return;
        }
        std::shared_ptr<prefetch_request> req;
        if (!g_demand_q.empty()) {
            req = g_demand_q.front();
            g_demand_q.pop_front();
        } else {
            req = g_work_q.front();
            g_work_q.pop_front();
        }
        int32_t queued = PREFETCH_QUEUED;
        if (req->cancel.load(std::memory_order_acquire) && !req->demand.load(std::memory_order_acquire)) {
            req->state.store(PREFETCH_CANCELED, std::memory_order_release);
            const uint64_t requested = req->cancel_requested_us.load(std::memory_order_acquire);
            if (requested > 0) {
                record_metric_sample(g_cancel_response_samples_us,
                        (uint64_t) std::max<int64_t>(0, ggml_time_us() - (int64_t) requested));
            }
            account_prefetch_bytes(req);
            auto it = g_request_by_expert.find(prefetch_key(req->il, req->expert));
            if (it != g_request_by_expert.end() && it->second.get() == req.get()) {
                g_request_by_expert.erase(it);
            }
            g_ready_cv.notify_all();
            continue;
        }
        if (!req->state.compare_exchange_strong(queued, PREFETCH_FILLING, std::memory_order_acq_rel)) {
            continue;
        }
        lk.unlock();
        prefetch_fill(req);
        lk.lock();
    }
}

static void pred_shutdown() {
    if (!g_workers.empty() || g_prediction_worker.joinable()) {
        {
            std::lock_guard<std::mutex> lk(g_work_mu);
            g_work_exit = true;
        }
        g_work_cv.notify_all();
        g_prediction_cv.notify_all();
        if (g_prediction_worker.joinable()) {
            g_prediction_worker.join();
        }
        for (auto & t : g_workers) {
            t.join();
        }
    }
}

static bool ticket_actual_contains(const prediction_ticket & ticket, int e) {
    return std::find(ticket.actual.begin(), ticket.actual.end(), e) != ticket.actual.end();
}

static void compute_prediction(const std::shared_ptr<prediction_ticket> & ticket) {
    ticket->compute_start_us = (uint64_t) ggml_time_us();
    const pred_layer & cur = g_pred[ticket->source_pi];
    const pred_layer & target = g_pred[ticket->target_pi];
    const int64_t ne0 = (int64_t) ticket->x.size();
    if ((int64_t) cur.norm.size() != ne0 || (int64_t) target.norm.size() != ne0 || ne0 == 0) {
        ticket->cancel.store(true, std::memory_order_release);
        return;
    }
    const int64_t n_exp = (int64_t) (target.gate.size()/ne0);
    std::vector<float> hv((size_t) ne0);
    std::vector<float> logits((size_t) n_exp);
    std::vector<float> probabilities((size_t) n_exp);
    std::vector<int32_t> idx((size_t) n_exp);
    for (int64_t i = 0; i < ne0; i++) {
        const float w = cur.norm[(size_t) i];
        hv[(size_t) i] = target.norm[(size_t) i]*(w*w > 1e-12f ? ticket->x[(size_t) i]/w : 0.0f);
    }
    float max_logit = -FLT_MAX;
    for (int64_t e = 0; e < n_exp; e++) {
        if (ticket->cancel.load(std::memory_order_acquire)) {
            g_prediction_canceled.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const float * gate = target.gate.data() + (size_t) e*ne0;
        float score = 0.0f;
        for (int64_t i = 0; i < ne0; i++) {
            score += gate[i]*hv[(size_t) i];
        }
        logits[(size_t) e] = score;
        max_logit = std::max(max_logit, score);
        idx[(size_t) e] = (int32_t) e;
    }
    double sum = 0.0;
    for (int64_t e = 0; e < n_exp; e++) {
        const float p = std::exp(logits[(size_t) e] - max_logit);
        probabilities[(size_t) e] = p;
        sum += p;
    }
    double entropy = 0.0;
    for (float & p : probabilities) {
        p = sum > 0.0 ? (float) (p/sum) : 0.0f;
        if (p > 0.0f) {
            entropy -= (double) p*std::log((double) p);
        }
    }
    const float normalized_entropy = n_exp > 1 ? (float) (entropy/std::log((double) n_exp)) : 0.0f;
    const int top_n = std::min((int) n_exp, g_prefetch_max);
    std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(),
            [&](int32_t a, int32_t b) { return logits[(size_t) a] > logits[(size_t) b]; });

    std::vector<int32_t> selected;
    std::vector<int32_t> top_experts;
    std::vector<float> top_probabilities;
    if (top_n > 0) {
        selected.push_back(idx[0]);
    }
    for (int k = 0; k < top_n; k++) {
        top_experts.push_back(idx[(size_t) k]);
        top_probabilities.push_back(probabilities[(size_t) idx[(size_t) k]]);
    }
    if (g_prefetch_entropy_threshold >= 0.0f && g_prefetch_probability_threshold >= 0.0f &&
            normalized_entropy <= g_prefetch_entropy_threshold) {
        for (int k = 1; k < top_n; k++) {
            if (probabilities[(size_t) idx[(size_t) k]] >= g_prefetch_probability_threshold) {
                selected.push_back(idx[(size_t) k]);
            }
        }
    }
    ticket->compute_end_us = (uint64_t) ggml_time_us();

    std::lock_guard<std::mutex> lk(ticket->mu);
    if (ticket->cancel.load(std::memory_order_acquire) || ticket->actual_known) {
        g_prediction_late.fetch_add(1, std::memory_order_relaxed);
        ticket->cancel.store(true, std::memory_order_release);
        return;
    }
    ticket->normalized_entropy = normalized_entropy;
    ticket->top_experts = std::move(top_experts);
    ticket->top_probabilities = std::move(top_probabilities);
    ticket->selected = std::move(selected);
    ticket->selection_reason = ticket->selected.size() > 1 ? "TOP1_PLUS_ENTROPY_PROBABILITY" :
            (g_prefetch_entropy_threshold < 0.0f || g_prefetch_probability_threshold < 0.0f ? "TOP1_ONLY" :
            (normalized_entropy > g_prefetch_entropy_threshold ? "TOP1_ENTROPY_HIGH" : "TOP1_PROBABILITY_LOW"));
    record_metric_sample(g_prediction_queue_samples_us,
            ticket->compute_start_us >= ticket->queued_us ? ticket->compute_start_us - ticket->queued_us : 0);
    record_metric_sample(g_prediction_compute_samples_us,
            ticket->compute_end_us - ticket->compute_start_us);
    if (!g_prefetch_calibrate) {
        bool queued_physical_io = false;
        for (size_t rank = 0; rank < ticket->selected.size(); rank++) {
            const int e = ticket->selected[rank];
            const char * residency = classify_prediction_locked(ticket, ticket->target_pi, e, (int) rank);
            ticket->residency.emplace_back(residency);
            queued_physical_io = queued_physical_io || strcmp(residency, "NONE_ENQUEUED") == 0;
        }
        if (queued_physical_io) {
            g_prediction_tickets_with_io.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ticket->prediction_done = true;
    g_pred_pushes++;
}

static void prediction_worker() {
    std::unique_lock<std::mutex> lk(g_prediction_mu);
    for (;;) {
        g_prediction_cv.wait(lk, [] { return g_work_exit.load(std::memory_order_acquire) || !g_prediction_q.empty(); });
        if (g_work_exit.load(std::memory_order_acquire)) {
            return;
        }
        auto ticket = g_prediction_q.front();
        g_prediction_q.pop_front();
        lk.unlock();
        if (!ticket->cancel.load(std::memory_order_acquire)) {
            compute_prediction(ticket);
        }
        lk.lock();
    }
}

// Called at the beginning of the fused FFN op. Only copying and queueing are
// synchronous; router prediction runs concurrently with gate/up computation.
static void pregate_hook(const ggml_tensor * counts, const ggml_tensor * x) {
    static uint64_t seq = 0;
    const uint64_t cur_seq = seq++;
    if (!counts || !counts->data || !x || x->type != GGML_TYPE_F32 ||
            x->nb[0] != sizeof(float) || x->ne[1] != 1 || x->ne[2] != 1) {
        return;
    }
    auto it = g_pred_ix.find(counts->data);
    if (it == g_pred_ix.end()) {
        return;
    }
    const int source_pi = it->second;
    const int target_pi = g_pred[source_pi].target_ix;
    if (target_pi < 0) {
        return;
    }
    if (g_pred[target_pi].historical_cold_rate < g_prefetch_min_cold_rate) {
        g_prediction_skipped_low_cold.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    auto ticket = std::make_shared<prediction_ticket>();
    ticket->id = g_prediction_id.fetch_add(1, std::memory_order_relaxed) + 1;
    ticket->seq = cur_seq;
    ticket->source_pi = source_pi;
    ticket->target_pi = target_pi;
    ticket->target_layer = g_pred[target_pi].il;
    ticket->queued_us = (uint64_t) ggml_time_us();
    const float * data = (const float *) x->data;
    ticket->x.assign(data, data + x->ne[0]);
    {
        std::lock_guard<std::mutex> lk(g_prediction_mu);
        if (ticket->target_layer >= (int) g_ticket_by_layer.size()) {
            return;
        }
        if (g_ticket_by_layer[(size_t) ticket->target_layer]) {
            g_ticket_by_layer[(size_t) ticket->target_layer]->cancel.store(true, std::memory_order_release);
        }
        g_ticket_by_layer[(size_t) ticket->target_layer] = ticket;
        g_prediction_q.push_back(ticket);
    }
    g_prediction_cv.notify_one();
}

static void pregate_predict_match(const ggml_tensor * counts, const ggml_tensor * ids) {
    if (!counts || !counts->data || !ids || !ids->data || ids->ne[1] != 1) {
        return;
    }
    auto it = g_pred_ix.find(counts->data);
    if (it == g_pred_ix.end()) {
        return;
    }
    const int il = g_pred[it->second].il;
    std::shared_ptr<prediction_ticket> ticket;
    {
        std::lock_guard<std::mutex> lk(g_prediction_mu);
        if (il < 0 || il >= (int) g_ticket_by_layer.size()) {
            return;
        }
        ticket = g_ticket_by_layer[(size_t) il];
    }
    if (!ticket) {
        return;
    }

    std::vector<std::shared_ptr<prefetch_request>> waits;
    {
        std::lock_guard<std::mutex> lk(ticket->mu);
        ticket->actual_known = true;
        ticket->actual.reserve((size_t) ids->ne[0]);
        for (int64_t id = 0; id < ids->ne[0]; id++) {
            ticket->actual.push_back(*(const int32_t *) ((const char *) ids->data + id*ids->nb[0]));
        }
        if (!ticket->prediction_done) {
            ticket->cancel.store(true, std::memory_order_release);
            g_prediction_late.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_predicted_experts += ticket->selected.size();
        for (size_t rank = 0; rank < ticket->selected.size(); rank++) {
            const int e = ticket->selected[rank];
            if (ticket_actual_contains(*ticket, e)) {
                g_predicted_experts_used++;
            }
            if (rank < PREFETCH_RANK_STATS) {
                g_rank_selected[rank].fetch_add(1, std::memory_order_relaxed);
                if (ticket_actual_contains(*ticket, e)) {
                    g_rank_used[rank].fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        for (hot_pin_ref & pin : ticket->hot_pins) {
            if (!ticket_actual_contains(*ticket, pin.expert)) {
                release_hot_pin(pin);
            }
        }
        for (const auto & req : ticket->requests) {
            if (ticket_actual_contains(*ticket, req->expert)) {
                req->actual_correct.store(1, std::memory_order_release);
                req->demand.store(true, std::memory_order_release);
                const int32_t state = req->state.load(std::memory_order_acquire);
                if (state == PREFETCH_QUEUED) {
                    std::lock_guard<std::mutex> qlk(g_work_mu);
                    g_demand_q.push_back(req);
                    g_prefetch_promotions.fetch_add(1, std::memory_order_relaxed);
                }
                if (state < PREFETCH_READY) {
                    waits.push_back(req);
                }
            } else if (req->state.load(std::memory_order_acquire) < PREFETCH_READY) {
                int32_t unknown = -1;
                req->actual_correct.compare_exchange_strong(unknown, 0, std::memory_order_acq_rel);
                request_prefetch_cancel(req, false);
            } else {
                int32_t unknown = -1;
                req->actual_correct.compare_exchange_strong(unknown, 0, std::memory_order_acq_rel);
            }
            account_prefetch_bytes(req);
        }
        FILE * pred_out = g_pred_log ? g_pred_log : (g_prefetch_calibrate ? stderr : nullptr);
        if (pred_out) {
            std::lock_guard<std::mutex> log_lk(g_prediction_mu);
            layer_tier & target_layer = g_layers[ticket->target_layer];
            std::vector<int32_t> actual_cold;
            std::vector<int32_t> prediction_hits;
            std::vector<int32_t> cold_prediction_hits;
            std::vector<bool> top_candidate_cold;
            {
                std::lock_guard<std::mutex> hot_lk(g_hot_mu);
                for (int e : ticket->actual) {
                    const bool hot = tier_atomic_load_i32(&target_layer.lut_host[e]) != target_layer.sentinel;
                    const bool pooled = target_layer.pool_lut &&
                            target_layer.pool_lut[e].load(std::memory_order_acquire) >= 0;
                    if (!hot && !pooled) {
                        actual_cold.push_back(e);
                    }
                    if (std::find(ticket->selected.begin(), ticket->selected.end(), e) != ticket->selected.end()) {
                        prediction_hits.push_back(e);
                        if (!hot && !pooled) {
                            cold_prediction_hits.push_back(e);
                        }
                    }
                }
                for (int e : ticket->top_experts) {
                    const bool hot = tier_atomic_load_i32(&target_layer.lut_host[e]) != target_layer.sentinel;
                    const bool pooled = target_layer.pool_lut &&
                            target_layer.pool_lut[e].load(std::memory_order_acquire) >= 0;
                    top_candidate_cold.push_back(!hot && !pooled);
                }
            }
            fprintf(pred_out, "%s{\"seq\":%llu,\"source_layer\":%d,\"target_layer\":%d,"
                    "\"prediction_queue_us\":%llu,\"prediction_compute_us\":%llu,"
                    "\"selection_reason\":\"%s\",\"expert_bytes\":%zu,"
                    "\"normalized_entropy\":%.8f,\"top_probabilities\":[",
                    pred_out == stderr ? "expert_prefetch_calibration " : "",
                    (unsigned long long) ticket->seq, g_pred[ticket->source_pi].il,
                    ticket->target_layer,
                    (unsigned long long) (ticket->compute_start_us - ticket->queued_us),
                    (unsigned long long) (ticket->compute_end_us - ticket->compute_start_us),
                    ticket->selection_reason.c_str(), target_layer.pool_slot_bytes, ticket->normalized_entropy);
            for (size_t i = 0; i < ticket->top_probabilities.size(); i++) {
                fprintf(pred_out, "%s%.8f", i ? "," : "", ticket->top_probabilities[i]);
            }
            fprintf(pred_out, "],\"top_experts\":[");
            for (size_t i = 0; i < ticket->top_experts.size(); i++) {
                fprintf(pred_out, "%s%d", i ? "," : "", ticket->top_experts[i]);
            }
            fprintf(pred_out, "],\"top_candidate_cold\":[");
            for (size_t i = 0; i < top_candidate_cold.size(); i++) {
                fprintf(pred_out, "%s%s", i ? "," : "", top_candidate_cold[i] ? "true" : "false");
            }
            fprintf(pred_out, "],\"selected\":[");
            for (size_t i = 0; i < ticket->selected.size(); i++) {
                fprintf(pred_out, "%s%d", i ? "," : "", ticket->selected[i]);
            }
            fprintf(pred_out, "],\"residency\":[");
            for (size_t i = 0; i < ticket->residency.size(); i++) {
                fprintf(pred_out, "%s\"%s\"", i ? "," : "", ticket->residency[i].c_str());
            }
            fprintf(pred_out, "],\"actual_experts\":[");
            for (size_t i = 0; i < ticket->actual.size(); i++) {
                fprintf(pred_out, "%s%d", i ? "," : "", ticket->actual[i]);
            }
            fprintf(pred_out, "],\"actual_cold_experts\":[");
            for (size_t i = 0; i < actual_cold.size(); i++) {
                fprintf(pred_out, "%s%d", i ? "," : "", actual_cold[i]);
            }
            fprintf(pred_out, "],\"prediction_hits\":[");
            for (size_t i = 0; i < prediction_hits.size(); i++) {
                fprintf(pred_out, "%s%d", i ? "," : "", prediction_hits[i]);
            }
            fprintf(pred_out, "],\"cold_prediction_hits\":[");
            for (size_t i = 0; i < cold_prediction_hits.size(); i++) {
                fprintf(pred_out, "%s%d", i ? "," : "", cold_prediction_hits[i]);
            }
            fprintf(pred_out, "]}\n");
        }
    }
    if (!waits.empty()) {
        const int64_t wait_start = ggml_time_us();
        note_demand_start();
        g_work_cv.notify_all();
        std::unique_lock<std::mutex> lk(g_work_mu);
        g_ready_cv.wait(lk, [&] {
            for (const auto & req : waits) {
                const int32_t state = req->state.load(std::memory_order_acquire);
                if (state < PREFETCH_READY) {
                    return false;
                }
            }
            return true;
        });
        lk.unlock();
        note_demand_end();
        const uint64_t waited = (uint64_t) (ggml_time_us() - wait_start);
        g_prefetch_waits.fetch_add(1, std::memory_order_relaxed);
        g_prefetch_wait_us.fetch_add(waited, std::memory_order_relaxed);
        record_metric_sample(g_wait_samples_us, waited);
        if (g_perf_trace.load(std::memory_order_relaxed)) {
            LLAMA_LOG_INFO("[PERF_TRACE][expert_prefetch_wait] layer=%d requests=%zu wait=%.3f ms\n",
                    il, waits.size(), waited/1000.0);
            LLAMA_LOG_INFO("[PERF_TRACE][expert_demand_block] kind=prefetch_wait layer=%d requests=%zu total=%.3f ms\n",
                    il, waits.size(), waited/1000.0);
        }
    }
}

static void pred_init(const llama_model & model) {
    if (!g_predict) {
        return;
    }
    const int n_layer = model.hparams.n_layer();
    std::vector<float> raw;
    size_t mirror_bytes = 0;
    for (int il = 0; il < n_layer; il++) {
        if (g_layers[il].ws.empty() || !g_layers[il].sd) {
            continue;
        }
        const ggml_tensor * gate = model.layers[il].ffn_gate_inp;
        // pre-router norm: ffn_norm on classic archs, attn_post_norm on the
        // hybrid qwen35/qwen36 archs (GGUF post_attention_norm)
        const ggml_tensor * norm = model.layers[il].ffn_norm;
        if (!norm) {
            norm = model.layers[il].attn_post_norm;
        }
        if (!gate || !norm || gate->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32) {
            continue;
        }
        const int64_t ne0 = gate->ne[0];
        const int64_t n_exp = gate->ne[1];
        if (n_exp != g_layers[il].n_expert || norm->ne[0] != ne0) {
            continue;
        }
        pred_layer pl;
        pl.il = il;
        pl.n_routed = model.hparams.n_expert_used;
        uint64_t historical_total = 0;
        uint64_t historical_cold = 0;
        for (int e = 0; e < g_layers[il].n_expert; e++) {
            historical_total += g_layers[il].cum[e];
            if (tier_atomic_load_i32(&g_layers[il].lut_host[e]) == g_layers[il].sentinel) {
                historical_cold += g_layers[il].cum[e];
            }
        }
        if (historical_total > 0) {
            pl.historical_cold_rate = (float) ((double) historical_cold/(double) historical_total);
        }
        pl.norm.resize(ne0);
        pl.gate.resize((size_t) n_exp*ne0);
        ggml_backend_tensor_get(norm, pl.norm.data(), 0, ne0*sizeof(float));
        ggml_backend_tensor_get(gate, pl.gate.data(), 0, (size_t) n_exp*ne0*sizeof(float));
        g_pred_ix[g_layers[il].sd->counts->data] = (int) g_pred.size();
        g_pred.push_back(std::move(pl));
        mirror_bytes += (size_t) (n_exp + 1)*ne0*sizeof(float);
    }
    std::vector<int> pred_ix_by_layer(n_layer, -1);
    for (int pi = 0; pi < (int) g_pred.size(); pi++) {
        pred_ix_by_layer[g_pred[pi].il] = pi;
    }
    for (pred_layer & source : g_pred) {
        const int target_il = source.il + PREGATE_LOOKAHEAD;
        if (target_il < n_layer && pred_ix_by_layer[target_il] >= 0) {
            source.target_ix = pred_ix_by_layer[target_il];
            g_pred_pairs++;
        }
    }
    const bool want_worker = g_predict && !g_prefetch_calibrate && g_poolB_alloc > 0 && g_pred_pairs > 0;
    g_predictor_mirror_bytes = mirror_bytes;
    const bool want_predictor = g_predict && g_pred_pairs > 0 &&
            (g_pred_log || want_worker || g_prefetch_calibrate || getenv("LLAMA_EXPERT_STATS") != nullptr);
    if (want_predictor) {
        g_ticket_by_layer.resize((size_t) model.hparams.n_layer());
        tier_resolve_moe_hooks();
        MOE_PREDICT_HOOK(pregate_hook);
        MOE_PREDICT_MATCH_HOOK(pregate_predict_match);
        g_prediction_worker = std::thread(prediction_worker);
    }
    if (want_worker) {
        for (auto & kv : g_stores) {
            store & st = kv.second;
            if (!st.ptrs || !st.ptrs->data) {
                continue;
            }
            layer_tier & L = g_layers[st.il];
            if (!L.poolB || L.n_slotsB <= 0) {
                continue;
            }
            probe_ctx c{ L.lutB.get(), L.stateB.get(), L.ownersB.data(),
                         L.poolB, (int64_t) L.pool_slot_bytes, (int64_t) st.pool_off };
            g_probe_ix[st.ptrs->data] = c;
        }
        tier_resolve_moe_hooks();
        MOE_PROBE_HOOK(pregate_probe);
        MOE_PREFETCH_USE_HOOK(pregate_prefetch_use);
        for (int i = 0; i < g_n_workers; i++) {
            g_workers.emplace_back(prefetch_worker);
        }
    }
    if (want_predictor) {
        atexit(pred_shutdown); // registered after dump_stats: runs first (reverse order)
    }
    char wrk[64] = "";
    if (want_worker) {
        snprintf(wrk, sizeof(wrk), ", prefetching (%d threads)", g_n_workers);
    }
    TIER_LOG("%s: pre-gate predictor: %zu layers, %zu exact L+%d routes, %.1f MiB mirrors%s\n", __func__,
            g_pred.size(), g_pred_pairs, PREGATE_LOOKAHEAD, (double) mirror_bytes/(1024.0*1024.0),
            want_worker ? wrk : (g_prefetch_calibrate ? ", calibration only" :
                    (g_pred_pairs > 0 && g_pred_log ? ", logging" : " (inactive)")));
}



// RAM pool (LLAMA_EXPERT_RAMPOOL): per-layer contiguous blocks, one tier
// below VRAM. cold ops read weights through store::ptrs (pool slot or mmap
// fallback). all mutations happen on the host between compute steps (post
// sched sync); the compute threads only read the tables, lock-free.
#define POOL_ALIGN 64

static size_t pool_align(size_t n) {
    return (n + POOL_ALIGN - 1) & ~(size_t) (POOL_ALIGN - 1);
}

// point every store's ptrs[e] at the expert's current location
static void pool_publish(layer_tier & L, int e) {
    const int k = L.pool_lut[e];
    for (auto & kv : L.ws) {
        store & st = *kv.second;
        if (!st.ptrs) {
            continue;
        }
        const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
        const char * addr = k >= 0 ? L.pool + (size_t) k*L.pool_slot_bytes + st.pool_off
                                   : (const char *) kv.first->data + slice*e;
        ((int64_t *) st.ptrs->data)[e] = (int64_t) (uintptr_t) addr;
    }
}

// copy expert e from the mmap source into pool slot k and publish it; the
// mmap copy is dropped afterwards: the pool is the sole resident copy
static void pool_fill(layer_tier & L, int k, int e) {
    int32_t free_state = 0;
    if (!L.pool_slot_state[k].compare_exchange_strong(free_state, 3, std::memory_order_acquire)) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    size_t fill_bytes = 0;
    for (auto & kv : L.ws) {
        store & st = *kv.second;
        const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
        const char * src = (const char *) kv.first->data + slice*e;
        memcpy(L.pool + (size_t) k*L.pool_slot_bytes + st.pool_off, src, slice);
        fill_bytes += slice;
        if (st.poolable) {
            tier_madvise(src, slice, true);
        }
    }
    const uint64_t fill_us = (uint64_t)(ggml_time_us() - t0);
    g_fetch_us += fill_us;
    if (g_perf_trace.load(std::memory_order_relaxed)) {
        g_trace_pool_fill_us.fetch_add(fill_us, std::memory_order_relaxed);
        g_trace_pool_fill_bytes.fetch_add(fill_bytes, std::memory_order_relaxed);
        g_trace_pool_fills.fetch_add(1, std::memory_order_relaxed);
        LLAMA_LOG_INFO("[PERF_TRACE][expert_pool_fill] source=demand_mmap layer=%d expert=%d bytes=%zu memcpy=%.3f ms\n",
                L.il, e, fill_bytes, fill_us/1000.0);
    }
    L.pool_slot_expert[k] = e;
    L.pool_lut[e] = k;
    L.pool_dwell[k] = 0;
    pool_publish(L, e);
    g_pool_fills++;
}

static void pool_evict(layer_tier & L, int k) {
    const int e = L.pool_slot_expert[k];
    if (e < 0) {
        return;
    }
    L.pool_slot_expert[k] = -1;
    L.pool_lut[e] = -1;
    L.pool_slot_state[k].store(0, std::memory_order_release);
    L.pool_dwell[k] = 0;
    pool_publish(L, e); // back to the mmap fallback
    g_pool_evictions++;
}

static bool parse_heat_csv(const std::string & path, int n_layer,
        std::vector<std::vector<std::pair<int64_t, int32_t>>> & heat) {
    std::ifstream in(path);
    if (!in) {
        LLAMA_LOG_ERROR("%s: cannot open heat csv '%s'\n", __func__, path.c_str());
        return false;
    }
    heat.resize(n_layer);
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<int64_t> vals;
        while (std::getline(ss, tok, ',')) {
            vals.push_back(std::stoll(tok));
        }
        if (vals.size() < 3) {
            continue;
        }
        const int il = (int) vals[0];
        if (il < 0 || il >= n_layer) {
            continue;
        }
        heat[il].emplace_back(vals[2], (int32_t) vals[1]);
    }
    return true;
}

// consume selection counts, update decayed scores + cumulative stats, and
// (LLAMA_EXPERT_ADAPT) repin at most one slot. called at graph build time,
// i.e. between compute steps; counts are written by the fused cold op.
static void maybe_update(layer_tier & L) {
    if (!L.sd || !L.sd->counts->data) {
        return;
    }
    int32_t * cnt = (int32_t *) L.sd->counts->data;
    const int n = L.n_expert;
    if (cnt[n] == 0) {
        return; // no compute since last visit
    }
    const int32_t * mask = (const int32_t *) L.sd->mask->data;

    static const float decay = [] {
        const char * e = getenv("LLAMA_EXPERT_DECAY");
        return e ? (float) atof(e) : 0.999f;
    }();

    for (int e = 0; e < n; e++) {
        if (cnt[e] > 0) {
            g_dirty = true;
        }
        L.score[e] = L.score[e]*decay + (float) cnt[e];
        L.cum[e]  += (uint64_t) cnt[e];
        if (mask[e]) {
            L.cum_cold += (uint64_t) cnt[e];
            if (L.pool) {
                g_pool_cold += (uint64_t) cnt[e];
                if (L.pool_lut[e] >= 0) {
                    g_pool_hits += (uint64_t) cnt[e];
                }
            }
        }
        cnt[e] = 0;
    }
    L.cum_total += (uint64_t) cnt[n];
    L.cum_graphs++;
    cnt[n] = 0;

    // publish completed prefetch fills (READY -> RESIDENT); recycle any whose
    // expert went hot mid-fill or that pool A picked up. then keep ~25% of
    // slots free for the worker by evicting the oldest-timestamped.
    if (L.poolB && L.n_slotsB > 0) {
        for (int k = 0; k < L.n_slotsB; k++) {
            if (L.stateB[k].load(std::memory_order_acquire) != 2) {
                continue;
            }
            const int32_t e = L.ownersB[k];
            if (tier_atomic_load_i32(&L.lut_host[e]) != L.sentinel ||
                    (L.pool_lut && L.pool_lut[e].load(std::memory_order_relaxed) >= 0)) {
                L.lutB[e].store(-1, std::memory_order_release);
                L.ownersB[k] = -1;
                L.stateB[k].store(0, std::memory_order_release);
                continue;
            }
            L.stateB[k].store(3, std::memory_order_release);
            g_pred_published++;
        }
        int free_n = 0;
        for (int k = 0; k < L.n_slotsB; k++) {
            if (L.stateB[k].load(std::memory_order_acquire) == 0) {
                free_n++;
            }
        }
        const int target = L.n_slotsB/4 + 1;
        while (free_n < target) {
            int kv = -1;
            int64_t tmin = INT64_MAX;
            for (int k = 0; k < L.n_slotsB; k++) {
                const int32_t st = L.stateB[k].load(std::memory_order_acquire);
                if (st != 2 && st != 3) {
                    continue;
                }
                const int64_t v = L.lastB[k].load(std::memory_order_acquire);
                if (v < tmin) {
                    tmin = v;
                    kv = k;
                }
            }
            if (kv < 0) {
                break;
            }
            const int32_t e = L.ownersB[kv];
            if (e >= 0) {
                L.lutB[e].store(-1, std::memory_order_release);
            }
            L.ownersB[kv] = -1;
            L.stateB[kv].store(0, std::memory_order_release);
            g_predB_evict++;
            free_n++;
        }
    }

    if (!g_adapt) {
        return;
    }

    {
        // Prediction completes on another thread. Serialize slot identity and
        // generation changes with pin/unpin so a pin can never attach to a
        // slot while that slot is being replaced.
        std::lock_guard<std::mutex> hot_lk(g_hot_mu);

        for (int s = 0; s < L.n_slots; s++) {
            L.dwell[s]++;
        }

    // coldest hot slot (empty slots count as score 0)
    int si = -1;
    float smin = FLT_MAX;
    for (int s = 0; s < L.n_slots; s++) {
        if (s == L.sentinel) {
            continue;
        }
        if (L.hot_pin_count && L.hot_pin_count[s].load(std::memory_order_acquire) > 0) {
            g_hot_eviction_blocks.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const float v = L.slot_expert[s] < 0 ? 0.0f : L.score[L.slot_expert[s]];
        if (v < smin) {
            smin = v;
            si = s;
        }
    }
    // hottest cold expert
    int ec = -1;
    float sc = 0.0f;
    for (int e = 0; e < n; e++) {
        if (mask[e] && L.score[e] > sc) {
            sc = L.score[e];
            ec = e;
        }
    }
        if (si >= 0 && ec >= 0) {
        const int eold = L.slot_expert[si];
        if (eold < 0 || (L.dwell[si] >= 32 && sc > 1.5f*smin)) {
            const bool ec_pooled = L.pool && L.pool_lut[ec] >= 0;
            tier_atomic_store_i32(&L.lut_host[ec], si); // pairs with worker reads
            if (eold >= 0) {
                tier_atomic_store_i32(&L.lut_host[eold], L.sentinel);
            }
            for (auto & kv : L.ws) {
                const ggml_tensor * w = kv.first;
                store & st = *kv.second;
                const size_t slice = ggml_nbytes(w)/n;
                // H2D source: the pool copy when resident (its mmap pages are dropped)
                const char * srcp = st.ptrs ? (const char *) (uintptr_t) ((const int64_t *) st.ptrs->data)[ec]
                                            : (const char *) w->data + slice*ec;
                const int64_t upload_start = g_perf_trace.load(std::memory_order_relaxed) ? ggml_time_us() : 0;
                ggml_backend_tensor_set(st.w_hot, srcp, slice*si, slice);
                if (upload_start) {
                    LLAMA_LOG_INFO("[PERF_TRACE][expert_hot_upload] layer=%d expert=%d tensor=%s bytes=%zu total=%.3f ms\n",
                            L.il, ec, st.w_hot->name, slice, (ggml_time_us() - upload_start)/1000.0);
                }
                if (st.discardable) {
                    tier_madvise((const char *) w->data + slice*ec, slice, true);
                    if (eold >= 0) {
                        tier_madvise((const char *) w->data + slice*eold, slice, false);
                    }
                }
                ggml_backend_tensor_set(st.lut, L.lut_host.data(), 0, n*sizeof(int32_t));
                int32_t * m = (int32_t *) st.mask->data;
                m[ec] = 0;
                if (eold >= 0) {
                    m[eold] = 1;
                }
            }
            if (ec_pooled) {
                // promotion consumed the pool copy; the slot goes back to empty.
                // a FILLING slot is the worker's: the publish pass recycles it
                // next window (the H2D above read ptrs = mmap, correct bytes).
                const int k = L.pool_lut[ec];
                if (L.pool_slot_state[k].load(std::memory_order_acquire) != 1) {
                    L.pool_slot_expert[k] = -1;
                    L.pool_slot_state[k].store(0, std::memory_order_release);
                    L.pool_lut[ec] = -1;
                    L.pool_dwell[k] = 0;
                    pool_publish(L, ec);
                    g_pool_evictions++;
                }
            }
            L.slot_expert[si] = ec;
            L.dwell[si] = 0;
            L.hot_generation[si].fetch_add(1, std::memory_order_acq_rel);
            g_repins++;
        }
        }
    }

    // RAM pool: same hysteresis shape as the hot tier, one level down
    if (L.pool && L.n_pool_slots > 0) {
        for (int k = 0; k < L.n_pool_slots; k++) {
            L.pool_dwell[k]++;
        }
        // coldest pooled slot (empty slots count as score 0); FILLING/READY
        // slots are the publish pass's (defensive: pool A never sees them)
        int ki = -1;
        float pmin = FLT_MAX;
        for (int k = 0; k < L.n_pool_slots; k++) {
            if (L.pool_slot_state[k].load(std::memory_order_acquire) == 1 ||
                L.pool_slot_state[k].load(std::memory_order_acquire) == 2) {
                continue;
            }
            const float v = L.pool_slot_expert[k] < 0 ? 0.0f : L.score[L.pool_slot_expert[k]];
            if (v < pmin) {
                pmin = v;
                ki = k;
            }
        }
        // hottest cold expert that is neither hot nor pooled
        int ep = -1;
        float sp = 0.0f;
        for (int e = 0; e < n; e++) {
            if (mask[e] && L.pool_lut[e] < 0 && L.score[e] > sp) {
                sp = L.score[e];
                ep = e;
            }
        }
        if (ki >= 0 && ep >= 0 && g_pool_fill_budget >= L.pool_slot_bytes) {
            const int ev = L.pool_slot_expert[ki];
            if (ev < 0 || (L.pool_dwell[ki] >= 32 && sp > 1.5f*pmin)) {
                pool_evict(L, ki);
                pool_fill(L, ki, ep);
                g_pool_fill_budget -= L.pool_slot_bytes;
            }
        }
    }
}

static void dump_stats() {
    if (const char * p = getenv("LLAMA_EXPERT_STATS")) {
        FILE * f = strcmp(p, "1") ? fopen(p, "w") : stderr;
        if (f) {
            fprintf(f, "expert_tier: repins %llu\n", (unsigned long long) g_repins);
            if (g_pool_bytes) {
                fprintf(f, "expert_pool: fills %llu evictions %llu hits %llu / %llu cold (%.1f%%) resident %.2f GiB\n",
                        (unsigned long long) g_pool_fills, (unsigned long long) g_pool_evictions,
                        (unsigned long long) g_pool_hits, (unsigned long long) g_pool_cold,
                        g_pool_cold ? 100.0*(double) g_pool_hits/(double) g_pool_cold : 0.0,
                        (double) g_pool_alloc/(double) (1 << 30));
            }
            {
                uint64_t routed = 0;
                uint64_t hot_hits = 0;
                for (const auto & L : g_layers) {
                    routed += L.cum_total;
                    hot_hits += L.cum_total - L.cum_cold;
                }
                fprintf(f, "expert_hot_tier: hits %llu / routed %llu (%.1f%%)\n",
                        (unsigned long long) hot_hits,
                        (unsigned long long) routed,
                        routed ? 100.0*(double) hot_hits/(double) routed : 0.0);
            }
            if (g_pred_pairs > 0) {
                const uint64_t prefetch_fills = g_pred_fills.load(std::memory_order_relaxed);
                const uint64_t prefetch_first_decode_used = g_pred_first_decode_used.load(std::memory_order_relaxed);
                fprintf(f, "expert_predict: pushes %llu fills %llu published %llu specevict %llu probe %llu/%llu drop hot %llu dup %llu budget %llu full %llu\n",
                        (unsigned long long) g_pred_pushes, (unsigned long long) prefetch_fills,
                        (unsigned long long) g_pred_published, (unsigned long long) g_predB_evict,
                        (unsigned long long) g_probe_hits.load(), (unsigned long long) g_probe_miss.load(),
                        (unsigned long long) g_pred_drop_hot.load(std::memory_order_relaxed),
                        (unsigned long long) g_pred_drop_dup.load(std::memory_order_relaxed),
                        (unsigned long long) g_pred_drop_budget.load(std::memory_order_relaxed),
                        (unsigned long long) g_pred_drop_full.load(std::memory_order_relaxed));
                fprintf(f, "expert_prefetch_effectiveness: first_decode_used %llu / completed %llu (%.1f%%)\n",
                        (unsigned long long) prefetch_first_decode_used,
                        (unsigned long long) prefetch_fills,
                        prefetch_fills ? 100.0*(double) prefetch_first_decode_used/(double) prefetch_fills : 0.0);
                fprintf(f, "expert_predict_accuracy: routed_used %llu / predicted %llu (%.1f%%)\n",
                        (unsigned long long) g_predicted_experts_used,
                        (unsigned long long) g_predicted_experts,
                        g_predicted_experts ? 100.0*(double) g_predicted_experts_used/(double) g_predicted_experts : 0.0);
                fprintf(f, "expert_prefetch_priority: promoted %llu waits %llu wait_us %llu chunk_cancels %llu demand_active %d\n",
                        (unsigned long long) g_prefetch_promotions.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_waits.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_wait_us.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_chunk_cancels.load(std::memory_order_relaxed),
                        (int) g_demand_active.load(std::memory_order_relaxed));
                fprintf(f, "expert_prefetch_wait_latency_us: p50 %llu p95 %llu p99 %llu samples %zu\n",
                        (unsigned long long) metric_percentile(g_wait_samples_us, 0.50),
                        (unsigned long long) metric_percentile(g_wait_samples_us, 0.95),
                        (unsigned long long) metric_percentile(g_wait_samples_us, 0.99), g_wait_samples_us.size());
                fprintf(f, "expert_prefetch_cancel_latency_us: p50 %llu p95 %llu p99 %llu samples %zu\n",
                        (unsigned long long) metric_percentile(g_cancel_response_samples_us, 0.50),
                        (unsigned long long) metric_percentile(g_cancel_response_samples_us, 0.95),
                        (unsigned long long) metric_percentile(g_cancel_response_samples_us, 0.99),
                        g_cancel_response_samples_us.size());
                fprintf(f, "expert_prediction_latency_us: queue_p50 %llu queue_p95 %llu queue_p99 %llu compute_p50 %llu compute_p95 %llu compute_p99 %llu samples %zu\n",
                        (unsigned long long) metric_percentile(g_prediction_queue_samples_us, 0.50),
                        (unsigned long long) metric_percentile(g_prediction_queue_samples_us, 0.95),
                        (unsigned long long) metric_percentile(g_prediction_queue_samples_us, 0.99),
                        (unsigned long long) metric_percentile(g_prediction_compute_samples_us, 0.50),
                        (unsigned long long) metric_percentile(g_prediction_compute_samples_us, 0.95),
                        (unsigned long long) metric_percentile(g_prediction_compute_samples_us, 0.99),
                        g_prediction_compute_samples_us.size());
                fprintf(f, "expert_prediction_work: compute_us %llu tickets %llu tickets_with_physical_io %llu selected_hot %llu selected_demand_pool %llu selected_prefetch_ready %llu selected_prefetch_filling %llu selected_enqueued %llu prefetch_copy_us %llu skipped_low_cold_rate %llu\n",
                        (unsigned long long) metric_sum(g_prediction_compute_samples_us),
                        (unsigned long long) g_pred_pushes,
                        (unsigned long long) g_prediction_tickets_with_io.load(std::memory_order_relaxed),
                        (unsigned long long) g_residency_hot.load(std::memory_order_relaxed),
                        (unsigned long long) g_residency_demand.load(std::memory_order_relaxed),
                        (unsigned long long) g_residency_prefetch_ready.load(std::memory_order_relaxed),
                        (unsigned long long) g_residency_prefetch_filling.load(std::memory_order_relaxed),
                        (unsigned long long) g_residency_enqueued.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_copy_us.load(std::memory_order_relaxed),
                        (unsigned long long) g_prediction_skipped_low_cold.load(std::memory_order_relaxed));
                for (int rank = 0; rank < std::min(g_prefetch_max, PREFETCH_RANK_STATS); rank++) {
                    fprintf(f, "expert_prediction_rank %d: selected %llu used %llu enqueued %llu correct_bytes %llu wrong_bytes %llu\n",
                            rank + 1,
                            (unsigned long long) g_rank_selected[rank].load(std::memory_order_relaxed),
                            (unsigned long long) g_rank_used[rank].load(std::memory_order_relaxed),
                            (unsigned long long) g_rank_enqueued[rank].load(std::memory_order_relaxed),
                            (unsigned long long) g_rank_correct_bytes[rank].load(std::memory_order_relaxed),
                            (unsigned long long) g_rank_wrong_bytes[rank].load(std::memory_order_relaxed));
                }
                fprintf(f, "expert_prefetch_lifecycle: merges %llu cancel_requests %llu expired %llu demand_deferred %llu hot_eviction_blocks %llu\n",
                        (unsigned long long) g_prefetch_merges.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_cancel_requests.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_expired_requests.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_demand_deferred.load(std::memory_order_relaxed),
                        (unsigned long long) g_hot_eviction_blocks.load(std::memory_order_relaxed));
                fprintf(f, "expert_prefetch_bytes: correct %llu wrong %llu\n",
                        (unsigned long long) g_prefetch_correct_bytes.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_wrong_bytes.load(std::memory_order_relaxed));
                uint64_t state_free = 0, state_filling = 0, state_ready = 0, state_resident = 0;
                for (const auto & layer : g_layers) {
                    for (int k = 0; k < layer.n_slotsB; k++) {
                        switch (layer.stateB[k].load(std::memory_order_acquire)) {
                            case 0: state_free++; break;
                            case 1: state_filling++; break;
                            case 2: state_ready++; break;
                            case 3: state_resident++; break;
                        }
                    }
                }
                fprintf(f, "expert_prefetch_safety: partial_probe_rejects %llu duplicate_claims %llu stale_unpins %llu slot_free %llu slot_filling %llu slot_ready %llu slot_resident %llu\n",
                        (unsigned long long) g_prefetch_partial_probe_rejects.load(std::memory_order_relaxed),
                        (unsigned long long) g_prefetch_duplicate_claims.load(std::memory_order_relaxed),
                        (unsigned long long) g_hot_stale_unpins.load(std::memory_order_relaxed),
                        (unsigned long long) state_free, (unsigned long long) state_filling,
                        (unsigned long long) state_ready, (unsigned long long) state_resident);
                fprintf(f, "expert_prefetch_hot_lock: pins %llu unpins %llu late_predictions %llu canceled_predictions %llu\n",
                        (unsigned long long) g_hot_pins.load(std::memory_order_relaxed),
                        (unsigned long long) g_hot_unpins.load(std::memory_order_relaxed),
                        (unsigned long long) g_prediction_late.load(std::memory_order_relaxed),
                        (unsigned long long) g_prediction_canceled.load(std::memory_order_relaxed));
            }
            fprintf(f, "expert_memory_bytes: model_mmap %zu hot_backend %zu demand_pool %zu prefetch_pool %zu pread_ring %zu predictor_mirror %zu\n",
                    g_model_mmap_bytes, g_hot_alloc, g_pool_alloc, g_poolB_alloc,
                    g_stage_alloc, g_predictor_mirror_bytes);
            fprintf(f, "expert_initialization_us: total %llu hot_fill %llu demand_pool_fill %llu\n",
                    (unsigned long long) g_init_total_us,
                    (unsigned long long) g_init_hot_fill_us,
                    (unsigned long long) g_init_demand_pool_fill_us);
            if (g_stage_n > 0) {
                fprintf(f, "expert_pread: fetches %llu bytes %llu stalls %llu fallbacks %llu fails %llu\n",
                        (unsigned long long) g_stage_fetches.load(),
                        (unsigned long long) g_stage_bytes.load(),
                        (unsigned long long) g_stage_stalls.load(),
                        (unsigned long long) g_stage_fallbacks.load(),
                        (unsigned long long) g_stage_fails.load());
            }
            {
                const moe_timing_snapshot timing = moe_timing_snapshot_get();
                const uint64_t down_tail = timing.total > timing.setup + timing.gate_up + timing.activation
                    ? timing.total - timing.setup - timing.gate_up - timing.activation : 0;
                fprintf(f, "expert_timers: steps %llu pool_fill %llu us (%.2f ms/step) cold_total %llu us (%.2f ms/step)\n",
                        (unsigned long long) g_steps,
                        (unsigned long long) g_fetch_us,
                        g_steps ? (double) g_fetch_us / (double) g_steps / 1000.0 : 0.0,
                        (unsigned long long) timing.total,
                        g_steps ? (double) timing.total / (double) g_steps / 1000.0 : 0.0);
                fprintf(f, "expert_timers_cold: setup %llu us gate_up %llu us activation %llu us down_and_sync %llu us\n",
                        (unsigned long long) timing.setup,
                        (unsigned long long) timing.gate_up,
                        (unsigned long long) timing.activation,
                        (unsigned long long) down_tail);
            }
            for (const auto & L : g_layers) {
                if (L.ws.empty() || L.cum_total == 0) {
                    continue;
                }
                fprintf(f, "layer %2d: cold %llu / %llu (%.1f%%) graphs %llu\n", L.il,
                        (unsigned long long) L.cum_cold, (unsigned long long) L.cum_total,
                        100.0*(double) L.cum_cold/(double) L.cum_total,
                        (unsigned long long) L.cum_graphs);
            }
            // invariant: mask agrees with lut, slot<->expert bijection holds
            for (const auto & L : g_layers) {
                if (!L.sd) {
                    continue;
                }
                const int32_t * m = (const int32_t *) L.sd->mask->data;
                int bad = 0;
                for (int e = 0; e < L.n_expert; e++) {
                    const int want = (L.lut_host[e] == L.sentinel) ? 1 : 0;
                    if (m[e] != want) {
                        bad++;
                    }
                }
                for (int s = 0; s < L.n_slots; s++) {
                    if (s == L.sentinel || L.slot_expert[s] < 0) {
                        continue;
                    }
                    if (L.lut_host[L.slot_expert[s]] != s) {
                        bad++;
                    }
                }
                if (L.pool) {
                    for (int k = 0; k < L.n_pool_slots; k++) {
                        if (L.pool_slot_expert[k] >= 0 && L.pool_lut[L.pool_slot_expert[k]] != k) {
                            bad++;
                        }
                    }
                    for (int e = 0; e < L.n_expert; e++) {
                        const int k = L.pool_lut[e];
                        if (k >= 0 && L.pool_slot_expert[k] != e) {
                            bad++; // pool<->expert bijection, all claimed states
                        } else if (k >= 0 && L.pool_slot_state[k] == 3 && m[e] != 1) {
                            bad++; // resident implies cold
                        }
                    }
                }
                if (L.poolB) {
                    for (int e = 0; e < L.n_expert; e++) {
                        const int k = L.lutB[e];
                        if (k >= 0 && L.ownersB[k] != e) {
                            bad++;
                        }
                    }
                    for (int k = 0; k < L.n_slotsB; k++) {
                        if (L.ownersB[k] >= 0 && (int32_t) L.lutB[L.ownersB[k]].load() != k) {
                            bad++;
                        }
                    }
                }
                if (bad) {
                    fprintf(f, "layer %2d: INVARIANT VIOLATIONS %d\n", L.il, bad);
                }
            }
            if (f != stderr) {
                fclose(f);
            }
        }
    }
    if (const char * p = getenv("LLAMA_EXPERT_USAGE")) {
        FILE * f = fopen(p, "w");
        if (f) {
            fprintf(f, "layer,expert,count\n");
            for (const auto & L : g_layers) {
                if (L.cum.empty()) {
                    continue;
                }
                for (int e = 0; e < L.n_expert; e++) {
                    if (L.cum[e] > 0) {
                        fprintf(f, "%d,%d,%llu\n", L.il, e, (unsigned long long) L.cum[e]);
                    }
                }
            }
            fclose(f);
        }
    }
}

void update() {
    std::vector<std::shared_ptr<prediction_ticket>> finished;
    {
        std::lock_guard<std::mutex> lk(g_prediction_mu);
        for (auto & ticket : g_ticket_by_layer) {
            if (ticket) {
                finished.push_back(std::move(ticket));
            }
        }
    }
    for (auto & ticket : finished) {
        ticket->cancel.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(ticket->mu);
        for (const auto & req : ticket->requests) {
            if (req->state.load(std::memory_order_acquire) < PREFETCH_READY &&
                    !req->demand.load(std::memory_order_acquire)) {
                request_prefetch_cancel(req, true);
            }
        }
        for (hot_pin_ref & pin : ticket->hot_pins) {
            release_hot_pin(pin);
        }
    }
    g_pred_step.fetch_add(1, std::memory_order_relaxed);
    g_steps++;
    for (auto & L : g_layers) {
        g_pool_fill_budget = 16 << 20;
        maybe_update(L);
    }
    if (g_timing_interval && g_steps % g_timing_interval == 0) {
        const moe_timing_snapshot timing = moe_timing_snapshot_get();
        const uint64_t down_tail = timing.total > timing.setup + timing.gate_up + timing.activation
            ? timing.total - timing.setup - timing.gate_up - timing.activation : 0;
        TIER_LOG("expert_timers: step %llu pool_fill %.2f ms cold %.2f ms [setup %.2f gate_up %.2f activation %.2f down_and_sync %.2f]\n",
                (unsigned long long) g_steps, g_fetch_us/1000.0, timing.total/1000.0,
                timing.setup/1000.0, timing.gate_up/1000.0, timing.activation/1000.0, down_tail/1000.0);
    }
}

static moe_timing_snapshot g_trace_cold_begin;

void set_perf_trace(bool enabled) {
    g_perf_trace.store(enabled, std::memory_order_relaxed);
    if (enabled) {
        tier_resolve_moe_hooks();
        LLAMA_LOG_INFO("[PERF_TRACE] expert tier tracing enabled\n");
    }
}

void perf_trace_begin() {
    if (!g_perf_trace.load(std::memory_order_relaxed)) {
        return;
    }
    g_trace_ssd_read_us.store(0, std::memory_order_relaxed);
    g_trace_ssd_read_bytes.store(0, std::memory_order_relaxed);
    g_trace_ssd_reads.store(0, std::memory_order_relaxed);
    g_trace_stage_wait_us.store(0, std::memory_order_relaxed);
    g_trace_stage_waits.store(0, std::memory_order_relaxed);
    g_trace_pool_fill_us.store(0, std::memory_order_relaxed);
    g_trace_pool_fill_bytes.store(0, std::memory_order_relaxed);
    g_trace_pool_fills.store(0, std::memory_order_relaxed);
    g_trace_cold_begin = moe_timing_snapshot_get();
}

void perf_trace_end() {
    if (!g_perf_trace.load(std::memory_order_relaxed)) {
        return;
    }
    const moe_timing_snapshot end = moe_timing_snapshot_get();
    const uint64_t cold_us = end.total - g_trace_cold_begin.total;
    const uint64_t setup_us = end.setup - g_trace_cold_begin.setup;
    const uint64_t gate_up_us = end.gate_up - g_trace_cold_begin.gate_up;
    const uint64_t activation_us = end.activation - g_trace_cold_begin.activation;
    const uint64_t down_sync_us = cold_us > setup_us + gate_up_us + activation_us
        ? cold_us - setup_us - gate_up_us - activation_us : 0;
    LLAMA_LOG_INFO("[PERF_TRACE][expert_io_summary] ssd_reads=%llu bytes=%llu read=%.3f ms "
            "stage_waits=%llu wait=%.3f ms pool_fills=%llu pool_bytes=%llu pool_memcpy=%.3f ms\n",
            (unsigned long long) g_trace_ssd_reads.load(std::memory_order_relaxed),
            (unsigned long long) g_trace_ssd_read_bytes.load(std::memory_order_relaxed),
            g_trace_ssd_read_us.load(std::memory_order_relaxed)/1000.0,
            (unsigned long long) g_trace_stage_waits.load(std::memory_order_relaxed),
            g_trace_stage_wait_us.load(std::memory_order_relaxed)/1000.0,
            (unsigned long long) g_trace_pool_fills.load(std::memory_order_relaxed),
            (unsigned long long) g_trace_pool_fill_bytes.load(std::memory_order_relaxed),
            g_trace_pool_fill_us.load(std::memory_order_relaxed)/1000.0);
    LLAMA_LOG_INFO("[PERF_TRACE][moe_cold_summary] total=%.3f ms setup=%.3f ms gate_up=%.3f ms "
            "activation=%.3f ms down_and_sync=%.3f ms\n",
            cold_us/1000.0, setup_us/1000.0, gate_up_us/1000.0,
            activation_us/1000.0, down_sync_us/1000.0);
}

size_t expert_weight_bytes(const llama_model & model) {
    size_t b = 0;
    for (int il = 0; il < (int) model.hparams.n_layer(); il++) {
        const llama_layer & l = model.layers[il];
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (w) {
                b += ggml_nbytes(w);
            }
        }
    }
    return b;
}

void configure_prefetch(int slots_per_layer, int max_experts, int chunk_mb, bool calibrate) {
    if (!g_layers.empty()) {
        const size_t chunk_bytes = (size_t) std::max(1, chunk_mb)*1024*1024;
        if (g_prefetch_slots == slots_per_layer &&
                g_prefetch_max == std::max(1, max_experts) &&
                g_prefetch_chunk_bytes == chunk_bytes &&
                g_prefetch_calibrate == calibrate) {
            return;
        }
        throw std::runtime_error("expert prefetch configuration changed after tier initialization");
    }
    g_prefetch_slots = slots_per_layer;
    g_prefetch_max = std::max(1, max_experts);
    g_prefetch_chunk_bytes = (size_t) std::max(1, chunk_mb)*1024*1024;
    g_prefetch_calibrate = calibrate;
}

void init(const llama_model & model) {
    if (!g_layers.empty()) {
        return; // already initialized
    }
    const int64_t init_start_us = ggml_time_us();
    g_model_mmap_bytes = model.mmap_size();
    const char * env_s   = getenv("LLAMA_EXPERT_S");
    const char * env_hot = getenv("LLAMA_EXPERT_HOT");
    if (const char * e = getenv("LLAMA_EXPERT_ADAPT")) {
        g_adapt = atoi(e) != 0;
    } else {
        g_adapt = true; // Auto-fit and online adaptation ON by default
    }
    if (const char * e = getenv("LLAMA_EXPERT_MADVISE")) {
        g_madvise = atoi(e) != 0;
    }
    if (const char * e = getenv("LLAMA_EXPERT_RAMPOOL")) {
        g_pool_bytes = (size_t) (atof(e)*1073741824.0);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREDICT")) {
        g_predict = atoi(e) != 0;
    }
    // Calibration is itself a request to run the predictor. It suppresses
    // physical prefetch below, so it is safe even without the environment flag.
    if (g_prefetch_calibrate) {
        g_predict = true;
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_ENTROPY")) {
        g_prefetch_entropy_threshold = (float) atof(e);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_PROBABILITY")) {
        g_prefetch_probability_threshold = (float) atof(e);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_MIN_COLD_RATE")) {
        g_prefetch_min_cold_rate = std::clamp((float) atof(e), 0.0f, 1.0f);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREDICT_LOG")) {
        if (e[0]) {
            g_pred_log = fopen(e, "w");
            if (!g_pred_log) {
                TIER_LOG("%s: cannot open predict log '%s'\n", __func__, e);
            }
            std::string rpath(e);
            const auto dot = rpath.rfind('.');
            if (dot != std::string::npos) {
                rpath = rpath.substr(0, dot) + ".route" + rpath.substr(dot);
            } else {
                rpath += ".route";
            }
            g_route_log = fopen(rpath.c_str(), "w");
            if (!g_route_log) {
                TIER_LOG("%s: cannot open route trace '%s'\n", __func__, rpath.c_str());
            }
        }
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_MB")) {
        g_work_budget = (int64_t) (atof(e)*1048576.0);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_GB")) {
        g_poolB_bytes = (size_t) (atof(e)*1073741824.0);
    }
    if (g_prefetch_calibrate) {
        g_poolB_bytes = 0;
        TIER_LOG("%s: expert prefetch calibration enabled; physical prefetch disabled\n", __func__);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_THREADS")) {
        g_n_workers = std::min(8, std::max(1, atoi(e)));
    }
    const bool want_pread = [] {
        const char * e = getenv("LLAMA_EXPERT_PREAD");
        return !e || atoi(e) != 0;
    }();

    bool manual_S = false;
    if (env_s) {
        g_S = std::max(0, atoi(env_s));
        manual_S = (env_s != nullptr);
    }

    if (const char * e = getenv("LLAMA_EXPERT_TMAX")) {
        g_tmax = std::max(0, atoi(e));
    }
    if (const char * e = getenv("LLAMA_EXPERT_TIMING")) {
        const int interval = atoi(e);
        g_timing_interval = interval > 0 ? (uint64_t) interval : 0;
    }

    const int n_layer  = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;

    if (g_route_log) {
        tier_resolve_moe_hooks();
        MOE_ROUTE_HOOK(g_route_log, n_layer);
    }

    std::vector<std::vector<std::pair<int64_t, int32_t>>> heat;
    if (env_hot && env_hot[0] && !parse_heat_csv(env_hot, n_layer, heat)) {
        return;
    }

    // sidecar persistence: compute fingerprint, set path, load if no HOT override
    g_fingerprint = compute_fingerprint(model);
    const std::string mpath = model.path();
    if (!mpath.empty()) {
        g_sidecar_path = mpath + ".tier";
    }
    if (!env_hot || !env_hot[0]) {
        const auto sidecar_heat = load_sidecar(model);
        if (!sidecar_heat.empty()) {
            heat.resize(n_layer);
            for (int il = 0; il < n_layer && il < (int) sidecar_heat.size(); il++) {
                for (int e = 0; e < n_expert && e < (int) sidecar_heat[il].size(); e++) {
                    if (sidecar_heat[il][e] > 0.0f) {
                        heat[il].emplace_back((int64_t) sidecar_heat[il][e], e);
                    }
                }
            }
        }
    } else {
        TIER_LOG("%s: LLAMA_EXPERT_HOT set, ignoring sidecar\n", __func__);
    }

    if (getenv("LLAMA_EXPERT_STATS") || getenv("LLAMA_EXPERT_USAGE")) {
        atexit(dump_stats);
    }
    if (getenv("LLAMA_EXPERT_STATS") || g_timing_interval) {
        // The CPU backend may be runtime-loaded. Resolve the timer even when
        // prediction/pread are disabled, otherwise timing would read as zero.
        tier_resolve_moe_hooks();
    }
    atexit(save_sidecar);

ggml_backend_dev_t dev =
    ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

// Intel / AMD 集成 GPU 在 Vulkan 后端中会被标记为 IGPU
if (!dev) {
    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

if (!dev) {
    TIER_LOG(
        "%s: expert tiering disabled (no GPU or IGPU device found)\n",
        __func__);
    return;
}

    // pread demand fetch: init + KAT
    if (want_pread) {
        const std::string mpath = model.path();
        void * mbase = model.mmap_base();
        const ggml_tensor * kat_w = nullptr;
        for (int il = 0; il < n_layer && !kat_w; il++) {
            const llama_layer & l = model.layers[il];
            for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
                if (w && w->data && ggml_backend_buffer_is_host(w->buffer)) {
                    kat_w = w;
                    break;
                }
            }
        }
        if (mpath.empty()) {
            TIER_LOG("%s: pread disabled (no model path)\n", __func__);
            g_pread_disabled = true;
        } else if (!mbase) {
            TIER_LOG("%s: pread disabled (no mmap; --no-mmap?)\n", __func__);
            g_pread_disabled = true;
        } else if (!kat_w) {
            TIER_LOG("%s: pread disabled (no host expert weights)\n", __func__);
            g_pread_disabled = true;
        } else if (!tier_pread_init(mpath.c_str())) {
            TIER_LOG("%s: pread disabled (open failed)\n", __func__);
        } else if (!tier_pread_kat(kat_w->data, (size_t) ((const char *) kat_w->data - (const char *) mbase))) {
            TIER_LOG("%s: pread disabled (KAT failed)\n", __func__);
            tier_pread_close();
        } else {
            TIER_LOG("%s: pread KAT passed\n", __func__);
            atexit(tier_pread_close);
        }
    }

    // Calculate per-expert slot memory size across all layers for Auto-Fit
    size_t bytes_per_slot_all_layers = 0;
    int n_tensors = 0;
    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (w && ggml_backend_buffer_is_host(w->buffer)) {
                n_tensors++;
                if (n_expert > 0) {
                    bytes_per_slot_all_layers += ggml_nbytes(w) / n_expert;
                }
            }
        }
    }

    if (n_tensors == 0) {
        TIER_LOG("%s: expert tiering disabled (no host expert weights found)\n", __func__);
        return;
    }

    // Reserve 512 MB for the CUDA runtime, display, alignment and the graph
    // capture buffers that allocate after this sizing (300 MB let capture OOM)
    const size_t safety_buffer = 512ULL * 1024 * 1024;
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    const size_t usable_vram = (free_vram > safety_buffer) ? (free_vram - safety_buffer) : 0;

    if (!manual_S && bytes_per_slot_all_layers > 0) {
        if (usable_vram >= bytes_per_slot_all_layers) {
            int autofit_s = (int) (usable_vram / bytes_per_slot_all_layers);
            g_S = std::clamp(autofit_s, 1, n_expert);
            TIER_LOG("%s: AUTO-FIT ENGINE -> set S = %d (Free VRAM: %.2f / %.2f GB, per-slot size: %.2f MB, total experts: %d)\n",
                           __func__, g_S, (double)free_vram / (1024.0*1024.0*1024.0),
                           (double)total_vram / (1024.0*1024.0*1024.0),
                           (double)bytes_per_slot_all_layers / (1024.0*1024.0), n_expert);
        } else {
            g_S = 0;
            TIER_LOG("%s: AUTO-FIT ENGINE -> Insufficient free VRAM (%.2f MB free vs %.2f MB required per slot), expert tiering on GPU disabled\n",
                           __func__, (double)free_vram / (1024.0*1024.0), (double)bytes_per_slot_all_layers / (1024.0*1024.0));
        }
    }

    // clamp a forced S to what fits; the unclamped path OOMs at graph capture
    if (manual_S && bytes_per_slot_all_layers > 0) {
        const int afford = (int) (usable_vram / bytes_per_slot_all_layers);
        if (g_S > afford) {
            TIER_LOG("%s: manual S = %d exceeds free VRAM, clamping to %d\n", __func__, g_S, afford);
            g_S = std::max(afford, 0);
        }
    }

    if (g_S <= 0) {
        TIER_LOG("%s: expert tiering disabled (S=0)\n", __func__);
        return;
    }

    TIER_LOG("%s: page hints: %s\n", __func__, g_madvise ? "on" : "off");
    ggml_backend_buffer_type_t buft_gpu = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();

    struct ggml_init_params ip_gpu = {
        /*.mem_size   =*/ ggml_tensor_overhead()*2*n_tensors + 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_init_params ip_cpu = {
        /*.mem_size   =*/ ggml_tensor_overhead()*3*n_tensors + 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    g_ctx_gpu = ggml_init(ip_gpu);
    g_ctx_cpu = ggml_init(ip_cpu);

    std::vector<int32_t> lut_host(n_expert);
    std::vector<int32_t> mask_host(n_expert);
    std::vector<char>    zeros;

    size_t total_bytes = 0;
    int64_t hits = 0, total = 0;
    const bool need_indirect_ptrs = g_pool_bytes > 0 ||
            (g_predict && !g_prefetch_calibrate && (g_prefetch_slots > 0 || g_poolB_bytes > 0));

    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];

        // top-S experts of this layer by heat
        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            int64_t layer_total = 0;
            for (const auto & p : h) {
                layer_total += p.first;
            }
            for (int i = 0; i < (int) h.size() && i < g_S; i++) {
                top.push_back(h[i].second);
                hits += h[i].first;
            }
            total += layer_total;
        }
        if (top.empty()) {
            if (!g_adapt) {
                continue;
            }
            for (int i = 0; i < std::min(g_S, n_expert); i++) {
                top.push_back(i);
            }
        }

        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) top.size());
        std::fill(mask_host.begin(), mask_host.end(), 1);
        for (int s = 0; s < (int) top.size(); s++) {
            lut_host[top[s]]  = s;
            mask_host[top[s]] = 0;
        }

        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (!w || !ggml_backend_buffer_is_host(w->buffer)) {
                continue;
            }
            if (w->ne[3] != 1 || (int) w->ne[2] != n_expert) {
                continue;
            }
            const size_t slice = ggml_nbytes(w)/n_expert;

            store s;
            s.w_hot  = ggml_new_tensor_3d(g_ctx_gpu, w->type, w->ne[0], w->ne[1], g_adapt ? g_S + 1 : (int) top.size() + 1);
            s.lut    = ggml_new_tensor_2d(g_ctx_gpu, GGML_TYPE_I32, 1, n_expert);
            s.mask   = ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I32, n_expert);
            s.counts = ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I32, n_expert + 1);
            s.il      = il;
            s.is_down = (w == l.ffn_down_exps);
            s.poolable    = w->data && model.weights_discardable(w->data);
            s.discardable = g_madvise && s.poolable;
            s.ptrs = need_indirect_ptrs ? ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I64, n_expert) : nullptr;

            ggml_set_name(s.w_hot, (std::string(w->name) + ".hot").c_str());

            // tensors are allocated lazily below per-context; collect first
            // (allocation happens after all tensors are created)
            g_stores[w] = s;
            total_bytes += ggml_nbytes(s.w_hot);
            (void) slice;
        }

        // per-layer tensors are ready; fill them after global allocation
    }

    // allocate all tensors
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors_from_buft(g_ctx_gpu, buft_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(g_ctx_cpu, buft_cpu);
    if (!buf_gpu) {
        TIER_LOG("%s: expert tiering GPU allocation failed (VRAM full), falling back to CPU host execution\n", __func__);
        return;
    }
    ggml_backend_buffer_set_usage(buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (buf_cpu) {
        ggml_backend_buffer_set_usage(buf_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // fill lut/mask/hot weights
    const int64_t hot_fill_start_us = ggml_time_us();
    g_layers.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];

        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int i = 0; i < (int) h.size() && i < g_S; i++) {
                top.push_back(h[i].second);
            }
        }
        if (top.empty()) {
            if (!g_adapt) {
                continue;
            }
            for (int i = 0; i < std::min(g_S, n_expert); i++) {
                top.push_back(i);
            }
        }

        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) top.size());
        std::fill(mask_host.begin(), mask_host.end(), 1);
        for (int s = 0; s < (int) top.size(); s++) {
            lut_host[top[s]]  = s;
            mask_host[top[s]] = 0;
        }

        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            auto it = g_stores.find(w);
            if (it == g_stores.end()) {
                continue;
            }
            store & s = it->second;
            const size_t slice = ggml_nbytes(w)/n_expert;

            ggml_backend_tensor_set(s.lut,  lut_host.data(),  0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(s.mask, mask_host.data(), 0, n_expert*sizeof(int32_t));
            if (s.ptrs) {
                for (int e = 0; e < n_expert; e++) {
                    ((int64_t *) s.ptrs->data)[e] = (int64_t) (uintptr_t) ((const char *) w->data + slice*e);
                }
            }

            if (zeros.size() < slice) {
                zeros.assign(slice, 0);
            }
            if (s.w_hot && s.w_hot->buffer) {
                ggml_backend_tensor_set(s.w_hot, zeros.data(), slice*top.size(), slice);
                for (int sl = 0; sl < (int) top.size(); sl++) {
                    const void * src_ptr = w->data ? (const char *) w->data + slice*top[sl] : zeros.data();
                    ggml_backend_tensor_set(s.w_hot, src_ptr, slice*sl, slice);
                    if (s.discardable && w->data) {
                        tier_madvise(src_ptr, slice, true);
                    }
                }
            }
        }

        // group the stores of this layer for stats/repin
        layer_tier L;
        L.il       = il;
        L.n_expert = n_expert;
        L.sentinel = (int) top.size();
        L.n_slots  = g_adapt ? g_S + 1 : (int) top.size() + 1;
        L.slot_expert.assign(L.n_slots, -1);
        L.dwell.assign(L.n_slots, 0);
        L.hot_pin_count.reset(new std::atomic<int32_t>[L.n_slots]);
        L.hot_generation.reset(new std::atomic<uint64_t>[L.n_slots]);
        for (int s = 0; s < L.n_slots; s++) {
            L.hot_pin_count[s].store(0, std::memory_order_relaxed);
            L.hot_generation[s].store(1, std::memory_order_relaxed);
        }
        L.lut_host = lut_host;
        L.score.assign(n_expert, 0.0f);
        L.cum.assign(n_expert, 0);
        if (il < (int) heat.size()) {
            for (const auto & p : heat[il]) {
                L.cum[p.second]   = (uint64_t) p.first;
                L.score[p.second] = (float) p.first; // warm start from seed
            }
        }
        for (int s = 0; s < (int) top.size(); s++) {
            L.slot_expert[s] = top[s];
        }
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            auto it = g_stores.find(w);
            if (it == g_stores.end()) {
                continue;
            }
            memset(it->second.counts->data, 0, (n_expert + 1)*sizeof(int32_t));
            L.ws.push_back({w, &it->second});
            if (it->second.is_down) {
                L.sd = &it->second;
            }
        }
        if (!L.ws.empty()) {
            g_layers[il] = std::move(L);
        }
    }

    g_init_hot_fill_us = (uint64_t) (ggml_time_us() - hot_fill_start_us);
    g_hot_alloc = total_bytes;

    // Shared per-layer slot layout. The demand pool, pread ring and
    // speculative pool may be enabled independently but must agree on where
    // gate/up/down live inside one expert slot.
    for (auto & L : g_layers) {
        if (!L.sd || !L.sd->poolable) {
            continue;
        }
        size_t slot = 0;
        for (auto & kv : L.ws) {
            kv.second->pool_off = slot;
            slot += pool_align(ggml_nbytes(kv.first)/L.n_expert);
        }
        L.pool_slot_bytes = slot;
    }

    // RAM pool: one block per layer; slot = one expert across all its tensors
    if (g_pool_bytes > 0) {
        const int64_t demand_pool_fill_start_us = ggml_time_us();
        int n_pool_layers = 0;
        for (auto & L : g_layers) {
            if (L.sd && L.sd->poolable) {
                n_pool_layers++;
            }
        }
        for (auto & L : g_layers) {
            if (!L.sd || !L.sd->poolable || n_pool_layers == 0) {
                continue;
            }
            const size_t per_layer = g_pool_bytes/(size_t) n_pool_layers;
            L.n_pool_slots = L.pool_slot_bytes > 0 ? (int) (per_layer/L.pool_slot_bytes) : 0;
            if (L.n_pool_slots <= 0) {
                continue;
            }
            const size_t bytes = (size_t) L.n_pool_slots*L.pool_slot_bytes;
            const int64_t alloc_start = g_perf_trace.load(std::memory_order_relaxed) ? ggml_time_us() : 0;
#if defined(_WIN32)
            L.pool = (char *) _aligned_malloc(bytes, POOL_ALIGN);
#else
            void * p = nullptr;
            if (posix_memalign(&p, POOL_ALIGN, bytes) != 0) {
                p = nullptr;
            }
            L.pool = (char *) p;
#endif
            if (alloc_start) {
                LLAMA_LOG_INFO("[PERF_TRACE][expert_host_buffer_alloc] kind=demand_pool layer=%d bytes=%zu total=%.3f ms\n",
                        L.il, bytes, (ggml_time_us() - alloc_start)/1000.0);
            }
            if (!L.pool) {
                TIER_LOG("%s: pool alloc failed for layer %d (%zu MiB)\n", __func__, L.il, bytes >> 20);
                L.n_pool_slots = 0;
                continue;
            }
            g_pool_alloc += bytes;
            L.pool_slot_expert.assign(L.n_pool_slots, -1);
            L.pool_dwell.assign(L.n_pool_slots, 0);
            L.pool_slot_state.reset(new std::atomic<int32_t>[L.n_pool_slots]);
            for (int k = 0; k < L.n_pool_slots; k++) {
                L.pool_slot_state[k].store(0, std::memory_order_relaxed);
            }
            L.pool_lut.reset(new std::atomic<int32_t>[L.n_expert]);
            for (int e = 0; e < L.n_expert; e++) {
                L.pool_lut[e].store(-1, std::memory_order_relaxed);
            }
            if (!heat.empty()) {
                // seed: hottest cold experts by score
                std::vector<int32_t> cold;
                for (int e = 0; e < L.n_expert; e++) {
                    if (L.lut_host[e] == L.sentinel) {
                        cold.push_back(e);
                    }
                }
                std::sort(cold.begin(), cold.end(), [&](int32_t a, int32_t b) {
                    return L.score[a] != L.score[b] ? L.score[a] > L.score[b] : a < b;
                });
                for (int k = 0; k < L.n_pool_slots && k < (int) cold.size(); k++) {
                    pool_fill(L, k, cold[k]);
                }
            }
        }
        TIER_LOG("%s: expert pool: %.2f GiB over %d layers\n",
                __func__, (double) g_pool_alloc/(double) (1 << 30), n_pool_layers);
        g_init_demand_pool_fill_us = (uint64_t) (ggml_time_us() - demand_pool_fill_start_us);
    }

    // pread staging ring (LLAMA_EXPERT_PREAD): bounded, page-aligned, owned
    // by the cold path; sized to hold one token's routed experts of the
    // widest layer. requires the ptrs path (LLAMA_EXPERT_RAMPOOL > 0).
    if (want_pread && !g_pread_disabled) {
        if (g_pool_bytes == 0) {
            TIER_LOG("%s: pread ring disabled (needs LLAMA_EXPERT_RAMPOOL > 0)\n", __func__);
        } else {
            g_stage_fail_test = [] {
                const char * e = getenv("LLAMA_EXPERT_PREAD_FAIL");
                return e && atoi(e) != 0;
            }();
            size_t max_slot = 0;
            for (auto & L : g_layers) {
                if (L.sd && L.sd->poolable && L.pool_slot_bytes > 0) {
                    max_slot = std::max(max_slot, L.pool_slot_bytes);
                }
            }
            const size_t page = 4096;
            g_stage_stride = max_slot > 0 ? (max_slot + page - 1) & ~(page - 1) : 0;
            size_t ring_bytes = (size_t) model.hparams.n_expert_used*g_stage_stride;
            if (const char * e = getenv("LLAMA_EXPERT_PREAD_RING_MB")) {
                ring_bytes = (size_t) (atof(e)*1048576.0);
            }
            g_stage_n = g_stage_stride > 0 ? (int) std::max((size_t) 1, ring_bytes/g_stage_stride) : 0;
            // single-file guard: every poolable tensor must lie in the first
            // file mapping; sharded models disable the ring
            const char * mbase = (const char *) model.mmap_base();
            const size_t msize = model.mmap_size();
            if (g_stage_n > 0 && (!mbase || msize == 0)) {
                g_stage_n = 0;
            }
            for (const auto & L : g_layers) {
                if (g_stage_n == 0) {
                    break;
                }
                if (!L.sd || !L.sd->poolable || L.pool_slot_bytes == 0) {
                    continue;
                }
                for (const auto & kv : L.ws) {
                    const char * d = (const char *) kv.first->data;
                    if (d < mbase || (size_t) (d - mbase) + ggml_nbytes(kv.first) > msize) {
                        g_stage_n = 0;
                        TIER_LOG("%s: pread ring disabled (tensor outside first file mapping - sharded?)\n", __func__);
                        break;
                    }
                }
            }
            if (g_stage_n > 0) {
                const size_t bytes = (size_t) g_stage_n*g_stage_stride;
                const int64_t alloc_start = g_perf_trace.load(std::memory_order_relaxed) ? ggml_time_us() : 0;
#if defined(_WIN32)
                g_stage_base = (char *) _aligned_malloc(bytes, page);
#else
                void * p = nullptr;
                if (posix_memalign(&p, page, bytes) != 0) {
                    p = nullptr;
                }
                g_stage_base = (char *) p;
#endif
                if (alloc_start) {
                    LLAMA_LOG_INFO("[PERF_TRACE][expert_host_buffer_alloc] kind=pread_ring layer=-1 bytes=%zu total=%.3f ms\n",
                            bytes, (ggml_time_us() - alloc_start)/1000.0);
                }
                if (!g_stage_base) {
                    TIER_LOG("%s: pread ring alloc failed (%d x %.2f MiB)\n", __func__,
                            g_stage_n, (double) g_stage_stride/(1024.0*1024.0));
                    g_stage_n = 0;
                } else {
                    g_stage_alloc = bytes;
                    g_stage_word.reset(new std::atomic<uint64_t>[g_stage_n]);
                    for (int k = 0; k < g_stage_n; k++) {
                        g_stage_word[k].store(0, std::memory_order_relaxed);
                    }
                    g_stage_mmap = mbase;
                    for (auto & L : g_layers) {
                        if (!L.sd || !L.sd->poolable || L.pool_slot_bytes == 0) {
                            continue;
                        }
                        for (auto & kv : L.ws) {
                            store & st = *kv.second;
                            if (!st.ptrs || !st.ptrs->data) {
                                continue;
                            }
                            g_stage_ix[st.ptrs->data] = stage_ctx{ L.il, (int64_t) st.pool_off,
                                    (const char *) kv.first->data, (int64_t) (ggml_nbytes(kv.first)/L.n_expert) };
                        }
                    }
                    tier_resolve_moe_hooks();
                    MOE_FETCH_HOOK(stage_fetch);
                    TIER_LOG("%s: pread ring: %d slots x %.2f MiB%s\n", __func__,
                            g_stage_n, (double) g_stage_stride/(1024.0*1024.0),
                            g_stage_fail_test ? " (FAIL-TEST)" : "");
                }
            }
        }
    }

    // Spec pool: an explicit slot count takes precedence over the legacy GiB
    // budget. Each layer owns its slots and state independently.
    const bool want_spec_pool = g_predict && !g_prefetch_calibrate &&
            (g_prefetch_slots > 0 || g_poolB_bytes > 0);
    if (want_spec_pool) {
        int n_pool_layers = 0;
        for (auto & L : g_layers) {
            if (L.sd && L.sd->poolable) {
                n_pool_layers++;
            }
        }
        for (auto & L : g_layers) {
            if (!L.sd || !L.sd->poolable || n_pool_layers == 0) {
                continue;
            }
            if (g_prefetch_slots > 0 && L.pool_slot_bytes == 0) {
                throw std::runtime_error("--expert-prefetch-slots cannot derive an expert slot layout for this layer");
            }
            const size_t per_layer = g_poolB_bytes/(size_t) n_pool_layers;
            L.n_slotsB = g_prefetch_slots > 0 ? g_prefetch_slots :
                    (L.pool_slot_bytes > 0 ? (int) (per_layer/L.pool_slot_bytes) : 0);
            if (L.n_slotsB <= 0) {
                if (g_prefetch_slots > 0) {
                    throw std::runtime_error("explicit expert prefetch slot allocation produced zero slots");
                }
                continue;
            }
            if (L.pool_slot_bytes > 0 && (size_t) L.n_slotsB > SIZE_MAX/L.pool_slot_bytes) {
                throw std::runtime_error("expert prefetch slot allocation size overflow");
            }
            const size_t bytes = (size_t) L.n_slotsB*L.pool_slot_bytes;
            const int64_t alloc_start = g_perf_trace.load(std::memory_order_relaxed) ? ggml_time_us() : 0;
#if defined(_WIN32)
            L.poolB = (char *) _aligned_malloc(bytes, POOL_ALIGN);
#else
            void * p = nullptr;
            if (posix_memalign(&p, POOL_ALIGN, bytes) != 0) {
                p = nullptr;
            }
            L.poolB = (char *) p;
#endif
            if (alloc_start) {
                LLAMA_LOG_INFO("[PERF_TRACE][expert_host_buffer_alloc] kind=prefetch_pool layer=%d bytes=%zu total=%.3f ms\n",
                        L.il, bytes, (ggml_time_us() - alloc_start)/1000.0);
            }
            if (!L.poolB) {
                TIER_LOG("%s: spec pool alloc failed for layer %d (%zu MiB)\n", __func__, L.il, bytes >> 20);
                L.n_slotsB = 0;
                if (g_prefetch_slots > 0) {
                    std::ostringstream err;
                    err << "failed to allocate " << g_prefetch_slots << " expert prefetch slots for layer "
                        << L.il << " (" << L.pool_slot_bytes << " bytes per slot, " << bytes << " bytes total)";
                    throw std::runtime_error(err.str());
                }
                continue;
            }
            g_poolB_alloc += bytes;
            L.ownersB.assign(L.n_slotsB, -1);
            L.stateB.reset(new std::atomic<int32_t>[L.n_slotsB]);
            for (int k = 0; k < L.n_slotsB; k++) {
                L.stateB[k].store(0, std::memory_order_relaxed);
            }
            L.lutB.reset(new std::atomic<int32_t>[L.n_expert]);
            for (int e = 0; e < L.n_expert; e++) {
                L.lutB[e].store(-1, std::memory_order_relaxed);
            }
            L.lastB.reset(new std::atomic<int64_t>[L.n_slotsB]);
            L.fillB.reset(new std::atomic<uint64_t>[L.n_slotsB]);
            L.checkedB.reset(new std::atomic<uint64_t>[L.n_slotsB]);
            for (int k = 0; k < L.n_slotsB; k++) {
                L.lastB[k].store(0, std::memory_order_relaxed);
                L.fillB[k].store(0, std::memory_order_relaxed);
                L.checkedB[k].store(0, std::memory_order_relaxed);
            }
        }
        TIER_LOG("%s: spec pool: %.2f GiB over %d layers (%d slots/layer, %d threads)\n",
                __func__, (double) g_poolB_alloc/(double) (1 << 30), n_pool_layers,
                g_prefetch_slots > 0 ? g_prefetch_slots : -1, g_n_workers);
    }

                TIER_LOG("%s: expert tiering on: %d slots/layer, %zu tensors, %.2f GiB pinned, seed coverage %.1f%%\n",
            __func__, g_S, g_stores.size(), (double) total_bytes/(1 << 30),
            total > 0 ? 100.0*hits/total : 0.0);

    pred_init(model);
    g_init_total_us = (uint64_t) (ggml_time_us() - init_start_us);
    TIER_LOG("%s: memory plan: model_mmap=%.2f GiB hot_backend=%.2f GiB demand_pool=%.2f GiB prefetch_pool=%.2f GiB pread_ring=%.2f MiB predictor_mirror=%.2f MiB\n",
            __func__, (double) model.mmap_size()/(double) (1ULL << 30),
            (double) g_hot_alloc/(double) (1ULL << 30),
            (double) g_pool_alloc/(double) (1ULL << 30),
            (double) g_poolB_alloc/(double) (1ULL << 30),
            (double) g_stage_alloc/(1024.0*1024.0),
            (double) g_predictor_mirror_bytes/(1024.0*1024.0));
    TIER_LOG("%s: initialization: total=%.3f s hot_fill=%.3f s demand_pool_fill=%.3f s\n",
            __func__, g_init_total_us/1000000.0, g_init_hot_fill_us/1000000.0,
            g_init_demand_pool_fill_us/1000000.0);
    if (g_prefetch_min_cold_rate > 0.0f) {
        TIER_LOG("%s: predictor layer filter: historical cold-rate >= %.3f\n",
                __func__, g_prefetch_min_cold_rate);
    }
}

ggml_tensor * build_mul_mat_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids) {
    auto it = g_stores.find(w);
    if (it == g_stores.end() || ids->ne[1] > (int64_t) g_tmax) {
        return ggml_mul_mat_id(ctx, w, x, ids);
    }

    const store & s = it->second;

    // get_rows wants matching trailing dims, so flatten ids to 1d first
    ggml_tensor * ids_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, ids), ids->ne[0]*ids->ne[1]);
    ggml_tensor * hot_flat = ggml_get_rows(ctx, s.lut, ids_flat); // [1, n_used*n_tokens]
    ggml_tensor * ids_hot  = ggml_reshape_2d(ctx, hot_flat, ids->ne[0], ids->ne[1]);
    ggml_tensor * hot      = ggml_mul_mat_id(ctx, s.w_hot, x, ids_hot);    // GPU

    if (g_hot_only) {
        return hot;
    }

    ggml_tensor * cold     = ggml_mul_mat_id_cold(ctx, w, x, ids, s.mask, s.ptrs); // CPU

    return ggml_add(ctx, hot, cold);
}

bool begin_moe_cold(bool eligible,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * ids) {
    g_hot_only = false;
    if (!eligible || g_stores.empty()) {
        return false;
    }
    auto ig = g_stores.find(gate_w);
    auto iu = g_stores.find(up_w);
    auto id = g_stores.find(down_w);
    if (ig == g_stores.end() || iu == g_stores.end() || id == g_stores.end() ||
        ids->ne[1] > (int64_t) g_tmax) {
        return false;
    }
    g_hot_only = true;
    return true;
}

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids) {
    if (!g_hot_only) {
        return nullptr;
    }
    store & sd = g_stores[down_w];
    return ggml_moe_cold(ctx, gate_w, up_w, down_w, x, ids, sd.mask, sd.counts,
            g_stores[gate_w].ptrs, g_stores[up_w].ptrs, sd.ptrs);
}

ggml_tensor * build_moe_count(ggml_context * ctx, ggml_tensor * down_w, ggml_tensor * ids) {
    // prompt-sized batches skip the fused path; harvest router decisions anyway
    if (g_hot_only || g_stores.empty()) {
        return nullptr;
    }
    auto it = g_stores.find(down_w);
    if (it == g_stores.end() || ids->ne[1] <= (int64_t) g_tmax) {
        return nullptr;
    }
    return ggml_moe_count(ctx, ids, it->second.counts);
}

}
