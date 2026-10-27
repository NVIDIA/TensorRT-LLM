// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT


#pragma once
#include <cstdint>

namespace gpu_trace {

/* ===== Compile-time FNV-1a hash (always available, CPU + GPU) ============= */

constexpr uint32_t hash(const char* s) {
    uint32_t h = 2166136261u;
    for (; *s; ++s) h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

static constexpr int RECORD_INTS = 4;
static constexpr int MAX_WARP_PER_BLOCK = 32;
static constexpr int MAX_SM_NUN = 256;
static constexpr int DATA_OFFSET = MAX_WARP_PER_BLOCK + MAX_SM_NUN * 4;


}  // namespace gpu_trace

/* ===== Scope registration (host-side, compile-time auto-register) ========= */

#ifdef GPU_TRACE_ENABLED

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu_trace {
namespace detail {

// One 8-bit scope id may map to multiple compile-time scope names (hash collision).
inline std::unordered_map<uint32_t, std::vector<std::string>>& scope_registry() {
    static std::unordered_map<uint32_t, std::vector<std::string>> reg;
    return reg;
}

inline bool do_register(uint32_t id, const char* name) {
    std::string s(name);
    auto& v = scope_registry()[id];
    if (std::find(v.begin(), v.end(), s) == v.end())
        v.push_back(std::move(s));
    return true;
}

/** Resolved label for Chrome trace: single name, or "A/B/C" when id collides. */
inline std::string scope_label_for_id(uint32_t id) {
    auto& reg = scope_registry();
    auto it = reg.find(id);
    if (it == reg.end() || it->second.empty())
        return "unknown";
    const auto& names = it->second;
    if (names.size() == 1)
        return names[0];
    std::vector<std::string> sorted(names.begin(), names.end());
    std::sort(sorted.begin(), sorted.end());
    std::string out = sorted[0];
    for (size_t i = 1; i < sorted.size(); ++i) {
        out.push_back('/');
        out += sorted[i];
    }
    return out;
}

}  // namespace detail
}  // namespace gpu_trace

#define GPU_TRACE_REGISTER_IMPL_(name, name_str)       \
    [[maybe_unused]] inline const bool _gtreg_##name = \
        ::gpu_trace::detail::do_register(::gpu_trace::hash(name_str), name_str)

#else
#define GPU_TRACE_REGISTER_IMPL_(name, name_str)
#endif

/* ===== GPU_TRACE_SCOPE_DEC ================================================ */

#ifdef GPU_TRACE_ENABLED

#define GPU_TRACE_SCOPE_DEC(name)                                      \
    static constexpr uint32_t GTEVT_##name = ::gpu_trace::hash(#name); \
    GPU_TRACE_REGISTER_IMPL_(name, #name)

GPU_TRACE_SCOPE_DEC(GPU_TRACE_TEST_SCOPE);
GPU_TRACE_SCOPE_DEC(GPU_TRACE_TEST);

#else

#define GPU_TRACE_SCOPE_DEC(name) static constexpr uint32_t GTEVT_##name = 0

#endif

/* ========================================================================== */
#ifndef GPU_TRACE_ENABLED
/* ===== Disabled: zero-overhead stubs ====================================== */

struct GPUTraceParam {};

namespace gpu_trace {
struct Recorder {};
template <typename S>
inline bool setup_from_env(GPUTraceParam&, S, const char* = nullptr, int = 0) {
    return false;
}
template <typename S>
inline void teardown(GPUTraceParam&, S, const char* = nullptr) {}
}  // namespace gpu_trace

#define GPU_TRACE_INIT
#define GET_GPU_TRACE(cond)
#define RELEASE_GPU_TRACE
#define GPU_TRACE_SCOPE(name)
#define GPU_TRACE_SCOPE_BEGIN(name)
#define GPU_TRACE_SCOPE_END(name)
#define GPU_TRACE_EXIT

#else /* GPU_TRACE_ENABLED */
/* ===== Enabled ============================================================ */

struct GPUTraceParam {
    uint32_t* buf = nullptr;
    int capacity = 0;
    int target_block = -1;
};

__constant__ GPUTraceParam g_gputraceParam;

/* ---- Device-side Recorder & Scope ---------------------------------------- */
#ifdef __CUDACC__

namespace gpu_trace {

__device__ __forceinline__ unsigned lane_id() {
    unsigned id;
    asm("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}

__device__ __forceinline__ unsigned warp_id() {
    return (threadIdx.x + threadIdx.y * blockDim.x +
            threadIdx.z * blockDim.x * blockDim.y) >>
           5;
}

struct Recorder {
    uint32_t* cur;
    uint32_t remaining;

    __device__ __forceinline__ bool init() {
        cur = nullptr;
        remaining = 0;
        const auto& p = g_gputraceParam;
        bool active = p.buf &&
                      (p.target_block < 0 || blockIdx.x == (unsigned)p.target_block) &&
                      (lane_id() == 0);
        if (!active)
            return false;
        remaining = p.capacity;
        cur = p.buf + DATA_OFFSET + warp_id() * p.capacity * RECORD_INTS;
        return true;
    }

    __device__ __forceinline__ void write_v4(uint32_t d0,
                                             uint32_t d1,
                                             uint32_t d2,
                                             uint32_t d3) {
        uint32_t active = (remaining != 0);
        asm volatile(
            "{\n\t"
            "  .reg .pred p;\n\t"
            "  setp.ne.u32 p, %0, 0;\n\t"
            "  @p st.global.cs.v4.b32 [%1], {%2, %3, %4, %5};\n\t"
            "}" ::"r"(active),
            "l"(cur), "r"(d0), "r"(d1), "r"(d2), "r"(d3)
            : "memory");
        cur += active * RECORD_INTS;
        remaining -= active;
    }

    __device__ __forceinline__ void load();
    __device__ __forceinline__ void save(bool force = false);
};

__device__ __forceinline__ uint32_t recorder_remaining_load() {
    uint32_t val;
    asm volatile("ld.global.ca.b32 %0, [%1];"
                 : "=r"(val)
                 : "l"(g_gputraceParam.buf + warp_id())
                 : "memory");
    return val;
}

__device__ __forceinline__ void recorder_remaining_store(uint32_t val) {
    uint32_t* ptr = g_gputraceParam.buf;
    if (ptr == nullptr)
        return;
    asm volatile("st.global.wb.b32 [%0], %1;" ::"l"(ptr + warp_id()), "r"(val)
                 : "memory");
}

__device__ __forceinline__ void Recorder::load() {
    if (lane_id() != 0)
        return;
    if (!init())
        return;
    uint32_t ar = recorder_remaining_load();
    cur += (remaining - ar) * RECORD_INTS;
    remaining = ar;
}

__device__ __forceinline__ void Recorder::save(bool force) {
    if (!force && !this->cur)
        return;
    recorder_remaining_store(remaining);
}

/*
 * v4 record layout (4 × uint32_t per event, zero GPU-side computation):
 *   Word 0 : begin_clk  (raw 32-bit clock from CS2R)
 *   Word 1 : end_clk    (raw 32-bit clock from CS2R)
 *   Word 2 : scope_id   (compile-time constant, 8-bit hash)
 *   Word 3 : reserved   (0)
 *
 * All packing / duration / timestamp computation is deferred to the CPU
 * in teardown().  GPU critical path: CS2R → SHF → STG.v4 (2 levels).
 */
template <uint32_t ScopeId>
struct Scope {
    static constexpr int scope_id = ScopeId;
    uint32_t begin_clk_;
    const uint32_t data_;

    __device__ __forceinline__ Scope(uint32_t data = 0) : begin_clk_(0), data_(data) {}

    __device__ __forceinline__ static uint32_t read_clock() {
        uint32_t clk;
        asm volatile("{ mov.u32 %0, %%clock; }" : "=r"(clk)::"memory");
        return clk;
    }

    __device__ __forceinline__ void begin() {
        begin_clk_ = read_clock();
    }

    __device__ __forceinline__ void end(Recorder& rec) {
        uint32_t end_clk = read_clock();
        rec.write_v4(begin_clk_, end_clk, (uint32_t)ScopeId, data_);
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

template <uint32_t ScopeId>
struct ScopeGuard : Scope<ScopeId> {
    Recorder& rec_;

    __device__ __forceinline__ ScopeGuard(Recorder& rec) : rec_(rec) {
        this->begin();
    }

    __device__ __forceinline__ ~ScopeGuard() {
        this->end(rec_);
    }
};

}  // namespace gpu_trace

#define GPU_TRACE_INIT                \
    {                                 \
        gpu_trace::Recorder _gt_r; \
        _gt_r.init();                 \
        _gt_r.save();                 \
        __syncthreads();              \
        if (threadIdx.x == 0)         \
        {                             \
            uint64_t c = clock64();   \
            g_gputraceParam.buf[gpu_trace::MAX_WARP_PER_BLOCK + blockIdx.x * 4] = uint32_t(c & 0xFFFFFFFF); \
            g_gputraceParam.buf[gpu_trace::MAX_WARP_PER_BLOCK + blockIdx.x * 4 + 1] = uint32_t(c >> 32); \
        }                             \
    }

#define GPU_TRACE_EXIT                \
        if (threadIdx.x == 0)         \
        {                             \
            uint64_t c = clock64();   \
            g_gputraceParam.buf[gpu_trace::MAX_WARP_PER_BLOCK + blockIdx.x * 4 + 2] = uint32_t(c & 0xFFFFFFFF); \
            g_gputraceParam.buf[gpu_trace::MAX_WARP_PER_BLOCK + blockIdx.x * 4 + 3] = uint32_t(c >> 32); \
        }                             \

#define GET_GPU_TRACE(cond)  \
    gpu_trace::Recorder _gt_rec{}; \
    if (cond)                                   \
    _gt_rec.load()

#define RELEASE_GPU_TRACE _gt_rec.save()

#define GPU_TRACE_SCOPE(name)                                                        \
    gpu_trace::ScopeGuard<GTEVT_##name> _gtscp_##name( \
        _gt_rec)

#define GPU_TRACE_SCOPE_BEGIN(name)               \
    gpu_trace::Scope<GTEVT_##name> _gtevt_##name; \
    _gtevt_##name.begin()

#define GPU_TRACE_SCOPE_END(name) _gtevt_##name.end(_gt_rec)

#endif /* __CUDACC__ */

/* ---- Host-side helpers --------------------------------------------------- */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>

namespace gpu_trace {

inline int64_t buf_total_ints(int per_thread_capacity) {
    return DATA_OFFSET + MAX_WARP_PER_BLOCK * per_thread_capacity * RECORD_INTS;
}

inline bool setup_from_env(GPUTraceParam& p,
                           cudaStream_t stream,
                           const char* env_var = "GPU_TRACE",
                           int per_thread_capacity = 65536) {
    const char* val = std::getenv(env_var);
    if (!val)
        return false;

    p.target_block = std::atoi(val);
    p.capacity = per_thread_capacity;

    int64_t total = buf_total_ints(p.capacity);
    int64_t bytes = total * (int64_t)sizeof(uint32_t);
    cudaMalloc(&p.buf, bytes);
    cudaMemsetAsync(p.buf, 0, bytes, stream);

    cudaMemcpyToSymbolAsync(g_gputraceParam, &p, sizeof(GPUTraceParam), 0,
                            cudaMemcpyHostToDevice, stream);
    return true;
}

/**
 * Synchronize, read back trace data, write Chrome trace JSON.
 *
 * v4 record layout (4 × uint32_t per event):
 *   Word 0 : begin_clk  (raw 32-bit clock)
 *   Word 1 : end_clk    (raw 32-bit clock)
 *   Word 2 : scope_id   (8-bit hash, stored as uint32_t)
 *   Word 3 : reserved   (0)
 *
 * Duration and timestamps are computed on the CPU from raw clocks.
 */
inline void teardown(GPUTraceParam& p,
                     cudaStream_t stream,
                     const char* default_path = "/tmp/gpu_trace.json") {
    if (!p.buf)
        return;

    cudaStreamSynchronize(stream);

    int device = 0;
    cudaGetDevice(&device);
    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);
    double scale = (clock_rate_khz > 0) ? (1e3 / (double)clock_rate_khz) : 1e-3;

    int64_t total = buf_total_ints(p.capacity);
    std::vector<uint32_t> h_buf(total);
    cudaMemcpy(h_buf.data(), p.buf, total * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(p.buf);
    p.buf = nullptr;

    GPUTraceParam empty{};
    cudaMemcpyToSymbol(g_gputraceParam, &empty, sizeof(GPUTraceParam));

    // --- Extract per-SM timing (clock64 stored as 4 × uint32_t per SM) ---
    struct SmTiming {
        int id;
        uint64_t start, end, dur;
    };
    std::vector<SmTiming> sm_timings;
    for (int i = 0; i < MAX_SM_NUN; ++i) {
        int base = MAX_WARP_PER_BLOCK + i * 4;
        uint64_t s = (uint64_t)h_buf[base + 1] << 32 | h_buf[base + 0];
        uint64_t e = (uint64_t)h_buf[base + 3] << 32 | h_buf[base + 2];
        if (s == 0 && e == 0)
            continue;
        sm_timings.push_back({i, 0, e - s, e - s});
    }


    std::string out(default_path);
    const char* env = std::getenv("GPU_TRACE");
    if (env) {
        const char* colon = std::strchr(env, ':');
        if (colon && colon[1])
            out = colon + 1;
    }

    struct Rec {
        int tid;
        std::string name;
        int64_t tb, te;
    };
    std::vector<Rec> records;
    int64_t t_origin = INT64_MAX;

    for (int t = 0; t < MAX_WARP_PER_BLOCK; ++t) {
        int remain = (int)h_buf[t];
        int used = p.capacity - remain;
        if (used <= 0)
            continue;
        int64_t warp_base = DATA_OFFSET + (int64_t)t * p.capacity * RECORD_INTS;
        for (int i = 0; i < used; ++i) {
            int64_t off = warp_base + (int64_t)i * RECORD_INTS;
            uint32_t begin_clk = h_buf[off + 0];
            uint32_t end_clk = h_buf[off + 1];
            uint32_t eid = h_buf[off + 2];
            if (begin_clk == 0 && end_clk == 0)
                break;

            int64_t dur = (int64_t)(uint32_t)(end_clk - begin_clk);
            int64_t tb = (int64_t)begin_clk;
            int64_t te = tb + dur;

            records.push_back({t, detail::scope_label_for_id(eid), tb, te});
            if (tb < t_origin)
                t_origin = tb;
        }
    }

    if (records.empty() && sm_timings.empty())
        return;

    std::map<int, std::vector<size_t>> by_tid;
    for (size_t i = 0; i < records.size(); ++i) by_tid[records[i].tid].push_back(i);

    for (auto& [tid, idx] : by_tid) {
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return records[a].tb < records[b].tb; });
    }

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f)
        return;

    std::fprintf(f, "{\"traceEvents\":[\n");
    bool first = true;
    auto comma = [&]() {
        if (!first)
            std::fprintf(f, ",\n");
        first = false;
    };

    int vpid = 0;
    for (auto& [tid, idx] : by_tid) {
        int pid = vpid++;

        comma();
        std::fprintf(f,
                     "{\"name\":\"process_name\",\"ph\":\"M\","
                     "\"pid\":%d,\"tid\":0,"
                     "\"args\":{\"name\":\"Block %d / Warp %d\"}}",
                     pid, p.target_block, tid);
        comma();
        std::fprintf(f,
                     "{\"name\":\"process_sort_index\",\"ph\":\"M\","
                     "\"pid\":%d,\"tid\":0,"
                     "\"args\":{\"sort_index\":%d}}",
                     pid, pid);

        std::set<std::string> name_set;
        for (size_t ri : idx) name_set.insert(records[ri].name);
        std::vector<std::string> lane_names(name_set.begin(), name_set.end());
        std::map<std::string, int> name_to_lane;
        for (int lane = 0; lane < (int)lane_names.size(); ++lane)
            name_to_lane[lane_names[lane]] = lane;

        for (int lane = 0; lane < (int)lane_names.size(); ++lane) {
            comma();
            std::fprintf(f,
                         "{\"name\":\"thread_name\",\"ph\":\"M\","
                         "\"pid\":%d,\"tid\":%d,"
                         "\"args\":{\"name\":\"%s\"}}",
                         pid, lane, lane_names[lane].c_str());
            comma();
            std::fprintf(f,
                         "{\"name\":\"sort_index\",\"ph\":\"M\","
                         "\"pid\":%d,\"tid\":%d,"
                         "\"args\":{\"sort_index\":%d}}",
                         pid, lane, lane);
        }

        for (size_t ri : idx) {
            auto& rec = records[ri];
            int lane = name_to_lane[rec.name];
            int64_t dur = rec.te - rec.tb;
            comma();
            std::fprintf(f,
                         "{\"name\":\"%s\",\"ph\":\"X\","
                         "\"ts\":%.3f,\"dur\":%.3f,"
                         "\"pid\":%d,\"tid\":%d,"
                         "\"args\":{\"cycles\":%lld}}",
                         rec.name.c_str(), (double)(rec.tb - t_origin) * scale,
                         (double)dur * scale, pid, lane, (long long)dur);
        }
    }

    // --- SM timing events in Chrome trace (separate process group) ---
    if (!sm_timings.empty()) {
        uint64_t sm_origin = sm_timings[0].start;
        uint64_t sum = 0;
        uint64_t min_dur = UINT64_MAX, max_dur = 0;
        int min_sm = 0, max_sm = 0;
        for (auto& t : sm_timings) {
            sum += t.dur;
            if (t.start < sm_origin) sm_origin = t.start;
            if (t.dur < min_dur) { min_dur = t.dur; min_sm = t.id; }
            if (t.dur > max_dur) { max_dur = t.dur; max_sm = t.id; }
        }
        double mean = (double)sum / sm_timings.size();
        double var = 0;
        for (auto& t : sm_timings) {
            double d = (double)t.dur - mean;
            var += d * d;
        }
        double stdev = std::sqrt(var / sm_timings.size());
        double imbalance = min_dur > 0 ? (double)max_dur / min_dur : 0.0;

        int sm_pid = vpid++;
        comma();
        std::fprintf(f,
                     "{\"name\":\"process_name\",\"ph\":\"M\","
                     "\"pid\":%d,\"tid\":0,"
                     "\"args\":{\"name\":\"SM Timing (%zu SMs)\"}}",
                     sm_pid, sm_timings.size());
        comma();
        std::fprintf(f,
                     "{\"name\":\"process_sort_index\",\"ph\":\"M\","
                     "\"pid\":%d,\"tid\":0,"
                     "\"args\":{\"sort_index\":%d}}",
                     sm_pid, sm_pid);

        // e2e thread at tid=MAX_SM_NUN, sort to top
        int e2e_tid = MAX_SM_NUN;
        comma();
        std::fprintf(f,
                     "{\"name\":\"thread_name\",\"ph\":\"M\","
                     "\"pid\":%d,\"tid\":%d,"
                     "\"args\":{\"name\":\"E2E\"}}",
                     sm_pid, e2e_tid);
        comma();
        std::fprintf(f,
                     "{\"name\":\"thread_sort_index\",\"ph\":\"M\","
                     "\"pid\":%d,\"tid\":%d,"
                     "\"args\":{\"sort_index\":-1}}",
                     sm_pid, e2e_tid);

        uint64_t max_end = 0;
        for (auto& t : sm_timings)
            if (t.end > max_end) max_end = t.end;
        uint64_t e2e_dur = max_end - sm_origin;

        comma();
        std::fprintf(f,
                     "{\"name\":\"E2E\",\"ph\":\"X\","
                     "\"ts\":0,\"dur\":%.3f,"
                     "\"pid\":%d,\"tid\":%d,"
                     "\"args\":{\"cycles\":%llu,\"us\":%.3f,"
                     "\"min_cycles\":%llu,\"min_us\":%.3f,\"min_sm\":%d,"
                     "\"max_cycles\":%llu,\"max_us\":%.3f,\"max_sm\":%d,"
                     "\"mean_cycles\":%.0f,\"mean_us\":%.3f,"
                     "\"stdev_cycles\":%.0f,\"stdev_us\":%.3f,"
                     "\"imbalance\":%.4f}}",
                     (double)e2e_dur * scale,
                     sm_pid, e2e_tid,
                     (unsigned long long)e2e_dur, (double)e2e_dur * scale,
                     (unsigned long long)min_dur, (double)min_dur * scale, min_sm,
                     (unsigned long long)max_dur, (double)max_dur * scale, max_sm,
                     mean, mean * scale,
                     stdev, stdev * scale,
                     imbalance);

        for (auto& t : sm_timings) {
            comma();
            std::fprintf(f,
                         "{\"name\":\"SM %d\",\"ph\":\"X\","
                         "\"ts\":%.3f,\"dur\":%.3f,"
                         "\"pid\":%d,\"tid\":%d,"
                         "\"args\":{\"cycles\":%llu,\"start\":%llu,\"end\":%llu}}",
                         t.id,
                         (double)(t.start - sm_origin) * scale,
                         (double)t.dur * scale,
                         sm_pid, t.id,
                         (unsigned long long)t.dur,
                         (unsigned long long)t.start,
                         (unsigned long long)t.end);
        }
    }

    std::fprintf(f, "\n]}");
    std::fclose(f);
    std::fprintf(stderr, "[GPUTrace] %zu records -> %s  (SM clock %d kHz)\n",
                 records.size(), out.c_str(), clock_rate_khz);
}

}  // namespace gpu_trace

#endif /* GPU_TRACE_ENABLED */
