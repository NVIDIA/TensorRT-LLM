/* CUDA TMA multicast-copy backend. C ABI; no CUDA or Python headers required. */
#ifndef MEGAMOE_TMA_COPY_H
#define MEGAMOE_TMA_COPY_H

#include <stdint.h>
#include "gpu_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MEGAMOE_TMA_COPY_ABI_VERSION 5
#define MEGAMOE_TMA_COPY_SLICE_BYTES 8192
#define MEGAMOE_TMA_COPY_SLOT_CONTROL_BYTES 32
#define MEGAMOE_TMA_COPY_SLOT_DATA_BYTES 8208
#define MEGAMOE_TMA_COPY_MAX_SMS 8
#define MEGAMOE_TMA_COPY_MAX_WARPS 32
/* Experimental candidate: warps counts logical producer/consumer worker pairs. */
#define MEGAMOE_TMA_COPY_THREADS_PER_WORKER 64

/* Zero requests the default; zero return denotes invalid geometry. The entire
 * payload/notification kernel stays within this per-GPU copy SM budget. */
static inline int megamoe_tma_copy_sm_count(int requested_sms, int device_sms) {
    if (device_sms <= 0 || requested_sms < 0 ||
        requested_sms > MEGAMOE_TMA_COPY_MAX_SMS || requested_sms > device_sms)
        return 0;
    return requested_sms ? requested_sms :
        (device_sms < MEGAMOE_TMA_COPY_MAX_SMS ? device_sms : MEGAMOE_TMA_COPY_MAX_SMS);
}

/* Each logical slice remains 8192B; physical slots add 16B for one aligned
 * load that straddles the logical slice boundary.
 * Maximum whole slots after control records and 128-byte launch alignment.
 * Capacity is independent of warp count; remainder slots are not discarded. */
static inline int megamoe_tma_copy_total_slots(int available_bytes) {
    if (available_bytes <= 0) return 0;
    const int slot_bytes = MEGAMOE_TMA_COPY_SLOT_DATA_BYTES + MEGAMOE_TMA_COPY_SLOT_CONTROL_BYTES;
    int slots = available_bytes / slot_bytes;
    while (slots > 0 && (((uint64_t)slots * slot_bytes + 127) & ~(uint64_t)127) >
                         (uint64_t)available_bytes)
        --slots;
    return slots;
}
static inline int megamoe_tma_copy_max_warps(int total_slots, int max_threads) {
    if (total_slots < 2 || max_threads < MEGAMOE_TMA_COPY_THREADS_PER_WORKER) return 0;
    int warps = total_slots / 2;
    if (warps > MEGAMOE_TMA_COPY_MAX_WARPS) warps = MEGAMOE_TMA_COPY_MAX_WARPS;
    if (warps > max_threads / MEGAMOE_TMA_COPY_THREADS_PER_WORKER) warps = max_threads / MEGAMOE_TMA_COPY_THREADS_PER_WORKER;
    return warps;
}

#if defined(__CUDACC__)
#define MEGAMOE_TMA_HD __host__ __device__
#else
#define MEGAMOE_TMA_HD
#endif
/* Validated geometry has warps>=1 and total_slots>=2*warps. The first
 * total_slots%warps warps own one extra slot; every slot has exactly one owner. */
static inline MEGAMOE_TMA_HD int megamoe_tma_warp_slot_count(int total_slots, int warps, int warp) {
    return total_slots / warps + (warp < total_slots % warps);
}
static inline MEGAMOE_TMA_HD int megamoe_tma_warp_slot_begin(int total_slots, int warps, int warp) {
    int extra = total_slots % warps;
    return warp * (total_slots / warps) + (warp < extra ? warp : extra);
}
/* Shared with CPU coverage checks: CTA/SM first, warp second, sequence last. */
static inline MEGAMOE_TMA_HD uint64_t megamoe_tma_worker_first_slice(
    int cta, int sms, int warp) {
    return (uint64_t)cta + (uint64_t)sms * (uint64_t)warp;
}
static inline MEGAMOE_TMA_HD uint64_t megamoe_tma_worker_slice_stride(int sms, int warps) {
    return (uint64_t)sms * (uint64_t)warps;
}
#undef MEGAMOE_TMA_HD

typedef struct MegamoeTmaCopyState MegamoeTmaCopyState;

/* A physical interval within one logically concatenated byte stream. Entries
 * in a range are contiguous in virtual space (no per-weight/plane rounding).
 * Addresses/lengths are multiples of four; source is unicast, dst is multicast.
 */
typedef struct MegamoeTmaCopySegment {
    uint64_t src;
    uint64_t dst;
    uint64_t bytes;
    uint64_t virtual_begin;
} MegamoeTmaCopySegment;

/* A whole routing domain: source-owned slots, or one scatter destination team.
 * Slice j spans [j*8192,min((j+1)*8192,bytes)) in its virtual byte stream and can
 * cross any number of segments. It keeps ONE GPU/CTA/warp owner throughout.
 * This rank owns floor((first_slice+j*slice_stride)/slice_divisor). Uniform
 * routing uses divisor=1, reciprocal=0 and retains the original integer path.
 * Otherwise reciprocal=UINT64_MAX/divisor enables an exact mulhi+correction.
 * prefix_end compacts rank-local slices across ranges for SM/worker round-robin.
 * No reset at weight boundaries.
 */
typedef struct MegamoeTmaCopyRange {
    uint64_t segment_begin;
    uint64_t segment_count;
    uint64_t bytes;
    uint64_t first_slice;
    uint64_t slice_stride;
    uint64_t slice_count;
    uint64_t prefix_end;
    uint64_t slice_divisor;
    uint64_t slice_reciprocal;
} MegamoeTmaCopyRange;

typedef struct MegamoeTmaCopyConfig {
    int abi_version;
    int device;
    int device_sm_count;
    int sms;
    int warps;
    int threads_per_cta;
    int slots_per_warp;       /* Minimum over all warps, floor(total_slots/warps). */
    int bank0_slots_per_warp; /* Minimum bank 0 capacity, ceil(slots_per_warp/2). */
    int bank1_slots_per_warp; /* Minimum bank 1 capacity, floor(slots_per_warp/2). */
    int slice_bytes;
    int dynamic_shared_bytes;
    int device_optin_shared_bytes;
    int device_shared_bytes_per_sm;
    int max_active_ctas_per_sm;
    int compute_major;
    int compute_minor;
    int total_slots;
    int max_slots_per_warp;
    int extra_slot_warps; /* First this many warps receive max_slots_per_warp. */
    int max_warps;        /* Shared-memory and kernel/device thread capacity. */
    uint64_t max_segments;
} MegamoeTmaCopyConfig;

/* Every function returns a CUDA runtime error integer (0 == success).
 * sms=0 selects min(device_sm_count,8); otherwise 1..min(device_sm_count,8).
 * Explicit counts above eight are rejected; READY stays in this same kernel.
 * warps=0 selects seven communication warps; explicit requests are 1..32.
 * Runtime geometry requires at least two 8 KiB slots per warp and respects
 * the active kernel and device limits.
 * All whole shared-memory slots are divided near-equally among warps, reserving
 * an mbarrier/phase record for every slot. Two banks use ALL allocated slots.
 * Excessive runtime warp counts return cudaErrorInvalidConfiguration.
 * Creation verifies that this shared-memory footprint permits one CTA/SM.
 */
int megamoe_tma_copy_create(uint64_t max_segments, int sms, int warps,
                          MegamoeTmaCopyState **out);

/* Copies caller descriptors into preallocated pinned storage before returning.
 * The stream is an opaque cudaStream_t, including NULL for the default stream.
 * Source/destination allocations must outlive completion on that stream.
 * Repeated submissions are ordered across streams using a completion event.
 * Reuse can wait for the preceding descriptor DMA, never for its full payload.
 * CUDA graph capture is unsupported. Calls require the state's current device.
 * Concurrent submits are serialized; destroy must not race any other call.
 * range_count=0 is an ordered no-op. The caller owns SYS READY/terminal publication
 * after this stream's kernel, which returns only after multicast writes finish.
 */
int megamoe_tma_copy_submit(MegamoeTmaCopyState *state,
                          const MegamoeTmaCopySegment *segments, uint64_t segment_count,
                          const MegamoeTmaCopyRange *ranges, uint64_t range_count,
                          void *cuda_stream);
/* Fused GPU completion publication. After every CTA's multicast payload is
 * complete, the last finishing CTA writes generation to this rank's 8-byte
 * terminal slot through flag_mc (a nonzero, 8-byte-aligned multicast VA).
 * generation must be nonzero. No extra notification kernel, stream barrier,
 * host generation-table upload, or copy-engine terminal write is submitted.
 * Empty work still launches one notification-only CTA. The counter is reset
 * by the last CTA; submissions on this state remain ordered across streams.
 * A consumer must acquire the expected generation from every rank before
 * reading the payload. READY does not itself grant next-generation reuse.
 */
int megamoe_tma_copy_submit_notify(MegamoeTmaCopyState *state,
                                 const MegamoeTmaCopySegment *segments, uint64_t segment_count,
                                 const MegamoeTmaCopyRange *ranges, uint64_t range_count,
                                 uint64_t flag_mc, uint64_t generation, void *cuda_stream);
/* GPU-direct cold setup: input config contains HOST pointer tables. They are
 * uploaded once; current HALO-Q outputs are bound by device address, never copied.
 * Submit is stream-ordered and does not read or wait for plan data on the CPU.
 * All bound tensors must remain alive and only be reused on the same stream
 * after the copy kernel has read them. The Python endpoint enforces this lease.
 */
int megamoe_tma_copy_configure_gpu_plan(MegamoeTmaCopyState *state,
                                      const MegamoeTmaGpuPlanConfig *config);
int megamoe_tma_copy_bind_gpu_direct(MegamoeTmaCopyState *state,
    uint64_t ids, uint64_t levels, uint64_t owners, uint64_t workspace, int capacity);
int megamoe_tma_copy_submit_gpu_direct(MegamoeTmaCopyState *state,
    uint64_t flag_mc, uint64_t generation, void *cuda_stream);
/* Explicit diagnostic only: synchronizes last work and copies one small result. */
int megamoe_tma_copy_gpu_plan_result(MegamoeTmaCopyState *state,
                                   MegamoeTmaGpuPlanResult *result);
int megamoe_tma_copy_destroy(MegamoeTmaCopyState *state);
int megamoe_tma_copy_config_info(const MegamoeTmaCopyState *state,
                               MegamoeTmaCopyConfig *out);
const char *megamoe_tma_copy_error_string(int error);

#ifdef __cplusplus
}
#endif
#endif
