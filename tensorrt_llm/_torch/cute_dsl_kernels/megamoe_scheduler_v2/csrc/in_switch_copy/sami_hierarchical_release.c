/* Hierarchical HALO-Q SAMI submitter.
 *
 * The hot path waits for the mapped scheduler copy plan, classifies all
 * helper slots in C, builds one mixed owner-direct/scatter descriptor batch,
 * submits exactly one cuMemcpyBatchAsync, performs one SYS release, and then
 * multicasts this rank's terminal generation. */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cuda.h>

#include "sami_shards.h"
#ifdef SAMI_ENABLE_TMA
#include "gpu_plan.h"
#include "tma_copy.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#define SAMI_SPIN_HINT() __asm__ __volatile__("yield" ::: "memory")
#else
#define SAMI_SPIN_HINT() __asm__ __volatile__("pause" ::: "memory")
#endif

#define SAMI_PLAN_MARKER 0
#define SAMI_GEN_TABLE_LEN 65536
#define SAMI_MAX_WORLD 32
#define SAMI_MAX_LEVELS 4

/* The global publisher selects direct writers. Explicit capacity zero
 * uniformly demotes local direct work to scatter; positive ratios weight
 * issuers which already carry direct payload during scatter waterfill. */
#ifndef SAMI_OWNER_CAPACITY_NUMERATOR
#define SAMI_OWNER_CAPACITY_NUMERATOR 1
#endif
#ifndef SAMI_OWNER_CAPACITY_DENOMINATOR
#define SAMI_OWNER_CAPACITY_DENOMINATOR 1
#endif
#if SAMI_OWNER_CAPACITY_NUMERATOR <= 0 || SAMI_OWNER_CAPACITY_DENOMINATOR <= 0
#error "SAMI owner capacity ratio must be positive"
#endif

typedef struct
{
    int world;
    int rank;
    int helper_count;
    int planes;
    int global_experts;
    int home_count;
    int owner_stride;
    int level_count;
    int partition;
    int mixed_direct_enabled; /* -1: auto; 0/1: explicit owner capacity */
    int striped_scatter;
    int scatter_concentrate;
    int owner_capacity_numerator;
    int owner_capacity_denominator;
    int group_sizes[SAMI_MAX_LEVELS];

    CUdeviceptr* src_table;                   /* [global_experts][planes] */
    CUdeviceptr* dst_table;                   /* [level][helper][planes] */
    CUdeviceptr* foreign_dst_table;           /* [target][helper][plane], external owner */
    int target_count;
    int* slot_scratch;                        /* [4][helper]: expert, level, owner, mode */
    uint32_t (*column_masks)[SAMI_MAX_WORLD]; /* [helper][member], column bits */
    int plan_abi_version;
    int plan_words;
    int route_features;
    int plan_on_device;  /* Direct HALO-Q outputs; no mapped CPU plan. */
    size_t* plane_bytes; /* [planes] */
    size_t* shard_off;   /* [level][plane][world] */
    size_t* shard_bytes; /* [level][plane][world] */

    CUdeviceptr* dsts;
    CUdeviceptr* srcs;
    size_t* sizes;
    CUmemcpyAttributes attr;
    size_t attr_idx;

    uint64_t generation;
    uint64_t generation_base;
    CUdeviceptr generation_table;
    uint64_t* generation_host;
    CUdeviceptr flag_mc;
    CUstreamBatchMemOpParams release_barrier;

    int32_t volatile* plan;
    size_t last_commands;
    int last_direct_slots;
    int last_scatter_slots;
#ifdef SAMI_ENABLE_TMA
    MegamoeTmaCopyState* tma;
    MegamoeTmaCopySegment* tma_segments;
    MegamoeTmaCopyRange* tma_ranges;
    size_t tma_segment_count, tma_range_count;
    size_t max_commands, tma_max_ranges;
    int tma_source_load_percent; /* 100: exact RR; 1..199: source percent/team. */
    uint64_t tma_source_reciprocal[SAMI_MAX_LEVELS];
    uint64_t tma_peer_reciprocal[SAMI_MAX_LEVELS];
    int last_tma_weighted_domains, last_tma_uniform_domains;
    int last_tma_mixed_source_domains, last_tma_foreign_source_domains;
    uint64_t last_tma_weighted_total_slices, last_tma_weighted_source_quota_slices;
    uint64_t last_tma_weighted_local_slices, last_tma_local_slices;
    int tma_route; /* 0: published SAMI modes, 1: source, 2: scatter */
#endif
} hierarchical_ctx;

#ifdef SAMI_ENABLE_TMA
/* Cold setup only. The hot builder emits one monotonic range per domain;
 * it never enumerates a periodic ownership table or allocates temporary data. */
static void initialize_tma_dealing(hierarchical_ctx* c)
{
    if (c->tma_source_load_percent == 100)
        return;
    for (int level = 0; level < c->level_count; ++level)
    {
        unsigned denominator = 100U * (unsigned) c->group_sizes[level];
        unsigned source_divisor = (unsigned) c->tma_source_load_percent;
        c->tma_source_reciprocal[level] = source_divisor == 1 ? 0 : UINT64_MAX / source_divisor;
        unsigned peer_divisor = denominator - (unsigned) c->tma_source_load_percent;
        c->tma_peer_reciprocal[level] = peer_divisor == 1 ? 0 : UINT64_MAX / peer_divisor;
    }
}

static uint64_t tma_weighted_source_quota(uint64_t slices, unsigned members, unsigned percent)
{
    /* A byte stream has at most ceil(UINT64_MAX/8192) slices. Multiplication
     * by percent<=199 is safe; the source gets percent/(100*team), not weight:1. */
    return slices * percent / (100U * members);
}

#endif

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static int raise_cu(CUresult rc, char const* what)
{
    char const *name = NULL, *description = NULL;
    cuGetErrorName(rc, &name);
    cuGetErrorString(rc, &description);
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s: %s", what, name ? name : "?", description ? description : "?");
    return -1;
}

static void ctx_free(hierarchical_ctx* c)
{
    if (!c)
        return;
#ifdef SAMI_ENABLE_TMA
    if (c->tma)
        megamoe_tma_copy_destroy(c->tma);
    free(c->tma_segments);
    free(c->tma_ranges);
#endif
    free(c->src_table);
    free(c->dst_table);
    free(c->foreign_dst_table);
    free(c->slot_scratch);
    free(c->column_masks);
    free(c->plane_bytes);
    free(c->shard_off);
    free(c->shard_bytes);
    free(c->dsts);
    free(c->srcs);
    free(c->sizes);
    if (c->generation_table)
        cuMemFree(c->generation_table);
    if (c->generation_host)
        cuMemFreeHost(c->generation_host);
    free(c);
}

static void capsule_destroy(PyObject* capsule)
{
    ctx_free((hierarchical_ctx*) PyCapsule_GetPointer(capsule, "sami_hierarchical_ctx"));
}

static int prepare_generation(hierarchical_ctx* c)
{
    c->generation = 1;
    c->generation_base = 1;
#ifndef SAMI_ENABLE_TMA
    CUresult rc = cuMemAlloc(&c->generation_table, (size_t) SAMI_GEN_TABLE_LEN * sizeof(uint64_t));
    if (rc == CUDA_SUCCESS)
        rc = cuMemAllocHost((void**) &c->generation_host, (size_t) SAMI_GEN_TABLE_LEN * sizeof(uint64_t));
    if (rc == CUDA_SUCCESS)
    {
        for (uint64_t index = 0; index < SAMI_GEN_TABLE_LEN; ++index)
            c->generation_host[index] = c->generation_base + index;
        rc = cuMemcpyHtoD(c->generation_table, c->generation_host, (size_t) SAMI_GEN_TABLE_LEN * sizeof(uint64_t));
    }
    if (rc != CUDA_SUCCESS)
        return raise_cu(rc, "hierarchical generation table setup");
#endif
    return 0;
}

static int owner_capacity_ratio(int* mixed_direct_enabled, int* numerator, int* denominator)
{
    char const* value = getenv("MEGAMOE_SAMI_OWNER_CAPACITY");
    *mixed_direct_enabled = 1;
    *numerator = SAMI_OWNER_CAPACITY_NUMERATOR;
    *denominator = SAMI_OWNER_CAPACITY_DENOMINATOR;
    if (!value || !value[0])
    {
        *mixed_direct_enabled = -1;
        return 0;
    }
    if (!strcmp(value, "0"))
    {
        *mixed_direct_enabled = 0;
        *numerator = 1;
        *denominator = 1;
    }
    else if (!strcmp(value, "1"))
    {
        *numerator = 1;
        *denominator = 1;
    }
    else if (!strcmp(value, "1.25"))
    {
        *numerator = 5;
        *denominator = 4;
    }
    else if (!strcmp(value, "1.5"))
    {
        *numerator = 3;
        *denominator = 2;
    }
    else if (!strcmp(value, "2"))
    {
        *numerator = 2;
        *denominator = 1;
    }
    else if (!strcmp(value, "2.5"))
    {
        *numerator = 5;
        *denominator = 2;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "MEGAMOE_SAMI_OWNER_CAPACITY must be 0, 1, 1.25, 1.5, 2, or 2.5");
        return -1;
    }
    return 0;
}

/* Explicit scatter controls are shared process configuration. Automatic
 * admission is published globally; native refinements only use an independent
 * group's complete, identical local row. */
static int scatter_concentrate(int* concentrate)
{
    char const* value = getenv("MEGAMOE_SAMI_SCATTER_CONCENTRATE");
    *concentrate = -1;
    if (!value || !value[0])
        return 0;
    if (value[1] || (value[0] != '0' && value[0] != '1'))
    {
        PyErr_SetString(PyExc_ValueError, "MEGAMOE_SAMI_SCATTER_CONCENTRATE must be 0 or 1");
        return -1;
    }
    *concentrate = value[0] - '0';
    return 0;
}

static int scatter_placement(int* striped)
{
    char const* value = getenv("MEGAMOE_SAMI_SCATTER_PLACEMENT");
    *striped = 0;
    if (!value || !value[0] || !strcmp(value, "packed"))
        return 0;
    if (!strcmp(value, "striped"))
    {
        *striped = 1;
        return 0;
    }
    PyErr_SetString(PyExc_ValueError, "MEGAMOE_SAMI_SCATTER_PLACEMENT must be packed or striped");
    return -1;
}

static PyObject* sami_create(PyObject* self, PyObject* args, PyObject* kwds)
{
    static char* keywords[] = {
        "world",
        "rank",
        "helper_count",
        "planes",
        "global_experts",
        "src_table",
        "dst_table",
        "plane_bytes",
        "plan_ptr",
        "level_count",
        "group_sizes",
        "owner_stride",
        "flag_mc",
        "partition",
        "plan_abi_version",
        "plan_words",
        "route_features",
        "foreign_dst_table",
        "tma_sm_count",
        "tma_warps",
        "tma_route",
        "tma_source_load_percent",
        "plan_on_device",
        NULL,
    };
    int world, rank, helper_count, planes, global_experts;
    int level_count, owner_stride;
    int partition = SAMI_PARTITION_BUNDLE;
    int plan_abi_version = 6, plan_words = 0, route_features = 0;
    int tma_sms = 0, tma_warps = 0, tma_route = 0, tma_source_load_percent = 100;
    int plan_on_device = 0;
    unsigned long long foreign_pointer = 0;
    unsigned long long src_pointer, dst_pointer, bytes_pointer, plan_pointer;
    unsigned long long groups_pointer, flag_mc;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiiiiKKKKiKiK|iiiiKiiiii", keywords, &world, &rank, &helper_count,
            &planes, &global_experts, &src_pointer, &dst_pointer, &bytes_pointer, &plan_pointer, &level_count,
            &groups_pointer, &owner_stride, &flag_mc, &partition, &plan_abi_version, &plan_words, &route_features,
            &foreign_pointer, &tma_sms, &tma_warps, &tma_route, &tma_source_load_percent, &plan_on_device))
        return NULL;
    if (tma_sms < 0 || tma_warps < 0 || tma_warps > 32 || tma_route < 0 || tma_route > 2 || tma_source_load_percent < 1
        || tma_source_load_percent > 199 || (plan_on_device != 0 && plan_on_device != 1))
    {
        PyErr_SetString(PyExc_ValueError, "invalid TMA copy configuration");
        return NULL;
    }
#ifdef SAMI_ENABLE_TMA
    if (tma_sms > MEGAMOE_TMA_COPY_MAX_SMS)
    {
        PyErr_Format(PyExc_ValueError, "tma_sm_count must be at most %d", MEGAMOE_TMA_COPY_MAX_SMS);
        return NULL;
    }
#endif
#ifndef SAMI_ENABLE_TMA
    if (tma_sms || tma_warps || tma_route || tma_source_load_percent != 100 || plan_on_device)
    {
        PyErr_SetString(PyExc_ValueError, "TMA settings require the TMA backend");
        return NULL;
    }
#endif
    if (world < 2 || world > SAMI_MAX_WORLD || (world & 1) || rank < 0 || rank >= world || helper_count <= 0
        || planes <= 0 || planes > SAMI_SHARDS_MAX_PLANES || global_experts <= 0 || global_experts > 384
        || global_experts % world || level_count <= 0 || level_count > SAMI_MAX_LEVELS || owner_stride < helper_count
        || !src_pointer || !dst_pointer || !bytes_pointer || (!plan_on_device && !plan_pointer) || !groups_pointer
        || !flag_mc || (partition != SAMI_PARTITION_PER_PLANE && partition != SAMI_PARTITION_BUNDLE))
    {
        PyErr_SetString(PyExc_ValueError, "invalid hierarchical SAMI geometry");
        return NULL;
    }

    const int64_t stride = ((int64_t) helper_count + 3) & ~INT64_C(3);
    const int64_t expected_words = 4 + 7 * stride;
    if (!plan_words && expected_words <= INT32_MAX)
        plan_words = (int) expected_words;
    if (plan_abi_version != 6 || expected_words > INT32_MAX || plan_words != expected_words || owner_stride != stride
        || (int64_t) world * (global_experts / world + (int64_t) helper_count) > INT32_MAX || (plan_pointer & 15)
        || (route_features != 0 && route_features != 1) || (!route_features && foreign_pointer)
        || (route_features && !foreign_pointer))
    {
        PyErr_SetString(PyExc_ValueError, "invalid hierarchical route channel ABI");
        return NULL;
    }

    hierarchical_ctx* c = (hierarchical_ctx*) calloc(1, sizeof(*c));
    if (!c)
        return PyErr_NoMemory();
    c->world = world;
    c->rank = rank;
    c->helper_count = helper_count;
    c->planes = planes;
    c->global_experts = global_experts;
    c->home_count = global_experts / world;
    c->owner_stride = owner_stride;
    c->level_count = level_count;
    c->partition = partition;
    c->plan_abi_version = plan_abi_version;
    c->plan_words = plan_words;
    c->route_features = route_features;
    c->plan_on_device = plan_on_device;
    if (owner_capacity_ratio(&c->mixed_direct_enabled, &c->owner_capacity_numerator, &c->owner_capacity_denominator)
        < 0)
    {
        ctx_free(c);
        return NULL;
    }
    if (scatter_concentrate(&c->scatter_concentrate) < 0)
    {
        ctx_free(c);
        return NULL;
    }
    if (scatter_placement(&c->striped_scatter) < 0)
    {
        ctx_free(c);
        return NULL;
    }
    memcpy(c->group_sizes, (void*) (uintptr_t) groups_pointer, (size_t) level_count * sizeof(int));
    for (int level = 0; level < level_count; ++level)
    {
        int size = c->group_sizes[level];
        if (size < (level ? 4 : 2) || size > world || world % size || (!level && size != world)
            || (level && (size >= c->group_sizes[level - 1] || c->group_sizes[level - 1] % size)))
        {
            ctx_free(c);
            PyErr_SetString(PyExc_ValueError, "invalid hierarchical SAMI group sizes");
            return NULL;
        }
    }

    for (int level = 1; level < level_count; ++level)
        c->target_count += world / c->group_sizes[level];
    if (route_features && !c->target_count)
    {
        ctx_free(c);
        PyErr_SetString(PyExc_ValueError, "external routes require proper groups");
        return NULL;
    }
    /* Bound every slot-sized allocation before multiplying on the host.
     * This also makes allocation failure safe on a narrower size_t host. */
    size_t slot_items = (size_t) level_count * (size_t) planes;
    size_t foreign_items = (size_t) c->target_count * (size_t) planes;
    if (route_features && foreign_items > slot_items)
        slot_items = foreign_items;
    if (2u * (size_t) planes > slot_items)
        slot_items = 2u * (size_t) planes;
    size_t slot_bytes = slot_items * sizeof(CUdeviceptr);
    if (sizeof(*c->column_masks) > slot_bytes)
        slot_bytes = sizeof(*c->column_masks);
    if ((size_t) helper_count > SIZE_MAX / slot_bytes)
    {
        ctx_free(c);
        return PyErr_NoMemory();
    }
    size_t source_count = (size_t) global_experts * (size_t) planes;
    size_t destination_count = (size_t) level_count * (size_t) helper_count * (size_t) planes;
    size_t shard_count = (size_t) level_count * (size_t) planes * (size_t) world;
    size_t outgoing_capacity
        = route_features ? (size_t) (helper_count < c->home_count ? helper_count : c->home_count) : 0;
    size_t max_commands = ((size_t) helper_count + outgoing_capacity) * (size_t) planes;
    size_t foreign_count = (size_t) c->target_count * (size_t) helper_count * (size_t) planes;
    c->slot_scratch = (int*) malloc(4u * (size_t) helper_count * sizeof(int));
    c->column_masks = calloc((size_t) helper_count, sizeof(*c->column_masks));
    c->src_table = (CUdeviceptr*) malloc(source_count * sizeof(CUdeviceptr));
    c->dst_table = (CUdeviceptr*) malloc(destination_count * sizeof(CUdeviceptr));
    if (route_features)
        c->foreign_dst_table = (CUdeviceptr*) malloc(foreign_count * sizeof(CUdeviceptr));
    c->plane_bytes = (size_t*) malloc((size_t) planes * sizeof(size_t));
    c->shard_off = (size_t*) calloc(shard_count, sizeof(size_t));
    c->shard_bytes = (size_t*) calloc(shard_count, sizeof(size_t));
    c->dsts = (CUdeviceptr*) malloc(max_commands * sizeof(CUdeviceptr));
    c->srcs = (CUdeviceptr*) malloc(max_commands * sizeof(CUdeviceptr));
    c->sizes = (size_t*) malloc(max_commands * sizeof(size_t));
    if (!c->src_table || !c->dst_table || !c->plane_bytes || !c->shard_off || !c->shard_bytes || !c->dsts || !c->srcs
        || !c->sizes || !c->slot_scratch || !c->column_masks || (route_features && !c->foreign_dst_table))
    {
        ctx_free(c);
        return PyErr_NoMemory();
    }
    memcpy(c->src_table, (void*) (uintptr_t) src_pointer, source_count * sizeof(uint64_t));
    memcpy(c->dst_table, (void*) (uintptr_t) dst_pointer, destination_count * sizeof(uint64_t));
    memcpy(c->plane_bytes, (void*) (uintptr_t) bytes_pointer, (size_t) planes * sizeof(uint64_t));
    size_t total_plane_bytes = 0;
    for (int plane = 0; plane < planes; ++plane)
    {
        size_t bytes = c->plane_bytes[plane];
        if (!bytes || bytes > SIZE_MAX / (size_t) world - total_plane_bytes)
        {
            ctx_free(c);
            PyErr_SetString(PyExc_ValueError, "invalid hierarchical plane sizes");
            return NULL;
        }
        total_plane_bytes += bytes;
    }
    for (size_t i = 0; i < source_count; ++i)
    {
        if (!c->src_table[i] || c->plane_bytes[i % planes] > UINT64_MAX - c->src_table[i])
        {
            ctx_free(c);
            PyErr_SetString(PyExc_ValueError, "invalid hierarchical source range");
            return NULL;
        }
    }
    for (size_t i = 0; i < destination_count; ++i)
    {
        if (!c->dst_table[i] || c->plane_bytes[i % planes] > UINT64_MAX - c->dst_table[i])
        {
            ctx_free(c);
            PyErr_SetString(PyExc_ValueError, "invalid hierarchical destination range");
            return NULL;
        }
    }
    if (route_features)
    {
        memcpy(c->foreign_dst_table, (void*) (uintptr_t) foreign_pointer, foreign_count * sizeof(CUdeviceptr));
        size_t index = 0;
        for (int level = 1; level < level_count; ++level)
        {
            int group = c->group_sizes[level];
            for (int begin = 0; begin < world; begin += group)
            {
                int member = rank >= begin && rank < begin + group;
                for (int slot = 0; slot < helper_count; ++slot)
                {
                    for (int plane = 0; plane < planes; ++plane, ++index)
                    {
                        CUdeviceptr ptr = c->foreign_dst_table[index];
                        if (member ? ptr != 0 : (!ptr || c->plane_bytes[plane] > UINT64_MAX - ptr))
                        {
                            ctx_free(c);
                            PyErr_SetString(PyExc_ValueError, "invalid external destination range");
                            return NULL;
                        }
                    }
                }
            }
        }
    }

    for (int level = 0; level < level_count; ++level)
    {
        int group = c->group_sizes[level];
        size_t compact_count = (size_t) planes * (size_t) group;
        size_t* offsets = (size_t*) calloc(compact_count, sizeof(size_t));
        size_t* lengths = (size_t*) calloc(compact_count, sizeof(size_t));
        if (!offsets || !lengths)
        {
            free(offsets);
            free(lengths);
            ctx_free(c);
            return PyErr_NoMemory();
        }
        int result
            = sami_compute_shards(c->plane_bytes, planes, group, SAMI_SHARD_ALIGNMENT, partition, offsets, lengths);
        if (result != SAMI_SHARDS_OK)
        {
            free(offsets);
            free(lengths);
            ctx_free(c);
            PyErr_Format(PyExc_ValueError, "hierarchical shard partition failed at level %d (rc=%d)", level, result);
            return NULL;
        }
        for (int plane = 0; plane < planes; ++plane)
        {
            for (int column = 0; column < group; ++column)
            {
                size_t source = (size_t) plane * (size_t) group + column;
                size_t target = ((size_t) level * (size_t) planes + plane) * (size_t) world + column;
                c->shard_off[target] = offsets[source];
                c->shard_bytes[target] = lengths[source];
            }
        }
        free(offsets);
        free(lengths);
    }

#ifdef SAMI_ENABLE_TMA
    c->max_commands = max_commands;
    c->tma_route = tma_route;
    c->tma_source_load_percent = tma_source_load_percent;
    initialize_tma_dealing(c);
    /* Every nonempty domain contains at least one physical segment and emits
     * at most one range. Both arrays fit the existing segment capacity. */
    if (max_commands > SIZE_MAX / sizeof(*c->tma_ranges))
    {
        ctx_free(c);
        return PyErr_NoMemory();
    }
    c->tma_max_ranges = max_commands;
    c->tma_segments = (MegamoeTmaCopySegment*) calloc(max_commands, sizeof(*c->tma_segments));
    c->tma_ranges = (MegamoeTmaCopyRange*) calloc(c->tma_max_ranges, sizeof(*c->tma_ranges));
    if (!c->tma_segments || !c->tma_ranges)
    {
        ctx_free(c);
        return PyErr_NoMemory();
    }
    int tma_result = megamoe_tma_copy_create(c->tma_max_ranges, tma_sms, tma_warps, &c->tma);
    if (tma_result)
    {
        PyErr_Format(PyExc_RuntimeError, "TMA copy setup failed (%d requested warps): %s", tma_warps,
            megamoe_tma_copy_error_string(tma_result));
        ctx_free(c);
        return NULL;
    }
    if (plan_on_device)
    {
        /* Cold setup copies immutable address tables, never a scheduler plan. */
        MegamoeTmaGpuPlanConfig config = {
            .world = c->world,
            .rank = c->rank,
            .helper_count = c->helper_count,
            .planes = c->planes,
            .global_experts = c->global_experts,
            .home_count = c->home_count,
            .owner_stride = c->owner_stride,
            .level_count = c->level_count,
            .target_count = c->target_count,
            .plan_abi_version = c->plan_abi_version,
            .plan_words = c->plan_words,
            .route_features = c->route_features,
            .tma_route = c->tma_route,
            .tma_source_load_percent = c->tma_source_load_percent,
            .src_table = (uint64_t const*) c->src_table,
            .dst_table = (uint64_t const*) c->dst_table,
            .foreign_dst_table = (uint64_t const*) c->foreign_dst_table,
            .plane_bytes = (uint64_t const*) c->plane_bytes,
        };
        memcpy(config.group_sizes, c->group_sizes, sizeof(config.group_sizes));
        tma_result = megamoe_tma_copy_configure_gpu_plan(c->tma, &config);
        if (tma_result)
        {
            PyErr_Format(
                PyExc_RuntimeError, "TMA GPU plan setup failed: %s", megamoe_tma_copy_error_string(tma_result));
            ctx_free(c);
            return NULL;
        }
    }
#endif
    c->plan = (int32_t volatile*) (uintptr_t) plan_pointer;
    c->flag_mc = (CUdeviceptr) flag_mc;
    memset(&c->attr, 0, sizeof(c->attr));
    c->attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    c->attr.srcLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
    c->attr.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
    c->attr.flags = CU_MEMCPY_FLAG_DEFAULT;
    memset(&c->release_barrier, 0, sizeof(c->release_barrier));
    c->release_barrier.operation = CU_STREAM_MEM_OP_BARRIER;
    c->release_barrier.memoryBarrier.operation = CU_STREAM_MEM_OP_BARRIER;
    c->release_barrier.memoryBarrier.flags = CU_STREAM_MEMORY_BARRIER_TYPE_SYS;
    if (prepare_generation(c) < 0)
    {
        ctx_free(c);
        return NULL;
    }
    PyObject* capsule = PyCapsule_New(c, "sami_hierarchical_ctx", capsule_destroy);
    if (!capsule)
        ctx_free(c);
    return capsule;
}

#ifdef SAMI_ENABLE_TMA
static PyObject* sami_bind_gpu_direct(PyObject* self, PyObject* args)
{
    PyObject* capsule;
    unsigned long long ids, levels, owners, workspace, status, grid_sync;
    int capacity;
    if (!PyArg_ParseTuple(
            args, "OKKKKKKi", &capsule, &ids, &levels, &owners, &workspace, &status, &grid_sync, &capacity))
        return NULL;
    hierarchical_ctx* c = (hierarchical_ctx*) PyCapsule_GetPointer(capsule, "sami_hierarchical_ctx");
    if (!c)
        return NULL;
    if (!c->plan_on_device || !ids || !levels || !owners || !workspace || !status || !grid_sync
        || ((ids | levels | owners | workspace | status | grid_sync) & 3) || capacity <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "invalid TMA GPU-direct binding");
        return NULL;
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS rc
        = megamoe_tma_copy_bind_gpu_direct(c->tma, ids, levels, owners, workspace, status, grid_sync, capacity);
    Py_END_ALLOW_THREADS if (rc)
    {
        PyErr_Format(PyExc_RuntimeError, "TMA GPU-direct binding failed: %s", megamoe_tma_copy_error_string(rc));
        return NULL;
    }
    Py_RETURN_NONE;
}
#endif

static PyObject* sami_set_loc_hint(PyObject* self, PyObject* args)
{
    PyObject* capsule;
    int source_device, destination_device;
    if (!PyArg_ParseTuple(args, "Oii", &capsule, &source_device, &destination_device))
        return NULL;
    hierarchical_ctx* c = (hierarchical_ctx*) PyCapsule_GetPointer(capsule, "sami_hierarchical_ctx");
    if (!c)
        return NULL;
    c->attr.srcLocHint.id = source_device;
    c->attr.dstLocHint.id = destination_device;
    Py_RETURN_NONE;
}

/* Target ids enumerate each proper hierarchy level, then its aligned groups. */
static int target_geometry(hierarchical_ctx const* c, int target, int* level_out, int* begin_out)
{
    if (target < 0 || target >= c->target_count)
        return -1;
    for (int level = 1; level < c->level_count; ++level)
    {
        int count = c->world / c->group_sizes[level];
        if (target < count)
        {
            *level_out = level;
            *begin_out = target * c->group_sizes[level];
            return 0;
        }
        target -= count;
    }
    return -1;
}

static inline void waterfill_scatter_capacity(int const* direct_units, int members, int scatter_units,
    int owner_capacity_numerator, int owner_capacity_denominator, int* capacity)
{
    int load[SAMI_MAX_WORLD];
    int weight[SAMI_MAX_WORLD];
    for (int member = 0; member < members; ++member)
    {
        load[member] = direct_units[member];
        capacity[member] = 0;
        weight[member] = direct_units[member] > 0 ? owner_capacity_numerator : owner_capacity_denominator;
    }
    for (int unit = 0; unit < scatter_units; ++unit)
    {
        int best = 0;
        for (int member = 1; member < members; ++member)
        {
            /* Compare projected normalized load without floating point.
             * Strict less plus ascending member order is the tie break. */
            if ((int64_t) (load[member] + 1) * weight[best] < (int64_t) (load[best] + 1) * weight[member])
                best = member;
        }
        ++load[best];
        ++capacity[best];
    }
}

#ifdef SAMI_ENABLE_TMA
/* Concatenate BYTES before slicing. A routing domain is either all source
 * work owned by this rank, or all scatter weights for its aligned target team.
 * Ranges preserve configured slice identity across physical plane/weight boundaries.
 */
static int append_tma_segment(
    hierarchical_ctx* c, size_t* segments, uint64_t* cursor, CUdeviceptr src, CUdeviceptr dst, size_t bytes)
{
    if (*segments >= c->max_commands || !src || !dst || !bytes || (src & 3) || (dst & 3) || (bytes & 3)
        || bytes > UINT64_MAX - *cursor || bytes > UINT64_MAX - src || bytes > UINT64_MAX - dst)
        return -1;
    MegamoeTmaCopySegment* s = &c->tma_segments[(*segments)++];
    s->src = src;
    s->dst = dst;
    s->bytes = bytes;
    s->virtual_begin = *cursor;
    *cursor += bytes;
    return 0;
}

/* The validated arena binds one UC object per (owner, plane) and one MC
 * object per (team, plane). The caller resets identity at plane/domain
 * boundaries and gives normal and external teams separate namespaces.
 * Numerical pointer adjacency alone is not enough to merge MC destinations. */
static int append_tma_coalesced_segment(hierarchical_ctx* c, size_t* segments, uint64_t* cursor, CUdeviceptr src,
    CUdeviceptr dst, size_t bytes, uint64_t identity, uint64_t* previous_identity)
{
    if (!src || !dst || !bytes || (src & 3) || (dst & 3) || (bytes & 3) || bytes > UINT64_MAX - *cursor
        || bytes > UINT64_MAX - src || bytes > UINT64_MAX - dst)
        return -1;
    if (*segments && *previous_identity == identity)
    {
        MegamoeTmaCopySegment* previous = &c->tma_segments[*segments - 1];
        if (previous->bytes <= UINT64_MAX - previous->src && previous->bytes <= UINT64_MAX - previous->dst
            && previous->bytes <= UINT64_MAX - previous->virtual_begin && previous->src + previous->bytes == src
            && previous->dst + previous->bytes == dst && previous->virtual_begin + previous->bytes == *cursor
            && bytes <= UINT64_MAX - previous->bytes)
        {
            previous->bytes += bytes;
            *cursor += bytes;
            return 0;
        }
    }
    if (append_tma_segment(c, segments, cursor, src, dst, bytes))
        return -1;
    *previous_identity = identity;
    return 0;
}

static int append_tma_range(hierarchical_ctx* c, size_t* ranges, uint64_t* prefix, size_t segment_begin,
    size_t segment_count, uint64_t bytes, uint64_t first, uint64_t step, uint64_t slices, uint64_t divisor,
    uint64_t reciprocal)
{
    /* These constant bounds also avoid runtime division for overflow checks.
     * Valid 8KiB streams are smaller than this conservative slice bound. */
    if (!step || !divisor || divisor > 100U * SAMI_MAX_WORLD || slices > UINT64_MAX / (100U * SAMI_MAX_WORLD))
        return -1;
    uint64_t scaled_limit = slices * divisor;
    if (first >= scaled_limit)
        return 0;
    uint64_t count = 1 + (scaled_limit - 1 - first) / step;
    if (*ranges >= c->tma_max_ranges || count > UINT64_MAX - *prefix)
        return -1;
    MegamoeTmaCopyRange* r = &c->tma_ranges[(*ranges)++];
    r->segment_begin = segment_begin;
    r->segment_count = segment_count;
    r->bytes = bytes;
    r->first_slice = first;
    r->slice_stride = step;
    r->slice_count = count;
    *prefix += count;
    r->prefix_end = *prefix;
    r->slice_divisor = divisor;
    r->slice_reciprocal = reciprocal;
    return 0;
}

static size_t build_tma_descriptors(hierarchical_ctx* c, int outgoing_count, int* plan_error)
{
    int slots = c->helper_count, stride = c->owner_stride;
    int *experts = c->slot_scratch, *levels = experts + slots;
    int *owners = levels + slots, *modes = owners + slots;
    size_t segments = 0, ranges = 0;
    uint64_t prefix = 0;
    int direct = 0, scatter = 0;
    c->last_tma_weighted_domains = c->last_tma_uniform_domains = 0;
    c->last_tma_mixed_source_domains = c->last_tma_foreign_source_domains = 0;
    c->last_tma_weighted_total_slices = c->last_tma_weighted_source_quota_slices = 0;
    c->last_tma_weighted_local_slices = c->last_tma_local_slices = 0;
    for (int slot = 0; slot < slots; ++slot)
    {
        if (experts[slot] < 0 || modes[slot] == 2)
            continue;
        if (c->tma_route == 1)
        {
            int group = c->group_sizes[levels[slot]];
            int begin = c->rank / group * group;
            if (owners[slot] < begin || owners[slot] >= begin + group)
            {
                *plan_error = 2;
                return 0;
            }
            modes[slot] = 1;
        }
        else if (c->tma_route == 2)
        {
            modes[slot] = 0;
        }
        direct += modes[slot] == 1;
        scatter += modes[slot] == 0;
    }
    /* Domain zero is source-owned work (including outgoing direct). A scatter
     * domain must have one common member set, so it is separated per level;
     * all ranks in that aligned team see the same plane/slot byte order. */
    for (int domain = 0; domain <= c->level_count; ++domain)
    {
        size_t begin_segment = segments;
        uint64_t cursor = 0;
        /* Keep one uninterrupted virtual stream per routing domain. Plane-major
         * order groups the large aligned weights before the scalar planes; the
         * physical arena and source ownership are unchanged. Within each plane,
         * normal source-owned slots precede existing external outgoing records. */
        for (int plane = 0; plane < c->planes; ++plane)
        {
            uint64_t previous_identity = UINT64_MAX;
            for (int slot = 0; slot < slots; ++slot)
            {
                if (experts[slot] < 0 || modes[slot] == 2)
                    continue;
                if (domain == 0 ? (modes[slot] != 1 || owners[slot] != c->rank)
                                : (modes[slot] != 0 || levels[slot] != domain - 1))
                    continue;
                int level = levels[slot];
                size_t si = (size_t) experts[slot] * c->planes + plane;
                size_t di = ((size_t) level * slots + slot) * c->planes + plane;
                uint64_t identity = ((uint64_t) owners[slot] << 32) | (uint32_t) level;
                if (append_tma_coalesced_segment(c, &segments, &cursor, c->src_table[si], c->dst_table[di],
                        c->plane_bytes[plane], identity, &previous_identity))
                    return 0;
            }
            if (domain == 0)
            {
                for (int i = 0; i < outgoing_count; ++i)
                {
                    int expert = c->plan[4 + 4 * stride + i];
                    int target = c->plan[4 + 5 * stride + i];
                    int helper = c->plan[4 + 6 * stride + i];
                    size_t si = (size_t) expert * c->planes + plane;
                    size_t di = ((size_t) target * slots + helper) * c->planes + plane;
                    uint64_t identity = ((uint64_t) c->rank << 32) | (uint32_t) (c->level_count + target);
                    if (append_tma_coalesced_segment(c, &segments, &cursor, c->src_table[si], c->foreign_dst_table[di],
                            c->plane_bytes[plane], identity, &previous_identity))
                        return 0;
                }
            }
        }
        if (!cursor)
            continue;
        uint64_t step = domain ? (uint64_t) c->group_sizes[domain - 1] : 1;
        uint64_t first = domain ? (uint64_t) c->rank % step : 0;
        uint64_t slices = cursor / MEGAMOE_TMA_COPY_SLICE_BYTES + (cursor % MEGAMOE_TMA_COPY_SLICE_BYTES != 0);
        size_t before_ranges = ranges;
        int common_source = -1;
        int weighted = 0;
        if (domain && c->tma_source_load_percent != 100)
        {
            for (int slot = 0; slot < slots; ++slot)
            {
                if (experts[slot] < 0 || modes[slot] != 0 || levels[slot] != domain - 1)
                    continue;
                if (common_source == -1)
                    common_source = owners[slot];
                else if (common_source != owners[slot])
                {
                    common_source = -2;
                    break;
                }
            }
            int team_begin = c->rank / (int) step * (int) step;
            if (common_source == -2)
                ++c->last_tma_mixed_source_domains;
            else if (common_source < team_begin || common_source >= team_begin + (int) step)
                ++c->last_tma_foreign_source_domains;
            else
                weighted = 1;
        }
        if (weighted)
        {
            unsigned members = (unsigned) step;
            unsigned source_local = (unsigned) common_source % members;
            unsigned percent = (unsigned) c->tma_source_load_percent;
            uint64_t denominator = 100U * members;
            uint64_t before_prefix = prefix;
            uint64_t source_quota = tma_weighted_source_quota(slices, members, percent);
            if (slices > UINT64_MAX - c->last_tma_weighted_total_slices
                || source_quota > UINT64_MAX - c->last_tma_weighted_source_quota_slices)
                return 0;
            ++c->last_tma_weighted_domains;
            c->last_tma_weighted_total_slices += slices;
            c->last_tma_weighted_source_quota_slices += source_quota;
            uint64_t divisor, reciprocal;
            if (c->rank == common_source)
            {
                first = denominator - 1;
                step = denominator;
                divisor = percent;
                reciprocal = c->tma_source_reciprocal[domain - 1];
            }
            else
            {
                unsigned peer_order = ((unsigned) c->rank % members + members - source_local) % members - 1;
                first = peer_order * denominator;
                step = (members - 1) * denominator;
                divisor = denominator - percent;
                reciprocal = c->tma_peer_reciprocal[domain - 1];
            }
            /* floor((first+j*step)/divisor) is the inverse of the source
             * quota events and peer round-robin. The whole existing virtual
             * stream stays intact, with increasing slices and a continuous
             * rank-local prefix across domains for SM/worker round-robin. */
            if (append_tma_range(c, &ranges, &prefix, begin_segment, segments - begin_segment, cursor, first, step,
                    slices, divisor, reciprocal))
                return 0;
            c->last_tma_weighted_local_slices += prefix - before_prefix;
        }
        else
        {
            if (domain)
                ++c->last_tma_uniform_domains;
            if (append_tma_range(
                    c, &ranges, &prefix, begin_segment, segments - begin_segment, cursor, first, step, slices, 1, 0))
                return 0;
        }
        if (ranges == before_ranges)
            segments = begin_segment; /* This rank owns none of this stream's slices. */
    }
    c->last_tma_local_slices = prefix;
    c->tma_segment_count = segments;
    c->tma_range_count = ranges;
    c->last_commands = segments;
    c->last_direct_slots = direct;
    c->last_scatter_slots = scatter;
    *plan_error = 0;
    return segments;
}
#endif

static inline size_t build_descriptors(hierarchical_ctx* c, int* plan_error)
{
    const volatile int32_t* plan = c->plan;
    int const slots = c->helper_count;
    int const stride = c->owner_stride;
    int const quota = slots < c->home_count ? slots : c->home_count;
    int* experts = c->slot_scratch;
    int* levels = experts + slots;
    int* owners = levels + slots;
    int* modes = owners + slots;
    uint32_t(*column_masks)[SAMI_MAX_WORLD] = c->column_masks;
    int owner_counts[SAMI_MAX_WORLD] = {0};
    unsigned char seen_experts[384] = {0};
    int direct_slots = 0, scatter_slots = 0, incoming_slots = 0;
    int active = 0, common_level = -1;
    int global_max_send = plan[1];
    int outgoing_count = plan[3];
    *plan_error = 1;
    if (c->plan_abi_version != 6 || c->plan_words != 4 + 7 * stride || plan[2] != 0 || global_max_send < 0
        || global_max_send > quota || outgoing_count < 0 || outgoing_count > quota
        || (!c->route_features && outgoing_count))
        return 0;
    memset(column_masks, 0, (size_t) slots * sizeof(*column_masks));

    /* Published modes are global decisions. Validate the complete local row
     * before constructing any descriptor; inactive capacity never becomes work. */
    for (int slot = 0; slot < slots; ++slot)
    {
        int expert = experts[slot] = plan[4 + slot];
        int level = levels[slot] = plan[4 + stride + slot];
        int owner = owners[slot] = plan[4 + 2 * stride + slot];
        int mode = modes[slot] = plan[4 + 3 * stride + slot];
        if (expert == -1)
        {
            if (level != -1 || owner != -1 || mode != 0)
                return 0;
            continue;
        }
        if (expert < 0 || expert >= c->global_experts || level < 0 || level >= c->level_count || owner < 0
            || owner >= c->world || owner != expert / c->home_count || mode < 0 || mode > 2 || seen_experts[expert])
            return 0;
        seen_experts[expert] = 1;
        if (++owner_counts[owner] > global_max_send)
            return 0;
        int group = c->group_sizes[level];
        int begin = c->rank / group * group;
        int local = owner >= begin && owner < begin + group;
        if ((mode == 1 && !local)
            || (mode == 2 && (!c->route_features || !c->foreign_dst_table || local || level == 0)))
            return 0;
        if (!active)
            common_level = level;
        else if (common_level != level)
            common_level = -2;
        ++active;
        incoming_slots += mode == 2;
    }
    /* Source-direct is valid whenever its owner belongs to the target team.
     * The scheduler may explicitly choose it for multiple weights or levels;
     * auto-policy heuristics are not descriptor safety constraints. */
    if (incoming_slots)
    {
        if (common_level <= 0 || global_max_send == 0 || global_max_send > 2 || (global_max_send == 2 && active < 3))
            return 0;
        int group = c->group_sizes[common_level];
        int begin = c->rank / group * group;
        for (int slot = 0; slot < slots; ++slot)
        {
            if (experts[slot] < 0)
                continue;
            int external = owners[slot] < begin || owners[slot] >= begin + group;
            if (external != (modes[slot] == 2))
                return 0;
        }
    }
    if (outgoing_count && (global_max_send == 0 || global_max_send > 2 || (active && common_level <= 0)))
        return 0;
    if (owner_counts[c->rank] + outgoing_count > global_max_send)
        return 0;
    for (int i = 0; i < outgoing_count; ++i)
    {
        int expert = plan[4 + 4 * stride + i];
        int target = plan[4 + 5 * stride + i];
        int helper = plan[4 + 6 * stride + i];
        int level, begin;
        if (expert < c->rank * c->home_count || expert >= (c->rank + 1) * c->home_count || seen_experts[expert]
            || helper < 0 || helper >= slots || target_geometry(c, target, &level, &begin))
            return 0;
        int group = c->group_sizes[level];
        if (c->rank >= begin && c->rank < begin + group)
            return 0;
        seen_experts[expert] = 1;
        for (int j = 0; j < i; ++j)
            if (plan[4 + 5 * stride + j] == target && plan[4 + 6 * stride + j] == helper)
                return 0;
    }
    for (int i = outgoing_count; i < stride; ++i)
        if (plan[4 + 4 * stride + i] != -1 || plan[4 + 5 * stride + i] != -1 || plan[4 + 6 * stride + i] != -1)
            return 0;
    for (int slot = slots; slot < stride; ++slot)
        if (plan[4 + slot] != -1 || plan[4 + stride + slot] != -1 || plan[4 + 2 * stride + slot] != -1
            || plan[4 + 3 * stride + slot] != 0)
            return 0;

#ifdef SAMI_ENABLE_TMA
    return build_tma_descriptors(c, outgoing_count, plan_error);
#endif

    /* A uniform explicit off setting may demote local direct work. Incoming
     * copies remain owned by their external sender and never regain relays. */
    for (int slot = 0; slot < slots; ++slot)
    {
        if (experts[slot] < 0)
            continue;
        if (c->mixed_direct_enabled == 0 && modes[slot] == 1)
            modes[slot] = 0;
        direct_slots += modes[slot] == 1;
        scatter_slots += modes[slot] == 0;
    }
    int concentrate = c->scatter_concentrate > 0;
    int numerator = c->owner_capacity_numerator;
    int denominator = c->owner_capacity_denominator;
    if (common_level >= 0 && c->mixed_direct_enabled < 0 && c->scatter_concentrate < 0 && !c->striped_scatter)
    {
        int group = c->group_sizes[common_level];
        int begin = c->rank / group * group;
        int external_scatter = 0;
        for (int slot = 0; slot < slots; ++slot)
            if (experts[slot] >= 0 && modes[slot] == 0 && (owners[slot] < begin || owners[slot] >= begin + group))
                ++external_scatter;
        if (active == 1 && external_scatter == 1)
            concentrate = 1;
        if (direct_slots == group - 1 && scatter_slots == 1 && external_scatter == 1 && !incoming_slots)
        {
            numerator = 2;
            denominator = 1;
        }
    }
    for (int level = 0; level < c->level_count; ++level)
    {
        int group = c->group_sizes[level];
        int begin = c->rank / group * group;
        if (c->mixed_direct_enabled == 0)
        {
            for (int slot = 0; slot < slots; ++slot)
            {
                if (experts[slot] < 0 || levels[slot] != level || modes[slot] != 0)
                    continue;
                int owner = owners[slot];
                int local = owner >= begin && owner < begin + group;
                int column = local ? (c->rank - owner + group) % group : c->rank - begin;
                column_masks[slot][c->rank - begin] = UINT32_C(1) << column;
            }
            continue;
        }
        int direct_units[SAMI_MAX_WORLD] = {0};
        int capacity[SAMI_MAX_WORLD];
        int scatter_count = 0;
        for (int slot = 0; slot < slots; ++slot)
        {
            if (experts[slot] < 0 || levels[slot] != level)
                continue;
            if (modes[slot] == 1)
                direct_units[owners[slot] - begin] += group;
            else if (modes[slot] == 0)
                ++scatter_count;
        }
        waterfill_scatter_capacity(direct_units, group, scatter_count * group, numerator, denominator, capacity);
        for (int slot = 0; slot < slots; ++slot)
        {
            if (experts[slot] < 0 || levels[slot] != level || modes[slot] != 0)
                continue;
            int owner = owners[slot];
            int owner_member = owner >= begin && owner < begin + group ? owner - begin : -1;
            int next_column = 0, columns_left = group;
            if (owner_member >= 0 && !concentrate)
            {
                int take = capacity[owner_member] < columns_left ? capacity[owner_member] : columns_left;
                for (int column = 0; column < take; ++column)
                    column_masks[slot][owner_member] |= UINT32_C(1) << next_column++;
                capacity[owner_member] -= take;
                columns_left -= take;
            }
            while (columns_left > 0)
            {
                int best = -1, active_members = 0;
                for (int member = 0; member < group; ++member)
                {
                    if (capacity[member] <= 0 || (concentrate && member == owner_member))
                        continue;
                    ++active_members;
                    if (best < 0 || capacity[member] > capacity[best])
                        best = member;
                }
                if (best < 0 && concentrate)
                {
                    for (int member = 0; member < group; ++member)
                        if (member != owner_member && (best < 0 || capacity[member] > capacity[best]))
                            best = member;
                }
                if (best < 0)
                    return 0;
                int take = capacity[best] < columns_left ? capacity[best] : columns_left;
                if (concentrate)
                    take = columns_left;
                else if (c->striped_scatter && active_members > 1)
                {
                    int fair = (columns_left + active_members - 1) / active_members;
                    if (take > fair)
                        take = fair;
                }
                for (int column = 0; column < take; ++column)
                    column_masks[slot][best] |= UINT32_C(1) << next_column++;
                capacity[best] -= take;
                columns_left -= take;
            }
            if (c->striped_scatter)
            {
                /* Fair placement selects column counts, but can revisit a
                 * member. Pack those unchanged counts into one run per member
                 * so every plane still has at most one descriptor per issuer. */
                int cursor = 0;
                int first = owner_member >= 0 ? owner_member : 0;
                for (int order = 0; order < group; ++order)
                {
                    int member = (first + order) % group;
                    int count = __builtin_popcount(column_masks[slot][member]);
                    uint32_t mask = 0;
                    for (int column = 0; column < count; ++column)
                        mask |= UINT32_C(1) << cursor++;
                    column_masks[slot][member] = mask;
                }
                if (cursor != group)
                    return 0;
            }
        }
    }
    size_t commands = 0;
    const size_t max_commands = ((size_t) slots + (c->route_features ? (size_t) quota : 0)) * (size_t) c->planes;
    for (int slot = 0; slot < slots; ++slot)
    {
        int expert = experts[slot];
        if (expert < 0 || modes[slot] == 2)
            continue;
        int level = levels[slot];
        CUdeviceptr const* source = c->src_table + (size_t) expert * (size_t) c->planes;
        CUdeviceptr const* destination = c->dst_table + ((size_t) level * (size_t) slots + slot) * (size_t) c->planes;
        if (modes[slot] == 1)
        {
            if (c->rank != owners[slot])
                continue;
            for (int plane = 0; plane < c->planes; ++plane)
            {
                if (commands >= max_commands)
                    return 0;
                c->dsts[commands] = destination[plane];
                c->srcs[commands] = source[plane];
                c->sizes[commands++] = c->plane_bytes[plane];
            }
            continue;
        }
        int group = c->group_sizes[level];
        int begin = c->rank / group * group;
        uint32_t mask = column_masks[slot][c->rank - begin];
        for (int plane = 0; plane < c->planes; ++plane)
        {
            size_t offset = 0, end = 0;
            int have_range = 0;
            for (int column = 0; column < group; ++column)
            {
                if (!(mask & (UINT32_C(1) << column)))
                    continue;
                size_t index = ((size_t) level * (size_t) c->planes + plane) * (size_t) c->world + column;
                size_t length = c->shard_bytes[index];
                if (!length)
                    continue;
                size_t shard_offset = c->shard_off[index];
                if (!have_range)
                {
                    offset = shard_offset;
                    end = shard_offset + length;
                    have_range = 1;
                }
                else if (shard_offset == end)
                {
                    end += length;
                }
                else
                {
                    /* One descriptor must cover a contiguous column run. */
                    return 0;
                }
            }
            if (!have_range)
                continue;
            if (commands >= max_commands)
                return 0;
            c->dsts[commands] = destination[plane] + offset;
            c->srcs[commands] = source[plane] + offset;
            c->sizes[commands++] = end - offset;
        }
    }
    /* Stable large-first order keeps each descriptor triplet together. */
    for (size_t index = 1; index < commands; ++index)
    {
        CUdeviceptr saved_dst = c->dsts[index], saved_src = c->srcs[index];
        size_t saved_size = c->sizes[index], insertion = index;
        while (insertion > 0 && c->sizes[insertion - 1] < saved_size)
        {
            c->dsts[insertion] = c->dsts[insertion - 1];
            c->srcs[insertion] = c->srcs[insertion - 1];
            c->sizes[insertion] = c->sizes[insertion - 1];
            --insertion;
        }
        c->dsts[insertion] = saved_dst;
        c->srcs[insertion] = saved_src;
        c->sizes[insertion] = saved_size;
    }
    for (int i = 0; i < outgoing_count; ++i)
    {
        int expert = plan[4 + 4 * stride + i];
        int target = plan[4 + 5 * stride + i];
        int helper = plan[4 + 6 * stride + i];
        CUdeviceptr const* source = c->src_table + (size_t) expert * (size_t) c->planes;
        CUdeviceptr const* destination
            = c->foreign_dst_table + ((size_t) target * (size_t) slots + helper) * (size_t) c->planes;
        for (int plane = 0; plane < c->planes; ++plane)
        {
            if (commands >= max_commands || !destination[plane]
                || c->plane_bytes[plane] > UINT64_MAX - destination[plane])
                return 0;
            c->dsts[commands] = destination[plane];
            c->srcs[commands] = source[plane];
            c->sizes[commands++] = c->plane_bytes[plane];
        }
    }
    c->last_commands = commands;
    c->last_direct_slots = direct_slots;
    c->last_scatter_slots = scatter_slots;
    *plan_error = 0;
    return commands;
}

static inline int wait_plan(hierarchical_ctx* c, int32_t wanted, uint64_t timeout_ns)
{
    uint64_t start = now_ns();
    for (;;)
    {
        int32_t marker = __atomic_load_n((int32_t const*) &c->plan[SAMI_PLAN_MARKER], __ATOMIC_ACQUIRE);
        if (marker - wanted >= 0)
            return 0;
        SAMI_SPIN_HINT();
        if (timeout_ns && now_ns() - start > timeout_ns)
            return -1;
    }
}

static PyObject* sami_submit(PyObject* self, PyObject* const* arguments, Py_ssize_t count)
{
    if (count < 2 || count > 6)
    {
        PyErr_SetString(PyExc_TypeError,
            "submit(ctx, stream, [start_event, end_event, timeout_ns, "
            "payload_end_event])");
        return NULL;
    }
    hierarchical_ctx* c = (hierarchical_ctx*) PyCapsule_GetPointer(arguments[0], "sami_hierarchical_ctx");
    if (!c)
        return NULL;
    CUstream stream = (CUstream) (uintptr_t) PyLong_AsUnsignedLongLong(arguments[1]);
    CUevent start_event = count > 2 ? (CUevent) (uintptr_t) PyLong_AsUnsignedLongLong(arguments[2]) : 0;
    CUevent end_event = count > 3 ? (CUevent) (uintptr_t) PyLong_AsUnsignedLongLong(arguments[3]) : 0;
    uint64_t timeout_ns = count > 4 ? PyLong_AsUnsignedLongLong(arguments[4]) : 2000000000ull;
    CUevent payload_end_event = count > 5 ? (CUevent) (uintptr_t) PyLong_AsUnsignedLongLong(arguments[5]) : 0;
    if (PyErr_Occurred())
        return NULL;

    CUresult result = CUDA_SUCCESS;
    char const* failed = "wait for hierarchical scheduler plan";
    int timed_out = 0;
    int tma_error = 0;
    int plan_error = 0;
    size_t commands = 0;
    Py_BEGIN_ALLOW_THREADS if (!c->plan_on_device)
    {
        timed_out = wait_plan(c, 0, timeout_ns);
        if (!timed_out)
        {
            commands = build_descriptors(c, &plan_error);
            __atomic_store_n((int32_t*) &c->plan[SAMI_PLAN_MARKER], -1, __ATOMIC_RELEASE);
        }
    }
#ifndef SAMI_ENABLE_TMA
    if (!timed_out && !plan_error && c->generation - c->generation_base >= SAMI_GEN_TABLE_LEN)
    {
        failed = "cuStreamSynchronize(hierarchical generation rollover)";
        result = cuStreamSynchronize(stream);
        if (result == CUDA_SUCCESS)
        {
            c->generation_base = c->generation;
            for (uint64_t index = 0; index < SAMI_GEN_TABLE_LEN; ++index)
                c->generation_host[index] = c->generation_base + index;
            failed = "cuMemcpyHtoDAsync(hierarchical generation rollover)";
            result = cuMemcpyHtoDAsync(
                c->generation_table, c->generation_host, (size_t) SAMI_GEN_TABLE_LEN * sizeof(uint64_t), stream);
        }
    }
#endif
    if (!timed_out && !plan_error && result == CUDA_SUCCESS && start_event)
    {
        failed = "cuEventRecord(hierarchical start)";
        result = cuEventRecord(start_event, stream);
    }
#ifdef SAMI_ENABLE_TMA
    if (!timed_out && !plan_error && result == CUDA_SUCCESS)
    {
        // Even zero-payload ranks publish this generation from the GPU.
        failed = "TMA multicast payload and GPU terminal";
        if (c->plan_on_device)
        {
            /* The same CUDA stream orders HALO-Q writes before this kernel.
             * No host wait, descriptor construction, or plan marker reset. */
            tma_error = megamoe_tma_copy_submit_gpu_direct(c->tma, c->flag_mc, c->generation, (void*) stream);
        }
        else
        {
            tma_error = megamoe_tma_copy_submit_notify(c->tma, c->tma_segments, c->tma_segment_count, c->tma_ranges,
                c->tma_range_count, c->flag_mc, c->generation, (void*) stream);
        }
        if (tma_error)
            result = CUDA_ERROR_UNKNOWN;
    }
#else
    if (!timed_out && !plan_error && result == CUDA_SUCCESS && commands)
    {
        failed = "cuMemcpyBatchAsync(hierarchical payload)";
        result = cuMemcpyBatchAsync(c->dsts, c->srcs, c->sizes, commands, &c->attr, &c->attr_idx, 1, stream);
    }
#endif
    if (!timed_out && !plan_error && result == CUDA_SUCCESS && payload_end_event)
    {
        failed = "cuEventRecord(hierarchical payload end)";
        result = cuEventRecord(payload_end_event, stream);
    }
#ifndef SAMI_ENABLE_TMA
    if (!timed_out && !plan_error && result == CUDA_SUCCESS && commands)
    {
        failed = "cuStreamBatchMemOp(hierarchical SYS release)";
        result = cuStreamBatchMemOp(stream, 1, &c->release_barrier, 0);
    }
    if (!timed_out && !plan_error && result == CUDA_SUCCESS)
    {
        failed = "hierarchical terminal multicast";
        result = cuMemcpyAsync(c->flag_mc,
            c->generation_table + sizeof(uint64_t) * (CUdeviceptr) (c->generation - c->generation_base),
            sizeof(uint64_t), stream);
    }
#endif
    if (!timed_out && !plan_error && result == CUDA_SUCCESS && end_event)
    {
        failed = "cuEventRecord(hierarchical end)";
        result = cuEventRecord(end_event, stream);
    }
    if (!timed_out && !plan_error && result == CUDA_SUCCESS)
        ++c->generation;
    Py_END_ALLOW_THREADS

        if (timed_out)
    {
        PyErr_SetString(PyExc_TimeoutError, "hierarchical scheduler plan publication timed out");
        return NULL;
    }
    if (plan_error)
    {
        PyErr_SetString(PyExc_ValueError,
            plan_error == 2 ? "TMA source route requires a multicast alias on the source rank; "
                              "the source is outside this destination group"
                            : "invalid hierarchical scheduler plan publication");
        return NULL;
    }
#ifdef SAMI_ENABLE_TMA
    if (tma_error)
    {
        PyErr_Format(PyExc_RuntimeError, "TMA multicast payload failed: %s", megamoe_tma_copy_error_string(tma_error));
        return NULL;
    }
#endif
    if (result != CUDA_SUCCESS)
    {
        raise_cu(result, failed);
        return NULL;
    }
    return PyLong_FromSize_t(commands);
}

static PyObject* sami_current_gen(PyObject* self, PyObject* capsule)
{
    hierarchical_ctx* c = (hierarchical_ctx*) PyCapsule_GetPointer(capsule, "sami_hierarchical_ctx");
    if (!c)
        return NULL;
    return PyLong_FromUnsignedLongLong(c->generation);
}

#ifdef SAMI_ENABLE_TMA
/* Explicit diagnostics only: never called by the submission hot path. */
static int refresh_gpu_plan_diagnostics(hierarchical_ctx* c)
{
    if (!c->plan_on_device)
        return 0;
    MegamoeTmaGpuPlanResult stats;
    int rc;
    Py_BEGIN_ALLOW_THREADS rc = megamoe_tma_copy_gpu_plan_result(c->tma, &stats);
    Py_END_ALLOW_THREADS if (rc)
    {
        PyErr_Format(PyExc_RuntimeError, "TMA GPU plan diagnostics failed: %s", megamoe_tma_copy_error_string(rc));
        return -1;
    }
    if (stats.error)
    {
        PyErr_Format(PyExc_ValueError, "invalid TMA GPU scheduler plan (error %d)", stats.error);
        return -1;
    }
    c->last_direct_slots = stats.direct_slots;
    c->last_scatter_slots = stats.scatter_slots;
    c->last_commands = (size_t) stats.segment_count;
    c->tma_segment_count = (size_t) stats.segment_count;
    c->tma_range_count = (size_t) stats.range_count;
    c->last_tma_weighted_domains = stats.weighted_domains;
    c->last_tma_uniform_domains = stats.uniform_domains;
    c->last_tma_mixed_source_domains = stats.mixed_source_domains;
    c->last_tma_foreign_source_domains = stats.foreign_source_domains;
    c->last_tma_weighted_total_slices = stats.weighted_total_slices;
    c->last_tma_weighted_source_quota_slices = stats.weighted_source_quota_slices;
    c->last_tma_weighted_local_slices = stats.weighted_local_slices;
    c->last_tma_local_slices = stats.total_slices;
    return 0;
}
#endif

static PyObject* sami_last_mode_counts(PyObject* self, PyObject* capsule)
{
    hierarchical_ctx* c = (hierarchical_ctx*) PyCapsule_GetPointer(capsule, "sami_hierarchical_ctx");
    if (!c)
        return NULL;
#ifdef SAMI_ENABLE_TMA
    if (refresh_gpu_plan_diagnostics(c) < 0)
        return NULL;
#endif
    return Py_BuildValue("iin", c->last_direct_slots, c->last_scatter_slots, (Py_ssize_t) c->last_commands);
}

static PyObject* sami_driver_symbol(PyObject* self, PyObject* unused)
{
#ifdef SAMI_ENABLE_TMA
    return PyLong_FromVoidPtr((void*) &megamoe_tma_copy_submit_notify);
#else
    return PyLong_FromVoidPtr((void*) &cuMemcpyBatchAsync);
#endif
}

#ifdef SAMI_ENABLE_TMA
static PyObject* sami_tma_config(PyObject* self, PyObject* capsule)
{
    hierarchical_ctx* c = (hierarchical_ctx*) PyCapsule_GetPointer(capsule, "sami_hierarchical_ctx");
    if (!c)
        return NULL;
    if (refresh_gpu_plan_diagnostics(c) < 0)
        return NULL;
    MegamoeTmaCopyConfig config;
    int rc = megamoe_tma_copy_config_info(c->tma, &config);
    if (rc)
    {
        PyErr_Format(PyExc_RuntimeError, "TMA config failed: %s", megamoe_tma_copy_error_string(rc));
        return NULL;
    }
    PyObject* result = Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:s}",
        "abi_version", config.abi_version, "device", config.device, "threads_per_cta", config.threads_per_cta,
        "device_sm_count", config.device_sm_count, "sm_count", config.sms, "warps", config.warps, "slots_per_warp",
        config.slots_per_warp, "bank0_slots", config.bank0_slots_per_warp, "bank1_slots", config.bank1_slots_per_warp,
        "slice_bytes", config.slice_bytes, "dynamic_shared_bytes", config.dynamic_shared_bytes,
        "max_active_ctas_per_sm", config.max_active_ctas_per_sm, "total_slots", config.total_slots,
        "max_slots_per_warp", config.max_slots_per_warp, "extra_slot_warps", config.extra_slot_warps, "max_warps",
        config.max_warps, "notification", "fused_gpu_last_cta");
    if (!result)
        return NULL;
    PyObject* dealing = Py_BuildValue("{s:i,s:d,s:K,s:i,s:i,s:i,s:i,s:K,s:K,s:K,s:K}", "source_load_percent",
        c->tma_source_load_percent, "source_load_factor", 0.01 * c->tma_source_load_percent, "max_ranges",
        (unsigned long long) c->tma_max_ranges, "weighted_domains", c->last_tma_weighted_domains, "uniform_domains",
        c->last_tma_uniform_domains, "fallback_mixed_source_domains", c->last_tma_mixed_source_domains,
        "fallback_foreign_source_domains", c->last_tma_foreign_source_domains, "weighted_total_slices",
        (unsigned long long) c->last_tma_weighted_total_slices, "weighted_source_quota_slices",
        (unsigned long long) c->last_tma_weighted_source_quota_slices, "weighted_local_slices",
        (unsigned long long) c->last_tma_weighted_local_slices, "local_slices",
        (unsigned long long) c->last_tma_local_slices);
    if (!dealing)
    {
        Py_DECREF(result);
        return NULL;
    }
    int dealing_status = PyDict_Update(result, dealing);
    Py_DECREF(dealing);
    if (dealing_status < 0)
    {
        Py_DECREF(result);
        return NULL;
    }
    PyObject* counts = PyTuple_New(config.warps);
    if (!counts)
    {
        Py_DECREF(result);
        return NULL;
    }
    for (int warp = 0; warp < config.warps; ++warp)
    {
        PyObject* count = PyLong_FromLong(megamoe_tma_warp_slot_count(config.total_slots, config.warps, warp));
        if (!count)
        {
            Py_DECREF(counts);
            Py_DECREF(result);
            return NULL;
        }
        PyTuple_SET_ITEM(counts, warp, count);
    }
    int status = PyDict_SetItemString(result, "warp_slot_counts", counts);
    Py_DECREF(counts);
    if (status < 0)
    {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
#endif

static PyMethodDef methods[] = {
#ifdef SAMI_ENABLE_TMA
    {"tma_config", sami_tma_config, METH_O, "Return TMA geometry; GPU-direct diagnostics explicitly synchronize."},
    {"bind_gpu_direct", sami_bind_gpu_direct, METH_VARARGS,
        "Bind existing HALO-Q device outputs and workspace without copying the plan."},
#endif
    {"create", (PyCFunction) (void*) sami_create, METH_VARARGS | METH_KEYWORDS,
        "Create one precomputed hierarchical hybrid context."},
    {"set_loc_hint", sami_set_loc_hint, METH_VARARGS, "Set CUDA source and destination device hints."},
    {"submit", (PyCFunction) (void*) sami_submit, METH_FASTCALL,
        "Wait, build, and submit one mixed hierarchical batch."},
    {"current_gen", sami_current_gen, METH_O, "Return the next terminal generation."},
    {"last_mode_counts", sami_last_mode_counts, METH_O, "Return (direct slots, scatter slots, command count)."},
    {"driver_symbol", sami_driver_symbol, METH_NOARGS, "Return the linked payload launcher symbol."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
#ifdef SAMI_ENABLE_TMA
    "megamoe_sami_tma",
#else
    "megamoe_sami_hierarchical",
#endif
    "HALO-Q hierarchical SAMI release path.",
    -1,
    methods,
};

#ifdef SAMI_ENABLE_TMA
PyMODINIT_FUNC PyInit_megamoe_sami_tma(void)
#else
PyMODINIT_FUNC PyInit_megamoe_sami_hierarchical(void)
#endif
{
    PyObject* result = PyModule_Create(&module);
    if (!result)
        return NULL;
    PyModule_AddIntConstant(result, "PARTITION_PER_PLANE", SAMI_PARTITION_PER_PLANE);
    PyModule_AddIntConstant(result, "PARTITION_BUNDLE", SAMI_PARTITION_BUNDLE);
    PyModule_AddIntConstant(result, "PLAN_ABI_VERSION", 6);
    PyModule_AddIntConstant(result, "REMOTE_VISIBILITY_RELEASE", 1);
    return result;
}
