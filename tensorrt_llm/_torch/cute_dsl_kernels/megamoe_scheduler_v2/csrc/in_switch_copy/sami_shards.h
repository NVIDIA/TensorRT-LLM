/* SAMI scatter shard partition -- pure C, no CUDA, no Python.
 *
 * Computes the [plane][column] (offset, length) tables that build_scatter()
 * consumes at megamoe_scheduler/csrc/in_switch_copy/sami_batch_release.c:289-298:
 *
 *     const size_t *soff = c->shard_off  + (size_t)col;
 *     const size_t *slen = c->shard_bytes + (size_t)col;
 *     for (int p = 0; p < planes; ++p) {
 *         size_t nbytes = slen[(size_t)p * world];
 *         if (!nbytes) continue;
 *         size_t off = soff[(size_t)p * world];
 *         c->dsts[n] = dst[p] + off;
 *         c->srcs[n] = src[p] + off;
 *
 * So: index is [p * world + col], and `off` is PLANE-RELATIVE -- it is added to
 * that plane's own base pointer.  Nothing here ever assumes the planes are
 * adjacent in memory; they are independent allocations in the live arena and
 * arena.py:372-380 actively rejects overlap between them.
 *
 * TWO STRATEGIES
 *
 *   PER_PLANE (legacy, what production does today)
 *     Cut each plane independently into `world` shards.  Reproduces
 *     _weighted_shards(nbytes, world, 1.0/world) from sami/broadcast.py:100-130.
 *     Every column touches every splittable plane => planes*world commands.
 *
 *   BUNDLE (new)
 *     Concatenate the splittable planes into one LOGICAL byte range and cut that
 *     into `world` runs.  The concatenation is a ruler, not an address: each run
 *     is projected back to (plane, plane-relative offset, length) before any
 *     pointer is formed, and no single command ever spans two planes.  A column
 *     therefore holds at most ONE run per plane, which is exactly what the
 *     existing [plane][column] table shape can express -- hence no ABI change.
 *     Command count drops from planes*world to (world + splittable - 1).
 *
 * DELIBERATELY DEPENDENCY-FREE so the partition can be unit tested on a host
 * with no GPU and no CUDA toolkit.  Include it from sami_batch_release.c.
 */

#ifndef SAMI_SHARDS_H
#define SAMI_SHARDS_H

#include <stddef.h>

#define SAMI_PARTITION_PER_PLANE 0
#define SAMI_PARTITION_BUNDLE    1

/* The shard alignment production already uses: the `alignment: int = 512`
 * default of _weighted_shards at megamoe_scheduler/sami/broadcast.py:101. */
#define SAMI_SHARD_ALIGNMENT 512

#define SAMI_SHARDS_OK      0
#define SAMI_SHARDS_EINVAL (-1)
#define SAMI_SHARDS_ECOVER (-2)  /* self-check failed: not an exact partition */
#define SAMI_SHARDS_EALIGN (-3)  /* self-check failed: unaligned plane offset  */

#define SAMI_SHARDS_MAX_PLANES 16
#define SAMI_SHARDS_MAX_WORLD  64

/* A plane participates in splitting iff it is large enough to be worth cutting.
 * Mirrors the `nbytes <= alignment * world` early-out at sami/broadcast.py:106,
 * derived from the size predicate rather than from plane indices so a future
 * geometry reorder cannot break the partition silently. */
static inline int sami_shards_splittable(size_t nbytes, int world, size_t alignment)
{
    return world > 1 && nbytes > alignment * (size_t)world;
}

/* Legacy per-plane cut, integer arithmetic.
 * At owner_share = 1/world the Python float expression int(nbytes * (1.0/world))
 * is exact for every nbytes < 2^53, so integer division reproduces it bit for
 * bit at any production geometry. */
static inline void sami_shards_per_plane(size_t nbytes, int world, size_t alignment,
                                  size_t *off, size_t *len, int stride)
{
    size_t owner_bytes, remainder, floor_bytes, cap_bytes;
    int index;

    for (index = 0; index < world; ++index) {
        off[(size_t)index * stride] = 0;
        len[(size_t)index * stride] = 0;
    }
    if (!sami_shards_splittable(nbytes, world, alignment)) {
        len[0] = nbytes;                    /* column 0 (the owner) takes it all */
        return;
    }

    owner_bytes = (nbytes / (size_t)world) / alignment * alignment;
    floor_bytes = alignment;
    cap_bytes = nbytes - alignment * (size_t)(world - 1);
    if (owner_bytes > cap_bytes) owner_bytes = cap_bytes;
    if (owner_bytes < floor_bytes) owner_bytes = floor_bytes;

    remainder = nbytes - owner_bytes;
    len[0] = owner_bytes;
    for (index = 0; index < world - 1; ++index) {
        size_t begin = owner_bytes
            + (remainder * (size_t)index / (size_t)(world - 1)) / alignment * alignment;
        size_t end = (index == world - 2)
            ? nbytes
            : owner_bytes
              + (remainder * (size_t)(index + 1) / (size_t)(world - 1)) / alignment * alignment;
        off[(size_t)(index + 1) * stride] = begin;
        len[(size_t)(index + 1) * stride] = end - begin;
    }
}

/* Snap a logical cut down to an alignment boundary IN PLANE-RELATIVE SPACE.
 *
 * Snapping in logical space would be wrong: the logical plane bases are only
 * alignment-multiples by coincidence at hidden=7168/intermediate=3072.
 * geometry.py:68-69 requires only multiples of 16, so a different model shape
 * can put a plane base off the boundary and a logical snap would then project
 * to an unaligned plane offset. */
static inline size_t sami_shards_snap(size_t logical, const size_t *base,
                               const size_t *bytes, const int *split,
                               int split_count, size_t alignment)
{
    int q;
    for (q = 0; q < split_count; ++q) {
        size_t lo = base[split[q]];
        size_t hi = lo + bytes[split[q]];
        if (logical >= lo && logical < hi) {
            size_t rel = logical - lo;
            return lo + rel / alignment * alignment;
        }
    }
    return logical;                          /* at or past the end: leave exact */
}

/* Fill off[planes*world] / len[planes*world].  Both must be caller-allocated.
 * Returns SAMI_SHARDS_OK, or a negative code with the tables left undefined. */
static inline int sami_compute_shards(const size_t *plane_bytes, int planes, int world,
                               size_t alignment, int strategy,
                               size_t *off, size_t *len)
{
    size_t base[SAMI_SHARDS_MAX_PLANES];
    size_t cut[SAMI_SHARDS_MAX_WORLD + 1];
    int split[SAMI_SHARDS_MAX_PLANES];
    int split_count = 0;
    size_t total = 0;
    int p, k, q;

    if (!plane_bytes || !off || !len) return SAMI_SHARDS_EINVAL;
    if (planes <= 0 || planes > SAMI_SHARDS_MAX_PLANES) return SAMI_SHARDS_EINVAL;
    if (world <= 0 || world > SAMI_SHARDS_MAX_WORLD) return SAMI_SHARDS_EINVAL;
    if (alignment == 0 || (alignment & (alignment - 1))) return SAMI_SHARDS_EINVAL;
    for (p = 0; p < planes; ++p)
        if (plane_bytes[p] == 0) return SAMI_SHARDS_EINVAL;

    if (strategy == SAMI_PARTITION_PER_PLANE) {
        for (p = 0; p < planes; ++p)
            sami_shards_per_plane(plane_bytes[p], world, alignment,
                                  off + (size_t)p * world, len + (size_t)p * world, 1);
    } else if (strategy == SAMI_PARTITION_BUNDLE) {
        for (p = 0; p < planes; ++p) {
            for (k = 0; k < world; ++k) {
                off[(size_t)p * world + k] = 0;
                len[(size_t)p * world + k] = 0;
            }
            if (sami_shards_splittable(plane_bytes[p], world, alignment)) {
                base[p] = total;
                total += plane_bytes[p];
                split[split_count++] = p;
            } else {
                base[p] = 0;
                len[(size_t)p * world] = plane_bytes[p];  /* owner column only */
            }
        }
        if (split_count == 0) return SAMI_SHARDS_OK;

        /* Cut the logical concatenation, snapping each interior boundary inside
         * whichever plane it lands in, and force monotonicity so a snap can
         * never make a column negative-length (it may make one empty, which the
         * consumer already skips at sami_batch_release.c:292-294). */
        cut[0] = 0;
        cut[world] = total;
        for (k = 1; k < world; ++k) {
            size_t raw = total * (size_t)k / (size_t)world;
            cut[k] = sami_shards_snap(raw, base, plane_bytes, split,
                                      split_count, alignment);
            if (cut[k] < cut[k - 1]) cut[k] = cut[k - 1];
        }
        if (cut[world - 1] > cut[world]) cut[world - 1] = cut[world];

        for (k = 0; k < world; ++k) {
            for (q = 0; q < split_count; ++q) {
                size_t lo, hi, a, z;
                p = split[q];
                lo = base[p];
                hi = lo + plane_bytes[p];
                a = cut[k] > lo ? cut[k] : lo;
                z = cut[k + 1] < hi ? cut[k + 1] : hi;
                if (z > a) {
                    off[(size_t)p * world + k] = a - lo;
                    len[(size_t)p * world + k] = z - a;
                }
            }
        }
    } else {
        return SAMI_SHARDS_EINVAL;
    }

    /* Fail closed: every plane must be covered exactly once, gapless, and every
     * emitted offset must be aligned.  Nothing in the shipped tree tests the
     * partition, so the check lives with the producer. */
    for (p = 0; p < planes; ++p) {
        size_t cursor = 0;
        for (k = 0; k < world; ++k) {
            size_t o = off[(size_t)p * world + k];
            size_t n = len[(size_t)p * world + k];
            if (n == 0) continue;
            if (o % alignment) return SAMI_SHARDS_EALIGN;
            if (o != cursor) return SAMI_SHARDS_ECOVER;
            cursor += n;
        }
        if (cursor != plane_bytes[p]) return SAMI_SHARDS_ECOVER;
    }
    return SAMI_SHARDS_OK;
}

/* Number of copy commands one rank emits for one slot under `strategy`.
 * Used by sami_create to size the descriptor arrays exactly instead of the
 * current worst-case S*planes bound (sami_batch_release.c:174). */
static inline int sami_shards_commands(const size_t *len, int planes, int world, int col)
{
    int p, n = 0;
    for (p = 0; p < planes; ++p)
        if (len[(size_t)p * world + col]) ++n;
    return n;
}

#endif /* SAMI_SHARDS_H */
