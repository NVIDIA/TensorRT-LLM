# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Kimi K3 routed-expert core for decode (M <= 8 tokens; up to 64 with m_max 64): one persistent CuTe DSL kernel.

k3_moe reads the top-k of trtllm::kimi_k3_noaux_tc_mxfp8_quant (global expert ids and
weights) and its MXFP8 activations, and computes this rank's routed partial [M, 3584]:
- With the route + quant folded in (fold): the kernel reads the router logits and the bf16
  latent instead. Every CTA computes the top-16 of every token itself (the sigmoid keys in
  shared memory aliased on the weight ring, one warp per token selecting, with the device
  code of trtllm::k3_route_quant, so the ids and weights are the same bits), and quantizes
  the latent to MXFP8 into its own scratch rows, which its activation producer gathers.
- Prologue: every CTA derives the same (local expert, <= 8 token slots) groups in shared
  memory: a token bitmask per local expert, a ballot prefix over the experts present, and
  each token's slots in top-k order for the combine.
- FC1 (gate_up, MXFP4 x MXFP8) + SiTU + MXFP8 requantization -> FC2 (down) through an
  FC1->FC2 Lamport handoff (the intermediate slab holds FP8 -0.0 / E8M0 NaN sentinels until
  FC1 writes it) or, with FC2_SYNC counter, per-group counts of the FC1 tiles done (hint:
  both, the count only deciding when to load), then a deterministic routing-weighted combine.
- The last FC2 tile that reads a group's intermediate re-arms it, so the slab is armed
  again when the kernel ends; the tile counters reset themselves the same way.
- The m_max 64 build (wide decode steps, e.g. R x 8 speculative verify tokens) gives an
  expert with t tokens ceil(t / 8) groups of consecutive tokens (64-bit token masks; the
  group's slots name their (token, top-k) pairs), keeps each FC2 task's per-token sums in
  TMEM columns, and combines each m-tile in 8-token chunk tasks. No fold, fused all-reduce,
  head flags or latent slab there.
- With the fused all-reduce (ar_world > 0), the partial is not stored: each m-tile's
  reducer pushes its bf16 rows into every rank's Lamport buffer through a multicast
  mapping; once a CTA has no tile left, its epilogue takes reduction tasks (one m-tile
  each, from their own cursor, one at a time), waits for all ranks' rows, sums them in rank
  order (as the MNNVL one-shot all-reduce does) into the output and empties the buffer.
  Two buffers alternate between calls; the grid's last task claim flips a flag.

Weights are read in place in the trtllm-gen W4A8_MXFP4_MXFP8 layout
(MXFP4WeightTRTLLMGenFusedMoEMethod):
    w3_w1_weight [E, 2I, H/2], rows [up ; gate], interleaved (2i = up_i, 2i+1 =
        gate_i), then shuffled in 32-row blocks: physical row p holds source row
        4*(p%8) + p//8 of its block
    w2_weight    [E, H, I/2], same 32-row block shuffle
    *_weight_scale: same row order, then block_scale_interleave (128x4), which is
        exactly the SF-atom layout the tcgen05 block-scaled MMA consumes.
So FC1 epilogue lane L of warp w in m-tile t holds intermediate column
64t + 16w + 2(L%8) + L//16 (up rows: bit 3 of L clear; the gate row is lane L^8),
and FC2 lane L holds output channel 128t + 32w + 4(L%8) + L//8.

Tile order: FC1 tiles of every group, then FC2 tiles. With the dynamic queue (default)
tiles are claimed from one global cursor, so a CTA only ever waits for FC1 tiles already
claimed by a running CTA: the kernel makes progress with any number of resident CTAs,
e.g. while other streams hold SMs. The static schedule (tile = CTA + k * grid) is faster
when every CTA is resident.

Launched with programmatic dependent launch: barrier setup and the TMEM allocation run
before griddepcontrol.wait, which precedes every read of the producer's outputs and every
global write.

Configuration is per module instance (the shapes are trace-time constants): the
loader injects K3_CONFIG = {"i_tp": ..., "num_ctas": ..., "num_local": ..., ...} before
executing the module; options it leaves out take their defaults.
"""

from __future__ import annotations

import os

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass import dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda.tensor_map import TensorMapDataType

_CFG = globals().get("K3_CONFIG") or {}


def _cfg(key: str, default):
    """A kernel option from the op's configuration (K3_CONFIG), else its default."""
    return type(default)(_CFG.get(key, default))


# =============================================================================
# Problem shape and tunables (trace-time constants).
# =============================================================================
N = 8  # token slots per group (MMA N)
H = 3584  # latent hidden = FC1 K = FC2 M
TOP_K = 16
# Tokens per call: 8 (decode), or up to 64 in the wide build (16, 32 or 64), which groups an
# expert's tokens 8 at a time and keeps the FC2 per-token sums in TMEM.
M_MAX = int(_cfg("m_max", 8))
WIDE = M_MAX > N
assert M_MAX == N or (WIDE and M_MAX % N == 0 and M_MAX <= 64), M_MAX
I_TP = int(_cfg("i_tp", 3072))
assert I_TP % 128 == 0, "I_TP must be a multiple of 128"
NUM_LOCAL = int(_cfg("num_local", 256))  # this rank's experts
SITU_GATE_CAP = float(_cfg("gate_cap", 4.0))
SITU_LINEAR_CAP = float(_cfg("linear_cap", 25.0))
SF_RECIPE = str(_cfg("sf_recipe", "ceil"))
assert SF_RECIPE in ("ceil", "ocp")
# Tile schedule. dynamic: every tile is claimed from one global cursor, in queue order.
# static: tile = CTA + k * grid. hybrid: each CTA's first tile is its own index, the rest are
# claimed. Only the dynamic queue makes progress with any number of resident CTAs.
SCHED = str(_cfg("sched", "dynamic" if int(_cfg("dyn_all", 1)) else "static"))
assert SCHED in ("dynamic", "hybrid", "static"), SCHED
DYN_ALL = SCHED != "static"  # tiles (all, or all but the first) come through the claim ring
HYBRID = SCHED == "hybrid"
PRECLAIM_T = 1 if HYBRID else 0  # the tile whose claim is issued in the prologue
USE_PDL = bool(int(_cfg("pdl", 0 if os.environ.get("TRTLLM_ENABLE_PDL") == "0" else 1)))
# PDL trigger: right after griddepcontrol.wait (0), so the next kernel's CTAs take SMs as this
# kernel's CTAs exit and may stream their weights while its last tasks run; or at CTA exit (1).
LATE_TRIGGER = bool(int(_cfg("late_trigger", 0)))
# All-reduce of the routed partial in the kernel (0: off, else the group size): each FC2
# m-tile's reducer pushes its bf16 rows into every rank's Lamport buffer through the
# multicast mapping, and 28 reduction tasks (one per m-tile), taken by CTAs with no tile
# left, reduce the ranks' rows in rank order into the output.
AR_WORLD = int(_cfg("ar_world", 0))
FUSED_AR = AR_WORLD > 0
# Push only (edge E7): the m-tiles' rows still go to every rank's buffer (half flags[0] & 1), but no reduction tasks
# run, the output is not written and nothing here writes the flags word: the consumer (sandwich (b)) sums the
# ranks in the reducers' order, empties the half it read and owns the word as its call count.
AR_PUSH_ONLY = bool(int(_cfg("ar_push_only", 0)))
assert not AR_PUSH_ONLY or FUSED_AR, "push-only is a mode of the fused all-reduce"
# The reduction tasks spin on other GPUs; with the static schedule, CTAs spinning there could
# hold SMs that this kernel's unscheduled tiles (whose pushes the peers wait for) need while
# another collective holds the rest.
assert SCHED == "dynamic" or not FUSED_AR, "the fused all-reduce needs the dynamic tile queue"
AR_POLL_NS = int(_cfg("ar_poll_ns", 64))
# Poll loops back off with nanosleep between polls (0) or spin tightly (1): a sleeping warp can
# wake well after the value it waits for has landed. 2 also spins in the FC2 start delay.
SPIN = int(_cfg("spin", 0))
assert SPIN in (0, 1, 2), SPIN
# Issue the prologue claim before griddepcontrol.wait: the queue is this layer's, reset by
# its previous call, which ended before the producer kernel (launched after it and waiting
# on it) triggered this launch. (The all-reduce buffer flag is read after the wait; its flip
# follows every CTA's last reduction task, not the tile queue.)
PREWAIT_CLAIM = bool(int(_cfg("prewait_claim", 1))) and DYN_ALL
# A CTA whose first task is an FC2 task claims its next task only once that task's MMAs are done
# (its last stage released). Claiming as soon as its loads are issued, such a CTA takes a second
# FC2 task ahead of the CTAs still issuing their FC1 tile and runs both back to back while those
# get none. Dynamic queue (the hybrid queue claims a CTA's second task up front), M <= 8 build.
FC2_FIRST_HOLD = bool(int(_cfg("fc2_first_hold", 1))) and SCHED == "dynamic" and not WIDE
# FC2 epilogue hand-off. last: each tile's arrival is an atomic round trip and the last
# arrival of an m-tile combines it (the last reader of a group re-arms it). designated: the
# tile of the last group combines each m-tile and the tile of the last m-tile re-arms each
# group, after waiting for the others' fire-and-forget arrivals; a tile only ever waits for
# tiles earlier in the queue, which every CTA processes in order.
EPI_SYNC = str(_cfg("epi_sync", "last"))
assert EPI_SYNC in ("last", "designated"), EPI_SYNC
# FC2 work split. tile: one task per (group, m-tile); its partial rows are stored and the
# m-tile's last (or designated) arrival combines them. slice: one task per (m-tile, slice of
# consecutive groups); the CTA accumulates its groups' rows per token in registers, stores one
# partial per task, and the m-tile's last slice sums the slices in slice order. The model runs
# slice, with or without the fused all-reduce; tile is a config choice.
FC2_MODE = str(_CFG.get("fc2", "slice"))
assert FC2_MODE in ("tile", "slice"), FC2_MODE
SLICE = FC2_MODE == "slice"
FC2_SLICES = int(_cfg("fc2_slices", 5))
SLICE_MAX = 8
# Wide build: at least one slice per FC2_GROUP_TARGET groups (up to WIDE_SLICE_MAX slices), so that the FC2 tasks
# left when the FC1 tiles run out stay short as the group count grows.
FC2_GROUP_TARGET = int(_cfg("fc2_group_target", 6))
WIDE_SLICE_MAX = 16
# Where the slice FC2 combines an m-tile and re-arms a slice's groups. inline: the m-tile's last
# slice task and the slice's m-tile-27 task do it after their own groups (and wait for the
# others). task: every FC2 task only publishes; 28 combine tasks and one re-arm task per slice
# follow the FC2 tasks in the queue, taken by CTAs whose FC2 work is done.
# last: every FC2 task's arrival is an atomic round trip on its m-tile's counter and the last
# arrival combines the m-tile at once; the re-arm tasks follow the FC2 tasks.
# chunks (the wide build's only mode): every FC2 task only publishes; 28 x ceil(M / 8) combine
# tasks, one per (m-tile, 8 tokens), then the re-arm tasks follow the FC2 tasks.
FC2_COMBINE = str(_cfg("fc2_combine", "chunks" if WIDE else "last"))
assert FC2_COMBINE in (("chunks",) if WIDE else ("inline", "task", "last")), FC2_COMBINE
SLICE_TASKS = SLICE and FC2_COMBINE in ("task", "last", "chunks")  # re-arm tasks in the queue
COMBINE_TASKS = SLICE and FC2_COMBINE == "task"
COMBINE_LAST = SLICE and FC2_COMBINE == "last"
COMBINE_CHUNKS = SLICE and FC2_COMBINE == "chunks"
assert 1 <= FC2_SLICES <= SLICE_MAX, FC2_SLICES
# FC1 -> FC2 handoff. scan: FC2's activation stages are loaded as soon as the ring allows and the
# Lamport warps re-load them until their sentinels are gone. counter (slice FC2): the FC1 epilogues
# count their tiles per group with a release; the activation producer acquires a group's count
# before loading its intermediate once, and the re-arm tasks reset the counts instead of re-arming.
# hint (slice FC2): the counts are relaxed and only decide when a group is loaded; the scan still
# validates what was loaded (no fences in the FC1 epilogue). The model runs hint (the slice FC2's re-arm
# tasks and the dynamic queue); builds without them default to scan.
FC2_SYNC = str(_CFG.get("fc2_sync", "hint" if SLICE_TASKS and SCHED == "dynamic" else "scan"))
assert FC2_SYNC in ("scan", "hint", "counter"), FC2_SYNC
FC1_COUNTS = FC2_SYNC != "scan"  # the FC1 epilogues count their tiles per group
SCAN_FC2 = FC2_SYNC != "counter"  # the Lamport warps validate FC2's activation stages
assert not FC1_COUNTS or (SLICE_TASKS and SCHED == "dynamic"), (
    "the FC1 counts need the slice FC2's re-arm tasks and the dynamic queue"
)
# CTAs that drew fewer FC1 tiles start FC2 later, so FC2's loads do not compete with the FC1
# tiles still streaming (worth 3-6 us at 11-32 groups with the slice FC2's scan).
FC2_DELAY_NS = int(_cfg("fc2_delay_ns", 0 if FC1_COUNTS else 1000))
FC2_DELAY_LAG_NS = int(_cfg("fc2_delay_lag_ns", 0 if FC1_COUNTS else 4000))
NUM_LAMPORT_WARPS = 4
# Route + quant in the prologue: the kernel takes the router logits and the bf16 latent instead
# of trtllm::k3_route_quant's outputs (that kernel's device code, imported, so the same bits).
FOLD = bool(int(_cfg("fold", 0)))
# The weights producer starts once grouping phase 3 is done (groups_ready) instead of at the prologue's CTA-wide
# sync; not with FOLD, whose routing scratch lives in the A stages until that sync.
EARLY_RELEASE = bool(int(_cfg("early_release", 1))) and not FOLD
# The routing comes from trtllm::k3_route_quant_ag of the same call without a grid boundary: the kernel
# does not wait for that grid before its prologue; it acquires route_quant_ag's per-token ready words
# (ready[t] for the ids and weights, ready[8 + t] for the MXFP8 row) against this call's epoch (the head
# workspace's hflags[2], advanced by this kernel's last claim, which also re-arms the words of the tokens
# past the call's), and waits for the grid only before writing memory it does not own (the output).
HEAD_FLAGS = bool(int(_cfg("head_flags", 0)))
# With FUSED_AR (Track 5 edge E5): the reduced latent rows also go into the consumer's slab (int32 [3][8][H/2], the
# all-ones word empty; a computed all-ones word is published as 0x7FC07FC0) in buffer ``lat_buf`` (the call's
# ordinal mod 3), and this call re-arms buffer (lat_buf + 1) % 3, plus buffer 0 when ``lat_rearm0`` (the step's last
# call), once its grid wait has returned: the buffer's previous reader completed before this grid could start.
LAT_SLAB = bool(int(_cfg("lat_slab", 0)))
assert not LAT_SLAB or (FUSED_AR and not AR_PUSH_ONLY), (
    "the latent slab is written by the reduction tasks"
)
LAT_SLAB_BUFS = 3
LAT_SLAB_EMPTY = -1
assert not (HEAD_FLAGS and FOLD), (
    "the ready words come from route_quant_ag, which the fold replaces"
)
assert not HEAD_FLAGS or DYN_ALL, (
    "the head epoch advances, and the unused ready words are re-armed, at the queue's last claim"
)
# With HEAD_FLAGS, the PDL trigger: at launch (launch) or once the routing is acquired (ready), where
# the grid wait would have returned.
HEAD_TRIGGER = str(_cfg("head_trigger", "launch"))
assert HEAD_TRIGGER in ("launch", "ready"), HEAD_TRIGGER
HEAD_TRIGGER_READY = HEAD_FLAGS and HEAD_TRIGGER == "ready"
assert not WIDE or (SLICE and SCHED == "dynamic" and FC2_SYNC == "hint"), (
    "the wide build runs the slice FC2 on the dynamic queue with the hint handoff"
)
assert not WIDE or not (FOLD or FUSED_AR or HEAD_FLAGS), (
    "the wide build takes route_quant's outputs and returns the partial"
)
if FOLD:
    from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import k3_route_quant_kernel as _rq


def _device_sm_count() -> int:
    (err,) = cuda_driver.cuInit(0)
    assert err == cuda_driver.CUresult.CUDA_SUCCESS, err
    err, ctx_dev = cuda_driver.cuCtxGetDevice()
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        err, ctx_dev = cuda_driver.cuDeviceGet(0)
    err, n = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, ctx_dev
    )
    assert err == cuda_driver.CUresult.CUDA_SUCCESS, err
    return int(n)


NUM_CTAS = int(_CFG["num_ctas"]) if "num_ctas" in _CFG else _device_sm_count()
CLAIM_BASE = NUM_CTAS if HYBRID else 0  # queue index of the first claim

a_dtype = cutlass.Float4E2M1FN
b_dtype = cutlass.Float8E4M3FN
sf_dtype = cutlass.Float8E8M0FNU
c_dtype = cutlass.BFloat16

a_smem_width = 8
sf_vec_size = 32
num_m0_per_sf_atom = 32
num_m1_per_sf_atom = 4
num_k_per_sf_atom = 4
num_elts_atom_sf_e8 = num_m0_per_sf_atom * num_m1_per_sf_atom * num_k_per_sf_atom
num_elts_atom_sf_fp16 = num_elts_atom_sf_e8 // 2
num_tmem_cols_per_sf_atom = 4
smem_capacity = cutlass.memory.get_smem_capacity_in_bytes("sm_100")
num_mbar_bytes = 1024
# Accumulator (8 columns) + SFA/SFB for every stage (2 x 12 x 4) fit in 128 columns, which
# leaves TMEM for other kernels' CTAs on the same SM. The wide build adds an FC2 task's per-token
# sums from column TOK_COL (one per token, then one scratch column for empty slots).
num_tmem_alloc_cols = 256 if WIDE else 128
TOK_COL = 128

epilog_warp_id = (0, 1, 2, 3)
producer_weights_warp_id = 4
producer_acts_warp_id = 5
scales_tmem_warp_id = 6
consumer_warp_id = 7
lamport_acts_warp_id = 8
EPI_THREADS = len(epilog_warp_id) * 32
EPI_BAR_ID = 1
GROUPING_BAR_ID = 2
threads_per_cta = (8 + NUM_LAMPORT_WARPS) * 32
NUM_WARPS = threads_per_cta // 32
# The prologue grouping runs on every warp but the claimer, whose first atomic claim is in
# flight meanwhile.
GROUPING_WARPS = NUM_WARPS - 1
GROUPING_THREADS = GROUPING_WARPS * 32

_LOG2E = 1.4426950408889634
MX_BLOCK = 32
SF_TMEM_COL = 4
SF_TMEM_DP = 4
E4M3_MAX = 448.0
FP8_SENTINEL_I8 = -128  # 0x80 = FP8 -0.0
SF_SENTINEL_I8 = -1  # 0xFF = E8M0 NaN (never produced)
C_ARM_WORD = -2139062144  # 0x80808080
DYN_RING = 8

MMA_M, MMA_TILER_N, MMA_TILE_K, MMA_INST_K = 128, 8, 128, 32
_mma_tiler_mnk = [MMA_M, MMA_TILER_N, MMA_TILE_K]
_mma_inst_mnk = [MMA_M, MMA_TILER_N, MMA_INST_K]
# SF atoms per tile / TMEM columns per MMA k-block for this tiling
# (compute_sf_rest of the CuTeDSL qmma example): one atom per tile in every dim.
REST_K_SF = REST_M_SF = REST_N_SF = 1
NUM_TMEM_COLS_PER_KBLOCK_SFA = NUM_TMEM_COLS_PER_KBLOCK_SFB = 4

NUM_BYTES_A = MMA_M * MMA_TILE_K * a_smem_width // 8
NUM_BYTES_A_GMEM = MMA_M * MMA_TILE_K * 4 // 8
NUM_BYTES_B = MMA_TILER_N * MMA_TILE_K * b_dtype.width // 8
NUM_BYTES_SFA = num_elts_atom_sf_e8 * sf_dtype.width // 8
NUM_BYTES_SFB = num_elts_atom_sf_e8 * sf_dtype.width // 8
SFB_GROUP_BYTES = num_tmem_cols_per_sf_atom * num_m1_per_sf_atom  # 16

K1_TILES = H // MMA_TILE_K  # 28
K2_TILES = I_TP // MMA_TILE_K
NUM_KBLOCKS = MMA_TILE_K // MMA_INST_K
NUM_TMA_LOAD_BYTES_WEIGHTS = NUM_BYTES_A_GMEM + NUM_BYTES_SFA
# The expert weights and their scales are read once per call: their TMA loads are L2 evict-first
# (createpolicy.fractional.L2::evict_first, fraction 1.0), so the layer's stream does not push
# out the code, states and small weights the other kernels of the step reuse. evict_first 0
# (A/B builds) loads them at normal priority.
EVICT_FIRST = 0x12F0000000000000
W_L2_HINT = EVICT_FIRST if int(_cfg("evict_first", 1)) else None
NUM_TMA_LOAD_BYTES_ACTS_FC2 = NUM_BYTES_B + N * SFB_GROUP_BYTES
SFB_SRC_STRIDE_FC1 = H // sf_vec_size  # 112 = the linear activation-scale row
SF_STRIDE0 = (I_TP // MX_BLOCK) * SF_TMEM_DP  # I/8: per-(group, slot) intermediate-scale bytes
assert SF_STRIDE0 == K2_TILES * SFB_GROUP_BYTES

M_TILES_FC1 = 2 * I_TP // MMA_M
M_TILES_FC2 = H // MMA_M  # 28
M_TILES_TOTAL = M_TILES_FC1 + M_TILES_FC2
# FC2_SYNC counter / hint: FC2 loads a group once this many of its FC1 tiles have counted (a negative
# control of counter sets fewer, so FC2 can read an incomplete intermediate).
FC2_SYNC_NEED = int(_cfg("fc2_sync_need", M_TILES_FC1))


def group_capacity(num_local_experts: int) -> int:
    """Groups this rank can see in one decode step (M <= 8, top-16). Wide build: an expert with
    t tokens has ceil(t / 8) <= 1 + (t - 1) / 8 groups, so P = M_MAX * 16 pairs on E present
    experts make at most E + (P - E) / 8 groups, largest with E = min(experts, P)."""
    if not WIDE:
        return min(num_local_experts, M_MAX * TOP_K)
    pairs = M_MAX * TOP_K
    present = min(num_local_experts, pairs)
    return present + (pairs - present) // N


G_CAP = group_capacity(NUM_LOCAL)
# Slice partials [m-tile][slice][token][128] in the partial buffer (G_CAP * N * H floats; the
# wide build's buffer holds exactly PART_ROWS rows of H).
S_CAP = WIDE_SLICE_MAX if WIDE else SLICE_MAX
PART_ROWS = S_CAP * M_MAX
assert WIDE or M_TILES_FC2 * S_CAP * M_MAX * MMA_M <= G_CAP * N * H
assert M_TILES_FC2 * MMA_M == H
NUM_CHUNKS = (NUM_LOCAL + 31) // 32  # 32 local experts per ballot
CHUNKS_PER_WARP = (NUM_CHUNKS + GROUPING_WARPS - 1) // GROUPING_WARPS
ROUTE_PAIRS = M_MAX * TOP_K  # (token, top-k slot) pairs, one per prologue thread
# Wide build: the pairs per grouping thread, and the second word of each expert's token mask.
PAIR_ITERS = (ROUTE_PAIRS + GROUPING_THREADS - 1) // GROUPING_THREADS
MASK_HI = NUM_CHUNKS * 32
PAIR_CHUNK = 16  # combine: pairs whose partial rows are loaded together
# A/B switches for the m-tile combine (batched: the local pairs compacted in the prologue,
# PAIR_CHUNK loads in flight; loop: per token, its slots one guarded load at a time) and for
# where the reducer resets its counters (before or after the epilogue barrier).
COMBINE = str(_cfg("combine", "batched"))
assert COMBINE in ("batched", "loop"), COMBINE
EPI_RESET = str(_cfg("epi_reset", "before"))
assert EPI_RESET in ("before", "after"), EPI_RESET
assert WIDE or ROUTE_PAIRS <= producer_weights_warp_id * 32

# Per-layer state words (int32). Every counter is back at zero when the kernel ends:
# [0] tile-queue cursor (reset by the last claim: every CTA claims until the queue is empty,
# so exactly one claim per CTA finds it empty),
# [4 + m] FC2 m-tile arrivals (reset by that m-tile's reducer),
# [32 + g] FC2 tiles done reading group g's intermediate (reset by the last one, which
# also re-arms that intermediate),
# [32 + G_CAP + g] FC1 tiles of group g done (FC2_SYNC counter or hint; reset by the slice's re-arm
# task).
ST_CURSOR = 0
ST_AR_CURSOR = 1
ST_MTILE = 4
ST_GROUP = 32
assert ST_MTILE + M_TILES_FC2 <= ST_GROUP
ST_FC1 = ST_GROUP + G_CAP
NUM_STATE = ST_FC1 + G_CAP


NUM_AB_STAGE = (smem_capacity - num_mbar_bytes - 2048 - 8192) // (
    NUM_BYTES_A + NUM_BYTES_B + NUM_BYTES_SFA + NUM_BYTES_SFB
)
NUM_AB_STAGE = (NUM_AB_STAGE // NUM_LAMPORT_WARPS) * NUM_LAMPORT_WARPS  # 12
assert not WIDE or 3 * MASK_HI * 4 <= NUM_BYTES_B * NUM_AB_STAGE  # the wide prologue's scratch

B_SCAN_BYTES_PER_LANE = NUM_BYTES_B // 32
B_SCAN_VEC = 16
B_SCAN_ITERS = B_SCAN_BYTES_PER_LANE // B_SCAN_VEC
SFB_SCAN_VEC = N * SFB_GROUP_BYTES // 32
REARM_VEC4 = N * I_TP // 16  # 16-byte stores per group intermediate
REARM_SF_WORDS = N * K2_TILES  # armed scale words per group

# Fused all-reduce buffers: 2 x [M_MAX][AR_WORLD][H] bf16 per rank, as int32 words. A word
# of -0.0 (fp32 bits 0x80000000) means "not written"; pushes turn bf16 -0.0 into +0.0, so a
# written pair of bf16 values never has that pattern. The reducer puts the pattern back.
AR_BUFS = 2
AR_BUF_WORDS = M_MAX * max(AR_WORLD, 1) * H // 2
AR_EMPTY_WORD = -2147483648  # 0x80000000
AR_RANK_CHUNK = 8  # ranks summed per chunk, as the MNNVL one-shot does for > 8 ranks
AR_TASKS = M_TILES_FC2 if FUSED_AR and not AR_PUSH_ONLY else 0

# Fold: routing scratch in the weight ring's A stages, which no TMA writes before the prologue's
# CTA-wide sync: keys (int32) and sigmoids (fp32) of [M_MAX][896], then the selected ids and
# weights (bf16 bits in int32) of [M_MAX][16].
NUM_EXPERTS = 896
KEY_ITERS = (NUM_EXPERTS + GROUPING_THREADS - 1) // GROUPING_THREADS  # experts per grouping thread
RS_KEY = 0
RS_SIG = RS_KEY + M_MAX * NUM_EXPERTS * 4
RS_ID = RS_SIG + M_MAX * NUM_EXPERTS * 4
RS_W = RS_ID + M_MAX * TOP_K * 4
assert not FOLD or RS_W + M_MAX * TOP_K * 4 <= NUM_BYTES_A * NUM_AB_STAGE
# Fold: the quantization runs on the epilogue and Lamport warps, idle until their first tile; the
# activation producer waits for their arrival on QUANT_BAR before its first gather.
QUANT_BAR_ID = 3
QUANT_THREADS = (len(epilog_warp_id) + NUM_LAMPORT_WARPS) * 32
VEC8_PER_ROW = H // 8
QUANT_ITERS = M_MAX * VEC8_PER_ROW // QUANT_THREADS
assert QUANT_ITERS * QUANT_THREADS == M_MAX * VEC8_PER_ROW and VEC8_PER_ROW % 64 == 0


# =============================================================================
# DSL helpers
# =============================================================================
@dsl_user_op
def _read_globaltimer(*, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int64(
        _llvm.inline_asm(
            _T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _atomic_fetch_add(addr_i64, val, *, loc=None, ip=None):
    """atom.acq_rel.gpu.add.u32 returning the old value."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _red_release_add(addr_i64, val, *, loc=None, ip=None):
    """red.release.gpu.global.add.u32 (no return value, no round trip)."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _red_relaxed_add(addr_i64, val, *, loc=None, ip=None):
    """red.relaxed.gpu.global.add.u32 (no ordering, no round trip)."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _load_relaxed(addr_i64, *, loc=None, ip=None):
    """ld.relaxed.gpu.global.u32."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];", "=r,l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _load_acquire(addr_i64, *, loc=None, ip=None):
    """ld.acquire.gpu.global.u32."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _store_release(addr_i64, val, *, loc=None, ip=None):
    """st.release.gpu.global.u32."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(bf16(hi) << 16) | bf16(lo), round to nearest even."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _bf16_lo(word):
    return (word << cutlass.Int32(16)).bitcast(cutlass.Float32)


def _bf16_hi(word):
    return (word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)


@cute.jit
def _ar_word(cur, tok, rank, ch):
    """Int32 word index of (buffer cur, token, rank, channel) in a fused all-reduce buffer."""
    return cur * AR_BUF_WORDS + ((tok * AR_WORLD + rank) * H + ch) // 2


@cute.jit
def _ar_reduce_tile(ar_uc, y_words, cur, m_tile, tidx, num_tokens, ar_rank, lat_slab, lat_buf):
    """One m-tile of the cross-rank reduction (epilogue warps): wait until every rank's
    rows are in, sum them in rank order (chunks of AR_RANK_CHUNK ranks, like the MNNVL
    one-shot), write the output and put the empty pattern back."""
    tok = tidx // 16
    ch0 = m_tile * MMA_M + (tidx % 16) * 8
    if tok < num_tokens:
        base = _ar_word(cur, tok, cutlass.Int32(0), ch0)
        # A task can be claimed long before its m-tile is done here: until this rank's own
        # rows arrive, poll only them (one load per thread) and back off longer.
        own = cutlass.Boolean(True)
        while own:
            v = ar_uc.load(
                idx=base + ar_rank * (H // 2), vector_size=4, alignment=16, is_volatile=True
            )
            dirty = cutlass.Boolean(False)
            for q in cutlass.range_constexpr(4):
                dirty = dirty | (v[q] == cutlass.Int32(AR_EMPTY_WORD))
            own = dirty
            if own:
                _backoff(4 * AR_POLL_NS)
        # Poll every rank's row until none holds an empty word; sum the rows of that last
        # poll: fp32, rank order, chunks of AR_RANK_CHUNK ranks summed from 0 and then added.
        a0 = cutlass.Float32(0.0)
        a1 = cutlass.Float32(0.0)
        a2 = cutlass.Float32(0.0)
        a3 = cutlass.Float32(0.0)
        a4 = cutlass.Float32(0.0)
        a5 = cutlass.Float32(0.0)
        a6 = cutlass.Float32(0.0)
        a7 = cutlass.Float32(0.0)
        pending = cutlass.Boolean(True)
        while pending:
            dirty = cutlass.Boolean(False)
            acc = [cutlass.Float32(0.0)] * 8
            for rb in cutlass.range_constexpr(0, AR_WORLD, AR_RANK_CHUNK):
                chunk = [cutlass.Float32(0.0)] * 8
                for rr in cutlass.range_constexpr(min(AR_RANK_CHUNK, AR_WORLD - rb)):
                    v = ar_uc.load(
                        idx=base + (rb + rr) * (H // 2),
                        vector_size=4,
                        alignment=16,
                        is_volatile=True,
                    )
                    for q in cutlass.range_constexpr(4):
                        w = cutlass.Int32(v[q])
                        dirty = dirty | (w == cutlass.Int32(AR_EMPTY_WORD))
                        chunk[2 * q] = chunk[2 * q] + _bf16_lo(w)
                        chunk[2 * q + 1] = chunk[2 * q + 1] + _bf16_hi(w)
                for e in cutlass.range_constexpr(8):
                    acc[e] = acc[e] + chunk[e]
            a0, a1, a2, a3, a4, a5, a6, a7 = acc
            pending = dirty
            if pending:
                _backoff(AR_POLL_NS)
        acc = [a0, a1, a2, a3, a4, a5, a6, a7]
        words = (
            _pack_bf16x2(acc[1], acc[0]),
            _pack_bf16x2(acc[3], acc[2]),
            _pack_bf16x2(acc[5], acc[4]),
            _pack_bf16x2(acc[7], acc[6]),
        )
        y_words.store(words, idx=(tok * H + ch0) // 2, alignment=16)
        if cutlass.const_expr(LAT_SLAB):
            ones = cutlass.Int32(LAT_SLAB_EMPTY)
            nan = cutlass.Int32(0x7FC07FC0)
            lat_slab.store(
                tuple(cutlass.select_(w == ones, nan, w) for w in words),
                idx=lat_buf * cutlass.Int32(M_MAX * (H // 2)) + (tok * H + ch0) // 2,
                alignment=16,
            )
        empty = cutlass.Int32(AR_EMPTY_WORD)
        for r in cutlass.range_constexpr(AR_WORLD):
            ar_uc.store((empty, empty, empty, empty), idx=base + r * (H // 2), alignment=16)


@cute.jit
def _emit_row(acc, tok, ch, lane, warp_idx, m_tile, y_tensor, ar_mc, ar_cur, ar_rank):
    """Epilogue: token tok's combined value of channel ch. Without the fused all-reduce it
    is stored; with it, lanes p, p+8, p+16, p+24, p+1, ... hold channels 4p .. 4p+7 of the
    warp's 32, so lanes 0, 2, 4, 6 gather them and push 16 bytes each (-0.0 as +0.0)."""
    if cutlass.const_expr(FUSED_AR):
        bits = _pack_bf16x2(cutlass.Float32(0.0), acc) & cutlass.Int32(0xFFFF)
        bits = cutlass.select_(bits == cutlass.Int32(0x8000), cutlass.Int32(0), bits)
        b1 = cute.arch.shuffle_sync_down(bits, 8)
        b2 = cute.arch.shuffle_sync_down(bits, 16)
        b3 = cute.arch.shuffle_sync_down(bits, 24)
        b4 = cute.arch.shuffle_sync_down(bits, 1)
        b5 = cute.arch.shuffle_sync_down(bits, 9)
        b6 = cute.arch.shuffle_sync_down(bits, 17)
        b7 = cute.arch.shuffle_sync_down(bits, 25)
        if (lane < 8) & (lane % 2 == 0):
            ar_mc.store(
                (
                    bits | (b1 << cutlass.Int32(16)),
                    b2 | (b3 << cutlass.Int32(16)),
                    b4 | (b5 << cutlass.Int32(16)),
                    b6 | (b7 << cutlass.Int32(16)),
                ),
                idx=_ar_word(ar_cur, tok, ar_rank, m_tile * MMA_M + warp_idx * 32 + 4 * lane),
                alignment=16,
            )
    else:
        y_tensor.store(acc.to(c_dtype), idx=tok * H + ch)


@cute.jit
def _wide_slices(num_groups):
    """The wide build's FC2 slice count for num_groups groups: at least min(groups, FC2_SLICES) and one per
    FC2_GROUP_TARGET groups, at most S_CAP, at least 1."""
    s = cutlass.select_(
        num_groups < cutlass.Int32(FC2_SLICES), num_groups, cutlass.Int32(FC2_SLICES)
    )
    per = (num_groups + cutlass.Int32(FC2_GROUP_TARGET - 1)) // cutlass.Int32(FC2_GROUP_TARGET)
    s = cutlass.select_(s < per, per, s)
    s = cutlass.select_(s > cutlass.Int32(S_CAP), cutlass.Int32(S_CAP), s)
    return cutlass.select_(s < cutlass.Int32(1), cutlass.Int32(1), s)


def _backoff(ns, delay=False):
    """Between two polls (delay: two timer reads of the FC2 start delay): nanosleep, or
    nothing when the spin option spins there."""
    if SPIN < (2 if delay else 1):
        prims.nanosleep(ns)


@cute.jit
def _smem_or(arr, idx, val):
    prims.inline_ptx_hl(
        "red.shared.or.b32 [{$r0}], {$r1};", read_only_args=[arr.subview(idx).data_ptr(), val]
    )


@cute.jit
def _tma_gather4_cta(smem_dst, tma_ptr, k_coord, row0, row1, row2, row3, barrier):
    prims.inline_ptx_hl(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];",
        read_only_args=[
            smem_dst.data_ptr(),
            tma_ptr,
            k_coord,
            row0,
            row1,
            row2,
            row3,
            barrier.data_ptr(),
        ],
    )


@cute.jit
def _claim(dyn_slot, dyn_ready, dyn_consumed, ctr_ptr, t):
    """Warp 4: take the next tile of the global queue and publish the raw index CTA-wide.
    Before reusing a ring slot, wait for every reader lane's release."""
    slot = t % cutlass.Int32(DYN_RING)
    if t >= cutlass.Int32(DYN_RING):
        phase = (t // cutlass.Int32(DYN_RING) - 1) % 2
        while not cute.arch.mbarrier_try_wait(dyn_consumed.subview(slot).data_ptr(), phase):
            pass
    if prims.elect_sync():
        v = _atomic_fetch_add(ctr_ptr, cutlass.Int32(1)) + cutlass.Int32(CLAIM_BASE)
        dyn_slot.store(v, idx=slot)
        prims.mbarrier_arrive(dyn_ready.subview(slot))
    cute.arch.sync_warp()
    value = dyn_slot.load(idx=slot, is_volatile=True)
    cute.arch.sync_warp()
    return value


@cute.jit
def _recv(dyn_slot, dyn_ready, dyn_consumed, t, phase):
    slot = t % cutlass.Int32(DYN_RING)
    while not cute.arch.mbarrier_try_wait(dyn_ready.subview(slot).data_ptr(), phase):
        pass
    value = dyn_slot.load(idx=slot, is_volatile=True)
    prims.mbarrier_arrive(dyn_consumed.subview(slot))
    return value


@cute.jit
def _next_tile(
    t,
    claimer: cutlass.Constexpr[bool],
    ph,
    bidx,
    total_tiles,
    dyn_slot,
    dyn_ready,
    dyn_consumed,
    ctr_ptr,
):
    """Merged-queue index of this CTA's t-th tile (FC1 tiles < tiles_fc1 <= FC2
    tiles), or -1. Dynamic: claimed from the global cursor. Static: bidx + t*grid.
    The claimer's own claims go through _claimer_next."""
    if cutlass.const_expr(DYN_ALL):
        # Hybrid too: its first tile (the CTA's index) goes through ring slot 0, so every
        # slot's publish and release counts match the dynamic queue's.
        raw = _recv(dyn_slot, dyn_ready, dyn_consumed, t, ph)
        return cutlass.select_(raw < total_tiles, raw, cutlass.Int32(-1))
    lin = bidx + t * NUM_CTAS
    return cutlass.select_(lin < total_tiles, lin, cutlass.Int32(-1))


@cute.jit
def _retire_claim(raw, total_tasks, state_ptr, ar_flags, ar_cur, hflags, head_e, ready, num_tokens):
    """Claimer warp, after each claim: every CTA claims until the queue is empty, so the
    grid's last claim is the one that finds it empty for the NUM_CTAS-th time. It resets
    the queue for this layer's next call. With HEAD_FLAGS it also advances the epoch, which
    every CTA read before its first claim, and stores the advanced epoch into the ready words
    of the tokens past this call's, which no CTA of this call polls; the routing kernel stores
    it into the others. Every word then holds the next call's epoch, never the epoch + 1 that
    call waits for, also across the int32 wrap (a word left at its initial 0 would match the
    epoch -1)."""
    succ = cutlass.select_(
        total_tasks > cutlass.Int32(CLAIM_BASE),
        total_tasks - cutlass.Int32(CLAIM_BASE),
        cutlass.Int32(0),
    )
    if raw == cutlass.Int32(CLAIM_BASE + NUM_CTAS - 1) + succ:
        if prims.elect_sync():
            _store_release(state_ptr + cutlass.Int64(ST_CURSOR * 4), cutlass.Int32(0))
            if cutlass.const_expr(HEAD_FLAGS):
                for tok in cutlass.range_constexpr(M_MAX):
                    if cutlass.Int32(tok) >= num_tokens:
                        ready.store(head_e + cutlass.Int32(1), idx=tok, is_volatile=True)
                        ready.store(head_e + cutlass.Int32(1), idx=tok + M_MAX, is_volatile=True)
                hflags.store(head_e + cutlass.Int32(1), idx=2, is_volatile=True)


@cute.jit
def _claim_ar_task(state_ptr, epi_flag, tidx, ar_flags, ar_cur):
    """Epilogue warps: the next cross-rank reduction task (an m-tile), or >= AR_TASKS.
    Every CTA claims until none is left, so the grid's last claim comes after every
    reduction and push of this call: it resets the cursor and hands the other buffer to
    the next call."""
    cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
    if tidx == 0:
        task = _atomic_fetch_add(state_ptr + cutlass.Int64(ST_AR_CURSOR * 4), cutlass.Int32(1))
        if task == cutlass.Int32(AR_TASKS + NUM_CTAS - 1):
            _store_release(state_ptr + cutlass.Int64(ST_AR_CURSOR * 4), cutlass.Int32(0))
            ar_flags.store(ar_cur ^ cutlass.Int32(1), idx=0, is_volatile=True)
        epi_flag.store(task, idx=0)
    cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
    return epi_flag.load(idx=0)


@cute.jit
def _claimer_next(t, first_raw, dyn_slot, dyn_ready, dyn_consumed, ctr_ptr, bidx, total_tasks,
                  state_ptr, ar_flags, ar_cur, hflags, head_e, ready, num_tokens):  # fmt: skip
    """The claimer's t-th tile (t >= 1): the claim issued in the prologue, or a new one."""
    if cutlass.const_expr(not DYN_ALL):
        lin = bidx + t * NUM_CTAS
        return cutlass.select_(lin < total_tasks, lin, cutlass.Int32(-1))
    raw = first_raw
    if t != cutlass.Int32(PRECLAIM_T):
        raw = _claim(dyn_slot, dyn_ready, dyn_consumed, ctr_ptr, t)
        _retire_claim(
            raw, total_tasks, state_ptr, ar_flags, ar_cur, hflags, head_e, ready, num_tokens
        )
    return cutlass.select_(raw < total_tasks, raw, cutlass.Int32(-1))


def _tanh_f32(x):
    e = cute.math.exp2(cute.math.abs(x) * cutlass.Float32(-2.0 * _LOG2E), fastmath=True)
    t = (cutlass.Float32(1.0) - e) * cute.arch.rcp_approx(cutlass.Float32(1.0) + e)
    return cutlass.select_(x < cutlass.Float32(0.0), -t, t)


def _sigmoid_f32(x):
    return cute.arch.rcp_approx(
        cutlass.Float32(1.0) + cute.math.exp2(x * cutlass.Float32(-_LOG2E), fastmath=True)
    )


def _situ(gate, up):
    g = (
        cutlass.Float32(SITU_GATE_CAP)
        * _tanh_f32(gate * cutlass.Float32(1.0 / SITU_GATE_CAP))
        * _sigmoid_f32(gate)
    )
    u = cutlass.Float32(SITU_LINEAR_CAP) * _tanh_f32(up * cutlass.Float32(1.0 / SITU_LINEAR_CAP))
    return g * u


def _block_e8m0(amax):
    """E8M0 byte of an MX block and 2^(127-byte) as f32 (ceil: trtllm-gen's SiTU cubin
    on sm_100; ocp: floor(log2(amax)) - 8, saturated by the caller)."""
    bits = cutlass.Int32(amax.bitcast(cutlass.Int32))
    expf = (bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)
    if cutlass.const_expr(SF_RECIPE == "ocp"):
        byte = expf - cutlass.Int32(8)
        byte = cutlass.select_(byte < cutlass.Int32(0), cutlass.Int32(0), byte)
    else:
        sf = amax * cutlass.Float32(1.0 / 448.0)
        sbits = cutlass.Int32(sf.bitcast(cutlass.Int32))
        sexp = (sbits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)
        mant = sbits & cutlass.Int32(0x7FFFFF)
        byte = sexp + cutlass.select_(mant != cutlass.Int32(0), cutlass.Int32(1), cutlass.Int32(0))
        byte = cutlass.select_(byte > cutlass.Int32(0xFE), cutlass.Int32(0xFE), byte)
        byte = cutlass.select_(amax > cutlass.Float32(0.0), byte, cutlass.Int32(0))
    inv = cutlass.Int32((cutlass.Int32(254) - byte) << cutlass.Int32(23)).bitcast(cutlass.Float32)
    return byte, inv


def _scan_b_sentinel(sB, stage, lane):
    base = stage * cutlass.Int32(NUM_BYTES_B) + lane * cutlass.Int32(B_SCAN_BYTES_PER_LANE)
    found = cutlass.Boolean(False)
    for i in range(B_SCAN_ITERS):
        v = sB.subview(base + cutlass.Int32(i * B_SCAN_VEC)).load(vector_size=B_SCAN_VEC)
        for j in range(B_SCAN_VEC):
            found = found | (v[j] == cutlass.Int8(FP8_SENTINEL_I8))
    return prims.vote_sync(0xFFFFFFFF, found, "any")


def _scan_sfb_sentinel(sSFB, stage, lane):
    base = stage * cutlass.Int32(NUM_BYTES_SFB) + lane * cutlass.Int32(SFB_SCAN_VEC)
    v = sSFB.subview(base).load(vector_size=SFB_SCAN_VEC)
    found = cutlass.Boolean(False)
    for j in range(SFB_SCAN_VEC):
        found = found | (v[j] == cutlass.Int8(SF_SENTINEL_I8))
    return prims.vote_sync(0xFFFFFFFF, found, "any")


@cute.jit
def _rearm_group(c_words, cs_words, group, tid):
    """Epilogue warps: put the sentinels back into group's intermediate values and into
    bytes 0..3 of each 16-byte scale group (bytes 4..15 are never written and stay 0)."""
    c_base = group * (REARM_VEC4 * 4)
    for j in cutlass.range_constexpr((REARM_VEC4 + EPI_THREADS - 1) // EPI_THREADS):
        q = j * EPI_THREADS + tid
        if q < REARM_VEC4:
            arm = cutlass.Int32(C_ARM_WORD)
            c_words.store((arm, arm, arm, arm), idx=c_base + q * 4, alignment=16)
    s_base = group * (N * SF_STRIDE0 // 4)
    for j in cutlass.range_constexpr((REARM_SF_WORDS + EPI_THREADS - 1) // EPI_THREADS):
        q = j * EPI_THREADS + tid
        if q < REARM_SF_WORDS:
            cs_words.store(cutlass.Int32(-1), idx=s_base + q * 4)


# =============================================================================
# k3_moe: host function (TMA descriptors + launch)
# =============================================================================
@cute.jit
def k3_moe(
    a1_tensor: cute.Tensor,  # w3_w1_weight viewed (H/2, 2I, E) FP4 bytes, K-major
    b1_tensor: cute.Tensor,  # MXFP8 activations viewed (H, M) FP8 (fold: the scratch, (H, NUM_CTAS*8))
    sfa1_tensor: cute.Tensor,  # w3_w1_weight_scale viewed (512, H/128, 2I/128, E)
    sfb1_tensor: cute.Tensor,  # activation scales (M, H/32) E8M0, linear (fold: (NUM_CTAS*8, H/32))
    c_tensor: cute.Tensor,  # intermediate values (G_cap, N, I) FP8, armed
    c_scale_tensor: cute.Tensor,  # intermediate scales (G_cap, N, I/8), armed
    c_words_tensor: cute.Tensor,  # int32 view of the intermediate values
    cs_words_tensor: cute.Tensor,  # int32 view of the intermediate scales
    a2_tensor: cute.Tensor,  # w2_weight viewed (I/2, H, E)
    b2_tensor: cute.Tensor,  # c_tensor viewed (I, N, G_cap)
    sfa2_tensor: cute.Tensor,  # w2_weight_scale viewed (512, I/128, H/128, E)
    sfb2_tensor: cute.Tensor,  # c_scale viewed (16, I/128, N, G_cap)
    y_tensor: cute.Tensor,  # out (M, H) bf16: the partial, or with FUSED_AR the reduced sum
    y_words_tensor: cute.Tensor,  # the same, as int32 words
    part_tensor: cute.Tensor,  # scratch fp32 (G_cap*N, H)
    topk_ids_tensor: cute.Tensor,  # int32 (M, 16) global expert ids
    topk_w_tensor: cute.Tensor,  # bf16 (M, 16) routing weights
    state_tensor: cute.Tensor,  # int32 (NUM_STATE,) per-layer counters, zero between calls
    ar_uc_tensor: cute.Tensor,  # int32 words: this rank's all-reduce buffers (FUSED_AR)
    ar_mc_tensor: cute.Tensor,  # int32 words: their multicast mapping (FUSED_AR)
    ar_flags_tensor: cute.Tensor,  # int32 [0] = buffer of the next call (FUSED_AR)
    logits_tensor: cute.Tensor,  # fp32 (M * 896,) router logits (FOLD)
    bias_tensor: cute.Tensor,  # fp32 (896,) routing bias (FOLD)
    xin_words_tensor: cute.Tensor,  # int32 view of the bf16 latent (M * H/2,) (FOLD)
    xq_words_tensor: cute.Tensor,  # int32 view of the e4m3 scratch (NUM_CTAS*8 * H/4,) (FOLD)
    xsf_tensor: cute.Tensor,  # uint8 scale scratch (NUM_CTAS*8 * H/32,) (FOLD)
    ready_tensor: cute.Tensor,  # int32 route_quant_ag ready words [16] (HEAD_FLAGS)
    hflags_tensor: cute.Tensor,  # int32 head workspace flags, [2] = epoch (HEAD_FLAGS)
    lat_slab_tensor: cute.Tensor,  # int32 words of the consumer's latent slab [3][8][H/2] (LAT_SLAB)
    num_tokens: cutlass.Int32,
    local_offset: cutlass.Int32,
    num_local: cutlass.Int32,
    ar_rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    lat_buf: cutlass.Int32,
    lat_rearm0: cutlass.Int32,
    stream: cuda_driver.CUstream,
) -> None:
    _kpp = a1_tensor.shape[0]
    _mw = a1_tensor.shape[1]
    _ew = a1_tensor.shape[2]
    tma_a1_desc = cuda.create_tensor_map_tiled(
        global_address=a1_tensor.iterator.toint(), dtype=a_dtype, global_dims=[_kpp * 2, _mw, _ew],
        global_strides=[_kpp // 16, (_mw * _kpp) // 16], box_dims=(MMA_TILE_K, MMA_M, 1),
        swizzle=cuda.TensorMapSwizzle.s128b, tma_format=TensorMapDataType.f416u4_align16b,
    )  # fmt: skip
    tma_b1_desc = cuda.create_tensor_map_tiled_from_view(
        b1_tensor,
        box_dims=(MMA_TILE_K, 1),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    sfa1_fp16 = cute.recast_tensor(sfa1_tensor, cutlass.Uint16)
    tma_sfa1_desc = cuda.create_tensor_map_tiled_from_view(
        sfa1_fp16,
        box_dims=(num_elts_atom_sf_fp16, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    sfb1_ptr = sfb1_tensor.iterator.toint()
    _kpp2 = a2_tensor.shape[0]
    _mw2 = a2_tensor.shape[1]
    _ew2 = a2_tensor.shape[2]
    tma_a2_desc = cuda.create_tensor_map_tiled(
        global_address=a2_tensor.iterator.toint(), dtype=a_dtype, global_dims=[_kpp2 * 2, _mw2, _ew2],
        global_strides=[_kpp2 // 16, (_mw2 * _kpp2) // 16], box_dims=(MMA_TILE_K, MMA_M, 1),
        swizzle=cuda.TensorMapSwizzle.s128b, tma_format=TensorMapDataType.f416u4_align16b,
    )  # fmt: skip
    tma_b2_desc = cuda.create_tensor_map_tiled_from_view(
        b2_tensor,
        box_dims=(MMA_TILE_K, MMA_TILER_N, 1),
        stride_order=(0, 1, 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    sfa2_fp16 = cute.recast_tensor(sfa2_tensor, cutlass.Uint16)
    tma_sfa2_desc = cuda.create_tensor_map_tiled_from_view(
        sfa2_fp16,
        box_dims=(num_elts_atom_sf_fp16, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    sfb2_fp16 = cute.recast_tensor(sfb2_tensor, cutlass.Uint16)
    tma_sfb2_desc = cuda.create_tensor_map_tiled_from_view(
        sfb2_fp16,
        box_dims=(SFB_GROUP_BYTES // 2, 1, N, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    state_ptr = state_tensor.iterator.toint()
    k3_moe_kernel(
        tma_a1_desc, tma_b1_desc, tma_sfa1_desc, tma_a2_desc, tma_b2_desc, tma_sfa2_desc, tma_sfb2_desc,
        sfb1_ptr, state_ptr, c_tensor, c_scale_tensor, c_words_tensor, cs_words_tensor,
        y_tensor, y_words_tensor, part_tensor, topk_ids_tensor, topk_w_tensor, ar_uc_tensor, ar_mc_tensor,
        ar_flags_tensor, logits_tensor, bias_tensor, xin_words_tensor, xq_words_tensor, xsf_tensor, ready_tensor,
        hflags_tensor, lat_slab_tensor, num_tokens, local_offset, num_local, ar_rank, routed_scaling_factor,
        lat_buf, lat_rearm0,
    ).launch(
        grid=[NUM_CTAS, 1, 1], block=[threads_per_cta, 1, 1], cluster=(1, 1, 1), stream=stream,
        use_pdl=USE_PDL,
    )  # fmt: skip
    return


# =============================================================================
# k3_moe kernel
# =============================================================================
@cute.kernel
def k3_moe_kernel(
    tma_a1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_a2_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b2_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa2_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfb2_desc: cutlass.GridConstant[cuda.TensorMap],
    sfb1_ptr: cutlass.Int64,
    state_ptr: cutlass.Int64,
    c_tensor: cutlass.Array,
    c_scale_tensor: cutlass.Array,
    c_words: cutlass.Array,
    cs_words: cutlass.Array,
    y_tensor: cutlass.Array,
    y_words: cutlass.Array,
    part_tensor: cutlass.Array,
    topk_ids: cutlass.Array,
    topk_w: cutlass.Array,
    ar_uc: cutlass.Array,
    ar_mc: cutlass.Array,
    ar_flags: cutlass.Array,
    logits: cutlass.Array,
    bias: cutlass.Array,
    xin_words: cutlass.Array,
    xq_words: cutlass.Array,
    xsf: cutlass.Array,
    ready: cutlass.Array,
    hflags: cutlass.Array,
    lat_slab: cutlass.Array,
    num_tokens: cutlass.Int32,
    local_offset: cutlass.Int32,
    num_local: cutlass.Int32,
    ar_rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    lat_buf: cutlass.Int32,
    lat_rearm0: cutlass.Int32,
) -> None:
    mma_tiler_mnk = _mma_tiler_mnk
    mma_inst_mnk = _mma_inst_mnk
    num_ab_stage = NUM_AB_STAGE
    rest_k_sf = REST_K_SF
    rest_m_sf = REST_M_SF
    rest_n_sf = REST_N_SF

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane_id = tidx % 32
    dyn_ctr_ptr = state_ptr + cutlass.Int64(ST_CURSOR * 4)

    ab_full_fc1 = cutlass.Array(
        cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem, alignment=8
    )
    ab_full_fc2 = cutlass.Array(
        cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem, alignment=8
    )
    ab_empty = cutlass.Array(
        cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem, alignment=8
    )
    scales_in_tmem = cutlass.Array(
        cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem, alignment=8
    )
    lamport_arrived = cutlass.Array(
        cutlass.Int64, num_ab_stage, space=cutlass.AddressSpace.smem, alignment=8
    )
    lamport_retry = cutlass.Array(
        cutlass.Int64, NUM_LAMPORT_WARPS, space=cutlass.AddressSpace.smem, alignment=8
    )
    dyn_slot = cutlass.Array(cutlass.Int32, DYN_RING, space=cutlass.AddressSpace.smem, alignment=4)
    dyn_ready = cutlass.Array(cutlass.Int64, DYN_RING, space=cutlass.AddressSpace.smem, alignment=8)
    dyn_consumed = cutlass.Array(
        cutlass.Int64, DYN_RING, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # The group -> expert map, the group count and the task count are in shared memory (grouping phase 3):
    # the weights producer starts on this while the other warps finish the pair slots (phases 4-5).
    groups_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if cutlass.const_expr(WIDE):
        # The epilogue finished an epilogue-only task (combine chunk, re-arm): the claimer may take the next.
        epi_free = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    localmax_smem = cutlass.Array(
        cutlass.Float32, 2 * len(epilog_warp_id), space=cutlass.AddressSpace.smem
    )
    epi_flag = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(WIDE):
        # Grouping (same in every CTA): per group its expert and token count; per (group, slot)
        # the pair index token * 16 + k of the slot's token (the routing weight is read from the
        # top-k weights). The token masks and each expert's first group live in the B stages
        # during the prologue (below).
        chunk_cnt = cutlass.Array(cutlass.Int32, 32, space=cutlass.AddressSpace.smem, alignment=16)
        g_expert = cutlass.Array(
            cutlass.Int32, G_CAP, space=cutlass.AddressSpace.smem, alignment=16
        )
        g_cnt = cutlass.Array(cutlass.Int32, G_CAP, space=cutlass.AddressSpace.smem, alignment=16)
        g_pair = cutlass.Array(
            cutlass.Int16, G_CAP * N, space=cutlass.AddressSpace.smem, alignment=16
        )
        # Per FC2 slice, the tokens its groups hold (tokens 0..31 at [2 s], 32..63 at [2 s + 1]): the slice tasks
        # store, and the combine reads, only those tokens' partial rows.
        slice_mask = cutlass.Array(
            cutlass.Int32, 2 * S_CAP, space=cutlass.AddressSpace.smem, alignment=16
        )
    else:
        # Grouping (same in every CTA): emap[e] = this step's token mask of local expert e, then
        # (group << 8) | mask; per group: expert, token count, the 8 slots' token rows (4 bits
        # each, padded with the first token) and routing weights (0 on pads); each token's
        # slots in top-k order; [0] = number of groups.
        emap = cutlass.Array(
            cutlass.Int32, NUM_CHUNKS * 32, space=cutlass.AddressSpace.smem, alignment=16
        )
        chunk_cnt = cutlass.Array(cutlass.Int32, 32, space=cutlass.AddressSpace.smem, alignment=16)
        g_expert = cutlass.Array(
            cutlass.Int32, G_CAP, space=cutlass.AddressSpace.smem, alignment=16
        )
        g_cnt = cutlass.Array(cutlass.Int32, G_CAP, space=cutlass.AddressSpace.smem, alignment=16)
        g_rows = cutlass.Array(cutlass.Int32, G_CAP, space=cutlass.AddressSpace.smem, alignment=16)
        g_rw = cutlass.Array(
            cutlass.Float32, G_CAP * N, space=cutlass.AddressSpace.smem, alignment=16
        )
        s_tok_slots = cutlass.Array(
            cutlass.Int32, ROUTE_PAIRS, space=cutlass.AddressSpace.smem, alignment=16
        )
        # The local pairs compacted in (token, top-k) order: slot | token << 16, for the combine.
        s_pairs = cutlass.Array(
            cutlass.Int32, ROUTE_PAIRS, space=cutlass.AddressSpace.smem, alignment=16
        )
    s_pcnt = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    # [0] groups, [1] all-reduce buffer of this call, [2] local pairs
    s_meta = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    sA = cutlass.Array(
        cutlass.Int8, NUM_BYTES_A * num_ab_stage, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sB = cutlass.Array(
        cutlass.Int8, NUM_BYTES_B * num_ab_stage, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sSFA = cutlass.Array(
        cutlass.Int8, NUM_BYTES_SFA * num_ab_stage, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sSFB = cutlass.Array(
        cutlass.Int8, NUM_BYTES_SFB * num_ab_stage, space=cutlass.AddressSpace.smem, alignment=1024
    )
    if cutlass.const_expr(WIDE):
        # Prologue only, in the B stages (no TMA writes them before the grouping's last barrier):
        # each local expert's token mask, tokens 0..31 at [e] and 32..63 at [MASK_HI + e], and
        # its first group.
        emask = cutlass.Array(
            sB.data_ptr(0), shape=(2 * MASK_HI,), dtype=cutlass.Int32, alignment=16
        )
        e_first = cutlass.Array(
            sB.data_ptr(2 * MASK_HI * 4), shape=(MASK_HI,), dtype=cutlass.Int32, alignment=16
        )
    # This thread's index among the grouping threads (every warp but the claimer).
    gtid = tidx - cutlass.select_(
        warp_idx > producer_weights_warp_id, cutlass.Int32(32), cutlass.Int32(0)
    )
    bias_vals = []
    if cutlass.const_expr(FOLD):
        rs_key = cutlass.Array(
            sA.data_ptr(RS_KEY), shape=(M_MAX * NUM_EXPERTS,), dtype=cutlass.Int32, alignment=16
        )
        rs_sig = cutlass.Array(
            sA.data_ptr(RS_SIG), shape=(M_MAX * NUM_EXPERTS,), dtype=cutlass.Float32, alignment=16
        )
        rs_id = cutlass.Array(
            sA.data_ptr(RS_ID), shape=(M_MAX * TOP_K,), dtype=cutlass.Int32, alignment=16
        )
        rs_w = cutlass.Array(
            sA.data_ptr(RS_W), shape=(M_MAX * TOP_K,), dtype=cutlass.Int32, alignment=16
        )
        # The routing bias is a weight: this thread's experts' values are read before the grid
        # dependency (the claimer warp's reads are in bounds and unused).
        for i in cutlass.range_constexpr(KEY_ITERS):
            e = gtid + cutlass.Int32(i * GROUPING_THREADS)
            bias_vals.append(
                bias.load(idx=cutlass.select_(e < NUM_EXPERTS, e, cutlass.Int32(NUM_EXPERTS - 1)))
            )

    # ------------------------------------------------ prologue, before the grid dependency
    if warp_idx == 0:
        if tidx < num_ab_stage:
            prims.mbarrier_init(ab_full_fc1.subview(tidx), 2 + N)
            prims.mbarrier_init(ab_full_fc2.subview(tidx), 2)
            prims.mbarrier_init(ab_empty.subview(tidx), 1)
            prims.mbarrier_init(scales_in_tmem.subview(tidx), 1)
            prims.mbarrier_init(lamport_arrived.subview(tidx), 1)
        if tidx < NUM_LAMPORT_WARPS:
            prims.mbarrier_init(lamport_retry.subview(tidx), 1)
        if cutlass.const_expr(DYN_ALL):
            if tidx < DYN_RING:
                prims.mbarrier_init(dyn_ready.subview(tidx), 1)
                prims.mbarrier_init(dyn_consumed.subview(tidx), threads_per_cta - 32)
        if tidx == 0:
            prims.mbarrier_init(acc_full.subview(0), 1)
            prims.mbarrier_init(acc_empty.subview(0), EPI_THREADS)
            prims.mbarrier_init(tmem_ready.subview(0), 32)
            prims.mbarrier_init(groups_ready.subview(0), 1)
            if cutlass.const_expr(WIDE):
                prims.mbarrier_init(epi_free.subview(0), 1)
    if cutlass.const_expr(WIDE):
        for j in cutlass.range_constexpr((2 * MASK_HI + threads_per_cta - 1) // threads_per_cta):
            if tidx + j * threads_per_cta < 2 * MASK_HI:
                emask.store(cutlass.Int32(0), idx=tidx + j * threads_per_cta)
        if tidx < 2 * S_CAP:
            slice_mask.store(cutlass.Int32(0), idx=tidx)
    else:
        for j in cutlass.range_constexpr(
            (NUM_CHUNKS * 32 + threads_per_cta - 1) // threads_per_cta
        ):
            if tidx + j * threads_per_cta < NUM_CHUNKS * 32:
                emap.store(cutlass.Int32(0), idx=tidx + j * threads_per_cta)
    if cutlass.const_expr(HEAD_FLAGS):
        # This call's epoch: k3_moe of the previous call advanced it and completed before
        # route_quant_ag (which never writes it) let this grid launch.
        if tidx == 0:
            s_meta.store(hflags.load(idx=2, is_volatile=True), idx=3)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    head_e = cutlass.Int32(0)
    if cutlass.const_expr(HEAD_FLAGS):
        head_e = s_meta.load(idx=3)
    if warp_idx == consumer_warp_id:
        prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_alloc_cols)
        prims.mbarrier_arrive(tmem_ready)
        prims.tcgen05_relinquish_alloc_permit()

    first_raw = cutlass.Int32(0)
    if cutlass.const_expr(HYBRID):
        if warp_idx == producer_weights_warp_id:
            if prims.elect_sync():
                dyn_slot.store(bidx, idx=0)
                prims.mbarrier_arrive(dyn_ready.subview(0))
            cute.arch.sync_warp()
    if cutlass.const_expr(PREWAIT_CLAIM):
        if warp_idx == producer_weights_warp_id:
            first_raw = _claim(
                dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr, cutlass.Int32(PRECLAIM_T)
            )
    if cutlass.const_expr(USE_PDL):
        if cutlass.const_expr(not HEAD_FLAGS):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not LATE_TRIGGER and not HEAD_TRIGGER_READY):
            # Every consumer of this kernel's outputs waits for the whole grid, so the next
            # kernel may launch (and set itself up) now.
            cute.arch.griddepcontrol_launch_dependents()
    # ------------------------------------------------ prologue: first claim + grouping
    if warp_idx == producer_weights_warp_id:
        if cutlass.const_expr(DYN_ALL and not PREWAIT_CLAIM):
            first_raw = _claim(
                dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr, cutlass.Int32(PRECLAIM_T)
            )
    else:
        gw = warp_idx - cutlass.select_(
            warp_idx > producer_weights_warp_id, cutlass.Int32(1), cutlass.Int32(0)
        )
        if cutlass.const_expr(FOLD):
            # (0) every token's selection keys and sigmoids (all logit loads in flight first, rows
            # clamped to M-1), then warp gw < M selects token gw's top-16 and their weights.
            logit = []
            for t in cutlass.range_constexpr(M_MAX):
                row = cutlass.select_(
                    cutlass.Int32(t) < num_tokens, cutlass.Int32(t), num_tokens - cutlass.Int32(1)
                )
                for i in cutlass.range_constexpr(KEY_ITERS):
                    e = gtid + cutlass.Int32(i * GROUPING_THREADS)
                    ec = cutlass.select_(e < NUM_EXPERTS, e, cutlass.Int32(NUM_EXPERTS - 1))
                    logit.append(logits.load(idx=row * NUM_EXPERTS + ec))
            for t in cutlass.range_constexpr(M_MAX):
                if cutlass.Int32(t) < num_tokens:
                    for i in cutlass.range_constexpr(KEY_ITERS):
                        e = gtid + cutlass.Int32(i * GROUPING_THREADS)
                        if e < NUM_EXPERTS:
                            sig = _rq.sigmoid_accurate(logit[t * KEY_ITERS + i])
                            rs_sig.store(sig, idx=t * NUM_EXPERTS + e)
                            rs_key.store(
                                _rq.selection_key(sig + bias_vals[i]), idx=t * NUM_EXPERTS + e
                            )
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            if gw < num_tokens:
                expert, weight_bits = _rq.top16_warp(
                    rs_key.subview(gw * NUM_EXPERTS),
                    rs_sig.subview(gw * NUM_EXPERTS),
                    lane_id,
                    routed_scaling_factor,
                )
                if lane_id < TOP_K:
                    rs_id.store(expert, idx=gw * TOP_K + lane_id)
                    rs_w.store(
                        cutlass.Int32(weight_bits) & cutlass.Int32(0xFFFF), idx=gw * TOP_K + lane_id
                    )
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
        if cutlass.const_expr(HEAD_FLAGS):
            # One lane per token acquires route_quant_ag's ready word (every CTA polling with all its
            # pair threads would hammer 8 L2 words with ~20K threads); the barrier hands the
            # acquired writes to the other grouping threads.
            if warp_idx == 0:
                if lane_id < num_tokens:
                    ready_addr = ready.data_ptr(lane_id).toint()
                    while _load_acquire(ready_addr) != head_e + cutlass.Int32(1):
                        pass
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            if cutlass.const_expr(USE_PDL and HEAD_TRIGGER_READY and not LATE_TRIGGER):
                # The CTA's first launch_dependents by any thread is its trigger.
                cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(FUSED_AR):
            # This call's all-reduce buffer: written by the previous call's last task claim, or with
            # AR_PUSH_ONLY the consumer's call count, whose low bit alternates the halves.
            if tidx == 0:
                s_meta.store(ar_flags.load(idx=0, is_volatile=True) & cutlass.Int32(1), idx=1)
        if cutlass.const_expr(WIDE):
            # (1) each local expert's token mask; this thread's pairs p = gtid + i * GROUPING_THREADS
            # (token p // 16, top-k slot p % 16).
            pair_experts = []
            for i in cutlass.range_constexpr(PAIR_ITERS):
                p = gtid + cutlass.Int32(i * GROUPING_THREADS)
                pt = p // TOP_K
                pe = cutlass.Int32(-1)
                if pt < num_tokens:
                    el = topk_ids.load(idx=p) - local_offset
                    if el >= cutlass.Int32(0):
                        if el < num_local:
                            pe = el
                if pe >= cutlass.Int32(0):
                    _smem_or(
                        emask,
                        pe + (pt >> cutlass.Int32(5)) * cutlass.Int32(MASK_HI),
                        cutlass.Int32(1) << (pt & cutlass.Int32(31)),
                    )
                pair_experts.append(pe)
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            # (2) groups per 32-expert chunk: ceil(tokens / 8) per expert (at most 8: four ballots).
            for j in cutlass.range_constexpr(CHUNKS_PER_WARP):
                c = gw + j * GROUPING_WARPS
                if c < NUM_CHUNKS:
                    x = c * 32 + lane_id
                    cnt = cute.arch.popc(emask.load(idx=x)) + cute.arch.popc(
                        emask.load(idx=x + MASK_HI)
                    )
                    ng = (cnt + cutlass.Int32(N - 1)) // cutlass.Int32(N)
                    chunk_groups = cutlass.Int32(0)
                    for b in cutlass.range_constexpr(4):
                        bal = cute.arch.vote_ballot_sync(
                            ((ng >> cutlass.Int32(b)) & cutlass.Int32(1)) != cutlass.Int32(0)
                        )
                        chunk_groups = chunk_groups + (cute.arch.popc(bal) << cutlass.Int32(b))
                    if lane_id == 0:
                        chunk_cnt.store(chunk_groups, idx=c)
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            # (3) an expert's first group = the groups of the experts before it (ascending id); its
            # groups take its tokens 8 at a time in ascending order.
            for j in cutlass.range_constexpr(CHUNKS_PER_WARP):
                c = gw + j * GROUPING_WARPS
                if c < NUM_CHUNKS:
                    x = c * 32 + lane_id
                    cnt = cute.arch.popc(emask.load(idx=x)) + cute.arch.popc(
                        emask.load(idx=x + MASK_HI)
                    )
                    ng = (cnt + cutlass.Int32(N - 1)) // cutlass.Int32(N)
                    g0 = cutlass.Int32(0)
                    for c2 in cutlass.range_constexpr(NUM_CHUNKS):
                        if cutlass.Int32(c2) < c:
                            g0 = g0 + chunk_cnt.load(idx=c2)
                    for b in cutlass.range_constexpr(4):
                        bal = cute.arch.vote_ballot_sync(
                            ((ng >> cutlass.Int32(b)) & cutlass.Int32(1)) != cutlass.Int32(0)
                        )
                        g0 = g0 + (
                            cute.arch.popc(bal & cutlass.Int32(cute.arch.lanemask_lt()))
                            << cutlass.Int32(b)
                        )
                    e_first.store(g0, idx=x)
                    for q in cutlass.range(ng, unroll=1):
                        rest = cnt - q * cutlass.Int32(N)
                        g_expert.store(x, idx=g0 + q)
                        g_cnt.store(
                            cutlass.select_(rest > cutlass.Int32(N), cutlass.Int32(N), rest),
                            idx=g0 + q,
                        )
            if tidx == 0:
                total = cutlass.Int32(0)
                for c2 in cutlass.range_constexpr(NUM_CHUNKS):
                    total = total + chunk_cnt.load(idx=c2)
                s_meta.store(total, idx=0)
                s_meta.store(_wide_slices(total), idx=2)
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            if cutlass.const_expr(EARLY_RELEASE):
                # Phases 1-3 and thread 0's totals are behind the barrier above; the weights
                # producer needs nothing from phase 4.
                if tidx == 0:
                    prims.mbarrier_arrive(groups_ready)
            # (4) slot of each pair: its expert's first group * N + the rank of its token among
            # the expert's tokens; the slot keeps the pair index.
            for i in cutlass.range_constexpr(PAIR_ITERS):
                pe = pair_experts[i]
                if pe >= cutlass.Int32(0):
                    p = gtid + cutlass.Int32(i * GROUPING_THREADS)
                    pt = p // TOP_K
                    below = (cutlass.Int32(1) << (pt & cutlass.Int32(31))) - cutlass.Int32(1)
                    below_lo = cutlass.select_(pt >= cutlass.Int32(32), cutlass.Int32(-1), below)
                    below_hi = cutlass.select_(pt > cutlass.Int32(32), below, cutlass.Int32(0))
                    rank = cute.arch.popc(emask.load(idx=pe) & below_lo) + cute.arch.popc(
                        emask.load(idx=pe + MASK_HI) & below_hi
                    )
                    first = e_first.load(idx=pe)
                    g_pair.store(cutlass.Int16(p), idx=first * N + rank)
                    # The slice of the pair's group: g in [s G / S, (s + 1) G / S).
                    grp = first + rank // N
                    n_sl = s_meta.load(idx=2)
                    sl_p = ((grp + cutlass.Int32(1)) * n_sl - cutlass.Int32(1)) // s_meta.load(
                        idx=0
                    )
                    _smem_or(
                        slice_mask,
                        sl_p * 2 + (pt >> cutlass.Int32(5)),
                        cutlass.Int32(1) << (pt & cutlass.Int32(31)),
                    )
            # The masks and first groups were written and read through the generic proxy in the B
            # stages; the activation producer's TMA writes (async proxy) follow the barrier below.
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
        else:
            # (1) token bitmask per local expert; this thread's pair (token pt, top-k slot).
            pt = tidx // TOP_K
            pe = cutlass.Int32(-1)
            pw = cutlass.Float32(0.0)
            if tidx < ROUTE_PAIRS:
                if pt < num_tokens:
                    if cutlass.const_expr(FOLD):
                        el = rs_id.load(idx=tidx) - local_offset
                    else:
                        el = topk_ids.load(idx=tidx) - local_offset
                    if el >= cutlass.Int32(0):
                        if el < num_local:
                            pe = el
                            if cutlass.const_expr(FOLD):
                                pw = (rs_w.load(idx=tidx) << cutlass.Int32(16)).bitcast(
                                    cutlass.Float32
                                )
                            else:
                                pw = cutlass.Float32(topk_w.load(idx=tidx))
                if pe >= cutlass.Int32(0):
                    _smem_or(emap, pe, cutlass.Int32(1) << pt)
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            # (2) experts present per 32-expert chunk.
            for j in cutlass.range_constexpr(CHUNKS_PER_WARP):
                c = gw + j * GROUPING_WARPS
                if c < NUM_CHUNKS:
                    m = emap.load(idx=c * 32 + lane_id)
                    bal = cute.arch.vote_ballot_sync(m != cutlass.Int32(0))
                    if lane_id == 0:
                        chunk_cnt.store(cute.arch.popc(bal), idx=c)
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            # (3) group index = rank of the expert among those present (ascending id).
            for j in cutlass.range_constexpr(CHUNKS_PER_WARP):
                c = gw + j * GROUPING_WARPS
                if c < NUM_CHUNKS:
                    x = c * 32 + lane_id
                    m = emap.load(idx=x)
                    bal = cute.arch.vote_ballot_sync(m != cutlass.Int32(0))
                    base = cutlass.Int32(0)
                    for c2 in cutlass.range_constexpr(NUM_CHUNKS):
                        if cutlass.Int32(c2) < c:
                            base = base + chunk_cnt.load(idx=c2)
                    g = base + cute.arch.popc(bal & cutlass.Int32(cute.arch.lanemask_lt()))
                    if m != cutlass.Int32(0):
                        emap.store((g << cutlass.Int32(8)) | m, idx=x)
                        g_expert.store(x, idx=g)
                        cnt = cutlass.Int32(0)
                        rows = cutlass.Int32(0)
                        first = cutlass.Int32(0)
                        for tb in cutlass.range_constexpr(M_MAX):
                            if ((m >> cutlass.Int32(tb)) & cutlass.Int32(1)) != cutlass.Int32(0):
                                rows = rows | (cutlass.Int32(tb) << (cnt * cutlass.Int32(4)))
                                if cnt == cutlass.Int32(0):
                                    first = cutlass.Int32(tb)
                                cnt = cnt + cutlass.Int32(1)
                        for n2 in cutlass.range_constexpr(N):
                            if cutlass.Int32(n2) >= cnt:
                                rows = rows | (first << cutlass.Int32(4 * n2))
                            g_rw.store(cutlass.Float32(0.0), idx=g * N + n2)
                        g_cnt.store(cnt, idx=g)
                        g_rows.store(rows, idx=g)
            if tidx == 0:
                total = cutlass.Int32(0)
                for c2 in cutlass.range_constexpr(NUM_CHUNKS):
                    total = total + chunk_cnt.load(idx=c2)
                s_meta.store(total, idx=0)
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            if cutlass.const_expr(EARLY_RELEASE):
                # Phases 1-3 and thread 0's totals are behind the barrier above; the arrive releases them to the
                # weights producer, which needs nothing from phases 4-5.
                if tidx == 0:
                    prims.mbarrier_arrive(groups_ready)
            # (4) slot of each pair: group * N + rank of the token among the expert's tokens.
            slot = cutlass.Int32(-1)
            pbal = cutlass.Int32(0)
            if tidx < ROUTE_PAIRS:
                if pe >= cutlass.Int32(0):
                    word = emap.load(idx=pe)
                    mask_lt = word & ((cutlass.Int32(1) << pt) - cutlass.Int32(1))
                    slot = (word >> cutlass.Int32(8)) * N + cute.arch.popc(
                        mask_lt & cutlass.Int32(0xFF)
                    )
                    g_rw.store(pw, idx=slot)
                s_tok_slots.store(slot, idx=tidx)
                if cutlass.const_expr(COMBINE == "batched"):
                    pbal = cute.arch.vote_ballot_sync(slot >= cutlass.Int32(0))
                    if lane_id == 0:
                        s_pcnt.store(cute.arch.popc(pbal), idx=warp_idx)
            if cutlass.const_expr(COMBINE == "batched"):
                cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
            # (5) the local pairs, compacted in (token, top-k) order.
            if cutlass.const_expr(COMBINE == "batched"):
                if tidx < ROUTE_PAIRS:
                    pbase = cutlass.Int32(0)
                    ptotal = cutlass.Int32(0)
                    for w2 in cutlass.range_constexpr(ROUTE_PAIRS // 32):
                        c2 = s_pcnt.load(idx=w2)
                        pbase = pbase + cutlass.select_(
                            cutlass.Int32(w2) < warp_idx, c2, cutlass.Int32(0)
                        )
                        ptotal = ptotal + c2
                    if slot >= cutlass.Int32(0):
                        s_pairs.store(
                            slot | (pt << cutlass.Int32(16)),
                            idx=pbase
                            + cute.arch.popc(pbal & cutlass.Int32(cute.arch.lanemask_lt())),
                        )
                    if tidx == 0:
                        s_meta.store(ptotal, idx=2)
    if cutlass.const_expr(not EARLY_RELEASE):
        if cutlass.const_expr(FOLD):
            # The routing scratch in the A stages was written and read through the generic proxy;
            # the ring's TMA writes (async proxy) start after the CTA-wide sync below.
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(0)
    else:
        if warp_idx == producer_weights_warp_id:
            while not cute.arch.mbarrier_test_wait(groups_ready.data_ptr(), 0):
                pass
        else:
            cute.arch.barrier(barrier_id=GROUPING_BAR_ID, number_of_threads=GROUPING_THREADS)
    num_groups = cute.arch.make_warp_uniform(s_meta.load(idx=0))
    ar_cur = cutlass.Int32(0)
    if cutlass.const_expr(FUSED_AR):
        ar_cur = cute.arch.make_warp_uniform(s_meta.load(idx=1))
    tiles_fc1 = M_TILES_FC1 * num_groups
    total_tiles = M_TILES_TOTAL * num_groups
    num_slices = cutlass.Int32(1)
    if cutlass.const_expr(SLICE):
        # FC2 tasks are (slice, m-tile), slice-major, so the first FC2 tasks read the first
        # groups, whose FC1 tiles come first in the queue; at most 32 groups per slice (one
        # lane each when the tasks count their groups' readers; the wide build loops).
        if cutlass.const_expr(WIDE):
            num_slices = _wide_slices(num_groups)
        else:
            num_slices = cutlass.select_(
                num_groups < cutlass.Int32(FC2_SLICES), num_groups, cutlass.Int32(FC2_SLICES)
            )
            min_slices = (num_groups + cutlass.Int32(31)) // cutlass.Int32(32)
            num_slices = cutlass.select_(num_slices < min_slices, min_slices, num_slices)
            num_slices = cutlass.select_(
                num_slices < cutlass.Int32(1), cutlass.Int32(1), num_slices
            )
        total_tiles = cutlass.select_(
            num_groups > cutlass.Int32(0),
            tiles_fc1 + cutlass.Int32(M_TILES_FC2) * num_slices,
            cutlass.Int32(0),
        )
    # Queue order: FC1 tiles, then FC2 tiles (then the slice FC2's combine and re-arm tasks). The
    # all-reduce tasks have their own cursor.
    total_tasks = total_tiles
    n_chunks = 0  # combine tasks per m-tile (chunks of 8 tokens)
    if cutlass.const_expr(SLICE_TASKS):
        if cutlass.const_expr(COMBINE_CHUNKS):
            n_chunks = (num_tokens + cutlass.Int32(N - 1)) // cutlass.Int32(N)
            total_tasks = total_tiles + cutlass.select_(
                num_groups > cutlass.Int32(0),
                cutlass.Int32(M_TILES_FC2) * n_chunks + num_slices,
                cutlass.Int32(0),
            )
        else:
            total_tasks = total_tiles + cutlass.select_(
                num_groups > cutlass.Int32(0),
                cutlass.Int32(M_TILES_FC2 if COMBINE_TASKS else 0) + num_slices,
                cutlass.Int32(0),
            )
    active = total_tasks > 0
    if cutlass.const_expr(DYN_ALL):
        if warp_idx == producer_weights_warp_id:
            _retire_claim(
                first_raw, total_tasks, state_ptr, ar_flags, ar_cur, hflags, head_e, ready,
                num_tokens,
            )  # fmt: skip
    coord_n = 0

    # Fold: the MXFP8 latent, quantized by this CTA into its own scratch rows [8 * bidx, 8 * bidx + M)
    # by the epilogue and Lamport warps (idle until their first tile), all loads in flight first
    # (vectors past M clamped to this thread's first one). The activation producer gathers FC1's B
    # rows and scales from there once QUANT_BAR completes.
    if cutlass.const_expr(FOLD):
        if active and ((warp_idx < len(epilog_warp_id)) | (warp_idx >= lamport_acts_warp_id)):
            qtid = tidx - cutlass.select_(
                warp_idx >= lamport_acts_warp_id,
                cutlass.Int32((lamport_acts_warp_id - len(epilog_warp_id)) * 32),
                cutlass.Int32(0),
            )
            n_vec = num_tokens * VEC8_PER_ROW
            xrow0 = bidx * M_MAX
            vecs = []
            for k in cutlass.range_constexpr(QUANT_ITERS):
                c = qtid + cutlass.Int32(k * QUANT_THREADS)
                cc = cutlass.select_(c < n_vec, c, qtid)
                tok = cc // VEC8_PER_ROW
                vecs.append(
                    xin_words.load(
                        idx=tok * (H // 2) + (cc - tok * VEC8_PER_ROW) * 4,
                        vector_size=4,
                        alignment=16,
                    )
                )
            for k in cutlass.range_constexpr(QUANT_ITERS):
                c = qtid + cutlass.Int32(k * QUANT_THREADS)
                # n_vec and k * QUANT_THREADS are multiples of 64: the condition is warp-uniform, as
                # the scale's 4-lane shuffles need.
                if c < n_vec:
                    tok = c // VEC8_PER_ROW
                    j = c - tok * VEC8_PER_ROW
                    v = vecs[k]
                    q_lo, q_hi, sf_byte = _rq.mxfp8_quant_vec8([v[0], v[1], v[2], v[3]])
                    xq_words.store((q_lo, q_hi), idx=(xrow0 + tok) * (H // 4) + j * 2, alignment=8)
                    if lane_id % 4 == 0:
                        xsf.store(
                            cutlass.Uint8(sf_byte), idx=(xrow0 + tok) * (H // MX_BLOCK) + j // 4
                        )
            # Generic-proxy stores read by the producer's TMA gathers (async proxy).
            prims.fence_proxy("async_global")
            prims.barrier_cta_arrive(QUANT_BAR_ID, QUANT_THREADS + 32)
    # Nothing routed to this rank: its partial is zero.
    if warp_idx < len(epilog_warp_id) and num_groups == 0 and bidx < M_TILES_FC2:
        if cutlass.const_expr(HEAD_FLAGS and USE_PDL):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(FUSED_AR):
            ztok = tidx // 16
            if ztok < num_tokens:
                zero = cutlass.Int32(0)
                ar_mc.store(
                    (zero, zero, zero, zero),
                    idx=_ar_word(ar_cur, ztok, ar_rank, bidx * MMA_M + (tidx % 16) * 8),
                    alignment=16,
                )
        else:
            zch = bidx * MMA_M + tidx
            for t in cutlass.range(num_tokens, unroll=1):
                y_tensor.store(cutlass.Float32(0.0).to(c_dtype), idx=t * H + zch)

    # ------------------------------------------------ producerWeights (4), claimer
    if warp_idx == producer_weights_warp_id and active:
        g = 0
        ab_empty_phase = 1
        t = cutlass.Int32(0)
        dyn_ph = 0
        v = cutlass.select_(bidx < total_tasks, bidx, cutlass.Int32(-1))
        if cutlass.const_expr(SCHED == "dynamic"):
            v = cutlass.select_(first_raw < total_tasks, first_raw, cutlass.Int32(-1))
        in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        while in_fc1:
            m_tile = v % M_TILES_FC1
            group = v // M_TILES_FC1
            coord_m = m_tile * mma_tiler_mnk[0]
            coord_m_sf = coord_m // (num_m0_per_sf_atom * num_m1_per_sf_atom)
            coord_expert = g_expert.load(idx=group)
            for k_tile in cutlass.range(K1_TILES, unroll=1):
                stage = g % num_ab_stage
                if stage == 0 and g != 0:
                    ab_empty_phase = ab_empty_phase ^ 1
                while not cute.arch.mbarrier_try_wait(
                    ab_empty.subview(stage).data_ptr(), ab_empty_phase
                ):
                    pass
                coord_k = k_tile * mma_tiler_mnk[2]
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(
                        ab_full_fc1.subview(stage), NUM_TMA_LOAD_BYTES_WEIGHTS
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sA.subview(NUM_BYTES_A * stage),
                        tma_a1_desc.get_ptr(),
                        (coord_k, coord_m, coord_expert),
                        ab_full_fc1.subview(stage),
                        l2_cache_hint=W_L2_HINT,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sSFA.subview(NUM_BYTES_SFA * stage),
                        tma_sfa1_desc.get_ptr(),
                        (cutlass.Int32(0), k_tile * rest_k_sf, coord_m_sf, coord_expert),
                        ab_full_fc1.subview(stage),
                        l2_cache_hint=W_L2_HINT,
                    )
                g = g + 1
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _claimer_next(
                t, first_raw, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr, bidx, total_tasks,
                state_ptr, ar_flags, ar_cur, hflags, head_e, ready, num_tokens,
            )  # fmt: skip
            in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while in_fc2:
            lin2 = v - tiles_fc1
            m_tile = lin2 % M_TILES_FC2
            g_lo = lin2 // M_TILES_FC2
            g_hi = g_lo + cutlass.Int32(1)
            if cutlass.const_expr(SLICE):
                sl = lin2 // M_TILES_FC2
                if cutlass.const_expr(WIDE):
                    sl = lin2 % num_slices
                    m_tile = lin2 // num_slices
                g_lo = sl * num_groups // num_slices
                g_hi = (sl + cutlass.Int32(1)) * num_groups // num_slices
            coord_m = m_tile * mma_tiler_mnk[0]
            coord_m_sf = coord_m // (num_m0_per_sf_atom * num_m1_per_sf_atom)
            for group in cutlass.range(g_lo, g_hi, unroll=1):
                coord_expert = g_expert.load(idx=group)
                for k_tile in cutlass.range(K2_TILES, unroll=1):
                    stage = g % num_ab_stage
                    if stage == 0 and g != 0:
                        ab_empty_phase = ab_empty_phase ^ 1
                    while not cute.arch.mbarrier_try_wait(
                        ab_empty.subview(stage).data_ptr(), ab_empty_phase
                    ):
                        pass
                    coord_k = k_tile * mma_tiler_mnk[2]
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            ab_full_fc2.subview(stage), NUM_TMA_LOAD_BYTES_WEIGHTS
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sA.subview(NUM_BYTES_A * stage),
                            tma_a2_desc.get_ptr(),
                            (coord_k, coord_m, coord_expert),
                            ab_full_fc2.subview(stage),
                            l2_cache_hint=W_L2_HINT,
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sSFA.subview(NUM_BYTES_SFA * stage),
                            tma_sfa2_desc.get_ptr(),
                            (cutlass.Int32(0), k_tile * rest_k_sf, coord_m_sf, coord_expert),
                            ab_full_fc2.subview(stage),
                            l2_cache_hint=W_L2_HINT,
                        )
                    g = g + 1
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            if cutlass.const_expr(FC2_FIRST_HOLD):
                if t == cutlass.Int32(1):  # this FC2 task was the CTA's first
                    while not cute.arch.mbarrier_try_wait(
                        ab_empty.subview((g - 1) % num_ab_stage).data_ptr(), ab_empty_phase ^ 1
                    ):
                        pass
            v = _claimer_next(
                t, first_raw, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr, bidx, total_tasks,
                state_ptr, ar_flags, ar_cur, hflags, head_e, ready, num_tokens,
            )  # fmt: skip
            in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        t_epi0 = t  # the claim of this CTA's first epilogue-only task
        while v >= cutlass.Int32(0):  # the all-reduce tasks are the epilogue's
            if cutlass.const_expr(WIDE):
                # The combine and re-arm tasks are the epilogue's alone: claim the next one only once
                # the epilogue has finished this one, so that they spread over the CTAs that are free.
                while not cute.arch.mbarrier_try_wait(
                    epi_free.data_ptr(), (t - t_epi0) % cutlass.Int32(2)
                ):
                    pass
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _claimer_next(
                t, first_raw, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr, bidx, total_tasks,
                state_ptr, ar_flags, ar_cur, hflags, head_e, ready, num_tokens,
            )  # fmt: skip

    # ------------------------------------------------ producerActivations (5)
    if warp_idx == producer_acts_warp_id and active:
        lane = tidx % 32
        sfb_lane = cutlass.select_(lane < cutlass.Int32(N), lane, cutlass.Int32(0))
        # Fold: B rows and scales come from this CTA's scratch rows, complete once every quantizing
        # thread has arrived.
        xrow0 = cutlass.Int32(0)
        if cutlass.const_expr(HEAD_FLAGS):
            # route_quant_ag's MXFP8 rows (its writers fenced the async proxy before releasing),
            # acquired by one lane per token.
            if lane < num_tokens:
                qready_addr = ready.data_ptr(lane + cutlass.Int32(M_MAX)).toint()
                while _load_acquire(qready_addr) != head_e + cutlass.Int32(1):
                    pass
            cute.arch.sync_warp()
        if cutlass.const_expr(FOLD):
            xrow0 = bidx * M_MAX
            prims.barrier_cta_sync(QUANT_BAR_ID, thread_count=QUANT_THREADS + 32)
        g = 0
        ab_empty_phase = 1
        t = cutlass.Int32(0)
        dyn_ph = 0
        n_fc1_done = cutlass.Int32(0)
        v = _next_tile(
            t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
        )
        in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        while in_fc1:
            group = v // M_TILES_FC1
            if cutlass.const_expr(WIDE):
                # The slots' token rows (empty slots: row 0).
                grp_cnt = g_cnt.load(idx=group)
                rw = []
                for n in cutlass.range_constexpr(N):
                    pair_n = cutlass.Int32(g_pair.load(idx=group * N + n))
                    rw.append(
                        cutlass.select_(
                            cutlass.Int32(n) < grp_cnt, pair_n // TOP_K, cutlass.Int32(0)
                        )
                    )
                r0, r1, r2, r3, r4, r5, r6, r7 = rw
                sfb_pair = cutlass.Int32(g_pair.load(idx=group * N + sfb_lane))
                sfb_token = cutlass.select_(sfb_lane < grp_cnt, sfb_pair // TOP_K, cutlass.Int32(0))
            else:
                rows = g_rows.load(idx=group)
                r0 = (rows & cutlass.Int32(0xF)) + xrow0
                r1 = ((rows >> cutlass.Int32(4)) & cutlass.Int32(0xF)) + xrow0
                r2 = ((rows >> cutlass.Int32(8)) & cutlass.Int32(0xF)) + xrow0
                r3 = ((rows >> cutlass.Int32(12)) & cutlass.Int32(0xF)) + xrow0
                r4 = ((rows >> cutlass.Int32(16)) & cutlass.Int32(0xF)) + xrow0
                r5 = ((rows >> cutlass.Int32(20)) & cutlass.Int32(0xF)) + xrow0
                r6 = ((rows >> cutlass.Int32(24)) & cutlass.Int32(0xF)) + xrow0
                r7 = ((rows >> cutlass.Int32(28)) & cutlass.Int32(0xF)) + xrow0
                grp_cnt = g_cnt.load(idx=group)
                sfb_token = ((rows >> (sfb_lane * cutlass.Int32(4))) & cutlass.Int32(0xF)) + xrow0
            sfb_cp_sz = cutlass.select_(lane < grp_cnt, cutlass.Int32(4), cutlass.Int32(0))
            for k_tile in cutlass.range(K1_TILES, unroll=1):
                stage = g % num_ab_stage
                if stage == 0 and g != 0:
                    ab_empty_phase = ab_empty_phase ^ 1
                while not cute.arch.mbarrier_try_wait(
                    ab_empty.subview(stage).data_ptr(), ab_empty_phase
                ):
                    pass
                coord_k = k_tile * mma_tiler_mnk[2]
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(ab_full_fc1.subview(stage), NUM_BYTES_B)
                    _tma_gather4_cta(
                        sB.subview(NUM_BYTES_B * stage),
                        tma_b1_desc.get_ptr(),
                        coord_k,
                        r0,
                        r1,
                        r2,
                        r3,
                        ab_full_fc1.subview(stage),
                    )
                    _tma_gather4_cta(
                        sB.subview(NUM_BYTES_B * stage + NUM_BYTES_B // 2),
                        tma_b1_desc.get_ptr(),
                        coord_k,
                        r4,
                        r5,
                        r6,
                        r7,
                        ab_full_fc1.subview(stage),
                    )
                if lane < cutlass.Int32(N):
                    sfb_gmem = (
                        sfb1_ptr
                        + cutlass.Int64(sfb_token) * cutlass.Int64(SFB_SRC_STRIDE_FC1)
                        + cutlass.Int64(k_tile * NUM_KBLOCKS)
                    )
                    sfb_gmem_ir = cutlass.inttoptr(sfb_gmem, mem_space=1, dtype=sf_dtype)
                    sfb_smem_ir = sSFB.subview(
                        NUM_BYTES_SFB * stage + lane * SFB_GROUP_BYTES
                    ).data_ptr()
                    prims.cp_async_shared_global(
                        sfb_smem_ir, sfb_gmem_ir, size=4, modifier="ca", cp_size=sfb_cp_sz
                    )
                    prims.cp_async_mbarrier_arrive(ab_full_fc1.subview(stage), noinc=True)
                g = g + 1
            n_fc1_done = n_fc1_done + cutlass.Int32(1)
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)

        # FC1 -> FC2 activation delay: CTAs that drew fewer FC1 tiles start later.
        if cutlass.const_expr(FC2_DELAY_NS > 0 or FC2_DELAY_LAG_NS > 0):
            if (v >= cutlass.Int32(0)) & (v < total_tiles):
                delay_ns = cutlass.Int64(FC2_DELAY_NS)
                if cutlass.const_expr(FC2_DELAY_LAG_NS > 0):
                    n1_max = (tiles_fc1 + NUM_CTAS - 1) // NUM_CTAS
                    lag = cutlass.select_(
                        n1_max > n_fc1_done, n1_max - n_fc1_done, cutlass.Int32(0)
                    )
                    delay_ns = delay_ns + cutlass.Int64(FC2_DELAY_LAG_NS) * cutlass.Int64(lag)
                t_end = _read_globaltimer() + delay_ns
                while _read_globaltimer() < t_end:
                    _backoff(32, delay=True)

        # counter: the loads complete on the MMA's barrier directly (the scan's arrival is this one).
        acts_fc2_full = lamport_arrived if SCAN_FC2 else ab_full_fc2
        in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while in_fc2:
            lin2 = v - tiles_fc1
            g_lo = lin2 // M_TILES_FC2
            g_hi = g_lo + cutlass.Int32(1)
            if cutlass.const_expr(SLICE):
                sl = lin2 // M_TILES_FC2
                if cutlass.const_expr(WIDE):
                    sl = lin2 % num_slices
                g_lo = sl * num_groups // num_slices
                g_hi = (sl + cutlass.Int32(1)) * num_groups // num_slices
            for group in cutlass.range(g_lo, g_hi, unroll=1):
                if cutlass.const_expr(FC1_COUNTS):
                    # The group's intermediate is complete once its FC1 tiles have all counted
                    # (every lane polls the same word: one request, no divergence).
                    fc1_ctr = state_ptr + cutlass.Int64((ST_FC1 + group) * 4)
                    if cutlass.const_expr(SCAN_FC2):
                        while _load_relaxed(fc1_ctr) < cutlass.Int32(FC2_SYNC_NEED):
                            pass
                    else:
                        while _load_acquire(fc1_ctr) < cutlass.Int32(FC2_SYNC_NEED):
                            pass
                        prims.fence_proxy("async_global")
                for k_tile in cutlass.range(K2_TILES, unroll=1):
                    stage = g % num_ab_stage
                    if stage == 0 and g != 0:
                        ab_empty_phase = ab_empty_phase ^ 1
                    while not cute.arch.mbarrier_try_wait(
                        ab_empty.subview(stage).data_ptr(), ab_empty_phase
                    ):
                        pass
                    coord_k = k_tile * mma_tiler_mnk[2]
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            acts_fc2_full.subview(stage), NUM_TMA_LOAD_BYTES_ACTS_FC2
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sB.subview(NUM_BYTES_B * stage),
                            tma_b2_desc.get_ptr(),
                            (coord_k, coord_n, group),
                            acts_fc2_full.subview(stage),
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sSFB.subview(NUM_BYTES_SFB * stage),
                            tma_sfb2_desc.get_ptr(),
                            (cutlass.Int32(0), k_tile, cutlass.Int32(0), group),
                            acts_fc2_full.subview(stage),
                        )
                    g = g + 1
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while v >= cutlass.Int32(0):  # the all-reduce tasks are the epilogue's
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )

    # ------------------------------------------------ producerScalesTmem (6)
    if warp_idx == scales_tmem_warp_id and active:
        while not cute.arch.mbarrier_try_wait(tmem_ready.data_ptr(), 0):
            pass
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        sfa_cols_per_stage = rest_k_sf * rest_m_sf * num_tmem_cols_per_sf_atom
        sfb_cols_per_stage = rest_k_sf * rest_n_sf * num_tmem_cols_per_sf_atom
        sfa_col_id0 = base_col_id + mma_tiler_mnk[1]
        sfb_col_id0 = sfa_col_id0 + num_ab_stage * sfa_cols_per_stage
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        g = 0
        ab_full_fc1_phase = 0
        t = cutlass.Int32(0)
        dyn_ph = 0
        v = _next_tile(
            t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
        )
        in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        while in_fc1:
            for k_tile in cutlass.range(K1_TILES, unroll=1):
                stage = g % num_ab_stage
                if stage == 0 and g != 0:
                    ab_full_fc1_phase = ab_full_fc1_phase ^ 1
                while not cute.arch.mbarrier_try_wait(
                    ab_full_fc1.subview(stage).data_ptr(), ab_full_fc1_phase
                ):
                    pass
                # The activation scales came by cp.async (generic proxy); tcgen05.cp reads them (async proxy).
                prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                sfa_tmem_ptr = cutlass.inttoptr(
                    (base_row_id << 16) | (sfa_col_id0 + stage * sfa_cols_per_stage),
                    6,
                    cutlass.Int32,
                )
                sfb_tmem_ptr = cutlass.inttoptr(
                    (base_row_id << 16) | (sfb_col_id0 + stage * sfb_cols_per_stage),
                    6,
                    cutlass.Int32,
                )
                desc_a = prims.Tcgen05SmemDesc.build(
                    sSFA.subview(stage * NUM_BYTES_SFA),
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    base_offset=0,
                    layout=0,
                )
                desc_b = prims.Tcgen05SmemDesc.build(
                    sSFB.subview(stage * NUM_BYTES_SFB),
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    base_offset=0,
                    layout=0,
                )
                if prims.elect_sync():
                    prims.tcgen05_cp(s2t_shape, sfa_tmem_ptr, desc_a, multicast=s2t_multicast)
                    prims.tcgen05_cp(s2t_shape, sfb_tmem_ptr, desc_b, multicast=s2t_multicast)
                    prims.tcgen05_commit(scales_in_tmem.subview(stage))
                g = g + 1
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        j = 0
        ab_full_fc2_phase = 0
        in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while in_fc2:
            lin2 = v - tiles_fc1
            g_lo = lin2 // M_TILES_FC2
            g_hi = g_lo + cutlass.Int32(1)
            if cutlass.const_expr(SLICE):
                sl = lin2 // M_TILES_FC2
                if cutlass.const_expr(WIDE):
                    sl = lin2 % num_slices
                g_lo = sl * num_groups // num_slices
                g_hi = (sl + cutlass.Int32(1)) * num_groups // num_slices
            for _group in cutlass.range(g_lo, g_hi, unroll=1):
                for k_tile in cutlass.range(K2_TILES, unroll=1):
                    stage = g % num_ab_stage
                    if j % num_ab_stage == 0 and j != 0:
                        ab_full_fc2_phase = ab_full_fc2_phase ^ 1
                    while not cute.arch.mbarrier_try_wait(
                        ab_full_fc2.subview(stage).data_ptr(), ab_full_fc2_phase
                    ):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    sfa_tmem_ptr = cutlass.inttoptr(
                        (base_row_id << 16) | (sfa_col_id0 + stage * sfa_cols_per_stage),
                        6,
                        cutlass.Int32,
                    )
                    sfb_tmem_ptr = cutlass.inttoptr(
                        (base_row_id << 16) | (sfb_col_id0 + stage * sfb_cols_per_stage),
                        6,
                        cutlass.Int32,
                    )
                    desc_a = prims.Tcgen05SmemDesc.build(
                        sSFA.subview(stage * NUM_BYTES_SFA),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    desc_b = prims.Tcgen05SmemDesc.build(
                        sSFB.subview(stage * NUM_BYTES_SFB),
                        leading_byte_offset=16,
                        stride_byte_offset=128,
                        base_offset=0,
                        layout=0,
                    )
                    if prims.elect_sync():
                        prims.tcgen05_cp(s2t_shape, sfa_tmem_ptr, desc_a, multicast=s2t_multicast)
                        prims.tcgen05_cp(s2t_shape, sfb_tmem_ptr, desc_b, multicast=s2t_multicast)
                        prims.tcgen05_commit(scales_in_tmem.subview(stage))
                    g = g + 1
                    j = j + 1
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while v >= cutlass.Int32(0):  # the all-reduce tasks are the epilogue's
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )

    # ------------------------------------------------ MMA (7)
    if warp_idx == consumer_warp_id and active:
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)
        idesc = prims.Tcgen05MxInstrDesc.build(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            scale_format=1,
            n_dim=mma_tiler_mnk[1],
            m_dim=mma_tiler_mnk[0],
        )
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        sfa_cols_per_stage = rest_k_sf * rest_m_sf * num_tmem_cols_per_sf_atom
        sfb_cols_per_stage = rest_k_sf * rest_n_sf * num_tmem_cols_per_sf_atom
        sfa_col_id0 = base_col_id + mma_tiler_mnk[1]
        sfb_col_id0 = sfa_col_id0 + num_ab_stage * sfa_cols_per_stage
        num_kblocks = mma_tiler_mnk[2] // mma_inst_mnk[2]
        num_sf_ids = num_k_per_sf_atom * sf_vec_size // mma_inst_mnk[2]
        g = 0
        scales_in_tmem_phase = 0
        acc_empty_phase = 1
        t = cutlass.Int32(0)
        dyn_ph = 0
        v = _next_tile(
            t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
        )
        in_tiles = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while in_tiles:
            # FC1 and FC2 tiles share the MMA body; only the k-tile count differs.
            k_tiles = cutlass.select_(
                v < tiles_fc1, cutlass.Int32(K1_TILES), cutlass.Int32(K2_TILES)
            )
            # One accumulation per FC1 tile and per group of an FC2 task (one in tile mode).
            n_acc = cutlass.Int32(1)
            if cutlass.const_expr(SLICE):
                lin2 = v - tiles_fc1
                sl = lin2 // M_TILES_FC2
                if cutlass.const_expr(WIDE):
                    sl = lin2 % num_slices
                n_acc = cutlass.select_(
                    v < tiles_fc1,
                    cutlass.Int32(1),
                    (sl + cutlass.Int32(1)) * num_groups // num_slices
                    - sl * num_groups // num_slices,
                )
            for _acc_i in cutlass.range(n_acc, unroll=1):
                while not cute.arch.mbarrier_try_wait(acc_empty.data_ptr(), acc_empty_phase):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                acc_empty_phase = acc_empty_phase ^ 1
                scale_d = False
                for k_tile in cutlass.range(k_tiles, unroll=1):
                    stage = g % num_ab_stage
                    if stage == 0 and g != 0:
                        scales_in_tmem_phase = scales_in_tmem_phase ^ 1
                    while not cute.arch.mbarrier_try_wait(
                        scales_in_tmem.subview(stage).data_ptr(), scales_in_tmem_phase
                    ):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    sfa_tmem_addr_base = (base_row_id << 16) | (
                        sfa_col_id0 + stage * sfa_cols_per_stage
                    )
                    sfb_tmem_addr_base = (base_row_id << 16) | (
                        sfb_col_id0 + stage * sfb_cols_per_stage
                    )
                    desc_a_mma_base = prims.Tcgen05SmemDesc.build(
                        sA.subview(stage * NUM_BYTES_A),
                        leading_byte_offset=16,
                        stride_byte_offset=1024,
                        base_offset=0,
                        layout=2,
                    )
                    desc_b_mma_base = prims.Tcgen05SmemDesc.build(
                        sB.subview(stage * NUM_BYTES_B),
                        leading_byte_offset=16,
                        stride_byte_offset=1024,
                        base_offset=0,
                        layout=2,
                    )
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        sf_inside = kblock_idx % num_sf_ids
                        sf_col = kblock_idx // num_sf_ids
                        sfa_tmem_ptr = cutlass.inttoptr(
                            sfa_tmem_addr_base + sf_col * NUM_TMEM_COLS_PER_KBLOCK_SFA,
                            6,
                            cutlass.Int32,
                        )
                        sfb_tmem_ptr = cutlass.inttoptr(
                            sfb_tmem_addr_base + sf_col * NUM_TMEM_COLS_PER_KBLOCK_SFB,
                            6,
                            cutlass.Int32,
                        )
                        idesc_u = idesc.set_sf_ids(a_sf_id=sf_inside, b_sf_id=sf_inside)
                        inc = ((mma_inst_mnk[2] * a_smem_width // 8) >> 4) * kblock_idx
                        if prims.elect_sync():
                            prims.tcgen05_mma_block_scale(
                                prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1, acc_tmem_ptr,
                                desc_a_mma_base + inc, desc_b_mma_base + inc, idesc_u, scale_d, sfa_tmem_ptr,
                                sfb_tmem_ptr,
                            )  # fmt: skip
                        scale_d = True
                    if prims.elect_sync():
                        prims.tcgen05_commit(ab_empty.subview(stage))
                    g = g + 1
                if prims.elect_sync():
                    prims.tcgen05_commit(acc_full)
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_tiles = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while v >= cutlass.Int32(0):  # the all-reduce tasks are the epilogue's
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )

    # ------------------------------------------------ lamport (8..11): FC2 only
    if warp_idx >= lamport_acts_warp_id and active:
        lane = tidx % 32
        my_group = warp_idx - cutlass.Int32(lamport_acts_warp_id)
        stages_per_warp = num_ab_stage // NUM_LAMPORT_WARPS
        g = 0
        t = cutlass.Int32(0)
        dyn_ph = 0
        v = _next_tile(
            t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
        )
        in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        while in_fc1:  # FC1 stages need no validation; keep ring position and claim ring in step
            g = g + K1_TILES
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        j = 0
        lam_phase = 0
        retry_phase = 0
        in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while in_fc2:
            lin2 = v - tiles_fc1
            g_lo = lin2 // M_TILES_FC2
            g_hi = g_lo + cutlass.Int32(1)
            if cutlass.const_expr(SLICE):
                sl = lin2 // M_TILES_FC2
                if cutlass.const_expr(WIDE):
                    sl = lin2 % num_slices
                g_lo = sl * num_groups // num_slices
                g_hi = (sl + cutlass.Int32(1)) * num_groups // num_slices
            if cutlass.const_expr(SCAN_FC2):  # counter: the producer acquired the group
                for group in cutlass.range(g_lo, g_hi, unroll=1):
                    for k_tile in cutlass.range(K2_TILES, unroll=1):
                        stage = g % num_ab_stage
                        if j % num_ab_stage == 0 and j != 0:
                            lam_phase = lam_phase ^ 1
                        if (stage // stages_per_warp) == my_group:
                            while not cute.arch.mbarrier_try_wait(
                                lamport_arrived.subview(stage).data_ptr(), lam_phase
                            ):
                                pass
                            coord_k = k_tile * mma_tiler_mnk[2]
                            need_retry = _scan_b_sentinel(sB, stage, lane) | _scan_sfb_sentinel(
                                sSFB, stage, lane
                            )
                            while need_retry:
                                if prims.elect_sync():
                                    prims.mbarrier_arrive_expect_tx(
                                        lamport_retry.subview(my_group), NUM_TMA_LOAD_BYTES_ACTS_FC2
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sB.subview(NUM_BYTES_B * stage),
                                        tma_b2_desc.get_ptr(),
                                        (coord_k, coord_n, group),
                                        lamport_retry.subview(my_group),
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sSFB.subview(NUM_BYTES_SFB * stage),
                                        tma_sfb2_desc.get_ptr(),
                                        (cutlass.Int32(0), k_tile, cutlass.Int32(0), group),
                                        lamport_retry.subview(my_group),
                                    )
                                while not cute.arch.mbarrier_try_wait(
                                    lamport_retry.subview(my_group).data_ptr(), retry_phase
                                ):
                                    pass
                                retry_phase = retry_phase ^ 1
                                need_retry = _scan_b_sentinel(sB, stage, lane) | _scan_sfb_sentinel(
                                    sSFB, stage, lane
                                )
                            if prims.elect_sync():
                                prims.mbarrier_arrive(ab_full_fc2.subview(stage))
                        g = g + 1
                        j = j + 1
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        while v >= cutlass.Int32(0):  # the all-reduce tasks are the epilogue's
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )

    # ------------------------------------------------ epilogue (0-3)
    if warp_idx < len(epilog_warp_id) and (active or (FUSED_AR and not AR_PUSH_ONLY)):
        # With HEAD_FLAGS, FC1 writes only the op's own buffers (c, its scales), so it runs while the
        # producer grid (route_quant_ag, or the MoE front's shared tiles) finishes; the grid wait comes
        # before the FC2 phase, whose combine and all-reduce write the output (the allocator's memory).
        while not cute.arch.mbarrier_try_wait(tmem_ready.data_ptr(), 0):
            pass
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        row_id_with_warp_offset = base_row_id + warp_idx * 32
        t2r_repx = min(32, mma_tiler_mnk[1])
        lane = tidx % 32
        is_up = ((lane // 8) % 2) == 0
        up_mask = cutlass.select_(is_up, cutlass.Float32(1.0), cutlass.Float32(0.0))
        fc1_col_in_tile = warp_idx * 16 + 2 * (lane % 8) + lane // 16
        fc2_ch_in_tile = warp_idx * 32 + 4 * (lane % 8) + lane // 8
        acc_full_phase = 0
        t = cutlass.Int32(0)
        dyn_ph = 0
        v = _next_tile(
            t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
        )
        in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)
        while in_fc1:
            m_tile = v % M_TILES_FC1
            group = v // M_TILES_FC1
            while not cute.arch.mbarrier_try_wait(acc_full.data_ptr(), acc_full_phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc_full_phase = acc_full_phase ^ 1
            tmem_ld = cutlass.inttoptr(
                (row_id_with_warp_offset << 16) | base_col_id, 6, cutlass.Float32
            )
            t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ld, num=t2r_repx)
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.mbarrier_arrive(acc_empty)
            out_col = m_tile * (MMA_M // 2) + fc1_col_in_tile
            block_idx = m_tile * (MMA_M // 2 // MX_BLOCK) + (warp_idx // 2)
            feat_off = (block_idx % SF_TMEM_COL) + (block_idx // SF_TMEM_COL) * (
                SF_TMEM_COL * SF_TMEM_DP
            )
            gbase = group * N
            for n in cutlass.range_constexpr(N):
                xn = cutlass.Float32(t2r_rmem[n])
                partner = cute.arch.shuffle_sync_bfly(xn, 8)
                res = _situ(partner, xn)
                absv = cute.math.abs(res) * up_mask
                warp_amax = prims.redux_sync(absv, prims.ReductionKind.FMAX, 0xFFFFFFFF, abs=True)
                if lane == 0:
                    localmax_smem.store(warp_amax, idx=(n % 2) * 4 + warp_idx)
                cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                partner_amax = localmax_smem.load(idx=(n % 2) * 4 + (warp_idx ^ 1))
                block_amax = cute.arch.fmax(warp_amax, partner_amax)
                byte, inv_scale = _block_e8m0(block_amax)
                qv = cute.arch.fmax(
                    cute.arch.fmin(res * inv_scale, cutlass.Float32(E4M3_MAX)),
                    cutlass.Float32(-E4M3_MAX),
                )
                fp8_i8 = cutlass.Float8E4M3FN(qv).bitcast(cutlass.Int8)
                if fp8_i8 == cutlass.Int8(FP8_SENTINEL_I8):
                    fp8_i8 = cutlass.Int8(0)
                if is_up:
                    c_tensor.store(fp8_i8, idx=(gbase + n) * I_TP + out_col, alignment=1)
                if (tidx % 64) == 0:
                    sc8 = cutlass.Int8(byte & cutlass.Int32(0xFF))
                    if sc8 == cutlass.Int8(SF_SENTINEL_I8):
                        sc8 = cutlass.Int8(0)
                    c_scale_tensor.store(sc8, idx=(gbase + n) * SF_STRIDE0 + feat_off, alignment=1)
            if cutlass.const_expr(FC1_COUNTS):
                fc1_ctr = state_ptr + cutlass.Int64((ST_FC1 + group) * 4)
                if cutlass.const_expr(SCAN_FC2):
                    # hint: counted once every thread has issued its stores; the scan validates.
                    cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                    if tidx == 0:
                        _red_relaxed_add(fc1_ctr, cutlass.Int32(1))
                else:
                    # Other CTAs' TMA loads read this tile's columns once the group's count is
                    # complete: every writer fences the async proxy, thread 0 releases after the barrier.
                    prims.fence_proxy("async_global")
                    cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                    if tidx == 0:
                        _red_release_add(fc1_ctr, cutlass.Int32(1))
            t = t + cutlass.Int32(1)
            if cutlass.const_expr(DYN_ALL):
                if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                    dyn_ph = dyn_ph ^ 1
            v = _next_tile(
                t, False, dyn_ph, bidx, total_tasks, dyn_slot, dyn_ready, dyn_consumed, dyn_ctr_ptr
            )
            in_fc1 = (v >= cutlass.Int32(0)) & (v < tiles_fc1)

        if cutlass.const_expr(HEAD_FLAGS and USE_PDL):
            cute.arch.griddepcontrol_wait()
        in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        if cutlass.const_expr(WIDE):
            zeros = cutlass.Array(cutlass.Float32, M_MAX, space=cutlass.AddressSpace.rmem)
            for i in cutlass.range_constexpr(M_MAX):
                zeros[i] = cutlass.Float32(0.0)
        if cutlass.const_expr(SLICE):
            while in_fc2:
                lin2 = v - tiles_fc1
                m_tile = lin2 % M_TILES_FC2
                sl = lin2 // M_TILES_FC2
                if cutlass.const_expr(WIDE):
                    # FC2 tasks are m-tile-major here: an m-tile's slices finish together, so its combine
                    # chunks run while later m-tiles stream.
                    sl = lin2 % num_slices
                    m_tile = lin2 // num_slices
                g_lo = sl * num_groups // num_slices
                g_hi = (sl + cutlass.Int32(1)) * num_groups // num_slices
                ch = m_tile * MMA_M + fc2_ch_in_tile
                zero = cutlass.Float32(0.0)
                if cutlass.const_expr(WIDE):
                    # Per-token sums of this thread's channel over the slice's groups, in group
                    # order, in TMEM column TOK_COL + token (TOK_COL + M_MAX takes empty slots).
                    tok_col0 = (row_id_with_warp_offset << 16) | (base_col_id + TOK_COL)
                    prims.tcgen05_st(
                        prims.Tcgen05LdStShape.SHAPE_32X32B,
                        cutlass.inttoptr(tok_col0, 6, cutlass.Float32),
                        zeros.load(0, M_MAX, alignment=32),
                    )
                    prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                    for group in cutlass.range(g_lo, g_hi, unroll=1):
                        grp_cnt = g_cnt.load(idx=group)
                        valids = []
                        cols = []
                        rws = []
                        for n in cutlass.range_constexpr(N):
                            valid = cutlass.Int32(n) < grp_cnt
                            pair_n = cutlass.select_(
                                valid,
                                cutlass.Int32(g_pair.load(idx=group * N + n)),
                                cutlass.Int32(0),
                            )
                            valids.append(valid)
                            cols.append(
                                tok_col0
                                + cutlass.select_(valid, pair_n // TOP_K, cutlass.Int32(M_MAX))
                            )
                            rws.append(cutlass.Float32(topk_w.load(idx=pair_n)))
                        while not cute.arch.mbarrier_try_wait(acc_full.data_ptr(), acc_full_phase):
                            pass
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc_full_phase = acc_full_phase ^ 1
                        tmem_ld = cutlass.inttoptr(
                            (row_id_with_warp_offset << 16) | base_col_id, 6, cutlass.Float32
                        )
                        t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ld, num=t2r_repx)
                        sums = [
                            prims.tcgen05_ld(
                                "32x32b", cutlass.inttoptr(cols[n], 6, cutlass.Float32), num=1
                            )
                            for n in range(N)
                        ]
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc_empty)
                        for n in cutlass.range_constexpr(N):
                            val = cutlass.select_(
                                valids[n], cutlass.Float32(t2r_rmem[n]) * rws[n], zero
                            )
                            prims.tcgen05_st(
                                prims.Tcgen05LdStShape.SHAPE_32X32B,
                                cutlass.inttoptr(cols[n], 6, cutlass.Float32),
                                cutlass.Float32(sums[n][0]) + val,
                            )
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                    # This task's partial rows, [m-tile][slice][token][128].
                    accs = prims.tcgen05_ld(
                        "32x32b", cutlass.inttoptr(tok_col0, 6, cutlass.Float32), num=M_MAX
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    pbase = (m_tile * S_CAP + sl) * (M_MAX * MMA_M) + fc2_ch_in_tile
                    mask_lo = slice_mask.load(idx=sl * 2)
                    mask_hi = slice_mask.load(idx=sl * 2 + 1)
                    for tok in cutlass.range_constexpr(M_MAX):
                        word = mask_lo if tok < 32 else mask_hi
                        if ((word >> cutlass.Int32(tok % 32)) & cutlass.Int32(1)) != cutlass.Int32(
                            0
                        ):
                            part_tensor.store(cutlass.Float32(accs[tok]), idx=pbase + tok * MMA_M)
                else:
                    # Per-token sums of this thread's channel over the slice's groups, in group order.
                    a0 = zero
                    a1 = zero
                    a2 = zero
                    a3 = zero
                    a4 = zero
                    a5 = zero
                    a6 = zero
                    a7 = zero
                    for group in cutlass.range(g_lo, g_hi, unroll=1):
                        rows = g_rows.load(idx=group)
                        grp_cnt = g_cnt.load(idx=group)
                        gbase = group * N
                        while not cute.arch.mbarrier_try_wait(acc_full.data_ptr(), acc_full_phase):
                            pass
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        acc_full_phase = acc_full_phase ^ 1
                        tmem_ld = cutlass.inttoptr(
                            (row_id_with_warp_offset << 16) | base_col_id, 6, cutlass.Float32
                        )
                        t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ld, num=t2r_repx)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.mbarrier_arrive(acc_empty)
                        for n in cutlass.range_constexpr(N):
                            valid = cutlass.Int32(n) < grp_cnt
                            val = cutlass.select_(
                                valid, cutlass.Float32(t2r_rmem[n]) * g_rw.load(idx=gbase + n), zero
                            )
                            tok = cutlass.select_(
                                valid,
                                (rows >> cutlass.Int32(4 * n)) & cutlass.Int32(0xF),
                                cutlass.Int32(-1),
                            )
                            a0 = a0 + cutlass.select_(tok == cutlass.Int32(0), val, zero)
                            a1 = a1 + cutlass.select_(tok == cutlass.Int32(1), val, zero)
                            a2 = a2 + cutlass.select_(tok == cutlass.Int32(2), val, zero)
                            a3 = a3 + cutlass.select_(tok == cutlass.Int32(3), val, zero)
                            a4 = a4 + cutlass.select_(tok == cutlass.Int32(4), val, zero)
                            a5 = a5 + cutlass.select_(tok == cutlass.Int32(5), val, zero)
                            a6 = a6 + cutlass.select_(tok == cutlass.Int32(6), val, zero)
                            a7 = a7 + cutlass.select_(tok == cutlass.Int32(7), val, zero)
                    # This task's partial rows, [m-tile][slice][token][128].
                    accs = [a0, a1, a2, a3, a4, a5, a6, a7]
                    pbase = (m_tile * S_CAP + sl) * (M_MAX * MMA_M) + fc2_ch_in_tile
                    for tok in cutlass.range_constexpr(M_MAX):
                        if cutlass.Int32(tok) < num_tokens:
                            part_tensor.store(accs[tok], idx=pbase + tok * MMA_M)
                mtile_ctr = state_ptr + cutlass.Int64((ST_MTILE + m_tile) * 4)
                is_combiner = sl == num_slices - cutlass.Int32(1)
                is_rearmer = m_tile == cutlass.Int32(M_TILES_FC2 - 1)
                if cutlass.const_expr(SLICE_TASKS):
                    is_combiner = cutlass.Boolean(False)
                    is_rearmer = cutlass.Boolean(False)
                # The partial rows are CTA-visible after the barrier; tid 0's release publishes
                # them GPU-wide. The m-tile's last slice combines, after the others' arrivals.
                cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                if cutlass.const_expr(COMBINE_LAST):
                    if tidx == 0:
                        arrived = _atomic_fetch_add(mtile_ctr, cutlass.Int32(1))
                        last_arrival = arrived == num_slices - cutlass.Int32(1)
                        epi_flag.store(
                            cutlass.select_(last_arrival, cutlass.Int32(1), cutlass.Int32(0)), idx=0
                        )
                        if last_arrival:
                            _store_release(mtile_ctr, cutlass.Int32(0))
                else:
                    if tidx == 0:
                        if is_combiner:
                            while _load_acquire(mtile_ctr) < num_slices - cutlass.Int32(1):
                                _backoff(32)
                            _store_release(mtile_ctr, cutlass.Int32(0))
                        else:
                            _red_release_add(mtile_ctr, cutlass.Int32(1))
                # Every group this task read counts one reader (its MMAs are done); the slice's
                # m-tile-27 task re-arms the slice's groups once the other 27 have counted.
                if warp_idx == 1:
                    if cutlass.const_expr(WIDE):
                        for gb in cutlass.range(g_lo, g_hi, 32, unroll=1):
                            if (gb + lane) < g_hi:
                                _red_release_add(
                                    state_ptr + cutlass.Int64((ST_GROUP + gb + lane) * 4),
                                    cutlass.Int32(1),
                                )
                    else:
                        if (g_lo + lane) < g_hi:
                            gctr = state_ptr + cutlass.Int64((ST_GROUP + g_lo + lane) * 4)
                            if is_rearmer:
                                while _load_acquire(gctr) < cutlass.Int32(M_TILES_FC2 - 1):
                                    _backoff(32)
                                _store_release(gctr, cutlass.Int32(0))
                            else:
                                _red_release_add(gctr, cutlass.Int32(1))
                # Only a combiner or a re-armer waits for its spinning thread(s); every other task
                # moves on to its next accumulator.
                if cutlass.const_expr(COMBINE_LAST):
                    cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                    is_combiner = epi_flag.load(idx=0) != cutlass.Int32(0)
                else:
                    if is_combiner | is_rearmer:
                        cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                if cutlass.const_expr(not WIDE):  # the wide build combines in chunk tasks
                    if is_combiner:
                        # The slices' partials summed in slice order.
                        cute.arch.fence_acq_rel_gpu()
                        cbase = m_tile * S_CAP * (M_MAX * MMA_M) + fc2_ch_in_tile
                        c0 = zero
                        c1 = zero
                        c2 = zero
                        c3 = zero
                        c4 = zero
                        c5 = zero
                        c6 = zero
                        c7 = zero
                        # 8 slices x 8 tokens of loads in flight per round.
                        for sb in cutlass.range(0, num_slices, 8, unroll=1):
                            for q in cutlass.range_constexpr(8):
                                sq = sb + cutlass.Int32(q)
                                ok = sq < num_slices
                                qbase = cbase + cutlass.select_(ok, sq, cutlass.Int32(0)) * (
                                    M_MAX * MMA_M
                                )
                                vq = [
                                    part_tensor.load(idx=qbase + tk * MMA_M, is_volatile=True)
                                    for tk in range(M_MAX)
                                ]
                                c0 = c0 + cutlass.select_(ok, vq[0], zero)
                                c1 = c1 + cutlass.select_(ok, vq[1], zero)
                                c2 = c2 + cutlass.select_(ok, vq[2], zero)
                                c3 = c3 + cutlass.select_(ok, vq[3], zero)
                                c4 = c4 + cutlass.select_(ok, vq[4], zero)
                                c5 = c5 + cutlass.select_(ok, vq[5], zero)
                                c6 = c6 + cutlass.select_(ok, vq[6], zero)
                                c7 = c7 + cutlass.select_(ok, vq[7], zero)
                        cs = [c0, c1, c2, c3, c4, c5, c6, c7]
                        for tok in cutlass.range_constexpr(M_MAX):
                            if cutlass.Int32(tok) < num_tokens:
                                _emit_row(
                                    cs[tok], cutlass.Int32(tok), ch, lane, warp_idx, m_tile, y_tensor, ar_mc,
                                    ar_cur, ar_rank,
                                )  # fmt: skip
                if is_rearmer:
                    for b in cutlass.range(g_hi - g_lo, unroll=1):
                        _rearm_group(c_words, cs_words, g_lo + b, tidx)
                t = t + cutlass.Int32(1)
                if cutlass.const_expr(DYN_ALL):
                    if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                        dyn_ph = dyn_ph ^ 1
                v = _next_tile(
                    t,
                    False,
                    dyn_ph,
                    bidx,
                    total_tasks,
                    dyn_slot,
                    dyn_ready,
                    dyn_consumed,
                    dyn_ctr_ptr,
                )
                in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
            if cutlass.const_expr(SLICE_TASKS):
                zero = cutlass.Float32(0.0)
                if cutlass.const_expr(not WIDE):
                    # Combine tasks: an m-tile's slices summed in slice order once all have published.
                    in_comb = (v >= total_tiles) & (v < total_tiles + cutlass.Int32(M_TILES_FC2))
                    if cutlass.const_expr(not COMBINE_TASKS):
                        in_comb = cutlass.Boolean(False)
                    while in_comb:
                        m_tile = v - total_tiles
                        mtile_ctr = state_ptr + cutlass.Int64((ST_MTILE + m_tile) * 4)
                        if tidx == 0:
                            while _load_acquire(mtile_ctr) < num_slices:
                                _backoff(32)
                            _store_release(mtile_ctr, cutlass.Int32(0))
                        cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                        cute.arch.fence_acq_rel_gpu()
                        ch = m_tile * MMA_M + fc2_ch_in_tile
                        cbase = m_tile * S_CAP * (M_MAX * MMA_M) + fc2_ch_in_tile
                        c0 = zero
                        c1 = zero
                        c2 = zero
                        c3 = zero
                        c4 = zero
                        c5 = zero
                        c6 = zero
                        c7 = zero
                        for sb in cutlass.range(0, num_slices, 8, unroll=1):
                            for q in cutlass.range_constexpr(8):
                                sq = sb + cutlass.Int32(q)
                                ok = sq < num_slices
                                qbase = cbase + cutlass.select_(ok, sq, cutlass.Int32(0)) * (
                                    M_MAX * MMA_M
                                )
                                vq = [
                                    part_tensor.load(idx=qbase + tk * MMA_M, is_volatile=True)
                                    for tk in range(M_MAX)
                                ]
                                c0 = c0 + cutlass.select_(ok, vq[0], zero)
                                c1 = c1 + cutlass.select_(ok, vq[1], zero)
                                c2 = c2 + cutlass.select_(ok, vq[2], zero)
                                c3 = c3 + cutlass.select_(ok, vq[3], zero)
                                c4 = c4 + cutlass.select_(ok, vq[4], zero)
                                c5 = c5 + cutlass.select_(ok, vq[5], zero)
                                c6 = c6 + cutlass.select_(ok, vq[6], zero)
                                c7 = c7 + cutlass.select_(ok, vq[7], zero)
                        cs = [c0, c1, c2, c3, c4, c5, c6, c7]
                        for tok in cutlass.range_constexpr(M_MAX):
                            if cutlass.Int32(tok) < num_tokens:
                                _emit_row(
                                    cs[tok], cutlass.Int32(tok), ch, lane, warp_idx, m_tile, y_tensor, ar_mc,
                                    ar_cur, ar_rank,
                                )  # fmt: skip
                        t = t + cutlass.Int32(1)
                        if cutlass.const_expr(DYN_ALL):
                            if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                                dyn_ph = dyn_ph ^ 1
                        v = _next_tile(
                            t,
                            False,
                            dyn_ph,
                            bidx,
                            total_tasks,
                            dyn_slot,
                            dyn_ready,
                            dyn_consumed,
                            dyn_ctr_ptr,
                        )
                        in_comb = (v >= total_tiles) & (
                            v < total_tiles + cutlass.Int32(M_TILES_FC2)
                        )
                if cutlass.const_expr(COMBINE_CHUNKS):
                    # Combine tasks, one per (m-tile, 8 tokens): the m-tile's slices summed in slice
                    # order once all have published; the m-tile's counter then counts the chunks that
                    # have read it, and the last one resets it.
                    n_comb = cutlass.Int32(M_TILES_FC2) * n_chunks
                    in_chunk = (v >= total_tiles) & (v < total_tiles + n_comb)
                    while in_chunk:
                        cidx = v - total_tiles
                        m_tile = cidx // n_chunks
                        tok0 = (cidx - m_tile * n_chunks) * cutlass.Int32(N)
                        mtile_ctr = state_ptr + cutlass.Int64((ST_MTILE + m_tile) * 4)
                        if tidx == 0:
                            while _load_acquire(mtile_ctr) < num_slices:
                                _backoff(32)
                        cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                        cute.arch.fence_acq_rel_gpu()
                        ch = m_tile * MMA_M + fc2_ch_in_tile
                        cbase = m_tile * S_CAP * (M_MAX * MMA_M) + tok0 * MMA_M + fc2_ch_in_tile
                        c0 = zero
                        c1 = zero
                        c2 = zero
                        c3 = zero
                        c4 = zero
                        c5 = zero
                        c6 = zero
                        c7 = zero
                        mword = tok0 >> cutlass.Int32(5)
                        mshift = tok0 & cutlass.Int32(31)
                        # 8 slices x 8 tokens of loads in flight per round; a slice without one of these
                        # tokens stored nothing for it, and it adds zero (as the stored zero would).
                        for sb in cutlass.range(0, num_slices, 8, unroll=1):
                            vals = []
                            for q in cutlass.range_constexpr(8):
                                sq = sb + cutlass.Int32(q)
                                ok = sq < num_slices
                                sq_c = cutlass.select_(ok, sq, cutlass.Int32(0))
                                bits = cutlass.select_(
                                    ok,
                                    (slice_mask.load(idx=sq_c * 2 + mword) >> mshift)
                                    & cutlass.Int32(0xFF),
                                    cutlass.Int32(0),
                                )
                                qbase = cbase + sq_c * (M_MAX * MMA_M)
                                for tk in cutlass.range_constexpr(N):
                                    pv = zero
                                    if (
                                        (bits >> cutlass.Int32(tk)) & cutlass.Int32(1)
                                    ) != cutlass.Int32(0):
                                        pv = part_tensor.load(
                                            idx=qbase + tk * MMA_M, is_volatile=True
                                        )
                                    vals.append(pv)
                            for q in cutlass.range_constexpr(8):
                                c0 = c0 + vals[q * N + 0]
                                c1 = c1 + vals[q * N + 1]
                                c2 = c2 + vals[q * N + 2]
                                c3 = c3 + vals[q * N + 3]
                                c4 = c4 + vals[q * N + 4]
                                c5 = c5 + vals[q * N + 5]
                                c6 = c6 + vals[q * N + 6]
                                c7 = c7 + vals[q * N + 7]
                        cs = [c0, c1, c2, c3, c4, c5, c6, c7]
                        for k in cutlass.range_constexpr(N):
                            if tok0 + cutlass.Int32(k) < num_tokens:
                                _emit_row(
                                    cs[k], tok0 + cutlass.Int32(k), ch, lane, warp_idx, m_tile,
                                    y_tensor, ar_mc, ar_cur, ar_rank,
                                )  # fmt: skip
                        if tidx == 0:
                            seen = _atomic_fetch_add(mtile_ctr, cutlass.Int32(1))
                            if seen == num_slices + n_chunks - cutlass.Int32(1):
                                _store_release(mtile_ctr, cutlass.Int32(0))
                            prims.mbarrier_arrive(epi_free)
                        t = t + cutlass.Int32(1)
                        if cutlass.const_expr(DYN_ALL):
                            if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                                dyn_ph = dyn_ph ^ 1
                        v = _next_tile(
                            t,
                            False,
                            dyn_ph,
                            bidx,
                            total_tasks,
                            dyn_slot,
                            dyn_ready,
                            dyn_consumed,
                            dyn_ctr_ptr,
                        )
                        in_chunk = (v >= total_tiles) & (v < total_tiles + n_comb)
                # Re-arm tasks: a slice's groups re-armed once all 28 of their readers have counted
                # (and their FC1 counts reset; counter: only the counts, nothing scans the intermediate).
                if cutlass.const_expr(COMBINE_CHUNKS):
                    rearm0 = total_tiles + cutlass.Int32(M_TILES_FC2) * n_chunks
                else:
                    rearm0 = total_tiles + cutlass.Int32(M_TILES_FC2 if COMBINE_TASKS else 0)
                in_rearm = v >= rearm0
                while in_rearm:
                    sl = v - rearm0
                    g_lo = sl * num_groups // num_slices
                    g_hi = (sl + cutlass.Int32(1)) * num_groups // num_slices
                    if warp_idx == 1:
                        if cutlass.const_expr(WIDE):
                            for gb in cutlass.range(g_lo, g_hi, 32, unroll=1):
                                if (gb + lane) < g_hi:
                                    gctr = state_ptr + cutlass.Int64((ST_GROUP + gb + lane) * 4)
                                    while _load_acquire(gctr) < cutlass.Int32(M_TILES_FC2):
                                        _backoff(32)
                                    _store_release(gctr, cutlass.Int32(0))
                                    _store_release(
                                        state_ptr + cutlass.Int64((ST_FC1 + gb + lane) * 4),
                                        cutlass.Int32(0),
                                    )
                        else:
                            if (g_lo + lane) < g_hi:
                                gctr = state_ptr + cutlass.Int64((ST_GROUP + g_lo + lane) * 4)
                                while _load_acquire(gctr) < cutlass.Int32(M_TILES_FC2):
                                    _backoff(32)
                                _store_release(gctr, cutlass.Int32(0))
                                if cutlass.const_expr(FC1_COUNTS):
                                    _store_release(
                                        state_ptr + cutlass.Int64((ST_FC1 + g_lo + lane) * 4),
                                        cutlass.Int32(0),
                                    )
                    if cutlass.const_expr(SCAN_FC2):
                        cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                        for b in cutlass.range(g_hi - g_lo, unroll=1):
                            _rearm_group(c_words, cs_words, g_lo + b, tidx)
                    if cutlass.const_expr(WIDE):
                        cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                        if tidx == 0:
                            prims.mbarrier_arrive(epi_free)
                    t = t + cutlass.Int32(1)
                    if cutlass.const_expr(DYN_ALL):
                        if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                            dyn_ph = dyn_ph ^ 1
                    v = _next_tile(
                        t,
                        False,
                        dyn_ph,
                        bidx,
                        total_tasks,
                        dyn_slot,
                        dyn_ready,
                        dyn_consumed,
                        dyn_ctr_ptr,
                    )
                    in_rearm = v >= rearm0
        else:
            while in_fc2:
                lin2 = v - tiles_fc1
                m_tile = lin2 % M_TILES_FC2
                group = lin2 // M_TILES_FC2
                grp_cnt = g_cnt.load(idx=group)
                gbase = group * N
                ch = m_tile * MMA_M + fc2_ch_in_tile
                while not cute.arch.mbarrier_try_wait(acc_full.data_ptr(), acc_full_phase):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                acc_full_phase = acc_full_phase ^ 1
                tmem_ld = cutlass.inttoptr(
                    (row_id_with_warp_offset << 16) | base_col_id, 6, cutlass.Float32
                )
                t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ld, num=t2r_repx)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.mbarrier_arrive(acc_empty)
                for n in cutlass.range_constexpr(N):
                    if n < grp_cnt:
                        rw = g_rw.load(idx=gbase + n)
                        part_tensor.store(
                            cutlass.Float32(t2r_rmem[n]) * rw, idx=(gbase + n) * H + ch
                        )
                mtile_ctr = state_ptr + cutlass.Int64((ST_MTILE + m_tile) * 4)
                group_ctr = state_ptr + cutlass.Int64((ST_GROUP + group) * 4)
                if cutlass.const_expr(EPI_SYNC == "last"):
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                    if tidx == 0:
                        # Both round trips in flight at once.
                        arrived = _atomic_fetch_add(mtile_ctr, cutlass.Int32(1))
                        readers = _atomic_fetch_add(group_ctr, cutlass.Int32(1))
                        epi_flag.store(
                            cutlass.select_(
                                arrived == num_groups - 1, cutlass.Int32(1), cutlass.Int32(0)
                            ),
                            idx=0,
                        )
                        epi_flag.store(
                            cutlass.select_(
                                readers == M_TILES_FC2 - 1, cutlass.Int32(1), cutlass.Int32(0)
                            ),
                            idx=1,
                        )
                        if cutlass.const_expr(EPI_RESET == "before"):
                            if arrived == num_groups - 1:
                                _store_release(mtile_ctr, cutlass.Int32(0))
                            if readers == M_TILES_FC2 - 1:
                                _store_release(group_ctr, cutlass.Int32(0))
                else:
                    # This tile's partial rows are CTA-visible after the barrier; tid 0's release
                    # publishes them GPU-wide.
                    cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                    if tidx == 0:
                        if group != num_groups - 1:
                            _red_release_add(mtile_ctr, cutlass.Int32(1))
                        if m_tile != M_TILES_FC2 - 1:
                            _red_release_add(group_ctr, cutlass.Int32(1))
                        if group == num_groups - 1:
                            while _load_acquire(mtile_ctr) < num_groups - 1:
                                _backoff(32)
                            _store_release(mtile_ctr, cutlass.Int32(0))
                        if m_tile == M_TILES_FC2 - 1:
                            while _load_acquire(group_ctr) < cutlass.Int32(M_TILES_FC2 - 1):
                                _backoff(32)
                            _store_release(group_ctr, cutlass.Int32(0))
                        epi_flag.store(
                            cutlass.select_(
                                group == num_groups - 1, cutlass.Int32(1), cutlass.Int32(0)
                            ),
                            idx=0,
                        )
                        epi_flag.store(
                            cutlass.select_(
                                m_tile == M_TILES_FC2 - 1, cutlass.Int32(1), cutlass.Int32(0)
                            ),
                            idx=1,
                        )
                cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
                if epi_flag.load(idx=1) != cutlass.Int32(0):
                    # Every FC2 tile of this group has read its intermediate: re-arm it.
                    _rearm_group(c_words, cs_words, group, tidx)
                    if cutlass.const_expr(EPI_SYNC == "last" and EPI_RESET == "after"):
                        if tidx == 0:
                            _store_release(group_ctr, cutlass.Int32(0))
                if epi_flag.load(idx=0) != cutlass.Int32(0):
                    if cutlass.const_expr(EPI_SYNC == "last" and EPI_RESET == "after"):
                        if tidx == 0:
                            _store_release(mtile_ctr, cutlass.Int32(0))
                    cute.arch.fence_acq_rel_gpu()
                    if cutlass.const_expr(COMBINE == "loop"):
                        for tok in cutlass.range(num_tokens, unroll=1):
                            acc = cutlass.Float32(0.0)
                            for s in cutlass.range_constexpr(TOP_K):
                                slot = s_tok_slots.load(idx=tok * TOP_K + s)
                                if slot >= cutlass.Int32(0):
                                    acc = acc + part_tensor.load(
                                        idx=slot * H + ch, is_volatile=True
                                    )
                            _emit_row(
                                acc,
                                tok,
                                ch,
                                lane,
                                warp_idx,
                                m_tile,
                                y_tensor,
                                ar_mc,
                                ar_cur,
                                ar_rank,
                            )
                    else:
                        # Every token's slots in top-k order; the loads of PAIR_CHUNK pairs (any
                        # tokens) are in flight together.
                        n_pairs = s_meta.load(idx=2)
                        acc0 = cutlass.Float32(0.0)
                        acc1 = cutlass.Float32(0.0)
                        acc2 = cutlass.Float32(0.0)
                        acc3 = cutlass.Float32(0.0)
                        acc4 = cutlass.Float32(0.0)
                        acc5 = cutlass.Float32(0.0)
                        acc6 = cutlass.Float32(0.0)
                        acc7 = cutlass.Float32(0.0)
                        for pb in cutlass.range(0, n_pairs, PAIR_CHUNK, unroll=1):
                            vals = []
                            toks = []
                            for q in cutlass.range_constexpr(PAIR_CHUNK):
                                pq = pb + q
                                valid = pq < n_pairs
                                packed = s_pairs.load(
                                    idx=cutlass.select_(valid, pq, cutlass.Int32(0))
                                )
                                part_v = part_tensor.load(
                                    idx=(packed & cutlass.Int32(0xFFFF)) * H + ch, is_volatile=True
                                )
                                vals.append(cutlass.select_(valid, part_v, cutlass.Float32(0.0)))
                                toks.append(
                                    cutlass.select_(
                                        valid, packed >> cutlass.Int32(16), cutlass.Int32(-1)
                                    )
                                )
                            for q in cutlass.range_constexpr(PAIR_CHUNK):
                                zero = cutlass.Float32(0.0)
                                acc0 = acc0 + cutlass.select_(
                                    toks[q] == cutlass.Int32(0), vals[q], zero
                                )
                                acc1 = acc1 + cutlass.select_(
                                    toks[q] == cutlass.Int32(1), vals[q], zero
                                )
                                acc2 = acc2 + cutlass.select_(
                                    toks[q] == cutlass.Int32(2), vals[q], zero
                                )
                                acc3 = acc3 + cutlass.select_(
                                    toks[q] == cutlass.Int32(3), vals[q], zero
                                )
                                acc4 = acc4 + cutlass.select_(
                                    toks[q] == cutlass.Int32(4), vals[q], zero
                                )
                                acc5 = acc5 + cutlass.select_(
                                    toks[q] == cutlass.Int32(5), vals[q], zero
                                )
                                acc6 = acc6 + cutlass.select_(
                                    toks[q] == cutlass.Int32(6), vals[q], zero
                                )
                                acc7 = acc7 + cutlass.select_(
                                    toks[q] == cutlass.Int32(7), vals[q], zero
                                )
                        accs = [acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7]
                        for tok in cutlass.range_constexpr(M_MAX):
                            if cutlass.Int32(tok) < num_tokens:
                                _emit_row(
                                    accs[tok], cutlass.Int32(tok), ch, lane, warp_idx, m_tile, y_tensor, ar_mc,
                                    ar_cur, ar_rank,
                                )  # fmt: skip
                t = t + cutlass.Int32(1)
                if cutlass.const_expr(DYN_ALL):
                    if t % cutlass.Int32(DYN_RING) == cutlass.Int32(0):
                        dyn_ph = dyn_ph ^ 1
                v = _next_tile(
                    t,
                    False,
                    dyn_ph,
                    bidx,
                    total_tasks,
                    dyn_slot,
                    dyn_ready,
                    dyn_consumed,
                    dyn_ctr_ptr,
                )
                in_fc2 = (v >= cutlass.Int32(0)) & (v < total_tiles)
        if cutlass.const_expr(FUSED_AR and not AR_PUSH_ONLY):
            # No tile is left for this CTA and the queue is empty, so every push of this rank
            # is issued or in a running CTA's epilogue. Reduction tasks, one at a time: an
            # idle CTA takes the next only when it is free again.
            if cutlass.const_expr(LAT_SLAB):
                # E5: re-arm the next buffer (and buffer 0 at the step's last call); this grid's wait returned.
                slab_words = M_MAX * (H // 2)
                ones = cutlass.Int32(LAT_SLAB_EMPTY)
                nxt = (lat_buf + cutlass.Int32(1)) % cutlass.Int32(LAT_SLAB_BUFS)
                for w in range(bidx * EPI_THREADS + tidx, slab_words, NUM_CTAS * EPI_THREADS):
                    lat_slab.store(ones, idx=nxt * cutlass.Int32(slab_words) + w)
                    if lat_rearm0 != cutlass.Int32(0):
                        lat_slab.store(ones, idx=w)
            task = _claim_ar_task(state_ptr, epi_flag, tidx, ar_flags, ar_cur)
            while task < cutlass.Int32(AR_TASKS):
                _ar_reduce_tile(
                    ar_uc, y_words, ar_cur, task, tidx, num_tokens, ar_rank, lat_slab, lat_buf
                )
                task = _claim_ar_task(state_ptr, epi_flag, tidx, ar_flags, ar_cur)
    # ------------------------------------------------ teardown
    # Every role is done: MMAs committed and waited on, TMEM loads waited on.
    if cutlass.const_expr(HEAD_FLAGS and USE_PDL):
        cute.arch.griddepcontrol_wait()
    prims.barrier_cta_sync(0)
    if warp_idx == consumer_warp_id:
        tmem_raw_addr = tmem_ptr_i32.load()
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32), num_tmem_alloc_cols
        )
