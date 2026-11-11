# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5 sparse-MLA FP8 prefill FMHA for SM103a.

This is the dedicated prefill/context counterpart of the GLM sparse-MLA
generation kernel.  It implements the model specialization H16, QK576
(latent512 + RoPE64), V512, top-k 2048, FP8 E4M3 Q/KV, and BF16 output with
one persistent 512-thread CTA per query token.  The schedule follows the
TensorRT-LLM Q16 tactic's warp-specialized QK, online-softmax, PV, correction,
gather, and Cluster Launch Control roles.

The four gather warps fill an eleven-slot shared K/V ring.  Each logical tile
is loaded in RoPE/L0/L1/L2/L3 order, allowing the next RoPE panel to advance
as soon as QK releases its slot.  Latent panels remain live until both QK and
PV finish.  RoPE transfers only 8 KiB at a 16-KiB slot base; each latent panel
transfers 16 KiB.  The kernel also uses hardware min/max instructions for the
softmax fragment reduction, a 256-bit global sparse-index load followed by
two 128-bit shared stores, and 256-bit global output stores.

Compile-time and runtime choices
--------------------------------
H16, QK576, V512, top-k 2048, the 128-row KV tile, pipeline depths, warp
roles, CTA/cluster size, PDL mode, and the invalid-index specialization are
compile-time choices.  ``total_q``, KV-pool rows, tensor addresses, the two
device BMM scale arrays, per-query ``cu_seqlen_ks``/``cu_seqlen_ke`` bounds,
and the CUDA stream are runtime values.  ``total_q`` and bounds capacity are
symbolic and are not compile-cache keys, so one binary covers arbitrary
positive query counts.  A different model shape or sparse top-k must fall
back; there is no runtime dispatcher inside this module.

Three further compile-time modes share the kernel body: ``dense_block_table``
(the sparse rows are computed from the paged block table instead of being
loaded) and, on top of it, ``dense_causal`` (``dense_causal_groups``): one
CTA per group of four consecutive query tokens of one request, MMA N = 64
(token, head) columns, K/V tiles of 128 consecutive positions loaded as
32-row paged blocks with plain TMA tile loads, and a per-token causal mask.
Its grid is the group count, not ``total_q``.  All three modes produce
bitwise-identical output for the dense causal index image; the sparse
device code is unchanged by the dense modes (every difference is a
``const_expr`` branch).

Direct FMHA-core ABI
--------------------
``prepare_runner`` consumes post-preprocessing tensors, not the original
BF16 Q from ``Fmha.forward``:

* contiguous FP8 ``quant_q_buffer[total_q,16,576]``;
* a contiguous flattened, layer-adjusted physical FP8 KV-pool view whose row
  size is 576;
* 32-byte-aligned contiguous physical
  ``int32 sparse_attn_indices[total_q,2048]``;
* contiguous CUDA ``int32 cu_seqlen_ks[capacity]`` and
  ``cu_seqlen_ke[capacity]``, with equal capacities at least ``total_q``;
* 32-byte-aligned contiguous BF16 output storage for
  ``total_q * 16 * 512`` values; and
* contiguous CUDA FP32 ``mla_bmm1_scale[2]`` and ``mla_bmm2_scale[1]``.

For query ``q``, the valid-prefix length is
``clamp(cu_seqlen_ke[q] - cu_seqlen_ks[q], 0, 2048)``.  The first
``valid_count`` sparse entries must be in-range physical pool rows, followed
by a contiguous ``-1`` suffix.  Interior holes are outside the ABI.  A zero
length row produces zero output without entering the QK/PV tile pipelines.
The invalid-safe specialization is the production default.  Select the
all-valid specialization only when the caller proves that all 2048 entries in
every row are valid.  Legacy ``cu_q_seqlens`` and ``sequence_length`` values
remain shape-validated for adapter compatibility but are not device inputs.

``mla_bmm1_scale[1]`` is the runtime base-2 softmax scale
``log2(e) / (q_scaling * sqrt(576))``; ``mla_bmm2_scale[0]`` is the runtime
output scale.  The FP8 probability operand is multiplied by 448 before E4M3
conversion, matching TensorRT-LLM.  Invalid positions are masked before
max/exp, so they contribute neither probability nor value.  The gather can
route their addresses to a safe physical row without zeroing that row in
shared memory.

This is a source-level semantic ABI, not TensorRT-LLM's binary
``KernelParams`` ABI.  A compiled CuTe DSL artifact cannot replace the native cubin
while receiving an unchanged ``KernelParams*``.

TensorRT-LLM integration
------------------------
The adapter serves only SM103 context-only GLM DSA requests with this exact
shape, FP8 KV cache, block-size-one sparse rows, BF16 output, and no chunked
prefill, cache reuse, speculative context, or CP-Helix.  It preserves this
order:

1. run ``mla_rope_append_paged_kv_assign_q`` to rotate Q and append KV;
2. run ``invokeMLAContextFp8Quantize`` to populate a non-aliasing FP8 Q
   workspace and both runtime scale arrays;
3. obtain the layer-adjusted physical KV view and global sparse indices from
   the existing DSA local-to-global conversion;
4. pass DSA's existing graph-stable ``cu_seqlen_ks`` and ``cu_seqlen_ke``
   capacity buffers directly to ``prepare_runner`` without slicing,
   copying, or host-reading them; and
5. invoke the prepared runner from the adapter's ``forward`` method.

The adapter adds no preprocessing launch or host synchronization for bounds:
DSA already produces both buffers on device.  After RoPE/KV mutation, any
kernel failure must be fatal rather than falling back and repeating
preprocessing.

Precompile the invalid-safe specialization and allocate all workspaces before
CUDA Graph capture.  A prepared runner binds addresses but refreshes the
current stream on each eager call/capture; rebuild it when any bound allocation
changes.  Graph replay reuses the captured kernel node, so every Q/KV/index/
output/scale/bounds address must remain stable and bounds contents must be
updated at the same address.  Cross-stream producers require an event wait.
Output must not overlap Q, KV, indices, bounds, or scales.
The FP8 Q workspace must not overlap either runtime scale array.

PDL has exactly one predecessor wait, executed by W9 before reading Q,
indices, bounds, or scales.  W9 publishes those values to the other roles via
CTA barriers.  ``LAUNCH_DEPENDENTS`` executes only after all persistent roles
reconverge and all output stores complete.

Performance
-----------
On B300 at the target shape the kernel is within about 1% of the native
TensorRT-LLM kernel.
"""

import math
from dataclasses import dataclass

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.experimental import primitives as prims

IO_DTYPE = cutlass.Float8E4M3FN
OUT_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32


NUM_HEADS: int = 16
L: int = 512
R: int = 64
TOPK: int = 2048
KV_STRIDE: int = L + R
FP8_P_SCALE: float = 448.0
LOG2_FP8_P_SCALE: float = math.log2(FP8_P_SCALE)

# A 16x256b TMEM load groups the H16 softmax statistics by fragment column:
# [0, 1, 8, 9, 2, 3, 10, 11, ...]. The PV accumulator is instead in logical
# head order. Keep online-softmax state packed, then apply this inverse
# permutation exactly once at the correction handoff.
H16_LOGICAL_TO_PACKED_STATS: tuple[int, ...] = (
    0,
    1,
    4,
    5,
    8,
    9,
    12,
    13,
    2,
    3,
    6,
    7,
    10,
    11,
    14,
    15,
)

VALID_ROWS: int = NUM_HEADS

QK_N: int = 128
QK_K: int = 128
SCORE_N: int = NUM_HEADS
PV_M: int = 128
PV_N: int = NUM_HEADS
N_V: int = L // PV_M

CTA_N: int = QK_N
N_LAT: int = L // QK_K
EB: int = 1

LBO_K: int = 16
SBO_LAT: int = 8 * QK_K * EB
SBO_ROPE: int = 8 * R * EB
LBO_P: int = 16
SBO_P: int = 8 * QK_N * EB
V_GROUP: int = 128
LBO_V: int = (QK_N // 2) * V_GROUP * EB
SBO_V: int = 8 * V_GROUP * EB

# TMEM columns 0-31 hold two score stages; columns 32-95 hold the V512 output.
TMEM_O_OFF: int = 32
NUM_TMEM_COLS: int = 128

# The first four K128 slices are the same 512 latent values consumed as V.
# Eleven 16-KiB addresses hold two complete five-panel tiles plus one panel of
# lookahead into the following tile. RoPE uses only the base 8 KiB of its
# address; latent panels retain the 128-byte swizzle.
K_HDIM_STAGES: int = 5
V_HDIM_STAGES: int = 4
ST_KV: int = 2 * K_HDIM_STAGES + 1
ST_I: int = 8
ST_S: int = 2
ST_P: int = 2
ST_C: int = 2

W_CORR0: int = 4
W_MMA_QK: int = 8
W_LOAD_KQ: int = 9
W_MMA_PV: int = 11
W_LOAD_GATHER0: int = 12
NUM_GATHER_WARPS: int = 4
THREADS: int = 16 * 32

BAR_SM0: int = 1  # g0 intra-group (128 thr)
BAR_CORR: int = 2
KV_STAGE_BYTES: int = QK_N * QK_K * EB
KVR_STAGE_BYTES: int = QK_N * R * EB
CLC_STAGES: int = 2
SCHEDULER_WARP: int = 10
# Four gather warps plus QK, PV, four softmax warps, and four correction warps
# read each two-stage work slot. The producer may reuse a slot only after all
# 14 consumers have retired its active-tile and invalid-count metadata.
WORK_SLOT_CONSUMERS: int = 14

# ---------------------------------------------------------------------------
# dense_causal mode constants.  Every use is under ``cutlass.const_expr(dense_causal)``
# or a Constexpr select, so the sparse/dense_block_table device code is
# unchanged.
# ---------------------------------------------------------------------------
# One work item is a group of G_TOK consecutive query tokens of one request
# (tail groups hold 1..3 real tokens; the remaining columns are dummies that
# copy the last real token's visibility and are never stored).  Because the
# tile's token dimension is the MMA N (= TMEM columns), the score
# tile is S^T[128 KV rows x N_DC (token, head) columns] with column
# ``n = t * 16 + head`` (token-major: the Q TMA box's middle dimension is the
# 16 heads), and the output accumulator is O^T[128 d x N_DC] per V stripe.
G_TOK: int = 4
N_DC: int = G_TOK * NUM_HEADS
# TMEM: two score stages at columns [stage * 64, +64) = 0..127, the four O^T
# stripes at 128 + s * 64 = 128..383; the allocation is the full 512 columns.
# G_TOK = 8 would need 4 x 128 = 512 columns for O^T alone.
TMEM_O_OFF_DC: int = ST_S * N_DC
NUM_TMEM_COLS_DC: int = 512
# K/V tile kt of a group = positions [kt * 128, kt * 128 + 128) = paged
# blocks kt * 4 + rank; each producer warp loads its 32-row block of every
# panel with one 2-D TMA tile load (box [inner, 32]).  32-row boxes at 4 KiB
# (latent) / 2 KiB (RoPE) offsets keep the 128-B / 64-B swizzle phase, so the
# smem image and the dk/dkr/dv descriptors are those of the sparse path.
KV_BOX_ROWS_DC: int = 32
# Eight 16-KiB ring slots = one complete five-panel tile plus three panels of
# look-ahead; nine do not fit the 227-KiB budget.
ST_KV_DC: int = 8
# Q is single-staged (N_DC rows x 576 B = 36 KiB).  q_empty is released by the
# last QK commit of a group, well before its epilogue, so the next group's Q
# load still overlaps the current group's softmax/PV tail.
Q_STAGES_DC: int = 1
Q_BYTES_DC: int = N_DC * KV_STRIDE * EB
# Per-work-slot metadata record published by W9 through index_full/index_empty:
# {q0, n_tok, req, num_kt, ke_0, ke_1, ke_2, ke_3} (int32).
GROUP_META_WORDS: int = 8


def _mma_idesc(n: int, a_mn_major: bool) -> int:
    """Return the tcgen05 ``kind::f8f6f4`` instruction descriptor for M=128, N=``n``.

    Bit 4: D is F32; bits 7-12 zero: A and B are E4M3; bit 15: A is MN-major
    (the PV MMA reads V^T); bits 17-22: N >> 3; bit 27: M >> 4 = 8 (M = 128).
    ``_mma_idesc(16, False) == 0x8040010`` and ``_mma_idesc(16, True) ==
    0x8048010`` are the literals the sparse path has always emitted;
    dense_causal uses N = 64 (``0x8100010`` / ``0x8108010``).
    """
    if n % 8 or not 16 <= n <= 256:
        raise ValueError(f"tcgen05 M=128 needs N a multiple of 8 in [16, 256], got {n}")
    return 0x8000010 | ((n >> 3) << 17) | (0x8000 if a_mn_major else 0)


@cute.jit
def _launch(
    q_storage: cute.Tensor,
    kv_storage: cute.Tensor,
    sparse_indices: cute.Tensor,
    cu_seqlen_ks: cute.Tensor,
    cu_seqlen_ke: cute.Tensor,
    output: cute.Tensor,
    mla_bmm1_scale: cute.Tensor,
    mla_bmm2_scale: cute.Tensor,
    req_idx: cute.Tensor,
    block_table: cute.Tensor,
    group_table: cute.Tensor,
    stride_factor: cutlass.Int32,
    layer_offset: cutlass.Int32,
    stream: cuda_driver.CUstream,
    all_sparse_indices_valid: cutlass.Constexpr[bool],
    dense_block_table: cutlass.Constexpr[bool],
    dense_causal: cutlass.Constexpr[bool],
) -> None:
    # ``box_rows`` is the number of query tokens (Q) or KV-pool rows (K/V) per
    # TMA box: 1 on the sparse path, G_TOK / KV_BOX_ROWS_DC in dense_causal
    # mode.
    def q_tensor_map(offset, inner, box_inner, swizzle, box_rows):
        return cuda.create_tensor_map_tiled(
            global_address=(q_storage.iterator + offset).toint(),
            dtype=IO_DTYPE,
            global_dims=[inner, NUM_HEADS, q_storage.shape[0]],
            global_strides=[
                KV_STRIDE * EB // 16,
                NUM_HEADS * KV_STRIDE * EB // 16,
            ],
            box_dims=[box_inner, NUM_HEADS, box_rows],
            swizzle=swizzle,
        )

    def kv_tensor_map(offset, inner, box_inner, swizzle, box_rows):
        return cuda.create_tensor_map_tiled(
            global_address=(kv_storage.iterator + offset).toint(),
            dtype=IO_DTYPE,
            global_dims=[inner, kv_storage.shape[0]],
            global_strides=[KV_STRIDE * EB // 16],
            box_dims=[box_inner, box_rows],
            swizzle=swizzle,
        )

    if cutlass.const_expr(dense_causal):
        # Dense-causal: a [128, 16, 4] / [64, 16, 4] Q box lands as 64 smem
        # rows ``n = t * 16 + head`` (the MMA B rows / TMEM columns), and the
        # K/V maps load one 32-row paged block per producer warp.  The grid is
        # one CTA per group-table row; the CLC scheduler is unchanged.
        tma_ql = q_tensor_map(0, L, QK_K, cuda.TensorMapSwizzle.s128b, G_TOK)
        tma_qr = q_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b, G_TOK)
        tma_kv = kv_tensor_map(0, KV_STRIDE, QK_K, cuda.TensorMapSwizzle.s128b, KV_BOX_ROWS_DC)
        tma_kvr = kv_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b, KV_BOX_ROWS_DC)
        num_work_items = group_table.shape[0]
    else:
        tma_ql = q_tensor_map(0, L, QK_K, cuda.TensorMapSwizzle.s128b, 1)
        tma_qr = q_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b, 1)
        tma_kv = kv_tensor_map(0, KV_STRIDE, QK_K, cuda.TensorMapSwizzle.s128b, 1)
        tma_kvr = kv_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b, 1)
        num_work_items = q_storage.shape[0]
    glm52_self_attn_fmha_sparse_mla_prefill_kernel(
        tma_ql,
        tma_qr,
        tma_kv,
        tma_kvr,
        sparse_indices,
        cu_seqlen_ks,
        cu_seqlen_ke,
        output,
        mla_bmm1_scale,
        mla_bmm2_scale,
        req_idx,
        block_table,
        group_table,
        stride_factor,
        layer_offset,
        all_sparse_indices_valid,
        dense_block_table,
        dense_causal,
    ).launch(
        grid=(num_work_items, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=True,
    )


_COMPILE_CACHE: dict[tuple[int, bool, bool, bool], object] = {}


def _device_index(tensor: torch.Tensor) -> int:
    device_index = tensor.device.index
    return device_index if device_index is not None else torch.cuda.current_device()


def _current_stream(device_index: int) -> cuda_driver.CUstream:
    return cuda_driver.CUstream(torch.cuda.current_stream(device_index).cuda_stream)


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two nonempty contiguous tensors overlap in device memory."""
    left_begin = left.data_ptr()
    left_end = left_begin + left.numel() * left.element_size()
    right_begin = right.data_ptr()
    right_end = right_begin + right.numel() * right.element_size()
    return left_begin < right_end and right_begin < left_end


def _fake_tensors() -> tuple[object, ...]:
    make_fake = cute.runtime.make_fake_compact_tensor
    total_q = cute.sym_int64()
    pool_rows = cute.sym_int64()
    bounds_capacity = cute.sym_int64()
    return (
        make_fake(
            cutlass.Uint8,
            (total_q, NUM_HEADS, KV_STRIDE),
            stride_order=(2, 1, 0),
            assumed_align=16,
        ),
        make_fake(
            cutlass.Uint8,
            (pool_rows, KV_STRIDE),
            stride_order=(1, 0),
            assumed_align=16,
        ),
        # The kernel addresses the sparse rows through a raw pointer
        # (query * 2048 + slot), so the row count is its own symbol: the
        # dense_block_table specialization binds a one-row placeholder here.
        make_fake(
            cutlass.Int32,
            (cute.sym_int64(), TOPK),
            stride_order=(1, 0),
            assumed_align=32,
        ),
        make_fake(
            cutlass.Int32,
            (bounds_capacity,),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake(
            cutlass.Int32,
            (bounds_capacity,),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake(
            OUT_DTYPE,
            (total_q, NUM_HEADS, L),
            stride_order=(2, 1, 0),
            assumed_align=32,
        ),
        make_fake(cutlass.Float32, (2,), stride_order=(0,), assumed_align=4),
        make_fake(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4),
        # dense_block_table mode operands: per-query request id and the paged
        # block table [num_requests, max_blocks_per_request]; unread otherwise.
        # Both carry their own symbols: the kernel reads req_idx[query] through
        # a raw pointer, and the route hands over a capacity slice whose length
        # is not the bounds capacity (a one-element placeholder in sparse mode).
        make_fake(
            cutlass.Int32,
            (cute.sym_int64(),),
            stride_order=(0,),
            assumed_align=4,
        ),
        make_fake(
            cutlass.Int32,
            (cute.sym_int64(), cute.sym_int64()),
            stride_order=(1, 0),
            assumed_align=4,
        ),
        # dense_causal mode operand: the group table ``[G, 2] = (q0, n_tok)``
        # its row count is the grid and its own symbol.
        # A one-row placeholder in the other modes.
        make_fake(
            cutlass.Int32,
            (cute.sym_int64(), 2),
            stride_order=(1, 0),
            assumed_align=4,
        ),
    )


def compile(  # noqa: A001
    *,
    all_sparse_indices_valid: bool = False,
    dense_block_table: bool = False,
    dense_causal: bool = False,
    device_index: int = 0,
) -> object:
    """Compile one symbolic-shape prefill specialization.

    :param all_sparse_indices_valid: Select the specialization that omits
        invalid-tail masking, defaults to False.
    :type all_sparse_indices_valid: bool, optional
    :param dense_block_table: Select the specialization whose W9 producer
        computes each sparse slot's physical pool row from the paged block
        table (``block_table[req_idx[q], pos >> 5] * stride_factor +
        (pos & 31) + layer_offset`` with ``pos = cu_seqlen_ks[q] + slot``)
        instead of loading it from ``sparse_attn_indices``; the rows a
        dense causal index tensor would hold, so the output is bitwise the
        index path's. Defaults to False.
    :type dense_block_table: bool, optional
    :param dense_causal: Select the dense-causal specialization: one CTA per
        group of ``G_TOK`` = 4 consecutive query tokens of one request
        (``group_table`` rows), MMA N = 64, K/V tiles of 128 consecutive
        positions loaded as 32-row paged blocks through the block table, and a
        per-token causal mask.  Requires ``dense_block_table=True`` (same
        operands) and ``all_sparse_indices_valid=False`` (the diagonal tile
        always needs the mask).  Output is bitwise the sparse path's for the
        dense causal index image with ``cu_seqlen_ks = 0``.  Defaults to
        False.
    :type dense_causal: bool, optional
    :param device_index: CUDA device on which to initialize compilation,
        defaults to 0.
    :type device_index: int, optional
    :raises ValueError: If ``dense_causal`` is combined with
        ``all_sparse_indices_valid`` or without ``dense_block_table``.
    :return: Cached compiled launcher for the requested validity mode.
    :rtype: object
    """
    if dense_causal:
        if not dense_block_table:
            raise ValueError("dense_causal requires dense_block_table")
        if all_sparse_indices_valid:
            raise ValueError(
                "dense_causal always masks the diagonal tile; "
                "all_sparse_indices_valid must be False"
            )
    key = (device_index, all_sparse_indices_valid, dense_block_table, dense_causal)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutlass.cuda.initialize_cuda_context(device_index)
    compiled = cute.compile(
        _launch,
        *_fake_tensors(),
        cutlass.Int32(1),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        all_sparse_indices_valid,
        dense_block_table,
        dense_causal,
        options="--enable-tvm-ffi --opt-level 2",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


@dataclass(frozen=True)
class PreparedMlaPrefillRunner:
    """Bind runtime tensors while refreshing the CUDA stream for every call.

    :param compiled: Compiled symbolic-shape launcher.
    :type compiled: object
    :param args: Runtime tensors in the compiled launch ABI order.
    :type args: tuple[torch.Tensor, ...]
    :param device_index: CUDA device containing all bound tensors.
    :type device_index: int
    """

    compiled: object
    args: tuple[object, ...]
    device_index: int

    def __call__(self) -> None:
        """Launch once on the device's current CUDA stream."""
        stream = _current_stream(self.device_index)
        self.compiled(*self.args, stream)


_PLACEHOLDERS: dict[tuple[int, str], torch.Tensor] = {}


def _placeholder(device: torch.device, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Return a cached zero int32 tensor that an unselected mode never reads."""
    key = (device.index, name)
    tensor = _PLACEHOLDERS.get(key)
    if tensor is None:
        tensor = torch.zeros(shape, dtype=torch.int32, device=device)
        _PLACEHOLDERS[key] = tensor
    return tensor


def prepare_runner(
    quant_q_buffer: torch.Tensor,
    kv_pool_view: torch.Tensor,
    sparse_attn_indices: torch.Tensor | None,
    output: torch.Tensor,
    cu_q_seqlens: torch.Tensor | None,
    sequence_length: torch.Tensor | None,
    *,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    mla_bmm1_scale: torch.Tensor,
    mla_bmm2_scale: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor | None = None,
    all_sparse_indices_valid: bool = False,
    block_table: torch.Tensor | None = None,
    req_idx: torch.Tensor | None = None,
    stride_factor: int | None = None,
    layer_offset: int = 0,
    dense_causal_groups: torch.Tensor | None = None,
) -> PreparedMlaPrefillRunner:
    """Validate and bind the TensorRT-LLM sparse-prefill semantic ABI.

    The sparse rows must already use physical KV-pool indices.  For query
    ``q``, the default specialization consumes the first
    ``clamp(cu_seqlen_ke[q] - cu_seqlen_ks[q], 0, 2048)`` entries.  That
    prefix must be valid and any remaining entries must be ``-1`` padding.
    Both bounds tensors may be graph-stable capacity allocations larger than
    ``total_q``; the runner neither slices nor host-reads them.

    Dense block-table mode (``block_table`` given): no index tensor is read.
    Slot ``s`` of query ``q`` is the position ``cu_seqlen_ks[q] + s`` of
    request ``req_idx[q]`` and maps to the physical row
    ``block_table[req_idx[q], pos >> 5] * stride_factor + (pos & 31)
    + layer_offset`` (``-1`` when ``s`` is past the window, the block is past
    ``block_table.shape[1]`` or the table entry is negative), i.e. exactly the
    ``req_index_to_global`` image of the dense-causal index tensor DSA builds
    when it skips top-k.  The output is bitwise identical to the sparse path
    fed with that tensor.

    Dense-causal mode (``dense_causal_groups`` given, implies block-table
    mode): one CTA per group of up to ``G_TOK`` = 4 consecutive query tokens
    of one request (:func:`dense_causal_groups` builds the table).  Token
    ``q`` attends positions ``0 .. min(cu_seqlen_ke[q], 2048) - 1`` of request
    ``req_idx[q]``; ``cu_seqlen_ks`` must be zero and is not read (see
    :func:`dense_causal_bounds`).  The block table must cover every position
    below the group's largest ``cu_seqlen_ke`` (in-window ``-1`` entries are
    outside this ABI).  The output is bitwise identical to the two other
    modes on the same inputs.  ``all_sparse_indices_valid`` must be False.

    :param quant_q_buffer: Contiguous FP8 Q tensor shaped
        ``[total_q, 16, 576]``.
    :type quant_q_buffer: torch.Tensor
    :param kv_pool_view: Contiguous FP8 physical KV-pool storage with rows of
        576 elements.
    :type kv_pool_view: torch.Tensor
    :param sparse_attn_indices: Contiguous, 32-byte-aligned physical sparse
        rows shaped ``[total_q, 2048]``; None only in dense block-table mode.
    :type sparse_attn_indices: torch.Tensor, optional
    :param output: Caller-owned, 32-byte-aligned BF16 output storage.
    :type output: torch.Tensor
    :param cu_q_seqlens: Legacy CUDA int32 cumulative-Q metadata, or None
        together with ``sequence_length``.
    :type cu_q_seqlens: torch.Tensor, optional
    :param sequence_length: Legacy CUDA int32 request lengths, or None
        together with ``cu_q_seqlens``.
    :type sequence_length: torch.Tensor, optional
    :param cu_seqlen_ks: CUDA int32 per-query sparse-window starts.
    :type cu_seqlen_ks: torch.Tensor
    :param cu_seqlen_ke: CUDA int32 per-query sparse-window ends.
    :type cu_seqlen_ke: torch.Tensor
    :param mla_bmm1_scale: CUDA FP32 BMM1 scale array with two elements.
    :type mla_bmm1_scale: torch.Tensor
    :param mla_bmm2_scale: CUDA FP32 BMM2 scale array with one element.
    :type mla_bmm2_scale: torch.Tensor
    :param host_past_key_value_lengths: Optional CPU int32 request metadata,
        defaults to None. The kernel does not read it.
    :type host_past_key_value_lengths: torch.Tensor, optional
    :param all_sparse_indices_valid: Omit invalid-tail masking only when every
        sparse row is known to contain 2048 valid entries, defaults to False.
    :type all_sparse_indices_valid: bool, optional
    :param block_table: CUDA int32 paged block table
        ``[num_requests, max_blocks_per_request]`` selecting dense block-table
        mode, defaults to None (sparse index mode).
    :type block_table: torch.Tensor, optional
    :param req_idx: CUDA int32 per-query request row of ``block_table``
        (capacity at least ``total_q``); required with ``block_table``.
    :type req_idx: torch.Tensor, optional
    :param stride_factor: Physical rows per block-table entry; required with
        ``block_table``.
    :type stride_factor: int, optional
    :param layer_offset: Row offset of this layer inside a block entry,
        defaults to 0.
    :type layer_offset: int, optional
    :param dense_causal_groups: int32 ``[G, 2]`` group table ``(q0, n_tok)``
        selecting dense-causal mode, defaults to None.  A CPU table is
        validated (``q0 + n_tok <= total_q``, ``1 <= n_tok <= 4``) and
        uploaded to the Q device; a CUDA table is bound as is (no host read,
        like the bounds tensors).
    :type dense_causal_groups: torch.Tensor, optional
    :raises TypeError: If a tensor dtype does not match the semantic ABI.
    :raises ValueError: If shape, layout, device, alignment, capacity, or
        ownership validation fails.
    :raises RuntimeError: If the selected device is not SM103.
    :return: Address-bound runner that refreshes the current stream per call.
    :rtype: PreparedMlaPrefillRunner
    """
    if quant_q_buffer.dtype == torch.float8_e4m3fn:
        q_storage = quant_q_buffer.view(torch.uint8)
    elif quant_q_buffer.dtype == torch.uint8:
        q_storage = quant_q_buffer
    else:
        raise TypeError("quant_q_buffer must be FP8 E4M3 or uint8 storage")
    if kv_pool_view.dtype == torch.float8_e4m3fn:
        kv_storage = kv_pool_view.view(torch.uint8)
    elif kv_pool_view.dtype == torch.uint8:
        kv_storage = kv_pool_view
    else:
        raise TypeError("kv_pool_view must be FP8 E4M3 or uint8 storage")

    if q_storage.ndim != 3 or tuple(q_storage.shape[1:]) != (
        NUM_HEADS,
        KV_STRIDE,
    ):
        raise ValueError(f"quant_q_buffer must have shape [total_q, {NUM_HEADS}, {KV_STRIDE}]")
    total_q = int(q_storage.shape[0])
    if total_q <= 0:
        raise ValueError("quant_q_buffer must contain at least one query token")
    if kv_storage.numel() % KV_STRIDE:
        raise ValueError("kv_pool_view has an invalid flattened row stride")
    pool_rows = kv_storage.numel() // KV_STRIDE
    if pool_rows <= 0:
        raise ValueError("kv_pool_view must contain at least one physical row")
    dense_block_table = block_table is not None
    dense_causal = dense_causal_groups is not None
    if dense_causal:
        # Same operands as block-table mode, never with
        # the all-valid specialization (the diagonal tile needs the mask).
        if not dense_block_table:
            raise ValueError("dense_causal_groups requires block_table mode")
        if all_sparse_indices_valid:
            raise ValueError("dense_causal_groups cannot be combined with all_sparse_indices_valid")
        if dense_causal_groups.dtype != torch.int32:
            raise TypeError("dense_causal_groups must be int32")
        if dense_causal_groups.ndim != 2 or dense_causal_groups.shape[1] != 2:
            raise ValueError("dense_causal_groups must be [num_groups, 2] = (q0, n_tok)")
        if dense_causal_groups.shape[0] <= 0:
            raise ValueError("dense_causal_groups must contain at least one group")
        if not dense_causal_groups.is_contiguous():
            raise ValueError("dense_causal_groups must be contiguous")
        if dense_causal_groups.device.type == "cpu":
            # Host-side content check on the CPU table (no device
            # synchronization): every group must address real query rows and
            # hold 1..G_TOK tokens.
            q0 = dense_causal_groups[:, 0].to(torch.int64)
            n_tok = dense_causal_groups[:, 1].to(torch.int64)
            if int(n_tok.min()) < 1 or int(n_tok.max()) > G_TOK:
                raise ValueError(f"dense_causal_groups n_tok must be in [1, {G_TOK}]")
            if int(q0.min()) < 0 or int((q0 + n_tok).max()) > total_q:
                raise ValueError("dense_causal_groups q0 + n_tok must not exceed total_q")
            group_table = dense_causal_groups.to(q_storage.device)
        else:
            group_table = dense_causal_groups
    else:
        group_table = _placeholder(q_storage.device, "group_table", (1, 2))
    if dense_block_table:
        if req_idx is None or stride_factor is None:
            raise ValueError("dense block-table mode needs req_idx and stride_factor")
        if block_table.dtype != torch.int32 or req_idx.dtype != torch.int32:
            raise TypeError("block_table and req_idx must be int32")
        if block_table.ndim != 2 or block_table.shape[1] <= 0:
            raise ValueError("block_table must be [num_requests, max_blocks_per_request]")
        if req_idx.ndim != 1 or req_idx.numel() < total_q:
            raise ValueError("req_idx must be rank one with capacity at least total_q")
        if int(stride_factor) <= 0:
            raise ValueError("stride_factor must be positive")
        if int(layer_offset) < 0:
            raise ValueError("layer_offset must be non-negative")
        if sparse_attn_indices is None:
            sparse_attn_indices = _placeholder(q_storage.device, "sparse", (1, TOPK))
    else:
        if req_idx is not None or stride_factor is not None or layer_offset:
            raise ValueError("req_idx, stride_factor and layer_offset require block_table")
        if sparse_attn_indices is None:
            raise ValueError("sparse_attn_indices is required without block_table")
        if sparse_attn_indices.dtype != torch.int32 or tuple(sparse_attn_indices.shape) != (
            total_q,
            TOPK,
        ):
            raise ValueError(f"sparse_attn_indices must be int32 [{total_q}, {TOPK}]")
        block_table = _placeholder(q_storage.device, "block_table", (1, 1))
        req_idx = _placeholder(q_storage.device, "req_idx", (1,))
        stride_factor = 1
    if output.dtype != torch.bfloat16 or output.numel() != total_q * NUM_HEADS * L:
        raise ValueError("output must be contiguous BF16 storage for total_q * 16 * 512 values")
    for name, tensor in (
        ("cu_seqlen_ks", cu_seqlen_ks),
        ("cu_seqlen_ke", cu_seqlen_ke),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be rank one")
        if tensor.numel() < total_q:
            raise ValueError(f"{name} capacity must be at least total_q")
    if cu_seqlen_ks.shape != cu_seqlen_ke.shape:
        raise ValueError("cu_seqlen_ks and cu_seqlen_ke capacities must match")
    if (cu_q_seqlens is None) != (sequence_length is None):
        raise ValueError("cu_q_seqlens and sequence_length must both be tensors or both be None")
    batch_size: int | None = None
    if cu_q_seqlens is not None:
        if cu_q_seqlens.dtype != torch.int32 or sequence_length.dtype != torch.int32:
            raise TypeError("cu_q_seqlens and sequence_length must be int32")
        if sequence_length.ndim != 1:
            raise ValueError("sequence_length must be rank one")
        batch_size = int(sequence_length.shape[0])
        if tuple(cu_q_seqlens.shape) != (batch_size + 1,):
            raise ValueError("cu_q_seqlens and sequence_length shapes are inconsistent")
    if host_past_key_value_lengths is not None:
        if host_past_key_value_lengths.dtype != torch.int32:
            raise TypeError("host_past_key_value_lengths must be int32")
        if host_past_key_value_lengths.device.type != "cpu":
            raise ValueError("host_past_key_value_lengths must live on CPU")
        if host_past_key_value_lengths.ndim != 1:
            raise ValueError("host_past_key_value_lengths must be rank one")
        if batch_size is not None and tuple(host_past_key_value_lengths.shape) != (batch_size,):
            raise ValueError("host_past_key_value_lengths has the wrong batch size")
    if tuple(mla_bmm1_scale.shape) != (2,) or mla_bmm1_scale.dtype != torch.float32:
        raise ValueError("mla_bmm1_scale must be FP32[2]")
    if tuple(mla_bmm2_scale.shape) != (1,) or mla_bmm2_scale.dtype != torch.float32:
        raise ValueError("mla_bmm2_scale must be FP32[1]")

    tensors = [
        q_storage,
        kv_storage,
        sparse_attn_indices,
        cu_seqlen_ks,
        cu_seqlen_ke,
        output,
        mla_bmm1_scale,
        mla_bmm2_scale,
        block_table,
        req_idx,
        group_table,
    ]
    if cu_q_seqlens is not None:
        tensors.extend((cu_q_seqlens, sequence_length))
    if any(t.device != q_storage.device for t in tensors):
        raise ValueError("all runtime tensors must be on the Q tensor device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all runtime tensors must be contiguous")
    for name, tensor in (
        ("quant_q_buffer", q_storage),
        ("kv_pool_view", kv_storage),
        ("sparse_attn_indices", sparse_attn_indices),
        ("cu_seqlen_ks", cu_seqlen_ks),
        ("cu_seqlen_ke", cu_seqlen_ke),
        ("mla_bmm1_scale", mla_bmm1_scale),
        ("mla_bmm2_scale", mla_bmm2_scale),
        ("block_table", block_table),
        ("req_idx", req_idx),
        ("group_table", group_table),
    ):
        if _overlaps(output, tensor):
            raise ValueError(f"output must not overlap {name}")
    for name, tensor in (
        ("mla_bmm1_scale", mla_bmm1_scale),
        ("mla_bmm2_scale", mla_bmm2_scale),
    ):
        if _overlaps(q_storage, tensor):
            raise ValueError(f"quant_q_buffer must not overlap {name}")
    for name, tensor in (
        ("quant_q_buffer", q_storage),
        ("kv_pool_view", kv_storage),
    ):
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be at least 16-byte aligned")
    if sparse_attn_indices.data_ptr() % 32:
        raise ValueError("sparse_attn_indices must be at least 32-byte aligned")
    if output.data_ptr() % 32:
        raise ValueError("output must be at least 32-byte aligned")
    for name, tensor in (
        ("cu_seqlen_ks", cu_seqlen_ks),
        ("cu_seqlen_ke", cu_seqlen_ke),
        ("mla_bmm1_scale", mla_bmm1_scale),
        ("mla_bmm2_scale", mla_bmm2_scale),
        ("block_table", block_table),
        ("req_idx", req_idx),
        ("group_table", group_table),
    ):
        if tensor.data_ptr() % 4:
            raise ValueError(f"{name} must be at least 4-byte aligned")

    if torch.cuda.get_device_capability(q_storage.device) != (10, 3):
        raise RuntimeError("GLM sparse-MLA prefill requires an SM103 GPU")

    device_index = _device_index(q_storage)
    compiled = compile(
        all_sparse_indices_valid=all_sparse_indices_valid,
        dense_block_table=dense_block_table,
        dense_causal=dense_causal,
        device_index=device_index,
    )
    return PreparedMlaPrefillRunner(
        compiled=compiled,
        args=(
            q_storage,
            kv_storage.reshape(pool_rows, KV_STRIDE),
            sparse_attn_indices,
            cu_seqlen_ks,
            cu_seqlen_ke,
            output.reshape(total_q, NUM_HEADS, L),
            mla_bmm1_scale,
            mla_bmm2_scale,
            req_idx,
            block_table,
            group_table,
            int(stride_factor),
            int(layer_offset),
        ),
        device_index=device_index,
    )


@cute.jit
def _consume_clc_response(response, full, empty, iteration):
    """Consume one response per warp and release the two-stage fanout slot."""
    stage = iteration % cutlass.Int32(CLC_STAGES)
    phase = (iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
    while not prims.mbarrier_try_wait_parity(full.data_ptr() + stage, phase):
        pass
    work_id = cutlass.Int32(0)
    valid_i32 = cutlass.Int32(0)
    if prims.elect_sync():
        opaque = prims.load_ext(response.data_ptr() + stage, dtype=cutlass.Int128)
        valid = prims.clusterlaunchcontrol_query_cancel(
            prims.ClusterLaunchControlQueryType.IS_CANCELED,
            opaque,
        )
        work_id = prims.clusterlaunchcontrol_query_cancel(
            prims.ClusterLaunchControlQueryType.GET_FIRST_CTA_ID_X,
            opaque,
        )
        valid_i32 = cutlass.Int32(valid)
        prims.fence_proxy(
            prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )
        prims.mbarrier_arrive(empty.data_ptr() + stage, relaxed=True)
    work_id = prims.shfl_sync(0xFFFFFFFF, work_id, 0, 0x1F, prims.Shfl.IDX)
    valid_i32 = prims.shfl_sync(0xFFFFFFFF, valid_i32, 0, 0x1F, prims.Shfl.IDX)
    return work_id, valid_i32 != cutlass.Int32(0)


@cute.jit
def _tma_gather4_cta1(
    smem_dst,
    tma_desc,
    mbarrier,
    col,
    row0,
    row1,
    row2,
    row3,
) -> None:
    """Issue one leader-only CTA1 gather4 without divergent Python control."""
    local_barrier = prims.cvta_to(mbarrier, prims.CvtaSpace.SHARED)
    prims.inline_ptx_hl(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, -1;\n\t"
        "@leader cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.L2::cache_hint"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}],"
        " [{$r7}], {$r8};\n\t"
        "}",
        read_only_args=[
            smem_dst,
            tma_desc,
            col,
            row0,
            row1,
            row2,
            row3,
            local_barrier,
            cutlass.Int64(0),
        ],
    )


@cute.jit
def _freeze_smem_descriptor(desc):
    """Copy an SMEM descriptor through a register to prevent rematerialization."""
    return cute.arch.inline_ptx(
        "mov.b64 {$w0}, {$r0};",
        write_only_types=[cutlass.Int64],
        read_only_args=[desc],
    )


@cute.jit
def _tcgen05_mma_qk4(tmem_addr, desc_k, desc_q, enable_input_d, n: cutlass.Constexpr[int]) -> None:
    """Issue one K128 QK group without rematerializing descriptors per K32.

    ``n`` is the MMA N (TMEM columns): 16 heads of one token on the sparse
    path, ``N_DC`` = 64 (token, head) columns in dense_causal mode.  The
    instruction descriptor is a trace-time literal, so the
    sparse call sites emit the historical ``mov.b32 idesc, 0x8040010`` text.
    """
    prims.inline_ptx_hl(
        "{\n\t"
        ".reg .pred leader, accumulate;\n\t"
        ".reg .b32 idesc;\n\t"
        ".reg .b64 kdesc, qdesc;\n\t"
        "elect.sync _|leader, -1;\n\t"
        f"mov.b32 idesc, {_mma_idesc(n, False):#x};\n\t"
        "mov.b64 kdesc, {$r1};\n\t"
        "mov.b64 qdesc, {$r2};\n\t"
        "setp.ne.b32 accumulate, {$r3}, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], kdesc, qdesc, idesc, accumulate;\n\t"
        "add.u64 kdesc, kdesc, 2;\n\t"
        "add.u64 qdesc, qdesc, 2;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], kdesc, qdesc, idesc, 1;\n\t"
        "add.u64 kdesc, kdesc, 2;\n\t"
        "add.u64 qdesc, qdesc, 2;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], kdesc, qdesc, idesc, 1;\n\t"
        "add.u64 kdesc, kdesc, 2;\n\t"
        "add.u64 qdesc, qdesc, 2;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], kdesc, qdesc, idesc, 1;\n\t"
        "}",
        read_only_args=[
            cutlass.Int32(tmem_addr),
            cutlass.Int64(desc_k),
            cutlass.Int64(desc_q),
            cutlass.Int32(enable_input_d),
        ],
    )


@cute.jit
def _tcgen05_mma_qk2(tmem_addr, desc_k, desc_q, n: cutlass.Constexpr[int]) -> None:
    """Issue the K64 rope QK group as two adjacent K32 MMAs (``n`` as in qk4)."""
    prims.inline_ptx_hl(
        "{\n\t"
        ".reg .pred leader;\n\t"
        ".reg .b32 idesc;\n\t"
        ".reg .b64 kdesc, qdesc;\n\t"
        "elect.sync _|leader, -1;\n\t"
        f"mov.b32 idesc, {_mma_idesc(n, False):#x};\n\t"
        "mov.b64 kdesc, {$r1};\n\t"
        "mov.b64 qdesc, {$r2};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], kdesc, qdesc, idesc, 1;\n\t"
        "add.u64 kdesc, kdesc, 2;\n\t"
        "add.u64 qdesc, qdesc, 2;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], kdesc, qdesc, idesc, 1;\n\t"
        "}",
        read_only_args=[
            cutlass.Int32(tmem_addr),
            cutlass.Int64(desc_k),
            cutlass.Int64(desc_q),
        ],
    )


@cute.jit
def _tcgen05_mma_pv4(tmem_addr, desc_v, desc_p, enable_input_d, n: cutlass.Constexpr[int]) -> None:
    """Issue one K128 PV group without rematerializing descriptors per K32.

    ``n`` is the MMA N exactly as in :func:`_tcgen05_mma_qk4`; A = V^T is
    MN-major (descriptor bit 15), so the sparse literal is ``0x8048010``.
    """
    prims.inline_ptx_hl(
        "{\n\t"
        ".reg .pred leader, accumulate;\n\t"
        ".reg .b32 idesc;\n\t"
        ".reg .b64 vdesc, pdesc;\n\t"
        "elect.sync _|leader, -1;\n\t"
        f"mov.b32 idesc, {_mma_idesc(n, True):#x};\n\t"
        "mov.b64 vdesc, {$r1};\n\t"
        "mov.b64 pdesc, {$r2};\n\t"
        "setp.ne.b32 accumulate, {$r3}, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], vdesc, pdesc, idesc, accumulate;\n\t"
        "add.u64 vdesc, vdesc, 256;\n\t"
        "add.u64 pdesc, pdesc, 2;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], vdesc, pdesc, idesc, 1;\n\t"
        "add.u64 vdesc, vdesc, 256;\n\t"
        "add.u64 pdesc, pdesc, 2;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], vdesc, pdesc, idesc, 1;\n\t"
        "add.u64 vdesc, vdesc, 256;\n\t"
        "add.u64 pdesc, pdesc, 2;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4"
        " [{$r0}], vdesc, pdesc, idesc, 1;\n\t"
        "}",
        read_only_args=[
            cutlass.Int32(tmem_addr),
            cutlass.Int64(desc_v),
            cutlass.Int64(desc_p),
            cutlass.Int32(enable_input_d),
        ],
    )


@cute.jit
def _float_to_u32_for_atomic_max(value: cutlass.Float32) -> cutlass.Uint32:
    """Encode a float so one unsigned atomic max preserves float ordering."""
    bits = prims.mov_b32(value, target_type=cutlass.Int32)
    mask = (bits >> cutlass.Int32(31)) | cutlass.Int32(0x80000000)
    return prims.mov_b32(bits ^ mask, target_type=cutlass.Uint32)


@cute.jit
def _u32_to_float_for_atomic_max(value: cutlass.Uint32) -> cutlass.Float32:
    """Decode the monotonic unsigned representation used by softmax max."""
    encoded = prims.mov_b32(value, target_type=cutlass.Int32)
    mask = (~(encoded >> cutlass.Int32(31))) | cutlass.Int32(0x80000000)
    return prims.mov_b32(encoded ^ mask, target_type=cutlass.Float32)


@cute.jit
def _smem_atomic_max_u32(ptr: cute.Pointer, value: cutlass.Uint32) -> None:
    """Update one CTA-scope encoded softmax maximum."""
    prims.atomicrmw(
        prims.AtomicOp.MAX,
        ptr,
        value,
        syncscope=prims.MemScope.CTA,
        space=prims.SharedSpace.shared_cta,
    )


@cute.jit
def _mask_scores_after_valid_prefix(scores, valid_kv, warp_idx, row_idx):
    """Set the positional sparse tail to -FLT_MAX before online softmax."""
    masked = cutlass.Array(ACC_DTYPE, NUM_HEADS, space=cutlass.AddressSpace.rmem)
    row_base = warp_idx * cutlass.Int32(32) + row_idx
    for i in cutlass.range_constexpr(NUM_HEADS):
        fragment_index: cutlass.Constexpr[int] = i % 16
        token_offset: cutlass.Constexpr[int] = (
            ((fragment_index % 4) // 2) + 2 * (fragment_index // 8)
        ) * 8
        value = scores[i]
        if row_base + cutlass.Int32(token_offset) >= valid_kv:
            value = ACC_DTYPE(-3.0e38)
        masked[i] = value
    return cutlass.Vector.from_elements(tuple(masked[i] for i in range(NUM_HEADS)), dtype=ACC_DTYPE)


@cute.jit
def _scale_scores_for_fp8_p(scores, new_max, scale_log2):
    """Apply the online-max shift and TRT FP8-P scale with packed FMAs."""
    scaled = cutlass.Array(ACC_DTYPE, NUM_HEADS, space=cutlass.AddressSpace.rmem)
    scale_pair = (scale_log2, scale_log2)
    neg_scale_pair = (-scale_log2, -scale_log2)
    for i in cutlass.range_constexpr(0, NUM_HEADS, 2):
        max_idx0: cutlass.Constexpr[int] = ((i % 8) // 4) * 2 + (i % 2)
        max_idx1: cutlass.Constexpr[int] = (((i + 1) % 8) // 4) * 2 + ((i + 1) % 2)
        bias = prims.fma_packed_f32x2(
            (new_max[max_idx0], new_max[max_idx1]),
            neg_scale_pair,
            (LOG2_FP8_P_SCALE, LOG2_FP8_P_SCALE),
            rnd=prims.FPRoundingMode.RN,
        )
        scaled_pair = prims.fma_packed_f32x2(
            (scores[i], scores[i + 1]),
            scale_pair,
            bias,
            rnd=prims.FPRoundingMode.RN,
        )
        scaled[i] = scaled_pair[0]
        scaled[i + 1] = scaled_pair[1]
    return cutlass.Vector.from_elements(tuple(scaled[i] for i in range(NUM_HEADS)), dtype=ACC_DTYPE)


@cute.kernel
def glm52_self_attn_fmha_sparse_mla_prefill_kernel(
    tma_ql: cutlass.GridConstant[cuda.TensorMap],
    tma_qr: cutlass.GridConstant[cuda.TensorMap],
    tma_kv: cutlass.GridConstant[cuda.TensorMap],
    tma_kvr: cutlass.GridConstant[cuda.TensorMap],
    sparse_indices: cute.Tensor,
    cu_seqlen_ks: cute.Tensor,
    cu_seqlen_ke: cute.Tensor,
    o: cute.Tensor,
    mla_bmm1_scale: cute.Tensor,
    mla_bmm2_scale: cute.Tensor,
    req_idx: cute.Tensor,
    block_table: cute.Tensor,
    group_table: cute.Tensor,
    stride_factor: cutlass.Int32,
    layer_offset: cutlass.Int32,
    all_sparse_indices_valid: cutlass.Constexpr[bool],
    dense_block_table: cutlass.Constexpr[bool],
    dense_causal: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    # The work id: one query token on the sparse paths, one group-table row
    # (G_TOK tokens of one request) in dense_causal mode.
    query_idx, _, _ = cute.arch.block_idx()

    # ---- dense_causal Constexpr selects ---------------------------------
    # In the sparse specializations every select folds to the historical
    # constant, so the traced device code is unchanged.
    mma_n: cutlass.Constexpr[int] = N_DC if dense_causal else SCORE_N
    tmem_o_off: cutlass.Constexpr[int] = TMEM_O_OFF_DC if dense_causal else TMEM_O_OFF
    num_tmem_cols: cutlass.Constexpr[int] = NUM_TMEM_COLS_DC if dense_causal else NUM_TMEM_COLS
    st_kv: cutlass.Constexpr[int] = ST_KV_DC if dense_causal else ST_KV
    # Q rows (= MMA B rows) per work slot and the number of Q slots.
    q_rows: cutlass.Constexpr[int] = N_DC if dense_causal else NUM_HEADS
    q_stages: cutlass.Constexpr[int] = Q_STAGES_DC if dense_causal else CLC_STAGES
    # The sparse index-tile ring is unused in dense_causal mode (one element).
    st_i: cutlass.Constexpr[int] = 1 if dense_causal else ST_I
    # Dense softmax re-reads the score stage per token chunk after the max
    # barrier, so all four softmax warps release s_empty (one arrival each);
    # the sparse path releases it from W0 lane 0 after the shared barrier.
    s_empty_arrivals: cutlass.Constexpr[int] = 4 if dense_causal else 1
    # Epilogue staging: one token (16 rows x 512 BF16) on the sparse path, a
    # half group of two tokens in dense_causal mode.
    partial_o_tokens: cutlass.Constexpr[int] = 2 if dense_causal else 1

    # ---- SMEM ------------------------------------------------------------
    sQ = cutlass.Array(
        IO_DTYPE,
        q_stages * q_rows * L,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    sQr = cutlass.Array(
        IO_DTYPE,
        q_stages * q_rows * R,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    sKv = cutlass.Array(
        IO_DTYPE,
        st_kv * QK_N * QK_K,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    sparse_stage_slots: cutlass.Constexpr[int] = 1 if dense_causal else ST_I * QK_N
    sparse_stage = cutlass.Array(
        cutlass.Int32,
        sparse_stage_slots,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    sP = cutlass.Array(
        IO_DTYPE, ST_P * QK_N * mma_n, space=cutlass.AddressSpace.smem, alignment=128
    )
    softmax_max = cutlass.Array(
        cutlass.Uint32,
        ST_S * mma_n,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    softmax_sum = cutlass.Array(
        ACC_DTYPE,
        4 * mma_n,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    compact_stats = cutlass.Array(
        ACC_DTYPE,
        ST_C * mma_n * 3,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    partial_o = cutlass.Array(
        OUT_DTYPE,
        partial_o_tokens * VALID_ROWS * L,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    # dense_causal masks from the per-token ``ke`` in group_meta instead.
    invalid_count_slots: cutlass.Constexpr[int] = (
        1 if (all_sparse_indices_valid or dense_causal) else (CLC_STAGES * (TOPK // QK_N))
    )
    invalid_counts = cutlass.Array(
        cutlass.Int32,
        invalid_count_slots,
        space=cutlass.AddressSpace.smem,
        alignment=64,
    )
    active_k_tiles = cutlass.Array(
        cutlass.Int32, CLC_STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    # dense_causal per-work-slot record {q0, n_tok, req, num_kt, ke_0..ke_3}
    # one element in the other modes.
    group_meta_words: cutlass.Constexpr[int] = CLC_STAGES * GROUP_META_WORDS if dense_causal else 1
    group_meta = cutlass.Array(
        cutlass.Int32,
        group_meta_words,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    q_full = cutlass.Array(cutlass.Int64, CLC_STAGES, space=cutlass.AddressSpace.smem)
    q_empty = cutlass.Array(cutlass.Int64, CLC_STAGES, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(cutlass.Int64, st_kv, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(cutlass.Int64, st_kv, space=cutlass.AddressSpace.smem)
    index_full = cutlass.Array(cutlass.Int64, CLC_STAGES, space=cutlass.AddressSpace.smem)
    index_empty = cutlass.Array(cutlass.Int64, CLC_STAGES, space=cutlass.AddressSpace.smem)
    tile_index_full = cutlass.Array(cutlass.Int64, st_i, space=cutlass.AddressSpace.smem)
    tile_index_empty = cutlass.Array(cutlass.Int64, st_i, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    p_empty = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    c_full = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    c_empty = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    o_empty = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    clc_full = cutlass.Array(
        cutlass.Int64, CLC_STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    clc_empty = cutlass.Array(
        cutlass.Int64, CLC_STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    clc_response = cutlass.Array(
        cutlass.Int128, CLC_STAGES, space=cutlass.AddressSpace.smem, alignment=16
    )
    tmem_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    runtime_scales = cutlass.Array(
        ACC_DTYPE,
        2,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )

    if warp_idx == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(CLC_STAGES):
                prims.mbarrier_init(q_full.data_ptr() + i, 1)
                prims.mbarrier_init(q_empty.data_ptr() + i, 1)
                prims.mbarrier_init(index_full.data_ptr() + i, 1)
                prims.mbarrier_init(index_empty.data_ptr() + i, WORK_SLOT_CONSUMERS)
                prims.mbarrier_init(clc_full.data_ptr() + i, 1)
                prims.mbarrier_init(clc_empty.data_ptr() + i, THREADS // 32)
            for i in cutlass.range_constexpr(st_kv):
                prims.mbarrier_init(kv_full.data_ptr() + i, 1)
                prims.mbarrier_init(kv_empty.data_ptr() + i, 2)
            for i in cutlass.range_constexpr(st_i):
                prims.mbarrier_init(tile_index_full.data_ptr() + i, 1)
                prims.mbarrier_init(tile_index_empty.data_ptr() + i, 4)
            for i in cutlass.range_constexpr(ST_S):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, s_empty_arrivals)
            for i in cutlass.range_constexpr(ST_P):
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
                prims.mbarrier_init(p_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_C):
                prims.mbarrier_init(c_full.data_ptr() + i, 1)
                prims.mbarrier_init(c_empty.data_ptr() + i, 4)
            prims.mbarrier_init(o_full, 1)
            prims.mbarrier_init(o_empty, 4)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    # Each logical token CTA owns one independent CTA1 TMEM allocation
    # (128 columns; the full 512 in dense_causal mode).
    if warp_idx == W_MMA_PV:
        prims.tcgen05_alloc(tmem_i32, num_tmem_cols, group="cta_1")
    if warp_idx == W_LOAD_KQ:
        # Q, sparse indices, and both scale arrays are predecessor-produced.
        # W9 waits once and publishes the small scalar payload at the same
        # setup barrier that already protects the TMEM allocation.
        if prims.elect_sync():
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            runtime_scales[0] = (mla_bmm1_scale.iterator.raw_ptr() + 1).load(alignment=4)
            runtime_scales[1] = mla_bmm2_scale.iterator.raw_ptr().load(alignment=4)
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, cutlass.Float32)
    scale_log2 = runtime_scales[0]
    out_scale = runtime_scales[1]
    if warp_idx == W_MMA_PV:
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    q_bytes = (NUM_HEADS * L + NUM_HEADS * R) * EB

    # WG2 contains W8 QK, W9 Q/page-offset, idle W10, and W11 PV. PTX
    # requires all four warps in a warpgroup to execute the same setmaxnreg.
    # Keeping idle W10 at the default allocation can exceed the per-SM
    # register budget once WG0 and WG1 increase their allocations.
    if (warp_idx >= cutlass.Int32(8)) and (warp_idx < cutlass.Int32(12)):
        prims.setmaxregister(80, prims.SetMaxRegisterAction.DECREASE)

    # W10 is a dedicated CLC scheduler.  Every response is published through
    # a two-stage ring and consumed once by each of the 16 warps.  This keeps
    # work discovery off the data-path warps and lets every role advance at
    # the pace permitted by its own resource barriers.
    if warp_idx == cutlass.Int32(SCHEDULER_WARP):
        scheduler_iteration = cutlass.Int32(0)
        scheduler_valid = cutlass.Boolean(True)
        while scheduler_valid:
            stage = scheduler_iteration % cutlass.Int32(CLC_STAGES)
            phase = (scheduler_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
            if prims.elect_sync():
                while not prims.mbarrier_try_wait_parity(
                    clc_empty.data_ptr() + stage,
                    phase ^ cutlass.Int32(1),
                ):
                    pass
                prims.mbarrier_arrive_expect_tx(clc_full.data_ptr() + stage, 16)
                prims.clusterlaunchcontrol_try_cancel(
                    smem_address=(clc_response.data_ptr() + stage).llvm_ptr,
                    mbarrier=(clc_full.data_ptr() + stage).llvm_ptr,
                    multicast="multicast",
                )
            _, scheduler_valid = _consume_clc_response(
                clc_response,
                clc_full,
                clc_empty,
                scheduler_iteration,
            )
            scheduler_iteration += cutlass.Int32(1)

    # =====================================================================
    # W9 - two-stage Q/index producer
    # =====================================================================
    if warp_idx == W_LOAD_KQ:
        if cutlass.const_expr(dense_causal):
            # Dense-causal W9.  The work id is a
            # group-table row.  Per group: load the four-token Q box into the
            # single Q slot, then publish the group record {q0, n_tok, req,
            # num_kt, ke_0..ke_3} through the two-stage work slot
            # (index_full / index_empty, 14 consumers unchanged).  There is
            # no index-tile ring: the producer warps address the K/V blocks
            # straight from the block table.
            work_iteration = cutlass.Int32(0)
            work_valid = cutlass.Boolean(True)
            cu_seqlen_ke_ptr = cu_seqlen_ke.iterator.raw_ptr()
            req_idx_ptr = req_idx.iterator.raw_ptr()
            group_table_ptr = group_table.iterator.raw_ptr()
            lane = tidx % cutlass.Int32(32)
            while work_valid:
                work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
                work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
                # Warp-uniform group record (every lane loads the same words).
                group_row = group_table_ptr + query_idx * cutlass.Int32(2)
                group_q0 = group_row.load(alignment=4)
                group_n_tok = (group_row + cutlass.Int32(1)).load(alignment=4)

                # Q is single-staged (Q_STAGES_DC = 1): barrier slot 0 with
                # the work-item parity as its phase.  q_empty is released by
                # the last QK commit of the previous group.
                q_phase = work_iteration % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(
                    q_empty.data_ptr(),
                    q_phase ^ cutlass.Int32(1),
                ):
                    pass
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(q_full.data_ptr(), Q_BYTES_DC)
                    # Box [128, 16, 4]: smem row n = t * 16 + head, 128 B each,
                    # i.e. the MMA B row / TMEM column order.  Rows past
                    # total_q (last tail group) are TMA zero fill.
                    for c in cutlass.range_constexpr(N_LAT):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sQ.data_ptr() + c * cutlass.Int32(N_DC * QK_K),
                            tma_ql.get_ptr(),
                            (c * QK_K, 0, group_q0),
                            q_full.data_ptr(),
                        )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sQr.data_ptr(),
                        tma_qr.get_ptr(),
                        (0, 0, group_q0),
                        q_full.data_ptr(),
                    )

                while not prims.mbarrier_try_wait_parity(
                    index_empty.data_ptr() + work_stage,
                    work_phase ^ cutlass.Int32(1),
                ):
                    pass

                # ke_t = clamp(cu_seqlen_ke[q0 + t], 1, 2048) for the real
                # tokens; dummy columns (t >= n_tok) copy the last real token
                # so they never reference KV beyond the request.  The 2048
                # clamp mirrors the sparse path's clamp(ke - ks, 0, TOPK)
                # (cu_seqlen_ks is 0 in this mode and is not read).  The
                # lower clamp makes num_kt >= 1 by construction, so the
                # num_kt == 0 zero-output path is unreachable here.
                token = lane
                if token > group_n_tok - cutlass.Int32(1):
                    token = group_n_tok - cutlass.Int32(1)
                token_ke = (cu_seqlen_ke_ptr + group_q0 + token).load(alignment=4)
                if token_ke < cutlass.Int32(1):
                    token_ke = cutlass.Int32(1)
                if token_ke > cutlass.Int32(TOPK):
                    token_ke = cutlass.Int32(TOPK)
                last_ke = prims.shfl_sync(
                    0xFFFFFFFF,
                    token_ke,
                    group_n_tok - cutlass.Int32(1),
                    0x1F,
                    prims.Shfl.IDX,
                )
                num_kt = (last_ke + cutlass.Int32(QK_N - 1)) // cutlass.Int32(QK_N)
                group_req = (req_idx_ptr + group_q0).load(alignment=4)
                meta_base = work_stage * cutlass.Int32(GROUP_META_WORDS)
                if lane < cutlass.Int32(G_TOK):
                    group_meta[meta_base + cutlass.Int32(G_TOK) + lane] = token_ke
                if prims.elect_sync():
                    group_meta[meta_base] = group_q0
                    group_meta[meta_base + cutlass.Int32(1)] = group_n_tok
                    group_meta[meta_base + cutlass.Int32(2)] = group_req
                    group_meta[meta_base + cutlass.Int32(3)] = num_kt
                    active_k_tiles[work_stage] = num_kt
                prims.fence_proxy(
                    prims.Proxy.ASYNC_SHARED,
                    space=prims.SharedSpace.shared_cta,
                )
                if prims.elect_sync():
                    prims.mbarrier_arrive(index_full.data_ptr() + work_stage, relaxed=True)

                query_idx, work_valid = _consume_clc_response(
                    clc_response,
                    clc_full,
                    clc_empty,
                    work_iteration,
                )
                work_iteration += cutlass.Int32(1)
        else:
            work_iteration = cutlass.Int32(0)
            tile_iteration = cutlass.Int32(0)
            work_valid = cutlass.Boolean(True)
            sparse_ptr = sparse_indices.iterator.raw_ptr()
            cu_seqlen_ks_ptr = cu_seqlen_ks.iterator.raw_ptr()
            cu_seqlen_ke_ptr = cu_seqlen_ke.iterator.raw_ptr()
            if cutlass.const_expr(dense_block_table):
                # Dense-causal mode: the sparse row of slot ``s`` for query ``q``
                # is the req_index_to_global image of position
                # ``cu_seqlen_ks[q] + s``, computed here instead of loaded, so
                # the route neither materializes the [T, 2048] index tensor nor
                # launches req_index_to_global. Invalid slots map to -1 exactly like it does.
                req_idx_ptr = req_idx.iterator.raw_ptr()
                block_table_ptr = block_table.iterator.raw_ptr()
                max_blocks_per_req = cutlass.Int32(cute.size(block_table, mode=[1]))
            lane = tidx % cutlass.Int32(32)
            while work_valid:
                work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
                work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(
                    q_empty.data_ptr() + work_stage,
                    work_phase ^ cutlass.Int32(1),
                ):
                    pass
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(q_full.data_ptr() + work_stage, q_bytes)
                    q_stage_base = work_stage * cutlass.Int32(NUM_HEADS * L)
                    for c in cutlass.range_constexpr(N_LAT):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sQ.data_ptr() + q_stage_base + c * cutlass.Int32(NUM_HEADS * QK_K),
                            tma_ql.get_ptr(),
                            (c * QK_K, 0, query_idx),
                            q_full.data_ptr() + work_stage,
                        )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sQr.data_ptr() + work_stage * cutlass.Int32(NUM_HEADS * R),
                        tma_qr.get_ptr(),
                        (0, 0, query_idx),
                        q_full.data_ptr() + work_stage,
                    )

                # Q has an independent lifetime and can be prefetched as soon as
                # QK releases its slot. Delay only the work metadata until every
                # consumer of the previous query has retired that metadata slot.
                while not prims.mbarrier_try_wait_parity(
                    index_empty.data_ptr() + work_stage,
                    work_phase ^ cutlass.Int32(1),
                ):
                    pass

                query_base = query_idx * cutlass.Int32(TOPK)
                num_active = cutlass.Int32(TOPK // QK_N)
                valid_count = cutlass.Int32(TOPK)
                if cutlass.const_expr(not all_sparse_indices_valid):
                    if prims.elect_sync():
                        valid_count = (cu_seqlen_ke_ptr + query_idx).load(alignment=4) - (
                            cu_seqlen_ks_ptr + query_idx
                        ).load(alignment=4)
                        if valid_count < cutlass.Int32(0):
                            valid_count = cutlass.Int32(0)
                        if valid_count > cutlass.Int32(TOPK):
                            valid_count = cutlass.Int32(TOPK)
                    valid_count = prims.shfl_sync(
                        0xFFFFFFFF,
                        valid_count,
                        0,
                        0x1F,
                        prims.Shfl.IDX,
                    )
                    num_active = (valid_count + cutlass.Int32(QK_N - 1)) // cutlass.Int32(QK_N)
                if cutlass.const_expr(dense_block_table):
                    # Warp-uniform per-query operands of the row formula.
                    dense_ks = (cu_seqlen_ks_ptr + query_idx).load(alignment=4)
                    dense_req = (req_idx_ptr + query_idx).load(alignment=4)
                    dense_table_row = dense_req * max_blocks_per_req
                if prims.elect_sync():
                    active_k_tiles[work_stage] = num_active
                prims.fence_proxy(
                    prims.Proxy.ASYNC_SHARED,
                    space=prims.SharedSpace.shared_cta,
                )
                if prims.elect_sync():
                    prims.mbarrier_arrive(index_full.data_ptr() + work_stage, relaxed=True)

                for kt in cutlass.range(0, num_active, 1, unroll=1):
                    index_stage = tile_iteration % cutlass.Int32(ST_I)
                    index_phase = (tile_iteration // cutlass.Int32(ST_I)) % cutlass.Int32(2)
                    while not prims.mbarrier_try_wait_parity(
                        tile_index_empty.data_ptr() + index_stage,
                        index_phase ^ cutlass.Int32(1),
                    ):
                        pass
                    if cutlass.const_expr(not all_sparse_indices_valid):
                        if prims.elect_sync():
                            tile_valid = valid_count - kt * cutlass.Int32(QK_N)
                            if tile_valid > cutlass.Int32(QK_N):
                                tile_valid = cutlass.Int32(QK_N)
                            if tile_valid < cutlass.Int32(0):
                                tile_valid = cutlass.Int32(0)
                            invalid_counts[work_stage * cutlass.Int32(TOPK // QK_N) + kt] = (
                                cutlass.Int32(QK_N) - tile_valid
                            )
                    if lane < cutlass.Int32(16):
                        row = lane * cutlass.Int32(8)
                        if cutlass.const_expr(dense_block_table):
                            mapped = []
                            for j in cutlass.range_constexpr(8):
                                slot = kt * cutlass.Int32(QK_N) + row + cutlass.Int32(j)
                                value = cutlass.Int32(-1)
                                if slot < valid_count:
                                    pos = dense_ks + slot
                                    block_id = pos >> 5
                                    if block_id < max_blocks_per_req:
                                        base = (block_table_ptr + dense_table_row + block_id).load(
                                            alignment=4
                                        )
                                        if base >= cutlass.Int32(0):
                                            value = base * stride_factor + (pos & 31) + layer_offset
                                mapped.append(value)
                            values_lo = cutlass.Vector.from_elements(
                                (mapped[0], mapped[1], mapped[2], mapped[3]),
                                dtype=cutlass.Int32,
                            )
                            values_hi = cutlass.Vector.from_elements(
                                (mapped[4], mapped[5], mapped[6], mapped[7]),
                                dtype=cutlass.Int32,
                            )
                        else:
                            values = (
                                sparse_ptr + query_base + kt * cutlass.Int32(QK_N) + row
                            ).load(count=8, alignment=32)
                            values_lo = cutlass.Vector.from_elements(
                                (values[0], values[1], values[2], values[3]),
                                dtype=cutlass.Int32,
                            )
                            values_hi = cutlass.Vector.from_elements(
                                (values[4], values[5], values[6], values[7]),
                                dtype=cutlass.Int32,
                            )
                        stage_ptr = (
                            sparse_stage.data_ptr() + index_stage * cutlass.Int32(QK_N) + row
                        )
                        stage_ptr.store(values_lo, alignment=16)
                        (stage_ptr + cutlass.Int32(4)).store(values_hi, alignment=16)
                    prims.fence_proxy(
                        prims.Proxy.ASYNC_SHARED,
                        space=prims.SharedSpace.shared_cta,
                    )
                    if prims.elect_sync():
                        prims.mbarrier_arrive(
                            tile_index_full.data_ptr() + index_stage,
                            relaxed=True,
                        )
                    tile_iteration += cutlass.Int32(1)

                query_idx, work_valid = _consume_clc_response(
                    clc_response,
                    clc_full,
                    clc_empty,
                    work_iteration,
                )
                work_iteration += cutlass.Int32(1)

    # =====================================================================
    # W12-W15 - four independent direct sparse gather4 producers. Each warp
    # owns one quarter of both gather trees, matching TRTLLM's sparse Load WG.
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_LOAD_GATHER0)) and (
        warp_idx < cutlass.Int32(W_LOAD_GATHER0 + NUM_GATHER_WARPS)
    ):
        prims.setmaxregister(80, prims.SetMaxRegisterAction.DECREASE)
        producer_rank = warp_idx - cutlass.Int32(W_LOAD_GATHER0)
        routed_rows = cutlass.Array(cutlass.Int32, 32, space=cutlass.AddressSpace.rmem)
        if cutlass.const_expr(dense_causal):
            block_table_ptr = block_table.iterator.raw_ptr()
            max_blocks_per_req = cutlass.Int32(cute.size(block_table, mode=[1]))
        lane = tidx % cutlass.Int32(32)
        tile_iteration = cutlass.Int32(0)
        work_iteration = cutlass.Int32(0)
        work_valid = cutlass.Boolean(True)
        while work_valid:
            work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
            work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(
                index_full.data_ptr() + work_stage, work_phase
            ):
                pass
            num_kt = active_k_tiles[work_stage]
            if cutlass.const_expr(dense_causal):
                # Group record published by W9: request row of the block
                # table and the largest ke of the group (ke_{n_tok-1}).
                meta_base = work_stage * cutlass.Int32(GROUP_META_WORDS)
                group_n_tok = group_meta[meta_base + cutlass.Int32(1)]
                group_table_row = group_meta[meta_base + cutlass.Int32(2)] * max_blocks_per_req
                group_last_ke = group_meta[
                    meta_base + cutlass.Int32(G_TOK) + group_n_tok - cutlass.Int32(1)
                ]

            for kt in cutlass.range(0, num_kt, 1, unroll=1):
                if cutlass.const_expr(dense_causal):
                    # Dense-causal producer: this warp
                    # owns paged block ``kt * 4 + rank`` = positions
                    # [block * 32, +32) of every panel, loaded with one 2-D
                    # TMA tile load per panel (20 per tile instead of ~160
                    # gather4).  Blocks at or past the group's largest ke,
                    # past the table, or with a negative entry fall back to
                    # pool row 0: every row they supply is causally masked
                    # (its positions are >= ke of every column).  The load is
                    # still issued so kv_full's transaction count completes.
                    block_id = kt * cutlass.Int32(NUM_GATHER_WARPS) + producer_rank
                    block_row0 = cutlass.Int32(0)
                    if block_id * cutlass.Int32(KV_BOX_ROWS_DC) < group_last_ke:
                        if block_id < max_blocks_per_req:
                            table_entry = (block_table_ptr + group_table_row + block_id).load(
                                alignment=4
                            )
                            if table_entry >= cutlass.Int32(0):
                                block_row0 = table_entry * stride_factor + layer_offset
                    block_row0 = cute.arch.make_warp_uniform(block_row0)

                    # Same RoPE-first ring order and slot/phase arithmetic as
                    # the sparse path, over ST_KV_DC slots.
                    base_panel = tile_iteration * cutlass.Int32(K_HDIM_STAGES)
                    base_slot = base_panel % cutlass.Int32(st_kv)
                    base_phase = (base_panel // cutlass.Int32(st_kv)) & cutlass.Int32(1)
                    for c in cutlass.range_constexpr(K_HDIM_STAGES):
                        slot = base_slot + cutlass.Int32(c)
                        panel_phase = base_phase
                        if slot >= cutlass.Int32(st_kv):
                            slot -= cutlass.Int32(st_kv)
                            panel_phase ^= cutlass.Int32(1)
                        while not prims.mbarrier_try_wait_parity(
                            kv_empty.data_ptr() + slot,
                            panel_phase ^ cutlass.Int32(1),
                        ):
                            pass
                        if producer_rank == cutlass.Int32(0):
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(
                                    kv_full.data_ptr() + slot,
                                    KVR_STAGE_BYTES if c == 0 else KV_STAGE_BYTES,
                                )
                        # 32-row boxes at rank * 2 KiB (RoPE, SW64) / rank *
                        # 4 KiB (latent, SW128) keep the swizzle phase, so the
                        # slot image equals the sparse path's.
                        if prims.elect_sync():
                            if c == 0:
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sKv.data_ptr()
                                    + slot * cutlass.Int32(CTA_N * QK_K)
                                    + producer_rank * cutlass.Int32(KV_BOX_ROWS_DC * R),
                                    tma_kvr.get_ptr(),
                                    (0, block_row0),
                                    kv_full.data_ptr() + slot,
                                )
                            else:
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sKv.data_ptr()
                                    + slot * cutlass.Int32(CTA_N * QK_K)
                                    + producer_rank * cutlass.Int32(KV_BOX_ROWS_DC * QK_K),
                                    tma_kv.get_ptr(),
                                    ((c - 1) * QK_K, block_row0),
                                    kv_full.data_ptr() + slot,
                                )
                    tile_iteration += cutlass.Int32(1)
                else:
                    index_stage = tile_iteration % cutlass.Int32(ST_I)
                    index_phase = (tile_iteration // cutlass.Int32(ST_I)) % cutlass.Int32(2)
                    while not prims.mbarrier_try_wait_parity(
                        tile_index_full.data_ptr() + index_stage,
                        index_phase,
                    ):
                        pass
                    staged_base = index_stage * cutlass.Int32(QK_N) + producer_rank * cutlass.Int32(
                        CTA_N // NUM_GATHER_WARPS
                    )
                    for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                        values = (
                            sparse_stage.data_ptr() + staged_base + cutlass.Int32(g * 4)
                        ).load(count=4, alignment=16)
                        for j in cutlass.range_constexpr(4):
                            routed_row = values[j]
                            if routed_row < cutlass.Int32(0):
                                routed_row = cutlass.Int32(0)
                            routed_rows[g * 4 + j] = cute.arch.make_warp_uniform(routed_row)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(
                            tile_index_empty.data_ptr() + index_stage,
                            relaxed=True,
                        )

                    # RoPE is first so its QK-only release gives the producer the
                    # fifth slot needed to finish the next tile before current PV.
                    base_panel = tile_iteration * cutlass.Int32(K_HDIM_STAGES)
                    base_slot = base_panel % cutlass.Int32(st_kv)
                    base_phase = (base_panel // cutlass.Int32(st_kv)) & cutlass.Int32(1)
                    for c in cutlass.range_constexpr(K_HDIM_STAGES):
                        slot = base_slot + cutlass.Int32(c)
                        panel_phase = base_phase
                        if slot >= cutlass.Int32(st_kv):
                            slot -= cutlass.Int32(st_kv)
                            panel_phase ^= cutlass.Int32(1)
                        while not prims.mbarrier_try_wait_parity(
                            kv_empty.data_ptr() + slot,
                            panel_phase ^ cutlass.Int32(1),
                        ):
                            pass
                        if producer_rank == cutlass.Int32(0):
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(
                                    kv_full.data_ptr() + slot,
                                    KVR_STAGE_BYTES if c == 0 else KV_STAGE_BYTES,
                                )
                        for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                            if c == 0:
                                _tma_gather4_cta1(
                                    sKv.data_ptr()
                                    + slot * cutlass.Int32(CTA_N * QK_K)
                                    + (
                                        producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
                                        + cutlass.Int32(g * 4)
                                    )
                                    * cutlass.Int32(R),
                                    tma_kvr.get_ptr(),
                                    kv_full.data_ptr() + slot,
                                    cutlass.Int32(0),
                                    routed_rows[g * 4],
                                    routed_rows[g * 4 + 1],
                                    routed_rows[g * 4 + 2],
                                    routed_rows[g * 4 + 3],
                                )
                            else:
                                _tma_gather4_cta1(
                                    sKv.data_ptr()
                                    + slot * cutlass.Int32(CTA_N * QK_K)
                                    + (
                                        producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
                                        + cutlass.Int32(g * 4)
                                    )
                                    * cutlass.Int32(QK_K),
                                    tma_kv.get_ptr(),
                                    kv_full.data_ptr() + slot,
                                    cutlass.Int32((c - 1) * QK_K),
                                    routed_rows[g * 4],
                                    routed_rows[g * 4 + 1],
                                    routed_rows[g * 4 + 2],
                                    routed_rows[g * 4 + 3],
                                )
                    tile_iteration += cutlass.Int32(1)

            if prims.elect_sync():
                prims.mbarrier_arrive(index_empty.data_ptr() + work_stage, relaxed=True)
            query_idx, work_valid = _consume_clc_response(
                clc_response,
                clc_full,
                clc_empty,
                work_iteration,
            )
            work_iteration += cutlass.Int32(1)

    # =====================================================================
    # W8 — independent CTA1 QK consumer
    # =====================================================================
    if warp_idx == W_MMA_QK:
        dq_root = prims.Tcgen05SmemDesc.build(
            start_address=sQ,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dqr_root = prims.Tcgen05SmemDesc.build(
            start_address=sQr,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_ROPE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        dk_root = prims.Tcgen05SmemDesc.build(
            start_address=sKv,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dk = _freeze_smem_descriptor(dk_root)
        dkr_root = prims.Tcgen05SmemDesc.build(
            start_address=sKv,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_ROPE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        dkr = _freeze_smem_descriptor(dkr_root)

        work_iteration = cutlass.Int32(0)
        tile_iteration = cutlass.Int32(0)
        work_valid = cutlass.Boolean(True)
        while work_valid:
            work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
            work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
            if cutlass.const_expr(dense_causal):
                # Single Q slot (Q_STAGES_DC = 1): barrier 0, phase = work
                # parity, mirroring W9's dense branch.
                q_slot = cutlass.Int32(0)
                q_phase = work_iteration % cutlass.Int32(2)
            else:
                q_slot = work_stage
                q_phase = work_phase
            while not prims.mbarrier_try_wait_parity(q_full.data_ptr() + q_slot, q_phase):
                pass
            while not prims.mbarrier_try_wait_parity(
                index_full.data_ptr() + work_stage, work_phase
            ):
                pass
            num_kt = active_k_tiles[work_stage]
            q_offset_bytes = q_slot * cutlass.Int32(q_rows * L * EB)
            qr_offset_bytes = q_slot * cutlass.Int32(q_rows * R * EB)
            dq = _freeze_smem_descriptor(dq_root.advance_start_address(q_offset_bytes))
            dqr = _freeze_smem_descriptor(dqr_root.advance_start_address(qr_offset_bytes))
            sst = tile_iteration % cutlass.Int32(ST_S)
            sph = (tile_iteration // cutlass.Int32(ST_S)) % cutlass.Int32(2)

            for kt in cutlass.range(0, num_kt, 1, unroll=1):
                while not prims.mbarrier_try_wait_parity(
                    s_empty.data_ptr() + sst,
                    sph ^ cutlass.Int32(1),
                ):
                    pass
                base_panel = tile_iteration * cutlass.Int32(K_HDIM_STAGES)
                base_slot = base_panel % cutlass.Int32(st_kv)
                base_phase = (base_panel // cutlass.Int32(st_kv)) & cutlass.Int32(1)
                # Initialize with L0, accumulate L1-L3, then append RoPE.
                for c in cutlass.range_constexpr(1, K_HDIM_STAGES):
                    slot = base_slot + cutlass.Int32(c)
                    panel_phase = base_phase
                    if slot >= cutlass.Int32(st_kv):
                        slot -= cutlass.Int32(st_kv)
                        panel_phase ^= cutlass.Int32(1)
                    while not prims.mbarrier_try_wait_parity(
                        kv_full.data_ptr() + slot, panel_phase
                    ):
                        pass
                    ko = slot * cutlass.Int32(CTA_N * QK_K * EB)
                    # Q panel pitch: q_rows x 128 B (16 rows sparse, 64 dense).
                    qo: cutlass.Constexpr[int] = (c - 1) * q_rows * QK_K * EB
                    _tcgen05_mma_qk4(
                        tmem_raw + sst * cutlass.Int32(mma_n),
                        dk + cutlass.Int64(ko // 16),
                        dq + cutlass.Int64(qo // 16),
                        c != 1,
                        mma_n,
                    )
                    if prims.elect_sync():
                        # First of two releases; PV supplies the second.
                        prims.tcgen05_commit(
                            kv_empty.data_ptr() + slot,
                            group=prims.CTAGroup.CTA_1,
                        )

                while not prims.mbarrier_try_wait_parity(
                    kv_full.data_ptr() + base_slot, base_phase
                ):
                    pass
                rope_offset = base_slot * cutlass.Int32(QK_N * QK_K * EB)
                _tcgen05_mma_qk2(
                    tmem_raw + sst * cutlass.Int32(mma_n),
                    dkr + cutlass.Int64(rope_offset // 16),
                    dqr,
                    mma_n,
                )
                if prims.elect_sync():
                    # RoPE has no PV consumer. The plain arrival follows the
                    # commit, so count=2 completes only after the QK read.
                    prims.tcgen05_commit(
                        kv_empty.data_ptr() + base_slot,
                        group=prims.CTAGroup.CTA_1,
                    )
                    prims.mbarrier_arrive(
                        kv_empty.data_ptr() + base_slot,
                        relaxed=True,
                    )
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        s_full.data_ptr() + sst,
                        group=prims.CTAGroup.CTA_1,
                    )
                sst += cutlass.Int32(1)
                if sst == cutlass.Int32(ST_S):
                    sst = cutlass.Int32(0)
                    sph ^= cutlass.Int32(1)
                tile_iteration += cutlass.Int32(1)

            if prims.elect_sync():
                if num_kt == cutlass.Int32(0):
                    # No tcgen05 operation exists for an empty row, so an
                    # empty commit has no completion event to release Q.
                    prims.mbarrier_arrive(
                        q_empty.data_ptr() + q_slot,
                        relaxed=True,
                    )
                else:
                    prims.tcgen05_commit(
                        q_empty.data_ptr() + q_slot,
                        group=prims.CTAGroup.CTA_1,
                    )
                prims.mbarrier_arrive(index_empty.data_ptr() + work_stage, relaxed=True)
            query_idx, work_valid = _consume_clc_response(
                clc_response,
                clc_full,
                clc_empty,
                work_iteration,
            )
            work_iteration += cutlass.Int32(1)

    # =====================================================================
    # W11 — CTA1 SwapsAb VP; retain all four V128 stripes in TMEM
    # =====================================================================
    if warp_idx == W_MMA_PV:
        dp_root = prims.Tcgen05SmemDesc.build(
            start_address=sP,
            leading_byte_offset=LBO_P,
            stride_byte_offset=SBO_P,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dv_root = prims.Tcgen05SmemDesc.build(
            start_address=sKv,
            leading_byte_offset=LBO_V,
            stride_byte_offset=SBO_V,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dv = _freeze_smem_descriptor(dv_root)
        work_iteration = cutlass.Int32(0)
        tile_iteration = cutlass.Int32(0)
        work_valid = cutlass.Boolean(True)
        while work_valid:
            work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
            work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(
                index_full.data_ptr() + work_stage, work_phase
            ):
                pass
            num_kt = active_k_tiles[work_stage]
            pst = tile_iteration % cutlass.Int32(ST_P)
            pph = (tile_iteration // cutlass.Int32(ST_P)) % cutlass.Int32(2)

            for kt in cutlass.range(0, num_kt, 1, unroll=1):
                while not prims.mbarrier_try_wait_parity(p_full.data_ptr() + pst, pph):
                    pass
                oph = tile_iteration % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(o_empty, oph ^ cutlass.Int32(1)):
                    pass
                # P^T stage pitch: mma_n (token, head) rows x 128 B of KV.
                pbase = pst * cutlass.Int32(QK_N * mma_n)
                dp = _freeze_smem_descriptor(dp_root.advance_start_address(pbase))
                base_panel = tile_iteration * cutlass.Int32(K_HDIM_STAGES)
                base_slot = base_panel % cutlass.Int32(st_kv)
                base_phase = (base_panel // cutlass.Int32(st_kv)) & cutlass.Int32(1)
                for s in cutlass.range_constexpr(V_HDIM_STAGES):
                    slot = base_slot + cutlass.Int32(s + 1)
                    panel_phase = base_phase
                    if slot >= cutlass.Int32(st_kv):
                        slot -= cutlass.Int32(st_kv)
                        panel_phase ^= cutlass.Int32(1)
                    while not prims.mbarrier_try_wait_parity(
                        kv_full.data_ptr() + slot, panel_phase
                    ):
                        pass
                    vso = slot * cutlass.Int32(QK_N * QK_K * EB)
                    # O^T stripe s at columns tmem_o_off + s * mma_n.
                    _tcgen05_mma_pv4(
                        tmem_raw + tmem_o_off + s * mma_n,
                        dv + cutlass.Int64(vso // 16),
                        dp,
                        kt > 0,
                        mma_n,
                    )
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            kv_empty.data_ptr() + slot,
                            group=prims.CTAGroup.CTA_1,
                        )
                if prims.elect_sync():
                    prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)
                    prims.tcgen05_commit(
                        p_empty.data_ptr() + pst,
                        group=prims.CTAGroup.CTA_1,
                    )
                pst += cutlass.Int32(1)
                if pst == cutlass.Int32(ST_P):
                    pst = cutlass.Int32(0)
                    pph ^= cutlass.Int32(1)
                tile_iteration += cutlass.Int32(1)

            if prims.elect_sync():
                prims.mbarrier_arrive(index_empty.data_ptr() + work_stage, relaxed=True)
            query_idx, work_valid = _consume_clc_response(
                clc_response,
                clc_full,
                clc_empty,
                work_iteration,
            )
            work_iteration += cutlass.Int32(1)

    # =====================================================================
    # W0-W3 — TRTLLM-style wide-tcgen05.ld online softmax warpgroup
    # =====================================================================
    if warp_idx < cutlass.Int32(4):
        prims.setmaxregister(152, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        col_idx = lane % cutlass.Int32(4)
        row_idx = lane // cutlass.Int32(4)
        local_row_idx = row_idx % cutlass.Int32(4)
        base = prims.TmemAddr(tmem_raw)
        row0 = base.row_id + warp_idx * cutlass.Int32(32)
        addr0 = (row0 << cutlass.Int32(16)) | base.col_id
        addr1 = ((row0 + cutlass.Int32(16)) << cutlass.Int32(16)) | base.col_id
        smem_swizzle = cutlass.Swizzle.from_name("s128b")
        if cutlass.const_expr(dense_causal):
            # Dense-causal softmax.  The score
            # tile is S^T[128 KV rows x 64 columns], column n = t * 16 + head.
            # Each thread owns the sparse path's 4 KV rows x 4 heads fragment
            # once per token chunk t (16 columns), so every per-token operation
            # (mask, max, exp2, sums, FP8 P) is the sparse path's element for
            # element -- the bitwise gate.  Per tile: phase A loads each chunk,
            # masks rows at or past the token's visibility, reduces the max
            # into the per-token smem slots, then one BAR_SM0; phase B reloads
            # each chunk from TMEM (keeps the live set at 16 scores + 48 stats
            # inside the 152-register budget), re-masks, scales, exp2, sums and
            # stores P.  Every softmax warp releases s_empty after its phase B
            # (count 4), so QK(kt+1) still runs into the other stage.
            prev_max_arr = cutlass.Array(ACC_DTYPE, G_TOK * 4, space=cutlass.AddressSpace.rmem)
            prev_sum_arr = cutlass.Array(ACC_DTYPE, G_TOK * 4, space=cutlass.AddressSpace.rmem)
            corr_arr = cutlass.Array(ACC_DTYPE, G_TOK * 4, space=cutlass.AddressSpace.rmem)
            work_iteration = cutlass.Int32(0)
            tile_iteration = cutlass.Int32(0)
            work_valid = cutlass.Boolean(True)
            while work_valid:
                work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
                work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(
                    index_full.data_ptr() + work_stage, work_phase
                ):
                    pass
                if warp_idx == cutlass.Int32(0):
                    # 64 running-max slots (t * 16 + packed head), two per lane.
                    softmax_max[lane] = _float_to_u32_for_atomic_max(ACC_DTYPE(-3.0e38))
                    softmax_max[lane + cutlass.Int32(32)] = _float_to_u32_for_atomic_max(
                        ACC_DTYPE(-3.0e38)
                    )
                prims.barrier_cta_sync(BAR_SM0, thread_count=128)

                num_kt = active_k_tiles[work_stage]
                meta_base = work_stage * cutlass.Int32(GROUP_META_WORDS)
                # ke_0..ke_3 (dummy tokens copy the last real one, W9).
                token_ke = (group_meta.data_ptr() + meta_base + cutlass.Int32(G_TOK)).load(
                    count=G_TOK, alignment=16
                )
                for i in cutlass.range_constexpr(G_TOK * 4):
                    prev_max_arr[i] = ACC_DTYPE(-3.0e38)
                    prev_sum_arr[i] = ACC_DTYPE(0.0)
                stage = tile_iteration % cutlass.Int32(ST_S)
                phase = (tile_iteration // cutlass.Int32(ST_S)) % cutlass.Int32(2)

                for kt in cutlass.range(0, num_kt, 1, unroll=1):
                    while not prims.mbarrier_try_wait_parity(s_full.data_ptr() + stage, phase):
                        pass
                    tile_row0 = kt * cutlass.Int32(QK_N)

                    # ---- phase A: per-token masked max into smem ----------
                    for t in cutlass.range_constexpr(G_TOK):
                        chunk_col = stage * cutlass.Int32(N_DC) + cutlass.Int32(t * NUM_HEADS)
                        frag0 = prims.tcgen05_ld(
                            "16x256b",
                            cutlass.inttoptr(addr0 + chunk_col, 6, ACC_DTYPE),
                            num=2,
                        )
                        frag1 = prims.tcgen05_ld(
                            "16x256b",
                            cutlass.inttoptr(addr1 + chunk_col, 6, ACC_DTYPE),
                            num=2,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        scores = cutlass.Vector.from_elements(
                            tuple(frag0[i] for i in range(8)) + tuple(frag1[i] for i in range(8)),
                            dtype=ACC_DTYPE,
                        )
                        # Causal mask: rows >= valid_rows_t = ke_t - kt * 128
                        # are not visible to token t (<= 0: whole tile masked,
                        # the lower tile of a straddling group; >= 128: none).
                        valid_rows = token_ke[t] - tile_row0
                        if valid_rows < cutlass.Int32(QK_N):
                            scores = _mask_scores_after_valid_prefix(
                                scores,
                                valid_rows,
                                warp_idx,
                                row_idx,
                            )
                        thread_max = _fragment_max4(scores)
                        reduced_max = cutlass.Vector.from_elements(
                            tuple(_warpgroup_row_max(thread_max[i]) for i in range(4)),
                            dtype=ACC_DTYPE,
                        )
                        atomic_value = reduced_max[0]
                        if local_row_idx == cutlass.Int32(1):
                            atomic_value = reduced_max[1]
                        elif local_row_idx == cutlass.Int32(2):
                            atomic_value = reduced_max[2]
                        elif local_row_idx == cutlass.Int32(3):
                            atomic_value = reduced_max[3]
                        max_slot = (
                            cutlass.Int32(t * NUM_HEADS)
                            + col_idx * cutlass.Int32(4)
                            + local_row_idx
                        )
                        _smem_atomic_max_u32(
                            softmax_max.data_ptr() + max_slot,
                            _float_to_u32_for_atomic_max(atomic_value),
                        )
                    prims.barrier_cta_sync(BAR_SM0, thread_count=128)

                    while not prims.mbarrier_try_wait_parity(
                        c_empty.data_ptr() + stage,
                        phase ^ cutlass.Int32(1),
                    ):
                        pass
                    while not prims.mbarrier_try_wait_parity(
                        p_empty.data_ptr() + stage,
                        phase ^ cutlass.Int32(1),
                    ):
                        pass

                    # ---- phase B, statistics: new max and corr per token --
                    is_final_tile = kt + cutlass.Int32(1) == num_kt
                    for t in cutlass.range_constexpr(G_TOK):
                        encoded_tile_max = (
                            softmax_max.data_ptr()
                            + cutlass.Int32(t * NUM_HEADS)
                            + col_idx * cutlass.Int32(4)
                        ).load(count=4, alignment=16)
                        new_max = cutlass.Vector.from_elements(
                            tuple(
                                _u32_to_float_for_atomic_max(encoded_tile_max[i]) for i in range(4)
                            ),
                            dtype=ACC_DTYPE,
                        )
                        prev_max = cutlass.Vector.from_elements(
                            tuple(prev_max_arr[t * 4 + i] for i in range(4)),
                            dtype=ACC_DTYPE,
                        )
                        # A fully masked chunk keeps new_max == prev_max, so
                        # corr == exp2(0) == 1 exactly.
                        corr = cute.math.exp2(
                            (prev_max - new_max) * scale_log2,
                            fastmath=True,
                        )
                        for i in cutlass.range_constexpr(4):
                            corr_arr[t * 4 + i] = corr[i]
                            prev_max_arr[t * 4 + i] = new_max[i]
                    if not is_final_tile:
                        # Publish corr (64 values) before exp/P so W4-7 can
                        # rescale the previous O tile in parallel.  Lanes 0-3
                        # of W0 hold col_idx = lane, i.e. packed heads lane*4..
                        if warp_idx == cutlass.Int32(0):
                            if lane < cutlass.Int32(4):
                                stat_base = stage * cutlass.Int32(N_DC * 3)
                                for t in cutlass.range_constexpr(G_TOK):
                                    corr_t = cutlass.Vector.from_elements(
                                        tuple(corr_arr[t * 4 + i] for i in range(4)),
                                        dtype=ACC_DTYPE,
                                    )
                                    (
                                        compact_stats.data_ptr()
                                        + stat_base
                                        + cutlass.Int32(2 * N_DC + t * NUM_HEADS)
                                        + lane * cutlass.Int32(4)
                                    ).store(corr_t, alignment=16)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if lane == cutlass.Int32(0):
                                prims.mbarrier_arrive(c_full.data_ptr() + stage)

                    # ---- phase B, per chunk: reload, scale, exp2, sum, P ---
                    for t in cutlass.range_constexpr(G_TOK):
                        chunk_col = stage * cutlass.Int32(N_DC) + cutlass.Int32(t * NUM_HEADS)
                        frag0 = prims.tcgen05_ld(
                            "16x256b",
                            cutlass.inttoptr(addr0 + chunk_col, 6, ACC_DTYPE),
                            num=2,
                        )
                        frag1 = prims.tcgen05_ld(
                            "16x256b",
                            cutlass.inttoptr(addr1 + chunk_col, 6, ACC_DTYPE),
                            num=2,
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        scores = cutlass.Vector.from_elements(
                            tuple(frag0[i] for i in range(8)) + tuple(frag1[i] for i in range(8)),
                            dtype=ACC_DTYPE,
                        )
                        valid_rows = token_ke[t] - tile_row0
                        if valid_rows < cutlass.Int32(QK_N):
                            scores = _mask_scores_after_valid_prefix(
                                scores,
                                valid_rows,
                                warp_idx,
                                row_idx,
                            )
                        new_max = cutlass.Vector.from_elements(
                            tuple(prev_max_arr[t * 4 + i] for i in range(4)),
                            dtype=ACC_DTYPE,
                        )
                        corr = cutlass.Vector.from_elements(
                            tuple(corr_arr[t * 4 + i] for i in range(4)),
                            dtype=ACC_DTYPE,
                        )
                        prev_sum = cutlass.Vector.from_elements(
                            tuple(prev_sum_arr[t * 4 + i] for i in range(4)),
                            dtype=ACC_DTYPE,
                        )
                        scaled = _scale_scores_for_fp8_p(scores, new_max, scale_log2)
                        probs = cute.math.exp2(scaled, fastmath=True)
                        thread_sum = _fragment_sum4(probs)
                        prev_sum = prev_sum * corr + thread_sum
                        for i in cutlass.range_constexpr(4):
                            prev_sum_arr[t * 4 + i] = prev_sum[i]

                        # P^T rows t * 16 + head of this stage; the SW128
                        # swizzle only reads address bits below the 2-KiB
                        # token pitch, so the sparse row pattern carries over.
                        for half in cutlass.range_constexpr(2):
                            half_probs = cutlass.Vector.from_elements(
                                tuple(probs[half * 8 + i] for i in range(8)),
                                dtype=ACC_DTYPE,
                            )
                            packed_p = half_probs.to(IO_DTYPE).bitcast(cutlass.Int32)
                            address_row = lane % cutlass.Int32(8)
                            address_segment = (lane // cutlass.Int32(8)) % cutlass.Int32(2)
                            row_start = (
                                stage * cutlass.Int32(N_DC * QK_N)
                                + cutlass.Int32(t * NUM_HEADS * QK_N)
                                + (address_segment * cutlass.Int32(8) + address_row)
                                * cutlass.Int32(QK_N)
                                + (warp_idx * cutlass.Int32(2) + cutlass.Int32(half))
                                * cutlass.Int32(16)
                            )
                            prims.stmatrix(
                                (sP.data_ptr() + row_start).apply_swizzle(smem_swizzle),
                                [packed_p[0], packed_p[1]],
                                prims.MMALayout.COL,
                                shape=prims.StoreShape.M16N8,
                            )
                    # This warp's TMEM reads of the stage are complete
                    # (tcgen05_wait LOAD above); one arrival per softmax warp.
                    if prims.elect_sync():
                        prims.mbarrier_arrive(s_empty.data_ptr() + stage, relaxed=True)
                    prims.fence_proxy(
                        prims.Proxy.ASYNC_SHARED,
                        space=prims.SharedSpace.shared_cta,
                    )
                    if prims.elect_sync():
                        prims.mbarrier_arrive(p_full.data_ptr() + stage)

                    if is_final_tile:
                        # Per token: the sparse path's shuffle tree (16, 8, 4)
                        # then the four warp partials in warp order.
                        for t in cutlass.range_constexpr(G_TOK):
                            warp_sum = cutlass.Vector.from_elements(
                                tuple(
                                    _warpgroup_row_sum(prev_sum_arr[t * 4 + i]) for i in range(4)
                                ),
                                dtype=ACC_DTYPE,
                            )
                            if lane < cutlass.Int32(4):
                                (
                                    softmax_sum.data_ptr()
                                    + warp_idx * cutlass.Int32(N_DC)
                                    + cutlass.Int32(t * NUM_HEADS)
                                    + lane * cutlass.Int32(4)
                                ).store(warp_sum, alignment=16)
                        prims.barrier_cta_sync(BAR_SM0, thread_count=128)
                        for t in cutlass.range_constexpr(G_TOK):
                            sum_base = cutlass.Int32(t * NUM_HEADS) + col_idx * cutlass.Int32(4)
                            prev_sum = (softmax_sum.data_ptr() + sum_base).load(
                                count=4, alignment=16
                            )
                            for sum_warp in cutlass.range_constexpr(1, 4):
                                prev_sum += (
                                    softmax_sum.data_ptr()
                                    + sum_warp * cutlass.Int32(N_DC)
                                    + sum_base
                                ).load(count=4, alignment=16)
                            for i in cutlass.range_constexpr(4):
                                prev_sum_arr[t * 4 + i] = prev_sum[i]
                        if warp_idx == cutlass.Int32(0):
                            if lane < cutlass.Int32(4):
                                stat_base = stage * cutlass.Int32(N_DC * 3)
                                for t in cutlass.range_constexpr(G_TOK):
                                    head_base = cutlass.Int32(t * NUM_HEADS) + lane * cutlass.Int32(
                                        4
                                    )
                                    sum_t = cutlass.Vector.from_elements(
                                        tuple(prev_sum_arr[t * 4 + i] for i in range(4)),
                                        dtype=ACC_DTYPE,
                                    )
                                    max_t = cutlass.Vector.from_elements(
                                        tuple(prev_max_arr[t * 4 + i] for i in range(4)),
                                        dtype=ACC_DTYPE,
                                    )
                                    corr_t = cutlass.Vector.from_elements(
                                        tuple(corr_arr[t * 4 + i] for i in range(4)),
                                        dtype=ACC_DTYPE,
                                    )
                                    (compact_stats.data_ptr() + stat_base + head_base).store(
                                        sum_t, alignment=16
                                    )
                                    (
                                        compact_stats.data_ptr()
                                        + stat_base
                                        + cutlass.Int32(N_DC)
                                        + head_base
                                    ).store(max_t, alignment=16)
                                    (
                                        compact_stats.data_ptr()
                                        + stat_base
                                        + cutlass.Int32(2 * N_DC)
                                        + head_base
                                    ).store(corr_t, alignment=16)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if lane == cutlass.Int32(0):
                                prims.mbarrier_arrive(c_full.data_ptr() + stage)
                    stage += cutlass.Int32(1)
                    if stage == cutlass.Int32(ST_S):
                        stage = cutlass.Int32(0)
                        phase ^= cutlass.Int32(1)
                    tile_iteration += cutlass.Int32(1)

                if prims.elect_sync():
                    prims.mbarrier_arrive(index_empty.data_ptr() + work_stage, relaxed=True)
                query_idx, work_valid = _consume_clc_response(
                    clc_response,
                    clc_full,
                    clc_empty,
                    work_iteration,
                )
                work_iteration += cutlass.Int32(1)
        else:
            work_iteration = cutlass.Int32(0)
            tile_iteration = cutlass.Int32(0)
            work_valid = cutlass.Boolean(True)
            while work_valid:
                work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
                work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(
                    index_full.data_ptr() + work_stage, work_phase
                ):
                    pass
                if (warp_idx == cutlass.Int32(0)) and (lane < cutlass.Int32(NUM_HEADS)):
                    softmax_max[lane] = _float_to_u32_for_atomic_max(ACC_DTYPE(-3.0e38))
                prims.barrier_cta_sync(BAR_SM0, thread_count=128)

                num_kt = active_k_tiles[work_stage]
                prev_max = cutlass.Vector.from_elements(
                    tuple(ACC_DTYPE(-3.0e38) for _ in range(4)),
                    dtype=ACC_DTYPE,
                )
                prev_sum = cutlass.Vector.from_elements(
                    tuple(ACC_DTYPE(0.0) for _ in range(4)),
                    dtype=ACC_DTYPE,
                )
                stage = tile_iteration % cutlass.Int32(ST_S)
                phase = (tile_iteration // cutlass.Int32(ST_S)) % cutlass.Int32(2)

                for kt in cutlass.range(0, num_kt, 1, unroll=1):
                    while not prims.mbarrier_try_wait_parity(s_full.data_ptr() + stage, phase):
                        pass
                    frag0 = prims.tcgen05_ld(
                        "16x256b",
                        cutlass.inttoptr(
                            addr0 + stage * cutlass.Int32(SCORE_N),
                            6,
                            ACC_DTYPE,
                        ),
                        num=2,
                    )
                    frag1 = prims.tcgen05_ld(
                        "16x256b",
                        cutlass.inttoptr(
                            addr1 + stage * cutlass.Int32(SCORE_N),
                            6,
                            ACC_DTYPE,
                        ),
                        num=2,
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    scores = cutlass.Vector.from_elements(
                        tuple(frag0[i] for i in range(8)) + tuple(frag1[i] for i in range(8)),
                        dtype=ACC_DTYPE,
                    )
                    if cutlass.const_expr(not all_sparse_indices_valid):
                        invalid_count = invalid_counts[
                            work_stage * cutlass.Int32(TOPK // QK_N) + kt
                        ]
                        if invalid_count > cutlass.Int32(0):
                            scores = _mask_scores_after_valid_prefix(
                                scores,
                                cutlass.Int32(QK_N) - invalid_count,
                                warp_idx,
                                row_idx,
                            )

                    thread_max = _fragment_max4(scores)
                    reduced_max = cutlass.Vector.from_elements(
                        tuple(_warpgroup_row_max(thread_max[i]) for i in range(4)),
                        dtype=ACC_DTYPE,
                    )
                    atomic_value = reduced_max[0]
                    if local_row_idx == cutlass.Int32(1):
                        atomic_value = reduced_max[1]
                    elif local_row_idx == cutlass.Int32(2):
                        atomic_value = reduced_max[2]
                    elif local_row_idx == cutlass.Int32(3):
                        atomic_value = reduced_max[3]
                    # TRT-LLM carries one running max across the full query.  Keep
                    # the shared atomic accumulator live across KV tiles so each
                    # tile needs one reduction barrier rather than a reduce/reset
                    # barrier pair.
                    max_slot = col_idx * cutlass.Int32(4) + local_row_idx
                    _smem_atomic_max_u32(
                        softmax_max.data_ptr() + max_slot,
                        _float_to_u32_for_atomic_max(atomic_value),
                    )
                    prims.barrier_cta_sync(BAR_SM0, thread_count=128)
                    encoded_tile_max = (softmax_max.data_ptr() + col_idx * cutlass.Int32(4)).load(
                        count=4, alignment=16
                    )
                    tile_max = cutlass.Vector.from_elements(
                        tuple(_u32_to_float_for_atomic_max(encoded_tile_max[i]) for i in range(4)),
                        dtype=ACC_DTYPE,
                    )
                    if (warp_idx == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                        prims.mbarrier_arrive(s_empty.data_ptr() + stage, relaxed=True)

                    while not prims.mbarrier_try_wait_parity(
                        c_empty.data_ptr() + stage,
                        phase ^ cutlass.Int32(1),
                    ):
                        pass
                    while not prims.mbarrier_try_wait_parity(
                        p_empty.data_ptr() + stage,
                        phase ^ cutlass.Int32(1),
                    ):
                        pass

                    new_max = tile_max
                    corr = cute.math.exp2(
                        (prev_max - new_max) * scale_log2,
                        fastmath=True,
                    )
                    is_final_tile = kt + cutlass.Int32(1) == num_kt
                    if not is_final_tile:
                        # Correction only consumes corr for intermediate tiles.  Publish it
                        # before exp/P so W4-7 can rescale the previous O tile in parallel
                        # with the remainder of this softmax step.  Do not touch this slot
                        # again after publication: correction may immediately release it.
                        if warp_idx == cutlass.Int32(0):
                            if lane < cutlass.Int32(4):
                                stat_base = stage * cutlass.Int32(NUM_HEADS * 3)
                                head_base = lane * cutlass.Int32(4)
                                (
                                    compact_stats.data_ptr()
                                    + stat_base
                                    + cutlass.Int32(2 * NUM_HEADS)
                                    + head_base
                                ).store(corr, alignment=16)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if lane == cutlass.Int32(0):
                                prims.mbarrier_arrive(c_full.data_ptr() + stage)
                    scaled = _scale_scores_for_fp8_p(scores, new_max, scale_log2)
                    probs = cute.math.exp2(scaled, fastmath=True)
                    thread_sum = _fragment_sum4(probs)
                    prev_sum = prev_sum * corr + thread_sum
                    if is_final_tile:
                        warp_sum = cutlass.Vector.from_elements(
                            tuple(_warpgroup_row_sum(prev_sum[i]) for i in range(4)),
                            dtype=ACC_DTYPE,
                        )
                        if lane < cutlass.Int32(4):
                            (
                                softmax_sum.data_ptr()
                                + warp_idx * cutlass.Int32(NUM_HEADS)
                                + lane * cutlass.Int32(4)
                            ).store(warp_sum, alignment=16)
                        prims.barrier_cta_sync(BAR_SM0, thread_count=128)
                        sum_base = col_idx * cutlass.Int32(4)
                        prev_sum = (softmax_sum.data_ptr() + sum_base).load(count=4, alignment=16)
                        for sum_warp in cutlass.range_constexpr(1, 4):
                            prev_sum += (
                                softmax_sum.data_ptr()
                                + sum_warp * cutlass.Int32(NUM_HEADS)
                                + sum_base
                            ).load(count=4, alignment=16)
                    prev_max = new_max

                    for half in cutlass.range_constexpr(2):
                        half_probs = cutlass.Vector.from_elements(
                            tuple(probs[half * 8 + i] for i in range(8)),
                            dtype=ACC_DTYPE,
                        )
                        packed_p = half_probs.to(IO_DTYPE).bitcast(cutlass.Int32)
                        address_row = lane % cutlass.Int32(8)
                        address_segment = (lane // cutlass.Int32(8)) % cutlass.Int32(2)
                        row_start = (
                            stage * cutlass.Int32(NUM_HEADS * QK_N)
                            + (address_segment * cutlass.Int32(8) + address_row)
                            * cutlass.Int32(QK_N)
                            + (warp_idx * cutlass.Int32(2) + cutlass.Int32(half))
                            * cutlass.Int32(16)
                        )
                        prims.stmatrix(
                            (sP.data_ptr() + row_start).apply_swizzle(smem_swizzle),
                            [packed_p[0], packed_p[1]],
                            prims.MMALayout.COL,
                            shape=prims.StoreShape.M16N8,
                        )
                    prims.fence_proxy(
                        prims.Proxy.ASYNC_SHARED,
                        space=prims.SharedSpace.shared_cta,
                    )
                    if prims.elect_sync():
                        prims.mbarrier_arrive(p_full.data_ptr() + stage)
                    if is_final_tile:
                        if warp_idx == cutlass.Int32(0):
                            if lane < cutlass.Int32(4):
                                stat_base = stage * cutlass.Int32(NUM_HEADS * 3)
                                head_base = lane * cutlass.Int32(4)
                                (compact_stats.data_ptr() + stat_base + head_base).store(
                                    prev_sum, alignment=16
                                )
                                (
                                    compact_stats.data_ptr()
                                    + stat_base
                                    + cutlass.Int32(NUM_HEADS)
                                    + head_base
                                ).store(prev_max, alignment=16)
                                (
                                    compact_stats.data_ptr()
                                    + stat_base
                                    + cutlass.Int32(2 * NUM_HEADS)
                                    + head_base
                                ).store(corr, alignment=16)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if lane == cutlass.Int32(0):
                                prims.mbarrier_arrive(c_full.data_ptr() + stage)
                    stage += cutlass.Int32(1)
                    if stage == cutlass.Int32(ST_S):
                        stage = cutlass.Int32(0)
                        phase ^= cutlass.Int32(1)
                    tile_iteration += cutlass.Int32(1)

                if prims.elect_sync():
                    prims.mbarrier_arrive(index_empty.data_ptr() + work_stage, relaxed=True)
                query_idx, work_valid = _consume_clc_response(
                    clc_response,
                    clc_full,
                    clc_empty,
                    work_iteration,
                )
                work_iteration += cutlass.Int32(1)

    # =====================================================================
    # W4-7 — correction / rescale / epilogue
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_CORR0)) and (warp_idx < cutlass.Int32(W_CORR0 + 4)):
        prims.setmaxregister(176, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        corr_warp = warp_idx - cutlass.Int32(W_CORR0)
        base = prims.TmemAddr(tmem_raw)
        my_addr = ((base.row_id + corr_warp * cutlass.Int32(32)) << 16) | base.col_id
        corr_thread = corr_warp * cutlass.Int32(32) + lane
        o_ptr = o.iterator.raw_ptr()
        if cutlass.const_expr(dense_causal):
            # Dense-causal correction / epilogue.  O^T
            # stripe s of token t occupies TMEM columns
            # TMEM_O_OFF_DC + s * 64 + t * 16 (lanes = d within the stripe,
            # column = logical head).  Rescale per tile: 4 stripes x 4 token
            # chunks of 32x32b.x16 ld / mul / st.  Epilogue: two halves of two
            # tokens through the 32-KiB partial_o staging; stores only for
            # real tokens (t < n_tok, CTA-uniform).  num_kt >= 1 always (W9),
            # so there is no zero-output path in this mode.
            r_sum_arr = cutlass.Array(ACC_DTYPE, G_TOK * 4, space=cutlass.AddressSpace.rmem)
            work_iteration = cutlass.Int32(0)
            tile_iteration = cutlass.Int32(0)
            work_valid = cutlass.Boolean(True)
            while work_valid:
                work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
                work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(
                    index_full.data_ptr() + work_stage, work_phase
                ):
                    pass
                num_kt = active_k_tiles[work_stage]
                meta_base = work_stage * cutlass.Int32(GROUP_META_WORDS)
                group_q0 = group_meta[meta_base]
                group_n_tok = group_meta[meta_base + cutlass.Int32(1)]
                col_group = lane & cutlass.Int32(3)
                cst = tile_iteration % cutlass.Int32(ST_C)
                cph = (tile_iteration // cutlass.Int32(ST_C)) % cutlass.Int32(2)
                for kt in cutlass.range(0, num_kt, 1, unroll=1):
                    while not prims.mbarrier_try_wait_parity(c_full.data_ptr() + cst, cph):
                        pass
                    stat_base = cst * cutlass.Int32(N_DC * 3)
                    if kt + cutlass.Int32(1) == num_kt:
                        # Per token, the four packed sums of this lane's
                        # 16x256b fragment (packed heads col_group * 4 ..).
                        for t in cutlass.range_constexpr(G_TOK):
                            sums = (
                                compact_stats.data_ptr()
                                + stat_base
                                + cutlass.Int32(t * NUM_HEADS)
                                + col_group * cutlass.Int32(4)
                            ).load(count=4, alignment=16)
                            for i in cutlass.range_constexpr(4):
                                r_sum_arr[t * 4 + i] = sums[i]
                    # corr of every token in logical head order (the O^T
                    # columns), read before releasing the stats slot.
                    corrs = []
                    for t in cutlass.range_constexpr(G_TOK):
                        packed_corr = (
                            compact_stats.data_ptr()
                            + stat_base
                            + cutlass.Int32(2 * N_DC + t * NUM_HEADS)
                        ).load(count=NUM_HEADS, alignment=16)
                        corrs.append(
                            cutlass.Vector.from_elements(
                                tuple(
                                    packed_corr[packed_head]
                                    for packed_head in H16_LOGICAL_TO_PACKED_STATS
                                ),
                                dtype=ACC_DTYPE,
                            )
                        )
                    if prims.elect_sync():
                        prims.mbarrier_arrive(c_empty.data_ptr() + cst, relaxed=True)

                    oph = (tile_iteration + kt) % cutlass.Int32(2)
                    if kt > cutlass.Int32(0):
                        while not prims.mbarrier_try_wait_parity(o_full, oph ^ cutlass.Int32(1)):
                            pass
                        for s in cutlass.range_constexpr(N_V):
                            for t in cutlass.range_constexpr(G_TOK):
                                tp = cutlass.inttoptr(
                                    my_addr + TMEM_O_OFF_DC + s * N_DC + t * NUM_HEADS,
                                    6,
                                    ACC_DTYPE,
                                )
                                tile_v = prims.tcgen05_ld("32x32b", tp, num=NUM_HEADS)
                                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                                prims.tcgen05_st("32x32b", tp, tile_v * corrs[t])
                        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(o_empty, relaxed=True)
                    cst += cutlass.Int32(1)
                    if cst == cutlass.Int32(ST_C):
                        cst = cutlass.Int32(0)
                        cph ^= cutlass.Int32(1)

                final_phase = (tile_iteration + num_kt - cutlass.Int32(1)) % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(o_full, final_phase):
                    pass
                for half in cutlass.range_constexpr(2):
                    # CTA-uniform skip of a half whose two tokens are dummies.
                    if cutlass.Int32(half * 2) < group_n_tok:
                        for t_half in cutlass.range_constexpr(2):
                            t: cutlass.Constexpr[int] = half * 2 + t_half
                            r_sum = cutlass.Vector.from_elements(
                                tuple(r_sum_arr[t * 4 + i] for i in range(4)),
                                dtype=ACC_DTYPE,
                            )
                            inv = out_scale / r_sum
                            scale = cutlass.Vector.from_elements(
                                (
                                    inv[0],
                                    inv[1],
                                    inv[0],
                                    inv[1],
                                    inv[2],
                                    inv[3],
                                    inv[2],
                                    inv[3],
                                ),
                                dtype=ACC_DTYPE,
                            )
                            for s in cutlass.range_constexpr(N_V):
                                tmem_addr = my_addr + TMEM_O_OFF_DC + s * N_DC + t * NUM_HEADS
                                tp_lo = cutlass.inttoptr(tmem_addr, 6, ACC_DTYPE)
                                tp_hi = cutlass.inttoptr(
                                    tmem_addr + (cutlass.Int32(16) << cutlass.Int32(16)),
                                    6,
                                    ACC_DTYPE,
                                )
                                tile_lo = prims.tcgen05_ld("16x256b", tp_lo, num=2)
                                tile_hi = prims.tcgen05_ld("16x256b", tp_hi, num=2)
                                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                                packed_lo = (tile_lo * scale).to(OUT_DTYPE).bitcast(cutlass.Int32)
                                packed_hi = (tile_hi * scale).to(OUT_DTYPE).bitcast(cutlass.Int32)

                                slice_idx = corr_warp // cutlass.Int32(2)
                                warp_in_slice = corr_warp % cutlass.Int32(2)
                                matrix_idx = lane // cutlass.Int32(8)
                                thread_row = lane % cutlass.Int32(8)
                                matrix_row = matrix_idx // cutlass.Int32(2)
                                matrix_col_base = warp_in_slice * cutlass.Int32(
                                    4
                                ) + matrix_idx % cutlass.Int32(2)
                                # partial_o[stripe][token-in-half][2 x 16 rows]
                                # [128 B]: the sparse token image at a 4-KiB
                                # token pitch and an 8-KiB stripe pitch.
                                for row_half in cutlass.range_constexpr(2):
                                    matrix_col = matrix_col_base + cutlass.Int32(row_half * 2)
                                    smem_offset_bytes = (
                                        cutlass.Int32(
                                            s * partial_o_tokens * NUM_HEADS * PV_M * 2
                                            + t_half * NUM_HEADS * PV_M * 2
                                        )
                                        + slice_idx * cutlass.Int32(16 * 128)
                                        + (matrix_row * cutlass.Int32(8) + thread_row)
                                        * cutlass.Int32(128)
                                        + (matrix_col ^ thread_row) * cutlass.Int32(16)
                                    )
                                    store_regs = packed_lo
                                    if cutlass.const_expr(row_half == 1):
                                        store_regs = packed_hi
                                    prims.stmatrix(
                                        partial_o.data_ptr()
                                        + smem_offset_bytes // cutlass.Int32(2),
                                        [
                                            store_regs[0],
                                            store_regs[1],
                                            store_regs[2],
                                            store_regs[3],
                                        ],
                                        prims.MMALayout.COL,
                                        shape=prims.StoreShape.M8N8,
                                    )
                        prims.fence_proxy(
                            prims.Proxy.ASYNC_SHARED,
                            space=prims.SharedSpace.shared_cta,
                        )
                        prims.barrier_cta_sync(BAR_CORR, thread_count=128)
                        for s in cutlass.range_constexpr(N_V):
                            for t_half in cutlass.range_constexpr(2):
                                t: cutlass.Constexpr[int] = half * 2 + t_half
                                # Only real tokens are stored (CTA-uniform).
                                if cutlass.Int32(t) < group_n_tok:
                                    head = corr_thread // cutlass.Int32(8)
                                    output_vector = corr_thread % cutlass.Int32(8)
                                    smem_row = head + (
                                        output_vector // cutlass.Int32(4)
                                    ) * cutlass.Int32(NUM_HEADS)
                                    logical_sector = (
                                        output_vector % cutlass.Int32(4)
                                    ) * cutlass.Int32(2)
                                    swizzle = smem_row & cutlass.Int32(7)
                                    smem_base = cutlass.Int32(
                                        s * partial_o_tokens * NUM_HEADS * PV_M * 2
                                        + t_half * NUM_HEADS * PV_M * 2
                                    ) + smem_row * cutlass.Int32(128)
                                    load_offset0 = smem_base + (
                                        logical_sector ^ swizzle
                                    ) * cutlass.Int32(16)
                                    load_offset1 = smem_base + (
                                        (logical_sector + cutlass.Int32(1)) ^ swizzle
                                    ) * cutlass.Int32(16)
                                    values0 = (
                                        partial_o.data_ptr() + load_offset0 // cutlass.Int32(2)
                                    ).load(count=8, alignment=16)
                                    values1 = (
                                        partial_o.data_ptr() + load_offset1 // cutlass.Int32(2)
                                    ).load(count=8, alignment=16)
                                    values = cutlass.Vector.from_elements(
                                        tuple(values0[i] for i in range(8))
                                        + tuple(values1[i] for i in range(8)),
                                        dtype=OUT_DTYPE,
                                    )
                                    output_row = (
                                        (group_q0 + cutlass.Int32(t)) * cutlass.Int32(NUM_HEADS)
                                        + head
                                    ) * cutlass.Int32(L)
                                    (
                                        o_ptr
                                        + output_row
                                        + cutlass.Int32(s * PV_M)
                                        + output_vector * cutlass.Int32(16)
                                    ).store(values, alignment=32)
                        prims.barrier_cta_sync(BAR_CORR, thread_count=128)
                if prims.elect_sync():
                    prims.mbarrier_arrive(o_empty, relaxed=True)

                if prims.elect_sync():
                    prims.mbarrier_arrive(index_empty.data_ptr() + work_stage, relaxed=True)
                query_idx, work_valid = _consume_clc_response(
                    clc_response,
                    clc_full,
                    clc_empty,
                    work_iteration,
                )
                work_iteration += cutlass.Int32(1)
                tile_iteration += num_kt
        else:
            work_iteration = cutlass.Int32(0)
            tile_iteration = cutlass.Int32(0)
            work_valid = cutlass.Boolean(True)
            while work_valid:
                work_stage = work_iteration % cutlass.Int32(CLC_STAGES)
                work_phase = (work_iteration // cutlass.Int32(CLC_STAGES)) % cutlass.Int32(2)
                while not prims.mbarrier_try_wait_parity(
                    index_full.data_ptr() + work_stage, work_phase
                ):
                    pass
                num_kt = active_k_tiles[work_stage]
                col_group = lane & cutlass.Int32(3)
                r_sum = cutlass.Vector.from_elements(
                    tuple(ACC_DTYPE(0.0) for _ in range(4)),
                    dtype=ACC_DTYPE,
                )
                cst = tile_iteration % cutlass.Int32(ST_C)
                cph = (tile_iteration // cutlass.Int32(ST_C)) % cutlass.Int32(2)
                for kt in cutlass.range(0, num_kt, 1, unroll=1):
                    while not prims.mbarrier_try_wait_parity(c_full.data_ptr() + cst, cph):
                        pass
                    stat_base = cst * cutlass.Int32(NUM_HEADS * 3)
                    if kt + cutlass.Int32(1) == num_kt:
                        # Match TRT-LLM's 16x256b fragment ownership: a lane only
                        # needs the four sums used by its TMEM fragment.
                        packed0 = col_group * cutlass.Int32(4)
                        packed1 = packed0 + cutlass.Int32(1)
                        packed2 = packed0 + cutlass.Int32(2)
                        packed3 = packed2 + cutlass.Int32(1)
                        r_sum = cutlass.Vector.from_elements(
                            (
                                compact_stats[stat_base + packed0],
                                compact_stats[stat_base + packed1],
                                compact_stats[stat_base + packed2],
                                compact_stats[stat_base + packed3],
                            ),
                            dtype=ACC_DTYPE,
                        )
                    packed_corr = (compact_stats.data_ptr() + stat_base + 2 * NUM_HEADS).load(
                        count=NUM_HEADS, alignment=16
                    )
                    corr = cutlass.Vector.from_elements(
                        tuple(
                            packed_corr[packed_head] for packed_head in H16_LOGICAL_TO_PACKED_STATS
                        ),
                        dtype=ACC_DTYPE,
                    )
                    if prims.elect_sync():
                        prims.mbarrier_arrive(c_empty.data_ptr() + cst, relaxed=True)

                    oph = (tile_iteration + kt) % cutlass.Int32(2)
                    if kt > cutlass.Int32(0):
                        while not prims.mbarrier_try_wait_parity(o_full, oph ^ cutlass.Int32(1)):
                            pass
                        for s in cutlass.range_constexpr(N_V):
                            tp = cutlass.inttoptr(
                                my_addr + TMEM_O_OFF + s * PV_N,
                                6,
                                ACC_DTYPE,
                            )
                            tile_v = prims.tcgen05_ld("32x32b", tp, num=NUM_HEADS)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                            prims.tcgen05_st("32x32b", tp, tile_v * corr)
                        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(o_empty, relaxed=True)
                    cst += cutlass.Int32(1)
                    if cst == cutlass.Int32(ST_C):
                        cst = cutlass.Int32(0)
                        cph ^= cutlass.Int32(1)

                if num_kt == cutlass.Int32(0):
                    # No PV operation exists to publish o_full for an all-invalid
                    # row. Store the mathematically defined zero output directly;
                    # the cyclic TMEM and SMEM barriers remain at their current phase.
                    zero16 = cutlass.Vector.from_elements(
                        tuple(OUT_DTYPE(0) for _ in range(16)), dtype=OUT_DTYPE
                    )
                    for vector_idx in cutlass.range(
                        corr_thread,
                        VALID_ROWS * (L // 16),
                        128,
                        unroll=1,
                    ):
                        head = vector_idx // cutlass.Int32(L // 16)
                        col = (vector_idx % cutlass.Int32(L // 16)) * cutlass.Int32(16)
                        output_row = (query_idx * cutlass.Int32(VALID_ROWS) + head) * cutlass.Int32(
                            L
                        )
                        (o_ptr + output_row + col).store(zero16, alignment=32)
                else:
                    final_phase = (tile_iteration + num_kt - cutlass.Int32(1)) % cutlass.Int32(2)
                    while not prims.mbarrier_try_wait_parity(o_full, final_phase):
                        pass
                    inv = out_scale / r_sum
                    scale = cutlass.Vector.from_elements(
                        (
                            inv[0],
                            inv[1],
                            inv[0],
                            inv[1],
                            inv[2],
                            inv[3],
                            inv[2],
                            inv[3],
                        ),
                        dtype=ACC_DTYPE,
                    )
                    for s in cutlass.range_constexpr(N_V):
                        tmem_addr = my_addr + TMEM_O_OFF + s * PV_N
                        tp_lo = cutlass.inttoptr(tmem_addr, 6, ACC_DTYPE)
                        tp_hi = cutlass.inttoptr(
                            tmem_addr + (cutlass.Int32(16) << cutlass.Int32(16)),
                            6,
                            ACC_DTYPE,
                        )
                        tile_lo = prims.tcgen05_ld("16x256b", tp_lo, num=2)
                        tile_hi = prims.tcgen05_ld("16x256b", tp_hi, num=2)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        packed_lo = (tile_lo * scale).to(OUT_DTYPE).bitcast(cutlass.Int32)
                        packed_hi = (tile_hi * scale).to(OUT_DTYPE).bitcast(cutlass.Int32)

                        slice_idx = corr_warp // cutlass.Int32(2)
                        warp_in_slice = corr_warp % cutlass.Int32(2)
                        matrix_idx = lane // cutlass.Int32(8)
                        thread_row = lane % cutlass.Int32(8)
                        matrix_row = matrix_idx // cutlass.Int32(2)
                        matrix_col_base = warp_in_slice * cutlass.Int32(
                            4
                        ) + matrix_idx % cutlass.Int32(2)
                        for row_half in cutlass.range_constexpr(2):
                            matrix_col = matrix_col_base + cutlass.Int32(row_half * 2)
                            smem_offset_bytes = (
                                cutlass.Int32(s * NUM_HEADS * PV_M * 2)
                                + slice_idx * cutlass.Int32(16 * 128)
                                + (matrix_row * cutlass.Int32(8) + thread_row) * cutlass.Int32(128)
                                + (matrix_col ^ thread_row) * cutlass.Int32(16)
                            )
                            store_regs = packed_lo
                            if cutlass.const_expr(row_half == 1):
                                store_regs = packed_hi
                            prims.stmatrix(
                                partial_o.data_ptr() + smem_offset_bytes // cutlass.Int32(2),
                                [
                                    store_regs[0],
                                    store_regs[1],
                                    store_regs[2],
                                    store_regs[3],
                                ],
                                prims.MMALayout.COL,
                                shape=prims.StoreShape.M8N8,
                            )
                    prims.fence_proxy(
                        prims.Proxy.ASYNC_SHARED,
                        space=prims.SharedSpace.shared_cta,
                    )
                    prims.barrier_cta_sync(BAR_CORR, thread_count=128)
                    for s in cutlass.range_constexpr(N_V):
                        # SW128 permutes each 16-byte sector.  Load two independent
                        # 128-bit sectors and combine them for one 256-bit GMEM store.
                        head = corr_thread // cutlass.Int32(8)
                        output_vector = corr_thread % cutlass.Int32(8)
                        smem_row = head + (output_vector // cutlass.Int32(4)) * cutlass.Int32(
                            NUM_HEADS
                        )
                        logical_sector = (output_vector % cutlass.Int32(4)) * cutlass.Int32(2)
                        swizzle = smem_row & cutlass.Int32(7)
                        smem_base = cutlass.Int32(
                            s * NUM_HEADS * PV_M * 2
                        ) + smem_row * cutlass.Int32(128)
                        load_offset0 = smem_base + (logical_sector ^ swizzle) * cutlass.Int32(16)
                        load_offset1 = smem_base + (
                            (logical_sector + cutlass.Int32(1)) ^ swizzle
                        ) * cutlass.Int32(16)
                        values0 = (partial_o.data_ptr() + load_offset0 // cutlass.Int32(2)).load(
                            count=8, alignment=16
                        )
                        values1 = (partial_o.data_ptr() + load_offset1 // cutlass.Int32(2)).load(
                            count=8, alignment=16
                        )
                        values = cutlass.Vector.from_elements(
                            tuple(values0[i] for i in range(8))
                            + tuple(values1[i] for i in range(8)),
                            dtype=OUT_DTYPE,
                        )
                        output_row = (query_idx * cutlass.Int32(VALID_ROWS) + head) * cutlass.Int32(
                            L
                        )
                        (
                            o_ptr
                            + output_row
                            + cutlass.Int32(s * PV_M)
                            + output_vector * cutlass.Int32(16)
                        ).store(values, alignment=32)
                    prims.barrier_cta_sync(BAR_CORR, thread_count=128)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(o_empty, relaxed=True)

                if prims.elect_sync():
                    prims.mbarrier_arrive(index_empty.data_ptr() + work_stage, relaxed=True)
                query_idx, work_valid = _consume_clc_response(
                    clc_response,
                    clc_full,
                    clc_empty,
                    work_iteration,
                )
                work_iteration += cutlass.Int32(1)
                tile_iteration += num_kt

    # The scheduler and all role warps consume the final invalid response
    # before reconverging for the single persistent TMEM deallocation.
    prims.barrier_cta_sync(0)
    if tidx == 0:
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    if warp_idx == W_MMA_PV:
        prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_1")


@cute.jit
def _fragment_max4(values):
    return cutlass.Vector.from_elements(
        (
            cute.arch.fmax(
                cute.arch.fmax(values[0], values[2]),
                cute.arch.fmax(values[8], values[10]),
            ),
            cute.arch.fmax(
                cute.arch.fmax(values[1], values[3]),
                cute.arch.fmax(values[9], values[11]),
            ),
            cute.arch.fmax(
                cute.arch.fmax(values[4], values[6]),
                cute.arch.fmax(values[12], values[14]),
            ),
            cute.arch.fmax(
                cute.arch.fmax(values[5], values[7]),
                cute.arch.fmax(values[13], values[15]),
            ),
        ),
        dtype=ACC_DTYPE,
    )


@cute.jit
def _fragment_sum4(values):
    return cutlass.Vector.from_elements(
        (
            values[0] + values[2] + values[8] + values[10],
            values[1] + values[3] + values[9] + values[11],
            values[4] + values[6] + values[12] + values[14],
            values[5] + values[7] + values[13] + values[15],
        ),
        dtype=ACC_DTYPE,
    )


@cute.jit
def _warpgroup_row_max(value: cutlass.Float32) -> cutlass.Float32:
    for offset in (4, 8):
        value = cute.arch.fmax(
            value,
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=value,
                offset=offset,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
            ),
        )
    return value


@cute.jit
def _warpgroup_row_sum(value: cutlass.Float32) -> cutlass.Float32:
    for offset in (16, 8, 4):
        value += prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=value,
            offset=offset,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        )
    return value


def dense_causal_bounds(
    seq_lens: torch.Tensor,
    cached_lens: torch.Tensor | None,
    total_q: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query ``(cu_seqlen_ks, cu_seqlen_ke)`` of a dense-causal context step (top-k skipped).

    ``seq_lens`` int32 ``[B]`` = new (uncached) query rows per request, ``cached_lens`` int32 ``[B]``
    or ``None`` = KV-block-reuse prefix rows already in the cache.  Row ``t`` of request ``b`` may
    attend ``ke = cached_lens[b] + local_pos + 1`` KV rows (``ks = 0``): the serving adapter passes
    DSA's global top-k indices, which are dense ``0 .. cached + local_pos`` for such rows, and the
    kernel reads ``clamp(ke - ks, 0, 2048)`` of them.  A re-sent prompt whose prefix is cached therefore
    gets, for its new rows, exactly the bounds the no-reuse run had for the same rows.
    """
    if seq_lens.dtype != torch.int32 or seq_lens.ndim != 1:
        raise ValueError(
            f"seq_lens must be int32 [B], got {seq_lens.dtype} {tuple(seq_lens.shape)}"
        )
    if int(seq_lens.sum().item()) != int(total_q):
        raise ValueError(f"seq_lens sum {int(seq_lens.sum().item())} != total_q {total_q}")
    device = seq_lens.device
    starts = torch.cumsum(seq_lens, 0) - seq_lens
    pos = torch.arange(total_q, dtype=torch.int32, device=device) - torch.repeat_interleave(
        starts, seq_lens, output_size=total_q
    )
    if cached_lens is not None:
        if cached_lens.dtype != torch.int32 or tuple(cached_lens.shape) != tuple(seq_lens.shape):
            raise ValueError(
                f"cached_lens must be int32 {tuple(seq_lens.shape)}, got {cached_lens.dtype} "
                f"{tuple(cached_lens.shape)}"
            )
        pos = pos + torch.repeat_interleave(cached_lens, seq_lens, output_size=total_q)
    ks = torch.zeros(total_q, dtype=torch.int32, device=device)
    ke = (pos + 1).to(torch.int32).contiguous()
    return ks, ke


def dense_causal_groups(seq_lens: torch.Tensor) -> torch.Tensor:
    """Build the dense-causal work-item table ``[G, 2] = (q0, n_tok)``.

    ``seq_lens`` is the CPU int32 ``[B]`` vector of new (uncached) query rows per request, in
    the order the rows are laid out in ``quant_q_buffer``.  Request ``r`` with rows
    ``[cu[r], cu[r+1])`` contributes the groups ``q0 = cu[r] + 4 * j``, ``n_tok = min(4, cu[r+1]
    - q0)`` in request-major order, so consecutive CTAs of one request share K/V tiles in L2.
    The result is a contiguous CPU int32 tensor with ``G = sum_r ceil(len_r / 4) >= 1`` rows;
    :func:`prepare_runner` validates it against ``total_q`` and uploads it to the Q
    device (one H2D copy per context step, shared by every layer).
    """
    if seq_lens.dtype != torch.int32 or seq_lens.ndim != 1:
        raise ValueError(
            f"seq_lens must be int32 [B], got {seq_lens.dtype} {tuple(seq_lens.shape)}"
        )
    if seq_lens.device.type != "cpu":
        raise ValueError("seq_lens must live on the CPU (the route holds host lengths)")
    rows: list[tuple[int, int]] = []
    q_start = 0
    for length in seq_lens.tolist():
        if length < 0:
            raise ValueError(f"seq_lens must be non-negative, got {length}")
        for local in range(0, length, G_TOK):
            rows.append((q_start + local, min(G_TOK, length - local)))
        q_start += length
    if not rows:
        raise ValueError("seq_lens must describe at least one query row")
    return torch.tensor(rows, dtype=torch.int32).contiguous()
