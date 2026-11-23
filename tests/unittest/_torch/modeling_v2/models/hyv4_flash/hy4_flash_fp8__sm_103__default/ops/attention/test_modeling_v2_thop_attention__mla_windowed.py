# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for Hy4-Flash's `mla_windowed` thop_attention variant.

The variant certifies the checkpoint's thirty sliding-window layers: MLA's
context and generation phases at 64 q heads on one rank (16 under tp4, 32
under tp2), over the same paged latent pool geometry as the twelve indexed
sparse layers, with a per-head fp32 sink, a 512-token window and the
interleaved-mRoPE table the model builds outside the op. It certifies the
target sparse layers' own decode too -- one generation row per request, the
indexer's selection as `sparse_attn_indices` -- and the DFly2 drafter's block
call over the same pool.

Every windowed cell drives a real paged latent pool through both phases: a
context call (a fresh append, or a chunk over a cached prefix on the
paged-context path), a decode row appended at the history's end, and one
generation call. The sparse cells drive one generation call each: rows
appended by the harness, their selections naming the pool rows to read. Each
output is gated against the entry's own `reference` over the operands the
kernel reads -- the append path's q/k rope slices rebuilt from the caller's
pre-call tensors, the paged-context path's operands passed through, the
windowed decode's K/V read out of the pool, and the sparse rows the row's own
selection names.

The pool is a ring, as the engine gives a 512-token windowed layer sixteen
pages per request: the append writes every token at its absolute page and
slot, a history longer than the window overwrites the aged-out slots in
place, and a decode therefore sees the last `min(history, 512)` latent rows.
`check_ring` pins that mapping by reading the pool back bitwise. Three
controls measure that the distinguishing inputs are live: the window against a
full-causal reference, the sink against a sink-free one, and the interleaved
mRoPE table against the plain duplicated one at text-only positions.

One shape is measured rather than certified: `mrope_rotary_cos_sin` /
`mrope_position_deltas` are carried but not read. An NVFP4 pool (`quant_mode`
8192) is uncertified here: its kernels exist for the paged-context path only
and its append needs a second e4m3 scale pool, which this variant does not
build -- the catalog entry certifies that shape.
"""

import math
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.cache_manager import (  # noqa: E501
    NO_PICK,
    DraftPicks,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.thop_attention__mla_windowed import (  # noqa: E501
    MLA_CONTEXT,
    MLA_GENERATION,
    QM_FP8_KV_CACHE,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.thop_attention__mla_windowed import (  # noqa: E501
    thop_attention as op,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.spec.dfly_block_prepare import (  # noqa: E501
    BLOCK_WIDTH,
    PAD_SLOT_ID,
    TARGET_TOPK,
    dfly_block_indices,
)

__extra_import_path__ = ["../../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the mla_windowed entry requires a CUDA device"

# Hy4-Flash's sliding-window MLA layer, from the checkpoint's text_config: the
# sparse layers run 32 heads (8 per rank at tp4), the sliding-window layers 64
# (16 at tp4). Latent rows are [compressed_kv 512 | k_pe 64], context qk
# [nope 192 | rope 64], context v 256.
NUM_HEADS_SWA = 64
NUM_HEADS_SWA_TP4 = 16
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 256
V_HEAD_DIM = 256
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
Q_LORA_RANK = 1024
TOKENS_PER_BLOCK = 32
# text_config.sliding_window, and the per-layer rope base of the SWA layers
# (the sparse layers use 1e7). Interleaved mRoPE sections (t, h, w).
WINDOW = 512
ROPE_THETA_SWA = 10_000.0
MROPE_SECTION = (12, 10, 10)
MASK_CAUSAL = 1
# QuantMode.FP8_KV_CACHE: the one quantized pool this site runs.
QUANT_MODE_FP8_KV_CACHE = QM_FP8_KV_CACHE
# PositionEmbeddingType.mrope, as the MLA backend passes it for Hy4.
POSITION_EMBEDDING_TYPE_MROPE = 9
# KvCacheConfig.max_num_tokens defaults to 8192, the largest row count one
# context call carries; it also bounds every position these cells rope at.
MAX_SEQ_LEN = 8192

# ── the DFly2 drafter's block, from the landed `ops/spec/dfly_block_prepare.py`
# and the call site that feeds it ──
# The block's width and the indexer's top-k are the entry's own constants
# (`BLOCK_WIDTH` 6, `TARGET_TOPK` 2048); the two sparse layers' heads are the
# checkpoint's 32 under tp4, and the block's five draft layers are
# `dfly_target_kv_layers` [25, 29, 33, 37, 41].
BLOCK_ROWS = BLOCK_WIDTH
SPARSE_HEADS_TP4 = 8
DRAFT_LAYERS = (25, 29, 33, 37, 41)
#: The width of the selection buffer the *caller* allocates per row: the
#: drafter's `dims.py` means to round `index_topk + block_width` up to its
#: `DRAFT_INDEX_ALIGN` 128 (`round_up(2048 + 6, 128) = 2176`), and the entry is
#: handed `num_sparse_topk = int(packed.shape[1])`, that same width. A
#: distribution over the *width* is therefore what the op is being asked for;
#: the picks themselves are only ever `TARGET_TOPK` wide.
INDEX_WIDTH = -(-(TARGET_TOPK + BLOCK_ROWS) // 128) * 128
#: The history each request brings to the block, the rows its top-`TARGET_TOPK`
#: selection is taken from: long enough that the selection is a selection, and a
#: whole number of pages so the ring below holds the block's rows without
#: wrapping. A request's picks state is `TARGET_TOPK` wide whatever its history.
DRAFT_HISTORY = 2304

# ── the sparse layers' own decode, from `dims.py` and `MlaLayer._attend` ──
# The sparse layers run `heads_sparse` 32 q heads, 32 / attn_tp per rank: 16 at
# the tp2 width the disaggregated deployment's context engine runs. One kv
# head, the absorbed latent as `head_size` and the key count per row from the
# device lengths.
SPARSE_HEADS_TP2 = 16
#: The two histories the cell's requests bring to their single decode row: one
#: longer than the selection, so its list is a real `TARGET_TOPK` of the row's
#: own history, and one shorter, so its list is the whole history followed by
#: the `-1` padding. Both are whole pages.
SPARSE_HISTORY_LONG = 2560
SPARSE_HISTORY_SHORT = 1024

# The bands. Both are the catalog test's for this same op and these same two
# paths, derived there: 8 bf16 ulp per element against the row's largest
# magnitude and 4 of relative RMS in aggregate for a bf16 pool (a two-GEMM
# chain's accumulation-order band), and the fp8 MLA band (atol 2**-3,
# rtol 2**-4, 4 bf16 ulp RMS) for an e4m3 pool, where the kernel additionally
# rounds the softmax probabilities to e4m3 -- an operand the fp8 references
# model with `e4m3_probabilities=True` (measured at 1.83 / 0.01 bf16 ulp RMS
# against 4.45 / 4.26 unmodelled).
BF16_ELEMENT_ULP = 8.0
BF16_RMS_ULP = 4.0
FP8_MLA_ATOL = 2**-3
FP8_MLA_RTOL = 2**-4
ULP_BF16 = 2.0**-8

# The append's roped k_pe half, gated as the catalog test gates it for this
# same kernel: one ulp of the pool dtype plus a bit-exact-fraction floor (8
# elements in 495360 measured one bf16 ulp off, the kernel evaluating the
# rotation in its own order; an e4m3 pool's coarser grid has never been
# approached). The compressed-kv half beside it is bitwise.
LATENT_ROPE_ULP_RTOL = 2**-7
LATENT_ROPE_E4M3_ULP_RTOL = 2**-3
LATENT_ROPE_MAX_INEXACT_FRACTION = 1e-3

_CAPABILITY_TO_ARCH = {
    (10, 0): Arch.SM_100,
    (10, 3): Arch.SM_103,
    (10, 7): Arch.SM_107,
}


def _arch_of_this_device() -> Arch:
    cap = torch.cuda.get_device_capability()
    arch = _CAPABILITY_TO_ARCH.get(cap)
    assert arch is not None, f"device capability {cap} has no Arch entry"
    assert arch in op.ARCHS, (
        f"this device is {arch.name}, and the variant certifies {sorted(a.name for a in op.ARCHS)}"
    )
    return arch


def _seed_of(cell) -> int:
    """A cell's own seed, stable across processes (str hash is not)."""
    return zlib.crc32(cell.why.encode())


# ─── the interleaved-mRoPE table, as the model builds it ───


def _mrope_table(
    inv_freq: torch.Tensor,
    plane_positions: torch.Tensor,
    sections: Tuple[int, ...] = MROPE_SECTION,
) -> torch.Tensor:
    """The interleaved three-plane mRoPE table in the op's duplicated layout.

    `HYV4FlashRotaryEmbedding.forward` expands the position ids to three
    (time, height, width) planes; `recomposition_frequencies` starts from the
    time plane and overwrites the stride-3 slots its section sizes select with
    the height and width planes' angles, then duplicates the block. Each slot
    is stored as a (cos, sin) float2 pair -- the layout
    `RopeParams(..., duplicate_data=True)` builds. `plane_positions` is
    [rows, 3]: the (t, h, w) position of the token whose absolute position
    indexes that table row.
    """
    angles = plane_positions.to(torch.float32).unsqueeze(-1) * inv_freq  # [P, 3, half]
    freq = angles[:, 0].clone()
    for plane, offset in enumerate((1, 2), start=1):
        length = sections[plane] * 3
        idx = slice(offset, length, 3)
        freq[:, idx] = angles[:, plane, idx]
    pairs = torch.stack((freq.cos(), freq.sin()), dim=-1)  # [P, half, 2]
    # The op takes the table flat, as `create_rope_const_params` returns it.
    return torch.cat((pairs, pairs), dim=1).reshape(1, -1).contiguous()


def _rope_const(theta: float, max_positions: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The engine's own frequency vector and plain duplicated table.

    `create_rope_const_params` returns `inv_freq=None` when the backend cached
    the table by weakref; rebuild the same vector rather than skip the case
    (the same fallback the target's `mla_rope_append_paged_kv_assign_q` test
    uses).
    """
    rope = RopeParams(
        dim=QK_ROPE_HEAD_DIM, theta=theta, max_positions=max_positions, duplicate_data=True
    )
    inv_freq, plain = rope.create_rope_const_params()
    if inv_freq is None:
        inv_freq = torch.tensor(
            1.0
            / (theta ** (np.arange(0, QK_ROPE_HEAD_DIM, 2) / QK_ROPE_HEAD_DIM)).astype(np.float32),
            device="cuda",
        )
    return inv_freq, plain


def _rope_ref(x: torch.Tensor, table: torch.Tensor, position: int) -> torch.Tensor:
    """GPT-J interleaved rotation of the last dim, fp32 math, input dtype out.

    The kernel's `invokeMLARopeContext` indexes `cos_sin_cache + ROPE_DIM *
    position` and rotates adjacent pairs; the table is the duplicated layout,
    so the first `half` pairs are the ones read.
    """
    half = QK_ROPE_HEAD_DIM // 2
    view = table.view(-1, QK_ROPE_HEAD_DIM, 2)
    cos = view[position, :half, 0]
    sin = view[position, :half, 1]
    pairs = x.float().reshape(*x.shape[:-1], half, 2)
    out = torch.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
    out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
    return out.reshape(x.shape).to(x.dtype)


def _rotate_probe(x: torch.Tensor, table: torch.Tensor, position: int) -> torch.Tensor:
    """The same rotation in fp32, for comparing two tables before rounding."""
    half = QK_ROPE_HEAD_DIM // 2
    view = table.view(-1, QK_ROPE_HEAD_DIM, 2)
    cos = view[position, :half, 0]
    sin = view[position, :half, 1]
    pairs = x.float().reshape(*x.shape[:-1], half, 2)
    out = torch.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
    out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
    return out.reshape(x.shape)


# ─── operands ───


def _random_context_inputs(
    num_tokens: int, num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production-shaped MLA context inputs: contiguous q and k (the k rope
    slice deliberately garbage -- the op fills it from latent_cache), v a
    strided split view of a packed kv_b_proj-style buffer, latent
    [ckv | k_pe]."""
    q = torch.randn(num_tokens, num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(
        num_tokens,
        num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_nope, v = kv.split([num_heads * QK_NOPE_HEAD_DIM, num_heads * V_HEAD_DIM], dim=-1)
    k = torch.empty_like(q).view(num_tokens, num_heads, QK_HEAD_DIM)
    k[..., :QK_NOPE_HEAD_DIM] = k_nope.view(num_tokens, num_heads, QK_NOPE_HEAD_DIM)
    k[..., QK_NOPE_HEAD_DIM:] = torch.randn(
        num_tokens, num_heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k = k.view(num_tokens, num_heads * QK_HEAD_DIM)
    latent = torch.randn(num_tokens, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    return q, k, v, latent


def _random_explicit_kv(num_tokens: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit K/V for the latent_cache=None context calls: a contiguous K and
    a packed [T, H*(nope+v)] kv_b_proj-style buffer whose split view is V.

    These stand in for what the caller's kv_b_proj produced; the pool rows are
    separate latents, so the two phases of a chunked cell are certified against
    their own operands rather than against each other.
    """
    k = torch.randn(num_tokens, num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    packed_kv = torch.randn(
        num_tokens,
        num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return k, packed_kv


def _v_split_view(packed_kv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """The [.., H*nope:] split view of a packed [T, H*(nope+v)] buffer. The
    context FMHA hard-codes V's row stride to H*(nope+v_head_dim) elements,
    so V must keep this stride -- a contiguous [T, H*v] V is misread."""
    return packed_kv.split([num_heads * QK_NOPE_HEAD_DIM, num_heads * V_HEAD_DIM], dim=-1)[1]


def _sinks(num_heads: int, seed: int) -> torch.Tensor:
    """Per-head fp32 sinks spanning the checkpoint's 0.0 initialisation.

    `learnable_sink=true` with the parameter initialised at 0.0; a served sink
    is any fp32 logit, so the cells drive -2..2 per head, which keeps 0.0
    inside what they cover.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.rand(num_heads, generator=generator, dtype=torch.float32, device="cuda") * 4.0 - 2.0
    )


# ─── the environment: one real paged latent pool, ring-addressed ───


class _Hy4MlaEnv:
    """Real MLA op state at Hy4's `mla_windowed` site.

    A caller-owned paged latent pool (kv_factor 1, one kv head, row width
    C+R) in which each request owns the window's sixteen pages as a ring:
    absolute page `j` lives at physical page `base + j % ring_pages`, so the
    append writes every token at its absolute page and slot and a history
    longer than the window overwrites aged-out slots in place. The
    block-offset table is sized for the request's longest history and filled
    with the same ring mapping at every entry, which makes the pool correct
    whether the kernel indexes the table by the absolute page or by the
    cyclic one. Test-computed latent rows are mirrored per request, so decode
    references are built from the true cache content and the pool is checked
    bitwise against it.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        max_batch: int,
        max_history: int,
        pool_dtype: torch.dtype = torch.bfloat16,
        quant_mode: int = 0,
        q_scaling: float = 1.0,
        q_lora_rank: Optional[int] = Q_LORA_RANK,
        use_paged_context_fmha: bool = False,
        window: int = WINDOW,
        ring_tokens: int = WINDOW,
        head_size: int = LATENT_DIM,
    ) -> None:
        self.num_heads = num_heads
        self.max_batch = max_batch
        self.window = window
        # The `head_size` argument a cell declares. It is not K's row width:
        # the op reads K/V widths from `qk_nope_head_dim + qk_rope_head_dim`
        # and `v_head_dim`, and uses `head_size` only to tell a generation MLA
        # op (`kv_lora_rank + qk_rope_head_dim`, the absorbed latent width) from
        # a context one (the qk width).
        self.head_size = head_size
        self.tokens_per_block = TOKENS_PER_BLOCK
        self.quant_mode = quant_mode
        self.q_scaling = q_scaling
        self.q_lora_rank = q_lora_rank
        self.use_paged_context_fmha = use_paged_context_fmha
        self.max_seq_len = MAX_SEQ_LEN
        self.max_blocks_per_seq = -(-max_history // TOKENS_PER_BLOCK)
        self.ring_pages = ring_tokens // TOKENS_PER_BLOCK
        # Both MLA phases scale QK^T by 1 / (q_scaling * sqrt(nope + R)); Hy4's
        # HF reference uses qk_head_dim ** -0.5 over the concatenated 256.
        self.softmax_scale = 1.0 / (q_scaling * math.sqrt(QK_HEAD_DIM))
        # Per-head sinks; set by the driver before the first call.
        self.sinks: Optional[torch.Tensor] = None
        # mRoPE side inputs, normally absent; a control drives garbage here.
        self.mrope_rotary_cos_sin: Optional[torch.Tensor] = None
        self.mrope_position_deltas: Optional[torch.Tensor] = None

        self.pool = torch.zeros(
            max_batch * self.ring_pages,
            TOKENS_PER_BLOCK,
            LATENT_DIM,
            dtype=pool_dtype,
            device="cuda",
        )
        self.pool_pointers = torch.zeros(1, 2, dtype=torch.int64, device="cpu")
        self.pool_pointers[0, 0] = self.pool.data_ptr()
        self.pool_mapping = torch.zeros(1, 2, dtype=torch.int32, device="cpu")
        self.block_offsets = torch.zeros(
            1, max_batch, 2, self.max_blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        # Auto-resized in place by the op on first call.
        self.workspace = torch.empty(0, dtype=torch.int8, device="cuda")

        # The model's interleaved-mRoPE table, built here exactly as
        # `HYV4FlashRotaryEmbedding` builds it and passed as `rotary_cos_sin`.
        self.rotary_inv_freq, self.rotary_cos_sin = self._mrope_const_params()

        self.base: Dict[int, int] = {}
        self.latent_rows: Dict[int, List[torch.Tensor]] = {}

    def _mrope_const_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq, _plain = _rope_const(ROPE_THETA_SWA, MAX_SEQ_LEN)
        positions = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(-1).repeat(1, 3)
        return inv_freq, _mrope_table(inv_freq, positions)

    def add_request(self, request_id: int) -> None:
        self.base[request_id] = request_id * self.ring_pages
        self.latent_rows[request_id] = []

    def cached_len(self, request_id: int) -> int:
        return sum(t.shape[0] for t in self.latent_rows[request_id])

    @property
    def fp8_pool(self) -> bool:
        return self.pool.dtype == torch.float8_e4m3fn

    def quantize_e4m3(self, x: torch.Tensor) -> torch.Tensor:
        """e4m3(x) at kv scale 1.0 -- the op's write-side quantization, and the
        same one the caller applies to the decode query."""
        return x.float().to(torch.float8_e4m3fn)

    def to_pool(self, rows: torch.Tensor) -> torch.Tensor:
        return self.quantize_e4m3(rows) if self.fp8_pool else rows

    def rope_ref(self, x: torch.Tensor, position: int) -> torch.Tensor:
        return _rope_ref(x, self.rotary_cos_sin, position)

    def rope_latent_rows(self, rows: torch.Tensor, first_position: int) -> torch.Tensor:
        """The rows as the append kernel lands them: [ckv | rope(k_pe)] at each
        token's absolute position.

        `_rope_ref`'s rotation over the whole block at once, one table row per
        row of `rows`: the same elementwise math, so the pooled rope half is
        bit-identical to rotating each row on its own, which is what one row per
        position reduces to. A drafter's history is thousands of rows, and the
        per-row form is thousands of launches.
        """
        out = rows.clone()
        count = rows.shape[0]
        half = QK_ROPE_HEAD_DIM // 2
        positions = torch.arange(first_position, first_position + count, device=rows.device)
        view = self.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        cos = view[positions, :half, 0]
        sin = view[positions, :half, 1]
        pairs = out[:, KV_LORA_RANK:].float().reshape(count, half, 2)
        rotated = torch.empty_like(pairs)
        rotated[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        rotated[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        out[:, KV_LORA_RANK:] = rotated.reshape(count, QK_ROPE_HEAD_DIM).to(rows.dtype)
        return out

    def write_latent_rows(self, request_id: int, rows: torch.Tensor) -> None:
        """Write rows already in pool form into the request's ring and mirror
        them. Stands in for the caller-side append the generation-preprocessing
        op performs in production (and for a cached prefix)."""
        start = self.cached_len(request_id)
        pooled = self.to_pool(rows)
        tpb = self.tokens_per_block
        pos = torch.arange(start, start + rows.shape[0], device=rows.device)
        pages = self.base[request_id] + (pos // tpb) % self.ring_pages
        # One scatter over the whole write: a call's positions are distinct, so
        # no page slot is written twice and the row order does not matter.
        self.pool[pages, pos % tpb] = pooled
        self.latent_rows[request_id].append(rows)

    def rows_of(self, request_id: int, positions: torch.Tensor) -> torch.Tensor:
        """The pool row ids of the request's logical positions: the flat
        `page * tokens_per_block + offset` slot that `SparseMlaDecode` and
        `DflyBlockIndices.block_slots` both index."""
        tpb = self.tokens_per_block
        pos = positions.long()
        page = self.base[request_id] + (pos // tpb) % self.ring_pages
        return (page * tpb + pos % tpb).to(torch.int32)

    def rows_at(self, row_ids: torch.Tensor) -> torch.Tensor:
        """The pool rows a selection names: a flat row id is `page * 32 + offset`
        of the pool slab, which is how the kernel addresses it."""
        return self.pool.view(-1, LATENT_DIM)[row_ids.long()]

    def append_latent_row(self, request_id: int, row: torch.Tensor) -> None:
        """One decode token's latent row, as `mla_rope_generation` lands it:
        the k_pe half already roped, so the generation kernel reads it as is."""
        self.write_latent_rows(request_id, row.unsqueeze(0))

    def gather_ring(self, request_id: int) -> torch.Tensor:
        """The pool's ring contents in logical order: the request's last
        min(history, ring_tokens) rows, gathered in one pass over the same
        page/slot mapping the writes use."""
        tpb = self.tokens_per_block
        total = self.cached_len(request_id)
        start = max(0, total - self.ring_pages * tpb)
        pos = torch.arange(start, total, device=self.pool.device)
        page = self.base[request_id] + (pos // tpb) % self.ring_pages
        return self.pool[page, pos % tpb]

    def check_ring(self, request_ids: List[int]) -> None:
        """The pool must hold exactly the last window's rows.

        The compressed-kv half is gated bitwise. The roped k_pe half is not:
        the kernel evaluates the rotation in its own order, which costs a
        handful of bf16 elements one ulp (`_assert_latent_rope` in the catalog
        test measured 8 in 495360), so it gets a one-ulp gate plus a
        bit-exact-fraction floor -- both the catalog test's, for this same
        append.
        """
        tpb = self.tokens_per_block
        for rid in request_ids:
            expected = torch.cat(self.latent_rows[rid])
            start = max(0, expected.shape[0] - self.ring_pages * tpb)
            expected = self.to_pool(expected[start:])
            got = self.gather_ring(rid)
            ckv, exp_ckv = got[:, :KV_LORA_RANK], expected[:, :KV_LORA_RANK]
            if self.fp8_pool:
                same = torch.equal(ckv.view(torch.uint8), exp_ckv.view(torch.uint8))
            else:
                same = torch.equal(ckv, exp_ckv)
            assert same, f"ring compressed_kv mismatch: request {rid}, {got.shape[0]} rows"
            a = got[:, KV_LORA_RANK:].float()
            b = expected[:, KV_LORA_RANK:].float()
            torch.testing.assert_close(
                a,
                b,
                rtol=LATENT_ROPE_E4M3_ULP_RTOL if self.fp8_pool else LATENT_ROPE_ULP_RTOL,
                atol=0.0,
                msg=lambda m: f"ring rope half, request {rid}\n{m}",
            )
            inexact = int(
                (
                    got[:, KV_LORA_RANK:].view(torch.uint8)
                    != expected[:, KV_LORA_RANK:].view(torch.uint8)
                ).sum()
            )
            allowed = max(1, int(LATENT_ROPE_MAX_INEXACT_FRACTION * a.numel()))
            assert inexact <= allowed, (
                f"ring rope half not bit-exact enough: request {rid}, "
                f"{inexact}/{a.numel()} elements differ (allowed {allowed})"
            )

    def check_unwritten_pool_zero(self, request_ids: List[int]) -> None:
        """Every page outside the requests' rings is still all-zero: this pins
        the slab geometry, since a page-stride or element-width mistake writes
        into pages nobody reserved."""
        used = set()
        for rid in request_ids:
            used.update(range(self.base[rid], self.base[rid] + self.ring_pages))
        rest = [p for p in range(self.pool.shape[0]) if p not in used]
        assert bool((self.pool[rest].float() == 0).all()), (
            "the append wrote outside the requests' pages"
        )

    def fp8_decode_buffers(
        self, fused_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three buffers an fp8-pool MLA generation call requires, filled
        the way the production generation-preprocessing step fills them:

        - quant_q_buffer: e4m3(fused_q) -- the query the decode kernel actually
          reads (`q` itself is not read);
        - mla_bmm1_scale: [x, x * log2(e)] with x = softmax_scale (s = 1.0, so
          the s**2 that undoes the write scale is 1); only element [1] is read;
        - mla_bmm2_scale: [1.0], undoing the V side's write scale.
        """
        num_gen = fused_q.shape[0]
        quant_q = self.quantize_e4m3(fused_q.view(num_gen, self.num_heads, LATENT_DIM)).contiguous()
        bmm1 = self.softmax_scale
        return (
            quant_q,
            torch.tensor([bmm1, bmm1 * math.log2(math.e)], dtype=torch.float32, device="cuda"),
            torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        )

    def _fill_offsets(self, request_ids: List[int], totals: Optional[List[int]] = None) -> None:
        # kv_factor-1 pool: the same block id in both the K and the V row. Every
        # table entry carries the ring mapping, so an absolute-page lookup and a
        # cyclic one name the same physical page and neither runs off the table.
        for s, rid in enumerate(request_ids):
            total = self.cached_len(rid) if totals is None else totals[s]
            n_pages = -(-total // self.tokens_per_block)
            assert n_pages <= self.max_blocks_per_seq, (n_pages, self.max_blocks_per_seq)
            row = self.block_offsets[0, s]
            for j in range(self.max_blocks_per_seq):
                row[0, j] = self.base[rid] + (j % self.ring_pages)
                row[1, j] = row[0, j]

    def _call_op(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        kv_lens: List[int],
        ctx_lens: List[int],
        num_contexts: int,
        num_ctx_tokens: int,
        attention_input_type: int,
        head_size: int,
        num_kv_heads: int,
        v_head_dim: int,
        latent_cache: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        cu_q_seqlens: Optional[torch.Tensor],
        cu_kv_seqlens: Optional[torch.Tensor],
        fmha_scheduler_counter: Optional[torch.Tensor],
        mask_type: int = MASK_CAUSAL,
        softmax_stats_tensor: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        predicted_tokens_per_seq: int = 1,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        num_sparse_topk: Optional[int] = None,
        attention_window_size: Optional[int] = None,
        kv_bound: Optional[int] = None,
    ) -> None:
        ns = len(kv_lens)
        req_types = [0 if i < num_contexts else 1 for i in range(ns)]
        host_kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
        if kv_bound is not None:
            # The entry's host KV length is a launch scalar, not the step's key
            # bound: on a sparse layer the core hands it the selection's width
            # (`_generation_args`), so it is what the kernel launches with while
            # the device lengths stay the step's own.
            host_kv_lens = torch.full_like(host_kv_lens, kv_bound)
        with validating(op):
            op(
                q=q,
                k=k,
                v=v,
                output=output,
                output_sf=None,
                workspace=self.workspace,
                sequence_length=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
                host_past_key_value_lengths=host_kv_lens,
                host_total_kv_lens=torch.tensor(
                    [sum(kv_lens[:num_contexts]), sum(kv_lens[num_contexts:])],
                    dtype=torch.int32,
                ),
                context_lengths=torch.tensor(ctx_lens, dtype=torch.int32, device="cuda"),
                host_context_lengths=torch.tensor(ctx_lens, dtype=torch.int32),
                host_request_types=torch.tensor(req_types, dtype=torch.int32),
                max_context_q_len_override=None,
                kv_cache_block_offsets=self.block_offsets,
                host_kv_cache_pool_pointers=self.pool_pointers,
                host_kv_cache_pool_mapping=self.pool_mapping,
                cache_indirection=None,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                out_scale=None,
                rotary_inv_freq=self.rotary_inv_freq,
                rotary_cos_sin=self.rotary_cos_sin,
                latent_cache=latent_cache,
                q_pe=q_pe,
                block_ids_per_seq=None,
                attention_sinks=self.sinks,
                is_fused_qkv=k is None,
                update_kv_cache=True,
                predicted_tokens_per_seq=predicted_tokens_per_seq,
                local_layer_idx=0,
                num_heads=self.num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                tokens_per_block=self.tokens_per_block,
                max_num_requests=self.max_batch,
                max_context_length=self.max_seq_len,
                max_seq_len=self.max_seq_len,
                attention_window_size=self.window
                if attention_window_size is None
                else attention_window_size,
                beam_width=1,
                mask_type=mask_type,
                quant_mode=self.quant_mode,
                q_scaling=self.q_scaling,
                position_embedding_type=POSITION_EMBEDDING_TYPE_MROPE,
                rope_dim=QK_ROPE_HEAD_DIM,
                rope_base=ROPE_THETA_SWA,
                rope_scale_type=0,
                rope_scale=1.0,
                rope_short_m_scale=1.0,
                rope_long_m_scale=1.0,
                rope_max_positions=MAX_SEQ_LEN,
                rope_original_max_positions=MAX_SEQ_LEN,
                use_paged_context_fmha=self.use_paged_context_fmha,
                attention_input_type=attention_input_type,
                is_mla_enable=True,
                chunked_prefill_buffer_batch_size=1,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=KV_LORA_RANK,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                v_head_dim=v_head_dim,
                rope_append=True,
                mrope_rotary_cos_sin=self.mrope_rotary_cos_sin,
                mrope_position_deltas=self.mrope_position_deltas,
                helix_position_offsets=None,
                helix_is_inactive_rank=None,
                attention_chunk_size=None,
                softmax_stats_tensor=softmax_stats_tensor,
                is_spec_decoding_enabled=False,
                use_spec_decoding=False,
                is_spec_dec_tree=False,
                spec_decoding_generation_lengths=None,
                spec_decoding_position_offsets=None,
                spec_decoding_packed_mask=None,
                spec_decoding_bl_tree_mask_offset=None,
                spec_decoding_bl_tree_mask=None,
                spec_bl_tree_first_sparse_mask_offset_kv=None,
                sparse_kv_indices=None,
                sparse_kv_offsets=None,
                sparse_attn_indices=sparse_attn_indices,
                sparse_attn_offsets=None,
                sparse_attn_indices_block_size=0,
                num_sparse_topk=num_sparse_topk,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                mla_bmm1_scale=mla_bmm1_scale,
                mla_bmm2_scale=mla_bmm2_scale,
                quant_q_buffer=quant_q_buffer,
                num_contexts=num_contexts,
                num_ctx_tokens=num_ctx_tokens,
            )
        torch.cuda.synchronize()

    def call_context(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        latent_cache: torch.Tensor,
    ) -> torch.Tensor:
        """context_only call over fresh sequences: the op ropes q's and k's
        rope slices in place, reads k_pe from latent_cache, and appends
        [ckv | rope(k_pe)] to the pool at each token's absolute position."""
        num_contexts = len(request_ids)
        num_ctx_tokens = sum(seq_lens)
        self._fill_offsets(request_ids, totals=list(seq_lens))
        output = torch.empty(
            num_ctx_tokens, self.num_heads * V_HEAD_DIM, dtype=q.dtype, device=q.device
        )
        self._call_op(
            q=q,
            k=k,
            v=v,
            output=output,
            kv_lens=list(seq_lens),
            ctx_lens=list(seq_lens),
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            attention_input_type=MLA_CONTEXT,
            head_size=self.head_size,
            num_kv_heads=self.num_heads,  # context runs as MHA
            v_head_dim=V_HEAD_DIM,
            latent_cache=latent_cache,
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
        )

        start = 0
        for rid, ln in zip(request_ids, seq_lens):
            rows = latent_cache[start : start + ln].clone()
            for i in range(ln):
                rows[i, KV_LORA_RANK:] = self.rope_ref(rows[i, KV_LORA_RANK:], i)
            self.latent_rows[rid].append(rows)
            start += ln
        return output

    def call_context_chunk(
        self,
        request_ids: List[int],
        new_lens: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """context_only call with latent_cache=None (cached KV / chunked
        prefill): no in-kernel RoPE and no cache append. q arrives pre-roped
        with `new_lens` rows per sequence, K/V arrive explicitly with
        `cached_len` rows per sequence, and the pool already holds every
        latent row the decode will read."""
        num_contexts = len(request_ids)
        num_ctx_tokens = sum(new_lens)
        totals = [self.cached_len(rid) for rid in request_ids]
        self._fill_offsets(request_ids)
        output = torch.empty(
            num_ctx_tokens, self.num_heads * V_HEAD_DIM, dtype=q.dtype, device=q.device
        )
        self._call_op(
            q=q,
            k=k,
            v=v,
            output=output,
            kv_lens=totals,
            ctx_lens=list(new_lens),
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            attention_input_type=MLA_CONTEXT,
            head_size=self.head_size,
            num_kv_heads=self.num_heads,  # context runs as MHA
            v_head_dim=V_HEAD_DIM,
            latent_cache=None,  # selects the no-RoPE / no-append context path
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
        )
        return output

    def call_generation(self, request_ids: List[int], fused_q: torch.Tensor) -> torch.Tensor:
        """generation_only call: latent MQA over the paged cache. The new
        token's latent row must already be in the pool (append_latent_row);
        latent_cache/q_pe are presence-validated but their data is not
        consumed, so they are garbage here on purpose."""
        rows = len(request_ids)
        totals = [self.cached_len(rid) for rid in request_ids]
        self._fill_offsets(request_ids)
        cu_q = torch.arange(rows + 1, dtype=torch.int32) * self.num_heads
        cu_kv = torch.zeros(rows + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(totals, dtype=torch.int32).cumsum(0)
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        garbage_latent = torch.randn(rows, LATENT_DIM, dtype=fused_q.dtype, device="cuda")
        garbage_q_pe = torch.randn(
            rows, self.num_heads, QK_ROPE_HEAD_DIM, dtype=fused_q.dtype, device="cuda"
        )
        output = torch.empty(
            rows, self.num_heads * KV_LORA_RANK, dtype=fused_q.dtype, device="cuda"
        )
        fp8_buffers = self.fp8_decode_buffers(fused_q) if self.fp8_pool else (None, None, None)
        quant_q, bmm1, bmm2 = fp8_buffers
        self._call_op(
            q=fused_q,
            k=None,
            v=None,
            output=output,
            kv_lens=totals,
            ctx_lens=totals,
            num_contexts=0,
            num_ctx_tokens=0,
            attention_input_type=MLA_GENERATION,
            head_size=LATENT_DIM,
            num_kv_heads=1,  # latent MQA
            v_head_dim=KV_LORA_RANK,
            latent_cache=garbage_latent,
            q_pe=garbage_q_pe,
            cu_q_seqlens=cu_q.cuda(),
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=counter,
            mla_bmm1_scale=bmm1,
            mla_bmm2_scale=bmm2,
            quant_q_buffer=quant_q,
        )
        return output

    def call_block_generation(
        self,
        request_ids: List[int],
        fused_q: torch.Tensor,
        selection: torch.Tensor,
        lengths: torch.Tensor,
        width: int,
        *,
        dev_kv_lens: Optional[List[int]] = None,
        num_contexts: int = 0,
        ctx_lens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The DFly2 drafter's block call (`Hyv4FlashDraft._block_attention`).

        Generation only, `width` rows per request in one call, each row reading
        its own `selection` row: the anchor's compacted picks followed by the
        block's causal prefix. `num_sparse_topk` is the selection's width, as
        the call site reads it off the buffer (`int(packed.shape[1])`), and the
        host KV length is that same width -- the core's `_generation_args`
        bound, which keeps the launch scalars identical across the steps of a
        captured graph. The window is the pool's maximum, because a sparse
        layer is bounded by its own selection and not by the sliding window.
        `latent_cache`/`q_pe` are presence-validated but not read, so they are
        garbage here on purpose; the block's rows are already in the pool.

        A request's device KV length is the count through the block's last row,
        not the rows the request holds in the pool: the kernel derives row `j`'s
        own count as `bound - width + 1 + j` -- the dense causal bound used as a
        count -- so a bound past the row's live prefix sends it into the `-1`
        padding the selection's writer put there. `lengths` is the per-row count
        `DflyBlockIndices` writes; a request's last block row is its bound, its
        default here. `dev_kv_lens` overrides the per-generation-row lengths
        with the step's own values, in the order of `request_ids`: a request
        whose history has overgrown the selection's width is handed the
        engine's `anchors + width`, what `spec_worker._write_block_kv_lens`
        leaves in `md.kv_lens_cuda_runtime` -- a bound past the row's live
        prefix, kept as the `kv_len` cells' refused control.

        `num_contexts` above 0 drives the same call inside a mixed batch: that
        many context sequences sit ahead of the generation rows, with
        `ctx_lens` their own lengths. The metadata's `seq_offset` is that
        count -- the generation phase reads `sequence_length[num_contexts:]`,
        `kv_cache_block_offsets[num_contexts:]` and `host_request_types`
        `[num_contexts:]` -- so only the trailing rows belong to this call,
        exactly as `FallbackFmha.attention` phases them.
        """
        gens = len(request_ids)
        rows = gens * width
        writer = [int(lengths[g * width + width - 1]) for g in range(gens)]
        assert torch.equal(
            torch.tensor(writer, dtype=torch.int32, device=selection.device),
            (selection[width - 1 :: width] >= 0).sum(dim=1, dtype=torch.int32),
        ), "the bound is not the block's last row's live count"
        totals = writer if dev_kv_lens is None else [int(v) for v in dev_kv_lens]
        assert len(totals) == gens, (len(totals), gens)
        context = [int(v) for v in (ctx_lens or [])]
        assert len(context) == num_contexts, (len(context), num_contexts)
        batch_ids = list(range(num_contexts)) + list(request_ids)
        kv_lens = context + totals
        if num_contexts or dev_kv_lens is not None:
            self._fill_offsets(batch_ids, totals=kv_lens)
        else:
            self._fill_offsets(request_ids)
        # The scheduler buffers are per *sequence*, not per row, whatever P is:
        # the prologue writes exactly len(kv_lens) + 1 entries, cu_q in q-row
        # units (num_heads * P per sequence) and cu_kv in tokens, each already
        # counting this step's P (`reference_scheduler_buffers`). Only the
        # generation rows are this call's: the context rows' entries are the
        # prefix the engine's `seq_offset` skips.
        cu_q = torch.arange(gens + 1, dtype=torch.int32) * (self.num_heads * width)
        cu_kv = torch.zeros(gens + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(totals, dtype=torch.int32).cumsum(0)
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        garbage_latent = torch.randn(rows, LATENT_DIM, dtype=fused_q.dtype, device="cuda")
        garbage_q_pe = torch.randn(
            rows, self.num_heads, QK_ROPE_HEAD_DIM, dtype=fused_q.dtype, device="cuda"
        )
        output = torch.empty(
            rows, self.num_heads * KV_LORA_RANK, dtype=fused_q.dtype, device="cuda"
        )
        self._call_op(
            q=fused_q,
            k=None,
            v=None,
            output=output,
            kv_lens=kv_lens,
            ctx_lens=kv_lens,
            num_contexts=num_contexts,
            num_ctx_tokens=0,
            attention_input_type=MLA_GENERATION,
            head_size=LATENT_DIM,
            num_kv_heads=1,  # latent MQA
            v_head_dim=KV_LORA_RANK,
            latent_cache=garbage_latent,
            q_pe=garbage_q_pe,
            cu_q_seqlens=cu_q.cuda(),
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=counter,
            predicted_tokens_per_seq=width,
            sparse_attn_indices=selection,
            num_sparse_topk=selection.shape[1],
            attention_window_size=MAX_SEQ_LEN,
            kv_bound=selection.shape[1],
        )
        return output

    def call_sparse_generation(
        self,
        request_ids: List[int],
        fused_q: torch.Tensor,
        selection: torch.Tensor,
        kv_lens: List[int],
    ) -> torch.Tensor:
        """The target sparse layers' own decode (`MlaLayer._attend`).

        One generation row per request over the sparse layers' bf16 pool, each
        row reading its own `sparse_attn_indices` row: the indexer's selection
        as the core writes it. `num_sparse_topk` is the selection's width, and
        the host KV bound is that same width -- the core's `_generation_args`
        bound -- while the *device* lengths stay the requests' own, which is
        where the kernel's per-row key count comes from: a row reads as many
        entries as its device length says, so a history shorter than the width
        reads exactly its live prefix and never the `-1` padding past it. The
        window is the pool's maximum, because a sparse layer is bounded by its
        own selection and not by the sliding window. `latent_cache`/`q_pe` are
        presence-validated but not read, so they are garbage here on purpose;
        the decode rows are already in the pool.
        """
        rows = len(request_ids)
        self._fill_offsets(request_ids)
        # The scheduler buffers are per *sequence*: cu_q in q-row units
        # (num_heads per row), cu_kv in tokens, one entry per request.
        cu_q = torch.arange(rows + 1, dtype=torch.int32) * self.num_heads
        cu_kv = torch.zeros(rows + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(kv_lens, dtype=torch.int32).cumsum(0)
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        garbage_latent = torch.randn(rows, LATENT_DIM, dtype=fused_q.dtype, device="cuda")
        garbage_q_pe = torch.randn(
            rows, self.num_heads, QK_ROPE_HEAD_DIM, dtype=fused_q.dtype, device="cuda"
        )
        output = torch.empty(
            rows, self.num_heads * KV_LORA_RANK, dtype=fused_q.dtype, device="cuda"
        )
        self._call_op(
            q=fused_q,
            k=None,
            v=None,
            output=output,
            kv_lens=list(kv_lens),
            ctx_lens=list(kv_lens),
            num_contexts=0,
            num_ctx_tokens=0,
            attention_input_type=MLA_GENERATION,
            head_size=LATENT_DIM,
            num_kv_heads=1,  # latent MQA
            v_head_dim=KV_LORA_RANK,
            latent_cache=garbage_latent,
            q_pe=garbage_q_pe,
            cu_q_seqlens=cu_q.cuda(),
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=counter,
            sparse_attn_indices=selection,
            num_sparse_topk=selection.shape[1],
            attention_window_size=MAX_SEQ_LEN,
            kv_bound=selection.shape[1],
        )
        return output


# ─── references over the operands the kernel reads ───


def _append_operands(
    env: _Hy4MlaEnv,
    q: torch.Tensor,
    k: torch.Tensor,
    latent: torch.Tensor,
    q_lens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The append path's q/k as the context FMHA reads them after the call's
    in-place rope: q's rope slice at the token's position, k's from
    latent_cache's k_pe at the token's position. Built from the pre-call
    tensors (the op clobbers q's and k's rope slices)."""
    q_out, k_out = q.clone(), k.clone()
    q_view = q.view(-1, env.num_heads, QK_HEAD_DIM)
    qo = q_out.view(-1, env.num_heads, QK_HEAD_DIM)
    ko = k_out.view(-1, env.num_heads, QK_HEAD_DIM)
    start = 0
    for ln in q_lens:
        for i in range(ln):
            t = start + i
            qo[t, :, QK_NOPE_HEAD_DIM:] = env.rope_ref(q_view[t, :, QK_NOPE_HEAD_DIM:], i)
            ko[t, :, QK_NOPE_HEAD_DIM:] = env.rope_ref(latent[t, KV_LORA_RANK:], i)
        start += ln
    return q_out, k_out


def _context_reference(
    env: _Hy4MlaEnv,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: List[int],
    kv_lens: List[int],
) -> torch.Tensor:
    return op.reference(
        q,
        k,
        v,
        q_lens,
        kv_lens,
        num_heads=env.num_heads,
        num_kv_heads=env.num_heads,  # context runs as MHA
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=env.window,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


def _generation_reference(
    env: _Hy4MlaEnv, request_ids: List[int], fused_q: torch.Tensor
) -> torch.Tensor:
    """Latent MQA over each sequence's pooled rows: K = [ckv | roped k_pe],
    V = K[:, :512], one kv head, bottom-right causal, the window keeping the
    newest 512 keys -- which is exactly what the ring holds."""
    k_all = torch.cat([torch.cat(env.latent_rows[rid]) for rid in request_ids])
    v_all = k_all[:, :KV_LORA_RANK].contiguous()
    return op.reference(
        fused_q,
        k_all,
        v_all,
        [1] * len(request_ids),
        [env.cached_len(rid) for rid in request_ids],
        num_heads=env.num_heads,
        num_kv_heads=1,
        qk_head_dim=LATENT_DIM,
        v_head_dim=KV_LORA_RANK,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=env.window,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


def _publish_picks(
    env: _Hy4MlaEnv,
    request_ids: List[int],
    counts: List[int],
    history: int = DRAFT_HISTORY,
) -> DraftPicks:
    """The picks state the target's core publishes, filled through its own writer.

    `Hyv4FlashCacheManager.draft_picks(layers, topk, rows=..., device=...)`
    allocates this object and hands it to both ends -- the core writes each
    sparse layer's selection into it from `_layer_forward`, the drafter reads it
    through `DflyBlockIndices`. The harness builds the same state, one row per
    request (`state_rows` below is that row for each request), and writes each
    draft layer's own selection through `DraftPicks.write`: `counts[g]` of the
    request's history rows as pool rows, `NO_PICK` past them. `history` is how
    many rows the request's indexer scored -- the picks are a top-`counts[g]` of
    that many rows, and `counts` never exceeds it.
    """
    gens, device = len(request_ids), env.pool.device
    picks = DraftPicks(DRAFT_LAYERS, gens, TARGET_TOPK, device)
    rows = torch.arange(gens, device=device)
    for layer in DRAFT_LAYERS:
        packed = torch.full((gens, TARGET_TOPK), NO_PICK, dtype=torch.int32, device=device)
        for g, rid in enumerate(request_ids):
            # This layer's indexer selection over the request's own history: the
            # top-`counts[g]` of its rows, in whatever order it scored them.
            chosen = torch.randperm(history, device=device)[: counts[g]]
            packed[g, : counts[g]] = env.rows_of(rid, chosen)
        picks.write(layer, packed, rows)
    return picks


def _block_indices(
    env: _Hy4MlaEnv,
    picks: DraftPicks,
    request_ids: List[int],
    state_rows: torch.Tensor,
    history: int = DRAFT_HISTORY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The k29 entry's selection: `packed` `[layers, rows, width]`, `lengths`.

    `block_slots` is where the block's rows already sit in the pool -- the
    anchor's bonus token and the mask rows, appended after the history -- and
    the entry writes every block row's selection: the anchor's compacted picks
    followed by the block's causal prefix, `-1` past it. `history` is where the
    block starts, the anchor's own position.
    """
    gens, width, device = len(request_ids), BLOCK_ROWS, env.pool.device
    first = torch.arange(history, history + width, device=device)
    block_slots = torch.stack([env.rows_of(rid, first) for rid in request_ids])
    active = torch.ones(gens, dtype=torch.bool, device=device)
    packed = torch.full(
        (len(DRAFT_LAYERS), gens * width, INDEX_WIDTH),
        PAD_SLOT_ID,
        dtype=torch.int32,
        device=device,
    )
    lengths = torch.zeros((len(DRAFT_LAYERS), gens * width), dtype=torch.int32, device=device)
    dfly_block_indices(picks.history, state_rows, block_slots, active, packed, lengths)
    return packed, lengths


def _sparse_selection(env: _Hy4MlaEnv, request_id: int, total: int) -> torch.Tensor:
    """One decode row's `sparse_attn_indices`, as the core writes them.

    `Indexer.pool_and_select`'s `_recent_rows(positions, index_topk)`: the
    row's own key position and the `TARGET_TOPK - 1` before it, descending,
    `-1` in the slots whose position falls before the document's start -- a
    valid prefix followed by the `-1` padding the kernel skips. `_pool_rows`
    then resolves the positions through the step's block table, which here is
    the harness's own page mapping (`rows_of`); the padded slots stay `-1`, as
    they resolve to nothing.
    """
    pos = total - 1 - torch.arange(TARGET_TOPK, dtype=torch.int64, device=env.pool.device)
    rows = env.rows_of(request_id, pos.clamp(min=0))
    rows[pos < 0] = NO_PICK
    return rows


def _sparse_decode_reference(
    env: _Hy4MlaEnv,
    selection: torch.Tensor,
    kv_lens: List[int],
    fused_q: torch.Tensor,
) -> torch.Tensor:
    """The sparse rows as `reference` reads them: one sequence per request.

    A row's key list is its own selection's live prefix, resolved back to pool
    rows (`rows_at`) -- which is what the kernel gathers -- and its `kv_len` is
    that prefix's length, the count the kernel derives from the device length.
    `reference`'s bottom-right causal mask over one query row makes a
    sequence's whole list visible; the list is a set, so the row's order (the
    core writes descending positions) does not enter. No window: a sparse row
    is bounded by its selection.
    """
    keys = [env.rows_at(sel[sel >= 0]) for sel in selection]
    k = torch.cat(keys)
    v = k[:, :KV_LORA_RANK].contiguous()
    return op.reference(
        fused_q,
        k,
        v,
        [1] * len(kv_lens),
        list(kv_lens),
        num_heads=env.num_heads,
        num_kv_heads=1,
        qk_head_dim=LATENT_DIM,
        v_head_dim=KV_LORA_RANK,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=None,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


def _block_reference(
    env: _Hy4MlaEnv,
    selection: torch.Tensor,
    counts: List[int],
    fused_q: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """The block's rows as `reference` reads them: one sequence per request.

    A request's key list is its picked rows followed by the block's own six rows,
    which is the last row's selection -- the widest live prefix the entry writes
    (`counts + width`). The reference's bottom-right causal mask then gives row
    `t` exactly `counts + t + 1` keys, the prefix the entry leaves live for that
    row, so a kernel that read every row's full width instead would attend the
    block's later slots and leave the band. No window: a sparse row is bounded by
    its selection, `attention_sinks` stays in the denominator only, and the pool
    rows are read by the ids the selection names, the way the kernel reads them.
    """
    gens = len(counts)
    rows = gens * width
    q = fused_q.view(rows, env.num_heads, LATENT_DIM)
    keys = []
    for g in range(gens):
        last = selection[g * width + width - 1]
        keys.append(env.rows_at(last[: counts[g] + width]))
    k = torch.cat(keys)
    v = k[:, :KV_LORA_RANK].contiguous()
    return op.reference(
        q.reshape(rows, env.num_heads * LATENT_DIM),
        k,
        v,
        [width] * gens,
        [counts[g] + width for g in range(gens)],
        num_heads=env.num_heads,
        num_kv_heads=1,
        qk_head_dim=LATENT_DIM,
        v_head_dim=KV_LORA_RANK,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
        window=None,
        e4m3_inputs=env.fp8_pool,
        e4m3_probabilities=env.fp8_pool,
    )


# ─── gates ───


def _compare(out: torch.Tensor, ref: torch.Tensor, fp8: bool) -> None:
    """The band for this pool: the catalog test's, derived there for this op's
    two MLA paths (see the constants above)."""
    if fp8:
        torch.testing.assert_close(out, ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
        err = (out.float() - ref.float()).pow(2).mean().sqrt()
        rel_rms = (err / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)).item()
        assert rel_rms <= BF16_RMS_ULP * ULP_BF16, (
            f"relative RMS {rel_rms:.3e} exceeds {BF16_RMS_ULP} bf16 ulp"
        )
    else:
        assert_within_ulp(out, ref, element_ulp=BF16_ELEMENT_ULP, rms_ulp=BF16_RMS_ULP)


def _measured_band(out: torch.Tensor, ref: torch.Tensor) -> str:
    """The two numbers `assert_within_ulp` gates on, in units of one bf16 ulp of
    the row scale: the worst element and the relative RMS.

    The gate prints nothing when it passes, so the cells whose band a report has
    to state print this themselves, from the same two quantities the gate
    computes. Printing does not weaken either gate: it sits beside them.
    """
    flat_out = out.float().reshape(-1, out.shape[-1])
    flat_ref = ref.float().reshape(-1, ref.shape[-1])
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    worst = ((flat_out - flat_ref).abs() / row_scale).max().item() / ULP_BF16
    rel_rms = (
        (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt().clamp_min(1e-9)
    ).item() / ULP_BF16
    return (
        f"worst element {worst:.4f} bf16 ulp (gate {BF16_ELEMENT_ULP}), "
        f"relative RMS {rel_rms:.4f} bf16 ulp (gate {BF16_RMS_ULP})"
    )


def _outside_fraction(out: torch.Tensor, rival: torch.Tensor) -> float:
    """Fraction of elements outside the bf16 element gate against a wrong
    reference: how much of the output a control moved."""
    o = out.float().reshape(-1, out.shape[-1])
    r = rival.float().reshape(-1, rival.shape[-1])
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    outside = (o - r).abs() / scale > BF16_ELEMENT_ULP * ULP_BF16
    return float(outside.float().mean())


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


# ─── the cells ───


def _run_cell(cell) -> None:
    """Drive one cell through the phases its spec names.

    A `block` cell is one of the two generation-only calls over the sparse
    layers' pool: the target's own decode (`predicted_tokens_per_seq` 1, 16 q
    heads at the tp2 shard) or the DFly2 drafter's six-row block.
    """
    if cell.spec.get("block"):
        if cell.spec["predicted_tokens_per_seq"] == 1:
            _run_sparse_cell(cell)
        else:
            _run_block_cell(cell)
        return
    spec = cell.spec
    seed = _seed_of(cell)
    torch.manual_seed(seed)
    heads = spec["num_heads"]
    assert spec["num_kv_heads"] == heads, spec["num_kv_heads"]
    assert spec["tokens_per_block"] == TOKENS_PER_BLOCK
    assert spec["window"] == WINDOW
    assert spec["head_size"] == LATENT_DIM
    assert spec["v_head_dim"] == V_HEAD_DIM
    assert spec["sinks"] is True
    quant_mode = spec["quant_mode"]
    assert quant_mode in (0, QUANT_MODE_FP8_KV_CACHE), quant_mode
    fp8 = quant_mode == QUANT_MODE_FP8_KV_CACHE
    chunked = spec["use_paged_context_fmha"]
    max_batch = spec["max_num_requests"]
    decode = spec["decode"]
    # A cell states its context as `prefill` (fresh) or as `cached` + `new`
    # (a chunk over a prefix); a paged-context cell with only `prefill` is one
    # chunk over an empty prefix.
    cached = spec.get("cached", 0)
    new = spec.get("new", spec.get("prefill", 0) - cached)
    prefill = cached + new
    assert prefill == spec.get("prefill", prefill), (cached, new, spec.get("prefill"))
    assert (cached == 0) or chunked, "a cached prefix needs the paged-context path"
    assert spec.get("rows", max_batch * new) == max_batch * new, (
        "the envelope cell's `rows` must be the batch's context row count"
    )
    history = prefill + decode

    env = _Hy4MlaEnv(
        heads,
        max_batch=max_batch,
        max_history=history,
        # The ring must hold the context phase's rows without aliasing: a
        # prefill longer than the ring overwrites rows its own early queries
        # read. The decode then wraps by its one new row, and every slot the
        # window no longer covers holds an aged-out row.
        ring_tokens=max(WINDOW, -(-prefill // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK),
        pool_dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        quant_mode=quant_mode,
        use_paged_context_fmha=chunked,
        head_size=spec["head_size"],
    )
    env.sinks = _sinks(heads, seed)
    request_ids = list(range(max_batch))
    for rid in request_ids:
        env.add_request(rid)

    # ── context phase
    if chunked:
        rows = torch.randn(max_batch, prefill, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        for rid in request_ids:
            env.write_latent_rows(rid, env.rope_latent_rows(rows[rid], 0))
        k, packed_kv = _random_explicit_kv(max_batch * prefill, heads)
        v = _v_split_view(packed_kv, heads)
        q = torch.randn(max_batch * new, heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        out = env.call_context_chunk(request_ids, [new] * max_batch, q, k, v)
        ref = _context_reference(env, q, k, v, [new] * max_batch, [prefill] * max_batch)
    else:
        q, k, v, latent = _random_context_inputs(max_batch * prefill, heads)
        q_pre, k_pre, latent_pre = q.clone(), k.clone(), latent.clone()
        out = env.call_context(request_ids, [prefill] * max_batch, q, k, v, latent)
        env.check_ring(request_ids)
        q_ref, k_ref = _append_operands(env, q_pre, k_pre, latent_pre, [prefill] * max_batch)
        ref = _context_reference(env, q_ref, k_ref, v, [prefill] * max_batch, [prefill] * max_batch)
    _compare(out, ref, fp8)
    if not fp8:  # the fp8 gate's numbers are a different pair; see `_compare`
        print(f"{cell.why} [context]: {_measured_band(out, ref)}", flush=True)

    # ── decode phase: one row at the history's end, then one generation call
    decode_rows = torch.randn(max_batch, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    for rid in request_ids:
        env.append_latent_row(rid, decode_rows[rid])
    fused_q = torch.randn(max_batch, heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    pool_before = env.pool.clone()
    out_gen = env.call_generation(request_ids, fused_q)
    assert _bitwise_equal(pool_before, env.pool), "the generation call wrote to the pool"
    env.check_ring(request_ids)
    ref_gen = _generation_reference(env, request_ids, fused_q)
    _compare(out_gen, ref_gen, fp8)
    if not fp8:
        print(f"{cell.why} [decode]: {_measured_band(out_gen, ref_gen)}", flush=True)
    env.check_unwritten_pool_zero(request_ids)


def _run_block_cell(cell) -> None:
    """Drive one DFly2 block cell: the drafter's call at the shape it uses.

    One generation-only call over the target sparse layers' bf16 pool, `width`
    rows per request, each row's `sparse_attn_indices` the k29 selection the
    engine's own producers wrote -- the family's `DraftPicks`, filled through
    its writer, read by `DflyBlockIndices`. The block's rows are already in the
    pool, so the call must not write it, and the output is gated against
    `reference` over exactly the rows each selection row names.

    A `kv_len` in the spec is the step's own device KV length: the block's rows
    then sit at the end of a `kv_len - width`-row history instead of the short
    cells' `DRAFT_HISTORY`, and the batch carries that length
    (`spec_worker._write_block_kv_lens` leaves `anchors + width` in
    `md.kv_lens_cuda_runtime`) while the entry's bound is the writer's count
    through each block's last row -- the two counts the engine carries for the
    same request. Such a cell gates the writer's bound and drives the same call
    once more with the step's device length as the bound, the other count, and
    asserts that control leaves the band, so the operands and the bound are
    measured apart from each other. `num_contexts` above 0 puts that
    many context sequences ahead of the generation rows -- a mixed batch, where
    the entry's `num_contexts` is the metadata's `seq_offset` and the context
    rows keep the step's own lengths.
    """
    spec = cell.spec
    seed = _seed_of(cell)
    torch.manual_seed(seed)
    heads = spec["num_heads"]
    assert heads == SPARSE_HEADS_TP4, heads  # the sparse layers' 32 q heads / tp4
    assert spec["num_kv_heads"] == 1, spec["num_kv_heads"]  # latent MQA
    assert spec["head_size"] == LATENT_DIM, spec["head_size"]
    assert spec["v_head_dim"] == KV_LORA_RANK, spec["v_head_dim"]
    assert spec["tokens_per_block"] == TOKENS_PER_BLOCK
    assert spec["quant_mode"] == 0, spec["quant_mode"]
    width = spec["predicted_tokens_per_seq"]
    assert width == BLOCK_ROWS, width
    batch = spec["max_num_requests"]
    num_contexts = int(spec.get("num_contexts", 0))
    assert 0 <= num_contexts < batch, (num_contexts, batch)
    gens = batch - num_contexts
    if "num_sparse_topk" in spec:
        assert spec["num_sparse_topk"] == INDEX_WIDTH, spec["num_sparse_topk"]
    kv_len = spec.get("kv_len")
    history = DRAFT_HISTORY if kv_len is None else kv_len - width

    env = _Hy4MlaEnv(
        heads,
        max_batch=batch,
        max_history=history + width,
        # A sparse layer is bounded by its own selection, not by a window, so
        # the ring holds the whole history and the block's rows after it. The
        # block wraps nothing: its slots are real rows of this request's ring.
        window=MAX_SEQ_LEN,
        ring_tokens=-(-(history + width) // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK,
        pool_dtype=torch.bfloat16,
        quant_mode=0,
        head_size=spec["head_size"],
    )
    env.sinks = _sinks(heads, seed)
    # The generation rows are the batch's tail; the context rows ahead of them
    # carry no block of their own, only the offset the entry has to skip.
    request_ids = list(range(num_contexts, batch))
    short_history = (
        torch.randn(gens, DRAFT_HISTORY, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        if kv_len is None
        else None
    )
    block = torch.randn(gens, width, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    for g, rid in enumerate(request_ids):
        env.add_request(rid)
        rows = (
            short_history[g]
            if short_history is not None
            # A long request's history is `anchors` rows: one request at a
            # time, so the run's peak is the pool's and not a batch of them.
            else torch.randn(history, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        )
        env.write_latent_rows(rid, env.rope_latent_rows(rows, 0))
        # The block's own rows land at the history's end, as the prologue's
        # append does; the target's verify step reads them back from here.
        env.write_latent_rows(rid, env.rope_latent_rows(block[g], history))
    for rid in range(num_contexts):
        env.add_request(rid)
    env.check_ring(request_ids)

    # The anchor's selection: a saturated pick list for the long requests, a
    # shorter one at the batch's tail, which is the spread a real step carries.
    # A history that has overgrown the selection's width keeps exactly
    # `TARGET_TOPK` valid picks -- the checkpoint indexer's own count there.
    counts = [TARGET_TOPK] * gens
    if kv_len is None:
        for g in range(max(0, gens - 4), gens):
            counts[g] = TARGET_TOPK // (2 ** (gens - g))
    picks = _publish_picks(env, request_ids, counts, history=history)
    state_rows = torch.arange(gens, dtype=torch.int64, device="cuda")
    packed, lengths = _block_indices(env, picks, request_ids, state_rows, history=history)
    slot = len(DRAFT_LAYERS) - 1  # one draft layer's own selection, as the op takes it
    selection = packed[slot]
    # What the entry wrote, before the op sees it: row `j` of request `g` is
    # live for `counts[g] + j + 1` columns, and every live column is a pool row
    # (the picks first, then the block's own rows, and `-1` past them).
    want = torch.tensor(
        [counts[g] + j + 1 for g in range(gens) for j in range(width)],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(lengths[slot], want), "the k29 entry's live counts moved"
    live = selection[:, : int(want.min())]
    assert bool((live >= 0).all()), "a live column of the selection names no pool row"

    fused_q = torch.randn(gens * width, heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    pool_before = env.pool.clone()
    ctx_lens = [kv_len] * num_contexts if kv_len is not None else None
    # `lengths[slot]` is the writer's count per block row; the entry reads the
    # request's bound off the block's last row, so the per-row counts the
    # reference masks with and the sequence bound the kernel is handed are the
    # same numbers -- what `draft.Hyv4FlashDraft._block_attention` hands every
    # generating request, its context rows keeping the step's own lengths.
    out = env.call_block_generation(
        request_ids,
        fused_q,
        selection,
        lengths[slot],
        width,
        num_contexts=num_contexts,
        ctx_lens=ctx_lens,
    )
    if kv_len is not None:
        # The refused control, one variable from the gated call: the same batch
        # and selection with the step's device length as the bound, the other
        # count the engine carries for the request. The kernel's derived count
        # `bound - width + 1 + j` then runs past the row's live prefix into the
        # `-1` tail the writer left, so this call leaves the band.
        out_dev = env.call_block_generation(
            request_ids,
            fused_q,
            selection,
            lengths[slot],
            width,
            dev_kv_lens=[kv_len] * gens,
            num_contexts=num_contexts,
            ctx_lens=ctx_lens,
        )
    assert _bitwise_equal(pool_before, env.pool), "the generation call wrote to the pool"
    env.check_ring(request_ids)
    env.check_unwritten_pool_zero(request_ids)
    ref = _block_reference(env, selection, counts, fused_q, width)
    print(f"{cell.why}: {_measured_band(out, ref)}", flush=True)
    _compare(out, ref, fp8=False)
    if kv_len is not None:
        print(
            f"{cell.why} [control, the device length {kv_len} as the bound]: "
            f"{_measured_band(out_dev, ref)}",
            flush=True,
        )
        with pytest.raises(AssertionError, match=r"not close|exceeds"):
            _compare(out_dev, ref, fp8=False)


def _run_sparse_cell(cell) -> None:
    """Drive the target sparse layers' own decode at the ctx deployment's shard.

    One generation-only call over the sparse layers' bf16 pool, one row per
    request, each row's `sparse_attn_indices` the selection the core passes at
    decode: `Indexer.pool_and_select`'s `_recent_rows(positions, index_topk)`,
    resolved to pool rows. The requests bring two histories, one on each side
    of the selection's width: a history longer than it, whose list is a real
    top-`TARGET_TOPK` of the row's own keys, and one shorter, whose list is the
    whole history with `-1` past the live prefix. The entry is handed
    `num_sparse_topk` and the host KV bound at the selection's width, the value
    the core's `_generation_args` writes, while the device lengths stay the
    requests' own -- the count the kernel derives a row's key list from: the
    selection's width on a long history (the device length clamped) and the
    live prefix itself on a short one. Both are gated against `reference` over
    exactly the rows each row's live prefix names.
    """
    spec = cell.spec
    seed = _seed_of(cell)
    torch.manual_seed(seed)
    heads = spec["num_heads"]
    assert heads == SPARSE_HEADS_TP2, heads  # the sparse layers' 32 q heads / tp2
    assert spec["num_kv_heads"] == 1, spec["num_kv_heads"]  # latent MQA
    assert spec["head_size"] == LATENT_DIM, spec["head_size"]
    assert spec["v_head_dim"] == KV_LORA_RANK, spec["v_head_dim"]
    assert spec["tokens_per_block"] == TOKENS_PER_BLOCK
    assert spec["quant_mode"] == 0, spec["quant_mode"]
    assert spec["predicted_tokens_per_seq"] == 1, spec["predicted_tokens_per_seq"]
    gens = spec["max_num_requests"]
    histories = [SPARSE_HISTORY_LONG if g % 4 != 3 else SPARSE_HISTORY_SHORT for g in range(gens)]

    env = _Hy4MlaEnv(
        heads,
        max_batch=gens,
        max_history=SPARSE_HISTORY_LONG + 1,
        # A sparse layer is bounded by its own selection, not by a window, so
        # the ring holds each request's whole history and the decode row after
        # it, and the selection addresses those rows directly: nothing wraps.
        window=MAX_SEQ_LEN,
        ring_tokens=-(-(SPARSE_HISTORY_LONG + 1) // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK,
        pool_dtype=torch.bfloat16,
        quant_mode=0,
        head_size=spec["head_size"],
    )
    env.sinks = _sinks(heads, seed)
    request_ids = list(range(gens))
    for g, rid in enumerate(request_ids):
        env.add_request(rid)
        history = torch.randn(histories[g], LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        env.write_latent_rows(rid, env.rope_latent_rows(history, 0))
        # The row's own key, at the history's end: the prologue appended it
        # before the generation call, so the kernel reads it from the pool.
        decode_row = torch.randn(1, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        env.write_latent_rows(rid, env.rope_latent_rows(decode_row, histories[g]))
    env.check_ring(request_ids)

    # One decode row per request: the row count the entry reads off the device
    # is the request's own, key count and all.
    totals = [h + 1 for h in histories]
    selection = torch.stack(
        [_sparse_selection(env, rid, total) for rid, total in zip(request_ids, totals)]
    )
    # The core's layout, before the op sees it: a valid prefix of the row's own
    # keys -- its whole history, or the newest `TARGET_TOPK` of it -- then `-1`.
    for g, total in enumerate(totals):
        live = min(total, TARGET_TOPK)
        assert bool((selection[g, :live] >= 0).all()), (
            f"a live column of request {g}'s selection names no pool row"
        )
        assert bool((selection[g, live:] < 0).all()), (
            f"request {g}'s selection is padded inside its live prefix"
        )
    kv_lens = [min(total, TARGET_TOPK) for total in totals]

    fused_q = torch.randn(gens, heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    pool_before = env.pool.clone()
    out = env.call_sparse_generation(request_ids, fused_q, selection, totals)
    assert _bitwise_equal(pool_before, env.pool), "the generation call wrote to the pool"
    env.check_ring(request_ids)
    ref = _sparse_decode_reference(env, selection, kv_lens, fused_q)
    print(f"{cell.why}: {_measured_band(out, ref)}", flush=True)
    long = [g for g in range(gens) if totals[g] > TARGET_TOPK]
    short = [g for g in range(gens) if totals[g] <= TARGET_TOPK]
    print(
        f"{cell.why} [{len(long)} rows over histories longer than the width]: "
        f"{_measured_band(out[long], ref[long])}",
        flush=True,
    )
    print(
        f"{cell.why} [{len(short)} rows over histories shorter than the width]: "
        f"{_measured_band(out[short], ref[short])}",
        flush=True,
    )
    _compare(out, ref, fp8=False)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    """Drive every configuration the variant claims, both phases, one pool.

    The cell's own seed drives the sinks and every operand, and the pool is
    checked bitwise against the rows the reference reads, so a passing cell is
    the append and the masked softmax together.
    """
    _arch_of_this_device()
    certifying(op)
    _run_cell(cell)


def test_context_accepts_the_qk_width(certifying) -> None:  # noqa: F811
    """A context call at the qk width 256 must run as well.

    `head_size` is not K's row width: the op reads K/V widths from
    `qk_nope_head_dim + qk_rope_head_dim` and `v_head_dim`, and uses
    `head_size` only to separate a generation MLA op (the absorbed latent
    width, 576) from a context one. The built-in MLA context module passes
    `qk_head_dim` (256) there, so that call shape is as real as the cells'
    576, and it runs the same 256/256 context geometry: `MLA::initContext`
    takes V's head width from `v_head_dim`.
    """
    _arch_of_this_device()
    torch.manual_seed(5)
    heads = NUM_HEADS_SWA
    env = _Hy4MlaEnv(heads, max_batch=1, max_history=64, head_size=QK_HEAD_DIM)
    certifying(op)
    env.sinks = _sinks(heads, 5)
    env.add_request(0)
    q, k, v, latent = _random_context_inputs(64, heads)
    q_pre, k_pre, latent_pre = q.clone(), k.clone(), latent.clone()
    out = env.call_context([0], [64], q, k, v, latent)
    q_ref, k_ref = _append_operands(env, q_pre, k_pre, latent_pre, [64])
    ref = _context_reference(env, q_ref, k_ref, v, [64], [64])
    _compare(out, ref, fp8=False)


# ─── controls: the distinguishing inputs are live ───


def test_window_is_applied() -> None:
    """The 512-token window clips, and the windowed reference is the right one.

    One 600-token prefill: the windowed reference matches within the band, and
    the full-causal reference of the same operands is outside it by a measured
    fraction -- a kernel that ignored `attention_window_size` would pass the
    second comparison and fail the first.
    """
    _arch_of_this_device()
    torch.manual_seed(11)
    heads = NUM_HEADS_SWA
    # 600 rows need nineteen pages; a 512-token ring would overwrite the rows
    # the prefill's first queries read before they read them.
    env = _Hy4MlaEnv(heads, max_batch=1, max_history=600, ring_tokens=608)
    env.sinks = _sinks(heads, 11)
    env.add_request(0)
    q, k, v, latent = _random_context_inputs(600, heads)
    q_pre, k_pre, latent_pre = q.clone(), k.clone(), latent.clone()
    out = env.call_context([0], [600], q, k, v, latent)
    q_ref, k_ref = _append_operands(env, q_pre, k_pre, latent_pre, [600])

    common = dict(
        num_heads=heads,
        num_kv_heads=heads,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        softmax_scale=env.softmax_scale,
        causal=True,
        attention_sinks=env.sinks,
    )
    windowed = op.reference(q_ref, k_ref, v, [600], [600], window=WINDOW, **common)
    _compare(out, windowed, False)
    full = op.reference(q_ref, k_ref, v, [600], [600], window=None, **common)
    fraction = _outside_fraction(out, full)
    assert fraction > 0.05, (
        f"the full-causal rival moved only {fraction:.3f} of the output past the "
        "element gate -- the window gate would not have caught an ignored window"
    )


def test_sinks_enter_the_denominator_only() -> None:
    """A per-head sink is a logit in the softmax denominator, dropped from the
    numerator.

    The sink-free reference is the numerator-inclusive rival (the sink's value
    vector would be zero): it must be outside the band, while the
    denominator-only reference matches. A kernel that ignored the sink, or one
    that renormalised the weights after adding it, fails one comparison each.

    The sinks sit near log(128), the logsumexp scale of the 128-key softmax
    this call computes, so the sink column keeps a resolvable share of the
    mass; a sink near zero is genuinely negligible there and would make the
    sink-free separation gate vacuous rather than the semantics different (the
    catalog test draws its sinks the same way).
    """
    _arch_of_this_device()
    torch.manual_seed(12)
    heads = NUM_HEADS_SWA
    env = _Hy4MlaEnv(heads, max_batch=1, max_history=128)
    env.sinks = _sinks(heads, 12) + math.log(128)
    env.add_request(0)
    q, k, v, latent = _random_context_inputs(128, heads)
    q_pre, k_pre, latent_pre = q.clone(), k.clone(), latent.clone()
    out = env.call_context([0], [128], q, k, v, latent)
    q_ref, k_ref = _append_operands(env, q_pre, k_pre, latent_pre, [128])

    common = dict(
        num_heads=heads,
        num_kv_heads=heads,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        softmax_scale=env.softmax_scale,
        causal=True,
    )
    with_sinks = op.reference(q_ref, k_ref, v, [128], [128], attention_sinks=env.sinks, **common)
    _compare(out, with_sinks, False)
    without = op.reference(q_ref, k_ref, v, [128], [128], attention_sinks=None, **common)
    fraction = _outside_fraction(out, without)
    assert fraction > 0.05, (
        f"the sink-free rival moved only {fraction:.3f} of the output past the "
        "element gate -- the sink would not have been measured"
    )


def test_mrope_side_inputs_are_not_read() -> None:
    """The MLA path reads `rotary_cos_sin` only.

    Hy4's caller also carries `mrope_rotary_cos_sin` and `mrope_position_deltas`
    (the interleaved table and the per-request position deltas the standard
    path reads). Driving the same call with garbage there changes the output
    bitwise not at all. The table the model actually passes is the interleaved
    mRoPE one, and at text-only positions it rotates exactly as the plain
    duplicated table does -- which is why one table serves both layer types.
    """
    _arch_of_this_device()
    heads = NUM_HEADS_SWA
    outputs = []
    for with_garbage in (False, True):
        torch.manual_seed(13)
        env = _Hy4MlaEnv(heads, max_batch=1, max_history=64)
        env.sinks = _sinks(heads, 13)
        if with_garbage:
            generator = torch.Generator(device="cuda").manual_seed(99)
            env.mrope_rotary_cos_sin = (
                torch.rand(1, MAX_SEQ_LEN * QK_ROPE_HEAD_DIM, generator=generator, device="cuda")
                * 2.0
                - 1.0
            )
            env.mrope_position_deltas = torch.zeros(1, dtype=torch.int32, device="cuda")
        env.add_request(0)
        q, k, v, latent = _random_context_inputs(64, heads)
        outputs.append(env.call_context([0], [64], q, k, v, latent))
    assert _bitwise_equal(outputs[0], outputs[1]), (
        "mrope_rotary_cos_sin / mrope_position_deltas changed the MLA context output"
    )

    # The interleaved table at text-only positions is the plain duplicated one.
    inv_freq, plain = _rope_const(ROPE_THETA_SWA, MAX_SEQ_LEN)
    text_planes = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(-1).repeat(1, 3)
    text_table = _mrope_table(inv_freq, text_planes)
    probe = torch.randn(4, heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    for position in (0, 1, 127, 1023):
        torch.testing.assert_close(
            _rotate_probe(probe, text_table, position),
            _rotate_probe(probe, plain, position),
            rtol=1e-6,
            atol=1e-5,
            msg=lambda m: f"interleaved mRoPE differs from the plain table at {position}\n{m}",
        )


# ─── the guard ───


def test_is_valid_rejects_misaligned_sinks() -> None:
    """The guard admits the site's sinks and refuses the shapes that are read
    as a raw buffer: a short tensor is read past its end, an empty one
    silently disables the sink."""
    _arch_of_this_device()
    heads = NUM_HEADS_SWA
    good = _sinks(heads, 15)
    op.is_valid(num_heads=heads, num_kv_heads=heads, attention_sinks=good)
    op.is_valid(num_heads=heads, num_kv_heads=1, attention_sinks=good)
    with pytest.raises(AssertionError, match="attention_sinks must be contiguous"):
        op.is_valid(
            num_heads=heads,
            num_kv_heads=heads,
            attention_sinks=torch.zeros(heads - 1, dtype=torch.float32, device="cuda"),
        )
    with pytest.raises(AssertionError, match="must be a multiple of num_kv_heads"):
        op.is_valid(num_heads=heads, num_kv_heads=3, attention_sinks=good)

    # And the guard is what the armed call runs: the same short tensor through
    # the full call is refused before the kernel reads it.
    torch.manual_seed(15)
    env = _Hy4MlaEnv(heads, max_batch=1, max_history=64)
    env.sinks = torch.zeros(heads - 1, dtype=torch.float32, device="cuda")
    env.add_request(0)
    q, k, v, latent = _random_context_inputs(64, heads)
    with pytest.raises(AssertionError, match="attention_sinks must be contiguous"):
        env.call_context([0], [64], q, k, v, latent)


def test_mla_requires_int_q_lora_rank() -> None:
    """The op's own loud validation: MLA needs the q-LoRA rank stated, 0 when
    there is none."""
    _arch_of_this_device()
    heads = NUM_HEADS_SWA
    torch.manual_seed(16)
    env = _Hy4MlaEnv(heads, max_batch=1, max_history=32, q_lora_rank=None)
    env.sinks = _sinks(heads, 16)
    env.add_request(0)
    q, k, v, latent = _random_context_inputs(32, heads)
    with pytest.raises(ValueError, match="q_lora_rank must be an int for MLA"):
        env.call_context([0], [32], q, k, v, latent)
