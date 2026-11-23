# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's mla_rope_generation variant.

Hy4-Flash's decode prologue: the catalog entry ``attention/mla_rope_generation``
copied into this target's ``ops/attention/`` and certified over the
checkpoint's own geometry -- a 512-wide latent plus a 64-wide rope half,
q_lora 1024, nope 192, value 256, at both of its head counts (64 for the SWA
layers, 32 for the sparse ones), their TP4 shards and the disaggregated context
engine's attn_tp-2 shards (32 for the SWA layers, 16 for the sparse ones) --
over a bf16 latent pool, at `P` up to 6 (DFly2's five drafts plus the anchor),
over an interleaved three-section mRoPE table at both of the checkpoint's RoPE
bases, with the
512-token SWA window below the KV length.

The last two cases are the DFly2 drafter's own call at tep4: 8 heads (the
sparse layers' TP4 shard), P=6, mRoPE at 1e7, over a 42-row pool -- one row per
text layer -- naming row 25, the first sparse layer the drafter reuses as its
KV source, with its longest sequence's block ending at the pool's served
length (5134, which a sparse layer also passes as its window). The second of
them leads the batch with a context row, so the generation rows resolve at
`num_contexts + g` rather than from 0. A single-row pool over a short KV cannot
show any of those three: the row a call names, the pool the row lives in, or
the shift of the generation tail.

The op reads paged-KV-cache addressing tensors and per-sequence length tensors
that the runtime derives from a KVCacheManagerV2 and a prepared
TrtllmAttentionMetadata, so the test builds that state for real rather than
hand-rolling offsets: the manager allocates the bf16 latent pool and the
metadata produces the device `sequence_length`, its host twin, the
block-offset table and the pool pointers. Every pool check then reads a token
back through the block table the manager handed the op.

The RoPE table is the engine's own. `RopeParams.create_rope_const_params()`
with `duplicate_data=True` builds the fp32 (cos, sin) float2 table the MLA
backend passes, and the mRoPE table is that same table gathered at
`position + offset(stream(pair))`: pair `i` reads the height stream when
`i % 3 == 1 and i < 3 * section[1]`, the width stream when `i % 3 == 2 and
i < 3 * section[2]`, the temporal one otherwise. That is the interleaved
layout, and at this checkpoint's section it splits the 32 pairs [12, 10, 10].
The offsets make the three streams differ, which is what a token whose streams
have advanced differently sees; at zero offsets the table is the plain
single-position one, and a control below measures how far apart the two are in
the bytes the kernel writes.

The roped halves are gated in ulps rather than bit-exactly. The kernel
evaluates `x*cos -+ y*sin` in fp32 and may contract it to FMA; the torch
mirror cannot, so the fp32 intermediate can differ by a rounding, which
occasionally survives into the bf16 result as a second ulp. The catalog's own
bf16 case is bit-exact only because its table is the identity rotation
(`cos, sin = 1, 0`), where the two forms agree exactly. The gates here allow a
measured band: at most `BF16_MAX_ULP` ulps on any element, and at most
`BF16_MAX_MULTI_ULP_FRACTION` of elements past one ulp. Everything that is not
a rotation -- `fused_q`'s absorbed-q head, `q_pe`, `latent_cache`, the copied
nope half of an appended row -- stays bit-exact.

Not driven: the op's quantized pool paths. No case here makes an fp8-e4m3
(`quant_mode=128`) or NVFP4 (`quant_mode=8192`) call, so `quant_q_buffer` and
both bmm scales are always None. `is_valid`'s guards for them are still
exercised, since a guard nothing drives is a guard nobody has shown admits a
real call.
"""

import inspect
import math
import zlib
from typing import List, Optional

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.attention.mla_rope_generation__mla_decode_prologue import (  # noqa: E501
    MROPE_OFFSETS,
    MROPE_SECTION,
    QM_FP8_KV_CACHE,
    MlaRopeGeneration,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.attention.mla_rope_generation__mla_decode_prologue import (  # noqa: E501
    mla_rope_generation as op,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "mla_rope_generation requires a CUDA device"

# ─── the checkpoint's MLA geometry ───────────────────────────────────────
# Sparse layers run 32 heads and SWA layers 64; the tep4 member shards both
# across four ranks, so 8 and 16 are the per-rank head counts it calls with.
NUM_HEADS_SWA = 64
NUM_HEADS_SPARSE = 32
NUM_HEADS_TP4_SWA = 16
NUM_HEADS_TP4_SPARSE = 8
# The disaggregated deployment's context engine runs tp2 (attn_tp 2), so its
# per-rank counts are 32 for the SWA layers and 16 for the sparse ones -- the
# same numbers the *other* layer kind carries at full width and at tp4, which is
# why each cell naming one of them names its base too.
NUM_HEADS_CTX_SWA = 32
NUM_HEADS_CTX_SPARSE = 16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
V_HEAD_DIM = 256
Q_LORA_RANK = 1024
GEN_HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
# MLA's softmax scale folded into `q_scaling`: (nope + rope) ** -0.5.
Q_SCALING = 0.0625
# Pool page size: 32 is the engine default, 64 a tuned target's opt-in.
TOKENS_PER_BLOCK = 32
PAGE64 = 64
MAX_SEQ_LEN = 1024
#: The SWA layers' window. Every case below keeps it below the KV length of at
#: least one sequence, which is the condition the write path is certified for.
WINDOW = 512
#: A KV length past the window. Positions stay under MAX_SEQ_LEN.
LONG_KV = 700
THETA_SWA = 10_000.0
THETA_SPARSE = 10_000_000.0
#: The primary pool's shape at tep4: the checkpoint's 42 text layers, one pool
#: row per decoder layer (`MlaLayer._pool_row` returns the decoder index). The
#: drafter's block appends to the pool of the sparse layer it reuses, and the
#: first of those is layer 25 (`dims.target_kv_layers`), well above row 0.
NUM_TEXT_LAYERS = 42
DRAFT_TARGET_LAYER = 25
#: The pool's served length at the gate: DFly2 logged "served length 5134"
#: (the 5120 `max_seq_len` grown by the draft block). A sparse layer has no
#: window and passes the pool's maximum length as `attention_window_size`, so
#: 5134 is also the window these cells run under -- at, not below, the KV
#: length, which is the other end of the axis `WINDOW` sits on.
DRAFT_POOL_LEN = 5134
#: The prompt a context row leads the batch with. A context row is a fresh
#: request presented with its whole prompt as its query length, and the gate's
#: own prompts are GSM8K's (105-471 characters, so at most ~118 tokens); 128 is
#: that rounded up to a page boundary. The op never reads a context row's
#: blocks -- it resolves the generation tails at `num_contexts + g` -- so what
#: the row has to be is a real, non-empty request ahead of them.
CTX_PROMPT = 128

#: The band a roped element may sit in, measured in bf16 ulps of the largest
#: value in the tensor being compared. One ulp is the mirror's own rounding; the
#: extra room is the kernel's freedom to contract the fp32 `x*cos -+ y*sin` into
#: an FMA, which can push the fp32 intermediate across a bf16 rounding boundary.
#: Four ulps is 1.6% of the largest value -- far below what a wrong table, a
#: wrong position or a swapped pair produces, which the plain-table control
#: below measures at O(1) relative.
#:
#: The band is anchored to the tensor's own largest value rather than to each
#: element's, because the rotation cancels: `x*cos - y*sin` of two O(1) operands
#: is near zero whenever the angle is near a multiple of pi/4, and there the
#: fp32 residue differs between the kernel's FMA and the mirror by a last-bit
#: amount that is *many ulps of the residue's own magnitude* while staying
#: ~1e-7 of the operands' scale. The distribution bound below is what still
#: holds that per-element noise to a small fraction of the output.
BF16_MAX_ULP = 4
#: How much of a roped output may sit past BF16_MAX_ULP ulps of its own
#: magnitude -- the cancellation residues, whose relative error is unbounded but
#: whose count is a last-fp32-bit effect. Never fewer than two elements, so a
#: 64-element k_pe row is not judged by a single residue.
BF16_MAX_MULTI_ULP_FRACTION = 1e-3


def _bytes(x: torch.Tensor) -> torch.Tensor:
    """The tensor's raw bytes, flattened, whatever its view's contiguity."""
    return x.contiguous().view(torch.uint8).reshape(-1)


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Bit-exact comparison, signed zeros included."""
    assert got.shape == want.shape, (got.shape, want.shape)
    torch.testing.assert_close(
        _bytes(got),
        _bytes(want),
        rtol=0.0,
        atol=0.0,
        msg=lambda m: f"{what} not bit-exact\n{m}",
    )


def _bf16_ulp_distance(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per-element distance in bf16 ulps, from the stored bit patterns.

    A bf16's low 15 bits order it by magnitude, so the distance between two
    magnitudes is the number of representable bf16 values between them --
    exactly the ulp count, with no division and nothing to blow up near zero.
    """
    assert got.shape == want.shape, (got.shape, want.shape)
    gi = got.view(torch.int16).to(torch.int32) & 0x7FFF
    wi = want.view(torch.int16).to(torch.int32) & 0x7FFF
    return (gi - wi).abs()


def _assert_bf16_roped(got: torch.Tensor, want: torch.Tensor, what: str) -> tuple[int, float]:
    """The rotation's band, as a distribution rather than a worst element.

    Every element must sit within BF16_MAX_ULP bf16 ulps *of the tensor's
    largest value* -- an absolute bound, so a rotation that cancels cannot fail
    it -- and no more than BF16_MAX_MULTI_ULP_FRACTION of the elements may sit
    past BF16_MAX_ULP ulps of their own magnitude. Returns how many elements
    missed bit-exactness and the worst deviation as a fraction of the scale, so
    a caller can report the measured distribution instead of asserting it.
    """
    dist = _bf16_ulp_distance(got, want)
    diff = (got.float() - want.float()).abs()
    scale = float(want.float().abs().max())
    worst = float(diff.max())
    worst_frac = worst / scale if scale else 0.0
    tol = BF16_MAX_ULP * 2.0**-8 * scale
    assert worst <= tol, (
        f"{what}: worst element differs by {worst:.4g} "
        f"({worst_frac:.4g} of the largest value {scale:.4g}; allowed "
        f"{BF16_MAX_ULP} bf16 ulps of it, {tol:.4g}). got {float(got.flatten()[int(diff.argmax())]):.6g}, "
        f"want {float(want.flatten()[int(diff.argmax())]):.6g}"
    )
    past = int((dist > BF16_MAX_ULP).sum())
    allowed = max(2, int(BF16_MAX_MULTI_ULP_FRACTION * got.numel()))
    assert past <= allowed, (
        f"{what}: {past}/{got.numel()} elements are past {BF16_MAX_ULP} ulps of their "
        f"own magnitude (allowed {allowed}); worst is {worst_frac:.4g} of the scale"
    )
    return int((dist > 0).sum()), worst_frac


def _mrope_streams(section=MROPE_SECTION) -> List[int]:
    """Which position stream each of the 32 rope pairs reads.

    The interleaved mRoPE layout: pair `i` reads row 1 (height) when
    `i % 3 == 1 and i < 3 * section[1]`, row 2 (width) when `i % 3 == 2 and
    i < 3 * section[2]`, and row 0 (temporal) otherwise. The counts come out as
    the config's section, which is what makes this table the checkpoint's.
    """
    streams = []
    for i in range(QK_ROPE_HEAD_DIM // 2):
        if i % 3 == 1 and i < 3 * section[1]:
            streams.append(1)
        elif i % 3 == 2 and i < 3 * section[2]:
            streams.append(2)
        else:
            streams.append(0)
    return streams


class _MlaEnv:
    """Real op state: an MLA KV cache manager and the engine's RoPE table.

    `quant_mode` is 0 -- the bf16 latent pool this target runs. The manager's
    element type and page layout are asserted against what the case claims,
    because the op sizes its page slabs from `tokens_per_block` and
    `quant_mode` and reads its table from the tensor it is handed.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        num_heads: int = NUM_HEADS_SWA,
        rope_theta: float = THETA_SWA,
        quant_mode: int = 0,
        mrope_section=MROPE_SECTION,
        mrope_offsets=MROPE_OFFSETS,
        num_layers: int = 1,
        max_seq_len: int = MAX_SEQ_LEN,
        max_tokens: int = 131072,
    ) -> None:
        assert quant_mode == 0, (
            f"quant_mode must be 0 (a bf16 latent pool); got {quant_mode}. "
            "The quantized pool paths are not certified by this target's cells"
        )
        self.max_batch_size = max_batch_size
        self.tokens_per_block = tokens_per_block
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section)
        self.mrope_offsets = tuple(mrope_offsets)
        # The pool's own axes: how many layers it holds (one row each, so a
        # layer above 0 is a pool row above 0) and the length it serves.
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
            CacheType.SELFKONLY,  # MLA latent cache: kv_factor=1, one kv head
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=GEN_HEAD_SIZE,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.BF16,
        )
        assert self.kv_cache_manager.tokens_per_block == tokens_per_block
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None and pool.dtype == torch.bfloat16
        # [pages, 1, slots, 1, head_dim]: the op addresses a token as
        # (page, 0, slot, 0), so the slot dim has to hold a page's tokens.
        assert pool.dim() == 5 and pool.shape[1] == 1 and pool.shape[3] == 1, pool.shape
        assert pool.shape[2] >= tokens_per_block, pool.shape

        # The engine's own duplicated-layout table: entry (p, j) is the float2
        # (cos, sin) of pair j // 2 at absolute position p, which is exactly how
        # the kernel reads `cos_sin_cache + ROPE_DIM * position`.
        rope = RopeParams(
            dim=QK_ROPE_HEAD_DIM,
            theta=rope_theta,
            max_positions=max_seq_len + max(self.mrope_offsets),
            duplicate_data=True,
        )
        _, base = rope.create_rope_const_params()
        assert base is not None
        self._base = base.view(-1, QK_ROPE_HEAD_DIM, 2)
        self.rotary_cos_sin = self.mrope_table(self.mrope_offsets)

    def mrope_table(self, offsets) -> torch.Tensor:
        """The table whose three position streams sit `offsets` apart.

        Built by gathering the engine's single-position table at
        `position + offsets[stream(pair)]`, so a stream at offset 0 is the
        engine's own bytes and the others differ only by the position they read
        -- no second derivation of `inv_freq` to disagree with the first.

        One position of this table is `2 * QK_ROPE_HEAD_DIM` floats:
        `create_sinusoidal_positions_for_attention_plugin(duplicate_data=True)`
        repeats the 32 `(cos, sin)` pairs, which is the shape the kernel's
        `cos_sin_cache + 2 * ROPE_DIM * position` strides over. The gather
        takes the first copy and the concatenation restores the second, so a
        zero-offset stream reproduces the engine's own bytes exactly.
        """
        streams = _mrope_streams(self.mrope_section)
        per_pair = torch.tensor(
            [offsets[s] for s in streams], dtype=torch.long, device=self._base.device
        )
        rows = torch.arange(self.max_seq_len, device=self._base.device)
        pairs = torch.arange(QK_ROPE_HEAD_DIM // 2, device=self._base.device)
        index = rows[:, None] + per_pair[None, :]
        first_copy = self._base[index, pairs[None, :]]  # [max_seq_len, 32, 2]
        return torch.cat([first_copy, first_copy], dim=1).reshape(1, -1).contiguous()

    def plain_table(self) -> torch.Tensor:
        """The single-position table: every stream at the token's own position."""
        return self._base[: self.max_seq_len].reshape(1, -1).contiguous()

    def prepare_metadata(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        num_contexts: int,
        cached_lens: List[int],
    ) -> TrtllmAttentionMetadata:
        metadata = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=8192,
            kv_cache_manager=self.kv_cache_manager,
        )
        metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
        metadata.num_contexts = num_contexts
        metadata.request_ids = request_ids
        metadata.prompt_lens = cached_lens
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_lens,
        )
        metadata.prepare()
        return metadata

    def rope_ref(self, x: torch.Tensor, position: int, table=None) -> torch.Tensor:
        """GPT-J interleaved rotation of the last dim, fp32 math, x's dtype."""
        table = self.rotary_cos_sin if table is None else table
        half = QK_ROPE_HEAD_DIM // 2
        rows = table.view(-1, QK_ROPE_HEAD_DIM, 2)
        cos = rows[position, :half, 0]
        sin = rows[position, :half, 1]
        pairs = x.float().reshape(*x.shape[:-1], half, 2)
        out = torch.empty_like(pairs)
        out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        return out.reshape(x.shape).to(x.dtype)

    def pool_tensor(self, layer_idx: int = 0) -> torch.Tensor:
        """One pool row's zero-copy view: the layer's own blocks, in its order.

        `get_buffers` is documented as the layer's view and
        `get_batch_cache_indices(..., layer_idx=...)` as the indices into that
        view, so the pair is the read-back for any pool row -- not just 0.
        """
        pool = self.kv_cache_manager.get_buffers(layer_idx)
        assert pool is not None
        return pool

    def layer_base(self, layer_idx: int) -> int:
        """Where the op must resolve `layer_idx` to: pool pointer + mapping row.

        The op addresses a layer as `poolPointers[pool] + offset * kv_factor *
        bytesPerBlock`, the pair `kv_cache_pool_mapping` holds -- the same
        arithmetic the model's own sparse path resolves a pool row with
        (`DSACacheManagerV2.get_primary_pool_page_index_params`). Measured
        against the layer's own view base, so a harness that read the wrong
        row would fail here rather than pass on a misread pool.
        """
        mapping = self.kv_cache_manager.kv_cache_pool_mapping
        pointers = self.kv_cache_manager.kv_cache_pool_pointers
        pool_id, offset = int(mapping[layer_idx][0]), int(mapping[layer_idx][1])
        bytes_per_block = self.tokens_per_block * GEN_HEAD_SIZE * 2  # bf16, kv_factor 1
        return int(pointers[pool_id][0]) + offset * bytes_per_block

    def pool_page(self, layer_idx: int, page: int) -> int:
        """Which page of the whole pool `page` of that pool row lands on.

        One pool holds every row: the coalescing key that forms a pool has no
        layer in it (`createStorageConfig`'s `(coalescingGroup, size,
        sparsity)`) and neither has `makeLifeCycle`, so a 42-layer MLA cache
        gets one pool whose index scale is the row count -- a slot id maps to
        42 page indices, one per row. The rows are therefore interleaved page
        by page: row `L`'s page `p` is the pool's page `mapping[L][1] + p`,
        which is the offset the op re-applies as a byte displacement. Row 0's
        view starts at the pool's page 0, so its index is the pool's page
        number and this is the map into it.
        """
        mapping = self.kv_cache_manager.kv_cache_pool_mapping
        return int(mapping[layer_idx][1]) + page

    def blocks(self, request_id: int, layer_idx: int = 0) -> List[int]:
        return list(
            self.kv_cache_manager.get_batch_cache_indices([request_id], layer_idx=layer_idx)[0]
        )

    def slot(self, request_id: int, position: int, layer_idx: int = 0) -> tuple[int, int]:
        tpb = self.tokens_per_block
        return self.blocks(request_id, layer_idx)[position // tpb], position % tpb

    def cache_row(self, request_id: int, position: int, layer_idx: int = 0) -> torch.Tensor:
        """Read one token's latent row back from a pool row."""
        page, offset = self.slot(request_id, position, layer_idx)
        return self.pool_tensor(layer_idx)[page, 0, offset, 0]

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


def _make_q_pe(rows: int, num_heads: int, contiguous: bool) -> torch.Tensor:
    if contiguous:
        return torch.randn(rows, num_heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    # The MLA projection's strided view: q_pe sliced out of a packed q tensor,
    # which is what the model hands the op.
    q = torch.randn(
        rows,
        num_heads,
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return q.split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)[1]


def _positions(
    kv_lens: List[int], num_contexts: int, num_gen: int, tokens_per_seq: int
) -> List[int]:
    """0-based absolute position of every generation row, in row order.

    Row `r` is generation sequence `r // P`'s `r % P`-th new token; the P new
    tokens of one sequence occupy consecutive positions ending at its total KV
    length minus one.
    """
    return [
        kv_lens[num_contexts + g] - tokens_per_seq + t
        for g in range(num_gen)
        for t in range(tokens_per_seq)
    ]


def _cached_lens(
    tokens_per_block: int,
    predicted_tokens_per_seq: int,
    num_seqs: int,
    long_kv: int = LONG_KV,
    max_seq_len: int = MAX_SEQ_LEN,
) -> List[int]:
    """KV lengths that put one sequence past the window and the rest on the
    page alignments a P-row block can land on.

    Sequence 0 sits at `long_kv` cached tokens, past the 512-token window (and,
    for a cell that names one, at the pool's served length). The next three put
    the P new rows two slots before a page's end, on its last slot, and on the
    first slot of the next page, so a `P > 1` block straddles a page boundary
    and a `P = 1` call lands on all three alignments. The rest walk one slot at
    a time past the boundary, which is what fills a 64-sequence row.
    """
    assert num_seqs >= 4, num_seqs
    cached = [long_kv, tokens_per_block - 2, tokens_per_block - 1, tokens_per_block]
    cached += [tokens_per_block + 1 + i for i in range(num_seqs - 4)]
    assert len(set(cached)) == num_seqs, cached
    assert max(cached) + predicted_tokens_per_seq <= max_seq_len, cached
    return cached


class _Step:
    """One prepared bf16 generation step, callable with a window override.

    The batch is `request_ids` order: `num_contexts` context rows first, then
    the generation sequences -- the layout the engine's own metadata presents
    and the op resolves its rows from. `call` is the only place this file names
    the op's positional arguments, so the window case, the mixed batch and every
    other case drive the same call.
    """

    def __init__(
        self,
        env: _MlaEnv,
        request_ids: List[int],
        cached_lens: List[int],
        predicted_tokens_per_seq: int,
        q_scaling: float = Q_SCALING,
        q_lora_rank: int = Q_LORA_RANK,
        q_pe_contiguous: bool = False,
        table: Optional[torch.Tensor] = None,
        window: int = WINDOW,
        num_contexts: int = 0,
        context_lens: Optional[List[int]] = None,
        layer_idx: int = 0,
    ) -> None:
        self.env = env
        self.p = predicted_tokens_per_seq
        self.q_scaling = q_scaling
        self.q_lora_rank = q_lora_rank
        self.window = window
        self.num_contexts = num_contexts
        self.layer_idx = layer_idx
        self.request_ids = list(request_ids)
        self.ctx_ids = self.request_ids[:num_contexts]
        self.gen_ids = self.request_ids[num_contexts:]
        self.ctx_cached = list(cached_lens[:num_contexts])
        self.cached_lens = list(cached_lens[num_contexts:])
        num_gen = len(self.gen_ids)
        for rid in self.gen_ids:
            cache = env.kv_cache_manager.kv_cache_map[rid]
            assert cache.resize(cache.capacity + self.p)
        # A context row carries its own prompt as its query length and, being a
        # fresh request, no history; a generation row carries `P` new tokens on
        # top of its cached length. The op reads the generation rows from
        # `num_contexts` on and must not touch the context ones.
        if context_lens is None:
            assert num_contexts == 0, (
                "a context row carries its own prompt as its query length; pass context_lens"
            )
            context_lens = []
        self.ctx_seq_lens = list(context_lens)
        seq_lens = self.ctx_seq_lens + [self.p] * num_gen
        self.ctx_kv_lens = [c + s for c, s in zip(self.ctx_cached, self.ctx_seq_lens)]
        self.gen_kv_lens = [c + self.p for c in self.cached_lens]
        self.kv_lens = self.ctx_kv_lens + self.gen_kv_lens
        self.metadata = env.prepare_metadata(
            self.request_ids, seq_lens, num_contexts, list(cached_lens)
        )
        self.positions = _positions(self.kv_lens, num_contexts, num_gen, self.p)
        self.rows = num_gen * self.p
        self.table = env.rotary_cos_sin if table is None else table
        num_heads = env.num_heads
        self.fused_q = torch.randn(
            self.rows, num_heads, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        self.q_pe = _make_q_pe(self.rows, num_heads, q_pe_contiguous)
        self.latent_cache = torch.randn(
            self.rows, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        self.fused_q_orig = self.fused_q.clone()
        self.q_pe_orig = self.q_pe.clone()
        self.latent_orig = self.latent_cache.clone()
        self.cu_q_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
        self.cu_kv_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
        self.fmha_scheduler_counter = torch.full((1,), 7, dtype=torch.uint32, device="cuda")
        self.roped_inexact = 0
        self.roped_total = 0
        self.roped_worst = 0.0

    def call(
        self, window: Optional[int] = None, table: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        env = self.env
        md = self.metadata
        with validating(op):
            out = op(
                self.fused_q,
                self.q_pe,
                self.latent_cache,
                self.table if table is None else table,
                self.cu_q_seqlens,
                self.cu_kv_seqlens,
                self.fmha_scheduler_counter,
                None,  # mla_bmm1_scale -- the bf16 pool writes neither scale
                None,  # mla_bmm2_scale
                None,  # quant_q_buffer
                md.kv_lens_cuda_runtime,
                md.kv_lens_runtime,
                md.prompt_lens_cpu_runtime,
                self.num_contexts,
                md.kv_cache_block_offsets,
                env.kv_cache_manager.kv_cache_pool_pointers,
                env.kv_cache_manager.kv_cache_pool_mapping,
                None,  # kv_scale_orig_quant
                None,  # kv_scale_quant_orig
                None,  # kv_cache_scale_orig_quant
                None,  # out_scale
                None,  # block_ids_per_seq
                [None, None],  # helix_tensor_params
                self.p,
                self.layer_idx,
                env.num_heads,
                1,  # num_kv_heads
                GEN_HEAD_SIZE,
                0,  # residual_dim
                env.tokens_per_block,
                self.window if window is None else window,
                1,  # beam_width
                0,  # quant_mode: the bf16 latent pool
                self.q_scaling,
                self.q_lora_rank,
                KV_LORA_RANK,
                QK_NOPE_HEAD_DIM,
                QK_ROPE_HEAD_DIM,
                V_HEAD_DIM,
                True,  # rope_append
            )
        torch.cuda.synchronize()
        return out


def _assert_scheduler_buffers(
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    num_gen: int,
    num_heads: int,
    gen_kv_lens: List[int],
    predicted_tokens_per_seq: int,
) -> None:
    # cu_q counts q rows: num_heads rows per generation token, and a
    # generation sequence contributes predicted_tokens_per_seq of them.
    expected_cu_q = (
        torch.arange(num_gen + 1, dtype=torch.int32) * num_heads * predicted_tokens_per_seq
    )
    expected_cu_kv = torch.zeros(num_gen + 1, dtype=torch.int32)
    expected_cu_kv[1:] = torch.tensor(gen_kv_lens, dtype=torch.int32).cumsum(0)
    torch.testing.assert_close(cu_q_seqlens.cpu(), expected_cu_q)
    torch.testing.assert_close(cu_kv_seqlens.cpu(), expected_cu_kv)
    assert fmha_scheduler_counter.item() == 0


def _verify(
    step: _Step, pool_before: torch.Tensor, other_before: Optional[torch.Tensor] = None
) -> None:
    """Every effect of one bf16 generation call, against the pool snapshot.

    1. The roped `q_pe` in `fused_q`'s rope tail, its absorbed-q head and both
       inputs bitwise untouched.
    2. The appended rows: `[compressed_kv | roped k_pe]` at each token's own
       position, the nope half copied verbatim, in the layer's own pool row.
    3. Nothing else in that pool row moved -- and, when the call names a row
       above 0, nothing moved anywhere else in the pool either. Row 0's view is
       the whole pool (`pool_page`), so the second half of this names the pool
       pages the call's own rows land on and refuses any other page moving.
    4. The scheduler buffers, over the generation sequences only.
    """
    env = step.env
    roped_q = torch.stack(
        [env.rope_ref(step.q_pe_orig[r], step.positions[r], step.table) for r in range(step.rows)]
    )
    inexact, worst = _assert_bf16_roped(
        step.fused_q[..., KV_LORA_RANK:], roped_q, "fused_q roped tail"
    )
    step.roped_inexact += inexact
    step.roped_worst = max(step.roped_worst, worst)
    step.roped_total += roped_q.numel()
    _assert_bytes_equal(
        step.fused_q[..., :KV_LORA_RANK],
        step.fused_q_orig[..., :KV_LORA_RANK],
        "fused_q absorbed-q head",
    )
    _assert_bytes_equal(step.q_pe, step.q_pe_orig, "q_pe")
    _assert_bytes_equal(step.latent_cache, step.latent_orig, "latent_cache")

    for r in range(step.rows):
        rid = step.gen_ids[r // step.p]
        row = env.cache_row(rid, step.positions[r], step.layer_idx)
        _assert_bytes_equal(
            row[:KV_LORA_RANK],
            step.latent_orig[r, :KV_LORA_RANK],
            f"appended compressed_kv (request {rid}, position {step.positions[r]}, "
            f"pool row {step.layer_idx})",
        )
        roped_k = env.rope_ref(step.latent_orig[r, KV_LORA_RANK:], step.positions[r], step.table)
        inexact, worst = _assert_bf16_roped(
            row[KV_LORA_RANK:],
            roped_k,
            f"appended k_pe (request {rid}, position {step.positions[r]})",
        )
        step.roped_inexact += inexact
        step.roped_worst = max(step.roped_worst, worst)
        step.roped_total += roped_k.numel()

    pool = env.pool_tensor(step.layer_idx)
    changed = (pool.view(torch.int16) != pool_before.view(torch.int16)).nonzero().cpu()
    got_slots = set(zip(changed[:, 0].tolist(), changed[:, 2].tolist()))
    expected_slots = {
        env.slot(step.gen_ids[r // step.p], step.positions[r], step.layer_idx)
        for r in range(step.rows)
    }
    assert got_slots <= expected_slots, (
        f"op touched pool slots {sorted(got_slots - expected_slots)} "
        f"outside {sorted(expected_slots)} in pool row {step.layer_idx}"
    )
    # Upper bound, not equality: a written element that happens to match the
    # one already there (a zero over a zeroed pool) is invisible to a snapshot
    # diff. The rows' contents are pinned above.
    assert changed.shape[0] <= step.rows * GEN_HEAD_SIZE, (
        f"op wrote {changed.shape[0]} pool elements, at most "
        f"{step.rows * GEN_HEAD_SIZE} rows' worth expected"
    )

    if other_before is not None:
        # A layer's row is where the op's mapping lookup must land. `other_before`
        # is row 0's view, which starts at the pool's page 0 and so holds every
        # row of the pool: an offset that missed the row writes pages the call's
        # own rows do not name, and they show up here. The rows' pages are named
        # through `pool_page` -- the pool's index scale is the row count, so a
        # layer's page index is not the pool's page number.
        pool_all = env.pool_tensor(0)
        moved = (pool_all.view(torch.int16) != other_before.view(torch.int16)).any(dim=(1, 2, 3, 4))
        got_pages = set(moved.nonzero().flatten().tolist())
        want_pages = {env.pool_page(step.layer_idx, page) for page, _ in expected_slots}
        assert got_pages <= want_pages, (
            f"a call naming pool row {step.layer_idx} moved pool pages "
            f"{sorted(got_pages - want_pages)} outside its own {sorted(want_pages)}"
        )

    _assert_scheduler_buffers(
        step.cu_q_seqlens,
        step.cu_kv_seqlens,
        step.fmha_scheduler_counter,
        len(step.gen_ids),
        env.num_heads,
        step.gen_kv_lens,
        step.p,
    )


def _run_and_check(env: _MlaEnv, request_ids, cached_lens, p, repeats: int = 1, **step_kwargs):
    """Prepare one step, drive it, verify every effect, then re-drive it.

    The repeat is the idempotence claim: the position comes from
    `sequence_length`, so a second call rewrites the same slots and the same
    bytes rather than double-appending.

    The snapshot is the pool row the call names -- and, for a call naming a row
    above 0, row 0's view as well, which is the whole pool (its rows are
    interleaved page by page, see `pool_page`), so an offset that misses the row
    shows up as pages moving that the call's own rows do not name rather than as
    a pass. `layer_base` pins that row's view to the address the op's own mapping
    arithmetic resolves, so a harness diffing the wrong row fails here instead of
    reading a row the kernel never wrote.
    """
    step = _Step(env, request_ids, cached_lens, p, **step_kwargs)
    layer = step.layer_idx
    pool = env.pool_tensor(layer)
    assert env.layer_base(layer) == pool.data_ptr(), (
        f"pool row {layer}'s view does not start where the op's mapping resolves it: "
        f"view {pool.data_ptr()}, op {env.layer_base(layer)}"
    )
    pool_before = pool.clone()
    other_before = None if layer == 0 else env.pool_tensor(0).clone()
    out = step.call()
    assert out is step.fused_q, "the op hands its q buffer back"
    _verify(step, pool_before, other_before)
    for _ in range(repeats - 1):
        pool_ref = pool.clone()
        q_ref = step.fused_q.clone()
        cu_ref = step.cu_kv_seqlens.clone()
        step.call()
        torch.testing.assert_close(pool.view(torch.int16), pool_ref.view(torch.int16))
        torch.testing.assert_close(step.fused_q, q_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(step.cu_kv_seqlens, cu_ref, rtol=0.0, atol=0.0)
    return step


# ─── the checkpoint's shapes ──────────────────────────────────────────────


def test_swa_decode_64_heads_page32() -> None:
    """The SWA layers' production decode: 64 heads, P=1, 32-token pages, a
    bf16 pool and the 1e4 table. Four sequences at positions 700 (past the
    512-token window), 30, 31 (a page's last slot) and 32 (the next page's
    first). Run twice."""
    torch.manual_seed(10)
    env = _MlaEnv(num_heads=NUM_HEADS_SWA, tokens_per_block=TOKENS_PER_BLOCK)
    try:
        cached = [LONG_KV, 30, 31, 32]
        rids = list(range(4))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        step = _run_and_check(env, rids, cached, 1, repeats=2)
        assert step.positions == [700, 30, 31, 32]
    finally:
        env.shutdown()


def test_swa_decode_64_heads_page64() -> None:
    """The same decode over 64-token pages: the page slab is twice as wide, so
    the append's vector stride and the block table's shift both change. The
    three short sequences land two slots before a page's end, on its last slot
    and on the next page's first."""
    torch.manual_seed(11)
    env = _MlaEnv(num_heads=NUM_HEADS_SWA, tokens_per_block=PAGE64)
    try:
        cached = [LONG_KV, 62, 63, 64]
        rids = list(range(4))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        step = _run_and_check(env, rids, cached, 1)
        assert step.positions == [700, 62, 63, 64]
    finally:
        env.shutdown()


def test_dfly2_verify_64_heads_p6() -> None:
    """DFly2's verify step: P=6, 64 heads, bf16 pool, the SWA layers' 1e4
    table. Five drafts plus the anchor occupy six consecutive positions per
    sequence, so sequence 1's block ends on page 0's last slot, sequence 2's
    straddles the boundary, and sequence 3's starts a fresh page. Run twice:
    a repeated P > 1 call is idempotent, not a double append."""
    torch.manual_seed(12)
    env = _MlaEnv(num_heads=NUM_HEADS_SWA, tokens_per_block=TOKENS_PER_BLOCK)
    try:
        cached = [LONG_KV, 30, 31, 32]
        rids = list(range(4))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        step = _run_and_check(env, rids, cached, 6, repeats=2)
        assert step.positions[:6] == [700, 701, 702, 703, 704, 705]
        assert step.positions[6:12] == [30, 31, 32, 33, 34, 35]
        # Sequence 2's six rows really do cross a 32-slot page boundary.
        assert step.positions[12] // 32 != step.positions[17] // 32
    finally:
        env.shutdown()


def test_sparse_decode_32_heads_p1() -> None:
    """The sparse layers' decode: 32 heads, P=1, bf16 pool, and the 1e7 base
    the sparse layers' table is built at. The base is what the op reads out of
    the table it is handed, so driving only one of the checkpoint's two bases
    would leave the other's table untested."""
    torch.manual_seed(13)
    env = _MlaEnv(
        num_heads=NUM_HEADS_SPARSE,
        tokens_per_block=TOKENS_PER_BLOCK,
        rope_theta=THETA_SPARSE,
    )
    try:
        cached = [LONG_KV, 30, 31, 32]
        rids = list(range(4))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        _run_and_check(env, rids, cached, 1)
    finally:
        env.shutdown()


def test_sparse_verify_32_heads_p6() -> None:
    """The sparse layers' verify step: 32 heads at P=6 over the 1e7 table."""
    torch.manual_seed(14)
    env = _MlaEnv(
        num_heads=NUM_HEADS_SPARSE,
        tokens_per_block=TOKENS_PER_BLOCK,
        rope_theta=THETA_SPARSE,
    )
    try:
        cached = [LONG_KV, 30, 31, 32]
        rids = list(range(4))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        _run_and_check(env, rids, cached, 6, repeats=2)
    finally:
        env.shutdown()


def test_per_rank_shard_heads() -> None:
    """The per-rank head counts this target is called with: the tep4 member's 16
    (SWA) and 8 (sparse), and the disaggregated context engine's attn_tp-2 32
    (SWA) and 16 (sparse). The head count is an op argument, so a per-rank shard
    is a different call rather than a scaled-down copy of the full one; the two
    ranks' numbers cross the layer kinds, and each runs at its kind's base."""
    torch.manual_seed(15)
    for heads, theta in (
        (NUM_HEADS_TP4_SWA, THETA_SWA),
        (NUM_HEADS_TP4_SPARSE, THETA_SPARSE),
        (NUM_HEADS_CTX_SWA, THETA_SWA),
        (NUM_HEADS_CTX_SPARSE, THETA_SPARSE),
    ):
        env = _MlaEnv(num_heads=heads, tokens_per_block=TOKENS_PER_BLOCK, rope_theta=theta)
        try:
            cached = [LONG_KV, 30, 31, 32]
            rids = list(range(4))
            env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
            _run_and_check(env, rids, cached, 1)
        finally:
            env.shutdown()


def test_mrope_sections_move_the_rope_bytes() -> None:
    """The interleaved mRoPE table is the one the kernel reads, not a plain one.

    The two tables are the same bytes on the temporal pairs and different on
    every height and width pair, which is what the section [12, 10, 10] says.
    Then the appended k_pe half is compared against the plain-table mirror: the
    fraction of bytes and the worst relative deviation it moves are what make
    the other cases' ulp-band comparison evidence that the kernel read the
    mRoPE table. (At zero offsets the two tables are the same table, so a test
    that built only the plain one would pass every other check in this file.)
    """
    torch.manual_seed(16)
    env = _MlaEnv(num_heads=NUM_HEADS_SWA, tokens_per_block=TOKENS_PER_BLOCK)
    try:
        cached = [LONG_KV, 30, 31, 32]
        rids = list(range(4))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        step = _run_and_check(env, rids, cached, 6)

        streams = _mrope_streams()
        assert list(torch.bincount(torch.tensor(streams), minlength=3).tolist()) == list(
            MROPE_SECTION
        )
        mrope = step.table.view(-1, QK_ROPE_HEAD_DIM, 2)
        plain = env.plain_table().view(-1, QK_ROPE_HEAD_DIM, 2)
        pos = step.positions[0]
        # The engine's table repeats the 32 pairs inside a position -- 2 *
        # QK_ROPE_HEAD_DIM floats, which is the stride the kernel reads it
        # with. A table holding only one copy is half as long and the kernel
        # reads past its end.
        assert (mrope[pos][: QK_ROPE_HEAD_DIM // 2] == mrope[pos][QK_ROPE_HEAD_DIM // 2 :]).all(), (
            "the engine's table does not carry the duplicated pair layout"
        )
        # One entry per pair: both of the pair's (cos, sin) components agree.
        same = (mrope[pos] == plain[pos]).all(-1).tolist()
        for i in range(QK_ROPE_HEAD_DIM // 2):
            pair_same = same[i]
            assert pair_same == (streams[i] == 0), (
                f"pair {i} (stream {streams[i]}) "
                f"{'agrees' if pair_same else 'differs'} with the plain table"
            )

        got = env.cache_row(step.gen_ids[0], pos)[KV_LORA_RANK:]
        mirror = env.rope_ref(step.latent_orig[0, KV_LORA_RANK:], pos, env.plain_table())
        frac = float((_bytes(got) != _bytes(mirror)).float().mean())
        # The same scale-anchored statistic the band uses, so the separation is
        # measured in the units the band is stated in: a wrong table moves
        # elements by O(scale), the kernel's own FMA noise by ~1e-7 of it.
        ctl_worst = float((got.float() - mirror.float()).abs().max()) / float(
            got.float().abs().max()
        )
        print(
            f"mRoPE control: {frac:.3f} of the k_pe bytes differ; worst {ctl_worst:.3g} "
            f"of the scale (band allows {BF16_MAX_ULP * 2.0**-8:.3g})"
        )
        # 20 of the 32 pairs sit off the temporal stream, so 40 of the 64 rope
        # elements are expected to move; the gates sit well below that.
        assert frac > 0.25, f"the plain-table mirror moved only {frac:.3f} of the k_pe bytes"
        assert ctl_worst > 0.25, (
            f"the plain-table mirror stayed within {ctl_worst:.3g} of the scale"
        )
    finally:
        env.shutdown()


def test_window_does_not_reach_the_append() -> None:
    """The 512-token window changes nothing this op writes.

    `attention_window_size` sizes the reader's cyclic remap and the block
    array's window fields; the append addresses the pool at the token's
    absolute position through the raw block pointer. So the same prepared step
    issued with the window at 512 and raised past the KV length writes the same
    pool bytes, the same `fused_q` and the same scheduler buffers -- which is
    what makes every other case here, all of them windowed below their KV
    length, evidence about the same write path. The batch really does hold a
    sequence past the window, or this would prove nothing.
    """
    torch.manual_seed(17)
    env = _MlaEnv(num_heads=NUM_HEADS_SWA, tokens_per_block=TOKENS_PER_BLOCK)
    try:
        cached = [LONG_KV, 30, 31, 32]
        rids = list(range(4))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        step = _Step(env, rids, cached, 6)
        assert max(step.kv_lens) > WINDOW
        pool = env.pool_tensor()
        images = []
        for window in (WINDOW, MAX_SEQ_LEN):
            pool.zero_()
            step.call(window=window)
            images.append(
                (
                    pool.view(torch.int16).clone(),
                    step.fused_q.clone(),
                    step.cu_q_seqlens.clone(),
                    step.cu_kv_seqlens.clone(),
                )
            )
        for got, want in zip(images[1:], images[:1]):
            for g, w in zip(got, want):
                torch.testing.assert_close(g, w, rtol=0.0, atol=0.0)
    finally:
        env.shutdown()


# ─── the entry's guard ────────────────────────────────────────────────────


def _scale_tensor(value: float) -> torch.Tensor:
    return torch.full((1,), value, dtype=torch.float32, device="cuda")


def _guard_args(**overrides: object) -> dict:
    """Every `is_valid` parameter, inert, with the named ones replaced.

    Read off the signature rather than written out. The guard mirrors a
    47-parameter op because `validating` forwards positionally, and a
    hand-counted positional list here would shift silently the day a parameter
    is inserted -- landing a scale tensor in `out_scale` and testing nothing.
    """
    args = dict.fromkeys(inspect.signature(op.is_valid).parameters)
    # The plain row layout and the checkpoint's rope geometry: what every call
    # this target makes passes, and what a guard that reads the geometry needs.
    args.update(residual_dim=0, quant_mode=0, qk_rope_head_dim=QK_ROPE_HEAD_DIM)
    args.update(overrides)
    return args


def test_is_valid_refuses_a_non_plain_row_and_a_folded_norm() -> None:
    """The two metadata guards this target's own calls must pass.

    A non-zero `residual_dim` describes the DSA PrimTS or residual record,
    whose block scales live inside the data row; this target writes plain rows,
    so a non-zero value would read bytes that are not there. A
    `kv_norm_weight` folds kv_a_layernorm into the kernel, which then reads
    `latent_cache` unnormalized -- and every caller here has normalized, so it
    would normalize twice, plausibly.
    """
    with pytest.raises(AssertionError, match="residual_dim"):
        op.is_valid(**_guard_args(residual_dim=QK_ROPE_HEAD_DIM))
    with pytest.raises(AssertionError, match="kv_norm_weight"):
        op.is_valid(
            **_guard_args(
                kv_norm_weight=torch.ones(KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
            )
        )
    # The call every case above makes is admitted.
    op.is_valid(**_guard_args())


def test_is_valid_refuses_fp8_without_a_quantized_query_buffer() -> None:
    """Over an fp8 pool the quantized query has nowhere to go without
    `quant_q_buffer`, and the op does not presence-check it: the kernel
    launches into a null pointer and the CUDA context dies. The guard is the
    only place that can be caught. (The target's bf16 pool needs no buffer,
    which is why the guard is keyed on the quant_mode bit.)"""
    with pytest.raises(AssertionError, match="quant_q_buffer"):
        op.is_valid(**_guard_args(quant_mode=QM_FP8_KV_CACHE, quant_q_buffer=None))
    op.is_valid(
        **_guard_args(
            quant_mode=QM_FP8_KV_CACHE,
            quant_q_buffer=torch.empty(1, dtype=torch.uint8, device="cuda"),
        )
    )
    op.is_valid(**_guard_args(quant_mode=0, quant_q_buffer=None))


def test_is_valid_refuses_a_non_reciprocal_scale_pair() -> None:
    """The two KV scales are independent: `w` quantizes what the call writes,
    `r` is folded into the scales the consuming FMHA dequantizes with. That
    cancellation only holds when `r = 1/w`, and a non-reciprocal pair
    mis-scales the decode with nothing downstream able to see it."""
    with pytest.raises(AssertionError, match="reciprocal"):
        op.is_valid(
            **_guard_args(
                kv_scale_orig_quant=_scale_tensor(0.25),
                kv_scale_quant_orig=_scale_tensor(3.0),
            )
        )
    op.is_valid(
        **_guard_args(
            kv_scale_orig_quant=_scale_tensor(0.5),
            kv_scale_quant_orig=_scale_tensor(2.0),
        )
    )
    # Both omitted is the production call: 1.0 and 1.0, which are reciprocal.
    op.is_valid(**_guard_args())


def test_is_valid_refuses_a_table_shorter_than_the_positions() -> None:
    """The kernel strides the rope table `2 * qk_rope_head_dim` floats per
    position -- the duplicated pair layout -- and reads it up to
    `max(sequence_length) - 1`. A `duplicate_data=False` table holds one copy
    per position, half the floats, and the read past its end is unchecked: it
    surfaces later as an illegal memory access that kills the CUDA context,
    with nothing pointing at the call, so the guard is the only place it can be
    caught. The unguarded probe is recorded in the entry's `note` rather than
    run here, because losing the context loses the rest of the file's results.
    """
    lengths = torch.tensor([LONG_KV, 8, 8, 8], dtype=torch.int32, device="cuda")
    plain = torch.zeros(MAX_SEQ_LEN * QK_ROPE_HEAD_DIM, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="rotary_cos_sin holds"):
        op.is_valid(**_guard_args(rotary_cos_sin=plain, sequence_length=lengths))
    # The duplicated layout -- what `RopeParams(duplicate_data=True)` builds --
    # is the same table twice as long, and is admitted.
    op.is_valid(
        **_guard_args(
            rotary_cos_sin=torch.zeros(
                MAX_SEQ_LEN * 2 * QK_ROPE_HEAD_DIM, dtype=torch.float32, device="cuda"
            ),
            sequence_length=lengths,
        )
    )


# ─── the entry's own claims ───────────────────────────────────────────────

# The entry's cells are the op's whoever calls it, so this target's variant
# carries the cells it certifies rather than a per-target overlay.
CELLS = op.CELLS


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    """Drive every configuration the entry claims, through the entry's own
    reference and its own gate.

    The cases above test the op; this one tests the *claim*. A cell list that
    nothing executes reads as coverage and is not, which is why a separate
    check refuses an entry whose cells no test names.

    Each cell cross-checks `op.reference` against this file's mirror before
    either is used to gate the kernel: two references disagreeing is a
    different failure from the kernel moving, and collapsing them into one
    assert loses which happened.
    """
    spec = cell.spec
    p = spec["predicted_tokens_per_seq"]
    # The keys only the drafter's cells carry: the pool row the call names (a
    # sparse layer's decoder index), how many context rows lead the presented
    # batch, and the pool's served length -- which is also the KV length the
    # longest sequence in the batch ends its block at.
    layer_idx = spec.get("layer_idx", 0)
    num_contexts = spec.get("num_contexts", 0)
    kv_len = spec.get("kv_len")
    num_layers = spec.get("num_layers", 1)
    max_seq_len = MAX_SEQ_LEN if kv_len is None else kv_len
    long_kv = LONG_KV if kv_len is None else kv_len - p

    # Every cell carries the checkpoint's geometry, so a cell that drifts from
    # it fails here rather than certifying a shape the model never runs.
    assert spec["qk_nope_head_dim"] == QK_NOPE_HEAD_DIM
    assert spec["qk_rope_head_dim"] == QK_ROPE_HEAD_DIM
    assert spec["kv_lora_rank"] == KV_LORA_RANK
    assert spec["q_lora_rank"] == Q_LORA_RANK
    assert spec["v_head_dim"] == V_HEAD_DIM
    assert spec["q_scaling"] == Q_SCALING
    # A sparse layer has no window of its own: the model passes the pool's
    # served length instead (`MlaLayer._window` -> `_pool_max`). A cell naming
    # 5134 is therefore the model's own call with its own number, and it has to
    # name the pool that length belongs to.
    assert spec["attention_window_size"] in (WINDOW, DRAFT_POOL_LEN)
    if spec["attention_window_size"] == DRAFT_POOL_LEN:
        assert kv_len == DRAFT_POOL_LEN, (kv_len, spec["attention_window_size"])
    # A row above 0 is the multi-layer pool the drafter writes into: one row per
    # text layer, at the first sparse layer it reuses. No other shape of this
    # target's produces one, so a cell that names a row has to name the pool.
    assert num_layers in (1, NUM_TEXT_LAYERS)
    assert layer_idx == (DRAFT_TARGET_LAYER if num_layers > 1 else 0)
    assert tuple(spec["mrope_section"]) == MROPE_SECTION
    assert tuple(spec["mrope_offsets"]) == MROPE_OFFSETS
    assert spec["rope_theta"] in (THETA_SWA, THETA_SPARSE)
    assert spec["quant_mode"] == 0, (
        "this target's latent pool is bf16; the quantized pool paths are not certified here"
    )
    assert spec["num_heads"] in (
        NUM_HEADS_SWA,
        NUM_HEADS_SPARSE,
        NUM_HEADS_TP4_SWA,
        NUM_HEADS_TP4_SPARSE,
        # The ctx engine's attn_tp-2 shards are numerically the same two values
        # as other calls (32 == the sparse full count, 16 == the SWA tp4 shard);
        # the cell's base is what tells them apart.
        NUM_HEADS_CTX_SWA,
        NUM_HEADS_CTX_SPARSE,
    )
    assert isinstance(op, MlaRopeGeneration)

    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    env = _MlaEnv(
        # The context rows are requests of the same batch, so the batch the
        # metadata is sized for is generation sequences plus context rows.
        max_batch_size=spec["num_seqs"] + num_contexts,
        tokens_per_block=spec["tokens_per_block"],
        num_heads=spec["num_heads"],
        rope_theta=spec["rope_theta"],
        quant_mode=spec["quant_mode"],
        mrope_section=spec["mrope_section"],
        mrope_offsets=spec["mrope_offsets"],
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        # A pool of its own is a cell's to size; the single-row pool the other
        # cells use takes the harness default.
        max_tokens=spec.get("max_tokens", 131072),
    )
    try:
        request_ids = list(range(spec["num_seqs"] + num_contexts))
        cached = _cached_lens(
            spec["tokens_per_block"],
            p,
            spec["num_seqs"],
            long_kv=long_kv,
            max_seq_len=max_seq_len,
        )
        # A context row is a fresh request presented with its whole prompt as its
        # query length, so what the manager holds for it is the prompt while the
        # metadata sees no history; a generation row's history is what the
        # harness grows by the block's own rows.
        cached_lens = [0] * num_contexts + cached
        token_nums = [CTX_PROMPT] * num_contexts + cached
        env.kv_cache_manager.add_dummy_requests(request_ids, token_nums=token_nums)
        step = _run_and_check(
            env,
            request_ids,
            cached_lens,
            p,
            repeats=2,
            q_scaling=spec["q_scaling"],
            q_lora_rank=spec["q_lora_rank"],
            window=spec["attention_window_size"],
            num_contexts=num_contexts,
            context_lens=[CTX_PROMPT] * num_contexts if num_contexts else None,
            layer_idx=layer_idx,
        )

        table = env.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        half = QK_ROPE_HEAD_DIM // 2

        # 1. The entry's reference against this file's, on the appended row.
        for r in range(step.rows):
            pos = step.positions[r]
            entry_ref = op.reference(
                step.latent_orig[r],
                table[pos, :half, 0],
                table[pos, :half, 1],
                QK_ROPE_HEAD_DIM,
                None,
                torch.bfloat16,
            )
            local_ref = torch.cat(
                [
                    step.latent_orig[r, :KV_LORA_RANK],
                    env.rope_ref(step.latent_orig[r, KV_LORA_RANK:], pos),
                ]
            )
            _assert_bytes_equal(entry_ref, local_ref, f"references disagree, row {r}")

        # 2. The entry's reference against the kernel, through the same gate.
        for r in range(step.rows):
            pos = step.positions[r]
            row = torch.cat([step.fused_q_orig[r, :, :KV_LORA_RANK], step.q_pe_orig[r]], dim=-1)
            _assert_bf16_roped(
                step.fused_q[r],
                op.reference(
                    row,
                    table[pos, :half, 0],
                    table[pos, :half, 1],
                    QK_ROPE_HEAD_DIM,
                    None,
                    torch.bfloat16,
                ),
                f"entry reference vs kernel, row {r}",
            )

        # 3. The scalar outputs the entry states, and the scheduler buffers it
        #    fills. The bmm scales belong to the quantized paths, so no buffer
        #    here holds them; the reference's own values are still checked.
        want1, want2 = op.reference_bmm_scales(
            None, spec["q_scaling"], spec["qk_nope_head_dim"], spec["qk_rope_head_dim"]
        )
        x = 1.0 / (
            spec["q_scaling"] * math.sqrt(spec["qk_nope_head_dim"] + spec["qk_rope_head_dim"])
        )
        assert want1[0].item() == pytest.approx(x, rel=1e-6)
        assert want1[1].item() == pytest.approx(x * math.log2(math.e), rel=1e-6)
        assert want2.item() == 1.0
        # The scheduler's buffers span the generation rows: a context row is a
        # row the op resolves the generation tail past, not a row it schedules.
        want_cu_q, want_cu_kv = op.reference_scheduler_buffers(
            step.gen_kv_lens, spec["num_heads"], p
        )
        assert want_cu_q.tolist() == step.cu_q_seqlens.tolist()
        assert want_cu_kv.tolist() == step.cu_kv_seqlens.tolist()
        print(
            f"{cell.why}: {step.roped_inexact}/{step.roped_total} roped elements "
            f"missed bit-exactness; worst deviation {step.roped_worst:.3g} of the scale"
        )
    finally:
        env.shutdown()
