# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Hy4-Flash target's mla_rope_append_paged_kv_assign_q entry.

The op reads paged-KV-cache addressing tensors and cumulative-length tensors
that the runtime normally derives from a KVCacheManagerV2 and a
TrtllmAttentionMetadata prepared with enable_context_mla_with_cached_kv=True.
The test builds that state for real, pre-writes known cached-prefix rows into
the paged pool, then checks the three kernel effects against the entry's own
`reference` (and, for the RoPE itself, against a transcription of the HF
`apply_rotary_pos_emb`): GPT-J RoPE of each new context token's q_pe written
back into q at the token's absolute position, the same RoPE of k_pe written
back into latent_cache, and `[compressed_kv | rope(k_pe)]` appended into the
paged latent cache at that position. Cached-prefix rows, q's nope slices and
latent_cache's compressed_kv slice must be bitwise untouched.

Hy4-Flash's geometry: q heads are `[nope 192 | rope 64]`, latent rows
`[compressed_kv 512 | k_pe 64]`, page 32. The sparse layers run 32 heads (8 per
rank at tp4), the sliding-window layers 64 (16 at tp4); the disaggregated
deployment's context engine runs attn_tp 2, 16 sparse / 32 SWA per rank. Two pool surfaces:
bf16 (`quant_mode=0`, the core KV cache) and fp8-e4m3 (`quant_mode=128`, the
kv_fp8 variant), both at bf16 activations.

The rope surface is Hy4's interleaved mRoPE: three position planes whose angles
are interleaved at stride 3 over the 32 pair slots (sections [12, 10, 10]),
duplicated into the 64-pair block the kernel reads, at base 1e7 on the sparse
layers and 1e4 on the sliding-window ones. Text-only positions coincide across
the planes, so every cell runs the plain duplicated table; one test drives a
table whose planes differ and checks the kernel against the HF formula, with
the engine's non-interleaved layout and swapped planes as measured controls.
"""

import zlib
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.attention.mla_rope_append_paged_kv_assign_q__r14 import (  # noqa: E501
    mla_rope_append_paged_kv_assign_q as op,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mla_rope_append_paged_kv_assign_q requires a CUDA device"

# Hy4-Flash MLA geometry. The sparse layers run 32 heads (8 per rank at tp4),
# the sliding-window layers 64 (16 at tp4).
NUM_HEADS_SPARSE = 32
NUM_HEADS_SWA = 64
NUM_HEADS_SPARSE_TP4 = 8
NUM_HEADS_SWA_TP4 = 16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
LATENT_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
TOKENS_PER_BLOCK = 32
# The engine's context-call envelope: KvCacheConfig.max_num_tokens defaults to
# 8192, which is the largest row count one context call carries. It also
# covers the 5120-token prefill chunk in the entry's cells.
MAX_SEQ_LEN = 8192
# Hy4's two rope bases, per layer type, and the interleaved mRoPE sections.
ROPE_THETA_SPARSE = 10_000_000.0
ROPE_THETA_SWA = 10_000.0
MROPE_SECTION = (12, 10, 10)
QUANT_MODE_FP8_KV_CACHE = 128  # QuantMode.FP8_KV_CACHE
QUANT_MODE_INT8_KV_CACHE = 64  # QuantMode.INT8_KV_CACHE — rejected, see below

# One e4m3 ulp. e4m3 keeps 3 mantissa bits, so 2**-3 relative is one ulp — the
# gate for the appended row's k_pe half, the only quantized output that passes
# through the in-kernel RoPE, where the kernel and a torch fp32 reference can
# evaluate x*cos -+ y*sin in different orders. Paired with a bit-exact-fraction
# floor because the coarse e4m3 rounding absorbs that difference in practice.
# That both halves still discriminate is measured, not assumed:
# test_fp8_kv_wrong_mirror_controls runs four wrong mirrors of the same bytes
# and asserts each blows the half it is able to blow.
E4M3_ULP_RTOL = 2**-3
E4M3_MAX_INEXACT_FRACTION = 1e-3

# The roped q must come back as an ordinary bf16 rotation, NOT quantized: this
# op leaves q to the context FMHA, which quantizes q itself. A bf16 value drawn
# from a continuous distribution survives an e4m3 round trip only when its
# mantissa already fits in 3 bits, which happens for ~6% of elements; a
# quantized q would sit at 100%. The gate is 25% — 4x the measured 6.2%, and 4x
# below a quantized tensor — and every fp8 case also compares its own untouched
# nope slice as an in-run baseline.
MAX_E4M3_ROUNDTRIP_FRACTION = 0.25


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Half an e4m3 ulp, *and* a floor on how much may miss bit-exactness.

    The rotation is where kernel and reference can disagree: both evaluate
    `x*cos -+ y*sin`, and they may do it in different orders. e4m3's coarse
    rounding absorbs that in practice, so neither half of this gate is expected
    to be approached.

    Both halves are kept because they fail differently: a tolerance alone
    passes a mirror that is uniformly a little wrong, and a bit-exact demand
    alone fails the first time a reordering does survive the rounding.
    """
    assert out.shape == ref.shape, (out.shape, ref.shape)
    torch.testing.assert_close(out.float(), ref.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
    inexact = int((out.reshape(-1).view(torch.uint8) != ref.reshape(-1).view(torch.uint8)).sum())
    allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * out.numel()))
    assert inexact <= allowed, (
        f"not bit-exact enough: {inexact}/{out.numel()} bytes differ (allowed {allowed})"
    )


class _MlaCtxEnv:
    """Real op state: MLA (kv_factor=1) paged KV cache manager + the duplicated
    RoPE table + context-MLA-with-cached-KV metadata.

    With fp8_pool the manager allocates an e4m3 latent pool (one byte per
    element) while the activations (`q`, `latent_cache`) stay bf16, and
    `orig_quant` becomes the op's write-side KV scaling factor. Without it the
    pool holds bf16 rows, which is the target's core KV cache.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        fp8_pool: bool = False,
        orig_quant: Optional[float] = None,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        theta: float = ROPE_THETA_SWA,
        max_seq_len: int = MAX_SEQ_LEN,
        cos_sin: Optional[torch.Tensor] = None,
        max_batch_size: int = 8,
        nope_size: int = QK_NOPE_HEAD_DIM,
        rope_size: int = QK_ROPE_HEAD_DIM,
        lora_size: int = KV_LORA_RANK,
    ) -> None:
        self.torch_dtype = torch.bfloat16
        self.num_heads = num_heads
        self.nope_size = nope_size
        self.rope_size = rope_size
        self.lora_size = lora_size
        # Per-head q width and latent row width.
        self.q_head_size = nope_size + rope_size
        self.latent_size = lora_size + rope_size
        self.max_batch_size = max_batch_size
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.fp8_pool = fp8_pool
        self.pool_dtype = torch.float8_e4m3fn if fp8_pool else self.torch_dtype
        self.quant_mode = QUANT_MODE_FP8_KV_CACHE if fp8_pool else 0
        self.kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=131072, enable_block_reuse=False),
            CacheType.SELFKONLY,  # MLA latent cache: kv_factor=1, one kv head
            num_layers=1,
            num_kv_heads=1,
            head_dim=self.latent_size,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8 if fp8_pool else DataType.BF16,
        )
        # The pool must really be paged at the size and element type the case
        # claims: the op sizes its page slabs from tokens_per_block and
        # quant_mode, not from anything the manager tells it.
        assert self.kv_cache_manager.tokens_per_block == tokens_per_block
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None and pool.dtype == self.pool_dtype
        # KV scaling factor. None is the op's "no scaling" input; a real tensor
        # is what the engine passes on the fp8 path.
        self.kv_scale_orig_quant = (
            None
            if orig_quant is None
            else torch.full((1,), orig_quant, dtype=torch.float32, device="cuda")
        )
        # The exact fp32 number the op sees, as a 0-dim tensor, so the mirror
        # cannot disagree with the kernel in the last bit.
        self.write_scale = torch.tensor(
            1.0 if orig_quant is None else orig_quant, dtype=torch.float32
        ).cuda()
        # Duplicated-layout fp32 (cos, sin) table, as the MLA backend builds it
        # (RopeParams.from_config sets duplicate_data=True for MLA models). A
        # private copy, because the constructor caches the tensor by params and
        # a test that rewrites its table must not reach another env.
        if cos_sin is None:
            rope = RopeParams(
                dim=rope_size,
                theta=theta,
                max_positions=max_seq_len,
                duplicate_data=True,
            )
            _, cos_sin = rope.create_rope_const_params()
        self.rotary_cos_sin = cos_sin.clone()

    def prepare_metadata(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        num_contexts: int,
        cached_lens: List[int],
    ) -> TrtllmAttentionMetadata:
        metadata = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=MAX_SEQ_LEN,
            kv_cache_manager=self.kv_cache_manager,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
        metadata.num_contexts = num_contexts
        metadata.request_ids = request_ids
        metadata.prompt_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_lens,
        )
        metadata.prepare()
        return metadata

    def rope_fp32(self, x: torch.Tensor, positions: Union[int, Sequence[int]]) -> torch.Tensor:
        """GPT-J interleaved rotation of the last dim in fp32, unrounded.
        x is [n, ..., R] with positions of length n (or a single position for
        x of shape [..., R])."""
        half = self.rope_size // 2
        table = self.rotary_cos_sin.view(-1, self.rope_size, 2)
        if isinstance(positions, int):
            cos = table[positions, :half, 0]
            sin = table[positions, :half, 1]
        else:
            pos = torch.tensor(list(positions), dtype=torch.long, device="cuda")
            # [n, half] broadcast over x's middle dims.
            shape = [len(positions)] + [1] * (x.dim() - 2) + [half]
            cos = table[pos, :half, 0].view(shape)
            sin = table[pos, :half, 1].view(shape)
        pairs = x.float().reshape(*x.shape[:-1], half, 2)
        out = torch.empty_like(pairs)
        out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        return out.reshape(x.shape)

    def rope_ref(self, x: torch.Tensor, positions: Union[int, Sequence[int]]) -> torch.Tensor:
        """rope_fp32 with one rounding back to x's dtype — what the kernel
        writes into q and latent_cache."""
        return self.rope_fp32(x, positions).to(x.dtype)

    def hf_apply_rotary(self, x: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        """HYV4Flash's `apply_rotary_pos_emb` on this env's table, in fp32.

        The ground truth for the interleaved mRoPE: `cos`/`sin` are the table's
        64-slot vectors for each position, halved, repeat_interleaved, and
        paired with `rotate_half`'s consecutive-element rotation.
        """
        return _hf_apply_rotary(self.rotary_cos_sin, x, positions, self.rope_size)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """e4m3(x * kv_scale_orig_quant) — the op's write-side quantization of
        the appended latent row."""
        return (x.float() * self.write_scale).to(torch.float8_e4m3fn)

    def pool_tensor(self, layer_idx: int) -> torch.Tensor:
        pool = self.kv_cache_manager.get_buffers(layer_idx)
        assert pool is not None
        return pool

    def blocks(self, request_id: int) -> List[int]:
        return list(self.kv_cache_manager.get_batch_cache_indices([request_id])[0])

    def slot(self, request_id: int, position: int) -> Tuple[int, int]:
        tpb = self.tokens_per_block
        return self.blocks(request_id)[position // tpb], position % tpb

    def fill_rows(self, layer_idx: int, request_id: int, length: int) -> torch.Tensor:
        """Write known random latent rows at positions [0, length) of one
        sequence's pages; return the written rows [length, C+R]."""
        draw_dtype = torch.float32 if self.fp8_pool else self.torch_dtype
        rows = torch.randn(length, self.latent_size, dtype=draw_dtype, device="cuda").to(
            self.pool_dtype
        )
        pool = self.pool_tensor(layer_idx)  # [pages,1,tpb,1,head]
        tpb = self.tokens_per_block
        blocks = self.blocks(request_id)
        for start in range(0, length, tpb):
            page = blocks[start // tpb]
            n = min(tpb, length - start)
            pool[page, 0, :n, 0].copy_(rows[start : start + n])
        return rows

    def read_rows(self, layer_idx: int, request_id: int, start: int, end: int) -> torch.Tensor:
        """Read latent rows at positions [start, end) of one sequence back
        from the paged pool as a dense [end-start, C+R] tensor."""
        pool = self.pool_tensor(layer_idx)
        tpb = self.tokens_per_block
        blocks = self.blocks(request_id)
        out = torch.empty(end - start, self.latent_size, dtype=self.pool_dtype, device="cuda")
        # Page by page rather than row by row: the 8192-token cell spans 256
        # pages and the 5120-token one 160.
        pos = start
        while pos < end:
            page, offset = blocks[pos // tpb], pos % tpb
            n = min(tpb - offset, end - pos)
            out[pos - start : pos - start + n] = pool[page, 0, offset : offset + n, 0]
            pos += n
        return out

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


def _hf_apply_rotary(
    table: torch.Tensor, x: torch.Tensor, positions: Sequence[int], rope_size: int
) -> torch.Tensor:
    """HYV4Flash's `apply_rotary_pos_emb` on `table`, in fp32 (unrounded).

    `table` is the op's duplicated `(cos, sin)` pair layout, so its 64 slots per
    position are `cat((freqs, freqs))`; HF halves that to 32 and interleaves
    each value twice before pairing consecutive elements. This is a different
    code path from the pair arithmetic in `rope_fp32`, which is the point: the
    interleaved mRoPE is certified against HF's formula, not against a second
    transcription of the same pair indexing.
    """
    pairs = table.view(-1, rope_size, 2)
    pos = torch.tensor(list(positions), dtype=torch.long, device="cuda")
    shape = [len(positions)] + [1] * (x.dim() - 2) + [rope_size]
    cos = pairs[pos, :, 0].view(shape)
    sin = pairs[pos, :, 1].view(shape)
    cos = cos[..., : rope_size // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : rope_size // 2].repeat_interleave(2, dim=-1)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x.float() * cos + rotated * sin


def _mrope_table(
    inv_freq: torch.Tensor,
    plane_positions: torch.Tensor,
    sections: Tuple[int, ...] = MROPE_SECTION,
    rope_size: int = QK_ROPE_HEAD_DIM,
) -> torch.Tensor:
    """The interleaved three-plane mRoPE table in the op's duplicated layout.

    `HYV4FlashRotaryEmbedding.forward` expands the position ids to three
    (time, height, width) planes; `recomposition_frequencies` starts from the
    time plane and overwrites the stride-3 slots its section sizes select with
    the height and width planes' angles, then duplicates the block. Each slot
    is stored as a (cos, sin) float2 pair — the layout
    `RopeParams(..., duplicate_data=True)` builds.

    `plane_positions` is [rows, 3]: the (t, h, w) position of the token whose
    absolute position indexes that table row.
    """
    angles = plane_positions.to(torch.float32).unsqueeze(-1) * inv_freq  # [P, 3, half]
    freq = angles[:, 0].clone()
    for plane, offset in enumerate((1, 2), start=1):
        length = sections[plane] * 3
        idx = slice(offset, length, 3)
        freq[:, idx] = angles[:, plane, idx]
    pairs = torch.stack((freq.cos(), freq.sin()), dim=-1)  # [P, half, 2]
    return torch.cat((pairs, pairs), dim=1).reshape(-1).contiguous()


def _mrope_table_concatenated(
    inv_freq: torch.Tensor,
    plane_positions: torch.Tensor,
    sections: Tuple[int, ...] = MROPE_SECTION,
    rope_size: int = QK_ROPE_HEAD_DIM,
) -> torch.Tensor:
    """The engine's `mrope_interleaved=False` layout: whole section blocks taken
    from each plane and concatenated (`MRotaryEmbedding.get_cos_sin`'s else
    branch). A plausible mis-derivation of the same angles, used as a control."""
    angles = plane_positions.to(torch.float32).unsqueeze(-1) * inv_freq
    freq = angles[:, 0].clone()
    offset = 0
    for plane, size in enumerate(sections):
        freq[:, offset : offset + size] = angles[:, plane, offset : offset + size]
        offset += size
    pairs = torch.stack((freq.cos(), freq.sin()), dim=-1)
    return torch.cat((pairs, pairs), dim=1).reshape(-1).contiguous()


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Bit-exact comparison of two tensors, signed zeros included."""
    torch.testing.assert_close(
        got.reshape(-1).view(torch.uint8),
        want.reshape(-1).view(torch.uint8),
        rtol=0.0,
        atol=0.0,
        msg=lambda m: f"{what} not bit-exact\n{m}",
    )


def _inexact_bytes(got: torch.Tensor, want: torch.Tensor) -> int:
    return int((got.reshape(-1).view(torch.uint8) != want.reshape(-1).view(torch.uint8)).sum())


def _fails_e4m3_tolerance(got: torch.Tensor, mirror: torch.Tensor) -> bool:
    """Whether the tolerance half of the e4m3 gate rejects this mirror."""
    try:
        torch.testing.assert_close(got.float(), mirror.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
    except AssertionError:
        return True
    return False


def _e4m3_roundtrip_fraction(x: torch.Tensor) -> float:
    """Fraction of elements of a bf16 tensor that survive an e4m3 round trip
    unchanged — ~6% for continuous data, 100% for a quantized tensor."""
    rt = x.float().to(torch.float8_e4m3fn).to(x.dtype)
    return float((x.reshape(-1) == rt.reshape(-1)).float().mean())


def _outside_band_fraction(got: torch.Tensor, mirror: torch.Tensor) -> float:
    """Fraction of elements differing by more than four bf16 ulps of their
    row's largest magnitude (2**-7 relative, a bf16 ulp being 2**-8) — the
    separation a control must show."""
    diff = (got.float() - mirror.float()).abs()
    scale = mirror.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    return float((diff / scale > 2.0**-7).float().mean())


def _band(got: torch.Tensor, ref: torch.Tensor) -> str:
    """Worst deviation from `ref` in units of the row's largest magnitude, and
    the fraction of bytes that are bit-exact -- the two units the assertions on
    these tensors bound.

    The row-scale unit is what `_validating`'s element gate uses and what
    `_outside_band_fraction` measures controls in; the bit-exact fraction is
    what the e4m3 floor and the pool-byte checks use. Reported, not asserted on.
    """
    scale = ref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    worst = float(((got.float() - ref.float()).abs() / scale).max())
    exact = 1.0 - _inexact_bytes(got, ref) / got.numel()
    return f"worst {worst:.3e} of row scale, {exact:.5f} bit-exact"


class _Run(NamedTuple):
    """What one op call was driven with and produced."""

    q: torch.Tensor  # post-call q, [T, H*(N+R)]
    q_orig: torch.Tensor  # pre-call q
    latent_cache: torch.Tensor  # post-call latent_cache, [T, C+R]
    latent_orig: torch.Tensor  # pre-call latent_cache
    ref_k_pe: torch.Tensor  # torch reference for rope(k_pe), activation dtype
    ref_k_pe_fp32: torch.Tensor  # the same reference before rounding
    rows: torch.Tensor  # appended pool rows in row order, [T, C+R]
    positions: List[int]  # absolute position of every row
    measured: Dict[str, str]  # this run's bands, by comparison, for the report


def _call(
    env: _MlaCtxEnv,
    metadata: TrtllmAttentionMetadata,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    num_contexts: int,
    residual_dim: int = 0,
    quant_mode: Optional[int] = None,
) -> None:
    """One op call at env's geometry, in the engine's argument order."""
    block_offsets = metadata.kv_cache_block_offsets
    pool_pointers = env.kv_cache_manager.kv_cache_pool_pointers
    pool_mapping = env.kv_cache_manager.kv_cache_pool_mapping
    assert block_offsets is not None
    assert pool_pointers is not None and pool_mapping is not None
    op(
        q,
        latent_cache,
        num_contexts,
        metadata.ctx_cached_token_indptr,
        metadata.ctx_kv_indptr,
        int(metadata.max_ctx_seq_len),
        env.rotary_cos_sin,
        env.num_heads,
        env.nope_size,
        env.rope_size,
        env.lora_size,
        block_offsets,
        pool_pointers,
        pool_mapping,
        env.kv_scale_orig_quant,  # None is the no-scale call
        residual_dim,  # 0 or rope_size; non-zero needs an FP4 KV pool
        0,  # layer_idx
        env.tokens_per_block,
        env.max_seq_len,  # attention_window_size
        1,  # beam_width
        env.quant_mode if quant_mode is None else quant_mode,
    )
    torch.cuda.synchronize()


def _run_and_check(
    env: _MlaCtxEnv,
    request_ids: List[int],
    seq_lens: List[int],
    num_contexts: int,
    cached_lens: List[int],
    layer_idx: int = 0,
) -> _Run:
    """Pre-fill every sequence's cached prefix, run the op over the context
    sequences' new tokens, and verify every kernel effect."""
    kv_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
    metadata = env.prepare_metadata(request_ids, seq_lens, num_contexts, cached_lens)
    # Known cached-prefix rows (all sequences) to check the op leaves them be.
    prefix_rows = [env.fill_rows(layer_idx, rid, c) for rid, c in zip(request_ids, cached_lens)]

    num_heads = env.num_heads
    nope, lora = env.nope_size, env.lora_size
    ctx_new = seq_lens[:num_contexts]
    num_tokens = sum(ctx_new)
    assert int(metadata.num_ctx_tokens) == num_tokens
    assert int(metadata.max_ctx_seq_len) == max(ctx_new)
    # Absolute position of every new context token, in q/latent row order.
    positions = [cached_lens[s] + i for s in range(num_contexts) for i in range(ctx_new[s])]

    q = torch.randn(num_tokens, num_heads * env.q_head_size, dtype=env.torch_dtype, device="cuda")
    # The op rejects other ranks (thop checks latent_cache.dim() == 2).
    latent_cache = torch.randn(num_tokens, env.latent_size, dtype=env.torch_dtype, device="cuda")
    q_orig = q.clone()
    latent_orig = latent_cache.clone()
    pool = env.pool_tensor(layer_idx)
    pool_before = pool.clone()

    _call(env, metadata, q, latent_cache, num_contexts)

    # 1. q: per head, the rope tail is rotated at the token's absolute
    # position; the nope slice is bitwise untouched. This holds on the fp8 path
    # too — q stays an unquantized activation tensor there.
    q_heads = q.view(num_tokens, num_heads, env.q_head_size)
    q_orig_heads = q_orig.view(num_tokens, num_heads, env.q_head_size)
    ref_q_pe = env.rope_ref(q_orig_heads[..., nope:], positions)
    torch.testing.assert_close(q_heads[..., nope:], ref_q_pe)
    torch.testing.assert_close(
        q_heads[..., :nope],
        q_orig_heads[..., :nope],
        rtol=0.0,
        atol=0.0,  # caller-owned region: must be bitwise untouched
    )

    # 2. latent_cache: k_pe rotated in place; compressed_kv bitwise untouched.
    ref_k_pe_fp32 = env.rope_fp32(latent_orig[:, lora:], positions)
    ref_k_pe = ref_k_pe_fp32.to(env.torch_dtype)
    torch.testing.assert_close(latent_cache[:, lora:], ref_k_pe)
    torch.testing.assert_close(
        latent_cache[:, :lora],
        latent_orig[:, :lora],
        rtol=0.0,
        atol=0.0,  # read-only region: must be bitwise untouched
    )

    if env.fp8_pool:
        # Neither in-place rotation is quantized, so this op followed by the
        # context FMHA's own quantization is not a double quantization.
        # Compared against the untouched nope slice of the same tensor, which
        # is the in-run baseline for un-quantized bf16 data.
        for name, values in (
            ("roped q", q_heads[..., nope:]),
            ("roped k_pe", latent_cache[:, lora:]),
        ):
            frac = _e4m3_roundtrip_fraction(values)
            assert frac < MAX_E4M3_ROUNDTRIP_FRACTION, (
                f"{name} looks e4m3-quantized: {frac:.4f} of elements survive "
                "an e4m3 round trip unchanged"
            )
        baseline = _e4m3_roundtrip_fraction(q_orig_heads[..., :nope])
        assert baseline < MAX_E4M3_ROUNDTRIP_FRACTION, (
            f"un-quantized baseline is already {baseline:.4f} — the "
            "round-trip check cannot discriminate at this input distribution"
        )

    # 3. Paged cache: rows [cached_s, kv_s) hold the entry's own reference for
    # [compressed_kv | rope(k_pe)] at that row's absolute position; the
    # pre-filled cached prefix [0, cached_s) is bitwise untouched.
    table = env.rotary_cos_sin.view(-1, env.rope_size, 2)
    pos = torch.tensor(positions, dtype=torch.long, device="cuda")
    cos_rows = table[pos, : env.rope_size // 2, 0]
    sin_rows = table[pos, : env.rope_size // 2, 1]
    appended = []
    wants = []
    row_start = 0
    for s in range(num_contexts):
        rid = request_ids[s]
        rows = env.read_rows(layer_idx, rid, cached_lens[s], kv_lens[s])
        appended.append(rows)
        row_end = row_start + ctx_new[s]
        want = op.reference(
            latent_orig[row_start:row_end],
            cos_rows[row_start:row_end],
            sin_rows[row_start:row_end],
            env.rope_size,
            env.kv_scale_orig_quant,
            env.pool_dtype,
        )
        if env.fp8_pool:
            _assert_bytes_equal(
                rows[:, :lora], want[:, :lora], f"appended compressed_kv (request {rid})"
            )
            _compare(rows[:, lora:], want[:, lora:])
        else:
            torch.testing.assert_close(
                rows[:, :lora],
                want[:, :lora],
                rtol=0.0,
                atol=0.0,  # dtype-preserving copy: must be bitwise equal
            )
            torch.testing.assert_close(rows[:, lora:], want[:, lora:])
        wants.append(want)
        row_start = row_end
    for s, rid in enumerate(request_ids):
        if cached_lens[s] == 0:
            continue
        kept = env.read_rows(layer_idx, rid, 0, cached_lens[s])
        # Cached prefix (and every generation sequence's rows): bitwise kept.
        _assert_bytes_equal(kept, prefix_rows[s], f"cached prefix (request {rid})")

    # 4. Nothing else in the pool moved: exactly one (C+R)-element row per new
    # context token, at the slot its position addresses. This is also what pins
    # the page-slab geometry, whose element width the op derives from quant_mode
    # alone (1 byte under quant_mode=128).
    changed = pool.reshape(-1).view(torch.uint8) != pool_before.reshape(-1).view(torch.uint8)
    row_bytes = pool.element_size() * env.latent_size
    idx = changed.nonzero().reshape(-1) // row_bytes
    got = set(
        zip(
            (idx // env.tokens_per_block).tolist(),
            (idx % env.tokens_per_block).tolist(),
        )
    )
    expected = {
        env.slot(request_ids[s], cached_lens[s] + i)
        for s in range(num_contexts)
        for i in range(ctx_new[s])
    }
    assert got <= expected, (
        f"op touched pool slots {sorted(got - expected)} outside the "
        f"{len(expected)} (page, slot) pairs its inputs address"
    )
    # Upper bound, not equality: a written byte that happens to match the byte
    # already there is invisible to a snapshot diff. The rows' contents are
    # pinned above.
    assert idx.numel() <= num_tokens * row_bytes, (
        f"op wrote {idx.numel()} pool bytes, at most {num_tokens * row_bytes} rows' worth expected"
    )

    rows_got = torch.cat(appended, dim=0)
    rows_ref = torch.cat(wants, dim=0)
    measured = {
        "q_rope": _band(q_heads[..., nope:], ref_q_pe),
        "k_pe_rope": _band(latent_cache[:, lora:], ref_k_pe),
        "pool_kv": _band(rows_got[:, :lora], rows_ref[:, :lora]),
        "pool_rope": _band(rows_got[:, lora:], rows_ref[:, lora:]),
    }

    return _Run(
        q=q,
        q_orig=q_orig,
        latent_cache=latent_cache,
        latent_orig=latent_orig,
        ref_k_pe=ref_k_pe,
        ref_k_pe_fp32=ref_k_pe_fp32,
        rows=rows_got,
        positions=positions,
        measured=measured,
    )


def _drive(
    env: _MlaCtxEnv, cached: List[int], new: List[int], num_contexts: Optional[int] = None
) -> _Run:
    """Allocate one request per sequence and run `_run_and_check` over them."""
    rids = list(range(len(new)))
    env.kv_cache_manager.add_dummy_requests(rids, token_nums=[c + n for c, n in zip(cached, new)])
    return _run_and_check(
        env,
        request_ids=rids,
        seq_lens=new,
        num_contexts=len(rids) if num_contexts is None else num_contexts,
        cached_lens=cached,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    """Every configuration the entry claims, through the same driver the
    hand-written cases use.

    Parametrized off the entry, so a cell added there is a case here. Without
    this the cell list would be a claim nothing executes, which is worse than
    no claim at all -- it reads as coverage.
    """
    spec = cell.spec
    # The driver builds bf16 and fp8-e4m3 pools only; a cell asking for another
    # layout would be certified against the wrong pool.
    assert spec["residual_dim"] == 0, spec
    assert spec["quant_mode"] in (0, QUANT_MODE_FP8_KV_CACHE), spec
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    fp8_pool = spec["quant_mode"] == QUANT_MODE_FP8_KV_CACHE
    env = _MlaCtxEnv(
        spec["head_num"],
        fp8_pool=fp8_pool,
        # The fp8 cell is the kv_fp8 variant's context call, which passes a
        # real per-tensor write scale; the bf16 cells pass none.
        orig_quant=1.0 if fp8_pool else None,
        tokens_per_block=spec["tokens_per_block"],
        nope_size=spec["nope_size"],
        rope_size=spec["rope_size"],
        lora_size=spec["lora_size"],
    )
    certifying(op)  # drives is_valid and requires the entry's reference to run
    try:
        run = _drive(env, spec["cached"], spec["new"])
    finally:
        env.shutdown()
    print(
        f"MEASURED {cell.why[:44]}: {spec['head_num']} heads, "
        f"{sum(spec['new'])} new over a {'fp8' if fp8_pool else 'bf16'} pool, "
        f"page {spec['tokens_per_block']} | "
        + "; ".join(f"{name} {band}" for name, band in run.measured.items())
    )


def test_bf16_pool_mixed_cached_lengths() -> None:
    """The core (bf16) context call at Hy4's sliding-window head count, H = 64
    and page size 32. Four context sequences append T = 402 rows in one call,
    with cached prefixes placing the first new token at every alignment a
    32-slot page has -- 0 (fresh prefill), 32 (page 1 slot 0), 100 (page 3 slot
    4) and 511 (page 15's last slot) -- and the 300-token sequence walking ten
    pages."""
    torch.manual_seed(20)
    env = _MlaCtxEnv(NUM_HEADS_SWA)
    try:
        _drive(env, cached=[0, 32, 100, 511], new=[37, 64, 300, 1])
    finally:
        env.shutdown()


def test_fp8_kv_mixed_cached_lengths() -> None:
    """The kv_fp8 variant's context call: the same mixed batch over an e4m3
    pool with a real, non-power-of-two per-tensor write scale (1/1.5), so the
    write-side multiply is a real rescale rather than an exponent shift."""
    torch.manual_seed(21)
    env = _MlaCtxEnv(NUM_HEADS_SWA, fp8_pool=True, orig_quant=1.0 / 1.5)
    try:
        _drive(env, cached=[0, 32, 100, 511], new=[37, 64, 300, 1])
    finally:
        env.shutdown()


def test_fp8_kv_wrong_mirror_controls() -> None:
    """What the e4m3 gate on the appended k_pe half discriminates.

    Every appended byte above matched the entry's reference within the gate,
    which is only meaningful if the same comparison moves for a wrong mirror.
    Four plausible mis-derivations of the same bytes, each asserted to blow the
    half of the gate it is able to blow (the numbers are recorded from this
    test's run in the report)."""
    torch.manual_seed(22)
    env = _MlaCtxEnv(NUM_HEADS_SWA, fp8_pool=True, orig_quant=1.0)
    try:
        run = _drive(env, cached=[0, 31], new=[33, 40])
        lora = env.lora_size
        got = run.rows[:, lora:]
        allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * got.numel()))
        shifted = [p + 1 for p in run.positions]
        blunt = {
            "un-roped k_pe": env.quantize(run.latent_orig[:, lora:]),
            "rope at position + 1": env.quantize(env.rope_ref(run.latent_orig[:, lora:], shifted)),
            "the previous token's row": env.quantize(run.ref_k_pe.roll(1, dims=0)),
        }
        for name, mirror in blunt.items():
            inexact = _inexact_bytes(got, mirror)
            assert inexact > 100 * allowed, (
                f"control '{name}' moved only {inexact}/{got.numel()} bytes — "
                "the bit-exact-fraction floor would not have caught it"
            )
            assert _fails_e4m3_tolerance(got, mirror), (
                f"control '{name}' stayed inside the one-ulp tolerance — that "
                "half of the gate would not have caught it"
            )
        # The rounding-level control: quantizing the unrounded fp32 RoPE result
        # instead of the bf16 one the kernel rounds to first differs by a single
        # rounding step, so only the bit-exact-fraction floor catches it.
        no_bf16 = (run.ref_k_pe_fp32 * env.write_scale).to(torch.float8_e4m3fn)
        inexact = _inexact_bytes(got, no_bf16)
        assert inexact > 8 * allowed, (
            f"control 'no bf16 rounding before e4m3' moved only {inexact}/"
            f"{got.numel()} bytes — the bit-exact-fraction floor would not "
            "have caught it"
        )
    finally:
        env.shutdown()


def test_fp8_kv_trailing_generation_seqs_ignored() -> None:
    """fp8 mixed batch: two context sequences followed by two generation
    sequences, all with cached prefixes. Sequence 0's new tokens fill page 3
    exactly (96..127); sequence 1 starts on page 0's last slot and crosses on
    its very first token (31..63). The op must touch only the context rows,
    index per-seq tensors over [0, num_contexts), and leave both generation
    sequences' pages bitwise untouched."""
    torch.manual_seed(23)
    env = _MlaCtxEnv(NUM_HEADS_SWA, fp8_pool=True, orig_quant=1.0 / 1.5)
    try:
        _drive(env, cached=[96, 31, 200, 77], new=[32, 33, 1, 1], num_contexts=2)
    finally:
        env.shutdown()


@pytest.mark.parametrize(
    "num_heads,theta",
    [(NUM_HEADS_SPARSE, ROPE_THETA_SPARSE), (NUM_HEADS_SWA, ROPE_THETA_SWA)],
    ids=["sparse_h32_base1e7", "swa_h64_base1e4"],
)
def test_mrope_interleaved_table(num_heads, theta, certifying) -> None:  # noqa: F811
    """Hy4's interleaved three-plane mRoPE table, through the kernel.

    The table interleaves the height and width planes into the time plane at
    stride 3 over the section-sized spans ([12, 10, 10]) and duplicates the
    block, at the layer type's base. Its planes differ per row (as an image
    token's grid position does), so a kernel that assumed split-half pairing,
    read the second 32 pairs, or transposed the table would disagree with HF's
    `apply_rotary_pos_emb` on the same table. The controls measure that
    disagreement for the engine's non-interleaved layout and for swapped
    height/width planes.
    """
    torch.manual_seed(31)
    cached, new = [128], [192]
    positions_all = torch.arange(MAX_SEQ_LEN, device="cuda")
    # The height and width planes run ahead of time, as an image token's grid
    # position does; the offsets are large enough that the differing slots'
    # angles move measurably at both bases.
    plane_positions = torch.stack((positions_all, positions_all + 512, positions_all + 1024), dim=1)
    rope = RopeParams(
        dim=QK_ROPE_HEAD_DIM,
        theta=theta,
        max_positions=MAX_SEQ_LEN,
        duplicate_data=True,
    )
    # Build the interleaved table from the engine's own frequency vector --
    # `1 / (base ** (arange(0, dim, 2) / dim))`, evaluated in fp64 and rounded
    # to fp32 by `create_sinusoidal_positions_for_attention_plugin`; HF's
    # `compute_default_rope_parameters` is the same formula in fp32. Sharing
    # the vector with the plain table below is what lets the text-only case
    # compare the two layouts rather than two roundings of `inv_freq`.
    inv_freq, plain = rope.create_rope_const_params()
    if inv_freq is None:
        # The backend caches this by weakref when model extra attrs are
        # installed; rebuild the same vector rather than skip the case.
        inv_freq = torch.tensor(
            1.0
            / (theta ** (np.arange(0, QK_ROPE_HEAD_DIM, 2) / QK_ROPE_HEAD_DIM)).astype(np.float32),
            device="cuda",
        )
    table = _mrope_table(inv_freq, plane_positions)
    env = _MlaCtxEnv(num_heads, theta=theta, cos_sin=table)
    certifying(op)
    try:
        run = _drive(env, cached, new)
    finally:
        env.shutdown()

    # 1. The kernel's in-place rotations are HF's apply_rotary_pos_emb on the
    # same interleaved table (one rounding to bf16, as the kernel writes).
    q_heads = run.q.view(-1, num_heads, env.q_head_size)
    q_orig_heads = run.q_orig.view(-1, num_heads, env.q_head_size)
    hf_q = env.hf_apply_rotary(q_orig_heads[..., env.nope_size :], run.positions)
    torch.testing.assert_close(q_heads[..., env.nope_size :], hf_q.to(env.torch_dtype))
    hf_k = env.hf_apply_rotary(run.latent_orig[:, env.lora_size :], run.positions)
    torch.testing.assert_close(run.latent_cache[:, env.lora_size :], hf_k.to(env.torch_dtype))
    # The driver's pair-math mirror and HF's formula are the same function on
    # this table (fp32, unrounded), so the two oracles agree before rounding.
    torch.testing.assert_close(
        env.rope_fp32(q_orig_heads[..., env.nope_size :], run.positions),
        hf_q,
        rtol=1e-6,
        atol=1e-5,
    )

    # 2. Controls: the comparison above moves for the engine's non-interleaved
    # layout and for swapped height/width planes.
    got = q_heads[..., env.nope_size :]
    wrong_layouts = {
        "non-interleaved sections": _mrope_table_concatenated(inv_freq, plane_positions),
        "swapped height/width planes": _mrope_table(inv_freq, plane_positions[:, [0, 2, 1]]),
    }
    # Measured on sm_103, fraction of q's rope elements past four bf16 ulps of
    # their head's largest magnitude (the gate is 0.05):
    #
    #   wrong layout                    theta 1e7   theta 1e4
    #   non-interleaved sections          0.4014      0.6678
    #   swapped height/width planes       0.3915      0.6070
    for name, wrong in wrong_layouts.items():
        mirror = _hf_apply_rotary(
            wrong, q_orig_heads[..., env.nope_size :], run.positions, env.rope_size
        )
        frac = _outside_band_fraction(got, mirror.to(env.torch_dtype))
        assert frac > 0.05, (
            f"control '{name}' moved only {frac:.3f} of q's rope elements past "
            "four bf16 ulps — the gate would not have caught the wrong layout"
        )

    # 3. With text-only positions the three planes coincide and the
    # interleaved construction is the plain duplicated table's rotation.
    text_planes = positions_all.unsqueeze(-1).repeat(1, 3)
    text_table = _mrope_table(inv_freq, text_planes)
    probe = torch.randn(4, num_heads, env.rope_size, dtype=env.torch_dtype, device="cuda")
    probe_positions = [0, 1, 127, 1023]
    torch.testing.assert_close(
        _hf_apply_rotary(text_table, probe, probe_positions, env.rope_size),
        _hf_apply_rotary(plain, probe, probe_positions, env.rope_size),
        rtol=1e-6,
        atol=1e-5,
    )


def _single_context_call_args(env: _MlaCtxEnv, tokens: int = 8):
    """Metadata, q and latent_cache for one fresh context sequence."""
    env.kv_cache_manager.add_dummy_requests([0], token_nums=[tokens])
    metadata = env.prepare_metadata([0], [tokens], 1, [0])
    q = torch.randn(tokens, env.num_heads * env.q_head_size, dtype=env.torch_dtype, device="cuda")
    latent_cache = torch.randn(tokens, env.latent_size, dtype=env.torch_dtype, device="cuda")
    return metadata, q, latent_cache


def test_rejects_int8_kv_cache_quant_mode() -> None:
    """A quant_mode asking for a non-fp8/non-fp4 quantized KV cache is rejected,
    not silently treated as fp8 or as high precision."""
    torch.manual_seed(25)
    env = _MlaCtxEnv(NUM_HEADS_SWA, fp8_pool=True, orig_quant=1.0)
    try:
        metadata, q, latent_cache = _single_context_call_args(env)
        raised = ""
        try:
            _call(env, metadata, q, latent_cache, 1, quant_mode=QUANT_MODE_INT8_KV_CACHE)
        except RuntimeError as exc:
            raised = str(exc)
        assert "Only FP8 and NVFP4 KV caches are supported for now" in raised, (
            f"quant_mode={QUANT_MODE_INT8_KV_CACHE} was not rejected as expected; got: {raised!r}"
        )
    finally:
        env.shutdown()


def test_residual_dim_guard() -> None:
    """The entry's `is_valid` refusal for `residual_dim != 0`, after the
    unguarded call is shown to fail on its own: the op rejects residual
    quantization on a non-FP4 pool loudly, and the guard states the same
    precondition before the kernel sees it."""
    torch.manual_seed(26)
    env = _MlaCtxEnv(NUM_HEADS_SWA)
    try:
        metadata, q, latent_cache = _single_context_call_args(env)
        raised = ""
        try:
            _call(env, metadata, q, latent_cache, 1, residual_dim=env.rope_size)
        except RuntimeError as exc:
            raised = str(exc)
        assert "MLA KV residual quantization requires an NVFP4 KV cache." in raised, (
            f"residual_dim={env.rope_size} on a bf16 pool was not rejected as "
            f"expected; got: {raised!r}"
        )
        with validating(op):
            with pytest.raises(AssertionError, match="residual_dim must be 0"):
                _call(env, metadata, q, latent_cache, 1, residual_dim=env.rope_size)
    finally:
        env.shutdown()
