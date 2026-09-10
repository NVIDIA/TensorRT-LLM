# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mla_rope_generation catalog entry.

The op reads paged-KV-cache addressing tensors and per-sequence length
tensors that the runtime normally derives from a KVCacheManager and a
prepared TrtllmAttentionMetadata. The test builds that state for real —
an actual MLA (SELFKONLY, kv_factor=1) KVCacheManager and a prepared
TrtllmAttentionMetadata — then checks the kernel effects against a torch
fp32 reference.

Two surfaces:

1. bf16 latent pool (`quant_mode=0`, all fp8 buffers None), H = 16: GPT-J
   RoPE of q_pe written into fused_q's tail, [compressed_kv | rope(k_pe)]
   appended into the paged latent cache, and the decode-FMHA scheduler
   buffers (cu_q/cu_kv/counter) filled. Covered at both pool page sizes:
   64, and 32 (the engine default), where the appended slot is placed at
   every 32-slot alignment and multi-step decodes cross a page boundary
   between steps.

2. fp8-e4m3 latent pool (`quant_mode=128`) at the DeepSeek-R1-0528 cell —
   H = 128, tokens_per_block = 32, C/R/nope/v = 512/64/128/128,
   beam_width = 1, rope_append = True. Here the op stops writing fused_q
   entirely and instead writes the three buffers the fp8 MLA decode FMHA
   consumes: an e4m3 copy of the fused query in `quant_q_buffer`, and the
   two folded FMHA scales in `mla_bmm1_scale` / `mla_bmm2_scale`; the cache
   append lands as e4m3. Swept over the KV scaling factor (omitted = the
   production call, and explicit 1.0 / 1.5 / 2.0), over q_scaling (1.0 and
   DeepSeek-R1's YaRN attention temperature), and over a deliberately
   inconsistent scale pair that separates which tensor drives the write side
   from which drives the read-side scales.

3. predicted_tokens_per_seq (`P`) > 1 — the MTP path — over both pools, at
   P in {1, 2, 3, 4}. One generation sequence then arrives with P query
   tokens at P consecutive absolute positions, and both helpers below are
   parameterised on P so P = 1 is the same code path it always was. What the
   P > 1 cases pin: each row's own rope position, the P cache rows per
   sequence (page-boundary straddles at every 32-slot alignment included),
   the scheduler-buffer fills, and which length tensor the position comes
   from. Two controls sit beside them — wrong-position mirrors, and the
   documented paged-append race armed at P > 1 so a clean pool comparison
   is distinguishable from a blind one.
"""

import math
from typing import List, NamedTuple, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

from .mla_rope_generation import mla_rope_generation

assert torch.cuda.is_available(), "mla_rope_generation requires a CUDA device"

# DeepSeek-V3 MLA head geometry (num_heads reduced to a TP-slice-like 16).
NUM_HEADS = 16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
GEN_HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
# Pool page size. 32 is the engine default (KvCacheConfig.tokens_per_block),
# 64 the value a tuned MLA target opts into; both are covered below.
TOKENS_PER_BLOCK = 64
PAGE32 = 32
MAX_SEQ_LEN = 1024

# ─── fp8-e4m3 latent pool surface ─────────────────────────────────────
# DeepSeek-R1-0528 cell: full 128-head MLA over an fp8 latent pool.
NUM_HEADS_R1 = 128
Q_LORA_RANK_R1 = 1536
QUANT_MODE_FP8_KV_CACHE = 128  # QuantMode.FP8_KV_CACHE
# DeepSeek-R1's YaRN attention temperature: mscale = 0.1 * mscale_all_dim *
# ln(factor) + 1 with mscale_all_dim = 1.0 and factor = 40. trtllm carries it
# as q_scaling = 1 / mscale**2, so the softmax scale mscale**2 / sqrt(nope + R)
# comes out of 1 / (q_scaling * sqrt(nope + R)). ~0.53366 — the value the
# target passes, and the reason q_scaling != 1 has to be certified here.
R1_MSCALE = 0.1 * math.log(40.0) + 1.0
Q_SCALING_R1 = 1.0 / (R1_MSCALE * R1_MSCALE)

# The op derives mla_bmm1_scale in fp32 on the device; the reference computes
# the same expression in double and rounds once. Max observed deviation over
# the swept (scale, q_scaling) grid: 7.2e-8 relative — 0.6 fp32 ulp, and zero
# on several of the cases. The gate is torch's default fp32 rtol with atol
# tightened to 0 (18x the observed deviation). It is not a loosening: every
# way of getting this scale wrong misses by a factor, not by ulps. In
# |got - want| / |want|, the metric assert_close gates on: a reference that
# drops the s**2 fold at s = 1.5 is off by 1.25 (9.6e5x the gate), one that
# drops q_scaling = 0.53366 by 0.874 (6.7e5x), one using sqrt(C + R) instead
# of sqrt(nope + R) by 0.732 (5.6e5x), and one leaving log2(e) off element
# [1] by 0.443 (3.4e5x).
BMM_SCALE_RTOL = 1.3e-6

# One e4m3 ulp. e4m3 keeps 3 mantissa bits, so 2**-3 relative is one ulp — the
# gate for the two parts of the fp8 output that go through the in-kernel RoPE
# (quant_q_buffer's tail and the appended row's k_pe half), where the kernel
# and a torch fp32 reference can evaluate x*cos -+ y*sin in different orders.
# Paired with a bit-exact-fraction floor, because in practice the coarse e4m3
# rounding absorbs that difference completely: every roped element measured on
# sm_100 under quant_mode 128 (all scales, all cases below) was bit-exact, so
# neither half of the gate has ever been approached. That both halves still
# discriminate is measured, not assumed — test_fp8_kv_explicit_unit_scale runs
# three wrong mirrors of the same bytes and asserts each blows both.
E4M3_ULP_RTOL = 2**-3
E4M3_MAX_INEXACT_FRACTION = 1e-3


class _MlaEnv:
    """Real op state: MLA KV cache manager + duplicated-layout RoPE table.

    With fp8_pool the manager allocates an e4m3 latent pool (one byte per
    element) and the two KV scaling-factor tensors are built; orig_quant and
    quant_orig are independent on purpose so a test can pass a deliberately
    inconsistent pair.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        tokens_per_block: int = TOKENS_PER_BLOCK,
        num_heads: int = NUM_HEADS,
        fp8_pool: bool = False,
        orig_quant: Optional[float] = None,
        quant_orig: Optional[float] = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.tokens_per_block = tokens_per_block
        self.num_heads = num_heads
        self.fp8_pool = fp8_pool
        self.kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=131072, enable_block_reuse=False),
            CacheType.SELFKONLY,  # MLA latent cache: kv_factor=1, one kv head
            num_layers=1,
            num_kv_heads=1,
            head_dim=GEN_HEAD_SIZE,
            tokens_per_block=tokens_per_block,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8 if fp8_pool else DataType.BF16,
        )
        # The pool must really be paged at the size and element type the case
        # claims: the op sizes its page slabs from tokens_per_block and
        # quant_mode, not from anything the manager tells it.
        assert self.kv_cache_manager.tokens_per_block == tokens_per_block
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None
        assert pool.dtype == (torch.float8_e4m3fn if fp8_pool else torch.bfloat16)
        # KV scaling-factor tensors. None for both is what the production call
        # site passes (TrtllmAttention.mla_rope_generation hard-codes None), and
        # the op then behaves as if both were 1.0.
        self.kv_scale_orig_quant = _scale_tensor(orig_quant)
        self.kv_scale_quant_orig = _scale_tensor(quant_orig)
        # The exact fp32 numbers the op sees, as 0-dim tensors, so the mirror
        # cannot disagree with the kernel in the last bit.
        self.write_scale = _scale_value(orig_quant)  # multiplies on quantize
        self.read_scale = _scale_value(quant_orig)  # folded into the bmm scales
        # Duplicated-layout fp32 (cos, sin) table, as the MLA backend builds
        # it (RopeParams.from_config sets duplicate_data=True for MLA models).
        rope = RopeParams(
            dim=QK_ROPE_HEAD_DIM,
            theta=10000.0,
            max_positions=MAX_SEQ_LEN,
            duplicate_data=True,
        )
        _, self.rotary_cos_sin = rope.create_rope_const_params()

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

    def rope_ref(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """GPT-J interleaved rotation of the last dim, fp32 math, bf16 result."""
        half = QK_ROPE_HEAD_DIM // 2
        table = self.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        cos = table[position, :half, 0]
        sin = table[position, :half, 1]
        pairs = x.float().reshape(*x.shape[:-1], half, 2)
        out = torch.empty_like(pairs)
        out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        return out.reshape(x.shape).to(x.dtype)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """e4m3(x * kv_scale_orig_quant) — the op's write-side quantization,
        applied to both the appended cache row and the fused query."""
        return (x.float() * self.write_scale).to(torch.float8_e4m3fn)

    def pool_tensor(self) -> torch.Tensor:
        pool = self.kv_cache_manager.get_buffers(0)
        assert pool is not None
        return pool

    def blocks(self, request_id: int) -> List[int]:
        return list(self.kv_cache_manager.get_batch_cache_indices([request_id])[0])

    def slot(self, request_id: int, position: int) -> tuple[int, int]:
        tpb = self.tokens_per_block
        return self.blocks(request_id)[position // tpb], position % tpb

    def cache_row(self, request_id: int, position: int) -> torch.Tensor:
        """Read one token's latent row back from the paged pool."""
        page, offset = self.slot(request_id, position)
        return self.pool_tensor()[page, 0, offset, 0]

    def shutdown(self) -> None:
        self.kv_cache_manager.shutdown()


def _scale_tensor(value: Optional[float]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return torch.full((1,), value, dtype=torch.float32, device="cuda")


def _scale_value(value: Optional[float]) -> torch.Tensor:
    """The fp32 number the op uses for this role — 1.0 when the tensor is
    omitted. 0-dim so it broadcasts into an fp32 multiply."""
    return torch.tensor(1.0 if value is None else value, dtype=torch.float32).cuda()


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Bit-exact comparison of two e4m3 tensors, signed zeros included."""
    torch.testing.assert_close(
        got.reshape(-1).view(torch.uint8),
        want.reshape(-1).view(torch.uint8),
        rtol=0.0,
        atol=0.0,  # pure scale-and-round: must be bit-exact
        msg=lambda m: f"{what} not bit-exact\n{m}",
    )


def _assert_e4m3_roped(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """One-e4m3-ulp gate plus a bit-exact-fraction floor, for the parts of the
    fp8 output that pass through the in-kernel RoPE."""
    torch.testing.assert_close(got.float(), want.float(), rtol=E4M3_ULP_RTOL, atol=0.0)
    inexact = int((got.reshape(-1).view(torch.uint8) != want.reshape(-1).view(torch.uint8)).sum())
    allowed = max(1, int(E4M3_MAX_INEXACT_FRACTION * got.numel()))
    assert inexact <= allowed, (
        f"{what} not bit-exact enough: {inexact}/{got.numel()} bytes differ (allowed {allowed})"
    )


class _Fp8Run(NamedTuple):
    """The tensors one fp8 generation step was driven with and produced.

    Rows of the tensors are generation tokens: row `r` belongs to generation
    sequence `r // predicted_tokens_per_seq` and sits at `positions[r]`."""

    fused_q: torch.Tensor
    q_pe: torch.Tensor
    latent_cache: torch.Tensor
    quant_q_buffer: torch.Tensor
    gen_ids: List[int]
    positions: List[int]
    predicted_tokens_per_seq: int


def _positions(
    kv_lens: List[int], num_contexts: int, num_gen: int, tokens_per_seq: int
) -> List[int]:
    """0-based absolute position of every generation row, in row order.

    Row `r` is generation sequence `r // P`'s `r % P`-th new token; the P new
    tokens of one sequence occupy consecutive positions ending at its total
    KV length minus one."""
    return [
        kv_lens[num_contexts + g] - tokens_per_seq + t
        for g in range(num_gen)
        for t in range(tokens_per_seq)
    ]


def _run_and_check(
    env: _MlaEnv,
    request_ids: List[int],
    seq_lens: List[int],
    num_contexts: int,
    cached_lens: List[int],
    q_pe_contiguous: bool,
    predicted_tokens_per_seq: int = 1,
) -> None:
    """One generation step over a bf16 latent pool: allocate P slots per
    generation sequence, run the op over the generation tokens, and verify
    every kernel effect."""
    gen_ids = request_ids[num_contexts:]
    num_gen = len(gen_ids)
    p = predicted_tokens_per_seq
    assert seq_lens[num_contexts:] == [p] * num_gen
    num_heads = env.num_heads
    for rid in gen_ids:
        for _ in range(p):
            env.kv_cache_manager.impl.add_token(rid)
    metadata = env.prepare_metadata(request_ids, seq_lens, num_contexts, cached_lens)
    kv_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
    positions = _positions(kv_lens, num_contexts, num_gen, p)
    rows = num_gen * p

    fused_q = torch.randn(rows, num_heads, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    q_pe = _make_q_pe(rows, num_heads, q_pe_contiguous)
    latent_cache = torch.randn(rows, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    fused_q_orig = fused_q.clone()
    q_pe_orig = q_pe.clone()
    cu_q_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
    cu_kv_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
    fmha_scheduler_counter = torch.full((1,), 7, dtype=torch.uint32, device="cuda")

    mla_rope_generation(
        fused_q,
        q_pe,
        latent_cache,
        env.rotary_cos_sin,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        None,  # mla_bmm1_scale: fp8-KV-cache path only
        None,  # mla_bmm2_scale
        None,  # quant_q_buffer
        metadata.kv_lens_cuda_runtime,
        metadata.kv_lens_runtime,
        metadata.prompt_lens_cpu_runtime,
        num_contexts,
        metadata.kv_cache_block_offsets,
        env.kv_cache_manager.kv_cache_pool_pointers,
        env.kv_cache_manager.kv_cache_pool_mapping,
        None,  # kv_scale_orig_quant
        None,  # kv_scale_quant_orig
        None,  # kv_cache_scale_orig_quant
        None,  # out_scale
        None,  # block_ids_per_seq
        [None, None],  # helix_tensor_params
        p,  # predicted_tokens_per_seq
        0,  # layer_idx
        num_heads,
        1,  # num_kv_heads
        GEN_HEAD_SIZE,
        0,  # residual_dim
        env.tokens_per_block,
        MAX_SEQ_LEN,  # attention_window_size
        1,  # beam_width
        0,  # quant_mode: bf16 KV cache
        1.0,  # q_scaling
        0,  # q_lora_rank
        KV_LORA_RANK,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        True,  # rope_append
    )
    torch.cuda.synchronize()

    # 1. fused_q tail = rope(q_pe) at that row's own position; the absorbed-q
    #    head slice is untouched.
    for r in range(rows):
        ref = env.rope_ref(q_pe_orig[r], positions[r])
        torch.testing.assert_close(fused_q[r, :, KV_LORA_RANK:], ref)
    torch.testing.assert_close(
        fused_q[..., :KV_LORA_RANK],
        fused_q_orig[..., :KV_LORA_RANK],
        rtol=0.0,
        atol=0.0,  # caller-owned region: must be bitwise untouched
    )
    # q_pe is an input only (mutable in the schema, not mutated in practice).
    torch.testing.assert_close(q_pe, q_pe_orig, rtol=0.0, atol=0.0)

    # 2. Cache append: [compressed_kv | rope(k_pe)] at each row's own slot.
    for r in range(rows):
        rid = gen_ids[r // p]
        row = env.cache_row(rid, positions[r])
        torch.testing.assert_close(
            row[:KV_LORA_RANK],
            latent_cache[r, :KV_LORA_RANK],
            rtol=0.0,
            atol=0.0,  # dtype-preserving copy: must be bitwise equal
        )
        ref_k = env.rope_ref(latent_cache[r, KV_LORA_RANK:], positions[r])
        torch.testing.assert_close(row[KV_LORA_RANK:], ref_k)

    # 3. Scheduler buffers over generation sequences only.
    _assert_scheduler_buffers(
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        num_gen,
        num_heads,
        kv_lens[num_contexts:],
        p,
    )


def _run_and_check_fp8(
    env: _MlaEnv,
    request_ids: List[int],
    seq_lens: List[int],
    num_contexts: int,
    cached_lens: List[int],
    q_pe_contiguous: bool,
    q_scaling: float,
    q_lora_rank: int = 0,
    pass_bmm1: bool = True,
    pass_bmm2: bool = True,
    repeats: int = 1,
    predicted_tokens_per_seq: int = 1,
) -> _Fp8Run:
    """One generation step over an fp8-e4m3 latent pool. Verifies the three
    fp8 buffers, the quantized cache append, the scheduler buffers, that the
    op writes nothing else in the pool, and that fused_q/q_pe/latent_cache all
    come back bitwise untouched."""
    gen_ids = request_ids[num_contexts:]
    num_gen = len(gen_ids)
    p = predicted_tokens_per_seq
    assert seq_lens[num_contexts:] == [p] * num_gen
    num_heads = env.num_heads
    for rid in gen_ids:
        for _ in range(p):
            env.kv_cache_manager.impl.add_token(rid)
    metadata = env.prepare_metadata(request_ids, seq_lens, num_contexts, cached_lens)
    kv_lens = [c + s for c, s in zip(cached_lens, seq_lens)]
    positions = _positions(kv_lens, num_contexts, num_gen, p)
    rows = num_gen * p

    fused_q = torch.randn(rows, num_heads, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    q_pe = _make_q_pe(rows, num_heads, q_pe_contiguous)
    latent_cache = torch.randn(rows, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    fused_q_orig = fused_q.clone()
    q_pe_orig = q_pe.clone()
    latent_orig = latent_cache.clone()
    cu_q_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
    cu_kv_seqlens = torch.full((num_gen + 1,), -1, dtype=torch.int32, device="cuda")
    fmha_scheduler_counter = torch.full((1,), 7, dtype=torch.uint32, device="cuda")
    # Production allocates quant_q_buffer as uint8 [tokens, heads, C + R].
    quant_q_buffer = torch.full(
        (rows, num_heads, GEN_HEAD_SIZE), 255, dtype=torch.uint8, device="cuda"
    )
    bmm1 = torch.full((2,), -99.0, dtype=torch.float32, device="cuda") if pass_bmm1 else None
    bmm2 = torch.full((1,), -99.0, dtype=torch.float32, device="cuda") if pass_bmm2 else None
    pool = env.pool_tensor()
    pool_before = pool.clone()

    def call() -> None:
        mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            env.rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            bmm1,
            bmm2,
            quant_q_buffer,
            metadata.kv_lens_cuda_runtime,
            metadata.kv_lens_runtime,
            metadata.prompt_lens_cpu_runtime,
            num_contexts,
            metadata.kv_cache_block_offsets,
            env.kv_cache_manager.kv_cache_pool_pointers,
            env.kv_cache_manager.kv_cache_pool_mapping,
            env.kv_scale_orig_quant,
            env.kv_scale_quant_orig,
            None,  # kv_cache_scale_orig_quant
            None,  # out_scale
            None,  # block_ids_per_seq
            [None, None],  # helix_tensor_params
            p,  # predicted_tokens_per_seq
            0,  # layer_idx
            num_heads,
            1,  # num_kv_heads
            GEN_HEAD_SIZE,
            0,  # residual_dim
            env.tokens_per_block,
            MAX_SEQ_LEN,  # attention_window_size
            1,  # beam_width
            QUANT_MODE_FP8_KV_CACHE,
            q_scaling,
            q_lora_rank,
            KV_LORA_RANK,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            V_HEAD_DIM,
            True,  # rope_append
        )
        torch.cuda.synchronize()

    call()

    # 1. fused_q is read-only on this path: the roped q goes to quant_q_buffer
    #    instead, so neither half of fused_q is written.
    torch.testing.assert_close(fused_q, fused_q_orig, rtol=0.0, atol=0.0)
    torch.testing.assert_close(q_pe, q_pe_orig, rtol=0.0, atol=0.0)
    torch.testing.assert_close(latent_cache, latent_orig, rtol=0.0, atol=0.0)

    # 2. quant_q_buffer = e4m3(fused_q * orig_quant) with the tail replaced by
    #    the roped q_pe (rounded to bf16 first, as the kernel's own dtype does).
    quant_q = quant_q_buffer.view(torch.float8_e4m3fn)
    roped_q = torch.stack([env.rope_ref(q_pe_orig[r], positions[r]) for r in range(rows)])
    _assert_bytes_equal(
        quant_q[..., :KV_LORA_RANK],
        env.quantize(fused_q_orig[..., :KV_LORA_RANK]),
        "quant_q_buffer absorbed-q head",
    )
    _assert_e4m3_roped(
        quant_q[..., KV_LORA_RANK:],
        env.quantize(roped_q),
        "quant_q_buffer roped tail",
    )

    # 3. The two decode-FMHA scales, folded with the read-side scaling factor.
    if bmm1 is not None:
        s = float(env.read_scale)
        x = s * s / (q_scaling * math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM))
        want1 = torch.tensor([x, x * math.log2(math.e)], dtype=torch.float32, device="cuda")
        torch.testing.assert_close(bmm1, want1, rtol=BMM_SCALE_RTOL, atol=0.0)
    if bmm2 is not None:
        torch.testing.assert_close(
            bmm2,
            env.read_scale.reshape(1),
            rtol=0.0,
            atol=0.0,  # a copy of kv_scale_quant_orig, not a computation
        )

    # 4. Cache append, quantized: e4m3([compressed_kv | rope(k_pe)] * orig_quant),
    #    one row per generation token at that token's own position.
    for r in range(rows):
        rid = gen_ids[r // p]
        row = env.cache_row(rid, positions[r])
        _assert_bytes_equal(
            row[:KV_LORA_RANK],
            env.quantize(latent_orig[r, :KV_LORA_RANK]),
            f"appended compressed_kv (request {rid}, position {positions[r]})",
        )
        ref_k = env.rope_ref(latent_orig[r, KV_LORA_RANK:], positions[r])
        _assert_e4m3_roped(
            row[KV_LORA_RANK:],
            env.quantize(ref_k),
            f"appended k_pe (request {rid}, position {positions[r]})",
        )

    # 5. Nothing else in the pool moved: exactly P C+R-byte rows per
    #    generation sequence, at the slots their positions address.
    changed = pool.view(torch.uint8) != pool_before.view(torch.uint8)
    expected_slots = {env.slot(gen_ids[r // p], positions[r]) for r in range(rows)}
    idx = changed.nonzero().cpu()
    got_slots = set(zip(idx[:, 0].tolist(), idx[:, 2].tolist()))
    assert got_slots == expected_slots, (
        f"op touched pool slots {sorted(got_slots)}, expected {sorted(expected_slots)}"
    )
    # Upper bound, not equality: a written byte that happens to match the byte
    # already there (an element quantizing to +0 over a zeroed pool) is
    # invisible to a snapshot diff. The rows' contents are pinned above.
    assert idx.shape[0] <= rows * GEN_HEAD_SIZE, (
        f"op wrote {idx.shape[0]} pool bytes, at most {rows * GEN_HEAD_SIZE} rows' worth expected"
    )

    # 6. Scheduler buffers over generation sequences only.
    _assert_scheduler_buffers(
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        num_gen,
        num_heads,
        kv_lens[num_contexts:],
        p,
    )

    # 7. Repeating the prepared step rewrites the same bytes: the position is
    #    derived from sequence_length, so the call is idempotent rather than
    #    double-appending, and the fp8 outputs are run-to-run stable.
    for _ in range(repeats - 1):
        quant_q_before = quant_q_buffer.clone()
        pool_after_first = pool.clone()
        call()
        torch.testing.assert_close(quant_q_buffer, quant_q_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            pool.view(torch.uint8),
            pool_after_first.view(torch.uint8),
            rtol=0.0,
            atol=0.0,
        )

    return _Fp8Run(
        fused_q=fused_q_orig,
        q_pe=q_pe_orig,
        latent_cache=latent_orig,
        quant_q_buffer=quant_q_buffer,
        gen_ids=gen_ids,
        positions=positions,
        predicted_tokens_per_seq=p,
    )


def _make_q_pe(rows: int, num_heads: int, contiguous: bool) -> torch.Tensor:
    if contiguous:
        return torch.randn(rows, num_heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    # mla.py-style strided view: q_pe sliced out of a packed q tensor.
    q = torch.randn(
        rows,
        num_heads,
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return q.split([QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM], dim=-1)[1]


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


def test_bf16_decode_batch_strided_q_pe() -> None:
    """Pure-decode batch; one sequence's new slot crosses a block boundary
    (cached 64 = one full 64-token block); q_pe is a packed-q strided view."""
    torch.manual_seed(0)
    env = _MlaEnv()
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1], token_nums=[64, 32])
        _run_and_check(
            env,
            request_ids=[0, 1],
            seq_lens=[1, 1],
            num_contexts=0,
            cached_lens=[64, 32],
            q_pe_contiguous=False,
        )
    finally:
        env.shutdown()


def test_bf16_mixed_batch_skips_context() -> None:
    """Context sequence leads the batch; the op consumes only the generation
    tokens and indexes length/block tensors starting at num_contexts."""
    torch.manual_seed(1)
    env = _MlaEnv()
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[40, 100, 7])
        _run_and_check(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[40, 1, 1],
            num_contexts=1,
            cached_lens=[0, 100, 7],
            q_pe_contiguous=True,
        )
    finally:
        env.shutdown()


def test_bf16_large_decode_batch_multi_step() -> None:
    """64-sequence decode batch (many tokens per call), two consecutive
    steps so the second call appends after the first call's tokens."""
    torch.manual_seed(2)
    env = _MlaEnv(max_batch_size=64)
    try:
        rids = list(range(64))
        cached = [(37 * (i + 1)) % 800 + 1 for i in range(64)]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        for step in range(2):
            _run_and_check(
                env,
                request_ids=rids,
                seq_lens=[1] * 64,
                num_contexts=0,
                cached_lens=[c + step for c in cached],
                q_pe_contiguous=True,
            )
    finally:
        env.shutdown()


def test_bf16_page32_decode_batch_alignments() -> None:
    """Page size 32 (the engine default): a pure-decode batch whose three
    new slots land at every alignment a 32-token page has — position 32
    (page 1 slot 0, a fresh page after one full page), 31 (page 0's last
    slot) and 64 (page 2 slot 0, after two full pages). q_pe is a packed-q
    strided view."""
    torch.manual_seed(3)
    env = _MlaEnv(tokens_per_block=PAGE32)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[32, 31, 64])
        _run_and_check(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[1, 1, 1],
            num_contexts=0,
            cached_lens=[32, 31, 64],
            q_pe_contiguous=False,
        )
    finally:
        env.shutdown()


def test_bf16_page32_mixed_batch_skips_context() -> None:
    """Page size 32, context sequence leading the batch: the op consumes
    only the generation tokens and indexes length/block tensors from
    num_contexts. The generation slots sit at position 96 (page 3 slot 0)
    and 7 (mid first page)."""
    torch.manual_seed(4)
    env = _MlaEnv(tokens_per_block=PAGE32)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[40, 96, 7])
        _run_and_check(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[40, 1, 1],
            num_contexts=1,
            cached_lens=[0, 96, 7],
            q_pe_contiguous=True,
        )
    finally:
        env.shutdown()


def test_bf16_page32_large_decode_batch_multi_step() -> None:
    """Page size 32, 64-sequence decode batch over two consecutive steps.
    The cached lengths spread the new slots over 25 pages; two sequences
    (i = 5, 37) sit on a page's last slot at step 0 and cross into the next
    page at step 1, and two (i = 18, 50) start a fresh page at step 0."""
    torch.manual_seed(5)
    env = _MlaEnv(max_batch_size=64, tokens_per_block=PAGE32)
    try:
        rids = list(range(64))
        cached = [(37 * (i + 1)) % 800 + 1 for i in range(64)]
        assert [i for i in range(64) if cached[i] % PAGE32 == PAGE32 - 1] == [5, 37]
        assert [i for i in range(64) if cached[i] % PAGE32 == 0] == [18, 50]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        for step in range(2):
            _run_and_check(
                env,
                request_ids=rids,
                seq_lens=[1] * 64,
                num_contexts=0,
                cached_lens=[c + step for c in cached],
                q_pe_contiguous=True,
            )
    finally:
        env.shutdown()


def _fp8_env(
    max_batch_size: int = 8,
    orig_quant: Optional[float] = None,
    quant_orig: Optional[float] = None,
) -> _MlaEnv:
    """DeepSeek-R1-0528 cell over an fp8-e4m3 latent pool: H = 128, page 32."""
    return _MlaEnv(
        max_batch_size=max_batch_size,
        tokens_per_block=PAGE32,
        num_heads=NUM_HEADS_R1,
        fp8_pool=True,
        orig_quant=orig_quant,
        quant_orig=quant_orig,
    )


def test_fp8_kv_decode_production_config() -> None:
    """The production fp8 generation call: both KV scale tensors omitted (what
    TrtllmAttention.mla_rope_generation passes), DeepSeek-R1's YaRN q_scaling,
    q_lora_rank 1536, q_pe a packed-q strided view. Three decode slots at
    positions 32 / 31 / 64 — a fresh page, a page's last slot, and a fresh
    page after two full ones. Run twice to pin run-to-run stability."""
    torch.manual_seed(10)
    env = _fp8_env()
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[32, 31, 64])
        _run_and_check_fp8(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[1, 1, 1],
            num_contexts=0,
            cached_lens=[32, 31, 64],
            q_pe_contiguous=False,
            q_scaling=Q_SCALING_R1,
            q_lora_rank=Q_LORA_RANK_R1,
            repeats=2,
        )
    finally:
        env.shutdown()


def test_fp8_kv_explicit_unit_scale() -> None:
    """Explicit scaling factor 1.0 (a checkpoint whose k_scale/v_scale are the
    production 1.0, passed as real tensors) at q_scaling 1.0, then the
    wrong-variant controls that measure what the e4m3-ulp gate discriminates.
    Every variant below is a plausible mis-derivation of the same buffer."""
    torch.manual_seed(11)
    env = _fp8_env(orig_quant=1.0, quant_orig=1.0)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1], token_nums=[32, 31])
        run = _run_and_check_fp8(
            env,
            request_ids=[0, 1],
            seq_lens=[1, 1],
            num_contexts=0,
            cached_lens=[32, 31],
            q_pe_contiguous=True,
            q_scaling=1.0,
        )
        # Controls. Every fp8 comparison above came back bit-exact, which is
        # only meaningful if the same comparison moves for a wrong mirror.
        # Measured on sm_100 at this case — the k_pe half of request 0's
        # appended row (position 32) against three plausible mis-derivations:
        #
        #   mirror                        bytes differing   max rel dev
        #   un-roped k_pe                    49/64 (76.6%)    92.4  (739x)
        #   rope at position + 1             28/64 (43.8%)     3.2  (25.6x)
        #   the other request's roped row    63/64 (98.4%)    16.6  (133x)
        #
        # with the multiplier against E4M3_ULP_RTOL. The bit-exact-fraction
        # floor allows 1 byte in a 64-element row, so even the weakest control
        # — the off-by-one position, where at position 32 the low-frequency
        # rope pairs barely move and over half the e4m3 bytes stay put — is 28x
        # past the fraction allowance and 25.6x past the tolerance.
        # quant_q_buffer's tail against fused_q's own never-roped q_pe differs
        # in 74.5% of 16 384 bytes, against an allowance of 16.
        gen_ids, positions = run.gen_ids, run.positions
        row0 = env.cache_row(gen_ids[0], positions[0])[KV_LORA_RANK:]
        row0_bytes = row0.view(torch.uint8)
        wrong = {
            "un-roped k_pe": env.quantize(run.latent_cache[0, KV_LORA_RANK:]),
            "rope at position + 1": env.quantize(
                env.rope_ref(run.latent_cache[0, KV_LORA_RANK:], positions[0] + 1)
            ),
            "other request's row": env.quantize(
                env.rope_ref(run.latent_cache[1, KV_LORA_RANK:], positions[1])
            ),
        }
        for name, mirror in wrong.items():
            frac = float((row0_bytes != mirror.view(torch.uint8)).float().mean())
            rel = float(
                ((row0.float() - mirror.float()).abs() / mirror.float().abs().clamp(min=1e-9)).max()
            )
            assert frac > 100 * E4M3_MAX_INEXACT_FRACTION, (
                f"control '{name}' moved only {frac:.3f} of bytes — the "
                "bit-exact-fraction floor would not have caught it"
            )
            assert rel > 4 * E4M3_ULP_RTOL, (
                f"control '{name}' stayed within {rel:.3g} relative — the "
                "one-ulp tolerance would not have caught it"
            )
        quant_tail = run.quant_q_buffer[..., KV_LORA_RANK:].view(torch.float8_e4m3fn)
        unroped = env.quantize(run.q_pe)
        frac = float((quant_tail.view(torch.uint8) != unroped.view(torch.uint8)).float().mean())
        assert frac > 0.5, f"quant_q tail control only moved {frac:.3f} of bytes"
    finally:
        env.shutdown()


def test_fp8_kv_scale_factor_sweep() -> None:
    """KV scaling factors other than the production 1.0, passed as the
    reciprocal pair a calibrated fp8 checkpoint would produce. 1.5 is not a
    power of two, so 1 / 1.5 is inexact in fp32 and the write-side multiply is
    a real rescale rather than an exponent shift. Both q_scaling values are
    exercised across the sweep."""
    for scale, q_scaling in ((1.5, Q_SCALING_R1), (2.0, 1.0)):
        torch.manual_seed(12)
        env = _fp8_env(orig_quant=1.0 / scale, quant_orig=scale)
        try:
            env.kv_cache_manager.add_dummy_requests([0, 1], token_nums=[32, 95])
            _run_and_check_fp8(
                env,
                request_ids=[0, 1],
                seq_lens=[1, 1],
                num_contexts=0,
                cached_lens=[32, 95],
                q_pe_contiguous=True,
                q_scaling=q_scaling,
            )
        finally:
            env.shutdown()


def test_fp8_kv_scale_tensors_are_independent() -> None:
    """A deliberately inconsistent scale pair (orig_quant 0.25, quant_orig 3.0)
    separates the two roles: orig_quant alone drives the write side (the
    appended row and quant_q_buffer), quant_orig alone drives both decode-FMHA
    scales. The reference is built the same way, so a passing run pins the
    split — the op does not derive either tensor from the other, and a caller
    passing a non-reciprocal pair gets a silently inconsistent round trip."""
    torch.manual_seed(13)
    env = _fp8_env(orig_quant=0.25, quant_orig=3.0)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1], token_nums=[32, 31])
        _run_and_check_fp8(
            env,
            request_ids=[0, 1],
            seq_lens=[1, 1],
            num_contexts=0,
            cached_lens=[32, 31],
            q_pe_contiguous=True,
            q_scaling=Q_SCALING_R1,
        )
    finally:
        env.shutdown()


def test_fp8_kv_mixed_batch_skips_context() -> None:
    """Context sequence leading the batch over an fp8 pool: the op consumes
    only the generation tokens and indexes length/block tensors from
    num_contexts. The generation slots sit at position 96 (page 3 slot 0) and
    7 (mid first page)."""
    torch.manual_seed(14)
    env = _fp8_env(orig_quant=1.0, quant_orig=1.0)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[40, 96, 7])
        _run_and_check_fp8(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[40, 1, 1],
            num_contexts=1,
            cached_lens=[0, 96, 7],
            q_pe_contiguous=True,
            q_scaling=Q_SCALING_R1,
        )
    finally:
        env.shutdown()


def test_fp8_kv_large_decode_batch_multi_step() -> None:
    """64-sequence fp8 decode batch (8192 quantized q rows per call) over two
    consecutive steps, at the production scale (both tensors omitted). The
    cached lengths spread the new slots over 25 pages; two sequences sit on a
    page's last slot at step 0 and cross into the next page at step 1, and two
    start a fresh page at step 0."""
    torch.manual_seed(15)
    env = _fp8_env(max_batch_size=64)
    try:
        rids = list(range(64))
        cached = [(37 * (i + 1)) % 800 + 1 for i in range(64)]
        assert [i for i in range(64) if cached[i] % PAGE32 == PAGE32 - 1] == [5, 37]
        assert [i for i in range(64) if cached[i] % PAGE32 == 0] == [18, 50]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        for step in range(2):
            _run_and_check_fp8(
                env,
                request_ids=rids,
                seq_lens=[1] * 64,
                num_contexts=0,
                cached_lens=[c + step for c in cached],
                q_pe_contiguous=True,
                q_scaling=Q_SCALING_R1,
            )
    finally:
        env.shutdown()


def test_fp8_kv_bmm_scale_buffers_are_optional() -> None:
    """Omitting mla_bmm1_scale or mla_bmm2_scale is accepted, not rejected:
    the op simply does not write that buffer and produces everything else
    unchanged. (quant_q_buffer is different — see the contract; omitting it is
    an illegal memory access, so it cannot be exercised in-process.)"""
    for pass_bmm1, pass_bmm2 in ((False, True), (True, False)):
        torch.manual_seed(16)
        env = _fp8_env(orig_quant=1.0, quant_orig=1.0)
        try:
            env.kv_cache_manager.add_dummy_requests([0], token_nums=[31])
            _run_and_check_fp8(
                env,
                request_ids=[0],
                seq_lens=[1],
                num_contexts=0,
                cached_lens=[31],
                q_pe_contiguous=True,
                q_scaling=Q_SCALING_R1,
                pass_bmm1=pass_bmm1,
                pass_bmm2=pass_bmm2,
            )
        finally:
            env.shutdown()


class _Fp8Step:
    """One prepared fp8 generation step, callable with argument overrides.

    `_run_and_check_fp8` allocates its own buffers and verifies every effect;
    this is its low-level twin, for the two cases that have to vary an
    argument it does not expose — the host length tensor and the block-offset
    table. Pure decode (`num_contexts = 0`), production KV scale tensors."""

    def __init__(
        self,
        env: _MlaEnv,
        request_ids: List[int],
        cached_lens: List[int],
        predicted_tokens_per_seq: int,
        q_scaling: float = Q_SCALING_R1,
    ) -> None:
        self.env = env
        self.p = predicted_tokens_per_seq
        self.q_scaling = q_scaling
        self.gen_ids = request_ids
        num_gen = len(request_ids)
        for rid in request_ids:
            for _ in range(self.p):
                env.kv_cache_manager.impl.add_token(rid)
        self.metadata = env.prepare_metadata(request_ids, [self.p] * num_gen, 0, cached_lens)
        kv_lens = [c + self.p for c in cached_lens]
        self.positions = _positions(kv_lens, 0, num_gen, self.p)
        self.rows = num_gen * self.p
        num_heads = env.num_heads
        self.fused_q = torch.randn(
            self.rows, num_heads, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        self.q_pe = torch.randn(
            self.rows, num_heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        self.latent_cache = torch.randn(
            self.rows, GEN_HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        self.quant_q_buffer = torch.zeros(
            self.rows, num_heads, GEN_HEAD_SIZE, dtype=torch.uint8, device="cuda"
        )
        self.cu_q_seqlens = torch.zeros(num_gen + 1, dtype=torch.int32, device="cuda")
        self.cu_kv_seqlens = torch.zeros(num_gen + 1, dtype=torch.int32, device="cuda")
        self.fmha_scheduler_counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        self.mla_bmm1_scale = torch.zeros(2, dtype=torch.float32, device="cuda")
        self.mla_bmm2_scale = torch.zeros(1, dtype=torch.float32, device="cuda")

    def call(
        self,
        host_past: Optional[torch.Tensor] = None,
        block_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        md = self.metadata
        mla_rope_generation(
            self.fused_q,
            self.q_pe,
            self.latent_cache,
            self.env.rotary_cos_sin,
            self.cu_q_seqlens,
            self.cu_kv_seqlens,
            self.fmha_scheduler_counter,
            self.mla_bmm1_scale,
            self.mla_bmm2_scale,
            self.quant_q_buffer,
            md.kv_lens_cuda_runtime,
            md.kv_lens_runtime if host_past is None else host_past,
            md.prompt_lens_cpu_runtime,
            0,  # num_contexts
            md.kv_cache_block_offsets if block_offsets is None else block_offsets,
            self.env.kv_cache_manager.kv_cache_pool_pointers,
            self.env.kv_cache_manager.kv_cache_pool_mapping,
            self.env.kv_scale_orig_quant,
            self.env.kv_scale_quant_orig,
            None,  # kv_cache_scale_orig_quant
            None,  # out_scale
            None,  # block_ids_per_seq
            [None, None],  # helix_tensor_params
            self.p,
            0,  # layer_idx
            self.env.num_heads,
            1,  # num_kv_heads
            GEN_HEAD_SIZE,
            0,  # residual_dim
            self.env.tokens_per_block,
            MAX_SEQ_LEN,  # attention_window_size
            1,  # beam_width
            QUANT_MODE_FP8_KV_CACHE,
            self.q_scaling,
            Q_LORA_RANK_R1,
            KV_LORA_RANK,
            QK_NOPE_HEAD_DIM,
            QK_ROPE_HEAD_DIM,
            V_HEAD_DIM,
            True,  # rope_append
        )
        torch.cuda.synchronize()

    def assert_rows_at_sequence_length_slots(self) -> None:
        """Every appended row is e4m3([compressed_kv | rope(k_pe)]) at the slot
        `sequence_length - P + t` addresses."""
        for r in range(self.rows):
            rid = self.gen_ids[r // self.p]
            row = self.env.cache_row(rid, self.positions[r])
            _assert_bytes_equal(
                row[:KV_LORA_RANK],
                self.env.quantize(self.latent_cache[r, :KV_LORA_RANK]),
                f"appended compressed_kv (request {rid}, row {r})",
            )
            _assert_e4m3_roped(
                row[KV_LORA_RANK:],
                self.env.quantize(
                    self.env.rope_ref(self.latent_cache[r, KV_LORA_RANK:], self.positions[r])
                ),
                f"appended k_pe (request {rid}, row {r})",
            )


def test_fp8_kv_mtp_production_sweep() -> None:
    """The production MTP generation call at predicted_tokens_per_seq 1, 2, 3
    and 4 — 1 as the regression check that the single-token path did not move,
    2-4 because a target sweeping max_draft_len over 1/2/3 passes
    max_draft_len + 1. Both KV scale tensors omitted (what the engine's own
    MLA call site passes), DeepSeek-R1's YaRN q_scaling, q_lora_rank 1536,
    q_pe a packed-q strided view. The three cached lengths put the P rows of
    each sequence at a different 32-slot alignment: ending on a page's last
    slot, starting a fresh page, and straddling the boundary. Every case runs
    twice to pin that a repeated call is idempotent at P > 1 too."""
    for p in (1, 2, 3, 4):
        torch.manual_seed(20 + p)
        env = _fp8_env()
        try:
            cached = [PAGE32 - p, PAGE32, 2 * PAGE32 - 1]
            env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=cached)
            _run_and_check_fp8(
                env,
                request_ids=[0, 1, 2],
                seq_lens=[p] * 3,
                num_contexts=0,
                cached_lens=cached,
                q_pe_contiguous=False,
                q_scaling=Q_SCALING_R1,
                q_lora_rank=Q_LORA_RANK_R1,
                repeats=2,
                predicted_tokens_per_seq=p,
            )
        finally:
            env.shutdown()


def test_fp8_kv_mtp_page32_alignments() -> None:
    """P = 2, 3 and 4 with the P rows of a sequence placed at every alignment
    a 32-slot page allows: entirely inside a page, ending on its last slot,
    straddling the boundary after each of the 1..P-1 possible splits, and
    starting a fresh page at slot 0. A straddling generation sequence is new
    at P > 1 — at P = 1 one call could only ever write one slot per
    sequence."""
    for p in (2, 3, 4):
        torch.manual_seed(30 + p)
        env = _fp8_env()
        try:
            # First-token position of sequence k, walking the boundary past
            # the whole P-row block.
            cached = [PAGE32 - p - 1 + k for k in range(p + 2)]
            rids = list(range(len(cached)))
            env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
            _run_and_check_fp8(
                env,
                request_ids=rids,
                seq_lens=[p] * len(rids),
                num_contexts=0,
                cached_lens=cached,
                q_pe_contiguous=True,
                q_scaling=Q_SCALING_R1,
                predicted_tokens_per_seq=p,
            )
        finally:
            env.shutdown()


def test_fp8_kv_mtp_mixed_batch_skips_context() -> None:
    """Context sequence leading a batch whose generation sequences each carry
    P = 3 tokens: the op consumes only the generation rows and indexes the
    length and block tensors from num_contexts. Both generation sequences
    straddle a page boundary."""
    torch.manual_seed(34)
    env = _fp8_env(orig_quant=1.0, quant_orig=1.0)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[40, 30, 63])
        _run_and_check_fp8(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[40, 3, 3],
            num_contexts=1,
            cached_lens=[0, 30, 63],
            q_pe_contiguous=True,
            q_scaling=Q_SCALING_R1,
            predicted_tokens_per_seq=3,
        )
    finally:
        env.shutdown()


def test_fp8_kv_mtp_scale_factor() -> None:
    """A calibrated fp8 checkpoint's reciprocal scale pair at P = 3, with
    q_scaling 1.0: the write-side factor applies to all P appended rows and
    all P quantized query rows, and the two decode-FMHA scales stay per-batch
    scalars that P does not enter."""
    torch.manual_seed(35)
    env = _fp8_env(orig_quant=1.0 / 1.5, quant_orig=1.5)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1], token_nums=[30, 95])
        _run_and_check_fp8(
            env,
            request_ids=[0, 1],
            seq_lens=[3, 3],
            num_contexts=0,
            cached_lens=[30, 95],
            q_pe_contiguous=True,
            q_scaling=1.0,
            predicted_tokens_per_seq=3,
        )
    finally:
        env.shutdown()


def test_fp8_kv_mtp_large_decode_batch_multi_step() -> None:
    """64-sequence fp8 MTP batch at P = 2 (16 384 quantized q rows and 128
    appended cache rows per call) over two consecutive steps, at the
    production scale. The cached lengths spread the rows over 25 pages; two
    sequences straddle a page boundary inside a single call at step 0, and two
    start a fresh page."""
    torch.manual_seed(36)
    env = _fp8_env(max_batch_size=64)
    try:
        rids = list(range(64))
        cached = [(37 * (i + 1)) % 800 + 1 for i in range(64)]
        assert [i for i in range(64) if cached[i] % PAGE32 == PAGE32 - 1] == [5, 37]
        assert [i for i in range(64) if cached[i] % PAGE32 == 0] == [18, 50]
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=cached)
        for step in range(2):
            _run_and_check_fp8(
                env,
                request_ids=rids,
                seq_lens=[2] * 64,
                num_contexts=0,
                cached_lens=[c + 2 * step for c in cached],
                q_pe_contiguous=True,
                q_scaling=Q_SCALING_R1,
                predicted_tokens_per_seq=2,
            )
    finally:
        env.shutdown()


def test_fp8_kv_mtp_position_controls() -> None:
    """P = 4 at the R1 cell, plus the wrong-position mirrors that measure what
    the per-row rope gate discriminates. The failure mode they guard is the
    one no shape check catches: every one of the P rows roped at a single
    position. Measured on sm_100 at this case (positions 29-32 and 40-43),
    against the appended k_pe half of the named row — 64 e4m3 bytes, gated at
    one e4m3 ulp with an allowance of 1 byte:

      row  mirror                                  bytes differ   max rel dev
      0    roped at the sequence's last position    35/64 (54.7%)   13.0 (104x)
      0    roped at the next row's position         21/64 (32.8%)   11.3  (90x)
      0    the next row's latent at row 0's pos     62/64 (96.9%)   1.4e9
      3    roped at the sequence's first position   30/64 (46.9%)   40.1 (321x)

    with the multiplier against E4M3_ULP_RTOL. The weakest control is the
    off-by-one position — at these positions the low-frequency rope pairs
    barely move, so two thirds of the e4m3 bytes are genuinely unchanged —
    and it is still 21x past the fraction allowance and 90x past the
    tolerance. (The row-mixup mirror's relative deviation is degenerate
    rather than informative: one of its elements quantizes to zero, so the
    ratio is bounded only by the reference's 1e-9 clamp. Its byte fraction is
    the meaningful half.) The single-position collapse is checked from both
    ends, because a bug pinning all P positions to L - 1 is invisible on the
    last row and one pinning them to L - P is invisible on the first."""
    torch.manual_seed(37)
    env = _fp8_env(orig_quant=1.0, quant_orig=1.0)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1], token_nums=[29, 40])
        run = _run_and_check_fp8(
            env,
            request_ids=[0, 1],
            seq_lens=[4, 4],
            num_contexts=0,
            cached_lens=[29, 40],
            q_pe_contiguous=True,
            q_scaling=Q_SCALING_R1,
            predicted_tokens_per_seq=4,
        )
        p = run.predicted_tokens_per_seq
        latent, positions = run.latent_cache, run.positions
        controls = [
            (0, "roped at the sequence's last position", positions[p - 1], 0),
            (0, "roped at the next row's position", positions[1], 0),
            (0, "the next row's latent at row 0's position", positions[0], 1),
            (p - 1, "roped at the sequence's first position", positions[0], p - 1),
        ]
        for row, name, mirror_pos, src_row in controls:
            got = env.cache_row(run.gen_ids[0], positions[row])[KV_LORA_RANK:]
            mirror = env.quantize(env.rope_ref(latent[src_row, KV_LORA_RANK:], mirror_pos))
            frac = float((got.view(torch.uint8) != mirror.view(torch.uint8)).float().mean())
            rel = float(
                ((got.float() - mirror.float()).abs() / mirror.float().abs().clamp(min=1e-9)).max()
            )
            assert frac > 100 * E4M3_MAX_INEXACT_FRACTION, (
                f"control '{name}' (row {row}) moved only {frac:.3f} of bytes — "
                "the bit-exact-fraction floor would not have caught it"
            )
            assert rel > 4 * E4M3_ULP_RTOL, (
                f"control '{name}' (row {row}) stayed within {rel:.3g} relative — "
                "the one-ulp tolerance would not have caught it"
            )
        # The same collapse in quant_q_buffer: row 0's roped tail against the
        # sequence's last position. 50.3% of its 8192 bytes moved (measured);
        # the floor is set well below that because the fraction is a property
        # of how far apart two rope angles are, not of the gate.
        quant_tail = run.quant_q_buffer[0, :, KV_LORA_RANK:].view(torch.float8_e4m3fn)
        mirror_q = env.quantize(env.rope_ref(run.q_pe[0], positions[p - 1]))
        frac = float((quant_tail.view(torch.uint8) != mirror_q.view(torch.uint8)).float().mean())
        assert frac > 0.2, f"quant_q single-position control moved {frac:.3f} of bytes"
    finally:
        env.shutdown()


def test_fp8_kv_mtp_position_basis_is_sequence_length() -> None:
    """Which length tensor supplies the P consecutive positions: the device
    `sequence_length`, not its host twin. The same prepared P = 3 step is
    issued twice — once as prepared, once with host_past_key_value_lengths
    perturbed to sequence_length - 1 — and both land the rows at the
    sequence_length-derived slots with bitwise identical pool, quant_q_buffer
    and scheduler buffers."""
    torch.manual_seed(38)
    env = _fp8_env()
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1], token_nums=[40, 31])
        step = _Fp8Step(env, [0, 1], [40, 31], predicted_tokens_per_seq=3)
        pool = env.pool_tensor()

        pool.zero_()
        step.call()
        step.assert_rows_at_sequence_length_slots()
        pool_ref = pool.view(torch.uint8).clone()
        quant_ref = step.quant_q_buffer.clone()
        cu_kv_ref = step.cu_kv_seqlens.clone()

        perturbed = step.metadata.kv_lens_runtime.clone() - 1
        pool.zero_()
        step.call(host_past=perturbed)
        step.assert_rows_at_sequence_length_slots()
        torch.testing.assert_close(pool.view(torch.uint8), pool_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(step.quant_q_buffer, quant_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(step.cu_kv_seqlens, cu_kv_ref, rtol=0.0, atol=0.0)
    finally:
        env.shutdown()


def test_fp8_kv_mtp_append_race_control() -> None:
    """Harness-blindness control for every pool comparison above.

    The documented defect this surface neighbours: when one call's new tokens
    do not map to distinct physical slots, the paged append's writes race and
    leave torn cache rows, nondeterministically and with no error. At P > 1 a
    single call writes P rows per sequence, so that arming sequence is
    reachable here — this replays it by aliasing every block-offset entry onto
    one physical page, so all 8 sequences' rows contend for the same two
    slots. It fires (6 of 6 identical armed calls left different pool images
    on sm_100, 380-1140 bytes apart), and the certified geometry repeated in
    the same process reproduces bitwise. Without the armed half, a clean pool
    comparison would be indistinguishable from a comparison that cannot see
    an append at all."""
    torch.manual_seed(39)
    env = _fp8_env(max_batch_size=8)
    try:
        rids = list(range(8))
        env.kv_cache_manager.add_dummy_requests(rids, token_nums=[40] * 8)
        step = _Fp8Step(env, rids, [40] * 8, predicted_tokens_per_seq=2)
        pool = env.pool_tensor()
        honest = step.metadata.kv_cache_block_offsets
        assert honest is not None
        aliased = torch.full_like(honest, int(honest[0, 0, 0, 0]))

        armed = []
        for _ in range(4):
            pool.zero_()
            step.call(block_offsets=aliased)
            armed.append(pool.view(torch.uint8).clone())
        assert any(not torch.equal(armed[0], a) for a in armed[1:]), (
            "the armed aliased-slot call reproduced bitwise across 4 runs — "
            "this harness cannot see the append race it is meant to detect"
        )

        certified = []
        for _ in range(4):
            pool.zero_()
            step.call()
            certified.append(pool.view(torch.uint8).clone())
        for run in certified[1:]:
            torch.testing.assert_close(run, certified[0], rtol=0.0, atol=0.0)
        step.assert_rows_at_sequence_length_slots()
    finally:
        env.shutdown()


def test_bf16_mtp_decode_batch() -> None:
    """bf16 latent pool at P = 2 and 3 over the engine-default page size: the
    roped q lands in fused_q's tail at each row's own position, and each
    sequence's P cache rows land at consecutive slots — one sequence per case
    inside a page, one straddling the boundary. The last case leads with a
    context sequence."""
    for p, q_pe_contiguous in ((2, True), (3, False)):
        torch.manual_seed(40 + p)
        env = _MlaEnv(tokens_per_block=PAGE32)
        try:
            cached = [PAGE32 - p, PAGE32 - 1, 2 * PAGE32 - 1]
            env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=cached)
            _run_and_check(
                env,
                request_ids=[0, 1, 2],
                seq_lens=[p] * 3,
                num_contexts=0,
                cached_lens=cached,
                q_pe_contiguous=q_pe_contiguous,
                predicted_tokens_per_seq=p,
            )
        finally:
            env.shutdown()
    torch.manual_seed(44)
    env = _MlaEnv(tokens_per_block=PAGE32)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[40, 31, 63])
        _run_and_check(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[40, 2, 2],
            num_contexts=1,
            cached_lens=[0, 31, 63],
            q_pe_contiguous=True,
            predicted_tokens_per_seq=2,
        )
    finally:
        env.shutdown()
