# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's mla_rope_generation entry.

The op reads paged-KV-cache addressing tensors and per-sequence length
tensors that the runtime normally derives from a KVCacheManager and a
prepared TrtllmAttentionMetadata. The test builds that state for real -- an
MLA (SELFKONLY, kv_factor=1) KVCacheManager over an fp8-e4m3 latent pool and
a prepared TrtllmAttentionMetadata -- then checks the kernel effects against
a torch fp32 reference.

One surface: the fp8 pool (`quant_mode=128`) at GLM-5.2's TP4 geometry,
H = 16, C/R/nope = 512/64/192, page 32, q_scaling 1, at P = 1 and the MTP
width P = 2. Over this pool the op leaves fused_q untouched and writes the
e4m3 query into `quant_q_buffer`, the two folded FMHA scales, the scheduler
buffers, and one e4m3 cache row per generation token.
"""

import inspect
import math
import zlib
from typing import List, NamedTuple, Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.mla_rope_generation import (  # noqa: E501
    mla_rope_generation as op,
)
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "mla_rope_generation requires a CUDA device"

# GLM-5.2 MLA geometry at TP4: 64 heads over 4 ranks, a 512/64 latent and a
# 192-wide nope half. V_HEAD_DIM is the engine's value (kv_lora_rank), which is
# what the target passes; the checkpoint's 256 never reaches this op.
NUM_HEADS = 16
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
Q_LORA_RANK = 2048
V_HEAD_DIM = KV_LORA_RANK
GEN_HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
# GLM-5.2's rope base. Which base the table was built at is visible in every
# roped output, so the test builds it at the checkpoint's value.
ROPE_THETA = 8_000_000.0
TOKENS_PER_BLOCK = 32
MAX_SEQ_LEN = 1024
QUANT_MODE_FP8_KV_CACHE = 128  # QuantMode.FP8_KV_CACHE

# The op derives mla_bmm1_scale in fp32 on the device; the reference computes
# the same expression in double and rounds once. Every way of getting this
# scale wrong misses by a factor, not by ulps, so torch's fp32 rtol with
# atol 0 is the gate.
BMM_SCALE_RTOL = 1.3e-6

# One e4m3 ulp, for the parts of the fp8 output that go through the in-kernel
# RoPE, paired with a bit-exact-fraction floor. See `_compare`. e4m3 carries 3
# explicit mantissa bits, so a round-to-nearest lands within 2**-3 relative.
E4M3_ULP_RTOL = 2**-3
# How much of a quantized output may miss bit-exactness before the gate calls it
# drift rather than rounding.
E4M3_MAX_INEXACT_FRACTION = 1e-3


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Half an e4m3 ulp, *and* a floor on how much may miss bit-exactness.

    The rotation is where kernel and reference can disagree: both evaluate
    `x*cos -+ y*sin` and may do it in different orders. e4m3's coarse
    rounding absorbs that in practice -- every quantized element measured at
    the 16-head GLM geometry on sm_100 came back bit-exact, so neither half
    of this gate has been approached.

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
        fp8_pool: bool = True,
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
            theta=ROPE_THETA,
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
        # Driving the guard over every production call is what proves it admits
        # what it is supposed to admit.
        with validating(op):
            op(
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


def test_fp8_kv_mixed_batch_skips_context() -> None:
    """A context sequence leading the batch: the op consumes only the
    generation tokens and indexes length and block tensors from num_contexts.
    The generation rows sit at a fresh page (position 96) and mid first page
    (position 7)."""
    torch.manual_seed(14)
    env = _MlaEnv(max_batch_size=4)
    try:
        env.kv_cache_manager.add_dummy_requests([0, 1, 2], token_nums=[40, 96, 7])
        _run_and_check_fp8(
            env,
            request_ids=[0, 1, 2],
            seq_lens=[40, 1, 1],
            num_contexts=1,
            cached_lens=[0, 96, 7],
            q_pe_contiguous=True,
            q_scaling=1.0,
            q_lora_rank=Q_LORA_RANK,
        )
    finally:
        env.shutdown()


def test_fp8_kv_mtp_page_straddles() -> None:
    """P = 2 at every alignment against a 32-token page: the two rows of one
    sequence either share a page or straddle into a fresh one."""
    torch.manual_seed(15)
    env = _MlaEnv(max_batch_size=4)
    try:
        cached = [30, 31, 32, 33]
        env.kv_cache_manager.add_dummy_requests([0, 1, 2, 3], token_nums=cached)
        _run_and_check_fp8(
            env,
            request_ids=[0, 1, 2, 3],
            seq_lens=[2, 2, 2, 2],
            num_contexts=0,
            cached_lens=cached,
            q_pe_contiguous=False,
            q_scaling=1.0,
            q_lora_rank=Q_LORA_RANK,
            predicted_tokens_per_seq=2,
            repeats=2,
        )
    finally:
        env.shutdown()


def _guard_args(**overrides: object) -> dict:
    """Every `is_valid` parameter, inert, with the named ones replaced.

    Read off the signature rather than written out, so a parameter inserted
    into the op cannot shift a hand-counted positional list.
    """
    args = dict.fromkeys(inspect.signature(op.is_valid).parameters)
    args.update(residual_dim=0, quant_mode=0)
    args.update(overrides)
    return args


def test_is_valid_refuses_a_non_reciprocal_scale_pair() -> None:
    """A non-reciprocal pair leaves the consuming FMHA undoing a write scale
    that was never applied, and every buffer this op writes still looks
    plausible."""
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


def test_is_valid_refuses_the_three_silent_configurations() -> None:
    """Each is accepted by the op and answered wrongly, or -- in
    quant_q_buffer's case -- answered with a dead process."""
    with pytest.raises(AssertionError, match="quant_q_buffer"):
        op.is_valid(**_guard_args(quant_mode=QUANT_MODE_FP8_KV_CACHE, quant_q_buffer=None))
    op.is_valid(
        **_guard_args(
            quant_mode=QUANT_MODE_FP8_KV_CACHE,
            quant_q_buffer=torch.empty(1, dtype=torch.uint8, device="cuda"),
        )
    )
    with pytest.raises(AssertionError, match="residual_dim"):
        op.is_valid(**_guard_args(residual_dim=QK_ROPE_HEAD_DIM))
    with pytest.raises(AssertionError, match="kv_norm_weight"):
        op.is_valid(
            **_guard_args(
                kv_norm_weight=torch.ones(KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
            )
        )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Drive every configuration the entry claims, through the entry's own
    reference and this file's gate.

    Each cell cross-checks `op.reference` against this file's `env.rope_ref` +
    `env.quantize` before either gates the kernel: two references disagreeing
    is a different failure from the kernel moving.
    """
    spec = cell.spec
    # The cell's geometry is this file's geometry; a drift would test one
    # shape and certify another.
    assert (
        spec["num_heads"],
        spec["kv_lora_rank"],
        spec["qk_rope_head_dim"],
        spec["qk_nope_head_dim"],
        spec["v_head_dim"],
    ) == (NUM_HEADS, KV_LORA_RANK, QK_ROPE_HEAD_DIM, QK_NOPE_HEAD_DIM, V_HEAD_DIM)
    p = spec["predicted_tokens_per_seq"]
    num_seqs = spec["num_seqs"]
    torch.manual_seed(zlib.crc32(cell.why.encode()) % 2**31)

    env = _MlaEnv(
        max_batch_size=max(4, num_seqs),
        tokens_per_block=spec["tokens_per_block"],
        num_heads=spec["num_heads"],
        fp8_pool=True,
        orig_quant=spec["write_scale"],
        quant_orig=spec["read_scale"],
    )
    try:
        # Cached lengths straddle page boundaries: the first two sit at a fresh
        # page and at a page's last slot, the rest walk through alignments.
        cached = [32, spec["tokens_per_block"] - 1] + [31 + 29 * i for i in range(2, num_seqs)]
        cached = cached[:num_seqs]
        request_ids = list(range(num_seqs))
        env.kv_cache_manager.add_dummy_requests(request_ids, token_nums=cached)

        run = _run_and_check_fp8(
            env,
            request_ids=request_ids,
            seq_lens=[p] * num_seqs,
            num_contexts=0,
            cached_lens=cached,
            q_pe_contiguous=False,
            q_scaling=spec["q_scaling"],
            q_lora_rank=spec["q_lora_rank"],
            predicted_tokens_per_seq=p,
        )

        # 1. The entry's reference against this file's, on the appended row.
        half = QK_ROPE_HEAD_DIM // 2
        table = env.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        for r in range(run.latent_cache.shape[0]):
            pos = run.positions[r]
            entry_ref = op.reference(
                run.latent_cache[r],
                table[pos, :half, 0],
                table[pos, :half, 1],
                QK_ROPE_HEAD_DIM,
                env.kv_scale_orig_quant,
            )
            local_ref = torch.cat(
                [
                    run.latent_cache[r, :KV_LORA_RANK],
                    env.rope_ref(run.latent_cache[r, KV_LORA_RANK:], pos),
                ]
            )
            _assert_bytes_equal(entry_ref, env.quantize(local_ref), f"references disagree, row {r}")

        # 2. The entry's reference against the kernel, through `_compare`.
        quant_q = run.quant_q_buffer.view(torch.float8_e4m3fn)
        for r in range(run.fused_q.shape[0]):
            pos = run.positions[r]
            row = torch.cat([run.fused_q[r, :, :KV_LORA_RANK], run.q_pe[r]], dim=-1)
            _compare(
                quant_q[r],
                op.reference(
                    row,
                    table[pos, :half, 0],
                    table[pos, :half, 1],
                    QK_ROPE_HEAD_DIM,
                    env.kv_scale_orig_quant,
                ),
            )

        # 3. The two scalar outputs the entry also states.
        want1, want2 = op.reference_bmm_scales(
            env.kv_scale_quant_orig, spec["q_scaling"], QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM
        )
        kv_lens = [c + p for c in cached]
        want_cu_q, want_cu_kv = op.reference_scheduler_buffers(kv_lens, spec["num_heads"], p)
        assert want_cu_q.tolist() == [i * spec["num_heads"] * p for i in range(num_seqs + 1)]
        assert want_cu_kv.tolist()[-1] == sum(kv_lens)
        assert want2.item() == pytest.approx(
            1.0 if spec["read_scale"] is None else spec["read_scale"]
        )
        assert want1[1].item() == pytest.approx(want1[0].item() * math.log2(math.e), rel=1e-6)
    finally:
        env.shutdown()
