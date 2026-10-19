# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the sliding-window decode entries at a tep4 rank's shape: 32 query / 4 KV heads.

``paged_decode_attention`` (``swa_paged_decode``) and ``swa_block_attention`` (``swa_block``)
take the head counts as compile-time constants (8 query heads per KV head); the tep8 op tests
certify them at 16 / 2. At 32 / 4, what is gated here:

* ``paged_decode_attention``'s cache write is the built-in append byte for byte
  (``applyBiasRopeUpdateKVCacheV2`` through ``thop.trtllm_gen_generation_preprocess``);
* its output against an fp64 reference, compared with the built-in trtllm-gen path at the same
  shape: with BF16 Q no farther in max and mean-abs (<= 1.0x); with ``q_fp8`` (the served build,
  the built-in's own E4M3 Q) mean-abs within 1.10x the built-in's at every context and max-abs
  within 1.10x where a whole window of keys is attended -- the tep8 test's policy;
* ``swa_block_attention`` is bit for bit the chain it replaces at 32 / 4 (``preattn_prep`` ->
  ``paged_decode_attention`` -> the built-in norm-gate kernel): the cache bytes and the output,
  with and without the fused prep and gate, checkpoint order and permuted, across window and
  page boundaries, and through one captured graph replayed at other contexts.

``MODELING_V2_TIMING=1`` also runs the microbenchmark (printed, never gated).
"""

import math
import os
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.paged_decode_attention import (  # noqa: E501
    HEAD_DIM,
    paged_decode_attention,
    window_splits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    CKPT_EPS,
    CKPT_Q_SCALE,
    preattn_prep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.swa_block import (  # noqa: E501
    swa_block_attention,
)
from tensorrt_llm._torch.models import modeling_musespark_open as builtin_model
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.quantization.mode import QuantMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the target's decode entries are certified on sm_103",
)

DEV = "cuda"
FP8 = torch.float8_e4m3fn
#: One tep4 rank: 128 / 4 query heads, 16 / 4 KV heads.
H, G = 32, 4
Q_WIDTH, KV_WIDTH = H * HEAD_DIM, G * HEAD_DIM
PROJ_WIDTH = 2 * (Q_WIDTH + KV_WIDTH)
WINDOW = 2048
PAGE = 32
Q_SCALE = 5.473
GATE_EPS = 1e-5
_CACHE = {}


def _sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _rope_table() -> torch.Tensor:
    """The checkpoint's RoPE table, as a tep4 rank's built-in sliding-window attention holds it."""
    if "table" not in _CACHE:
        from tensorrt_llm._torch.configs.musespark_open import MuseSparkOpenConfig
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.mapping import Mapping

        config = MuseSparkOpenConfig(
            num_attention_heads=H,
            num_key_value_heads=G,
            moe={"topk": 8, "num_experts": 256, "override_routed_hidden_dim": 4096},
        )
        model_config = ModelConfig(
            pretrained_config=config, mapping=Mapping(world_size=1, tp_size=1, rank=0)
        )
        with torch.device(DEV):
            attention = builtin_model.MuseSparkOpenAttention(model_config, 0)
        assert attention.uses_rope and attention.is_sliding
        assert attention.num_heads == H and attention.num_key_value_heads == G
        _CACHE["table"] = attention.rotary_emb.rotary_cos_sin
    return _CACHE["table"]


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8 if t.element_size() == 1 else torch.int16)


def _assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype, (what, got.shape, want.shape)
    diff = _bits(got) != _bits(want)
    if diff.any():
        first = diff.nonzero()[0].tolist()
        raise AssertionError(f"{what}: {int(diff.sum())} of {diff.numel()} differ, first {first}")


# --------------------------------------------------------------------------
# paged_decode_attention against the built-in path
# --------------------------------------------------------------------------
class _Step:
    """One decode step: a fused-QKV operand ``[q | k | v]`` at 32 / 4 heads and an FP8 pool in
    the TRT-LLM paged layout (logical page ``i`` is K page ``2 perm[i]``, V page
    ``2 perm[i] + 1``)."""

    def __init__(self, rows, ctx, *, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx = rows, ctx
        self.splits = window_splits(WINDOW)
        pages = -(-ctx // PAGE)
        spare = pages + 4
        perm = torch.randperm(spare, generator=g, device=DEV)[:pages].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = torch.randn(2 * spare, G, PAGE, HEAD_DIM, generator=g, device=DEV).to(FP8)
        q = torch.randn(rows, Q_WIDTH, generator=g, device=DEV) * Q_SCALE
        kv = torch.randn(rows, 2 * KV_WIDTH, generator=g, device=DEV)
        self.qkv = torch.cat((q, kv), dim=1).to(torch.bfloat16)
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.scale = torch.tensor([1.0], dtype=torch.float32, device=DEV)
        self.inv_scale = 1.0 / self.scale

    def args(self, pool, q_fp8=False, pdl=False):
        return (self.qkv, pool, self.table, self.seq_lens, self.scale, H, G, WINDOW,
                self.splits, q_fp8, pdl)  # fmt: skip


_BUILTIN_BUFFERS = {}


def _builtin_step(step: _Step, pool: torch.Tensor, fmha: bool = True):
    """The built-in sliding-window decode step: thop's generation preprocess (the K/V append,
    Q to E4M3), then flashinfer's trtllm-gen decode with the window."""
    import flashinfer.decode as fdec

    if not _BUILTIN_BUFFERS:
        _BUILTIN_BUFFERS["workspace"] = torch.zeros(64 << 20, dtype=torch.uint8, device=DEV)
        _BUILTIN_BUFFERS["counter"] = torch.zeros(4096, dtype=torch.int32, device=DEV)
    workspace, counter = _BUILTIN_BUFFERS["workspace"], _BUILTIN_BUFFERS["counter"]
    rows, ctx = step.rows, step.ctx
    (q_processed, kv_pool, block_tables, _, bmm1_scale, bmm2_scale, fmha_workspace, _, _,
     max_kv_len, window_left, _) = thop.trtllm_gen_generation_preprocess(
        step.qkv, workspace, step.seq_lens, None, None, step.table.view(1, 1, 2, -1),
        torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64),
        torch.tensor([[0, 0]], dtype=torch.int32), step.scale, step.inv_scale, None, None, None,
        None, 0, 0, H, G, HEAD_DIM, PAGE, int(QuantMode.FP8_KV_CACHE), WINDOW, WINDOW, rows,
        1, rows, ctx, 0, 10000.0, 0, 1.0, 1024, 0, 1.0 / math.sqrt(HEAD_DIM), 1.0, True, rows, 0,
        _sm_count(), pool.shape[0], 2, True, False)  # fmt: skip
    if not fmha:
        return None
    q8 = q_processed.view(torch.uint8).flatten()[: rows * Q_WIDTH].view(FP8).view(rows, H, HEAD_DIM)
    out = torch.empty(rows, Q_WIDTH, dtype=torch.bfloat16, device=DEV)
    fdec.trtllm_batch_decode_with_kv_cache(
        query=q8,
        kv_cache=(kv_pool, kv_pool),
        workspace_buffer=fmha_workspace,
        block_tables=block_tables,
        seq_lens=step.seq_lens,
        max_seq_len=max_kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=window_left,
        out=out.view(rows, H, HEAD_DIM),
        kv_layout="HND",
        enable_pdl=False,
        backend="trtllm-gen",
        q_len_per_req=rows,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=bmm1_scale.narrow(0, 1, 1),
        multi_ctas_kv_counter_buffer=counter,
    )
    return out


def _fp64_reference(step: _Step, pool: torch.Tensor) -> torch.Tensor:
    """fp64 windowed attention of BF16 Q over the FP8 K/V the cache holds after the step."""
    rows, ctx = step.rows, step.ctx
    keys = torch.arange(ctx, device=DEV)
    raw = _bytes(pool)

    def gather(kv):
        return raw[step.table[kv, keys // PAGE].long(), :, keys % PAGE].view(FP8).double()

    k, v = gather(0), gather(1)
    q = step.qkv[:, :Q_WIDTH].double().view(rows, H, HEAD_DIM)
    out = torch.empty(rows, H, HEAD_DIM, dtype=torch.float64, device=DEV)
    group = H // G
    for t in range(rows):
        pos = ctx - rows + t
        lo = max(0, pos - WINDOW + 1)
        for h in range(G):
            s = q[t, h * group : (h + 1) * group] @ k[lo : pos + 1, h].T / math.sqrt(HEAD_DIM)
            out[t, h * group : (h + 1) * group] = torch.softmax(s, -1) @ v[lo : pos + 1, h]
    return out.reshape(rows, Q_WIDTH)


def _errors(out, ref):
    d = (out.double() - ref).abs()
    return float(d.max()), float(d.mean())


_CASES = [
    (8, 100000),
    (8, 131072),
    (8, 2100),
    (8, 2056),
    (8, 1000),
    (3, 100003),
    (1, 100000),
    (8, 40),
    (3, 3),
    (1, 1),
]


@pytest.mark.parametrize("rows, ctx", _CASES)
def test_paged_decode_cache_write_is_the_builtin_append(rows, ctx) -> None:
    step = _Step(rows, ctx, seed=rows * 7919 + ctx)
    builtin_pool = step.pool.clone()
    _builtin_step(step, builtin_pool, fmha=False)
    paged_decode_attention(*step.args(step.pool))
    assert torch.equal(_bytes(step.pool), _bytes(builtin_pool)), (rows, ctx)


@pytest.mark.parametrize("rows, ctx", _CASES)
def test_paged_decode_is_no_farther_from_fp64_than_the_builtin(rows, ctx) -> None:
    """Tolerances: BF16 Q <= 1.0x the built-in's max and mean-abs error; ``q_fp8`` (the served
    build) mean-abs <= 1.10x, and max-abs <= 1.10x where at least a window of keys is
    attended (with fewer keys the max is one element of a softmax over a handful of keys)."""
    step = _Step(rows, ctx, seed=ctx + rows)
    builtin_pool = step.pool.clone()
    builtin = _builtin_step(step, builtin_pool)
    ref = _fp64_reference(step, builtin_pool)
    b_max, b_mean = _errors(builtin, ref)
    for q_fp8 in (False, True):
        out = paged_decode_attention(*step.args(step.pool.clone(), q_fp8=q_fp8))
        o_max, o_mean = _errors(out, ref)
        print(
            f"\ntep4 rows {rows} ctx {ctx} q_fp8 {q_fp8}: op max {o_max:.4g} mean {o_mean:.3g}; "
            f"builtin max {b_max:.4g} mean {b_mean:.3g}"
        )
        if not q_fp8:
            assert o_max <= b_max and o_mean <= b_mean, (o_max, b_max, o_mean, b_mean)
        else:
            assert o_mean <= 1.10 * b_mean, (o_mean, b_mean)
            if ctx >= WINDOW:
                assert o_max <= 1.10 * b_max, (o_max, b_max)


# --------------------------------------------------------------------------
# swa_block_attention: bit for bit the chain at 32 / 4 heads
# --------------------------------------------------------------------------
def _permutation() -> torch.Tensor:
    """Checkpoint column order -> low-latency layout ``[q | k | v | gate]``."""
    interleaved = torch.arange(2 * Q_WIDTH, device=DEV).reshape(H, 2 * HEAD_DIM)
    return torch.cat(
        (
            interleaved[:, :HEAD_DIM].reshape(-1),
            torch.arange(2 * Q_WIDTH, PROJ_WIDTH, device=DEV),
            interleaved[:, HEAD_DIM:].reshape(-1),
        )
    )


class _BlockStep:
    """One sliding-window decode step at 32 / 4 heads: the raw BF16 projection (each head at
    its own log-normal scale), positions, an FP8 paged pool."""

    def __init__(self, rows, ctx, *, seed=0, permuted=False):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx, self.permuted = rows, ctx, permuted
        self.splits = window_splits(WINDOW)
        pages = -(-ctx // PAGE)
        spare = pages + 4
        perm = torch.randperm(spare, generator=g, device=DEV)[:pages].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = torch.randn(2 * spare, G, PAGE, HEAD_DIM, generator=g, device=DEV).to(FP8)
        heads = PROJ_WIDTH // HEAD_DIM
        x = torch.randn(rows, heads, HEAD_DIM, generator=g, device=DEV)
        x = x * torch.exp(2.0 * torch.randn(rows, heads, 1, generator=g, device=DEV))
        proj = x.reshape(rows, PROJ_WIDTH).to(torch.bfloat16)
        self.proj = proj[:, _permutation()].contiguous() if permuted else proj
        self.positions = torch.arange(ctx - rows, ctx, device=DEV, dtype=torch.int32).reshape(1, -1)
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.scale = torch.tensor([1.0], dtype=torch.float32, device=DEV)

    def raw_gate(self):
        if self.permuted:
            return self.proj[:, Q_WIDTH + 2 * KV_WIDTH :].view(self.rows, H, HEAD_DIM)
        return self.proj[:, : 2 * Q_WIDTH].view(self.rows, H, 2 * HEAD_DIM)[..., HEAD_DIM:]


def _builtin_norm_gate(attention: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The launch ``MuseSparkOpenAttention.apply_output_gate`` makes at decode row counts."""
    out = torch.empty_like(attention)
    builtin_model._attention_norm_gate_kernel[(attention.shape[0] * H,)](
        out, attention, gate, attention.stride(0), gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else HEAD_DIM, GATE_EPS, HEADS=H, HEAD_DIM=HEAD_DIM,
        num_warps=4,
    )  # fmt: skip
    return out


def _chain(step, pool, q_fp8, gate_on, pdl=False):
    fused, gate = preattn_prep(
        step.proj, _rope_table(), step.positions, H, G, CKPT_EPS, CKPT_Q_SCALE, step.permuted, pdl
    )
    out = paged_decode_attention(
        fused, pool, step.table, step.seq_lens, step.scale, H, G, WINDOW, step.splits, q_fp8, pdl
    )
    return fused, (_builtin_norm_gate(out, gate) if gate_on else out)


def _block(step, pool, q_fp8, fused_prep, gate_on, fused=None, pdl=False):
    return swa_block_attention(
        step.proj if fused_prep else fused,
        pool,
        step.table,
        step.seq_lens,
        step.scale,
        H,
        G,
        WINDOW,
        step.splits,
        q_fp8,
        fused_prep=fused_prep,
        cos_sin=_rope_table() if fused_prep else None,
        position_ids=step.positions if fused_prep else None,
        prep_eps=CKPT_EPS,
        q_scale=CKPT_Q_SCALE,
        qkv_permuted=step.permuted,
        gate=step.raw_gate() if gate_on else None,
        gate_eps=GATE_EPS,
        enable_pdl=pdl,
    )


_BLOCK_CTXS = (1, 7, 8, 9, 31, 32, 33, 40, 2047, 2048, 2049, 2055, 2056, 2057, 4096, 100000, 100003)
_BLOCK_MODES = (
    # (q_fp8, fused_prep, gate_on, permuted)
    (True, True, True, False),
    (True, True, True, True),
    (True, False, False, False),
    (False, True, False, False),
    (True, True, False, True),
)


@pytest.mark.parametrize("mode", _BLOCK_MODES, ids=lambda m: "q8%d-prep%d-gate%d-perm%d" % m)
@pytest.mark.parametrize("ctx", _BLOCK_CTXS)
@pytest.mark.parametrize("rows", (1, 3, 8))
def test_swa_block_is_the_chain_bit_for_bit(rows, ctx, mode) -> None:
    if rows > ctx:
        pytest.skip("more rows than keys")
    q_fp8, fused_prep, gate_on, permuted = mode
    step = _BlockStep(
        rows, ctx, seed=zlib.crc32(f"{rows} {ctx} {mode}".encode()), permuted=permuted
    )
    chain_pool, op_pool = step.pool.clone(), step.pool.clone()
    fused, want = _chain(step, chain_pool, q_fp8, gate_on)
    got = _block(step, op_pool, q_fp8, fused_prep, gate_on, fused=fused)
    _assert_same_bits(op_pool, chain_pool, f"rows {rows} ctx {ctx} {mode}: cache")
    _assert_same_bits(got, want, f"rows {rows} ctx {ctx} {mode}: output")


@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
def test_swa_block_replays_a_captured_graph_at_other_contexts(pdl) -> None:
    """One capture of the served build (fused prep and gate, Q in E4M3) replays at other
    contexts with the bits of eager calls."""
    top = 131072
    template = _BlockStep(8, top, seed=2)
    seq_lens, positions = template.seq_lens, template.positions
    pool = template.pool

    def step_fn():
        return _block(template, pool, True, True, True, pdl=pdl)

    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step_fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step_fn()
    for ctx in (8, 40, 2049, 4096, 100003, top):
        seq_lens.fill_(ctx)
        positions.copy_(torch.arange(ctx - 8, ctx, device=DEV, dtype=torch.int32).reshape(1, -1))
        snapshot = pool.clone()
        graph.replay()
        torch.cuda.synchronize()
        got, graph_pool = graph_out.clone(), pool.clone()
        pool.copy_(snapshot)
        eager = step_fn()
        _assert_same_bits(got, eager, f"ctx {ctx}: replay vs eager")
        _assert_same_bits(graph_pool, pool, f"ctx {ctx}: cache")


# --------------------------------------------------------------------------
# Microbenchmark (MODELING_V2_TIMING=1): one sliding-window layer's decode attention at tep4
# --------------------------------------------------------------------------
def _graph_us(fn, iters: int = 20, reps: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    return sorted(times)[len(times) // 2]


@pytest.mark.skipif(not os.environ.get("MODELING_V2_TIMING"), reason="MODELING_V2_TIMING unset")
@pytest.mark.parametrize("ctx", (8192, 100000))
def test_microbenchmark_sliding_window_tep4(ctx) -> None:
    """Per sliding-window layer, B=1, 8 rows: the attention (built-in generation preprocess +
    trtllm-gen FMHA vs ``paged_decode_attention`` with E4M3 Q) and the block from the raw
    projection to the gated output (prep + built-in attention + norm-gate vs ``swa_block``'s
    one fused op). The built-in block's prep is timed as ``preattn_prep`` (the built-in's bits
    in fewer launches), so the built-in block time is a lower bound. Printed, never gated."""
    step = _Step(8, ctx, seed=1)
    block = _BlockStep(8, ctx, seed=1)
    pool = step.pool.clone()
    bpool = block.pool.clone()
    fused = preattn_prep(
        block.proj, _rope_table(), block.positions, H, G, CKPT_EPS, CKPT_Q_SCALE, False
    )[0]
    bstep = _Step(8, ctx, seed=1)
    bstep.qkv = fused
    bstep.pool = bpool
    bstep.table = block.table

    def builtin_attention():
        _builtin_step(step, pool)

    def paged():
        paged_decode_attention(*step.args(pool, q_fp8=True, pdl=True))

    def builtin_block():
        f, gate = preattn_prep(
            block.proj, _rope_table(), block.positions, H, G, CKPT_EPS, CKPT_Q_SCALE, False, True
        )
        bstep.qkv = f
        out = _builtin_step(bstep, bpool)
        _builtin_norm_gate(out, gate)

    def chain_block():
        _chain(block, bpool, True, True, pdl=True)

    def fused_block():
        _block(block, bpool, True, True, True, pdl=True)

    results = {name: _graph_us(fn) for name, fn in (
        ("builtin preprocess+fmha", builtin_attention),
        ("paged_decode_attention", paged),
        ("builtin block prep+preprocess+fmha+norm_gate", builtin_block),
        ("swa_paged_decode block prep+paged+norm_gate", chain_block),
        ("swa_block fused", fused_block),
    )}  # fmt: skip
    line = ", ".join(f"{name} {us:.1f} us" for name, us in results.items())
    print(f"\nTIMING tep4 sliding-window layer ctx {ctx} rows 8: {line}")
