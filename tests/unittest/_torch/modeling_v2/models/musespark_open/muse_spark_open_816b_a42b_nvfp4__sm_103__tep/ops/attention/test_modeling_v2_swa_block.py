# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the musespark_open swa_block entry (pre-wait loads, fused prep, fused gate).

The entry's claim is bit identity with the chain it replaces on a
sliding-window decode step: ``preattn_prep`` -> ``paged_decode_attention``
-> the built-in ``_attention_norm_gate_kernel`` (what
``MuseSparkOpenAttention.apply_output_gate`` launches at decode row counts, and
what the PDL norm-gate copy computes statement for statement). Every check compares raw
bits: the KV cache bytes after the step, the attention output and the gated
o_proj input. Besides the cells (also gated against the entry's own reference
within a band), the tests cover window and page boundaries at 1..8 rows,
special values, CUDA-graph replays at changing contexts, a PDL predecessor that
writes the operand late (the negative control for every read that must follow
the wait) -- directly in front of the block, and two kernels back through
prep's entry trigger -- a PDL predecessor that rewrites the cached K/V late
(which shows that the pre-wait loads really run while a predecessor does), a
PTX audit that traces every global load before ``griddepcontrol.wait`` to the
parameters it may read there, and a SASS check that ptxas kept that order.

``MODELING_V2_TIMING=1`` also runs a small isolated timing (printed, never gated).
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import zlib

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    preattn_prep as prep_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    swa_block as swa_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.paged_decode_attention import (  # noqa: E501
    paged_decode_attention,
    window_splits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    CKPT_EPS,
    CKPT_KV_HEADS,
    CKPT_MAX_POSITIONS,
    CKPT_Q_HEADS,
    CKPT_Q_SCALE,
    HEAD_DIM,
    preattn_prep,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.swa_block import (  # noqa: E501
    swa_block_attention as op,
)
from tensorrt_llm._torch.models import modeling_musespark_open as builtin_model

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "swa_block requires a CUDA device"

DEV = "cuda"
FP8 = torch.float8_e4m3fn
H, G = CKPT_Q_HEADS, CKPT_KV_HEADS
Q_WIDTH, KV_WIDTH = H * HEAD_DIM, G * HEAD_DIM
PROJ_WIDTH = 2 * (Q_WIDTH + KV_WIDTH)
#: The checkpoint's post-attention head-norm epsilon (rms_norm_eps).
GATE_EPS = 1e-5


def _rope_table() -> torch.Tensor:
    """The checkpoint's RoPE table, as the built-in sliding-window attention holds it."""
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
        assert attention.scale_query_by == CKPT_Q_SCALE
        assert attention.q_norm.variance_epsilon == CKPT_EPS
        _CACHE["table"] = attention.rotary_emb.rotary_cos_sin
        _CACHE["gate_eps"] = attention.o_head_norm.variance_epsilon
    return _CACHE["table"]


_CACHE = {}


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


class _Step:
    """One sliding-window decode step: a raw BF16 projection, positions, an FP8 paged pool.

    The projection's 64-wide heads each carry their own log-normal scale (six
    decades, as the prep test's), so the norms see a wide input range. Logical
    page ``i`` is K page ``2 perm[i]`` and V page ``2 perm[i] + 1``.
    """

    def __init__(
        self,
        rows,
        ctx,
        *,
        page=32,
        window=2048,
        seed=0,
        scale=1.0,
        permuted=False,
        positions=None,
        special=None,
    ):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.rows, self.ctx, self.page, self.window = rows, ctx, page, window
        self.permuted = permuted
        self.splits = window_splits(window)
        pages = -(-ctx // page)
        spare = pages + 4
        perm = torch.randperm(spare, generator=g, device=DEV)[:pages].to(torch.int32)
        self.table = torch.stack((2 * perm, 2 * perm + 1)).contiguous()
        self.pool = torch.randn(2 * spare, G, page, HEAD_DIM, generator=g, device=DEV).to(FP8)
        heads = PROJ_WIDTH // HEAD_DIM
        x = torch.randn(rows, heads, HEAD_DIM, generator=g, device=DEV)
        x = x * torch.exp(2.0 * torch.randn(rows, heads, 1, generator=g, device=DEV))
        if special is not None:
            special(x)
        proj = x.reshape(rows, PROJ_WIDTH).to(torch.bfloat16)
        if permuted:
            proj = proj[:, _permutation()].contiguous()
        self.proj = proj
        if positions is None:
            positions = torch.arange(ctx - rows, ctx, device=DEV, dtype=torch.int32)
        self.positions = positions.to(torch.int32).reshape(1, rows)
        self.seq_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        self.scale = torch.tensor([scale], dtype=torch.float32, device=DEV)

    def raw_gate(self, proj=None):
        proj = self.proj if proj is None else proj
        if self.permuted:
            return proj[:, Q_WIDTH + 2 * KV_WIDTH :].view(self.rows, H, HEAD_DIM)
        return proj[:, : 2 * Q_WIDTH].view(self.rows, H, 2 * HEAD_DIM)[..., HEAD_DIM:]


def _chain(step, pool, q_fp8, gate_mode, pdl=False, proj=None):
    """The prep + paged decode (+ norm-gate) chain the block replaces. Returns (fused, out)."""
    proj = step.proj if proj is None else proj
    fused, gate = preattn_prep(
        proj, _rope_table(), step.positions, H, G, CKPT_EPS, CKPT_Q_SCALE, step.permuted, pdl
    )
    out = paged_decode_attention(
        fused,
        pool,
        step.table,
        step.seq_lens,
        step.scale,
        H,
        G,
        step.window,
        step.splits,
        q_fp8,
        pdl,
    )
    if gate_mode is None:
        return fused, out
    return fused, _builtin_norm_gate(out, gate)


def _builtin_norm_gate(attention: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The launch ``MuseSparkOpenAttention.apply_output_gate`` makes at 1..128 rows."""
    out = torch.empty_like(attention)
    builtin_model._attention_norm_gate_kernel[(attention.shape[0] * H,)](
        out,
        attention,
        gate,
        attention.stride(0),
        gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else HEAD_DIM,
        GATE_EPS,
        HEADS=H,
        HEAD_DIM=HEAD_DIM,
        num_warps=4,
    )
    return out


def _op_args(step, pool, q_fp8, fused_prep, gate_mode, pdl=False, proj=None, fused=None, gate=None):
    proj = step.proj if proj is None else proj
    if gate is not None:
        pass
    elif gate_mode == "3d":
        gate = step.raw_gate(proj)
    elif gate_mode == "2d":
        gate = preattn_prep(
            proj, _rope_table(), step.positions, H, G, CKPT_EPS, CKPT_Q_SCALE, step.permuted
        )[1]
    else:
        gate = None
    operand = proj if fused_prep else fused
    return (
        operand,
        pool,
        step.table,
        step.seq_lens,
        step.scale,
        H,
        G,
        step.window,
        step.splits,
        q_fp8,
    ), dict(
        fused_prep=fused_prep,
        cos_sin=_rope_table() if fused_prep else None,
        position_ids=step.positions if fused_prep else None,
        prep_eps=CKPT_EPS,
        q_scale=CKPT_Q_SCALE,
        qkv_permuted=step.permuted,
        gate=gate,
        gate_eps=GATE_EPS,
        enable_pdl=pdl,
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8 if t.element_size() == 1 else torch.int16)


def _assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype, (what, got.shape, want.shape)
    diff = _bits(got) != _bits(want)
    if diff.any():
        first = diff.nonzero()[0].tolist()
        raise AssertionError(
            f"{what}: {int(diff.sum())} of {diff.numel()} differ, first at {first}: "
            f"{got[tuple(first)].item()} vs {want[tuple(first)].item()}"
        )


def _check(step, q_fp8, fused_prep, gate_mode, pdl=False, what=""):
    """The op against the chain: cache bytes, output bits. Returns the op output."""
    chain_pool, op_pool = step.pool.clone(), step.pool.clone()
    fused, want = _chain(step, chain_pool, q_fp8, gate_mode, pdl)
    args, kwargs = _op_args(step, op_pool, q_fp8, fused_prep, gate_mode, pdl, fused=fused)
    before = step.proj.clone()
    got = op(*args, **kwargs)
    torch.cuda.synchronize()
    _assert_same_bits(step.proj, before, "the projection is read-only")
    _assert_same_bits(op_pool, chain_pool, f"{what}: cache after the step")
    _assert_same_bits(got, want, f"{what}: output")
    return got


# --------------------------------------------------------------------------
# Cells: bitwise against the chain, and within a band of the entry's reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:56] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    """Bitwise the chain; within (6 scaled ulps, 0.8 ulp RMS) of the entry's reference.

    The reference is the three entries' references in sequence (fp64 prep
    between its BF16 roundings, fp32 attention, fp64 norm-gate between its
    roundings). The band is paged_decode_attention's own (3 scaled ulps, 0.35
    RMS: one BF16 output rounding plus the BF16 P) widened for the gate:
    normalizing a head whose elements each moved by up to 3 scaled ulps moves
    the normed head by as much, then two more BF16 roundings (norm, product)
    and the reference's own. Each cell prints its measured values.
    """
    certifying(op)
    spec = cell.spec
    step = _Step(
        spec["rows"],
        spec["ctx"],
        page=spec["page"],
        window=spec["window"],
        seed=zlib.crc32(cell.why.encode()),
        permuted=spec.get("permuted", False),
    )
    q_fp8, fused_prep, gate_mode = spec["q_fp8"], spec["fused_prep"], spec["gate"]
    got = _check(step, q_fp8, fused_prep, gate_mode, what=cell.why)
    ref_pool = step.pool.clone()
    fused, _ = _chain(step, step.pool.clone(), q_fp8, None)
    args, kwargs = _op_args(step, ref_pool, q_fp8, fused_prep, gate_mode, fused=fused)
    ref = op.reference(*args, **kwargs)
    g, w = got.reshape(-1, HEAD_DIM).float(), ref.reshape(-1, HEAD_DIM).float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    worst = float(((g - w).abs() / scale).max()) / 2.0**-8
    rms = float((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt().clamp_min(1e-30)) / 2.0**-8
    print(f"\n{cell.why}: vs reference worst {worst:.2f} scaled ulps, relative RMS {rms:.3f} ulp")
    assert_within_ulp(got.reshape(-1, HEAD_DIM), ref.reshape(-1, HEAD_DIM), 6.0, 0.8)


# --------------------------------------------------------------------------
# Bit identity across the boundaries the pre-wait loads and the folds touch
# --------------------------------------------------------------------------
_CTXS = (
    1,
    2,
    7,
    8,
    9,
    31,
    32,
    33,
    40,
    1000,
    2040,
    2047,
    2048,
    2049,
    2055,
    2056,
    2057,
    2063,
    2064,
    2080,
    2081,
    4095,
    4096,
    4097,
    100000,
    100003,
    131072,
    449000,
)
_MODES = (
    ("swa_prewait", False, None),
    ("swa_fused_prep", True, None),
    ("swa_fused_gate", False, "2d"),
    ("swa_fused_prep_and_gate", True, "3d"),
)


@pytest.mark.parametrize("mode, fused_prep, gate_mode", _MODES, ids=[m[0] for m in _MODES])
@pytest.mark.parametrize("rows", range(1, 9))
def test_bits_across_window_and_page_boundaries(rows, mode, fused_prep, gate_mode) -> None:
    """Every context of ``_CTXS`` that holds the step's rows, both Q modes, 32- and 64-token pages."""
    for ctx in _CTXS:
        if ctx < rows:
            continue
        for page in (32, 64):
            for q_fp8 in (True, False):
                step = _Step(rows, ctx, page=page, seed=ctx * 131 + rows * 7 + page)
                _check(
                    step,
                    q_fp8,
                    fused_prep,
                    gate_mode,
                    what=f"{mode} rows {rows} ctx {ctx} page {page} q_fp8 {q_fp8}",
                )


@pytest.mark.parametrize("rows", (1, 4, 8))
def test_bits_with_other_positions_and_scales(rows) -> None:
    """Positions anywhere in the RoPE table (not just the cache slots), KV scales other than 1,
    the low-latency weight layout and smaller windows."""
    g = torch.Generator(device=DEV).manual_seed(rows)
    for seed in range(6):
        positions = torch.randint(0, CKPT_MAX_POSITIONS, (rows,), generator=g, device=DEV)
        for scale in (1.0, 0.37, 3.0):
            for permuted in (False, True):
                for window, ctx in ((2048, 100000 + seed), (512, 5000 + seed), (128, 333)):
                    step = _Step(
                        rows,
                        ctx,
                        window=window,
                        seed=seed,
                        scale=scale,
                        permuted=permuted,
                        positions=positions,
                    )
                    _check(
                        step,
                        True,
                        True,
                        "3d",
                        what=f"rows {rows} seed {seed} scale {scale} permuted {permuted} "
                        f"window {window}",
                    )


def _special_heads(kind):
    def apply(x):
        heads = x.shape[1]
        if kind == "zero":
            x[:, ::3] = 0.0
        elif kind == "tiny":
            x[:, ::3] = x[:, ::3] * 1e-35
        elif kind == "huge":
            x[:, 1::4] = x[:, 1::4].sign() * 3.0e38
        elif kind == "one_big":
            x[:, ::2] = 1e-10
            x[:, ::2, 0] = 1e30
        elif kind == "inf":
            x[:, heads // 3, 5] = float("inf")
        elif kind == "nan":
            x[:, heads // 2, 7] = float("nan")

    return apply


@pytest.mark.parametrize("kind", ("zero", "tiny", "huge", "one_big", "inf", "nan"))
def test_bits_on_special_values(kind) -> None:
    """Special heads through prep, attention and gate: the same bits as the chain (NaN and
    infinite results included, compared as raw bits)."""
    for rows, ctx in ((8, 100000), (3, 2050), (1, 40)):
        step = _Step(rows, ctx, seed=rows + ctx, special=_special_heads(kind))
        _check(step, True, True, "3d", what=f"{kind} rows {rows} ctx {ctx}")
        _check(step, False, False, "2d", what=f"{kind} rows {rows} ctx {ctx} (fused gate, BF16 Q)")


def test_the_norm_gate_tree_is_the_builtin_kernels() -> None:
    """The fused norm-gate's sum of squares is the built-in kernel's tree: on heads whose sum
    depends on the add order (24 octaves of magnitude), the gated output equals the built-in
    kernel's for every head."""
    rows = 8
    attn = torch.empty(rows, Q_WIDTH, device=DEV, dtype=torch.bfloat16)
    gate = torch.randn(rows, Q_WIDTH, device=DEV).to(torch.bfloat16)
    # Drive the combine directly: one split whose partial output is the head itself.
    # Split 0 carries the head (m = 0, l = 1); split 1 is empty (l = 0, zeros).
    splits = 2
    part_o = torch.zeros(splits * G * rows * 8, HEAD_DIM, device=DEV, dtype=torch.float32)
    part_ml = torch.zeros(splits * G * rows * 8, 2, device=DEV, dtype=torch.float32)
    part_ml[: G * rows * 8, 1] = 1.0
    for it in range(64):
        e = torch.randint(-12, 12, (rows, Q_WIDTH), device=DEV).float()
        m = 1.0 + torch.rand(rows, Q_WIDTH, device=DEV)
        s = torch.where(torch.rand(rows, Q_WIDTH, device=DEV) < 0.5, -1.0, 1.0)
        attn.copy_((s * m * torch.exp2(e)).to(torch.bfloat16))
        # part row = (split * G + h) * (rows * 8) + token * 8 + head-in-group
        heads = attn.float().view(rows, G, 8, HEAD_DIM).permute(1, 0, 2, 3).reshape(-1, HEAD_DIM)
        part_o[: G * rows * 8].copy_(heads)
        out = torch.empty_like(attn)
        swa_module._swa_combine_kernel[(rows * 8, G)](
            part_o,
            part_ml,
            out,
            out.stride(0),
            gate,
            gate.stride(0),
            HEAD_DIM,
            GATE_EPS,
            splits,
            M=rows,
            HQ=H,
            HKV=G,
            D=HEAD_DIM,
            BLOCK_S=2,
            FUSED_GATE=True,
            LAUNCH_WITH_PDL=False,
            num_warps=4,
        )
        _assert_same_bits(out, _builtin_norm_gate(attn, gate), f"iteration {it}")


# --------------------------------------------------------------------------
# Launch properties
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pdl", (False, True), ids=("plain", "pdl"))
@pytest.mark.parametrize("mode, fused_prep, gate_mode", _MODES, ids=[m[0] for m in _MODES])
def test_a_captured_graph_replays_at_other_contexts(pdl, mode, fused_prep, gate_mode) -> None:
    """Context length, block table, positions and cache are read on the device: one capture
    serves every step, bit for bit the chain at each step's context."""
    rows = 8
    first = _Step(rows, 100000, seed=1)
    max_pages = -(-449100 // 32) + 4
    table = torch.zeros(2, max_pages, dtype=torch.int32, device=DEV)
    pool = torch.zeros(2 * max_pages + 8, G, 32, HEAD_DIM, device=DEV).to(FP8)
    proj = first.proj.clone()
    seq_lens = first.seq_lens.clone()
    positions = first.positions.clone()

    def load(step):
        table.zero_()
        table[:, : step.table.shape[1]].copy_(step.table)
        pool.view(torch.uint8).zero_()
        pool[: step.pool.shape[0]].view(torch.uint8).copy_(step.pool.view(torch.uint8))
        proj.copy_(step.proj)
        seq_lens.copy_(step.seq_lens)
        positions.copy_(step.positions)

    def run():
        shadow = types_step(first, table, pool, proj, seq_lens, positions)
        fused = None
        if not fused_prep:
            fused = preattn_prep(
                proj,
                _rope_table(),
                positions,
                H,
                G,
                CKPT_EPS,
                CKPT_Q_SCALE,
                False,
                pdl,
                trigger_at_entry=pdl,
            )[0]
        args, kwargs = _op_args(
            shadow, pool, True, fused_prep, gate_mode, pdl, proj=proj, fused=fused
        )
        return op(*args, **kwargs)

    load(first)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()  # compiles outside the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for i, ctx in enumerate((100013, 2049, 449000, 8, 2056, 131072)):
        later = _Step(rows, ctx, seed=10 + i)
        load(later)
        graph.replay()
        torch.cuda.synchronize()
        want_pool = later.pool.clone()
        _, want = _chain(later, want_pool, True, gate_mode)
        _assert_same_bits(captured, want, f"{mode} replay at {ctx}")
        _assert_same_bits(pool[: later.pool.shape[0]], want_pool, f"{mode} cache at {ctx}")


def types_step(template, table, pool, proj, seq_lens, positions):
    """A ``_Step`` view over the static buffers a captured graph reads."""
    step = _Step.__new__(_Step)
    step.rows, step.ctx, step.page, step.window = template.rows, None, 32, template.window
    step.permuted, step.splits = False, template.splits
    step.table, step.pool, step.proj = table, pool, proj
    step.positions, step.seq_lens, step.scale = positions, seq_lens, template.scale
    return step


@triton.jit
def _globaltimer(dummy):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _late_writer_kernel(src, dst, n, delay_ns, BLOCK: tl.constexpr):
    """A PDL predecessor that releases its dependents at entry and writes its output late."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def _late_writer(src: torch.Tensor, dst: torch.Tensor, delay_ns: int) -> None:
    n = src.numel()
    _late_writer_kernel[(triton.cdiv(n, 1024),)](
        src, dst, n, delay_ns, BLOCK=1024, num_warps=4, launch_pdl=True
    )


@triton.jit
def _late_bulk_writer_kernel(src, dst, n, delay_ns, BLOCK: tl.constexpr):
    """As ``_late_writer_kernel``, grid-strided: every program is resident at once, so the
    dependents launch at once and every byte lands only after the delay."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    for first in range(tl.program_id(0) * BLOCK, n, tl.num_programs(0) * BLOCK):
        offsets = first + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def _late_bulk_writer(src: torch.Tensor, dst: torch.Tensor, delay_ns: int) -> None:
    s, d = src.reshape(-1).view(torch.uint8), dst.reshape(-1).view(torch.uint8)
    n = s.numel()
    grid = min(triton.cdiv(n, 4096), 96)
    _late_bulk_writer_kernel[(grid,)](s, d, n, delay_ns, BLOCK=4096, num_warps=4, launch_pdl=True)


@pytest.mark.parametrize("mode, fused_prep, gate_mode", _MODES, ids=[m[0] for m in _MODES])
@torch.inference_mode()
def test_a_late_operand_never_reaches_the_block(mode, fused_prep, gate_mode) -> None:
    """Negative control: the operand (raw projection, or prep's output) lands 40 us after the
    block's grids have launched, over NaN; the block's output, cache bytes and gated output are
    still the chain's on the clean operand, eagerly and over 6 replays of one graph."""
    rows = 8
    step = _Step(rows, 100000, seed=5)
    src = step.proj.clone()
    operand = torch.empty_like(src)
    pool = step.pool.clone()

    def chain():
        if fused_prep:
            args, kwargs = _op_args(step, pool, True, True, gate_mode, True, proj=operand)
            _late_writer(src, operand, 40_000)
        else:
            # Everything but the operand is ready before the late writer runs, so the
            # block's grids are its direct PDL dependents.
            clean, gate = preattn_prep(
                src, _rope_table(), step.positions, H, G, CKPT_EPS, CKPT_Q_SCALE, False, True
            )
            late = torch.empty_like(clean)
            late.fill_(float("nan"))
            args, kwargs = _op_args(
                step,
                pool,
                True,
                False,
                gate_mode,
                True,
                proj=src,
                fused=late,
                gate=gate if gate_mode else None,
            )
            _late_writer(clean, late, 40_000)
        return op(*args, **kwargs)

    def load(seed):
        # Contexts at or below the step's own, so its block table covers every key.
        fresh = _Step(rows, 100000 - seed, seed=seed)
        src.copy_(fresh.proj)
        step.positions.copy_(fresh.positions)
        step.seq_lens.copy_(fresh.seq_lens)
        operand.fill_(float("nan"))
        pool.view(torch.uint8).copy_(step.pool.view(torch.uint8))
        return fresh

    def want(fresh):
        ref_pool = step.pool.clone()
        clean = types_step(step, step.table, ref_pool, src.clone(), step.seq_lens, step.positions)
        clean.ctx = int(fresh.seq_lens[0])
        _, out = _chain(clean, ref_pool, True, gate_mode)
        return out, ref_pool

    fresh = load(1)
    out = chain()
    torch.cuda.synchronize()
    w, w_pool = want(fresh)
    _assert_same_bits(out, w, f"{mode} eager output")
    _assert_same_bits(pool, w_pool, f"{mode} eager cache")
    load(2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = chain()
    for seed in range(3, 9):
        fresh = load(seed)
        graph.replay()
        torch.cuda.synchronize()
        w, w_pool = want(fresh)
        _assert_same_bits(captured, w, f"{mode} replay {seed} output")
        _assert_same_bits(pool, w_pool, f"{mode} replay {seed} cache")


#: The late writers' delay in the two tests below that need the host to launch the block while
#: the writer still runs: 2 ms, far above any host launch latency of the block's op.
_LATE_NS = 2_000_000


@pytest.mark.parametrize("gate_mode", (None, "2d"), ids=("swa_prewait", "swa_fused_gate"))
@torch.inference_mode()
def test_a_late_projection_reaches_the_split_only_through_prep(gate_mode) -> None:
    """Transitivity through prep's entry trigger: a PDL predecessor that releases its
    dependents at entry writes the raw projection 2 ms late, over NaN; prep with its entry
    trigger follows, then the split / combine (with the gate fused for ``swa_fused_gate``). The split grid
    launches while prep still waits on the predecessor; output and cache bytes must still be
    the chain's on the clean projection, eagerly and over 6 replays of one graph."""
    rows = 8
    step = _Step(rows, 100000, seed=5)
    src = step.proj.clone()
    raw = torch.empty_like(src)
    pool = step.pool.clone()

    def chain():
        _late_writer(src, raw, _LATE_NS)
        fused, gate = preattn_prep(
            raw,
            _rope_table(),
            step.positions,
            H,
            G,
            CKPT_EPS,
            CKPT_Q_SCALE,
            False,
            True,
            trigger_at_entry=True,
        )
        args, kwargs = _op_args(
            step, pool, True, False, gate_mode, True, proj=raw, fused=fused, gate=gate
        )
        if gate_mode is None:
            kwargs["gate"] = None
        return op(*args, **kwargs)

    def load(seed):
        fresh = _Step(rows, 100000 - seed, seed=seed)
        src.copy_(fresh.proj)
        step.positions.copy_(fresh.positions)
        step.seq_lens.copy_(fresh.seq_lens)
        raw.fill_(float("nan"))
        pool.view(torch.uint8).copy_(step.pool.view(torch.uint8))
        return fresh

    def want(fresh):
        ref_pool = step.pool.clone()
        clean = types_step(step, step.table, ref_pool, src.clone(), step.seq_lens, step.positions)
        clean.ctx = int(fresh.seq_lens[0])
        _, out = _chain(clean, ref_pool, True, gate_mode)
        return out, ref_pool

    fresh = load(1)
    out = chain()
    torch.cuda.synchronize()
    w, w_pool = want(fresh)
    _assert_same_bits(out, w, "eager output")
    _assert_same_bits(pool, w_pool, "eager cache")
    load(2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = chain()
    for seed in range(3, 9):
        fresh = load(seed)
        graph.replay()
        torch.cuda.synchronize()
        w, w_pool = want(fresh)
        _assert_same_bits(captured, w, f"replay {seed} output")
        _assert_same_bits(pool, w_pool, f"replay {seed} cache")


@pytest.mark.parametrize("fused_prep", (False, True), ids=("swa_prewait", "swa_fused_prep"))
@torch.inference_mode()
def test_the_pre_wait_loads_overlap_a_running_predecessor(fused_prep) -> None:
    """The pre-wait window is real, so the memory-order argument is load-bearing: a PDL
    predecessor rewrites the window's *cached* K/V 2 ms after it starts. The paged decode split (every
    load after its wait) always reads the new bytes; the swa_block split, whose cached-K/V loads
    precede its wait, reads the old bytes (or a mix) in at least one of 20 runs. Any in-graph
    writer of the buffers the split reads before its wait must therefore have completed
    before the split grid starts (``kernels/swa_block/attention.py``'s module docstring)."""
    rows = 8
    step = _Step(rows, 100000, seed=11)
    old = step.pool.clone()
    new = _Step(rows, 100000, seed=12).pool.clone()
    assert new.shape == old.shape
    fused, _gate = preattn_prep(
        step.proj, _rope_table(), step.positions, H, G, CKPT_EPS, CKPT_Q_SCALE, False
    )

    def paged_decode_on(pool, pdl):
        return paged_decode_attention(
            fused,
            pool,
            step.table,
            step.seq_lens,
            step.scale,
            H,
            G,
            step.window,
            step.splits,
            True,
            pdl,
        )

    want_old, want_new = paged_decode_on(old.clone(), False), paged_decode_on(new.clone(), False)
    assert not torch.equal(_bits(want_old), _bits(want_new))
    seen = {}
    for name in ("paged_decode", "swa_block"):
        counts = {"old": 0, "new": 0, "mixed": 0}
        launch_us = []
        for _ in range(20):
            pool = old.clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            _late_bulk_writer(new, pool, _LATE_NS)
            if name == "paged_decode":
                out = paged_decode_on(pool, True)
            else:
                args, kwargs = _op_args(step, pool, True, fused_prep, None, True, fused=fused)
                out = op(*args, **kwargs)
            launch_us.append((time.perf_counter() - start) * 1e6)
            torch.cuda.synchronize()
            if torch.equal(_bits(out), _bits(want_old)):
                counts["old"] += 1
            elif torch.equal(_bits(out), _bits(want_new)):
                counts["new"] += 1
            else:
                counts["mixed"] += 1
        counts["host_launch_us_max"] = round(max(launch_us), 1)
        seen[name] = counts
    print(f"\n[pre-wait window fused_prep={fused_prep}] {seen}")
    assert seen["paged_decode"]["new"] == 20, seen
    assert seen["swa_block"]["old"] + seen["swa_block"]["mixed"] >= 1, seen


def test_pages_outside_the_window_are_never_read() -> None:
    """Every page wholly before the window points at a page of NaNs: nothing moves."""
    step = _Step(8, 100000, seed=3)
    clean = op(
        *_op_args(step, step.pool.clone(), True, True, "3d")[0],
        **_op_args(step, step.pool.clone(), True, True, "3d")[1],
    )
    lo = step.ctx - step.rows - step.window + 1
    pool = step.pool.clone()
    poison = pool.shape[0] - 1
    assert poison not in step.table[:, lo // step.page :].flatten().tolist()
    pool.view(torch.uint8)[poison] = 0x7F  # E4M3 NaN
    step.table[:, : lo // step.page] = poison
    args, kwargs = _op_args(step, pool, True, True, "3d")
    with validating(op):
        poisoned = op(*args, **kwargs)
    _assert_same_bits(poisoned, clean, "poisoned pages outside the window")


def test_the_op_is_deterministic() -> None:
    step = _Step(8, 131072, seed=9)
    a = op(
        *_op_args(step, step.pool.clone(), True, True, "3d")[0],
        **_op_args(step, step.pool.clone(), True, True, "3d")[1],
    )
    b = op(
        *_op_args(step, step.pool.clone(), True, True, "3d")[0],
        **_op_args(step, step.pool.clone(), True, True, "3d")[1],
    )
    _assert_same_bits(a, b, "second run")


def test_too_few_splits_is_refused() -> None:
    step = _Step(8, 100000, seed=4)
    args, kwargs = _op_args(step, step.pool.clone(), True, True, "3d")
    args = args[:8] + (step.splits - 1,) + args[9:]
    with pytest.raises(ValueError, match="split"):
        op(*args, **kwargs)


# --------------------------------------------------------------------------
# PTX audit: what is read before griddepcontrol.wait, traced to kernel parameters
# --------------------------------------------------------------------------
_LOAD = re.compile(r"\bld\.global[\w.:]*\s+(?:\{[^}]*\}|%\w+)\s*,\s*\[\s*(%\w+)")
_STORE = re.compile(r"\bst\.global")
_PARAM = re.compile(r"ld\.param\.\w+\s+(%\w+),\s*\[(\w+)\]")
_INSTR = re.compile(r"^(?:@!?%\w+\s+)?([a-z][\w.]*)\s+(.*?);?\s*$")
_REG = re.compile(r"%[a-z]+\d+")


def _compiled(fn):
    """Every compiled variant of a Triton JIT function in this process, newest last."""
    found = []
    for cache in getattr(fn, "device_caches", {}).values():
        kernels = cache[0] if isinstance(cache, tuple) else cache
        found += list(kernels.values())
    return found


def _param_slice(lines, register, params, defs, seen=None):
    """Parameters ``register`` is computed from (a backward slice through the PTX)."""
    seen = set() if seen is None else seen
    if register in seen:
        return set()
    seen.add(register)
    if register in params:
        return {params[register]}
    found = set()
    for index in defs.get(register, ()):
        operands = _REG.findall(lines[index].split(",", 1)[1]) if "," in lines[index] else []
        for reg in operands:
            found |= _param_slice(lines, reg, params, defs, seen)
    return found


def _audit(ptx: str, names):
    """(wait line, trigger lines, [(line, params of the address)] before the wait, stores before)."""
    lines = [line.strip() for line in ptx.splitlines()]
    params, defs = {}, {}
    for i, line in enumerate(lines):
        m = _PARAM.search(line)
        if m:
            index = int(m.group(2).rsplit("_", 1)[1])
            params[m.group(1)] = names[index] if index < len(names) else m.group(2)
            continue
        for part in line.split(";"):
            part = part.strip()
            mi = _INSTR.match(part)
            if not mi or mi.group(1).startswith(("st.", "bra", "setp", "griddep")):
                continue
            first = mi.group(2).split(",", 1)[0]
            for reg in _REG.findall(first):
                defs.setdefault(reg, []).append(i)
    waits = [i for i, line in enumerate(lines) if "griddepcontrol.wait" in line]
    triggers = [i for i, line in enumerate(lines) if "griddepcontrol.launch_dependents" in line]
    assert len(waits) == 1, waits
    early = []
    for i in range(waits[0]):
        for m in _LOAD.finditer(lines[i]):
            early.append((i, _param_slice(lines, m.group(1), params, defs)))
    stores = [i for i in range(waits[0]) if _STORE.search(lines[i])]
    return waits[0], triggers, early, stores


def _arg_names(fn):
    return [p.name for p in fn.params if not p.is_constexpr]


@pytest.mark.parametrize("fused_prep", (False, True), ids=("swa_prewait", "swa_fused_prep"))
def test_ptx_split_reads_only_inputs_and_old_cache_before_the_wait(fused_prep) -> None:
    """Before ``griddepcontrol.wait`` the split kernel loads only through seq_lens, the block
    table, the pool, the scale and (fused prep) the positions and cos/sin table; never through
    the operand or the partial buffers; it stores nothing there; it triggers after the wait."""
    step = _Step(8, 100000, seed=6)
    args, kwargs = _op_args(
        step,
        step.pool.clone(),
        True,
        fused_prep,
        "3d",
        True,
        fused=_chain(step, step.pool.clone(), True, None)[0],
    )
    op(*args, **kwargs)
    torch.cuda.synchronize()
    names = _arg_names(swa_module._swa_split_kernel)
    allowed = {
        "seq_len_ptr",
        "bt_ptr",
        "pool_ptr",
        "scale_ptr",
        "cos_sin_ptr",
        "positions_ptr",
        "bt_stride_kv",
        "num_splits",
        "qkv_stride",
    }
    forbidden = {"qkv_ptr", "part_o_ptr", "part_ml_ptr"}
    audited = 0
    for kernel in _compiled(swa_module._swa_split_kernel):
        ptx = kernel.asm["ptx"]
        if "griddepcontrol.wait" not in ptx:
            continue
        wait, triggers, early, stores = _audit(ptx, names)
        reached = set().union(*[p for _, p in early]) if early else set()
        print(
            f"\n[ptx split fused_prep={fused_prep}] wait at {wait}, triggers {triggers}, "
            f"{len(early)} loads before the wait through {sorted(reached)}"
        )
        assert early, "no load before the wait: the pre-wait reads were moved after it"
        assert not stores, f"stores before the wait at lines {stores}"
        assert not reached & forbidden, f"a pre-wait load reaches {reached & forbidden}"
        assert reached <= allowed, f"unexpected parameters before the wait: {reached - allowed}"
        assert {"pool_ptr", "bt_ptr", "seq_len_ptr"} <= reached
        assert triggers and min(triggers) > wait
        audited += 1
    assert audited, "no PDL variant of the split kernel was compiled"


def test_ptx_combine_and_prep_wait_before_every_load() -> None:
    """The combine kernel and preattn_prep read nothing before their wait; the combine triggers
    after it with the gate fused; prep with ``trigger_at_entry`` triggers before it."""
    step = _Step(8, 100000, seed=7)
    for gate_mode in ("3d", None):
        args, kwargs = _op_args(step, step.pool.clone(), True, True, gate_mode, True)
        op(*args, **kwargs)
    preattn_prep(
        step.proj,
        _rope_table(),
        step.positions,
        H,
        G,
        CKPT_EPS,
        CKPT_Q_SCALE,
        False,
        True,
        trigger_at_entry=True,
    )
    torch.cuda.synchronize()
    seen = 0
    for fn in (swa_module._swa_combine_kernel, prep_module._preattn_prep_kernel):
        names = _arg_names(fn)
        for kernel in _compiled(fn):
            ptx = kernel.asm["ptx"]
            if "griddepcontrol.wait" not in ptx:
                continue
            wait, triggers, early, stores = _audit(ptx, names)
            print(f"\n[ptx {fn.__name__}] wait at {wait}, triggers {triggers}, early loads {early}")
            assert not early and not stores
            seen += 1
    assert seen >= 3


def _cuobjdump() -> str:
    path = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
    return path if os.path.exists(path) else (shutil.which("cuobjdump") or "")


def _sass_opcodes(cubin: bytes) -> list:
    """``[(opcode, line)]`` of a cubin's SASS, in program order."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin)
        path = f.name
    try:
        sass = subprocess.run(
            [_cuobjdump(), "-sass", path], capture_output=True, text=True, check=True
        ).stdout
    finally:
        os.unlink(path)
    found = []
    for line in sass.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m:
            found.append((m.group(1), line.strip()))
    return found


@triton.jit
def _wait_probe_kernel(x, y, PDL: tl.constexpr):
    if PDL:
        tl.extra.cuda.gdc_wait()
    tl.store(y + tl.arange(0, 32), tl.load(x + tl.arange(0, 32)) + 1.0)


@pytest.mark.skipif(not _cuobjdump(), reason="cuobjdump not found")
@torch.inference_mode()
def test_sass_keeps_the_ptx_pre_wait_loads_and_no_non_coherent_load() -> None:
    """The PTX audit carries to the machine code: in the SASS of every PDL variant of the split
    kernel the instruction ``griddepcontrol.wait`` becomes (found by diffing a probe kernel with
    and without it) is present, no more global loads precede it than precede the wait in the
    PTX (ptxas hoisted no operand load above it), and no load anywhere in the kernel goes
    through the non-coherent or constant path (``ld.global.nc``, ``LDG.*.CONSTANT``)."""
    x, y = torch.zeros(32, device=DEV), torch.zeros(32, device=DEV)
    plain = _wait_probe_kernel[(1,)](x, y, PDL=False)
    waiting = _wait_probe_kernel[(1,)](x, y, PDL=True, launch_pdl=True)
    without = {o for o, _ in _sass_opcodes(plain.asm["cubin"])}
    wait_ops = {o for o, _ in _sass_opcodes(waiting.asm["cubin"])} - without
    assert wait_ops, "griddepcontrol.wait leaves no SASS opcode of its own"
    step = _Step(8, 100000, seed=6)
    fused = _chain(step, step.pool.clone(), True, None)[0]
    for fused_prep in (False, True):
        args, kwargs = _op_args(step, step.pool.clone(), True, fused_prep, "3d", True, fused=fused)
        op(*args, **kwargs)
    torch.cuda.synchronize()
    audited = 0
    for kernel in _compiled(swa_module._swa_split_kernel):
        ptx = kernel.asm["ptx"]
        if "griddepcontrol.wait" not in ptx:
            continue
        lines = ptx.splitlines()
        wait = next(i for i, line in enumerate(lines) if "griddepcontrol.wait" in line)
        ptx_before = sum(1 for line in lines[:wait] if re.search(r"\bld\.global", line))
        ptx_nc = sum(1 for line in lines if re.search(r"\bld\.global[\w.]*\.nc\b", line))
        ops = _sass_opcodes(kernel.asm["cubin"])
        waits = [i for i, (o, _) in enumerate(ops) if o in wait_ops]
        assert waits, f"no {sorted(wait_ops)} in the split kernel's SASS"
        sass_before = sum(1 for o, _ in ops[: waits[0]] if o.startswith("LDG"))
        constant = [line for o, line in ops if o.startswith("LDG") and "CONSTANT" in o]
        print(
            f"\n[sass split regs={kernel.n_regs}] wait {ops[waits[0]][1]!r}; global loads before "
            f"it: PTX {ptx_before}, SASS {sass_before}; LDG total "
            f"{sum(1 for o, _ in ops if o.startswith('LDG'))}"
        )
        assert 0 < sass_before <= ptx_before, (sass_before, ptx_before)
        assert ptx_nc == 0 and not constant, (ptx_nc, constant[:3])
        audited += 1
    assert audited >= 2, "both PDL variants of the split kernel should have been compiled"


# --------------------------------------------------------------------------
# Isolated timing (printed, never gated)
# --------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
def test_isolated_timing() -> None:
    step = _Step(8, 100000, seed=8)
    pool = step.pool.clone()
    fused = _chain(step, pool.clone(), True, None)[0]
    arms = {
        "chain prep+paged_decode+gate": lambda: _chain(step, pool, True, "3d", True),
        "swa_prewait (prep entry trigger + split pre-wait)": lambda: op(
            *_op_args(
                step,
                pool,
                True,
                False,
                None,
                True,
                fused=preattn_prep(
                    step.proj,
                    _rope_table(),
                    step.positions,
                    H,
                    G,
                    CKPT_EPS,
                    CKPT_Q_SCALE,
                    False,
                    True,
                    trigger_at_entry=True,
                )[0],
            )[0],
            **_op_args(step, pool, True, False, None, True, fused=fused)[1],
        ),
        "swa_fused_prep_and_gate": lambda: op(
            *_op_args(step, pool, True, True, "3d", True)[0],
            **_op_args(step, pool, True, True, "3d", True)[1],
        ),
    }
    for name, fn in arms.items():
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(20):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000 / 20)
        times.sort()
        print(f"\n[timing] {name}: {times[len(times) // 2]:.2f} us per block (median of 20)")
