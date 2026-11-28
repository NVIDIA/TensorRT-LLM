# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_mxfp8_linear entry: attention projections over the checkpoint's MXFP8 weights.

Small decode-graph steps (<= 16 rows under `flashinfer_mxfp8_decode_graph_capture`, cells with `graph`) are exact
FP8-weight products in fp32, gated at a relative RMS of 4e-3 (bf16 rounding of the output). Larger steps, context chunks
and eager steps quantize the activation to MXFP8 as well (the engine's gate); their gate is 3e-2 against the unquantized
product.
"""

import contextlib

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (  # noqa: E501
    SMALL_M_ROWS,
    csa2_mxfp8_gemm_prewarm,
    csa2_mxfp8_gemm_tune_rows,
    csa2_mxfp8_graph_rows,
    csa2_mxfp8_linear,
    mxfp8_row_scales,
    mxfp8_swizzled_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (
    csa2_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_stock_rope_table,
)
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture


def _ids(c):
    return [x.why[:48] for x in c]


def _weights(n, k, gen):
    w = (torch.randn(n, k, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    blocks = torch.randint(
        121, 127, ((n + 31) // 32, k // 32), generator=gen, device=DEV, dtype=torch.uint8
    )  # 2^-6 .. 2^-1
    scale = mxfp8_row_scales(blocks, (n, k))
    return w, scale, mxfp8_swizzled_scales(scale)


def _rel(a, b):
    return (
        (a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt()
    ).item()


@pytest.mark.parametrize("cell", csa2_mxfp8_linear.CELLS, ids=_ids(csa2_mxfp8_linear.CELLS))
def test_cells(cell, certifying):  # noqa: F811
    certifying(csa2_mxfp8_linear)
    s, gen = cell.spec, gen_for(cell)
    m, n, k = s["m"], s["n"], s["k"]
    w, scale, sw = _weights(n, k, gen)
    x = torch.randn(m, k, generator=gen, device=DEV).bfloat16()
    alpha = torch.ones(1, device=DEV)
    kw = {}
    if s.get("split_n"):
        kw["split_n"] = s["split_n"]
    if s.get("rope"):
        table = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV)).view(
            -1, 2, 32
        )
        kw["rope"] = (
            torch.randint(0, 4000, (m,), generator=gen, device=DEV, dtype=torch.int32),
            table,
            s["rope"],
        )
    if s.get("rms"):
        kw["rms_norm"] = (torch.randn(k, generator=gen, device=DEV).bfloat16(), 1e-6)
    args = (x, w, scale, sw, alpha, kw.get("split_n", 0), kw.get("rope"), kw.get("rms_norm"))
    with (
        flashinfer_mxfp8_decode_graph_capture() if s.get("graph") else contextlib.nullcontext()
    ):  # a decode-graph step
        got = csa2_mxfp8_linear(*args, native_eager=s.get("native", False))
    ref = csa2_mxfp8_linear.reference(*args)
    lowm = m <= SMALL_M_ROWS and s.get("graph") and w.numel() <= 16 << 20
    gate = (1e-2 if kw.get("rope") or kw.get("rms_norm") else 4e-3) if lowm else 3e-2
    for g, r in zip(
        got if isinstance(got, tuple) else (got,), ref if isinstance(ref, tuple) else (ref,)
    ):
        assert g.shape == r.shape and g.dtype == torch.bfloat16
        assert _rel(g, r) < gate, _rel(g, r)


@pytest.mark.parametrize("m", [2, 6, 14])
@pytest.mark.parametrize(
    "form",
    ["qkv_split", "q_b_norm_rope", "q_b_norm", "q_b_rope", "i_q_b_norm_rope", "split_norm_rope"],
)
def test_rows_of_different_scale(form, m):
    """Real activations: every row its own RMS (here spread over three orders of magnitude), checked row by row.

    Decode-graph steps of up to 16 rows (tep8 decode of 6 or 14 rows, dep small decode steps; run here under
    `flashinfer_mxfp8_decode_graph_capture`) take the low-M kernel with these forms (q_b and the index q_b fold the
    q_norm and the RoPE). The kernel's epilogue RoPE rotates with an un-normalized
    partner column when the norm is folded too: rows of RMS ~1 hide it, these do not.
    """
    gen = torch.Generator(device=DEV).manual_seed(100 + m)
    # split_norm_rope: a split projection with the norm folded and a separate RoPE (rotated whole, then split)
    n, k, split = {
        "qkv_split": (1792, 5120, 1280),
        "i_q_b_norm_rope": (8192, 1280, 0),
        "split_norm_rope": (4096, 1280, 1280),
    }.get(form, (4096, 1280, 0))
    w, scale, sw = _weights(n, k, gen)
    x = torch.randn(m, k, generator=gen, device=DEV)
    x = (
        x
        * torch.logspace(-1.5, 1.5, m, device=DEV)[
            torch.randperm(m, generator=gen, device=DEV)
        ].unsqueeze(1)
    ).bfloat16()
    alpha = torch.ones(1, device=DEV)
    rope = rms = None
    if "rope" in form:
        table = csa2_stock_rope_table(csa2_rope_table(64, 4096, 0, 10000.0, 1.0, 32, 1, DEV)).view(
            -1, 2, 32
        )
        head = 128 if form.startswith("i_") else 512  # the index heads are 128 wide
        rope = (
            torch.randint(0, 4000, (m,), generator=gen, device=DEV, dtype=torch.int32),
            table,
            head,
        )
    if "norm" in form:
        rms = ((torch.rand(k, generator=gen, device=DEV) + 0.5).bfloat16(), 1e-6)
    args = (x, w, scale, sw, alpha, split, rope, rms)
    with validating(csa2_mxfp8_linear), flashinfer_mxfp8_decode_graph_capture():
        got = csa2_mxfp8_linear(*args)
    ref = csa2_mxfp8_linear.reference(*args)
    for g, r in zip(
        got if isinstance(got, tuple) else (got,), ref if isinstance(ref, tuple) else (ref,)
    ):
        per_row = [_rel(g[i], r[i]) for i in range(m)]
        assert max(per_row) < 1e-2, (form, m, [f"{e:.2e}" for e in per_row])


def test_low_m_only_in_a_decode_graph():
    """Five rows (a short context chunk) of q_a | kv: the low-M kernel (bf16 activations) only under the decode-graph
    flag; outside it the activation is quantized to MXFP8 and the result equals the engine's prefill projection
    (`mxfp8_quantize_activation` + `mxfp8_mxfp8_gemm`) bit for bit."""
    from tensorrt_llm._torch.modules.linear import mxfp8_quantize_activation

    gen = torch.Generator(device=DEV).manual_seed(13)
    w, scale, sw = _weights(1792, 5120, gen)
    x = torch.randn(5, 5120, generator=gen, device=DEV).bfloat16()
    alpha = torch.ones(1, device=DEV)
    eager = csa2_mxfp8_linear(x, w, scale, sw, alpha)
    data, a_scale = mxfp8_quantize_activation(x)
    stock = torch.ops.trtllm.mxfp8_mxfp8_gemm(data, a_scale, w, sw, alpha, torch.bfloat16)
    assert _rel(eager, stock) < 1e-6, _rel(eager, stock)
    with flashinfer_mxfp8_decode_graph_capture():
        lowm = csa2_mxfp8_linear(x, w, scale, sw, alpha)
    ref = csa2_mxfp8_linear.reference(x, w, scale, sw, alpha)
    assert _rel(lowm, ref) < 4e-3 < _rel(eager, ref)


@pytest.mark.parametrize("m", [5, 24, 2048])
def test_native_eager_is_stocks_native_gemm(m):
    """`native_eager`: outside a decode graph the projection is stock's native autotuned GEMM
    (`trtllm::mxfp8_mxfp8_gemm_autotuned` over `mxfp8_quantize_activation`, its CSA2 module's `native_mxfp8_gemm`) bit
    for bit at every row count; under the decode-graph flag the option selects nothing (the graph cells' kernels)."""
    from tensorrt_llm._torch.modules.linear import mxfp8_quantize_activation

    gen = torch.Generator(device=DEV).manual_seed(17)
    w, scale, sw = _weights(1792, 5120, gen)
    x = torch.randn(m, 5120, generator=gen, device=DEV).bfloat16()
    alpha = torch.ones(1, device=DEV)
    qr, kv = csa2_mxfp8_linear(x, w, scale, sw, alpha, 1280, native_eager=True)
    data, a_scale = mxfp8_quantize_activation(x)
    stock = torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned(data, a_scale, w, sw, alpha, torch.bfloat16)
    assert torch.equal(torch.cat([qr, kv], 1), stock)
    if m <= SMALL_M_ROWS:
        with flashinfer_mxfp8_decode_graph_capture():
            assert torch.equal(
                csa2_mxfp8_linear(x, w, scale, sw, alpha, native_eager=True),
                csa2_mxfp8_linear(x, w, scale, sw, alpha),
            )


@pytest.mark.parametrize("graph", [False, True])
def test_cuda_graph_replay(graph):
    with flashinfer_mxfp8_decode_graph_capture() if graph else contextlib.nullcontext():
        _graph_replay()


def _graph_replay():
    gen = torch.Generator(device=DEV).manual_seed(3)
    w, scale, sw = _weights(1792, 5120, gen)
    alpha = torch.ones(1, device=DEV)
    for m in (6, 64):
        x = torch.randn(m, 5120, generator=gen, device=DEV).bfloat16()
        run = lambda a: csa2_mxfp8_linear(a, w, scale, sw, alpha, 1280)  # noqa: E731
        run(x)
        torch.cuda.synchronize()
        static = x.clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run(static)
        static.copy_(x * 0.5)
        g.replay()
        torch.cuda.synchronize()
        ref = run(x * 0.5)
        assert all(torch.equal(a, b) for a, b in zip(out, ref))


def test_rejections():
    gen = torch.Generator(device=DEV).manual_seed(1)
    w, scale, sw = _weights(256, 256, gen)
    x = torch.zeros(4, 256, device=DEV, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=DEV)
    csa2_mxfp8_linear.is_valid(x, w, scale, sw, alpha)
    with pytest.raises(AssertionError, match="E4M3"):
        csa2_mxfp8_linear.is_valid(x, w.to(torch.bfloat16), scale, sw, alpha)
    with pytest.raises(AssertionError):
        csa2_mxfp8_linear.is_valid(x, w, scale[:, :4], sw, alpha)


def test_prewarm_compiles_untuned_row_kernels():
    """After `csa2_mxfp8_gemm_prewarm`, quantized projections of any untuned row count (a real projection shape, every
    MMA-tile class of the fallback GEMM) compile nothing more, and the results stay correct."""
    from flashinfer.gemm import gemm_base

    from tensorrt_llm._torch.custom_ops.mxfp8_gemm import _flashinfer_runner
    from tensorrt_llm._torch.modules.mxfp8_swapab import _COMPILED as swapab

    cache = gemm_base._CUTE_DSL_MM_MXFP8_KERNEL_CACHE
    rows = csa2_mxfp8_gemm_prewarm(torch.device(DEV))
    assert rows or len(cache)  # a second call in the same process is a no-op
    assert csa2_mxfp8_gemm_prewarm(torch.device(DEV)) == ()
    compiled, modules, swapped = len(cache), _flashinfer_runner.cache_info().currsize, len(swapab)
    assert modules >= 1 and swapped >= 1, (
        "the prewarm loads the FlashInfer CUTLASS module and compiles the swap-AB GEMM"
    )
    gen = torch.Generator(device=DEV).manual_seed(5)
    w, scale, sw = _weights(
        14336, 1280, gen
    )  # past the low-M weight cap (every row count quantizes), below the wide-N dense rule
    alpha = torch.ones(1, device=DEV)
    for m in (3, 7, 13, 24, 31, 40, 150, 256, 300):
        x = torch.randn(m, 1280, generator=gen, device=DEV).bfloat16()
        got = csa2_mxfp8_linear(x, w, scale, sw, alpha)
        ref = csa2_mxfp8_linear.reference(x, w, scale, sw, alpha)
        assert _rel(got, ref) < 3e-2, (m, _rel(got, ref))
    torch.cuda.synchronize()
    assert len(swapab) == swapped, (
        f"compiled {len(swapab) - swapped} more swap-AB GEMM kernels after the prewarm"
    )
    assert len(cache) == compiled, (
        f"compiled {len(cache) - compiled} more GEMM kernels after the prewarm"
    )
    assert _flashinfer_runner.cache_info().currsize == modules, (
        "a FlashInfer GEMM module was built after the prewarm"
    )


def test_graph_rows():
    class Spec:
        max_draft_len = max_total_draft_tokens = 3

    class Cfg:
        spec_config = Spec()
        cuda_graph_batch_sizes = (1, 2, 4, 64, 2)

    assert csa2_mxfp8_graph_rows(Cfg()) == (4, 8, 16, 256)
    Cfg.spec_config = None
    assert csa2_mxfp8_graph_rows(Cfg()) == (1, 2, 4, 64)


def test_warmup_tunes_registered_rows():
    """Inside the engine's autotuner pass the first projection of a weight profiles every registered row count; serving
    calls of those rows then hit the tuner's cache (a tuned tactic, not the fallback), with unchanged results."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.custom_ops.mxfp8_gemm import MXFP8GemmUnifiedRunner
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        _MXFP8_GEMM_AUTOTUNED_OP,
        _choose_mxfp8_tactic,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
        _backends,
    )

    gen = torch.Generator(device=DEV).manual_seed(7)
    w, scale, sw = _weights(
        5120, 1024, gen
    )  # o_b of one group: a graph step of any row count runs the autotuned GEMM
    alpha = torch.ones(1, device=DEV)
    rows = (
        24,
        160,
        300,
        1280,
    )  # 1280: the dep8 decode step (eager steps of 1024+ rows run the dense GEMM)
    try:
        assert csa2_mxfp8_gemm_tune_rows(rows) == rows
        with autotune():
            assert AutoTuner.get().is_tuning_mode
            csa2_mxfp8_linear(
                torch.randn(2048, 1024, generator=gen, device=DEV).bfloat16(), w, scale, sw, alpha
            )
        for m in rows:
            x = torch.randn(m, 1024, generator=gen, device=DEV).bfloat16()
            data, a_scale = torch.ops.trtllm.mxfp8_quantize(x, True)
            runner = MXFP8GemmUnifiedRunner(torch.bfloat16, _backends(m))
            tactic = _choose_mxfp8_tactic(
                _MXFP8_GEMM_AUTOTUNED_OP, runner, [data, a_scale, w, sw], False
            )
            assert tactic != -1, f"{m} rows: no tuned tactic"
            assert (
                _rel(
                    csa2_mxfp8_linear(x, w, scale, sw, alpha),
                    csa2_mxfp8_linear.reference(x, w, scale, sw, alpha),
                )
                < 3e-2
            )
        # a captured 1280-row step runs that tuned tactic (as stock's decode-graph projections), within the same band
        x = torch.randn(1280, 1024, generator=gen, device=DEV).bfloat16()
        csa2_mxfp8_linear(x, w, scale, sw, alpha)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = csa2_mxfp8_linear(x, w, scale, sw, alpha)
        g.replay()
        torch.cuda.synchronize()
        assert _rel(out, csa2_mxfp8_linear.reference(x, w, scale, sw, alpha)) < 3e-2
        # a row count registered later for the same weight is tuned too (the weight shape was already done)
        assert csa2_mxfp8_gemm_tune_rows(rows + (416,)) == tuple(sorted(rows + (416,)))
        with autotune():
            csa2_mxfp8_linear(
                torch.randn(2048, 1024, generator=gen, device=DEV).bfloat16(), w, scale, sw, alpha
            )
        x = torch.randn(416, 1024, generator=gen, device=DEV).bfloat16()
        data, a_scale = torch.ops.trtllm.mxfp8_quantize(x, True)
        runner = MXFP8GemmUnifiedRunner(torch.bfloat16, _backends(416))
        assert (
            _choose_mxfp8_tactic(_MXFP8_GEMM_AUTOTUNED_OP, runner, [data, a_scale, w, sw], False)
            != -1
        ), "416 rows registered after the weight was tuned: no tuned tactic"
    finally:
        csa2_mxfp8_gemm_tune_rows(())


def test_tactics_agree_once_after_tuning():
    """`csa2_mxfp8_agree_tactics` runs once, outside the tuning pass and any capture, after something was tuned (single
    process: the agreement itself is the identity); later calls do nothing."""
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention import (
        csa2_mxfp8_linear as lin,
    )

    gen = torch.Generator(device=DEV).manual_seed(11)
    w, scale, sw = _weights(5120, 1024, gen)
    alpha = torch.ones(1, device=DEV)
    saved = lin._agreed
    try:
        lin._agreed = False
        csa2_mxfp8_gemm_tune_rows((300,))
        with autotune():
            assert not lin.csa2_mxfp8_agree_tactics()  # inside the tuning pass: not yet
            csa2_mxfp8_linear(
                torch.randn(300, 1024, generator=gen, device=DEV).bfloat16(), w, scale, sw, alpha
            )
        assert lin._tuned
        assert lin.csa2_mxfp8_agree_tactics()
        assert not lin.csa2_mxfp8_agree_tactics()
    finally:
        lin._agreed = saved
        csa2_mxfp8_gemm_tune_rows(())
