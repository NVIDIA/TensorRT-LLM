# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention projection over the checkpoint's MXFP8 weights, as the engine's CSA2 module selects the kernel.

`weight` is the checkpoint's E4M3 `[N, K]` tensor and `scale` its UE8M0 per-row 1x32 scales `[N, K/32]` (uint8;
`mxfp8_row_scales` expands the checkpoint's 32x32 blocks, exactly). Small decode-graph steps (at most 16 rows, under
`flashinfer_mxfp8_decode_graph_capture`, as the engine gates its small-M projections) run the engine's
low-M warp-MMA kernel (`low_m_gemm.mxfp8_lowm_linear`: BF16 activations times the FP8 weights dequantized exactly in
registers, FP32 accumulation; it can also split the output, rotate the RoPE channels of every head in its epilogue and
RMS-normalise its input rows). Larger steps, context chunks and eager steps quantize the activation to MXFP8
(`trtllm::mxfp8_quantize`) and run the CUTLASS MXFP8 GEMM
(`trtllm::mxfp8_mxfp8_gemm`) on the CUTLASS-swizzled weight scales `scale_sw`.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_rope import csa2_rope_tail
from .csa2_store_rows import csa2_rope_inplace

SMALL_M_ROWS = 16  # rows the low-M kernel serves (the engine's `SMALL_M_ROWS`)
_LOWM_MAX_WEIGHT_BYTES = 16 << 20  # past it the tensor-core GEMM streams the weight faster
_BLOCK = 32


def mxfp8_row_scales(scale: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """The checkpoint's `[ceil(R/32), ceil(C/32)]` UE8M0 block scales -> per-row 1x32 scales `[R, C/32]` uint8.

    A setup helper, not an entry (after the engine's `expand_mxfp8_scales`): scale metadata only, the E4M3 weight
    bytes are never touched.
    """
    r, c = shape
    assert c % _BLOCK == 0
    if scale.dtype == torch.uint8:
        enc = scale
    elif scale.dtype == getattr(torch, "float8_e8m0fnu", None):
        enc = scale.view(torch.uint8)
    else:
        v = scale.float()
        assert bool((torch.isfinite(v) & (v > 0)).all()), (
            "MXFP8 scales must be finite positive powers of two"
        )
        e = torch.log2(v)
        assert bool(((e == e.round()) & (e >= -127) & (e <= 127)).all()), (
            "scales must be exact UE8M0 values"
        )
        enc = (e + 127).to(torch.uint8)
    return enc.repeat_interleave(_BLOCK, 0)[:r].contiguous()[:, : c // _BLOCK].contiguous()


def mxfp8_swizzled_scales(scale: torch.Tensor) -> torch.Tensor:
    """`[N, K/32]` uint8 -> the CUTLASS 128x4-swizzled 1-D scales the GEMM reads (a setup helper)."""
    return torch.ops.trtllm.block_scale_interleave(scale.contiguous()).reshape(-1)


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = torch.exp2(scale.float() - 127).repeat_interleave(_BLOCK, 1)
    return weight.float() * s


def _lowm_fits(m: int, n: int, k: int, weight: torch.Tensor) -> bool:
    """Whether `m` rows of an `[N, K]` weight run the low-M kernel: only in a decode-graph step (warm-up or capture,
    `flashinfer_mxfp8_decode_graph_capture_active`), as the engine's small-M projections (`_lowm_projects_qkv`,
    `_lowm_projects`, `_lowm_o_a`); context chunks and eager steps of any row count quantize the activation, as stock's
    prefill projection and its eager linears do."""
    from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture_active
    from tensorrt_llm._torch.modules.low_m_gemm import lowm_fp8_mma_fits

    return (
        0 < m <= SMALL_M_ROWS
        and flashinfer_mxfp8_decode_graph_capture_active()
        and lowm_fp8_mma_fits(m, n, k)
        and weight.numel() <= _LOWM_MAX_WEIGHT_BYTES
    )


def csa2_mxfp8_epilogue_rope(m: int, weight: torch.Tensor, rms_norm: bool = False) -> bool:
    """Whether `csa2_mxfp8_linear` rotates the RoPE channels inside its GEMM for `m` rows (the low-M kernel), so a
    caller that rotates the query elsewhere (the direct decode does it as it loads) must not rotate it twice. Never with
    a folded `rms_norm` (see `Csa2Mxfp8Linear.__call__`). A setup-free predicate."""
    return not rms_norm and _lowm_fits(m, weight.shape[0], weight.shape[1], weight)


_DENSE_MIN_ROWS = 1024  # the persistent CuTe DSL GEMM beats the autotuned CUTLASS one from here on (GB300, O-B shape)
_WIDE_N, _WIDE_DENSE_MIN_ROWS = (
    16384,
    128,
)  # ... and from 128 rows for very wide outputs (the 64-head q_b: N 32768, K 1280)
_SWAP_AB_MAX_ROWS = (
    256  # up to here the swap-AB CuTe DSL GEMM is the fastest candidate with a cold L2 (GB300)
)


def _dense(m: int, n: int, k: int, graph: bool = False) -> bool:
    """Whether `m` rows of an `[N, K]` weight run the dense CuTe GEMM (shape rule only). `graph`: a decode-graph step
    (warm-up or capture, `flashinfer_mxfp8_decode_graph_capture_active`), whose projections other than the wide q_b take
    the autotuned GEMM at every row count, as stock's decode-graph projections do (QKV-A
    `mxfp8_quantize_gemm_autotuned`, O-B and index Q-B `mxfp8_linear_quantized` under the decode capture: the unified
    MXFP8 tactic cache, cold-L2 profiled; the dense GEMM is its context / eager route)."""
    if graph and n < _WIDE_N:
        return False
    return (
        m >= (_WIDE_DENSE_MIN_ROWS if n >= _WIDE_N else _DENSE_MIN_ROWS)
        and n % 256 == 0
        and k % 128 == 0
    )


def _backends(m: int) -> Tuple[str, ...]:
    """The autotuned GEMM's candidates for `m` rows."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import projection_backends
    from tensorrt_llm._torch.custom_ops.mxfp8_gemm import MXFP8_DEFAULT_BACKENDS

    # A decode step streams the weight from HBM; the autotuner times the candidates with it resident in L2 and picks the
    # FlashInfer CuTe DSL GEMM, while with a cold L2 the swap-AB CuTe DSL one is faster up to 256 rows on GB300. Larger
    # steps keep the engine's candidates.
    if m <= _SWAP_AB_MAX_ROWS and "cutedsl_swap_ab" in MXFP8_DEFAULT_BACKENDS:
        return ("cutedsl_swap_ab",)
    return projection_backends(m, MXFP8_DEFAULT_BACKENDS)


# Row counts whose tactics `_quantized_gemm` tunes at warm-up (`csa2_mxfp8_gemm_tune_rows`), and the weight shapes done.
_tune_rows: Tuple[int, ...] = ()
_tuned: set = set()


def csa2_mxfp8_gemm_tune_rows(rows) -> Tuple[int, ...]:
    """Register the row counts a target's steps run the autotuned MXFP8 GEMM at (its CUDA-graph step rows), a setup
    helper. While the engine's warm-up autotuner pass runs (`AutoTuner.is_tuning_mode`), the first `_quantized_gemm` of
    every weight shape then profiles the candidates at each of these rows, so serving steps of these rows run tuned
    tactics instead of the runner's fallback (stock's MXFP8 linears are tuned per graph shape the same way). Returns the
    registered rows."""
    global _tune_rows
    _tune_rows = tuple(sorted({int(r) for r in rows if int(r) > 0}))
    return _tune_rows


def csa2_mxfp8_graph_rows(model_config) -> Tuple[int, ...]:
    """Rows of a generation target's CUDA-graph steps: every captured batch size times the tokens a request adds per
    step (one plus the draft tokens). A setup helper for `csa2_mxfp8_gemm_tune_rows`."""
    spec = getattr(model_config, "spec_config", None)
    draft = (
        int(getattr(spec, "max_total_draft_tokens", None) or getattr(spec, "max_draft_len", 0) or 0)
        if spec
        else 0
    )
    return tuple(
        sorted(
            {
                int(b) * (1 + draft)
                for b in (getattr(model_config, "cuda_graph_batch_sizes", None) or ())
            }
        )
    )


def _tune_weight(weight: torch.Tensor, scale_sw: torch.Tensor) -> None:
    """Profile `weight`'s autotuned GEMM at every registered row count (the engine's tuner, exact rows, cached)."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_gemm_autotuned

    n, k = weight.shape
    # per weight shape and registered row set: rows registered later are tuned for a shape done before
    key = (weight.device.index, n, k, scale_sw.numel(), _tune_rows)
    if key in _tuned:
        return
    import time

    from tensorrt_llm._torch.custom_ops.mxfp8_gemm import MXFP8GemmUnifiedRunner
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        _MXFP8_GEMM_AUTOTUNED_OP,
        _choose_mxfp8_tactic,
    )
    from tensorrt_llm.logger import logger

    t0, picks = time.perf_counter(), {}
    for m in _tune_rows:  # graph rows: the rows the decode graphs capture
        if _dense(m, n, k, graph=True):
            continue
        data, a_scale = torch.ops.trtllm.mxfp8_quantize(
            torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device), True
        )
        mxfp8_gemm_autotuned(
            data, a_scale, weight, scale_sw, torch.bfloat16, _backends(m), tune=True
        )
        inputs = [data, a_scale, weight, scale_sw]
        picks[m] = _choose_mxfp8_tactic(
            _MXFP8_GEMM_AUTOTUNED_OP,
            MXFP8GemmUnifiedRunner(torch.bfloat16, _backends(m)),
            inputs,
            False,
        )
    _tuned.add(key)  # only once every row count was profiled: a failed pass is retried
    logger.info(
        f"DeepSeek-V4.1 MXFP8 GEMM [{n}, {k}] tuned at {len(picks)} row counts in {time.perf_counter() - t0:.1f} s: "
        + ", ".join(f"{m}:{t}" for m, t in picks.items())
    )


def csa2_mxfp8_tune_engine_gemm(device, n: int, k: int, rows) -> dict:
    """Profile the engine's autotuned MXFP8 GEMM (`mxfp8_gemm_autotuned`, its default candidates: the op key and runner
    every engine caller shares) for an `[n, k]` weight at each of `rows` (exact rows: the runner's tuning config has no
    row buckets), on a zero weight of that shape, once per process. A setup helper for engine code inside a target's
    forward that calls that GEMM with an untuned shape: the DSpark draft's MXFP8 LM head (`CompactTPArgmax`), which
    stock tunes in its warm-up (its own MXFP8 linears switch on the engine's FlashInfer MXFP8 autotune pass, which this
    target, with no such linears, does not get), and which otherwise runs the runner's fallback CUTLASS tactic. Does
    nothing outside the engine's warm-up autotuner pass (`AutoTuner.is_tuning_mode`) or under graph capture, so a target
    may call it on every forward. Returns `{rows: tactic}` (empty when it did nothing)."""
    key = ("engine", torch.device(device).index, n, k)
    if key in _tuned or not rows or k % _BLOCK or torch.cuda.is_current_stream_capturing():
        return {}
    import time

    from tensorrt_llm._torch.autotuner import AutoTuner

    if not AutoTuner.get().is_tuning_mode:
        return {}

    from tensorrt_llm._torch.custom_ops.mxfp8_gemm import MXFP8GemmUnifiedRunner
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        _MXFP8_GEMM_AUTOTUNED_OP,
        _choose_mxfp8_tactic,
        mxfp8_gemm_autotuned,
    )
    from tensorrt_llm.logger import logger

    device = torch.device(device)
    _tuned.add(key)
    t0, picks = time.perf_counter(), {}
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device=device)
    # the swizzled scales exactly as the draft head builds them (`block_scale_interleave` of [n, k/32] uint8)
    scale_sw = torch.ops.trtllm.block_scale_interleave(
        torch.full((n, k // _BLOCK), 127, dtype=torch.uint8, device=device)
    )
    for m in sorted({int(r) for r in rows if int(r) > 0}):
        data, a_scale = torch.ops.trtllm.mxfp8_quantize(
            torch.zeros(m, k, dtype=torch.bfloat16, device=device), True
        )
        mxfp8_gemm_autotuned(data, a_scale, weight, scale_sw, torch.bfloat16, tune=True)
        picks[m] = _choose_mxfp8_tactic(
            _MXFP8_GEMM_AUTOTUNED_OP,
            MXFP8GemmUnifiedRunner(torch.bfloat16),
            [data, a_scale, weight, scale_sw],
            False,
        )
    del weight, scale_sw
    logger.info(
        f"DeepSeek-V4.1 engine MXFP8 GEMM [{n}, {k}] tuned at {len(picks)} row counts in "
        f"{time.perf_counter() - t0:.1f} s: " + ", ".join(f"{m}:{t}" for m, t in picks.items())
    )
    return picks


_agreed = False


def csa2_mxfp8_agree_tactics() -> bool:
    """Once per process, at the first forward after the warm-up's tuning pass (outside it and any graph capture): every
    rank of the tuner's tensor-parallel group takes the autotuned MXFP8 GEMM tactic most ranks profiled per shape
    (`agree_mxfp8_tactics`), as stock's decode-graph warm-up does for its MXFP8 linears before it captures: an
    attention-DP step waits for its slowest rank, so a rank left on a slower kernel by profiling noise costs them all.
    A setup helper and a collective: the target calls it at the same point of every rank's forward (warm-up forwards
    are collective under attention DP). Returns whether it ran."""
    global _agreed
    if _agreed or not _tuned or torch.cuda.is_current_stream_capturing():
        return False
    from tensorrt_llm._torch.autotuner import AutoTuner

    if AutoTuner.get().is_tuning_mode:
        return False
    from tensorrt_llm._torch.custom_ops.mxfp8_gemm import agree_mxfp8_tactics
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import _MXFP8_GEMM_AUTOTUNED_OP

    _agreed = True
    agree_mxfp8_tactics(_MXFP8_GEMM_AUTOTUNED_OP)
    return True


def _quantized_gemm(data, a_scale, weight, scale_sw, alpha, out=None, native_eager=False):
    """The engine's choice for a quantized activation: the dense CuTe GEMM for wide steps (once an eager call compiled
    it, as graph capture cannot compile), else the autotuned CUTLASS / FlashInfer GEMM (a tactic per row count, tuned at
    warm-up for the registered rows, `csa2_mxfp8_gemm_tune_rows`). A contiguous bf16 `out` receives the result (the
    dense GEMM writes it directly, as stock's `mxfp8_linear_quantized`). `native_eager`: outside decode graphs, stock's
    native projection GEMM at every row count (`native_mxfp8_gemm`: `trtllm::mxfp8_mxfp8_gemm_autotuned`, the linear's
    own autotuned CUTLASS runner), as a CSA2 module whose eager q_a | kv and O-B take it (tep8)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import (
        dense_mxfp8_gemm,
        dense_mxfp8_gemm_compiled,
    )
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_gemm_autotuned

    m, (n, k) = data.shape[0], weight.shape
    capturing = torch.cuda.is_current_stream_capturing()
    if _tune_rows and not capturing and AutoTuner.get().is_tuning_mode:
        _tune_weight(weight, scale_sw)
    from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture_active

    graph = capturing or flashinfer_mxfp8_decode_graph_capture_active()
    if native_eager and not graph:
        y = torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned(
            data, a_scale, weight, scale_sw, alpha, torch.bfloat16
        )
        return y if out is None else out.copy_(y)
    if _dense(m, n, k, graph=graph) and (
        not capturing or dense_mxfp8_gemm_compiled(data.device, n, k)
    ):
        return dense_mxfp8_gemm(data, a_scale, weight, scale_sw, alpha, out=out)
    y = mxfp8_gemm_autotuned(data, a_scale, weight, scale_sw, torch.bfloat16, _backends(m))
    return y if out is None else out.copy_(y)


# Row counts that reach every kernel `_quantized_gemm` can launch without a tuned tactic (a row bucket the AutoTuner
# holds no entry for takes the runner's fallback): up to `_SWAP_AB_MAX_ROWS` rows the swap-AB CuTe DSL GEMM (its first
# tile), past it (below the dense GEMM's rows) the FlashInfer CUTLASS GEMM, whose module the first call JIT-builds with
# nvcc (about 100 s). Both take N, K and M at run time, so one small weight compiles them for every projection; the
# dense CuTe GEMM is compiled per weight by the warm-up's own wide steps.
_PREWARM_ROWS = (16, 200, 512)
_prewarmed: set = set()


def csa2_mxfp8_gemm_prewarm(device: torch.device, n: int = 1024, k: int = 1024) -> Tuple[int, ...]:
    """Compile, once per process and device, every GEMM kernel `_quantized_gemm` (hence `csa2_mxfp8_linear` and
    `csa2_o_projection`) launches for an untuned row count below the dense CuTe GEMM's (a setup helper, the analogue
    of the engine's warm-up prewarms; outside any CUDA-graph capture). Without it the first serving step with such a
    row count compiles them inside the forward (about 20 s per kernel) while the attention-DP peers wait.
    Returns the row counts run (empty when already done)."""
    device = torch.device(device)
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key in _prewarmed or torch.cuda.is_current_stream_capturing():
        return ()
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device=device)
    scale_sw = mxfp8_swizzled_scales(
        torch.full((n, k // _BLOCK), 127, dtype=torch.uint8, device=device)
    )
    alpha = torch.ones(1, dtype=torch.float32, device=device)
    for m in _PREWARM_ROWS:
        data, a_scale = torch.ops.trtllm.mxfp8_quantize(
            torch.zeros(m, k, dtype=torch.bfloat16, device=device), True
        )
        _quantized_gemm(data, a_scale, weight, scale_sw, alpha)
    torch.cuda.synchronize(device)
    _prewarmed.add(key)  # only after every row ran: an interrupted prewarm is retried
    return _PREWARM_ROWS


class Csa2Mxfp8Linear(OpWrapper):
    """`y = x @ W.T` for the checkpoint's MXFP8 `W`, bf16 out; with
    `split_n` the pair `(y[:, :split_n], y[:, split_n:])`.

    `x` bf16 `[M, K]` (unit column stride), `weight` E4M3 `[N, K]`, `scale` uint8 `[N, K/32]`, `scale_sw` its swizzled
    form, `alpha` fp32 `[1]` ones. `rms_norm = (gamma bf16 [K], eps)` projects `rms_norm(x) * gamma`.
    `rope = (pos int32 [M], table fp32 [P, 2, 32], head_dim)` rotates (interleaved pairs, the engine's table) the last
    64 columns of every `head_dim`-wide head of the output. Any M
    works; the kernel is chosen by M (see the module docstring).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: q_a | kv fused projection of six rows, split output",
            spec=dict(m=6, n=1792, k=5120, split_n=1280, graph=True),
        ),
        Cell(
            why="tep8 decode: q_b of 8 heads with the q_norm folded in and the RoPE in the epilogue",
            spec=dict(m=6, n=4096, k=1280, rope=512, rms=True, graph=True),
        ),
        Cell(
            why="tep8 decode: index q_b (32 heads of 128) with RoPE, norm folded in",
            spec=dict(m=6, n=4096, k=1280, rope=128, rms=True, graph=True),
        ),
        Cell(
            why="tep8 decode: o_b, a column slice of one group",
            spec=dict(m=6, n=5120, k=1024, graph=True),
        ),
        Cell(
            why="tep8 decode: the larger part (1664) of this rank's 3200 Engram WKV columns, six rows",
            spec=dict(m=6, n=1664, k=6144, graph=True),
        ),
        Cell(
            why="tep8 verify of four requests (24 rows): the quantized GEMM",
            spec=dict(m=24, n=1792, k=5120, split_n=1280),
        ),
        Cell(
            why="a five-token context chunk (eager): q_a | kv on the quantized GEMM, as stock's prefill projection",
            spec=dict(m=5, n=1792, k=5120, split_n=1280),
        ),
        Cell(
            why=(
                "a five-token context chunk (eager): q_b over the "
                "fused q_norm + MXFP8 quantization, RoPE after the GEMM"
            ),
            spec=dict(m=5, n=4096, k=1280, rope=512, rms=True),
        ),
        Cell(
            why=(
                "tep8 eager q_a | kv (`native_eager`): a 24-row verify "
                "step on stock's native autotuned projection GEMM"
            ),
            spec=dict(m=24, n=1792, k=5120, split_n=1280, native=True),
        ),
        Cell(
            why=(
                "tep8 eager q_a | kv (`native_eager`): a 2048-row context "
                "chunk on stock's native autotuned projection GEMM"
            ),
            spec=dict(m=2048, n=1792, k=5120, split_n=1280, native=True),
        ),
        Cell(
            why="dep32 step of 150 rows: 64-head q_b with RoPE after the GEMM",
            spec=dict(m=150, n=32768, k=1280, rope=512),
        ),
        Cell(
            why="dep8 step of 1280 rows: q_a | kv and o_b over 8 groups",
            spec=dict(m=1280, n=1792, k=5120, split_n=1280),
        ),
        Cell(why="dep8 step of 1280 rows: o_b over 8 groups", spec=dict(m=1280, n=5120, k=8192)),
    )
    note = """
    Small decode-graph steps (cells with `graph`, run under `flashinfer_mxfp8_decode_graph_capture`) are exact
    FP8-weight x bf16-activation products in fp32 (the test gates them against the dequantized fp32 product within bf16
    rounding). Larger steps also quantize the activation to MXFP8 (a 3-bit mantissa), as the engine's own projections do
    (as do small context chunks and eager steps), so their gate is a relative RMS error of 3% against the same product.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        scale_sw: torch.Tensor,
        alpha: torch.Tensor,
        split_n: int = 0,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
        rms_norm: Optional[Tuple[torch.Tensor, float]] = None,
        x_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        native_eager: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        m, k = x.shape
        n = weight.shape[0]
        if _lowm_fits(m, n, k, weight):
            x = x if x.is_contiguous() else x.contiguous()
            from tensorrt_llm._torch.modules.low_m_gemm import mxfp8_lowm_linear

            # The kernel's epilogue RoPE rotates each column with its pair partner, and only the column itself carries
            # the folded norm's per-row 1/rms: with both folded in, rows whose RMS is not ~1 come out wrong. As the
            # engine's small-M query projection does, fold the norm only and rotate after the GEMM.
            fold_rope = rope is not None and rms_norm is None
            if rope is None or fold_rope:
                return mxfp8_lowm_linear(
                    x,
                    weight,
                    scale,
                    split_n=split_n,
                    rope=rope if fold_rope else None,
                    rms_norm=rms_norm,
                )
            # a separate RoPE rotates the whole output, then it is split (as the eager path and the reference do)
            y = mxfp8_lowm_linear(x, weight, scale, rms_norm=rms_norm)
            csa2_rope_inplace(
                y.view(m, n // rope[2], rope[2]), rope[0], rope[1].reshape(rope[1].shape[0], -1)
            )
            return (y[:, :split_n], y[:, split_n:]) if split_n else y
        if x_mxfp8 is not None and rms_norm is None:
            # `x` as its producer already quantized it (`mxfp8_quantize(x, True)`: the mHC boundary's MXFP8 output)
            data, a_scale = x_mxfp8
        elif rms_norm is not None and k % 128 == 0 and k <= 8192:
            # the norm and the MXFP8 quantization of its output in one launch (the engine's fused query norm)
            from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8

            _, data, a_scale = rms_norm_mxfp8(x, rms_norm[0], rms_norm[1], normalized=False)
        else:
            if rms_norm is not None:
                from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
                    flashinfer_rmsnorm,
                )

                x = flashinfer_rmsnorm(x, rms_norm[0], rms_norm[1])
            # the engine's activation quantizer (`mxfp8_quantize_activation`: the CuTe-DSL kernel for wide steps, the
            # native one below them), as stock's prefill projections quantize their input
            from tensorrt_llm._torch.modules.linear import mxfp8_quantize_activation

            data, a_scale = mxfp8_quantize_activation(x if x.is_contiguous() else x.contiguous())
        y = _quantized_gemm(data, a_scale, weight, scale_sw, alpha, native_eager=native_eager)
        if rope is not None:
            csa2_rope_inplace(
                y.view(m, n // rope[2], rope[2]), rope[0], rope[1].reshape(rope[1].shape[0], -1)
            )
        return (y[:, :split_n], y[:, split_n:]) if split_n else y

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        scale_sw: torch.Tensor,
        alpha: torch.Tensor,
        split_n: int = 0,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
        rms_norm: Optional[Tuple[torch.Tensor, float]] = None,
        x_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        native_eager: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """fp32 product with the dequantized weight (activation unquantized),
        then the RoPE of `csa2_rope_tail.reference`."""
        m, k = x.shape
        a = x.float()
        if rms_norm is not None:
            a = (
                a
                * torch.rsqrt(a.square().mean(-1, keepdim=True) + rms_norm[1])
                * rms_norm[0].float()
            )
        y = (a @ _dequant(weight, scale).t()).to(torch.bfloat16)
        if rope is not None:
            n, hd = y.shape[1], rope[2]
            table = torch.stack([rope[1][:, 0], rope[1][:, 1]], dim=-1)  # [P, 32, 2] (cos, sin)
            y = csa2_rope_tail.reference(y.view(m, n // hd, hd), table, rope[0], 64).reshape(m, n)
        return (y[:, :split_n], y[:, split_n:]) if split_n else y

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        scale_sw: torch.Tensor,
        alpha: torch.Tensor,
        split_n: int = 0,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
        rms_norm: Optional[Tuple[torch.Tensor, float]] = None,
        x_mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        native_eager: bool = False,
    ) -> None:
        m, k = x.shape
        n = weight.shape[0]
        assert x.dtype == torch.bfloat16 and x.stride(-1) == 1, (
            "x is bf16 with unit column stride (rows may be strided)"
        )
        assert (
            weight.dtype == torch.float8_e4m3fn
            and weight.shape[1] == k
            and k % 128 == 0
            and n % 128 == 0
        ), "weight is E4M3 [N, K] with N, K multiples of 128 (whole swizzle tiles)"
        assert scale.dtype == torch.uint8 and scale.shape == (n, k // 32) and scale.is_contiguous()
        assert alpha.dtype == torch.float32 and alpha.numel() == 1
        assert split_n % 8 == 0 and split_n < n or split_n == 0
        if rope is not None:
            assert (
                rope[0].dtype == torch.int32
                and rope[1].dtype == torch.float32
                and rope[1].shape[1:] == (2, 32)
            )
            assert n % rope[2] == 0 and rope[2] > 64
        if rms_norm is not None:
            assert rms_norm[0].dtype == torch.bfloat16 and rms_norm[0].shape == (k,)


csa2_mxfp8_linear = Csa2Mxfp8Linear()
