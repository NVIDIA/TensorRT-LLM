# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One lagged Hyper-Connections boundary in a single DeepGEMM launch (`sm100_mega_mhc_impl`): the previous
sub-layer's post update, this sub-layer's mixes and Sinkhorn, the collapse with the previous `pre`, and the RMSNorm.
Stock runs it for context-sized boundaries (`mhc_fused_hc`, from 1024 rows)."""

from __future__ import annotations

from functools import lru_cache

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_mapping import (
    hc_post_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre_mapping import (
    hc_pre_mapping,
)

MIN_ROWS = 1024  # stock's floor for this kernel: below it the split-K waves are mostly empty
MAX_ROWS = 1 << 20

# Streams on which the kernel already ran eagerly: its first launch on a stream allocates that stream's split
# barriers, which a graph capture forbids, so a capture may take it only on one of these.
_WARM_STREAMS: set = set()


# DeepGEMM compiles the kernel per split-K count (a template parameter), which it derives from the row count's
# 64-row blocks and the SM count; a count first met in a serving step compiles there (~3.5 s while the attention-DP
# peer waits). Every count a row count of 1024 .. 2^20 can take is already taken by 1024 .. 4096 rows (16 .. 64
# blocks, for 100 .. 200 SMs), so `hc_mega_boundary_prewarm` runs those once.
PREWARM_ROWS = tuple(64 * m for m in range(MIN_ROWS // 64, 65))
_prewarmed: set = set()


@lru_cache(maxsize=None)
def _kernel():
    try:
        from tensorrt_llm.deep_gemm import mega_mhc
    except ImportError:
        return None
    return mega_mhc


class HcMegaBoundary(OpWrapper):
    """`(x_prev [n, d], residual [n, hc, d], post_prev [n, hc], comb_prev [n, hc, hc], pre_mix [n, hc], ...)` ->
    `(residual_cur [n, hc, d], pre [n, hc], post [n, hc], comb [n, hc, hc], h [n, d])`.

    The same function as `hc_boundary` with a post update, in DeepGEMM's one-kernel form:

        residual_cur = hc_post_mapping(x_prev, residual, post_prev, comb_prev)
        pre, post, comb, h = hc_pre_mapping(residual_cur, pre_mix, ...)

    i.e. the previous sub-layer's output mixed into the residual, this sub-layer's own mixes, and
    `h = rmsnorm(collapse(residual_cur, pre_mix))` collapsed with the *previous* `pre`. `pre`, `post`, `comb` fp32,
    `h` and `residual_cur` bf16. `hc_fn` is the checkpoint's `[24, 4 * d]` fp32 mixer, used as is.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="the smallest context boundary the target sends here (1024 rows)",
            spec=dict(n=1024, d=5120),
        ),
        Cell(
            why="dep2 decoder half of a bounded replay: 34 requests x 140 rows / 2 ranks",
            spec=dict(n=2380, d=5120),
        ),
        Cell(why="dep2 context tile: 8192 rows", spec=dict(n=8192, d=5120)),
        Cell(why="dep2 context step: 32300 rows (34 x 1900 / 2 ranks)", spec=dict(n=32300, d=5120)),
    )
    note = """
    The mixer product is TF32-class with split-K partial sums reduced in a fixed order (the split count follows the
    row count, so a row's mixes may change in the last bits with the batch size; a given size repeats bitwise).
    Sinkhorn and the RMS statistics are fp32 in-kernel. The op allocates only its outputs and syncs nothing on the
    host. The first launch on a stream allocates that stream's split barriers: `available` refuses a capture on a
    stream the kernel has not run on eagerly, and an eager launch also runs once on `torch.cuda.graph`'s default
    capture stream (as stock does), so later captures there can take it. With `mxfp8` the same launch also writes `h`
    in MXFP8 as `mxfp8_quantize(h, True)` gives it (E4M3 values, swizzled UE8M0 scales; the padding rows of the last
    128-row scale tile are zeroed, as stock's fused lagged boundary leaves them), the q_a | kv GEMM's input.
    """

    @staticmethod
    def available(n: int, hidden: int) -> bool:
        """Whether a call over `n` rows can be made now: this device, the row range, and inside a capture, a stream
        the kernel already ran on."""
        from tensorrt_llm._torch.autotuner import AutoTuner

        if not (MIN_ROWS <= n <= MAX_ROWS and hidden % 1024 == 0):
            return False
        if (
            torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7))
            or _kernel() is None
        ):
            return False
        if AutoTuner.get().is_tuning_mode:
            return False
        return (
            not torch.cuda.is_current_stream_capturing()
            or torch.cuda.current_stream().cuda_stream in _WARM_STREAMS
        )

    def __call__(
        self,
        x_prev: torch.Tensor,
        residual: torch.Tensor,
        post_prev: torch.Tensor,
        comb_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
        mxfp8: bool = False,
    ) -> tuple:
        """-> (residual_cur, pre, post, comb, h), with `mxfp8` also `(h E4M3, swizzled scales)`."""
        n, hc, d = residual.shape
        res = torch.empty_like(residual)
        pre = torch.empty(n, hc, dtype=torch.float32, device=residual.device)
        post = torch.empty(n, hc, dtype=torch.float32, device=residual.device)
        comb = torch.empty(n, hc, hc, dtype=torch.float32, device=residual.device)
        h = torch.empty_like(x_prev)
        fp8 = {}
        if mxfp8:
            values = torch.empty_like(x_prev, dtype=torch.float8_e4m3fn)
            scales = torch.empty(
                -(-n // 128) * 128 * (d // 32), dtype=torch.uint8, device=x_prev.device
            )
            if n % 128:
                scales[n // 128 * 128 * (d // 32) :].zero_()
            fp8 = dict(y_fp8=values, y_tiled_sf=scales)

        def launch():
            _kernel()(
                x=x_prev, residual=residual, shifted_prev_mix=pre_mix.view(n, hc, 1), post_mix=post_prev.view(n, hc, 1),
                comb_res_mix=comb_prev.view(n, hc, hc), fn=hc_fn, mix_scales=hc_scale, mix_bases=hc_base,
                hc_mult=hc_mult, hc_norm_eps=rms_eps, hc_pre_eps=hc_eps, hc_post_scale=post_mult, sinkhorn_eps=hc_eps,
                num_sinkhorn_iters=sinkhorn_iters, rmsnorm_weight=norm_weight, rmsnorm_eps=norm_eps,
                rmsnorm_scale=1.0, new_residual=res, new_prev_mix=pre.view(n, hc, 1), new_post_mix=post.view(n, hc, 1),
                new_comb_res_mix=comb, y_bf16=h, **fp8,
            )  # fmt: skip

        launch()
        if not torch.cuda.is_current_stream_capturing():
            _WARM_STREAMS.add(torch.cuda.current_stream().cuda_stream)
            capture = torch.cuda.graph.default_capture_stream
            if capture is not None and capture.cuda_stream not in _WARM_STREAMS:
                # The same outputs again on the capture stream: its barriers are allocated outside any graph.
                capture.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture):
                    launch()
                torch.cuda.current_stream().wait_stream(capture)
                _WARM_STREAMS.add(capture.cuda_stream)
        return (res, pre, post, comb, h) + (((fp8["y_fp8"], fp8["y_tiled_sf"]),) if mxfp8 else ())

    def reference(
        self,
        x_prev: torch.Tensor,
        residual: torch.Tensor,
        post_prev: torch.Tensor,
        comb_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
        mxfp8: bool = False,
    ) -> tuple:
        """The two entries it fuses, composed: `hc_post_mapping` then `hc_pre_mapping` (both float64 references); with
        `mxfp8` also `mxfp8_quantize(h, True)`."""
        cur = hc_post_mapping.reference(x_prev, residual, post_prev, comb_prev)
        pre, post, comb, h = hc_pre_mapping.reference(
            cur, pre_mix, hc_fn, hc_scale, hc_base, norm_weight, hc_mult, sinkhorn_iters, rms_eps, norm_eps,
            hc_eps, post_mult,
        )  # fmt: skip
        if mxfp8:
            return (
                cur,
                pre,
                post,
                comb,
                h,
                torch.ops.trtllm.mxfp8_quantize(h.bfloat16().contiguous(), True),
            )
        return cur, pre, post, comb, h

    def is_valid(
        self,
        x_prev: torch.Tensor,
        residual: torch.Tensor,
        post_prev: torch.Tensor,
        comb_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
        mxfp8: bool = False,
    ) -> None:
        assert residual.dim() == 3 and residual.shape[1] == hc_mult == 4, "residual is [n, 4, d]"
        n, _, d = residual.shape
        assert residual.dtype == torch.bfloat16 and residual.is_cuda
        assert d % 1024 == 0, "the kernel is instantiated for hidden sizes in 1024 steps"
        assert MIN_ROWS <= n <= MAX_ROWS, "rows outside the range the target sends here"
        assert x_prev.shape == (n, d) and x_prev.dtype == torch.bfloat16
        assert post_prev.shape == (n, hc_mult) and post_prev.dtype == torch.float32
        assert comb_prev.shape == (n, hc_mult, hc_mult) and comb_prev.dtype == torch.float32
        assert pre_mix.shape == (n, hc_mult) and pre_mix.dtype == torch.float32
        tensors = [x_prev, residual, post_prev, comb_prev, pre_mix, norm_weight, hc_fn]
        assert all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors), (
            "16-byte aligned, contiguous"
        )
        assert norm_weight.shape == (d,) and norm_weight.dtype == torch.bfloat16
        mix = (2 + hc_mult) * hc_mult
        assert hc_fn.shape == (mix, hc_mult * d) and hc_fn.dtype == torch.float32
        assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
        assert hc_base.shape == (mix,) and hc_base.dtype == torch.float32
        assert sinkhorn_iters >= 1 and post_mult == 2.0, (
            "the reference entries fix the post scale at 2"
        )


hc_mega_boundary = HcMegaBoundary()


def hc_mega_boundary_prewarm(device: torch.device, hidden: int, hc_mult: int = 4) -> tuple:
    """Compile, once per process and device, the kernel of every split-K count (a setup helper, the analogue of the
    engine's warm-up prewarms; outside any CUDA-graph capture and autotuning): one launch over zeros per row count of
    `PREWARM_ROWS`, for the bf16 output and for the bf16 plus MXFP8 output. Returns the row counts run (empty when
    already done or the kernel is not available)."""
    device = torch.device(device)
    key = (device.index if device.index is not None else torch.cuda.current_device(), hidden)
    if key in _prewarmed or not hc_mega_boundary.available(PREWARM_ROWS[-1], hidden):
        return ()
    if torch.cuda.is_current_stream_capturing():
        return ()
    _prewarmed.add(key)
    n, mix = PREWARM_ROWS[-1], (2 + hc_mult) * hc_mult
    f32 = dict(dtype=torch.float32, device=device)
    x = torch.zeros(n, hidden, dtype=torch.bfloat16, device=device)
    residual = torch.zeros(n, hc_mult, hidden, dtype=torch.bfloat16, device=device)
    post, comb, pre = (
        torch.zeros(n, hc_mult, **f32),
        torch.zeros(n, hc_mult, hc_mult, **f32),
        torch.zeros(n, hc_mult, **f32),
    )
    fn, scale, base = (
        torch.zeros(mix, hc_mult * hidden, **f32),
        torch.ones(3, **f32),
        torch.zeros(mix, **f32),
    )
    norm = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    # both outputs the targets take: bf16 only, and bf16 plus the next GEMM's MXFP8 input (another kernel variant)
    for mx in (False, True):
        for r in PREWARM_ROWS:
            hc_mega_boundary(
                x[:r],
                residual[:r],
                post[:r],
                comb[:r],
                pre[:r],
                fn,
                scale,
                base,
                norm,
                hc_mult,
                1,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                mxfp8=mx,
            )
    torch.cuda.synchronize(device)
    return PREWARM_ROWS
