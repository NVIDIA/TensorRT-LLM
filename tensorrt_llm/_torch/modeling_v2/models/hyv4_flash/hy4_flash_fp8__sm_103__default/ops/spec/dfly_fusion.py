# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFly2's slot-0 fusion, its per-position hidden correction and the confidence head.

Two entries, because the drafter calls them at different points of one draft
block: `dfly_fusion` once, on the anchor row of each request, before the five
draft layers; `dfly_hidden_correction` once per drafted position, on that
position's draft hidden and the previous token's embedding. Both are **macro
entries**: `__call__` composes the target's `cublas_mm`, the catalog's
`flashinfer_rmsnorm` and native torch elementwise ops, because no single kernel
in this tree, in FlashInfer, in vLLM or in SGLang computes either formula (both
engines compose the same pieces: vLLM `HYV4DFlyFusion`/`HYV4DFlyHiddenCorrection`,
SGLang `HYV4FlashDFlyFusion`/`HYV4FlashDFlyHiddenCorrection`).

The checkpoint stores the tensors as ``model.dfly_layers.mtp_fusion.{enorm,
hnorm,eh_proj,gate}``, ``model.dfly_layers.hidden_correction.{hidden_norm,
embed_norm,gate_proj,up_proj,down_proj}`` and ``model.dfly_layers.
confidence_head.proj.{weight,bias}``, all BF16 at hidden 3072 with the
correction's intermediate width also 3072 (``gate_proj``/``up_proj`` are
``[3072, 6144]``: they read the two concatenated 3072-wide normalized streams,
they are not a packed ``gate_up``). Weights are passed in the checkpoint's own
``[out, in]`` orientation; the transpose into `cublas_mm`'s column-major
operand happens here, so no transpose kernel runs.

The confidence head is ``AcceptRatePredictor`` (AngelSpec
``models/draft/dspark.py``): a linear projection of the **corrected** hidden to
one logit per row, trained with BCE-with-logits. It returns the logit, as the
checkpointed module does; the accept probability is its sigmoid, which the
caller applies (the engine's calibration bins in `accept_stats.py` are in
[0, 1]).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.gemm.cublas_mm__r11 import (  # noqa: E501
    cublas_mm,
)

#: `rms_norm_eps` of this checkpoint's DFly config (`config.json`, and AngelSpec's
#: `dfly2_hy_v4_flash_draft_config.json`). Passed in, never read off `self`.
DEFAULT_EPS = 1e-5


def _rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`x / sqrt(mean(x^2) + eps) * weight` in fp32 -- `reference`'s norm."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()


def _assert_last_dim_contiguous(x: torch.Tensor, name: str) -> None:
    # The rmsnorm kernel reads `stride(-1) == 1` and answers with the wrong
    # elements otherwise rather than raising; a `[rows, width, hidden]` block's
    # `[:, i]` slice is contiguous on the last dim and is the shape a caller has.
    assert x.stride(-1) == 1, (
        f"{name}'s last dim must be contiguous; stride {x.stride(-1)} reads the "
        f"wrong elements in the normalization rather than raising"
    )


def _assert_weight(w: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    # `cublas_mm` reads its second operand column-major and takes its leading
    # dimension from the constant `k` rather than from the operand's stride
    # (`cpp/tensorrt_llm/thop/cublasScaledMM.cpp`, `cublas_gemm_caller`), while
    # its own check is only `mat_b.strides()[0] == 1` (`cublas_mm_out`). A
    # weight sliced from a wider buffer -- same shape, `w.t()` still
    # `stride(0) == 1` -- is therefore accepted and read at the wrong leading
    # dimension: a plausible wrong product, not an error. Only a dense
    # row-major `w` makes `w.t()` the operand the kernel reads correctly.
    assert w.shape == shape, f"{name} must be {shape}; got {tuple(w.shape)}"
    assert w.is_contiguous(), f"{name} must be dense row-major; `w.t()` is its column-major operand"


class DflyFusion(OpWrapper):
    """`slot0 = anchors + gate * eh_proj(cat(enorm(anchors), hnorm(target_hidden)))`.

    `anchors [rows, H]` is the anchor token's embedding, `target_hidden [rows, H]`
    the target model's hidden for the same row; `enorm`/`hnorm [H]` are RMSNorm
    weights, `eh_proj [H, 2H]` the fused projection and `gate [H]` the per-channel
    gate on its output. vLLM and SGLang compute exactly this and both apply it to
    the block's slot 0 only (`embeddings[::width]`), so `rows` is the number of
    draft requests, not the number of block rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The rows the drafter runs at: one request, one width-6 draft block, the
    # largest cuda-graph batch this run serves, and a whole 64-request block
    # (64 x width 6).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="slot-0 fusion: one draft request, the per-step decode shape",
            spec=dict(rows=1, dtype=torch.bfloat16),
        ),
        Cell(
            why="slot-0 fusion: one width-6 draft block",
            spec=dict(rows=6, dtype=torch.bfloat16),
        ),
        Cell(
            why="slot-0 fusion: the largest cuda-graph batch, 64 draft requests",
            spec=dict(rows=64, dtype=torch.bfloat16),
        ),
        Cell(
            why="slot-0 fusion: 64 requests x width 6, the largest block the engine materializes",
            spec=dict(rows=384, dtype=torch.bfloat16),
        ),
    )

    note = """
    A macro entry. `__call__` launches, in order: two `flashinfer_rmsnorm`
    calls (`torch.ops.trtllm.flashinfer_rmsnorm`, a compiled-in trtllm op that
    dispatches to flashinfer's CuTe DSL path), one `cublas_mm`
    (`torch.ops.trtllm.cublas_mm`, compiled-in C++/cuBLASLt), a `torch.cat`, and
    the gate multiply and residual add as torch elementwise kernels. Searched
    for one fused kernel instead: `torch.ops.trtllm` (no norm+concat+gemm+gated
    residual fusion exists), the catalog (`norm/`, `gemm/`, `residual/` -- the
    residual category is empty), FlashInfer (`rmsnorm`, `silu_and_mul`; no
    gated-residual fusion), vLLM and SGLang (both compose the same calls). The
    composition is therefore the entry.

    The reference is the same formula in fp32, rounded once; the kernel path
    rounds after each norm and after the projection, which is what the test's
    band covers.

    Certified on sm_103 at 1, 6, 64 and 384 rows. Measured against the
    reference, worst across the cells: 2.60 of the test's 4-ulp element band
    and 0.92 of its 2-ulp relative-RMS band; the wrong-formula controls (a
    different gate, a dropped residual, the swapped norms) land at 296 to 484
    element ulp, so the band separates them. `gate` is a [H] vector, not a
    scalar and not [rows, 1]: the checkpoint's `mtp_fusion.gate` is [3072] with
    rms 0.163 and absmax 0.236. fp16 is accepted by every constituent and is not
    certified here; this checkpoint is BF16 throughout. `eps` is the config's
    1e-5.
    """

    def __call__(
        self,
        anchors: torch.Tensor,
        target_hidden: torch.Tensor,
        enorm: torch.Tensor,
        hnorm: torch.Tensor,
        eh_proj: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        fused = cublas_mm(
            torch.cat(
                (
                    flashinfer_rmsnorm(anchors, enorm, eps),
                    flashinfer_rmsnorm(target_hidden, hnorm, eps),
                ),
                -1,
            ),
            eh_proj.t(),
        )
        return anchors + gate * fused

    def reference(
        self,
        anchors: torch.Tensor,
        target_hidden: torch.Tensor,
        enorm: torch.Tensor,
        hnorm: torch.Tensor,
        eh_proj: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """fp32 throughout, rounded once -- the more accurate side of the band."""
        with true_fp32_matmul():
            fused = (
                torch.cat(
                    (
                        _rmsnorm_fp32(anchors, enorm, eps),
                        _rmsnorm_fp32(target_hidden, hnorm, eps),
                    ),
                    -1,
                )
                @ eh_proj.float().t()
            )
            out = anchors.float() + gate.float() * fused
        return out.to(anchors.dtype)

    def is_valid(
        self,
        anchors: torch.Tensor,
        target_hidden: torch.Tensor,
        enorm: torch.Tensor,
        hnorm: torch.Tensor,
        eh_proj: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> None:
        # Only the inputs the entry takes and answers plausibly wrongly with.
        # Row/width/dtype mismatches between the operands raise in `cat`, the
        # norm or the gemm, so they are not restated here.
        _assert_last_dim_contiguous(anchors, "anchors")
        _assert_last_dim_contiguous(target_hidden, "target_hidden")
        hidden = enorm.numel()
        assert hnorm.shape == (hidden,), f"hnorm must be [{hidden}]; got {tuple(hnorm.shape)}"
        _assert_weight(eh_proj, (hidden, 2 * hidden), "eh_proj")
        # A [rows, 1] or [1, H] gate broadcasts into a per-row or uniform scale
        # and returns a plausible wrong fusion instead of raising.
        assert gate.shape == (hidden,), (
            f"gate must be the per-channel [{hidden}] vector the checkpoint stores; "
            f"{tuple(gate.shape)} broadcasts and answers a different fusion"
        )


class DflyHiddenCorrection(OpWrapper):
    """`corrected = hidden + down(silu(gate(combined)) * up(combined))`,
    `confidence = confidence_proj @ corrected + confidence_bias`, with
    `combined = cat(hidden_norm(hidden), embed_norm(previous_embed))`.

    `hidden [rows, H]` is the draft hidden at the position being corrected,
    `previous_embed [rows, H]` the embedding of the token before it (the anchor
    token's for the first drafted position); both norms and the three
    projections are the checkpoint's `hidden_correction` tensors, and
    `confidence_proj [1, H]`/`confidence_bias [1]` the `confidence_head.proj`
    linear. The confidence head reads the corrected hidden, as AngelSpec's
    `_compute_draft_logits` does, and returns a logit per row (`[rows]`).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The same rows: the correction runs once per drafted position, so its row
    # count is the block's row count at every width.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="hidden correction: one draft request, the per-step decode shape",
            spec=dict(rows=1, dtype=torch.bfloat16),
        ),
        Cell(
            why="hidden correction: one width-6 draft block",
            spec=dict(rows=6, dtype=torch.bfloat16),
        ),
        Cell(
            why="hidden correction: the largest cuda-graph batch, 64 draft requests",
            spec=dict(rows=64, dtype=torch.bfloat16),
        ),
        Cell(
            why="hidden correction: 64 requests x width 6, the largest block the engine materializes",
            spec=dict(rows=384, dtype=torch.bfloat16),
        ),
    )

    note = """
    A macro entry. `__call__` launches: two `flashinfer_rmsnorm`, a `torch.cat`,
    two `cublas_mm` (the checkpoint's separate `gate_proj` and `up_proj`, not a
    packed `gate_up`), `F.silu` and the multiply, one more `cublas_mm` for
    `down_proj`, the residual add, and one `F.linear` for the 3072->1 confidence
    head. Searched for a fused replacement: the catalog's `silu_and_mul` takes
    one packed `[rows, 2*intermediate]` tensor and would need a `cat` of the two
    projections first, which is the same two launches; no in-tree, FlashInfer,
    vLLM or SGLang kernel fuses norm+concat+gate_up+silu+mul+down+residual.

    The confidence head stays a plain `F.linear`: at `n = 1` the target's
    `cublas_mm` has no certified cell and a GEMV kernel is the faster path, but
    the only GEMV entries in this tree are NVFP4 (`qwen4_exp/ops/gemm/
    nvfp4_gemv.py`) and this checkpoint's DFly tensors are BF16.

    Certified on sm_103 at 1, 6, 64 and 384 rows. Measured against the
    reference, worst across the cells: the corrected hidden 1.98 of the test's
    8-ulp element band and 1.08 of its 4-ulp relative-RMS band, the confidence
    logit 1.82 of its 4-ulp element band and 1.67 of its 2-ulp RMS band. The
    wrong-formula controls (swapped gate/up, a dropped residual, the confidence
    head on the uncorrected hidden) land at 194 to 710 element ulp.
    `gate_proj` and `up_proj` have the same shape, so swapping them is not
    detectable here and is a load-time mapping, not a guard. fp16 is accepted by
    the constituents and is not certified; `eps` is the config's 1e-5.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        previous_embed: torch.Tensor,
        hidden_norm: torch.Tensor,
        embed_norm: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        confidence_proj: torch.Tensor,
        confidence_bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat(
            (
                flashinfer_rmsnorm(hidden, hidden_norm, eps),
                flashinfer_rmsnorm(previous_embed, embed_norm, eps),
            ),
            -1,
        )
        delta = cublas_mm(
            F.silu(cublas_mm(combined, gate_proj.t())) * cublas_mm(combined, up_proj.t()),
            down_proj.t(),
        )
        corrected = hidden + delta
        confidence = F.linear(corrected, confidence_proj, confidence_bias).squeeze(-1)
        return corrected, confidence

    def reference(
        self,
        hidden: torch.Tensor,
        previous_embed: torch.Tensor,
        hidden_norm: torch.Tensor,
        embed_norm: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        confidence_proj: torch.Tensor,
        confidence_bias: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 throughout, rounded once -- the more accurate side of the band."""
        with true_fp32_matmul():
            combined = torch.cat(
                (
                    _rmsnorm_fp32(hidden, hidden_norm, eps),
                    _rmsnorm_fp32(previous_embed, embed_norm, eps),
                ),
                -1,
            )
            delta = (
                F.silu(combined @ gate_proj.float().t()) * (combined @ up_proj.float().t())
            ) @ (down_proj.float().t())
            corrected = hidden.float() + delta
            confidence = corrected @ confidence_proj.float().t() + confidence_bias.float()
        return corrected.to(hidden.dtype), confidence.squeeze(-1).to(hidden.dtype)

    def is_valid(
        self,
        hidden: torch.Tensor,
        previous_embed: torch.Tensor,
        hidden_norm: torch.Tensor,
        embed_norm: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        confidence_proj: torch.Tensor,
        confidence_bias: torch.Tensor,
        eps: float,
    ) -> None:
        # Only the inputs the entry takes and answers plausibly wrongly with.
        _assert_last_dim_contiguous(hidden, "hidden")
        _assert_last_dim_contiguous(previous_embed, "previous_embed")
        width = hidden_norm.numel()
        assert embed_norm.shape == (width,), (
            f"embed_norm must be [{width}]; got {tuple(embed_norm.shape)}"
        )
        for name, weight in (
            ("gate_proj", gate_proj),
            ("up_proj", up_proj),
            ("down_proj", down_proj),
        ):
            _assert_weight(weight, (width, 2 * width if name != "down_proj" else width), name)
        _assert_weight(confidence_proj, (1, width), "confidence_proj")
        assert confidence_bias.shape == (1,), (
            f"confidence_bias must be [1]; got {tuple(confidence_bias.shape)}"
        )


dfly_fusion = DflyFusion()
dfly_hidden_correction = DflyHiddenCorrection()
