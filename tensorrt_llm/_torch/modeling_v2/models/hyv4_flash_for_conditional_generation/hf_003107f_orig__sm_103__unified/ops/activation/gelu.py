# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-tower GELU: the exact (erf) form `ACT2FN["gelu"]` evaluates.

This checkpoint's `vision_config.hidden_act` is `"gelu"` (`config.json`), and the
reference tower resolves it through `ACT2FN` at both call sites --
`HYV4VLVisionMLP.activation_fn` and `HYV4VLVisionPatchMerger.proj_act` in
`hy_v4_vl/modeling_hy_v4_vl.py` -- i.e. `nn.GELU(approximate="none")`:
`0.5 * x * (1 + erf(x / sqrt(2)))`. The tanh approximation
(`gelu_pytorch_tanh`) is a different function and is not this entry.

Native torch (rung 4 of the sourcing ladder): one `torch.nn.functional.gelu(x)`
call, `approximate="none"` -- one elementwise CUDA kernel in the torch wheel,
the same call the reference tower makes. Adapted from
`models/qwen4_exp/ops/activation/gelu_erf.py`, which wraps the same call for
qwen4_exp's vision merger; the cells here are this checkpoint's two call sites
and row counts.

Searched, nothing better: `torch.ops.trtllm.moe_gelu` is this function but is
the MoE tail's fused activation -- `moe_gelu(Tensor input, Tensor
tile_idx_to_mn_limit, Tensor num_non_exiting_tiles, int tile_tokens_dim)` in
`cpp/tensorrt_llm/thop/moe/cuteDsl/cuteDslMoeUtilsOp.cpp`, where the int32 tile
metadata steers which rows are computed, so it cannot stand in for a plain
elementwise call. FlashInfer's `activation` module exports only gated forms
(`gelu_and_mul`, `gelu_tanh_and_mul`, `silu_and_mul`), as does
`torch.ops.trtllm.flashinfer_gelu_tanh_and_mul` -- and that one is the tanh
approximation. The in-tree CuTe DSL `cute_dsl_nvfp4_dense_gemm_gelu_*` ops fuse
a tanh GELU into a GEMM epilogue.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The vision tower's `intermediate_size`: the block MLP's fc1 output width.
MLP_WIDTH = 4304

#: The merger's `hidden_size * spatial_merge_size**2`: its 2x2 conv output width.
MERGER_WIDTH = 4608

_SQRT_2 = math.sqrt(2.0)


class Gelu(OpWrapper):
    """`0.5 * x * (1 + erf(x / sqrt(2)))` elementwise, any shape, a new tensor
    out in `x`'s dtype.

    Fusion boundary: the activation only. The block MLP's `fc1`/`fc2` and the
    merger's 2x2 `proj.0` / 1x1 `proj.2` around it are the caller's; there is no
    gate half and no multiply. This is the erf GELU and not the tanh one.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: shape, dtype; `dist` "uniform" over [lo, hi] (default), "tower" (the
    # block MLP's fc1 output: a LayerNorm'd hidden at the checkpoint's gamma
    # scale through a synthetic dense_h_to_4h at its measured weight/bias
    # statistics) or "merger" (the merger's proj.0 output: perceive.norm's
    # gamma tiled over the 2x2 window through a synthetic 4608-wide projection
    # at its measured statistics).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="block MLP at 512x512: 1024 patch rows x 4304, the tower's own activand",
            spec=dict(shape=(1024, MLP_WIDTH), dtype=torch.bfloat16, dist="tower"),
        ),
        Cell(
            why="block MLP at 1024x1024: 4096 patch rows x 4304 bf16",
            spec=dict(shape=(4096, MLP_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
        Cell(
            why="block MLP envelope: 16384 patch rows (max_vit_seq_len) x 4304, the tower's own activand",
            spec=dict(shape=(16384, MLP_WIDTH), dtype=torch.bfloat16, dist="tower"),
        ),
        Cell(
            why="merger at 512x512: 256 merged blocks x 4608 bf16",
            spec=dict(shape=(256, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
        Cell(
            why="merger at 1024x1024: 1024 merged blocks x 4608, built the way the merger builds it",
            spec=dict(shape=(1024, MERGER_WIDTH), dtype=torch.bfloat16, dist="merger"),
        ),
        Cell(
            why="merger envelope: 4096 merged blocks from 16384 patches x 4608 bf16",
            spec=dict(shape=(4096, MERGER_WIDTH), dtype=torch.bfloat16, lo=-16.0, hi=16.0),
        ),
    )

    note = """
    Metadata consumed: none. Out-of-place; `x` is not mutated. Source:
    `torch.nn.functional.gelu` with `approximate="none"` -- native torch (rung 4
    of the sourcing ladder), one ATen elementwise CUDA kernel compiled into the
    torch wheel: no JIT path, no autotuner, no native library of this tree's.

    Identity: `ACT2FN["gelu"]` is the exact erf form, which is what the
    reference tower computes at both call sites (`HYV4VLVisionMLP.activation_fn`
    and `HYV4VLVisionPatchMerger.proj_act`). The tanh approximation differs by
    4.7e-4 at its worst in fp32, tens of times outside torch's fp32 band, and by
    less than one bf16 ulp at the output scale -- so a bf16 match cannot say
    which GELU was computed; the test's fp32 identity test is what pins it.

    Call sites (reference: `hy_v4_vl/modeling_hy_v4_vl.py`; this target's
    `Hyv4FlashVisionTower`):
      - the block MLP's activation, `[patch_rows, 4304]` bf16, once per block
        per frame: 1024, 4096 and 16384 patch rows for a 512^2, 1024^2 and
        2048^2 image (patch 16, `max_vit_seq_len` 16384);
      - the merger's activation, `[merged_blocks, 4608]` bf16, once per frame:
        256, 1024 and 4096 blocks (spatial merge 2, `img_max_token_num` 4096).
    The vision tower is replicated, so every rank runs the same shapes.

    Input distribution, measured from the checkpoint's bf16 tensors:
    `vit.layers.0.mlp.dense_h_to_4h.weight` std 0.018955, `.bias` mean -2.416808
    std 1.668815; `vit.layers.0.post_attention_layernorm.weight` mean 1.511802
    std 2.353820; `vit.perceive.proj.0.weight` std 0.008602, `.bias` mean
    -0.000401 std 0.003034; `vit.perceive.norm.weight` mean 1.084490 std
    0.029593. The block MLP's activand is therefore centred near -2.4 with a
    spread of about 2.5, the merger's near 0 with a spread of about 0.6; two
    cells build each through a synthetic layer with those statistics, the rest
    sweep [-16, 16], wider than either.

    Measured on GB300 (sm_103) against `reference`, bitwise-equal share /
    relative L2 / worst element as a fraction of one bf16 ulp of its row's
    largest element: block MLP 1024 rows 81.564% / 3.42e-08 / 0.000, 4096 rows
    88.219% / 1.89e-09 / 0.000, 16384 rows 82.100% / 3.38e-08 / 0.000; merger
    256 rows 88.234% / 1.88e-09 / 0.000, 1024 rows 100.000% / 0.00e+00 / 0.000,
    4096 rows 88.242% / 1.89e-09 / 0.000. The elements that differ are
    single-rounding ties that flipped between fp32-internal evaluation (the
    kernel) and fp64 evaluation (the reference), each within one ulp of its own
    value; the test's 2-ulp element gate and 0.1-ulp RMS gate are met with the
    margin above.

    The identity is measured in fp32, where the question has an answer: the
    correct path deviates 3.29e-07 from the closed form; the tanh approximation
    4.73e-04 (1439x that, 31x torch's fp32 band), sigmoid 1.702x 2.03e-02, silu
    1.93e-01, relu 1.70e-01, and erf's scale truncated to 1/1.414 4.45e-05
    (2.9x the band). In bf16 the tanh approximation differs from the output by
    1.06x the correct path's own rounding, so it is not separable there -- the
    fp32 test is what pins the identity.

    A contiguous 4-D operand is measured equal to its 2-D view (one test, no
    cell): the reference applies the activation to the merger's
    `[B, 4608, H, W]` conv output, while this target's call site is the 2-D row
    form.

    Not certified: fp32 beyond the identity test; fp16, float8 operands; a
    non-contiguous `x`; widths other than 4304 and 4608; row counts between the
    cells and above the envelope (the op has no row- or column-indexed
    argument -- the kernel's grid follows `numel`, so no row-count invariance
    is claimed beyond what was driven); numel past 2^31 (the largest cell is
    7.05e7 elements).
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """The closed form in fp64 from primitive ops, rounded once to `x.dtype`."""
        x64 = x.double()
        return (0.5 * x64 * (1.0 + torch.erf(x64 / _SQRT_2))).to(x.dtype)


gelu = Gelu()
