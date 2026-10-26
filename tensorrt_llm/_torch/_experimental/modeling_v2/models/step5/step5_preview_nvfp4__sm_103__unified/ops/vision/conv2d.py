# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""2-D convolution of the Step-5 vision tower: the patch embedding and the two downsamplers.

Tier 4, one `torch.nn.functional.conv2d` call: no engine kernel computes a 2-D
convolution. Rejected at the cheaper rungs:

- tier 1: no `torch.ops.trtllm` op is a 2-D convolution; the only convolutions
  registered are `causal_conv1d_fwd` / `causal_conv1d_update` (1-D, depthwise,
  causal).
- tier 2: nothing under `tensorrt_llm/_torch/modules/` computes one.
- tier 3: FlashInfer's public API has no convolution.
- The CuTe DSL implicit-GEMM fprop kernels in
  `tensorrt_llm/_torch/visual_gen/kernels/blackwell/conv/` are not a registered
  op and sit outside every admitted rung (a plain Python callable needs the run's
  authorization); they are 3-D only, and their bf16 path needs
  `in_channels % 8 == 0`, which the patch embedding's 3 channels fail.
- qwen4_exp's `ops/vision/patch_embed_conv3d` is a 3-D convolution over
  pre-flattened patch rows at its own geometry, not a strided 2-D convolution over
  an image.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: `vision_config`: width 1536, patch_size 14, image_size 728; crops are 504 px.
WIDTH = 1536
PATCH = 14
OVERVIEW_SIDE = 728
CROP_SIDE = 504

#: The three convolutions: (in_channels, out_channels, kernel, stride, padding, bias).
PATCH_EMBED = (3, WIDTH, PATCH, PATCH, 0, False)
DOWNSAMPLER1 = (WIDTH, 2 * WIDTH, 3, 2, 1, True)
DOWNSAMPLER2 = (2 * WIDTH, 4 * WIDTH, 3, 2, 1, True)

#: Tiles per call at the ISL 8192 ceiling. A crop is 83 prompt tokens (81
#: `<im_patch>` and its two markers) and an image's overview 171; the input
#: processor cuts at most 6 x 6 = 36 crops from one image (`_MAX_SIDE` 3024 over
#: 504-px windows). 92 crops take 3 images: 92 * 83 + 3 * 171 = 8149 <= 8192.
#: 47 crop-less images take 47 * 171 = 8037 tokens.
MAX_CROPS = 92
MAX_OVERVIEWS = 47


def _cell(why: str, conv: tuple, tiles: int, side: int, layout: str) -> Cell:
    in_ch, out_ch, kernel, stride, padding, bias = conv
    return Cell(
        why=why,
        spec=dict(
            tiles=tiles,
            side=side,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel=kernel,
            stride=stride,
            padding=padding,
            bias=bias,
            layout=layout,
            dtype=torch.bfloat16,
        ),
    )


class Conv2d(OpWrapper):
    """`out[n, k, p, q] = bias[k] + sum_{c,r,s} x[n, c, p*stride + r - padding,
    q*stride + s - padding] * weight[k, c, r, s]`, zero outside the image.

    Cross-correlation (no kernel flip), like `nn.Conv2d`, with dilation 1 and one
    group. `input` is `[N, C, H, W]` in any memory format; the result is a new
    `[N, K, P, Q]` tensor in `input`'s dtype and, as torch propagates it,
    `input`'s memory format (a channels-last input gives a channels-last output,
    whose `flatten(2).transpose(1, 2)` is a free `[N, P*Q, K]` row view).

    Fusion boundary: the convolution and its bias. Turning tower rows into an
    image view before a downsampler, and the image view back into rows after it,
    are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: tiles (N), side (H = W), in/out_channels, kernel, stride, padding,
    # bias, layout ("nchw" contiguous, or "nhwc" for channels-last: the
    # zero-copy `rows.view(N, H, W, C).permute(0, 3, 1, 2)` of the tower's
    # `[N, H*W, C]` rows), dtype.
    CELLS: tuple[Cell, ...] = (
        # Patch embedding, 3 -> 1536, k14 s14, no bias: the input processor's
        # contiguous pixel tiles.
        _cell("patch embed: 1 overview 728 px", PATCH_EMBED, 1, OVERVIEW_SIDE, "nchw"),
        _cell("patch embed: 3 overviews 728 px", PATCH_EMBED, 3, OVERVIEW_SIDE, "nchw"),
        _cell("patch embed: 1 crop 504 px", PATCH_EMBED, 1, CROP_SIDE, "nchw"),
        _cell("patch embed: 9 crops 504 px", PATCH_EMBED, 9, CROP_SIDE, "nchw"),
        _cell("patch embed: 36 crops, one 3024 px image", PATCH_EMBED, 36, CROP_SIDE, "nchw"),
        _cell(
            "patch embed: 92 crops, the ISL 8192 ceiling",
            PATCH_EMBED,
            MAX_CROPS,
            CROP_SIDE,
            "nchw",
        ),
        _cell(
            "patch embed: 47 overviews, the ISL 8192 ceiling",
            PATCH_EMBED,
            MAX_OVERVIEWS,
            OVERVIEW_SIDE,
            "nchw",
        ),
        _cell(
            "patch embed: channels-last pixels, 1 overview", PATCH_EMBED, 1, OVERVIEW_SIDE, "nhwc"
        ),
        # Downsampler 1, 1536 -> 3072, k3 s2 p1, bias, on the tower's grid.
        _cell("downsampler1: 1 overview 52x52 channels-last", DOWNSAMPLER1, 1, 52, "nhwc"),
        _cell("downsampler1: 9 crops 36x36 channels-last", DOWNSAMPLER1, 9, 36, "nhwc"),
        _cell("downsampler1: 36 crops 36x36 channels-last", DOWNSAMPLER1, 36, 36, "nhwc"),
        _cell("downsampler1: 92 crops 36x36, the ISL ceiling", DOWNSAMPLER1, MAX_CROPS, 36, "nhwc"),
        _cell(
            "downsampler1: 47 overviews 52x52, ISL ceiling",
            DOWNSAMPLER1,
            MAX_OVERVIEWS,
            52,
            "nhwc",
        ),
        _cell("downsampler1: 1 overview 52x52 contiguous", DOWNSAMPLER1, 1, 52, "nchw"),
        # Downsampler 2, 3072 -> 6144, k3 s2 p1, bias, on downsampler 1's output.
        _cell("downsampler2: 1 overview 26x26 channels-last", DOWNSAMPLER2, 1, 26, "nhwc"),
        _cell("downsampler2: 9 crops 18x18 channels-last", DOWNSAMPLER2, 9, 18, "nhwc"),
        _cell("downsampler2: 36 crops 18x18 channels-last", DOWNSAMPLER2, 36, 18, "nhwc"),
        _cell("downsampler2: 92 crops 18x18, the ISL ceiling", DOWNSAMPLER2, MAX_CROPS, 18, "nhwc"),
        _cell(
            "downsampler2: 47 overviews 26x26, ISL ceiling",
            DOWNSAMPLER2,
            MAX_OVERVIEWS,
            26,
            "nhwc",
        ),
        _cell("downsampler2: 1 overview 26x26 contiguous", DOWNSAMPLER2, 1, 26, "nchw"),
    )

    note = """
    Metadata consumed: none. Out-of-place; `input` is not mutated (asserted per
    cell). All operands share one dtype and device; torch raises otherwise. One
    `F.conv2d` call, routed by torch to cuDNN (cudnn.benchmark off, torch's
    default, in every measurement here). Profiled launches on sm_103: the patch
    embedding (contiguous pixels) converts input and weight to NHWC, runs a
    `cutlass_tensorop_bf16_s16816fprop` kernel and converts the output back to
    NCHW; a downsampler on channels-last input runs one elementwise launch,
    then a `cutlass3x_sm100 ... implicit_gemm_fprop_bf16_bf16_f32` kernel, then
    the bias add as a second elementwise launch. The first launch is, by
    inference, the contiguous weight's copy to channels-last (85 MB for
    downsampler 1, 340 MB for downsampler 2, every call); a channels-last
    weight would skip it, and no cell holds one.

    Caller preconditions: `input` is `[N, C, H, W]`. A downsampler's input is
    the tower's `[N, H*W, C]` rows viewed as `rows.view(N, H, W, C).permute(0,
    3, 1, 2)`: channels-last, no copy; StepFun's `transpose(1, 2).reshape(...)`
    is the same strides. The output keeps the input's memory format.

    Numerics (sm_103, cuDNN): the conv accumulates in fp32 and rounds to bf16;
    the bias is then added in bf16, a second rounding (StepFun's `nn.Conv2d` in
    bf16 does the same). Against the once-rounded fp32 reference: no-bias cells
    at most 1.98 units (2^-8 of the row max), relative RMS 0.017 units, 4.4e-4
    of elements differing; bias cells at most 2.00 units, relative RMS 0.738
    units, about 0.26 of elements differing, all of it the second rounding
    (`test_bias_is_added_after_rounding`). The test gates 3 / 1.0 units.
    Contiguous and channels-last inputs gave bitwise-equal outputs at
    downsampler 1 with 9 crops, and a tile alone equalled the same tile inside a
    9-tile batch bitwise there; neither is certified elsewhere (cuDNN picks its
    algorithm per shape: a 730 px tile and its 728 px crop, same 52 x 52 output,
    were not bitwise equal).

    Measured cost on sm_103 (eager, mean of 5 after warm-up): patch embedding
    0.17 ms per overview, 5.3 ms at 92 crops, 5.7 ms at 47 overviews;
    downsampler 1 1.8 ms at 92 crops; downsampler 2 1.9 ms at 92 crops and
    2.1 ms at 47 overviews. Peak allocation above the operands at most 1.10 GiB
    (patch embedding, 47 overviews: the output). Not tuned.

    Guards (`is_valid`): a patchifying convolution (stride equal to the kernel,
    no padding) over a tile whose side is not a whole number of patches; torch
    drops the remainder (730 px gives 52 x 52 from the top-left 728 px) with no
    error.

    Hard limits (computed, not probed; the envelope is far below each)
      2^31 elements of one tensor: input pixels at 1351 overviews / 2819 crops;
      patch-embedding output at 518 overviews / 1079 crops; downsampler-1
      output at 1035 overviews / 2158 crops; downsampler-2 output at 2069 /
      4316. The ISL 8192 ceiling is 47 overviews or 92 crops per call.

    Certified: bf16, the three convolutions of the cells (3 -> 1536 k14 s14 no
    bias; 1536 -> 3072 and 3072 -> 6144 k3 s2 p1 with bias), at the tiles,
    sides and layouts the cells name.

    Not certified: tile counts between and beyond the cells (per-request
    ceiling 92 crops / 47 overviews; a vision call that batches several
    requests' tiles can pass more), fp16 and fp32 (fp32 under cuDNN runs TF32
    unless the caller turns `torch.backends.cudnn.allow_tf32` off), other
    kernels, strides, paddings and channel counts, dilation and groups (not
    parameters of `__call__`), memory formats other than contiguous and
    channels-last, cudnn.benchmark on, CUDA-graph capture (the call site is
    eager), torch.compile.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        return F.conv2d(input, weight, bias, stride, padding)

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """im2col (`F.unfold`) in fp32, a true-fp32 matmul against the flattened
        filter, the bias added in fp32, rounded once to `input.dtype`. Returns a
        contiguous `[N, K, P, Q]`."""
        n, _, h, w = input.shape
        out_channels, _, kh, kw = weight.shape
        p = (h + 2 * padding - kh) // stride + 1
        q = (w + 2 * padding - kw) // stride + 1
        cols = F.unfold(input.float(), (kh, kw), padding=padding, stride=stride)
        filters = weight.reshape(out_channels, -1).float()
        with true_fp32_matmul():
            out = filters @ cols  # [N, K, P*Q]
        if bias is not None:
            out = out + bias.float()[:, None]
        return out.view(n, out_channels, p, q).to(input.dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int,
        padding: int,
    ) -> None:
        # A patchifying convolution (stride == kernel, no padding) over an image
        # that is not a whole number of patches drops the remainder rows and
        # columns: a 730 px tile gives the same 52 x 52 grid as 728 px from the
        # top-left 728 x 728 pixels, with no error.
        kh, kw = weight.shape[-2:]
        if padding == 0 and (stride, stride) == (kh, kw):
            h, w = input.shape[-2:]
            assert h % kh == 0 and w % kw == 0, (
                f"conv2d: a {kh}x{kw} patch embedding needs whole patches; "
                f"{h}x{w} leaves {h % kh}x{w % kw} pixels that would be dropped"
            )


conv2d = Conv2d()
