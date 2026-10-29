# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm over the last dim -- mean subtraction, biased variance, weight and
bias -- in one Triton kernel, through the FLA module callable
`tensorrt_llm._torch.modules.fla.layernorm_gated.layernorm_fn`.

One call site: the Mistral Large 4 DSA indexer's q_input_norm (hidden 6144,
bf16 weight and bias, eps 1e-6, rows 1..9216 on the FP8 checkpoint's target,
1..32768 on the NVFP4 checkpoint's).

Tier-2 module callable, not a `torch.ops.trtllm.*` op: the registered norm
family is RMS-norm throughout, and `fused_adaptive_layernorm` refuses every
width but 5120. One call reshapes to 2-D, allocates the output and launches
`_layer_norm_fwd_1pass_kernel` once.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modules.fla.layernorm_gated import layernorm_fn as _layernorm_fn

from .._op import Arch, Cell, OpWrapper

#: The hidden-6144 MLA + DSA target (tep4): its lightning indexer's
#: q_input_norm, a LayerNorm(6144) with a bias at the checkpoint's
#: indexer_norm_eps 1e-6, on each of the 44 indexer layers. Its input is the
#: layer's attention-normed hidden state, one contiguous bf16 row per token:
#: every row of the step, up to max_num_tokens (9216 for the FP8 checkpoint's
#: target, 32768 for the NVFP4 checkpoint's, whose chunked prefill is off so
#: one context step carries all 32768 rows).
HIDDEN = 6144


class LayerNorm(OpWrapper):
    """`(x - mean) / sqrt(var + eps) * weight + bias` per row of the last dim,
    variance over `N` (not `N - 1`), new contiguous tensor in `x`'s dtype.

    Every operand is widened to fp32 on load, the reduction and the affine run
    in fp32, and only the store rounds. `bias=None` drops the additive term (a
    compile-time branch of the kernel, not a zero bias).

    Fusion boundary: the norm and its affine only -- no residual, no gate, no
    output quantization. The callable's `z`, `group_size`, `norm_before_gate`
    and `is_rms_norm` are fixed at "no gate, one group, LayerNorm"; the gated
    and RMS forms are different computations, not options of this entry.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: shape, dtype; `eps` (default 1e-6). `site` "indexer_q" (rows shaped
    # like the attention-normed hidden state, a quarter of them each at 1e-6,
    # 1e-4, 1e-2 and 1 times its variance -- 1e-6 of it is about 2 eps -- so
    # the gate sees eps); weight and bias are drawn over that site's checkpoint
    # ranges. `checkpoint` "fp8" (default) or "nvfp4": whose q_input_norm
    # weight and bias ranges are drawn (two training snapshots).
    # `graph_rows`: each row count captured in a CUDA graph and replayed on new
    # data. `sweep_rows` (lo, hi): every row count lo..hi, eagerly.
    #
    # The kernel runs one program per row with the whole row in registers:
    # BLOCK_N is the width's next power of two (8192 at 6144, 2048 lanes of it
    # masked) at 8 warps. Triton also specializes the row count, compiling
    # M == 1, M divisible by 16 and any other M separately; the graph batches
    # reach all three (1; 2, 4 and 8; 16) and so does the sweep.
    #
    # The indexer's eager cells drive every row count up to each target's
    # max_num_tokens, each as its own call: 1..9216 for the FP8 checkpoint's
    # target, 1..32768 for the NVFP4 checkpoint's. On sm_103 the 1..32768
    # sweep costs ~6 s, about half of it the bitwise compare; 1..9216 costs
    # ~1.5 s.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="indexer q_input_norm, CUDA graphs 1/2/4/8/16: hidden 6144 bf16 at eps "
            "1e-6, the captured decode batches, each replayed on new data bitwise equal "
            "to eager",
            spec=dict(
                shape=(16, HIDDEN),
                dtype=torch.bfloat16,
                eps=1e-6,
                site="indexer_q",
                graph_rows=(1, 2, 4, 8, 16),
            ),
        ),
        Cell(
            why="indexer q_input_norm, eager rows 1..9216: hidden 6144 bf16 at eps 1e-6, "
            "every row count up to the token budget, each bitwise the leading rows of the "
            "9216-row call",
            spec=dict(
                shape=(9216, HIDDEN),
                dtype=torch.bfloat16,
                eps=1e-6,
                site="indexer_q",
                sweep_rows=(1, 9216),
            ),
        ),
        Cell(
            why="indexer q_input_norm, eager rows 1..32768: hidden 6144 bf16 at eps 1e-6, "
            "the NVFP4 target's max_num_tokens 32768 with chunked prefill off (one 32768-row "
            "prompt, or up to 16 packed), every row count, each bitwise the leading rows of "
            "the 32768-row call, at the NVFP4 checkpoint's q_input_norm ranges",
            spec=dict(
                shape=(32768, HIDDEN),
                dtype=torch.bfloat16,
                eps=1e-6,
                site="indexer_q",
                checkpoint="nvfp4",
                sweep_rows=(1, 32768),
            ),
        ),
    )

    note = """
    Metadata consumed: none. Inputs are not mutated; two scratch fp32 [M]
    tensors (mean, rstd) are allocated per call and discarded.

    Layout: `x.stride(-1) == 1` is read in place through its row stride; any
    other `x`, and non-contiguous `weight`/`bias`, are silently `.contiguous()`d
    first (one extra copy, correct result). `weight`/`bias` dtype is free.

    Loud failures (left to the callable): weight or bias length != N raises a
    bare AssertionError with no message; `next_pow2(N) * x.element_size() >
    64KB` raises "This layer norm doesn't support feature dim >= 64KB" (N =
    65536 in bf16); CPU tensors fail before or at the Triton launch.

    Accuracy: at the indexer's affine (weights 0.89..1.14, biases within 0.09)
    over 9216 hidden-state rows of 6144 the bf16 result is the correctly
    rounded value on all but 1.6e-5..1.8e-5 of elements, and one bf16 step off
    at most except where `x_hat * weight` and `bias` cancel: 21 elements of
    the sweep cell's input sit more than one step off (up to 26; 92 in another
    draw), each a result within 2e-6 of zero and off by at most 1.2e-8.
    torch's `F.layer_norm` on the same bf16 operands (fp32 inside, one
    rounding) measures the same and differs from this op on 1.9e-5 of
    elements. Over 32768 such rows at the NVFP4 checkpoint's affine (weights
    0.89..1.13, biases -0.087..0.090): all but 1.77e-5 of elements (3,562 of
    201,326,592) correctly rounded, 117 more than one step off, each a result
    within 1.9e-6 of zero and off by at most 1.5e-8; `F.layer_norm` 1.93e-5
    off, differing from this op on 1.94e-5. A constant row (zero included)
    returns `bias` bitwise. `eps` sits inside the sqrt.

    Rows are computed independently: at 6144, every row count in 1..32768 is
    bitwise the leading rows of the 32768-row call (and every one in 1..9216
    of the 9216-row call), and a CUDA-graph replay at each of 1..16 rows is
    bitwise equal to an eager call on the same data. One 32768-row call is one
    launch; back to back, an eager call of 1 or 16 rows takes as long as the
    host's launch path, not the kernel.

    Compilation: each (dtype, width, row-count class) is compiled by Triton
    on its first call in a process, or loaded from Triton's on-disk cache; a
    cold compile measured 0.3-0.9 s. The graph cells capture after one
    warm-up call at the same row count, as the engine's warm-up does; capture
    with no warm-up was not measured.

    Accepted but not certified: an fp8-e4m3 `x` returns fp8-e4m3 (output is
    `empty_like(x)`), within one e4m3 ulp of the exact value; fp16, fp32 and
    fp64 `x`; `weight`/`bias` in a dtype other than `x`'s; `bias=None`; an
    `x` of more than two dims, or with a row stride other than N; widths
    other than 6144; more than 32768 rows at 6144. Forward only: the autograd
    Function has no backward. Runs unchanged under `inference_mode`, and the
    graph cells capture and replay inside it.

    Hard limits (sm_103). Computed, not probed: the envelopes' largest call,
    32768 x 6144 on the NVFP4 target (9216 x 6144 on the FP8 target), is
    201,326,592 elements, and its last row starts at 201,320,448, 9.4% of
    2^31. The op reads no KV pool, so no pool size bounds it.
      row offset   `row * stride_x_row` in int32: the program id is 32-bit
                   and the compiled signature types the row stride i32.
                   Wraps at row * stride >= 2^31: 349,526 rows at 6144
      columns      one block of next_pow2(N) lanes; N is bounded by the 64KB
                   check above, not by an index width
      rows         the grid's first dimension and an i32 argument: 2^31 - 1
      fp32         no index passes through fp32
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        return _layernorm_fn(x, weight, bias, z=None, eps=eps)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """fp64 from primitive ops (not `F.layer_norm`), rounded once to `x.dtype`."""
        xd = x.double()
        mean = xd.mean(dim=-1, keepdim=True)
        var = (xd - mean).square().mean(dim=-1, keepdim=True)
        out = (xd - mean) / torch.sqrt(var + eps) * weight.double()
        if bias is not None:
            out = out + bias.double()
        return out.to(x.dtype)


layer_norm = LayerNorm()
