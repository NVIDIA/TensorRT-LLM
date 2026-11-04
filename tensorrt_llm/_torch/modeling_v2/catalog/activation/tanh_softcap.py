# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final-logit tanh soft-cap, ``cap * tanh(x / cap)``: the Mistral Large 4 checkpoints'
``softmax_tanh_gating`` (30.0) applied to the gathered fp32 logits.

Sourcing tier 4, composed from native torch: no engine kernel computes it.
Rejected at the cheaper rungs, each searched on this build: tier 1,
``torch.ops.trtllm`` registers no soft-cap op (its only tanh op,
``flashinfer_gelu_tanh_and_mul``, is a gated GELU); tier 2, no callable under
``tensorrt_llm/_torch/modules`` (``LogitsProcessor`` in
``modules/logits_processor.py`` has no cap, and ``modeling_gemma4.py`` applies
``torch.tanh(logits / cap) * cap`` inline rather than through a module); tier 3,
the ``flashinfer.logits_processor`` operators (``Temperature``, ``Softmax``,
``TopK``, ``TopP``, ``MinP``, ``Sample``) carry no soft-cap. Correct and slow by
design: three elementwise launches, which a tuning pass may fold into the
logits epilogue.
"""

from __future__ import annotations

import math

import torch

from .._op import Arch, Cell, OpWrapper

#: The checkpoint's vocabulary; the call site holds every rank's gathered row.
VOCAB_SIZE = 204800
#: ``softmax_tanh_gating`` in the checkpoint's config.
SOFTCAP = 30.0
#: The Mistral Large 4 NVFP4 target's token budget (``max_num_tokens``; the FP8
#: target's is 9216): the most rows a step can hand over when a context request
#: asks for its context logits.
MAX_NUM_TOKENS = 32768
#: The most rows whose 204800-wide logits stay below 2^31 elements.
INT32_ROWS = (2**31 - 1) // VOCAB_SIZE

_CKPT = dict(vocab=VOCAB_SIZE, dtype=torch.float32, cap=SOFTCAP)


class TanhSoftcap(OpWrapper):
    """`cap * tanh(x / cap)`, elementwise, any shape, a new tensor in `x`'s dtype.

    The three torch ops the reference implementation's logits processor runs,
    in its order, each rounding to `x.dtype`: `x / cap` (a Python-float divisor
    is applied on CUDA as `x * fl(1 / cap)`, bitwise), `torch.tanh`, `* cap`.

    Fusion boundary: the cap only. The lm_head projection, the vocabulary
    all-gather and the fp32 upcast before it are the caller's (the shell's
    logits processor); temperature, softmax and sampling after it are the
    sampler's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, vocab, dtype, cap; values uniform over [lo, hi]. `graph` True
    # captures the call in a CUDA graph and checks a replay on fresh logits
    # written into the captured input, bitwise against an eager call on them;
    # False runs it eagerly. `windows` are (start, stop) row ranges whose output
    # must be bitwise the op run on those rows alone. A `large` cell reaches a
    # bound past the per-cell budget and carries pytest.mark.large.
    #
    # Rows per call: one per sequence in the step (1..16, max_batch_size 16),
    # or, when a context request asks for its context logits, every token of
    # the step (up to max_num_tokens 32768). Decode steps are captured: the
    # gates capture 1/2/4/8/16 padded, the LLM API default 1..16 unpadded.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"decode graph batch {rows}: captured, replayed; |x| <= 400 crosses saturation",
            spec=dict(rows=rows, lo=-400.0, hi=400.0, graph=True, **_CKPT),
        )
        for rows in range(1, 17)
    ) + (
        Cell(
            why="context step, 3 requests: eager, one last-token row per request",
            spec=dict(rows=3, lo=-400.0, hi=400.0, graph=False, **_CKPT),
        ),
        Cell(
            why="16 rows over [-60, 60]: the bend, |x / cap| <= 2, nothing saturates",
            spec=dict(rows=16, lo=-60.0, hi=60.0, graph=False, **_CKPT),
        ),
        Cell(
            why="16 rows over [-1e-3, 1e-3]: relative precision where the cap is identity",
            spec=dict(rows=16, lo=-1e-3, hi=1e-3, graph=False, **_CKPT),
        ),
        Cell(
            why=f"context logits, {INT32_ROWS} rows: the most below 2^31 elements, "
            "leading and last rows bitwise; 26 GiB peak",
            spec=dict(
                rows=INT32_ROWS,
                lo=-400.0,
                hi=400.0,
                graph=False,
                windows=((0, 1), (0, 16), (0, 4096), (INT32_ROWS - 16, INT32_ROWS)),
                **_CKPT,
            ),
        ),
        Cell(
            why=f"context logits, {MAX_NUM_TOKENS} rows = max_num_tokens (large): 6.7e9 "
            "elements, past 2^31 and 2^32; 81 GiB peak",
            spec=dict(
                rows=MAX_NUM_TOKENS,
                lo=-400.0,
                hi=400.0,
                graph=False,
                windows=(
                    (0, 16),
                    (8184, 8200),
                    (10480, 10492),  # element 2^31 is row 10485, column 155648
                    (16376, 16392),
                    (20966, 20978),  # element 2^32 is row 20971, column 106496
                    (24568, 24584),
                    (MAX_NUM_TOKENS - 16, MAX_NUM_TOKENS),
                ),
                large=True,
                **_CKPT,
            ),
        ),
    )

    note = """
    Metadata consumed: none. Out-of-place; `x` is not mutated. Three launches,
    allocating the quotient, the tanh and the output, each of `x`'s size, at
    most two live at once; the first two are freed on return. That is 12.5 MiB
    each at 16 x 204800 fp32, but 25 GiB each at a 32768-row context-logits
    step: the call then holds 50 GiB beyond its 25 GiB input at its peak
    (measured), which a caller serving context logits at the token budget must
    leave free. Under CUDA-graph capture they come from the graph's pool and
    `cap` is baked into the captured kernels. Rows are independent: a padded
    batch's dummy rows, nan or inf included, never reach the live ones, and a
    row's result does not depend on how many rows the call holds (measured:
    row windows of a 10485- and a 32768-row call are bitwise the op run on
    those rows alone, and every captured batch 1..16 replays bitwise eager).

    `cap` is a Python float. A 0-d CUDA tensor `cap` would take torch's true
    division instead of the reciprocal multiply: within one ulp of it, not
    bitwise equal (measured); not a certified form.

    Numerics (fp32): at most 4 ulp from `reference` over every cell, under a
    derived 7.5 ulp bound (the test states the derivation). |x| >=
    270.327392578125 returns exactly +-30.0 (the fp32 tanh rounds to 1);
    +-inf -> +-30.0; nan -> nan; -0.0 -> -0.0. Inputs below 30 * 2^-126
    (about 3.5e-37) in magnitude pass through a subnormal quotient and lose
    relative precision (1e-40 -> 1.0001e-40), absolute error below 1e-43.

    `cap` must be finite and non-zero: `is_valid` refuses 0, +-inf and nan,
    which the op accepts and answers with zeros and nan (cap 0: every logit
    becomes +-0 and an exact 0 becomes nan; an infinite or nan cap: nan
    everywhere). A negative cap returns the |cap| result bitwise (tanh is odd).

    Hard limits (sm_103)
      x.numel()  elementwise TensorIterator launches  boundary 2^31 elements
                 (crossed from 10486 rows at width 204800)  probed at 32768 x 204800 =
                 6710886400 elements, past 2^31 and 2^32  passed: every element
                 within the bound, and rows straddling elements 2^31 and 2^32
                 and the quarter points bitwise the op run on those rows alone

    Not certified: bf16/fp16 input (each step would round to the input dtype),
    non-contiguous input, a width other than 204800, torch.compile.
    """

    def __call__(self, x: torch.Tensor, cap: float) -> torch.Tensor:
        return torch.tanh(x / cap) * cap

    def reference(self, x: torch.Tensor, cap: float) -> torch.Tensor:
        """The closed form in fp64, rounded once to `x.dtype`."""
        return (cap * torch.tanh(x.double() / cap)).to(x.dtype)

    def is_valid(self, x: torch.Tensor, cap: float) -> None:
        if not (math.isfinite(cap) and cap != 0.0):
            raise ValueError(
                f"tanh_softcap: cap must be finite and non-zero, got {cap!r}; "
                "a zero cap zeroes every logit and an infinite or nan one makes them nan"
            )


tanh_softcap = TanhSoftcap()
