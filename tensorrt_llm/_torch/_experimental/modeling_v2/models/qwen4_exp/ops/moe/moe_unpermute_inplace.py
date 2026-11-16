# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE unpermute: each token's routing-weighted sum of its top-k expert rows, taken
from the grouped GEMM's permuted output in a fixed slot order, in one trtllm call.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _ckpt(tokens: int, values: str = "normal") -> dict:
    """The checkpoint's routed-expert combine: top-10 of 512 experts, hidden 2560,
    bf16 FC2 rows, fp32 routing weights.

    The test lays `permuted_input` out exactly as the CuteDSL FC2 receives it: it
    draws each row's top-10 of uniform scores over the 512 experts, sorts the ids
    with `moe_sort` at `tile_tokens_dim` and writes each (token, slot) row at the
    permuted index the sort gave it. `values` is how the FC2 rows are drawn:
    `normal` = standard normal; `wide` = standard normal times 2^u, u uniform in
    [-30, 30] per element, so one sum adds terms up to 2^60 apart in magnitude.
    The scales are a softmax over each row's ten slots, as the renormalizing
    router emits them.
    """
    return dict(
        tokens=tokens,
        tile_tokens_dim=128,
        values=values,
        num_experts=512,
        top_k=10,
        hidden_size=2560,
        dtype=torch.bfloat16,
        scales_dtype=torch.float32,
    )


def _fma_f32(acc: torch.Tensor, prod: torch.Tensor) -> torch.Tensor:
    """`acc + prod` rounded once to fp32, as a fused multiply-add rounds it.

    `acc` is fp32; `prod` is the fp64 product of a bf16 and an fp32 value, which is
    exact (at most 8 + 24 significant bits). The fp64 sum is rounded to odd -- its
    exact rounding error from TwoSum, then the last bit forced to 1 if the sum was
    inexact -- so that the conversion to fp32 is one correct rounding of the exact
    sum rather than two roundings in a row (53 >= 24 + 2 bits).
    """
    a = acc.double()
    s = a + prod
    bb = s - a
    err = (a - (s - bb)) + (prod - bb)
    inexact = (err != 0) & torch.isfinite(s)
    # A sum rounded away from zero truncates to its neighbour one step toward zero.
    toward_zero = inexact & ((err < 0) != (s < 0))
    bits = s.view(torch.int64)
    bits = torch.where(inexact, (bits - toward_zero.long()) | 1, bits)
    return bits.view(torch.float64).float()


class MoeUnpermuteInplace(OpWrapper):
    """`torch.ops.trtllm.moe_unpermute_inplace`: the routing-weighted top-k combine.

    For each token `t`, writes row `t` of `output`:

        output[t] = bf16( sum over k = 0 .. top_k - 1, in that order, of
                          permuted_input[expanded_idx_to_permuted_idx[t, k]]
                          * topk_scales[t, k] )

    accumulated in fp32 with one rounding per slot (each product fused into the
    running sum) and rounded once to the output dtype. A slot whose permuted index
    is negative is skipped; a token with every slot skipped gets a zero row.

    Inputs, as the CuteDSL MoE hands them over: `permuted_input` `[M, H]` bf16 is
    the FC2 grouped GEMM's output, one row per (token, expert) pair in permuted
    order, `M >= T * top_k` (the sort's padded row count); `output` `[T, H]` bf16
    is written in full; `expanded_idx_to_permuted_idx` `[T, top_k]` int32 is the
    third output of `moe_sort`; `topk_scales` `[T, top_k]` fp32 are the routing
    weights. Call it with these keyword names, the op's own. Returns `output`.

    Fusion boundary: the gather, the weighting and the sum. The FC2 GEMM and the
    sort are the caller's, and so is anything added to the routed output (the
    shared expert). The op is deterministic: one thread block per token sums its
    slots in slot order, so the result does not depend on where the sort placed
    each row, on which other tokens share the batch, or on the call. That is what
    separates it from the FC2 GEMM's fused finalize, whose reduction order varies
    between launches.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, 1 request without MTP; rows from the sort's block kernel",
            spec=_ckpt(1),
        ),
        Cell(why="8192 rows, max_num_tokens prefill", spec=_ckpt(8192)),
        Cell(
            why="FC2 rows spanning 2^-30..2^30; terms 2^60 apart in one sum",
            spec=_ckpt(8192, values="wide"),
        ),
    )

    note = """
    Metadata consumed: none. Stateless; allocates nothing, writes `output` in place
    on the current stream with one kernel, `moeUnpermuteKernel<__nv_bfloat16,
    float, 256>` (`cpp/tensorrt_llm/kernels/moe/cuteDsl/moeUtils.cu`, registered
    in `cpp/tensorrt_llm/thop/moe/cuteDsl/cuteDslMoeUtilsOp.cpp`), one thread block
    per token. No host synchronization: it captures into a CUDA graph, and replay
    on new rows, scales and ids (moe_sort captured with it) is bitwise equal to
    eager and to `reference`.

    Call sites: the engine's CuteDSL MoE calls it when the FC2 finalize is not
    fused (`run_moe_nvfp4_impl`, `_torch/moe/fused_moe/fused_moe_cute_dsl.py`), and
    this target's `_moe_block` calls it in the prefill and the decode forward with
    the same four tensors: the FC2 output, the forward's dense `[num_tokens,
    hidden]` routed-output buffer, the sort's third output and the router's fp32
    weights. Prefill runs eagerly at up to max_num_tokens (8192) rows; decode runs
    captured at requests x MTP verify width, up to 128.

    What it reads: only the `T * top_k` rows `expanded_idx_to_permuted_idx` names,
    and the scales. The padding rows of each tile and the tiles past the sort's
    `num_non_exiting_tiles` are never read (every cell fills every other row of
    `permuted_input` with NaN, and under a graph those rows hold the previous
    replay's rows), so the FC2 output's unwritten regions need no memset before
    this op. What it writes: every element of `output`, whatever it held, and none
    of its inputs.

    Determinism (bitwise, measured at bring-up; the test re-drives only graph
    replay against eager): repeated calls; a second moe_sort of the same ids, which
    places an expert's rows in another order; a random permutation of all permuted
    rows; a prefix of a larger batch, with or without a fresh sort of the prefix
    alone; a shuffle of the tokens. The FC2 GEMM's fused finalize
    (`cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell`) differs from its
    own first call in about half the output elements at decode, while the plain FC2
    GEMM followed by this op does not, and is bitwise equal to the slot-order fp32
    sum of the real FC2 rows.

    Accuracy against the exact sum: the fp32 sum's error is bounded by its terms'
    magnitudes, so an element whose terms nearly cancel can miss the float64 sum
    rounded once to bf16 by many ulps of the element itself (827 of 20971520
    elements at T = 8192, standard-normal rows and softmax weights, up to 137.5).

    Silent, refused by `is_valid`: a non-contiguous `permuted_input`, `output`,
    `expanded_idx_to_permuted_idx` or `topk_scales` (all four are read or written
    from `data_ptr()` as dense row-major); an `output` whose dtype is not
    `permuted_input`'s (an fp32 output receives the bf16 bit patterns in the first
    half of its bytes); 2^31 or more tokens (nothing is written and nothing
    raises). The test holds the `permuted_input` refusal.

    Data hazards `is_valid` cannot see (values): a permuted index at or past
    `permuted_input.size(0)` is read without a bounds check; moe_sort never emits
    one.

    Loud (measured, not re-driven): int64 indices; hidden size not a multiple of 8; fp32 rows, or fp16 scales
    with bf16 rows; fewer than `T * top_k` permuted rows; a scales or index shape
    that disagrees with `output`.

    Accepted, not certified: T = 0 returns and writes nothing; bf16 scales; fp16
    rows; a CPU `output`.

    Certified only at the checkpoint's values: top_k 10, hidden 2560, bf16 rows and
    output, fp32 scales, the sort's layout at tile 128, every slot local; T 1 and
    8192 eager, 128 under CUDA-graph replay. Slots marked -1 (another rank's
    expert) are not certified.

    Hard limits (sm_103)
      tokens           one thread block per token, the count narrowed to int32:
                       boundary 2^31, `is_valid` refuses it
      output elements  T * H, offset in int64: exact past 2^31 and 2^32
      permuted rows    int32 index, offset in int64; no index passes through fp32
    """

    def __call__(
        self,
        permuted_input: torch.Tensor,
        output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        topk_scales: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.trtllm.moe_unpermute_inplace(
            permuted_input=permuted_input,
            output=output,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            topk_scales=topk_scales,
        )
        return output

    def reference(
        self,
        permuted_input: torch.Tensor,
        output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        topk_scales: torch.Tensor,
    ) -> torch.Tensor:
        """The slot-order fp32 sum, one rounding per slot, rounded once at the end.

        Each product of a row and its weight is exact in float64; `_fma_f32` adds
        it to the fp32 running sum with one rounding, as a fused multiply-add
        does. A slot with a negative index leaves the sum untouched. `output` is
        ignored: the result has `permuted_input`'s dtype, which is what the op
        writes.
        """
        idx = expanded_idx_to_permuted_idx.long()
        acc = torch.zeros(
            idx.shape[0], permuted_input.shape[1], dtype=torch.float32, device=idx.device
        )
        for k in range(idx.shape[1]):
            slot = idx[:, k]
            rows = permuted_input.index_select(0, slot.clamp_min(0)).double()
            prod = rows * topk_scales[:, k : k + 1].double()
            acc = torch.where((slot >= 0).unsqueeze(1), _fma_f32(acc, prod), acc)
        return acc.to(permuted_input.dtype)

    def is_valid(
        self,
        permuted_input: torch.Tensor,
        output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        topk_scales: torch.Tensor,
    ) -> None:
        # One thread block per token, the count narrowed to int32: from 2^31
        # tokens nothing is written and no error surfaces.
        assert output.size(0) < 2**31, (
            f"output has {output.size(0)} >= 2^31 rows; moe_unpermute_inplace launches one "
            "block per token from an int32 count and silently writes nothing"
        )
        # The kernel reads and writes all four through data_ptr() as dense
        # row-major buffers, and writes `output` with permuted_input's dtype.
        for name, t in (
            ("permuted_input", permuted_input),
            ("output", output),
            ("expanded_idx_to_permuted_idx", expanded_idx_to_permuted_idx),
            ("topk_scales", topk_scales),
        ):
            assert t.is_contiguous(), (
                f"{name} must be contiguous; moe_unpermute_inplace addresses it as a dense "
                "row-major buffer and silently combines or writes the wrong elements"
            )
        assert output.dtype == permuted_input.dtype, (
            f"output dtype {output.dtype} must be permuted_input's ({permuted_input.dtype}); "
            "the op writes permuted_input's element type into it without checking"
        )


moe_unpermute_inplace = MoeUnpermuteInplace()
