# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode GEMV: ``x @ weight^T`` for at most 8 bf16 rows on a CuTe DSL kernel (short-K or split-K, chosen
by the weight's shape)."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_decode_gemv.op  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class K3DecodeGemv(OpWrapper):
    """Return ``x @ weight^T`` as a new bf16 ``[M, N]`` tensor (fp32 accumulation, one bf16 rounding) in one
    k3_decode_gemv call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="KDA o_proj, short K resident in one CTA per weight tile",
            spec=dict(n=7168, k=768, ms=range(1, 9), trigger_early=(True, False)),
        ),
        Cell(
            why="KDA input projection, split K and an eight-row final output tile",
            spec=dict(n=3208, k=7168, ms=range(1, 9), trigger_early=(True, False)),
        ),
    )

    note = """
    Each TP16 cell covers M=1..8 and both trigger_early values, with
    x~N(0,1) and weight~0.03*N(0,1), rounded to bf16. Input tensors are
    read-only; every output element is written in a new contiguous tensor.
    Bias, activation, scaling, quantization, tensor-parallel reduction and
    flattening leading dimensions are the caller's.

    Short K uses one CTA per 128-row weight tile with the whole K resident,
    accumulating in k order. Split K uses four-CTA clusters: rank r sums
    128-column k-tiles r,r+4,... and owns rows 32r..32r+31. Other ranks
    push fp32 partials to it, then it adds them as ((p0+p1)+p2)+p3 and
    rounds once to bf16. Partial output tiles are zero-filled and not stored
    past N. The MMA always has eight token columns, zero-filled past M.
    Repeated calls and rows across M and trigger_early are bit-identical;
    cuBLAS may differ in summation order. Short K uses N/128 CTAs of 256
    threads; split K uses 4*ceil(N/128) CTAs in clusters of four.

    No attention metadata or module state is consumed. The current CUDA
    stream is used. A result-neutral process compile cache is keyed by
    (N, K, trigger_early, PDL); M is runtime. Warm every key eagerly before
    graph capture or the first call raises RuntimeError. cutlass and
    cuda.bindings must be importable. The op loudly rejects unsupported
    dtypes, layouts and shapes; the tests retain these refusals and warmed
    graph replay at M=1,8 for both shapes.

    TRTLLM_ENABLE_PDL defaults to 1 and is read every call. It changes
    scheduling, not results. Before griddepcontrol.wait, short K loads its
    whole weight slice; split K fills up to six shared-memory stages and
    prefetches the rest into L2. Weight must not be written by preceding
    in-flight work; x is read and y written after the wait. trigger_early
    releases dependents after initial weight loads, rather than at grid
    completion; PDL dependents must wait before reading y. Weight loads
    use L2 EVICT_FIRST. k3_decode_gemv_tail is a separate op, not this entry.
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, trigger_early: bool = True
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_decode_gemv(x, weight, trigger_early)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, trigger_early: bool = True
    ) -> torch.Tensor:
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).bfloat16()

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, trigger_early: bool = True) -> None:
        assert x.data_ptr() % 16 == 0, "x must start at a 16-byte-aligned address"
        assert weight.data_ptr() % 16 == 0, "weight must start at a 16-byte-aligned address"
        assert weight.device == x.device, "weight must be on x's device"


k3_decode_gemv = K3DecodeGemv()
