# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's decode-step embedding and its first layer's input RMSNorm in one CuTe DSL launch: the rows ``table[ids]``
written into a caller's buffer, and their RMSNorm."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_embed.op  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class K3EmbedNorm(OpWrapper):
    """Write ``table[ids]`` into ``raw`` (zero rows for ids outside ``[0, V)``) and return
    ``raw * rsqrt(mean(raw^2) + eps) * weight`` as a new bf16 tensor, in one k3_embed_norm call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"N={tokens} int32: K3 decode/DSpark token count and row independence",
                spec=dict(
                    tokens=tokens, dtype=torch.int32, vocab=163840, hidden=7168, eps=1e-6, slot=0
                ),
            )
            for tokens in (*range(1, 9), *range(16, 65, 8))
        ),
        *(
            Cell(
                why=f"N={tokens} int64: id-width parity with int32",
                spec=dict(
                    tokens=tokens, dtype=torch.int64, vocab=163840, hidden=7168, eps=1e-6, slot=0
                ),
            )
            for tokens in (1, 8, 64)
        ),
        Cell(
            why="eps=1e-5 and snapshot slot 1: norm root and raw offset",
            spec=dict(tokens=8, dtype=torch.int32, vocab=163840, hidden=7168, eps=1e-5, slot=1),
        ),
        *(
            Cell(
                why=f"{dtype} outside-vocabulary group {start // 4}: zero rows without id narrowing",
                spec=dict(
                    tokens=8,
                    dtype=dtype,
                    vocab=163840,
                    hidden=7168,
                    eps=1e-6,
                    slot=0,
                    outside=tuple(bad[(start + i) % len(bad)] for i in range(4)),
                ),
            )
            for dtype, bad in (
                (torch.int32, (-1, 163840, 163847, -163840, 2**31 - 1, -(2**31))),
                (
                    torch.int64,
                    (
                        -1,
                        163840,
                        163847,
                        -163840,
                        2**31 - 1,
                        -(2**31),
                        2**32,
                        2**32 + 5,
                        3 - 2**32,
                        2**40,
                    ),
                ),
            )
            for start in range(0, len(bad), 4)
        ),
    )
    note = """
    The caller allocates raw, usually a snapshot-bank slot; every element is
    overwritten, while ids, table and weight are read-only. No residual add,
    (1+weight) scaling or quantization. The replicated K3 table is not sharded.
    Out-of-range ids read no table row; negative weights can produce -0.0.
    k3_embed (gather alone) is a separate op.

    One 128-thread CTA per token reduces H/128 values per thread across four
    warps. Fast-math rsqrt and a fixed reduction order make identical calls and
    repeated ids at different token counts bit-identical. This geometry matches
    the CuTe flashinfer RMSNorm for 6144<H<=16384, not FLASHINFER_USE_CUDA_NORM=1;
    the native op test checks gather-plus-norm bit equality.

    No metadata beyond arguments and the current CUDA stream. A result-neutral
    process compile cache keys (N,V,H,ids dtype,PDL); eps and pointers are runtime
    arguments. Warm each key eagerly: cold CUDA-graph capture raises RuntimeError.
    TRTLLM_ENABLE_PDL defaults to 1 and is read each call. PDL releases dependents
    at entry, then waits before reading inputs; dependent kernels must wait before
    reading raw or output. PDL needs SM>=9.0. Missing cutlass or cuda.bindings fails
    on import. Strided ids are copied by the op; supported tensors are otherwise
    contiguous and 16-byte aligned, with H a multiple of 1024 in (6144,16384].
    """

    def __call__(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_embed_norm(ids, table, weight, eps, raw)

    def reference(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        valid = (ids >= 0) & (ids < table.shape[0])
        rows = table.index_select(0, torch.where(valid, ids, torch.zeros_like(ids)))
        rows = rows.masked_fill(~valid[:, None], 0.0)
        raw.copy_(rows)
        x = rows.float()
        return (x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + eps) * weight.float()).to(
            table.dtype
        )

    def is_valid(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        raw: torch.Tensor,
    ) -> None:
        assert all(tensor.device == ids.device for tensor in (table, weight, raw)), (
            "table, weight and raw must be on ids' device"
        )
        raw_start = raw.data_ptr()
        raw_end = raw_start + raw.numel() * raw.element_size()
        if raw.numel() == 0:
            return
        for tensor in (table, weight):
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            assert raw_end <= start or end <= raw_start, "raw must not overlap table or weight"
        if ids.numel():
            start = ids.data_ptr()
            stride = ids.stride(0) * ids.element_size()
            index = max(0, (raw_start - start - ids.element_size()) // stride + 1) if stride else 0
            assert (
                index >= ids.numel()
                or start + index * stride >= raw_end
                or (start + index * stride + ids.element_size() <= raw_start)
            ), "raw must not overlap ids"


k3_embed_norm = K3EmbedNorm()
