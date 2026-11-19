# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer index-key LayerNorm with one fixed reduction order (CuTe DSL)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_index_key_norm import (  # noqa: E501
    HEAD_DIM,
    index_key_norm_into,
    params_supported,
    rows_supported,
)

#: Columns of the packed hidden-state projection row ``[q_a | kv_a | k_pe | wk | weights_proj | g_proj]`` on a
#: TP16 rank (1536 + 512 + 64 + 128 + 32 + 768) and where its wk block starts.
_PACKED_WIDTH, _WK_COLUMN = 3040, 2112


def _torch_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, num_tokens: int
) -> torch.Tensor:
    """The LayerNorm of ``x[:num_tokens]`` in fp32 (torch's kernel), rounded to bf16."""
    return torch.nn.functional.layer_norm(
        x[:num_tokens].float(), (x.shape[-1],), weight.float(), bias.float(), eps
    ).to(torch.bfloat16)


def _spec(tokens, **overrides) -> dict:
    spec = dict(
        tokens=tokens,
        row_stride=_PACKED_WIDTH,  # the raw key is a column view of the packed projection output
        column=_WK_COLUMN,
        eps=1e-5,
        data="randn",
    )
    spec.update(overrides)
    return spec


class IndexKeyNorm(OpWrapper):
    """``LayerNorm(x[:num_tokens])`` over each 128-wide bf16 row with fp32 ``weight`` / ``bias``, bf16
    out (``[num_tokens, 128]``, contiguous), the indexer's ``k_norm``.

    ``x`` has a unit inner stride and 16-byte aligned rows (any row stride: the target passes the wk
    columns of the packed projection output); ``weight`` / ``bias`` are fp32 ``[128]``, contiguous and
    16-byte aligned. Other inputs run the plain torch LayerNorm (deterministic, not this kernel's bits).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"a {tokens}-token step: wk columns of the packed TP16 projection",
                spec=_spec(tokens),
            )
            for tokens in (1, 5, 40, 98, 224)
        ),
        Cell(why="a 16384-token prefill chunk", spec=_spec(16384)),
        Cell(why="a contiguous [T, 128] input", spec=_spec(77, row_stride=HEAD_DIM, column=0)),
        Cell(why="eps 1e-6 (the module default)", spec=_spec(98, eps=1e-6)),
        Cell(why="rows with a large mean: the two-pass variance", spec=_spec(98, data="offset")),
        Cell(why="all-zero and constant rows: zero variance", spec=_spec(24, data="flat")),
        Cell(why="large-magnitude rows", spec=_spec(40, data="wide")),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The indexer's key LayerNorm (``k_norm`` holds its weight, bias and epsilon). A torch.compile'd
    LayerNorm builds one persistent-reduction Triton kernel and picks its launch configuration per
    process by timing it, and the configurations sum the row in different orders: two TP ranks could
    normalize the same key a bf16 ulp apart and write different index-key cache bytes. This op has one
    order, the one of the configuration most ranks pick (``XBLOCK=8, num_warps=2``): its output equals
    that kernel's bit for bit, on any rank, run or GPU. Against the plain torch LayerNorm (``reference``)
    it differs by at most one bf16 ulp in a few elements per million.

    The indexer calls it on its key, on the same stream (beside the kv_a norm on the auxiliary stream
    under CUDA graphs, in line in eager steps). One plain launch (no PDL); compiles once, outside CUDA
    graph capture (the target's warmup step).
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        n = x.shape[0] if num_tokens is None else num_tokens
        if not (rows_supported(x) and params_supported(weight, bias)):
            return _torch_layer_norm(x, weight, bias, eps, n)
        out = torch.empty(n, HEAD_DIM, dtype=torch.bfloat16, device=x.device)
        index_key_norm_into(x, weight, bias, eps, out, n)
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """The LayerNorm in fp32 (torch's kernel), rounded to bf16."""
        return _torch_layer_norm(
            x, weight, bias, eps, x.shape[0] if num_tokens is None else num_tokens
        )

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        num_tokens: Optional[int] = None,
    ) -> None:
        assert x.dim() == 2 and x.shape[-1] == HEAD_DIM, f"x rows must be {HEAD_DIM} wide"
        assert num_tokens is None or x.shape[0] >= num_tokens, "num_tokens exceeds x"
        assert weight.numel() == HEAD_DIM and bias.numel() == HEAD_DIM, (
            "one weight and bias per column"
        )


index_key_norm = IndexKeyNorm()


def indexer_k_norm_args(indexer) -> tuple[torch.Tensor, torch.Tensor, float]:
    """``(weight, bias, eps)`` of the indexer's ``k_norm`` LayerNorm, as fp32 ``[128]`` contiguous tensors (the
    module's own parameters when they already are)."""
    norm = indexer.k_norm
    weight, bias = norm.weight, norm.bias
    if not params_supported(weight, bias):
        weight, bias = weight.float().contiguous(), bias.float().contiguous()
    return weight, bias, float(norm.variance_epsilon)


def indexer_k_norm(indexer, x: torch.Tensor, num_tokens: Optional[int] = None) -> torch.Tensor:
    """The indexer's ``k_norm`` on ``x`` through :data:`index_key_norm`."""
    return index_key_norm(x, *indexer_k_norm_args(indexer), num_tokens)
