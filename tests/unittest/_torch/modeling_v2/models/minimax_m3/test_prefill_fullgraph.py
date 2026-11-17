# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile real prefill host dispatch, substituting CPU math for GPU kernels."""

from dataclasses import dataclass
from types import ModuleType

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling as tp2,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (
    modeling as tp4,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8 import (
    modeling as tp8,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.m3_dense_swiglu import (
    ALPHA,
    BETA,
    LIMIT,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.gemm.mxfp8_state import (
    _mx_mode,
    _Mxfp8GemmState,
)
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture
from tensorrt_llm._torch.utils import torch_compiling

pytestmark = pytest.mark.cpu_only
_Prefill = tp2.PrefillTarget | tp4.PrefillTarget | tp8.PrefillTarget


@dataclass
class _Metadata:
    num_contexts: int
    num_ctx_tokens: int
    num_tokens: int
    kv_cache_manager: object = None
    msa_idx_k_cache: object = None
    msa_out_cache_loc: object = None


@pytest.mark.parametrize("modeling,tp", [(tp2, 2), (tp4, 4), (tp8, 8)])
def test_prefill_fullgraph_keeps_token_rows_dynamic(
    monkeypatch: pytest.MonkeyPatch, modeling: ModuleType, tp: int
) -> None:
    # Construct the real core without allocating checkpoint weights or GPU resources.
    core = object.__new__(getattr(modeling, f"MiniMaxM3Tp{tp}ModelingV2"))
    torch.nn.Module.__init__(core)
    core.num_layers = 1
    core.mxfp8 = _Mxfp8GemmState()
    core.mx_alpha = torch.ones(1)
    core.swiglu_alpha = ALPHA
    core.swiglu_limit = LIMIT
    core.swiglu_beta = BETA
    generator = torch.Generator().manual_seed(7090784)
    core.w = {
        "mlp_gu_w": torch.randn(32, 16, generator=generator),
        "mlp_gu_s": torch.empty(0),
        "mlp_dn_w": torch.randn(16, 16, generator=generator),
        "mlp_dn_s": torch.empty(0),
    }

    def gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        alpha: torch.Tensor,
        dtype: torch.dtype,
        tuned: bool,
        tune: bool,
        flashinfer: bool,
        native_tuning: bool,
    ) -> torch.Tensor:
        assert not tuned and not tune and not flashinfer
        return x @ weight.t()

    def swiglu(x: torch.Tensor, limit: float, alpha: float, beta: float) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up.clamp(-limit, limit) + beta)

    # Exercise the real forward, projection dispatch and dense activation policy.
    # Attention, collectives and native kernel numerics have separate GPU tests.
    def layer(
        self: _Prefill,
        i: int,
        position_ids: torch.Tensor | None,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        step: dict[str, object],
        spec_metadata: object,
        capture: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._swiglu_mlp(hidden, "mlp", False, dense=True), residual

    monkeypatch.setattr(modeling, "m3_mxfp8_linear", gemm)
    monkeypatch.setattr(modeling, "m3_silu_and_mul", swiglu)
    monkeypatch.setattr(modeling.PrefillTarget, "_layer", layer)
    monkeypatch.setattr(modeling, "_require", [False])
    target = modeling.PrefillTarget(core)

    def forward(x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        return target.forward(_Metadata(1, rows, rows), None, None, x, None, None, None)

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(forward, fullgraph=True, dynamic=True, backend=counter)
    # The compile policy must win over a surrounding decode-graph tuning scope,
    # just as it does for the built-in MXFP8 linear implementation.
    with flashinfer_mxfp8_decode_graph_capture():
        assert _mx_mode(core.mxfp8).decode_graph
        with torch_compiling(True):
            assert not _mx_mode(core.mxfp8).decode_graph
            for rows in range(9, 49):
                x = torch.randn(rows, 16, generator=generator)
                expected = (
                    swiglu(
                        x @ core.w["mlp_gu_w"].t(),
                        core.swiglu_limit,
                        core.swiglu_alpha,
                        core.swiglu_beta,
                    )
                    @ core.w["mlp_dn_w"].t()
                )
                torch.testing.assert_close(compiled(x), expected)
        assert _mx_mode(core.mxfp8).decode_graph
    assert counter.frame_count == 1
