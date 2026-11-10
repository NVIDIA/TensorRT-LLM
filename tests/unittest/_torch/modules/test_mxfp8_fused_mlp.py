# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the SM107 MXFP8 shared-expert MLP (fused FC12 and two-GEMM fallback)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import triton
from torch import nn
from utils.util import getSMVersion

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.modules.mxfp8_fused_mlp import Mxfp8SharedExpertMLP
from tensorrt_llm._torch.modules.situ import SituAndMul, situ_and_mul_kernel

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
skip_not_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or getSMVersion() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="needs SM107 and the CuTe DSL Rubin package",
)

_BETA = 4.0
_LINEAR_BETA = 25.0
# Mirrors GATE_UP_INTERLEAVE, which only imports where the CuTe DSL package exists.
_INTERLEAVE = 32


@skip_no_cuda
@pytest.mark.parametrize("num_tokens,intermediate_size", [(1, 64), (7, 3072)])
def test_situ_kernel_reads_interleaved_gate_up(num_tokens: int, intermediate_size: int) -> None:
    torch.manual_seed(0)
    gate_up = torch.randn((num_tokens, 2 * intermediate_size), dtype=torch.bfloat16, device="cuda")
    blocks = gate_up.view(num_tokens, 2, -1, _INTERLEAVE)
    interleaved = blocks.transpose(1, 2).reshape(num_tokens, -1).contiguous()
    output = gate_up.new_empty((num_tokens, intermediate_size))

    situ_and_mul_kernel[(num_tokens, triton.cdiv(intermediate_size, 1024))](
        output,
        output.stride(0),
        interleaved,
        interleaved.stride(0),
        intermediate_size,
        _BETA,
        _LINEAR_BETA,
        BLOCK_SIZE=1024,
        HAS_LINEAR_BETA=True,
        LAUNCH_WITH_PDL=False,
        GATE_UP_INTERLEAVE=_INTERLEAVE,
    )

    reference = SituAndMul(beta=_BETA, linear_beta=_LINEAR_BETA)(gate_up)
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@skip_not_sm107
def test_fused_fc12_dispatch_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.custom_ops.cute_dsl_mxfp8_shared_expert_custom_op import (
        GATE_UP_INTERLEAVE,
        use_fused_fc12,
    )
    from tensorrt_llm._torch.modules.dwdp import fabric_gather
    from tensorrt_llm._torch.modules.dwdp.dense import manager as dense_manager

    assert GATE_UP_INTERLEAVE == _INTERLEAVE
    assert not use_fused_fc12(1023, 6144)
    assert use_fused_fc12(1024, 6144)
    assert use_fused_fc12(2048, 3072)
    with with_multi_stream(True):
        assert not use_fused_fc12(2048, 3072)
    # Expert and dense DWDP prefetch on a copy stream beside the kernel.
    with monkeypatch.context() as patch:
        patch.setattr(dense_manager, "get_global_dense_dwdp_manager", lambda: object())
        assert not use_fused_fc12(2048, 3072)
    monkeypatch.setenv(fabric_gather._ENABLE_ENV, "1")
    assert not use_fused_fc12(2048, 3072)


@skip_not_sm107
def test_fused_plan_tracks_sm_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch import cute_dsl_utils
    from tensorrt_llm._torch.custom_ops import cute_dsl_mxfp8_shared_expert_custom_op as op

    monkeypatch.setattr(op, "_COMPILED_PLAN_CACHE", {})
    device = torch.device("cuda", torch.cuda.current_device())
    previous = cute_dsl_utils.sm_reserve()
    full = op._get_plan(device, 2048, 3072, _BETA, _LINEAR_BETA)
    cute_dsl_utils.set_sm_reserve(previous + 16)
    try:
        reserved = op._get_plan(device, 2048, 3072, _BETA, _LINEAR_BETA)
    finally:
        cute_dsl_utils.set_sm_reserve(previous)
    # The persistent grid must shrink once SMs are reserved after the first call.
    assert reserved is not full
    assert reserved.max_active_clusters < full.max_active_clusters


def _make_gated_mlp(
    hidden_size: int, intermediate_size: int
) -> tuple[GatedMLP, torch.Tensor, torch.Tensor]:
    mlp = GatedMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bias=False,
        dtype=torch.bfloat16,
        reduce_output=False,
        activation=SituAndMul(beta=_BETA, linear_beta=_LINEAR_BETA, use_fused_activation=True),
    ).cuda()
    gate = torch.randn((intermediate_size, hidden_size), dtype=torch.bfloat16) * 0.02
    up = torch.randn((intermediate_size, hidden_size), dtype=torch.bfloat16) * 0.02
    down = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16) * 0.02
    mlp.gate_up_proj.load_weights([{"weight": gate}, {"weight": up}])
    mlp.down_proj.load_weights([{"weight": down}])
    return mlp, torch.cat((gate, up)).cuda(), down.cuda()


def _reference(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    activation = SituAndMul(beta=_BETA, linear_beta=_LINEAR_BETA)
    return F.linear(activation(F.linear(x, gate_up)), down)


@skip_not_sm107
@torch.inference_mode()
def test_mxfp8_shared_expert_matches_bf16() -> None:
    from tensorrt_llm._torch.custom_ops.cute_dsl_mxfp8_shared_expert_custom_op import (
        _unfused_mxfp8_gated_mlp,
        use_fused_fc12,
    )

    torch.manual_seed(0)
    hidden_size, intermediate_size = 2048, 3072
    mlp, reference_gate_up, reference_down = _make_gated_mlp(hidden_size, intermediate_size)
    assert Mxfp8SharedExpertMLP.unsupported_reason(mlp) is None
    expert = Mxfp8SharedExpertMLP(mlp)
    assert expert.bf16_fallback is None

    empty = expert(torch.empty((0, hidden_size), dtype=torch.bfloat16, device="cuda"))
    assert empty.shape == (0, hidden_size)

    weights = (expert.fc1_weight, expert.fc1_scale, expert.fc2_weight, expert.fc2_scale)
    # 32 and 1024 run the two-GEMM path; 2048 and 2100 (padded to 2176 rows)
    # run the fused FC12 kernel in eager mode.
    for num_tokens in (32, 1024, 2048, 2100):
        x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
        reference = _reference(x, reference_gate_up, reference_down)
        output = expert(x)
        assert output.shape == reference.shape
        relative_l2 = (output.float() - reference.float()).norm() / reference.float().norm()
        assert relative_l2 < 0.1, (num_tokens, relative_l2.item())

        # CUDA graphs always take the two-GEMM path.
        unfused = _unfused_mxfp8_gated_mlp(x, *weights, _BETA, _LINEAR_BETA)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = expert(x)
        graph.replay()
        torch.testing.assert_close(captured, unfused, rtol=0, atol=0)
        if not use_fused_fc12(num_tokens, intermediate_size):
            torch.testing.assert_close(output, unfused, rtol=0, atol=0)


@skip_not_sm107
@torch.inference_mode()
def test_tuning_fused_fc12_also_runs_the_two_gemm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.custom_ops import cute_dsl_mxfp8_shared_expert_custom_op as op

    torch.manual_seed(0)
    hidden_size, intermediate_size = 2048, 3072
    expert = Mxfp8SharedExpertMLP(_make_gated_mlp(hidden_size, intermediate_size)[0])
    x = torch.randn((2048, hidden_size), dtype=torch.bfloat16, device="cuda")
    assert op.use_fused_fc12(x.shape[0], intermediate_size)
    # The two-GEMM path tunes its GEMMs when it runs under autotune; only the call is checked here.
    calls = []
    monkeypatch.setattr(
        op,
        "_unfused_mxfp8_gated_mlp",
        lambda input, *args: calls.append(input.shape) or torch.zeros_like(input),
    )

    expert(x)
    assert calls == []
    with autotune():
        expert(x)
    assert calls == [x.shape]


@skip_not_sm107
@torch.inference_mode()
def test_compiled_graphs_never_select_fused_fc12(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    expert = Mxfp8SharedExpertMLP(_make_gated_mlp(2048, 3072)[0])
    calls = []
    monkeypatch.setattr(expert, "_op", lambda *args: calls.append(args[-1]) or args[0])
    x = torch.randn((2048, 2048), dtype=torch.bfloat16, device="cuda")

    expert(x)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    expert(x)
    assert calls == [True, False]


@skip_not_sm107
@torch.inference_mode()
def test_kimi_k3_load_converts_shared_experts(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.models import modeling_kimi_linear as kimi

    torch.manual_seed(0)
    hidden_size = 2048
    model = nn.Module()
    model.layers = nn.ModuleList()
    references = []
    # The second shared expert is too narrow for MXFP8 and keeps the FP8 weight read.
    for intermediate_size in (3072, 1536):
        layer = nn.Module()
        layer.block_sparse_moe = nn.Module()
        mlp, gate_up, down = _make_gated_mlp(hidden_size, intermediate_size)
        layer.block_sparse_moe.shared_experts = mlp
        model.layers.append(layer)
        references.append((gate_up, down))
    original = model.layers[0].block_sparse_moe.shared_experts

    # Both variables set: MXFP8 runs first, the FP8 weight read takes the rest.
    converted, kept = kimi._convert_shared_experts_to_mxfp8(model)
    assert converted == 1
    assert kept == {"local intermediate size below 3072": 1}
    assert kimi._convert_moe_mlps_to_fp8_weight_read(model) == 2
    expert = model.layers[0].block_sparse_moe.shared_experts
    assert isinstance(expert, Mxfp8SharedExpertMLP)
    # The replicated expert frees its BF16 weights.
    assert original.gate_up_proj.weight.numel() == 0
    assert original.down_proj.weight.numel() == 0

    # The loader then walks post-load hooks parents first.
    for module in model.modules():
        if hasattr(module, "post_load_weights"):
            module.post_load_weights()
    x = torch.randn((32, hidden_size), dtype=torch.bfloat16, device="cuda")
    reference = _reference(x, *references[0])
    relative_l2 = (expert(x).float() - reference.float()).norm() / reference.float().norm()
    assert relative_l2 < 0.1, relative_l2.item()
    with pytest.raises(NotImplementedError):
        expert.pre_reload_weights()
    # Beside the routed grid, the CuTe DSL GEMMs take no SM carveout.
    moe = SimpleNamespace(shared_experts=expert, _shared_sms=8)
    with with_multi_stream(True):
        assert kimi.KimiK3MoERuntime._shared_sms_now(moe) == 0
