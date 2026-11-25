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
"""Registration contract of fast_custom_op and of the trtllm ops registered through it."""

import itertools
import operator
import os
import subprocess
import sys
from collections.abc import Callable

import pytest
import torch
import torch.fx
from torch._library.simple_registry import singleton as simple_registry

import tensorrt_llm  # noqa: F401

# Ops registered outside the custom_ops package, which does not import these modules.
import tensorrt_llm._torch.attention.backends.sparse.dsa.custom_ops  # noqa: F401
import tensorrt_llm._torch.attention.mla  # noqa: F401
import tensorrt_llm._torch.modules.mamba.layernorm_gated  # noqa: F401
import tensorrt_llm._torch.modules.qwen4_exp.hyper_connection_kernels  # noqa: F401
import tensorrt_llm._torch.modules.situ  # noqa: F401
from tensorrt_llm._torch.compilation.recover_pass import recover_pass
from tensorrt_llm._torch.custom_ops import fast_custom_op as fast_custom_op_module
from tensorrt_llm._torch.custom_ops.fast_custom_op import FastCustomOp, fast_custom_op

pytestmark = pytest.mark.cpu_only

_NAMESPACE = "trtllm_fast_custom_op_test"
_op_ids = itertools.count()


def _qualname(stem: str) -> str:
    # Ops cannot be redefined, so every registration in this file gets a fresh name.
    return f"{_NAMESPACE}::{stem}_{next(_op_ids)}"


def _register_add_one(**kwargs: object) -> FastCustomOp:
    @fast_custom_op(_qualname("add_one"), **kwargs)
    def add_one(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    @add_one.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    return add_one


def test_device_agnostic_registration_runs_on_cpu() -> None:
    op = _register_add_one(device_types=None)
    torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))


@pytest.mark.parametrize("device_types", ["cuda", "CUDA", ("cuda", "cpu")])
def test_accepts_device_types_and_dispatch_keys(device_types: str | tuple[str, ...]) -> None:
    op = _register_add_one(device_types=device_types)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(op.qualname, "CUDA")
    if "cpu" in device_types:
        torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))


def test_default_registers_cuda_only() -> None:
    op = _register_add_one()
    assert torch._C._dispatch_has_kernel_for_dispatch_key(op.qualname, "CUDA")
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(op.qualname, "CPU")
    with pytest.raises(NotImplementedError):
        op(torch.zeros(3))


def test_schema_has_pt2_compliant_tag_and_the_given_tags() -> None:
    op = _register_add_one(device_types=None, tags=(torch.Tag.needs_fixed_stride_order,))
    tags = op._op.tags
    assert torch.Tag.pt2_compliant_tag in tags
    assert torch.Tag.needs_fixed_stride_order in tags


def test_call_traces_to_the_default_overload() -> None:
    op = _register_add_one(device_types=None)
    targets = []

    def capture(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
        targets.extend(n.target for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(lambda x: op(x) * 2, backend=capture, fullgraph=True)
    torch.testing.assert_close(compiled(torch.zeros(3)), torch.full((3,), 2.0))
    assert op._op in targets
    assert isinstance(op._op, torch._ops.OpOverload)


def _register_mm_out(**kwargs: object) -> FastCustomOp:
    @fast_custom_op(_qualname("mm_out"), mutates_args=("out",), device_types=None, **kwargs)
    def mm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        torch.mm(a, b, out=out)

    @mm_out.register_fake
    def _(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        return None

    return mm_out


def test_body_runs_with_the_callers_grad_mode_by_default() -> None:
    op = _register_mm_out()
    a = torch.randn(2, 3, requires_grad=True)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="out="):
        op(a, torch.randn(3, 4), torch.empty(2, 4))


def test_disable_grad_runs_the_body_without_grad() -> None:
    op = _register_mm_out(disable_grad=True)
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(3, 4)
    out = torch.empty(2, 4)
    with torch.enable_grad():
        op(a, b, out)
    torch.testing.assert_close(out, (a @ b).detach())
    assert not out.requires_grad


def test_validate_mode_restores_the_alias_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_VALIDATE_CUSTOM_OPS", "1")

    @fast_custom_op(_qualname("returns_input"), device_types=None)
    def returns_input(x: torch.Tensor) -> torch.Tensor:
        return x

    assert isinstance(returns_input, torch.library.CustomOpDef)
    assert returns_input._name not in fast_custom_op_module._FAST_OPS
    with pytest.raises(RuntimeError, match="alias"):
        returns_input(torch.zeros(3))


@pytest.mark.parametrize("device_types", [None, "CUDA", "cuda", ("CUDA", "CPU")])
def test_validate_mode_device_types_and_tags(
    monkeypatch: pytest.MonkeyPatch, device_types: str | tuple[str, ...] | None
) -> None:
    monkeypatch.setenv("TLLM_VALIDATE_CUSTOM_OPS", "1")
    op = _register_add_one(device_types=device_types, tags=(torch.Tag.needs_fixed_stride_order,))
    assert isinstance(op, torch.library.CustomOpDef)
    overload = op._opoverload
    assert torch.Tag.pt2_compliant_tag in overload.tags
    assert torch.Tag.needs_fixed_stride_order in overload.tags
    if device_types is None or "CPU" in device_types:
        torch.testing.assert_close(op(torch.zeros(3)), torch.ones(3))


def test_validate_mode_imports_the_custom_ops() -> None:
    env = dict(os.environ, TLLM_VALIDATE_CUSTOM_OPS="1")
    subprocess.run(
        [sys.executable, "-c", "import tensorrt_llm._torch.custom_ops"],
        env=env,
        check=True,
        timeout=600,
    )


# Fast ops whose modules import on any machine. Listed so that an import guard turning off
# cannot empty the registry without this file noticing.
_EXPECTED_FAST_OPS = {
    "trtllm::bmm_out",
    "trtllm::fused_moe",
    "trtllm::fp8_rowwise_gemm",
    "trtllm::fp8_swap_ab_gemm",
    "trtllm::finegrained_mixed_dtype_gemm",
    "trtllm::weight_only_quant_gemm",
    "trtllm::w4a8_mxfp4_fp8_gemm",
    "trtllm::silu_and_mul",
    "trtllm::nvfp4_gemm",
    "trtllm::nvfp4_sfc_finalize",
    "trtllm::fp4_block_scale_moe_runner",
    "trtllm::fp8_block_scale_moe_runner",
    "trtllm::mxe4m3_mxe2m1_block_scale_moe_runner",
    "trtllm::create_mla_outputs",
    "trtllm::mla_custom_op_inplace",
    "trtllm::mla_dsa_proj",
    "trtllm::mla_dsa_attn_inplace",
    "trtllm::rms_norm_gated_token_major",
    "trtllm::situ_and_mul",
    "trtllm::qwen4_exp_hc_silu",
    "trtllm::qwen4_exp_hc_gate_mix",
    "trtllm::qwen4_exp_hc_combine_norm",
}


def test_expected_trtllm_fast_ops_are_registered() -> None:
    missing = _EXPECTED_FAST_OPS - fast_custom_op_module._FAST_OPS.keys()
    assert not missing, f"Not registered through fast_custom_op: {sorted(missing)}"


_TRTLLM_FAST_OPS = sorted(
    name for name in fast_custom_op_module._FAST_OPS if name.startswith("trtllm::")
)


@pytest.mark.parametrize("qualname", _TRTLLM_FAST_OPS)
def test_trtllm_fast_op_registration(qualname: str) -> None:
    op = fast_custom_op_module._FAST_OPS[qualname]
    # An op without a fake that is registered for every device would run its real body on
    # fake tensors during tracing instead of failing.
    assert simple_registry.find(qualname).fake_impl.kernel is not None, f"{qualname} has no fake"
    assert torch.Tag.pt2_compliant_tag in op._op.tags
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(qualname, "Autograd")


@pytest.mark.skipif(
    not hasattr(torch.ops.trtllm, "flashinfer_fused_add_rmsnorm"), reason="needs flashinfer"
)
@pytest.mark.parametrize("overload", [False, True], ids=["packet", "default"])
def test_recover_pass_splits_fused_add_rmsnorm(overload: bool) -> None:
    fused = torch.ops.trtllm.flashinfer_fused_add_rmsnorm
    graph = torch.fx.Graph()
    x, residual, weight = (graph.placeholder(name) for name in ("x", "residual", "weight"))
    graph.call_function(fused.default if overload else fused, (x, residual, weight, 1e-6))
    graph.output((x, residual))
    gm = recover_pass(torch.fx.GraphModule(torch.nn.Module(), graph))
    calls = [n for n in gm.graph.nodes if n.op == "call_function"]
    assert [n.target for n in calls] == [operator.add, torch.ops.trtllm.flashinfer_rmsnorm.default]
    add_node, norm_node = calls
    assert add_node.args == (x, residual)
    assert norm_node.args == (add_node, weight, 1e-6)
    (output,) = (n for n in gm.graph.nodes if n.op == "output")
    assert output.args == ((norm_node, add_node),)
