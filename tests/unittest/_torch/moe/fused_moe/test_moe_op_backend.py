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

from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops
from tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant import add_rmsnorm_fp4_quant
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoECommPlan, MoERunContext
from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import FlashinferOpBackend, TRTLLMOpBackend
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.fp4_block_scale import TRTLLMGenFp4BlockScaleBase
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.trtllm_nvfp4 import TrtllmTrtllmGenNvfp4Impl
from tensorrt_llm._torch.utils import ActivationType

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def backend():
    # Dispatch uses no instance state; avoid initializing communication helpers.
    return TRTLLMOpBackend.__new__(TRTLLMOpBackend)


@pytest.mark.parametrize("padded_hidden_size", [64, 80])
@pytest.mark.parametrize("supports_padding", [False, True])
def test_add_rmsnorm_nvfp4_outputs_are_zero_copy_uint8_views(
    padded_hidden_size: int, supports_padding: bool
) -> None:
    x = torch.zeros((2, 64), dtype=torch.bfloat16)
    residual = torch.zeros_like(x)
    direct_padding = supports_padding and padded_hidden_size != x.shape[1]
    output_width = padded_hidden_size if direct_padding else x.shape[1]
    fp4 = (
        torch.arange(output_width, dtype=torch.uint8)
        .view(2, output_width // 2)
        .view(torch.float4_e2m1fn_x2)
    )
    sf = (
        torch.arange(output_width // 8, dtype=torch.uint8)
        .view(2, output_width // 16)
        .view(torch.float8_e4m3fn)
    )
    storage = [fp4, sf]

    def pad(payload, scales, width):
        assert payload is fp4 and scales is sf
        assert width == padded_hidden_size
        storage[:] = [
            torch.nn.functional.pad(t.view(torch.uint8), (0, width // divisor - t.shape[1])).view(
                t.dtype
            )
            for t, divisor in ((payload, 2), (scales, 16))
        ]
        return tuple(storage)

    module = "tensorrt_llm._torch.modules.fused_ops.rmsnorm_fp4_quant"
    with (
        patch(f"{module}._ADD_RMSNORM_FP4QUANT_SUPPORTS_PADDING", supports_padding),
        patch(f"{module}._add_rmsnorm_fp4quant", return_value=(fp4, sf)) as fused,
        patch.object(torch.ops.trtllm, "pad_nvfp4_activation", side_effect=pad) as padding,
    ):
        packed, scales, norm_out = add_rmsnorm_fp4_quant(
            x, residual, torch.ones(64, dtype=x.dtype), 1e-5, torch.ones(1), padded_hidden_size
        )

    fused.assert_called_once()
    assert norm_out is fused.call_args.kwargs["y_out"]
    assert norm_out.shape == x.shape
    assert norm_out.dtype == x.dtype
    if direct_padding:
        assert fused.call_args.kwargs["output_hidden_size"] == padded_hidden_size
    else:
        assert "output_hidden_size" not in fused.call_args.kwargs
    if padded_hidden_size != x.shape[1] and not direct_padding:
        padding.assert_called_once_with(fp4, sf, padded_hidden_size)
    else:
        padding.assert_not_called()
    for actual, original in zip((packed, scales), storage):
        assert actual.dtype == torch.uint8
        assert actual.shape == original.shape
        assert actual.data_ptr() == original.data_ptr()
        torch.testing.assert_close(actual, original.view(torch.uint8), rtol=0, atol=0)


def _fp4_moe_args():
    # Only dispatch is under test: the native CUDA runners are mocked below.
    return dict(
        router_logits=None,
        routing_bias=None,
        hidden_states=torch.zeros((1, 32), dtype=torch.uint8),
        hidden_states_scale=torch.zeros((1, 4), dtype=torch.uint8),
        gemm1_weights=torch.zeros((2, 128, 32), dtype=torch.uint8),
        gemm1_weights_scale=torch.zeros((2, 128, 4), dtype=torch.uint8),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=torch.zeros((2, 64, 32), dtype=torch.uint8),
        gemm2_weights_scale=torch.zeros((2, 64, 4), dtype=torch.uint8),
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=2,
        top_k=1,
        n_group=None,
        topk_group=None,
        intermediate_size=64,
        local_expert_offset=0,
        local_num_experts=2,
        routed_scaling_factor=None,
        routing_method_type=0,
        output=torch.full((1, 64), float("nan"), dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("do_finalize", [False, True])
def test_nvfp4_uint8_dispatch_writes_output(do_finalize, backend):
    args = _fp4_moe_args()
    output = args["output"]

    def run(*op_args, **op_kwargs):
        assert op_args[2] is args["hidden_states"]
        assert op_args[3].dtype == torch.float8_e4m3fn
        assert op_args[3].data_ptr() == args["hidden_states_scale"].data_ptr()
        assert op_kwargs["output"] is output
        assert op_kwargs["do_finalize"] is do_finalize
        output.fill_(1)
        return [output]

    with patch.object(torch.ops.trtllm, "fp4_block_scale_moe_runner", side_effect=run) as runner:
        result = backend.run_fp4_block_scale_moe(**args, do_finalize=do_finalize)

    runner.assert_called_once()
    assert (result if do_finalize else result[0]) is output
    torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.parametrize("dtype", [torch.float4_e2m1fn_x2, torch.float32])
def test_fp4_moe_rejects_unsupported_activation_dtype(dtype, backend):
    args = _fp4_moe_args()
    args["hidden_states"] = torch.empty((1, 32), dtype=dtype)
    with patch.object(torch.ops.trtllm, "fp4_block_scale_moe_runner") as runner:
        with pytest.raises(ValueError, match="Unsupported FP4 MoE input: hidden_states.dtype"):
            backend.run_fp4_block_scale_moe(**args)

    runner.assert_not_called()
    assert torch.isnan(args["output"]).all()


@pytest.mark.parametrize("scale_width", [None, 3])
def test_fp4_moe_rejects_unsupported_weight_scale_layout(scale_width, backend):
    args = _fp4_moe_args()
    args["gemm1_weights_scale"] = (
        None if scale_width is None else torch.zeros((2, 128, scale_width), dtype=torch.uint8)
    )
    with patch.object(torch.ops.trtllm, "fp4_block_scale_moe_runner") as runner:
        with pytest.raises(ValueError, match="Unsupported FP4 MoE input:.*gemm1_weights_scale"):
            backend.run_fp4_block_scale_moe(**args)

    runner.assert_not_called()
    assert torch.isnan(args["output"]).all()


@pytest.mark.parametrize(
    "dtype, op_name",
    [
        (torch.float8_e4m3fn, "mxe4m3_mxe2m1_block_scale_moe_runner"),
        (torch.bfloat16, "bf16_mxe2m1_block_scale_moe_runner"),
    ],
)
def test_fp4_moe_preserves_other_supported_dispatch_paths(dtype, op_name, backend):
    args = _fp4_moe_args()
    args["hidden_states"] = torch.empty((1, 64), dtype=dtype)
    args["gemm1_weights_scale"] = torch.zeros((2, 128, 2), dtype=torch.uint8)
    output = args["output"]
    with patch.object(torch.ops.trtllm, op_name, return_value=output) as runner:
        result = backend.run_fp4_block_scale_moe(**args)
    runner.assert_called_once()
    assert runner.call_args.args[2] is args["hidden_states"]
    assert runner.call_args.kwargs["output"] is output
    assert "output_hidden_size" not in runner.call_args.kwargs
    assert result is output


def _gptoss_run_state(
    op_backend: TRTLLMOpBackend, hidden_size: int, padded_hidden_size: int
) -> SimpleNamespace:
    # Exercise run_moe without GPU-only construction or checkpoint loading.
    state = SimpleNamespace(
        op_backend=op_backend,
        _enable_direct_moe_output=True,
        has_any_quant=True,
        has_deepseek_fp8_block_scales=False,
        has_nvfp4=True,
        has_w4a16_mxfp4=False,
        has_w4a8_mxfp4_mxfp8=False,
        use_flashinfer=False,
        bias=True,
        activation_type=ActivationType.SwigluBias,
        hidden_size=hidden_size,
        mapping=SimpleNamespace(moe_tp_size=1, moe_ep_size=1),
        w3_w1_weight=torch.zeros((2, 128, padded_hidden_size // 2), dtype=torch.uint8),
        w3_w1_weight_scale=torch.zeros((2, 128, padded_hidden_size // 16), dtype=torch.uint8),
        w3_w1_bias=torch.zeros((2, 128)),
        w2_weight=torch.zeros((2, padded_hidden_size, 32), dtype=torch.uint8),
        w2_weight_scale=torch.zeros((2, padded_hidden_size, 4), dtype=torch.uint8),
        w2_bias=torch.zeros((2, padded_hidden_size)),
        act_alpha=torch.ones(2),
        act_beta=torch.ones(2),
        act_clamp=torch.ones(2),
        quant_method=SimpleNamespace(intermediate_size_per_partition_lean=64),
        num_slots=2,
        slot_start=0,
        expert_size_per_partition=2,
        max_num_tokens=129,
        use_dp=False,
        reduce_results=False,
        routing_method=SimpleNamespace(top_k=1, routing_method_type=0),
        _routes_outside_the_kernel=lambda: False,
    )
    state._get_direct_moe_output_hidden_size = MethodType(
        TrtllmTrtllmGenNvfp4Impl._get_direct_moe_output_hidden_size, state
    )
    state._unfinalized = MethodType(TRTLLMGenFp4BlockScaleBase._unfinalized, state)
    return state


def _gptoss_run_context(num_tokens: int, padded_hidden_size: int) -> MoERunContext:
    x = torch.zeros((num_tokens, padded_hidden_size // 2), dtype=torch.uint8)
    x[:, 0] = torch.arange(num_tokens, dtype=torch.int64).remainder(16).to(torch.uint8)
    return MoERunContext(
        x=x,
        x_sf=torch.zeros((num_tokens, padded_hidden_size // 16), dtype=torch.uint8),
        router_logits=torch.zeros((num_tokens, 2)),
        token_selected_experts=None,
        token_final_scales=None,
        comm_plan=MoECommPlan(
            input_sf_swizzled=True,
            enable_alltoall=False,
            moe_output=None,
            payload_in_workspace=False,
        ),
    )


def _finalizer_values(x: torch.Tensor, width: int) -> torch.Tensor:
    # Small exactly representable BF16 values make row/column corruption visible.
    return (x[:, :1].float() + torch.arange(width).remainder(8)).to(torch.bfloat16)


def _run_mock_finalizer(
    state: SimpleNamespace, ctx: MoERunContext
) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor, Mock]:
    native_outputs = []
    op_name = (
        "bf16_mxe2m1_block_scale_moe_runner"
        if state.has_w4a16_mxfp4
        else "fp4_block_scale_moe_runner"
    )

    def finalize(*op_args, **op_kwargs) -> torch.Tensor | list[torch.Tensor]:
        x = op_args[2]
        output = op_kwargs["output"]
        if output is None:
            width = op_kwargs.get("output_hidden_size")
            if width is None:
                width = state.w3_w1_weight.shape[-1] * 2
            output = _finalizer_values(x, width)
        else:
            output.copy_(_finalizer_values(x, output.shape[1]))
        native_outputs.append(output)
        if op_kwargs["output"] is not None and op_kwargs["do_finalize"]:
            # Match the functional custom-op sentinel for supplied workspaces.
            return [output.new_empty((0,))]
        return output if state.has_w4a16_mxfp4 else [output]

    with patch.object(torch.ops.trtllm, op_name, side_effect=finalize) as op:
        result = TRTLLMGenFp4BlockScaleBase.run_moe(state, ctx)
    op.assert_called_once()
    return result, native_outputs[0], op


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 32, 129])
@pytest.mark.parametrize("hidden_size,padded_hidden_size", [(32, 64), (64, 96), (2880, 3072)])
def test_gptoss_direct_output_preserves_exact_logical_values(
    num_tokens: int, hidden_size: int, padded_hidden_size: int, backend: TRTLLMOpBackend
) -> None:
    state = _gptoss_run_state(backend, hidden_size, padded_hidden_size)
    ctx = _gptoss_run_context(num_tokens, padded_hidden_size)
    input_before = ctx.x.clone()
    state._enable_direct_moe_output = False
    incumbent, padded, old_op = _run_mock_finalizer(state, ctx)
    state._enable_direct_moe_output = True
    direct, native, new_op = _run_mock_finalizer(state, ctx)

    assert old_op.call_args.kwargs["output_hidden_size"] is None
    assert new_op.call_args.kwargs["output_hidden_size"] == hidden_size
    assert padded.shape == (num_tokens, padded_hidden_size)
    assert direct is native
    assert direct.shape == incumbent.shape == (num_tokens, hidden_size)
    assert direct.dtype == incumbent.dtype == torch.bfloat16
    assert direct.is_contiguous() and incumbent.is_contiguous()
    torch.testing.assert_close(direct, incumbent, rtol=0, atol=0)
    torch.testing.assert_close(direct, _finalizer_values(ctx.x, hidden_size), rtol=0, atol=0)
    torch.testing.assert_close(ctx.x, input_before, rtol=0, atol=0)
    assert ctx.comm_plan.moe_output is None
    for op in (old_op, new_op):
        assert op.call_args.args[4] is state.w3_w1_weight
        assert op.call_args.kwargs["tune_max_num_tokens"] == state.max_num_tokens


@pytest.mark.parametrize(
    "excluded",
    ["disabled", "flashinfer", "no_bias", "swiglu", "tp2", "ep8", "alltoall", "mxfp4"],
)
def test_gptoss_direct_output_preserves_ineligible_paths(
    excluded: str, backend: TRTLLMOpBackend
) -> None:
    state = _gptoss_run_state(backend, 32, 64)
    ctx = _gptoss_run_context(2, 64)
    if excluded == "disabled":
        state._enable_direct_moe_output = False
    elif excluded == "flashinfer":
        state.use_flashinfer = True
    elif excluded == "no_bias":
        state.bias = False
    elif excluded == "swiglu":
        state.activation_type = ActivationType.Swiglu
    elif excluded == "tp2":
        state.mapping.moe_tp_size = 2
    elif excluded == "ep8":
        state.mapping.moe_ep_size = 8
    elif excluded == "alltoall":
        ctx = replace(ctx, comm_plan=replace(ctx.comm_plan, enable_alltoall=True))
    elif excluded == "mxfp4":
        state.has_nvfp4 = False
        state._get_direct_moe_output_hidden_size = MethodType(
            TRTLLMGenFp4BlockScaleBase._get_direct_moe_output_hidden_size, state
        )
        state.has_w4a16_mxfp4 = True
        state.w3_w1_weight_scale = torch.zeros((2, 128, 2), dtype=torch.uint8)
        ctx = replace(ctx, x=torch.zeros((2, 64), dtype=torch.bfloat16))

    result, native, op = _run_mock_finalizer(state, ctx)
    assert op.call_args.kwargs.get("output_hidden_size") is None
    assert native.shape == (2, 64)
    assert result.shape == (2, 32) and result.is_contiguous()
    torch.testing.assert_close(result, _finalizer_values(ctx.x, 32), rtol=0, atol=0)


@pytest.mark.parametrize("hidden_size,padded_hidden_size", [(64, 64), (48, 64)])
def test_gptoss_direct_output_preserves_unpadded_and_unaligned_widths(
    hidden_size: int, padded_hidden_size: int, backend: TRTLLMOpBackend
) -> None:
    state = _gptoss_run_state(backend, hidden_size, padded_hidden_size)
    ctx = _gptoss_run_context(2, padded_hidden_size)
    result, native, op = _run_mock_finalizer(state, ctx)
    assert op.call_args.kwargs["output_hidden_size"] is None
    assert result.shape == (2, hidden_size) and result.is_contiguous()
    if hidden_size == padded_hidden_size:
        assert result is native
    torch.testing.assert_close(result, _finalizer_values(ctx.x, hidden_size), rtol=0, atol=0)


@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("enable_alltoall", [False, True])
def test_gptoss_direct_output_preserves_supplied_workspace(
    ep_size: int, enable_alltoall: bool, backend: TRTLLMOpBackend
) -> None:
    state = _gptoss_run_state(backend, 32, 64)
    state.mapping.moe_ep_size = ep_size
    ctx = _gptoss_run_context(2, 64)
    storage = torch.full((4, 32), float("nan"), dtype=torch.bfloat16)
    output = storage[1:3]
    plan = replace(
        ctx.comm_plan,
        moe_output=output,
        enable_alltoall=enable_alltoall,
        payload_in_workspace=True,
    )
    ctx = replace(ctx, comm_plan=plan)
    result, native, op = _run_mock_finalizer(state, ctx)

    assert result is native is output is plan.moe_output
    assert op.call_args.kwargs["output"] is output
    assert op.call_args.kwargs["output_hidden_size"] is None
    assert result.is_contiguous()
    assert torch.isnan(storage[[0, 3]]).all()
    torch.testing.assert_close(output, _finalizer_values(ctx.x, 32), rtol=0, atol=0)


def test_gptoss_direct_output_preserves_nonfinalized_output(backend: TRTLLMOpBackend) -> None:
    state = _gptoss_run_state(backend, 32, 64)
    ctx = replace(_gptoss_run_context(2, 64), do_finalize=False)
    result, native, op = _run_mock_finalizer(state, ctx)
    assert op.call_args.kwargs["output_hidden_size"] is None
    assert isinstance(result, list) and result[0] is native
    assert native.shape == (2, 64)


def test_flashinfer_rejects_direct_output_metadata() -> None:
    backend = FlashinferOpBackend.__new__(FlashinferOpBackend)
    backend._fused_moe = Mock()
    args = _fp4_moe_args()
    args["output"] = None
    with pytest.raises(ValueError, match="output_hidden_size"):
        backend.run_fp4_block_scale_moe(**args, output_hidden_size=32)
    backend._fused_moe.trtllm_fp4_block_scale_moe.assert_not_called()
    backend._fused_moe.trtllm_fp4_block_scale_routed_moe.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_mxfp4_rejects_direct_output_metadata(dtype, backend: TRTLLMOpBackend) -> None:
    args = _fp4_moe_args()
    args["hidden_states"] = torch.empty((1, 64), dtype=dtype)
    args["gemm1_weights_scale"] = torch.zeros((2, 128, 2), dtype=torch.uint8)
    args["output"] = None
    with pytest.raises(ValueError, match="output_hidden_size"):
        backend.run_fp4_block_scale_moe(**args, output_hidden_size=32)


def _functional_fp4_moe_args() -> dict:
    args = _fp4_moe_args()
    args["routing_logits"] = args.pop("router_logits")
    args["output"] = None
    args["do_finalize"] = True
    return args


@pytest.mark.parametrize("output_kind", ["padded", "direct", "supplied"])
def test_fp4_custom_op_allocation_and_fake_contract(output_kind: str) -> None:
    args = _functional_fp4_moe_args()
    args["hidden_states"] = torch.full((2, 32), 3, dtype=torch.uint8)
    args["hidden_states_scale"] = torch.zeros((2, 4), dtype=torch.uint8)
    width = 64 if output_kind == "padded" else 32
    if output_kind == "direct":
        args["output_hidden_size"] = width
    elif output_kind == "supplied":
        args["output"] = torch.full((2, width), float("nan"), dtype=torch.bfloat16)

    events = []
    native_outputs = []
    tensor_new_empty = torch.Tensor.new_empty

    def allocate(
        tensor: torch.Tensor, shape: tuple[int, ...], *allocation_args, **allocation_kwargs
    ) -> torch.Tensor:
        if output_kind == "direct" and shape == (2, width):
            assert events == ["tune"]
            events.append("allocate")
        return tensor_new_empty(tensor, shape, *allocation_args, **allocation_kwargs)

    def finalize(
        input_tensors: list[torch.Tensor | None], *, tactic: list[int], output: torch.Tensor | None
    ) -> list[torch.Tensor]:
        assert events == (["tune", "allocate"] if output_kind == "direct" else ["tune"])
        assert tactic == [-1, -1]
        if output_kind == "padded":
            assert output is None
            output = torch.empty((2, width), dtype=torch.bfloat16)
        elif output_kind == "direct":
            assert output is not args["output"]
            assert output.shape == (2, width)
            assert output.dtype == torch.bfloat16 and output.is_contiguous()
        else:
            assert output is args["output"]
        output.copy_(_finalizer_values(input_tensors[2], width))
        native_outputs.append(output)
        events.append("finalize")
        return [output]

    kernel_runner = Mock(side_effect=finalize)
    kernel_runner.tuning_config = None

    def choose_one(
        name: str, runners: list[Mock], config: object, input_tensors: list[torch.Tensor | None]
    ) -> tuple[Mock, int]:
        assert name == "trtllm::fp4_block_scale_moe_runner"
        assert runners == [kernel_runner]
        assert input_tensors[2] is args["hidden_states"]
        assert input_tensors[4] is args["gemm1_weights"]
        events.append("tune")
        return kernel_runner, -1

    tuner = Mock()
    tuner.choose_one.side_effect = choose_one
    with (
        patch.object(torch.Tensor, "new_empty", new=allocate),
        patch.object(trtllm_gen_custom_ops.AutoTuner, "get", return_value=tuner),
        patch.object(trtllm_gen_custom_ops, "FP4BlockScaleMoERunner", return_value=kernel_runner),
        patch.object(
            trtllm_gen_custom_ops,
            "prepare_dummy_topk_and_hook",
            return_value=(None, None, None, None),
        ),
    ):
        result = trtllm_gen_custom_ops.fp4_block_scale_moe_runner(**args)

    assert events == (
        ["tune", "allocate", "finalize"] if output_kind == "direct" else ["tune", "finalize"]
    )
    actual = native_outputs[0] if output_kind == "supplied" else result[0]
    torch.testing.assert_close(
        actual, _finalizer_values(args["hidden_states"], width), rtol=0, atol=0
    )
    assert actual.is_contiguous()
    if output_kind == "supplied":
        assert result[0].shape == (0,)
        assert result[0].dtype == args["output"].dtype
        assert actual is args["output"]
    else:
        assert result[0] is native_outputs[0]
    with FakeTensorMode(allow_non_fake_inputs=True):
        fake_result = trtllm_gen_custom_ops.fp4_block_scale_moe_runner(**args)
    assert fake_result[0].shape == result[0].shape
    assert fake_result[0].dtype == result[0].dtype
    assert fake_result[0].stride() == result[0].stride()


@pytest.mark.parametrize(
    "output_hidden_size,do_finalize,supplied",
    [
        (0, True, False),
        (-32, True, False),
        (48, True, False),
        (96, True, False),
        (32, False, False),
        (32, True, True),
    ],
)
def test_fp4_custom_op_rejects_invalid_direct_output_metadata(
    output_hidden_size: int, do_finalize: bool, supplied: bool
) -> None:
    args = _functional_fp4_moe_args()
    args["output_hidden_size"] = output_hidden_size
    args["do_finalize"] = do_finalize
    if supplied:
        args["output"] = torch.zeros((1, 32), dtype=torch.bfloat16)
    with patch.object(trtllm_gen_custom_ops.AutoTuner, "get") as tuner:
        with pytest.raises(ValueError, match="output_hidden_size"):
            trtllm_gen_custom_ops.fp4_block_scale_moe_runner(**args)
    tuner.assert_not_called()
    with FakeTensorMode(allow_non_fake_inputs=True):
        with pytest.raises(ValueError, match="output_hidden_size"):
            trtllm_gen_custom_ops.fp4_block_scale_moe_runner(**args)
