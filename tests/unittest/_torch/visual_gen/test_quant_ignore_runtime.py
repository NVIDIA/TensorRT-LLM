# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step publication and actual execution routes through compile and CUDA Graph replay."""

from types import SimpleNamespace

import pytest
import torch
from test_common.visual_gen_quantization import (
    ToyDiT,
    assert_quantization_routes,
    observe_quantization_paths,
)

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.quantization.quant_dispatcher import LinearQuantDispatcher


def _pipeline(model: ToyDiT, second: ToyDiT | None = None) -> SimpleNamespace:
    pipeline = SimpleNamespace(
        _is_warmup=True,
        transformer=model,
        transformer_components=["transformer"],
        pipeline_config=DiffusionPipelineConfig(),
    )
    if second is not None:
        pipeline.transformer_2 = second
        pipeline.transformer_components.append("transformer_2")
    pipeline._publish_denoise_step = lambda value: BasePipeline._publish_denoise_step(
        pipeline, value
    )
    return pipeline


@pytest.mark.cpu_only
def test_pipeline_installs_quant_dispatcher() -> None:
    model = ToyDiT(
        linear_quant_ignored_steps=(0,), quantized_attention=False, install_dispatch=False
    )
    pipeline = _pipeline(model)
    BasePipeline._setup_linear_quant_dispatcher(pipeline)
    assert all(
        isinstance(block.linear.quant_method, LinearQuantDispatcher) for block in model.blocks
    )
    with observe_quantization_paths(model) as trace:
        trace.step = 0
        pipeline._publish_denoise_step((0, 1))
        model(torch.randn(1, 8, 128, dtype=torch.bfloat16))
    assert_quantization_routes(
        trace,
        {
            (0, f"blocks.{block}.{name}", kind): "bf16"
            for block in range(2)
            for name, kind in (("linear", "linear"), ("attention", "attention"))
        },
    )


@pytest.mark.cpu_only
def test_linear_and_attention_steps_are_independent() -> None:
    model = ToyDiT(linear_quant_ignored_steps=(0, 1), attention_quant_ignored_steps=(0, 2))
    second = ToyDiT(linear_quant_ignored_steps=(0, 1), attention_quant_ignored_steps=(0, 2))
    pipeline = _pipeline(model, second)
    keys = {}
    runner = SimpleNamespace(register_extra_key_fn=lambda name, fn: keys.update({name: fn}))
    BaseDiffusionModel.register_cuda_graph_extra_key_fns(model, runner)
    phases = ((True, True), (True, False), (False, True), (False, False))
    with observe_quantization_paths(model) as trace:
        for step, phase in enumerate(phases):
            trace.step = step
            pipeline._publish_denoise_step((step, len(phases)))
            assert keys["quant_ignored"]() == phase
            assert (
                second.model_config.attention_metadata_state
                == model.model_config.attention_metadata_state
            )
            assert (
                second.model_config.attention_metadata_state
                is not model.model_config.attention_metadata_state
            )
            model(torch.randn(1, 8, 128, dtype=torch.bfloat16))
    assert_quantization_routes(
        trace,
        {
            (step, f"blocks.{block}.{name}", kind): "bf16" if phase[index] else "quantized"
            for step, phase in enumerate(phases)
            for block in range(2)
            for index, (name, kind) in enumerate((("linear", "linear"), ("attention", "attention")))
        },
    )


@pytest.mark.cpu_only
def test_cfg_branches_share_quant_ignore_state() -> None:
    model = ToyDiT(linear_quant_ignored_steps=(1,), attention_quant_ignored_steps=(0,))
    pipeline = _pipeline(model)
    inputs = torch.randn(1, 8, 128, dtype=torch.bfloat16)
    with observe_quantization_paths(model) as trace:
        for step, _ in BasePipeline._profile_denoise_steps(pipeline, range(2)):
            trace.step = step
            model(inputs)
            model(inputs)  # Conditional and unconditional calls must enter the same methods.
    expected = {
        (step, f"blocks.{block}.{name}", kind): "bf16" if step == ignored_step else "quantized"
        for step in range(2)
        for block in range(2)
        for name, kind, ignored_step in (("linear", "linear", 1), ("attention", "attention", 0))
    }
    assert_quantization_routes(trace, expected)
    assert len(trace.events) == 2 * len(expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("exit_mode", ["normal", "exception"])
def test_quant_ignore_state_is_cleared(exit_mode: str) -> None:
    model = ToyDiT(linear_quant_ignored_steps=(0,), attention_quant_ignored_steps=(0,))
    pipeline = _pipeline(model)
    iterator = BasePipeline._profile_denoise_steps(pipeline, range(2))
    next(iterator)
    state = model.model_config.attention_metadata_state
    assert state["linear_quant_ignored"] and state["attention_quant_ignored"]
    if exit_mode == "exception":
        with pytest.raises(RuntimeError, match="inference failed"):
            iterator.throw(RuntimeError("inference failed"))
    else:
        list(iterator)
    assert state["denoise_step"] is None
    assert not state["linear_quant_ignored"] and not state["attention_quant_ignored"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("execution_mode", ["compile", "cuda_graph"])
def test_execution_mode_preserves_quantization_routes(execution_mode: str) -> None:
    from torch._dynamo.testing import CompileCounterWithBackend

    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        pytest.skip("The representative NVFP4 kernels require SM100/SM103")
    torch.manual_seed(42)
    with torch.device("cuda"):
        model = ToyDiT(
            block_count=1,
            linear_quant_ignored_steps=(0, 1),
            attention_quant_ignored_steps=(0, 2),
            real_kernels=True,
        )
    inputs = torch.randn(1, 128, 128, dtype=torch.bfloat16, device="cuda")
    state = model.model_config.attention_metadata_state
    phases = ((True, True), (True, False), (False, True), (False, False))
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    BaseDiffusionModel.register_cuda_graph_extra_key_fns(model, runner)
    try:
        with torch.inference_mode(), observe_quantization_paths(model) as eager_trace:
            for step, (linear, attention) in enumerate(phases):
                eager_trace.step = step
                state["linear_quant_ignored"] = linear
                state["attention_quant_ignored"] = attention
                model(inputs)
        assert_quantization_routes(
            eager_trace,
            {
                (step, f"blocks.0.{name}", kind): "bf16" if phase[index] else "quantized"
                for step, phase in enumerate(phases)
                for index, (name, kind) in enumerate(
                    (("linear", "linear"), ("attention", "attention"))
                )
            },
        )
        with torch.inference_mode(), observe_quantization_paths(model, device_only=True) as trace:
            compile_counter = CompileCounterWithBackend("inductor")
            execute = (
                torch.compile(model, backend=compile_counter)
                if execution_mode == "compile"
                else runner.wrap(model)
            )
            # Compile/capture every phase before resetting counters. Subsequent calls exercise replay.
            for linear, attention in phases:
                state["linear_quant_ignored"] = linear
                state["attention_quant_ignored"] = attention
                execute(inputs)
            for linear, attention in (*phases, phases[0]):
                state["linear_quant_ignored"] = linear
                state["attention_quant_ignored"] = attention
                trace.reset()
                execute(inputs)
                counts = trace.device_counts.cpu().tolist()
                for offset, quant_ignored in ((0, linear), (2, attention)):
                    assert counts[offset + int(not quant_ignored)] > 0, counts
                    assert counts[offset + int(quant_ignored)] == 0, counts
            if execution_mode == "cuda_graph":
                assert len(runner.graphs) == 4
            else:
                assert compile_counter.frame_count > 0, "The forward path was not compiled"
    finally:
        runner.clear()
        torch._dynamo.reset()
