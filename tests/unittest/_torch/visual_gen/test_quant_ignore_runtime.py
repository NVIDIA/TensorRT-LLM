# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed Transformer routing, native path ownership, and compiled replay selection."""

from unittest.mock import patch

import pytest
import torch
from test_common.visual_gen_quantization import (
    ToyDiT,
    assert_quantization_routes,
    make_toy_quant_dispatcher,
    observe_quantization_paths,
)
from torch import nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.offloading import ModuleOffloadManager
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.quantization.transformer_quant_dispatcher import (
    bind_model_weights,
    fixed_quant_config,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, LinearQuantConfig, QuantAttentionConfig


class _TinyModel(ToyDiT):
    def __init__(self, config):
        super().__init__(block_count=1, model_config=config)
        self._has_stage2 = False

    def post_load_weights(self):
        pass

    def prepare_text_cache(self, **kwargs):
        return self.blocks[0].linear.weight.dtype


class _TinyPipeline(BasePipeline):
    def _init_transformer(self):
        self.transformer = _TinyModel(self.pipeline_config.model_configs["transformer"])

    def load_weights(self, weights):
        with torch.no_grad():
            for block in self.transformer.blocks:
                block.linear.weight.copy_(weights)


class _MultiTinyPipeline(_TinyPipeline):
    @property
    def transformer_components(self):
        return list(self.pipeline_config.model_configs)

    def _init_transformer(self):
        for name, config in self.pipeline_config.model_configs.items():
            setattr(self, name, _TinyModel(config))


def _config(linear=(0, 1), attention=(0, 2)):
    return DiffusionPipelineConfig(
        cuda_graph={"enable": False},
        model_configs={
            "transformer": DiffusionModelConfig(
                component_name="transformer",
                quant_config=LinearQuantConfig(quant_algo="FP8", ignore_steps=list(linear)),
                dynamic_weight_quant=True,
                attention=AttentionConfig(
                    backend="CUTEDSL",
                    quant_attention_config=QuantAttentionConfig(
                        qk_dtype="nvfp4", v_dtype="fp8", ignore_steps=list(attention)
                    ),
                ),
            )
        },
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "linear,attention,count", [((), (), 1), ((0,), (), 2), ((), (0,), 2), ((0, 1), (0, 2), 4)]
)
def test_fixed_transformer_paths_preserve_executed_precision(linear, attention, count):
    dispatcher = make_toy_quant_dispatcher(
        block_count=1, linear_quant_ignore_steps=linear, attention_quant_ignore_steps=attention
    )
    assert len(dispatcher.variants) == count
    with observe_quantization_paths(dispatcher) as trace:
        for step in range(4):
            trace.step = step
            dispatcher.select(step)(torch.randn(1, 8, 128, dtype=torch.bfloat16))
    assert_quantization_routes(
        trace,
        {
            (step, f"blocks.0.{name}", kind): "bf16" if step in ignored else "quantized"
            for step in range(4)
            for name, kind, ignored in (
                ("linear", "linear", linear),
                ("attention", "attention", attention),
            )
        },
    )
    for _, model in dispatcher.models():
        assert not getattr(model.model_config.quant_config, "ignore_steps", [])
        recipe = model.model_config.attention.quant_attention_config
        assert recipe is None or not recipe.ignore_steps
        assert all("_unquantized_weight" not in module._buffers for module in model.modules())


@pytest.mark.cpu_only
def test_fixed_config_preserves_resolved_activation_and_loading_settings():
    original = _config().model_configs["transformer"]
    original.force_dynamic_quantization = False
    original.quant_config.group_size = 128
    fixed = fixed_quant_config(original, (False, False))
    assert fixed.dynamic_weight_quant
    assert not fixed.force_dynamic_quantization
    assert fixed.quant_config.group_size == 128
    assert original.quant_config.ignore_steps == [0, 1]
    bf16 = fixed_quant_config(original, (True, False))
    assert bf16.quant_config.quant_algo is None
    assert not bf16.dynamic_weight_quant and not bf16.force_dynamic_quantization
    assert bf16.attention.quant_attention_config is not None
    assert fixed.attention_metadata_state is not bf16.attention_metadata_state


@pytest.mark.cpu_only
def test_pipeline_selects_real_models_and_resets_on_loop_exit():
    pipeline = _TinyPipeline(_config())
    dispatcher = pipeline._transformer_quant_dispatchers["transformer"]
    first = pipeline.transformer
    pipeline._is_warmup = True
    for request in range(2):
        for step, _ in pipeline._profile_denoise_steps(range(4)):
            assert pipeline.transformer is dispatcher.select(step)
            assert pipeline.transformer.model_config.attention_metadata_state["denoise_step"] == (
                step,
                4,
            )
        assert pipeline.transformer is first
    with pytest.raises(RuntimeError, match="stop"):
        for step, _ in pipeline._profile_denoise_steps(range(4)):
            raise RuntimeError("stop")
    assert pipeline.transformer is first


@pytest.mark.cpu_only
def test_multiple_components_keep_unquantized_component_single_instance():
    config = _config()
    config.model_configs["transformer_2"] = DiffusionModelConfig(component_name="transformer_2")
    pipeline = _MultiTinyPipeline(config)
    secondary = pipeline.transformer_2
    assert len(pipeline._transformer_quant_dispatchers["transformer"].variants) == 4
    assert len(pipeline._transformer_quant_dispatchers["transformer_2"].variants) == 1
    selected = set()
    for step in range(4):
        pipeline._publish_denoise_step((step, 4))
        selected.add(id(pipeline.transformer))
        assert pipeline.transformer_2 is secondary
        assert secondary.model_config.quant_config.quant_algo is None
        assert secondary.model_config.attention.quant_attention_config is None
    assert len(selected) == 4
    pipeline._publish_denoise_step(None)
    with patch("torch.compile", side_effect=lambda model, **kwargs: model) as compile_model:
        pipeline.run_transformer_quant_hook("torch_compile")
        assert compile_model.call_count == 5


@pytest.mark.cpu_only
def test_warmup_visits_every_path_even_when_first_steps_have_the_same_state():
    pipeline = _TinyPipeline(_config(linear=(10,), attention=(11,)))
    visited = set()

    def warmup(*args):
        for step, _ in pipeline._profile_denoise_steps(range(2)):
            visited.add(id(pipeline.transformer))

    pipeline._is_warmup = True
    with patch.object(pipeline, "_run_warmup", warmup), patch("torch.cuda.synchronize"):
        pipeline._run_warmup_pass([(8, 8, 1)], 2)
    dispatcher = pipeline._transformer_quant_dispatchers["transformer"]
    assert visited == {id(model) for _, model in dispatcher.models()}
    assert dispatcher.forced_key is None
    pipeline._is_warmup = False
    with pytest.raises(ValueError, match="out-of-range"):
        list(pipeline._profile_denoise_steps(range(2)))


@pytest.mark.cpu_only
@pytest.mark.parametrize("component", ["transformer", "transformer_2"])
@pytest.mark.parametrize("kind", ["linear", "attention"])
def test_quant_ignore_steps_reject_out_of_range_indices_and_reset(component, kind):
    config = _config(
        linear=(4,) if kind == "linear" else (), attention=(4,) if kind == "attention" else ()
    )
    if component == "transformer_2":
        config.model_configs[component] = config.model_configs["transformer"].model_copy(
            update={"component_name": component}
        )
        config.model_configs["transformer"] = DiffusionModelConfig(component_name="transformer")
    pipeline = _MultiTinyPipeline(config)
    dispatcher = pipeline._transformer_quant_dispatchers[component]
    pipeline._publish_denoise_step((4, 5))
    field = "quant_config" if kind == "linear" else "attention_config.quant_attention_config"
    with pytest.raises(ValueError, match="out-of-range") as error:
        next(pipeline._profile_denoise_steps(range(4)))
    assert f"{field}.ignore_steps" in str(error.value)
    assert component in str(error.value)
    assert "[4]" in str(error.value) and "4 steps" in str(error.value)
    assert dispatcher.step is None
    assert all(
        model.model_config.attention_metadata_state["denoise_step"] is None
        for _, model in dispatcher.models()
    )
    assert list(pipeline._profile_denoise_steps(range(5))) == list(enumerate(range(5)))


@pytest.mark.cpu_only
@pytest.mark.parametrize("as_iterator", [False, True])
def test_quant_ignore_steps_accept_last_step_for_repeated_requests(as_iterator):
    pipeline = _TinyPipeline(_config(linear=(3,), attention=(0, 3)))
    for _ in range(2):
        timesteps = iter(range(4)) if as_iterator else range(4)
        for step, _ in pipeline._profile_denoise_steps(timesteps):
            dispatcher = pipeline._transformer_quant_dispatchers["transformer"]
            assert pipeline.transformer is dispatcher.select(step)
            assert pipeline.transformer.model_config.attention_metadata_state["denoise_step"] == (
                step,
                4,
            )
        assert dispatcher.step is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("as_iterator", [False, True])
def test_multistage_exclusions_preserve_local_steps(as_iterator):
    linear, attention = (7, 9), (8, 10)
    pipeline = _TinyPipeline(_config(linear=linear, attention=attention))
    dispatcher = pipeline._transformer_quant_dispatchers["transformer"]
    original_schedule = dispatcher.schedule
    for key in dispatcher.schedule.keys:
        model = _TinyModel(
            fixed_quant_config(pipeline.pipeline_config.model_configs["transformer"], key)
        )
        dispatcher.add(key, model, stage="refinement")
    inputs = torch.randn(1, 8, 128, dtype=torch.bfloat16)
    with observe_quantization_paths(dispatcher) as trace:
        for _ in range(2):
            for stage, offset, count in (("base", 0, 8), ("refinement", 8, 3)):
                timesteps = iter(range(count)) if as_iterator else range(count)
                with pipeline.transformer_quant_stage(stage, ignore_steps_offset=offset):
                    for local_step, timestep in enumerate(timesteps):
                        pipeline._publish_denoise_step((local_step, count))
                        step = offset + local_step
                        assert timestep == local_step
                        assert dispatcher.step == local_step
                        assert pipeline.transformer.model_config.attention_metadata_state[
                            "denoise_step"
                        ] == (local_step, count)
                        trace.step = step
                        pipeline.transformer(inputs)
                    pipeline._publish_denoise_step(None)
                assert dispatcher.stage == "base" and dispatcher.step is None
                assert dispatcher.schedule is original_schedule
                assert all(
                    model.model_config.attention_metadata_state["denoise_step"] is None
                    for _, model in dispatcher.models()
                )
    assert_quantization_routes(
        trace,
        {
            (step, f"blocks.0.{name}", kind): "bf16" if step in ignored else "quantized"
            for step in range(11)
            for name, kind, ignored in (
                ("linear", "linear", linear),
                ("attention", "attention", attention),
            )
        },
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["linear", "attention"])
def test_refinement_exclusions_validate_local_range_and_restore(kind):
    pipeline = _TinyPipeline(
        _config(
            linear=(11,) if kind == "linear" else (), attention=(11,) if kind == "attention" else ()
        )
    )
    dispatcher = pipeline._transformer_quant_dispatchers["transformer"]
    original_schedule = dispatcher.schedule
    for key in dispatcher.schedule.keys:
        dispatcher.add(
            key,
            _TinyModel(
                fixed_quant_config(pipeline.pipeline_config.model_configs["transformer"], key)
            ),
            stage="refinement",
        )
    with (
        pipeline.transformer_quant_stage("refinement", ignore_steps_offset=8),
        pytest.raises(ValueError, match="3 steps"),
    ):
        next(pipeline._profile_denoise_steps(range(3)))
    assert dispatcher.stage == "base" and dispatcher.step is None
    assert dispatcher.schedule is original_schedule


@pytest.mark.cpu_only
def test_variant_request_inputs_follow_the_selected_model():
    pipeline = _TinyPipeline(_config())
    caches = pipeline.prepare_transformer_inputs("prepare_text_cache")
    for step in range(4):
        pipeline._publish_denoise_step((step, 4))
        assert (
            pipeline.resolve_transformer_inputs(caches)
            == pipeline.transformer.blocks[0].linear.weight.dtype
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("share_parameters", [False, True])
def test_completed_linear_storage_is_shared_and_survives_offload(share_parameters):
    config = DiffusionModelConfig(component_name="transformer")

    def model():
        result = nn.Module()
        result.model_config = config
        result.proj = Linear(8, 8, dtype=torch.bfloat16, bias=False, quant_config=QuantConfig())
        return result

    source, target = model(), model()
    source.proj.weight.data.normal_()
    source.proj.post_load_weights()
    source.proj.rebuild_tensor_metadata = {
        "weight": {"meta": source.proj.weight.to("meta"), "param": source.proj.weight}
    }
    expected = source.proj.weight.detach().clone()
    bind_model_weights(source, target, share_parameters=share_parameters)
    assert (source.proj.weight is target.proj.weight) == share_parameters
    assert target.proj.rebuild_tensor_metadata["weight"]["param"] is target.proj.weight
    assert (
        target.proj.rebuild_tensor_metadata["weight"]
        is not source.proj.rebuild_tensor_metadata["weight"]
    )
    assert source.proj is not target.proj
    assert source.proj.quant_method is not target.proj.quant_method
    assert (
        source.proj.weight.untyped_storage().data_ptr()
        == target.proj.weight.untyped_storage().data_ptr()
    )
    manager = ModuleOffloadManager(
        {"transformer": nn.ModuleList([source, target])}, device="cpu", pin_memory=False
    )
    manager.initialize()
    for _ in range(2):
        manager.stage("transformer")
        assert (
            source.proj.weight.untyped_storage().data_ptr()
            == target.proj.weight.untyped_storage().data_ptr()
        )
        assert torch.equal(source.proj.weight, expected)
        for model in (source, target):
            assert model.proj.rebuild_tensor_metadata["weight"]["param"] is model.proj.weight
        manager._rebind_to_cpu("transformer")
        manager.active_stage_name = None
    if not share_parameters:
        target.proj.weight.data = torch.empty_like(target.proj.weight)
        assert source.proj.weight.data_ptr() != target.proj.weight.data_ptr()
        assert torch.equal(source.proj.weight, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("algorithm", ["FP8_BLOCK_SCALES", "NVFP4"])
def test_final_linear_layout_and_scales_are_shared_without_requantization(algorithm):
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.visual_gen.quantization import loader as weight_loader
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        pytest.skip("Representative quantized layouts require Blackwell")

    class Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.model_config = config
            self.mlp = GatedMLP(
                hidden_size=128,
                intermediate_size=128,
                bias=False,
                dtype=torch.bfloat16,
                config=config,
                use_cute_dsl_blockscaling_mm=algorithm == "NVFP4",
                fuse_bf16_gate_up_swiglu=True,
            )
            self.mlp.down_proj.all_reduce.register_buffer("routing_workspace", torch.empty(1))

        def post_load_weights(self):
            for layer in (self.mlp.gate_up_proj, self.mlp.down_proj):
                layer.post_load_weights()

    class Pipeline(_TinyPipeline):
        def _init_transformer(self):
            self.transformer = Model(self.pipeline_config.model_configs["transformer"])

        def load_weights(self, weights):
            loader = weight_loader.DynamicLinearWeightLoader(self.transformer.model_config)
            # Short checkpoint aliases deliberately differ from registered module paths.
            loader.load_linear_weights(self.transformer.mlp.gate_up_proj, "gate_up", weights[:2])
            loader.load_linear_weights(self.transformer.mlp.down_proj, "down", weights[2:])

    config = _config()
    model_config = config.model_configs["transformer"]
    (
        model_config.quant_config,
        model_config.quant_config_dict,
        model_config.dynamic_weight_quant,
        model_config.force_dynamic_quantization,
    ) = DiffusionPipelineConfig._resolve_quant_config(
        LinearQuantConfig(quant_algo=algorithm, ignore_steps=[0, 1]), None
    )
    with torch.device("cuda"):
        pipeline = Pipeline(config)
        weights = [{"weight": torch.randn(128, 128, dtype=torch.bfloat16)} for _ in range(3)]
    deferred = pipeline.deferred_transformer_quant_tensor_ids()
    for _, model in pipeline.transformer_quant_models():
        assert id(model.mlp.down_proj.all_reduce.routing_workspace) not in deferred
    quantizer = "quantize_nvfp4" if algorithm == "NVFP4" else "quantize_fp8_blockwise"
    with patch.object(
        weight_loader, quantizer, wraps=getattr(weight_loader, quantizer)
    ) as quantize:
        pipeline.load_transformer_quant_weights(weights)
        assert quantize.call_count == (2 if algorithm == "NVFP4" else 3)
    models = list(pipeline.transformer_quant_models())
    assert len(models) == 4
    source, target = models[0][1], models[1][1]
    assert source.mlp.down_proj.all_reduce is not target.mlp.down_proj.all_reduce
    assert (
        source.mlp.down_proj.all_reduce.routing_workspace
        is not target.mlp.down_proj.all_reduce.routing_workspace
    )
    for name in ("gate_up_proj", "down_proj"):
        left, right = getattr(source.mlp, name), getattr(target.mlp, name)
        assert left is not right and left.quant_method is not right.quant_method
        assert left._weights_transformed and right._weights_transformed
        for tensor_name, tensor in left.named_parameters():
            shared = dict(right.named_parameters())[tensor_name]
            assert tensor.untyped_storage().data_ptr() == shared.untyped_storage().data_ptr()
            assert tensor.dtype == shared.dtype and tensor.stride() == shared.stride()
        for tensor_name, metadata in right.rebuild_tensor_metadata.items():
            assert metadata["param"] is getattr(right, tensor_name)
    assert source.mlp._can_fuse_gate_up_swiglu() == target.mlp._can_fuse_gate_up_swiglu()
    assert source.mlp._can_fuse_gate_up_swiglu() == (algorithm == "NVFP4")
    assert source.mlp._can_fuse_swiglu_fp8_quant() == (algorithm == "FP8_BLOCK_SCALES")
    inputs = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
    dispatcher = pipeline._transformer_quant_dispatchers["transformer"]
    with torch.inference_mode(), observe_quantization_paths(dispatcher) as trace:
        for step in range(4):
            trace.step = step
            pipeline._publish_denoise_step((step, 4))
            pipeline.transformer.mlp(inputs)
    assert_quantization_routes(
        trace,
        {
            (step, f"mlp.{projection}", "linear"): "bf16" if step in (0, 1) else "quantized"
            for step in range(4)
            for projection in ("gate_up_proj", "down_proj")
        },
    )
    pipeline._publish_denoise_step(None)
    expected = source.mlp.gate_up_proj.weight.detach().view(torch.uint8).cpu().clone()
    manager = ModuleOffloadManager(
        {"transformer": nn.ModuleList([source, target])}, device="cuda", pin_memory=False
    )
    manager.initialize()
    for _ in range(2):
        manager.stage("transformer")
        left, right = source.mlp.gate_up_proj.weight, target.mlp.gate_up_proj.weight
        assert left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        assert torch.equal(left.detach().view(torch.uint8).cpu(), expected)
        for model in (source, target):
            for projection in (model.mlp.gate_up_proj, model.mlp.down_proj):
                for name, metadata in projection.rebuild_tensor_metadata.items():
                    assert metadata["param"] is getattr(projection, name)
        manager._rebind_to_cpu("transformer")
        manager.active_stage_name = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("execution_mode", ["compile", "cuda_graph"])
def test_execution_mode_preserves_quantization_routes(execution_mode):
    from torch._dynamo.testing import CompileCounterWithBackend

    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() not in (100, 103):
        pytest.skip("Representative NVFP4 kernels require SM100/SM103")
    with torch.device("cuda"):
        dispatcher = make_toy_quant_dispatcher(
            block_count=1,
            linear_quant_ignore_steps=(0, 1),
            attention_quant_ignore_steps=(0, 2),
            real_kernels=True,
        )
    inputs = torch.randn(1, 128, 128, dtype=torch.bfloat16, device="cuda")
    runners = []
    try:
        with (
            torch.inference_mode(),
            observe_quantization_paths(dispatcher, device_only=True) as trace,
        ):
            counter = CompileCounterWithBackend("inductor")
            execute = {}
            for _, model in dispatcher.models():
                if execution_mode == "compile":
                    execute[id(model)] = torch.compile(model, backend=counter)
                else:
                    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
                    runners.append(runner)
                    execute[id(model)] = runner.wrap(model)
            for _ in range(4):
                for step in range(4):
                    model = dispatcher.select(step)
                    execute[id(model)](inputs)
            for step in (0, 1, 2, 3, 0):
                model = dispatcher.select(step)
                trace.reset()
                execute[id(model)](inputs)
                counts = trace.device_counts.cpu().tolist()
                for offset, ignored in ((0, step in (0, 1)), (2, step in (0, 2))):
                    assert counts[offset + int(not ignored)] > 0, counts
                    assert counts[offset + int(ignored)] == 0, counts
            if execution_mode == "compile":
                assert counter.frame_count > 0
            else:
                assert all(runner.graphs for runner in runners)
    finally:
        for runner in runners:
            runner.clear()
        torch._dynamo.reset()
