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
"""Tests for visual generation module parameter offloading."""

from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.config import CpuOffloadConfig, VisualGenArgs
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
    COSMOS3_GENERATOR_OFFLOAD_COMPONENT,
    COSMOS3_REASONER_OFFLOAD_COMPONENT,
    COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT,
    COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT,
    Cosmos3OmniMoTPipeline,
)
from tensorrt_llm._torch.visual_gen.offloading import ModuleOffloadManager, OffloadPipeline
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


class _ToyModule(nn.Module):
    def __init__(self, weight_value: float, bias_value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2, 2), weight_value))
        self.register_buffer("bias", torch.full((2,), bias_value))


class _AlignmentToyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Parameter(torch.ones((1,), dtype=torch.bfloat16))
        self.alignment_sensitive = nn.Parameter(torch.ones((3,), dtype=torch.bfloat16))


class _ToyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = _ToyModule(weight_value=3.0, bias_value=30.0)

    @property
    def device(self):
        return next(self.parameters()).device


class _OffloadCudaGraphPipeline(BasePipeline):
    def _init_transformer(self) -> None:
        self.transformer = _ToyModule(weight_value=1.0, bias_value=10.0)

    def default_offload_stages(self) -> tuple[tuple[str, ...], ...]:
        return (("denoising_transformer",),)


class _CustomOffloadPipeline(BasePipeline):
    def _init_transformer(self) -> None:
        self.transformer = _ToyTransformer()
        self.text_encoder = _ToyModule(weight_value=1.0, bias_value=10.0)
        self.vae = _ToyModule(weight_value=2.0, bias_value=20.0)


def _make_config(
    *,
    offload_enable: bool = True,
    offload_stages: list[str | list[str]] | None = None,
    device: str = "cuda",
):
    return SimpleNamespace(
        primary_pretrained_config=SimpleNamespace(),
        device=device,
        cuda_graph=SimpleNamespace(enable=False),
        torch_compile=SimpleNamespace(enable=False),
        cpu_offload_config=CpuOffloadConfig(
            enable=offload_enable,
            stages=offload_stages,
        ),
    )


def test_pipeline_device_uses_config_not_transformer_parameter_device():
    pipeline = _CustomOffloadPipeline(_make_config(device="cuda:3"))

    assert pipeline.transformer.device == torch.device("cpu")
    assert pipeline.device == torch.device("cuda:3")


def _make_manager() -> tuple[ModuleOffloadManager, _ToyModule, _ToyModule]:
    stage_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    stage_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    manager = ModuleOffloadManager(
        stages={
            "stage_a": stage_a,
            "stage_b": stage_b,
        },
        device="cpu",
        pin_memory=False,
    )
    manager.initialize()
    return manager, stage_a, stage_b


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_cuda_graphs_with_offload_raise_not_implemented():
    config = _make_config()
    config.cuda_graph.enable = True

    with pytest.raises(
        NotImplementedError,
        match="CUDA graphs are not supported with visual generation offloading",
    ):
        _OffloadCudaGraphPipeline(config)


def test_offload_context_requires_initialized_pipeline_when_configured():
    pipeline = _OffloadCudaGraphPipeline(_make_config())

    with pipeline.offloader.context("transformer", enable=False):
        pass

    with pytest.raises(
        RuntimeError,
        match="offload pipeline has not been initialized",
    ):
        with pipeline.offloader.context("transformer"):
            pass


def test_configured_offload_stages_override_model_defaults_and_expose_vae():
    pipeline = _CustomOffloadPipeline(
        _make_config(
            offload_stages=[
                "text_encoder",
                ["denoising_transformer", "vae"],
                "denoising_transformer_2",
            ]
        )
    )

    assert pipeline.offloader.stages() == (
        ("text_encoder",),
        ("denoising_transformer", "vae"),
        ("denoising_transformer_2",),
    )

    components = pipeline.offload_pipeline_components()
    assert components["text_encoder"] is pipeline.text_encoder
    assert components["denoising_transformer"] is pipeline.transformer.blocks
    assert components["vae"] is pipeline.vae

    assert pipeline.offloader.filter_available_stages(pipeline.offloader.stages(), components) == (
        ("text_encoder",),
        ("denoising_transformer", "vae"),
    )


def test_offload_context_resolves_component_to_stageed_stage():
    pipeline = _CustomOffloadPipeline(
        _make_config(offload_stages=[["denoising_transformer", "vae"]])
    )
    pipeline.initialize_offload_pipeline()

    with pipeline.offloader.context("vae"):
        assert pipeline.offloader.offload_pipeline is not None
        assert (
            pipeline.offloader.offload_pipeline.manager.active_stage_name
            == "denoising_transformer+vae"
        )

    pipeline.cleanup()


def test_initialize_rejects_configured_stage_components_that_are_unavailable_for_model():
    pipeline = _CustomOffloadPipeline(
        _make_config(offload_stages=[["denoising_transformer", "denoising_transformer_2"]])
    )

    with pytest.raises(
        ValueError,
        match=r"Unknown cpu_offload_config\.stages entries.*denoising_transformer_2.*denoising_transformer",
    ):
        pipeline.initialize_offload_pipeline()


def test_visual_gen_args_loads_yaml_offload_stages(tmp_path):
    config_path = tmp_path / "visual_gen.yaml"
    config_path.write_text(
        """
cpu_offload_config:
  enable: true
  stages:
    - text_encoder
    - [denoising_transformer, vae]
""",
        encoding="utf-8",
    )

    args = VisualGenArgs.from_yaml(config_path)

    assert args.cpu_offload_config.enable is True
    assert args.cpu_offload_config.stages == [
        "text_encoder",
        ["denoising_transformer", "vae"],
    ]


def test_visual_gen_args_allows_model_specific_offload_stage_names():
    args = VisualGenArgs(
        cpu_offload_config={
            "enable": True,
            "stages": ["future_model_stage"],
        }
    )

    assert args.cpu_offload_config.stages == ["future_model_stage"]


def test_pipeline_rejects_unknown_offload_stage_names_for_this_model():
    pipeline = _CustomOffloadPipeline(_make_config(offload_stages=["transformer.blocks"]))

    with pytest.raises(
        ValueError,
        match=r"Unknown cpu_offload_config\.stages entries.*transformer\.blocks.*denoising_transformer",
    ):
        pipeline.initialize_offload_pipeline()


def test_initialize_reports_cpu_storage_allocation_context():
    stage = _ToyModule(weight_value=1.0, bias_value=10.0)
    manager = ModuleOffloadManager(
        stages={"stage": stage},
        device="cpu",
        pin_memory=False,
    )
    gpu_arena = torch.empty(1024, dtype=torch.uint8, device="cpu")

    def fail_cpu_storage(*args, **kwargs):
        raise RuntimeError("cpu allocation failed")

    with (
        patch.object(manager, "_allocate_gpu_arena", return_value=gpu_arena),
        patch("torch.empty", side_effect=fail_cpu_storage),
    ):
        with pytest.raises(
            RuntimeError,
            match="Failed to allocate packed CPU storage for visual generation offload",
        ) as exc_info:
            manager.initialize()

    message = str(exc_info.value)
    assert "Failed to allocate packed CPU storage for visual generation offload" in message
    assert "pin_memory=False" in message
    assert "stages=[stage=" in message


def test_initialize_reports_cuda_arena_allocation_hint():
    stage = _ToyModule(weight_value=1.0, bias_value=10.0)
    manager = ModuleOffloadManager(
        stages={"stage": stage},
        device="cuda",
        pin_memory=False,
    )
    original_empty = torch.empty

    def fail_cuda_arena(*args, **kwargs):
        device = torch.device(kwargs.get("device", "cpu"))
        if device.type == "cuda":
            raise RuntimeError("cuda allocation failed")
        return original_empty(*args, **kwargs)

    with patch("torch.empty", side_effect=fail_cuda_arena):
        with pytest.raises(
            RuntimeError,
            match="Failed to allocate GPU arena for visual generation offload",
        ) as exc_info:
            manager.initialize()

    message = str(exc_info.value)
    assert "Failed to allocate GPU arena for visual generation offload" in message
    assert "device=cuda" in message
    assert "stages=[stage=" in message
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in message


def test_initialize_packs_stage_before_rebinding_to_cpu():
    stage_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    stage_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    manager = ModuleOffloadManager(
        stages={
            "stage_a": stage_a,
            "stage_b": stage_b,
        },
        device="cpu",
        pin_memory=False,
    )
    events = []
    original_rebind_to_cpu = manager._rebind_to_cpu
    original_allocate_cpu_storage = manager._allocate_cpu_storage

    def record_rebind_to_cpu(name):
        events.append(("rebind_cpu", name))
        return original_rebind_to_cpu(name)

    def record_allocate_cpu_storage(num_bytes, stage_name=None):
        events.append(("allocate_cpu", stage_name))
        return original_allocate_cpu_storage(num_bytes, stage_name=stage_name)

    with (
        patch.object(manager, "_rebind_to_cpu", side_effect=record_rebind_to_cpu),
        patch.object(
            manager,
            "_allocate_cpu_storage",
            side_effect=record_allocate_cpu_storage,
        ),
    ):
        manager.initialize()

    assert events.index(("allocate_cpu", "stage_a")) < events.index(("rebind_cpu", "stage_a"))
    assert events.index(("allocate_cpu", "stage_b")) < events.index(("rebind_cpu", "stage_b"))


def test_packed_tensor_views_are_sufficiently_aligned():
    stage = _AlignmentToyModule()
    manager = ModuleOffloadManager(
        stages={"stage": stage},
        device="cpu",
        pin_memory=False,
    )
    manager.initialize()

    layout = manager.layouts["stage"]
    for spec in layout.specs:
        assert spec.offset % 16 == 0

    manager.stage("stage")
    assert stage.prefix.data_ptr() % 16 == 0
    assert stage.alignment_sensitive.data_ptr() % 16 == 0


def test_offload_pipeline_context_stages_requested_stage():
    stage_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    stage_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    pipeline = OffloadPipeline(
        stages=(("stage_a",), ("stage_b",)),
        components={
            "stage_a": stage_a,
            "stage_b": stage_b,
        },
        device="cpu",
        pin_memory=False,
    )
    pipeline.initialize()

    assert pipeline.manager.active_stage_name is None
    with pipeline.context("stage_a"):
        assert pipeline.manager.active_stage_name == "stage_a"
    assert pipeline.manager.active_stage_name == "stage_a"

    with pipeline.context("stage_b"):
        assert pipeline.manager.active_stage_name == "stage_b"


def test_inactive_stage_stays_cpu_backed_after_staging_another_stage():
    manager, stage_a, stage_b = _make_manager()
    assert manager.gpu_arena is not None
    stage_a_cpu_storage = manager.layouts["stage_a"].cpu_storage
    stage_b_cpu_storage = manager.layouts["stage_b"].cpu_storage
    assert stage_a_cpu_storage is not None
    assert stage_b_cpu_storage is not None
    stage_a_cpu_storage_ptr = _storage_ptr(stage_a_cpu_storage)
    stage_b_cpu_storage_ptr = _storage_ptr(stage_b_cpu_storage)
    gpu_arena_ptr = _storage_ptr(manager.gpu_arena)

    manager.stage("stage_a")

    assert manager.active_stage_name == "stage_a"
    assert _storage_ptr(stage_a.weight) == gpu_arena_ptr
    assert _storage_ptr(stage_a.bias) == gpu_arena_ptr
    assert _storage_ptr(stage_b.weight) == stage_b_cpu_storage_ptr
    assert _storage_ptr(stage_b.bias) == stage_b_cpu_storage_ptr
    torch.testing.assert_close(stage_b.weight, torch.full((2, 2), 2.0))
    torch.testing.assert_close(stage_b.bias, torch.full((2,), 20.0))

    manager.stage("stage_b")

    assert manager.active_stage_name == "stage_b"
    assert _storage_ptr(stage_a.weight) == stage_a_cpu_storage_ptr
    assert _storage_ptr(stage_a.bias) == stage_a_cpu_storage_ptr
    assert _storage_ptr(stage_b.weight) == gpu_arena_ptr
    assert _storage_ptr(stage_b.bias) == gpu_arena_ptr
    torch.testing.assert_close(stage_a.weight, torch.full((2, 2), 1.0))
    torch.testing.assert_close(stage_a.bias, torch.full((2,), 10.0))


def test_state_dict_reads_correct_inactive_stage_data():
    manager, stage_a, stage_b = _make_manager()

    manager.stage("stage_a")
    stage_b_state = stage_b.state_dict()
    torch.testing.assert_close(stage_b_state["weight"], torch.full((2, 2), 2.0))
    torch.testing.assert_close(stage_b_state["bias"], torch.full((2,), 20.0))

    manager.stage("stage_b")
    stage_a_state = stage_a.state_dict()
    torch.testing.assert_close(stage_a_state["weight"], torch.full((2, 2), 1.0))
    torch.testing.assert_close(stage_a_state["bias"], torch.full((2,), 10.0))


def test_rebinding_reuses_cached_view_objects():
    manager, stage_a, _ = _make_manager()
    cpu_weight = stage_a.weight
    cpu_bias = stage_a.bias

    manager.stage("stage_a")
    gpu_weight = stage_a.weight
    gpu_bias = stage_a.bias

    assert gpu_weight is not cpu_weight
    assert gpu_bias is not cpu_bias

    manager.stage("stage_b")
    assert stage_a.weight is cpu_weight
    assert stage_a.bias is cpu_bias

    manager.stage("stage_a")
    assert stage_a.weight is gpu_weight
    assert stage_a.bias is gpu_bias


# ============================================================================
# Cosmos3-specific offload wiring (no GPU / no checkpoint)
# ============================================================================


class _FakeRunner:
    """Stand-in for cosmos_guardrail.GuardrailRunner (a plain object)."""

    def __init__(self, models: list) -> None:
        self._models = models

    @property
    def models(self) -> list:
        return self._models


class _FakeSafetyChecker:
    def __init__(self, text_models: list, video_models: list) -> None:
        self.text_guardrail = _FakeRunner(text_models)
        self.video_guardrail = _FakeRunner(video_models)


class _FakeCosmos3Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Linear(2, 2)])
        self.gen_layers = nn.ModuleList([nn.Linear(2, 2)])


class _Cosmos3OffloadUnitPipeline(Cosmos3OmniMoTPipeline):
    """Cosmos3 pipeline with a fake transformer to avoid loading the model."""

    def _init_transformer(self) -> None:
        self.transformer = _FakeCosmos3Transformer()


def _make_cosmos3_unit_config(
    *,
    offload_enable: bool = True,
    offload_stages: Optional[list] = None,
    device: str = "cpu",
):
    return SimpleNamespace(
        primary_pretrained_config=SimpleNamespace(),
        device=device,
        cuda_graph=SimpleNamespace(enable=False),
        torch_compile=SimpleNamespace(enable=False),
        cpu_offload_config=CpuOffloadConfig(enable=offload_enable, stages=offload_stages),
    )


def _make_cosmos3_unit_pipeline(**config_kwargs) -> _Cosmos3OffloadUnitPipeline:
    return _Cosmos3OffloadUnitPipeline(_make_cosmos3_unit_config(**config_kwargs))


def test_cosmos3_offload_components_expose_towers_vae_and_guardrails():
    pipeline = _make_cosmos3_unit_pipeline()
    pipeline.vae = nn.Linear(2, 2)
    text_model = nn.Linear(2, 2)
    video_model = nn.Linear(2, 2)
    pipeline.safety_checker = _FakeSafetyChecker(
        text_models=[text_model], video_models=[video_model]
    )

    components = pipeline.offload_pipeline_components()

    assert (
        components[COSMOS3_REASONER_OFFLOAD_COMPONENT] is pipeline.transformer.language_model.layers
    )
    assert components[COSMOS3_GENERATOR_OFFLOAD_COMPONENT] is pipeline.transformer.gen_layers
    assert components["vae"] is pipeline.vae

    text_component = components[COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT]
    video_component = components[COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT]
    assert isinstance(text_component, nn.ModuleList) and list(text_component) == [text_model]
    assert isinstance(video_component, nn.ModuleList) and list(video_component) == [video_model]


def test_cosmos3_guardrail_components_absent_without_safety_checker():
    pipeline = _make_cosmos3_unit_pipeline()
    # safety_checker is loaded on rank 0 only; emulate ranks where it is missing.
    components = pipeline.offload_pipeline_components()

    assert COSMOS3_REASONER_OFFLOAD_COMPONENT in components
    assert COSMOS3_GENERATOR_OFFLOAD_COMPONENT in components
    assert COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT not in components
    assert COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT not in components


def test_cosmos3_guardrail_components_wrap_only_nn_modules():
    pipeline = _make_cosmos3_unit_pipeline()
    nn_model = nn.Linear(2, 2)
    # Blocklist is a plain object (not nn.Module) and must be skipped.
    pipeline.safety_checker = _FakeSafetyChecker(
        text_models=[object(), nn_model], video_models=[object()]
    )

    components = pipeline.offload_pipeline_components()

    assert list(components[COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT]) == [nn_model]
    # video guardrail has no nn.Module members -> no component registered.
    assert COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT not in components


def test_cosmos3_default_offload_stages_exclude_guardrails():
    pipeline = _make_cosmos3_unit_pipeline()

    assert pipeline.default_offload_stages() == (
        (COSMOS3_REASONER_OFFLOAD_COMPONENT,),
        (COSMOS3_GENERATOR_OFFLOAD_COMPONENT,),
    )


def test_cosmos3_validate_accepts_guardrail_names_even_when_not_loaded():
    # Emulates a non-rank-0 process: guardrail components are configured but absent
    # from offload_pipeline_components. Validation must not raise on the guardrails.
    pipeline = _make_cosmos3_unit_pipeline(
        offload_stages=[
            COSMOS3_REASONER_OFFLOAD_COMPONENT,
            COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT,
            COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT,
        ]
    )
    available_components = {
        COSMOS3_REASONER_OFFLOAD_COMPONENT: pipeline.transformer.language_model.layers
    }

    # Guardrail names are tolerated even though they are not in available_components.
    pipeline.offloader.validate_configured_stages(pipeline.offloader.stages(), available_components)


def test_cosmos3_validate_still_rejects_unknown_component_names():
    pipeline = _make_cosmos3_unit_pipeline(offload_stages=["not_a_real_component"])
    available_components = {
        COSMOS3_REASONER_OFFLOAD_COMPONENT: pipeline.transformer.language_model.layers
    }

    with pytest.raises(
        ValueError, match=r"Unknown cpu_offload_config\.stages entries.*not_a_real_component"
    ):
        pipeline.offloader.validate_configured_stages(
            pipeline.offloader.stages(), available_components
        )
