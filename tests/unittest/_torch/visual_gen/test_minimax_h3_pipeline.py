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

"""Synthetic pipeline-level tests for MiniMax-H3."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen.config import discover_pipeline_components
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import pipeline_minimax_h3 as h3_pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import (
    MINIMAX_H3_TEXT_TAG,
    resolve_canvas_size,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.scheduler import MiniMaxH3Scheduler
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline, PipelineComponent
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs


class _FakeMiniMaxH3Transformer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(patch_size=(1, 2, 2))
        self.training = True
        self.static_context_calls = 0
        self.forward_calls = 0
        self.inference_modes: list[bool] = []

    def eval(self) -> "_FakeMiniMaxH3Transformer":
        self.training = False
        return self

    def prepare_static_context(
        self,
        prompt_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.static_context_calls += 1
        return prompt_embeds, position_ids

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        static_context: tuple[torch.Tensor, torch.Tensor],
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        assert static_context is not None
        self.forward_calls += 1
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)


class _SyntheticMiniMaxH3Pipeline(MiniMaxH3Pipeline):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.transformer = _FakeMiniMaxH3Transformer()
        self.vae = SimpleNamespace(
            config=SimpleNamespace(latent_channels=2),
            spatial_compression_ratio=2,
        )
        self.audio_vae = SimpleNamespace(
            config=SimpleNamespace(latent_channels=3, sampling_rate=32000)
        )
        self.scheduler = MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
        self._is_warmup = True

    def _encode_prompt(
        self,
        prompt: str,
        keyframes: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del prompt, keyframes
        return torch.ones(1, 2, 4), torch.full((2,), MINIMAX_H3_TEXT_TAG)

    def _encode_keyframes(
        self,
        keyframes: list[Image.Image],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
    ) -> None:
        del keyframes, latent_height, latent_width, generator
        return None

    def _decode_video(
        self,
        rows: torch.Tensor,
        num_condition_rows: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        del rows, num_condition_rows, num_latent_frames, latent_height, latent_width
        return torch.zeros(1, 124, 32, 32, 3, dtype=torch.uint8)

    def _decode_audio(self, rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
        del rows
        return torch.zeros(1, 2, num_audio_latents * 800)


def test_pipeline_reuses_static_context_across_joint_denoise_steps() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()

    output = pipeline.forward(
        prompt="a synthetic prompt",
        seed=42,
        height=32,
        width=32,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=4,
    )

    assert pipeline.transformer.static_context_calls == 1
    assert pipeline.transformer.forward_calls == 3
    assert output.video.shape == (1, 124, 32, 32, 3)
    assert output.video.dtype == torch.uint8
    assert output.audio.shape[0:2] == (1, 2)
    assert output.audio_sample_rate == 32000
    assert output.frame_rate == 24.0


def test_pipeline_runs_generation_in_inference_mode() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()

    pipeline.forward(
        prompt="a synthetic prompt",
        seed=42,
        height=32,
        width=32,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=2,
    )

    assert pipeline.transformer.inference_modes == [True]


def test_request_generator_preserves_released_cpu_rng_contract() -> None:
    generator = MiniMaxH3Pipeline._request_generator(123)

    assert generator.device.type == "cpu"
    assert torch.equal(
        torch.randn(8, generator=generator),
        torch.randn(8, generator=torch.Generator().manual_seed(123)),
    )


def test_pipeline_unwraps_component_scoped_transformer_weights() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    captured: dict[str, object] = {}
    expected_weights = {"proj_in.weight": torch.ones(1)}
    pipeline.transformer.load_weights = lambda weights: captured.update(weights=weights)

    pipeline.load_weights({PipelineComponent.TRANSFORMER: expected_weights})

    assert captured["weights"] is expected_weights
    assert not pipeline.transformer.training


def test_pipeline_keeps_torch_compile_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        h3_pipeline.BasePipeline,
        "__init__",
        lambda self, config: torch.nn.Module.__init__(self),
    )

    def _config(torch_compile: TorchCompileConfig) -> SimpleNamespace:
        return SimpleNamespace(
            mapping=SimpleNamespace(world_size=1),
            attention=SimpleNamespace(backend="VANILLA"),
            cache=None,
            cuda_graph=SimpleNamespace(enable=False),
            torch_compile=torch_compile,
        )

    implicit = _config(TorchCompileConfig())
    MiniMaxH3Pipeline(implicit)
    assert implicit.torch_compile.enable

    explicit = _config(TorchCompileConfig(enable=True))
    MiniMaxH3Pipeline(explicit)
    assert explicit.torch_compile.enable


def test_modular_manifest_discovers_h3_pipeline_and_transformer_config(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"hidden_size": 32}),
        encoding="utf-8",
    )
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
            }
        ),
        encoding="utf-8",
    )

    components = discover_pipeline_components(tmp_path)

    assert components == {"transformer": tmp_path / "transformer" / "config.json"}
    assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == ("MiniMaxH3ModularPipeline")


def test_public_visual_gen_lists_minimax_h3() -> None:
    from tensorrt_llm import VisualGen

    assert "MiniMaxAI/MiniMax-H3" in VisualGen.supported_models()


def test_hf_download_is_scoped_to_the_supported_fl2va_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _capture_partial_download(
        model: str,
        allow_patterns: list[str],
        revision: str | None = None,
    ) -> Path:
        captured.update(
            model=model,
            allow_patterns=allow_patterns,
            revision=revision,
        )
        return tmp_path

    def _reject_full_download(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("MiniMax-H3 must not download the full 354 GB repository")

    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_loader.download_hf_partial",
        _capture_partial_download,
        raising=False,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.visual_gen.pipeline_loader.download_hf_model",
        _reject_full_download,
    )
    loader = PipelineLoader(
        VisualGenArgs(
            model="MiniMaxAI/MiniMax-H3",
            revision="5d9b308a59ab12e67147f191e184baf704185bd1",
        ),
        device="cpu",
    )

    resolved = loader._resolve_checkpoint_dir("MiniMaxAI/MiniMax-H3")

    assert resolved == str(tmp_path)
    assert captured == {
        "model": "MiniMaxAI/MiniMax-H3",
        "allow_patterns": [
            "modular_model_index.json",
            "LICENSE",
            "README.md",
            "transformer/*",
            "text_encoder/*",
            "tokenizer/*",
            "processor/*",
            "vae/*",
            "audio_vae/*",
            "scheduler/*",
            "audio_scheduler/*",
        ],
        "revision": "5d9b308a59ab12e67147f191e184baf704185bd1",
    }


def test_infer_supports_a_last_frame_without_a_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    captured: dict[str, object] = {}
    marker = object()

    def _return_marker(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return marker

    monkeypatch.setattr(h3_pipeline, "load_image", _return_marker)

    def _capture_forward(**kwargs: object) -> str:
        captured.update(kwargs)
        return "output"

    pipeline.forward = _capture_forward
    request = SimpleNamespace(
        prompt="finish at this frame",
        params=SimpleNamespace(
            negative_prompt=None,
            num_images_per_prompt=1,
            image=None,
            extra_params={"last_image": "/tmp/last.png"},
            seed=42,
            height=768,
            width=1344,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=4,
        ),
    )

    assert pipeline.infer(request) == "output"
    assert captured["keyframes"] == [marker]
    assert captured["keyframe_anchors"] == ("last",)
    assert pipeline.extra_param_specs["last_image"].type == "str"


def test_infer_unwraps_the_executor_single_prompt_list() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    captured: dict[str, object] = {}

    def _capture_forward(**kwargs: object) -> str:
        captured.update(kwargs)
        return "output"

    pipeline.forward = _capture_forward
    request = SimpleNamespace(
        prompt=["one public API prompt"],
        prepared_inputs={"keyframes": [], "keyframe_anchors": ()},
        params=SimpleNamespace(
            negative_prompt=None,
            num_images_per_prompt=1,
            image=None,
            extra_params=None,
            seed=0,
            height=128,
            width=128,
            num_frames=124,
            frame_rate=24.0,
            num_inference_steps=2,
        ),
    )

    assert pipeline.infer(request) == "output"
    assert captured["prompt"] == "one public API prompt"


def test_prepare_request_derives_default_canvas_from_last_only_keyframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    last_frame = Image.new("RGB", (512, 1024))

    def _return_last_frame(*args: object, **kwargs: object) -> Image.Image:
        del args, kwargs
        return last_frame

    monkeypatch.setattr(h3_pipeline, "load_image", _return_last_frame)
    request = SimpleNamespace(
        params=SimpleNamespace(
            image=None,
            height=None,
            width=None,
            extra_params={"last_image": "/tmp/portrait.png"},
        ),
        prepared_inputs={},
    )

    pipeline.prepare_request(request)

    assert (request.params.height, request.params.width) == resolve_canvas_size(512, 1024)
    assert request.prepared_inputs["keyframes"] == [last_frame]
    assert request.prepared_inputs["keyframe_anchors"] == ("last",)


def test_prepare_request_uses_default_t2va_canvas_and_preserves_explicit_size() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    default_request = SimpleNamespace(
        params=SimpleNamespace(
            image=None,
            height=None,
            width=None,
            extra_params=None,
        ),
        prepared_inputs={},
    )
    explicit_request = SimpleNamespace(
        params=SimpleNamespace(
            image=None,
            height=512,
            width=768,
            extra_params=None,
        ),
        prepared_inputs={},
    )

    pipeline.prepare_request(default_request)
    pipeline.prepare_request(explicit_request)

    assert (default_request.params.height, default_request.params.width) == resolve_canvas_size(
        16, 9
    )
    assert (explicit_request.params.height, explicit_request.params.width) == (512, 768)
    assert pipeline.default_generation_params["height"] is None
    assert pipeline.default_generation_params["width"] is None


def test_prepare_request_rejects_partial_canvas_override() -> None:
    pipeline = _SyntheticMiniMaxH3Pipeline()
    request = SimpleNamespace(
        params=SimpleNamespace(
            image=None,
            height=768,
            width=None,
            extra_params=None,
        ),
        prepared_inputs={},
    )

    with pytest.raises(ValueError, match="height and width must be set together"):
        pipeline.prepare_request(request)
