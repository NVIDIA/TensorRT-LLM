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

"""Registry and configuration tests for the MiniMax-H3 pipeline.

Weight-free: these cover discovery (``_class_name`` -> pipeline), the
``pipeline_config`` knob surface and the geometry the pipeline exposes to its
blocks, so a registration or knob regression fails without a 465 GB checkpoint.

Run:
    pytest tests/unittest/_torch/visual_gen/test_minimax_h3_registry.py -v
"""

import json
import os
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline

MINIMAX_H3_TRANSFORMER_CONFIG = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "ffn_dim": 14336,
    "in_channels": 24,
    "audio_in_channels": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,
    "time_embed_hidden_dim": 5376,
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
}


class TestRegistration:
    def test_registered_under_its_class_name(self):
        assert "MiniMaxH3Pipeline" in PIPELINE_REGISTRY
        entry = PIPELINE_REGISTRY["MiniMaxH3Pipeline"]
        assert entry.pipeline_cls is MiniMaxH3Pipeline
        assert "MiniMaxAI/MiniMax-H3" in entry.hf_ids
        assert entry.doc

    def test_conditioner_knobs_are_declared(self):
        # The loader rejects any `pipeline_config` key not declared here.
        assert PIPELINE_REGISTRY["MiniMaxH3Pipeline"].defaults == {
            "conditioner_offload": "auto",
            "conditioner_device": "",
        }

    def test_detected_from_model_index(self, tmp_path):
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
        )
        assert AutoPipeline._detect_from_checkpoint(str(tmp_path)) == "MiniMaxH3Pipeline"


class TestConditionerOffloadKnob:
    @staticmethod
    def _pipeline(knob=None, **extra):
        from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
        from tensorrt_llm._torch.visual_gen.config import (
            DiffusionModelConfig,
            DiffusionPipelineConfig,
        )

        extra_attrs = {} if knob is None else {"conditioner_offload": knob}
        extra_attrs.update(extra)
        model_config = DiffusionModelConfig(
            component_name="transformer",
            pretrained_config=SimpleNamespace(**MINIMAX_H3_TRANSFORMER_CONFIG),
        )
        config = DiffusionPipelineConfig(
            model_configs={"transformer": model_config}, extra_attrs=extra_attrs
        )
        with MetaInitMode():
            return MiniMaxH3Pipeline(config)

    def test_defaults_to_auto(self):
        assert self._pipeline()._conditioner_offload_mode == "auto"

    @pytest.mark.parametrize("knob", ["auto", "always", "never"])
    def test_accepts_documented_values(self, knob):
        assert self._pipeline(knob)._conditioner_offload_mode == knob

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="conditioner_offload"):
            self._pipeline("sometimes")

    def test_always_and_never_ignore_device_memory(self):
        always = self._pipeline("always")
        never = self._pipeline("never")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert always._resolve_conditioner_offload(device) is True
        assert never._resolve_conditioner_offload(device) is False

    def test_auto_keeps_the_conditioner_resident_on_the_host(self):
        # With no accelerator in play there is nothing to swap.
        assert self._pipeline("auto")._resolve_conditioner_offload(torch.device("cpu")) is False

    def test_no_conditioner_device_by_default(self):
        assert self._pipeline()._conditioner_device is None

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs a second GPU")
    def test_conditioner_device_is_parsed(self):
        pipeline = self._pipeline("never", conditioner_device="cuda:1")
        assert pipeline._conditioner_device == torch.device("cuda:1")

    def test_conditioner_device_rejects_an_invisible_card(self):
        with pytest.raises(ValueError, match="conditioner_device"):
            self._pipeline("never", conditioner_device=f"cuda:{torch.cuda.device_count() + 8}")


class TestGeometry:
    """The geometry every block keys off, from the released checkpoint's config."""

    @staticmethod
    def _pipeline():
        return TestConditionerOffloadKnob._pipeline()

    def test_generation_envelope(self):
        pipeline = self._pipeline()
        assert pipeline.fps == 24
        assert (pipeline.min_duration, pipeline.max_duration) == (5.0, 15.0)
        assert pipeline.audio_channels == 2
        assert pipeline.audio_sampling_rate == 32000

    def test_canvas_multiple_follows_vae_and_patch(self):
        pipeline = self._pipeline()
        # 16x spatial compression times the transformer's width patch of 2.
        assert pipeline.canvas_multiple == 32
        assert pipeline.canvas_short_edge == 768

    def test_conditioning_layer_is_not_the_last(self):
        pipeline = self._pipeline()
        # MiniMax-H3 reads hidden_states[50]; the final layer is post-norm and
        # is not what the released weights were trained against.
        assert pipeline.text_encoder_layer == 50

    def test_pixel_normalization_is_imagenet(self):
        pipeline = self._pipeline()
        assert pipeline.pixel_mean == (0.485, 0.456, 0.406)
        assert pipeline.pixel_std == (0.229, 0.224, 0.225)
