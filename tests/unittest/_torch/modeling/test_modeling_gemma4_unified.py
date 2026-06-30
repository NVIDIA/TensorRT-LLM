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
"""Unit tests for the Gemma 4 12B *Unified* (encoder-free) text path.

Gemma 4 12B uses HF ``architectures = ["Gemma4UnifiedForConditionalGeneration"]``
/ ``model_type = "gemma4_unified"``. Its text backbone is a standard dense
Gemma 4 text model nested under ``text_config``; the multimodal front-end is
encoder-free (lightweight projections). These tests cover the Phase-A (text)
onboarding: that the new architecture is *registered* and that the wrapper's
weight-key routing drops the encoder-free MM tensors and remaps the text
backbone keys into the reused ``Gemma4ForCausalLM``.

(Full forward/accuracy parity is covered by the end-to-end smoke on the real
12B checkpoint, and — once the encoder-free MM front-end lands in Phase B — by a
``TestModelingMultimodal`` subclass per the multimodal-onboarding skill.)
"""

import unittest
from types import SimpleNamespace

import pytest
import transformers
from packaging.version import Version

# Gemma 4 12B Unified (gemma4_unified) requires transformers>=5.10 (the release
# that ships the gemma4_unified config/model classes). TRT-LLM does NOT change
# its transformers pin (still 5.5.4) for this; so under the pinned CI env this
# test skips cleanly, and it only runs once the environment has transformers
# >=5.10 (which the user installs at runtime to use the 12B model).
pytestmark = pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.10"),
    reason=f"gemma4_unified requires transformers>=5.10 (installed: {transformers.__version__})",
)

_UNIFIED_ARCH = "Gemma4UnifiedForConditionalGeneration"


class TestGemma4UnifiedRegistration(unittest.TestCase):
    """The new arch must be discoverable by the auto-model + weight-mapper
    registries and by get_model_architecture (no GPU / no checkpoint needed)."""

    def test_auto_model_registered(self):
        # Importing the package runs the @register_auto_model decorator.
        import tensorrt_llm._torch.models  # noqa: F401
        from tensorrt_llm._torch.models.modeling_utils import (
            _GEMMA4_ARCHITECTURES,
            MODEL_CLASS_MAPPING,
        )

        self.assertIn(_UNIFIED_ARCH, MODEL_CLASS_MAPPING)
        self.assertIn(_UNIFIED_ARCH, _GEMMA4_ARCHITECTURES)

    def test_get_model_architecture_resolves_wrapper(self):
        from tensorrt_llm._torch.models.modeling_gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )
        from tensorrt_llm._torch.models.modeling_utils import get_model_architecture

        cfg = SimpleNamespace(architectures=[_UNIFIED_ARCH])
        cls, arch = get_model_architecture(cfg)
        self.assertIs(cls, Gemma4UnifiedForConditionalGeneration)
        self.assertEqual(arch, _UNIFIED_ARCH)

    def test_weight_mapper_registered(self):
        # The Gemma4 HF weight mapper must claim the unified arch too, otherwise
        # the generic mapper would mishandle the per-layer head_dim / k_eq_v /
        # layer_scalar logic.
        from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import (
            Gemma4HfWeightMapper,
        )
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPER_MAPPING

        # register_mapper("HF", name) stores the mapper under key f"{name}_{format}".
        mapper_cls = MODEL_CLASS_MAPPER_MAPPING.get(f"{_UNIFIED_ARCH}_HF")
        self.assertIs(mapper_cls, Gemma4HfWeightMapper)


class TestGemma4UnifiedWeightRouting(unittest.TestCase):
    """The wrapper must hand the text core only the ``model.language_model.*``
    weights (remapped to ``model.*``) and drop the 11 encoder-free multimodal
    projection tensors."""

    def test_load_weights_filters_and_remaps(self):
        from tensorrt_llm._torch.models.modeling_gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )

        captured = {}

        class _FakeLLM:
            def load_weights(self, weights, weight_mapper=None):
                captured.update(weights)

        # Bypass __init__ (which builds the real text core on GPU); we only
        # exercise the pure-Python key routing in load_weights.
        wrapper = Gemma4UnifiedForConditionalGeneration.__new__(
            Gemma4UnifiedForConditionalGeneration
        )
        wrapper.llm = _FakeLLM()
        # __new__ bypasses __init__; the encoder-free MM embedders are consulted
        # in load_weights only when present. None = text-only routing (Phase A).
        wrapper.embed_vision = None
        wrapper.embed_audio = None

        raw = {
            "model.language_model.embed_tokens.weight": 0,
            "model.language_model.layers.0.self_attn.q_proj.weight": 1,
            "model.language_model.layers.0.layer_scalar": 2,
            "model.language_model.norm.weight": 3,
            # encoder-free MM projection tensors — must be dropped in Phase A:
            "model.vision_embedder.patch_dense.weight": 90,
            "model.embed_vision.embedding_projection.weight": 91,
            "model.embed_audio.embedding_projection.weight": 92,
        }
        wrapper.load_weights(raw, weight_mapper=None)

        # 1) exactly the four text-backbone tensors reach the LLM
        self.assertEqual(len(captured), 4)
        # 2) "model.language_model." was remapped to "model."
        self.assertIn("model.embed_tokens.weight", captured)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", captured)
        self.assertIn("model.layers.0.layer_scalar", captured)
        self.assertIn("model.norm.weight", captured)
        # 3) none of the encoder-free MM projection tensors leaked through
        for k in captured:
            self.assertNotIn("vision_embedder", k)
            self.assertNotIn("embed_vision", k)
            self.assertNotIn("embed_audio", k)


if __name__ == "__main__":
    unittest.main()
