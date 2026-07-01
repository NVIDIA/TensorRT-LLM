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

Gemma 4 12B uses HF `architectures = ["Gemma4UnifiedForConditionalGeneration"]`
/ `model_type = "gemma4_unified"`. Its text backbone is a standard dense Gemma 4
text model nested under `text_config`; the multimodal front-end is encoder-free
(lightweight projections). These tests cover the text-path onboarding: that the
new architecture is registered and that the wrapper's weight-key routing drops
the encoder-free multimodal tensors and remaps the text-backbone keys into the
reused `Gemma4ForCausalLM`. (Full forward/accuracy parity is covered by the
end-to-end smoke on the real 12B checkpoint.)
"""

from types import SimpleNamespace

import pytest
import transformers
from packaging.version import Version

# gemma4_unified needs transformers>=5.10 (its config classes ship then); the repo
# pin stays at 5.5.4. A module-level pytestmark.skipif (not importorskip) keeps
# collection clean under the pin ("N collected, N skipped, exit 0"), matching
# test_modeling_gemma4.py.
_HAS_GEMMA4_UNIFIED = Version(transformers.__version__) >= Version("5.10")

pytestmark = pytest.mark.skipif(
    not _HAS_GEMMA4_UNIFIED,
    reason=f"gemma4_unified requires transformers>=5.10 (installed: {transformers.__version__})",
)

if _HAS_GEMMA4_UNIFIED:
    from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import (
        Gemma4HfWeightMapper,  # noqa: E402
    )
    from tensorrt_llm._torch.models.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,  # noqa: E402
    )
    from tensorrt_llm._torch.models.modeling_utils import (  # noqa: E402
        _GEMMA4_ARCHITECTURES,
        MODEL_CLASS_MAPPER_MAPPING,
        MODEL_CLASS_MAPPING,
        get_model_architecture,
    )

_UNIFIED_ARCH = "Gemma4UnifiedForConditionalGeneration"


def test_auto_model_registered():
    """The new arch is discoverable in the auto-model + Gemma4 arch registries
    (importing the package above ran the @register_auto_model decorator)."""
    assert _UNIFIED_ARCH in MODEL_CLASS_MAPPING
    assert _UNIFIED_ARCH in _GEMMA4_ARCHITECTURES


def test_get_model_architecture_resolves_wrapper():
    config = SimpleNamespace(architectures=[_UNIFIED_ARCH])
    cls, arch = get_model_architecture(config)
    assert cls is Gemma4UnifiedForConditionalGeneration
    assert arch == _UNIFIED_ARCH


def test_weight_mapper_registered():
    # The Gemma4 HF weight mapper must claim the unified arch too, otherwise the
    # generic mapper would mishandle the per-layer head_dim / k_eq_v / layer_scalar
    # logic. register_mapper("HF", name) stores the mapper under f"{name}_{format}".
    mapper_cls = MODEL_CLASS_MAPPER_MAPPING.get(f"{_UNIFIED_ARCH}_HF")
    assert mapper_cls is Gemma4HfWeightMapper


def test_load_weights_filters_and_remaps():
    """The wrapper hands the text core only the `model.language_model.*` weights
    (remapped to `model.*`) and drops the encoder-free multimodal projection
    tensors."""
    captured = {}

    class _FakeLLM:
        def load_weights(self, weights, weight_mapper=None):
            captured.update(weights)

    # __new__ bypasses __init__ (which builds the real text core on GPU); this
    # exercises only the pure-Python key routing in load_weights. The encoder-free
    # MM embedders are consulted only when present, so None = text-only routing.
    wrapper = Gemma4UnifiedForConditionalGeneration.__new__(Gemma4UnifiedForConditionalGeneration)
    wrapper.llm = _FakeLLM()
    wrapper.embed_vision = None
    wrapper.embed_audio = None

    raw_weights = {
        "model.language_model.embed_tokens.weight": 0,
        "model.language_model.layers.0.self_attn.q_proj.weight": 1,
        "model.language_model.layers.0.layer_scalar": 2,
        "model.language_model.norm.weight": 3,
        # Encoder-free MM projection tensors -- must be dropped by text-only routing:
        "model.vision_embedder.patch_dense.weight": 90,
        "model.embed_vision.embedding_projection.weight": 91,
        "model.embed_audio.embedding_projection.weight": 92,
    }
    wrapper.load_weights(raw_weights, weight_mapper=None)

    # Exactly the four text-backbone tensors reach the LLM.
    assert len(captured) == 4
    # "model.language_model." was remapped to "model.".
    assert "model.embed_tokens.weight" in captured
    assert "model.layers.0.self_attn.q_proj.weight" in captured
    assert "model.layers.0.layer_scalar" in captured
    assert "model.norm.weight" in captured
    # None of the encoder-free MM projection tensors leaked through.
    for key in captured:
        assert "vision_embedder" not in key
        assert "embed_vision" not in key
        assert "embed_audio" not in key
