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
"""Which DSpark drafter ``decoding_type: DSpark`` builds, and what DFlash refuses.

``DSpark`` ships in two flavours -- embedded in the target checkpoint
(DeepSeek-V4-Pro's ``mtp.*``) or standalone with its own checkpoint -- and one
builder picks between them, then picks the standalone backbone by the draft
checkpoint's ``model_type``.

The DFlash half matters just as much: DFlash no longer implements the DSpark
head set, so a drafter that declares it must be refused rather than served
without it. Silently dropping the Markov head does not fail anything; it lowers
the acceptance rate, which no test would attribute back to this function.

Everything here asserts *which class is selected*, never the object it builds:
constructing a real drafter needs GPUs and checkpoints.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models import modeling_dflash, modeling_dspark
from tensorrt_llm._torch.models.modeling_dflash import declares_dspark_heads

_DSV4_SENTINEL = object()
_QWEN3_SENTINEL = object()
_DFLASH_SENTINEL = object()
_LAGUNA_SENTINEL = object()

_DSPARK_HEADS = {
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "use_confidence_head": True,
    "shift_label": True,
    "projector_type": "dspark",
}


def _configs(
    *, model_type="qwen3", dflash_config=None, architectures=None, attention_backend="TRTLLM"
):
    """Duck-typed (target ModelConfig, draft ModelConfig) for dispatch-only asserts."""
    model_config = SimpleNamespace(
        spec_config=SimpleNamespace(
            speculative_model="/nonexistent/drafter",
            block_size=7,
            attention_backend=attention_backend,
        )
    )
    draft_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type=model_type,
            architectures=architectures,
            dflash_config=dflash_config,
        )
    )
    return model_config, draft_config


@pytest.fixture
def stub_dspark(monkeypatch):
    """Replace the DSpark drafter classes with sentinel-returning stubs."""
    monkeypatch.setattr(modeling_dspark, "DSv4DSparkForCausalLM", lambda *a, **k: _DSV4_SENTINEL)
    monkeypatch.setattr(
        modeling_dspark,
        "_DSPARK_DRAFTERS_BY_MODEL_TYPE",
        {"qwen3": lambda *a, **k: _QWEN3_SENTINEL},
    )
    monkeypatch.setattr(modeling_dspark, "validate_dspark_eplb_layer_base", lambda *a, **k: None)


@pytest.fixture
def stub_dflash(monkeypatch):
    monkeypatch.setattr(modeling_dflash, "DFlashForCausalLM", lambda *a, **k: _DFLASH_SENTINEL)
    monkeypatch.setattr(
        modeling_dflash, "DFlashLagunaForCausalLM", lambda *a, **k: _LAGUNA_SENTINEL
    )


def _embedded(monkeypatch, stages):
    """Force the level-1 probe: ``stages`` is the mtp.* count, None if standalone."""
    monkeypatch.setattr(modeling_dspark, "count_dspark_stages", lambda _p: stages)


# --------------------------------------------------------------------------
# decoding_type: DSpark
# --------------------------------------------------------------------------


def test_standalone_qwen3_drafter_selects_qwen3_dspark(monkeypatch, stub_dspark):
    _embedded(monkeypatch, None)
    model_config, draft_config = _configs(model_type="qwen3")

    built = modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    assert built is _QWEN3_SENTINEL


def test_embedded_draft_selects_dsv4_dspark(monkeypatch, stub_dspark):
    # The mtp.* namespace in the checkpoint index is the level-1 probe.
    _embedded(monkeypatch, 3)
    model_config, draft_config = _configs(model_type="deepseek_v4")

    built = modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    assert built is _DSV4_SENTINEL


def test_deepseek_v4_without_weight_index_still_selects_dsv4(monkeypatch, stub_dspark):
    # Fallback arm of the probe: a V4 checkpoint whose index file is absent must
    # not fall through to the standalone lineage, which has no V4 drafter.
    _embedded(monkeypatch, None)
    model_config, draft_config = _configs(model_type="deepseek_v4")

    built = modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    assert built is _DSV4_SENTINEL


def test_unknown_standalone_model_type_lists_supported(monkeypatch, stub_dspark):
    _embedded(monkeypatch, None)
    model_config, draft_config = _configs(model_type="llama")

    with pytest.raises(NotImplementedError) as excinfo:
        modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    message = str(excinfo.value)
    assert "llama" in message
    assert "qwen3" in message, "the error must list the supported draft model_type values"


def test_standalone_drafter_receives_the_attention_backend(monkeypatch, stub_dspark):
    _embedded(monkeypatch, None)
    seen = {}

    def _capture(draft_config, *, dflash_attention_backend):
        seen["backend"] = dflash_attention_backend
        return _QWEN3_SENTINEL

    monkeypatch.setattr(modeling_dspark, "_DSPARK_DRAFTERS_BY_MODEL_TYPE", {"qwen3": _capture})
    model_config, draft_config = _configs(attention_backend="TRTLLM")

    modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    assert seen["backend"] == "TRTLLM"


# --------------------------------------------------------------------------
# decoding_type: DFlash
# --------------------------------------------------------------------------


def test_dflash_refuses_a_dspark_drafter(stub_dflash):
    model_config, draft_config = _configs(dflash_config=dict(_DSPARK_HEADS))

    with pytest.raises(ValueError) as excinfo:
        modeling_dflash._build_dflash_draft(model_config, draft_config, None, None)

    message = str(excinfo.value)
    assert "DSpark" in message
    assert "decoding_type" in message, "the error must say how to fix the config"


@pytest.mark.parametrize(
    "field,value",
    [
        ("markov_rank", 256),
        ("use_confidence_head", True),
        ("shift_label", True),
        ("projector_type", "dspark"),
    ],
)
def test_any_single_dspark_field_is_enough_to_refuse(stub_dflash, field, value):
    # Each field alone means the drafter was trained under the DSpark
    # convention; serving it as plain DFlash degrades it silently.
    model_config, draft_config = _configs(dflash_config={"mask_token_id": 7, field: value})

    with pytest.raises(ValueError, match="DSpark"):
        modeling_dflash._build_dflash_draft(model_config, draft_config, None, None)


def test_plain_dflash_drafter_is_unchanged(stub_dflash):
    model_config, draft_config = _configs(
        dflash_config={"mask_token_id": 7, "target_layer_ids": [0, 1]}
    )

    built = modeling_dflash._build_dflash_draft(model_config, draft_config, None, None)

    assert built is _DFLASH_SENTINEL


def test_laguna_drafter_is_unchanged(stub_dflash):
    model_config, draft_config = _configs(
        architectures=["DFlashLagunaForCausalLM"],
        dflash_config={"mask_token_id": 7},
    )

    built = modeling_dflash._build_dflash_draft(model_config, draft_config, None, None)

    assert built is _LAGUNA_SENTINEL


def test_legacy_causal_dflash_config_is_not_mistaken_for_dspark(stub_dflash):
    # Laguna configs carry ``causal`` without any DSpark field; the legacy
    # decode path handles it, so this must not trip the refusal.
    model_config, draft_config = _configs(dflash_config={"mask_token_id": 7, "causal": True})

    built = modeling_dflash._build_dflash_draft(model_config, draft_config, None, None)

    assert built is _DFLASH_SENTINEL


# --------------------------------------------------------------------------
# The predicate itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dflash_config,expected",
    [
        (None, False),
        ({}, False),
        ({"mask_token_id": 7}, False),
        ({"causal": True}, False),
        ({"markov_rank": 0}, False),
        ({"shift_label": False}, False),
        ({"markov_rank": 256}, True),
        ({"shift_label": True}, True),
        ({"use_confidence_head": True}, True),
        ({"projector_type": "dspark"}, True),
        ({"projector_type": "DSpark"}, True),
    ],
)
def test_declares_dspark_heads(dflash_config, expected):
    config = SimpleNamespace(dflash_config=dflash_config)
    assert declares_dspark_heads(config) is expected
