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

Selection is all this file checks -- which class each factory returns, with the
classes stubbed. That is worth pinning, but it cannot catch a worker and a
drafter that agree on paper and diverge on first contact; for that see
``test_dspark_drafter_worker_contract.py``, which builds the real drafter and
drives the real worker's lazy init. The flavour probe at the bottom is the one
exception here -- it reads checkpoints written to ``tmp_path``, because it is
the single source every dispatch above consults.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models import modeling_dflash, modeling_dspark
from tensorrt_llm._torch.models.modeling_dflash import declares_dspark_heads
from tensorrt_llm._torch.speculative import utils as spec_utils
from tensorrt_llm._torch.speculative.interface import (
    SpeculativeDecodingMode,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig

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
    *,
    model_type="qwen3",
    dflash_config=None,
    architectures=None,
    attention_backend="TRTLLM",
    embedded=False,
    top_level=None,
):
    """Duck-typed (target ModelConfig, draft ModelConfig) for dispatch-only asserts."""
    model_config = SimpleNamespace(
        spec_config=SimpleNamespace(
            speculative_model="/nonexistent/drafter",
            block_size=7,
            attention_backend=attention_backend,
            draft_is_embedded_in_target=embedded,
        )
    )
    draft_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            model_type=model_type,
            architectures=architectures,
            dflash_config=dflash_config,
            **(top_level or {}),
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


# --------------------------------------------------------------------------
# decoding_type: DSpark
# --------------------------------------------------------------------------


def test_standalone_qwen3_drafter_selects_qwen3_dspark(monkeypatch, stub_dspark):
    model_config, draft_config = _configs(model_type="qwen3", embedded=False)

    built = modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    assert built is _QWEN3_SENTINEL


def test_embedded_draft_selects_dsv4_dspark(monkeypatch, stub_dspark):
    monkeypatch.setattr(modeling_dspark, "count_dspark_stages", lambda _p: 3)
    model_config, draft_config = _configs(model_type="deepseek_v4", embedded=True)

    built = modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    assert built is _DSV4_SENTINEL


def test_unknown_standalone_model_type_lists_supported(monkeypatch, stub_dspark):
    model_config, draft_config = _configs(model_type="llama", embedded=False)

    with pytest.raises(NotImplementedError) as excinfo:
        modeling_dspark._build_dspark_draft(model_config, draft_config, None, None)

    message = str(excinfo.value)
    assert "llama" in message
    assert "qwen3" in message, "the error must list the supported draft model_type values"


def test_standalone_drafter_receives_the_attention_backend(monkeypatch, stub_dspark):
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


def test_dflash_refuses_a_top_level_spelling_dspark_drafter(stub_dflash):
    # RadixArk/Kimi-K3-DSpark as published: the head switches sit at the top
    # level and dflash_config carries only mask_token_id / target_layer_ids.
    # A reader that looks in dflash_config alone misses exactly the drafter
    # this guard exists to catch.
    model_config, draft_config = _configs(
        dflash_config={"mask_token_id": 163824, "target_layer_ids": [7, 23, 51, 67, 83]},
        top_level={
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "block_size": 7,
        },
    )

    with pytest.raises(ValueError, match="DSpark"):
        modeling_dflash._build_dflash_draft(model_config, draft_config, None, None)


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


# --------------------------------------------------------------------------
# Runtime routing
#
# The worker, the spec metadata and the separate-draft-KV-cache decision must
# follow the same flavour flag the builder follows. When they disagree,
# DSv4DSparkWorker gets a standalone drafter and dies reaching for V4-draft-only
# attributes (num_stages, write_context_windows) -- only at the first forward,
# long after the engine reported a successful build.
# --------------------------------------------------------------------------

_WORKER_SENTINELS = {
    "DFlashWorker": object(),
    "DSparkWorker": object(),
    "DSv4DSparkWorker": object(),
}
# SimpleNamespace rather than object(): get_spec_metadata assigns
# ``metadata.enable_penalty`` on whatever it built, which a bare object rejects.
_METADATA_SENTINELS = {
    "DFlashSpecMetadata": SimpleNamespace(),
    "DSparkSpecMetadata": SimpleNamespace(),
}


@pytest.fixture
def stub_runtime(monkeypatch):
    """Stub the worker/metadata classes: constructing the real ones needs CUDA."""
    for name, sentinel in _WORKER_SENTINELS.items():
        monkeypatch.setattr(spec_utils, name, lambda *a, _s=sentinel, **k: _s)
    for name, sentinel in _METADATA_SENTINELS.items():
        monkeypatch.setattr(spec_utils, name, lambda *a, _s=sentinel, **k: _s)


def _spec_config(mode, *, embedded, allow_separate_kv=True):
    """Duck-typed spec config carrying only what the routing functions read."""
    return SimpleNamespace(
        spec_dec_mode=mode,
        draft_is_embedded_in_target=embedded,
        _use_shared_kv_cache=False,
        _allow_separate_draft_kv_cache=allow_separate_kv,
        max_draft_len=7,
        max_total_draft_tokens=7,
        tokens_per_gen_step=8,
        target_layer_ids=[7, 23, 51, 67, 83],
        advanced_sampling_mode=None,
        # Read by get_spec_metadata for the occurrence-penalty workspace; the
        # routing under test does not depend on it.
        enable_penalty=False,
    )


@pytest.mark.parametrize(
    "mode,embedded,expected",
    [
        (SpeculativeDecodingMode.DSPARK, True, "DSv4DSparkWorker"),
        (SpeculativeDecodingMode.DSPARK, False, "DSparkWorker"),
        (SpeculativeDecodingMode.DFLASH, False, "DFlashWorker"),
    ],
)
def test_worker_follows_the_flavour_not_the_mode(stub_runtime, mode, embedded, expected):
    worker = spec_utils.get_spec_worker(
        _spec_config(mode, embedded=embedded),
        model_config=None,
        mapping=None,
        use_separate_draft_kv_cache=False,
    )
    assert worker is _WORKER_SENTINELS[expected]


@pytest.mark.parametrize(
    "mode,embedded,expected",
    [
        (SpeculativeDecodingMode.DSPARK, True, "DSparkSpecMetadata"),
        (SpeculativeDecodingMode.DSPARK, False, "DFlashSpecMetadata"),
        (SpeculativeDecodingMode.DFLASH, False, "DFlashSpecMetadata"),
    ],
)
def test_spec_metadata_follows_the_flavour_not_the_mode(stub_runtime, mode, embedded, expected):
    metadata = spec_utils.get_spec_metadata(
        _spec_config(mode, embedded=embedded),
        SimpleNamespace(hidden_size=7168, torch_dtype=None, vocab_size=163840),
        max_num_requests=8,
        max_num_tokens=4096,
    )
    assert metadata is _METADATA_SENTINELS[expected]


@pytest.mark.parametrize(
    "mode,embedded,expected",
    [
        # The embedded draft opts out: it owns a rolling captured-context window.
        (SpeculativeDecodingMode.DSPARK, True, False),
        # A standalone DSpark drafter runs on DFlashWorker's paged draft KV --
        # the path K3 used before its decoding_type moved to DSpark.
        (SpeculativeDecodingMode.DSPARK, False, True),
        (SpeculativeDecodingMode.DFLASH, False, True),
    ],
)
def test_separate_draft_kv_cache_follows_the_flavour(mode, embedded, expected):
    config = _spec_config(mode, embedded=embedded)
    assert should_use_separate_draft_kv_cache(config) is expected


# --------------------------------------------------------------------------
# The flavour probe itself. This is the single source every dispatch above
# reads, so it is the one place the embedded/standalone question is decided.
# --------------------------------------------------------------------------


def _write_ckpt(tmp_path, *, weight_map=None, model_type=None):
    if weight_map is not None:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    if model_type is not None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return DSparkDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path))


def test_probe_reads_the_mtp_namespace_from_the_weight_index(tmp_path):
    config = _write_ckpt(
        tmp_path,
        weight_map={
            "mtp.0.attn.wq_a.weight": "x.safetensors",
            "layers.0.q.weight": "x.safetensors",
        },
        model_type="deepseek_v4",
    )
    assert config.draft_is_embedded_in_target is True


def test_probe_treats_a_standalone_drafter_index_as_standalone(tmp_path):
    config = _write_ckpt(
        tmp_path,
        weight_map={"layers.0.self_attn.q_proj.weight": "x.safetensors"},
        model_type="qwen3",
    )
    assert config.draft_is_embedded_in_target is False


def test_probe_falls_back_to_model_type_without_an_index(tmp_path):
    # A V4 checkpoint whose index file is absent must not be read as
    # standalone: the standalone lineage has no V4 drafter.
    config = _write_ckpt(tmp_path, model_type="deepseek_v4")
    assert config.draft_is_embedded_in_target is True


def test_probe_is_standalone_when_nothing_can_be_read(tmp_path):
    # Fail soft: an unreadable or not-yet-downloaded checkpoint must not crash
    # config validation, and standalone is the safe default (it is the flavour
    # whose worker probes the DSpark heads defensively).
    config = DSparkDecodingConfig(max_draft_len=7, speculative_model=str(tmp_path / "missing"))
    assert config.draft_is_embedded_in_target is False
