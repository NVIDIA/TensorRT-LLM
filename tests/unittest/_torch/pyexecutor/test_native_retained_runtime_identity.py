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
"""Actual native Attention/RopeParams types, model-free CPU ownership regressions."""

import weakref
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata, RopeParams
from tensorrt_llm._torch.pyexecutor.retained_identity import _snapshot_extra_attrs

pytestmark = pytest.mark.cpu_only


def bound_engine():
    model = torch.nn.Module()
    attention = Attention.__new__(Attention)
    torch.nn.Module.__init__(attention)
    attention.layer_idx = 0
    attention.layer_idx_str = "0"
    rope = RopeParams(dim=128, max_positions=40960)
    attention.pos_embd_params = SimpleNamespace(rope=rope)
    model.add_module("attention", attention)
    attrs = {"attn_layers": {"0": weakref.ref(attention)}, "kv_cache_dtype": "auto"}
    model.model_config = SimpleNamespace(extra_attrs=attrs)
    engine = SimpleNamespace(model=model, attn_metadata=None)
    return SimpleNamespace(model_engine=engine, device_id=0), attention, attrs


def test_native_registry_has_stable_layer_and_effective_rope_identity():
    executor, attention, attrs = bound_engine()
    snapshot = _snapshot_extra_attrs(executor)
    descriptor = snapshot["attn_layers"]["0"]
    assert descriptor["module"] == "attention"
    assert descriptor["global_layer"] == 0
    assert descriptor["rope"]["dim"] == 128
    attention.pos_embd_params.rope.theta = 12345.0
    assert _snapshot_extra_attrs(executor) != snapshot


def test_metadata_presence_and_dead_rope_cache_do_not_change_identity():
    executor, attention, attrs = bound_engine()
    baseline = _snapshot_extra_attrs(executor)
    metadata = AttentionMetadata(
        max_num_requests=1, max_num_tokens=64, seq_lens=None, num_contexts=0, seq_lens_kv=None
    )
    executor.model_engine.attn_metadata = metadata
    attrs["attention_metadata"] = weakref.ref(metadata)
    attrs["spec_metadata"] = None
    constants_type = namedtuple(
        "RopeConstParams",
        "inv_freq cos_sin",
        module="tensorrt_llm._torch.attention.backends.interface",
    )
    temporary = torch.nn.Module()
    expired = weakref.ref(temporary)
    del temporary
    attrs["rope_const_params"] = {
        (attention.pos_embd_params.rope, True): constants_type(None, expired)
    }
    assert _snapshot_extra_attrs(executor) == baseline
    attrs["rope_const_params"] = {}
    assert _snapshot_extra_attrs(executor) == baseline


@pytest.mark.parametrize("kind", ["dead", "foreign", "wrong-key", "not-weakref"])
def test_native_registry_rejects_unowned_or_misidentified_layers(kind):
    executor, attention, attrs = bound_engine()
    if kind == "dead":
        other = torch.nn.Module()
        attrs["attn_layers"]["0"] = weakref.ref(other)
        del other
    elif kind == "foreign":
        foreign, foreign_attention, _ = bound_engine()
        attrs["attn_layers"]["0"] = weakref.ref(foreign_attention)
    elif kind == "wrong-key":
        attrs["attn_layers"] = {"1": weakref.ref(attention)}
    else:
        attrs["attn_layers"]["0"] = attention
    with pytest.raises(ValueError):
        _snapshot_extra_attrs(executor)


def test_foreign_live_metadata_rejected():
    executor, attention, attrs = bound_engine()
    foreign = torch.nn.Module()
    attrs["attention_metadata"] = weakref.ref(foreign)
    with pytest.raises(ValueError, match="actual engine"):
        _snapshot_extra_attrs(executor)


def test_foreign_rope_parameters_rejected():
    executor, attention, attrs = bound_engine()
    attrs["rope_const_params"] = {(RopeParams(), True): ()}
    with pytest.raises(ValueError, match="foreign"):
        _snapshot_extra_attrs(executor)


def test_unknown_state_is_not_silently_omitted_and_semantic_knobs_remain():
    executor, attention, attrs = bound_engine()
    attrs["future_semantic_knob"] = {"nested": 7}
    baseline = _snapshot_extra_attrs(executor)
    assert baseline["future_semantic_knob"] == {"nested": 7}
    attrs["future_semantic_knob"]["nested"] = 8
    assert _snapshot_extra_attrs(executor) != baseline
    attrs["unknown_resource"] = object()
    with pytest.raises(ValueError, match="unqualified"):
        _snapshot_extra_attrs(executor)


def test_live_host_rope_tensor_rejected():
    executor, attention, attrs = bound_engine()
    tensor = torch.zeros(4, dtype=torch.float32)
    constants_type = namedtuple(
        "RopeConstParams",
        "inv_freq cos_sin",
        module="tensorrt_llm._torch.attention.backends.interface",
    )
    attrs["rope_const_params"] = {
        (attention.pos_embd_params.rope, True): constants_type(None, weakref.ref(tensor))
    }
    with pytest.raises(ValueError, match="live native RoPE"):
        _snapshot_extra_attrs(executor)


@pytest.mark.parametrize("kind", ["missing-key", "empty", "duplicate-target"])
def test_registry_requires_exact_owned_attention_population(kind):
    executor, attention, attrs = bound_engine()
    if kind == "missing-key":
        del attrs["attn_layers"]
    elif kind == "empty":
        attrs["attn_layers"] = {}
    else:
        attrs["attn_layers"]["0_duplicate"] = weakref.ref(attention)
    with pytest.raises(ValueError):
        _snapshot_extra_attrs(executor)


def test_attention_aliases_are_rejected():
    executor, attention, attrs = bound_engine()
    executor.model_engine.model.add_module("duplicate_alias", attention)
    with pytest.raises(ValueError):
        _snapshot_extra_attrs(executor)


def test_global_layer_id_and_registry_key_must_agree():
    executor, attention, attrs = bound_engine()
    attention.layer_idx = 17
    with pytest.raises(ValueError):
        _snapshot_extra_attrs(executor)


def test_partial_nonempty_registry_is_rejected():
    executor, attention, attrs = bound_engine()
    other_executor, other, _ = bound_engine()
    other.layer_idx = 1
    other.layer_idx_str = "1"
    executor.model_engine.model.add_module("unregistered_attention", other)
    with pytest.raises(ValueError):
        _snapshot_extra_attrs(executor)


def test_owned_arbitrary_metadata_type_is_rejected():
    executor, attention, attrs = bound_engine()
    arbitrary = torch.nn.Module()
    executor.model_engine.attn_metadata = arbitrary
    attrs["attention_metadata"] = weakref.ref(arbitrary)
    with pytest.raises(ValueError):
        _snapshot_extra_attrs(executor)


@pytest.mark.parametrize("global_id", [True, "0", -1])
def test_global_layer_id_requires_nonnegative_exact_integer(global_id):
    executor, attention, attrs = bound_engine()
    attention.layer_idx = global_id
    attention.layer_idx_str = str(global_id)
    attrs["attn_layers"] = {str(global_id): weakref.ref(attention)}
    with pytest.raises(ValueError):
        _snapshot_extra_attrs(executor)
