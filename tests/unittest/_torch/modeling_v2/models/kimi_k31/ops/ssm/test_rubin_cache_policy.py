# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 KV layer selection of the Kimi K3.1 Rubin targets (``ops/ssm/rubin_cache_policy.py``).

The selection is the ``fp8_kv_layers`` argument a target binds. The module is loaded from its
path, so the test needs neither torch nor the family's other modules.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# CPU test (l0_cpu); a GPU host other than sm_107 skips it.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability(0) != (10, 7),
        reason="Kimi K3.1 Rubin: sm_107 or CPU only",
    ),
]

REPO = Path(__file__).resolve().parents[8]
POLICY = (
    REPO
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k31/ops/ssm/rubin_cache_policy.py"
)

#: The production K3.1 text config: 93 layers, a DSA (MLA) layer every fourth layer (one-based
#: 4, 8, ..., 88) and the last two (92, 93), so zero-based 3, 7, ..., 87, 91 and 92.
K31_NUM_LAYERS = 93
K31_FULL_ATTN_LAYERS = [*range(4, 89, 4), 92, 93]


@pytest.fixture
def policy(monkeypatch):
    """The policy module, loaded from its path."""
    spec = importlib.util.spec_from_file_location("rubin_cache_policy_under_test", POLICY)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _text_config(
    num_layers=K31_NUM_LAYERS,
    full_attn_layers=K31_FULL_ATTN_LAYERS,
    *,
    dsa=True,
    model_type="kimi_linear",
    linear_attn_config=True,
):
    return types.SimpleNamespace(
        model_type=model_type,
        architectures=["KimiLinearForCausalLM"] if model_type == "kimi_linear" else ["Other"],
        num_hidden_layers=num_layers,
        linear_attn_config={
            "full_attn_layers": list(full_attn_layers),
            "kda_layers": [n for n in range(1, num_layers + 1) if n not in full_attn_layers],
        }
        if linear_attn_config
        else None,
        dsa_config={"dsa_layers": list(full_attn_layers)} if dsa else None,
    )


def test_default_is_the_last_two_dsa_layers_with_nvfp4(policy):
    assert policy.get_kimi_k31_fp8_kv_layer_ids(_text_config(), nvfp4_enabled=True) == {91, 92}
    assert policy.get_kimi_k31_fp8_kv_layer_ids(
        _text_config(), nvfp4_enabled=True, fp8_kv_layers=None
    ) == {91, 92}


def test_default_selects_nothing_without_nvfp4(policy):
    assert policy.get_kimi_k31_fp8_kv_layer_ids(_text_config(), nvfp4_enabled=False) == frozenset()
    assert policy.get_kimi_k31_fp8_kv_layer_ids(_text_config()) == frozenset()


@pytest.mark.parametrize("nvfp4", [True, False])
def test_empty_selects_nothing(policy, nvfp4):
    assert (
        policy.get_kimi_k31_fp8_kv_layer_ids(_text_config(), nvfp4_enabled=nvfp4, fp8_kv_layers=())
        == frozenset()
    )


@pytest.mark.parametrize("layers", [(91, 92), (3,), (87, 91), (92, 3, 47)])
@pytest.mark.parametrize("nvfp4", [True, False])
def test_explicit_layers_are_taken_as_given(policy, layers, nvfp4):
    # Without NVFP4 the call sites reject an explicit selection; the policy returns it.
    assert policy.get_kimi_k31_fp8_kv_layer_ids(
        _text_config(), nvfp4_enabled=nvfp4, fp8_kv_layers=layers
    ) == set(layers)


@pytest.mark.parametrize(
    "layers, match",
    [
        ((93,), r"outside \[0, 93\): \[93\]"),
        ((-1, 91), r"outside \[0, 93\): \[-1\]"),
        ((0, 91), r"only MLA layers; these layers use KDA instead: \[0\]"),
        ((90, 91, 92), r"only MLA layers; these layers use KDA instead: \[90\]"),
    ],
)
def test_explicit_layers_must_be_mla_layers_in_range(policy, layers, match):
    with pytest.raises(ValueError, match=r"fp8_kv_layers .*" + match):
        policy.get_kimi_k31_fp8_kv_layer_ids(
            _text_config(), nvfp4_enabled=True, fp8_kv_layers=layers
        )


@pytest.mark.parametrize("value", [[91, 92], {91, 92}, (91.0,), ("91",), (True,), "91,92", 91])
def test_the_selection_must_be_a_tuple_of_ints(policy, value):
    with pytest.raises(TypeError, match="fp8_kv_layers must be None or a tuple"):
        policy.get_kimi_k31_fp8_kv_layer_ids(
            _text_config(), nvfp4_enabled=True, fp8_kv_layers=value
        )


@pytest.mark.parametrize("layers", [None, (91,)])
def test_a_config_without_dsa_selects_nothing(policy, layers):
    config = _text_config(dsa=False)
    assert (
        policy.get_kimi_k31_fp8_kv_layer_ids(config, nvfp4_enabled=True, fp8_kv_layers=layers)
        == frozenset()
    )


def test_a_nested_text_config_is_resolved(policy):
    composite = types.SimpleNamespace(
        model_type="kimi_k31", architectures=["KimiK31ForConditionalGeneration"]
    )
    composite.text_config = _text_config()
    assert policy.get_kimi_k31_fp8_kv_layer_ids(composite, nvfp4_enabled=True) == {91, 92}
    assert policy.get_kimi_k31_fp8_kv_layer_ids(
        composite, nvfp4_enabled=True, fp8_kv_layers=(87,)
    ) == {87}


def test_another_model_selects_nothing(policy):
    config = _text_config(model_type="deepseek_v32")
    assert policy.get_kimi_k31_fp8_kv_layer_ids(config, nvfp4_enabled=True) == frozenset()


def test_a_kimi_config_needs_its_linear_attn_config(policy):
    with pytest.raises(ValueError, match="fp8_kv_layers requires Kimi K3's linear_attn_config"):
        policy.get_kimi_k31_fp8_kv_layer_ids(
            _text_config(linear_attn_config=False), nvfp4_enabled=True
        )


def test_the_default_follows_the_layer_schedule(policy):
    # A small config: MLA layers one-based 2 and 4 of 4, so zero-based 1 and 3.
    config = _text_config(4, [2, 4])
    assert policy.get_kimi_k31_fp8_kv_layer_ids(config, nvfp4_enabled=True) == {1, 3}
    config = _text_config(6, [2, 4, 6])
    assert policy.get_kimi_k31_fp8_kv_layer_ids(config, nvfp4_enabled=True) == {3, 5}


def test_the_module_reads_no_settings_or_environment(policy):
    # The selection is an argument: no feature module or environment variable.
    assert not {"torch", "os"} & set(vars(policy))
    source = POLICY.read_text()
    for name in ("import torch", "import os", "features", "environ"):
        assert name not in source, name
