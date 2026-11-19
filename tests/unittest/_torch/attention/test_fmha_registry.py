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

import pytest
import torch
from fmha_test_utils import FakeFmha

from tensorrt_llm._torch.attention.backends.fmha import registry
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha
from tensorrt_llm._torch.attention.backends.fmha.prims_ts_dsa_sparse_mla import (
    PrimsTSDsaSparseMlaFmha,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

PRIMS_TS = "prims_ts"
PRIMS_TS_BLOCK_SPARSE = "prims_ts_block_sparse"
PRIMS_TS_SPARSE_MLA = "prims_ts_sparse_mla"
PRIMS_TS_DSA_SPARSE_MLA = "prims_ts_dsa_sparse_mla"


def _enabled_names() -> tuple[str, ...]:
    classes = registry.get_enabled_fmha_lib_classes()
    names_by_class = {cls: name for name, cls in registry.FMHA_LIBS.items()}
    return tuple(names_by_class[cls] for cls in classes)


def test_sparse_prims_ts_precedes_trtllm_gen_in_default_order() -> None:
    names = registry.DEFAULT_FMHA_LIBS
    assert names.index(PRIMS_TS_SPARSE_MLA) < names.index("flashinfer_trtllm_gen")
    assert names.index(PRIMS_TS_SPARSE_MLA) < names.index("fallback")


def test_default_fmha_libs_exclude_prims_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)

    assert PRIMS_TS not in registry.DEFAULT_FMHA_LIBS
    assert PRIMS_TS_SPARSE_MLA in registry.DEFAULT_FMHA_LIBS
    assert PRIMS_TS_BLOCK_SPARSE in registry.DEFAULT_FMHA_LIBS
    assert set(registry.DEFAULT_FMHA_LIBS) <= set(registry.FMHA_LIBS)
    assert _enabled_names() == registry.DEFAULT_FMHA_LIBS


def test_dsa_sparse_mla_is_registered_by_default() -> None:
    assert registry.FMHA_LIBS[PRIMS_TS_DSA_SPARSE_MLA] is PrimsTSDsaSparseMlaFmha
    assert PRIMS_TS_DSA_SPARSE_MLA in registry.DEFAULT_FMHA_LIBS


def test_only_the_block_sparse_fmha_declares_block_sparse_support() -> None:
    assert Fmha.supports_block_sparse_inputs is False
    for name, fmha_cls in registry.FMHA_LIBS.items():
        assert fmha_cls.supports_block_sparse_inputs is (name == PRIMS_TS_BLOCK_SPARSE), name


@pytest.mark.parametrize("value", ["", "   ", ", ,"])
def test_empty_fmha_lib_env_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", value)

    assert _enabled_names() == registry.DEFAULT_FMHA_LIBS


def test_exact_fmha_lib_env_preserves_order_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_name = "fallback"
    monkeypatch.setenv("TLLM_FMHA_LIBS", f" {other_name}, {PRIMS_TS}, {other_name} ")

    assert _enabled_names() == (other_name, PRIMS_TS)


def test_fmha_lib_env_prepends_registered_names_in_written_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", f" +fallback, +{PRIMS_TS}, +fallback ")

    assert _enabled_names() == (
        "fallback",
        PRIMS_TS,
        *(name for name in registry.DEFAULT_FMHA_LIBS if name != "fallback"),
    )


def test_delta_fmha_lib_env_removes_default_library(monkeypatch: pytest.MonkeyPatch) -> None:
    removed_name = registry.DEFAULT_FMHA_LIBS[-1]
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"-{removed_name}")

    assert _enabled_names() == registry.DEFAULT_FMHA_LIBS[:-1]


def test_mixed_exact_and_delta_fmha_lib_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"{PRIMS_TS},-{PRIMS_TS}")

    with pytest.raises(ValueError, match="either an exact comma-separated list"):
        registry.get_enabled_fmha_lib_classes()


def test_unknown_fmha_lib_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    unknown_name = "unknown"
    while unknown_name in registry.FMHA_LIBS:
        unknown_name += "_"
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"{PRIMS_TS},{unknown_name}")

    with pytest.raises(ValueError, match=f"Unknown FMHA library '{unknown_name}'"):
        registry.get_enabled_fmha_lib_classes()


def test_empty_delta_fmha_lib_env_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+")

    with pytest.raises(ValueError, match="Invalid empty FMHA library entry"):
        registry.get_enabled_fmha_lib_classes()


class _LocalFmha(Fmha):
    pass


@pytest.mark.parametrize(
    ("fmha_libs", "env", "expected"),
    [
        # Prepend in environment order and deduplicate names and classes.
        (
            [PRIMS_TS_SPARSE_MLA, registry.FMHA_LIBS["fallback"], "fallback"],
            f"+fallback,+{PRIMS_TS},+fallback",
            ["fallback", PRIMS_TS, PRIMS_TS_SPARSE_MLA],
        ),
        # A final removal excludes the class from both the prefix and the base list.
        (["fallback", PRIMS_TS], f"+{PRIMS_TS},+fallback,-{PRIMS_TS}", ["fallback"]),
        # A later addition overrides an earlier removal with the default candidates.
        (
            None,
            f"-{PRIMS_TS},+fallback,+{PRIMS_TS}",
            [
                "fallback",
                PRIMS_TS,
                *(name for name in registry.DEFAULT_FMHA_LIBS if name != "fallback"),
            ],
        ),
        # Removing and re-adding a class places it at its new position in the prefix.
        (
            ["fallback", PRIMS_TS],
            f"+{PRIMS_TS},+fallback,-{PRIMS_TS},+{PRIMS_TS}",
            ["fallback", PRIMS_TS],
        ),
        # An exact list replaces all configured candidates, including an unnamed class.
        ([_LocalFmha, PRIMS_TS_SPARSE_MLA], f"{PRIMS_TS},fallback", [PRIMS_TS, "fallback"]),
    ],
    ids=["prepend_dedup", "remove_last", "add_last", "readd_order", "exact_override"],
)
def test_fmha_lib_env_overrides_candidates(
    monkeypatch: pytest.MonkeyPatch,
    fmha_libs: list[str | registry.FmhaCls] | None,
    env: str,
    expected: list[str],
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", env)

    assert registry.get_enabled_fmha_lib_classes(fmha_libs) == [
        registry.FMHA_LIBS[name] for name in expected
    ]


def test_explicit_list_can_disable_fmha_lib_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "invalid")
    fallback = registry.FMHA_LIBS["fallback"]

    assert registry.get_enabled_fmha_lib_classes(["fallback"], use_fmha_libs_env=False) == [
        fallback,
    ]
    assert registry.get_enabled_fmha_lib_classes([], use_fmha_libs_env=False) == []


def test_beam_shared_prefix_is_opt_in_and_precedes_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beam = "beam_shared_prefix"
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    assert beam not in registry.DEFAULT_FMHA_LIBS
    assert beam not in _enabled_names()

    assert beam in registry.FMHA_LIBS

    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{beam}")
    enabled = _enabled_names()
    assert enabled[0] == beam
    assert "fallback" in enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("deferred_init", [False, True])
def test_create_attention_keeps_explicit_fmha_configuration(
    monkeypatch: pytest.MonkeyPatch, deferred_init: bool
) -> None:
    class _AnonymousFmha(FakeFmha):
        def __init__(self, attn: TrtllmAttention) -> None:
            super().__init__(attn, "anonymous", [])

    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    attn = create_attention(
        "TRTLLM",
        layer_idx=0,
        num_heads=8,
        num_kv_heads=2,
        head_dim=128,
        fmha_libs=[_AnonymousFmha, "fallback"],
        use_fmha_libs_env=False,
        skip_create_weights_in_init=deferred_init,
    )
    assert isinstance(attn, TrtllmAttention)
    if deferred_init:
        assert attn._fmha_manager is None
        attn.update_quant_config(None)

    first_manager = attn._fmha_manager
    assert [type(lib) for lib in first_manager.fmha_libs] == [_AnonymousFmha, FallbackFmha]
    first_anonymous = first_manager.fmha_libs[0]

    attn.update_quant_config(QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8))

    assert attn._fmha_manager is not first_manager
    assert [type(lib) for lib in attn._fmha_manager.fmha_libs] == [_AnonymousFmha, FallbackFmha]
    anonymous = attn._fmha_manager.fmha_libs[0]
    assert anonymous.attn is attn
    assert anonymous is not first_anonymous
