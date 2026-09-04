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

from tensorrt_llm._torch.attention_backend.fmha import registry
from tensorrt_llm._torch.attention_backend.fmha.interface import Fmha

PRIMS_TS = "prims_ts"
PRIMS_TS_BLOCK_SPARSE = "prims_ts_block_sparse"


def _canonical_names() -> tuple[str, ...]:
    return tuple(registry.FMHA_LIBS)


def _enabled_names() -> tuple[str, ...]:
    classes = registry.get_enabled_fmha_lib_classes()
    names_by_class = {cls: name for name, cls in registry.FMHA_LIBS.items()}
    return tuple(names_by_class[cls] for cls in classes)


def test_prims_ts_precedes_trtllm_gen_in_canonical_order() -> None:
    names = _canonical_names()
    assert names.index(PRIMS_TS) < names.index("flashinfer_trtllm_gen")


def test_default_fmha_libs_exclude_prims_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)

    assert PRIMS_TS not in registry.DEFAULT_FMHA_LIBS
    assert PRIMS_TS_BLOCK_SPARSE in registry.DEFAULT_FMHA_LIBS
    assert set(registry.DEFAULT_FMHA_LIBS) <= set(registry.FMHA_LIBS)
    assert _enabled_names() == registry.DEFAULT_FMHA_LIBS


@pytest.mark.parametrize("name", [PRIMS_TS, "fallback"])
def test_dense_fmhas_reject_unconsumed_block_sparse_inputs(name: str) -> None:
    attention = type("Attention", (), {})()
    fmha = object.__new__(registry.FMHA_LIBS[name])
    Fmha.__init__(fmha, attention)
    forward_args = type("ForwardArgs", (), {"block_sparse_inputs": object()})()

    assert not fmha.is_supported(
        object(),
        None,
        None,
        object(),
        forward_args,
    )


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
    other_name = next(name for name in reversed(_canonical_names()) if name != PRIMS_TS)
    monkeypatch.setenv("TLLM_FMHA_LIBS", f" {other_name}, {PRIMS_TS}, {other_name} ")

    assert _enabled_names() == (other_name, PRIMS_TS)


def test_delta_fmha_lib_env_adds_prims_ts_in_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{PRIMS_TS}")

    expected_names = set(registry.DEFAULT_FMHA_LIBS) | {PRIMS_TS}
    assert _enabled_names() == tuple(name for name in _canonical_names() if name in expected_names)


def test_delta_fmha_lib_env_removes_default_library(monkeypatch: pytest.MonkeyPatch) -> None:
    removed_name = registry.DEFAULT_FMHA_LIBS[-1]
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"-{removed_name}")

    expected_names = set(registry.DEFAULT_FMHA_LIBS) - {removed_name}
    assert _enabled_names() == tuple(name for name in _canonical_names() if name in expected_names)


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
