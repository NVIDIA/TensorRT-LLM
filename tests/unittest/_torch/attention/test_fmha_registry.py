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

EXPECTED_DEFAULT_LIBS = (
    "msa_sparse_gqa",
    "prims_ts",
    "cute_dsl_mla",
    "flashinfer_trtllm_gen",
    "fallback",
)


def _enabled_names() -> tuple[str, ...]:
    classes = registry.get_enabled_fmha_lib_classes()
    names_by_class = {cls: name for name, cls in registry.FMHA_LIBS.items()}
    return tuple(names_by_class[cls] for cls in classes)


def test_default_fmha_lib_order_prioritizes_prims_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)

    assert registry.DEFAULT_FMHA_LIBS == EXPECTED_DEFAULT_LIBS
    assert tuple(registry.FMHA_LIBS) == EXPECTED_DEFAULT_LIBS
    assert _enabled_names() == EXPECTED_DEFAULT_LIBS


@pytest.mark.parametrize("value", ["", "   ", ", ,"])
def test_empty_fmha_lib_env_uses_default(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", value)

    assert _enabled_names() == EXPECTED_DEFAULT_LIBS


def test_exact_fmha_lib_env_preserves_order_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TLLM_FMHA_LIBS",
        " fallback, prims_ts, fallback, flashinfer_trtllm_gen ",
    )

    assert _enabled_names() == (
        "fallback",
        "prims_ts",
        "flashinfer_trtllm_gen",
    )


def test_delta_fmha_lib_env_applies_entries_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "TLLM_FMHA_LIBS",
        "-prims_ts, -fallback, +fallback, +prims_ts",
    )

    assert _enabled_names() == (
        "msa_sparse_gqa",
        "cute_dsl_mla",
        "flashinfer_trtllm_gen",
        "fallback",
        "prims_ts",
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("prims_ts,-fallback", "either an exact comma-separated list"),
        ("prims_ts,unknown", "Unknown FMHA library 'unknown'"),
        ("+", "Invalid empty FMHA library entry"),
    ],
)
def test_invalid_fmha_lib_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", value)

    with pytest.raises(ValueError, match=message):
        registry.get_enabled_fmha_lib_classes()
