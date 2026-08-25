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

"""Unit tests for bench_moe parallel-mode name resolution.

Covers the four legacy names (``DEP`` / ``TEP`` / ``DTP`` / ``TTP``), the
hybrid ``(D|T)TP<k>EP<m>`` grammar, and ``CUSTOM``. Pure CPU: no GPU, no MPI,
no model weights -- only the ``ConfigSpec`` -> ``(moe_ep, moe_tp, enable_dp)``
resolution and the search-time comm-axis collapse.
"""

from __future__ import annotations

import pytest

from .mapping import (
    _resolve_mapping_layout,
    is_named_parallel_mode,
    parallel_mode_enable_attention_dp,
    resolve_named_layout,
)
from .search import _comm_axis_for_parallel_mode
from .specs import ConfigSpec


def _config(mode: str, **kwargs) -> ConfigSpec:
    return ConfigSpec(backend="CUTLASS", parallel_mode=mode, **kwargs)


# --------------------------------------------------------------------------
# Legacy modes: regression guard -- resolution must be bit-for-bit unchanged.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,world_size,expected",
    [
        ("DEP", 4, (4, 1, True)),
        ("TEP", 4, (4, 1, False)),
        ("DTP", 4, (1, 4, True)),
        ("TTP", 4, (1, 4, False)),
        ("DEP", 8, (8, 1, True)),
        ("TTP", 1, (1, 1, False)),
    ],
)
def test_legacy_modes_unchanged(mode, world_size, expected):
    assert _resolve_mapping_layout(_config(mode), world_size) == expected


# --------------------------------------------------------------------------
# Hybrid (D|T)TP<k>EP<m>.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,world_size,expected_ep,expected_tp,expected_dp",
    [
        ("DTP2EP2", 4, 2, 2, True),
        ("TTP2EP2", 4, 2, 2, False),
        ("DTP2EP4", 8, 4, 2, True),
        ("TTP4EP2", 8, 2, 4, False),
        ("DTP8EP4", 32, 4, 8, True),
    ],
)
def test_hybrid_modes(mode, world_size, expected_ep, expected_tp, expected_dp):
    moe_ep, moe_tp, enable_dp = _resolve_mapping_layout(_config(mode), world_size)
    assert (moe_ep, moe_tp, enable_dp) == (expected_ep, expected_tp, expected_dp)


def test_hybrid_mode_is_case_insensitive():
    assert _resolve_mapping_layout(_config("dtp2ep2".upper()), 4) == (2, 2, True)
    assert resolve_named_layout("dtp2ep2", 4) == (2, 2, True)


def test_hybrid_mode_equivalent_to_custom():
    """A named hybrid must resolve identically to the pre-existing CUSTOM form."""
    named = _resolve_mapping_layout(_config("TTP2EP2"), 4)
    custom = _resolve_mapping_layout(
        _config("CUSTOM", moe_ep_size=2, moe_tp_size=2, enable_attention_dp=False), 4
    )
    assert named == custom


def test_hybrid_mode_rejects_world_size_mismatch():
    with pytest.raises(ValueError, match="must equal world_size=8"):
        _resolve_mapping_layout(_config("DTP2EP2"), 8)


@pytest.mark.parametrize("mode", ["TP2EP2", "DEP2", "DTP2", "XTP2EP2", "DTP2EP", "HYBRID"])
def test_unknown_or_prefixless_names_rejected(mode):
    assert not is_named_parallel_mode(mode)
    with pytest.raises(ValueError, match="Unknown parallel_mode"):
        _resolve_mapping_layout(_config(mode), 4)


def test_legacy_names_not_captured_by_hybrid_regex():
    """``DTP``/``TTP`` have no digits, so the table lookup must win."""
    assert resolve_named_layout("DTP", 4) == (1, 4, True)
    assert resolve_named_layout("TTP", 4) == (1, 4, False)


# --------------------------------------------------------------------------
# CUSTOM: unchanged behavior.
# --------------------------------------------------------------------------


def test_custom_requires_both_sizes():
    with pytest.raises(ValueError, match="requires explicit moe_ep_size and moe_tp_size"):
        _resolve_mapping_layout(_config("CUSTOM", moe_ep_size=2), 4)


def test_custom_defaults_attention_dp_to_false():
    assert _resolve_mapping_layout(_config("CUSTOM", moe_ep_size=2, moe_tp_size=2), 4) == (
        2,
        2,
        False,
    )


def test_custom_is_not_a_named_mode():
    assert not is_named_parallel_mode("CUSTOM")
    assert parallel_mode_enable_attention_dp("CUSTOM") is None


# --------------------------------------------------------------------------
# Search-time comm-axis collapse.
# --------------------------------------------------------------------------


_COMM = ("AUTO", "ALLGATHER", "DEEPEP")


@pytest.mark.parametrize("mode", ["TEP", "TTP", "TTP2EP2", "TTP4EP2"])
def test_comm_axis_collapses_without_attention_dp(mode):
    assert _comm_axis_for_parallel_mode(mode, _COMM) == ("AUTO",)


@pytest.mark.parametrize("mode", ["DEP", "DTP", "DTP2EP2", "DTP2EP4"])
def test_comm_axis_preserved_with_attention_dp(mode):
    assert _comm_axis_for_parallel_mode(mode, _COMM) == _COMM


def test_comm_axis_passthrough_for_custom():
    assert _comm_axis_for_parallel_mode("CUSTOM", _COMM) == _COMM
