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

import argparse

import pytest

from .mapping import (
    _resolve_mapping_layout,
    default_hybrid_parallel_modes,
    is_named_parallel_mode,
    parallel_mode_enable_attention_dp,
    resolve_named_layout,
)
from .search import (
    _axis_values_from_args,
    _comm_axis_for_parallel_mode,
    _default_parallel_axis_values,
)
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


# --------------------------------------------------------------------------
# Default --search parallel expansion.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "world_size,expected",
    [
        # No non-degenerate split: the historical four presets, unchanged.
        (1, ()),
        (2, ()),
        # Perfect squares: one even split, so one grid x two attention flavours.
        (4, ("DTP2EP2", "TTP2EP2")),
        (16, ("DTP4EP4", "TTP4EP4")),
        (64, ("DTP8EP8", "TTP8EP8")),
        # Not a perfect square: both neighbouring grids are offered.
        (8, ("DTP2EP4", "TTP2EP4", "DTP4EP2", "TTP4EP2")),
        (32, ("DTP4EP8", "TTP4EP8", "DTP8EP4", "TTP8EP4")),
        # Non-power-of-two world sizes still resolve.
        (6, ("DTP2EP3", "TTP2EP3", "DTP3EP2", "TTP3EP2")),
    ],
)
def test_default_hybrid_parallel_modes(world_size, expected):
    assert default_hybrid_parallel_modes(world_size) == expected


@pytest.mark.parametrize("world_size", [1, 2, 4, 6, 8, 16, 32, 64])
def test_default_axis_starts_with_the_four_presets(world_size):
    """Adding hybrids must not disturb the historical prefix or its order."""
    assert _default_parallel_axis_values(world_size)[:4] == ("DEP", "TEP", "DTP", "TTP")


@pytest.mark.parametrize("world_size", [4, 6, 8, 16, 32, 64])
def test_default_axis_hybrids_are_resolvable(world_size):
    """Every generated name must resolve to a legal grid for its world size."""
    for mode in default_hybrid_parallel_modes(world_size):
        moe_ep, moe_tp, _enable_dp = _resolve_mapping_layout(_config(mode), world_size)
        assert moe_ep * moe_tp == world_size
        assert moe_ep > 1 and moe_tp > 1  # never degenerates to a preset


def test_default_axis_has_no_duplicates():
    for world_size in (1, 2, 4, 6, 8, 16, 32, 64):
        values = _default_parallel_axis_values(world_size)
        assert len(values) == len(set(values))


# --------------------------------------------------------------------------
# Regressions: world size / config-file plumbing into the default axis.
# --------------------------------------------------------------------------


def _args(**kwargs) -> argparse.Namespace:
    """Minimal Namespace for the search-axis resolution helpers."""
    base = {
        "search": ("parallel",),
        "world_size": None,
        "backend": ("TRTLLM",),
        "parallel_mode": ("DEP",),
        "comm_method": ("AUTO",),
        "_cli_provided": set(),
        "_config_search_axes": {},
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_external_world_size_reaches_the_default_axis():
    """The detected world size must reach the default axis.

    ``--world_size`` is optional under external mpirun; worker.main pins the
    detected size onto args before the context is built. If that write-back is
    lost, the axis silently resolves against world_size=1 and drops hybrids.
    """
    resolved = _axis_values_from_args(
        _args(world_size=8),
        cli_dest="parallel_mode",
        cli_flag_name="--parallel_mode",
        config_key="parallel_mode",
        full_set=_default_parallel_axis_values(8),
    )
    assert resolved == _default_parallel_axis_values(8)
    assert "DTP2EP4" in resolved and "DTP4EP2" in resolved


def test_world_size_none_would_lose_the_hybrids():
    """Guards the failure mode above: world_size=1 legitimately has no split."""
    assert _default_parallel_axis_values(1) == ("DEP", "TEP", "DTP", "TTP")


@pytest.mark.parametrize("modes", [("CUSTOM",), ("DEP",), ("DTP2EP4",)])
def test_single_value_config_axis_is_not_replaced_by_the_default_set(modes):
    """A single-value config axis must survive axis resolution.

    A single-value ``search.parallel_mode`` in the JSON config is an explicit
    search set, not a scalar default, so it must not fall through to the
    world-size default expansion.
    """
    resolved = _axis_values_from_args(
        _args(world_size=8, _config_search_axes={"parallel_mode": modes}),
        cli_dest="parallel_mode",
        cli_flag_name="--parallel_mode",
        config_key="parallel_mode",
        full_set=_default_parallel_axis_values(8),
    )
    assert resolved == modes


def test_cli_parallel_mode_still_wins_over_config_axis():
    resolved = _axis_values_from_args(
        _args(
            world_size=8,
            parallel_mode=("TEP", "DEP"),
            _cli_provided={"parallel_mode"},
            _config_search_axes={"parallel_mode": ("CUSTOM",)},
        ),
        cli_dest="parallel_mode",
        cli_flag_name="--parallel_mode",
        config_key="parallel_mode",
        full_set=_default_parallel_axis_values(8),
    )
    assert resolved == ("TEP", "DEP")
