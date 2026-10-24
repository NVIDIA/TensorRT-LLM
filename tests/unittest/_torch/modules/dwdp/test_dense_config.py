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
"""CPU tests of the dense DWDP switches, how their values are parsed, and the launch
gate's uGPU and parallel-layout requirements.

* Booleans: ``1/true/yes/on`` and ``0/false/no/off`` in any case and with blanks around them;
  unset or blank keeps the default; any other value logs a warning and keeps the default, so the
  default-off flags stay off and ``TRTLLM_DWDP_DENSE_REQUIRED`` fails safe (stays on).
* Integers that do not parse, or are negative, log a warning and keep the default.
* ``TRTLLM_DWDP_DENSE_SEGMENTS`` is a comma-separated, case-insensitive list that must include
  ``attn``; an invalid list raises.  ``dense_dwdp_requested`` reads ``TRTLLM_DWDP_DENSE`` only,
  so a bad list cannot break a run with dense DWDP off.
* ``TRTLLM_DWDP_DENSE_DEBUG``: 1 fills the slots with NaN at commit; 2 also runs the slot
  watchdog.
* The launch gate rejects ``enable_ugpu`` (``Linear`` and ``MLA`` would keep ``_ugpu_*`` shard
  copies of their weights replicated) and ``pp_size`` or ``cp_size`` above 1 (a rank must hold
  every layer, unsharded), and names the value it found.  ``test_dwdp_creator.py`` checks every
  gate requirement, these included, and the order the gate checks them in.

The switch tests pass an ``environ`` dict and do not touch the process environment; the gate reads
it, so the gate tests remove every ``TRTLLM_DWDP_DENSE*`` variable from it first.
"""

import math
import os
import re

import pytest

from tensorrt_llm._torch.modules.dwdp import vmm
from tensorrt_llm._torch.modules.dwdp.dense import config as dense_config
from tensorrt_llm._torch.modules.dwdp.dense.config import (
    DENSE_DEBUG_ENV,
    DENSE_ENV,
    DENSE_MIN_BYTES_ENV,
    DENSE_REQUIRED_ENV,
    DENSE_SEGMENTS_ENV,
    DENSE_STORE_CHUNK_MB_ENV,
    DENSE_STRICT_ENV,
    DenseConfig,
    dense_dwdp_requested,
    parse_segments,
)
from tensorrt_llm._torch.modules.dwdp.dense.gate import validate_dense_gate
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    DEBUG_FILL_BYTE,
    DEFAULT_MIN_BYTES,
    SEGMENT_ORDER,
    SegmentKind,
)
from tensorrt_llm.llmapi.llm_args import DwdpConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE

TRUE_VALUES = ["1", "true", "TRUE", "Yes", "on", " On "]
FALSE_VALUES = ["0", "false", "No", "OFF", " off "]
UNSET_VALUES = [None, "", "   "]
JUNK_VALUES = ["2", "enable", "y", "onn", "-1"]


class _Warnings:
    """Stands in for the config module's ``logger``; keeps the warnings."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    capture = _Warnings()
    monkeypatch.setattr(dense_config, "logger", capture)
    return capture.messages


def _env(name: str, value: str | None) -> dict[str, str]:
    """An environment with ``name`` set to ``value``, or empty when ``value`` is None."""
    return {} if value is None else {name: value}


# ---------------------------------------------------------------------------
# Booleans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", TRUE_VALUES)
def test_dense_is_on_for_an_explicit_true_value(warnings, value):
    assert dense_dwdp_requested(_env(DENSE_ENV, value)) is True
    assert DenseConfig.from_env(_env(DENSE_ENV, value)).enabled is True
    assert warnings == []


@pytest.mark.parametrize("value", FALSE_VALUES + UNSET_VALUES)
def test_dense_is_off_for_a_false_or_unset_value(warnings, value):
    assert dense_dwdp_requested(_env(DENSE_ENV, value)) is False
    assert warnings == []


@pytest.mark.parametrize("value", JUNK_VALUES)
def test_any_other_value_keeps_dense_off_with_a_warning(warnings, value):
    assert dense_dwdp_requested(_env(DENSE_ENV, value)) is False
    assert len(warnings) == 1
    assert f"{DENSE_ENV}={value!r} is not one of 1/true/yes/on or 0/false/no/off" in warnings[0]
    assert warnings[0].endswith("using 0")


def test_dense_dwdp_requested_reads_no_other_switch(warnings):
    environ = {
        DENSE_ENV: "1",
        DENSE_SEGMENTS_ENV: "bogus",
        DENSE_REQUIRED_ENV: "maybe",
        DENSE_MIN_BYTES_ENV: "1MB",
    }
    assert dense_dwdp_requested(environ) is True
    assert warnings == []
    with pytest.raises(ValueError, match=DENSE_SEGMENTS_ENV):
        DenseConfig.from_env(environ)


@pytest.mark.parametrize(
    "value, required",
    [(None, True), ("", True), ("1", True), ("on", True), ("0", False), (" False ", False)],
)
def test_required_defaults_on(warnings, value, required):
    assert DenseConfig.from_env(_env(DENSE_REQUIRED_ENV, value)).required is required
    assert warnings == []


@pytest.mark.parametrize("value", JUNK_VALUES)
def test_an_unreadable_required_fails_safe(warnings, value):
    assert DenseConfig.from_env(_env(DENSE_REQUIRED_ENV, value)).required is True
    assert len(warnings) == 1 and warnings[0].endswith("using 1")


@pytest.mark.parametrize(
    "value, strict, warned",
    [(None, False, False), ("yes", True, False), ("0", False, False), ("maybe", False, True)],
)
def test_strict_is_off_unless_explicitly_on(warnings, value, strict, warned):
    assert DenseConfig.from_env(_env(DENSE_STRICT_ENV, value)).strict is strict
    assert len(warnings) == int(warned)


# ---------------------------------------------------------------------------
# Integers
# ---------------------------------------------------------------------------

INTEGER_SWITCHES = [
    pytest.param(DENSE_MIN_BYTES_ENV, "min_bytes", DEFAULT_MIN_BYTES, id="min_bytes"),
    pytest.param(DENSE_STORE_CHUNK_MB_ENV, "store_chunk_mb", 0, id="store_chunk_mb"),
    pytest.param(DENSE_DEBUG_ENV, "debug", 0, id="debug"),
]


@pytest.mark.parametrize("name, field, default", INTEGER_SWITCHES)
@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("0", 0), (" 4096 ", 4096)])
def test_integers_parse_or_take_the_default(warnings, name, field, default, value, expected):
    config = DenseConfig.from_env(_env(name, value))
    assert getattr(config, field) == (default if expected is None else expected)
    assert warnings == []


@pytest.mark.parametrize("name, field, default", INTEGER_SWITCHES)
@pytest.mark.parametrize("value", ["1MB", "-5", "1.5", "0x10", "many"])
def test_integers_that_do_not_parse_or_are_negative_keep_the_default(
    warnings, name, field, default, value
):
    assert getattr(DenseConfig.from_env(_env(name, value)), field) == default
    assert len(warnings) == 1
    assert f"{name}={value!r} is not a non-negative integer; using {default}" in warnings[0]


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, segments",
    [
        (None, (ATTN, MOE)),
        ("", (ATTN, MOE)),
        ("attn", (ATTN,)),
        (" ATTN , attn ", (ATTN,)),
        ("moe,attn", (ATTN, MOE)),
        ("Attn,MoE,", (ATTN, MOE)),
    ],
)
def test_segment_lists_are_case_insensitive_and_canonical(warnings, value, segments):
    config = DenseConfig.from_env(_env(DENSE_SEGMENTS_ENV, value))
    assert config.segments == segments
    assert config.has_moe is (MOE in segments)
    assert warnings == []


@pytest.mark.parametrize("value", ["moe", "atn", "attention", "attn,mlp", "attn;moe", ","])
def test_an_invalid_segment_list_raises(value):
    with pytest.raises(ValueError, match=f"{DENSE_SEGMENTS_ENV}={value!r} must list attn"):
        DenseConfig.from_env(_env(DENSE_SEGMENTS_ENV, value))
    with pytest.raises(ValueError, match="must list attn"):
        parse_segments(value)


def test_the_constructor_normalizes_and_checks_its_fields():
    assert DenseConfig(segments=["moe", "attn"]).segments == SEGMENT_ORDER
    assert DenseConfig(segments=("attn",)).segments == (ATTN,)
    with pytest.raises(ValueError, match="attention segment"):
        DenseConfig(segments=[MOE])
    with pytest.raises(ValueError, match="unknown segment"):
        DenseConfig(segments=["attn", "mlp"])
    for field in ("min_bytes", "store_chunk_mb", "debug"):
        with pytest.raises(ValueError, match=f"DenseConfig.{field} must be non-negative"):
            DenseConfig(**{field: -1})


# ---------------------------------------------------------------------------
# Derived settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "debug, fill_byte, watchdog",
    [(0, 0, False), (1, DEBUG_FILL_BYTE, False), (2, DEBUG_FILL_BYTE, True), (3, 0xFF, True)],
)
def test_debug_levels(debug, fill_byte, watchdog):
    config = DenseConfig.from_env({DENSE_DEBUG_ENV: str(debug)})
    assert config.slot_fill_byte == fill_byte
    assert config.slot_watchdog is watchdog


def test_the_canary_profile_and_the_layout_switches():
    """The attention-only canary's switches, and what the layout digest depends on."""
    config = DenseConfig.from_env(
        {
            DENSE_ENV: "1",
            DENSE_REQUIRED_ENV: "1",
            DENSE_SEGMENTS_ENV: "attn",
            DENSE_DEBUG_ENV: "1",
            DENSE_STORE_CHUNK_MB_ENV: "1024",
        }
    )
    assert (config.enabled, config.required, config.strict) == (True, True, False)
    assert config.segments == (ATTN,) and not config.has_moe
    assert (config.slot_fill_byte, config.slot_watchdog) == (DEBUG_FILL_BYTE, False)
    assert config.store_chunk_bytes == 1024 << 20
    switches = config.digest_switches(16)
    assert (switches.segments, switches.min_bytes, switches.store_chunk_mb) == (
        (ATTN,),
        DEFAULT_MIN_BYTES,
        1024,
    )
    assert switches.num_ranks == 16
    assert config.describe() == (
        f"segments=attn required=1 min_bytes={DEFAULT_MIN_BYTES} store_chunk_mb=1024 strict=0 "
        f"debug=1"
    )


# ---------------------------------------------------------------------------
# Launch gate: uGPU and the parallel layout
# ---------------------------------------------------------------------------

_FABRIC = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC


def _gate(monkeypatch, overrides: dict, sizes: dict) -> None:
    """Run the gate on a dense DWDP context-worker launch changed by the arguments.

    The launch is a ``TorchLlmArgs`` the gate accepts, with ``overrides`` applied without
    validation, on a ``Mapping`` with ``sizes``; the host shares FABRIC handles, and the process
    environment has no ``TRTLLM_DWDP_DENSE*`` variable.
    """
    for name in [name for name in os.environ if name.startswith(DENSE_ENV)]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(vmm, "peer_handle_type", lambda: _FABRIC)
    llm_args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        dwdp_config=DwdpConfig(
            dwdp_size=16, num_groups=1, num_experts_per_worker=56, num_prefetch_experts=56
        ),
    ).model_copy(update=overrides)
    mapping = Mapping(world_size=math.prod(sizes.values()), rank=0, **sizes)
    validate_dense_gate(llm_args, mapping)


def test_the_gate_accepts_a_launch_without_ugpu_on_whole_replicas(monkeypatch):
    _gate(monkeypatch, {}, {})


@pytest.mark.parametrize(
    "overrides, sizes, message",
    [
        ({"enable_ugpu": True}, {}, "requires enable_ugpu=False, but enable_ugpu=True"),
        ({}, {"pp_size": 2}, "requires pp_size=1, but pp_size=2"),
        ({}, {"cp_size": 2}, "requires cp_size=1, but cp_size=2"),
        ({}, {"pp_size": 2, "cp_size": 4}, "requires pp_size=1, but pp_size=2"),
    ],
    ids=["enable_ugpu", "pp_size", "cp_size", "pp_and_cp_size"],
)
def test_the_gate_rejects_ugpu_and_ranks_holding_part_of_a_replica(
    monkeypatch, overrides, sizes, message
):
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        _gate(monkeypatch, overrides, sizes)
    assert f"unset {DENSE_ENV}" in str(error.value)
