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
"""CPU tests for where the MegaMoE local backend places the shared expert (``impl_delta``, ``local_plan``)."""

import os

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    TrtllmCutedslMegaMoeLocalNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

pytestmark = pytest.mark.cpu_only

DISABLE = plan.SHARED_SERIAL_DISABLE_ENV
ROWS = plan.SHARED_SERIAL_MAX_ROWS_ENV
MIN_EP = plan.SHARED_SERIAL_MIN_EP_SIZE_ENV
PIN_ABOVE = plan.SHARED_CORUN_PIN_ABOVE_ROWS_ENV
PIN_OFF = plan.SHARED_CORUN_PIN_DISABLE_ENV
PIN_KERNEL = plan.SHARED_CORUN_PIN_KERNEL_ENV
BAND = plan.SHARED_CORUN_BAND_ENV
BAND_OFF = plan.SHARED_CORUN_BAND_DISABLE_ENV
LADDER = (32, 64, 128)


@pytest.fixture(autouse=True)
def _no_scope_env(monkeypatch):
    for name in (DISABLE, ROWS, MIN_EP, PIN_ABOVE, PIN_OFF, PIN_KERNEL, BAND, BAND_OFF):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("knob", [ROWS, MIN_EP, BAND])
def test_an_invalid_scope_knob_fails_when_the_backend_reads_it(monkeypatch, knob):
    monkeypatch.setenv(knob, "many")
    with pytest.raises(ValueError, match=knob):
        _backend()._init_shared_expert_plan()


@pytest.mark.parametrize(
    "bucket, environ, serial",
    [
        (32, {}, True),
        (64, {}, True),
        (128, {}, True),
        (192, {}, True),
        (1024, {}, True),
        (4096, {}, False),
        (None, {}, True),
        (32, {ROWS: "1"}, False),
        (64, {ROWS: "1"}, False),
        (32, {ROWS: "32"}, True),
        (64, {ROWS: "32"}, False),
        (128, {ROWS: "128"}, True),
        (64, {DISABLE: "1"}, False),
        (32, {MIN_EP: "16"}, False),
        (64, {BAND: "32:64"}, False),
        (32, {BAND: "32:64"}, True),
        (128, {BAND: "32:64"}, True),
        (None, {BAND: "32:64"}, True),
        (64, {BAND: "64:128"}, True),
        (128, {BAND: "64:128"}, False),
        (64, {BAND: "32:64", MIN_EP: "16"}, False),
    ],
)
def test_serial_scope_on_the_launch_bucket(bucket, environ, serial):
    assert plan.shared_expert_serial(bucket, 8, environ) is serial


@pytest.mark.parametrize("value, band", [("32:64", (32, 64)), ("0:64", (0, 64))])
def test_the_corun_band_parses_lo_below_hi(value, band):
    assert plan.shared_corun_band({}) is None
    assert plan.shared_corun_band({BAND: value}) == band


@pytest.mark.parametrize("bad", ["64", "64:32", "32:32", "-1:64", "a:b", "32:64:96", ""])
def test_the_corun_band_must_be_lo_below_hi(bad):
    with pytest.raises(ValueError, match=BAND):
        plan.shared_corun_band({BAND: bad})


def test_the_corun_band_off_switch_wins_over_a_band():
    for value in ("32:64", "64:32"):
        assert plan.shared_corun_band({BAND: value, BAND_OFF: "1"}) is None
    assert plan.shared_corun_band({BAND: "32:64", BAND_OFF: "0"}) == (32, 64)
    assert plan.shared_expert_serial(64, 8, {BAND: "32:64", BAND_OFF: "1"})
    assert plan.lever_environ({BAND_OFF: "1", "TRTLLM_M31_UNRELATED": "1"}) == {BAND_OFF: "1"}


def test_a_backend_takes_a_default_band_only_where_its_knobs_set_none(monkeypatch):
    backend_cls = TrtllmCutedslMegaMoeLocalNvfp4Impl
    backend = _backend(backend_cls=backend_cls)
    assert backend.shared_corun_band() is None
    assert backend.default_shared_corun_band((32, 64))
    assert backend.shared_corun_band() == (32, 64)
    assert [backend.shared_expert_serial(rows) for rows in (9, 64, 96, 128)] == [
        True,
        False,
        True,
        True,
    ]
    assert not backend.default_shared_corun_band((0, 128))
    assert backend.shared_corun_band() == (32, 64)
    monkeypatch.setenv(BAND, "64:128")
    explicit = _backend(backend_cls=backend_cls)
    assert not explicit.default_shared_corun_band((32, 64))
    assert explicit.shared_corun_band() == (64, 128)
    monkeypatch.delenv(BAND)
    monkeypatch.setenv(BAND_OFF, "1")
    disabled = _backend(backend_cls=backend_cls)
    assert not disabled.default_shared_corun_band((32, 64))
    assert disabled.shared_corun_band() is None
    assert all(disabled.shared_expert_serial(rows) for rows in (9, 64, 96, 128))


def test_a_default_band_is_checked_as_the_knob_is():
    backend = _backend(backend_cls=TrtllmCutedslMegaMoeLocalNvfp4Impl)
    with pytest.raises(ValueError, match=BAND):
        backend.default_shared_corun_band((64, 32))
    assert backend.shared_corun_band() is None


def test_ep_size_knob():
    assert plan.shared_serial_min_ep_size({}) == plan.SHARED_SERIAL_MIN_EP_SIZE == 1
    for ep_size in (1, 8, 16):
        assert plan.shared_expert_serial(32, ep_size, {})
    assert not plan.shared_expert_serial(32, 8, {MIN_EP: "16"})
    assert plan.shared_expert_serial(32, 16, {MIN_EP: "16"})
    assert plan.shared_expert_serial(128, 16, {MIN_EP: "16"})
    assert not plan.shared_expert_serial(128, 16, {MIN_EP: "16", ROWS: "32"})


@pytest.mark.parametrize(
    "knob, parse",
    [
        (ROWS, plan.shared_serial_max_rows),
        (MIN_EP, plan.shared_serial_min_ep_size),
        (PIN_ABOVE, plan.shared_corun_pin_above_rows),
    ],
)
@pytest.mark.parametrize("bad", ["-1", "many", "6.4"])
def test_scope_knobs_must_be_non_negative_integers(knob, parse, bad):
    with pytest.raises(ValueError, match=knob):
        parse({knob: bad})


@pytest.mark.parametrize(
    "bucket, environ, pinned",
    [
        (32, {}, False),
        (64, {}, True),
        (128, {}, True),
        (192, {}, True),
        (None, {}, True),
        (64, {ROWS: "32"}, True),
        (32, {PIN_ABOVE: "0"}, True),
        (64, {PIN_ABOVE: "32"}, True),
        (128, {PIN_ABOVE: "128"}, False),
        (128, {PIN_OFF: "1"}, False),
        (None, {PIN_OFF: "1"}, False),
        (64, {PIN_KERNEL: "cutlass"}, False),
        (128, {PIN_KERNEL: "cutlass"}, True),
        (64, {PIN_KERNEL: "cutlass", PIN_ABOVE: "32"}, True),
    ],
)
def test_corun_pin_on_the_launch_bucket(bucket, environ, pinned):
    assert plan.shared_corun_pin(bucket, environ) is pinned


def test_the_pin_kernel_is_the_cute_dsl_gemm_unless_set_to_cutlass():
    assert plan.shared_corun_pin_kernel({}) == "cute_dsl"
    assert plan.shared_corun_pin_kernel({PIN_KERNEL: "cutlass"}) == "cutlass"
    assert plan.shared_corun_pin_kernel({PIN_KERNEL: " CUTLASS "}) == "cutlass"
    with pytest.raises(ValueError, match=PIN_KERNEL):
        plan.shared_corun_pin_kernel({PIN_KERNEL: "cublas"})


def _backend(ladder=LADDER, backend_cls=TrtllmCutedslMegaMoeLocalNvfp4Impl, ep_size=8):
    """A backend with the state construction gives what these decisions read: the bucket ladder,
    the expert-parallel size, and the knobs as they are now."""
    backend = backend_cls.__new__(backend_cls)
    backend._maxt_buckets = list(ladder)
    backend.tactic_autotune = False
    backend.ep_size = ep_size
    backend._lever_environ = plan.lever_environ(os.environ)
    return backend


@pytest.mark.parametrize(
    "max_rows, bucket",
    [(1, 32), (9, 32), (32, 32), (33, 64), (64, 64), (65, 128), (96, 128), (128, 128)],
)
def test_a_step_within_the_top_bucket_reports_its_launch_bucket(max_rows, bucket):
    assert _backend().megamoe_launch_bucket(max_rows) == bucket


def test_an_unknown_step_reports_the_full_bucket():
    assert _backend().megamoe_launch_bucket(None) == LADDER[-1]


@pytest.mark.parametrize("ladder", [LADDER, (32, 64, 128, 192, 256), (32, 64, 100)])
def test_a_chunked_step_reports_no_more_than_its_smallest_chunk_bucket(ladder):
    backend, top = _backend(ladder), ladder[-1]
    for max_rows in range(top + 1, 8 * top + 1):
        num_chunks = -(-max_rows // top)
        chunks = ConfigurableMoE.split_chunk(None, max_rows, num_chunks)
        assert max(chunks) <= top
        smallest = min(backend.megamoe_launch_bucket(rows) for rows in chunks)
        assert backend.megamoe_launch_bucket(max_rows) <= smallest, max_rows


@pytest.mark.parametrize(
    "flip",
    [((PIN_OFF, "1"),), ((PIN_ABOVE, "128"),), ((PIN_ABOVE, "32"),)],
    ids=["disabled", "above-128", "above-32"],
)
def test_the_corun_pin_keeps_the_knobs_the_backend_was_built_with(monkeypatch, flip):
    backend = _backend()
    for name, value in flip:
        monkeypatch.setenv(name, value)
    rebuilt = _backend()
    for rows in (9, 40, 64, 96, 128):
        bucket = backend.megamoe_launch_bucket(rows)
        assert backend.shared_corun_pin(rows) is plan.shared_corun_pin(bucket, {})
        assert rebuilt.shared_corun_pin(rows) is plan.shared_corun_pin(bucket, dict(flip))
