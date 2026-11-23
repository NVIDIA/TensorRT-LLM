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
"""CPU-only tests for the opt-in TRTLLM_KV_BOUNCE_MAX_FREE_FRACTION share of free memory.

The free-memory query and the transport constructor are replaced by fakes.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.disaggregation.native.bounce import config as bcfg
from tensorrt_llm._torch.disaggregation.native.bounce import impl as btr

_ENV = "TRTLLM_KV_BOUNCE_MAX_FREE_FRACTION"
_DIR_ENV = "TRTLLM_KV_BOUNCE_DIRECTION"
_MIB = 1024 * 1024


@pytest.mark.parametrize(
    "raw, expected",
    [(None, 0.5), ("", 0.5), ("  ", 0.5), ("0.55", 0.55), (" 0.6 ", 0.6), ("0.9", 0.9)],
)
def test_max_free_fraction_from_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    assert bcfg.max_free_fraction_from_env() == expected


def test_unset_is_the_legacy_default(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert bcfg.max_free_fraction_from_env() == bcfg._HEADROOM_FRACTION == 0.5


@pytest.mark.parametrize("raw", ["0", "-0.1", "0.91", "1", "nan", "inf", "half", "0,55"])
def test_max_free_fraction_rejects_bad_value(monkeypatch, raw):
    monkeypatch.setenv(_ENV, raw)
    with pytest.raises(ValueError, match=_ENV):
        bcfg.max_free_fraction_from_env()


def test_larger_share_lifts_the_clamp():
    # 11,740 MiB free at setup: half of it cannot hold one 6,144 MiB receive region, 0.55 can.
    chunk = 32 * _MIB
    free = 11_740 * _MIB
    want = 6_144 * _MIB
    legacy = bcfg.fit_within_free(want, free_bytes=free, chunk_bytes=chunk, num_regions=1)
    assert legacy == (free // 2 // chunk) * chunk < want
    wider = bcfg.fit_within_free(
        want, free_bytes=free, chunk_bytes=chunk, max_free_fraction=0.55, num_regions=1
    )
    assert wider == want


def _sized(monkeypatch, free, size_mb=6144):
    chunk = 32 * _MIB
    monkeypatch.setattr(btr, "cudart", SimpleNamespace(cudaMemGetInfo=lambda: (0, free, 4 * free)))
    monkeypatch.setattr(btr, "CUASSERT", lambda ret: ret[1:])
    captured = {}

    def _init(self, agent, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(btr.VmmBounceTransport, "__init__", _init)
    cfg = bcfg.config_from_size(size_mb)
    btr.VmmBounceTransport.from_config(None, cfg, device_id=0, block_bytes_per_group=[1024])
    return captured, chunk


@pytest.mark.parametrize("direction, regions", [("both", 2), ("recv", 1), ("send", 1)])
@pytest.mark.parametrize("free_mib", [11_740, 12_400, 20_000, 40_000])
def test_from_config_unset_matches_legacy_sizing(monkeypatch, direction, regions, free_mib):
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.setenv(_DIR_ENV, direction)
    free = free_mib * _MIB
    captured, chunk = _sized(monkeypatch, free)
    budget = (free // 2 // regions // chunk) * chunk
    assert captured["capacity_bytes"] == min(6144 * _MIB, budget)


def test_from_config_uses_the_share(monkeypatch):
    monkeypatch.setenv(_ENV, "0.55")
    monkeypatch.setenv(_DIR_ENV, "recv")
    free = 11_740 * _MIB
    captured, chunk = _sized(monkeypatch, free)
    assert captured["capacity_bytes"] == 6144 * _MIB  # not clamped at 0.55
    monkeypatch.setenv(_ENV, "0.3")
    captured, chunk = _sized(monkeypatch, free)
    assert captured["capacity_bytes"] == (int(free * 0.3) // chunk) * chunk


def test_create_bounce_bad_share_raises_instead_of_disabling(monkeypatch):
    monkeypatch.setenv(_ENV, "2")

    def _no_build(*args, **kwargs):
        raise AssertionError("from_config must not run for a bad share")

    monkeypatch.setattr(btr.VmmBounceTransport, "from_config", classmethod(_no_build))
    with pytest.raises(ValueError, match=_ENV):
        btr.create_bounce(SimpleNamespace(), SimpleNamespace(), device_id=0, page_table=None)


def test_create_bounce_without_config_ignores_share(monkeypatch):
    monkeypatch.setenv(_ENV, "2")
    nb = btr.create_bounce(None, None, device_id=0, page_table=None)
    assert isinstance(nb, btr.NoBounceTransport)
