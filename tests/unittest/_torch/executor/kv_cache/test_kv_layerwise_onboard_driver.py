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
"""Python driver of layer-wise onboarding: which waits reach the core, and when."""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache import kv_layerwise_onboard as lw


class FakeImpl:
    def __init__(self, flushes: bool = True):
        self.flushes = flushes
        self.calls = []
        self.has_onboard_events = False

    def set_layerwise_onboard(self, enable, layer_groups):
        self.calls.append(("set", enable, layer_groups))

    def flush_onboard(self, stream):
        self.calls.append(("flush", stream))
        self.has_onboard_events = self.flushes
        return self.flushes

    def wait_layer_onboard(self, layer_id, stream):
        self.calls.append(("layer", layer_id, stream))

    def wait_all_onboard(self, stream):
        self.calls.append(("all", stream))
        self.has_onboard_events = False


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(lw, "_consumer_registered", False)
    monkeypatch.setattr(lw, "_armed", lw.weakref.WeakSet())
    monkeypatch.setattr(lw.torch.cuda, "is_current_stream_capturing", lambda: False)


def _driver(impl, last=3, offsets=None):
    offsets = offsets if offsets is not None else {i: i for i in range(4)}
    return lw.LayerwiseOnboard(impl, offsets.get, lambda: last)


def test_off_by_default_in_the_kv_cache_config():
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    assert KvCacheConfig().layerwise_host_onboard is False
    assert KvCacheConfig(layerwise_host_onboard=True).layerwise_host_onboard is True


def test_one_event_per_layer():
    impl = FakeImpl()
    _driver(impl)
    assert impl.calls[0] == ("set", True, lw.LAYERWISE_ONBOARD_GROUPS)


def test_without_a_consumer_every_flush_waits_for_everything():
    impl = FakeImpl()
    driver = _driver(impl)
    driver.flush(7)
    assert impl.calls[-2:] == [("flush", 7), ("all", 7)]
    assert not driver.armed


def test_layers_wait_individually_and_the_last_target_layer_retires_all():
    lw.register_layerwise_consumer()
    target, draft = FakeImpl(), FakeImpl()
    target_driver = _driver(target, last=13, offsets={10: 0, 11: 1, 12: 2, 13: 3})
    draft_driver = _driver(draft, last=None, offsets={0: 0})
    target_driver.flush(5)
    draft_driver.flush(5)
    assert target_driver.armed and draft_driver.armed
    for layer in (10, 11, 12):
        target_driver.wait_layer(layer, 5)
    assert [c for c in target.calls if c[0] == "layer"] == [("layer", i, 5) for i in range(3)]
    assert ("all", 5) not in target.calls
    target_driver.wait_layer(13, 5)
    assert target.calls[-2:] == [("layer", 3, 5), ("all", 5)]
    assert ("all", 5) in draft.calls, "the draft manager's copies retire at the last target layer"
    assert not target_driver.armed and not draft_driver.armed


def test_nothing_flushed_leaves_the_driver_idle():
    lw.register_layerwise_consumer()
    impl = FakeImpl(flushes=False)
    driver = _driver(impl)
    driver.flush(1)
    assert not driver.armed
    driver.wait_layer(0, 1)
    assert [c for c in impl.calls if c[0] in ("layer", "all")] == []


def test_capture_skips_waits(monkeypatch):
    lw.register_layerwise_consumer()
    impl = FakeImpl()
    driver = _driver(impl)
    driver.flush(1)
    monkeypatch.setattr(lw.torch.cuda, "is_current_stream_capturing", lambda: True)
    driver.wait_layer(3, 1)
    assert [c for c in impl.calls if c[0] in ("layer", "all")] == []
