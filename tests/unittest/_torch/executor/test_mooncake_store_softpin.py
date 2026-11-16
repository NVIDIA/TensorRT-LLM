# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M3.1 soft-pin port (TRTLLM_MOONCAKE_STORE_SOFTPIN): pinned puts, fail-safes, worker arming.

CPU only: the store is the in-process fake from test_mooncake_store_connector.
"""

from types import SimpleNamespace

import pytest
from test_mooncake_store_connector import (  # noqa: F401 (fixtures)
    fake_store,
    make_layout,
    make_worker,
    store_config,
)

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import softpin
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.softpin import (
    SOFTPIN_ENV,
    SoftPinStore,
    maybe_softpin_store,
    replicate_config,
    softpin_settings,
)

pytestmark = pytest.mark.cpu_only


class _FakeMod:
    """Stands in for `mooncake.store`'s soft-pin surface."""

    class SoftPinAction:
        ENABLE = "enable"

    class ReplicateConfig:
        def __init__(self):
            self.soft_pin_action = None
            self.soft_pin_ttl_ms = 0
            self.group_ids = []


def _factory(ttl_ms, group_ids=None):
    return replicate_config(ttl_ms, group_ids, _mod=_FakeMod)


class _Inner:
    def __init__(self, results=None, typeerror=False):
        self.calls = []
        self.results = results
        self.typeerror = typeerror
        self.closed = False

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *rest):
        if rest and self.typeerror:
            raise TypeError("takes 3 positional arguments")
        self.calls.append((list(keys), rest))
        if self.results is not None:
            return self.results(keys, rest)
        return [8] * len(keys)

    def batch_is_exist(self, keys):
        return [0] * len(keys)

    def close(self):
        self.closed = True


class _BindingLikeInner(_Inner):
    """Like the pybind `MooncakeDistributedStore`: setting an attribute the class does not define
    raises AttributeError, where `_Inner` (and the connector tests' fake store) accept any name."""

    _OWN = frozenset({"calls", "results", "typeerror", "closed"})

    def __setattr__(self, key, value):
        if key not in self._OWN:
            raise AttributeError(
                f"'mooncake._store.MooncakeDistributedStore' object has no attribute '{key}'"
            )
        object.__setattr__(self, key, value)


def test_settings_are_off_by_default():
    assert softpin_settings({}) is None
    assert softpin_settings({SOFTPIN_ENV: "0"}) is None
    assert softpin_settings({SOFTPIN_ENV: "1"}) == (True, softpin.DEFAULT_TTL_MS, False)
    assert softpin_settings(
        {SOFTPIN_ENV: "1", softpin.TTL_ENV: "5000", softpin.GROUP_BLOCKS_ENV: "1"}
    ) == (
        True,
        5000,
        True,
    )
    with pytest.raises(ValueError):
        softpin_settings({SOFTPIN_ENV: "1", softpin.TTL_ENV: "0"})


def test_off_or_non_saving_returns_the_same_handle():
    inner = _Inner()
    kw = dict(
        rank=0, role="both", namespace="ns", model_key="m", world_size=4, config_factory=_factory
    )
    assert maybe_softpin_store(inner, saves=True, environ={}, **kw) is inner
    assert maybe_softpin_store(inner, saves=False, environ={SOFTPIN_ENV: "1"}, **kw) is inner
    wrapped = maybe_softpin_store(inner, saves=True, environ={SOFTPIN_ENV: "1"}, **kw)
    assert isinstance(wrapped, SoftPinStore) and wrapped.pin


def test_three_argument_puts_get_the_pin_config_and_other_calls_pass_through():
    inner = _Inner()
    store = SoftPinStore(inner, 0, 1234, config_factory=_factory)
    assert store.batch_put_from_multi_buffers(["a", "b"], [[1], [2]], [[8], [8]]) == [8, 8]
    ((keys, rest),) = inner.calls
    assert keys == ["a", "b"] and len(rest) == 1
    config = rest[0]
    assert config.soft_pin_action == "enable" and config.soft_pin_ttl_ms == 1234
    # A caller's own config is left alone, and unknown attributes reach the handle.
    store.batch_put_from_multi_buffers(["c"], [[3]], [[8]], "theirs")
    assert inner.calls[-1][1] == ("theirs",)
    assert store.batch_is_exist(["x"]) == [0]
    store.close()
    assert inner.closed
    assert store.pinned_keys == 2 and store.keys == 3 and store.calls == 2
    assert "pinned_keys=2" in store.put_summary()


def test_group_blocks_put_every_page_of_a_block_in_one_group():
    inner = _Inner()
    group_of = softpin._group_from_key("ns", "model", 4)
    store = SoftPinStore(inner, 0, 10, group_of=group_of, config_factory=_factory)
    store.batch_put_from_multi_buffers(
        ["ns/model/w4r0/lg0/t32b9/abcd", "ns/model/w4r1/lg1/t32b7/abcd"], [[1], [2]], [[8], [8]]
    )
    config = inner.calls[0][1][0]
    assert config.group_ids == ["ns/model/w4/abcd", "ns/model/w4/abcd"]


def test_a_binding_without_soft_pin_leaves_puts_unpinned():
    def missing(ttl_ms, group_ids=None):
        return replicate_config(ttl_ms, group_ids, _mod=SimpleNamespace())

    inner = _Inner()
    store = SoftPinStore(inner, 0, 10, config_factory=missing)
    assert not store.pin
    store.batch_put_from_multi_buffers(["a"], [[1]], [[8]])
    assert inner.calls == [(["a"], ())]


def test_a_binding_that_rejects_the_config_argument_unpins_for_good():
    inner = _Inner(typeerror=True)
    store = SoftPinStore(inner, 0, 10, config_factory=_factory)
    assert store.batch_put_from_multi_buffers(["a"], [[1]], [[8]]) == [8]
    assert not store.pin
    store.batch_put_from_multi_buffers(["b"], [[1]], [[8]])
    assert [rest for _, rest in inner.calls] == [(), ()]


def test_refused_pinned_puts_fall_back_to_plain_once_before_any_landed():
    inner = _Inner(results=lambda keys, rest: [-1] * len(keys) if rest else [8] * len(keys))
    store = SoftPinStore(inner, 0, 10, config_factory=_factory)
    assert store.batch_put_from_multi_buffers(["a"], [[1]], [[8]]) == [8]
    assert not store.pin
    assert [rest != () for _, rest in inner.calls] == [True, False]


def test_capacity_refusals_before_any_landed_keep_the_pin():
    """3685966: the pool was empty until the first donor mounted; a -200 is not a verdict on soft pin."""
    mounted = {"yes": False}

    def results(keys, rest):
        # Every put is refused for want of space until a segment mounts; then pinned and plain puts both land.
        return [8] * len(keys) if mounted["yes"] else [-200] * len(keys)

    inner = _Inner(results=results)
    store = SoftPinStore(inner, 2, 10, config_factory=_factory)
    assert store.batch_put_from_multi_buffers(["a", "b"], [[1], [2]], [[8], [8]]) == [-200, -200]
    # No plain repeat: under the hook a segment mounting between the two calls unpinned this rank for good.
    assert [rest != () for _, rest in inner.calls] == [True]
    assert store.pin and store.capacity_refusals == 1 and store.failed_keys == 2
    mounted["yes"] = True
    assert store.batch_put_from_multi_buffers(["a"], [[1]], [[8]]) == [8]
    assert store.pin and store._confirmed and store.pinned_keys == 1
    assert all(rest != () for _, rest in inner.calls)
    assert "capacity_refusals=1" in store.put_summary() and "softpin=on" in store.put_summary()


def test_the_donor_mount_race_no_longer_unpins_one_rank():
    """The pinned put is refused -200, a segment mounts, the plain repeat (old behaviour) would land."""
    state = {"calls": 0}

    def results(keys, rest):
        state["calls"] += 1
        return [-200] * len(keys) if state["calls"] == 1 else [8] * len(keys)

    inner = _Inner(results=results)
    store = SoftPinStore(inner, 2, 10, config_factory=_factory)
    assert store.batch_put_from_multi_buffers(["a"], [[1]], [[8]]) == [-200]
    assert store.pin
    assert store.batch_put_from_multi_buffers(["a"], [[1]], [[8]]) == [8]
    assert store.pin and store.pinned_keys == 1
    assert [rest != () for _, rest in inner.calls] == [True, True]


def test_a_non_capacity_refusal_still_unpins_and_says_so_at_error_level(monkeypatch, capfd):
    seen = []
    monkeypatch.setattr(softpin.logger, "error", lambda msg: seen.append(("error", msg)))
    monkeypatch.setattr(softpin.logger, "warning", lambda msg: seen.append(("warning", msg)))
    inner = _Inner(
        results=lambda keys, rest: [-200, -600][: len(keys)] if rest else [8] * len(keys)
    )
    store = SoftPinStore(inner, 1, 10, config_factory=_factory)
    assert store.batch_put_from_multi_buffers(["a", "b"], [[1], [2]], [[8], [8]]) == [8, 8]
    assert not store.pin
    assert [level for level, msg in seen if "M31_SOFTPIN_DISABLED" in msg] == ["error"]
    assert "M31_SOFTPIN_DISABLED rank=1 reason=refused" in capfd.readouterr().err


def test_refusals_after_a_pinned_put_landed_are_returned_as_is():
    state = {"n": 0}

    def results(keys, rest):
        state["n"] += 1
        return [8] * len(keys) if state["n"] == 1 else [-1] * len(keys)

    inner = _Inner(results=results)
    store = SoftPinStore(inner, 0, 10, config_factory=_factory)
    assert store.batch_put_from_multi_buffers(["a"], [[1]], [[8]]) == [8]
    assert store.batch_put_from_multi_buffers(["b"], [[1]], [[8]]) == [-1]
    assert store.pin and len(inner.calls) == 2 and store.failed_keys == 1


def test_the_proxy_keeps_its_counters_over_a_binding_that_rejects_new_attributes():
    """Real-pool gate 3688430: the proxy forwards attribute sets it does not keep to the store
    handle, and the pybind store refuses a name it lacks, so `capacity_refusals = 0` in the
    constructor raised AttributeError on a real store while the permissive fakes passed."""
    inner = _BindingLikeInner(
        results=lambda keys, rest: [-200] * len(keys) if rest else [8] * len(keys)
    )
    with pytest.raises(AttributeError):
        inner.capacity_refusals = 0
    store = SoftPinStore(inner, 2, 10, config_factory=_factory)
    assert store.pin and "capacity_refusals=0" in store.put_summary()
    assert store.batch_put_from_multi_buffers(["a"], [[1]], [[8]]) == [-200]
    assert store.pin and store.capacity_refusals == 1 and store.failed_keys == 1
    assert "capacity_refusals=1" in store.put_summary()
    # Unpinning mid-run sets `pin` on the proxy too; names the proxy does not keep still reach the handle.
    inner.typeerror = True
    assert store.batch_put_from_multi_buffers(["b"], [[1]], [[8]]) == [8]
    assert not store.pin and "softpin=off" in store.put_summary()
    store.closed = "via proxy"
    assert inner.closed == "via proxy"
    # The worker's entry point builds the proxy over the binding the same way.
    wrapped = maybe_softpin_store(
        _BindingLikeInner(),
        rank=0,
        role="ctx",
        saves=True,
        namespace="ns",
        model_key="m",
        world_size=4,
        config_factory=_factory,
        environ={SOFTPIN_ENV: "1"},
    )
    assert isinstance(wrapped, SoftPinStore) and wrapped.pin
    assert "capacity_refusals=0" in wrapped.put_summary()


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_arms_soft_pin_only_when_switched_on(
    store_config,  # noqa: F811 (pytest fixture)
    fake_store,  # noqa: F811 (pytest fixture)
    monkeypatch,
    capfd,
    enabled,
):
    if enabled:
        monkeypatch.setenv(SOFTPIN_ENV, "1")
        # No mooncake binding on a CPU test host: the soft-pin surface is faked.
        monkeypatch.setattr(softpin, "replicate_config", _factory)
    else:
        monkeypatch.delenv(SOFTPIN_ENV, raising=False)
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        hashes = [bytes([i]) * 16 for i in range(3)]
        transfers = [RequestTransfers(1, [PageTransfer(h, 0, i) for i, h in enumerate(hashes)])]
        assert worker._put(transfers)[:2] == (3, 0)
        if enabled:
            assert isinstance(worker._store, SoftPinStore) and worker._store._inner is fake_store
            assert worker._store.pin and worker._store.pinned_keys == 3
        else:
            assert worker._store is fake_store
    err = capfd.readouterr().err
    assert ("M31_SOFTPIN_ARMED rank=" in err) == enabled
    if enabled:
        assert "softpin=on" in err and "src=intree" in err
