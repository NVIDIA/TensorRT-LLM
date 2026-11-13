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
"""Bounded put retries of the Mooncake store connector's saves."""

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import save_retry

pytestmark = pytest.mark.cpu_only


class _Store:
    """Returns scripted put and probe answers; an exception in the script is raised."""

    def __init__(self, results, probes=()):
        self.results = iter(results)
        self.probes = iter(probes)
        self.puts = []
        self.put_extra = []
        self.probe_calls = []

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *extra):
        self.puts.append((list(keys), list(addresses), list(sizes)))
        self.put_extra.append(extra)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result

    def batch_is_exist(self, keys):
        self.probe_calls.append(list(keys))
        result = next(self.probes, [0] * len(keys))
        if isinstance(result, Exception):
            raise result
        return result


def _put(store, clock=lambda: 0.0, **kwargs):
    return save_retry.put_with_retries(
        store, ["a", "b"], [[11], [22]], [[4], [4]], clock=clock, **kwargs
    )


def test_a_stored_batch_is_one_put_without_a_probe():
    store = _Store([[0, 0]])
    result = _put(store)
    assert (result.put_attempts, result.failed_indices) == (1, ())
    assert len(store.puts) == 1
    assert store.probe_calls == []


def test_only_the_refused_key_is_retried():
    store = _Store([[0, -200], [0]])
    result = _put(store)
    assert [keys for keys, _, _ in store.puts] == [["a", "b"], ["b"]]
    assert store.puts[1][1:] == ([[22]], [[4]])
    assert (result.recovered_keys, result.retried_keys) == (1, 1)


def test_a_probe_finds_a_put_that_raised_but_landed():
    store = _Store([RuntimeError("transport")], probes=[[1, 1]])
    result = _put(store)
    assert (len(store.puts), result.recovered_by_probe, result.failed_indices) == (1, 2, ())


def test_retries_are_bounded():
    store = _Store([[-200, -200]] * save_retry.MAX_PUT_ATTEMPTS)
    result = _put(store)
    assert len(store.puts) == save_retry.MAX_PUT_ATTEMPTS
    assert result.failed_indices == (0, 1)
    assert result.failure_codes == (-200,)


def test_raising_puts_are_retried_and_bounded():
    store = _Store([RuntimeError(str(index)) for index in range(save_retry.MAX_PUT_ATTEMPTS)])
    result = _put(store)
    assert result.exception_attempts == len(store.puts) == save_retry.MAX_PUT_ATTEMPTS


@pytest.mark.parametrize(
    "first, retried",
    [
        ([], ["a", "b"]),
        ([0], ["b"]),
        ([True, 0.0], ["a", "b"]),
        ([0, 0, 0], ["a", "b"]),
        (None, ["a", "b"]),
        (["0", 0], ["a"]),
    ],
)
def test_a_missing_or_malformed_status_is_not_taken_as_stored(first, retried):
    store = _Store([first, [0] * len(retried)])
    result = _put(store)
    assert store.puts[1][0] == retried
    assert result.failed_indices == ()


def test_no_retry_starts_after_the_window():
    times = iter([0.0, save_retry.RETRY_START_WINDOW_SECONDS * 2])
    store = _Store([[-200, -200]])
    result = _put(store, clock=lambda: next(times))
    assert (result.put_attempts, result.failed_indices) == (1, (0, 1))


def test_a_slow_probe_stops_the_retry():
    times = iter([0.0, 0.0, save_retry.RETRY_START_WINDOW_SECONDS * 2])
    store = _Store([[-200, -200]])
    result = _put(store, clock=lambda: next(times))
    assert result.put_attempts == 1


def test_a_failed_probe_keeps_the_keys_unresolved():
    store = _Store([[-200, -200], [0, 0]], probes=[RuntimeError("probe")])
    result = _put(store)
    assert result.failed_indices == ()
    assert result.exception_attempts == 1


def test_the_caller_buffers_are_passed_through_untouched():
    addresses = [[11], [22]]
    sizes = [[4], [4]]
    store = _Store([[-200, 0], [0]])
    save_retry.put_with_retries(store, ["a", "b"], addresses, sizes, clock=lambda: 0.0)
    assert store.puts[1][1][0] is addresses[0]
    assert addresses == [[11], [22]]
    assert sizes == [[4], [4]]


def test_each_put_gets_the_config_for_the_keys_it_carries():
    store = _Store([[-200, 0], [0]])
    _put(store, config_for=lambda indices: ("config", tuple(indices)))
    assert store.put_extra == [(("config", (0, 1)),), (("config", (0,)),)]


def test_no_config_keeps_the_three_argument_put():
    store = _Store([[0, 0]])
    _put(store, config_for=lambda indices: None)
    assert store.put_extra == [()]


def test_mismatched_lengths_call_no_store():
    store = _Store([])
    with pytest.raises(ValueError):
        save_retry.put_with_retries(store, ["a"], [], [])
    assert store.puts == []


@pytest.mark.parametrize("value, enabled", [(None, True), ("0", True), ("1", False)])
def test_the_retry_is_on_unless_disabled(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv(save_retry.DISABLE_ENV, raising=False)
    else:
        monkeypatch.setenv(save_retry.DISABLE_ENV, value)
    assert save_retry.save_retry_enabled() is enabled
