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
"""The commit after a pass can swap a request's page for the radix tree's (TRTLLM_MOONCAKE_STORE_COMMIT_SWAP_GUARD).

KV cache manager V2 rebases a full block the tree already holds onto the tree's page at commit and frees the page the
request computed, so that slot can be handed out again. A save list built before the pass must not be read from the
freed slot: the worker holds it until the commit has run (`get_finished`) and resolves every page again.

CPU only: the store is the in-process fake from test_mooncake_store_connector; the save thread runs for real.
"""

import time

import pytest
import torch
from test_mooncake_store_connector import (  # noqa: F401 (fixtures)
    fake_store,
    make_layout,
    make_window_layout,
    make_worker,
    store_config,
    wait_for,
)

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import worker as worker_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.addressing import PageAddressing
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    RequestTransfers,
    WindowRetire,
)

pytestmark = pytest.mark.cpu_only

H1, H2 = b"\x11" * 16, b"\x12" * 16


class _Event:
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


@pytest.fixture
def fake_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: _Event())


def put_addresses(fake_store):
    out = {}
    for keys, addresses, _ in fake_store.put_calls:
        for key, address in zip(keys, addresses):
            out[key] = list(address)
    return out


def bind_saves(worker, pages, request_id=9, retires=()):
    worker.bind_connector_meta(
        MooncakeStoreMetadata(saves=[RequestTransfers(request_id, list(pages))], retires=list(retires))
    )


def finished(worker, request_id):
    done = []
    return wait_for(lambda: done.extend(worker.get_finished([], [])[0]) or request_id in done)


def test_a_save_list_waits_for_the_commit_and_follows_a_swapped_page(store_config, fake_store, fake_events):
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None  # device addresses go to the store as they are
        table = {9: [[0, 4, 5]]}
        worker.bind_local_page_indices(lambda request_id: table[request_id])
        bind_saves(worker, [PageTransfer(H1, 0, 4, 1), PageTransfer(H2, 0, 5, 2)])
        worker.wait_for_save(None)
        time.sleep(0.2)
        # Nothing is read before the commit has run, and the request's pages stay pinned meanwhile.
        assert fake_store.put_calls == []
        assert worker._outstanding_saves[9] == 1
        # The commit rebased block 1 onto the tree's page 7 and freed page 4, which may be reused at once.
        table[9] = [[0, 7, 5]]
        assert worker.get_finished([9], []) == ([], [])
        assert finished(worker, 9)
        addressing = PageAddressing(make_layout())
        got = put_addresses(fake_store)
        assert got[worker._namespaces[0].key(H1)] == list(addressing.buffers(0, 7)[0])
        assert got[worker._namespaces[0].key(H2)] == list(addressing.buffers(0, 5)[0])
        assert list(addressing.buffers(0, 4)[0]) not in got.values()
        assert (worker._swap_checked, worker._swap_retargeted, worker._swap_dropped) == (2, 1, 0)


def test_a_block_without_a_page_after_the_commit_is_not_saved(store_config, fake_store, fake_events):
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        table = {9: [[0, 4, 5]]}
        worker.bind_local_page_indices(lambda request_id: table[request_id])
        bind_saves(worker, [PageTransfer(H1, 0, 4, 1), PageTransfer(H2, 0, 5, 2)])
        worker.wait_for_save(None)
        table[9] = [[0, -1, 5]]
        worker.get_finished([9], [])
        assert finished(worker, 9)
        assert set(put_addresses(fake_store)) == {worker._namespaces[0].key(H2)}
        assert (worker._swap_retargeted, worker._swap_dropped) == (0, 1)


def test_with_the_guard_off_the_list_goes_out_at_once_from_the_listed_page(
    store_config, fake_store, fake_events, monkeypatch
):
    """The pre-fix hand-off (kept behind COMMIT_SWAP_GUARD=0 for A/B and repro runs): it reads the listed slot."""
    monkeypatch.setenv(worker_module.COMMIT_SWAP_GUARD_ENV, "0")
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        table = {9: [[0, 4]]}
        worker.bind_local_page_indices(lambda request_id: table[request_id])
        bind_saves(worker, [PageTransfer(H1, 0, 4, 1)])
        worker.wait_for_save(None)
        table[9] = [[0, 7]]
        assert wait_for(lambda: fake_store.put_calls)
        got = put_addresses(fake_store)
        assert got[worker._namespaces[0].key(H1)] == list(PageAddressing(make_layout()).buffers(0, 4)[0])
        assert worker._swap_checked == 0


def test_the_fallback_hand_off_at_the_next_pass_drops_a_moved_page(store_config, fake_store, fake_events):
    """Without a `get_finished` in between, the next scheduling pass may have run: a moved slot is not followed."""
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        table = {9: [[0, 4, 5]]}
        worker.bind_local_page_indices(lambda request_id: table[request_id])
        bind_saves(worker, [PageTransfer(H1, 0, 4, 1), PageTransfer(H2, 0, 5, 2)])
        worker.wait_for_save(None)
        table[9] = [[0, 7, 5]]
        worker.bind_connector_meta(MooncakeStoreMetadata())
        worker.wait_for_save(None)
        worker.get_finished([9], [])
        assert finished(worker, 9)
        assert set(put_addresses(fake_store)) == {worker._namespaces[0].key(H2)}
        assert (worker._swap_retargeted, worker._swap_dropped) == (0, 1)


def test_snapshotted_and_unordered_pages_are_kept_as_listed(store_config, fake_store):
    with make_worker(fake_store, layout=make_layout(num_groups=2)) as worker:
        saves = [
            RequestTransfers(
                9,
                [
                    PageTransfer(H1, 1, 4, 1),  # its window snapshot was taken on the pass's stream
                    PageTransfer(H2, 0, 3),  # no block ordinal
                ],
            )
        ]
        out = worker._recheck_after_commit(saves, {(1, 4): (0, 0xABC000, 4096)}, lambda rid: [[9, 9], [9, 9]], True)
        assert [(p.layer_group_id, p.page_index) for p in out[0].pages] == [(1, 4), (0, 3)]
        assert worker._swap_checked == 0


def test_an_emptied_entry_still_retires_its_request(store_config, fake_store, fake_events):
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        table = {9: [[0, 4]]}
        worker.bind_local_page_indices(lambda request_id: table[request_id])
        bind_saves(worker, [PageTransfer(H1, 0, 4, 1)])
        worker.wait_for_save(None)
        table[9] = [[]]  # the request's cache is gone by the commit
        worker.get_finished([9], [])
        assert finished(worker, 9)
        assert fake_store.put_calls == [] and worker._swap_dropped == 1


def test_window_retirements_stay_behind_their_saves(store_config, fake_store, fake_events, monkeypatch):
    monkeypatch.setenv(worker_module.WINDOW_KEEP_ENV, "2")
    with make_worker(fake_store, layout=make_window_layout(num_slots=32)) as worker:
        worker._save_staging = None
        worker.bind_local_page_indices(lambda request_id: [[0, 4], [0, 4]])
        retire = WindowRetire(9, 3, H2, 1, [H1])
        bind_saves(worker, [PageTransfer(H1, 0, 4, 1)], retires=[retire])
        worker.wait_for_save(None)
        time.sleep(0.2)
        assert worker._retire_stats.ends == 0 and fake_store.put_calls == []
        worker.get_finished([9], [])
        assert wait_for(lambda: worker._retire_stats.ends == 1)
        assert fake_store.put_calls


def test_shutdown_hands_over_what_is_held(store_config, fake_store, fake_events):
    with make_worker(fake_store, layout=make_layout()) as worker:
        worker._save_staging = None
        worker.bind_local_page_indices(lambda request_id: [[0, 4]])
        bind_saves(worker, [PageTransfer(H1, 0, 4, 1)])
        worker.wait_for_save(None)
    assert set(put_addresses(fake_store)) == {worker._namespaces[0].key(H1)}
    assert worker._outstanding_saves.get(9, 0) == 0


def test_the_guard_is_on_by_default(store_config, fake_store, monkeypatch):
    monkeypatch.delenv(worker_module.COMMIT_SWAP_GUARD_ENV, raising=False)
    with make_worker(fake_store, layout=make_layout()) as worker:
        assert worker._swap_guard is True
