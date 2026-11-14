# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for independent-process tensor exports."""

import base64
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from torch.multiprocessing import get_sharing_strategy

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("shape", [(8,), (0,), (3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_filename_roundtrip_preserves_storage_views(shape, dtype):
    source = torch.arange(torch.Size(shape).numel(), dtype=dtype).reshape(shape)
    if source.ndim == 2:
        source = source.T
    strategy = get_sharing_strategy()
    exported = SharedTensorContainer.from_tensor(source).dump_to_dict()
    restored = SharedTensorContainer.from_dict(exported).get_local_view()
    torch.testing.assert_close(restored, source)
    assert restored.stride() == source.stride()
    assert get_sharing_strategy() == strategy


def test_concurrent_filename_exports_do_not_change_global_strategy(monkeypatch):
    strategy = get_sharing_strategy()
    barrier = threading.Barrier(3, timeout=5)
    original = torch.storage.TypedStorage._share_filename_cpu_

    def share(storage):
        assert get_sharing_strategy() == strategy
        barrier.wait()
        result = original(storage)
        barrier.wait()
        return result

    monkeypatch.setattr(torch.storage.TypedStorage, "_share_filename_cpu_", share)

    def export(value):
        source = torch.full((1024,), value)
        return SharedTensorContainer.from_tensor(source).dump_to_dict()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(export, value) for value in (1, 2)]
        barrier.wait()
        assert get_sharing_strategy() == strategy
        barrier.wait()
        exports = [future.result(timeout=5) for future in futures]
    for value, exported in zip((1, 2), exports):
        restored = SharedTensorContainer.from_dict(exported).get_local_view()
        torch.testing.assert_close(restored, torch.full((1024,), value))


def test_export_failure_does_not_change_global_strategy(monkeypatch):
    strategy = get_sharing_strategy()

    def fail(*args):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(SharedTensorContainer, "cpu_handle_to_dict", fail)
    with pytest.raises(RuntimeError, match="serialization failed"):
        SharedTensorContainer.from_tensor(torch.ones(8)).dump_to_dict()
    assert get_sharing_strategy() == strategy


def test_release_unconsumed_export_reclaims_shared_memory():
    exported = SharedTensorContainer.from_tensor(torch.ones(1024 * 1024)).dump_to_dict()
    path = Path("/dev/shm") / base64.b64decode(exported["storage_handle"]).decode().lstrip("/")
    assert path.exists()
    gc.collect()
    assert path.exists()
    SharedTensorContainer.release_dict(exported)
    gc.collect()
    assert not path.exists()


def test_release_cuda_export_does_not_reopen_producer_ipc(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.UntypedStorage,
        "_release_ipc_counter",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    SharedTensorContainer.release_dict(
        {
            "method_key": 1,
            "ref_counter_handle": base64.b64encode(b"counter"),
            "ref_counter_offset": 2,
            "storage_device": 0,
        }
    )
    assert calls == [((b"counter", 2), {"device": 0})]
