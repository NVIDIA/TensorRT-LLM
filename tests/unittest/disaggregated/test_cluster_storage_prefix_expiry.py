# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import tensorrt_llm.serve.cluster_storage as cluster_storage
from tensorrt_llm.serve.cluster_storage import HttpClusterStorageServer, StorageItem

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
async def test_get_prefix_omits_expired_entries(monkeypatch):
    storage = HttpClusterStorageServer("", "")
    storage._storage = {
        "worker/live": StorageItem(key="worker/live", value="live", expire_time=-1),
        "worker/ttl-live": StorageItem(key="worker/ttl-live", value="fresh", expire_time=15),
        "worker/exact": StorageItem(key="worker/exact", value="boundary", expire_time=10),
        "worker/expired": StorageItem(key="worker/expired", value="stale", expire_time=5),
    }
    monkeypatch.setattr(cluster_storage, "key_time", lambda: 10)

    assert await storage.get_prefix("worker/") == {
        "worker/live": "live",
        "worker/ttl-live": "fresh",
    }
    assert await storage.get_prefix("worker/", keys_only=True) == {
        "worker/live": "",
        "worker/ttl-live": "",
    }
