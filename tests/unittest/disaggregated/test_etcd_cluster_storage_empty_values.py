# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from tensorrt_llm.serve.cluster_storage import Etcd3ClusterStorage

pytestmark = pytest.mark.cpu_only


def _make_storage() -> Etcd3ClusterStorage:
    storage = object.__new__(Etcd3ClusterStorage)
    storage._client = mock.Mock()
    storage._watch_handles = {}
    return storage


@pytest.mark.asyncio
async def test_get_preserves_existing_empty_string() -> None:
    storage = _make_storage()
    storage.client.get.return_value = (b"", mock.Mock())

    assert await storage.get("worker/empty") == ""


@pytest.mark.asyncio
async def test_get_still_returns_none_for_missing_key() -> None:
    storage = _make_storage()
    storage.client.get.return_value = (None, mock.Mock())

    assert await storage.get("worker/missing") is None
