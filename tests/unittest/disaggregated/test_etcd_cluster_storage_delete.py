# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from tensorrt_llm.serve.cluster_storage import Etcd3ClusterStorage

pytestmark = pytest.mark.cpu_only


def make_storage():
    storage = object.__new__(Etcd3ClusterStorage)
    storage._client = mock.Mock()
    storage._watch_handles = {}
    return storage


@pytest.mark.asyncio
@pytest.mark.parametrize("client_result", [True, False])
async def test_delete_returns_etcd_client_result(client_result):
    storage = make_storage()
    storage.client.delete.return_value = client_result

    assert await storage.delete("worker/key") is client_result
    storage.client.delete.assert_called_once_with("worker/key")
