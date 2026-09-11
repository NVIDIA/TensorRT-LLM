# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    RequestData,
    SchedulerOutput,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def example() -> Iterator[ModuleType]:
    path = Path(__file__).resolve().parents[3] / "examples/llm-api/llm_kv_cache_connector.py"
    spec = importlib.util.spec_from_file_location("adp_persistent_connector_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        tensor_parallel_size=2,
        enable_attention_dp=True,
        kv_cache_config=SimpleNamespace(tokens_per_block=4),
    )


def test_restore_from_another_owner(
    example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish on owner 0, then restore into different block IDs on owner 1."""
    monkeypatch.setenv(example.CONNECTOR_CACHE_FOLDER_KEY, str(tmp_path))
    producer = example.PersistentKvCacheConnectorLeader(_args())
    consumer = example.PersistentKvCacheConnectorLeader(_args())
    src_worker = example.PersistentKvCacheConnectorWorker(_args())
    dst_worker = example.PersistentKvCacheConnectorWorker(_args())
    src_cache = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dst_cache = torch.zeros_like(src_cache)
    src_worker.register_kv_caches(src_cache)
    dst_worker.register_kv_caches(dst_cache)
    tokens = list(range(9))
    request = MagicMock(cache_salt="tenant-a")
    request.request_id = 5
    request.get_tokens.return_value = tokens
    assert producer.get_num_new_matched_tokens(request, 0) == (0, False)
    source_meta = producer.build_connector_meta(
        SchedulerOutput(
            new_requests=[RequestData(5, tokens, [2, 0, 1], 0, 9, cache_salt="tenant-a")],
            attention_dp_rank=0,
        )
    )
    src_worker.bind_connector_meta(source_meta)
    src_worker.wait_for_save(MagicMock())
    assert consumer.get_num_new_matched_tokens(request, 0) == (8, False)
    target_meta = consumer.build_connector_meta(
        SchedulerOutput(
            new_requests=[RequestData(5, tokens, [0, 1, 2], 0, 1, cache_salt="tenant-a")],
            attention_dp_rank=1,
        )
    )
    dst_worker.bind_connector_meta(target_meta)
    dst_worker.start_load_kv(MagicMock())
    torch.testing.assert_close(dst_cache[0], src_cache[2])
    torch.testing.assert_close(dst_cache[1], src_cache[0])
    assert torch.count_nonzero(dst_cache[2]) == 0

    # Same suffix under a different prefix or tenant is not reusable KV.
    request.get_tokens.return_value = [99] + tokens[1:]
    assert consumer.get_num_new_matched_tokens(request, 0) == (0, False)
    request.get_tokens.return_value = tokens
    request.cache_salt = "tenant-b"
    assert consumer.get_num_new_matched_tokens(request, 0) == (0, False)


def test_store_does_not_publish_incomplete_files(example: ModuleType, tmp_path: Path) -> None:
    worker = example.PersistentKvCacheConnectorWorker(_args())
    worker.register_kv_caches(torch.ones(1, 4))
    target = tmp_path / "shared-prefix.pt"
    worker.bind_connector_meta(example.PersistentKvCacheConnectorMetadata(save=[(target, 0)]))
    save = torch.save

    def inspect_publish(tensor: torch.Tensor, temporary_path: Path) -> None:
        save(tensor, temporary_path)
        assert not target.exists()

    with patch.object(example.torch, "save", side_effect=inspect_publish):
        worker.wait_for_save(MagicMock())
    torch.testing.assert_close(torch.load(target, weights_only=True), torch.ones(4))
    assert list(tmp_path.iterdir()) == [target]


def test_failed_store_removes_temporary_file(example: ModuleType, tmp_path: Path) -> None:
    worker = example.PersistentKvCacheConnectorWorker(_args())
    worker.register_kv_caches(torch.ones(1, 4))
    worker.bind_connector_meta(
        example.PersistentKvCacheConnectorMetadata(save=[(tmp_path / "shared-prefix.pt", 0)])
    )
    with patch.object(example.torch, "save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            worker.wait_for_save(MagicMock())
    assert not list(tmp_path.iterdir())
