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
"""Creation-time failure modes of the Python (V2) KV-cache transceiver.

The 'auto' resolver adopts a model's runtime preference verbatim and only
falls back to C++ for conditions it can decide from
``cache_transceiver_config`` alone (non-NIXL backend, infinite timeout, and
only when no preference was expressed). Everything else must fail loudly at
transceiver creation — these tests pin those creation-time errors.
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation import kv_cache_transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.native.rank_info import manager_page_table
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import create_cache_reuse_adapter
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    CppMambaHybridCacheManager,
    MixedMambaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("backend", ["UCX", "MPI", "MOONCAKE"])
def test_python_runtime_rejects_non_nixl_backend_at_creation(backend: str) -> None:
    """An explicit or preference-adopted 'PYTHON' runtime on a non-NIXL
    backend raises at creation instead of being silently rerouted."""
    config = CacheTransceiverConfig(backend=backend, transceiver_runtime="PYTHON")

    with pytest.raises(ValueError, match="only supports NIXL backend"):
        transceiver_module.create_kv_cache_transceiver(Mock(), Mock(), Mock(), Mock(), config)


def test_python_runtime_rejects_non_helix_cp_at_creation() -> None:
    """cp_size > 1 without helix is rejected by _check_compatible at
    creation; the resolver deliberately does not gate on CP (helix is
    supported, and a per-server cp check would resolve ctx and gen servers
    to different runtimes)."""
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")
    mapping = Mock()
    mapping.cp_size = 2
    mapping.has_cp_helix.return_value = False
    mapping.cp_config = {"cp_type": "star_attention"}

    with pytest.raises(ValueError, match="only cp_size == 1 or helix CP"):
        transceiver_module.create_kv_cache_transceiver(
            mapping, Mock(), Mock(spec=KVCacheManagerV2), Mock(), config
        )


@pytest.mark.parametrize(
    "manager_cls", [KVCacheManager, CppMambaHybridCacheManager, MixedMambaHybridCacheManager]
)
def test_python_runtime_rejects_v1_manager_at_creation(
    manager_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = object.__new__(manager_cls)
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")
    constructor = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.KvCacheTransceiverV2", constructor
    )

    with pytest.raises(ValueError, match="does not support KV cache manager V1"):
        transceiver_module.create_kv_cache_transceiver(None, None, manager, None, config)

    constructor.assert_not_called()


@pytest.mark.parametrize(
    "manager_cls", [KVCacheManager, CppMambaHybridCacheManager, MixedMambaHybridCacheManager]
)
def test_direct_python_transceiver_rejects_v1_before_resource_setup(
    manager_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = object.__new__(manager_cls)
    worker = Mock()
    monkeypatch.setattr("tensorrt_llm._torch.disaggregation.transceiver.TransferWorker", worker)

    with pytest.raises(ValueError, match="does not support KV cache manager V1"):
        KvCacheTransceiverV2(None, None, manager, None)

    worker.assert_not_called()


@pytest.mark.parametrize("factory", [manager_page_table, create_cache_reuse_adapter])
def test_python_transfer_resources_reject_v1_manager(factory) -> None:
    with pytest.raises(ValueError, match="requires KVCacheManagerV2; V1 is unsupported"):
        factory(object.__new__(KVCacheManager))


def test_python_runtime_accepts_v2_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = object.__new__(KVCacheManagerV2)
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")
    constructor = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.KvCacheTransceiverV2", constructor
    )

    result = transceiver_module.create_kv_cache_transceiver(None, None, manager, None, config)

    assert result is constructor.return_value
    constructor.assert_called_once_with(None, None, manager, config)


def test_python_runtime_rejects_separate_legacy_mamba_manager() -> None:
    manager = object.__new__(KVCacheManagerV2)
    legacy_mamba = object.__new__(MixedMambaHybridCacheManager)
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")

    with pytest.raises(ValueError, match="does not support KV cache manager V1"):
        transceiver_module.create_kv_cache_transceiver(
            None, None, manager, None, config, mamba_cache_manager=legacy_mamba
        )
