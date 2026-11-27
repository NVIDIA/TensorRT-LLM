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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation import kv_cache_transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import create_cache_reuse_adapter
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor import _util as util_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    CppMambaHybridCacheManager,
    MambaHybridCacheManagerV2,
    MixedMambaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, ResourceManagerType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, SchedulerConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _CreationReached(Exception):
    """Stops the resource factory at its call of the transceiver creation."""


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


@pytest.mark.parametrize("factory", [build_page_table_from_manager, create_cache_reuse_adapter])
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
    monkeypatch.setattr(transceiver_module.logger, "_appeared_keys", set())
    emitted = Mock()
    monkeypatch.setattr(transceiver_module.logger, "log", emitted)

    result = transceiver_module.create_kv_cache_transceiver(None, None, manager, None, config)

    assert result is constructor.return_value
    constructor.assert_called_once_with(None, None, manager, config)
    assert not any(
        "deprecated" in str(argument)
        for recorded in emitted.call_args_list
        for argument in recorded.args[1:]
    )


@pytest.mark.parametrize("runtime", ["CPP", None, "auto"])
def test_cpp_deprecation_warning_is_emitted_once(
    runtime: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = object.__new__(KVCacheManager)
    manager.impl = Mock()
    manager.total_num_kv_heads_per_layer = [1]
    manager.num_kv_heads_per_layer = [1]
    manager.head_dim = 128
    manager.tokens_per_block = 32
    manager.dtype = transceiver_module.tensorrt_llm.bindings.DataType.HALF
    manager.enable_indexer_k_cache = False
    dist = Mock()
    dist.pp_allgather.return_value = [1]
    native_constructor = Mock()
    emitted = Mock()
    monkeypatch.setattr(transceiver_module, "CacheTransceiverCpp", native_constructor)
    monkeypatch.setattr(transceiver_module, "_disagg_inflight_cancel_enabled_cache", False)
    monkeypatch.delenv("TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT", raising=False)
    monkeypatch.setattr(transceiver_module.logger, "_appeared_keys", set())
    monkeypatch.setattr(transceiver_module.logger, "log", emitted)

    for _ in range(2):
        config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime=runtime)
        result = transceiver_module.create_kv_cache_transceiver(
            Mapping(), dist, manager, transceiver_module.AttentionTypeCpp.DEFAULT, config
        )
        assert isinstance(result, transceiver_module.BindKvCacheTransceiver)
        assert config.transceiver_runtime == ("CPP" if runtime == "CPP" else None)

    warnings = [
        recorded
        for recorded in emitted.call_args_list
        if recorded.args[0] == transceiver_module.logger.WARNING
        and "The C++ cache transceiver" in recorded.args[1]
        and "deprecated" in recorded.args[1]
    ]
    assert len(warnings) == 1
    assert native_constructor.call_count == 2


def test_python_runtime_rejects_separate_legacy_mamba_manager() -> None:
    manager = object.__new__(KVCacheManagerV2)
    legacy_mamba = object.__new__(MixedMambaHybridCacheManager)
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")

    with pytest.raises(ValueError, match="does not support KV cache manager V1"):
        transceiver_module.create_kv_cache_transceiver(
            None, None, manager, None, config, mamba_cache_manager=legacy_mamba
        )


@pytest.mark.parametrize(
    ("manager_cls", "keeps_recurrent_state"),
    [
        (KVCacheManager, False),
        (KVCacheManagerV2, False),
        (CppMambaHybridCacheManager, True),
        (MixedMambaHybridCacheManager, True),
        (MambaHybridCacheManagerV2, True),
    ],
)
def test_resource_factory_names_a_recurrent_cache_as_the_mamba_manager(
    manager_cls: type, keeps_recurrent_state: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The creation rejects the hybrid managers it cannot serve by its ``mamba_cache_manager``
    argument, and the C++ transceiver takes the recurrent layers of the rank from it, so the
    executor's resource factory passes a cache that keeps recurrent state there and no other."""
    manager = object.__new__(manager_cls)
    manager.impl = Mock()
    creation = Mock(side_effect=_CreationReached)
    monkeypatch.setattr(util_module, "set_low_latency_dispatch", lambda enabled: None)
    monkeypatch.setattr(util_module, "KVCacheV2Scheduler", Mock())
    monkeypatch.setattr(util_module, "SimpleUnifiedScheduler", Mock())
    monkeypatch.setattr(util_module, "create_kv_cache_transceiver", creation)
    engine = SimpleNamespace(
        mm_encoder_item_scheduling_enabled=False,
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                pretrained_config=SimpleNamespace(model_type="llama"), is_encoder_decoder=False
            )
        ),
    )

    with pytest.raises(_CreationReached):
        util_module.create_py_executor_resources(
            dist=SimpleNamespace(),
            resources={ResourceManagerType.KV_CACHE_MANAGER: manager},
            mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
            llm_args=SimpleNamespace(extra_resource_managers={}),
            ctx_chunk_config=None,
            model_engine=engine,
            max_batch_size=2,
            max_num_tokens=128,
            max_num_sequences=4,
            scheduler_config=SchedulerConfig(use_python_scheduler=True),
        )

    assert creation.call_args.args[2] is manager
    assert creation.call_args.args[5] is (manager if keeps_recurrent_state else None)
