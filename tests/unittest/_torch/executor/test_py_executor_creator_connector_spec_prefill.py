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
"""Which speculative configs may run next to a KV connector.

A disaggregated context server with a one-model drafter (MTP or EAGLE3) serves
context-only requests, which never reject a draft token, so a connector whose
two halves declare `supports_speculative_prefill` stays valid there. Every other
combination keeps the speculative-decoding refusal, and an explicit
KVCacheManagerV1 selection keeps V1's unchanged behaviour. `mooncake-store` is
the connector shipped with such declarations.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorManager,
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    MooncakeStoreConnectorConfig,
    StoreRole,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.worker import (
    MooncakeStoreConnectorWorker,
)
from tensorrt_llm._torch.pyexecutor.py_executor import _validate_connector_admission
from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _connector_runs_speculative_prefill_only,
)
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    SADecodingConfig,
)

pytestmark = pytest.mark.cpu_only

DISAGG = CacheTransceiverConfig(backend="DEFAULT")


class _Declaring:
    supports_speculative_prefill = True


class _Plain:
    pass


def _mtp_eagle() -> MTPDecodingConfig:
    config = MTPDecodingConfig(max_draft_len=1)
    assert config.spec_dec_mode.is_mtp_eagle_one_model()
    return config


def _mtp_vanilla() -> MTPDecodingConfig:
    # Not `num_nextn_predict_layers=2`: that deprecated input is remapped to
    # max_draft_len and leaves the mode at MTP-Eagle.
    config = MTPDecodingConfig(max_draft_len=2, use_mtp_vanilla=True)
    assert config.spec_dec_mode.is_mtp_vanilla()
    return config


def _eagle3() -> Eagle3DecodingConfig:
    config = Eagle3DecodingConfig(max_draft_len=3, speculative_model="draft")
    assert config.spec_dec_mode.is_eagle3_one_model()
    return config


MOONCAKE = (MooncakeStoreConnectorWorker, MooncakeStoreConnectorScheduler)


@pytest.mark.parametrize("use_v2", [True, "auto"], ids=["v2", "auto"])
@pytest.mark.parametrize(
    "connector_classes",
    [(_Declaring, _Declaring), MOONCAKE],
    ids=["declaring", "mooncake_store"],
)
@pytest.mark.parametrize(
    "make_spec_config",
    [_mtp_eagle, _mtp_vanilla, _eagle3],
    ids=["mtp_eagle", "mtp_vanilla", "eagle3"],
)
def test_one_model_drafter_runs_prefill_only_on_a_declaring_disagg_connector(
    make_spec_config, connector_classes, use_v2
) -> None:
    worker_cls, scheduler_cls = connector_classes
    assert _connector_runs_speculative_prefill_only(
        make_spec_config(), worker_cls, scheduler_cls, DISAGG, use_v2
    )


@pytest.mark.parametrize(
    "spec_config, transceiver",
    [
        (_eagle3(), None),
        (_mtp_eagle(), None),
        (_eagle3(), CacheTransceiverConfig()),
        (NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2), DISAGG),
    ],
    ids=["eagle3_aggregated", "mtp_aggregated", "transceiver_without_backend", "ngram"],
)
def test_mooncake_store_keeps_the_refusal_off_a_disaggregated_context_server(
    spec_config, transceiver
) -> None:
    assert not _connector_runs_speculative_prefill_only(spec_config, *MOONCAKE, transceiver, True)


def test_a_capacity_role_mooncake_server_runs_mtp_with_attention_dp() -> None:
    """A disaggregated generation server lends memory and serves generation work.

    Its classes declare the mode, so one-model MTP on a disaggregated server
    selects it, and attention DP is accepted too. A role that transfers no KV is
    exempt from the context-only refusal, so generation requests are admitted.
    """
    assert _connector_runs_speculative_prefill_only(_mtp_eagle(), *MOONCAKE, DISAGG, True)
    KvCacheConnectorManager.validate_attention_dp(*MOONCAKE, pp_size=1, cp_size=1)

    # The halves as an attention-DP owner holds them, without opening a pool.
    worker = object.__new__(MooncakeStoreConnectorWorker)
    worker._config = MooncakeStoreConnectorConfig(
        master_server_address="127.0.0.1:50051", role=StoreRole.CAPACITY
    )
    scheduler = object.__new__(MooncakeStoreConnectorScheduler)
    with patch(
        "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector.mpi_rank", return_value=1
    ):
        manager = KvCacheConnectorManager(
            worker, scheduler, enable_attention_dp=True, speculative_prefill_only=True
        )
    assert manager.capacity_only is True
    assert manager.speculative_prefill_only is True
    generation_request = SimpleNamespace(
        is_context_only_request=False,
        is_generation_only_request=True,
        is_dummy_request=False,
        is_attention_dp_dummy=False,
    )
    _validate_connector_admission(manager, generation_request)


@pytest.mark.parametrize(
    "spec_config, worker_cls, scheduler_cls, transceiver, use_v2",
    [
        (None, _Declaring, _Declaring, DISAGG, True),
        (MTPDecodingConfig(), _Declaring, _Declaring, DISAGG, True),
        (
            NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2),
            _Declaring,
            _Declaring,
            DISAGG,
            True,
        ),
        (SADecodingConfig(max_draft_len=4), _Declaring, _Declaring, DISAGG, True),
        (_eagle3(), _Declaring, _Plain, DISAGG, True),
        (_eagle3(), _Plain, _Declaring, DISAGG, True),
        (_eagle3(), KvCacheConnectorWorker, KvCacheConnectorScheduler, DISAGG, True),
        (_eagle3(), _Declaring, _Declaring, None, True),
        (_eagle3(), _Declaring, _Declaring, CacheTransceiverConfig(), True),
        (_mtp_eagle(), _Declaring, _Declaring, None, "auto"),
        (_mtp_eagle(), _Declaring, _Declaring, DISAGG, False),
        (_eagle3(), _Declaring, _Declaring, DISAGG, False),
    ],
    ids=[
        "no_spec",
        "unresolved_mtp_draft_len",
        "ngram",
        "suffix_automaton",
        "worker_only_declares",
        "scheduler_only_declares",
        "base_classes",
        "aggregated",
        "transceiver_without_backend",
        "mtp_aggregated",
        "mtp_on_v1",
        "eagle3_on_v1",
    ],
)
def test_other_combinations_keep_the_speculative_refusal(
    spec_config, worker_cls, scheduler_cls, transceiver, use_v2
) -> None:
    assert not _connector_runs_speculative_prefill_only(
        spec_config, worker_cls, scheduler_cls, transceiver, use_v2
    )


def test_declares_speculative_prefill() -> None:
    assert KvCacheConnectorManager.declares_speculative_prefill(_Declaring, _Declaring)
    # A TP follower rank holds no scheduler, so only its worker is checked.
    assert KvCacheConnectorManager.declares_speculative_prefill(_Declaring, None)
    assert not KvCacheConnectorManager.declares_speculative_prefill(_Declaring, _Plain)
    assert not KvCacheConnectorManager.declares_speculative_prefill(_Plain, _Declaring)
    assert not KvCacheConnectorManager.declares_speculative_prefill(_Plain, None)
    assert not KvCacheConnectorManager.declares_speculative_prefill(
        KvCacheConnectorWorker, KvCacheConnectorScheduler
    )


def test_only_an_explicit_true_declares() -> None:
    class Truthy:
        supports_speculative_prefill = "yes"

    assert not KvCacheConnectorManager.declares_speculative_prefill(Truthy, Truthy)
