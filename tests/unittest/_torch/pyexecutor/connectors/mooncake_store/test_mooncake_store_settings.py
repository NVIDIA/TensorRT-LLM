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
"""Unit tests for the effective settings a mooncake-store deployment runs with.

The settings are settled in the LLM constructor rather than in any one entry
point: `trtllm-serve`, `trtllm-bench` and a direct `LLM(...)` all reach the
connector through it, and only the first provisions the pool. So most of these
go straight at `apply_effective_settings`, with one covering the constructor
that both the ranks and the usage report read the result of.

Runs without a Mooncake installation and without a GPU. The model is never
built: the constructor test replaces the build with a recorder, which stands
where the ranks would be spawned.
"""

import json

import pytest

import tensorrt_llm.usage as usage
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import settings
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    CLIENT_CONFIG_NAME,
    CONFIG_PATH_ENV,
)
from tensorrt_llm.llmapi.llm import LLM, _TorchLLM
from tensorrt_llm.llmapi.llm_args import (
    KvCacheConfig,
    KvCacheConnectorConfig,
    MooncakeStoreConfig,
    TorchLlmArgs,
)
from tensorrt_llm.usage import usage_lib

pytestmark = pytest.mark.cpu_only

GIB = 1 << 30


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient pool config: these tests set what they mean to test."""
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)


def write_client_config(path, **settings_in_file):
    """A Mooncake client config on disk, as an external launcher renders one."""
    path.write_text(json.dumps({"master_server_address": "10.0.0.7:50051", **settings_in_file}))
    return path


def llm_args(*, pool=None, connector="mooncake-store", **kv_cache_settings):
    """Args for a deployment that selected `connector`, with no pool provisioned."""
    return TorchLlmArgs(
        model="/tmp/dummy_model",
        kv_connector_config=KvCacheConnectorConfig(connector=connector, mooncake_store=pool),
        kv_cache_config=KvCacheConfig(**kv_cache_settings),
    )


def mooncake_store_config(**overrides):
    return MooncakeStoreConfig(
        pool="file:///shared/pool.json", model_key="our-checkpoint", **overrides
    )


def test_an_inherited_config_supersedes_the_block_it_ignores(monkeypatch, tmp_path):
    """`MOONCAKE_CONFIG_PATH` decides what every rank opens its handle with.

    Nothing provisions a pool here, which is how `trtllm-bench` and a direct
    `LLM(...)` meet an externally managed one.
    """
    inherited = write_client_config(
        tmp_path / "external.json",
        role="capacity",
        global_segment_size="32GiB",
        namespace="external",
        model_key="their-checkpoint",
        transfer_batch_size=128,
        stage_through_host=True,
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(inherited))
    pool = mooncake_store_config(role="both", segment_size="16GiB")

    settings.apply_effective_settings(llm_args(pool=pool))

    assert pool.role == "capacity"
    assert pool.segment_size == 32 * GIB
    assert pool.namespace == "external"
    assert pool.model_key == "their-checkpoint"
    assert pool.transfer_batch_size == 128
    assert pool.stage_through_host is True

    # How to reach the pool and where this run's files go are the server's own,
    # so an inherited config has nothing to say about them.
    assert pool.pool == "file:///shared/pool.json"


def test_a_config_rendered_into_the_run_directory_is_read_too(tmp_path):
    """Ranks an external launcher started never inherited the exported path.

    They read the config out of the run directory, so the settings restated
    here have to come from the same place.
    """
    write_client_config(tmp_path / CLIENT_CONFIG_NAME, role="producer")
    pool = mooncake_store_config(run_dir=str(tmp_path))

    settings.apply_effective_settings(llm_args(pool=pool))

    assert pool.role == "producer"
    assert pool.run_dir == str(tmp_path)


def test_an_unreadable_inherited_config_leaves_the_block_alone(monkeypatch, tmp_path):
    """The workers report a bad config; this path only describes one."""
    inherited = tmp_path / "truncated.json"
    inherited.write_text("{not json")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(inherited))
    pool = mooncake_store_config(segment_size="16GiB")

    settings.apply_effective_settings(llm_args(pool=pool))

    assert pool.role == "both"
    assert pool.segment_size == "16GiB"


def test_only_the_values_the_deployment_chose_are_reported_as_overridden(monkeypatch, tmp_path):
    """A default being resolved is not an override, and reads as noise as one."""
    inherited = write_client_config(
        tmp_path / "external.json", role="capacity", namespace="external"
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(inherited))
    # `namespace` is left to the pool, so resolving it is not a contradiction.
    pool = mooncake_store_config(role="both")

    warnings = []
    monkeypatch.setattr(settings.logger, "warning", warnings.append)
    settings.apply_effective_settings(llm_args(pool=pool))

    assert pool.namespace == "external"
    overrides = [line for line in warnings if "joins the pool as" in line]
    assert len(overrides) == 1
    assert "role='capacity'" in overrides[0]
    assert "namespace" not in overrides[0]


def test_the_pool_replaces_this_engines_own_offload_tiers(monkeypatch, tmp_path):
    """A native tier beside the pool claims a second share of the node's DRAM."""
    monkeypatch.setenv(
        CONFIG_PATH_ENV, str(write_client_config(tmp_path / "external.json", role="both"))
    )
    args = llm_args(
        pool=mooncake_store_config(),
        host_cache_size=4 * GIB,
        disk_cache_size=8 * GIB,
        disk_cache_path=str(tmp_path),
        enable_partial_reuse=True,
    )

    settings.apply_effective_settings(args)

    assert args.kv_cache_config.host_cache_size == 0
    assert args.kv_cache_config.disk_cache_size == 0
    assert args.kv_cache_config.enable_partial_reuse is False


def test_a_capacity_role_keeps_what_it_was_configured_with(monkeypatch, tmp_path):
    """It transfers nothing, so neither override has anything to protect.

    No page of its cache is registered with the pool, so a tier migration
    invalidates no address, and it looks nothing up, so it has no lookup for a
    partial match to spoil.
    """
    monkeypatch.setenv(
        CONFIG_PATH_ENV,
        str(write_client_config(tmp_path / "external.json", role="capacity")),
    )
    args = llm_args(
        pool=mooncake_store_config(),
        host_cache_size=4 * GIB,
        enable_partial_reuse=True,
    )

    settings.apply_effective_settings(args)

    assert args.kv_cache_config.host_cache_size == 4 * GIB
    assert args.kv_cache_config.enable_partial_reuse is True


def test_a_capacity_role_still_leaves_its_host_tier_to_be_sized(monkeypatch, tmp_path):
    """An unset `host_cache_size` is what asks for the auto host tier.

    That tier is where the V2 `MAX_UTILIZATION` scheduler suspends pages, and
    a generation server, which this role is for, has no other reclaim path.
    """
    monkeypatch.setenv(
        CONFIG_PATH_ENV,
        str(write_client_config(tmp_path / "external.json", role="capacity")),
    )
    args = llm_args(pool=mooncake_store_config())

    settings.apply_effective_settings(args)

    assert args.kv_cache_config.host_cache_size is None


def test_the_tiers_go_off_before_any_pool_has_been_provisioned():
    """There is no client config yet, and the role comes off the block."""
    args = llm_args(pool=mooncake_store_config(role="consumer"), host_cache_size=4 * GIB)

    settings.apply_effective_settings(args)

    assert args.kv_cache_config.host_cache_size == 0
    assert args.kv_cache_config.enable_partial_reuse is False


def test_a_settled_deployment_is_restated_without_a_word(monkeypatch, tmp_path):
    """Every caller is free to apply these, so a repeat has to be quiet."""
    monkeypatch.setenv(
        CONFIG_PATH_ENV,
        str(write_client_config(tmp_path / "external.json", role="producer")),
    )
    pool = mooncake_store_config(role="both")
    args = llm_args(pool=pool, host_cache_size=4 * GIB)
    settings.apply_effective_settings(args)

    logged = []
    monkeypatch.setattr(settings.logger, "warning", logged.append)
    monkeypatch.setattr(settings.logger, "info", logged.append)
    settings.apply_effective_settings(args)

    assert logged == []
    assert pool.role == "producer"
    assert args.kv_cache_config.host_cache_size == 0


def test_another_connector_keeps_its_own_cache_tiers():
    """Only this connector claims to be the deployment's offload tier."""
    args = llm_args(connector="lmcache", host_cache_size=4 * GIB)

    settings.apply_effective_settings(args)

    assert args.kv_cache_config.host_cache_size == 4 * GIB


def test_the_llm_constructor_settles_them_before_it_spawns_its_ranks(monkeypatch, tmp_path):
    """The ranks and the usage report have to describe one deployment.

    The constructor spawns the ranks from these args and the report reads them
    afterwards, so settling them anywhere later leaves the two disagreeing.
    """
    monkeypatch.setenv(
        CONFIG_PATH_ENV,
        str(
            write_client_config(
                tmp_path / "external.json", role="both", global_segment_size="32GiB"
            )
        ),
    )
    # Keep the process-wide usage session out of it, and record what the report
    # would have been sent.
    monkeypatch.setattr(usage_lib, "apply_usage_session_config", lambda *_a, **_k: False)
    reported = []
    monkeypatch.setattr(usage, "report_usage", lambda **kwargs: reported.append(kwargs))

    at_spawn = {}

    def record_instead_of_building(self):
        pool = self.args.kv_connector_config.mooncake_store
        at_spawn.update(
            role=pool.role,
            segment_size=pool.segment_size,
            host_cache_size=self.args.kv_cache_config.host_cache_size,
        )

    monkeypatch.setattr(_TorchLLM, "_build_model", record_instead_of_building)

    llm = LLM(
        model="/tmp/dummy_model",
        gpus_per_node=1,
        kv_connector_config=KvCacheConnectorConfig(
            connector="mooncake-store",
            mooncake_store=mooncake_store_config(role="capacity", segment_size="16GiB"),
        ),
        kv_cache_config=KvCacheConfig(host_cache_size=4 * GIB),
    )
    try:
        assert at_spawn == {
            "role": "both",
            "segment_size": 32 * GIB,
            "host_cache_size": 0,
        }
        assert len(reported) == 1
        assert reported[0]["llm_args"] is llm.args
    finally:
        llm.shutdown()
