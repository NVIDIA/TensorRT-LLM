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

from .open_search_db_utils import _match
from .test_perf_sanity import ClientConfig, ServerConfig


def _server_config(**overrides) -> ServerConfig:
    config = {
        "model_name": "example_model",
        "gpus": 8,
        "tensor_parallel_size": 8,
        "max_batch_size": 32,
        "kv_cache_config": {"dtype": "fp8"},
    }
    config.update(overrides)
    return ServerConfig(config)


def _client_config(**overrides) -> ClientConfig:
    config = {
        "concurrency": 32,
        "iterations": 10,
        "isl": 1024,
        "osl": 1024,
        "random_range_ratio": 0.0,
        "backend": "openai",
        "streaming": True,
    }
    config.update(overrides)
    return ClientConfig(config, model_name="example_model")


def _benchmark_data(server_config: ServerConfig, client_config: ClientConfig) -> dict:
    return {
        "s_gpu_type": "b200",
        "s_runtime": "aggr_server",
        **server_config.to_db_data(),
        **client_config.to_db_data(),
    }


def _benchmark_match_keys(server_config: ServerConfig, client_config: ClientConfig) -> list[str]:
    return [
        "s_gpu_type",
        "s_runtime",
        *server_config.to_match_keys(),
        *client_config.to_match_keys(),
    ]


def test_scenario_matching_ignores_recipe_tuning_changes():
    previous_config = _server_config()
    updated_config = _server_config(
        match_mode="scenario",
        tensor_parallel_size=4,
        moe_expert_parallel_size=2,
        max_batch_size=64,
        kv_cache_config={"dtype": "auto"},
    )
    client_config = _client_config()
    match_keys = _benchmark_match_keys(updated_config, client_config)

    assert updated_config.to_match_keys() == ["s_model_name", "l_gpus"]
    assert {"l_concurrency", "l_isl", "l_osl"}.issubset(match_keys)
    assert _match(
        _benchmark_data(previous_config, client_config),
        _benchmark_data(updated_config, client_config),
        match_keys,
    )


def test_config_matching_keeps_recipe_tuning_keys():
    previous_config = _server_config()
    updated_config = _server_config(
        tensor_parallel_size=4,
        moe_expert_parallel_size=2,
        max_batch_size=64,
        kv_cache_config={"dtype": "auto"},
    )
    client_config = _client_config()

    assert not _match(
        _benchmark_data(previous_config, client_config),
        _benchmark_data(updated_config, client_config),
        _benchmark_match_keys(updated_config, client_config),
    )


def test_scenario_matching_distinguishes_gpu_count():
    previous_config = _server_config()
    updated_config = _server_config(match_mode="scenario", gpus=4)
    client_config = _client_config()

    assert not _match(
        _benchmark_data(previous_config, client_config),
        _benchmark_data(updated_config, client_config),
        _benchmark_match_keys(updated_config, client_config),
    )


def test_scenario_matching_distinguishes_client_workload():
    server_config = _server_config(match_mode="scenario")
    previous_client_config = _client_config()
    updated_client_config = _client_config(concurrency=64)

    assert not _match(
        _benchmark_data(server_config, previous_client_config),
        _benchmark_data(server_config, updated_client_config),
        _benchmark_match_keys(server_config, updated_client_config),
    )
