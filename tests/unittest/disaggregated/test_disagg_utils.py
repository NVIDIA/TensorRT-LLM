import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

# isort: off
from tensorrt_llm.llmapi.disagg_utils import (
    MIN_GLOBAL_ID, CtxGenServerConfig, DisaggServerConfig, extract_ctx_gen_cfgs,
    extract_router_config, extract_disagg_cfg, get_global_disagg_request_id,
    get_local_request_id, get_server_configs_dict, parse_disagg_config_file)
# isort: on


def get_yaml_config():
    config = {
        "hostname": "test_host",
        "port": 9000,
        "context_servers": {
            "max_batch_size": 1,
            "num_instances": 2,
            "urls": ["host1:8001", "host2:8002"],
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
        },
        "generation_servers": {
            "router": {
                "type": "load_balancing",
                "use_tokens": False,
            },
            "max_batch_size": 1,
            "num_instances": 1,
            "urls": ["host3:8003"],
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
    }
    return config


def get_yaml_config_with_disagg_cluster():
    config = {
        "hostname": "test_host",
        "port": 9000,
        "context_servers": {
            "max_batch_size": 1,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
        },
        "generation_servers": {
            "router": {
                "type": "load_balancing",
                "use_tokens": False,
            },
            "max_batch_size": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
        "disagg_cluster": {
            "cluster_uri": "http://test_host:9000",
            "cluster_name": "test_cluster",
            "minimal_instances": {
                "context_servers": 2,
                "generation_servers": 2,
            },
            "heartbeat_interval_sec": 1,
            "inactive_timeout_sec": 2,
        },
    }
    return config


@pytest.fixture
def sample_yaml_config(request):
    if request.param == "disagg_cluster":
        config = get_yaml_config_with_disagg_cluster()
    else:
        config = get_yaml_config()
    return config


@pytest.fixture
def sample_yaml_file(sample_yaml_config, tmp_path):
    config = sample_yaml_config

    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(config, f)
    return yaml_file


def verify_disagg_config(config: DisaggServerConfig,
                         sample_yaml_config: str = ""):
    assert config.hostname == "test_host"
    assert config.port == 9000
    assert config.ctx_router_config.type == "round_robin"
    assert config.gen_router_config.type == "load_balancing"
    if sample_yaml_config == "":
        assert len(config.server_configs) == 3


@pytest.mark.parametrize("sample_yaml_config", ["disagg_cluster", ""],
                         indirect=True)
def test_parse_disagg_config_file(sample_yaml_file, sample_yaml_config):
    config = parse_disagg_config_file(sample_yaml_file)
    assert isinstance(config, DisaggServerConfig)
    verify_disagg_config(config, sample_yaml_config)


@pytest.mark.parametrize("sample_yaml_config", ["disagg_cluster", ""],
                         indirect=True)
def test_extract_disagg_cfg(sample_yaml_config):
    config = extract_disagg_cfg(**sample_yaml_config)
    assert isinstance(config, DisaggServerConfig)
    verify_disagg_config(config, sample_yaml_config)


def test_extract_ctx_gen_cfgs():
    configs = extract_ctx_gen_cfgs(
        type="ctx",
        num_instances=2,
        urls=["host1:8001", "host2:8002"],
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    assert len(configs) == 2
    assert configs[0].hostname == "host1"
    assert configs[0].port == 8001
    assert configs[0].instance_num_ranks == 2


@pytest.mark.parametrize("sample_yaml_config", [""], indirect=True)
def test_extract_router_config(sample_yaml_config):
    ctx_server_config = sample_yaml_config["context_servers"]
    gen_server_config = sample_yaml_config["generation_servers"]
    ctx_router_config = extract_router_config(ctx_server_config)
    gen_router_config = extract_router_config(gen_server_config)
    assert ctx_router_config.type == "round_robin"  # use default
    assert gen_router_config.type == "load_balancing"
    assert gen_router_config.args["use_tokens"] == False
    assert gen_router_config.args["max_batch_size"] == 1
    assert "max_num_tokens" not in gen_router_config.args


def test_get_server_configs_dict():
    server_configs = [
        CtxGenServerConfig(type="ctx",
                           hostname="host1",
                           port=8001,
                           instance_num_ranks=2),
        CtxGenServerConfig(type="gen",
                           hostname="host2",
                           port=8002,
                           instance_num_ranks=1),
    ]
    num_workers, server_dict = get_server_configs_dict(server_configs)
    assert num_workers == 3
    assert len(server_dict) == 2
    assert ("host1", 8001) in server_dict
    assert ("host2", 8002) in server_dict


# test get_global_disagg_request_id
@pytest.mark.parametrize("multithread", [True, False],
                         ids=["multithread", "singlethread"])
def test_get_global_disagg_request_id(multithread):
    iter = 10000
    node_ids = list(range(10))
    thread_num = len(node_ids)

    def get_ids(node_ids):
        all_node_ids = [[] for _ in range(len(node_ids))]
        for i in range(iter):
            if i % (4000 // thread_num) == 0:
                time.sleep(0.001)
            for i, node_id in enumerate(node_ids):
                all_node_ids[i].append(get_global_disagg_request_id(node_id))
        return all_node_ids

    if multithread:
        with ThreadPoolExecutor(max_workers=len(node_ids)) as executor:
            all_node_ids = [
                ids[0] for ids in executor.map(get_ids, [[i] for i in node_ids])
            ]
    else:
        all_node_ids = get_ids(node_ids)

    all_ids = set(i for ids in all_node_ids for i in ids)
    assert len(all_ids) == iter * len(node_ids)
    assert all(id >= MIN_GLOBAL_ID and id < ((1 << 63) - 1) for id in all_ids)


def test_get_local_request_id():
    last_id = MIN_GLOBAL_ID - 100
    ids = set()
    for i in range(1000):
        last_id = get_local_request_id(last_id)
        assert last_id >= 0
        assert last_id < MIN_GLOBAL_ID
        ids.add(last_id)
    assert len(ids) == 1000
    assert min(ids) == 0
    assert max(ids) == MIN_GLOBAL_ID - 1
    assert max(ids) - min(ids) > (
        1 << 40)  # ensure there is enough space for local ids
