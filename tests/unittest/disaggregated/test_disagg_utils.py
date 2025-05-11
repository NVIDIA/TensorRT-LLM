import pytest
import yaml

# isort: off
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig, DisaggServerConfig, extract_ctx_gen_cfgs,
    extract_disagg_cfg, get_server_configs_dict, parse_disagg_config_file)
# isort: on


def get_yaml_config():
    config = {
        "hostname": "test_host",
        "port": 9000,
        "context_servers": {
            "router_type": "round_robin",
            "num_instances": 2,
            "urls": ["host1:8001", "host2:8002"],
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
        },
        "generation_servers": {
            "router_type": "requests_load_balancing",
            "num_instances": 1,
            "urls": ["host3:8003"],
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
    }
    return config


@pytest.fixture
def sample_yaml_config():
    config = get_yaml_config()
    return config


@pytest.fixture
def sample_yaml_file(tmp_path):
    config = get_yaml_config()

    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(config, f)
    return yaml_file


def test_parse_disagg_config_file(sample_yaml_file):
    config = parse_disagg_config_file(sample_yaml_file)
    assert isinstance(config, DisaggServerConfig)
    assert config.hostname == "test_host"
    assert config.port == 9000
    assert config.ctx_router_type == "round_robin"
    assert config.gen_router_type == "requests_load_balancing"
    assert len(config.server_configs) == 3


def test_extract_disagg_cfg(sample_yaml_config):
    config = extract_disagg_cfg(**sample_yaml_config)
    assert isinstance(config, DisaggServerConfig)
    assert config.hostname == "test_host"
    assert config.port == 9000
    assert config.ctx_router_type == "round_robin"
    assert config.gen_router_type == "requests_load_balancing"
    assert len(config.server_configs) == 3


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
