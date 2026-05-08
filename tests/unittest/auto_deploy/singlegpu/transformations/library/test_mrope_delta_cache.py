from pathlib import Path

import torch
import torch.nn as nn
import yaml

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import StateResourceHandler
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
from tensorrt_llm._torch.auto_deploy.transform.library.mrope_delta_cache import (
    InitializeMropeDeltaCache,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def test_initialize_mrope_delta_cache_registers_state_resource():
    cm = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=18,
        device="cpu",
    )
    transform = InitializeMropeDeltaCache.from_kwargs(stage="cache_init")

    mod, info = transform._apply_to_full_model(
        nn.Module(),
        cm,
        factory=None,
        shared_config=SharedConfig(),
    )

    assert isinstance(mod, nn.Module)
    assert not info.skipped
    assert info.num_matches == 1
    assert len(cm._resource_lookup) == 1
    resource_name, resource_handler = next(iter(cm._resource_lookup.items()))
    assert resource_name.endswith("_mrope_delta_cache")
    assert resource_handler == StateResourceHandler(1, dtype=torch.int32)


def test_initialize_mrope_delta_cache_disabled_in_default_config():
    config_path = (
        _repo_root() / "tensorrt_llm" / "_torch" / "auto_deploy" / "config" / "default.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["transforms"]["initialize_mrope_delta_cache"]["enabled"] is False


def test_qwen_registry_configs_explicitly_enable_mrope_delta_cache():
    config_dir = _repo_root() / "examples" / "auto_deploy" / "model_registry" / "configs"
    # 35b enables mrope_delta_cache; 400b explicitly disables it (NVFP4 accuracy)
    expected = {
        "qwen3.5_moe_35b.yaml": True,
        "qwen3.5_moe_400b.yaml": False,
    }
    for config_name, expected_enabled in expected.items():
        with open(config_dir / config_name) as f:
            config = yaml.safe_load(f)

        assert config["transforms"]["initialize_mrope_delta_cache"]["enabled"] is expected_enabled
