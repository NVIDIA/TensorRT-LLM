import tempfile
from pathlib import Path

import pytest
import torch

from tensorrt_llm.builder import PluginConfig
from tensorrt_llm.llmapi.llm_utils import *

# isort: off
from test_llm import llama_model_path
# isort: on


def test_ConfigArbitrator_basic():
    # the performance and functionality have conflict plugins config, keep the functionalities and disable the performance's
    arb = _ConfigArbitrator()
    arb.claim_perf("chunked_context",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)
    arb.claim_func("block_reuse",
                   config_name="plugin_config",
                   use_paged_context_fmha=False)

    plugin_config = PluginConfig()
    arb(plugin_config=plugin_config)

    assert plugin_config.use_paged_context_fmha == False


def test_ConfigArbitrator_perf_conflict():
    # When performance-related plugins conflict, some performance-related feature maybe disabled to avoid conflict
    # No exception should be raised in this case
    arb = _ConfigArbitrator()
    arb.claim_perf("perf0",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)
    arb.claim_perf("perf1",
                   config_name="plugin_config",
                   use_paged_context_fmha=False)

    plugin_config = PluginConfig()
    arb(plugin_config=plugin_config)

    # The perf0 is claimed first, so the feature should be enabled
    assert plugin_config.use_paged_context_fmha == True


def test_ConfigArbitrator_func_conflict():
    # When functional-related plugins conflict, an exception should be raised to remind the user to resolve the conflict
    arb = _ConfigArbitrator()
    arb.claim_func("func0",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)
    arb.claim_func("func1",
                   config_name="plugin_config",
                   use_paged_context_fmha=False)

    plugin_config = PluginConfig()
    with pytest.raises(ConfigArbitrateError):
        arb(plugin_config=plugin_config)


def test_ConfigArbitrator_setup():
    # Setup some pre-defined plugins configures
    arb = _ConfigArbitrator()
    arb.setup("pre-ampere is not supported",
              config_name="plugin_config",
              use_paged_context_fmha=False)
    arb.claim_func("func0",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)

    plugin_config = PluginConfig()
    with pytest.raises(ConfigArbitrateError):
        arb(plugin_config=plugin_config)


def test_ConfigArbitor_multi_configs():
    # A func claims two different configures, and the arbiter should be able to handle it
    arb = _ConfigArbitrator()

    arb.claim_func("func0",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)
    arb.claim_func("func0",
                   config_name="kv_cache_config",
                   enable_block_reuse=False)

    plugin_config = PluginConfig()
    kv_cache_config = KvCacheConfig()

    arb(plugin_config=plugin_config, kv_cache_config=kv_cache_config)
    assert plugin_config.use_paged_context_fmha == True
    assert kv_cache_config.enable_block_reuse == False


def test_ConfigArbitor_multi_configs_func_conflict():
    # A func claims two different configures with conflict options, the arbiter should raise an exception
    arb = _ConfigArbitrator()

    arb.claim_func("func0",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)
    arb.claim_func("func0",
                   config_name="kv_cache_config",
                   enable_block_reuse=True)

    arb.claim_func("func1",
                   config_name="plugin_config",
                   use_paged_context_fmha=False)

    plugin_config = PluginConfig()
    kv_cache_config = KvCacheConfig()

    with pytest.raises(ConfigArbitrateError):
        arb(plugin_config=plugin_config, kv_cache_config=kv_cache_config)


def test_ConfigArbitor_multi_configs_perf_conflict():
    arb = _ConfigArbitrator()

    arb.claim_func("func0",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)
    arb.claim_func("func0",
                   config_name="kv_cache_config",
                   enable_block_reuse=True)

    arb.claim_perf("perf0",
                   config_name="plugin_config",
                   use_paged_context_fmha=False)  # conflict with func0
    arb.claim_perf("perf0", config_name="plugin_config",
                   paged_kv_cache=False)  # default value is True

    plugin_config = PluginConfig()
    kv_cache_config = KvCacheConfig()

    old_paged_kv_cache = plugin_config.paged_kv_cache
    arb(plugin_config=plugin_config, kv_cache_config=kv_cache_config)

    assert plugin_config.use_paged_context_fmha == True  # perf0 is disabled
    assert kv_cache_config.enable_block_reuse == True
    assert plugin_config.paged_kv_cache == old_paged_kv_cache  # perf0 is disabled


def test_ConfigArbitor_perf_fallback():
    arb = _ConfigArbitrator()

    fallback_triggered = False

    def fallback():
        nonlocal fallback_triggered
        fallback_triggered = True

    arb.claim_perf("perf0",
                   config_name="plugin_config",
                   use_paged_context_fmha=True)
    arb.claim_perf("perf1",
                   config_name="plugin_config",
                   use_paged_context_fmha=False,
                   fallback=fallback)

    plugin_config = PluginConfig()
    arb(plugin_config=plugin_config)

    assert plugin_config.use_paged_context_fmha == True
    assert fallback_triggered == True


def test_ModelLoader():
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    args = LlmArgs(llama_model_path, kv_cache_config=kv_cache_config)
    args.setup()

    # Test with HF model
    temp_dir = tempfile.TemporaryDirectory()

    def build_engine():
        model_loader = ModelLoader(args)
        engine_dir = model_loader(engine_dir=Path(temp_dir.name))
        assert engine_dir
        return engine_dir

    # Test with engine
    args.model = build_engine()
    args.setup()
    assert args.model_format is _ModelFormatKind.TLLM_ENGINE
    print(f'engine_dir: {args.model}')
    model_loader = ModelLoader(args)
    engine_dir = model_loader()
    assert engine_dir == args.model


def test_CachedModelLoader():
    # CachedModelLoader enables engine caching and multi-gpu building
    args = LlmArgs(llama_model_path,
                   kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))
    args.enable_build_cache = True
    args.setup()
    stats = LlmBuildStats()
    model_loader = CachedModelLoader(args, llm_build_stats=stats)
    engine_dir, _ = model_loader()
    assert engine_dir
    assert engine_dir.exists() and engine_dir.is_dir()
    model_format = ModelLoader.get_model_format(engine_dir)
    assert model_format is _ModelFormatKind.TLLM_ENGINE


def test_LlmArgs_default_gpus_per_node():
    # default
    llm_args = LlmArgs(llama_model_path)
    assert llm_args.gpus_per_node == torch.cuda.device_count()

    # set explicitly
    llm_args = LlmArgs(llama_model_path, gpus_per_node=6)
    assert llm_args.gpus_per_node == 6


if __name__ == '__main__':
    #test_ModelLoader()
    test_CachedModelLoader()
