import tempfile
from pathlib import Path

import torch

from tensorrt_llm.llmapi.llm_utils import *

# isort: off
from .test_llm import llama_model_path
# isort: on

from tensorrt_llm.llmapi.llm_utils import *


def test_ModelLoader():
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    args = LlmArgs(model=llama_model_path, kv_cache_config=kv_cache_config)
    args._setup()

    # Test with HF model
    temp_dir = tempfile.TemporaryDirectory()

    def build_engine():
        model_loader = ModelLoader(args)
        engine_dir = model_loader(engine_dir=Path(temp_dir.name))
        assert engine_dir
        return engine_dir

    # Test with engine
    args.model = build_engine()
    args._setup()
    assert args.model_format is _ModelFormatKind.TLLM_ENGINE
    print(f'engine_dir: {args.model}')
    model_loader = ModelLoader(args)
    engine_dir = model_loader()
    assert engine_dir == args.model


def test_CachedModelLoader():
    # CachedModelLoader enables engine caching and multi-gpu building
    args = LlmArgs(model=llama_model_path,
                   kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))
    args.enable_build_cache = True
    args._setup()
    stats = LlmBuildStats()
    model_loader = CachedModelLoader(args, llm_build_stats=stats)
    engine_dir, _ = model_loader()
    assert engine_dir
    assert engine_dir.exists() and engine_dir.is_dir()
    model_format = ModelLoader.get_model_format(engine_dir)
    assert model_format is _ModelFormatKind.TLLM_ENGINE


def test_LlmArgs_default_gpus_per_node():
    # default
    llm_args = LlmArgs(model=llama_model_path)
    assert llm_args.gpus_per_node == torch.cuda.device_count()

    # set explicitly
    llm_args = LlmArgs(model=llama_model_path, gpus_per_node=6)
    assert llm_args.gpus_per_node == 6
