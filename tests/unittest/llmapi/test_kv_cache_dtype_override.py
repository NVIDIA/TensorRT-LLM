import json

import pytest

from tensorrt_llm.commands.serve import get_llm_args
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _write_hf_quant_config(model_dir, kv_cache_quant_algo: str = "FP8"):
    with open(model_dir / "hf_quant_config.json", "w") as f:
        json.dump(
            {
                "quantization": {
                    "quant_algo": "FP8",
                    "kv_cache_quant_algo": kv_cache_quant_algo,
                }
            },
            f,
        )


def test_get_llm_args_plumbs_kv_cache_dtype():
    llm_args, _ = get_llm_args(model="dummy", kv_cache_dtype="nvfp4")
    assert llm_args["kv_cache_config"].dtype == "nvfp4"


def test_kv_cache_config_dtype_validation():
    cfg = KvCacheConfig(dtype="NVFP4")
    assert cfg.dtype == "nvfp4"

    cfg = KvCacheConfig(dtype="float16")
    assert cfg.dtype == "float16"

    with pytest.raises(ValueError, match="kv_cache_config.dtype must be one of"):
        KvCacheConfig(dtype="invalid_dtype")


def test_torch_llm_args_syncs_nvfp4_kv_cache_dtype(tmp_path):
    llm_args = TorchLlmArgs(model=str(tmp_path), kv_cache_config=KvCacheConfig(dtype="nvfp4"))
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_update_from_hf_quant_config_keeps_auto_strict(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="FP8")

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="auto"),
    )
    llm_args.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)
    model_loader = ModelLoader(llm_args)

    with pytest.raises(ValueError, match="conflicting with kv_cache_quant_algo"):
        model_loader._update_from_hf_quant_config()


def test_update_from_hf_quant_config_explicit_dtype_overrides(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="FP8")

    llm_args = TorchLlmArgs(model=str(tmp_path), kv_cache_config=KvCacheConfig(dtype="nvfp4"))
    model_loader = ModelLoader(llm_args)

    assert model_loader._update_from_hf_quant_config() is True
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
