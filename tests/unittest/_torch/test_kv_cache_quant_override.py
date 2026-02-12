import pytest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.model_loader import \
    validate_and_set_kv_cache_quant
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _make_model_config(kv_cache_quant_algo):
    return ModelConfig(quant_config=QuantConfig(
        kv_cache_quant_algo=kv_cache_quant_algo))


def test_validate_and_set_kv_cache_quant_auto_uses_checkpoint():
    model_config = _make_model_config(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "auto")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_explicit_dtype_overrides():
    model_config = _make_model_config(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_validate_and_set_kv_cache_quant_rejects_invalid_dtype():
    model_config = _make_model_config(QuantAlgo.FP8)
    with pytest.raises(ValueError, match="Accepted types are"):
        validate_and_set_kv_cache_quant(model_config, "invalid_dtype")
