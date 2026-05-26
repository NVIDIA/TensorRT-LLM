import json
from unittest import mock

import pytest
import torch

from tensorrt_llm.commands.serve import get_llm_args
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import (
    AttentionDpConfig,
    CacheTransceiverConfig,
    CpConfig,
    CudaGraphConfig,
    KvCacheConnectorConfig,
    NGramDecodingConfig,
    RocketSparseAttentionConfig,
    TorchCompileConfig,
    TorchLlmArgs,
    TrtLlmArgs,
    update_llm_args_with_extra_dict,
)
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


def _compressed_tensors_nvfp4_config(**overrides):
    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
                "input_activations": {
                    "strategy": "tensor_group",
                },
            },
        },
    }
    config.update(overrides)
    return config


def _torch_llm_args_with_quant_config(
    tmp_path, quant_config: QuantConfig, **kwargs
) -> TorchLlmArgs:
    llm_args = TorchLlmArgs(model=str(tmp_path), **kwargs)
    llm_args.quant_config = quant_config
    llm_args.sync_quant_config_with_kv_cache_config_dtype()
    return llm_args.sync_attn_backend_with_turboquant4_kv_cache()


def test_get_llm_args_plumbs_kv_cache_dtype():
    llm_args, _ = get_llm_args(model="dummy", gpus_per_node=1, kv_cache_dtype="nvfp4")
    assert llm_args["kv_cache_config"].dtype == "nvfp4"


def test_get_llm_args_plumbs_turboquant4_kv_cache_dtype():
    llm_args, _ = get_llm_args(
        model="dummy", gpus_per_node=1, kv_cache_dtype="turboquant4"
    )
    assert llm_args["kv_cache_config"].dtype == "turboquant4"


def test_kv_cache_config_dtype_validation():
    cfg = KvCacheConfig(dtype="NVFP4")
    assert cfg.dtype == "nvfp4"

    cfg = KvCacheConfig(dtype="TURBOQUANT4")
    assert cfg.dtype == "turboquant4"

    cfg = KvCacheConfig(dtype="float16")
    assert cfg.dtype == "float16"

    cfg = KvCacheConfig(dtype=torch.float16)
    assert cfg.dtype == "float16"

    with pytest.raises(ValueError, match="kv_cache_config.dtype must be one of"):
        KvCacheConfig(dtype="invalid_dtype")


def test_torch_llm_args_syncs_nvfp4_kv_cache_dtype(tmp_path):
    llm_args = TorchLlmArgs(model=str(tmp_path), kv_cache_config=KvCacheConfig(dtype="nvfp4"))
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4


def test_torch_llm_args_syncs_turboquant4_kv_cache_dtype(tmp_path):
    llm_args = TorchLlmArgs(model=str(tmp_path), kv_cache_config=KvCacheConfig(dtype="turboquant4"))
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.attn_backend == "TRTLLM"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="turboquant4"),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1]),
    )
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.attn_backend == "TRTLLM"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_torch_llm_args_accepts_turboquant4_without_attention_dp_config(tmp_path):
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="turboquant4"),
        attention_dp_config=None,
    )

    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.kv_cache_config.dtype == "turboquant4"


def test_torch_llm_args_syncs_turboquant4_quant_config_backend(tmp_path):
    llm_args = _torch_llm_args_with_quant_config(
        tmp_path, QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)
    )
    assert llm_args.attn_backend == "TRTLLM"
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_torch_llm_args_accepts_turboquant4_quant_config_field(tmp_path):
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        kv_cache_config=KvCacheConfig(dtype=torch.float16),
    )

    assert "quant_config" in TorchLlmArgs.model_fields
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_extra_options_turboquant4_quant_config_normalizes_kv_dtype(tmp_path):
    llm_args_dict = {
        "model": str(tmp_path),
        "backend": "pytorch",
        "kv_cache_config": KvCacheConfig(dtype=torch.float16),
    }
    extra_dict = {"quant_config": {"kv_cache_quant_algo": "TURBOQUANT4"}}

    merged = update_llm_args_with_extra_dict(llm_args_dict, extra_dict)
    llm_args = TorchLlmArgs(**merged)

    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_extra_options_accepts_constructed_turboquant4_quant_config(tmp_path):
    llm_args_dict = {
        "model": str(tmp_path),
        "backend": "pytorch",
        "kv_cache_config": KvCacheConfig(dtype=torch.float16),
    }
    extra_dict = {"quant_config": QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4)}

    merged = update_llm_args_with_extra_dict(llm_args_dict, extra_dict)
    llm_args = TorchLlmArgs(**merged)

    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2


def test_torch_llm_args_normalizes_turboquant4_concrete_kv_dtype(tmp_path):
    llm_args = _torch_llm_args_with_quant_config(
        tmp_path,
        QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        kv_cache_config=KvCacheConfig(dtype=torch.float16),
    )
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_torch_llm_args_clears_turboquant4_compile_and_block_reuse(tmp_path):
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="turboquant4", enable_block_reuse=True),
        torch_compile_config=TorchCompileConfig(enable_piecewise_cuda_graph=True),
    )

    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_torch_llm_args_refreshes_cached_quant_modes_for_turboquant4(tmp_path):
    quant_config = QuantConfig()
    assert not quant_config.layer_quant_mode.has_turboquant4_kv_cache()

    llm_args = _torch_llm_args_with_quant_config(
        tmp_path,
        quant_config,
        kv_cache_config=KvCacheConfig(dtype="turboquant4"),
    )

    assert llm_args.quant_config.layer_quant_mode.has_turboquant4_kv_cache()
    assert llm_args.quant_config.quant_mode.has_turboquant4_kv_cache()


def test_torch_llm_args_rewrites_unsupported_turboquant4_backend(tmp_path):
    llm_args = _torch_llm_args_with_quant_config(
        tmp_path,
        QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        attn_backend="FLASHINFER",
    )
    assert llm_args.attn_backend == "TRTLLM"
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_torch_llm_args_rejects_turboquant4_sparse_attention(tmp_path):
    with pytest.raises(ValueError, match="sparse attention"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            sparse_attention_config=RocketSparseAttentionConfig(),
        )


def test_torch_llm_args_rejects_turboquant4_speculative_decoding(tmp_path):
    with pytest.raises(ValueError, match="speculative decoding"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            speculative_config=NGramDecodingConfig(max_draft_len=1),
        )


def test_torch_llm_args_rejects_turboquant4_sliding_window(tmp_path):
    with pytest.raises(ValueError, match="sliding-window"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4", max_attention_window=[128]),
        )


def test_torch_llm_args_rejects_turboquant4_context_parallelism(tmp_path):
    with pytest.raises(ValueError, match="context parallelism"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            context_parallel_size=2,
        )


def test_torch_llm_args_rejects_turboquant4_cp_config(tmp_path):
    with pytest.raises(ValueError, match="context parallelism"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            cp_config=CpConfig(),
        )


def test_torch_llm_args_rejects_turboquant4_beam_search(tmp_path):
    with pytest.raises(ValueError, match="beam search"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            max_beam_width=2,
        )


def test_torch_llm_args_accepts_turboquant4_unset_beam_width(tmp_path):
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="turboquant4"),
        max_beam_width=None,
    )

    assert llm_args.max_beam_width is None
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4


def test_torch_llm_args_rejects_turboquant4_event_buffer(tmp_path):
    with pytest.raises(ValueError, match="event buffers"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4", event_buffer_max_size=1),
        )


def test_torch_llm_args_rejects_turboquant4_cache_transceiver(tmp_path):
    with pytest.raises(ValueError, match="cache transceiver"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            cache_transceiver_config=CacheTransceiverConfig(backend="NIXL"),
        )


def test_torch_llm_args_rejects_turboquant4_kv_cache_connector(tmp_path):
    with pytest.raises(ValueError, match="KV cache connector"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            kv_connector_config=KvCacheConnectorConfig(
                connector_module="example.connector",
                connector_scheduler_class="Scheduler",
                connector_worker_class="Worker",
            ),
        )


def test_torch_llm_args_rejects_turboquant4_kv_cache_aware_routing(tmp_path):
    with pytest.raises(ValueError, match="KV-cache-aware attention DP routing"):
        TorchLlmArgs(
            model=str(tmp_path),
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
            attention_dp_config=AttentionDpConfig(enable_kv_cache_aware_routing=True),
        )


def test_autodeploy_llm_args_rejects_turboquant4_kv_cache(tmp_path):
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs as AutoDeployLlmArgs

    device_props = mock.Mock()
    device_props.major = 9
    device_props.minor = 0

    with pytest.raises(
        ValueError, match="TurboQuant4 KV cache is not supported with AutoDeploy"
    ), mock.patch("torch.cuda.get_device_properties", return_value=device_props):
        AutoDeployLlmArgs(
            model=str(tmp_path),
            backend="_autodeploy",
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
        )


def test_trt_llm_args_rejects_turboquant4_kv_cache(tmp_path):
    device_props = mock.Mock()
    device_props.major = 9
    device_props.minor = 0

    with pytest.raises(
        ValueError, match="TurboQuant4 KV cache is supported only by the PyTorch backend"
    ), mock.patch("torch.cuda.get_device_properties", return_value=device_props):
        TrtLlmArgs(
            model="dummy",
            skip_tokenizer_init=True,
            kv_cache_config=KvCacheConfig(dtype="turboquant4"),
        )


def test_trt_llm_args_rejects_turboquant4_quant_config(tmp_path):
    device_props = mock.Mock()
    device_props.major = 9
    device_props.minor = 0

    with pytest.raises(
        ValueError, match="TurboQuant4 KV cache is supported only by the PyTorch backend"
    ), mock.patch("torch.cuda.get_device_properties", return_value=device_props):
        TrtLlmArgs(
            model="dummy",
            skip_tokenizer_init=True,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.TURBOQUANT4),
        )


def test_update_from_hf_quant_config_rejects_turboquant4_trt_args(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    device_props = mock.Mock()
    device_props.major = 9
    device_props.minor = 0

    with mock.patch("torch.cuda.get_device_properties", return_value=device_props):
        llm_args = TrtLlmArgs(model=str(tmp_path), skip_tokenizer_init=True)
    model_loader = ModelLoader(llm_args)

    error = "TurboQuant4 KV cache is supported only by the PyTorch backend"
    with pytest.raises(ValueError, match=error):
        model_loader._update_from_hf_quant_config()


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


def test_update_from_hf_quant_config_parses_compressed_tensors_model_kwargs(tmp_path):
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        model_kwargs={
            "quantization_config": _compressed_tensors_nvfp4_config(
                kv_cache_scheme={
                    "num_bits": 8,
                    "type": "float",
                }
            ),
        },
    )
    model_loader = ModelLoader(llm_args)

    assert model_loader._update_from_hf_quant_config() is True
    assert llm_args.quant_config.quant_algo == QuantAlgo.NVFP4
    assert llm_args.quant_config.group_size == 16
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_update_from_hf_quant_config_rejects_compressed_tensors_kv_conflict(tmp_path):
    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        model_kwargs={
            "quantization_config": _compressed_tensors_nvfp4_config(
                kv_cache_scheme={
                    "num_bits": 8,
                    "type": "float",
                }
            ),
        },
    )
    llm_args.quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)
    model_loader = ModelLoader(llm_args)

    with pytest.raises(ValueError, match="conflicting with FP8 KV cache"):
        model_loader._update_from_hf_quant_config()


def test_update_from_hf_quant_config_explicit_turboquant4_overrides(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="FP8")

    llm_args = TorchLlmArgs(model=str(tmp_path), kv_cache_config=KvCacheConfig(dtype="turboquant4"))
    model_loader = ModelLoader(llm_args)

    assert model_loader._update_from_hf_quant_config() is True
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4


def test_update_from_hf_quant_config_turboquant4_keeps_trtllm_backend(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    llm_args = TorchLlmArgs(model=str(tmp_path), kv_cache_config=KvCacheConfig(dtype="auto"))
    model_loader = ModelLoader(llm_args)

    assert model_loader._update_from_hf_quant_config() is True
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.attn_backend == "TRTLLM"
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse


def test_update_from_hf_quant_config_turboquant4_normalizes_concrete_kv_dtype(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype=torch.float16),
        torch_compile_config=TorchCompileConfig(enable_piecewise_cuda_graph=True),
    )
    assert not llm_args.quant_config.layer_quant_mode.has_turboquant4_kv_cache()
    model_loader = ModelLoader(llm_args)

    assert model_loader._update_from_hf_quant_config() is True
    assert llm_args.quant_config.kv_cache_quant_algo == QuantAlgo.TURBOQUANT4
    assert llm_args.kv_cache_config.dtype == "turboquant4"
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2
    assert llm_args.cuda_graph_config is None
    assert llm_args.torch_compile_config is None
    assert not llm_args.kv_cache_config.enable_block_reuse
    assert llm_args.quant_config.layer_quant_mode.has_turboquant4_kv_cache()
    assert llm_args.quant_config.quant_mode.has_turboquant4_kv_cache()


def test_update_from_hf_quant_config_rejects_turboquant4_sparse_attention(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="auto"),
        sparse_attention_config=RocketSparseAttentionConfig(),
    )
    model_loader = ModelLoader(llm_args)

    with pytest.raises(ValueError, match="sparse attention"):
        model_loader._update_from_hf_quant_config()


def test_update_from_hf_quant_config_rejects_turboquant4_speculative_decoding(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="auto"),
        speculative_config=NGramDecodingConfig(max_draft_len=1),
    )
    model_loader = ModelLoader(llm_args)

    with pytest.raises(ValueError, match="speculative decoding"):
        model_loader._update_from_hf_quant_config()


def test_update_from_hf_quant_config_rejects_turboquant4_kv_cache_connector(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="auto"),
        kv_connector_config=KvCacheConnectorConfig(
            connector_module="example.connector",
            connector_scheduler_class="Scheduler",
            connector_worker_class="Worker",
        ),
    )
    model_loader = ModelLoader(llm_args)

    with pytest.raises(ValueError, match="KV cache connector"):
        model_loader._update_from_hf_quant_config()


def test_update_from_hf_quant_config_rejects_turboquant4_kv_cache_aware_routing(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="auto"),
        attention_dp_config=AttentionDpConfig(enable_kv_cache_aware_routing=True),
    )
    model_loader = ModelLoader(llm_args)

    with pytest.raises(ValueError, match="KV-cache-aware attention DP routing"):
        model_loader._update_from_hf_quant_config()


def test_update_from_hf_quant_config_rejects_turboquant4_sliding_window(tmp_path):
    _write_hf_quant_config(tmp_path, kv_cache_quant_algo="TURBOQUANT4")

    llm_args = TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=KvCacheConfig(dtype="auto", max_attention_window=[128]),
    )
    model_loader = ModelLoader(llm_args)

    with pytest.raises(ValueError, match="sliding-window"):
        model_loader._update_from_hf_quant_config()
