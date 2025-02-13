from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from random import choices, shuffle
from typing import Dict, List, Tuple, Union

import yaml
from torch.cuda import device_count

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.pyexecutor.model_engine import \
    validate_and_set_kv_cache_quant
from tensorrt_llm.bench.build.build import (get_benchmark_engine_settings,
                                            get_model_config)
from tensorrt_llm.bench.dataclasses.general import (DatasetMetadata,
                                                    InferenceRequest)
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

_KV_CACHE_MAP = {
    QuantAlgo.FP8.value: "fp8",
    QuantAlgo.NVFP4.value: "fp8",
}


def get_executor_requests(
    requests: List[InferenceRequest],
    streaming: bool,
    eos_id: int,
    pad_id: int,
) -> List[trtllm.Request]:
    """Generate a list of TRT-LLM Executor requests.

    Args:
        requests (List[InferenceRequest]): A list of inference requests for processing.
        pad_id (int): Padding token identifier
        eos_id (int): End of sequence token identifier.
        streaming (bool, optional): Enable streaming for this request. Defaults to False.

    Returns:
        List[trtllm.Request]:  A list of TRT-LLM Executor request instance.
    """
    executor_requests = []
    while requests:
        request = requests.pop()
        executor_requests.append(
            get_executor_request(request,
                                 pad_id=pad_id,
                                 eos_id=eos_id,
                                 streaming=streaming))
        del request

    return executor_requests


def get_executor_request(request: InferenceRequest,
                         pad_id: int,
                         eos_id: int,
                         streaming: bool = False) -> trtllm.Request:
    """Generate a TRT-LLM Executor request.

    Args:
        request (InferenceRequest): An inference request for processing.
        pad_id (int): Padding token identifier
        eos_id (int): End of sequence token identifier.
        streaming (bool, optional): Enable streaming for this request. Defaults to False.

    Returns:
        trtllm.Request: A TRT-LLM Executor request instance.
    """
    return trtllm.Request(
        input_token_ids=request.input_ids,
        max_tokens=request.output_tokens,
        stop_words=[],
        bad_words=[],
        streaming=streaming,
        output_config=trtllm.OutputConfig(exclude_input_from_output=True),
        pad_id=pad_id,
        end_id=eos_id,
    )


def get_settings_from_engine(
    engine_path: Path
) -> Tuple[Dict[str, Union[str, int]], Dict[str, Union[str, int]]]:
    """Retrieve basic engine information.

    Args:
        engine_path (Path): Path to a TRT-LLM engine directory.

    Returns:
        Tuple[Dict[str, Union[str, int]], Dict[str, Union[str, int]]]: Engine
        properties parsed from the engine at engine_path.
    """
    config_path = engine_path / "config.json"
    runtime_config = {}

    with open(config_path, "r") as config_json:
        config = json.load(config_json)

    engine_world_map = config["pretrained_config"]["mapping"]
    engine_build_cfg = config["build_config"]
    engine_parallel_map = engine_build_cfg["auto_parallel_config"]

    world_config = {
        "pp_size": engine_world_map["pp_size"],
        "tp_size": engine_world_map["tp_size"],
        "world_size": engine_world_map["world_size"],
        "gpus_per_node": engine_parallel_map["gpus_per_node"],
    }

    executor_settings = {
        "max_batch_size": engine_build_cfg["max_batch_size"],
        "max_num_tokens": engine_build_cfg["max_num_tokens"],
    }

    runtime_config.update({
        "sw_version": config["version"],
        "engine_dir": str(engine_path.absolute()),
        "settings_config": executor_settings,
        "world_config": world_config,
    })

    runtime_config["performance_options"] = {}
    runtime_config["decoding_config"] = {
        "decoding_mode": engine_build_cfg["speculative_decoding_mode"]
    }
    return runtime_config, engine_build_cfg


def get_settings(params: dict, dataset_metadata: DatasetMetadata, model: str,
                 model_path: Union[Path, None]) -> Dict[str, Union[str, int]]:
    """Retrieve basic runtime config for pytorch backend path

    Args:
        params (dict): Configuration parameters.
        model (str): Model name.
        model_path (Union[Path, None]): Path to the model.
    Returns:
        Dict[str, Union[str, int]]: Properties for runtime config.
    """
    extra_llm_api_options = params.get("extra_llm_api_options")
    kv_cache_dtype = "auto"
    if extra_llm_api_options:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)
            if "pytorch_backend_config" in llm_args_dict:
                if "kv_cache_dtype" in llm_args_dict["pytorch_backend_config"]:
                    kv_cache_dtype = llm_args_dict["pytorch_backend_config"][
                        "kv_cache_dtype"]

    world_config = {
        "pp_size": params.get("pp"),
        "tp_size": params.get("tp"),
        "world_size": params.get("pp") * params.get("tp"),
        "ep_size": params.get("ep"),
        "gpus_per_node": device_count()
    }

    if params.get("max_batch_size") and params.get("max_num_tokens"):
        logger.info("Use user-provided max batch size and max num tokens.")
        max_batch_size, max_num_tokens = params.get(
            "max_batch_size"), params.get("max_num_tokens")
    else:
        model_config = get_model_config(model, model_path)

        from tensorrt_llm._torch.model_config import ModelConfig
        model = model_path or model
        tllm_model_config = ModelConfig.from_pretrained(model,
                                                        trust_remote_code=True)

        if (kv_cache_dtype is None
                and tllm_model_config.quant_config.kv_cache_quant_algo is None):
            kv_cache_dtype = _KV_CACHE_MAP.get(
                tllm_model_config.quant_config.quant_algo, "auto")

        validate_and_set_kv_cache_quant(tllm_model_config, kv_cache_dtype)

        max_batch_size, max_num_tokens = get_benchmark_engine_settings(
            model_config,
            tllm_model_config.quant_config,
            params.get("tp"),
            params.get("pp"),
            dataset_metadata.avg_isl,
            dataset_metadata.avg_osl,
        )
        # NOTE: This max is because the Pytorch backend does not support
        # chunking yet. We need to force the max number of tokens to the
        # max ISL we expect to see + max batch size. This means we can always
        # handle the longest context and generation.
        max_num_tokens = max(dataset_metadata.max_isl + max_batch_size,
                             max_num_tokens)
        logger.info(
            f"Max batch size and max num tokens not provided. "
            f"Using heuristics or pre-defined settings: max_batch_size={max_batch_size}, max_num_tokens={max_num_tokens}."
        )

    pyt_options = {
        "use_cuda_graph": True,
        "enable_overlap_scheduler": True,
        "kv_cache_dtype": kv_cache_dtype,
    }

    return {
        "sw_version": version("tensorrt_llm"),
        "model_path": model_path,
        "settings_config": {
            "max_batch_size": max_batch_size,
            "max_num_tokens": max_num_tokens,
            "chunking": False,
        },
        "world_config": world_config,
        "backend": "pytorch",
        "decoding_config": {},
        "performance_options": {
            "cuda_graphs": True,
            "pytorch_config": pyt_options,
        }
    }


def generate_warmup_dataset(requests, steps) -> List[InferenceRequest]:
    """Warm up the benchmarker."""
    warm_up_dataset = choices(requests, k=steps)
    shuffle(warm_up_dataset)
    return warm_up_dataset
