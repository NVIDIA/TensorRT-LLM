from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.dataclasses.general import InferenceRequest


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
