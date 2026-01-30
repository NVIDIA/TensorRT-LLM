from typing import Any

import yaml

from tensorrt_llm._torch.auto_deploy.llm import LLM as AutoDeployLLM
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs as AutoDeployLlmArgs
from tensorrt_llm.llmapi.llm import _TorchLLM, _TrtLLM
from tensorrt_llm.llmapi.llm_args import BaseLlmArgs, TorchLlmArgs, TrtLlmArgs
from tensorrt_llm.logger import logger


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_llm_args_from_cli_params(model: str, **params: dict[str, Any]) -> BaseLlmArgs:
    """Construct an instance of LlmArgs from CLI params and optionally a config YAML file.

    This function constructs LlmArgs according to the following logic:
    1. CLI params are parsed into a dict that aligns with the structure of LlmArgs.
    2. If a config YAML file is provided, it is deep-merged on top.
    3. The LlmArgs class for the correct backend is instantiated with the merged dict.

    This means that the config file takes precedence over CLI params.
    """
    extra_llm_api_options = params.pop("extra_llm_api_options", None)
    llm_args_dict = {}

    # Handle CLI params that map to nested config fields
    params.setdefault("kv_cache_config", {})
    if "free_gpu_memory_fraction" in params:
        params["kv_cache_config"]["free_gpu_memory_fraction"] = params.pop(
            "free_gpu_memory_fraction"
        )
    # TODO: align CLI param naming with trtllm-serve / trtllm-bench to avoid duplication with above
    if "kv_cache_free_gpu_memory_fraction" in params:
        params["kv_cache_config"]["free_gpu_memory_fraction"] = params.pop(
            "kv_cache_free_gpu_memory_fraction"
        )
    if "disable_kv_cache_reuse" in params:
        params["kv_cache_config"]["enable_block_reuse"] = not params.pop("disable_kv_cache_reuse")

    # Override with config file
    config_dict = {}
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, "r") as f:
            config_dict = yaml.safe_load(f) or {}
    llm_args_dict = _deep_merge(params, config_dict)

    # Determine backend and the corresponding LlmArgs class
    backend = llm_args_dict.get("backend", "pytorch")
    if backend == "pytorch":
        llm_args_cls = TorchLlmArgs
    elif backend == "_autodeploy":
        llm_args_cls = AutoDeployLlmArgs
    elif backend == "tensorrt":
        llm_args_cls = TrtLlmArgs
    else:
        raise ValueError(f"Invalid backend: {backend}")

    # Remove model from llm_args_dict if present, since we pass it explicitly as the first argument
    llm_args_dict.pop("model", None)

    logger.info(f"Creating LLM args from dict: {llm_args_dict}")
    return llm_args_cls(model=model, **llm_args_dict)


def create_llm_from_llm_args(llm_args: BaseLlmArgs):
    """Create and return the appropriate LLM instance for the backend corresponding to llm_args."""
    if llm_args.backend == "pytorch":
        llm_cls = _TorchLLM
    elif llm_args.backend == "_autodeploy":
        llm_cls = AutoDeployLLM
    else:
        llm_cls = _TrtLLM

    # TODO: llm_args is converted to a dict here and then immediately parsed back into an LlmArgs instance
    # in the LLM constructor, which is a bit hacky. This is due to the fact that the LLM constructor needs to
    # accept kwargs (rather than an LlmArgs instance) for the offline API.
    llm = llm_cls(**llm_args.model_dump())
    return llm
