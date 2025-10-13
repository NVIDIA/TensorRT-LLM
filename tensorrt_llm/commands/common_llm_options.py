from tensorrt_llm.llmapi.llm_args import BaseLlmArgs, KvCacheConfig


def get_llm_args_from_cli_params(**params: dict) -> BaseLlmArgs:
    """Construct an instance of LlmArgs from CLI params."""
    llm_kwargs = {}

    # TODO: logic to choose either TrtLlmArgs / TorchLlmArgs / etc.

    # Get valid LLM arg field names from BaseLlmArgs
    valid_llm_fields = set(BaseLlmArgs.model_fields.keys())

    for key, value in params.items():
        if key in valid_llm_fields and value is not None:
            llm_kwargs[key] = value

    # Handle special cases that map to nested configs
    if "kv_cache_free_gpu_memory_fraction" in params and params[
            "kv_cache_free_gpu_memory_fraction"] is not None:
        if "kv_cache_config" not in llm_kwargs:
            llm_kwargs["kv_cache_config"] = KvCacheConfig()
        llm_kwargs["kv_cache_config"].free_gpu_memory_fraction = params[
            "kv_cache_free_gpu_memory_fraction"]

    if "disable_kv_cache_reuse" in params:
        if "kv_cache_config" not in llm_kwargs:
            llm_kwargs["kv_cache_config"] = KvCacheConfig()
        llm_kwargs["kv_cache_config"].enable_block_reuse = not params[
            "disable_kv_cache_reuse"]

    return llm_kwargs
