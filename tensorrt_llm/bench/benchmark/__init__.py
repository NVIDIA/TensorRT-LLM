from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.logger import logger


def ignore_trt_only_args(kwargs: dict, backend: str):
    trt_only_args = [
        "batching_type",
        "normalize_log_probs",
        "extended_runtime_perf_knob_config",
    ]
    for arg in trt_only_args:
        if kwargs.pop(arg, None):
            logger.warning(f"Ignore {arg} for {backend} backend.")


def get_llm(runtime_config: RuntimeConfig, kwargs: dict):
    llm_cls = LLM

    if runtime_config.backend == 'pytorch':
        ignore_trt_only_args(kwargs)

        if runtime_config.iteration_log is not None:
            kwargs["enable_iter_perf_stats"] = True

        llm_cls = PyTorchLLM
    elif runtime_config.backend == "_autodeploy":
        ignore_trt_only_args(kwargs)
        kwargs["world_size"] = kwargs.pop("tensor_parallel_size", None)

        if runtime_config.iteration_log is not None:
            kwargs["enable_iter_perf_stats"] = True

        llm_cls = AutoDeployLLM

    llm = llm_cls(**kwargs)
    return llm
