import json
from pathlib import Path
from typing import Callable, Dict, Optional

from pydantic import AliasChoices, BaseModel, Field

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter
from tensorrt_llm.bench.build.build import get_model_config
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.logger import logger


class GeneralExecSettings(BaseModel):
    model_config = {
        "extra": "ignore"
    }  # Ignore extra fields not defined in the model

    backend: str = Field(
        default="pytorch",
        description="The backend to use when running benchmarking")
    beam_width: int = Field(default=1, description="Number of search beams")
    model_path: Optional[Path] = Field(default=None,
                                       description="Path to model checkpoint")
    concurrency: int = Field(
        default=-1, description="Desired concurrency rate, <=0 for no limit")
    dataset_path: Optional[Path] = Field(default=None,
                                         validation_alias=AliasChoices(
                                             "dataset_path", "dataset"),
                                         description="Path to dataset file")
    engine_dir: Optional[Path] = Field(
        default=None, description="Path to a serialized TRT-LLM engine")
    eos_id: int = Field(
        default=-1, description="End-of-sequence token ID, -1 to disable EOS")
    iteration_log: Optional[Path] = Field(
        default=None, description="Path where iteration logging is written")
    kv_cache_percent: float = Field(
        default=0.90,
        validation_alias=AliasChoices("kv_cache_percent",
                                      "kv_cache_free_gpu_mem_fraction"),
        description="Percentage of memory for KV Cache after model load")
    max_input_len: int = Field(default=4096,
                               description="Maximum input sequence length")
    max_seq_len: Optional[int] = Field(default=None,
                                       description="Maximum sequence length")
    modality: Optional[str] = Field(
        default=None, description="Modality of multimodal requests")
    model: Optional[str] = Field(default=None, description="Model name or path")
    num_requests: int = Field(
        default=0, description="Number of requests to cap benchmark run at")
    output_json: Optional[Path] = Field(
        default=None, description="Path where output should be written")
    report_json: Optional[Path] = Field(
        default=None, description="Path where report should be written")
    request_json: Optional[Path] = Field(
        default=None,
        description="Path where per request information is written")
    streaming: bool = Field(default=False,
                            description="Whether to use streaming mode")
    warmup: int = Field(default=2,
                        description="Number of requests to warm up benchmark")

    @property
    def iteration_writer(self) -> IterationWriter:
        return IterationWriter(self.iteration_log)

    @property
    def model_type(self) -> str:
        return get_model_config(self.model, self.checkpoint_path).model_type

    @property
    def checkpoint_path(self) -> Path:
        return self.model_path or self.model


def ignore_trt_only_args(kwargs: dict, backend: str):
    """Ignore TensorRT-only arguments for non-TensorRT backends.

    Args:
        kwargs: Dictionary of keyword arguments to be passed to the LLM constructor.
        backend: The backend type (e.g., "pytorch", "_autodeploy").
    """
    trt_only_args = [
        "batching_type",
        "normalize_log_probs",
        "extended_runtime_perf_knob_config",
    ]
    for arg in trt_only_args:
        if kwargs.pop(arg, None):
            logger.warning(f"Ignore {arg} for {backend} backend.")


def get_llm(runtime_config: RuntimeConfig, kwargs: dict):
    """Create and return an appropriate LLM instance based on the backend configuration.

    Args:
        runtime_config: Runtime configuration containing backend selection and settings.
        kwargs: Additional keyword arguments to pass to the LLM constructor.

    Returns:
        An instance of the appropriate LLM class for the specified backend.
    """
    llm_cls = LLM

    if runtime_config.backend != "tensorrt":
        ignore_trt_only_args(kwargs, runtime_config.backend)

    if runtime_config.backend == 'pytorch':
        llm_cls = PyTorchLLM

        if runtime_config.iteration_log is not None:
            kwargs["enable_iter_perf_stats"] = True

    elif runtime_config.backend == "_autodeploy":
        kwargs["world_size"] = kwargs.pop("tensor_parallel_size", None)
        llm_cls = AutoDeployLLM

    llm = llm_cls(**kwargs)
    return llm


def get_general_cli_options(
        params: Dict, bench_env: BenchmarkEnvironment) -> GeneralExecSettings:
    """Get general execution settings from command line parameters.

    Args:
        params: Dictionary of command line parameters.
        bench_env: Benchmark environment containing model and checkpoint information.

    Returns:
        An instance of GeneralExecSettings containing general execution settings.
    """
    # Create a copy of params to avoid modifying the original
    settings_dict = params.copy()

    # Add derived values that need to be computed from bench_env
    model_path = bench_env.checkpoint_path
    model = bench_env.model
    # Override/add the computed values
    settings_dict.update({
        "model_path": model_path,
        "model": model,
    })

    # Create and return the settings object, ignoring any extra fields
    return GeneralExecSettings(**settings_dict)


def generate_json_report(report_path: Optional[Path], func: Callable):
    if report_path is None:
        logger.debug("No report path provided, skipping report generation.")
    else:
        logger.info(f"Writing report information to {report_path}...")
        with open(report_path, "w") as f:
            f.write(json.dumps(func(), indent=4))
        logger.info(f"Report information written to {report_path}.")
