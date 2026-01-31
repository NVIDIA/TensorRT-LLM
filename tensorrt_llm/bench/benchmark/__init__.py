import json
from pathlib import Path
from typing import Callable, Dict, Optional

from pydantic import AliasChoices, BaseModel, Field

from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter
from tensorrt_llm.bench.build.build import get_model_config
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.logger import logger


class GeneralExecSettings(BaseModel):
    model_config = {
        "extra": "ignore"
    }  # Ignore extra fields not defined in the model

    backend: str = Field(
        default="pytorch",
        description="The backend to use when running benchmarking")
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
    max_input_len: int = Field(default=4096,
                               description="Maximum input sequence length")
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
