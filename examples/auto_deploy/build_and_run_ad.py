"""Main entrypoint to build, test, and prompt AutoDeploy inference models."""

from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag

from tensorrt_llm._torch.auto_deploy import LLM, DemoLLM, LlmArgs
from tensorrt_llm._torch.auto_deploy.llm_args import _try_decode_dict_with_str_values
from tensorrt_llm._torch.auto_deploy.utils.benchmark import benchmark, store_benchmark_results
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.sampling_params import SamplingParams

# Global torch config, set the torch compile cache to fix up to llama 405B
torch._dynamo.config.cache_size_limit = 20


class PromptConfig(BaseModel):
    """Prompt configuration."""

    batch_size: int = Field(default=2, description="Number of queries")
    queries: Union[str, List[str]] = Field(
        default_factory=lambda: [
            "How big is the universe? ",
            "In simple words and in a single sentence, explain the concept of gravity: ",
            "How to fix slicing in golf? ",
            "Where is the capital of Iceland? ",
            "How big is the universe? ",
            "In simple words and in a single sentence, explain the concept of gravity: ",
            "How to fix slicing in golf? ",
            "Where is the capital of Iceland? ",
        ]
    )
    sp_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"max_tokens": 100, "top_k": 200, "temperature": 1.0},
        description="Sampling parameter kwargs passed on the SamplingParams class. "
        "Defaults are set to the values used in the original model.",
    )

    def model_post_init(self, __context: Any):
        """Cut queries to batch_size.

        NOTE (lucaslie): has to be done with model_post_init to ensure it's always run. field
        validators are only run if a value is provided.
        """
        queries = [self.queries] if isinstance(self.queries, str) else self.queries
        batch_size = self.batch_size
        queries = queries * (batch_size // len(queries) + 1)
        self.queries = queries[:batch_size]

    @field_validator("sp_kwargs", mode="after")
    @classmethod
    def validate_sp_kwargs(cls, sp_kwargs):
        """Insert desired defaults for sampling params and try parsing string values as JSON."""
        sp_kwargs = {**cls.model_fields["sp_kwargs"].default_factory(), **sp_kwargs}
        sp_kwargs = _try_decode_dict_with_str_values(sp_kwargs)
        return sp_kwargs


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    enabled: bool = Field(default=False, description="If true, run simple benchmark")
    num: int = Field(default=10, ge=1, description="By default run 10 times and get average")
    isl: int = Field(default=2048, ge=1, description="Input seq length for benchmarking")
    osl: int = Field(default=128, ge=1, description="Output seq length for benchmarking")
    bs: int = Field(default=1, ge=1, description="Batch size for benchmarking")
    results_path: Optional[str] = Field(default="./benchmark_results.json")
    store_results: bool = Field(
        default=False, description="If True, store benchmark res in benchmark_results_path"
    )


class ExperimentConfig(BaseSettings):
    """Experiment Configuration based on Pydantic BaseModel."""

    model_config = ConfigDict(
        extra="forbid",
        cli_kebab_case=True,
    )

    ### CORE ARGS ##################################################################################
    # The main LLM arguments - contains model, tokenizer, backend configs, etc.
    args: LlmArgs = Field(
        description="The main LLM arguments containing model, tokenizer, backend configs, etc."
    )

    # Optional model field for convenience - if provided, will be used to initialize args.model
    model: Optional[str] = Field(
        default=None,
        description="The path to the model checkpoint or the model name from the Hugging Face Hub. "
        "If provided, will be passed through to initialize args.model",
    )

    ### SIMPLE PROMPTING CONFIG ####################################################################
    prompt: PromptConfig = Field(default_factory=PromptConfig)

    ### BENCHMARKING CONFIG ########################################################################
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    ### CONFIG DEBUG FLAG ##########################################################################
    dry_run: CliImplicitFlag[bool] = Field(default=False, description="Show final config and exit")

    ### VALIDATION #################################################################################
    @model_validator(mode="before")
    @classmethod
    def setup_args_from_model(cls, data: Dict) -> Dict:
        """Check for model being provided directly or via args.model."""
        msg = '"model" must be provided directly or via "args.model"'
        if not isinstance(data, dict):
            raise ValueError(msg)
        if not ("model" in data or "model" in data.get("args", {})):
            raise ValueError(msg)

        data["args"] = data.get("args", {})
        if "model" in data:
            data["args"]["model"] = data["model"]
        return data

    @field_validator("model", mode="after")
    @classmethod
    def sync_model_with_args(cls, model_value, info):
        args: LlmArgs = info.data["args"]
        return args.model if args is not None else model_value

    @field_validator("prompt", mode="after")
    @classmethod
    def sync_prompt_batch_size_with_args_max_batch_size(cls, prompt: PromptConfig, info):
        args: LlmArgs = info.data["args"]
        if args.max_batch_size < prompt.batch_size:
            args.max_batch_size = prompt.batch_size
        return prompt

    @field_validator("benchmark", mode="after")
    @classmethod
    def adjust_args_for_benchmark(cls, benchmark: BenchmarkConfig, info):
        args: LlmArgs = info.data["args"]
        if benchmark.enabled:
            # propagate benchmark settings to args
            args.max_batch_size = max(benchmark.bs, args.max_batch_size)
            args.max_seq_len = max(args.max_seq_len, benchmark.isl + benchmark.osl)
        return benchmark


def build_llm_from_config(config: ExperimentConfig) -> LLM:
    """Builds a LLM object from our config."""
    # construct llm high-level interface object
    llm_lookup = {
        "demollm": DemoLLM,
        "trtllm": LLM,
    }
    ad_logger.info(f"{config.args._parallel_config=}")
    llm = llm_lookup[config.args.runtime](**config.args.to_dict())
    return llm


def print_outputs(outs: Union[RequestOutput, List[RequestOutput]]) -> List[List[str]]:
    prompts_and_outputs: List[List[str]] = []
    if isinstance(outs, RequestOutput):
        outs = [outs]
    for i, out in enumerate(outs):
        prompt, output = out.prompt, out.outputs[0].text
        ad_logger.info(f"[PROMPT {i}] {prompt}: {output}")
        prompts_and_outputs.append([prompt, output])
    return prompts_and_outputs


def main(config: Optional[ExperimentConfig] = None):
    if config is None:
        config = CliApp.run(ExperimentConfig)
    ad_logger.info(f"{config=}")

    if config.dry_run:
        return

    llm = build_llm_from_config(config)

    # prompt the model and print its output
    ad_logger.info("Running example prompts...")
    outs = llm.generate(
        config.prompt.queries,
        sampling_params=SamplingParams(**config.prompt.sp_kwargs),
    )
    results = {"prompts_and_outputs": print_outputs(outs)}

    # run a benchmark for the model with batch_size == config.benchmark_bs
    if config.benchmark.enabled and config.args.runtime != "trtllm":
        ad_logger.info("Running benchmark...")
        keys_from_args = ["compile_backend", "attn_backend", "mla_backend"]
        fields_to_show = [f"benchmark={config.benchmark}"]
        fields_to_show.extend([f"{k}={getattr(config.args, k)}" for k in keys_from_args])
        results["benchmark_results"] = benchmark(
            func=lambda: llm.generate(
                torch.randint(0, 100, (config.benchmark.bs, config.benchmark.isl)).tolist(),
                sampling_params=SamplingParams(
                    max_tokens=config.benchmark.osl,
                    top_k=None,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            ),
            num_runs=config.benchmark.num,
            log_prefix="Benchmark with " + ", ".join(fields_to_show),
            results_path=config.benchmark.results_path,
        )
    elif config.benchmark.enabled:
        ad_logger.info("Skipping simple benchmarking for trtllm...")

    if config.benchmark.store_results:
        store_benchmark_results(results, config.benchmark.results_path)

    llm.shutdown()


if __name__ == "__main__":
    main()
