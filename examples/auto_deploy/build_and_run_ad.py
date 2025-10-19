"""Main entrypoint to build, test, and prompt AutoDeploy inference models."""

from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliImplicitFlag,
    CliUnknownArgs,
    SettingsConfigDict,
)

from tensorrt_llm._torch.auto_deploy import LLM, AutoDeployConfig, DemoLLM
from tensorrt_llm._torch.auto_deploy.utils._config import (
    DynamicYamlMixInForSettings,
    deep_merge_dicts,
)
from tensorrt_llm._torch.auto_deploy.utils.benchmark import benchmark, store_benchmark_results
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.sampling_params import SamplingParams

# Global torch config, set the torch compile cache to fix up to llama 405B
torch._dynamo.config.cache_size_limit = 20

# simple string, TRT-LLM style text-only prompt or full-scale HF message template
PromptInput = Union[str, Dict, List[Dict]]


class PromptConfig(BaseModel):
    """Prompt configuration.

    This configuration class can be used for this example script to configure the example prompts
    and the sampling parameters.
    """

    batch_size: int = Field(default=2, description="Number of queries")
    queries: Union[PromptInput, List[PromptInput]] = Field(
        default_factory=lambda: [
            # OPTION 1: simple text prompt
            "How big is the universe? ",
            # OPTION 2: wrapped text prompt for TRT-LLM
            {"prompt": "In simple words and a single sentence, explain the concept of gravity: "},
            # OPTION 3: a full-scale HF message template (this one works for text-only models!)
            # Learn more about chat templates: https://huggingface.co/docs/transformers/en/chat_templating
            # and multi-modal templates: https://huggingface.co/docs/transformers/en/chat_templating_multimodal
            [
                {
                    "role": "user",
                    "content": "How to fix slicing in golf?",
                }
            ],
            # More prompts...
            {"prompt": "Where is the capital of Iceland? "},
        ],
        description="Example queries to prompt the model with. We support both TRT-LLM text-only "
        "queries via the 'prompt' key and full-scale HF message template called via "
        "apply_chat_template.",
    )
    sp_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"max_tokens": 100, "top_k": None, "temperature": 1.0},
        description="Sampling parameter kwargs passed on the SamplingParams class. "
        "Defaults are set to the values used in the original model.",
    )

    def model_post_init(self, __context: Any):
        """Cut queries to batch_size.

        NOTE (lucaslie): has to be done with model_post_init to ensure it's always run. field
        validators are only run if a value is provided.
        """
        queries = self.queries if isinstance(self.queries, list) else [self.queries]
        batch_size = self.batch_size
        queries = queries * (batch_size // len(queries) + 1)
        queries = queries[:batch_size]

        # now let's standardize the queries for the LLM api to understand them
        queries_processed = []
        for query in queries:
            if isinstance(query, str):
                queries_processed.append({"prompt": query})
            elif isinstance(query, dict):
                queries_processed.append(query)
            elif isinstance(query, list):
                queries_processed.append(
                    {
                        "prompt": "Fake prompt. Check out messages field for the HF chat template.",
                        "messages": query,  # contains the actual HF chat template
                    }
                )
            else:
                raise ValueError(f"Invalid query type: {type(query)}")
        self.queries = queries_processed

    @field_validator("sp_kwargs", mode="after")
    @classmethod
    def validate_sp_kwargs(cls, sp_kwargs):
        """Insert desired defaults for sampling params and try parsing string values as JSON."""
        default = cls.model_fields["sp_kwargs"].get_default(call_default_factory=True)
        return deep_merge_dicts(default, sp_kwargs)


class BenchmarkConfig(BaseModel):
    """Benchmark configuration.

    This configuration class can be used for this example script to configure the simple
    benchmarking we run at the end of the script.
    """

    enabled: bool = Field(default=False, description="If true, run simple benchmark")
    num: int = Field(default=10, ge=1, description="By default run 10 times and get average")
    isl: int = Field(default=2048, ge=1, description="Input seq length for benchmarking")
    osl: int = Field(default=128, ge=1, description="Output seq length for benchmarking")
    bs: int = Field(default=1, ge=1, description="Batch size for benchmarking")
    results_path: Optional[str] = Field(default="./benchmark_results.json")
    store_results: bool = Field(
        default=False, description="If True, store benchmark res in benchmark_results_path"
    )


class ExperimentConfig(DynamicYamlMixInForSettings, BaseSettings):
    """Experiment Configuration for the example script.

    This configuration aggregates all relevant configurations for this example script. It is also
    used to auto-generate the CLI interface.
    """

    model_config = SettingsConfigDict(
        extra="forbid",
        cli_kebab_case=True,
        cli_ignore_unknown_args=True,
        nested_model_default_partial_update=True,
    )
    extra_cli_args: CliUnknownArgs

    ### CORE ARGS ##################################################################################
    # The main AutoDeploy arguments - contains model, tokenizer, backend configs, etc.
    args: AutoDeployConfig = Field(
        description="The main AutoDeploy arguments containing model, tokenizer, backend configs, etc. "
        "Please check `tensorrt_llm._torch.auto_deploy.llm_args.AutoDeployConfig` for more details."
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

    @model_validator(mode="before")
    @classmethod
    def process_extra_cli_args(cls, data: Dict) -> Dict:
        """Process extra CLI args.

        This model validator enables the user to provide additional CLI args that may not be
        auto-generated by the CLI app. A common use case for this would to modify graph transforms
        dynamically via CLI arguments.

        For example, the user can provide a CLI argument for raw dictionaries like this, e.g., for
        ``model_kwargs``: ``--args.model-kwargs.num-hidden-layers=10``.
        """
        # build a clean dotlist: ["a.b=1","c.d.e=foo",…]
        raw: List[str] = data.pop("extra_cli_args", [])
        dotlist = []
        it: Iterator[str] = iter(raw)
        for tok in it:
            if not tok.startswith("--"):
                continue
            body = tok[2:]
            if "=" in body:
                body, val = body.split("=", 1)
            else:
                # flag + separate value
                val = next(it, None)
            # ensure kebab-case is converted to snake_case
            dotlist.append(f"{body.replace('-', '_')}={val}")

        return deep_merge_dicts(data, OmegaConf.from_dotlist(dotlist))

    @field_validator("model", mode="after")
    @classmethod
    def sync_model_with_args(cls, model_value, info):
        if "args" not in info.data:
            return model_value
        args: AutoDeployConfig = info.data["args"]
        return args.model

    @field_validator("prompt", mode="after")
    @classmethod
    def sync_prompt_batch_size_with_args_max_batch_size(cls, prompt: PromptConfig, info):
        if "args" not in info.data:
            return prompt
        args: AutoDeployConfig = info.data["args"]
        if args.max_batch_size < prompt.batch_size:
            args.max_batch_size = prompt.batch_size
        return prompt

    @field_validator("benchmark", mode="after")
    @classmethod
    def adjust_args_for_benchmark(cls, benchmark: BenchmarkConfig, info):
        if "args" not in info.data:
            return benchmark
        args: AutoDeployConfig = info.data["args"]
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
    llm = llm_lookup[config.args.runtime](**config.args.to_llm_kwargs())
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
        config: ExperimentConfig = CliApp.run(ExperimentConfig)
    ad_logger.info(f"AutoDeploy Experiment Config:\n{yaml.dump(config.model_dump())}")

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
        keys_from_args = []
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
