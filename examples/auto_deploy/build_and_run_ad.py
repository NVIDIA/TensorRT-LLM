"""Main entrypoint to build, test, and prompt AutoDeploy inference models."""

import argparse
import json
from typing import List, Optional, Union

import torch
from simple_config import SimpleConfig

from tensorrt_llm._torch.auto_deploy.models import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.shim import AutoDeployConfig, DemoLLM
from tensorrt_llm._torch.auto_deploy.utils.benchmark import benchmark
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm import LLM, RequestOutput
from tensorrt_llm.sampling_params import SamplingParams

# Global torch config, set the torch compile cache to fix up to llama 405B
torch._dynamo.config.cache_size_limit = 20


def get_config_and_check_args() -> SimpleConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=json.loads)
    parser.add_argument("-m", "--model-kwargs", type=json.loads)
    args = parser.parse_args()
    configs_from_args = args.config or {}
    configs_from_args["model_kwargs"] = getattr(args, "model_kwargs") or {}
    config = SimpleConfig(**configs_from_args)
    ad_logger.info(f"Simple Config: {config}")
    return config


def build_llm_from_config(config: SimpleConfig) -> LLM:
    """Builds a LLM object from our config."""
    # set up builder config
    build_config = BuildConfig(max_seq_len=config.max_seq_len, max_batch_size=config.max_batch_size)
    build_config.plugin_config.tokens_per_block = config.page_size

    # setup AD config
    ad_config = AutoDeployConfig(
        use_cuda_graph=config.compile_backend == "torch-opt",
        torch_compile_enabled=config.compile_backend == "torch-opt",
        model_kwargs=config.model_kwargs,
        attn_backend=config.attn_backend,
        skip_loading_weights=config.skip_loading_weights,
        cuda_graph_max_batch_size=config.max_batch_size,
    )
    ad_logger.info(f"AutoDeploy Config: {ad_config}")

    # TODO (lliebenwein): let's see if prefetching can't be done through the LLM api?
    # I believe the "classic workflow" invoked via the LLM api can do that.
    # put everything into the HF model Factory and try pre-fetching the checkpoint
    factory = ModelFactoryRegistry.get("hf")(model=config.model, model_kwargs=config.model_kwargs)

    # construct llm high-level interface object
    llm_lookup = {
        "demollm": DemoLLM,
        "trtllm": LLM,
    }
    llm = llm_lookup[config.runtime](
        model=factory.ckpt_path,
        backend="autodeploy",
        build_config=build_config,
        pytorch_backend_config=ad_config,
        tensor_parallel_size=config.world_size,
    )

    return llm


def print_outputs(outs: Union[RequestOutput, List[RequestOutput]]):
    if isinstance(outs, RequestOutput):
        outs = [outs]
    for i, out in enumerate(outs):
        ad_logger.info(f"[PROMPT {i}] {out.prompt}: {out.outputs[0].text}")


@torch.inference_mode()
def main(config: Optional[SimpleConfig] = None):
    if config is None:
        config = get_config_and_check_args()

    llm = build_llm_from_config(config)

    # prompt the model and print its output
    outs = llm.generate(
        config.prompt,
        sampling_params=SamplingParams(
            max_tokens=config.max_tokens,
            top_k=config.top_k,
            temperature=config.temperature,
        ),
    )
    print_outputs(outs)

    # run a benchmark for the model with batch_size == config.benchmark_bs
    if config.benchmark:
        token_ids = torch.randint(0, 100, (config.benchmark_bs, config.benchmark_isl)).tolist()
        sampling_params = SamplingParams(max_tokens=config.benchmark_osl, top_k=None)
        keys = ["compile_backend", "attn_backend"]
        benchmark(
            lambda: llm.generate(token_ids, sampling_params=sampling_params, use_tqdm=False),
            config.benchmark_num,
            "Benchmark with " + ", ".join(f"{k}={getattr(config, k)}" for k in keys),
            results_path=config.benchmark_results_path,
        )


if __name__ == "__main__":
    main()
