"""Main entrypoint to build, test, and prompt AutoDeploy inference models."""

import argparse
import json
from typing import List, Optional, Union

import torch
from simple_config import SimpleConfig

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy.models import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.shim import DemoLLM
from tensorrt_llm._torch.auto_deploy.utils.benchmark import benchmark, store_benchmark_results
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.llm_args import TorchCompileConfig
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
    # TODO: let's see if prefetching can't be done through the LLM api?
    # I believe the "classic workflow" invoked via the LLM api can do that.
    # put everything into the HF model Factory and try pre-fetching the checkpoint
    factory = ModelFactoryRegistry.get(config.model_factory)(
        model=config.model,
        model_kwargs=config.model_kwargs,
        tokenizer=config.tokenizer,
        tokenizer_kwargs=config.tokenizer_kwargs,
        skip_loading_weights=config.skip_loading_weights,
    )
    ad_logger.info(f"Prefetched model : {factory.model}")

    # construct llm high-level interface object
    llm_lookup = {
        "demollm": DemoLLM,
        "trtllm": LLM,
    }
    llm = llm_lookup[config.runtime](
        model=factory.model,
        backend="_autodeploy",
        max_seq_len=config.max_seq_len,
        max_batch_size=config.max_batch_size,
        # AutoDeploy-specific parameters
        use_cuda_graph=config.compile_backend in ["torch-opt", "torch-cudagraph"],
        torch_compile_config=TorchCompileConfig()
        if config.compile_backend in ["torch-opt", "torch-compile"]
        else None,
        model_factory=config.model_factory,
        model_kwargs=config.model_kwargs,
        attn_backend=config.attn_backend,
        mla_backend=config.mla_backend,
        skip_loading_weights=config.skip_loading_weights,
        cuda_graph_max_batch_size=config.max_batch_size,
        free_mem_ratio=config.free_mem_ratio,
        simple_shard_only=config.simple_shard_only,
        attn_page_size=config.attn_page_size,  # Now passed directly as AutoDeploy parameter
        tensor_parallel_size=config.world_size,
        tokenizer=factory.init_tokenizer() if config.customize_tokenizer else None,
        checkpoint_device=config.checkpoint_device,
    )

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


@torch.inference_mode()
def main(config: Optional[SimpleConfig] = None):
    if config is None:
        config = get_config_and_check_args()

    llm = build_llm_from_config(config)

    # prompt the model and print its output
    ad_logger.info("Running example prompts...")
    outs = llm.generate(
        config.prompt,
        sampling_params=SamplingParams(
            max_tokens=config.max_tokens,
            top_k=config.top_k,
            temperature=config.temperature,
        ),
    )
    results = {"prompts_and_outputs": print_outputs(outs)}

    # run a benchmark for the model with batch_size == config.benchmark_bs
    if config.benchmark and config.runtime != "trtllm":
        ad_logger.info("Running benchmark...")
        keys = [
            "compile_backend",
            "attn_backend",
            "mla_backend",
            "benchmark_bs",
            "benchmark_isl",
            "benchmark_osl",
            "benchmark_num",
        ]
        results["benchmark_results"] = benchmark(
            func=lambda: llm.generate(
                torch.randint(0, 100, (config.benchmark_bs, config.benchmark_isl)).tolist(),
                sampling_params=SamplingParams(
                    max_tokens=config.benchmark_osl,
                    top_k=None,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            ),
            num_runs=config.benchmark_num,
            log_prefix="Benchmark with " + ", ".join(f"{k}={getattr(config, k)}" for k in keys),
            results_path=config.benchmark_results_path,
        )
    elif config.benchmark:
        ad_logger.info("Skipping simple benchmarking for trtllm...")

    if config.benchmark_store_results:
        store_benchmark_results(results, config.benchmark_results_path)

    llm.shutdown()


if __name__ == "__main__":
    main()
