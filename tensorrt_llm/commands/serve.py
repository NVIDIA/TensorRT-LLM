import asyncio
from typing import Optional

import click
from transformers import AutoTokenizer

from tensorrt_llm.llmapi import LLM, BuildConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_utils import LlmArgs
from tensorrt_llm.serve import OpenAIServer


@click.command("trtllm-serve")
@click.argument("model", type=str)
@click.option("--tokenizer",
              type=str,
              default=None,
              help="Path | Name of the tokenizer."
              "Specify this value only if using TensorRT engine as model.")
@click.option("--host",
              type=str,
              default="localhost",
              help="Hostname of the server.")
@click.option("--port", type=int, default=8000, help="Port of the server.")
@click.option("--backend",
              type=click.Choice(["pytorch"]),
              default=None,
              help="Set to 'pytorch' for pytorch path. Default is cpp path.")
@click.option("--max_beam_width",
              type=int,
              default=BuildConfig.max_beam_width,
              help="Maximum number of beams for beam search decoding.")
@click.option("--max_batch_size",
              type=int,
              default=BuildConfig.max_batch_size,
              help="Maximum number of requests that the engine can schedule.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=BuildConfig.max_num_tokens,
    help=
    "Maximum number of batched input tokens after padding is removed in each batch."
)
@click.option(
    "--max_seq_len",
    type=int,
    default=BuildConfig.max_seq_len,
    help="Maximum total length of one request, including prompt and outputs. "
    "If unspecified, the value is deduced from the model config.")
@click.option("--tp_size", type=int, default=1, help='Tensor parallelism size.')
@click.option("--pp_size",
              type=int,
              default=1,
              help='Pipeline parallelism size.')
@click.option("--ep_size",
              type=int,
              default=None,
              help="expert parallelism size")
@click.option("--gpus_per_node",
              type=int,
              default=None,
              help="Number of GPUs per node. Default to None, and it will be "
              "detected automatically.")
@click.option("--kv_cache_free_gpu_memory_fraction",
              type=float,
              default=0.9,
              help="Free GPU memory fraction reserved for KV Cache, "
              "after allocating model weights and buffers.")
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
def main(model: str, tokenizer: str, host: str, port: int, backend: str,
         max_beam_width: int, max_batch_size: int, max_num_tokens: int,
         max_seq_len: int, tp_size: int, pp_size: int, ep_size: Optional[int],
         gpus_per_node: Optional[int], kv_cache_free_gpu_memory_fraction: float,
         trust_remote_code: bool):
    """Running an OpenAI API compatible server

    MODEL: model name | HF checkpoint path | TensorRT engine path
    """
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_num_tokens=max_num_tokens,
                               max_beam_width=max_beam_width,
                               max_seq_len=max_seq_len)

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction)

    llm_args = LlmArgs.from_kwargs(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        moe_expert_parallel_size=ep_size,
        gpus_per_node=gpus_per_node,
        trust_remote_code=trust_remote_code,
        build_config=build_config,
        kv_cache_config=kv_cache_config,
        backend=backend if backend == "pytorch" else None)

    llm = LLM(**llm_args.to_dict())

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer or model)

    server = OpenAIServer(llm=llm, model=model, hf_tokenizer=hf_tokenizer)

    asyncio.run(server(host, port))


if __name__ == "__main__":
    main()
