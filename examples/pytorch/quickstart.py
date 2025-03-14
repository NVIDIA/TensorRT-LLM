import argparse

from datasets import load_dataset

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bindings.executor import KvCacheConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--enable_overlap_scheduler',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_chunked_prefill',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_attention_dp',
                        default=False,
                        action='store_true')
    parser.add_argument('--kv_cache_enable_block_reuse',
                        default=False,
                        action='store_true')
    parser.add_argument('--kv_cache_dtype', type=str, default='auto')
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--moe_tp_size', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    pytorch_config = PyTorchConfig(
        enable_overlap_scheduler=args.enable_overlap_scheduler,
        kv_cache_dtype=args.kv_cache_dtype)

    llm = LLM(model=args.model_dir,
              tensor_parallel_size=args.tp_size,
              enable_chunked_prefill=args.enable_chunked_prefill,
              pytorch_backend_config=pytorch_config,
              moe_expert_parallel_size=args.moe_ep_size,
              moe_tensor_parallel_size=args.moe_tp_size,
              enable_attention_dp=args.enable_attention_dp,
              kv_cache_config=KvCacheConfig(
                  enable_block_reuse=args.kv_cache_enable_block_reuse))

    dataset_name = "ccdv/cnn_dailymail"
    dataset_revision = "3.0.0"
    dataset_input_key = 'article'
    dataset_split = 'test'

    dataset = load_dataset(dataset_name, dataset_revision, split=dataset_split)

    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is",
        "Please summarize the following article: " +
        dataset[0:1][dataset_input_key][0]
    ] * 8
    sampling_params = SamplingParams(max_tokens=32,
                                     end_id=100000,
                                     pad_id=100001)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, \nGenerated text: {generated_text!r}")


if __name__ == '__main__':
    main()
