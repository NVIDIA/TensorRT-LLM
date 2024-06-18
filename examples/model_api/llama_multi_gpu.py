import argparse
import os
from pathlib import Path

from cuda import cudart
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm import BuildConfig, Mapping, build, mpi_barrier
from tensorrt_llm.executor import GenerationExecutorWorker, SamplingParams
from tensorrt_llm.models import LLaMAForCausalLM


def dataset():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    return input_text


def build_and_run_llama(hf_model_dir, engine_dir, tp_size, rank):
    tensorrt_llm.logger.set_level('verbose')
    status, = cudart.cudaSetDevice(rank)
    assert status == cudart.cudaError_t.cudaSuccess, f"cuda set device to {rank} errored: {status}"

    ## Build engine
    build_config = BuildConfig(max_input_len=256,
                               max_seq_len=512,
                               max_batch_size=8)
    build_config.builder_opt = 0  # fast build for demo, pls avoid using this in production, since inference might be slower
    build_config.plugin_config.gemm_plugin = 'float16'  # for fast build, tune inference perf based on your needs
    mapping = Mapping(world_size=tp_size, rank=rank, tp_size=tp_size)
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, mapping=mapping)
    engine = build(llama, build_config)
    engine.save(engine_dir)
    mpi_barrier()  # make sure every rank engine build finished

    ## Generation
    tokenizer_dir = hf_model_dir
    generate_len = 20  # change on your needs, hard code for simplicity here
    executor = GenerationExecutorWorker(Path(engine_dir), tokenizer_dir)

    sampling_params = SamplingParams(max_new_tokens=generate_len)

    output_streams = executor.generate_async(dataset(),
                                             True,
                                             sampling_params=sampling_params)
    if rank == 0:
        for stream in output_streams:
            for state in stream:
                print(f"Output: {state.text}")
    mpi_barrier()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Llama single model example")
    parser.add_argument(
        "--engine_dir",
        type=str,
        required=True,
        help=
        "Directory to save and load the engine. When -c is specified, always rebuild and save to this dir. When -c is not specified, load engine when the engine_dir exists, rebuild otherwise"
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Read the model data and tokenizer from this directory")
    parser.add_argument("-n",
                        "--tp_size",
                        type=int,
                        default=2,
                        help="TP size to run the model")
    return parser.parse_args()


def run_llama(args):
    status, gpus = cudart.cudaGetDeviceCount()
    assert status == 0 and gpus >= args.tp_size, f"The test needs at least {args.tp_size} GPUs, skipping"

    if not os.path.exists(args.engine_dir):
        os.makedirs(args.engine_dir, exist_ok=True)

    with MPIPoolExecutor(max_workers=args.tp_size) as pool:
        results = pool.map(build_and_run_llama,
                           [args.hf_model_dir] * args.tp_size,
                           [args.engine_dir] * args.tp_size,
                           [args.tp_size] * args.tp_size, range(args.tp_size))
        for r in results:
            assert r == True


if __name__ == "__main__":
    args = parse_args()
    run_llama(args)
