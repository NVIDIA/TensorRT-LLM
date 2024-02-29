import argparse
import os

import torch
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm import Mapping, mpi_barrier
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.models import LLaMAForCausalLM


def dataset():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    return input_text


def build_and_run_llama(hf_model_dir, engine_dir, tp_size, rank, clean_build):
    tensorrt_llm.logger.set_level('verbose')
    torch.cuda.set_device(rank)

    tokenizer_dir = hf_model_dir
    max_batch_size, max_isl, max_osl = 8, 256, 256

    mapping = Mapping(world_size=tp_size, rank=rank, tp_size=tp_size)
    if clean_build or not os.path.exists(engine_dir):
        os.makedirs(engine_dir, exist_ok=True)
        llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                                   dtype='float16',
                                                   mapping=mapping)
        llama.to_trt(max_batch_size, max_isl, max_osl)
        llama.save(engine_dir)
    mpi_barrier()  # make sure every rank engine build finished

    generate_len = 20  # change on your needs, hard code for simplicity here
    executor = GenerationExecutor(engine_dir, tokenizer_dir)

    output_streams = executor.generate_async(dataset(), True,
                                             [generate_len] * len(dataset()))
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
    parser.add_argument(
        "-c",
        "--clean_build",
        default=False,
        action="store_true",
        help=
        "Clean build the engine even if the engine_dir exists, be careful, this overwrites the engine_dir!!"
    )
    parser.add_argument("-n",
                        "--tp_size",
                        type=int,
                        default=2,
                        help="TP size to run the model")
    return parser.parse_args()


def run_llama(args):
    assert torch.cuda.device_count(
    ) >= args.tp_size, f"The test needs at least {args.tp_size} GPUs, skipping"

    with MPIPoolExecutor(max_workers=args.tp_size) as pool:
        results = pool.map(build_and_run_llama,
                           [args.hf_model_dir] * args.tp_size,
                           [args.engine_dir] * args.tp_size,
                           [args.tp_size] * args.tp_size, range(args.tp_size),
                           [args.clean_build] * args.tp_size)
        for r in results:
            assert r == True


if __name__ == "__main__":
    args = parse_args()
    run_llama(args)
