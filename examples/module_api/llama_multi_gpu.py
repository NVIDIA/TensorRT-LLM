import sys

import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm import Mapping
from tensorrt_llm.models import LLaMAForCausalLM


def dataset():
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    return input_text


def build_and_run_llama(hf_model_dir, tp_size, rank):
    tensorrt_llm.logger.set_level('verbose')
    torch.cuda.set_device(rank)

    tokenizer_dir = hf_model_dir
    max_batch_size, max_isl, max_osl = 8, 256, 256

    mapping = Mapping(world_size=tp_size, rank=rank, tp_size=tp_size)
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               dtype='float16',
                                               mapping=mapping)
    llama.to_trt(max_batch_size, max_isl, max_osl)
    MPI.COMM_WORLD.Barrier()  # make sure every rank engine build finished

    generate_len = 10  # change on your needs, hard code for simplicity here
    for (inp, output) in llama._generate(dataset(),
                                         generate_len,
                                         tokenizer_dir=tokenizer_dir):
        print(f"Rank {rank}: input:{inp} output: {output}")
        MPI.COMM_WORLD.Barrier()

    MPI.COMM_WORLD.Barrier()
    return True


def run_llama():
    hf_model_dir = sys.argv[1] if len(
        sys.argv
    ) >= 2 else "/home/scratch.trt_llm_data/llm-models/llama-models/llama-7b-hf"

    tp_size = sys.argv[2] if len(sys.argv) >= 3 else 2  # default TP2
    assert torch.cuda.device_count(
    ) >= tp_size, f"The test needs at least {tp_size} GPUs, skipping"

    with MPIPoolExecutor(max_workers=2) as executor:
        results = executor.map(build_and_run_llama, [hf_model_dir] * tp_size,
                               [tp_size] * tp_size, range(tp_size))
        for r in results:
            assert r == True


if __name__ == "__main__":
    run_llama()
