import torch
from llm_data import llm_models_root
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm import Mapping
from tensorrt_llm._utils import mpi_barrier
from tensorrt_llm.models import LLaMAForCausalLM

tensorrt_llm.logger.set_level('verbose')


# 76s on ipp1-1197, loading weights 18s (varies based on network speed), network/engine creation 27s
def build_and_run_tp2(rank):
    '''Do not save the engine, all in one LLaMAForCausalLM object
    '''
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    expected_output = [
        "chef in Paris and London before moving to New York",
        "\nLarge language model is a model that is"
    ]

    tensorrt_llm.logger.set_level('verbose')
    TP_SIZE = 2
    torch.cuda.set_device(rank)

    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir
    mapping = Mapping(world_size=TP_SIZE, rank=rank, tp_size=TP_SIZE)

    # build and run by one llama object
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               mapping=mapping)
    llama.to_trt(max_batch_size, max_isl, max_osl)
    mpi_barrier()
    tensorrt_llm.logger.warning(f"Build finished for rank {rank}")
    for idx, (inp, output) in enumerate(
            llama._generate(input_text, 10, tokenizer_dir=tokenizer_dir)):
        # print(f"Input: {inp}")
        tensorrt_llm.logger.info(f"{rank} input: {inp}")
        # print(f'Output: {output}')
        tensorrt_llm.logger.info(f"{rank} output: {output}")
        assert output == expected_output[
            idx], f"Expecting {expected_output[idx]}, got {output}"
        mpi_barrier()
    return True


def test_multi_gpu():
    if torch.cuda.device_count() < 2:
        print("The test needs at least 2 GPUs, skipping")
        return
    with MPIPoolExecutor(max_workers=2) as executor:
        results = executor.map(build_and_run_tp2, (0, 1))
        for r in results:
            assert r == True


if __name__ == "__main__":
    test_multi_gpu()
