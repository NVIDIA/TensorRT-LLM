import os
import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm import Mapping
from tensorrt_llm._utils import mpi_barrier
from tensorrt_llm.hlapi.utils import print_traceback_on_error
from tensorrt_llm.models import LLaMAForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.llm_data
from utils.llm_data import llm_models_root

# MPIPoolExecutor only serializes function name and let workers find it in Python path.
# Since all tests are not installed in Python path, workers will fail.
# As workaround, use cloudpickle to serialize current module and all dependencies in tests folder,
# so workers can access them directly instead of searching Python path.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(utils.llm_data)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

tensorrt_llm.logger.set_level('verbose')
TP_SIZE = 2


# 76s on ipp1-1197, loading weights 18s (varies based on network speed), network/engine creation 27s
@print_traceback_on_error
def build_and_run_tp2(rank, model_name):
    '''Do not save the engine, all in one LLaMAForCausalLM object
    '''
    input_text = [
        'Born in north-east France, Soyer trained as a',
        "What is large language model?"
    ]
    default_output = [
        "chef in Paris and London before moving to New York",
        "\nLarge language model is a model that is"
    ]
    expected_outputs = {
        "llama-models/llama-7b-hf":
        default_output,
        "Mixtral-8x7B-v0.1": [
            "painter in Paris and then in Rome. He was",
            "\n\nLarge language models (LLMs) are"
        ]
    }
    expected_output = expected_outputs.get(model_name, default_output)

    tensorrt_llm.logger.set_level('verbose')
    torch.cuda.set_device(rank)

    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = llm_models_root() / model_name
    tokenizer_dir = hf_model_dir
    mapping = Mapping(world_size=TP_SIZE, rank=rank, tp_size=TP_SIZE)

    # build and run by one llama object
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir,
                                               'float16',
                                               mapping=mapping)
    llama.to_trt(max_batch_size, max_isl, max_osl, strongly_typed=True)
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


@pytest.mark.parametrize("model_name",
                         ["llama-models/llama-7b-hf", "Mixtral-8x7B-v0.1"])
def test_multi_gpu(model_name):
    if torch.cuda.device_count() < TP_SIZE:
        print(f"The test needs at least ${TP_SIZE} GPUs, skipping")
        return
    with MPIPoolExecutor(max_workers=TP_SIZE) as executor:
        results = executor.map(build_and_run_tp2, (0, 1), [model_name] * 2)
        for r in results:
            assert r == True


if __name__ == "__main__":
    pytest.main(sys.argv)
