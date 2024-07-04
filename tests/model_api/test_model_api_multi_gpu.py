import os
import pickle
import sys
import tempfile

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm import Mapping
from tensorrt_llm._utils import mpi_barrier
from tensorrt_llm.auto_parallel import AutoParallelConfig, infer_cluster_config
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.executor import ExecutorBindingsWorker
from tensorrt_llm.hlapi.utils import SamplingParams, print_traceback_on_error
from tensorrt_llm.models import LLaMAForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import llm_data
from utils.llm_data import llm_models_root
from utils.util import force_ampere

# MPIPoolExecutor only serializes function name and let workers find it in Python path.
# Since all tests are not installed in Python path, workers will fail.
# As workaround, use cloudpickle to serialize current module and all dependencies in tests folder,
# so workers can access them directly instead of searching Python path.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(llm_data)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

tensorrt_llm.logger.set_level('verbose')
TP_SIZE = 2

batch_input_text = [
    "Born in north-east France, Soyer trained as a",
    "What is large language model?"
]
batch_output_text_expected_default = [
    "chef in Paris and London before moving to New York",
    "\nLarge language model is a model that is"
]
batch_output_text_expected_mixtral = [
    "painter in Paris and then in Rome. He was",
    "\n\nLarge language models (LLMs) are"
]


def get_batch_output_text_expected(model_name):
    if model_name == "Mixtral-8x7B-v0.1":
        return batch_output_text_expected_mixtral
    else:
        return batch_output_text_expected_default


# 76s on ipp1-1197, loading weights 18s (varies based on network speed), network/engine creation 27s
@print_traceback_on_error
def build_and_run_tp2(rank, model_name, engine_dir, use_auto_parallel):
    '''Do not save the engine, all in one LLaMAForCausalLM object
    '''
    batch_output_text_expected = get_batch_output_text_expected(model_name)

    tensorrt_llm.logger.set_level('verbose')
    torch.cuda.set_device(rank)

    max_batch_size, max_isl, max_osl = 8, 256, 256
    hf_model_dir = str(llm_models_root() / model_name)
    mapping = Mapping(world_size=TP_SIZE, rank=rank, tp_size=TP_SIZE)
    auto_parallel_config = AutoParallelConfig()
    if use_auto_parallel:
        mapping = Mapping()
        mapping.rank = rank
        auto_parallel_config = AutoParallelConfig(
            world_size=TP_SIZE,
            sharded_io_allowlist=[
                "past_key_value_\\d+",
                "present_key_value_\\d*",
            ],
            same_buffer_io={
                "past_key_value_(\\d+)": "present_key_value_\\1",
            },
            **infer_cluster_config(),
        )

    # build and run by one llama object
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, mapping=mapping)
    engine = build(
        llama,
        BuildConfig(max_batch_size=max_batch_size,
                    max_input_len=max_isl,
                    max_seq_len=max_osl + max_isl,
                    strongly_typed=True,
                    auto_parallel_config=auto_parallel_config))
    engine.save(engine_dir)
    mpi_barrier()
    tensorrt_llm.logger.warning(f"Build finished for rank {rank}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
    with ExecutorBindingsWorker(engine_dir) as executor:
        executor.block_subordinates()

        batch_input_ids = [tokenizer.encode(inp) for inp in batch_input_text]
        outputs = executor.generate(
            batch_input_ids, sampling_params=SamplingParams(max_new_tokens=10))

        for idx, output in enumerate(outputs):
            tensorrt_llm.logger.info(f"{rank} input: {batch_input_text[idx]}")
            output_text = tokenizer.decode(output.outputs[0].token_ids)
            tensorrt_llm.logger.info(f"{rank} output: {output_text}")
            assert output_text.endswith(
                batch_output_text_expected[idx]
            ), f"Expecting {batch_output_text_expected[idx]!r}, got {output_text!r}"
    mpi_barrier()
    return True


@force_ampere
@pytest.mark.parametrize("use_auto_parallel", [True, False],
                         ids=["enable_auto_parallel", "disable_auto_parallel"])
@pytest.mark.parametrize("model_name",
                         ["llama-models/llama-7b-hf", "Mixtral-8x7B-v0.1"])
def test_multi_gpu(model_name, use_auto_parallel):
    if torch.cuda.device_count() < TP_SIZE:
        print(f"The test needs at least ${TP_SIZE} GPUs, skipping")
        return
    if "Mixtral" in model_name and use_auto_parallel:
        pytest.skip("Auto parallel is not supported for Mixtral models")
    engine_dir = tempfile.TemporaryDirectory().name

    with MPIPoolExecutor(max_workers=TP_SIZE) as executor:
        results = executor.map(build_and_run_tp2, (0, 1), [model_name] * 2,
                               [engine_dir] * 2, [use_auto_parallel] * 2)
        for r in results:
            assert r is True


if __name__ == "__main__":
    pytest.main(sys.argv)
