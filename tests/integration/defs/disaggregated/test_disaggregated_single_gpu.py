import asyncio
import os
import pickle
import sys

import cloudpickle
import pytest
from defs.conftest import skip_no_hopper
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm import DisaggregatedParams, SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.llmapi import KvCacheConfig, MpiCommSession

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

MPI_TAG = 9999
MPI_READY = MPI_TAG + 2
MPI_REQUEST = MPI_TAG
MPI_RESULT = MPI_TAG + 1


def model_path(model_name):
    llm_models_root = os.environ["LLM_MODELS_ROOT"]
    if 'DeepSeek-V3-Lite-fp8' in model_name:
        return os.path.join(llm_models_root, 'DeepSeek-V3-Lite', 'fp8')
    elif 'TinyLlama-1.1B-Chat-v1.0' in model_name:
        return os.path.join(llm_models_root, 'llama-models-v2',
                            'TinyLlama-1.1B-Chat-v1.0')
    else:
        raise ValueError(f"Unknown model: {model_name}")


async def run_worker(kv_cache_config, pytorch_config, model_name, rank):
    assert isinstance(pytorch_config, dict)
    print(f"Running worker {rank}")
    port_name = MPI.Lookup_name('my_port')
    intercomm = MPI.COMM_WORLD.Connect(port_name)

    session = MPI.COMM_WORLD.Split(color=rank, key=0)
    set_mpi_comm(session)
    mpi_session = MpiCommSession(comm=session, n_workers=session.Get_size())

    try:
        llm = LLM(tensor_parallel_size=1,
                  auto_parallel=False,
                  model=model_name,
                  enable_chunked_prefill=False,
                  **pytorch_config,
                  _mpi_session=mpi_session,
                  kv_cache_config=kv_cache_config)
        print(f"LLM created")
    except Exception as e:
        print(f"Error creating LLM: {e}")
        raise e

    # Send ready signal
    print(f"Sending ready signal to main process")
    intercomm.send(intercomm.Get_rank(), dest=0, tag=MPI_READY)

    print(f"Waiting for requests")
    while True:
        try:
            requests = intercomm.recv(source=MPI.ANY_SOURCE, tag=MPI_REQUEST)
            print(f"Received requests: {requests}")
            if requests is None:
                break

            futures = []
            for request in requests:
                futures.append(
                    llm.generate_async(request[0],
                                       sampling_params=request[1],
                                       disaggregated_params=request[2]))

            for future in futures:
                result = await future
                intercomm.send(result.outputs, dest=0, tag=MPI_RESULT)
        except Exception as e:
            print(f"Worker {rank} error: {e}")
    llm.shutdown()


def send_requests_to_worker(requests, worker_rank, intercomm):
    print(f"Sending {len(requests)} requests to worker {worker_rank}")
    intercomm.send(requests, dest=worker_rank, tag=MPI_REQUEST)

    responses = []
    for _ in range(len(requests)):
        responses.append(intercomm.recv(source=worker_rank, tag=MPI_RESULT))
        print(f"Received response {responses[-1]} from worker {worker_rank}")
    return responses


def worker_entry_point(kv_cache_config, pytorch_config, model_name, rank):
    return asyncio.run(
        run_worker(kv_cache_config, pytorch_config, model_name, rank))


def verify_disaggregated(model, generation_overlap, enable_cuda_graph, prompt,
                         expected_output, expected_output_ids):
    worker_pytorch_configs = []

    # Context worker
    worker_pytorch_configs.append(
        dict(disable_overlap_scheduler=True,
             kv_cache_dtype="auto",
             use_cuda_graph=enable_cuda_graph))

    # Generation worker
    worker_pytorch_configs.append(
        dict(disable_overlap_scheduler=not generation_overlap,
             kv_cache_dtype="auto",
             use_cuda_graph=enable_cuda_graph))

    kv_cache_configs = [KvCacheConfig(max_tokens=2048 * 8) for _ in range(2)]
    model_names = [model_path(model) for _ in range(2)]
    ranks = [0, 1]
    worker_args = list(
        zip(kv_cache_configs, worker_pytorch_configs, model_names, ranks))

    port_name = MPI.Open_port()
    MPI.Publish_name('my_port', port_name)

    with MPIPoolExecutor(max_workers=2, env={"TRTLLM_USE_MPI_KVCACHE":
                                             "1"}) as executor:
        futures = []
        try:
            for worker_arg in worker_args:
                future = executor.submit(worker_entry_point, *worker_arg)
                futures.append(future)
        except Exception as e:
            print(f"Error in worker {worker_arg}: {e}")
            raise e

        try:
            print("Launched all the workers.")
            intercomm = MPI.COMM_SELF.Accept(port_name)

            for _ in range(2):
                intercomm.recv(tag=MPI_READY)
                print("Received ready signal.")
            max_tokens = 25

            requests = []
            requests.append(
                (prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True),
                 DisaggregatedParams(request_type="context_only")))

            responses = send_requests_to_worker(requests, 0, intercomm)
            output = responses[0]
            print(f"Output: {output}")
            print(f"Output: {output[0].disaggregated_params}")
            assert output[0].disaggregated_params is not None
            print(f"Output: {output[0].disaggregated_params.request_type}")
            assert output[0].disaggregated_params.request_type == "context_only"
            assert output[0].token_ids[0] == expected_output_ids[0]
            assert len(output[0].token_ids) == 1

            generation_request_disagg_params = output[0].disaggregated_params
            generation_request_disagg_params.request_type = "generation_only"
            requests = []
            requests.append(
                (prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True),
                 generation_request_disagg_params))

            responses = send_requests_to_worker(requests, 1, intercomm)
            output = responses[0]
            assert output[0].text == expected_output
            assert output[0].token_ids == expected_output_ids

        finally:
            # Send termination requests
            intercomm.send(None, dest=0, tag=MPI_REQUEST)
            intercomm.send(None, dest=1, tag=MPI_REQUEST)
            print("Sent termination requests to the workers.")

            # Wait for all futures to complete
            for future in futures:
                future.result()
            print("All workers terminated.")


@pytest.mark.parametrize("model", ["TinyLlama-1.1B-Chat-v1.0"])
@pytest.mark.parametrize("generation_overlap", [False, True])
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_disaggregated_simple_llama(model, generation_overlap,
                                    enable_cuda_graph):
    verify_disaggregated(
        model, generation_overlap, enable_cuda_graph,
        "What is the capital of Germany?",
        "\n<|assistant|>\nThe capital of Germany is Berlin. \n<|user|>", [
            2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13, 1576,
            7483, 310, 9556, 338, 5115, 29889, 2, 29871, 13, 29966, 29989, 1792,
            29989, 29958
        ])


@skip_no_hopper
@pytest.mark.parametrize("model", ["DeepSeek-V3-Lite-fp8/fp8"])
@pytest.mark.parametrize("generation_overlap", [False, True])
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_disaggregated_simple_deepseek(model, generation_overlap,
                                       enable_cuda_graph):
    verify_disaggregated(
        model, generation_overlap, enable_cuda_graph,
        "What is the capital of Germany?",
        " | Berlin \nWhat is the capital of France? | Paris \nWhat is the capital of Italy? | Rome \nWhat is",
        [
            369, 17575, 539, 3085, 344, 270, 6102, 294, 8760, 33, 369, 11111,
            539, 3085, 344, 270, 6102, 294, 14251, 33, 369, 16235, 539, 3085,
            344
        ])


@pytest.mark.parametrize("model", ["DeepSeek-V3-Lite-fp8/fp8"])
@pytest.mark.parametrize("enable_cuda_graph", [False])
@pytest.mark.parametrize("generation_overlap", [False])
def test_disaggregated_llama_context_capacity(model, enable_cuda_graph,
                                              generation_overlap):
    # Test the case where the context worker capacity is exceeded and
    # needs to wait for the generation worker to complete.
    worker_pytorch_configs = []

    # Context worker
    worker_pytorch_configs.append(
        dict(disable_overlap_scheduler=True,
             kv_cache_dtype="auto",
             use_cuda_graph=enable_cuda_graph))

    # Generation worker
    worker_pytorch_configs.append(
        dict(disable_overlap_scheduler=not generation_overlap,
             kv_cache_dtype="auto",
             use_cuda_graph=enable_cuda_graph))

    kv_cache_configs = [
        KvCacheConfig(max_tokens=128, enable_block_reuse=False)
        for _ in range(2)
    ]
    model_names = [model_path(model) for _ in range(2)]
    ranks = [0, 1]
    worker_args = list(
        zip(kv_cache_configs, worker_pytorch_configs, model_names, ranks))

    port_name = MPI.Open_port()
    MPI.Publish_name('my_port', port_name)

    prompt = "European Union is a political and economic union of 27 countries. The European Union is headquartered in Brussels, Belgium. The first president of the European Union was Jean-Claude Juncker. The current president is Ursula von der Leyen. The European Union is a major economic and political entity."

    with MPIPoolExecutor(max_workers=2, env={"TRTLLM_USE_MPI_KVCACHE":
                                             "1"}) as executor:
        futures = []
        try:
            for worker_arg in worker_args:
                future = executor.submit(worker_entry_point, *worker_arg)
                futures.append(future)
        except Exception as e:
            print(f"Error in worker {worker_arg}: {e}")
            raise e

        try:
            print("Launched all the workers.")
            intercomm = MPI.COMM_SELF.Accept(port_name)

            for _ in range(2):
                intercomm.recv(tag=MPI_READY)
                print("Received ready signal.")
            max_tokens = 25

            requests = []
            # Send 256 requests to make sure the context worker is saturated
            for _ in range(256):
                requests.append(
                    (prompt, SamplingParams(max_tokens=1, ignore_eos=True),
                     DisaggregatedParams(request_type="context_only")))

            intercomm.send(requests, dest=0, tag=MPI_REQUEST)

            for _ in range(len(requests)):
                output = intercomm.recv(source=0, tag=MPI_RESULT)
                assert output[0].disaggregated_params is not None
                assert output[
                    0].disaggregated_params.request_type == "context_only"
                assert len(output[0].token_ids) == 1

                generation_request_disagg_params = output[
                    0].disaggregated_params
                generation_request_disagg_params.request_type = "generation_only"
                requests = []
                requests.append((prompt,
                                 SamplingParams(max_tokens=max_tokens,
                                                ignore_eos=True),
                                 generation_request_disagg_params))

                intercomm.send(requests, dest=1, tag=MPI_REQUEST)
                output = intercomm.recv(source=1, tag=MPI_RESULT)

        finally:
            # Send termination requests
            intercomm.send(None, dest=0, tag=MPI_REQUEST)
            intercomm.send(None, dest=1, tag=MPI_REQUEST)
            print("Sent termination requests to the workers.")

            # Wait for all futures to complete
            for future in futures:
                future.result()
            print("All workers terminated.")


if __name__ == "__main__":
    pytest.main()
