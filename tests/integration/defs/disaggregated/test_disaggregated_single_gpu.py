import asyncio
import os
import pickle
import sys

import cloudpickle
import pytest
from defs.conftest import skip_no_hopper
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm import LLM, DisaggregatedParams, SamplingParams
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.llmapi import (CacheTransceiverConfig, CudaGraphConfig,
                                 KvCacheConfig, MpiCommSession)
from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig

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

MODEL_PATHS = {
    "DeepSeek-V3-Lite-fp8": "DeepSeek-V3-Lite/fp8",
    "TinyLlama-1.1B-Chat-v1.0": "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
    "Llama-3.1-8B-Instruct": "llama-3.1-model/Llama-3.1-8B-Instruct/",
    "EAGLE3-LLaMA3.1-Instruct-8B": "EAGLE3-LLaMA3.1-Instruct-8B",
    "Qwen3-8B-FP8": "Qwen3/Qwen3-8B-FP8",
}


def mpi_publish_name():
    port_name = None
    try:
        port_name = MPI.Open_port()
        MPI.Publish_name('my_port', port_name)
    except MPI.Exception as e:
        print(f"Error publishing port name: {e}")
        raise e
    except Exception as e:
        print(f"Unexpected error publishing port name: {e}")
        raise e

    return port_name


def mpi_initialize_intercomm(port_name):
    intercomm = None
    try:
        intercomm = MPI.COMM_SELF.Accept(port_name)
    except MPI.Exception as e:
        print(f"Error accepting intercomm: {e}", flush=True)
        raise
    except Exception as e:
        print(f"Unexpected error accepting intercomm: {e}", flush=True)
        raise
    return intercomm


def mpi_send_termination_request(intercomm):
    if intercomm is not None:
        # Send termination requests
        intercomm.send(None, dest=0, tag=MPI_REQUEST)
        intercomm.send(None, dest=1, tag=MPI_REQUEST)
        print("Sent termination requests to the workers.")


def model_path(model_name):
    llm_models_root = os.environ["LLM_MODELS_ROOT"]
    for name, path in MODEL_PATHS.items():
        if name in model_name:
            return os.path.join(llm_models_root, path)
    raise ValueError(f"Unknown model: {model_name}")


async def run_worker(kv_cache_config, cache_transceiver_config, pytorch_config,
                     model_name, rank):
    assert isinstance(pytorch_config, dict)
    print(f"Running worker {rank}")
    try:
        port_name = MPI.Lookup_name('my_port')
        intercomm = MPI.COMM_WORLD.Connect(port_name)
    except MPI.Exception as e:
        print(f"Error publishing port name: {e}")
        raise e
    except Exception as e:
        print(f"Unexpected error publishing port name: {e}")
        raise e

    session = MPI.COMM_WORLD.Split(color=rank, key=0)
    set_mpi_comm(session)
    mpi_session = MpiCommSession(comm=session, n_workers=session.Get_size())

    try:
        llm = LLM(tensor_parallel_size=1,
                  model=model_name,
                  enable_chunked_prefill=False,
                  **pytorch_config,
                  _mpi_session=mpi_session,
                  kv_cache_config=kv_cache_config,
                  cache_transceiver_config=cache_transceiver_config)
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


def worker_entry_point(kv_cache_config, cache_transceiver_config,
                       pytorch_config, model_name, rank):
    return asyncio.run(
        run_worker(kv_cache_config, cache_transceiver_config, pytorch_config,
                   model_name, rank))


def verify_disaggregated(model, generation_overlap, enable_cuda_graph, prompt,
                         expected_output, expected_output_ids):
    worker_pytorch_configs = []

    # Context worker
    worker_pytorch_configs.append(
        dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig() if enable_cuda_graph else None))

    # Generation worker
    worker_pytorch_configs.append(
        dict(
            disable_overlap_scheduler=not generation_overlap,
            cuda_graph_config=CudaGraphConfig() if enable_cuda_graph else None))

    kv_cache_configs = [KvCacheConfig(max_tokens=2048 * 8) for _ in range(2)]
    cache_transceiver_configs = [
        CacheTransceiverConfig(backend="DEFAULT") for _ in range(2)
    ]
    model_names = [model_path(model) for _ in range(2)]
    ranks = [0, 1]
    worker_args = list(
        zip(kv_cache_configs, cache_transceiver_configs, worker_pytorch_configs,
            model_names, ranks))

    port_name = mpi_publish_name()

    with MPIPoolExecutor(max_workers=2, env={"UCX_TLS": "^ib"}) as executor:
        futures = []
        try:
            for worker_arg in worker_args:
                future = executor.submit(worker_entry_point, *worker_arg)
                futures.append(future)
        except Exception as e:
            print(f"Error in worker {worker_arg}: {e}")
            raise e

        intercomm = None
        try:
            print("Launched all the workers.", flush=True)
            intercomm = mpi_initialize_intercomm(port_name)

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
        except Exception as e:
            print(f"Exception encountered: {e}", flush=True)
            raise e
        finally:
            print("Sending termination request", flush=True)
            mpi_send_termination_request(intercomm)

            # Wait for all futures to complete
            print("Waiting for all workers to terminate. ", flush=True)
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


@skip_no_hopper
@pytest.mark.parametrize("model", ["Qwen3-8B-FP8"])
@pytest.mark.parametrize("generation_overlap", [False, True])
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_disaggregated_simple_qwen3(model, generation_overlap,
                                    enable_cuda_graph):
    verify_disaggregated(
        model, generation_overlap, enable_cuda_graph,
        " What is the capital of China?",
        " The capital of China is Beijing. 2. What is the population of China? The population of China is about 1",
        [
            576, 6722, 315, 5616, 374, 26549, 13, 220, 17, 13, 3555, 374, 279,
            7042, 315, 5616, 30, 576, 7042, 315, 5616, 374, 911, 220, 16
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
        dict(
            disable_overlap_scheduler=True,
            cuda_graph_config=CudaGraphConfig() if enable_cuda_graph else None))

    # Generation worker
    worker_pytorch_configs.append(
        dict(
            disable_overlap_scheduler=not generation_overlap,
            cuda_graph_config=CudaGraphConfig() if enable_cuda_graph else None))

    kv_cache_configs = [
        KvCacheConfig(max_tokens=128, enable_block_reuse=False, dtype="auto")
        for _ in range(2)
    ]
    cache_transceiver_configs = [
        CacheTransceiverConfig(backend="DEFAULT") for _ in range(2)
    ]
    model_names = [model_path(model) for _ in range(2)]
    ranks = [0, 1]
    worker_args = list(
        zip(kv_cache_configs, cache_transceiver_configs, worker_pytorch_configs,
            model_names, ranks))

    port_name = mpi_publish_name()

    prompt = "European Union is a political and economic union of 27 countries. The European Union is headquartered in Brussels, Belgium. The first president of the European Union was Jean-Claude Juncker. The current president is Ursula von der Leyen. The European Union is a major economic and political entity."

    with MPIPoolExecutor(max_workers=2, env={"UCX_TLS": "^ib"}) as executor:
        futures = []
        try:
            for worker_arg in worker_args:
                future = executor.submit(worker_entry_point, *worker_arg)
                futures.append(future)
        except Exception as e:
            print(f"Error in worker {worker_arg}: {e}")
            raise e

        intercomm = None
        try:
            print("Launched all the workers.")
            intercomm = mpi_initialize_intercomm(port_name)

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

        except MPI.Exception as e:
            print(f"MPI Error")
            raise e
        finally:
            mpi_send_termination_request(intercomm)

            # Wait for all futures to complete
            for future in futures:
                future.result()
            print("All workers terminated.")


@pytest.mark.parametrize("model", ["Llama-3.1-8B-Instruct"])
@pytest.mark.parametrize("spec_dec_model_path", ["EAGLE3-LLaMA3.1-Instruct-8B"])
@pytest.mark.parametrize("generation_overlap", [False])
@pytest.mark.parametrize("eagle3_one_model", [True, False])
def test_disaggregated_spec_dec_batch_slot_limit(model, spec_dec_model_path,
                                                 generation_overlap,
                                                 eagle3_one_model):
    # Test whether the batch slots are properly released when using speculative decoding
    # with disaggregated serving.
    spec_dec_config = EagleDecodingConfig(
        speculative_model_dir=model_path(spec_dec_model_path),
        eagle3_one_model=eagle3_one_model,
        max_draft_len=3)

    worker_pytorch_configs = []

    # Context worker
    worker_pytorch_configs.append(
        dict(disable_overlap_scheduler=True,
             speculative_config=spec_dec_config,
             max_batch_size=1))

    # Generation worker
    worker_pytorch_configs.append(
        dict(disable_overlap_scheduler=not generation_overlap,
             speculative_config=spec_dec_config,
             max_batch_size=1))

    kv_cache_configs = [
        KvCacheConfig(max_tokens=128, enable_block_reuse=False)
        for _ in range(2)
    ]
    cache_transceiver_configs = [
        CacheTransceiverConfig(backend="DEFAULT") for _ in range(2)
    ]
    model_names = [model_path(model) for _ in range(2)]
    ranks = [0, 1]
    worker_args = list(
        zip(kv_cache_configs, cache_transceiver_configs, worker_pytorch_configs,
            model_names, ranks))

    port_name = mpi_publish_name()

    prompt = "What is the capital of Germany?"
    mpi_info = MPI.Info.Create()
    mpi_info.Set("oversubscribe", "true")
    with MPIPoolExecutor(max_workers=2,
                         env={
                             "UCX_TLS": "^ib",
                             "OMPI_MCA_rmaps_base_oversubscribe": "1"
                         },
                         mpi_info=mpi_info) as executor:
        futures = []
        try:
            for worker_arg in worker_args:
                future = executor.submit(worker_entry_point, *worker_arg)
                futures.append(future)
        except Exception as e:
            print(f"Error in worker {worker_arg}: {e}")
            raise e

        intercomm = None
        try:
            print("Launched all the workers.")
            intercomm = mpi_initialize_intercomm(port_name)

            for _ in range(2):
                intercomm.recv(tag=MPI_READY)
                print("Received ready signal.")
            max_tokens = 25

            requests = []
            for _ in range(10):
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

        except MPI.Exception as e:
            print(f"MPI Error")
            raise e
        finally:
            mpi_send_termination_request(intercomm)

            # Wait for all futures to complete
            for future in futures:
                future.result()
            print("All workers terminated.")


if __name__ == "__main__":
    pytest.main()
