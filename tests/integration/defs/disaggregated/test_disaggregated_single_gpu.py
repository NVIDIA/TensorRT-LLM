import asyncio
import logging
import sys

import pytest
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm.logger as logger
from tensorrt_llm import DisaggregatedParams, SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.llmapi import KvCacheConfig, MpiCommSession

print(sys.path)

MPI_TAG = 9999
MPI_READY = MPI_TAG + 2
MPI_REQUEST = MPI_TAG
MPI_RESULT = MPI_TAG + 1


async def run_worker(kv_cache_config, pytorch_config, model_name, rank):
    logging.info(f"Running worker {rank}")
    port_name = MPI.Lookup_name('my_port')
    intercomm = MPI.COMM_WORLD.Connect(port_name)

    session = MPI.COMM_WORLD.Split(color=rank, key=0)
    set_mpi_comm(session)
    mpi_session = MpiCommSession(comm=session, n_workers=session.Get_size())
    logger.set_level(logging.INFO)

    try:
        llm = LLM(tensor_parallel_size=1,
                  auto_parallel=False,
                  model=model_name,
                  enable_chunked_prefill=False,
                  pytorch_backend_config=pytorch_config,
                  _mpi_session=mpi_session,
                  kv_cache_config=kv_cache_config)
        logging.info(f"LLM created")
    except Exception as e:
        logging.error(f"Error creating LLM: {e}")
        raise e

    # Send ready signal
    logging.info(f"Sending ready signal to main process")
    intercomm.send(intercomm.Get_rank(), dest=0, tag=MPI_READY)

    logging.info(f"Waiting for requests")
    while True:
        try:
            requests = intercomm.recv(source=MPI.ANY_SOURCE, tag=MPI_REQUEST)
            logging.info(f"Received requests: {requests}")
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
            logging.error(f"Worker {rank} error: {e}")
    llm.shutdown()


def send_requests_to_worker(requests, worker_rank, intercomm):
    logging.info(f"Sending {len(requests)} requests to worker {worker_rank}")
    intercomm.send(requests, dest=worker_rank, tag=MPI_REQUEST)

    responses = []
    for _ in range(len(requests)):
        responses.append(intercomm.recv(source=worker_rank, tag=MPI_RESULT))
        logging.info(
            f"Received response {responses[-1]} from worker {worker_rank}")
    return responses


def worker_entry_point(kv_cache_config, pytorch_config, model_name, rank):
    logging.info(f"Running worker {rank}")
    return asyncio.run(
        run_worker(kv_cache_config, pytorch_config, model_name, rank))


@pytest.mark.parametrize("model_dir", [
    "/home/scratch.trt_llm_data/llm-models/llama-modeljs-v2/TinyLlama-1.1B-Chat-v1.0/"
])
@pytest.mark.parametrize("generation_overlap", [False, True])
def test_disaggregated_simple(model_dir, generation_overlap):
    worker_pytorch_configs = []

    # Context worker
    worker_pytorch_configs.append(
        PyTorchConfig(enable_overlap_scheduler=False, kv_cache_dtype="auto"))

    # Generation worker
    worker_pytorch_configs.append(
        PyTorchConfig(enable_overlap_scheduler=generation_overlap,
                      kv_cache_dtype="auto"))

    kv_cache_configs = [KvCacheConfig(max_tokens=2048 * 8) for _ in range(2)]
    model_names = [model_dir for _ in range(2)]
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
            for future in futures:
                future.result()
        except Exception as e:
            logging.error(f"Error in worker {worker_arg}: {e}")
            raise e

        logging.info("Launched all the workers.")
        intercomm = MPI.COMM_SELF.Accept(port_name)

        for _ in range(2):
            intercomm.recv(tag=MPI_READY)
            logging.info("Received ready signal.")

        requests = []
        requests.append(
            ("The capital of France is", SamplingParams(max_tokens=1),
             DisaggregatedParams(request_type="context_only")))

        responses = send_requests_to_worker(requests, 0, intercomm)
        output = responses[0]
        assert output[0].text == "Paris"
        assert output[0].disaggregated_params is not None
        assert output[0].disaggregated_params.request_type == "context_only"

        generation_request_disagg_params = output[0].disaggregated_params
        generation_request_disagg_params.request_type = "generation_only"
        requests = []
        requests.append(
            ("The capital of France is", SamplingParams(max_tokens=10),
             generation_request_disagg_params))

        responses = send_requests_to_worker(requests, 1, intercomm)
        output = responses[0]
        assert output[0].text.startswith("Paris")

        # Send a non-disaggregated request for output verification
        requests = []
        requests.append(
            ("The capital of France is", SamplingParams(max_tokens=10), None))

        response_ref = send_requests_to_worker(requests, 0, intercomm)
        output_ref = response_ref[0]
        assert output_ref[0].text.startswith("Paris")
        assert output_ref[0].text == output[
            0].text, f"Output mismatch: {output_ref[0].text} != {output[0].text}"

        # Send termination requests
        intercomm.send(None, dest=0, tag=MPI_REQUEST)
        intercomm.send(None, dest=1, tag=MPI_REQUEST)
        logging.info("Sent termination requests to the workers.")

        # Wait for all futures to complete
        for future in futures:
            future.result()
        logging.info("All workers terminated.")


if __name__ == "__main__":
    pytest.main()
