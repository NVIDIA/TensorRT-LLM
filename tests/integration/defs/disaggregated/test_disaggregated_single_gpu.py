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
from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig

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
MPI_CANCEL = MPI_TAG + 3
MPI_STARTED = MPI_TAG + 4

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


async def run_worker(kv_cache_config,
                     cache_transceiver_config,
                     pytorch_config,
                     model_name,
                     rank,
                     support_cancel=False):
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
            for i, request in enumerate(requests):
                print(f"Worker {rank}: submitting request {i}/{len(requests)}",
                      flush=True)
                futures.append(
                    llm.generate_async(request[0],
                                       sampling_params=request[1],
                                       disaggregated_params=request[2]))
            print(f"Worker {rank}: all {len(futures)} requests submitted",
                  flush=True)

            if support_cancel:
                intercomm.send(len(futures), dest=0, tag=MPI_STARTED)
                cancel_indices = intercomm.recv(source=MPI.ANY_SOURCE,
                                                tag=MPI_CANCEL)
                if cancel_indices:
                    print(
                        f"Worker {rank}: cancelling {len(cancel_indices)} requests: {cancel_indices}",
                        flush=True)
                    for idx in cancel_indices:
                        futures[idx].abort()

                for i, future in enumerate(futures):
                    try:
                        print(
                            f"Worker {rank}: awaiting future {i}/{len(futures)}",
                            flush=True)
                        result = await future
                        print(f"Worker {rank}: got result {i}, sending",
                              flush=True)
                        intercomm.send(result.outputs, dest=0, tag=MPI_RESULT)
                    except Exception as e:
                        print(f"Worker {rank}: error on future {i}: {e}",
                              flush=True)
                        intercomm.send(str(e), dest=0, tag=MPI_RESULT)
            else:
                for i, future in enumerate(futures):
                    try:
                        print(
                            f"Worker {rank}: awaiting future {i}/{len(futures)}",
                            flush=True)
                        result = await future
                        print(f"Worker {rank}: got result {i}, sending",
                              flush=True)
                        intercomm.send(result.outputs, dest=0, tag=MPI_RESULT)
                    except Exception as e:
                        print(f"Worker {rank}: error on future {i}: {e}",
                              flush=True)
                        intercomm.send(str(e), dest=0, tag=MPI_RESULT)
        except Exception as e:
            print(f"Unexpected error: {e}", flush=True)
            raise e


def send_requests_to_worker(requests, worker_rank, intercomm):
    print(f"Sending {len(requests)} requests to worker {worker_rank}")
    intercomm.send(requests, dest=worker_rank, tag=MPI_REQUEST)

    responses = []
    for _ in range(len(requests)):
        responses.append(intercomm.recv(source=worker_rank, tag=MPI_RESULT))
        print(f"Received response {responses[-1]} from worker {worker_rank}")
    return responses


def worker_entry_point(kv_cache_config,
                       cache_transceiver_config,
                       pytorch_config,
                       model_name,
                       rank,
                       support_cancel=False):
    return asyncio.run(
        run_worker(kv_cache_config,
                   cache_transceiver_config,
                   pytorch_config,
                   model_name,
                   rank,
                   support_cancel=support_cancel))


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

    kv_cache_configs = [
        KvCacheConfig(max_tokens=2048 * 8, use_kv_cache_manager_v2=False)
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

    with MPIPoolExecutor(max_workers=2,
                         env={
                             "UCX_TLS": "^ib,gdr_copy",
                             "UCX_MM_ERROR_HANDLING": "y"
                         }) as executor:
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
        KvCacheConfig(max_tokens=128,
                      enable_block_reuse=False,
                      dtype="auto",
                      use_kv_cache_manager_v2=False) for _ in range(2)
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

    with MPIPoolExecutor(max_workers=2,
                         env={
                             "UCX_TLS": "^ib,gdr_copy",
                             "UCX_MM_ERROR_HANDLING": "y"
                         }) as executor:
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
            # Send 32 requests to make sure the context worker is saturated
            for _ in range(32):
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
    spec_dec_config = Eagle3DecodingConfig(
        speculative_model=model_path(spec_dec_model_path),
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
        KvCacheConfig(max_tokens=128,
                      enable_block_reuse=False,
                      free_gpu_memory_fraction=0.4,
                      use_kv_cache_manager_v2=False) for _ in range(2)
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
                             "UCX_TLS": "^ib,gdr_copy",
                             "UCX_MM_ERROR_HANDLING": "y",
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


@pytest.mark.parametrize("model", ["TinyLlama-1.1B-Chat-v1.0"])
@pytest.mark.parametrize("generation_overlap", [False, True])
def test_disaggregated_logprobs(model, generation_overlap):
    """Verify that logprobs propagate correctly from prefill to decode.
    Ensures first_gen_log_probs is carried in DisaggregatedParams
    so the generation_only worker receives one logprob per token.
    """
    worker_pytorch_configs = [
        dict(disable_overlap_scheduler=True),
        dict(disable_overlap_scheduler=not generation_overlap),
    ]

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
    max_tokens = 10
    prompt = "What is the capital of Germany?"

    with MPIPoolExecutor(max_workers=2,
                         env={
                             "UCX_TLS": "^ib,gdr_copy",
                             "UCX_MM_ERROR_HANDLING": "y"
                         }) as executor:
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
            intercomm = mpi_initialize_intercomm(port_name)
            for _ in range(2):
                intercomm.recv(tag=MPI_READY)

            # --- Context-only phase (prefill) with logprobs ---
            ctx_requests = [(prompt,
                             SamplingParams(max_tokens=max_tokens,
                                            ignore_eos=True,
                                            logprobs=1),
                             DisaggregatedParams(request_type="context_only"))]

            ctx_responses = send_requests_to_worker(ctx_requests, 0, intercomm)
            ctx_output = ctx_responses[0][0]

            assert ctx_output.disaggregated_params is not None
            assert ctx_output.disaggregated_params.request_type == "context_only"
            assert len(ctx_output.token_ids) == 1

            # The context phase must populate first_gen_log_probs.
            dp = ctx_output.disaggregated_params
            assert dp.first_gen_log_probs is not None, (
                "first_gen_log_probs should be populated by the context phase")
            assert len(dp.first_gen_log_probs) >= 1
            for lp_entry in dp.first_gen_log_probs:
                assert isinstance(lp_entry, dict)
                for token_id, logprob_obj in lp_entry.items():
                    assert isinstance(token_id, int)
                    assert logprob_obj.logprob <= 0.0, (
                        "Log probabilities must be non-positive")

            # --- Generation-only phase (decode) with logprobs ---
            dp.request_type = "generation_only"
            gen_requests = [(prompt,
                             SamplingParams(max_tokens=max_tokens,
                                            ignore_eos=True,
                                            logprobs=1), dp)]

            gen_responses = send_requests_to_worker(gen_requests, 1, intercomm)
            gen_output = gen_responses[0][0]

            # Without first_gen_log_probs propagation this either crashes
            # (AttributeError) or returns fewer logprobs than tokens.
            assert gen_output.logprobs is not None, (
                "Generation phase should return logprobs")
            assert len(gen_output.logprobs) == len(gen_output.token_ids), (
                f"Expected one logprob per token: got {len(gen_output.logprobs)}"
                f" logprobs for {len(gen_output.token_ids)} tokens")

            for pos_idx, lp_entry in enumerate(gen_output.logprobs):
                assert isinstance(
                    lp_entry, dict), (f"logprobs[{pos_idx}] should be a dict")
                for token_id, logprob_obj in lp_entry.items():
                    assert isinstance(token_id, int)
                    assert logprob_obj.logprob <= 0.0

        except Exception as e:
            print(f"Exception encountered: {e}", flush=True)
            raise e
        finally:
            mpi_send_termination_request(intercomm)
            for future in futures:
                future.result()


@pytest.mark.parametrize("model", ["TinyLlama-1.1B-Chat-v1.0"])
def test_disaggregated_cancel_gen_requests(model):
    # Test that cancelling generation requests on a saturated generation
    # worker completes without hangs or resource leaks.
    worker_pytorch_configs = []

    # Context worker
    worker_pytorch_configs.append(
        dict(disable_overlap_scheduler=True, cuda_graph_config=None))

    # Generation worker
    worker_pytorch_configs.append(dict(cuda_graph_config=None))

    kv_cache_configs = [
        KvCacheConfig(max_tokens=2048, enable_block_reuse=False)
        for _ in range(2)
    ]
    cache_transceiver_configs = [
        CacheTransceiverConfig(backend="DEFAULT") for _ in range(2)
    ]
    model_names = [model_path(model) for _ in range(2)]

    port_name = mpi_publish_name()

    prompt = "What is the capital of Germany?"
    num_requests = 16
    num_cancel = 8
    max_tokens = 50

    with MPIPoolExecutor(max_workers=2,
                         env={
                             "UCX_TLS": "^ib,gdr_copy",
                             "UCX_MM_ERROR_HANDLING": "y",
                         }) as executor:
        futures = []
        try:
            futures.append(
                executor.submit(worker_entry_point, kv_cache_configs[0],
                                cache_transceiver_configs[0],
                                worker_pytorch_configs[0], model_names[0], 0))
            futures.append(
                executor.submit(worker_entry_point, kv_cache_configs[1],
                                cache_transceiver_configs[1],
                                worker_pytorch_configs[1], model_names[1], 1,
                                True))
        except Exception as e:
            print(f"Error submitting workers: {e}")
            raise e

        intercomm = None
        try:
            print("Launched all workers.", flush=True)
            intercomm = mpi_initialize_intercomm(port_name)

            for _ in range(2):
                intercomm.recv(tag=MPI_READY)
                print("Received ready signal.")

            context_requests = []
            for _ in range(num_requests):
                context_requests.append(
                    (prompt, SamplingParams(max_tokens=1, ignore_eos=True),
                     DisaggregatedParams(request_type="context_only")))

            intercomm.send(context_requests, dest=0, tag=MPI_REQUEST)

            gen_requests = []
            for _ in range(num_requests):
                output = intercomm.recv(source=0, tag=MPI_RESULT)
                assert output[0].disaggregated_params is not None
                assert output[
                    0].disaggregated_params.request_type == "context_only"
                assert len(output[0].token_ids) == 1

                disagg_params = output[0].disaggregated_params
                disagg_params.request_type = "generation_only"
                gen_requests.append(
                    (prompt,
                     SamplingParams(max_tokens=max_tokens,
                                    ignore_eos=True), disagg_params))

            intercomm.send(gen_requests, dest=1, tag=MPI_REQUEST)

            num_started = intercomm.recv(source=1, tag=MPI_STARTED)
            assert num_started == num_requests
            print(f"Generation worker started {num_started} requests.")

            cancel_indices = list(range(num_cancel))
            intercomm.send(cancel_indices, dest=1, tag=MPI_CANCEL)
            print(f"Sent cancel for indices {cancel_indices}.")

            for i in range(num_requests):
                output = intercomm.recv(source=1, tag=MPI_RESULT)
                print(f"Received result {i}/{num_requests}.")

        except Exception as e:
            print(f"Exception encountered: {e}", flush=True)
        finally:
            print("Sending termination request", flush=True)
            mpi_send_termination_request(intercomm)

            print("Waiting for all workers to terminate.", flush=True)
            for future in futures:
                future.result()
            print("All workers terminated.")


if __name__ == "__main__":
    pytest.main()
