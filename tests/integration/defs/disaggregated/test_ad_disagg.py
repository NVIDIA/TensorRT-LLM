# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import pickle
import sys
import traceback
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import replace

import cloudpickle
import pytest
import torch
from defs.conftest import check_device_contain, get_sm_version, skip_pre_hopper
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm import DisaggregatedParams, SamplingParams
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm._utils import set_mpi_comm
from tensorrt_llm.llmapi import Eagle3DecodingConfig

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@pytest.fixture(autouse=True)
def skip_b300():
    if check_device_contain(["B300"]):
        pytest.skip(
            "AutoDeploy disagg tests are disabled on B300/GB300 until capacity is available: "
            "https://nvbugs/6301621"
        )


WORKER_READY = "ready"
REQUEST_MODE_AGGREGATE = "aggregate"
MPI_REQUEST = 9999
MPI_RESULT = MPI_REQUEST + 1
OMPI_COMM_WORLD_ENV_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "OMPI_COMM_WORLD_NODE_RANK",
    "OMPI_UNIVERSE_SIZE",
)
AUTODEPLOY_DISAGG_SEED = 1234
REDUCED_TINYLLAMA_LAYERS = 2
REDUCED_DEEPSEEK_LAYERS = 2
LLAMA_EAGLE3_EXPECTED_TEXT = " Berlin\nWhat is the capital of France? Paris\nWhat is the capital of"
LLAMA_EAGLE3_EXPECTED_TOKEN_IDS = [
    20437,
    198,
    3923,
    374,
    279,
    6864,
    315,
    9822,
    30,
    12366,
    198,
    3923,
    374,
    279,
    6864,
    315,
]


MODEL_PATHS = {
    "EAGLE3-LLaMA3.1-Instruct-8B": "EAGLE3-LLaMA3.1-Instruct-8B",
    "Llama-3.1-8B-Instruct": "llama-3.1-model/Llama-3.1-8B-Instruct/",
    "TinyLlama-1.1B-Chat-v1.0": "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
    "DeepSeek-V3-Lite": "DeepSeek-V3-Lite/bf16",
}


def model_path(model_name):
    llm_models_root = os.environ["LLM_MODELS_ROOT"]
    for name, path in MODEL_PATHS.items():
        if name in model_name:
            return os.path.join(llm_models_root, path)
    raise ValueError(f"Unknown model: {model_name}")


def response_summary(response):
    """Summarize values returned by AutoDeploy test workers.

    Inputs:
        response: Payload from an AutoDeploy worker. It can be a formatted exception
            string or a list of normal LLM output objects.

    Outputs:
        A string representing the input response.

    This is useful because subprocess workers send results through MPI, so
    assertion failures otherwise lose the key fields
    needed to debug the disaggregated handoff: generated text/tokens, request
    type, context request id, draft-token count, and logits shape when present.
    """
    if isinstance(response, str):
        return f"error={response}"
    if isinstance(response, list) and response and hasattr(response[0], "token_ids"):
        summaries = []
        for idx, output in enumerate(response):
            disaggregated_params = output.disaggregated_params
            if disaggregated_params is None:
                request_type = REQUEST_MODE_AGGREGATE
                ctx_request_id = None
            else:
                request_type = disaggregated_params.request_type
                ctx_request_id = disaggregated_params.ctx_request_id
            draft_tokens = (
                len(disaggregated_params.draft_tokens)
                if disaggregated_params is not None
                and disaggregated_params.draft_tokens is not None
                else 0
            )
            logits = output.generation_logits
            logits_shape = tuple(logits.shape) if logits is not None else None
            summaries.append(
                f"{idx}: text={output.text!r}, token_ids={output.token_ids}, "
                f"disagg_type={request_type}, ctx_request_id={ctx_request_id}, "
                f"draft_tokens={draft_tokens}, logits_shape={logits_shape}"
            )
        return "[" + "; ".join(summaries) + "]"
    return repr(response)


def seed_disagg():
    torch.manual_seed(AUTODEPLOY_DISAGG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(AUTODEPLOY_DISAGG_SEED)


def disable_piecewise_cuda_graph_for_speculation(config: dict) -> dict:
    """Disable piecewise CUDA graph capture for speculative AutoDeploy tests."""
    config.setdefault("transforms", {}).setdefault("compile_model", {})["piecewise_enabled"] = False
    return config


def base_config(extra_config=None):
    common_config = dict(
        runtime="trtllm",
        attn_backend="trtllm",
        max_batch_size=4,
        max_seq_len=2048,
        max_num_tokens=512,
        trust_remote_code=True,
        kv_cache_config={"max_tokens": 2048},
        compile_backend="torch-cudagraph",
        cuda_graph_config={"batch_sizes": [1, 2, 4]},
    )
    if extra_config:
        common_config.update(extra_config)
    if common_config.get("speculative_config") is not None:
        disable_piecewise_cuda_graph_for_speculation(common_config)

    return common_config


def disagg_config(extra_config=None):
    return dict(
        base_config(extra_config),
        cache_transceiver_config={"backend": "DEFAULT"},
    )


def context_config(extra_config=None):
    # Context-only transfer happens after the request completes its context phase,
    # so keep the context worker on the non-overlap scheduling path.
    return dict(
        disagg_config(extra_config),
        disable_overlap_scheduler=True,
    )


def generation_config(generation_overlap, extra_config=None):
    config = disagg_config(extra_config)
    if not generation_overlap:
        config["disable_overlap_scheduler"] = True

    return config


def single_output(response):
    if isinstance(response, str):
        raise RuntimeError(response)
    if not isinstance(response, list) or not response:
        raise RuntimeError(f"Expected a non-empty output list, got {response_summary(response)}")
    return response[0]


def first_output(responses):
    if len(responses) != 1:
        raise RuntimeError(f"Expected one response, got {response_summary(responses)}")
    return single_output(responses[0])


def generation_params_from_context(context_output):
    context_params = context_output.disaggregated_params
    if context_params is None:
        raise RuntimeError(
            f"Context output has no disaggregated params: {response_summary([context_output])}"
        )
    return replace(context_params, request_type="generation_only")


def has_draft_tokens(output):
    params = output.disaggregated_params
    return params is not None and params.draft_tokens is not None and len(params.draft_tokens) > 0


def has_handoff_transport_metadata(params):
    # C++ transceiver carries handoff state in opaque_state; Python/native
    # transceiver carries the context endpoint in ctx_info_endpoint.
    return params.opaque_state is not None or params.ctx_info_endpoint is not None


def run_aggregate_generation(
    model,
    world_size,
    prompt,
    sampling_params_kwargs=None,
    extra_config=None,
):
    """Run one non-disaggregated AutoDeploy generation request."""
    if sampling_params_kwargs is None:
        sampling_params_kwargs = {"max_tokens": 25, "ignore_eos": True}

    seed_disagg()
    with AutoDeployLLM(
        model=model_path(model),
        world_size=world_size,
        **base_config(extra_config),
    ) as llm:
        seed_disagg()
        result = llm.generate(
            prompt,
            sampling_params=SamplingParams(**sampling_params_kwargs),
            use_tqdm=False,
        )

    output = result.outputs[0]
    print(f"[AD DISAGG TEST] aggregate output: {response_summary([output])}")
    return output


# ---------------------------------------------------------------------------
# Sequential live-pair tests.
#
# These tests run context-only and generation-only AutoDeploy instances in one
# process, one after the other, while keeping the context instance alive for
# the generation handoff. They usually run in the 1-GPU stage. Unlike the unit
# smoke tests, these load real weights. Keep exact output comparisons to
# single-request cases; batched handoff uses semantic slot checks because IFB can
# make multi-request generation differ from aggregate output even when handoff is
# correct.
# ---------------------------------------------------------------------------


def reduced_tinyllama_config(extra_config=None):
    config = {
        "model_kwargs": {"num_hidden_layers": REDUCED_TINYLLAMA_LAYERS},
        "max_batch_size": 4,
        "max_seq_len": 512,
        "max_num_tokens": 256,
        "kv_cache_config": {"max_tokens": 1024},
    }
    if extra_config:
        config.update(extra_config)
    return config


def reduced_deepseek_v3_mla_config():
    return {
        "model_kwargs": {"num_hidden_layers": REDUCED_DEEPSEEK_LAYERS},
        "max_batch_size": 4,
        "max_seq_len": 512,
        "max_num_tokens": 256,
        "kv_cache_config": {"max_tokens": 1024, "free_gpu_memory_fraction": 0.05},
        "transforms": {
            "insert_cached_mla_attention": {"backend": "trtllm_mla"},
            "fuse_rope_into_trtllm_mla": {"enabled": True},
            "multi_stream_mla_attn": {"stage": "compile", "enabled": False},
        },
    }


def long_context_prompt():
    return (
        "TensorRT-LLM disaggregated serving separates context prefill from token generation. "
        "The context worker computes the prompt KV cache, sends the cache state to the "
        "generation worker, and returns the first generated token metadata. "
    )


def capital_completion_prompts():
    return [
        "The capital of Germany is",
        "The capital of France is",
        "The capital of Italy is",
        "The capital of Spain is",
    ]


def assert_context_handoff_metadata(context_output, expect_logits=False):
    context_params = context_output.disaggregated_params
    assert context_params is not None
    assert context_params.request_type == "context_only"
    assert len(context_output.token_ids) == 1
    assert context_params.ctx_request_id is not None
    assert context_params.first_gen_tokens is not None
    if expect_logits:
        assert context_params.first_gen_logits is not None
    assert has_handoff_transport_metadata(context_params)


def run_sequential_handoff(
    model,
    generation_overlap,
    prompt,
    sampling_params_kwargs=None,
    extra_config=None,
):
    if sampling_params_kwargs is None:
        sampling_params_kwargs = {"max_tokens": 25, "ignore_eos": True}

    model_name = model_path(model)
    with AutoDeployLLM(
        model=model_name,
        world_size=1,
        **context_config(extra_config),
    ) as context_llm:
        seed_disagg()
        context_output = context_llm.generate(
            prompt,
            sampling_params=SamplingParams(**sampling_params_kwargs),
            disaggregated_params=DisaggregatedParams(request_type="context_only"),
            use_tqdm=False,
        ).outputs[0]
        print(f"[AD DISAGG TEST] context output: {response_summary([context_output])}")
        generation_params = generation_params_from_context(context_output)

        # Keep the context-side sender alive while generation consumes the
        # handoff params.
        with AutoDeployLLM(
            model=model_name,
            world_size=1,
            **generation_config(generation_overlap, extra_config),
        ) as generation_llm:
            seed_disagg()
            generation_output = generation_llm.generate(
                prompt,
                sampling_params=SamplingParams(**sampling_params_kwargs),
                disaggregated_params=generation_params,
                use_tqdm=False,
            ).outputs[0]
            print(f"[AD DISAGG TEST] generation output: {response_summary([generation_output])}")

    return {
        "context": context_output,
        "generation": generation_output,
    }


async def run_async_requests(llm, prompts, sampling_params_kwargs, disaggregated_params):
    futures = []
    for prompt, params in zip(prompts, disaggregated_params, strict=True):
        seed_disagg()
        futures.append(
            llm.generate_async(
                prompt,
                sampling_params=SamplingParams(**sampling_params_kwargs),
                disaggregated_params=params,
            )
        )

    outputs = []
    for future in futures:
        result = await future
        outputs.append(result.outputs[0])
    return outputs


def run_sequential_batch_handoff(
    model,
    generation_overlap,
    prompts,
    sampling_params_kwargs=None,
    extra_config=None,
):
    if sampling_params_kwargs is None:
        sampling_params_kwargs = {"max_tokens": 25, "ignore_eos": True}

    model_name = model_path(model)
    context_params = [DisaggregatedParams(request_type="context_only") for _ in range(len(prompts))]
    with AutoDeployLLM(
        model=model_name,
        world_size=1,
        **context_config(extra_config),
    ) as context_llm:
        context_outputs = asyncio.run(
            run_async_requests(context_llm, prompts, sampling_params_kwargs, context_params)
        )
        print(f"[AD DISAGG TEST] context batch output: {response_summary(context_outputs)}")
        generation_params = [
            generation_params_from_context(context_output) for context_output in context_outputs
        ]

        # Submit all generation-only requests before awaiting them so this
        # remains a batch slot-transfer test while avoiding the async queue
        # infrastructure used by the multi-GPU tests.
        with AutoDeployLLM(
            model=model_name,
            world_size=1,
            **generation_config(generation_overlap, extra_config),
        ) as generation_llm:
            generation_outputs = asyncio.run(
                run_async_requests(
                    generation_llm,
                    prompts,
                    sampling_params_kwargs,
                    generation_params,
                )
            )
            print(
                f"[AD DISAGG TEST] generation batch output: {response_summary(generation_outputs)}"
            )

    return {
        "context": context_outputs,
        "generation": generation_outputs,
    }


def reduced_model_config(model, extra_config=None):
    if "DeepSeek-V3-Lite" in model:
        config = reduced_deepseek_v3_mla_config()
    else:
        config = reduced_tinyllama_config()
    if extra_config:
        config.update(extra_config)
    return config


def reduced_model_cases():
    return [
        pytest.param(
            "TinyLlama-1.1B-Chat-v1.0",
            id="tinyllama",
        ),
        pytest.param(
            "DeepSeek-V3-Lite",
            id="deepseek_v3_mla",
            marks=skip_pre_hopper,
        ),
    ]


@pytest.mark.parametrize(
    "model",
    reduced_model_cases(),
)
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.timeout(600)
def test_reduced_layer_handoff_matches_aggregate(model):
    """Check single-request disaggregated handoff matches aggregate generation."""
    prompt = "What is the capital of Germany?"
    sampling_params_kwargs = {
        "max_tokens": 8,
        "ignore_eos": True,
        "top_k": 1,
        "seed": AUTODEPLOY_DISAGG_SEED,
    }
    extra_config = reduced_model_config(model)
    # Keep real weights loaded, but reduce the decoder stack so this still
    # exercises the model-specific attention/cache path without full-model cost.
    aggregate_output = run_aggregate_generation(
        model,
        world_size=1,
        prompt=prompt,
        sampling_params_kwargs=sampling_params_kwargs,
        extra_config=extra_config,
    )
    outputs = run_sequential_handoff(
        model,
        generation_overlap=True,
        prompt=prompt,
        sampling_params_kwargs=sampling_params_kwargs,
        extra_config=extra_config,
    )

    context_output = outputs["context"]
    generation_output = outputs["generation"]
    assert_context_handoff_metadata(context_output)
    assert context_output.token_ids == aggregate_output.token_ids[:1]
    assert generation_output.text == aggregate_output.text
    assert generation_output.token_ids == aggregate_output.token_ids


@pytest.mark.parametrize(
    "model",
    reduced_model_cases(),
)
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.timeout(600)
def test_disaggregated_logits(model):
    # Keep weighted but reduced layers so the test focuses on logits
    # transfer/equality rather than full-model compile and memory cost.
    extra_config = reduced_model_config(model, {"gather_generation_logits": True})
    sampling_params_kwargs = {
        "max_tokens": 10,
        "ignore_eos": True,
        "return_generation_logits": True,
    }
    prompt = "What is the capital of Germany?"
    aggregate_output = run_aggregate_generation(
        model,
        world_size=1,
        prompt=prompt,
        sampling_params_kwargs=sampling_params_kwargs,
        extra_config=extra_config,
    )
    outputs = run_sequential_handoff(
        model,
        generation_overlap=True,
        prompt=prompt,
        sampling_params_kwargs=sampling_params_kwargs,
        extra_config=extra_config,
    )

    context_output = outputs["context"]
    generation_output = outputs["generation"]
    assert_context_handoff_metadata(context_output, expect_logits=True)
    assert context_output.token_ids == aggregate_output.token_ids[:1]
    assert generation_output.text == aggregate_output.text
    assert generation_output.token_ids == aggregate_output.token_ids
    assert aggregate_output.generation_logits is not None
    assert generation_output.generation_logits is not None
    assert aggregate_output.generation_logits.shape == generation_output.generation_logits.shape
    # The MLA generation worker reconstructs logits from the compressed KV latent
    # through a different kernel/batching path than the single aggregate pass, so
    # bf16 rounding yields ~1-ULP logit differences. Use a looser tolerance for the
    # MLA (DeepSeek) case; MHA (tinyllama) stays tight. The functional checks above
    # (text/token_ids equality) remain strict for both.
    if "DeepSeek-V3-Lite" in model:
        rtol, atol = 1e-1, 1e-1
    else:
        rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(
        generation_output.generation_logits,
        aggregate_output.generation_logits,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.timeout(600)
def test_tinyllama_batch_handoff_semantic_slots():
    prompts = capital_completion_prompts()
    expected_capitals = ["Berlin", "Paris", "Rome", "Madrid"]
    sampling_params_kwargs = {
        "max_tokens": 12,
        "ignore_eos": True,
        "top_k": 1,
        "seed": AUTODEPLOY_DISAGG_SEED,
    }
    outputs = run_sequential_batch_handoff(
        "TinyLlama-1.1B-Chat-v1.0",
        generation_overlap=True,
        prompts=prompts,
        sampling_params_kwargs=sampling_params_kwargs,
    )

    for expected_capital, context_output, generation_output in zip(
        expected_capitals, outputs["context"], outputs["generation"], strict=True
    ):
        assert_context_handoff_metadata(context_output)
        assert expected_capital.lower() in generation_output.text.lower(), response_summary(
            outputs["generation"]
        )


@pytest.mark.parametrize(
    "model",
    reduced_model_cases(),
)
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.timeout(600)
def test_chunked_prefill_handoff(model):
    # Chunked prefill needs real weights for aggregate-vs-disaggregated
    # comparison, but not a full decoder stack. Use reduced layers so the test
    # focuses on chunk-boundary handoff behavior with cuda graph enabled.
    extra_config = reduced_model_config(
        model,
        {
            "enable_chunked_prefill": True,
            "max_num_tokens": 96,
        },
    )
    prompt = long_context_prompt() * 4
    sampling_params_kwargs = {"max_tokens": 8, "ignore_eos": True}
    aggregate_output = run_aggregate_generation(
        model,
        world_size=1,
        prompt=prompt,
        sampling_params_kwargs=sampling_params_kwargs,
        extra_config=extra_config,
    )
    outputs = run_sequential_handoff(
        model,
        generation_overlap=True,
        prompt=prompt,
        sampling_params_kwargs=sampling_params_kwargs,
        extra_config=extra_config,
    )

    context_output = outputs["context"]
    generation_output = outputs["generation"]
    assert_context_handoff_metadata(context_output)
    assert generation_output.token_ids
    assert context_output.token_ids == aggregate_output.token_ids[:1]
    assert generation_output.text == aggregate_output.text
    assert generation_output.token_ids == aggregate_output.token_ids


# ---------------------------------------------------------------------------
# Async MPI worker tests.
#
# These tests launch separate context and generation worker processes and pass
# requests through an MPI intercommunicator. They are closer to the real
# disaggregated deployment shape because context and generation models can live on
# different GPUs, and some cases shard each worker across multiple GPUs.
# ---------------------------------------------------------------------------


def llama_eagle3_config():
    return {
        "speculative_config": Eagle3DecodingConfig(
            max_draft_len=3,
            speculative_model=model_path("EAGLE3-LLaMA3.1-Instruct-8B"),
            eagle3_one_model=True,
            eagle3_layers_to_capture={1, 15, 28},
        ),
        # Force the Eagle3 draft to match the BF16 Llama 3.1 target. Shared KV
        # cache management requires matching target and draft KV dtypes.
        "speculative_model_kwargs": {"torch_dtype": "bfloat16"},
    }


def get_ucx_tls():
    if get_sm_version() < 90:
        return "^cuda_ipc,ib,gdr_copy"
    return "^ib,gdr_copy"


def worker_cuda_devices(worker_world_sizes, visible_devices):
    required_devices = sum(worker_world_sizes)
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        if len(devices) < required_devices:
            pytest.skip(
                f"AutoDeploy disaggregated world sizes {worker_world_sizes} require "
                f"{required_devices} visible GPUs, got {len(devices)}"
            )
    else:
        devices = [str(device) for device in range(required_devices)]

    cuda_visible_devices = []
    start = 0
    for world_size in worker_world_sizes:
        end = start + world_size
        cuda_visible_devices.append(",".join(devices[start:end]))
        start = end
    return cuda_visible_devices


def worker_error(error):
    return f"{type(error).__name__}: {error}\n{traceback.format_exc()}"


def isolate_ad_worker_from_outer_mpi():
    """Hide pytest's MPI transport from AutoDeploy's distributed init."""
    rank = MPI.COMM_WORLD.Get_rank()
    ad_comm = MPI.COMM_WORLD.Split(color=rank, key=0)
    set_mpi_comm(ad_comm)
    for key in OMPI_COMM_WORLD_ENV_KEYS:
        os.environ.pop(key, None)
    return ad_comm


async def run_worker(
    config,
    model_name,
    world_size,
    cuda_visible_devices,
    service_name,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("UCX_TLS", get_ucx_tls())
    os.environ.setdefault("UCX_MM_ERROR_HANDLING", "y")

    intercomm = MPI.COMM_WORLD.Connect(MPI.Lookup_name(service_name))
    ad_comm = isolate_ad_worker_from_outer_mpi()
    try:
        seed_disagg()
        with AutoDeployLLM(
            model=model_name,
            world_size=world_size,
            **config,
        ) as llm:
            intercomm.send(WORKER_READY, dest=0, tag=MPI_RESULT)
            while True:
                requests = intercomm.recv(source=0, tag=MPI_REQUEST)
                if requests is None:
                    break

                futures = []
                for request in requests:
                    seed_disagg()
                    try:
                        result = llm.generate_async(
                            request[0],
                            sampling_params=request[1],
                            disaggregated_params=request[2],
                        )
                        futures.append(result)
                    except Exception as e:
                        intercomm.send(worker_error(e), dest=0, tag=MPI_RESULT)

                for result in futures:
                    try:
                        output = await result
                        intercomm.send(output.outputs, dest=0, tag=MPI_RESULT)
                    except Exception as e:
                        intercomm.send(worker_error(e), dest=0, tag=MPI_RESULT)
    except Exception as e:
        intercomm.send(worker_error(e), dest=0, tag=MPI_RESULT)
        raise
    finally:
        intercomm.Disconnect()
        ad_comm.Free()


def worker_entry_point(
    config,
    model_name,
    world_size,
    cuda_visible_devices,
    service_name,
):
    return asyncio.run(
        run_worker(
            config,
            model_name,
            world_size,
            cuda_visible_devices,
            service_name,
        )
    )


def mpi_publish_name():
    service_name = f"ad_disagg_{uuid.uuid4()}"
    port_name = MPI.Open_port()
    MPI.Publish_name(service_name, port_name)
    return service_name, port_name


def send_requests_to_worker(requests, worker_rank, intercomms):
    intercomm = intercomms[worker_rank]
    intercomm.send(requests, dest=0, tag=MPI_REQUEST)
    responses = []
    for _ in range(len(requests)):
        responses.append(intercomm.recv(source=0, tag=MPI_RESULT))
    return responses


@contextmanager
def worker_pool(worker_configs, model_names, world_sizes):
    """Start async MPI workers and always tear them down after the test body.

    MPI cloudpickle serialization keeps worker callables by value, so CI workers
    do not re-import this pytest module from the source checkout before the
    installed TensorRT-LLM wheel is on the import path.
    """
    if len(worker_configs) != len(model_names) or len(model_names) != len(world_sizes):
        raise ValueError("worker_configs, model_names, and world_sizes must have the same length")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_visible_devices = worker_cuda_devices(world_sizes, visible_devices)
    services = [mpi_publish_name() for _ in world_sizes]
    futures = []
    intercomms = []
    with ExitStack() as stack:
        try:
            for (
                config,
                model_name,
                world_size,
                worker_cuda_visible_devices,
                (service_name, _),
            ) in zip(
                worker_configs,
                model_names,
                world_sizes,
                cuda_visible_devices,
                services,
                strict=True,
            ):
                executor = stack.enter_context(
                    MPIPoolExecutor(
                        max_workers=1,
                        path=sys.path,
                        env={
                            "UCX_TLS": get_ucx_tls(),
                            "UCX_MM_ERROR_HANDLING": "y",
                        },
                    )
                )
                futures.append(
                    executor.submit(
                        worker_entry_point,
                        config,
                        model_name,
                        world_size,
                        worker_cuda_visible_devices,
                        service_name,
                    )
                )

            for _, port_name in services:
                intercomms.append(MPI.COMM_SELF.Accept(port_name))
            for intercomm in intercomms:
                ready_response = intercomm.recv(source=0, tag=MPI_RESULT)
                if ready_response != WORKER_READY:
                    raise RuntimeError(
                        f"Unexpected AutoDeploy worker startup response: {ready_response}"
                    )
            yield intercomms
        finally:
            for intercomm in intercomms:
                intercomm.send(None, dest=0, tag=MPI_REQUEST)
                intercomm.Disconnect()
            for service_name, port_name in services:
                MPI.Unpublish_name(service_name, port_name)
                MPI.Close_port(port_name)
            for future in futures:
                future.result()


def run_context_then_generation_handoff(
    model,
    worker_world_sizes,
    generation_overlap,
    prompt,
    sampling_params_kwargs=None,
    extra_config=None,
):
    """Run one AutoDeploy disaggregated context-to-generation handoff.

    This launches a context worker and a generation worker. It sends one
    context-only request to the context worker, turns the returned
    ``DisaggregatedParams`` into a generation-only request, and sends that to
    the generation worker.

    Returns:
        dict with ``context`` and ``generation`` outputs. The caller owns all
        behavioral assertions, including output text, handoff metadata, logits,
        or draft-token checks.
    """
    worker_configs = [
        context_config(extra_config),
        generation_config(generation_overlap, extra_config),
    ]
    print(
        "[AD DISAGG TEST] "
        f"scenario start: model={model}, worker_world_sizes={worker_world_sizes}, "
        f"generation_overlap={generation_overlap}, compile_backend=torch-cudagraph, "
    )
    if sampling_params_kwargs is None:
        sampling_params_kwargs = {"max_tokens": 25, "ignore_eos": True}

    model_names = [model_path(model) for _ in range(2)]
    world_sizes = list(worker_world_sizes)

    with worker_pool(worker_configs, model_names, world_sizes) as intercomms:
        context_requests = [
            (
                prompt,
                SamplingParams(**sampling_params_kwargs),
                DisaggregatedParams(request_type="context_only"),
            )
        ]
        context_responses = send_requests_to_worker(context_requests, 0, intercomms)
        context_output = first_output(context_responses)
        print(
            f"[AD DISAGG TEST] context output: {response_summary([context_output])}",
        )

        generation_request_disagg_params = generation_params_from_context(context_output)
        generation_requests = [
            (prompt, SamplingParams(**sampling_params_kwargs), generation_request_disagg_params)
        ]

        generation_responses = send_requests_to_worker(generation_requests, 1, intercomms)
        generation_output = first_output(generation_responses)
        print(
            f"[AD DISAGG TEST] generation output: {response_summary([generation_output])}",
        )

        return {
            "context": context_output,
            "generation": generation_output,
        }


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.skip_less_device(2)
@pytest.mark.timeout(600)
def test_async_generation_matches_aggregate():
    aggregate_output = run_aggregate_generation(
        "TinyLlama-1.1B-Chat-v1.0",
        world_size=1,
        prompt="What is the capital of Germany?",
        sampling_params_kwargs={"max_tokens": 10, "ignore_eos": True},
    )
    outputs = run_context_then_generation_handoff(
        "TinyLlama-1.1B-Chat-v1.0",
        worker_world_sizes=(1, 1),
        generation_overlap=True,
        prompt="What is the capital of Germany?",
        sampling_params_kwargs={"max_tokens": 10, "ignore_eos": True},
    )
    context_params = outputs["context"].disaggregated_params
    assert context_params is not None
    assert context_params.request_type == "context_only"
    assert len(outputs["context"].token_ids) == 1
    assert context_params.ctx_request_id is not None
    assert context_params.first_gen_tokens is not None
    assert has_handoff_transport_metadata(context_params)
    assert outputs["generation"].token_ids
    assert outputs["context"].token_ids == aggregate_output.token_ids[:1]
    assert outputs["generation"].text == aggregate_output.text
    assert outputs["generation"].token_ids == aggregate_output.token_ids


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.skip_less_device(2)
@pytest.mark.timeout(600)
def test_async_generation_no_overlap_matches_aggregate():
    """Match aggregate generation with the generation worker on overlap=off.

    Same shape as test_async_generation_matches_aggregate but with the
    generation worker on the non-overlap scheduling path. Covers MHA disagg
    with overlap=off against the aggregate baseline.
    """
    sampling_params_kwargs = {"max_tokens": 10, "ignore_eos": True}
    aggregate_output = run_aggregate_generation(
        "TinyLlama-1.1B-Chat-v1.0",
        world_size=1,
        prompt="What is the capital of Germany?",
        sampling_params_kwargs=sampling_params_kwargs,
    )
    outputs = run_context_then_generation_handoff(
        "TinyLlama-1.1B-Chat-v1.0",
        worker_world_sizes=(1, 1),
        generation_overlap=False,
        prompt="What is the capital of Germany?",
        sampling_params_kwargs=sampling_params_kwargs,
    )
    assert_context_handoff_metadata(outputs["context"])
    assert outputs["generation"].token_ids
    assert outputs["context"].token_ids == aggregate_output.token_ids[:1]
    assert outputs["generation"].text == aggregate_output.text
    assert outputs["generation"].token_ids == aggregate_output.token_ids


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.skip_less_device(4)
@pytest.mark.timeout(900)
def test_async_sharded_generation_handoff():
    aggregate_output = run_aggregate_generation(
        "TinyLlama-1.1B-Chat-v1.0",
        world_size=2,
        prompt="What is the capital of Germany?",
        sampling_params_kwargs={"max_tokens": 10, "ignore_eos": True},
    )
    outputs = run_context_then_generation_handoff(
        "TinyLlama-1.1B-Chat-v1.0",
        worker_world_sizes=(2, 2),
        generation_overlap=True,
        prompt="What is the capital of Germany?",
        sampling_params_kwargs={"max_tokens": 10, "ignore_eos": True},
    )
    assert_context_handoff_metadata(outputs["context"])
    assert outputs["generation"].token_ids
    assert outputs["context"].token_ids == aggregate_output.token_ids[:1]
    assert outputs["generation"].text == aggregate_output.text
    assert outputs["generation"].token_ids == aggregate_output.token_ids


@skip_pre_hopper
@pytest.mark.threadleak(enabled=False)
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(2)
@pytest.mark.timeout(900)
def test_async_eagle3_full_model_handoff():
    sampling_params_kwargs = {
        "max_tokens": 16,
        "ignore_eos": True,
        "top_k": 1,
        "seed": AUTODEPLOY_DISAGG_SEED,
    }
    extra_config = llama_eagle3_config()
    outputs = run_context_then_generation_handoff(
        "Llama-3.1-8B-Instruct",
        worker_world_sizes=(1, 1),
        generation_overlap=True,
        prompt="What is the capital of Germany?",
        sampling_params_kwargs=sampling_params_kwargs,
        extra_config=extra_config,
    )
    context_params = outputs["context"].disaggregated_params
    assert context_params is not None
    assert context_params.request_type == "context_only"
    assert len(outputs["context"].token_ids) == 1
    assert context_params.ctx_request_id is not None
    assert context_params.first_gen_tokens is not None
    assert has_handoff_transport_metadata(context_params)
    assert outputs["generation"].token_ids
    assert has_draft_tokens(outputs["context"])
    assert has_draft_tokens(outputs["generation"])
    assert outputs["context"].text == " Berlin"
    assert outputs["context"].token_ids == LLAMA_EAGLE3_EXPECTED_TOKEN_IDS[:1]
    assert outputs["generation"].text == LLAMA_EAGLE3_EXPECTED_TEXT
    assert outputs["generation"].token_ids == LLAMA_EAGLE3_EXPECTED_TOKEN_IDS
