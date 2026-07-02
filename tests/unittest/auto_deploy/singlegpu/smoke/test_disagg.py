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
from dataclasses import replace

import pytest
from _model_test_utils import get_small_model_config
from utils.util import skip_pre_hopper

from tensorrt_llm import DisaggregatedParams, SamplingParams
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.llmapi import Eagle3DecodingConfig

LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EAGLE3_MODEL_ID = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V3"
DEEPSEEK_DISAGG_TRANSFORMS = {
    "insert_cached_attention": {"backend": "triton"},
    "insert_cached_mla_attention": {"backend": "trtllm_mla"},
    "fuse_rope_into_trtllm_mla": {"enabled": True},
    "compile_model": {"backend": "torch-simple"},
}


def small_model_config_disagg(model_id, attn_backend, compile_backend, **overrides):
    args = get_small_model_config(model_id)["args"]
    kv_cache_config = args["kv_cache_config"]
    kv_cache_config.update(
        {
            "tokens_per_block": 4,
            "max_tokens": 64,
            "free_gpu_memory_fraction": 0.001,
        }
    )
    args.update(
        {
            "world_size": 1,
            "runtime": "trtllm",
            "skip_tokenizer_init": True,
            "attn_backend": attn_backend,
            "compile_backend": compile_backend,
            "cuda_graph_config": {"max_batch_size": 2}
            if compile_backend == "torch-cudagraph"
            else None,
            "max_batch_size": 2,
            "max_seq_len": 64,
            "max_num_tokens": 16,
            "cache_transceiver_config": {"backend": "DEFAULT"},
        }
    )
    args.update(overrides)
    return args


def _sampling_params():
    return SamplingParams(
        max_tokens=4,
        ignore_eos=True,
        add_special_tokens=False,
        end_id=2,
        pad_id=0,
    )


def has_handoff_transport_metadata(params):
    # C++ transceiver carries handoff state in opaque_state; Python/native
    # transceiver carries the context endpoint in ctx_info_endpoint.
    return params.opaque_state is not None or params.ctx_info_endpoint is not None


def context_output_valid(output):
    params = output.disaggregated_params
    return (
        params is not None
        and params.request_type == "context_only"
        and len(output.token_ids) == 1
        and params.ctx_request_id is not None
        and params.first_gen_tokens is not None
        and has_handoff_transport_metadata(params)
    )


def create_generation_params(context_output):
    assert context_output_valid(context_output)
    params = context_output.disaggregated_params
    assert params is not None
    return replace(params, request_type="generation_only")


def has_draft_tokens(output):
    params = output.disaggregated_params
    return params is not None and params.draft_tokens is not None and len(params.draft_tokens) > 0


def run_live_disagg_smoke(
    model_id,
    attn_backend,
    compile_backend,
    common_config_overrides=None,
    context_config_overrides=None,
    generation_config_overrides=None,
):
    common_config_overrides = common_config_overrides or {}
    context_config_overrides = context_config_overrides or {}
    generation_config_overrides = generation_config_overrides or {}

    with AutoDeployLLM(
        **small_model_config_disagg(
            model_id,
            attn_backend,
            compile_backend,
            **common_config_overrides,
            disable_overlap_scheduler=True,
            **context_config_overrides,
        )
    ) as context_llm:
        context_output = context_llm.generate(
            [1, 2, 3, 4],
            sampling_params=_sampling_params(),
            disaggregated_params=DisaggregatedParams(request_type="context_only"),
        ).outputs[0]
        assert context_output_valid(context_output)
        disaggregated_params = create_generation_params(context_output)

        # Keep the context LLM alive while the generation LLM consumes the
        # handoff params. The real cache transceiver uses the context-side
        # sender endpoint, so this is the meaningful generation-only smoke.
        with AutoDeployLLM(
            **small_model_config_disagg(
                model_id,
                attn_backend,
                compile_backend,
                **common_config_overrides,
                **generation_config_overrides,
            )
        ) as generation_llm:
            generation_output = generation_llm.generate(
                [1, 2, 3, 4],
                sampling_params=_sampling_params(),
                disaggregated_params=disaggregated_params,
            ).outputs[0]
            assert generation_output.token_ids
            return context_output, generation_output


async def run_async_requests(llm, prompts, sampling_params, disaggregated_params):
    futures = []
    for prompt, params in zip(prompts, disaggregated_params, strict=True):
        futures.append(
            llm.generate_async(
                prompt,
                sampling_params=sampling_params,
                disaggregated_params=params,
            )
        )
    return [(await future).outputs[0] for future in futures]


def run_live_batch_disagg_smoke(model_id, attn_backend, compile_backend, config_overrides):
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8]]
    context_params = [DisaggregatedParams(request_type="context_only") for _ in prompts]

    with AutoDeployLLM(
        **small_model_config_disagg(
            model_id,
            attn_backend,
            compile_backend,
            **config_overrides,
            disable_overlap_scheduler=True,
        )
    ) as context_llm:
        context_outputs = asyncio.run(
            run_async_requests(context_llm, prompts, _sampling_params(), context_params)
        )
        generation_params = [
            create_generation_params(context_output) for context_output in context_outputs
        ]

        with AutoDeployLLM(
            **small_model_config_disagg(
                model_id,
                attn_backend,
                compile_backend,
                **config_overrides,
            )
        ) as generation_llm:
            generation_outputs = asyncio.run(
                run_async_requests(
                    generation_llm,
                    prompts,
                    _sampling_params(),
                    generation_params,
                )
            )

    for context_output, generation_output in zip(context_outputs, generation_outputs, strict=True):
        assert context_output_valid(context_output)
        assert generation_output.token_ids


GENERIC_DISAGG_SMOKE_CASES = [
    pytest.param(LLAMA_MODEL_ID, "trtllm", "torch-simple", {}, id="llama-trtllm-simple"),
    pytest.param(LLAMA_MODEL_ID, "trtllm", "torch-cudagraph", {}, id="llama-trtllm-cudagraph"),
    pytest.param(LLAMA_MODEL_ID, "flashinfer", "torch-simple", {}, id="llama-flashinfer-simple"),
    pytest.param(
        LLAMA_MODEL_ID,
        "flashinfer",
        "torch-cudagraph",
        {},
        id="llama-flashinfer-cudagraph",
    ),
    pytest.param(
        DEEPSEEK_MODEL_ID,
        "trtllm",
        "torch-simple",
        {"transforms": DEEPSEEK_DISAGG_TRANSFORMS},
        marks=skip_pre_hopper,
        id="deepseek-trtllm-simple",
    ),
]


@pytest.mark.parametrize(
    ("model_id", "attn_backend", "compile_backend", "config_overrides"),
    GENERIC_DISAGG_SMOKE_CASES,
)
def test_autodeploy_disaggregated_smoke(model_id, attn_backend, compile_backend, config_overrides):
    if model_id == DEEPSEEK_MODEL_ID:
        pytest.importorskip(
            "transformers.models.deepseek_v3.configuration_deepseek_v3",
            reason="DeepseekV3Config requires a newer transformers version",
        )

    run_live_disagg_smoke(model_id, attn_backend, compile_backend, config_overrides)


@pytest.mark.parametrize(
    ("model_id", "attn_backend", "compile_backend", "config_overrides"),
    GENERIC_DISAGG_SMOKE_CASES,
)
def test_autodeploy_disaggregated_batch_smoke(
    model_id, attn_backend, compile_backend, config_overrides
):
    if model_id == DEEPSEEK_MODEL_ID:
        pytest.importorskip(
            "transformers.models.deepseek_v3.configuration_deepseek_v3",
            reason="DeepseekV3Config requires a newer transformers version",
        )

    run_live_batch_disagg_smoke(model_id, attn_backend, compile_backend, config_overrides)


def test_autodeploy_disaggregated_eagle3_smoke():
    target_model_config = get_small_model_config(LLAMA_MODEL_ID)
    eagle3_model_config = get_small_model_config(EAGLE3_MODEL_ID)
    target_model_kwargs = {
        **target_model_config["args"]["model_kwargs"],
        "num_hidden_layers": 3,
    }
    speculative_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model=eagle3_model_config["args"]["model"],
        eagle3_one_model=True,
        eagle3_layers_to_capture={0, 1, 2},
    )
    speculative_model_kwargs = {
        **target_model_kwargs,
        **eagle3_model_config["args"]["model_kwargs"],
        "torch_dtype": "bfloat16",
    }

    # This is intentionally a smoke test: small_model_config_disagg uses
    # skip_loading_weights=True, so the meaningful assertions are that one-model
    # Eagle builds with a reduced target/draft pair and carries draft-token
    # metadata through the live disaggregated handoff. Force the draft dtype to
    # match the BF16 Llama target because shared KV cache management requires
    # target and draft KV resources to have the same dtype. Use three reduced
    # target layers to match Llama Eagle3's default three-layer capture.
    # Weighted acceptance and quality coverage belong in integration tests.
    context_output, generation_output = run_live_disagg_smoke(
        LLAMA_MODEL_ID,
        "flashinfer",
        "torch-simple",
        common_config_overrides={
            "model_kwargs": target_model_kwargs,
            "speculative_config": speculative_config,
            "speculative_model_kwargs": speculative_model_kwargs,
        },
    )
    assert has_draft_tokens(context_output)
    assert has_draft_tokens(generation_output)
