# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import global_kvcache_config, llm_test_harness, prompts
from .model_paths import llama_model_path


def run_llm_with_postprocess_parallel(tp_size: int = 1):
    sampling_params = SamplingParams(max_tokens=6)

    postproc_settings = dict(num_postprocess_workers=2, postprocess_tokenizer_dir=llama_model_path)

    llm_test_harness(
        llama_model_path,
        prompts,
        ["D E F G H I J K"],
        sampling_params=sampling_params,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
        **postproc_settings,
    )


def run_llm_with_postprocess_parallel_and_result_handler(streaming, backend, tp_size: int = 1):
    # avoid import error when running in CI
    from tensorrt_llm.executor.postproc_worker import PostprocParams
    from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer
    from tensorrt_llm.serve.postprocess_handlers import ChatPostprocArgs, chat_stream_post_processor

    from .run_llm_with_postproc import get_concatenated_content

    sampling_params = SamplingParams(max_tokens=6)
    tokenizer = load_hf_tokenizer(llama_model_path)
    post_proc_args = ChatPostprocArgs(tokenizer=tokenizer, role="assistant", model=llama_model_path)
    post_proc_params = PostprocParams(
        post_processor=chat_stream_post_processor, postproc_args=post_proc_args
    )

    llm = LLM(
        model=llama_model_path,
        backend=backend,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
        num_postprocess_workers=2,
        postprocess_tokenizer_dir=llama_model_path,
    )
    # Qwen3's byte-level BPE tokenizer folds the space before the first
    # generated token into that token itself, so the decoded continuation
    # of "A B C" carries a leading space that TinyLlama's tokenizer didn't.
    golden_result = " D E F G H I"
    outputs = []
    for output in llm.generate_async(
        prompts[0],
        sampling_params=sampling_params,
        _postproc_params=post_proc_params,
        streaming=streaming,
    ):
        outputs.append(output.outputs[0]._postprocess_result)
    actual_result = get_concatenated_content(outputs)
    assert actual_result == golden_result, f"Expected: {golden_result}, Actual: {actual_result}"
