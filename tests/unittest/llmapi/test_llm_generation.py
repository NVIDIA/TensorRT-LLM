# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from utils.util import force_ampere

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import (
    _test_llm_generate_async,
    global_kvcache_config,
    llm_check_output,
    prompts,
)
from .model_paths import get_model_path, llama_model_path, qwen3_tokenizer_model_name

pytestmark = pytest.mark.threadleak(enabled=False)


def test_llm_generate_async():
    _test_llm_generate_async()


@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.mpi_ray_parity
def test_llm_generate_async_with_stream_interval(chunked):
    model_path = get_model_path("llama-models-v2/llama-v2-7b-hf")
    max_num_tokens = 256
    with LLM(
        model_path, max_num_tokens=max_num_tokens, stream_interval=4, enable_chunked_prefill=chunked
    ) as llm:
        sampling_params = SamplingParams(max_tokens=13, ignore_eos=True, detokenize=False)
        step = 0
        last_step_len = 0
        prompt = "The capital of France is "
        if chunked:
            prompt = prompt * max_num_tokens
        for output in llm.generate_async(prompt, sampling_params=sampling_params, streaming=True):
            current_step_len = len(output.outputs[0].token_ids)
            # The output lens of each step need to be [1, 3, 4, 4, 1]
            if step == 0:
                assert current_step_len == 1
            elif step == 1:
                assert current_step_len - last_step_len == 3
            elif step == 2 or step == 3:
                assert current_step_len - last_step_len == 4
            else:
                assert current_step_len - last_step_len == 1
            step += 1
            last_step_len = current_step_len


@force_ampere
def test_generate_with_stop_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    )
    # The model continues "A B C" with space-separated letters, so the token
    # it generates for N is the space-prefixed " N"; with BPE tokenizers
    # (e.g. Qwen3) that differs from the token of a bare "N".
    stop_id = llm.tokenizer.encode(" N", add_special_tokens=False)[-1]

    llm_check_output(
        llm,
        prompts,
        ["D E F G H I J K L M"],
        sampling_params=SamplingParams(end_id=stop_id),
        finish_reasons=["stop"],
        stop_reasons=[None],
    )

    llm_check_output(
        llm,
        prompts,
        ["D E F G H"],
        sampling_params=SamplingParams(max_tokens=5),
        finish_reasons=["length"],
        stop_reasons=[None],
    )

    llm_check_output(
        llm,
        prompts,
        ["D E F G H I J K L M"],
        sampling_params=SamplingParams(stop_token_ids=[stop_id]),
        finish_reasons=["stop"],
        stop_reasons=[stop_id],
    )

    llm_check_output(
        llm,
        prompts,
        ["D E F G H I J K L M N"],
        sampling_params=SamplingParams(stop_token_ids=[stop_id], include_stop_str_in_output=True),
        finish_reasons=["stop"],
        stop_reasons=[stop_id],
    )

    llm_check_output(
        llm,
        prompts,
        ["D E F G H"],
        sampling_params=SamplingParams(stop="I J"),
        finish_reasons=["stop"],
        stop_reasons=["I J"],
    )

    llm_check_output(
        llm,
        prompts,
        ["D E F G H I J K L M"],
        sampling_params=SamplingParams(stop="I E", max_tokens=10),
        finish_reasons=["length"],
        stop_reasons=[None],
    )

    llm_check_output(
        llm,
        prompts,
        ["D E F G H I J"],
        sampling_params=SamplingParams(stop="I J", include_stop_str_in_output=True),
        finish_reasons=["stop"],
        stop_reasons=["I J"],
    )

    llm_check_output(
        llm,
        prompts,
        ["D E F G H"],
        sampling_params=SamplingParams(stop=["F E", "I J"], stop_token_ids=[stop_id]),
        finish_reasons=["stop"],
        stop_reasons=["I J"],
    )


@force_ampere
@pytest.mark.parametrize(
    "model_path",
    [
        get_model_path(qwen3_tokenizer_model_name),
    ],
)
def test_generate_with_detokenization_stop_words(model_path):
    llm = LLM(
        model=model_path,
        kv_cache_config=global_kvcache_config,
    )

    # Format the prompt using chat template
    messages = [{"role": "user", "content": "Say exactly: Hello there! How can I help"}]

    formatted_prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    detokenization_prompts = [formatted_prompt]

    # Test case 1: Stop word "How" should be detected after detokenization.
    # Use an exact text match (rather than llm_check_output's fuzzy
    # similarity check) to confirm the stop string is actually stripped from
    # the output, not merely close enough to pass a SequenceMatcher ratio.
    outputs = llm.generate(
        detokenization_prompts, sampling_params=SamplingParams(stop="How", max_tokens=10)
    )
    out = outputs[0].outputs[0]
    assert out.finish_reason == "stop"
    assert out.stop_reason == "How"
    # Truncation happens in text space (`text[:text.find(stop)]`), so the
    # whitespace that separated "there!" from the stop string belongs to the
    # retained prefix -- matching OpenAI/vLLM stop-string semantics.
    assert out.text == "Hello there! ", (
        f"Stop string 'How' must not be retained in output text, got: {out.text!r}"
    )

    # Test case 2: Stop word "there" should be detected after detokenization
    llm_check_output(
        llm,
        detokenization_prompts,
        ["Hello"],
        sampling_params=SamplingParams(stop="there", max_tokens=10),
        finish_reasons=["stop"],
        stop_reasons=["there"],
    )

    # Test case 3: Stop word that should not be found after detokenization
    llm_check_output(
        llm,
        detokenization_prompts,
        ["Hello there! How can I help"],
        sampling_params=SamplingParams(stop="XYZ", max_tokens=10),
        finish_reasons=["length"],
        stop_reasons=[None],
    )

    # Test case 4: Multiple stop words, one should be found after detokenization
    llm_check_output(
        llm,
        detokenization_prompts,
        ["Hello"],
        sampling_params=SamplingParams(stop=["XYZ", "there"], max_tokens=10),
        finish_reasons=["stop"],
        stop_reasons=["there"],
    )


@force_ampere
@pytest.mark.parametrize(
    "model_path",
    [
        get_model_path(qwen3_tokenizer_model_name),
    ],
)
def test_generate_with_detokenization_stop_words_streaming(model_path):
    llm = LLM(
        model=model_path,
        kv_cache_config=global_kvcache_config,
    )

    # Format the prompt using chat template
    messages = [{"role": "user", "content": "Say exactly: Hello there! How can I help"}]

    formatted_prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    sampling_params = SamplingParams(stop="How", max_tokens=10)

    # NOTE: without tracking `found_stop` and asserting on it after the loop,
    # this test can silently pass if generate_async yields no outputs at all
    # or never reaches a terminal finish_reason.
    found_stop = False
    for output in llm.generate_async(
        formatted_prompt, sampling_params=sampling_params, streaming=True
    ):
        if output.outputs[0].finish_reason == "stop":
            assert output.outputs[0].stop_reason == "How"
            # Trailing space is retained by text-space truncation; see the
            # non-streaming test above.
            assert output.outputs[0].text == "Hello there! "
            found_stop = True
            break
        elif output.outputs[0].finish_reason == "length":
            assert False, (
                f"Expected to find stop word 'How' but reached max_tokens. Generated: {output.outputs[0].text}"
            )
        elif output.outputs[0].finish_reason not in (None, ""):
            assert False, f"Unexpected finish_reason: {output.outputs[0].finish_reason}"

    assert found_stop, (
        "Expected to observe finish_reason='stop' with stop_reason='How', but no such output was produced"
    )

    # Test case: unmatched stop string should not trigger early stopping
    sampling_params_no_match = SamplingParams(stop="XYZ", max_tokens=10)
    found_length = False
    for output in llm.generate_async(
        formatted_prompt, sampling_params=sampling_params_no_match, streaming=True
    ):
        if output.outputs[0].finish_reason == "length":
            assert output.outputs[0].stop_reason is None
            found_length = True
            break
        elif output.outputs[0].finish_reason not in (None, ""):
            assert False, f"Unexpected finish_reason: {output.outputs[0].finish_reason}"

    assert found_length, (
        "Expected to observe finish_reason='length' with stop_reason=None, but no such output was produced"
    )


@force_ampere
def test_generate_with_bad_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    )

    # The model continues "A B C" with space-separated letters, so the tokens
    # it generates are the space-prefixed ones (" N", " I", " J"); with BPE
    # tokenizers (e.g. Qwen3) these differ from the tokens of bare "N" / "I J".
    bad_id = llm.tokenizer.encode(" N", add_special_tokens=False)[-1]

    llm_check_output(
        llm,
        prompts,
        ["D E F G H I J K L M\nA B C D"],
        sampling_params=SamplingParams(max_tokens=15, bad_token_ids=[bad_id]),
    )

    # Similarity alone also accepts the unbanned " ... I J K ..." output, so
    # also check that the banned sequence was not generated.
    outputs = llm_check_output(
        llm,
        prompts,
        ["D E F G H I K L M N O P Q R S"],
        sampling_params=SamplingParams(max_tokens=15, bad=" I J"),
    )
    assert " I J" not in outputs[0].outputs[0].text

    outputs = llm_check_output(
        llm,
        prompts,
        ["D E F G H I K L M N O P Q R S"],
        sampling_params=SamplingParams(max_tokens=15, bad=[" F E", " I J"]),
    )
    assert " I J" not in outputs[0].outputs[0].text
