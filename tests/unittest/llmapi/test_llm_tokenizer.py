# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import datasets
import pytest
from utils.util import similar

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer

from .llm_test_utils import alpaca_chinese_path, cnn_dailymail_path
from .model_paths import get_model_path, llama_model_path, qwen3_tokenizer_model_name

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.mark.parametrize("backend", ["HF", "TRTLLM"])
@pytest.mark.parametrize(
    "tokenizer_dir, clean_up_tokenization_spaces, threshold",
    [
        (get_model_path("gpt2"), False, 0.95),  # BPE
        (get_model_path("bert/bert-base-uncased"), True, 0.95),  # WordPiece
        (get_model_path("t5-small"), True, 0.95),  # SentencePiece
        (get_model_path("llama-models-v2/llama-v2-7b-hf"), False, 0.95),
        (get_model_path("codellama/CodeLlama-7b-Instruct-hf"), False, 0.95),
        (llama_model_path, False, 0.95),
        (get_model_path(qwen3_tokenizer_model_name), False, 0.95),
        (get_model_path("DeepSeek-R1/DeepSeek-R1"), False, 0.95),
    ],
)
def test_tokenizer_decode_incrementally(
    tokenizer_dir: str, clean_up_tokenization_spaces: bool, threshold: float, backend: str, mocker
):
    import tensorrt_llm.llmapi.tokenizer

    mocker.patch.object(
        tensorrt_llm.llmapi.tokenizer, "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND", backend
    )
    assert tensorrt_llm.llmapi.tokenizer.TLLM_INCREMENTAL_DETOKENIZATION_BACKEND == backend

    random.seed(42)

    num_samples = 100
    cnn_dailymail = datasets.load_dataset(cnn_dailymail_path, name="3.0.0", split="train")
    alpaca_chinese = datasets.load_dataset(
        alpaca_chinese_path, split="train", trust_remote_code=True
    )
    dataset = (
        cnn_dailymail["article"][: num_samples // 2]
        + alpaca_chinese["output_zh"][: num_samples // 2]
    )

    tokenizer = TransformersTokenizer.from_pretrained(
        tokenizer_dir,
        legacy=False,
        padding_side="left",
        truncation_side="left",
        trust_remote_code=True,
        use_fast=True,
    )

    num_perfect = 0
    for text in dataset:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        seq_len = len(token_ids)
        prompt_len = random.randint(1, seq_len // 2)
        decoded_text, states = tokenizer.decode_incrementally(token_ids[:prompt_len])
        for i in range(prompt_len, len(token_ids)):
            decoded_text, states = tokenizer.decode_incrementally(
                [token_ids[i]], decoded_text, states
            )

        if clean_up_tokenization_spaces and tokenizer.clean_up_tokenization_spaces:
            decoded_text = tokenizer.clean_up_tokenization(decoded_text)
        reference = tokenizer.decode(token_ids)
        if decoded_text == reference:
            num_perfect += 1
        else:
            # For non-perfect matching cases, decoded_text should also be very similar to the reference
            assert similar(decoded_text, reference, 0.99)
    print(f"Perfect matching ratio: {num_perfect / num_samples * 100}%")
    assert num_perfect / num_samples >= threshold
