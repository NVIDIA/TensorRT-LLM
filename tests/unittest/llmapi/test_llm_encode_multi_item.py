# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from typing import Generator, Optional, TypeVar, cast

import pytest
import torch
from transformers import AutoTokenizer
from utils.util import assert_no_cuda_sync

import tensorrt_llm._torch.attention_backend.flashinfer as trtllm_flashinfer
from tensorrt_llm import LLM
from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.distributed import AllReduceParams
from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Model
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm.llmapi.llm import EncoderOutput, PromptInputs

from .test_llm import get_model_path


def _build_prompt(
    question_str: str, answer_str: str, *, max_num_items: int = -1
) -> tuple[list[int], list[int]]:
    delimiter = "A"  # chosen to consist of a single token
    prefix = f"Is the following a good answer to {question_str}?"
    items = [
        f"Candidate answer 1: {answer_str}",
        f"Longer candidate answer 2: {answer_str}",
        f"Even longer candidate answer 3: {answer_str}",
    ]
    items = items[:max_num_items]

    tokenizer = AutoTokenizer.from_pretrained(_QWEN3_PATH)

    def _tokenize(s: str) -> list[int]:
        return tokenizer(s, return_attention_mask=False)["input_ids"]

    delimiter_tokens, prefix_tokens, *items_tokens = map(
        _tokenize,
        [
            delimiter,
            prefix,
            *items,
        ],
    )
    (delimiter_token,) = delimiter_tokens

    prompt_tokens = (
        prefix_tokens
        + [delimiter_token]
        + [
            item_token
            for item_tokens in items_tokens
            for item_token in item_tokens + [delimiter_token]
        ]
    )
    prefix_len = len(prefix_tokens)
    items_lens = list(map(len, items_tokens))

    # sanity check
    assert sum(items_lens) + len(items_lens) + prefix_len + 1 == len(prompt_tokens)

    return prompt_tokens, [prefix_len] + items_lens


T = TypeVar("T", torch.Tensor, list[int])


def _split_multi_item_sequence(sequence: T, prompt_components_lens: list[int]) -> tuple[T, list[T]]:
    # Slice multi-item sequence (prefix, delim, item1, delim, ..., itemN, delim) to extract
    # positions corresponding to (prefix, [item1, ..., itemN]).
    prefix_len, *prompt_components_lens = prompt_components_lens
    item_start = prefix_len + 1  # skip delimiter
    item_seqs = []
    for item_len in prompt_components_lens:
        item_end = item_start + item_len
        item_seqs.append(sequence[item_start:item_end])
        item_start = item_end + 1  # skip delimiter
    return sequence[:prefix_len], item_seqs


def _evaluate_multi_item(
    llm: LLM,
    prompt_tokens_and_prompt_components_lens: list[tuple[list[int], list[int]]],
    *,
    copy_logits_to_host: bool = True,
    padded_num_tokens: Optional[int] = None,
) -> list[list[torch.Tensor]]:
    input_len = sum(
        len(prompt_tokens) for prompt_tokens, _ in prompt_tokens_and_prompt_components_lens
    )
    prompt_inputs: list[PromptInputs] = [
        {
            "prompt_token_ids": prompt_tokens,
            "multi_item_part_lens": prompt_components_lens,
        }
        for prompt_tokens, prompt_components_lens in prompt_tokens_and_prompt_components_lens
    ]
    if padded_num_tokens is not None:
        prompt_inputs.append(
            {
                "prompt_token_ids": [0] * (padded_num_tokens - input_len),
                "multi_item_part_lens": [padded_num_tokens - input_len - 3, 1],
            }
        )
    with assert_no_cuda_sync(sync_timeout_s=5) if not copy_logits_to_host else nullcontext():
        results = cast(
            list,
            llm.encode(
                prompt_inputs,
                batch_indexed_model_output=False,
                copy_logits_to_host=copy_logits_to_host,
            ),
        )
    req_logits: list[list[torch.Tensor]] = []
    for req_result, (_, prompt_components_lens) in zip(
        results, prompt_tokens_and_prompt_components_lens
    ):
        logits = req_result.logits
        assert (logits.device == torch.device("cpu")) == copy_logits_to_host
        if not copy_logits_to_host:
            logits = logits.cpu()
        prompt_logits, item_logits_list = _split_multi_item_sequence(logits, prompt_components_lens)
        req_logits.append(
            [torch.cat((prompt_logits, item_logits), dim=0) for item_logits in item_logits_list]
        )
    return req_logits


def _evaluate_single_item(
    llm: LLM,
    prompt_tokens_and_prompt_components_lens: list[tuple[list[int], list[int]]],
    *,
    padded_num_tokens: int,
) -> list[list[torch.Tensor]]:
    single_iten_prompt_tokens_list = []
    for prompt_tokens, prompt_components_lens in prompt_tokens_and_prompt_components_lens:
        prefix_tokens, item_tokens_list = _split_multi_item_sequence(
            prompt_tokens, prompt_components_lens
        )
        single_iten_prompt_tokens_list += [
            prefix_tokens + item_tokens for item_tokens in item_tokens_list
        ]
    assert padded_num_tokens == sum(
        len(single_iten_prompt_tokens)
        for single_iten_prompt_tokens in single_iten_prompt_tokens_list
    )
    results = cast(
        list[EncoderOutput],
        llm.encode(
            [
                {
                    "prompt_token_ids": single_iten_prompt_tokens,
                }
                for single_iten_prompt_tokens in single_iten_prompt_tokens_list
            ],
            batch_indexed_model_output=False,
        ),
    )
    req_logits: list[list[torch.Tensor]] = []
    for _, prompt_components_lens in prompt_tokens_and_prompt_components_lens:
        req_logits.append(
            [
                item_result.logits.clone()
                for item_result in results[: len(prompt_components_lens) - 1]
            ]
        )
        results = results[len(prompt_components_lens) - 1 :]
    return req_logits


@pytest.fixture(scope="module")
def force_fa2() -> Generator[None, None, None]:
    # Multi-item scoring uses FA2. This forces regular ragged prefill to also use FA2
    # attention, which makes results numerically more comparable.
    with pytest.MonkeyPatch.context() as patcher:
        assert not trtllm_flashinfer._FORCE_RAGGED_FA2
        patcher.setattr(trtllm_flashinfer, "_FORCE_RAGGED_FA2", True)
        yield


_QWEN3_PATH = get_model_path("Qwen3/Qwen3-0.6B")


@pytest.fixture(scope="module")
def llm() -> Generator[LLM, None, None]:
    llm = LLM(model=_QWEN3_PATH, encode_only=True, attn_backend="FLASHINFER")
    with llm:
        yield llm


@pytest.fixture
def attention_only_qwen3(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    # Compute only self-attention, for only one layer. This is done to
    # limit error propagation.
    #
    # NB: Code adapted from tensorrt_llm/_torch/models/modeling_qwen3.py
    def _forward_attn_only(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        mrope_config: Optional[dict] = None,
        # args for deepstack
        deepstack_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self.layers[0].self_attn(
            position_ids=position_ids,
            hidden_states=inputs_embeds,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.layers[0].disable_attn_allreduce
            ),
            mrope_config=mrope_config,
            **kwargs,
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(Qwen3Model, "forward", _forward_attn_only)
        yield


@pytest.fixture
def multi_item_prompts(batch_size: int) -> list[tuple[list[int], list[int]]]:
    qa_strs = [
        ("the important questions", "something", 2),
        ("another, not so important question", "something less important but more verbose", 3),
    ]
    _multi_item_prompts = [
        _build_prompt(question_str, answer_str, max_num_items=num_items)
        for question_str, answer_str, num_items in qa_strs[:batch_size]
    ]

    # Ensure that padding of 'token_pos_in_items' will be necessary and there is variability
    # in prefix length and total prompt length.
    if batch_size > 1:
        assert len(set(len(prompt_tokens) for prompt_tokens, _ in _multi_item_prompts)) > 1
        assert (
            len(set(prompt_components_lens[0] for _, prompt_components_lens in _multi_item_prompts))
            > 1
        )
        assert (
            len(
                set(
                    sum(prompt_components_lens[1:])
                    for _, prompt_components_lens in _multi_item_prompts
                )
            )
            > 1
        )
        assert (
            len(
                set(
                    len(prompt_components_lens[1:])
                    for _, prompt_components_lens in _multi_item_prompts
                )
            )
            > 1
        )

    return _multi_item_prompts


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("copy_logits_to_host", [True, False])
def test_multi_item_and_single_item_agreement(
    llm: LLM,
    batch_size: int,
    attention_only_qwen3: None,
    force_fa2: None,
    multi_item_prompts: list[tuple[list[int], list[int]]],
    copy_logits_to_host: bool,
):
    # Need to pad batch size for numerical agreement in linear layers
    batch_num_tokens = sum(
        (
            sum(prompt_components_lens[1:])
            + len(prompt_components_lens[1:]) * prompt_components_lens[0]
        )
        for _, prompt_components_lens in multi_item_prompts
    )

    single_item_logits_list = _evaluate_single_item(
        llm, multi_item_prompts, padded_num_tokens=batch_num_tokens
    )
    multi_item_logits_list = _evaluate_multi_item(
        llm,
        multi_item_prompts,
        copy_logits_to_host=copy_logits_to_host,
        padded_num_tokens=batch_num_tokens,
    )

    for req_idx, (_, prompt_components_lens) in enumerate(multi_item_prompts):
        for item_idx in range(len(prompt_components_lens) - 1):
            # NB: Even with .plan(fixed_split_size=4096), rtol needs to be rather relaxed.
            torch.testing.assert_close(
                multi_item_logits_list[req_idx][item_idx],
                single_item_logits_list[req_idx][item_idx],
                rtol=1e-2,
                atol=1e-3,
            )


@pytest.mark.parametrize("batch_size", [1])
def test_invalid_multi_item_part_lens(
    llm: LLM,
    batch_size: int,
    multi_item_prompts: list[tuple[list[int], list[int]]],
):
    # construct input with invalid length
    prompt_tokens, prompt_components_lens = multi_item_prompts[0]
    prompt_components_lens = prompt_components_lens.copy()
    prompt_components_lens[-1] += 1

    with pytest.raises(
        ValueError, match='.*"multi_item_part_lens" inconsistent with prompt length.*'
    ):
        _evaluate_multi_item(llm, [(prompt_tokens, prompt_components_lens)])


@pytest.mark.parametrize("batch_size", [1])
def test_mixed_batch_caught(
    llm: LLM,
    batch_size: int,
    multi_item_prompts: list[tuple[list[int], list[int]]],
):
    prompt_inputs: list[PromptInputs] = [
        {
            "prompt_token_ids": prompt_tokens,
            "multi_item_part_lens": prompt_components_lens,
        }
        for prompt_tokens, prompt_components_lens in multi_item_prompts
    ]
    prompt_inputs.append(
        {
            "prompt_token_ids": [0] * 128,
        }
    )
    with pytest.raises(
        ValueError,
        match='.*"multi_item_part_lens" must either be provided for all prompts or for none.*',
    ):
        llm.encode(prompt_inputs, batch_indexed_model_output=False)


@pytest.mark.parametrize("batch_size", [1])
def test_other_llmapi_method_guards(
    batch_size: int,
    multi_item_prompts: list[tuple[list[int], list[int]]],
):
    prompt_inputs: list[PromptInputs] = [
        {
            "prompt_token_ids": prompt_tokens,
            "multi_item_part_lens": prompt_components_lens,
        }
        for prompt_tokens, prompt_components_lens in multi_item_prompts
    ]
    llm = LLM(model=_QWEN3_PATH)
    with llm:
        with pytest.raises(
            ValueError,
            match=".*Multi-item scoring is only supported via LLM.encode.*",
        ):
            llm.generate(prompt_inputs)

        with pytest.raises(
            ValueError,
            match=".*Multi-item scoring is only supported via LLM.encode.*",
        ):
            llm.generate_async(prompt_inputs[0])


@pytest.mark.parametrize("batch_size, attention_backend", [(1, "VANILLA"), (1, "TRTLLM")])
def test_unsupported_attention_backend(
    batch_size: int,
    attention_backend: str,
    multi_item_prompts: list[tuple[list[int], list[int]]],
):
    prompt_inputs: list[PromptInputs] = [
        {
            "prompt_token_ids": prompt_tokens,
            "multi_item_part_lens": prompt_components_lens,
        }
        for prompt_tokens, prompt_components_lens in multi_item_prompts
    ]
    llm = LLM(model=_QWEN3_PATH, encode_only=True, attn_backend=attention_backend)
    with (
        llm,
        pytest.raises(
            ValueError,
            match=".*Attention does not support multi-item scoring.*",
        ),
    ):
        llm.encode(prompt_inputs, batch_indexed_model_output=False)
