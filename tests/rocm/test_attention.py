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

import pytest
import torch
from transformers import LlamaConfig, PretrainedConfig

from tensorrt_llm.rocm.attention import attention_forward, configure_native_attention
from tensorrt_llm.rocm.ops import attention

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("queries", [1, 3, 5])
def test_attention_interface_padding_cached_decode_and_layout(queries) -> None:
    module = torch.nn.Module().eval()
    module.is_causal = True
    query = torch.randn(2, 4, queries, 16)
    key, value = torch.randn(2, 2, 5, 16), torch.randn(2, 2, 5, 16)
    padding = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 0, 1, 1]])
    actual, weights = attention_forward(module, query, key, value, padding, backend="torch")
    expected = attention(query, key, value, mask=padding[:, None, None, :].bool(), causal=True)
    torch.testing.assert_close(actual, expected.transpose(1, 2))
    assert actual.shape == (2, queries, 4, 16)
    assert actual.is_contiguous() and weights is None


def test_4d_mask_is_authoritative_for_static_cache_geometry() -> None:
    module = torch.nn.Module().eval()
    module.is_causal = True
    query = torch.randn(1, 2, 1, 8)
    key, value = torch.randn(1, 1, 7, 8), torch.randn(1, 1, 7, 8)
    mask = torch.tensor(
        [[[[0.0, 0.0, 0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf")]]]]
    )
    actual, _ = attention_forward(module, query, key, value, mask, backend="torch")
    expected = attention(query, key, value, mask=mask, causal=False)
    torch.testing.assert_close(actual, expected.transpose(1, 2))


def test_sliding_window_without_prebuilt_mask() -> None:
    module = torch.nn.Module().eval()
    module.is_causal = True
    query = torch.randn(1, 2, 3, 8)
    key, value = torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)
    actual, _ = attention_forward(
        module, query, key, value, None, sliding_window=2, backend="torch"
    )
    positions = torch.arange(3) + 2
    mask = (torch.arange(5)[None, :] <= positions[:, None]) & (
        torch.arange(5)[None, :] > positions[:, None] - 2
    )
    expected = attention(query, key, value, mask=mask)
    torch.testing.assert_close(actual, expected.transpose(1, 2))


def test_register_attention_and_reject_unsupported_models() -> None:
    assert configure_native_attention(LlamaConfig()) == "rdna4_hip"
    with pytest.raises(NotImplementedError, match="Llama"):
        configure_native_attention(PretrainedConfig())
    with pytest.raises(ValueError, match="dimensions"):
        configure_native_attention(LlamaConfig(hidden_size=512, num_attention_heads=1))


def test_training_dropout_and_attention_maps_rejected() -> None:
    module = torch.nn.Module()
    tensor = torch.ones(1, 1, 1, 4)
    with pytest.raises(ValueError, match="inference-only"):
        attention_forward(module, tensor, tensor, tensor, None, backend="torch")
    module.eval()
    with pytest.raises(NotImplementedError, match="weights"):
        attention_forward(
            module, tensor, tensor, tensor, None, output_attentions=True, backend="torch"
        )


def test_hf_native_callback_frontend_cache_and_padding_with_cpu_reference() -> None:
    """Qualify mask/interface integration, not native GPU arithmetic."""
    from copy import deepcopy
    from functools import partial

    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from tensorrt_llm.rocm.validation import tiny_model_and_tokenizer

    reference, _ = tiny_model_and_tokenizer()
    candidate = deepcopy(reference)
    candidate.config._attn_implementation = configure_native_attention(candidate.config)
    callback = ALL_ATTENTION_FUNCTIONS["rdna4_hip"]
    AttentionInterface.register("rdna4_hip", partial(attention_forward, backend="torch"))
    try:
        with torch.inference_mode():
            ids = torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]])
            mask = ids.ne(0).long()
            expected = reference(input_ids=ids, attention_mask=mask, use_cache=True)
            actual = candidate(input_ids=ids, attention_mask=mask, use_cache=True)
            torch.testing.assert_close(
                actual.logits[:, -1:], expected.logits[:, -1:], rtol=3e-5, atol=3e-5
            )
            next_ids = torch.tensor([[10], [11]])
            mask = torch.cat((mask, torch.ones(2, 1, dtype=torch.long)), dim=1)
            expected = reference(
                input_ids=next_ids,
                attention_mask=mask,
                past_key_values=expected.past_key_values,
                use_cache=True,
            )
            actual = candidate(
                input_ids=next_ids,
                attention_mask=mask,
                past_key_values=actual.past_key_values,
                use_cache=True,
            )
            torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-5, atol=3e-5)
    finally:
        AttentionInterface.register("rdna4_hip", callback)
