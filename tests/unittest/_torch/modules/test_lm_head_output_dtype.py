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
"""LMHead(output_dtype=torch.float32), which backs LlmArgs.lm_head_dtype="float32".

The weights and inputs stay bf16; the GEMM writes its float32 accumulator instead of
rounding the logits to bf16. The float64 reference uses the same bf16 values, so the
float32 path should match it up to accumulation order, while the default path is off
by the bf16 rounding of each logit.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

VOCAB_SIZE = 32000
HIDDEN_SIZE = 1024
DTYPE = torch.bfloat16


def _make_inputs(num_tokens: int):
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, dtype=DTYPE)
    # Logits with a std of about 4, where bf16 rounding is on the order of 1e-2.
    weight = (torch.randn(VOCAB_SIZE, HIDDEN_SIZE) * 4 / HIDDEN_SIZE**0.5).to(DTYPE)
    return hidden.cuda(), weight


def _make_lm_head(weight: torch.Tensor, output_dtype, **kwargs) -> LMHead:
    lm_head = LMHead(VOCAB_SIZE, HIDDEN_SIZE, dtype=DTYPE, output_dtype=output_dtype, **kwargs)
    lm_head.load_weights([{"weight": weight}])
    return lm_head.cuda()


def _reference_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden.double() @ weight.cuda().double().t()


@pytest.mark.parametrize(
    "tp_mode", [None, TensorParallelMode.COLUMN], ids=["no_tp", "column_gather"]
)
@pytest.mark.parametrize("num_tokens", [0, 1, 7, 128])
def test_float32_output_matches_reference(num_tokens: int, tp_mode):
    hidden, weight = _make_inputs(num_tokens)
    kwargs = dict(
        mapping=Mapping(), tensor_parallel_mode=tp_mode, gather_output=True, reduce_output=False
    )
    logits_bf16 = _make_lm_head(weight, None, **kwargs)(hidden)
    logits_fp32 = _make_lm_head(weight, torch.float32, **kwargs)(hidden)
    reference = _reference_logits(hidden, weight)

    assert logits_bf16.dtype == DTYPE
    assert logits_fp32.dtype == torch.float32
    assert logits_fp32.shape == logits_bf16.shape == reference.shape
    if num_tokens == 0:
        return

    torch.testing.assert_close(logits_fp32.double(), reference, rtol=1e-4, atol=1e-3)
    # The logits are no longer bf16 values ...
    assert not torch.equal(logits_fp32, logits_fp32.to(DTYPE).float())
    # ... and the error left is far below the bf16 rounding error, for logits and logprobs.
    err_bf16 = (logits_bf16.double() - reference).abs().max()
    err_fp32 = (logits_fp32.double() - reference).abs().max()
    assert err_fp32 * 50 < err_bf16
    logprobs_ref = torch.log_softmax(reference, dim=-1)
    lp_err_bf16 = (torch.log_softmax(logits_bf16.double(), dim=-1) - logprobs_ref).abs().max()
    lp_err_fp32 = (torch.log_softmax(logits_fp32.double(), dim=-1) - logprobs_ref).abs().max()
    assert lp_err_fp32 * 10 < lp_err_bf16


def test_float32_output_one_dim_input():
    # LogitsProcessor passes hidden_states[-1] (1-D) when there is no attention metadata.
    hidden, weight = _make_inputs(1)
    logits = _make_lm_head(weight, torch.float32)(hidden[0])
    assert logits.shape == (VOCAB_SIZE,)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(
        logits.double(), _reference_logits(hidden, weight)[0], rtol=1e-4, atol=1e-3
    )


def test_float32_output_spec_decoding_head_slice():
    # With lm_head TP in attention DP, the spec-decoding head slices the vocab at forward time.
    hidden, weight = _make_inputs(16)
    mapping = Mapping(enable_attention_dp=True, enable_lm_head_tp_in_adp=True)
    lm_head = _make_lm_head(
        weight,
        torch.float32,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        reduce_output=False,
    )
    lm_head_tp = Mapping(world_size=2, tp_size=2, rank=1)
    logits = lm_head(hidden, mapping_lm_head_tp=lm_head_tp, is_spec_decoding_head=True)
    reference = _reference_logits(hidden, weight)[:, VOCAB_SIZE // 2 :]
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits.double(), reference, rtol=1e-4, atol=1e-3)


def test_output_dtype_rejects_quantized_head():
    with pytest.raises(NotImplementedError, match="unquantized"):
        LMHead(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            dtype=DTYPE,
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
            output_dtype=torch.float32,
        )


def test_output_dtype_rejects_row_parallel():
    with pytest.raises(NotImplementedError, match="ROW"):
        LMHead(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            dtype=DTYPE,
            tensor_parallel_mode=TensorParallelMode.ROW,
            output_dtype=torch.float32,
        )


def test_llm_args_lm_head_dtype(tmp_path):
    assert TorchLlmArgs(model=str(tmp_path)).lm_head_dtype == "auto"
    assert TorchLlmArgs(model=str(tmp_path), lm_head_dtype="float32").lm_head_dtype == "float32"
    with pytest.raises(ValueError):
        TorchLlmArgs(model=str(tmp_path), lm_head_dtype="float16")
