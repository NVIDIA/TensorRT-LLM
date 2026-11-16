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
"""LMHead(output_dtype=torch.float32), which backs LlmArgs.lm_head_output_dtype="float32".

The weights and inputs stay bf16; the GEMM writes its float32 accumulator instead of
rounding the logits to bf16. The float64 reference uses the same bf16 values, so the
float32 path should match it up to accumulation order, while the default path is off
by the bf16 rounding of each logit.
"""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from transformers import LlamaConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.embedding import Embedding, LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_module
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader, validate_lm_head_output_dtype
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

VOCAB_SIZE = 32000
HIDDEN_SIZE = 1024
DTYPE = torch.bfloat16


def _make_inputs(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate BF16 inputs that expose final-output rounding."""
    torch.manual_seed(0)
    hidden = torch.randn(num_tokens, HIDDEN_SIZE, dtype=DTYPE)
    # Logits with a std of about 4, where bf16 rounding is on the order of 1e-2.
    weight = (torch.randn(VOCAB_SIZE, HIDDEN_SIZE) * 4 / HIDDEN_SIZE**0.5).to(DTYPE)
    return hidden.cuda(), weight


def _make_lm_head(weight: torch.Tensor, output_dtype: torch.dtype | None, **kwargs) -> LMHead:
    """Load the same unquantized weights into either output-dtype path."""
    lm_head = LMHead(VOCAB_SIZE, HIDDEN_SIZE, dtype=DTYPE, output_dtype=output_dtype, **kwargs)
    lm_head.load_weights([{"weight": weight}])
    return lm_head.cuda()


def _reference_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute an independent FP64 reference from the same BF16 inputs."""
    return hidden.double() @ weight.cuda().double().t()


@pytest.mark.parametrize(
    "tp_mode", [None, TensorParallelMode.COLUMN], ids=["no_tp", "column_gather"]
)
@pytest.mark.parametrize("num_tokens", [0, 1, 7, 128])
def test_float32_output_matches_reference(
    num_tokens: int, tp_mode: TensorParallelMode | None
) -> None:
    """Avoid BF16 output rounding for empty, single-token, and batched inputs."""
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


def test_float32_output_one_dim_input() -> None:
    """Accept the one-dimensional hidden state used without attention metadata."""
    # LogitsProcessor passes hidden_states[-1] (1-D) when there is no attention metadata.
    hidden, weight = _make_inputs(1)
    logits = _make_lm_head(weight, torch.float32)(hidden[0])
    assert logits.shape == (VOCAB_SIZE,)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(
        logits.double(), _reference_logits(hidden, weight)[0], rtol=1e-4, atol=1e-3
    )


@pytest.mark.parametrize("output_dtype", [None, DTYPE, torch.float32], ids=["auto", "bf16", "fp32"])
@pytest.mark.parametrize("num_tokens", [0, 1, 16])
def test_output_dtype_spec_decoding_head_slice(
    num_tokens: int, output_dtype: torch.dtype | None
) -> None:
    """Preserve default and FP32 output precision for a sliced speculative head."""
    # With lm_head TP in attention DP, the spec-decoding head slices the vocab at forward time.
    hidden, weight = _make_inputs(num_tokens)
    mapping = Mapping(enable_attention_dp=True, enable_lm_head_tp_in_adp=True)
    lm_head = _make_lm_head(
        weight,
        output_dtype,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        reduce_output=False,
    )
    lm_head_tp = Mapping(world_size=2, tp_size=2, rank=1)
    logits = lm_head(hidden, mapping_lm_head_tp=lm_head_tp, is_spec_decoding_head=True)
    reference = _reference_logits(hidden, weight)[:, VOCAB_SIZE // 2 :]
    assert logits.dtype == (output_dtype or DTYPE)
    assert logits.shape == reference.shape
    if output_dtype == torch.float32:
        torch.testing.assert_close(logits.double(), reference, rtol=1e-4, atol=1e-3)
    else:
        torch.testing.assert_close(logits, reference.to(DTYPE))


def test_output_dtype_rejects_quantized_head() -> None:
    """Reject the FP32-output option for unsupported quantized weights."""
    with pytest.raises(NotImplementedError, match="unquantized"):
        LMHead(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            dtype=DTYPE,
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
            output_dtype=torch.float32,
        )


def test_output_dtype_rejects_row_parallel() -> None:
    """Reject ROW parallelism rather than silently changing reduction precision."""
    with pytest.raises(NotImplementedError, match="ROW"):
        LMHead(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            dtype=DTYPE,
            tensor_parallel_mode=TensorParallelMode.ROW,
            output_dtype=torch.float32,
        )


def test_llm_args_lm_head_output_dtype(tmp_path: Path) -> None:
    """Validate the public option's default, FP32 value, and unsupported values."""
    assert TorchLlmArgs(model=str(tmp_path)).lm_head_output_dtype == "auto"
    assert (
        TorchLlmArgs(model=str(tmp_path), lm_head_output_dtype="float32").lm_head_output_dtype
        == "float32"
    )
    with pytest.raises(ValueError):
        TorchLlmArgs(model=str(tmp_path), lm_head_output_dtype="float16")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@torch.inference_mode()
def test_float32_output_strided_batched_input(dtype: torch.dtype) -> None:
    """Preserve leading dimensions and FP32 precision for strided hidden states."""
    generator = torch.Generator(device="cuda").manual_seed(42)
    hidden = torch.randn(2, 3, 128, device="cuda", dtype=dtype, generator=generator)[..., ::2]
    head = LMHead(127, 64, dtype=dtype, output_dtype=torch.float32).cuda()
    head.weight.copy_(torch.randn(127, 64, device="cuda", dtype=dtype, generator=generator))
    assert not hidden.is_contiguous()

    logits = head(hidden)
    reference = hidden.double() @ head.weight.double().t()
    assert logits.shape == (2, 3, 127)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits.double(), reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@torch.inference_mode()
def test_float32_output_cuda_graph(dtype: torch.dtype) -> None:
    """Replay the native FP32-output GEMM with updated hidden states."""
    generator = torch.Generator(device="cuda").manual_seed(42)
    hidden = torch.randn(7, 64, device="cuda", dtype=dtype, generator=generator)
    head = LMHead(127, 64, dtype=dtype, output_dtype=torch.float32).cuda()
    head.weight.copy_(torch.randn(127, 64, device="cuda", dtype=dtype, generator=generator))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            head(hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = head(hidden)

    for _ in range(2):
        hidden.copy_(torch.randn(hidden.shape, device="cuda", dtype=dtype, generator=generator))
        graph.replay()
        reference = hidden.double() @ head.weight.double().t()
        assert logits.dtype == torch.float32
        torch.testing.assert_close(logits.double(), reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("output_dtype", [None, torch.bfloat16, torch.float32])
def test_built_model_lm_head_output_dtype(nested: bool, output_dtype: torch.dtype | None) -> None:
    """Validate real LM heads, including nested ones, without mistaking embeddings for heads."""
    model = torch.nn.Module()
    model.embed_tokens = Embedding(16, 8, dtype=DTYPE)
    owner = model
    if nested:
        model.language_model = torch.nn.Module()
        owner = model.language_model
    owner.lm_head = LMHead(16, 8, dtype=DTYPE, output_dtype=output_dtype)

    validate_lm_head_output_dtype(model, "auto")
    if output_dtype == torch.float32:
        validate_lm_head_output_dtype(model, "float32")
    else:
        with pytest.raises(ValueError, match=r"lm_head_output_dtype=.*lm_head.output_dtype"):
            validate_lm_head_output_dtype(model, "float32")


def test_built_model_lm_head_output_dtype_checks_all_heads() -> None:
    """Reject an ignored option in a nested head even if another head honours it."""
    model = torch.nn.Module()
    model.lm_head = LMHead(16, 8, dtype=DTYPE, output_dtype=torch.float32)
    model.draft = torch.nn.Module()
    model.draft.lm_head = LMHead(16, 8, dtype=DTYPE)
    with pytest.raises(ValueError, match=r"draft.lm_head.output_dtype"):
        validate_lm_head_output_dtype(model, "float32")


def test_built_model_lm_head_output_dtype_requires_a_head() -> None:
    """Reject an explicit output dtype when only input embeddings were constructed."""
    model = torch.nn.Module()
    model.embed_tokens = Embedding(16, 8, dtype=DTYPE)
    validate_lm_head_output_dtype(model, "auto")
    with pytest.raises(ValueError, match="no LMHead was constructed"):
        validate_lm_head_output_dtype(model, "float32")


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("fallback", [False, True], ids=["meta_init", "regular_init"])
def test_model_loader_rejects_ignored_lm_head_output_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, fallback: bool
) -> None:
    """Reject unsupported constructed heads before any checkpoint weights are loaded."""
    model = torch.nn.Module()
    owner = model
    if nested:
        model.language_model = torch.nn.Module()
        owner = model.language_model
    owner.lm_head = LMHead(16, 8, dtype=DTYPE)
    config = ModelConfig(
        pretrained_config=LlamaConfig(architectures=["LlamaForCausalLM"], dtype=DTYPE),
        lm_head_output_dtype=torch.float32,
    )
    loader = ModelLoader(
        llm_args=TorchLlmArgs(model=str(tmp_path), lm_head_output_dtype="float32"),
        mapping=Mapping(),
        spec_config=None,
        sparse_attention_config=None,
        max_num_tokens=8,
        max_seq_len=8,
    )
    monkeypatch.setattr(loader, "_load_and_validate_config", lambda *_args: config)
    monkeypatch.setattr(model_loader_module, "MetaInitMode", nullcontext)
    monkeypatch.setattr(
        model_loader_module, "maybe_create_moe_load_balancer", lambda *_args: nullcontext()
    )
    monkeypatch.setattr(model_loader_module, "_checkpoint_startup_metadata", lambda *_args: {})
    factory = MagicMock(
        side_effect=[RuntimeError("meta initialization unavailable"), model]
        if fallback
        else [model]
    )
    monkeypatch.setattr(model_loader_module.AutoModelForCausalLM, "from_config", factory)
    checkpoint_loader = MagicMock()

    with pytest.raises(ValueError, match=r"lm_head_output_dtype=.*lm_head.output_dtype"):
        loader.load(str(tmp_path), checkpoint_loader)

    assert factory.call_count == (2 if fallback else 1)
    checkpoint_loader.load_weights.assert_not_called()
