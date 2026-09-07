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
"""CPU-only tests for the attention-backend kernel-failure diagnostics.

The backends wrap their kernel call so an opaque CUDA/cuBLAS RuntimeError is
preceded by a line naming the batch, the sequence lengths and the attention
window. These tests pin the two properties that matter: the dump never raises
(so it cannot mask or replace the original error), and it reports the fields a
reader needs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends import interface
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
    log_attention_failure_context,
)


class _CapturingLogger:
    """Captures error lines. tensorrt_llm's logger sets propagate=False, so the
    stdlib `caplog` fixture never sees them; swap the module's logger instead."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def error(self, *msg) -> None:
        self.messages.append(" ".join(str(m) for m in msg))


def _fake_metadata(**overrides):
    metadata = SimpleNamespace(
        num_contexts=2,
        num_generations=1,
        num_tokens=2050,
        max_seq_len=4096,
        seq_lens=torch.tensor([2047, 1, 1], dtype=torch.int32),
        seq_lens_kv=torch.tensor([2047, 2048, 5], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([2047, 2048, 5], dtype=torch.int32),
        request_ids=[11, 12, 13],
        kv_cache_block_offsets=torch.zeros((1, 3, 2, 8), dtype=torch.int32),
        workspace=torch.zeros(1024, dtype=torch.int8),
        kv_cache_manager=SimpleNamespace(
            max_attention_window_vec=[2048, None], tokens_per_block=64
        ),
    )
    for key, value in overrides.items():
        setattr(metadata, key, value)
    return metadata


def test_failure_context_reports_shapes_and_window(monkeypatch) -> None:
    capturing_logger = _CapturingLogger()
    monkeypatch.setattr(interface, "logger", capturing_logger)

    log_attention_failure_context(
        "TrtllmAttention", 3, _fake_metadata(), 2048, RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")
    )

    assert len(capturing_logger.messages) == 1
    message = capturing_logger.messages[0]
    assert "backend=TrtllmAttention" in message
    assert "layer_idx=3" in message
    assert "attention_window_size=2048" in message
    assert "num_contexts=2" in message
    assert "num_generations=1" in message
    assert "kv_lens(n=3, min=5, max=2048)" in message
    assert "kv_cache_block_offsets_shape=(1, 3, 2, 8)" in message
    assert "workspace_numel=1024" in message
    assert "max_attention_window_vec=[2048, None]" in message
    assert "RuntimeError" in message


def test_failure_context_tolerates_missing_fields() -> None:
    """A metadata object missing every optional field must not raise."""
    log_attention_failure_context(
        "FlashInferAttention", None, SimpleNamespace(), None, RuntimeError("illegal memory access")
    )


def test_failure_context_tolerates_unreadable_tensors() -> None:
    class _Unreadable:
        def detach(self):
            raise RuntimeError("device is in a bad state")

    log_attention_failure_context(
        "FlashInferAttention",
        0,
        _fake_metadata(seq_lens=_Unreadable()),
        2048,
        RuntimeError("illegal memory access"),
    )


def test_reraise_preserves_the_original_exception() -> None:
    """Mirrors the backends' wrapper: log, then `raise` the same object."""
    original = RuntimeError("CUDA error: an illegal memory access was encountered")

    try:
        try:
            raise original
        except RuntimeError as exc:
            log_attention_failure_context("TrtllmAttention", 0, _fake_metadata(), 2048, exc)
            raise
    except RuntimeError as caught:
        assert caught is original
    else:
        raise AssertionError("the exception must propagate")


def _spy_on_failure_context(monkeypatch, module):
    """Replace the backend module's `log_attention_failure_context` with a spy
    that records its positional args and still calls the real (resilient)
    implementation, so a test can prove the wrapper reached it."""
    calls: list[tuple] = []
    real = module.log_attention_failure_context

    def spy(*args) -> None:
        calls.append(args)
        real(*args)

    monkeypatch.setattr(module, "log_attention_failure_context", spy)
    return calls


def test_trtllm_forward_logs_and_reraises_kernel_failure(monkeypatch) -> None:
    """The real `TrtllmAttention.forward` wrapper must log the failure context
    and re-raise the *same* RuntimeError instance when the FMHA kernel throws.

    CPU-only: the FMHA selection/launch is mocked, so no GPU kernel runs."""
    from tensorrt_llm._torch.attention.backends import trtllm as trtllm_backend
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

    calls = _spy_on_failure_context(monkeypatch, trtllm_backend)
    # `prepare_sparse_runtime_params` touches sparse scheduler state we don't
    # build here; the wrapper under test is downstream of it.
    monkeypatch.setattr(trtllm_backend, "prepare_sparse_runtime_params", lambda *a, **k: None)

    num_heads, num_kv_heads, head_dim = 2, 2, 8
    qkv_hidden = (num_heads + 2 * num_kv_heads) * head_dim
    num_tokens, batch = 3, 3

    original = RuntimeError("CUDA error: an illegal memory access was encountered")

    class _ThrowingFmha:
        def forward(self, *args, **kwargs):
            raise original

    backend = TrtllmAttention.__new__(TrtllmAttention)
    backend.sparse_params = None
    backend.is_mla_enable = False
    backend.num_heads = num_heads
    backend.num_kv_heads = num_kv_heads
    backend.head_dim = head_dim
    backend.layer_idx = 5
    backend.local_layer_idx = 0
    backend.kv_scale_orig_quant = None
    backend.kv_scale_quant_orig = None
    backend.print_skip_softmax_stat = False
    backend._fmha_manager = SimpleNamespace(select=lambda *a, **k: _ThrowingFmha())
    # Skip rope-table growth; it dereferences rope_params we do not construct.
    backend._ensure_rope_table_size = lambda *a, **k: None

    seq_lens = torch.tensor([2047, 1, 1], dtype=torch.int32)
    metadata = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
    # `is_cross` is `seq_lens is not seq_lens_kv`; share the object to stay self.
    metadata.seq_lens = seq_lens
    metadata.seq_lens_kv = seq_lens
    metadata.kv_lens_runtime = torch.tensor([2047, 2048, 5], dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = torch.tensor([2047, 2048, 5], dtype=torch.int32)
    metadata.prompt_lens_cuda_runtime = torch.zeros(batch, dtype=torch.int32)
    metadata.prompt_lens_cpu_runtime = torch.zeros(batch, dtype=torch.int32)
    metadata.host_request_types_runtime = torch.zeros(batch, dtype=torch.int32)
    metadata.request_ids = [11, 12, 13]
    metadata.num_contexts = 2
    metadata.num_generations = 1
    metadata.num_tokens = num_tokens
    metadata.kv_cache_block_offsets = torch.zeros((1, 3, 2, 8), dtype=torch.int32)
    metadata.workspace = torch.zeros(1024, dtype=torch.int8)
    metadata.cu_q_seqlens = torch.zeros(batch + 1, dtype=torch.int32)
    metadata.cu_kv_seqlens = torch.zeros(batch + 1, dtype=torch.int32)
    metadata.enable_flash_mla = False
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = None
    metadata.spec_decoding_bl_tree_mask = None
    metadata.max_context_q_len_override = None
    metadata.kv_cache_manager = SimpleNamespace(
        max_attention_window_vec=[2048, None],
        tokens_per_block=64,
        max_seq_len=4096,
    )

    q = torch.zeros(num_tokens, qkv_hidden)
    forward_args = AttentionForwardArgs(
        output=torch.zeros(num_tokens, num_heads * head_dim),
        attention_mask=PredefinedAttentionMask.CAUSAL,
    )

    with pytest.raises(RuntimeError) as excinfo:
        TrtllmAttention.forward(backend, q, None, None, metadata, forward_args)

    # The wrapper must not wrap or replace the kernel's exception.
    assert excinfo.value is original
    assert len(calls) == 1
    backend_name, layer_idx, logged_metadata, window, exc = calls[0]
    assert backend_name == "TrtllmAttention"
    assert layer_idx == 5
    # Window defaults from the KV cache manager's per-layer vector (exclusive).
    assert window == 2048
    assert logged_metadata is metadata
    assert exc is original


def test_flashinfer_forward_logs_and_reraises_kernel_failure(monkeypatch) -> None:
    """The real `FlashInferAttention.forward` wrapper must log the failure
    context and re-raise the *same* RuntimeError when the kernel call throws.

    CPU-only: `forward_impl` (the kernel launch) is mocked to raise."""
    from tensorrt_llm._torch.attention.backends import flashinfer as flashinfer_backend
    from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention

    calls = _spy_on_failure_context(monkeypatch, flashinfer_backend)

    original = RuntimeError("CUDA error: an illegal memory access was encountered")

    backend = FlashInferAttention.__new__(FlashInferAttention)
    backend.is_mla_enable = False
    backend.layer_idx = 7

    def _throwing_forward_impl(**kwargs):
        raise original

    backend.forward_impl = _throwing_forward_impl

    metadata = _fake_metadata()
    q = torch.zeros(4, 16)
    forward_args = AttentionForwardArgs(
        output=torch.zeros(4, 16),
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=2048,
    )

    with pytest.raises(RuntimeError) as excinfo:
        FlashInferAttention.forward(backend, q, None, None, metadata, forward_args)

    assert excinfo.value is original
    assert len(calls) == 1
    backend_name, layer_idx, logged_metadata, window, exc = calls[0]
    assert backend_name == "FlashInferAttention"
    assert layer_idx == 7
    # The wrapper logs the TRTLLM-convention (exclusive) window, not the
    # decremented FlashInfer one.
    assert window == 2048
    assert logged_metadata is metadata
    assert exc is original
