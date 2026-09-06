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

import torch

from tensorrt_llm._torch.attention.backends import interface
from tensorrt_llm._torch.attention.backends.interface import log_attention_failure_context


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
