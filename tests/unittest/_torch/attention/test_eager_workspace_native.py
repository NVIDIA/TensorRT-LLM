# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Requires bindings built from this checkout; no workspace sizing mocks."""

from unittest.mock import patch

import pytest
import torch
from backend_case import BackendCase, generate_inputs, run_backend

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal import thop


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_native_workspace_report_is_requirement_not_capacity() -> None:
    if get_sm_version() != 90:
        pytest.skip("The regression explicitly exercises Hopper fallback FMHA")
    case = BackendCase(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        seq_lens=[64, 64],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype="bfloat16",
        page_size=32,
    )
    inputs = generate_inputs(case, seed=1234)
    report = torch.zeros(1, dtype=torch.int64, device="cpu")
    capacity = 64 * 1024**2
    native_attention = thop.attention
    calls = 0

    def attention_with_report(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        workspace = kwargs["workspace_"]
        assert isinstance(workspace, torch.Tensor)
        workspace.resize_(capacity)
        kwargs["workspace_required_bytes"] = report
        native_attention(**kwargs)
        required = report.item()
        assert 0 < required < capacity
        # A second layer must preserve the maximum accumulated so far.
        report.fill_(capacity)
        native_attention(**kwargs)
        assert report.item() == capacity
        assert workspace.untyped_storage().nbytes() == capacity

    with torch.inference_mode():
        golden = run_backend(case, "VANILLA", inputs, kv_dtype=case.kv_torch_dtype, kv_layout="NHD")
        with patch.object(thop, "attention", side_effect=attention_with_report):
            output = run_backend(
                case, "TRTLLM", inputs, kv_dtype=case.kv_torch_dtype, kv_layout="HND"
            )
        torch.testing.assert_close(output, golden, atol=3e-2, rtol=3e-3)
    assert calls == 1
