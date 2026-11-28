# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.trtllm_context_attention import (
    TrtllmContextWorkspace,
    trtllm_context_attention,
)
from tensorrt_llm.quantization.mode import QuantMode


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="TRTLLM context attention requires SM100 or SM103",
)
@pytest.mark.parametrize("cell", trtllm_context_attention.CELLS, ids=lambda c: c.why)
def test_trtllm_context_attention(cell, monkeypatch):
    spec = cell.spec
    if "skip_threshold" in spec:
        _check_skipped_tiles(spec, monkeypatch)
        return
    sage_qk_int8 = spec.get("sage_qk_int8", False)
    if sage_qk_int8 and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("The Blackwell INT8 Q/K context kernel requires SM100.")
    batch, length = spec["batch"], spec["length"]
    heads, kv_heads, dim = spec["heads"], spec["kv_heads"], 128
    generator = torch.Generator(device="cuda").manual_seed(42)

    def tensor(h):
        return torch.randn(
            batch, length, h, dim, device="cuda", dtype=torch.bfloat16, generator=generator
        )

    q, k, v = tensor(heads), tensor(kv_heads), tensor(kv_heads)
    kwargs = dict(num_heads=heads)
    if spec.get("sage"):
        k += 32 * torch.randn(
            1, 1, kv_heads, dim, dtype=k.dtype, device=k.device, generator=generator
        )
        kwargs.update(
            sage_block_sizes=(1, 16, 1),
            sage_smooth_k=True,
            sage_qk_int8=sage_qk_int8,
            quant_mode=int(QuantMode.FP8_QDQ),
            layer_idx=3,
        )
    else:
        q, k, v = torch.cat((q, k, v), dim=2), None, None
    workspace = TrtllmContextWorkspace(batch, length, q.device)
    counters = torch.zeros(2, device=q.device, dtype=torch.uint32)
    args = (q, k, v, workspace, counters)
    trtllm_context_attention.is_valid(*args, **kwargs)
    expected = trtllm_context_attention.reference(*args, **kwargs)
    actual = trtllm_context_attention(*args, **kwargs)
    if spec.get("sage"):
        relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert relative_l2 < 0.08
    else:
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def _check_skipped_tiles(spec, monkeypatch):
    length, heads = spec["length"], spec["heads"]
    q = torch.zeros(1, length, heads, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    q[..., 0] = 128**0.5
    k[:, 128:256, :, 0], k[:, 256:384, :, 0], k[:, 384:, :, 0] = -4, -5, -9
    v[:, 128:256], v[:, 256:384], v[:, 384:] = 1, 8, 64
    qkv = torch.cat((q, k, v), dim=2)
    workspace = TrtllmContextWorkspace(1, length, q.device)
    initial = [17, 5]
    counters = torch.tensor(initial, device=q.device, dtype=torch.uint32)
    # exp(-4) exceeds 8/512; exp(-5) and exp(-9) fall below it.
    kept = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k[:, :256].transpose(1, 2).float(),
            v[:, :256].transpose(1, 2).float(),
        )
        .transpose(1, 2)
        .to(q.dtype)
    )

    def run():
        return trtllm_context_attention(
            qkv,
            None,
            None,
            workspace,
            counters,
            num_heads=heads,
            skip_threshold=spec["skip_threshold"],
        )

    monkeypatch.setenv("TRTLLM_PRINT_SKIP_SOFTMAX_STAT", "1")
    try:
        actual = run()
    except RuntimeError as error:
        if "please run build_wheel.py with -DSKIP_SOFTMAX_STAT" not in str(error):
            raise
        monkeypatch.delenv("TRTLLM_PRINT_SKIP_SOFTMAX_STAT")
        actual = run()
    torch.testing.assert_close(actual, kept, rtol=0, atol=1e-3)
    # The SM100-family TRTLLM-Gen context runner accepts the threshold and leaves counters untouched.
    assert counters.tolist() == initial


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("q_heads", "kv_length", "blocks", "error"),
    [
        pytest.param(8, None, (1, 16, 1), "separate Q, K and V", id="packed-sage"),
        pytest.param(5, None, (0, 0, 0), "same number of K and V heads", id="odd-packed-width"),
        pytest.param(2, 2, (1, 16, 1), "share shape", id="short-kv-storage"),
    ],
)
def test_invalid_qkv_rejected_before_launch(q_heads, kv_length, blocks, error):
    q = torch.empty((1, 4, q_heads, 128), dtype=torch.bfloat16)
    kv = q.new_empty((1, kv_length, 2, 128)) if kv_length is not None else None
    workspace = TrtllmContextWorkspace(1, 4, q.device)
    counters = torch.zeros(2, dtype=torch.uint32)
    with pytest.raises(ValueError, match=error):
        trtllm_context_attention(
            q, kv, kv, workspace, counters, num_heads=2, sage_block_sizes=blocks
        )
