# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention import _flashinfer_runtime
from tensorrt_llm._torch.visual_gen.catalog.attention import flashinfer_prefill as prefill_module
from tensorrt_llm._torch.visual_gen.catalog.attention.flashinfer_prefill import (
    FlashInferPrefillRunner,
    flashinfer_prefill,
)


@pytest.mark.parametrize("cell", flashinfer_prefill.CELLS, ids=lambda cell: cell.why)
def test_prefill_matches_reference(cell):
    spec = cell.spec
    batch, heads, dim = spec["batch"], spec.get("heads", 4), spec.get("dim", 64)
    kv_heads = spec.get("kv_heads", heads)
    recipe = spec.get("recipe")
    if recipe and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100/SM103 block-scaled FlashInfer kernel")
    generator = torch.Generator(device="cuda").manual_seed(0)
    options = dict(device="cuda", dtype=spec["dtype"], generator=generator)
    q = torch.randn((batch, spec["q"], heads, dim), **options) * 0.2
    k = torch.randn((batch, spec["kv"], kv_heads, dim), **options) * 0.2
    v = torch.randn(k.shape, **options)
    mask = None
    if spec.get("padding"):
        mask = torch.ones(batch, spec["kv"], device="cuda", dtype=torch.bool)
        mask[0, -4:] = False
        if batch > 1:
            mask[1, -2:] = False
    runner = FlashInferPrefillRunner(
        num_heads=heads, num_kv_heads=kv_heads, head_dim=dim, plan_cache={}
    )
    kwargs = dict(
        runner=runner,
        key_padding_mask=mask,
        is_causal=spec.get("causal", False),
        qk_dtype=recipe,
        v_dtype="fp8" if recipe else None,
    )
    flashinfer_prefill.is_valid(q, k, v, **kwargs)
    output, lse = flashinfer_prefill(q, k, v, **kwargs)
    expected, expected_lse = flashinfer_prefill.reference(q, k, v, **kwargs)
    if recipe:
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine > 0.98
    else:
        torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, expected_lse, rtol=0, atol=6e-2)


def test_quantized_prefill_rejects_key_padding_mask():
    q = torch.empty(1, 5, 2, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 5, device="cuda", dtype=torch.bool)
    runner = FlashInferPrefillRunner(num_heads=2, head_dim=128, plan_cache={})
    with pytest.raises(ValueError, match="quantized attention does not support key padding masks"):
        flashinfer_prefill(
            q, q, q, runner=runner, key_padding_mask=mask, qk_dtype="nvfp4", v_dtype="fp8"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("capability", "expected_runner"), [((10, 0), "batch"), ((10, 3), "batch"), ((12, 0), "single")]
)
def test_batch_size_one_dispatches_by_architecture(monkeypatch, capability, expected_runner):
    q = torch.empty((1, 4, 2, 64), dtype=torch.bfloat16)
    calls = {name: Mock(return_value=(q, torch.empty(0))) for name in ("single", "batch")}
    monkeypatch.setattr(prefill_module, "_validate_inputs", Mock())
    for name, fn in calls.items():
        monkeypatch.setattr(_flashinfer_runtime, f"run_{name}_prefill", fn)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    flashinfer_prefill(
        q, q, q, runner=FlashInferPrefillRunner(num_heads=2, head_dim=64, plan_cache={})
    )
    assert {name: fn.call_count for name, fn in calls.items()} == {
        name: int(name == expected_runner) for name in calls
    }
