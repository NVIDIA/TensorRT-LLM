# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.minimax_m3.embedding_firstnorm import op as embedding
from tensorrt_llm._torch.cute_dsl_kernels.minimax_m3.moe import op as moe
from tensorrt_llm._torch.cute_dsl_kernels.minimax_m3.mx_gemv import op as gemv


@pytest.mark.cpu_only
@pytest.mark.parametrize("n,k", [(1537, 6144), (1536, 6145), (0, 6144), (1536, 0)])
def test_explicit_gemv_config_declines_incomplete_tiles(n, k):
    assert gemv._declines(n, k, 1, False, (2, 12))


@pytest.mark.cpu_only
def test_frozen_warmup_retries_failed_compilation(monkeypatch):
    monkeypatch.setattr(gemv, "_ensured", set())
    monkeypatch.setattr(gemv, "_frozen", [True])
    weight = torch.empty(1536, 6144, dtype=torch.float8_e4m3fn)
    scales = torch.empty(0)
    calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(gemv, "precompile", fail)
    with pytest.raises(RuntimeError, match="compile failed"):
        gemv.ensure_compiled(weight, scales, False)
    monkeypatch.setattr(
        gemv, "precompile", lambda *args, **kw: calls.append((args[2], kw["x_bf16"]))
    )
    gemv.ensure_compiled(weight, scales, False)
    assert calls == [(1, False), (16, False), (32, False), (48, False), (1, True)]
    gemv.ensure_compiled(weight, scales, False)
    assert len(calls) == 5


@pytest.mark.cpu_only
@pytest.mark.parametrize("inference", [False, True])
def test_reloaded_shared_weights_refresh_the_cached_tiles(monkeypatch, inference):
    monkeypatch.setattr(moe, "_PRETILED", {})
    monkeypatch.setattr(moe, "_pretile_shared", lambda *sources: torch.cat(sources))
    with torch.inference_mode(inference):
        sources = [torch.arange(4) for _ in range(4)]
        before = moe._pretiled(*sources)
        sources[0].add_(10)
        after = moe._pretiled(*sources)
        assert torch.equal(after, torch.cat(sources))
        assert after.data_ptr() == before.data_ptr()
    # A later state can be constructed outside the context that created its weights.
    assert torch.equal(moe._pretiled(*sources), torch.cat(sources))


@pytest.mark.cpu_only
def test_shared_state_default_reserves_shared_ctas(monkeypatch):
    class ReachedKernelSelection(Exception):
        pass

    def kernel(config, path):
        assert config["num_ctas"] == 148 - moe.SHARED_CTAS
        raise ReachedKernelSelection()

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: Mock(spec_set=["multi_processor_count"], multi_processor_count=148),
    )
    monkeypatch.setattr(moe, "is_supported", lambda device: (True, ""))
    monkeypatch.setattr(moe, "make_phase_plan", lambda *args: (torch.zeros(1), 1))
    monkeypatch.setattr(moe, "_kernel_module", kernel)
    with pytest.raises(ReachedKernelSelection):
        moe.M3MoeState(torch.device("cuda", 0), num_tokens=6, shared=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor storage required")
@pytest.mark.parametrize("bad", ["offset", "stride", "cpu"])
def test_embedding_rejects_unsupported_buffers(bad):
    h = embedding.HIDDEN
    ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    table = torch.empty(2, h, dtype=torch.bfloat16, device="cuda")
    weight = torch.empty(h, dtype=torch.bfloat16, device="cuda")
    raw = torch.empty(1, h, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(raw)
    embedding._check(ids, table, weight, raw, out)
    if bad == "offset":
        out = torch.empty(h + 1, dtype=torch.bfloat16, device="cuda")[1:].view(1, h)
    elif bad == "stride":
        weight = torch.empty(2 * h, dtype=torch.bfloat16, device="cuda")[::2]
    else:
        ids = ids.cpu()
    with pytest.raises(ValueError):
        embedding._check(ids, table, weight, raw, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor storage required")
@pytest.mark.parametrize("bad", ["shape", "dtype", "stride", "device", "offset"])
def test_moe_rejects_invalid_output_storage(bad):
    device = torch.device("cuda", torch.cuda.current_device())
    out = torch.empty(2, moe.HIDDEN, device=device, dtype=torch.bfloat16)
    moe._validate_output(out, 2, device)
    if bad == "offset":
        out = torch.empty(2 * moe.HIDDEN + 1, device=device, dtype=torch.bfloat16)[1:].view(
            2, moe.HIDDEN
        )
    elif bad == "shape":
        out = out[:1]
    elif bad == "dtype":
        out = out.float()
    elif bad == "stride":
        out = torch.empty(2, 2 * moe.HIDDEN, device=device, dtype=torch.bfloat16)[:, ::2]
    else:
        out = out.cpu()
    with pytest.raises(ValueError):
        moe._validate_output(out, 2, device)


@pytest.mark.cpu_only
def test_gemv_declines_transposed_weights_and_strided_activations():
    x = torch.empty(1, 6144, dtype=torch.float8_e4m3fn)
    weight = torch.empty(6144, 1536, dtype=torch.float8_e4m3fn).T
    scale = torch.empty(0)
    assert gemv.m3_mx_gemv(x, scale, weight, scale) is None
    x = torch.empty(1, 12288, dtype=torch.float8_e4m3fn)[:, ::2]
    assert gemv.m3_mx_gemv(x, scale, weight.contiguous(), scale) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA binding validation required")
@pytest.mark.parametrize("geometry", [(0, 64, 1), (1, 0, 1), (1, 64, 0), (1 << 32, 64, 1)])
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_full_context_reduce_rejects_invalid_geometry(geometry, tp):
    partial = torch.zeros(1, 1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    stats = torch.zeros(1, 1, 1, 1, device="cuda")
    lens = torch.ones(1, device="cuda", dtype=torch.int32)
    out = torch.empty(1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="producer geometry"):
        prefix = "sm107_tp2" if tp == 2 else f"sm107tp{tp}"
        getattr(torch.ops.trtllm, f"minimax_m3_{prefix}_fullctx_reduce")(
            partial, stats, lens, out, 1, *geometry, True, False
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA binding validation required")
@pytest.mark.parametrize("one_warp", [False, True])
@pytest.mark.parametrize(
    "undersized", ["src_new", "slots", "ids", "draft_ids", "pos_off", "kv_off"]
)
@pytest.mark.parametrize("tp", [4, 8])
def test_fold_tail_rejects_undersized_buffers(one_warp, undersized, tp):
    def ints(*shape):
        return torch.zeros(*shape, device="cuda", dtype=torch.int32)

    args = dict(
        ctrl=ints(1),
        slots=torch.arange(2, device="cuda", dtype=torch.int64),
        rows=2,
        src_new=ints(2, 1),
        dst_new=ints(2, 1),
        src_next=ints(2, 2),
        dst_next=ints(2, 2),
        src_draft=ints(2, 1),
        dst_draft=ints(2, 1),
        src_lens=ints(2),
        dst_lens=ints(2),
        ids=ints(4),
        draft_ids=ints(2),
        pos_off=ints(4),
        kv_off=ints(2),
        tokens_per_step=2,
    )
    args[undersized] = args[undersized][:1]
    suffix = "_one_warp" if one_warp else ""
    op = getattr(torch.ops.trtllm, f"minimax_m3_sm107_tp{tp}_fold_tail{suffix}")
    with pytest.raises(RuntimeError, match="must hold rows|staging outputs"):
        op(**args)
