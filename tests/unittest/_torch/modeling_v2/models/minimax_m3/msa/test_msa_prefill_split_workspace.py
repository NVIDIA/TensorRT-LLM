# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP2CTX_E4M3X prefill's split workspace on a GPU with several memory locality domains (SM107).

msa_prefill._prefill_split_workspace hands the fork's PrefillWorkspace to run_prefill: the partials go to pages that
alternate between the domains and the combine reads every tile from the SMs of its own domain. The output must be
bitwise the per-call-partials output, on compact and padded slabs, with the MXFP8 sink, and after the workspace grew.
"""

import math

import pytest
import torch

_HEAD_Q, _HEAD_KV, _D, _PAGE, _TOPK = 64, 4, 128, 128, 16


def _modules():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SM107 only")
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels.msa_utils import (
        require_msa_module,
    )

    require_msa_module()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.msa_kernels import (
        prefill_bf16x,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.msa_kernels.fmha_sm100 import (
        sparse,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.msa_kernels.prefill_bf16x import (
        jit as bf16x_jit,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa import msa_prefill

    if not bf16x_jit.prebuilt_available() and bf16x_jit._cutlass_version() < (4, 8):
        pytest.skip(
            "the SM107 BF16X prefill needs its ahead-of-time build or CUTLASS >= 4.8 (CUTLASS_ROOT)"
        )
    return msa_prefill, prefill_bf16x, sparse


def _workspace_or_skip(msa_prefill, monkeypatch):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_PREFILL_SPLIT_WORKSPACE", raising=False)
    workspace = msa_prefill._prefill_split_workspace(
        torch.device("cuda", torch.cuda.current_device())
    )
    if workspace is None:
        pytest.skip("the device has one memory locality domain")
    return workspace


def _topk_blocks(q_len, kv_len, gen, device):
    """[Hkv, q_len, TopK] batch-local causal block ids (own block, block 0, random others), -1 padded."""
    pos = torch.arange(kv_len - q_len, kv_len, device=device)
    nb = pos // _PAGE + 1
    maxnb = int(nb.max())
    j = torch.arange(maxnb, device=device)
    keys = torch.rand((_HEAD_KV, q_len, maxnb), generator=gen, device=device)
    keys = torch.where(j[None, None, :] >= nb[None, :, None], torch.full_like(keys, 2.0), keys)
    keys = torch.where(
        (j[None, None, :] == (nb - 1)[None, :, None]) | (j == 0)[None, None, :],
        torch.full_like(keys, -1.0),
        keys,
    )
    k = min(_TOPK, maxnb)
    idx = torch.topk(keys, k, dim=-1, largest=False).indices
    idx = torch.where(idx >= nb[None, :, None], torch.full_like(idx, 1 << 30), idx)
    idx = torch.sort(idx, dim=-1).values
    idx = torch.where(idx >= (1 << 30), torch.full_like(idx, -1), idx)
    if k < _TOPK:
        idx = torch.cat(
            [idx, torch.full((_HEAD_KV, q_len, _TOPK - k), -1, dtype=idx.dtype, device=device)], -1
        )
    return idx.to(torch.int32)


def _inputs(sparse, segments, seed):
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    q_lens, kv_lens = [s[0] for s in segments], [s[1] for s in segments]
    pages_per = [-(-kv // _PAGE) for kv in kv_lens]
    pages = sum(pages_per) + 8
    perm = torch.randperm(pages, generator=gen, device=device).to(torch.int32)
    table = torch.zeros((len(segments), max(pages_per)), dtype=torch.int32, device=device)
    start = 0
    for b, n in enumerate(pages_per):
        table[b, :n] = perm[start : start + n]
        start += n
    data_shape = (pages, _HEAD_KV, _PAGE, _D // 2)
    scale_shape = (pages, _HEAD_KV, _PAGE, _D // 16)

    def scales():
        return (
            torch.empty(scale_shape, device=device)
            .uniform_(0.25, 1.0, generator=gen)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )

    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32, device=device
    )
    cu_k = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), dtype=torch.int32, device=device
    )
    q2k = torch.cat([_topk_blocks(ql, kl, gen, device) for ql, kl in segments], dim=1).contiguous()
    k2q_row_ptr, _, schedule = sparse.build_k2q_csr(
        q2k,
        cu_q,
        cu_k,
        _PAGE,
        total_k=sum(kv_lens),
        max_seqlen_k=max(kv_lens),
        max_seqlen_q=max(q_lens),
        total_rows=sum(pages_per),
        qhead_per_kv=_HEAD_Q // _HEAD_KV,
        return_schedule=True,
    )
    return dict(
        q=torch.randn((sum(q_lens), _HEAD_Q, _D), device=device, generator=gen).to(
            torch.float8_e4m3fn
        ),
        paged_kv_cache=(
            torch.randint(0, 256, data_shape, dtype=torch.uint8, device=device, generator=gen),
            torch.randint(0, 256, data_shape, dtype=torch.uint8, device=device, generator=gen),
        ),
        kv_cache_sf=(scales(), scales()),
        kv_indices=table,
        kv_indptr=None,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        k2q_row_ptr=k2q_row_ptr,
        schedule=schedule,
        sm_scale=1.0 / math.sqrt(_D),
        topk=_TOPK,
        seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        k_global_scale=torch.tensor([0.5], dtype=torch.float32, device=device),
        v_global_scale=torch.tensor([0.75], dtype=torch.float32, device=device),
    )


def _run(prefill_bf16x, inputs, workspace, sink):
    total_q = int(inputs["q"].shape[0])
    outs = {
        "out": torch.empty((total_q, _HEAD_Q, _D), dtype=torch.bfloat16, device=inputs["q"].device)
    }
    if sink:
        width = _HEAD_Q * _D
        outs["o_fp8"] = torch.empty((total_q, width), dtype=torch.uint8, device=inputs["q"].device)
        outs["sf_out"] = torch.zeros(
            (-(-total_q // 128) * 128 * width // 32,), dtype=torch.uint8, device=inputs["q"].device
        )
    prefill_bf16x.run_prefill(**inputs, workspace=workspace, **outs)
    return outs


@pytest.mark.parametrize(
    "segments,sink",
    [
        ([(8192, 8192)], False),
        ([(8192, 8192)], True),
        ([(8492, 8492)], False),
        ([(5000, 5000), (3192, 7000)], False),
        ([(2048, 16384)], False),
    ],
    ids=["8k", "8k_mxfp8_sink", "padded_slabs", "batch2", "chunk_on_history"],
)
def test_split_workspace_is_bitwise(segments, sink, monkeypatch):
    msa_prefill, prefill_bf16x, sparse = _modules()
    workspace = _workspace_or_skip(msa_prefill, monkeypatch)
    inputs = _inputs(sparse, segments, seed=len(segments) * 7 + int(sink))
    ref = _run(prefill_bf16x, inputs, None, sink)
    got = _run(prefill_bf16x, inputs, workspace, sink)
    for name in ref:
        assert torch.equal(ref[name].view(torch.uint8), got[name].view(torch.uint8)), name


def test_split_workspace_regrows_and_reuses(monkeypatch):
    msa_prefill, prefill_bf16x, sparse = _modules()
    workspace = _workspace_or_skip(msa_prefill, monkeypatch)
    small = _inputs(sparse, [(2048, 2048)], seed=11)
    ref = _run(prefill_bf16x, small, None, False)["out"]
    _run(prefill_bf16x, _inputs(sparse, [(16384, 16384)], seed=12), workspace, False)
    assert torch.equal(
        ref.view(torch.uint8), _run(prefill_bf16x, small, workspace, False)["out"].view(torch.uint8)
    )


def test_split_workspace_kill_switch_and_capture(monkeypatch):
    msa_prefill, _, _ = _modules()
    workspace = _workspace_or_skip(msa_prefill, monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setenv("TRTLLM_M31_DISABLE_PREFILL_SPLIT_WORKSPACE", "1")
    assert msa_prefill._prefill_split_workspace(device) is None
    monkeypatch.delenv("TRTLLM_M31_DISABLE_PREFILL_SPLIT_WORKSPACE")
    assert msa_prefill._prefill_split_workspace(device) is workspace
    graph, x, captured = torch.cuda.CUDAGraph(), torch.zeros(16, device=device), []
    with torch.cuda.graph(graph):
        x.add_(1)
        captured.append(msa_prefill._prefill_split_workspace(device))
    assert captured == [None]


@pytest.mark.parametrize("segments", [[(8192, 8192)], [(8492, 8492)], [(5000, 5000), (3192, 7000)]])
def test_drop_bf16_keeps_the_mxfp8_copy_and_materializes_the_bf16_bytes(segments):
    """COMBINE_DROP_BF16 through msa_prefill: the combine leaves ``out`` unwritten, the MXFP8 copy is the bytes of the
    full chain, and the sink's ``materialize`` writes exactly the BF16 output the skipped store would have."""
    from types import SimpleNamespace

    msa_prefill, prefill_bf16x, sparse = _modules()
    inputs = _inputs(sparse, segments, seed=len(segments) * 13 + 5)
    ref = _run(prefill_bf16x, inputs, None, True)
    total_q = int(inputs["q"].shape[0])
    width = _HEAD_Q * _D
    device = inputs["q"].device
    out = torch.full((total_q, _HEAD_Q, _D), float("nan"), dtype=torch.bfloat16, device=device)
    unwritten = out.clone()
    o_fp8 = torch.empty((total_q, width), dtype=torch.uint8, device=device)
    sf_out = torch.zeros(
        (-(-total_q // 128) * 128 * width // 32,), dtype=torch.uint8, device=device
    )
    got = []
    k_data, v_data = inputs["paged_kv_cache"]
    k_sf, v_sf = inputs["kv_cache_sf"]
    metadata = SimpleNamespace(
        msa_block_table=inputs["kv_indices"], msa_seq_lens_cuda=inputs["seqused_k"]
    )
    msa_prefill._run_tp2ctx_bf16x(
        prefill_bf16x.run_prefill,
        inputs["q"],
        k_data,
        v_data,
        torch.stack((k_sf, v_sf), dim=1),
        metadata,
        len(segments),
        inputs["cu_seqlens_q"],
        inputs["cu_seqlens_k"],
        inputs["k2q_row_ptr"],
        inputs["schedule"],
        inputs["sm_scale"],
        inputs["topk"],
        inputs["k_global_scale"],
        inputs["v_global_scale"],
        out,
        True,
        mxfp8=(o_fp8, sf_out),
        mxfp8_sink=lambda o, q, sf, materialize=None: got.append((o, q, sf, materialize)),
        drop_bf16=True,
    )
    torch.cuda.synchronize()
    assert len(got) == 1 and got[0][3] is not None
    assert torch.equal(
        out.view(torch.uint8), unwritten.view(torch.uint8)
    )  # the BF16 store was skipped
    assert torch.equal(o_fp8, ref["o_fp8"]) and torch.equal(sf_out, ref["sf_out"])
    got[0][3]()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), ref["out"].view(torch.uint8))
    assert torch.equal(o_fp8, ref["o_fp8"])  # materialize writes only the BF16 output
