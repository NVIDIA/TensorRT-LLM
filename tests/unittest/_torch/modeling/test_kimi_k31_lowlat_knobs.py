# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gates of the Kimi K3.1 low-batch decode scheduling knobs in ``modeling_kimi_linear``.

The knobs only move work between CUDA streams (same kernels, same inputs), so their contract
is where they apply: Kimi K3.1 (a text config with ``dsa_config``) generation servers with the
routed-expert backend they were written for. Kimi K3 must never pick them up.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

_K31 = SimpleNamespace(dsa_config={"dsa_layers": [4], "index_topk": 2048})
_K3 = SimpleNamespace(dsa_config=None)


@pytest.mark.parametrize(
    "switch, backend, expected",
    [
        (True, "TRTLLM", True),
        (False, "TRTLLM", False),
        # The CuteDSL backends always run the latent chain on the aux stream; the switch is
        # only for the TRTLLM-Gen backend.
        (True, "CUTEDSL", False),
        (True, "CUTLASS", False),
    ],
)
def test_latent_down_stream_gate(monkeypatch, switch, backend, expected):
    monkeypatch.setattr(mkl, "_K31_LATENT_DOWN_STREAM", switch)
    assert mkl._k31_latent_down_stream_requested(backend) is expected


_DECODE_SWITCHES = (
    "_K31_LATENT_DOWN_STREAM",
    "_K31_FUSED_MOE_TAIL",
    "_K31_SEPARATED_ROUTING",
    "_K31_L2PF",
    "_K31_PACKED_MOE_FRONT",
    "_K31_SHARDED_MOE_FRONT",
    "_K31_SHARDED_MOE_FRONT_SHARED",
    "_K31_SHARDED_MOE_FRONT_SORT",
    "_K3_AR_PUSH_AMID",
    "_K3_AR_FINISH_FUSED",
    "_K3_ULL_OPROJ_SANDWICH",
    "_K3_ATTN_RES_PDL_MODE",
    "_K3_ATTN_RES_PDL",
    "_K3_ATTN_RES_PDL_MAX_M",
    "_K3_FUSE_MOE_RESIDUAL",
)


@pytest.mark.parametrize(
    "cfg, role, adp, max_num_tokens, expected",
    [
        # TEP32 generation server (MNT 192): every TEP decode switch, the routing sort in the front.
        (_K31, "generation", False, 192, ("tep", "1", 192, True)),
        # TEP16 generation server of the 2 x TEP8 configuration (MNT 288): the PDL kernel stays at 192 rows.
        (_K31, "generation", False, 288, ("tep", "1", 192, True)),
        # DEP16 gen (MNT 256): mode 2 + the MoE block's residual.
        (_K31, "generation", True, 256, ("dep", "2", 256, False)),
        # Context servers, aggregated serving and Kimi K3: nothing.
        (_K31, "context", False, 8192, (None, "0", 0, False)),
        (_K31, "", False, 8192, (None, "0", 0, False)),
        (_K3, "generation", False, 192, (None, "0", 0, False)),
    ],
)
def test_configure_k31_decode(monkeypatch, cfg, role, adp, max_num_tokens, expected):
    from tensorrt_llm._torch.distributed import ops

    for name in _DECODE_SWITCHES:
        monkeypatch.setattr(mkl, name, getattr(mkl, name))
    for name in ("_K3_AR_PUSH", "_K31_BF16_OPROJ_AR_FINISH", "_K31_BF16_OPROJ_AR_PUSH"):
        monkeypatch.setattr(ops, name, getattr(ops, name))
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    layout, mode, max_m, sort = expected
    model_config = SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=adp, tp_size=32 if layout == "tep" else 16),
        max_num_tokens=max_num_tokens,
    )
    mkl._configure_k31_decode(cfg, model_config)
    tep = layout == "tep"
    for name in _DECODE_SWITCHES[:7] + _DECODE_SWITCHES[8:11]:
        assert getattr(mkl, name) is tep, name
    assert mkl._K31_SHARDED_MOE_FRONT_SORT is sort
    assert mkl._K3_ATTN_RES_PDL_MODE == mode
    assert mkl._K3_ATTN_RES_PDL is (layout is not None)
    assert mkl._K3_ATTN_RES_PDL_MAX_M == max_m
    assert mkl._K3_FUSE_MOE_RESIDUAL is (layout == "dep")
    assert ops._K3_AR_PUSH is ops._K31_BF16_OPROJ_AR_FINISH is ops._K31_BF16_OPROJ_AR_PUSH is tep


_SORT = (16, 0, 112, 112, 129)  # (tile, slot start, local experts, slots, multi-CTA from)


def _moe_runtime_stub(accepts_mxfp8, sort_finalized, sort_unfinalized):
    backend = SimpleNamespace(accepts_mxfp8_input=accepts_mxfp8)
    targets = {True: sort_finalized, False: sort_unfinalized}
    return SimpleNamespace(
        routed_experts=SimpleNamespace(backend=backend),
        _k31_runner_sort_target=lambda num_tokens, do_finalize: targets[do_finalize],
    )


@pytest.mark.parametrize(
    "knob, dtype, accepts_mxfp8, sort_finalized, sort_unfinalized, expected",
    [
        (True, torch.bfloat16, True, _SORT, _SORT, True),
        (False, torch.bfloat16, True, _SORT, _SORT, False),
        (True, torch.float16, True, _SORT, _SORT, False),
        # Backends without MXFP8 activations keep their own input quantization.
        (True, torch.bfloat16, False, _SORT, _SORT, False),
        # Only where the routing sort routes the forward in both finalize modes (else the
        # top-k / quantization chain of the MoE op runs and quantizes itself).
        (True, torch.bfloat16, True, None, _SORT, False),
        (True, torch.bfloat16, True, _SORT, None, False),
    ],
)
def test_latent_stream_quant_gate(
    monkeypatch, knob, dtype, accepts_mxfp8, sort_finalized, sort_unfinalized, expected
):
    monkeypatch.setattr(mkl, "_K31_LATENT_STREAM_QUANT", knob)
    stub = _moe_runtime_stub(accepts_mxfp8, sort_finalized, sort_unfinalized)
    latent = torch.zeros(28, 3584, dtype=dtype)
    assert mkl.KimiK3MoERuntime._k31_latent_stream_quant_applies(stub, latent) is expected


def _l2pf_model(front_weight, shared_cols=0):
    """A one-layer stand-in of the model for ``_k3_l2pf_tensors``: an MoE layer whose sharded front (if any) has
    ``front_weight`` (head rows first: 384 at TP16) and computes ``shared_cols`` shared gate_up columns."""
    gate = torch.empty(4, 8)
    down = SimpleNamespace(weight=torch.empty(16, 8))
    shared = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=torch.empty(6, 8)),
        down_proj=SimpleNamespace(weight=torch.empty(8, 3)),
    )
    moe = SimpleNamespace(
        gate=SimpleNamespace(weight=gate),
        routed_expert_down_proj=down,
        shared_experts=shared,
        latent_down_stream=None,
        _k31f_weight=front_weight,
        _k31f_ws={"world": 16},
        _k31f_shared=(None, shared_cols, 4.0, 25.0) if shared_cols else None,
    )
    runtime = mkl.KimiK3MoERuntime
    moe._k31_sharded_front_shared_cols = lambda n: runtime._k31_sharded_front_shared_cols(moe, n)
    moe.k31_sharded_front_weight = lambda n: runtime.k31_sharded_front_weight(moe, n)
    moe.k31_sharded_front_reads_shared_gate_up = (
        lambda n: runtime.k31_sharded_front_reads_shared_gate_up(moe, n)
    )
    layer = SimpleNamespace(layer_idx=1, is_moe=True, block_sparse_moe=moe)
    return SimpleNamespace(layers=[layer]), moe


def _same(a, b):
    return [(t.data_ptr(), tuple(t.shape)) for t in a] == [
        (t.data_ptr(), tuple(t.shape)) for t in b
    ]


@pytest.mark.parametrize("front", ["off", "head", "shared"])
@pytest.mark.parametrize("num_tokens", [8, 16, 32, 64])
def test_l2pf_front_follows_the_sharded_front(monkeypatch, front, num_tokens):
    """TRTLLM_K3_L2PF_V2 prefetches the weight the MoE front reads: where the sharded front serves the rows (up to
    TRTLLM_KIMI_K31_SHARDED_MOE_FRONT_MAX_M) its head rows, plus its shared gate_up rows where it computes them (up to
    16 rows; the shared gate_up projection is then not prefetched), else the router and latent_down."""
    monkeypatch.setattr(mkl, "_K3_MOE_L2_PREFETCH", "")
    monkeypatch.setattr(
        mkl, "_k3_l2pf_linear_tensors", lambda m: [m.weight] if m is not None else []
    )
    weight = None if front == "off" else torch.empty(384 + (768 if front == "shared" else 0), 8)
    model, moe = _l2pf_model(weight, 384 if front == "shared" else 0)
    ref = lambda: model  # noqa: E731
    shared_gate_up, shared_down = (
        moe.shared_experts.gate_up_proj.weight,
        moe.shared_experts.down_proj.weight,
    )
    serves = front != "off" and num_tokens <= mkl._K31_SHARDED_MOE_FRONT_MAX_M
    with_shared = (
        serves and front == "shared" and num_tokens <= mkl._K31_SHARDED_MOE_FRONT_SHARED_MAX_M
    )
    got = mkl._k3_l2pf_tensors(ref, "oproj_down", 1, num_tokens)
    if serves:
        assert _same(got, [weight if with_shared else weight[:384]])
    else:
        assert _same(got, [moe.gate.weight, moe.routed_expert_down_proj.weight])
    got = mkl._k3_l2pf_tensors(ref, "attention_moe", 1, num_tokens)
    if with_shared:
        assert _same(got[-1:], [shared_down]) and not _same(got[-2:-1], [shared_gate_up])
    else:
        assert _same(got[-2:], [shared_gate_up, shared_down])
