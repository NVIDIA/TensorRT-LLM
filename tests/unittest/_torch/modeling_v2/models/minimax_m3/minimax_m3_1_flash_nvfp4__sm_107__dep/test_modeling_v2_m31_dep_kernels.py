# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 DEP target's three kernels, on CPU tensors and stub layers.

DEP_LC2 takes a sparse decode call only at the attention-DP geometry it serves (4 KV heads, E4M3 q, packed NVFP4
cache, 7-token verify, at most 448 rows) and raises on every other call, naming the reason (no Triton fallback; a
build without the op fails at install); DEP_OWNT runs OWN-T, the block-owner decode, on the CUDA calls LC2 takes,
with LC2's operands and no MXFP8 hand-off, and leaves LC2 every call it declines and every call when off or unbuilt;
SOFTMAX_MC routes 1..128 rows to the multi-CTA kernel and 129..448 rows of vocab 200064 to Path B, and leaves the
rest to FlashInfer; TOPP_DET needs an op with V4K8 and keep_all_ge1 and launches it with keep_all_ge1 off, so top_p
>= 1 rows are FlashInfer's. CPU only.
"""

import sys
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        lc2_decode,
        softmax_mc,
        topp_det,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

_STOCK_MODULE = (
    "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode"
)
_VOCAB = 200064


def _decode_call(reqs=40, dql=8, kv_heads=4, q_dtype=torch.float8_e4m3fn, pages=6):
    """The arguments MsaDecodeFmha._run_sparse hands the hook on an NVFP4 sparse layer: HND cache views, the block
    scales' per-page views, the head-major view of the token-major top-k table and the aligned global scales."""
    total_q = reqs * dql
    heads = 16 * kv_heads
    cache = torch.zeros(pages, 2, kv_heads, 128, 64, dtype=torch.uint8)
    scales = torch.zeros(pages, 2, kv_heads, 128, 8, dtype=torch.uint8)
    globals_ = torch.ones(2, 4, dtype=torch.float32)
    args = (
        torch.zeros(total_q, heads, 128, dtype=q_dtype),
        cache[:, 0],
        cache[:, 1],
        torch.full((total_q, kv_heads, 16), -1, dtype=torch.int32).permute(1, 0, 2),
        torch.zeros(reqs, pages, dtype=torch.int32),
        torch.zeros(reqs, dtype=torch.int32),
    )
    kwargs = dict(
        sm_scale=128**-0.5,
        output=torch.empty(total_q, heads, 128, dtype=torch.bfloat16),
        decode_query_len=dql,
        k_block_scale=scales[:, 0],
        v_block_scale=scales[:, 1],
        k_global_scale=globals_[0, :1],
        v_global_scale=globals_[1, :1],
    )
    return args, kwargs


def _reason(args, kwargs):
    q, k, v, topk, bt, seq = args
    return lc2_decode.decline_reason(
        q,
        k,
        v,
        topk,
        bt,
        seq,
        kwargs["output"],
        kwargs["decode_query_len"],
        kwargs.get("kv_scale"),
        kwargs.get("k_block_scale"),
        kwargs.get("v_block_scale"),
        kwargs.get("k_global_scale"),
        kwargs.get("v_global_scale"),
        kwargs.get("num_topk_chunks"),
    )


@pytest.mark.parametrize("reqs", [1, 4, 19, 35, 40, 45, 48, 56])
def test_lc2_takes_the_dep_k7_steps(reqs):
    assert _reason(*_decode_call(reqs=reqs)) is None


@pytest.mark.parametrize(
    "change, why",
    [
        (dict(reqs=57), "rows"),
        (dict(dql=4), "decode_query_len"),
        (dict(q_dtype=torch.bfloat16), "E4M3"),
        (dict(kv_heads=2), "q / output"),
    ],
)
def test_lc2_declines_outside_the_measured_geometry(change, why):
    assert why in _reason(*_decode_call(**change))


def test_lc2_declines_the_fp8_cache_and_test_only_split_factor():
    args, kwargs = _decode_call()
    assert _reason(args, dict(kwargs, kv_scale=torch.ones(1))) == "not the packed-NVFP4 cache call"
    assert _reason(args, dict(kwargs, num_topk_chunks=4)) == "not the packed-NVFP4 cache call"
    assert _reason(args, dict(kwargs, k_global_scale=None)) == "not the packed-NVFP4 cache call"


def test_lc2_declines_layouts_the_kernel_would_misread():
    args, kwargs = _decode_call()
    q, k, v, topk, bt, seq = args
    wide = torch.zeros(k.shape[0], 4, 128, 128, dtype=torch.uint8)[..., :64]
    assert "dense rows" in _reason((q, wide, wide, topk, bt, seq), kwargs)
    assert "differ" in _reason((q, k, k.contiguous(), topk, bt, seq), kwargs)
    assert "span" in _reason((q, k, v, topk, bt, seq[:-1]), kwargs)
    assert "q / output" in _reason(args, dict(kwargs, output=kwargs["output"].float()))
    assert "layout" in _reason((q, k, v, topk.long(), bt, seq), kwargs)
    scales = torch.zeros(k.shape[0], 4, 128, 16, dtype=torch.uint8)[..., :8]
    assert "block scales" in _reason(args, dict(kwargs, k_block_scale=scales, v_block_scale=scales))


def test_lc2_hook_runs_the_op_on_a_taken_call(monkeypatch):
    calls = []
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: lambda *a: calls.append(a))
    monkeypatch.setattr(lc2_decode, "DEP_LC2_PDL", True)
    args, kwargs = _decode_call(reqs=40)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    (call,) = calls
    q, k, v, topk, bt, seq = args
    assert call[0].dtype == torch.uint8 and call[0].data_ptr() == q.data_ptr()
    assert call[1] is k and call[2] is v and call[7] is topk and call[8] is bt and call[9] is seq
    assert call[3] is kwargs["k_block_scale"] and call[6] is kwargs["v_global_scale"]
    out = call[10]
    assert out.shape == (320, 8192) and out.data_ptr() == kwargs["output"].data_ptr()
    assert call[11:] == (None, None, 64, 8, kwargs["sm_scale"], False, True, False, None)


@pytest.mark.parametrize(
    "reqs, enabled, use_pdl",
    [(4, True, False), (5, True, True), (56, True, True), (5, False, False), (56, False, False)],
)
def test_lc2_pdl_launches_only_the_split_grid(monkeypatch, reqs, enabled, use_pdl):
    calls = []
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: lambda *a: calls.append(a))
    monkeypatch.setattr(lc2_decode, "DEP_LC2_PDL", enabled)
    args, kwargs = _decode_call(reqs=reqs)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    (call,) = calls
    assert call[16:] == (False, use_pdl, False, None)


def _no_stock_decode(monkeypatch):
    """A stock (Triton) decode module that fails the test if the hook ever reaches it."""
    fake = types.ModuleType(_STOCK_MODULE)
    fake.minimax_m3_sparse_attn_decode = lambda *a, **kw: pytest.fail(
        "the DEP hook has no Triton fallback"
    )
    monkeypatch.setitem(sys.modules, _STOCK_MODULE, fake)


def _declined_call(case):
    """A decode call LC2 declines for ``case``, and the text its decline reason carries."""
    args, kwargs = _decode_call(**case.get("call", {}))
    q, k, v, topk, bt, seq = args
    if case.get("seq") == "short":
        args = (q, k, v, topk, bt, seq[:-1])
    if case.get("topk") == "long":
        args = (q, k, v, topk.long(), bt, seq)
    if case.get("cache") == "wide":
        wide = torch.zeros(k.shape[0], 4, 128, 128, dtype=torch.uint8)[..., :64]
        args = (q, wide, wide, topk, bt, seq)
    if case.get("cache") == "differ":
        args = (q, k, k.contiguous(), topk, bt, seq)
    if case.get("scales") == "wide":
        scales = torch.zeros(k.shape[0], 4, 128, 16, dtype=torch.uint8)[..., :8]
        kwargs = dict(kwargs, k_block_scale=scales, v_block_scale=scales)
    if case.get("output") == "float":
        kwargs = dict(kwargs, output=kwargs["output"].float())
    if "global" in case:
        kwargs = dict(kwargs, k_global_scale=kwargs["k_global_scale"].double())
    kwargs = dict(kwargs, **case.get("kwargs", {}))
    return args, kwargs


_DECLINES = {
    "fp8_cache": (dict(kwargs=dict(kv_scale=torch.ones(1))), "not the packed-NVFP4 cache call"),
    "split_factor": (dict(kwargs=dict(num_topk_chunks=4)), "not the packed-NVFP4 cache call"),
    "no_global_scale": (dict(kwargs=dict(k_global_scale=None)), "not the packed-NVFP4 cache call"),
    "bf16_q": (dict(call=dict(q_dtype=torch.bfloat16)), "not E4M3"),
    "k4_verify": (dict(call=dict(dql=4)), "decode_query_len 4"),
    "456_rows": (dict(call=dict(reqs=57)), "456 rows (outside 1..448)"),
    "span": (dict(seq="short"), "q does not cover exactly the batch's decode span"),
    "2_kv_heads": (dict(call=dict(kv_heads=2)), "q / output not [T, 64, 128]"),
    "float_output": (dict(output="float"), "q / output not [T, 64, 128]"),
    "wide_cache": (dict(cache="wide"), "cache not [pages, 4, 128, 64]"),
    "kv_differ": (dict(cache="differ"), "k and v views differ"),
    "wide_scales": (dict(scales="wide"), "block scales not [pages, 4, 128, 8]"),
    "double_global": (dict(**{"global": True}), "global scales not one float32 each"),
    "int64_topk": (dict(topk="long"), "top-k table / block table / seq_lens layout"),
}


@pytest.mark.parametrize("name", sorted(_DECLINES))
def test_lc2_hook_raises_each_decline_reason(monkeypatch, name):
    """Every reason LC2 declines a call for raises with its text and runs no kernel: no LC2, no OWN-T, no Triton."""
    _no_stock_decode(monkeypatch)
    monkeypatch.setattr(
        lc2_decode, "lc2_op", lambda: lambda *a: pytest.fail("a declined call must not run LC2")
    )
    monkeypatch.setattr(
        lc2_decode, "own_t_op", lambda: lambda *a: pytest.fail("a declined call must not run OWN-T")
    )
    case, why = _DECLINES[name]
    args, kwargs = _declined_call(case)
    assert why in _reason(args, kwargs)
    with pytest.raises(RuntimeError, match="DEP_LC2 cannot take this sparse decode call") as err:
        lc2_decode.dep_lc2_decode(*args, **kwargs)
    assert why in str(err.value) and "TRTLLM_M31_DISABLE_DEP_LC2=1" in str(err.value)


def test_lc2_hook_raises_without_the_op(monkeypatch):
    _no_stock_decode(monkeypatch)
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: None)
    args, kwargs = _decode_call()
    with pytest.raises(RuntimeError, match="the op is not built"):
        lc2_decode.dep_lc2_decode(*args, **kwargs)


def test_the_dep_hook_imports_no_triton_decode():
    import inspect

    assert "triton" not in inspect.getsource(lc2_decode)


class _Fmha:
    def __init__(self, hook=None):
        self.sparse_decode_hook = hook


def _layer(kv_heads, libs):
    manager = types.SimpleNamespace(fmha_libs=libs)
    backend = types.SimpleNamespace(num_kv_heads=kv_heads, _fmha_manager=manager)
    return types.SimpleNamespace(self_attn=types.SimpleNamespace(attn=backend))


def test_install_hooks_every_4_head_msa_decode_library(monkeypatch):
    monkeypatch.setattr(lc2_decode, "DEP_LC2", True)
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: object())
    own = object()
    decode, prefill, taken = _Fmha(), object(), _Fmha(own)
    layers = [
        _layer(4, [prefill, decode]),
        _layer(4, [taken]),
        _layer(2, [_Fmha()]),
        types.SimpleNamespace(),
    ]
    assert lc2_decode.install(layers) == 1
    assert decode.sparse_decode_hook is lc2_decode.dep_lc2_decode
    assert taken.sparse_decode_hook is own
    assert layers[2].self_attn.attn._fmha_manager.fmha_libs[0].sparse_decode_hook is None


def test_install_again_counts_the_hooks_it_set(monkeypatch):
    monkeypatch.setattr(lc2_decode, "DEP_LC2", True)
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: object())
    layers = [_layer(4, [_Fmha()]), _layer(4, [_Fmha()]), _layer(4, [_Fmha(object())])]
    assert lc2_decode.install(layers) == 2
    assert lc2_decode.install(layers) == 2


def test_the_model_hooks_the_layers_once_their_fmha_managers_exist(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        modeling,
    )

    monkeypatch.setattr(lc2_decode, "DEP_LC2", True)
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: object())
    monkeypatch.setattr(
        modeling.MiniMaxM3VLForConditionalGeneration, "cache_derived_state", lambda self: None
    )
    # cache_derived_state also runs DEP_IDXK_PRODUCER_NVFP4 and DEP_IDX_PDL, which switch process-wide settings of the
    # built-in model: give them a built-in model without those settings (they install nothing) and restore the
    # reader's pre-wait prefetch afterwards.
    monkeypatch.setattr(modeling._ik, "_mm3", types.SimpleNamespace())
    monkeypatch.setattr(modeling._ipdl, "_mm3", types.SimpleNamespace())
    monkeypatch.setattr(modeling._ipdl._ik, "_DECODE_PREWAIT_PREFETCH", 0)
    model = object.__new__(modeling.ModelingV2MiniMaxM31FlashNvfp4Sm107Dep)
    torch.nn.Module.__init__(model)
    model.register_parameter("w", torch.nn.Parameter(torch.zeros(1)))
    # As built with deferred weight creation: no layer has its FMHA manager until create_weights.
    backend = types.SimpleNamespace(num_kv_heads=4)
    # The built-in model is an nn.Module; the NATIVE_ADD_RMSNORM plain-norm install walks its modules().
    model.model = types.SimpleNamespace(
        layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(attn=backend))],
        modules=lambda: iter(()),
    )
    model._rank, model._lc2_layers = 0, 0
    model.model_config = types.SimpleNamespace(
        use_cuda_graph=True, max_num_tokens=384, moe_max_num_tokens=384, extra_attrs={}
    )
    model._pad_mask, model._pad_layers = modeling._pad.PadMask(), 0
    decode = _Fmha()
    backend._fmha_manager = types.SimpleNamespace(fmha_libs=[decode])
    model.cache_derived_state()
    assert model._lc2_layers == 1
    assert decode.sparse_decode_hook is lc2_decode.dep_lc2_decode


def test_install_sets_no_hook_when_off(monkeypatch):
    monkeypatch.setattr(lc2_decode, "DEP_LC2", False)
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: object())
    decode = _Fmha()
    assert lc2_decode.install([_layer(4, [decode])]) == 0
    assert decode.sparse_decode_hook is None


def test_install_fails_on_a_build_without_the_op(monkeypatch):
    monkeypatch.setattr(lc2_decode, "DEP_LC2", True)
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: None)
    decode = _Fmha()
    with pytest.raises(RuntimeError, match="no Triton fallback.*TRTLLM_M31_DISABLE_DEP_LC2=1"):
        lc2_decode.install([_layer(4, [decode])])
    assert decode.sparse_decode_hook is None


def test_the_kill_switch_is_the_dep_lc2_item():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
    )

    assert lc2_decode.DEP_LC2 == (not m31_switches.m31_disabled("DEP_LC2"))
    assert lc2_decode.DEP_LC2_PDL == (not m31_switches.m31_disabled("DEP_LC2_PDL"))


def test_reserve_asks_for_4_kv_heads(monkeypatch):
    asked = []
    prepare = lambda anchor, kv_heads: asked.append(kv_heads) or 236716032  # noqa: E731
    monkeypatch.setattr(
        lc2_decode,
        "family_ops",
        lambda: types.SimpleNamespace(minimax_m3_sm107_tp2_sparse_decode_lc_prepare=prepare),
    )
    assert lc2_decode.reserve_workspace(torch.device("cpu")) == 236716032
    assert asked == [4]


def test_reserve_names_the_switch_on_a_build_that_refuses_4_heads(monkeypatch):
    def refuse(anchor, kv_heads):
        raise RuntimeError("LC2 split requires the TP2 geometry")

    monkeypatch.setattr(
        lc2_decode,
        "family_ops",
        lambda: types.SimpleNamespace(minimax_m3_sm107_tp2_sparse_decode_lc_prepare=refuse),
    )
    with pytest.raises(RuntimeError, match="TRTLLM_M31_DISABLE_DEP_LC2=1"):
        lc2_decode.reserve_workspace(torch.device("cpu"))


def _own_t_route(monkeypatch, own_t, reason=None):
    """LC2 and OWN-T stubs: OWN-T takes every call (``reason`` None) or declines it with ``reason``; returns the
    (LC2 calls, OWN-T calls) lists."""
    lc2_calls, own_t_calls = [], []
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: lambda *a: lc2_calls.append(a))
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", own_t)
    monkeypatch.setattr(lc2_decode, "own_t_op", lambda: lambda *a: own_t_calls.append(a))
    monkeypatch.setattr(lc2_decode, "own_t_decline_reason", lambda *a: reason)
    return lc2_calls, own_t_calls


@pytest.mark.parametrize("pdl", [True, False])
def test_own_t_runs_the_calls_lc2_takes_with_lc2s_operands(monkeypatch, pdl):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        merge_mxfp8,
    )

    lc2_calls, own_t_calls = _own_t_route(monkeypatch, True)
    monkeypatch.setattr(lc2_decode, "DEP_LC2_PDL", pdl)
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", True)
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    merge_mxfp8.reserve(torch.device("cpu"))
    args, kwargs = _decode_call(reqs=40)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    assert lc2_calls == []
    (call,) = own_t_calls
    q, k, v, topk, bt, seq = args
    assert call[0].dtype == torch.uint8 and call[0].data_ptr() == q.data_ptr()
    assert call[1] is k and call[2] is v and call[7] is topk and call[8] is bt and call[9] is seq
    assert call[3] is kwargs["k_block_scale"] and call[4] is kwargs["v_block_scale"]
    assert call[5] is kwargs["k_global_scale"] and call[6] is kwargs["v_global_scale"]
    out = call[10]
    assert out.shape == (320, 8192) and out.data_ptr() == kwargs["output"].data_ptr()
    q8, sf = call[11], call[12]
    assert q8.shape == (320, 8192) and sf.numel() == merge_mxfp8.scale_bytes(320)
    assert call[13:] == (64, 8, kwargs["sm_scale"], pdl)
    assert merge_mxfp8.SLOT.take(out) == (q8, sf)


def test_own_t_without_buffers_hands_none_and_publishes_nothing(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        merge_mxfp8,
    )

    _, own_t_calls = _own_t_route(monkeypatch, True)
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    args, kwargs = _decode_call(reqs=40)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    (call,) = own_t_calls
    assert call[11:14] == (None, None, 64)
    assert merge_mxfp8.SLOT.pending is None


@pytest.mark.parametrize(
    "own_t, reason",
    [(False, None), (True, "block scales not 16-byte aligned")],
    ids=["off", "declined"],
)
def test_lc2_runs_the_calls_own_t_does_not_take(monkeypatch, own_t, reason):
    lc2_calls, own_t_calls = _own_t_route(monkeypatch, own_t, reason)
    args, kwargs = _decode_call(reqs=40)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    assert own_t_calls == [] and len(lc2_calls) == 1


def test_lc2_runs_every_call_on_a_build_without_own_t(monkeypatch):
    lc2_calls = []
    monkeypatch.setattr(lc2_decode, "lc2_op", lambda: lambda *a: lc2_calls.append(a))
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", True)
    monkeypatch.setattr(lc2_decode, "own_t_op", lambda: None)
    args, kwargs = _decode_call(reqs=40)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    assert len(lc2_calls) == 1


def test_own_t_does_not_take_the_calls_lc2_declines(monkeypatch):
    _no_stock_decode(monkeypatch)
    lc2_calls, own_t_calls = _own_t_route(monkeypatch, True)
    args, kwargs = _decode_call(dql=4)
    with pytest.raises(RuntimeError, match="decode_query_len 4"):
        lc2_decode.dep_lc2_decode(*args, **kwargs)
    assert lc2_calls == [] and own_t_calls == []


def test_own_t_reads_the_masked_lengths(monkeypatch):
    _, own_t_calls = _own_t_route(monkeypatch, True)
    masked = torch.zeros(40, dtype=torch.int32)
    monkeypatch.setattr(
        lc2_decode, "_PAD_ROWS", types.SimpleNamespace(lc2_seq_lens=lambda seq: masked)
    )
    args, kwargs = _decode_call(reqs=40)
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    assert own_t_calls[0][9] is masked


def _own_t_operands(reqs=40):
    args, kwargs = _decode_call(reqs=reqs)
    cuda_q = types.SimpleNamespace(is_cuda=True)
    return [cuda_q, kwargs["k_block_scale"], kwargs["v_block_scale"], args[5], kwargs["output"]]


def test_own_t_takes_the_dep_geometry_up_to_64_requests():
    assert lc2_decode.own_t_decline_reason(*_own_t_operands(reqs=56)) is None
    assert lc2_decode.own_t_decline_reason(*_own_t_operands(reqs=64)) is None
    assert "65 requests" in lc2_decode.own_t_decline_reason(*_own_t_operands(reqs=65))


def test_own_t_declines_a_cpu_call():
    args, kwargs = _decode_call(reqs=40)
    reason = lc2_decode.own_t_decline_reason(
        args[0], kwargs["k_block_scale"], kwargs["v_block_scale"], args[5], kwargs["output"]
    )
    assert reason == "not a CUDA call"


def test_own_t_declines_views_its_copies_would_misalign():
    q, ksc, vsc, seq, out = _own_t_operands()
    # The scales' page rows 8 bytes past a 16-byte boundary.
    shifted = torch.zeros(ksc.shape[0] * 2 * 4 * 128 * 8 + 8, dtype=torch.uint8)[8:].view(
        ksc.shape[0], 2, 4, 128, 8
    )
    assert "block scales" in lc2_decode.own_t_decline_reason(
        q, shifted[:, 0], shifted[:, 1], seq, out
    )
    # Output rows 2 bytes past an 8-byte boundary.
    rows = out.shape[0]
    odd = torch.zeros(rows * 64 * 128 + 1, dtype=torch.bfloat16)[1:].view(rows, 64, 128)
    assert "output rows" in lc2_decode.own_t_decline_reason(q, ksc, vsc, seq, odd)
    # Output rows 8 bytes past a 16-byte boundary: the main kernel's 16-byte stores would misalign.
    half = torch.zeros(rows * 64 * 128 + 4, dtype=torch.bfloat16)[4:].view(rows, 64, 128)
    assert "output rows" in lc2_decode.own_t_decline_reason(q, ksc, vsc, seq, half)


def test_the_own_t_switch_is_the_dep_ownt_item():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
        m31_switches,
    )

    assert lc2_decode.DEP_OWNT == (not m31_switches.m31_disabled("DEP_OWNT"))


@pytest.mark.parametrize("own_t, device", [(True, "cpu"), (False, "cuda")])
def test_own_t_reserves_nothing_off_cuda_or_when_off(monkeypatch, own_t, device):
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", own_t)
    monkeypatch.setattr(lc2_decode, "own_t_op", lambda: object())
    monkeypatch.setattr(
        lc2_decode,
        "native_ops",
        types.SimpleNamespace(
            minimax_m3_sm107_sparse_decode_own_t_prepare=lambda anchor, max_items: pytest.fail(
                "nothing to prepare"
            )
        ),
    )
    assert lc2_decode.reserve_own_t_workspace(torch.device(device)) == 0


@pytest.mark.parametrize(
    "rows, vocab, path",
    [
        (1, _VOCAB, "mc"),
        (19, _VOCAB, "mc"),
        (48, _VOCAB, "mc"),
        (128, _VOCAB, "mc"),
        (129, _VOCAB, "pathb"),
        (152, _VOCAB, "pathb"),
        (384, _VOCAB, "pathb"),
        (448, _VOCAB, "pathb"),
        (449, _VOCAB, None),
        (152, 151936, None),
        (8, 16384, None),
    ],
)
def test_softmax_routes(monkeypatch, rows, vocab, path):
    monkeypatch.setattr(softmax_mc, "PATHB", True)
    assert softmax_mc.route(rows, vocab) == path


def test_softmax_pathb_switch_keeps_wide_calls_on_flashinfer(monkeypatch):
    monkeypatch.setattr(softmax_mc, "PATHB", False)
    assert softmax_mc.route(384, _VOCAB) is None
    assert softmax_mc.route(48, _VOCAB) == "mc"


@pytest.mark.parametrize("rows, ctas", [(1, 16), (12, 16), (13, 8), (24, 8), (25, 4), (128, 4)])
def test_softmax_write_ctas(rows, ctas):
    assert softmax_mc.write_ctas(rows) == ctas


def test_softmax_temperature_contract():
    logits = torch.zeros(4, 8)
    assert softmax_mc._temperature_args(None, logits) == (None, 1.0)
    assert softmax_mc._temperature_args(0.7, logits) == (None, 0.7)
    temps = torch.ones(4)
    assert softmax_mc._temperature_args(temps, logits)[0] is temps
    assert softmax_mc._temperature_args(torch.ones(4, dtype=torch.float64), logits) is None
    assert softmax_mc._temperature_args(torch.ones(3), logits) is None
    assert softmax_mc._temperature_args(True, logits) is None


def test_softmax_leaves_cpu_and_off_contract_logits_to_flashinfer():
    assert softmax_mc.softmax_mc(torch.zeros(8, _VOCAB), None) is None
    assert softmax_mc.softmax_mc(torch.zeros(8, _VOCAB, dtype=torch.float16), None) is None


class _V3Op:
    """A stand-in for minimax_m31_topp_renorm_det_v3 with the given schema arguments, recording its launches."""

    def __init__(self, args):
        schema = ", ".join(args)
        self.default = types.SimpleNamespace(
            _schema=f"minimax_m31_topp_renorm_det_v3({schema}) -> ()"
        )
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


_V3_ARGS = (
    "Tensor probs",
    "Tensor top_p",
    "Tensor(a!) workspace",
    "Tensor(b!) out",
    "int ctas_per_row",
    "bool pdl",
    "bool candidate_list",
    "bool list_path=False",
    "bool wide_hist=False",
)


@pytest.fixture
def v3_op(monkeypatch):
    def install(args):
        op = _V3Op(args)
        monkeypatch.setattr(
            topp_det,
            "family_ops",
            lambda: types.SimpleNamespace(
                minimax_m31_topp_renorm_det_v3=op,
                minimax_m31_topp_renorm_det_v3_ws_bytes=lambda: 212992,
            ),
        )
        monkeypatch.setattr(topp_det, "_STATE", {})
        return op

    return install


def test_topp_det_needs_the_keep_all_ge1_switch(v3_op):
    v3_op(_V3_ARGS)
    assert topp_det.topp_det_op() is None
    assert "not taken" in topp_det.engagement() or not topp_det.TOPP_DET


def test_topp_det_needs_v4k8(v3_op):
    v3_op(_V3_ARGS[:7] + ("bool keep_all_ge1=True",))
    assert topp_det.topp_det_op() is None


def test_topp_det_launches_flashinfer_rows_on_top_p_1(v3_op):
    op = v3_op(_V3_ARGS + ("bool keep_all_ge1=True",))
    assert topp_det.topp_det_op() is op
    probs, top_p, ws = torch.zeros(48, 8), torch.ones(48), torch.zeros(1, dtype=torch.uint8)
    out = topp_det._launch(op, probs, top_p, ws, 2)
    ((call),) = op.calls
    assert call[0] is probs and call[1] is top_p and call[2] is ws and call[3] is out
    assert call[4:] == (2, True, True, True, True, False)


@pytest.mark.parametrize(
    "rows, sms, ctas", [(1, 160, 16), (8, 160, 16), (19, 160, 8), (48, 160, 2), (384, 160, 1)]
)
def test_topp_det_ctas_per_row(rows, sms, ctas):
    assert topp_det.ctas_per_row(rows, sms) == ctas


@pytest.mark.parametrize(
    "rows, plan",
    [
        (96, [(0, 96, 2)]),
        (106, [(0, 106, 2)]),
        (107, [(0, 106, 2), (106, 107, 16)]),
        (128, [(0, 106, 2), (106, 128, 8)]),
        (159, [(0, 106, 2), (106, 159, 4)]),
        (160, [(0, 160, 1)]),
        (212, [(0, 212, 1)]),
        (384, [(0, 384, 1)]),
    ],
)
def test_topp_split_plan_at_212_sms(rows, plan):
    assert topp_det.launch_plan(rows, 212) == plan
    assert topp_det.launch_plan(rows, 212, split=False) == [
        (0, rows, topp_det.ctas_per_row(rows, 212))
    ]
    assert all(1024 % ctas == 0 and (r1 - r0) * ctas <= 212 or ctas == 1 for r0, r1, ctas in plan)


def test_topp_split_launches_row_slices(v3_op):
    op = v3_op(_V3_ARGS + ("bool keep_all_ge1=True",))
    assert topp_det.topp_det_op() is op
    rows, wsb = 128, 16
    probs, top_p = torch.rand(rows, 8), torch.rand(rows)
    ws = torch.zeros(rows * wsb, dtype=torch.uint8)
    out = topp_det._launch_plan(op, probs, top_p, ws, topp_det.launch_plan(rows, 212), wsb)
    assert [c[4] for c in op.calls] == [2, 8]
    for (r0, r1), call in zip(((0, 106), (106, 128)), op.calls):
        assert call[0].data_ptr() == probs[r0].data_ptr() and call[0].shape[0] == r1 - r0
        assert call[1].data_ptr() == top_p[r0].data_ptr() and call[1].shape[0] == r1 - r0
        assert call[2].data_ptr() == ws[r0 * wsb].data_ptr() and call[2].numel() == (r1 - r0) * wsb
        assert call[3].data_ptr() == out[r0].data_ptr() and call[3].shape[0] == r1 - r0
        assert call[5:] == (True, True, True, True, False)


def test_topp_det_contract():
    probs, top_p = torch.zeros(48, 200064), torch.full((48,), 0.95)
    assert not topp_det.outside_contract(probs, top_p)
    assert topp_det.outside_contract(torch.zeros(449, 8), torch.ones(449))
    assert topp_det.outside_contract(probs, 0.95)
    assert topp_det.outside_contract(probs, top_p.double())
    assert topp_det.outside_contract(probs[:, :-1], top_p)
    assert topp_det.outside_contract(probs.half(), top_p)
    assert topp_det.top_p_renorm_det(probs, top_p) is None
