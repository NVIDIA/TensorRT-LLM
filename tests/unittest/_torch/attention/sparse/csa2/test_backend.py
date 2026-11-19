# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native, PrimTS and Flash provider numerics, dispatch and metadata lifecycle."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import pytest
import torch
from utils.util import skip_blackwell_geforce

from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (
    CSA2FlashInfer,
    CSA2FlashMLA,
    _fused_context_supported,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Params

from ._utils import _assert_fp8_close, _plan_indexer

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.models.modeling_utils import QuantConfig

skip_prims_ts_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="CSA2 PrimTS requires SM100, SM103 or SM107",
)
skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)
skip_no_fused_context = pytest.mark.skipif(
    not torch.cuda.is_available() or not _fused_context_supported(torch.cuda.current_device()),
    reason="The fused CSA2 context kernel requires an SM100-family GPU",
)


def _reference(
    q: torch.Tensor,
    swa: torch.Tensor,
    extra: torch.Tensor | None,
    swa_valid: torch.Tensor,
    extra_valid: torch.Tensor | None,
    sink: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    kv, valid = swa, swa_valid
    if extra is not None:
        kv = torch.cat((kv, extra), dim=1)
        valid = torch.cat((valid, extra_valid), dim=1)
    kv = torch.where(valid[..., None], kv, 0)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * scale
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    probs = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1).softmax(-1)[
        ..., :-1
    ]
    return torch.einsum("qhk,qkd->qhd", probs, kv.float()).to(q.dtype)


@pytest.mark.parametrize("shape", [(2, 512), (2, 1, 8, 512)])
@pytest.mark.cpu_only
def test_fp8_accuracy_accepts_small_error_and_zero_queries(shape: tuple[int, ...]) -> None:
    expected = torch.ones(shape, dtype=torch.bfloat16)
    expected[0].zero_()
    actual = expected.clone()
    actual[1].mul_(1.02)
    _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
@pytest.mark.parametrize("operand", ["actual", "expected"])
@pytest.mark.cpu_only
def test_fp8_accuracy_rejects_nonfinite(value: float, operand: str) -> None:
    actual = torch.ones(2, 512, dtype=torch.bfloat16)
    expected = actual.clone()
    (actual if operand == "actual" else expected)[0, 0] = value
    with pytest.raises(AssertionError, match="must be finite"):
        _assert_fp8_close(actual, expected)


@pytest.mark.parametrize(
    "failure,match",
    [
        ("gain", "per-query L2"),
        ("one_query", "per-query L2"),
        ("outlier", "FP8 pointwise"),
        ("zero_reference", "zero-reference"),
        ("tiny_nonzero", "zero-reference"),
    ],
)
@pytest.mark.cpu_only
def test_fp8_accuracy_rejects_numerical_regressions(failure: str, match: str) -> None:
    expected = torch.ones(16, 1, 512, dtype=torch.bfloat16)
    actual = expected.clone()
    if failure == "gain":
        actual.mul_(1.1)
    elif failure == "one_query":
        actual[0].mul_(1.1)
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.05
    elif failure == "outlier":
        actual[0, 0, 0] = 1.5
        assert (actual[0].float() - expected[0].float()).norm() / expected[0].float().norm() < 0.05
    else:
        expected[0].zero_()
        actual[0].fill_(1e-30 if failure == "tiny_nonzero" else 0.01)
    if failure != "outlier":
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.12, rtol=0.04)
    with pytest.raises(AssertionError, match=match):
        _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
@pytest.mark.cpu_only
def test_fp8_accuracy_rejects_metadata_mismatch(mismatch: str) -> None:
    actual = torch.ones(2, 512, dtype=torch.bfloat16)
    expected = actual.clone()
    if mismatch == "shape":
        actual = actual[:1]
    elif mismatch == "dtype":
        actual = actual.float()
    else:
        actual = actual.to("meta")
    with pytest.raises(AssertionError, match="mismatch"):
        _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("gain,accepted", [(1.046875, True), (1.0546875, False)])
@pytest.mark.cpu_only
def test_fp8_accuracy_relative_l2_budget(gain: float, accepted: bool) -> None:
    expected = torch.ones(2, 512, dtype=torch.bfloat16)
    actual = expected * gain
    if accepted:
        _assert_fp8_close(actual, expected)
    else:
        with pytest.raises(AssertionError, match="per-query L2"):
            _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("amplitude", [0.25, 1.0, 16.0, 128.0])
@pytest.mark.parametrize("outlier", [False, True])
@pytest.mark.cpu_only
def test_fp8_accuracy_pointwise_budget_scales_with_reference(
    amplitude: float, outlier: bool
) -> None:
    expected = torch.full((2, 512), amplitude, dtype=torch.bfloat16)
    expected[:, 0] = 0
    actual = expected.clone()
    actual[:, 0] = (0.2 if outlier else 0.1) * max(1.0, amplitude)
    relative_l2 = (actual.float() - expected.float()).norm(dim=1) / expected.float().norm(dim=1)
    assert (relative_l2 < 0.05).all()
    if outlier:
        with pytest.raises(AssertionError, match="FP8 pointwise"):
            _assert_fp8_close(actual, expected)
    else:
        _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("other_query_peak", [1.0, 128.0])
@pytest.mark.cpu_only
def test_fp8_accuracy_pointwise_scale_uses_each_reference_query(other_query_peak: float) -> None:
    expected = torch.ones(2, 512, dtype=torch.bfloat16)
    expected[1].mul_(other_query_peak)
    actual = expected.clone()
    actual[0, 0] = 1.171875
    assert (actual[0].float() - expected[0].float()).norm() / expected[0].float().norm() < 0.05
    with pytest.raises(AssertionError, match="FP8 pointwise"):
        _assert_fp8_close(actual, expected)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("staging_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("shared", [False, True])
def test_staging_keeps_minimum_tma_backing(staging_dtype, shared):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    q = torch.empty(1, 8, 512, dtype=torch.bfloat16, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(
        q,
        0,
        staging_dtype=staging_dtype,
        context_lengths=[1] if shared else None,
        shared_rows=75 if shared else None,
    )
    if shared:
        assert metadata.shared_pool.shape == (75, 512)
        assert metadata.swa_pool is metadata.extra_pool is metadata.shared_pool
    else:
        assert metadata.swa_pool.shape == (1, 128, 512)
        assert metadata.extra_pool.shape == (1, 1, 512)
    for pool in (metadata.swa_pool, metadata.extra_pool):
        remaining_storage = pool.untyped_storage().nbytes() - (
            pool.storage_offset() * pool.element_size()
        )
        assert remaining_storage >= 128 << 10


def _backend(
    heads: int,
    layer_idx: int = 20,
    layout: CSA2Layout | None = None,
    compute_backend: str = "trtllm",
    use_packed: bool = False,
    kv_cache_dtype: str = "auto",
    quant_config: QuantConfig | None = None,
    fp8_staging: bool | None = None,
    q_scaling: float = 1.0,
) -> CSA2TrtllmAttention:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import get_csa2_backend
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend
    from tensorrt_llm._utils import is_sm_100f

    if compute_backend in ("auto", "trtllm") and not is_sm_100f():
        pytest.skip("TRTLLM dynamic sparse MLA requires SM100-family")
    if compute_backend == "prims_ts" and torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("CSA2 PrimTS requires SM100, SM103 or SM107")
    params = CSA2Params(
        layout=layout,
        compute_backend=compute_backend,
        use_packed_sparse_attention=use_packed,
        # ``None`` leaves staging at its default, so the backend resolves it from
        # its own support the way a served request does.
        use_fp8_staging=fp8_staging,
    )
    assert get_attention_backend("TRTLLM", params) is get_csa2_backend(params)
    attn = create_attention(
        "TRTLLM",
        layer_idx,
        heads,
        512,
        num_kv_heads=1,
        is_mla_enable=True,
        q_lora_rank=1280,
        kv_lora_rank=448,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        rope_append=False,
        sparse_params=params,
        kv_cache_dtype=kv_cache_dtype,
        quant_config=quant_config,
        q_scaling=q_scaling,
    )
    assert isinstance(attn, TrtllmAttention)
    return attn


def _helper_forward(
    attn: CSA2TrtllmAttention,
    q: torch.Tensor,
    metadata: CSA2TrtllmMetadata,
    args: AttentionForwardArgs,
) -> torch.Tensor:
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import CSA2PrimsTS

    if attn.sparse_params.use_packed_sparse_attention:
        return attn.forward_packed(q, metadata, args)
    if not hasattr(attn, "_test_compute_helper"):
        helper = {
            "flash_mla": CSA2FlashMLA,
            "flashinfer": CSA2FlashInfer,
            "prims_ts": CSA2PrimsTS,
        }[attn.compute_backend]
        attn._test_compute_helper = helper(attn)
    compute = (
        attn._test_compute_helper.forward_context
        if args.attention_input_type == AttentionInputType.context_only
        else attn._test_compute_helper.forward_generation
    )
    return compute(q, metadata, args)


def _inputs(q, swa, extra, swa_valid, extra_valid, sink, *, swa_cache_format="swa"):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    swa_ids = torch.arange(swa.shape[0] * swa.shape[1], device=q.device).reshape_as(swa_valid)
    main_ids = (
        None
        if extra is None
        else torch.arange(extra.shape[0] * extra.shape[1], device=q.device).reshape_as(extra_valid)
    )
    inputs = CSA2BackendForwardArgs(
        swa_cache_format=swa_cache_format,
        swa_pool=pack_rows(swa.flatten(0, 1), swa_cache_format),
        swa_indices=torch.where(swa_valid, swa_ids, -1),
        main_pool=None if extra is None else pack_rows(extra.flatten(0, 1), "main"),
        topk_indices=None if extra is None else torch.where(extra_valid, main_ids, -1),
    )
    return AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        attention_sinks=sink,
        sparse_backend_args=inputs,
    )


def _decoded_reference(
    q: torch.Tensor,
    args: AttentionForwardArgs,
    swa_valid: torch.Tensor,
    extra_valid: torch.Tensor | None,
    sink: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    inputs = args.sparse_backend_args
    swa = unpack_rows(inputs.swa_pool, 512, inputs.swa_cache_format or "swa").reshape(
        q.shape[0], -1, 512
    )
    main = (
        None
        if inputs.main_pool is None
        else unpack_rows(inputs.main_pool, 512, "main").reshape(q.shape[0], -1, 512)
    )
    return _reference(q, swa, main, swa_valid, extra_valid, sink, scale=scale)


def _cache_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _cache_tensors(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _cache_tensors(item)]
    return []


def _forbid_prims_ts_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel, metadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention

    def unexpected_staging(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "PrimTS must not allocate/materialize decoded KV or use custom packed attention"
        )

    monkeypatch.setattr(metadata, "_tma_backed_empty", unexpected_staging)
    monkeypatch.setattr(metadata.CSA2TrtllmMetadata, "stage_selected", unexpected_staging)
    monkeypatch.setattr(CSA2TrtllmAttention, "prepare_selected", unexpected_staging)
    monkeypatch.setattr(CSA2TrtllmAttention, "forward_packed", unexpected_staging)
    monkeypatch.setattr(kernel, "packed_sparse_attention", unexpected_staging)


def _prims_ts_query_count(query_band: str) -> int:
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    lower = (sms + 3) // 4
    return {"below": max(1, lower - 1), "inside": lower, "above": (sms + 1) // 2}[query_band]


@skip_prims_ts_unsupported
@pytest.mark.parametrize("query_band", ["below", "inside", "above"])
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
@torch.inference_mode()
def test_prims_ts_independent_generation_and_output(
    query_band: str, kv_dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    def unexpected_native(*args: object, **kwargs: object) -> None:
        pytest.fail("PrimTS compute reached native attention")

    monkeypatch.setattr(FallbackFmha, "forward", unexpected_native)
    _forbid_prims_ts_staging(monkeypatch)
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_packed import PackedSparseMLA

    pools = []
    original_run = PackedSparseMLA.run

    def record_raw_pools(self, query, swa_cache, *args, **kwargs):
        pools.append((swa_cache, kwargs["extra_kv_cache"]))
        return original_run(self, query, swa_cache, *args, **kwargs)

    monkeypatch.setattr(PackedSparseMLA, "run", record_raw_pools)
    torch.manual_seed(490)
    count, heads, q_scaling = _prims_ts_query_count(query_band), 64, 1.7
    attn = _backend(heads, compute_backend="prims_ts", kv_cache_dtype=kv_dtype, q_scaling=q_scaling)
    # Generation tiles stay packed; only context tiles with a shared bank of the fp8 KV cache
    # take the fused kernel.
    assert attn.staging_dtype == torch.bfloat16 and attn.kv_input_mode == "packed"
    assert attn.fused_context == (
        kv_dtype == "fp8" and _fused_context_supported(torch.cuda.current_device())
    )
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(count, 512, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(count, 512, device="cuda", dtype=torch.bool)
    swa_valid[:, 1::11] = False
    main_valid[:, 3::17] = False
    swa_valid[count // 2] = main_valid[count // 2] = False
    sink = torch.linspace(-1.75, 2.25, heads, device="cuda")
    args = _inputs(
        q, swa, main, swa_valid, main_valid, sink, swa_cache_format=attn.swa_cache_format
    )
    guarded = torch.full((count + 2, heads * 512), 23, device=q.device, dtype=q.dtype)
    args.output = guarded[1:-1]
    args.output.fill_(torch.nan)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 512, kv_input_mode="packed")
    assert metadata.num_sparse_topk == 640 and metadata.shared_plan is None
    assert not any(hasattr(metadata, name) for name in ("swa_pool", "extra_pool", "shared_pool"))
    actual = _helper_forward(attn, q.flatten(1), metadata, args)
    assert len(pools) == 1
    assert pools[0][0] is args.sparse_backend_args.swa_pool
    assert pools[0][1] is args.sparse_backend_args.main_pool
    assert actual.data_ptr() == args.output.data_ptr()
    expected = _decoded_reference(
        q, args, swa_valid, main_valid, sink, scale=1.0 / (q_scaling * math.sqrt(512))
    )
    torch.testing.assert_close(actual.view_as(q), expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(
        actual.view_as(q)[count // 2], torch.zeros_like(q[0]), atol=0, rtol=0
    )
    assert bool((guarded[0] == 23).all()) and bool((guarded[-1] == 23).all())


@skip_prims_ts_unsupported
@pytest.mark.parametrize("has_main", [True, False])
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
@torch.inference_mode()
def test_prims_ts_early_selection_matches_serial(
    kv_dtype: str, has_main: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_metadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import CSA2PrimsTS

    _forbid_prims_ts_staging(monkeypatch)
    # Only compacted plans prepare metadata; raw-selection plans have none to move.
    monkeypatch.setattr(CSA2PrimsTS, "raw_selections", False)
    streams = []
    original = _prims_ts_metadata.prepare_prims_ts_metadata

    def record_stream(*args, **kwargs):
        streams.append(torch.cuda.current_stream())
        return original(*args, **kwargs)

    monkeypatch.setattr(_prims_ts_metadata, "prepare_prims_ts_metadata", record_stream)
    torch.manual_seed(497)
    count, heads = 3, 16
    attn = _backend(heads, compute_backend="prims_ts", kv_cache_dtype=kv_dtype)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    swa_valid[:, 1::7] = False
    main = main_valid = None
    if has_main:
        main = torch.randn(count, 512, 512, device="cuda", dtype=torch.bfloat16)
        main_valid = torch.ones(count, 512, device="cuda", dtype=torch.bool)
        main_valid[:, 2::13] = False
    sink = torch.linspace(-1.0, 1.0, heads, device="cuda")
    args = _inputs(
        q, swa, main, swa_valid, main_valid, sink, swa_cache_format=attn.swa_cache_format
    )
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 512 if has_main else 0, kv_input_mode="packed")
    serial = _helper_forward(attn, q.flatten(1), metadata, args).clone()
    assert streams == [torch.cuda.current_stream()]
    ready = torch.cuda.Event()
    ready.record()
    early_args = replace(
        args, sparse_backend_args=replace(args.sparse_backend_args, selection_ready=ready)
    )
    early = _helper_forward(attn, q.flatten(1), metadata, early_args)
    torch.testing.assert_close(early, serial, atol=0, rtol=0)
    # Final selections are compacted on the side stream for both SWA formats.
    assert len(streams) == 2
    assert streams[1] != torch.cuda.current_stream()


@skip_prims_ts_unsupported
@pytest.mark.parametrize("early", [False, True], ids=["serial", "early-selection"])
@torch.inference_mode()
def test_prims_ts_selection_bank_claims_in_compaction(
    early: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compaction claims a deduplicated selection bank's owner slots, on either stream.

    The bank decode launches no claim kernel of its own, and the attention matches the
    selection bank without deduplication bit for bit. With final selections, both
    decodes run on the compaction's side stream. The native NVFP4 MAIN kernel, which
    needs no bank, is held off.
    """
    from dataclasses import replace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_bank, _prims_ts_packed
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _forbid_prims_ts_staging(monkeypatch)
    monkeypatch.setattr(_prims_ts_packed, "decodes_nvfp4_main", lambda *args, **kwargs: False)
    decodes = []
    dedup, plain_decode = _prims_ts_bank.dedup_main_records, _prims_ts_bank.decode_main_records

    def record_dedup(*args, **kwargs):
        decodes.append(("dedup", kwargs.get("claimed", False), torch.cuda.current_stream()))
        return dedup(*args, **kwargs)

    def record_plain(*args, **kwargs):
        decodes.append(("plain", False, torch.cuda.current_stream()))
        return plain_decode(*args, **kwargs)

    def no_claim_kernel():
        pytest.fail("the compaction claims the owner slots")

    monkeypatch.setattr(_prims_ts_bank, "dedup_main_records", record_dedup)
    monkeypatch.setattr(_prims_ts_bank, "decode_main_records", record_plain)
    monkeypatch.setattr(_prims_ts_bank, "_claim_kernel", no_claim_kernel)
    torch.manual_seed(503)
    count, heads = 192, 64
    attn = _backend(heads, compute_backend="prims_ts", kv_cache_dtype="fp8")
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(count, 512, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(count, 512, device="cuda", dtype=torch.bool)
    swa_valid[:, 1::7] = False
    main_valid[:, 2::13] = False
    sink = torch.linspace(-1.0, 1.0, heads, device="cuda")
    args = _inputs(
        q, swa, main, swa_valid, main_valid, sink, swa_cache_format=attn.swa_cache_format
    )
    if early:
        ready = torch.cuda.Event()
        ready.record()
        args = replace(
            args, sparse_backend_args=replace(args.sparse_backend_args, selection_ready=ready)
        )
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 512, kv_input_mode="packed")
    model_stream = torch.cuda.current_stream()
    deduplicated = _helper_forward(attn, q.flatten(1), metadata, args).clone()
    attn._test_compute_helper.dedup_selection_bank = False
    plain = _helper_forward(attn, q.flatten(1), metadata, args)
    torch.testing.assert_close(deduplicated, plain, atol=0, rtol=0)
    assert [(kind, claimed) for kind, claimed, _ in decodes] == [("dedup", True), ("plain", False)]
    assert all((stream != model_stream) == early for _, _, stream in decodes)


@skip_prims_ts_unsupported
@pytest.mark.parametrize(
    ("early", "band"),
    [(False, "below"), (True, "below"), (True, "limit")],
    ids=["serial", "early-selection", "early-selection-limit"],
)
@torch.inference_mode()
def test_prims_ts_generation_decodes_nvfp4_main_in_kernel(
    early: bool, band: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 64-head generation tile over E4M3 SWA rows reads its NVFP4 MAIN records in the kernel.

    No selection bank is decoded, except for a tile of fewer than
    ``_EARLY_SELECTION_BANK_ROWS`` rows whose selections are final before its projections,
    which keeps it. Either way the attention matches the selection bank's bit for bit.
    """
    from dataclasses import replace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import (
        _prims_ts_bank,
        _prims_ts_packed,
        prims_ts,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    _forbid_prims_ts_staging(monkeypatch)
    decodes = []
    for name in ("dedup_main_records", "decode_main_records"):
        original = getattr(_prims_ts_bank, name)

        def record(*args, _original=original, _name=name, **kwargs):
            decodes.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(_prims_ts_bank, name, record)
    torch.manual_seed(509)
    count = {"below": 192, "limit": prims_ts._EARLY_SELECTION_BANK_ROWS}[band]
    heads = 64
    attn = _backend(heads, compute_backend="prims_ts", kv_cache_dtype="fp8")
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(count, 512, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(count, 512, device="cuda", dtype=torch.bool)
    swa_valid[:, 1::7] = False
    main_valid[:, 2::13] = False
    sink = torch.linspace(-1.0, 1.0, heads, device="cuda")
    args = _inputs(
        q, swa, main, swa_valid, main_valid, sink, swa_cache_format=attn.swa_cache_format
    )
    if early:
        ready = torch.cuda.Event()
        ready.record()
        args = replace(
            args, sparse_backend_args=replace(args.sparse_backend_args, selection_ready=ready)
        )
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 512, kv_input_mode="packed")
    routed = _helper_forward(attn, q.flatten(1), metadata, args).clone()
    keeps_bank = early and band == "below"
    assert decodes == (["dedup_main_records"] if keeps_bank else [])
    monkeypatch.setattr(_prims_ts_packed, "decodes_nvfp4_main", lambda *args, **kwargs: False)
    banked = _helper_forward(attn, q.flatten(1), metadata, args)
    assert decodes == ["dedup_main_records"] * (2 if keeps_bank else 1)
    torch.testing.assert_close(routed, banked, atol=0, rtol=0)


@skip_prims_ts_unsupported
@pytest.mark.parametrize("route", ["nvfp4", "bank", "hybrid"])
@torch.inference_mode()
def test_prims_ts_group_layers_share_main_compaction(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layers of one staging group compact their MAIN selections and decode their bank once.

    Three layers read one MAIN pool through the same selections, each with its own SWA
    rows. After the group's first compaction (and selection bank decode, where the layers
    read one), the others of the forward compact only their SWA rows, on either stream,
    and match their own full compaction bit for bit. ``nvfp4``: every tile decodes NVFP4
    MAIN in the kernel, the later two from final selections on the side stream. ``bank``:
    every tile reads one selection bank on the model stream, also replayed from a CUDA
    Graph over changed inputs. ``hybrid``: the index source decodes NVFP4 MAIN and its
    Reuse layers keep the selection bank, so the first of them still decodes it. A new
    forward, or another group's compaction in between, compacts again.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import (
        _prims_ts_bank,
        _prims_ts_metadata,
        _prims_ts_packed,
        prims_ts,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    _forbid_prims_ts_staging(monkeypatch)
    compactions, decodes = [], []
    compact = _prims_ts_metadata.prepare_prims_ts_metadata

    def record_compaction(*args, **kwargs):
        compactions.append(kwargs.get("swa_only", False))
        return compact(*args, **kwargs)

    monkeypatch.setattr(_prims_ts_metadata, "prepare_prims_ts_metadata", record_compaction)
    for name in ("dedup_main_records", "decode_main_records"):
        original = getattr(_prims_ts_bank, name)

        def record_decode(*args, _original=original, **kwargs):
            decodes.append(torch.cuda.current_stream())
            return _original(*args, **kwargs)

        monkeypatch.setattr(_prims_ts_bank, name, record_decode)
    if route == "nvfp4":
        monkeypatch.setattr(prims_ts, "_EARLY_SELECTION_BANK_ROWS", 1)
    elif route == "bank":
        monkeypatch.setattr(_prims_ts_packed, "decodes_nvfp4_main", lambda *args, **kwargs: False)
    torch.manual_seed(521)
    count, heads, layers = 192, 64, 3
    attn = _backend(heads, compute_backend="prims_ts", kv_cache_dtype="fp8")
    swa_format = attn.swa_cache_format
    sink = torch.linspace(-1.0, 1.0, heads, device="cuda")
    main_ids = torch.arange(count * 512, device="cuda").view(count, 512)

    def main_selections(seed: int) -> torch.Tensor:
        valid = torch.ones(count, 512, device="cuda", dtype=torch.bool)
        valid[:, seed % 13 :: 13] = False
        return torch.where(valid, main_ids, -1)

    main_pool = pack_rows(torch.randn(count * 512, 512, device="cuda").bfloat16(), "main")
    topk = main_selections(2)
    queries = [torch.randn(count, heads * 512, device="cuda").bfloat16() for _ in range(layers)]
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    layer_args = []
    for layer in range(layers):
        swa_valid[:, layer + 1 :: 7] = False
        swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
        args = _inputs(
            queries[layer].view(count, heads, 512),
            swa,
            None,
            swa_valid.clone(),
            None,
            sink,
            swa_cache_format=swa_format,
        )
        args.sparse_backend_args = replace(
            args.sparse_backend_args, main_pool=main_pool, topk_indices=topk
        )
        args.output = torch.empty(count, heads * 512, device="cuda", dtype=torch.bfloat16)
        layer_args.append(args)
    # Later layers of the nvfp4 and hybrid groups have final selections before their projections.
    early = [False] + [route != "bank"] * (layers - 1)
    source = SimpleNamespace(csa2_indices={}, _csa2_stage_epoch=1)
    group = [(1, 1, 0)]
    prepare = attn.prepare_sparse_inputs

    def grouped_inputs(q, metadata, forward_args):
        # The module resolves each layer's staging group and forward state.
        return replace(
            prepare(q, metadata, forward_args),
            stage_group=group[0],
            state=None if group[0] is None else SimpleNamespace(metadata=source),
        )

    monkeypatch.setattr(attn, "prepare_sparse_inputs", grouped_inputs)
    metadata = CSA2TrtllmMetadata.for_query_tile(
        queries[0].view(count, heads, 512), 512, kv_input_mode="packed"
    )

    def run(layer: int) -> torch.Tensor:
        args = layer_args[layer]
        if early[layer]:
            ready = torch.cuda.Event()
            ready.record()
            args = replace(
                args, sparse_backend_args=replace(args.sparse_backend_args, selection_ready=ready)
            )
        return _helper_forward(attn, queries[layer], metadata, args)

    def references() -> list[torch.Tensor]:
        group[0] = None
        try:
            return [run(layer).clone() for layer in range(layers)]
        finally:
            group[0] = (1, 1, 0)

    def new_forward() -> None:
        source.csa2_indices = {}
        source._csa2_stage_epoch += 1

    expected = references()
    compactions.clear()
    decodes.clear()
    for layer in range(layers):
        torch.testing.assert_close(run(layer), expected[layer], atol=0, rtol=0)
    first_reuse = 2 if route == "hybrid" else 1
    assert compactions == [layer >= first_reuse for layer in range(layers)]
    if route == "nvfp4":
        assert decodes == []
    else:
        # One decode per group: on the model stream, or beside a Reuse layer's projections.
        assert len(decodes) == 1
        assert (decodes[0] == torch.cuda.current_stream()) == (route == "bank")

    # A new forward compacts again; so does a layer after another group's compaction.
    new_forward()
    compactions.clear()
    torch.testing.assert_close(run(layers - 1), expected[-1], atol=0, rtol=0)
    group[0] = (2, 1, 0)
    run(0)
    group[0] = (1, 1, 0)
    torch.testing.assert_close(run(layers - 1), expected[-1], atol=0, rtol=0)
    assert compactions == [False, False, False]

    if route != "bank":
        return
    # Graph replay follows live selections, pool rows and SWA rows of every layer.
    new_forward()
    compactions.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [run(layer) for layer in range(layers)]
    assert compactions == [False, True, True]
    topk.copy_(main_selections(5))
    main_pool.copy_(pack_rows(torch.randn(count * 512, 512, device="cuda").bfloat16(), "main"))
    for args in layer_args:
        rows = torch.randn(count * 128, 512, device="cuda").bfloat16()
        args.sparse_backend_args.swa_pool.copy_(pack_rows(rows, swa_format))
    for output in captured:
        output.fill_(torch.nan)
    graph.replay()
    replayed = [output.clone() for output in captured]
    for actual, reference in zip(replayed, references()):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@skip_prims_ts_unsupported
@pytest.mark.parametrize("holes", [False, True], ids=["dense", "holes"])
@pytest.mark.parametrize("has_main", [True, False])
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
@pytest.mark.parametrize("heads,count", [(8, 6), (64, 12)])
@torch.inference_mode()
def test_prims_ts_raw_selections_match_compacted(
    heads: int,
    count: int,
    kv_dtype: str,
    has_main: bool,
    holes: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generation tile reads its published selections without compacting them first.

    INT64 SWA slots and INT32 logical MAIN positions resolve in the kernel exactly as
    ``prepare_prims_ts_metadata`` resolves them: dense selections give the compacted plan's
    output bit for bit, and holes (-1, out-of-pool slots, positions past the visible span,
    the position limit or the table, unowned requests and unbacked pages) are masked. A tile
    that a native kernel (NVFP4 MAIN or TMA gather) serves, or of more than 32 heads, keeps
    its compacted lists.
    """
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import (
        _prims_ts_metadata,
        _prims_ts_packed,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import (
        CSA2PrimsTS,
        _reads_raw_selections,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    _forbid_prims_ts_staging(monkeypatch)
    compactions, mappings = [], []
    prepare = _prims_ts_metadata.prepare_prims_ts_metadata
    run = _prims_ts_packed.PackedSparseMLA.run

    def record_compaction(*args, **kwargs):
        compactions.append(True)
        return prepare(*args, **kwargs)

    def record_run(self, *args, **kwargs):
        mappings.append(kwargs.get("main_mapping"))
        return run(self, *args, **kwargs)

    monkeypatch.setattr(_prims_ts_metadata, "prepare_prims_ts_metadata", record_compaction)
    monkeypatch.setattr(_prims_ts_packed.PackedSparseMLA, "run", record_run)
    g = torch.Generator(device="cuda").manual_seed(4930 + heads + count)
    attn = _backend(heads, compute_backend="prims_ts", kv_cache_dtype=kv_dtype)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16, generator=g)
    swa_rows = 3 * count + 160
    swa_pool = pack_rows(
        torch.randn(swa_rows, 512, device="cuda", generator=g).bfloat16(), attn.swa_cache_format
    )
    swa = torch.stack(
        [torch.randperm(swa_rows, device="cuda", generator=g)[:128] for _ in range(count)]
    )
    main_pool = mapping = logical = None
    if has_main:
        # Two requests own shuffled 16-row pages of a paged MAIN cache; with holes, the pool
        # stops one page short, so the table's last physical page is unbacked.
        page, pages = 16, 40
        table = torch.randperm(2 * pages, device="cuda", generator=g).view(2, pages).int()
        pool_rows = 2 * pages * page - (page if holes else 0)
        main_storage = torch.full((pool_rows, 320), 7, device="cuda", dtype=torch.uint8)
        main_pool = main_storage[:, :288]
        main_pool.copy_(
            pack_rows(
                torch.randn(main_pool.shape[0], 512, device="cuda", generator=g).bfloat16(), "main"
            )
        )
        requests = (torch.arange(count, device="cuda") % 2).long()
        visible = torch.full((count,), pages * page, device="cuda", dtype=torch.int64)
        logical = torch.stack(
            [torch.randperm(pages * page, device="cuda", generator=g)[:512] for _ in range(count)]
        ).int()
        mapping = (table, page, 600 if holes else pages * page, requests, visible)
        if holes:
            visible[1] = 300
            requests[2] = 2
            logical[:, 5::23] = -1
            logical[3, :9] = pages * page + 5
            logical[count // 2] = -1
    if holes:
        swa[:, 2::19] = -1
        swa[0, :4] = swa_rows + torch.arange(4, device="cuda")
        swa[count // 2] = -1
    sink = torch.linspace(-1.25, 1.5, heads, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 512 if has_main else 0, kv_input_mode="packed")

    def forward(raw: bool) -> torch.Tensor:
        monkeypatch.setattr(CSA2PrimsTS, "raw_selections", raw)
        args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            attention_sinks=sink,
            output=torch.full_like(q, torch.nan).flatten(1),
            sparse_backend_args=CSA2BackendForwardArgs(
                swa_cache_format=attn.swa_cache_format,
                swa_pool=swa_pool,
                swa_indices=swa.contiguous(),
                main_pool=main_pool,
                topk_indices=logical,
                main_logical_indices=logical,
                main_mapping=mapping,
            ),
        )
        return _helper_forward(attn, q.flatten(1), metadata, args).view_as(q).clone()

    compacted = forward(False)
    assert compactions == [True] and mappings == [None]
    raw = forward(True)
    if not _reads_raw_selections(
        q.device, count, heads, metadata.num_sparse_topk, attn.swa_cache_format, "main"
    ):
        assert compactions == [True, True] and mappings == [None, None]
        torch.testing.assert_close(raw, compacted, atol=0, rtol=0)
        return
    assert compactions == [True], "raw selections must not compact"
    assert len(mappings) == 2 and (mappings[1] is not None) == has_main
    if not holes:
        torch.testing.assert_close(raw, compacted, atol=0, rtol=0)
        return
    # Holes move rows across tiles, so the online softmax rounds differently.
    torch.testing.assert_close(raw, compacted, atol=2e-2, rtol=2e-2)
    # A row without a valid key attends only the sink.
    torch.testing.assert_close(raw[count // 2], torch.zeros_like(q[0]), atol=0, rtol=0)
    swa_valid = (swa >= 0) & (swa < swa_rows)
    keys = unpack_rows(swa_pool, 512, attn.swa_cache_format)[swa.clamp(0, swa_rows - 1)]
    main_keys = main_valid = None
    if has_main:
        table, page, limit, requests, visible = mapping
        owned = (requests >= 0) & (requests < table.shape[0])
        span = torch.minimum(visible.clamp(max=limit), torch.tensor(table.shape[1] * page))
        span = torch.where(owned, span, 0)
        inside = (logical >= 0) & (logical < span[:, None])
        position = torch.where(inside, logical, 0).long()
        entry = table[requests.clamp(0, table.shape[0] - 1)[:, None], position // page]
        row = entry.long() * page + position % page
        main_valid = inside & (entry >= 0) & (row < main_pool.shape[0])
        main_keys = unpack_rows(main_pool, 512, "main")[torch.where(main_valid, row, 0)]
    expected = _reference(q, keys, main_keys, swa_valid, main_valid, sink)
    _assert_fp8_close(raw, expected)


@skip_prims_ts_unsupported
@pytest.mark.parametrize("has_main", [True, False], ids=["main", "swa-only"])
@pytest.mark.parametrize("heads,count", [(8, 6), (16, 12)])
@torch.inference_mode()
def test_prims_ts_runahead_matches_ordered(
    heads: int, count: int, has_main: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw-selection kernel that runs ahead of its preceding launch attends as an ordered one.

    It rotates its BF16 Q before the wait, then reads SWA rows after it; its MAIN selections
    and rows either before it (``main_ready``) or after it. Where it waits changes no
    arithmetic, so both MAIN orders give the same bits, and both match a kernel that reads
    a Q the projection rotated.
    """
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import CSA2PrimsTS
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import gptj_rope_small_m
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    _forbid_prims_ts_staging(monkeypatch)
    monkeypatch.setattr(CSA2PrimsTS, "raw_selections", True)
    g = torch.Generator(device="cuda").manual_seed(5110 + heads + count)
    attn = _backend(heads, compute_backend="prims_ts", kv_cache_dtype="fp8")
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16, generator=g)
    swa_rows = 3 * count + 160
    swa_pool = pack_rows(
        torch.randn(swa_rows, 512, device="cuda", generator=g).bfloat16(), attn.swa_cache_format
    )
    swa = torch.stack(
        [torch.randperm(swa_rows, device="cuda", generator=g)[:128] for _ in range(count)]
    )
    main_pool = logical = mapping = None
    if has_main:
        page, pages = 16, 40
        table = torch.randperm(pages, device="cuda", generator=g).view(1, pages).int()
        main_pool = pack_rows(
            torch.randn(pages * page, 512, device="cuda", generator=g).bfloat16(), "main"
        )
        logical = torch.stack(
            [torch.randperm(pages * page, device="cuda", generator=g)[:512] for _ in range(count)]
        ).int()
        mapping = (
            table,
            page,
            pages * page,
            torch.zeros(count, device="cuda", dtype=torch.int64),
            torch.full((count,), pages * page, device="cuda", dtype=torch.int64),
        )
    positions = torch.randint(0, 4096, (count,), device="cuda", generator=g, dtype=torch.int32)
    angle = torch.rand(4096, 32, device="cuda", generator=g) * 6.2831853
    cos_sin = torch.cat((angle.cos(), angle.sin()), 1).view(4096, 2, 32).contiguous()
    sink = torch.linspace(-1.25, 1.5, heads, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 512 if has_main else 0, kv_input_mode="packed")

    def forward(kernel_rope: bool, inputs_final: bool = False, main_ready: bool = True):
        monkeypatch.setattr(CSA2PrimsTS, "_inputs_final", lambda self, raw, rope: inputs_final)
        monkeypatch.setattr(CSA2PrimsTS, "_main_ready", lambda self: main_ready)
        query = q.clone()
        if not kernel_rope:
            gptj_rope_small_m(query, positions, cos_sin)
        args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            attention_sinks=sink,
            output=torch.full_like(q, torch.nan).flatten(1),
            sparse_backend_args=CSA2BackendForwardArgs(
                swa_cache_format=attn.swa_cache_format,
                swa_pool=swa_pool,
                swa_indices=swa.contiguous(),
                main_pool=main_pool,
                topk_indices=logical,
                main_logical_indices=logical,
                main_mapping=mapping,
                query_rope=(positions, cos_sin) if kernel_rope else None,
            ),
        )
        return _helper_forward(attn, query.flatten(1), metadata, args).view_as(q).clone()

    rotated = forward(False)
    runahead = forward(True, inputs_final=True)
    torch.testing.assert_close(runahead, rotated, atol=2e-2, rtol=2e-2)
    if has_main:
        main_after_wait = forward(True, inputs_final=True, main_ready=False)
        torch.testing.assert_close(main_after_wait, runahead, atol=0, rtol=0)


@skip_prims_ts_unsupported
@pytest.mark.parametrize(
    "main_bank", ["", "all", "selected"], ids=["in-kernel-decode", "main-bank", "selected-rows"]
)
@pytest.mark.parametrize("query_band", ["below", "inside", "above"])
@torch.inference_mode()
def test_prims_ts_packed_context_and_output(
    query_band: str, main_bank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import (
        CSA2PrimsTSMainBank,
        CSA2TrtllmMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        gather_rows,
        pack_rows,
    )

    _forbid_prims_ts_staging(monkeypatch)
    torch.manual_seed(491)
    count, heads, q_scaling = _prims_ts_query_count(query_band), 64, 1.3
    attn = _backend(heads, compute_backend="prims_ts", q_scaling=q_scaling)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(639, 639 + count, device="cuda", dtype=torch.int64)
    swa_rows = count + 127
    page_count = (swa_rows + 127) // 128
    swa_pages = torch.full((1, 4 + page_count), -1, device="cuda", dtype=torch.int32)
    swa_pages[0, 4:] = torch.arange(page_count - 1, -1, -1, device="cuda", dtype=torch.int32)
    main_pages = torch.tensor([[2, 0, 3, 1]], device="cuda", dtype=torch.int32)
    swa = pack_rows(torch.randn(page_count * 128, 512, device="cuda", dtype=torch.bfloat16), "swa")
    main_storage = torch.full((512, 320), 73, device="cuda", dtype=torch.uint8)
    main = main_storage[:, :288]
    main.copy_(pack_rows(torch.randn(512, 512, device="cuda", dtype=torch.bfloat16), "main"))
    swa_logical = positions[:, None] - 127 + torch.arange(128, device="cuda")
    swa_slots = swa_pages[0, swa_logical // 128].long() * 128 + swa_logical.remainder(128)
    main_logical = torch.arange(512, device="cuda").expand(count, -1).clone()
    main_logical[:, 7] = main_logical[:, 6]
    main_slots = main_pages[0, main_logical // 128].long() * 128 + main_logical.remainder(128)
    swa_slots[:, 1::11] = -1
    main_slots[:, 3::17] = main_logical[:, 3::17] = -1
    swa_slots[count // 2] = main_slots[count // 2] = main_logical[count // 2] = -1
    metadata = CSA2TrtllmMetadata.for_query_tile(
        q, 512, context_lengths=[count], kv_input_mode="packed"
    )
    sink = torch.linspace(-2.25, 1.75, heads, device="cuda")
    args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.context_only,
        attention_sinks=sink,
        output=torch.full_like(q, torch.nan).flatten(1),
        sparse_backend_args=CSA2BackendForwardArgs(
            swa_pool=swa,
            swa_indices=swa_slots,
            main_pool=main,
            topk_indices=main_logical,
            main_logical_indices=main_logical,
            main_mapping=(
                main_pages,
                128,
                512,
                torch.zeros(count, device="cuda", dtype=torch.int64),
                torch.full((count,), 512, device="cuda", dtype=torch.int64),
            ),
        ),
    )
    in_kernel = None
    if main_bank:
        # The same phase decoding every selection inside the kernel.
        in_kernel = _helper_forward(attn, q.flatten(1), metadata, args).clone()
        args.output = torch.full_like(q, torch.nan).flatten(1)
        # One request owns the four MAIN pages; its bank rows follow a nonzero base.
        metadata.prims_ts_main_bank = CSA2PrimsTSMainBank(
            offsets=torch.tensor([9, 9 + 512], device="cuda", dtype=torch.int64),
            base=9,
            rows=512,
            pages=main_pages,
            page_size=128,
            num_requests=1,
            selected_only=main_bank == "selected",
        )
    if main_bank == "selected":
        from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
            decode_main_bank,
            main_bank_marks,
            main_bank_storage,
        )
        from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import _ARENA_CACHE_KEY

        # Poison the retained bank, so a row the phase reads but does not decode
        # changes the output.
        arenas = metadata.fmha_plan_caches.setdefault(_ARENA_CACHE_KEY, {})
        stored = main_bank_storage(arenas, q.device, 512)
        stored.fill_(0x5A)
        full = torch.empty_like(stored)
        decode_main_bank(main, metadata.prims_ts_main_bank, full)
    actual = _helper_forward(attn, q.flatten(1), metadata, args)
    if main_bank == "selected":
        # Logical rows 7 and 3 + 17 k are never selected; the others decode as a full bank.
        chosen = torch.zeros(512, device="cuda", dtype=torch.bool)
        chosen[main_logical[main_logical >= 0].unique()] = True
        assert not chosen[7] and not chosen[3::17].any() and chosen.sum() > 400
        torch.testing.assert_close(stored[chosen], full[chosen], atol=0, rtol=0)
        assert bool((stored[~chosen] == 0x5A).all())
        assert torch.count_nonzero(main_bank_marks(arenas, q.device, 512)) == 0
    assert actual.data_ptr() == args.output.data_ptr()
    assert metadata.shared_plan is None
    formats = {key[5] for key in metadata.fmha_plan_caches["csa2_prims_ts"]}
    assert formats == ({"main", "main_fp8"} if main_bank else {"main"})
    if in_kernel is not None:
        # Bank rows are the operands the kernel would decode, so attention is identical.
        torch.testing.assert_close(actual, in_kernel, atol=0, rtol=0, equal_nan=True)
    assert not any(hasattr(metadata, name) for name in ("swa_pool", "extra_pool", "shared_pool"))
    expected = _reference(
        q,
        gather_rows(swa, swa_slots, 512, "swa"),
        gather_rows(main, main_slots, 512, "main"),
        swa_slots >= 0,
        main_slots >= 0,
        sink,
        scale=1.0 / (q_scaling * math.sqrt(512)),
    )
    torch.testing.assert_close(actual.view_as(q), expected, atol=0.03, rtol=0.03)
    torch.testing.assert_close(
        actual.view_as(q)[count // 2], torch.zeros_like(q[0]), atol=0, rtol=0
    )
    assert bool((main_storage[:, 288:] == 73).all())


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("extra_width", [0, 17, 512])
@torch.inference_mode()
def test_native_dual_pool(heads, extra_width, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

    calls = []
    original = FallbackFmha.forward

    def record(attn, *args, **kwargs):
        calls.append(attn.attn.sparse_params.algorithm)
        return original(attn, *args, **kwargs)

    monkeypatch.setattr(FallbackFmha, "forward", record)
    torch.manual_seed(451)
    attn = _backend(heads)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(3, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(3, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 2:] = False
    swa_valid[1] = False
    extra = (
        torch.randn(3, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(3, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[0, 1::2] = False  # holes before later valid entries
        extra_valid[1] = False
    sink = torch.randn(heads, device="cuda")
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    metadata = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    out = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    torch.cuda.synchronize()
    assert calls == ["csa2"]
    torch.testing.assert_close(
        out, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_trtllm_graph_replay_resets_sparse_state():
    attn = _backend(64)
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    extra_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    sink = torch.zeros(64, device="cuda")

    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    frame = CSA2TrtllmMetadata.for_query_tile(q, 17)

    def run():
        frame.is_cuda_graph = torch.cuda.is_current_stream_capturing()
        args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
        return attn.forward(q.flatten(1), None, None, frame, forward_args=args).view_as(q)

    for _ in range(3):
        run()
    pointers = frame.pool_pointers.clone()
    workspace_ptr = frame.workspace.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    # Larger eager queries use independent caller-owned metadata. No recursive
    # warmup or backend-private frame allocation is needed.
    large_q = q.repeat(8, 1, 1)
    large_frame = CSA2TrtllmMetadata.for_query_tile(large_q, 17)
    large_args = _inputs(
        large_q,
        swa.repeat(8, 1, 1),
        extra.repeat(8, 1, 1),
        swa_valid.repeat(8, 1),
        extra_valid.repeat(8, 1),
        sink,
    )
    attn.forward(large_q.flatten(1), None, None, large_frame, forward_args=large_args)
    for width in (17, 0, 3, 17):
        extra_valid.zero_()
        extra_valid[:, :width] = True
        swa_valid[0, 1:] = width != 0
        q.mul_(-1)
        extra.mul_(-1)
        graph.replay()
        # Quantize/dequantize independently of the prediction hook's gathering.
        from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

        swa_ref = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
        extra_ref = unpack_rows(pack_rows(extra, "main"), 512, "main")
        torch.testing.assert_close(
            output,
            _reference(q, swa_ref, extra_ref, swa_valid, extra_valid, sink),
            atol=0.03,
            rtol=0.03,
        )
        torch.testing.assert_close(frame.pool_pointers, pointers, atol=0, rtol=0)
        assert frame.workspace.data_ptr() == workspace_ptr
        assert large_frame.workspace.data_ptr() != workspace_ptr


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_standard_backend_multiple_tiles_and_reuse(monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
        CSA2CacheManager,
        CSA2CacheRole,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
        CSA2BackendForwardArgs,
        CSA2ForwardState,
        CSA2Layout,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    predictions = []
    original_predict = CSA2Indexer.sparse_attn_indexer

    def predict(indexer, metadata, hidden_states, *args, **kwargs):
        predictions.append((indexer.layer_idx, hidden_states.shape[0]))
        return original_predict(indexer, metadata, hidden_states, *args, **kwargs)

    monkeypatch.setattr(CSA2Indexer, "sparse_attn_indexer", predict)
    torch.manual_seed(419)
    layout = CSA2Layout((1, 1, 1), (0,), (0, 2), index_topk=4, window_size=4)
    count, heads = 19, 8
    manager = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, host_cache_size=0, dtype="fp8_ds_mla"),
        CacheType.SELFKONLY,
        num_layers=3,
        tokens_per_block=128,
        max_seq_len=64,
        max_batch_size=1,
        max_num_tokens=count,
        mapping=Mapping(),
        vocab_size=8192,
        layout=layout,
    )
    try:
        request = manager._create_kv_cache(100, None, [])
        assert manager._resume_and_restore(100, request)
        assert request.resize(count)
        runtime = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=count, kv_cache_manager=manager
        )
        positions = torch.arange(count, device="cuda")
        q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
        swa = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
        main = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
        index_k = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
        index_q = torch.randn(count, 2, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.ones(count, 2, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(heads, device="cuda")
        runtime.reset_routing()
        global_base = manager.get_cache_indices(100, 0, CSA2CacheRole.GLOBAL)[0] * 128
        global_slots = global_base + torch.arange(6, device="cuda")
        runtime.csa2_token_requests = torch.zeros(count, dtype=torch.int64, device="cuda")
        runtime.csa2_request_query_ranges = ((0, count),)
        runtime.csa2_request_start_positions = (0,)
        runtime.csa2_num_context_requests = 1
        runtime.csa2_global_page_tables = {
            0: torch.tensor([[global_base // 128]], dtype=torch.int32, device="cuda")
        }
        runtime.csa2_global_page_sizes = {0: 128}
        runtime.csa2_global_max_positions = {0: 6}
        runtime.csa2_main_write_slots = {0: global_slots}
        _plan_indexer(runtime)
        runtime.csa2_kv_sources = {0: 0, 1: 0, 2: 0}
        runtime.csa2_swa_indices = {}
        runtime.csa2_swa_write_slots = {}
        runtime.csa2_visible_lengths = {}
        for layer in range(3):
            swa_base = manager.get_cache_indices(100, layer, CSA2CacheRole.SWA)[0] * 128
            local = positions[:, None] - torch.arange(4, device="cuda")[None, :]
            swa_slots = torch.where(local >= 0, local + swa_base, -1)
            runtime.csa2_swa_indices[layer] = swa_slots
            runtime.csa2_swa_write_slots[layer] = positions + swa_base
            runtime.csa2_visible_lengths[layer] = positions.remainder(6) + 1
            args_dict = {}
            if layer != 1:
                args_dict.update(index_q=index_q * (-1 if layer else 1), index_weights=weights)
            if layer == 0:
                args_dict.update(main_kv=main, index_k=index_k)
            state = CSA2ForwardState(metadata=runtime, swa_kv=swa, **args_dict)
            backend = _backend(heads, layer, layout)
            outputs = []
            for start in range(0, count, 16):
                tile = q[start : start + 16]
                frame = runtime.get_query_tile_metadata(tile, layout.index_topk)
                args = AttentionForwardArgs(
                    attention_input_type=AttentionInputType.generation_only,
                    attention_sinks=sink,
                    sparse_backend_args=CSA2BackendForwardArgs(state=state, query_start=start),
                )
                outputs.append(
                    backend.forward(tile.flatten(1), None, None, frame, forward_args=args).view_as(
                        tile
                    )
                )
            actual = torch.cat(outputs)
            logical = runtime.csa2_indices[0 if layer == 1 else layer]
            slots = runtime.global_slot_tile(layer, 0, count, logical)
            swa_values = gather_rows(manager.get_swa_buffer(layer), swa_slots, 512, "swa")
            main_values = gather_rows(manager.get_main_buffer(0), slots, 512, "main")
            expected = _reference(q, swa_values, main_values, swa_slots >= 0, slots >= 0, sink)
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
            frame = runtime.get_query_tile_metadata(q[-3:], 4)
            assert frame.host_total_kv_lens.tolist() == [0, 3 * 256]
        assert predictions == [(0, count), (2, count)]
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shared_metadata_resets_global_to_swa_inputs():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    sink = torch.zeros(8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17)
    first, second = _backend(8, 20), _backend(8, 21)
    args = _inputs(q, swa, main, swa_valid, main_valid, sink)
    first.forward(q.flatten(1), None, None, metadata, forward_args=args)
    assert args.sparse_runtime_params.aux_kv_cache_pool_ptr is not None
    # Reuse the runtime carrier and metadata across distinct layers, but remove
    # main selection. No stale main pointer or sparse length may survive.
    args.sparse_backend_args = _inputs(q, -swa, None, swa_valid, None, sink).sparse_backend_args
    args.output = None
    output = second.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    assert args.sparse_runtime_params.aux_kv_cache_pool_ptr is None
    torch.testing.assert_close(metadata.prepared_lens, torch.full_like(metadata.prepared_lens, 4))
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, None, sink), atol=0.03, rtol=0.03
    )


@skip_prims_ts_unsupported
@pytest.mark.parametrize("heads,extra_width", [(1, 0), (8, 17), (32, 0), (128, 17)])
@torch.inference_mode()
def test_prims_ts_packed_source_edges(
    heads: int, extra_width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        gather_rows,
        pack_rows,
    )

    _forbid_prims_ts_staging(monkeypatch)
    torch.manual_seed(494)
    attn = _backend(heads, compute_backend="prims_ts")
    q = torch.randn(4, heads, 512, device="cuda", dtype=torch.bfloat16)
    channel_scale = (2.0 ** torch.arange(-2, 2, device="cuda")).repeat_interleave(128)
    swa = (torch.randn(4, 8, 512, device="cuda") * channel_scale).bfloat16()
    main = (
        (torch.randn(4, extra_width, 512, device="cuda") * channel_scale.flip(0)).bfloat16()
        if extra_width
        else None
    )
    swa_valid = torch.ones(4, 8, device="cuda", dtype=torch.bool)
    swa_valid[1] = swa_valid[3] = False
    swa_valid[0, 1::2] = False
    main_valid = (
        torch.ones(4, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if main_valid is not None:
        main_valid[1] = main_valid[2] = False
        main_valid[:, -1] = False
    sink = torch.linspace(-2, 2, heads, device="cuda")
    args = _inputs(q, swa, main, swa_valid, main_valid, sink)
    inputs = args.sparse_backend_args
    # Odd outer strides exercise aligned chunks and the byte-load fallback.
    for name, stride in (("swa_pool", 609), ("main_pool", 321)):
        pool = getattr(inputs, name)
        if pool is not None:
            padded = torch.empty((pool.shape[0], stride), dtype=pool.dtype, device=pool.device)
            view = padded[:, : pool.shape[1]]
            view.copy_(pool)
            setattr(inputs, name, view)
    # Preserve duplicate source rows and reject large IDs before INT32 narrowing.
    inputs.swa_indices[0, 2] = 0
    inputs.swa_indices[0, 1] = 2**32
    if extra_width:
        inputs.topk_indices[:, -1] = 2**32 + 3
        inputs.topk_indices[0, 2] = inputs.topk_indices[0, 0]
    metadata = CSA2TrtllmMetadata.for_query_tile(q, extra_width, kv_input_mode="packed")

    def reference():
        return _reference(
            q,
            gather_rows(inputs.swa_pool, inputs.swa_indices, 512, "swa"),
            gather_rows(inputs.main_pool, inputs.topk_indices, 512, "main")
            if inputs.main_pool is not None
            else None,
            swa_valid,
            main_valid if inputs.main_pool is not None else None,
            sink,
        )

    output = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    _assert_fp8_close(output, reference())
    torch.testing.assert_close(output[1], torch.zeros_like(q[1]), atol=0, rtol=0)
    cache = metadata.fmha_plan_caches["csa2_prims_ts"]
    pointers = tuple(tensor.data_ptr() for tensor in _cache_tensors(cache))
    # Reusable plans must bind the next owner's raw pool, not retain the first.
    inputs.swa_pool = pack_rows((swa * 0.5).flatten(0, 1), "swa")
    inputs.main_pool = inputs.topk_indices = None
    output = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    _assert_fp8_close(output, reference())
    assert tuple(tensor.data_ptr() for tensor in _cache_tensors(cache)) == pointers
    # A genuinely empty packed source is safe even with stale positive IDs.
    inputs.swa_pool = inputs.swa_pool[:0]
    output = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    torch.testing.assert_close(output, torch.zeros_like(q), atol=0, rtol=0)


@skip_prims_ts_unsupported
@pytest.mark.parametrize("rows", [256, 65535, 65536])
@torch.inference_mode()
def test_prims_ts_packed_launch_boundaries(rows: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_packed import (
        PackedSparseMLA,
        get_packed_sparse_mla_workspace_size,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    _forbid_prims_ts_staging(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    size = get_packed_sparse_mla_workspace_size(device, rows, 1, max_extra_topk=17)
    workspace = torch.empty(size, dtype=torch.uint8, device=device)
    wrapper = PackedSparseMLA(workspace)
    wrapper.plan(device, rows, 1, max_extra_topk=17, has_sinks=True)
    q = torch.zeros((rows, 1, 1, 512), dtype=torch.bfloat16, device=device)
    output = torch.full_like(q, torch.nan)
    swa = pack_rows(torch.ones((1, 512), dtype=q.dtype, device=device), "swa")
    main = pack_rows(torch.ones((1, 512), dtype=q.dtype, device=device), "main")
    indices = torch.full((rows, 128), -1, dtype=torch.int32, device=device)
    extra_indices = torch.full((rows, 17), -1, dtype=torch.int32, device=device)
    indices[:, 0] = extra_indices[:, 0] = 0
    lengths = torch.ones(rows, dtype=torch.int32, device=device)
    extra_lengths = torch.ones_like(lengths)
    lengths[1::4] = lengths[3::4] = 0
    extra_lengths[2::4] = extra_lengths[3::4] = 0
    sinks = torch.zeros((1,), dtype=torch.float32, device=device)

    def launch(*, validate=False, scale=2e38):
        return wrapper.run(
            q,
            swa,
            indices,
            lengths,
            extra_kv_cache=main,
            extra_sparse_indices=extra_indices,
            extra_sparse_lens=extra_lengths,
            softmax_scale=scale,
            output_scale=1.0,
            sinks=sinks,
            out=output,
            validate=validate,
        )

    def reference():
        numerator = lengths[:, None].float() * unpack_rows(swa, 512, "swa").float()
        numerator += extra_lengths[:, None].float() * unpack_rows(main, 512, "main").float()
        denominator = lengths.float() + extra_lengths.float() + sinks.exp()
        return (numerator / denominator[:, None]).to(q.dtype).view_as(q)

    launch(validate=True)
    torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        launch()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    lengths.copy_(lengths.roll(1))
    extra_lengths.bitwise_xor_(1)
    swa.copy_(pack_rows(torch.full((1, 512), -2.0, dtype=q.dtype, device=device), "swa"))
    sinks.fill_(0.5)
    graph.replay()
    torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)
    assert workspace.numel() == size
    # A finite FP32 input can still overflow its native base-2 conversion.
    with pytest.raises(ValueError, match="FP32 base-2 conversion"):
        launch(scale=3e38)


@skip_prims_ts_unsupported
@pytest.mark.parametrize("gather", [False, True])
@torch.inference_mode()
def test_prims_ts_packed_precompile_covers_every_row_count(
    gather: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    compiled = []
    gathered = []
    monkeypatch.setattr(packed, "_compile_packed", lambda spec: compiled.append(spec) or spec)
    monkeypatch.setattr(
        packed, "_get_compiled_mla_decode", lambda spec: gathered.append(spec) or spec
    )
    monkeypatch.setattr(packed.UnitScaleQuery, "plan", lambda *_, **__: None)
    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    geometry = dict(max_extra_topk=512, swa_cache_format="swa_fp8")
    if gather:
        geometry.update(main_cache_format="main_fp8", valid_prefix=True)
    # Enough rows to step a 640-wide MAIN plan through several split-KV counts
    # and, with 64 heads, through grids beyond one wave.
    heads, max_rows = (64, sm_count + 8) if gather else (16, 96)
    kernels = packed.precompile_packed_sparse_mla(device, max_rows, heads, **geometry)
    precompiled, pregathered = set(compiled), set(gathered)
    assert kernels == len(compiled) + len(gathered) == len(precompiled) + len(pregathered)
    assert len({spec.split_kv for spec in precompiled | pregathered}) > 1
    assert bool(pregathered) == gather
    for rows in range(1, max_rows + 1):
        # The arena size depends on neither the formats nor the prefix promise.
        size = packed.get_packed_sparse_mla_workspace_size(
            device, rows, heads, max_extra_topk=512, swa_cache_format="swa_fp8"
        )
        wrapper = packed.PackedSparseMLA(torch.empty(size, dtype=torch.uint8, device=device))
        wrapper.plan(device, rows, heads, **geometry)
        state = wrapper._state
        if state.gather is None:
            assert state.spec in precompiled, rows
        else:
            assert state.gather.compiled in pregathered and state.compiled is None, rows
    with pytest.raises(ValueError, match="max_rows"):
        packed.precompile_packed_sparse_mla(device, 0, heads, **geometry)


@skip_prims_ts_unsupported
@torch.inference_mode()
def test_prims_ts_precompiles_once_per_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import CSA2PrimsTS

    calls = []
    monkeypatch.setattr(
        packed,
        "precompile_packed_sparse_mla",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 1,
    )
    q = torch.zeros((2, 3, 512), device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode="packed")
    metadata._csa2_prims_ts_max_queries = 6
    helper = CSA2PrimsTS(_backend(3, compute_backend="prims_ts"))
    formats = [("swa", "main"), ("swa", "main"), ("swa_fp8", "main")]
    # A bank plan precompiles its MAIN format once more and shares the arena.
    formats += [("swa_fp8", "main_fp8"), ("swa_fp8", "main_fp8")]
    records = [
        helper._get_plan(q.flatten(1), metadata, swa_cache_format, main_cache_format)
        for swa_cache_format, main_cache_format in formats
    ]
    assert calls == [
        (
            (q.device, 6, 3),
            dict(
                max_topk=128,
                max_extra_topk=0,
                swa_cache_format=swa_cache_format,
                main_cache_format=main_cache_format,
                valid_prefix=True,
                output_groups=None,
                raw_selections=False,
                query_rope=False,
                inputs_final=False,
                main_ready=True,
            ),
        )
        for swa_cache_format, main_cache_format in (
            ("swa", "main"),
            ("swa_fp8", "main"),
            ("swa_fp8", "main_fp8"),
        )
    ]
    assert records[3] is not records[2] and records[3]["arena"] is records[2]["arena"]


@skip_prims_ts_unsupported
@torch.inference_mode()
def test_prims_ts_plans_ignore_main_ready_without_runahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import CSA2PrimsTS

    calls = []
    monkeypatch.setattr(
        packed,
        "precompile_packed_sparse_mla",
        lambda *args, **kwargs: calls.append((kwargs["inputs_final"], kwargs["main_ready"])) or 1,
    )
    monkeypatch.setattr(packed, "_compile_packed", lambda spec: spec)
    monkeypatch.setattr(packed, "_get_compiled_mla_decode", lambda spec: spec)
    monkeypatch.setattr(packed.UnitScaleQuery, "plan", lambda *_, **__: None)
    q = torch.zeros((2, 3, 512), device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode="packed")
    metadata._csa2_prims_ts_max_queries = 6
    helper = CSA2PrimsTS(_backend(3, compute_backend="prims_ts"))

    def plan(inputs_final: bool, main_ready: bool) -> dict:
        return helper._get_plan(
            q.flatten(1),
            metadata,
            "swa_fp8",
            "main",
            inputs_final=inputs_final,
            main_ready=main_ready,
        )

    # Only a kernel that runs ahead of its preceding launch reads MAIN before the wait.
    assert plan(False, False) is plan(False, True)
    assert plan(True, False) is not plan(True, True)
    assert calls == [(False, True), (True, False), (True, True)]


class _Forgetful(dict):
    """A memo that keeps nothing, so every lookup builds afresh."""

    def __setitem__(self, key, value) -> None:
        pass


@skip_prims_ts_unsupported
@pytest.mark.parametrize(
    "heads,geometry",
    [
        # Context tiles: a decoded MAIN bank and a SWA-only layer storing the grouped MXFP8
        # output, and NVFP4 MAIN records.
        (
            64,
            dict(
                max_extra_topk=512,
                swa_cache_format="swa_fp8",
                main_cache_format="main_fp8",
                valid_prefix=True,
                output_groups=8,
            ),
        ),
        (64, dict(max_extra_topk=512, swa_cache_format="swa_fp8", valid_prefix=True)),
        (
            64,
            dict(max_extra_topk=0, swa_cache_format="swa_fp8", valid_prefix=True, output_groups=8),
        ),
        # Split-KV producer plans, and runahead raw-selection plans that rotate Q.
        (16, dict(max_extra_topk=512)),
        (
            16,
            dict(
                max_extra_topk=512,
                swa_cache_format="swa_fp8",
                raw_selections=True,
                query_rope=True,
                inputs_final=True,
                main_ready=False,
            ),
        ),
    ],
)
@torch.inference_mode()
def test_prims_ts_launches_share_kernels_across_row_counts(
    heads: int, geometry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    capacity = packed._validate_geometry(1, heads, 128, geometry["max_extra_topk"])
    row_counts = [*range(1, sm_count + 9), 1024, 4095, 4096, 16383, 32768]
    if geometry.get("output_groups") is not None:
        # Only single-split, single-V-CTA plans store the MXFP8 output.
        row_counts = [
            rows
            for rows in row_counts
            if packed._quantizes_output(
                packed._select_packed_profile(rows, heads, capacity, sm_count)
            )
        ]
    shared = [packed._resolve_launch(device, rows, heads, 128, **geometry) for rows in row_counts]
    monkeypatch.setattr(packed, "_LAUNCH_KERNELS", _Forgetful())
    fresh = [packed._resolve_launch(device, rows, heads, 128, **geometry) for rows in row_counts]
    for rows, a, b in zip(row_counts, shared, fresh):
        # A plan of shared kernels compiles, sizes and binds what one of its own would.
        assert a.profile == b.profile, rows
        assert a.signature == b.signature == b.kernel.compile_signature(), rows
        assert (a.core_bytes, a.workspace_bytes, a.sections) == (
            b.core_bytes,
            b.workspace_bytes,
            b.sections,
        ), rows
        for x, y in ((a.gather, b.gather), (a.nvfp4, b.nvfp4)):
            assert (x is None) == (y is None), rows
            if x is not None:
                assert (x.profile, x.spec, x.core_bytes) == (y.profile, y.spec, y.core_bytes), rows
    assert len({id(launch.kernel) for launch in shared}) < len(row_counts)


def _packed_mla_reference(
    q, swa, indices, lengths, main, extra_indices, extra_lengths, scale, output_scale, sinks
):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    primary = unpack_rows(swa, 512, "swa").float()
    extra = unpack_rows(main, 512, "main").float() if main is not None else None
    result = torch.zeros_like(q)
    for row in range(q.shape[0]):
        selected_banks = []
        for bank, bank_indices, bank_lengths in (
            (primary, indices, lengths),
            (extra, extra_indices, extra_lengths),
        ):
            if bank is not None and bank_indices is not None:
                selected = bank_indices[row, : int(bank_lengths[row])].long()
                selected = selected[(selected >= 0) & (selected < bank.shape[0])]
                selected_banks.append(bank[selected])
        kv = torch.cat(selected_banks, dim=0)
        scores = q[row, 0].float() @ kv.T * scale
        if sinks is not None:
            logits = torch.cat((scores, sinks[:, None]), dim=1)
            weights = logits.softmax(-1)[:, : kv.shape[0]]
        else:
            weights = scores.softmax(-1)
        result[row, 0] = (weights @ kv * output_scale).bfloat16()
    return result


@skip_prims_ts_unsupported
@pytest.mark.parametrize(
    "heads,rows,topk,extra_topk,replay_iteration,has_main,has_sinks",
    [
        (16, 5, 137, 19, 2, True, True),
        (64, 64, 128, 129, 2, True, True),
        (128, 3, 129, 257, 2, True, True),
        (32, 2, 11, 0, 2, False, False),
        (16, 2, 128, 3, 2, False, True),
    ],
)
@torch.inference_mode()
def test_prims_ts_packed_amplified_graph_reference(
    heads: int,
    rows: int,
    topk: int,
    extra_topk: int,
    replay_iteration: int,
    has_main: bool,
    has_sinks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_packed import (
        PackedSparseMLA,
        get_packed_sparse_mla_workspace_size,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    _forbid_prims_ts_staging(monkeypatch)
    torch.manual_seed(47)
    device = torch.device("cuda", torch.cuda.current_device())
    swa_values = torch.randn(16, 512, device=device, dtype=torch.bfloat16) * torch.exp2(
        torch.arange(16, device=device).repeat_interleave(32) % 3 - 1
    )
    swa = pack_rows(swa_values, "swa")
    main_values = torch.randn(19, 512, device=device, dtype=torch.bfloat16) * torch.exp2(
        torch.arange(32, device=device).repeat_interleave(16) % 4 - 2
    )
    main_backing = torch.empty((19, 320), device=device, dtype=torch.uint8)
    main = main_backing[:, :288]
    main.copy_(pack_rows(main_values, "main"))
    indices = torch.randint(0, 16, (rows, topk), device=device, dtype=torch.int32)
    indices[:, 0], indices[:, 2] = 0, -1
    lengths = torch.full((rows,), topk, device=device, dtype=torch.int32)
    extra_indices = (
        torch.randint(0, 19, (rows, extra_topk), device=device, dtype=torch.int32)
        if extra_topk
        else None
    )
    extra_lengths = (
        torch.full((rows,), extra_topk, device=device, dtype=torch.int32) if extra_topk else None
    )
    if extra_topk:
        extra_indices[:, 0], extra_indices[:, 2] = 0, -1
    if not has_main:
        main = None
        if extra_topk:
            extra_lengths.zero_()
            extra_indices.fill_(-1)
    sinks = (
        torch.linspace(-1.5, 1.5, heads, device=device, dtype=torch.float32) if has_sinks else None
    )
    q = torch.randn(rows, 1, heads, 512, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(q)
    # The captured graph replays these host scales. The amplified replays raise
    # SWA logits eightfold, so the softmax scale keeps them within the FP8
    # reference tolerance that E4M3 Q.K rounding allows.
    scale, output_scale = 0.012, 1.2
    size = get_packed_sparse_mla_workspace_size(
        device, rows, heads, max_topk=topk, max_extra_topk=extra_topk
    )
    workspace = torch.empty(size, device=device, dtype=torch.uint8)
    wrapper = PackedSparseMLA(workspace)
    wrapper.plan(device, rows, heads, max_topk=topk, max_extra_topk=extra_topk, has_sinks=has_sinks)

    def launch():
        wrapper.run(
            q,
            swa,
            indices,
            lengths,
            extra_kv_cache=main,
            extra_sparse_indices=extra_indices,
            extra_sparse_lens=extra_lengths,
            softmax_scale=scale,
            output_scale=output_scale,
            sinks=sinks,
            out=output,
        )

    def reference():
        return _packed_mla_reference(
            q, swa, indices, lengths, main, extra_indices, extra_lengths, scale, output_scale, sinks
        )

    launch()
    _assert_fp8_close(output, reference())
    assert wrapper.workspace_size_bytes == size
    pointers = tuple(tensor.data_ptr() for tensor in wrapper.workspace_tensors)
    assert all(
        tensor.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
        for tensor in wrapper.workspace_tensors
    )
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    launch()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.max_memory_allocated() == allocated
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for iteration in range(replay_iteration + 1):
        if sinks is not None:
            sinks.add_(0.4)
        q.normal_()
        swa[:, 512:].add_(1)
        swa[:, :512].bitwise_xor_(128)
        if main is not None:
            main[:, :256].bitwise_xor_(1)
            main[:, 256:].copy_(pack_rows(main_values * (iteration + 1), "main")[:, 256:])
        indices[:, 0] = (iteration + 1) % 16
        lengths[1] = 0
        if extra_topk:
            extra_indices[:, 0] = (iteration + 2) % 19
            extra_lengths[0] = min(extra_topk, iteration + 1)
        output.fill_(torch.nan)
        graph.replay()
    torch.cuda.synchronize()
    captured, expected = output.clone(), reference()
    launch()
    torch.testing.assert_close(output, captured, rtol=0, atol=0)
    lengths.zero_()
    if extra_topk:
        extra_lengths.zero_()
    graph.replay()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert tuple(tensor.data_ptr() for tensor in wrapper.workspace_tensors) == pointers
    _assert_fp8_close(captured, expected)


@skip_prims_ts_unsupported
@torch.inference_mode()
def test_prims_ts_scale_overflow_rejects_source_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8, compute_backend="prims_ts", q_scaling=1.0 / (3e38 * math.sqrt(512)))

    def unexpected_source_preparation(*args, **kwargs):
        pytest.fail("Invalid scale must fail before cache writes or indexer execution")

    monkeypatch.setattr(attn, "prepare_sparse_inputs", unexpected_source_preparation)
    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.ones(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(8, device="cuda"))
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode="packed")
    before = metadata.get_workspace_bytes()
    with pytest.raises(ValueError, match="FP32 base-2 conversion"):
        _helper_forward(attn, q.flatten(1), metadata, args)
    assert metadata.get_workspace_bytes() == before
    assert "csa2_prims_ts_arenas" not in metadata.fmha_plan_caches


@skip_prims_ts_unsupported
@torch.inference_mode()
def test_prims_ts_cold_plan_rejects_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(64, compute_backend="prims_ts")

    def unexpected_source_preparation(*args, **kwargs):
        pytest.fail("Cold capture must fail before cache writes or indexer execution")

    monkeypatch.setattr(attn, "prepare_sparse_inputs", unexpected_source_preparation)
    q = torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(2, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(64, device="cuda"))
    args.output = torch.empty_like(q).flatten(1)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode="packed")
    before = metadata.get_workspace_bytes()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(
        RuntimeError, match="Warm up the CSA2 PrimTS profile before CUDA Graph capture"
    ):
        with torch.cuda.graph(graph):
            _helper_forward(attn, q.flatten(1), metadata, args)
    assert not metadata.fmha_plan_caches["csa2_prims_ts"]
    assert "csa2_prims_ts_arenas" not in metadata.fmha_plan_caches
    assert metadata.get_workspace_bytes() == before


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cold_metadata_rejects_capture():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8)
    q = torch.zeros(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(2, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, 1, device="cuda", dtype=torch.bool)
    sink = torch.zeros(8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    args = _inputs(q, swa, None, valid, None, sink)
    metadata.is_cuda_graph = True
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="Warm up CSA2 metadata"):
        with torch.cuda.graph(graph):
            attn.forward(q.flatten(1), None, None, metadata, forward_args=args)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation",
    [pytest.param("flash_mla", marks=skip_blackwell_geforce), "flashinfer", "prims_ts"],
)
@pytest.mark.parametrize("context", [False, True])
@torch.inference_mode()
def test_alternative_helper_dispatch(
    implementation: str, context: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Selected CSA2 implementation fell through to native fallback")

    monkeypatch.setattr(FallbackFmha, "forward", unexpected_fallback)
    attn = _backend(8, compute_backend=implementation)
    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    main_valid[0, ::2] = False
    swa_valid[1] = False
    main_valid[1] = False
    sink = torch.linspace(-2, 2, 8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(
        q, 17, kv_input_mode=attn.kv_input_mode, context_lengths=[q.shape[0]] if context else None
    )
    args = _inputs(q, swa, main, swa_valid, main_valid, sink)
    if context:
        from tensorrt_llm._torch.attention.backends.interface import AttentionInputType

        args.attention_input_type = AttentionInputType.context_only
    args.output = torch.empty_like(q).flatten(1)
    actual = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    assert actual.data_ptr() == args.output.data_ptr()
    expected = _decoded_reference(q, args, swa_valid, main_valid, sink)
    if implementation == "prims_ts":
        _assert_fp8_close(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["flashinfer", "prims_ts"])
@pytest.mark.parametrize(
    "option", ["custom_mask", "out_scale", "output_sf", "output_shape", "output_dtype"]
)
@torch.inference_mode()
def test_alternative_backend_rejects_unsupported_options(option: str, implementation: str) -> None:
    from tensorrt_llm._torch.attention.backends.interface import CustomAttentionMask
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8, compute_backend=implementation)
    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(8, device="cuda"))
    if option == "custom_mask":
        args.attention_mask = CustomAttentionMask.CUSTOM
    elif option == "out_scale":
        args.out_scale = torch.ones(1, device="cuda")
    elif option == "output_shape":
        args.output = torch.empty(2, q.shape[1] * 512, device=q.device, dtype=q.dtype)
    elif option == "output_dtype":
        args.output = torch.empty_like(q, dtype=torch.float32).flatten(1)
    else:
        args.output = torch.empty_like(q).flatten(1)
        args.output_sf = torch.empty(1, device="cuda", dtype=torch.uint8)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode=attn.kv_input_mode)
    with pytest.raises(ValueError, match="(do not support.*mask/output format|output must match)"):
        _helper_forward(attn, q.flatten(1), metadata, args)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["flashinfer", "prims_ts"])
@torch.inference_mode()
def test_alternative_helper_revalidates_output_scale(implementation: str) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8, compute_backend=implementation)
    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(8, device="cuda"))
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode=attn.kv_input_mode)
    _helper_forward(attn, q.flatten(1), metadata, args)
    torch.cuda.synchronize()
    args.out_scale = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="do not support.*mask/output format"):
        _helper_forward(attn, q.flatten(1), metadata, args)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("extra_width", [0, 17, 512])
@pytest.mark.parametrize("prefix", [0, 1024])
@pytest.mark.parametrize("tile_start", [0, 2])
@torch.inference_mode()
def test_native_context_preserves_real_query_groups(extra_width, prefix, tile_start, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(459)
    heads, count = 64, 5
    attn = _backend(heads)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 1::2] = False
    swa_valid[2] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[1, ::3] = False
        extra_valid[2] = False
    sink = torch.randn(heads, device="cuda")
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count + tile_start)
    source._num_contexts = 2
    source._num_ctx_tokens = source._num_tokens = count + tile_start
    source.csa2_num_context_requests = 2
    source.csa2_request_query_ranges = ((0, 3 + tile_start), (3 + tile_start, count + tile_start))
    source.csa2_request_start_positions = (prefix, prefix * 2)
    source.csa2_request_lengths = (3 + tile_start, 2)
    metadata = source.get_query_tile_metadata(q, extra_width, query_start=tile_start)
    assert metadata.num_contexts == 2
    assert metadata.num_ctx_tokens == count
    assert metadata.num_generations == 0
    torch.testing.assert_close(
        metadata.cu_q_seqlens.cpu(), torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    # Native context's causal mask sees a virtual compacted K domain. Every
    # physical selected row was already filtered by source logical causality.
    topk = metadata.num_sparse_topk
    torch.testing.assert_close(
        metadata.kv_lens_runtime.cpu(), torch.tensor([topk + 2, topk + 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.cu_kv_seqlens.cpu(), torch.tensor([0, topk + 2, 2 * topk + 3], dtype=torch.int32)
    )
    calls = []
    original = FallbackFmha.forward

    def record(provider, query, key, value, meta, forward_args):
        calls.append((meta.num_contexts, meta.num_ctx_tokens, forward_args.attention_input_type))
        assert forward_args.latent_cache is None
        return original(provider, query, key, value, meta, forward_args)

    monkeypatch.setattr(FallbackFmha, "forward", record)
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    args.attention_input_type = AttentionInputType.context_only
    output = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    torch.cuda.synchronize()
    assert calls == [(2, count, AttentionInputType.context_only)]
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )

    monkeypatch.setattr(FallbackFmha, "forward", original)
    generation = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    generation_args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    reference_generation = attn.forward(
        q.flatten(1), None, None, generation, forward_args=generation_args
    ).view_as(q)
    torch.testing.assert_close(output, reference_generation, atol=0.03, rtol=0.03)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("extra_width", [0, 17])
@torch.inference_mode()
def test_packed_helper_skips_bf16_staging(extra_width, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")

    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(460)
    heads, count = 32, 3
    attn = _backend(heads, use_packed=True)
    assert not any(isinstance(provider, FallbackFmha) for provider in attn._fmha_manager.fmha_libs)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 1::2] = False
    swa_valid[1] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[1] = False
    sink = torch.randn(heads, device="cuda")
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    records = None
    if extra is not None:
        packed = args.sparse_backend_args.main_pool
        records = torch.full((packed.shape[0], 356), 73, dtype=torch.uint8, device="cuda")
        records[:, :288].copy_(packed)
        args.sparse_backend_args.main_pool = records[:, :288]
    metadata = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    monkeypatch.setattr(
        metadata, "stage_selected", lambda *a: pytest.fail("Packed FMHA staged BF16 rows")
    )
    output = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )
    if records is not None:
        assert bool((records[:, 288:] == 73).all())
    # Reusing the selected library must revalidate unsupported output options.
    args.out_scale = torch.ones(1, device="cuda")
    with pytest.raises(
        ValueError, match="(Packed CSA2 does not support|do not support.*mask/output format)"
    ):
        _helper_forward(attn, q.flatten(1), metadata, args)

    args.out_scale = None
    args.sparse_backend_args.output_position_ids = torch.arange(
        count, device="cuda", dtype=torch.int32
    )
    with pytest.raises(
        ValueError, match="(Packed CSA2 does not support|do not support.*mask/output format)"
    ):
        _helper_forward(attn, q.flatten(1), metadata, args)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,packed",
    [("trtllm", False), ("flashinfer", False), ("prims_ts", False), ("trtllm", True)],
)
def test_csa2_quant_update_rebuilds_local_provider_policy(
    implementation: str, packed: bool
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

    attn = _backend(32, compute_backend=implementation, use_packed=packed)
    native = implementation == "trtllm" and not packed
    first = attn._fmha_manager
    assert bool(first.fmha_libs) == native
    assert all(type(provider) is FallbackFmha for provider in first.fmha_libs)
    attn.update_quant_config(attn.quant_config)
    assert bool(attn._fmha_manager.fmha_libs) == native
    assert all(type(provider) is FallbackFmha for provider in attn._fmha_manager.fmha_libs)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_backend_rejects_disabled_fallback(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    with pytest.raises(ValueError, match="requires the FallbackFmha"):
        _backend(32, compute_backend="trtllm")


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_context_graph_uses_fixed_generation_frame():
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(461)
    heads, count, extra_width = 64, 5, 17
    attn = _backend(heads)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    extra_valid = torch.ones(count, extra_width, device="cuda", dtype=torch.bool)
    sink = torch.randn(heads, device="cuda")
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count)
    source._num_contexts = 2
    source._num_ctx_tokens = source._num_tokens = count
    source.csa2_num_context_requests = 2
    source.csa2_request_query_ranges = ((0, 3), (3, 5))
    source.csa2_request_start_positions = (1024, 2048)
    source.csa2_request_lengths = (3, 2)
    source.is_cuda_graph = True
    metadata = source.get_query_tile_metadata(q, extra_width, query_start=0)
    assert metadata.num_contexts == 0
    assert metadata.num_generations == count
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    args.attention_input_type = (
        AttentionInputType.context_only
        if metadata.num_contexts
        else AttentionInputType.generation_only
    )
    for _ in range(3):
        attn.forward(q.flatten(1), None, None, metadata, forward_args=args)
    pointers = (
        metadata.swa_pool.data_ptr(),
        metadata.extra_pool.data_ptr(),
        metadata.workspace.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = source.get_query_tile_metadata(q, extra_width, query_start=0)
        assert captured is metadata
        output = attn.forward(q.flatten(1), None, None, captured, forward_args=args).view_as(q)
    swa_ids = torch.arange(count * 128, device="cuda").reshape(count, 128)
    main_ids = torch.arange(count * extra_width, device="cuda").reshape(count, extra_width)
    for active in (1, 0, 17):
        q.neg_()
        swa_valid.fill_(True)
        swa_valid[0, 1::2] = False
        swa_valid[1] = False
        extra_valid.copy_(torch.arange(extra_width, device="cuda")[None, :] < active)
        extra_valid[1] = False
        args.sparse_backend_args.swa_indices.copy_(torch.where(swa_valid, swa_ids, -1))
        args.sparse_backend_args.topk_indices.copy_(torch.where(extra_valid, main_ids, -1))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
        )
        assert pointers == (
            metadata.swa_pool.data_ptr(),
            metadata.extra_pool.data_ptr(),
            metadata.workspace.data_ptr(),
        )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,packed", [("flashinfer", False), ("prims_ts", False), ("trtllm", True)]
)
@torch.inference_mode()
def test_direct_compute_rejects_native_forward_misuse(implementation: str, packed: bool) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(32, compute_backend=implementation, use_packed=packed)
    q = torch.zeros(1, 32, 512, device="cuda", dtype=torch.bfloat16)
    values = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    args = _inputs(q, values, None, valid, None, torch.zeros(32, device="cuda"))
    with pytest.raises(ValueError, match="explicit module helper"):
        attn.forward(q.flatten(1), None, None, metadata, forward_args=args)


def test_segmented_mask_packing():
    mask = torch.tensor(
        [
            [True, False, True, False, False, False, False, True, True],
            [False, True, False, False, False, False, False, False, False],
        ]
    )
    assert CSA2FlashInfer.pack_query_masks(mask).tolist() == [133, 1, 2, 0]


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads,width,shared", [(8, 17, False), (64, 640, False), (64, 640, True)])
@torch.inference_mode()
def test_flashinfer_bf16_and_graph(heads, width, shared):
    torch.manual_seed(544)
    attn = CSA2FlashInfer()
    q = torch.randn(2, heads, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(2, width, 512, dtype=torch.bfloat16, device="cuda")
    valid = torch.ones(2, width, dtype=torch.bool, device="cuda")
    sink = torch.linspace(-2, 2, heads, device="cuda")

    def reference():
        scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * 512**-0.5
        scores.masked_fill_(~valid[:, None, :], -torch.inf)
        probs = torch.cat((scores, sink[None, :, None].expand(2, -1, -1)), -1).softmax(-1)[..., :-1]
        return torch.einsum("qhk,qkd->qhd", probs, kv.float()).bfloat16()

    indices = torch.arange(2 * width, device="cuda", dtype=torch.int32).reshape(2, width)

    def run():
        if shared:
            return attn.run_shared(q, kv.flatten(0, 1), indices, sink, 512**-0.5)
        return attn(q, kv, valid, sink, 512**-0.5)

    for _ in range(3):
        output = run()
    torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for end in (0, 9, width):
        valid.zero_()
        valid[:, :end] = True
        valid[1, ::2] = False
        kv.mul_(-1)
        if shared:
            physical = torch.arange(2 * width, device="cuda", dtype=torch.int32).reshape(2, width)
            indices.copy_(torch.where(valid, physical, -1))
        graph.replay()
        torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_fixed_plan_ignores_dirty_workspace_padding(monkeypatch):
    """A one-query no-split plan must not launch uninitialized padded CTAs."""
    q = torch.zeros(1, 8, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(1, 128, 512, dtype=torch.bfloat16, device="cuda")
    valid = torch.zeros(1, 128, dtype=torch.bool, device="cuda")
    valid[:, 0] = True
    sink = torch.zeros(8, device="cuda")
    original_empty = torch.empty

    def dirty_workspace(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if tensor.is_cuda and tensor.dtype == torch.uint8 and tensor.numel() >= 1024 * 1024:
            tensor.fill_(127)
        return tensor

    monkeypatch.setattr(torch, "empty", dirty_workspace)
    attn = CSA2FlashInfer()
    result = attn(q, kv, valid, sink, 512**-0.5)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.zeros_like(q), atol=0, rtol=0)


@skip_pre_blackwell
@skip_blackwell_geforce
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 64])
@torch.inference_mode()
def test_flash_mla_combined_pool_and_sink(heads):
    torch.manual_seed(345)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(3, 17, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(3, 17, device="cuda", dtype=torch.bool)
    valid[0, 9:] = False
    valid[1] = False
    sink = torch.linspace(-1, 2, heads, device="cuda")
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    weights = torch.cat((scores, sink[None, :, None].expand(3, -1, -1)), -1).softmax(-1)
    expected = torch.einsum("qhk,qkd->qhd", weights[..., :-1], kv.float()).bfloat16()
    actual = CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@skip_pre_blackwell
@skip_blackwell_geforce
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_flash_mla_cuda_graph_changes_selection():
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 128, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, 128, device="cuda", dtype=torch.bool)
    sink = torch.zeros(64, device="cuda")
    for _ in range(3):
        CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
    for end in (128, 3, 64):
        valid.zero_()
        valid[:, :end] = True
        kv.mul_(-1)
        graph.replay()
        expected = CSA2FlashMLA.run(q, kv, valid, sink, 512**-0.5)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_global_slot_graph_mapping_folds_visibility(monkeypatch):
    """Graph-owned slot mapping folds per-query visibility and tracks live inputs."""
    from ._utils import _page_metadata

    table = torch.arange(32, dtype=torch.int32, device="cuda").reshape(8, 4)
    request_storage = torch.empty(16, dtype=torch.int64, device="cuda")
    requests = request_storage[::2]
    requests.copy_(torch.arange(8, device="cuda"))
    metadata = _page_metadata(table, requests, 128, 512)
    metadata.is_cuda_graph = False
    storage = torch.zeros((8, 1024), dtype=torch.int64, device="cuda")
    logical = storage[:, ::2]
    logical.copy_(torch.arange(512, device="cuda"))
    visible = torch.full((8,), 512, dtype=torch.int64, device="cuda")
    metadata.global_slot_tile(0, 0, 8, logical, visible_lengths=visible)
    metadata.is_cuda_graph = True
    metadata.global_slot_tile(0, 0, 8, logical)
    for _ in range(3):
        metadata.global_slot_tile(0, 0, 8, logical, visible_lengths=visible)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        slots = metadata.global_slot_tile(0, 0, 8, logical, visible_lengths=visible)
    for step in range(3):
        table[0, 1] = -1 if step == 0 else 1
        requests.copy_(
            torch.arange(7, -1, -1, device="cuda") if step == 2 else torch.arange(8, device="cuda")
        )
        logical.copy_(torch.arange(512, device="cuda").roll(step * 17))
        logical[:, 7::29] = 1 << 40
        visible.fill_(0 if step == 1 else 511)
        graph.replay()
        # Frozen CPU mapping expression supplies the oracle independently of the helper.
        ids, req, pages, limits = logical.cpu(), requests.cpu(), table.cpu(), visible.cpu()
        columns = ids.clamp_min(0) // 128
        valid = (ids >= 0) & (ids < 512) & (columns < pages.shape[1])
        valid &= ((req >= 0) & (req < pages.shape[0]))[:, None]
        physical = pages[req.clamp(0, 7)[:, None], columns.clamp(max=3)]
        expected = torch.where(valid & (physical >= 0), physical.long() * 128 + ids % 128, -1)
        expected = torch.where(ids < limits[:, None], expected, -1).cuda()
        torch.testing.assert_close(slots, expected, atol=0, rtol=0)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "phase,extra_width,heads,kv_dtype",
    [
        pytest.param(phase, width, 64, dtype, id=f"{prefix}{phase}-{width}-64")
        for dtype, prefix in (("fp8_ds_mla", ""), ("fp8", "fp8-"))
        for phase, width in (
            ("generation", 0),
            ("generation", 512),
            ("context", 512),
            ("captured_context", 512),
            ("captured_generation", 512),
        )
    ],
)
@torch.inference_mode()
def test_native_fp8_quantized_attention(
    phase, extra_width, heads, kv_dtype, monkeypatch, record_property
):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    # FP8 staging path: keep the direct decode, which takes <= 16-row generation tiles, out.
    monkeypatch.setattr(CSA2TrtllmAttention, "direct_decode_serves", lambda *args: False)

    torch.manual_seed(482)
    count = 5
    attn = _backend(heads, kv_cache_dtype=kv_dtype)
    assert attn.native_fp8 and attn.staging_dtype == torch.float8_e4m3fn
    assert attn.has_fp8_kv_cache
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    q[:, 0, :8] = torch.tensor(
        [0.0, -0.0, 1.0625, -1.0625, 1.1875, -1.1875, 0.001, -0.001],
        device="cuda",
        dtype=torch.bfloat16,
    )
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    sv = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    sv[0, 1::2] = False
    sv[2] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    ev = torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    if ev is not None:
        ev[1, ::3] = False
        ev[2] = False
    sink = torch.randn(heads, device="cuda")
    args = _inputs(q, swa, extra, sv, ev, sink, swa_cache_format=attn.swa_cache_format)
    # Staging derives both scales from the packed pools; the oracle below reads
    # them back off the metadata and the forward args.
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count)
    if phase in ("generation", "captured_generation"):
        metadata = CSA2TrtllmMetadata.for_query_tile(
            q, extra_width, staging_dtype=attn.staging_dtype
        )
    else:
        source._num_contexts = 2
        source._num_ctx_tokens = source._num_tokens = count
        source.csa2_num_context_requests = 2
        source.csa2_request_query_ranges = ((0, 3), (3, 5))
        source.csa2_request_start_positions = (1024, 2048)
        source.csa2_request_lengths = (3, 2)
        source.is_cuda_graph = phase == "captured_context"
        metadata = source.get_query_tile_metadata(
            q, extra_width, query_start=0, staging_dtype=attn.staging_dtype
        )
    args.attention_input_type = (
        AttentionInputType.context_only
        if metadata.num_contexts
        else AttentionInputType.generation_only
    )
    assert metadata.num_contexts == (2 if phase == "context" else 0)
    calls = []
    seen = []
    native = FallbackFmha.forward

    def observe(provider, query, k, v, meta, forward_args):
        calls.append(forward_args.attention_input_type)
        assert query.dtype == torch.bfloat16
        assert meta.swa_pool.dtype == meta.extra_pool.dtype == torch.float8_e4m3fn
        if meta.num_contexts:
            assert forward_args.quant_q_buffer is None
            assert forward_args.quant_scale_qkv is forward_args.kv_scale_orig_quant
            # Context runs on a copy of the caller's args, so the scales it was
            # handed are observable only from in here.
            seen.append((forward_args.kv_scale_orig_quant, forward_args.kv_scale_quant_orig))
        if not meta.num_contexts:
            assert forward_args.quant_q_buffer.dtype == torch.float8_e4m3fn
            assert forward_args.latent_cache.dtype == torch.bfloat16
        return native(provider, query, k, v, meta, forward_args)

    monkeypatch.setattr(FallbackFmha, "forward", observe)

    def run():
        if phase == "captured_context":
            captured = source.get_query_tile_metadata(
                q, extra_width, query_start=0, staging_dtype=attn.staging_dtype
            )
            assert captured is metadata
        result = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
        if phase == "context":
            # The native-only context multiplier must not leak into caller args:
            # the next inherited forward rejects an unpaired multiplier.
            assert args.quant_scale_qkv is None
            assert args.quant_q_buffer is None
        return result

    # As in the graph runner, captured generation warms up and captures with multi-stream
    # work on, so it quantizes Q on a stream of its own.
    with with_multi_stream(phase == "captured_generation"):
        for _ in range(3):
            output = run()
        assert calls and all(p == args.attention_input_type for p in calls)
        graph = None
        pointers = None
        if phase.startswith("captured"):
            pointers = (
                metadata.swa_pool.data_ptr(),
                metadata.extra_pool.data_ptr(),
                metadata.workspace.data_ptr(),
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
    for replay in range(2 if graph else 1):
        if graph:
            q.neg_()
            args.sparse_backend_args.swa_indices[0, : 3 * (replay + 1)] = -1
            sv[0, : 3 * (replay + 1)] = False
            graph.replay()
        torch.cuda.synchronize()
        kv_dq = metadata.stage_kv_scales[1]
        scale_kv = kv_dq.item()
        # A power of two costs no relative precision: it only moves the exponent.
        assert scale_kv > 0.0 and math.log2(scale_kv).is_integer()
        # These pools decode well inside the E4M3 range, so the derivation asks
        # for no extra room and every staged byte stays what the unit scale
        # produced before the range-aware scale existed.
        assert scale_kv == 1.0
        if metadata.num_contexts:
            # The C++ op points dequant_scale_q at dequant_scale_kv, so context
            # cannot give Q a scale of its own: it gets the staging pair as is.
            orig_quant, q_dq = seen[-1]
            assert (orig_quant, q_dq) == metadata.stage_kv_scales
        else:
            # Q takes the reciprocal, which keeps BMM1's dequant product at one.
            # The in-tree split-KV reduction no longer requires that, but the same
            # correction factors live inside the cubins serving the other two
            # MultiCtasKvMode values, whose selection is not auditable.
            q_dq = args.kv_scale_quant_orig
            torch.testing.assert_close(q_dq * kv_dq, torch.ones(1, device="cuda"), atol=0, rtol=0)
        packed_q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, q_dq)
        if not metadata.num_contexts:
            torch.testing.assert_close(
                args.quant_q_buffer.reshape_as(q).view(torch.uint8),
                packed_q.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            scale = q_dq.item() * scale_kv * 512**-0.5
            torch.testing.assert_close(
                metadata.mla_bmm1_scale,
                torch.tensor([scale, scale * 1.4426950408889634], device="cuda"),
                atol=0,
                rtol=1e-6,
            )
            torch.testing.assert_close(metadata.mla_bmm2_scale, kv_dq, atol=0, rtol=0)
        inp = args.sparse_backend_args
        decoded_swa = unpack_rows(inp.swa_pool, 512, attn.swa_cache_format).reshape(count, 128, 512)
        qswa, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_swa, kv_dq)
        qextra = None
        if extra_width:
            decoded_extra = unpack_rows(inp.main_pool, 512, "main").reshape(count, extra_width, 512)
            qe, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_extra, kv_dq)
            qextra = qe.float() * scale_kv
        # BMM1 and BMM2 undo the staging scales, so the oracle runs on the
        # dequantized magnitudes that the kernel's own scale factors restore.
        reference = _reference(
            packed_q.float() * q_dq.item(), qswa.float() * scale_kv, qextra, sv, ev, sink
        ).to(torch.bfloat16)
        # The oracle accounts for input quantization. This is the existing
        # native BF16-output accumulation budget, not an extra quantization budget.
        torch.testing.assert_close(output, reference, atol=0.03, rtol=0.03)
        bf16 = _decoded_reference(q, args, sv, ev, sink)
        record_property(
            f"fp8_vs_bf16_max_abs_replay_{replay}",
            (output.float() - bf16.float()).abs().max().item(),
        )
        record_property(
            f"fp8_vs_quantized_reference_max_abs_replay_{replay}",
            (output.float() - reference.float()).abs().max().item(),
        )
        if graph:
            assert pointers == (
                metadata.swa_pool.data_ptr(),
                metadata.extra_pool.data_ptr(),
                metadata.workspace.data_ptr(),
            )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "phase,extra_width",
    [("generation", 128), ("generation", 512), ("context", 512)],
)
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
@torch.inference_mode()
def test_native_fp8_staging_keeps_rows_beyond_the_e4m3_range(
    phase, extra_width, kv_dtype, monkeypatch
):
    """Staged rows whose real magnitudes run past 448 must not be clamped to it.

    MAIN and legacy SWA carry group scales that can represent magnitudes above
    448. Ordinary FP8 SWA is already unit-scale E4M3, but MAIN must still determine
    a shared staging range that does not clamp its decoded values. The reference
    keeps the pools' exactly decoded values, which makes the reported error the
    staging round trip alone. The 512-wide case also runs past the KV length where
    trtllm-gen splits its reduction across CTAs, whose correction factors have to
    be rescaled by the same BMM1 scale the main kernel used.
    """
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (
        _CONTEXT_STAGE_SCALE_MAX_EXPONENT,
        CSA2TrtllmAttention,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    # FP8 staging path: keep the direct decode, which takes <= 16-row generation tiles, out.
    monkeypatch.setattr(CSA2TrtllmAttention, "direct_decode_serves", lambda *args: False)
    torch.manual_seed(482)
    count, heads = 5, 64
    attn = _backend(heads, kv_cache_dtype=kv_dtype)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    # Roughly the magnitude an image token reaches: inside what the packed pools
    # represent, far outside what a unit staging scale can hold.
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16) * 600.0
    sv = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    sv[0, 1::2] = False
    extra = torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16) * 600.0
    ev = torch.ones(count, extra_width, device="cuda", dtype=torch.bool)
    ev[1, ::3] = False
    sink = torch.randn(heads, device="cuda")
    args = _inputs(q, swa, extra, sv, ev, sink, swa_cache_format=attn.swa_cache_format)
    if phase == "generation":
        metadata = CSA2TrtllmMetadata.for_query_tile(
            q, extra_width, staging_dtype=attn.staging_dtype
        )
    else:
        source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count)
        source._num_contexts = 2
        source._num_ctx_tokens = source._num_tokens = count
        source.csa2_num_context_requests = 2
        source.csa2_request_query_ranges = ((0, 3), (3, 5))
        source.csa2_request_start_positions = (1024, 2048)
        source.csa2_request_lengths = (3, 2)
        metadata = source.get_query_tile_metadata(
            q, extra_width, query_start=0, staging_dtype=attn.staging_dtype
        )
    args.attention_input_type = (
        AttentionInputType.context_only
        if metadata.num_contexts
        else AttentionInputType.generation_only
    )
    output = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    kv_dq = metadata.stage_kv_scales[1]
    scale_kv = kv_dq.item()
    inp = args.sparse_backend_args
    decoded_swa = (
        unpack_rows(inp.swa_pool, 512, attn.swa_cache_format).reshape(count, 128, 512).float()
    )
    decoded_extra = unpack_rows(inp.main_pool, 512, "main").reshape(count, extra_width, 512).float()
    decoded_max = max(decoded_swa.abs().amax().item(), decoded_extra.abs().amax().item())
    assert decoded_max > 448.0
    if metadata.num_contexts:
        # Context shares one dequant tensor between Q and KV inside the C++ op, so Q
        # is divided by this scale instead of multiplying by its reciprocal. That
        # direction cannot saturate Q -- it spends Q's resolution below one -- so
        # context buys range too, as far as Q's own amax affords it. These pools ask
        # for more than either bound allows, so the tighter bound itself lands here,
        # well above the unit scale this path used to be pinned to; whatever still
        # runs past 448 * scale needs a decoupled dequant_scale_q in the C++ op.
        # Only the output comparison below stays generation-only, because nothing
        # here models a context reference.
        q_amax = q.abs().amax().item()
        affordable = 2.0 ** math.floor(math.log2(q_amax))
        assert 1.0 < scale_kv == min(affordable, 2.0**_CONTEXT_STAGE_SCALE_MAX_EXPONENT)
        assert q_amax / scale_kv >= 1.0  # Q gave up resolution, not magnitude.
        return
    # The derived scale must cover what the pools decode to, which is the whole
    # point: 448 * scale is the largest magnitude staging can still represent.
    assert scale_kv * 448.0 >= decoded_max
    q_dq = args.kv_scale_quant_orig
    packed_q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, q_dq)
    reference = _reference(
        packed_q.float() * q_dq.item(), decoded_swa, decoded_extra, sv, ev, sink
    ).to(torch.bfloat16)
    # Magnitudes this large make the softmax nearly a hard argmax, so a single
    # flipped winner dominates the maximum error; the mean is the stable measure.
    # A unit scale scores above 0.1 here, an order of magnitude worse.
    error = (output.float() - reference.float()).abs().mean().item()
    assert error / reference.float().abs().amax().item() < 0.03


@pytest.mark.cpu_only
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla", "auto"])
@pytest.mark.parametrize("native_fp8", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compute_backend", ["trtllm", "prims_ts"])
def test_direct_decode_admission_by_swa_format_and_provider(
    monkeypatch: pytest.MonkeyPatch,
    kv_dtype: str,
    native_fp8: bool,
    packed: bool,
    compute_backend: str,
) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import direct_decode
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import resolve_swa_cache_format

    monkeypatch.setattr(direct_decode, "direct_decode_device_supported", lambda device: True)
    attn = SimpleNamespace(
        compute_backend=compute_backend,
        native_fp8=native_fp8,
        swa_cache_format=resolve_swa_cache_format(kv_dtype),
        sparse_params=CSA2Params(use_packed_sparse_attention=packed),
        num_heads=16,
    )
    serves = CSA2TrtllmAttention.direct_decode_serves
    device = torch.device("cpu")
    # Native FP8 staging takes it for scaled SWA records only; PrimTS for either SWA format.
    expected = not packed and ((native_fp8 and kv_dtype != "fp8") or compute_backend == "prims_ts")
    for rows in (1, direct_decode.MAX_ROWS):
        assert serves(attn, rows, 0, 512, device) is expected
    assert not serves(attn, 0, 0, 512, device)
    assert not serves(attn, direct_decode.MAX_ROWS + 1, 0, 512, device)
    assert not serves(attn, 1, 1, 512, device)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("heads", "mode", "rows"),
    [
        (8, "SWA", 12),
        (8, "REUSE", 8),
        (8, "FULL", 1),
        (8, "REINDEX", 16),
        (16, "SWA", 1),
        (64, "FULL", 4),
        (64, "REINDEX", 16),
    ],
)
def test_prims_ts_direct_decode_serves_every_layer_mode(
    monkeypatch: pytest.MonkeyPatch, heads: int, mode: str, rows: int
) -> None:
    """A PrimTS generation step the direct decode fits takes it whatever the layer's mode."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import direct_decode
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Mode

    monkeypatch.setattr(direct_decode, "direct_decode_device_supported", lambda device: True)
    layout = SimpleNamespace(
        compress_ratios=(0, 4), layer=lambda idx: SimpleNamespace(mode=getattr(CSA2Mode, mode))
    )
    attn = SimpleNamespace(
        compute_backend="prims_ts",
        native_fp8=False,
        swa_cache_format="swa_fp8",
        sparse_params=SimpleNamespace(layout=layout, use_packed_sparse_attention=False),
        num_heads=heads,
        layer_idx=1,
    )
    assert CSA2TrtllmAttention.direct_decode_serves(attn, rows, 0, 512, torch.device("cpu"))
    assert not CSA2TrtllmAttention.direct_decode_serves(
        attn, direct_decode.MAX_ROWS + 1, 0, 512, torch.device("cpu")
    )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kv_cache_dtype_selects_persistent_format_independently_of_staging():
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    fp8_kv = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    # Both FP8 spellings and the auto/checkpoint algorithm can request E4M3
    # staging, but only the explicit raw fp8 dtype selects payload-only SWA.
    for kv_dtype, quant_config, persistent in (
        ("fp8", None, "swa_fp8"),
        ("fp8_ds_mla", None, "swa"),
        ("auto", fp8_kv, "swa"),
    ):
        for staging in (True, False):
            attn = _backend(
                64, kv_cache_dtype=kv_dtype, quant_config=quant_config, fp8_staging=staging
            )
            assert attn.swa_cache_format == persistent
            assert attn.native_fp8 is staging
            assert attn.staging_dtype == (torch.float8_e4m3fn if staging else torch.bfloat16)
            assert attn.has_fp8_kv_cache is staging
            assert attn.kv_cache_dtype == ("fp8" if staging else "auto")
    # Weight quantization in the same config is preserved for the parent.
    attn = _backend(
        64,
        kv_cache_dtype="auto",
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, kv_cache_quant_algo=QuantAlgo.FP8),
        fp8_staging=True,
    )
    assert attn.native_fp8 and attn.quant_config.quant_algo == QuantAlgo.MXFP8
    assert attn.swa_cache_format == "swa"
    # BF16 staging keeps the inherited state free of an FP8 KV cache, and an
    # explicit bfloat16 request wins over a checkpoint FP8 KV algo.
    for kv_dtype, quant_config in (("auto", None), ("bfloat16", fp8_kv)):
        attn = _backend(64, kv_cache_dtype=kv_dtype, quant_config=quant_config, fp8_staging=True)
        assert not attn.native_fp8 and attn.staging_dtype == torch.bfloat16
        assert not attn.has_fp8_kv_cache
        assert attn.swa_cache_format == "swa"
    # A later quantization config cannot change the staging chosen at
    # construction; FP8 KV is normalized, anything else is rejected.
    attn.update_quant_config(fp8_kv)
    assert not attn.has_fp8_kv_cache and attn.quant_config.kv_cache_quant_algo is None
    with pytest.raises(ValueError, match="only FP8"):
        attn.update_quant_config(QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4))
    with pytest.raises(ValueError, match="only FP8"):
        _backend(64, quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4))
    with pytest.raises(ValueError, match="dtype must be"):
        _backend(64, kv_cache_dtype="float16")
    # Packed attention reads persistent rows and has no staging: a checkpoint
    # FP8 KV algo just stays BF16, and opting into FP8 staging is contradictory.
    attn = _backend(64, use_packed=True, quant_config=fp8_kv)
    assert not attn.native_fp8 and not attn.has_fp8_kv_cache
    with pytest.raises(ValueError, match="FP8 staging requires"):
        _backend(64, use_packed=True, kv_cache_dtype="fp8", fp8_staging=True)


@skip_pre_blackwell
@pytest.mark.parametrize("implementation", ["trtllm", "prims_ts"])
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
@pytest.mark.parametrize("mismatch", ["carrier", "manager", "inferred_manager"])
def test_swa_format_mismatch_rejects_before_cache_publication(
    implementation: str, kv_dtype: str, mismatch: str
) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
        CSA2BackendForwardArgs,
        CSA2ForwardState,
    )

    attn = _backend(64, compute_backend=implementation, kv_cache_dtype=kv_dtype, fp8_staging=False)
    expected = attn.swa_cache_format
    other = "swa" if expected == "swa_fp8" else "swa_fp8"

    def unexpected_publication(*args, **kwargs) -> None:
        raise AssertionError("A format mismatch must be rejected before cache/indexer mutation")

    manager = SimpleNamespace(
        swa_cache_format=expected if mismatch == "carrier" else other,
        write_layer_rows=unexpected_publication,
    )
    source = SimpleNamespace(
        kv_cache_manager=manager, enter_layer=unexpected_publication, num_ctx_tokens=0
    )
    state = CSA2ForwardState(
        metadata=source, swa_kv=torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16)
    )
    tag = None if mismatch == "inferred_manager" else other if mismatch == "carrier" else expected
    args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        sparse_backend_args=CSA2BackendForwardArgs(state=state, swa_cache_format=tag),
    )
    q = torch.zeros(1, 64, 512, device="cuda", dtype=torch.bfloat16)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0, kv_input_mode=attn.kv_input_mode)
    message = "input SWA format" if mismatch == "carrier" else "cache manager SWA format"
    with pytest.raises(ValueError, match=message):
        attn.prepare_sparse_inputs(q.flatten(1), metadata, args)


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["flashinfer", "prims_ts"])
@pytest.mark.parametrize("staging", [None, False])
def test_library_backends_keep_bf16_queries_for_fp8_requests(
    implementation: str, staging: bool | None
) -> None:
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    attn = _backend(64, compute_backend=implementation, kv_cache_dtype="fp8", fp8_staging=staging)
    assert not attn.native_fp8 and attn.staging_dtype == torch.bfloat16
    assert attn.kv_input_mode == ("packed" if implementation == "prims_ts" else "staged")
    assert not attn.has_fp8_kv_cache and attn.kv_cache_dtype == "auto"
    assert attn.swa_cache_format == "swa_fp8"
    with pytest.raises(ValueError, match="FP8 staging requires"):
        _backend(64, compute_backend=implementation, kv_cache_dtype="fp8", fp8_staging=True)
    attn.update_quant_config(QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8))
    assert not attn.has_fp8_kv_cache


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
def test_default_staging_follows_the_backend_that_can_serve_it(kv_dtype):
    # Left at its default, FP8 staging follows the native path's own support
    # rather than the request: trtllm-gen stages E4M3 for an FP8 KV dtype, and
    # every configuration that cannot consume it keeps BF16 instead of failing.
    attn = _backend(64, kv_cache_dtype=kv_dtype)
    persistent = "swa_fp8" if kv_dtype == "fp8" else "swa"
    assert attn.native_fp8 and attn.staging_dtype == torch.float8_e4m3fn
    assert attn.swa_cache_format == persistent
    for kwargs in (
        {"use_packed": True},
        {"compute_backend": "flash_mla"},
        {"compute_backend": "flashinfer"},
    ):
        if kwargs.get("compute_backend") == "flashinfer":
            pytest.importorskip("flashinfer")
        attn = _backend(64, kv_cache_dtype=kv_dtype, **kwargs)
        assert not attn.native_fp8 and attn.staging_dtype == torch.bfloat16
        assert not attn.has_fp8_kv_cache and attn.kv_cache_dtype == "auto"
        assert attn.swa_cache_format == persistent


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_fp8_rejects_incompatible_options():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    # Packed attention reads persistent rows directly and has no FP8 staging.
    with pytest.raises(ValueError):
        _backend(64, kv_cache_dtype="fp8", use_packed=True, fp8_staging=True)
    # A BF16-staged frame must never be consumed by an FP8 backend.
    attn = _backend(64, kv_cache_dtype="fp8")
    q = torch.zeros(1, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 128, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 128, device="cuda", dtype=torch.bool)
    args = _inputs(
        q,
        swa,
        None,
        valid,
        None,
        torch.zeros(64, device="cuda"),
        swa_cache_format=attn.swa_cache_format,
    )
    with pytest.raises(ValueError):
        attn.forward(
            q.flatten(1), None, None, CSA2TrtllmMetadata.for_query_tile(q, 0), forward_args=args
        )


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,kv_dtype,decisive",
    [
        ("trtllm", "auto", "main"),
        ("trtllm", "fp8_ds_mla", "main"),
        ("trtllm", "fp8_ds_mla", "swa"),
        ("trtllm", "fp8_ds_mla", "none"),
        ("trtllm", "fp8", "main"),
        ("trtllm", "fp8", "none"),
        ("flashinfer", "auto", "main"),
    ],
)
@torch.inference_mode()
def test_shared_native_context_matches_selected_rows(
    implementation, kv_dtype, decisive, monkeypatch, record_property
):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import (
        CSA2SharedKVPlan,
        CSA2TrtllmMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        gather_rows,
        pack_rows,
    )

    torch.manual_seed(489)
    count, heads, width = 5, 64, 512
    attn = _backend(heads, kv_cache_dtype=kv_dtype, compute_backend=implementation)
    # Shared and selected staging must feed trtllm-gen identical operands; the fused FP8
    # context kernel serves shared tiles by default (test_fused_context_matches_trtllm_gen).
    attn.fused_context = False
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 1024, 1025], device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 0, 0, 1, 1], device="cuda", dtype=torch.int64)
    swa_pages = torch.full((2, 9), -1, device="cuda", dtype=torch.int32)
    swa_pages[0, 0] = 2
    swa_pages[1, 7:] = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    main_pages = torch.full((2, 9), -1, device="cuda", dtype=torch.int32)
    main_pages[0, 0] = 9
    main_pages[1] = torch.arange(9, device="cuda", dtype=torch.int32)
    swa = pack_rows(
        torch.randn(384, 512, device="cuda", dtype=torch.bfloat16), attn.swa_cache_format
    )
    main_storage = torch.empty(1280, 360, device="cuda", dtype=torch.uint8)
    main = main_storage[:, :288]
    main.copy_(pack_rows(torch.randn(1280, 512, device="cuda", dtype=torch.bfloat16), "main"))
    # A selected row whose group scales need a non-unit staging scale.
    if decisive == "main":
        main[0, 256:] = 126
    elif decisive == "swa":
        swa[256, 512:] = 129
    swa_logical = positions[:, None] - 127 + torch.arange(128, device="cuda")
    main_logical = torch.arange(width, device="cuda").expand(count, -1).clone()

    def physical(logical, pages):
        page = pages[requests[:, None], logical.clamp_min(0) // 128].long()
        return torch.where((logical >= 0) & (page >= 0), page * 128 + logical.remainder(128), -1)

    swa_slots = physical(swa_logical, swa_pages)
    main_logical.masked_fill_(main_logical > positions[:, None], -1)
    main_slots = physical(main_logical, main_pages)
    # Sparse holes cross the native128-row pool boundary; one query is empty.
    swa_slots[:, 1::11] = -1
    main_slots[:, 3::13] = -1
    swa_slots[2] = main_slots[2] = -1
    sink = torch.randn(heads, device="cuda")
    plan = CSA2SharedKVPlan(
        query_requests=requests,
        query_positions=positions,
        swa_starts=torch.tensor([0, 897], device="cuda", dtype=torch.int64),
        swa_offsets=torch.tensor([1, 131, 260], device="cuda", dtype=torch.int64),
        main_offsets=torch.tensor([260, 263, 1289], device="cuda", dtype=torch.int64),
        swa_pages=swa_pages,
        main_pages=main_pages,
        swa_page_size=128,
        main_page_size=128,
        num_requests=2,
        swa_rows=259,
        main_rows=1029,
        swa_first=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
        swa_floors=torch.tensor([0, 897], device="cuda", dtype=torch.int64),
        window=1 << 40,
    )
    calls = []
    native_stage = kernel.stage_shared_rows

    def observe(*args, **kwargs):
        calls.append(True)
        return native_stage(*args, **kwargs)

    monkeypatch.setattr(kernel, "stage_shared_rows", observe)
    outputs = []
    for shared in (False, True):
        metadata = CSA2TrtllmMetadata.for_query_tile(
            q,
            width,
            staging_dtype=attn.staging_dtype,
            context_lengths=[3, 2],
            shared_rows=1289 if shared else None,
        )
        if shared:
            metadata.shared_plan = plan
            assert metadata.swa_pool is metadata.extra_pool is metadata.shared_pool
            assert metadata.shared_pool.shape == (1289, 512)
            assert metadata.shared_pool.numel() < count * (128 + width) * 512
        args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.context_only,
            attention_sinks=sink,
            sparse_backend_args=CSA2BackendForwardArgs(
                swa_cache_format=attn.swa_cache_format,
                swa_pool=swa,
                swa_indices=swa_slots,
                main_pool=main,
                topk_indices=main_slots,
                main_logical_indices=main_logical,
            ),
        )
        output = (
            attn.forward(q.flatten(1), None, None, metadata, forward_args=args)
            if implementation == "trtllm"
            else _helper_forward(attn, q.flatten(1), metadata, args)
        )
        outputs.append(output.view_as(q))
    torch.cuda.synchronize()
    assert calls == [True]
    if implementation == "trtllm":
        torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    else:
        # FlashInfer ragged and paged launches can use different reduction orders.
        torch.testing.assert_close(outputs[1], outputs[0], atol=0.03, rtol=0.03)
    decoded_swa = gather_rows(swa, swa_slots, 512, attn.swa_cache_format)
    decoded_main = gather_rows(main, main_slots, 512, "main")
    ref_q = q
    if attn.native_fp8:
        scale = torch.ones(1, device="cuda", dtype=torch.float32)
        ref_q = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, scale)[0].float()
        decoded_swa = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_swa, scale)[
            0
        ].float()
        decoded_main = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(decoded_main, scale)[
            0
        ].float()
    expected = _reference(
        ref_q, decoded_swa, decoded_main, swa_slots >= 0, main_slots >= 0, sink
    ).to(torch.bfloat16)
    for name, actual in zip(("selected", "shared"), outputs):
        if implementation == "trtllm" and attn.native_fp8:
            # Native FP8 also quantizes P before BMM2; the FP32-P oracle does
            # not model that arithmetic. Exact native storage parity above is
            # the regression gate; retain independent FP32-P error as evidence.
            record_property(
                f"{name}_fp8_vs_fp32_p_max_abs",
                (actual.float() - expected.float()).abs().max().item(),
            )
        else:
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
        torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), atol=0, rtol=0)


@pytest.mark.cpu_only
def test_flashinfer_shared_plans_bound_eager_context_and_keep_protected(monkeypatch):
    """Eager context phases must not accumulate one paged plan per query count."""
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )

    heads, width = 8, 129
    bank = torch.zeros(7, 512, dtype=torch.bfloat16)
    sink = torch.zeros(heads)
    plans = []

    class Paged:
        def __init__(self, workspace, **kwargs):
            pass

        def plan(self, qo, indptr, page_indices, *args, **kwargs):
            plans.append(qo.numel() - 1)
            self._paged_kv_indices_buf = page_indices.clone()
            self._custom_mask_buf = torch.empty(
                (qo.numel() - 1) * ((width + 7) // 8), dtype=torch.uint8
            )

        def run(self, query, pools, *, return_lse):
            return torch.zeros_like(query), torch.zeros(query.shape[:2])

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.prefill",
        SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=Paged),
    )
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    helper = CSA2FlashInfer()
    selection = [None]
    helper.attention = SimpleNamespace(
        prepare_selected=lambda query, metadata, args: (
            query.view(-1, heads, 512),
            bank,
            selection[0],
            sink,
            512**-0.5,
        )
    )

    def phase(n, contexts, graph_reserved=False):
        selection[0] = torch.arange(n * width, dtype=torch.int32).reshape(n, width) % 7
        query = torch.zeros(n, heads * 512, dtype=torch.bfloat16)
        metadata = SimpleNamespace(
            num_contexts=contexts, shared_plan=True, _csa2_graph_reserved=graph_reserved
        )
        if contexts:
            args = AttentionForwardArgs(attention_input_type=AttentionInputType.context_only)
            return helper.forward_context(query, metadata, args)
        args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)
        return helper.forward_generation(query, metadata, args)

    key = lambda n: (n, heads, width, torch.device("cpu"), 512**-0.5)  # noqa: E731
    phase(3, contexts=1)
    capturing[0] = True
    phase(3, contexts=1, graph_reserved=True)  # captured: must survive later eviction
    capturing[0] = False
    captured = helper._paged_plans[key(3)]
    phase(1, contexts=0, graph_reserved=True)  # graph generation plan is protected as well
    generation = helper._paged_plans[key(1)]
    phase(6, contexts=0)  # eager generation (e.g. a mixed step) is not protected
    for n in (2, 4, 5):
        phase(n, contexts=1)
        assert set(helper._paged_plans) == {key(3), key(1), key(n)}
    assert helper._paged_plans[key(3)] is captured
    assert helper._paged_plans[key(1)] is generation
    assert plans == [3, 1, 6, 2, 4, 5]


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp8_ds_mla"])
@torch.inference_mode()
def test_native_fp8_staging_groups_scan_main_scales_once(monkeypatch, kv_dtype):
    """Each staging group scans its selected MAIN scale bytes once per forward and tile.

    The Reuse layer shares its owner's selections and MAIN cache, so it keeps the
    owner's MAIN bits but never its SWA bits; the Reindex layer selects its own rows
    and scans again. Every tile's scale, and the attention it feeds, must match a
    forward that scans each layer on its own, and the next forward must scan the
    rewritten cache.
    """
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import patch

    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import kernel
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
        CSA2CacheManager,
        CSA2CacheRole,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import (
        CSA2IndexerGroup,
        CSA2TrtllmMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
        CSA2BackendForwardArgs,
        CSA2ForwardState,
        CSA2Layout,
        CSA2Mode,
    )
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    # FP8 staging path: keep the direct decode, which takes <= 16-row generation tiles, out.
    monkeypatch.setattr(CSA2TrtllmAttention, "direct_decode_serves", lambda *args: False)
    torch.manual_seed(1230)
    layout = CSA2Layout((1, 1, 1), (0,), (0, 2), index_topk=4, window_size=4)
    assert [layout.layer(i).mode for i in range(3)] == [
        CSA2Mode.FULL,
        CSA2Mode.REUSE,
        CSA2Mode.REINDEX,
    ]
    count, heads = 19, 8
    manager = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, host_cache_size=0, dtype=kv_dtype),
        CacheType.SELFKONLY,
        num_layers=3,
        tokens_per_block=128,
        max_seq_len=64,
        max_batch_size=1,
        max_num_tokens=count,
        mapping=Mapping(),
        vocab_size=8192,
        layout=layout,
    )
    try:
        request = manager._create_kv_cache(100, None, [])
        assert manager._resume_and_restore(100, request)
        assert request.resize(count)
        runtime = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=count, kv_cache_manager=manager
        )
        positions = torch.arange(count, device="cuda")
        q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
        swa = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
        main = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
        index_k = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
        index_q = torch.randn(count, 2, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.ones(count, 2, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(heads, device="cuda")
        global_base = manager.get_cache_indices(100, 0, CSA2CacheRole.GLOBAL)[0] * 128
        runtime.csa2_token_requests = torch.zeros(count, dtype=torch.int64, device="cuda")
        runtime.csa2_request_query_ranges = ((0, count),)
        runtime.csa2_request_start_positions = (0,)
        runtime.csa2_num_context_requests = 1
        runtime.csa2_indexer_groups = {
            0: (
                CSA2IndexerGroup(
                    is_prefill=True,
                    first=0,
                    last=count,
                    lo=0,
                    hi=count,
                    sizes=None,
                    width=6,
                    keys=6,
                    descriptors=torch.tensor([[0, 6, 0, 0, 0]], dtype=torch.int32, device="cuda"),
                ),
            )
        }
        runtime.csa2_global_page_tables = {
            0: torch.tensor([[global_base // 128]], dtype=torch.int32, device="cuda")
        }
        runtime.csa2_global_page_sizes = {0: 128}
        runtime.csa2_global_max_positions = {0: 6}
        runtime.csa2_main_write_slots = {0: global_base + torch.arange(6, device="cuda")}
        runtime.csa2_kv_sources = {0: 0, 1: 0, 2: 0}
        runtime.csa2_swa_indices = {}
        runtime.csa2_swa_write_slots = {}
        runtime.csa2_visible_lengths = {}
        for layer in range(3):
            swa_base = manager.get_cache_indices(100, layer, CSA2CacheRole.SWA)[0] * 128
            local = positions[:, None] - torch.arange(4, device="cuda")[None, :]
            runtime.csa2_swa_indices[layer] = torch.where(local >= 0, local + swa_base, -1)
            runtime.csa2_swa_write_slots[layer] = positions + swa_base
            runtime.csa2_visible_lengths[layer] = positions.remainder(6) + 1
        backends = [_backend(heads, layer, layout, kv_cache_dtype=kv_dtype) for layer in range(3)]
        assert all(backend.native_fp8 for backend in backends)

        main_scans = []
        original_launch = kernel._launch

        def launch(key, *values):
            if key[0] == "stage_scales" and key[2]:
                main_scans.append(key)
            return original_launch(key, *values)

        original_scratch = CSA2TrtllmMetadata._stage_scratch

        def unshared(self, source, group, count, field="serial"):
            entry, compact = original_scratch(self, source, group, count, field)
            return entry, compact or field == "scale_serial"

        monkeypatch.setattr(kernel, "_launch", launch)

        def run(main_scale, owner_swa_scale, share):
            main_scans.clear()
            runtime.reset_routing()
            scales, outputs = [], []
            scope = (
                nullcontext()
                if share
                else patch.object(CSA2TrtllmMetadata, "_stage_scratch", unshared)
            )
            with scope:
                for layer, backend in enumerate(backends):
                    rows = {}
                    if layer != 1:
                        rows.update(index_q=index_q * (-1 if layer else 1), index_weights=weights)
                    if layer == 0:
                        rows.update(main_kv=main * main_scale, index_k=index_k)
                    layer_swa = swa * owner_swa_scale if layer == 0 else swa
                    state = CSA2ForwardState(metadata=runtime, swa_kv=layer_swa, **rows)
                    for start in range(0, count, 16):
                        tile = q[start : start + 16]
                        frame = runtime.get_query_tile_metadata(
                            tile, layout.index_topk, staging_dtype=backend.staging_dtype
                        )
                        args = AttentionForwardArgs(
                            attention_input_type=AttentionInputType.generation_only,
                            attention_sinks=sink,
                            sparse_backend_args=CSA2BackendForwardArgs(
                                state=state, query_start=start
                            ),
                        )
                        output = backend.forward(
                            tile.flatten(1), None, None, frame, forward_args=args
                        )
                        outputs.append(output.clone())
                        scales.append(frame.stage_kv_scales[1].item())
            return scales, outputs, len(main_scans)

        # Large MAIN rows set a nonunit exponent for either SWA format. Only
        # legacy SWA can store a nonunit persistent group scale; ordinary SWA
        # remains unit-scale E4M3 even when its unquantized input is large. Reuse
        # must share the owner's MAIN bound, never its private SWA bound.
        for main_scale, owner_swa_scale in ((900.0, 1.0), (1.0, 2000.0), (1.0, 1.0)):
            shared_scales, shared_outputs, shared_scans = run(
                main_scale, owner_swa_scale, share=True
            )
            alone_scales, alone_outputs, alone_scans = run(main_scale, owner_swa_scale, share=False)
            # Owner and Reindex layers scan both tiles; the Reuse layer scans none.
            assert (shared_scans, alone_scans) == (4, 6)
            assert shared_scales == alone_scales
            owner, reuse = shared_scales[:2], shared_scales[2:4]
            if main_scale > 1.0:
                assert owner == reuse == [8.0, 8.0]
            elif owner_swa_scale > 1.0 and kv_dtype == "fp8_ds_mla":
                assert min(owner) > 1.0 and reuse == [1.0, 1.0]
            else:
                assert owner == reuse == [1.0, 1.0]
            for shared, alone in zip(shared_outputs, alone_outputs):
                torch.testing.assert_close(shared, alone, atol=0, rtol=0)
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


def _fused_context_call(lengths, full, seed):
    """One eager FP8 context phase over the shared bank (Full: SWA window + top-512 of ratio-2 rows)."""
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import (
        CSA2SharedKVPlan,
        CSA2TrtllmMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    g = torch.Generator(device="cuda").manual_seed(seed)
    heads, dim, window, topk = 64, 512, 128, 512
    page, ratio = 128, 2
    requests, count = len(lengths), sum(lengths)
    swa_pages = [math.ceil(length / page) for length in lengths]
    order = torch.randperm(sum(swa_pages) + 4, device="cuda", generator=g).int()
    swa_table = torch.full((requests, max(swa_pages)), -1, device="cuda", dtype=torch.int32)
    used = 0
    for r, n in enumerate(swa_pages):
        swa_table[r, :n] = order[used : used + n]
        used += n
    swa_pool = pack_rows(
        torch.randn(order.numel() * page, dim, device="cuda", generator=g).bfloat16(), "swa"
    )
    request = torch.cat(
        [torch.full((n,), r, device="cuda", dtype=torch.int64) for r, n in enumerate(lengths)]
    )
    position = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    logical_swa = position[:, None] - (window - 1) + torch.arange(window, device="cuda")[None, :]
    swa_page = swa_table[request[:, None], logical_swa.clamp_min(0) // page].long()
    swa_slots = torch.where(
        (logical_swa >= 0) & (swa_page >= 0), swa_page * page + logical_swa % page, -1
    )
    swa_counts = [n + window - 1 for n in lengths]
    swa_offsets = [1]
    for n in swa_counts:
        swa_offsets.append(swa_offsets[-1] + n)
    total = swa_offsets[-1]
    main_offsets = [total] * (requests + 1)
    main_rows, main_table, main_pool, logical, mapping = 0, None, None, None, None
    if full:
        main_counts = [n // ratio for n in lengths]
        main_offsets = [total]
        for n in main_counts:
            main_offsets.append(main_offsets[-1] + n)
        main_rows = sum(main_counts)
        main_page = page // ratio
        pages = [max(1, math.ceil(n / main_page)) for n in main_counts]
        morder = torch.randperm(sum(pages) + 4, device="cuda", generator=g).int()
        main_table = torch.full((requests, max(pages)), -1, device="cuda", dtype=torch.int32)
        used = 0
        for r, n in enumerate(pages):
            main_table[r, :n] = morder[used : used + n]
            used += n
        main_pool = pack_rows(
            torch.randn(morder.numel() * main_page, dim, device="cuda", generator=g).bfloat16(),
            "main",
        )
        visible = (position + 1) // ratio
        scores = torch.rand(count, max(main_counts + [1]), device="cuda", generator=g)
        column = torch.arange(scores.shape[1], device="cuda")[None, :]
        scores = scores.masked_fill(column >= visible[:, None], -1.0)
        top = scores.topk(min(topk, scores.shape[1]), dim=1)
        logical = torch.where(top.values >= 0, top.indices, 1 << 30).sort(dim=1).values
        logical = torch.where(logical >= 1 << 30, -1, logical)
        logical = torch.nn.functional.pad(logical, (0, topk - logical.shape[1]), value=-1)
        logical = logical.int().contiguous()
        mapping = (main_table, main_page, 8192, request, visible)
    plan = CSA2SharedKVPlan(
        query_requests=request,
        query_positions=position.int(),
        swa_starts=torch.zeros(requests, device="cuda", dtype=torch.int64),
        swa_offsets=torch.tensor(swa_offsets, device="cuda", dtype=torch.int64),
        main_offsets=torch.tensor(main_offsets, device="cuda", dtype=torch.int64),
        swa_pages=swa_table,
        main_pages=main_table,
        swa_page_size=page,
        main_page_size=page // ratio,
        num_requests=requests,
        swa_rows=sum(swa_counts),
        main_rows=main_rows,
        swa_first=torch.tensor([0, *itertools.accumulate(lengths)][:-1], device="cuda"),
        swa_floors=torch.zeros(requests, device="cuda", dtype=torch.int64),
        window=window,
    )
    q = (torch.randn(count, heads, dim, device="cuda", generator=g) * 0.5).bfloat16()
    metadata = CSA2TrtllmMetadata.for_query_tile(
        q,
        topk if full else 0,
        staging_dtype=torch.float8_e4m3fn,
        context_lengths=list(lengths),
        shared_rows=1 + sum(swa_counts) + main_rows,
    )
    metadata.shared_plan = plan
    sink = torch.randn(heads, device="cuda", generator=g)
    inputs = CSA2BackendForwardArgs(
        swa_pool=swa_pool,
        swa_indices=swa_slots,
        main_pool=main_pool,
        topk_indices=logical,
        main_logical_indices=logical,
        main_mapping=mapping,
    )

    def call(attn):
        out = torch.empty(count, heads * dim, device="cuda", dtype=torch.bfloat16)
        args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.context_only,
            attention_sinks=sink,
            output=out,
            sparse_backend_args=inputs,
        )
        attn.forward(q.flatten(1), None, None, metadata, forward_args=args)
        return out.view(count, heads, dim)

    return call, metadata, q, sink


@skip_no_fused_context
@pytest.mark.parametrize(
    "lengths,full",
    [([300, 1, 129, 1500], True), ([5, 127, 640, 1], False)],
    ids=["lengths0-True", "lengths2-False"],
)
@torch.inference_mode()
def test_fused_context_matches_trtllm_gen(monkeypatch, lengths, full):
    """Shared-bank FP8 context tiles run the fused kernel on exactly the staged operands."""
    reference_attn = _backend(64, kv_cache_dtype="fp8_ds_mla")
    reference_attn.fused_context = False
    fused_attn = _backend(64, kv_cache_dtype="fp8_ds_mla")
    assert fused_attn.fused_context
    call, metadata, q, sink = _fused_context_call(lengths, full, seed=sum(lengths))
    op = torch.ops.trtllm.csa2_sparse_context
    calls = []

    def record(*args):
        calls.append(args[1].shape[0])
        return op(*args)

    monkeypatch.setattr(torch.ops.trtllm, "csa2_sparse_context", record)
    production = call(reference_attn)
    assert not calls
    count = q.shape[0]
    staged = (
        metadata.shared_pool.clone(),
        q,
        metadata.stage_kv_scales[0].clone(),
        metadata.prepared_indices[:count].clone(),
        metadata.prepared_lens[:count].clone(),
        metadata.mla_bmm1_scale.clone(),
        metadata.mla_bmm2_scale.clone(),
        sink,
    )
    fused = call(fused_attn)
    assert calls == [count]
    # The kernel reads exactly the staged operands, with Q quantized as the context quantizer does.
    direct = torch.empty_like(fused)
    op(*staged, direct)
    torch.cuda.synchronize()
    error = (fused.float() - production.float()).norm() / production.float().norm()
    assert error <= 2e-4
    assert torch.equal(fused.view(torch.int16), direct.view(torch.int16))
