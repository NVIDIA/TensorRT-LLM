# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise INT8 cache writes and reads through the native attention backend."""

from dataclasses import replace

import pytest
import torch
from backend_case import (
    BackendCase,
    _assert_cache_contains_new_tokens,
    _build_kv_cache_manager,
    generate_inputs,
)

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_PROMPT_LENS = [63, 31]
_DECODE_STEPS = 3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Native INT8 conversion uses round-to-nearest-even with signed saturation.
    """Match native signed INT8 rounding and saturation for exact cache assertions."""
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8)


def _metadata(
    case: BackendCase, manager: KVCacheManager, prompt_lens: list[int]
) -> TrtllmAttentionMetadata:
    """Prepare actual cache-manager metadata for a context or decode batch."""
    metadata = TrtllmAttentionMetadata(
        num_contexts=case.num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=case.num_cached_tokens
        ),
        seq_lens=torch.tensor(case.seq_lens, dtype=torch.int32),
        max_num_requests=case.num_seqs,
        max_num_tokens=case.max_num_tokens,
        kv_cache_manager=manager,
        request_ids=list(range(case.num_seqs)),
        prompt_lens=prompt_lens,
        kv_layout="HND",
    )
    metadata.prepare()
    assert not metadata.use_paged_context_fmha
    return metadata


def _backend(case: BackendCase) -> TrtllmAttention:
    """Construct the production TRTLLM backend with INT8 KV quantization enabled."""
    return create_attention(
        "TRTLLM",
        layer_idx=0,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=case.head_dim,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.INT8),
    )


def _forward(
    attention: TrtllmAttention,
    metadata: TrtllmAttentionMetadata,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Pass packed QKV and explicit reciprocal scales through native attention."""
    forward_args = AttentionForwardArgs(
        attention_mask=PredefinedAttentionMask.CAUSAL,
        kv_scale_orig_quant=scale.reciprocal(),
        kv_scale_quant_orig=scale,
    )
    qkv = torch.cat((q, k, v), dim=-1)
    result = attention.forward(qkv, None, None, metadata, forward_args=forward_args)
    if isinstance(result, tuple):
        result = result[0]
    return result[: q.shape[0]]


def _reference(
    case: BackendCase,
    q: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> torch.Tensor:
    """Compute causal attention in float32 over the explicit logical cache."""
    outputs = []
    offset = 0
    repeats = case.num_heads // case.num_kv_heads
    for q_len, cached_len, key, value in zip(
        case.seq_lens, case.num_cached_tokens, keys, values, strict=True
    ):
        query = q[offset : offset + q_len].view(q_len, case.num_heads, case.head_dim)
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        scores = query.float().transpose(0, 1) @ key.float().permute(1, 2, 0)
        scores *= case.head_dim**-0.5
        q_positions = torch.arange(q_len, device=q.device) + cached_len
        kv_positions = torch.arange(key.shape[0], device=q.device)
        scores.masked_fill_(kv_positions[None, :] > q_positions[:, None], -torch.inf)
        output = scores.softmax(dim=-1) @ value.float().transpose(0, 1)
        outputs.append(output.transpose(0, 1).reshape(q_len, -1))
        offset += q_len
    return torch.cat(outputs)


def _assert_cache(
    case: BackendCase,
    manager: KVCacheManager,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> None:
    """Compare the complete logical K/V cache against expected integer contents."""
    _assert_cache_contains_new_tokens(
        manager,
        0,
        list(range(case.num_seqs)),
        case.token_nums,
        [0] * case.num_seqs,
        [torch.stack((key, value)) for key, value in zip(keys, values, strict=True)],
        kv_layout="HND",
        cache_kind="kv",
    )


@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("scale_value", [1 / 32, 1 / 16], ids=["scale32", "scale16"])
@torch.inference_mode()
def test_int8_kv_cache_prefill_and_decode(
    dtype: str, num_kv_heads: int, scale_value: float, use_v2: bool
) -> None:
    """Verify real cache writes, saturation, and reads across a page boundary."""
    case = BackendCase(
        num_heads=4,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        seq_lens=_PROMPT_LENS,
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype=dtype,
        page_size=64,
        use_kv_cache_manager_v2=use_v2,
    )
    allocation = replace(case, seq_lens=[n + _DECODE_STEPS for n in _PROMPT_LENS])
    manager = _build_kv_cache_manager(allocation, "TRTLLM", torch.int8)
    reference_manager = _build_kv_cache_manager(allocation, "TRTLLM", case.compute_dtype)
    try:
        request_ids = list(range(case.num_seqs))
        manager.add_dummy_requests(request_ids, allocation.token_nums)
        cache = manager.get_buffers(0, kv_layout="HND")
        reference_cache = reference_manager.get_buffers(0, kv_layout="HND")
        assert cache.dtype == torch.int8
        assert cache.shape[1:] == reference_cache.shape[1:]
        assert cache[0].nbytes * 2 == reference_cache[0].nbytes
        if not use_v2:
            assert cache.shape == reference_cache.shape
            assert cache.nbytes * 2 == reference_cache.nbytes
        # V2 rounds the backing allocation to an arena size, so the same
        # minimum-size arena can expose more INT8 pages than FP16/BF16 pages.
        cache.zero_()

        attention = _backend(case)
        scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")
        inputs = generate_inputs(case, seed=17)
        q, k, v = inputs["q"], inputs["new_k"], inputs["new_v"]
        # Cover both saturation directions and ties-to-even in the cache writes.
        k[0, :4] = torch.tensor([20, -20, 2.5 * scale_value, 3.5 * scale_value])
        v[0, :4] = torch.tensor([-20, 20, -2.5 * scale_value, -3.5 * scale_value])
        keys = [x.view(-1, num_kv_heads, case.head_dim) for x in k.split(case.seq_lens)]
        values = [x.view(-1, num_kv_heads, case.head_dim) for x in v.split(case.seq_lens)]

        metadata = _metadata(case, manager, _PROMPT_LENS)
        actual = _forward(attention, metadata, q, k, v, scale)
        expected = _reference(case, q, keys, values)
        atol, rtol = (0.04, 0.01) if dtype == "bfloat16" else (0.015, 0.005)
        torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)

        quantized_keys = [_quantize(key, scale) for key in keys]
        quantized_values = [_quantize(value, scale) for value in values]
        _assert_cache(case, manager, quantized_keys, quantized_values)

        for step in range(_DECODE_STEPS):
            decode = replace(
                case,
                seq_lens=[1, 1],
                num_cached_tokens=[n + step for n in _PROMPT_LENS],
                num_contexts=0,
            )
            inputs = generate_inputs(decode, seed=31 + step)
            q = inputs["q"]
            # Put the new token on the quantization grid: MMHA may use the
            # unquantized new K/V while XQA reads it back from the cache.
            k = (_quantize(inputs["new_k"], scale).float() * scale).to(case.compute_dtype)
            v = (_quantize(inputs["new_v"], scale).float() * scale).to(case.compute_dtype)
            for index in request_ids:
                new_key = _quantize(k[index].view(1, num_kv_heads, case.head_dim), scale)
                new_value = _quantize(v[index].view(1, num_kv_heads, case.head_dim), scale)
                quantized_keys[index] = torch.cat((quantized_keys[index], new_key))
                quantized_values[index] = torch.cat((quantized_values[index], new_value))

            metadata = _metadata(decode, manager, _PROMPT_LENS)
            actual = _forward(attention, metadata, q, k, v, scale)
            expected = _reference(
                decode,
                q,
                [key.float() * scale for key in quantized_keys],
                [value.float() * scale for value in quantized_values],
            )
            torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
            _assert_cache(decode, manager, quantized_keys, quantized_values)
    finally:
        reference_manager.shutdown()
        manager.shutdown()


@pytest.mark.parametrize("cached_tokens,paged_context", [(7, False), (0, True), (7, True)])
@torch.inference_mode()
def test_int8_kv_cache_rejects_cached_context(cached_tokens: int, paged_context: bool) -> None:
    """Cached context must fail with either packed or paged-context metadata."""
    case = BackendCase(
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[2],
        num_cached_tokens=[cached_tokens],
        num_contexts=1,
        page_size=64,
    )
    manager = _build_kv_cache_manager(case, "TRTLLM", torch.int8)
    try:
        manager.add_dummy_requests([0], case.token_nums)
        manager.get_buffers(0, kv_layout="HND").zero_()
        attention = _backend(case)
        metadata = _metadata(case, manager, case.token_nums)
        metadata.use_paged_context_fmha = paged_context
        inputs = generate_inputs(case, seed=23)
        scale = torch.tensor([1 / 32], dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="INT8"):
            _forward(attention, metadata, inputs["q"], inputs["new_k"], inputs["new_v"], scale)
    finally:
        manager.shutdown()


@torch.inference_mode()
def test_int8_kv_mixed_prefill_and_decode() -> None:
    # Cached tokens are valid for the decode request in a mixed batch.
    """Cached decode tokens remain valid beside an uncached context request."""
    case = BackendCase(
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[5, 1],
        num_cached_tokens=[0, 7],
        num_contexts=1,
        page_size=64,
    )
    manager = _build_kv_cache_manager(case, "TRTLLM", torch.int8)
    try:
        manager.add_dummy_requests([0, 1], case.token_nums)
        manager.get_buffers(0, kv_layout="HND").zero_()
        scale = torch.tensor([1 / 32], dtype=torch.float32, device="cuda")
        inputs = generate_inputs(case, seed=42)
        q = inputs["q"]
        k = (_quantize(inputs["new_k"], scale).float() * scale).to(case.compute_dtype)
        v = (_quantize(inputs["new_v"], scale).float() * scale).to(case.compute_dtype)
        keys = [
            k[:5].view(5, 2, 128),
            torch.cat((torch.zeros(7, 2, 128, device="cuda"), k[5:].view(1, 2, 128))),
        ]
        values = [
            v[:5].view(5, 2, 128),
            torch.cat((torch.zeros(7, 2, 128, device="cuda"), v[5:].view(1, 2, 128))),
        ]
        actual = _forward(_backend(case), _metadata(case, manager, [5, 7]), q, k, v, scale)
        torch.testing.assert_close(
            actual.float(), _reference(case, q, keys, values), atol=0.015, rtol=0.005
        )
        _assert_cache(
            case,
            manager,
            [_quantize(x, scale) for x in keys],
            [_quantize(x, scale) for x in values],
        )
    finally:
        manager.shutdown()


@pytest.mark.parametrize("scale_kind", ["missing", "cpu", "float16", "vector"])
@pytest.mark.parametrize("scale_field", ["kv_scale_orig_quant", "kv_scale_quant_orig"])
@pytest.mark.parametrize("after_warmup", [False, True])
@torch.inference_mode()
def test_int8_kv_rejects_invalid_scale_tensor(
    scale_kind: str, scale_field: str, after_warmup: bool
) -> None:
    """Validate each scale independently, including replacements after warmup."""
    case = BackendCase(
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[2],
        num_cached_tokens=[0],
        num_contexts=1,
        page_size=64,
    )
    manager = _build_kv_cache_manager(case, "TRTLLM", torch.int8)
    try:
        manager.add_dummy_requests([0], case.token_nums)
        inputs = generate_inputs(case, seed=29)
        attention = _backend(case)
        metadata = _metadata(case, manager, case.token_nums)
        scale = torch.tensor([1 / 32], dtype=torch.float32, device="cuda")
        forward_args = AttentionForwardArgs(
            attention_mask=PredefinedAttentionMask.CAUSAL,
            kv_scale_orig_quant=scale.reciprocal(),
            kv_scale_quant_orig=scale,
        )
        qkv = torch.cat((inputs["q"], inputs["new_k"], inputs["new_v"]), dim=-1)
        if after_warmup:
            attention.forward(qkv, None, None, metadata, forward_args=forward_args)
        invalid_scale = (
            None
            if scale_kind == "missing"
            else torch.ones(
                2 if scale_kind == "vector" else 1,
                dtype=torch.float16 if scale_kind == "float16" else torch.float32,
                device="cpu" if scale_kind == "cpu" else "cuda",
            )
        )
        setattr(forward_args, scale_field, invalid_scale)
        with pytest.raises(ValueError, match="scalar float32 KV scales"):
            attention.forward(qkv, None, None, metadata, forward_args=forward_args)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        ("float32", "FP16 or BF16"),
        ("missing_cache", "active KV cache"),
        ("inactive_cache", "active KV cache"),
        ("cross_attention", "cross-attention"),
        ("helix", "context parallelism"),
    ],
)
@pytest.mark.parametrize("after_warmup", [False, True])
@torch.inference_mode()
def test_int8_kv_rejects_unsupported_forward_inputs(
    invalid: str, error: str, after_warmup: bool
) -> None:
    """Each validation fails in the real backend before native attention runs."""
    case = BackendCase(
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        seq_lens=[2],
        num_cached_tokens=[0],
        num_contexts=1,
        page_size=64,
    )
    manager = _build_kv_cache_manager(case, "TRTLLM", torch.int8)
    try:
        manager.add_dummy_requests([0], case.token_nums)
        metadata = _metadata(case, manager, case.token_nums)
        inputs = generate_inputs(case, seed=29)
        q, k, v = inputs["q"], inputs["new_k"], inputs["new_v"]
        attention = _backend(case)
        scale = torch.tensor([1 / 32], dtype=torch.float32, device="cuda")
        if after_warmup:
            _forward(attention, metadata, q, k, v, scale)
        if invalid == "float32":
            q, k, v = q.float(), k.float(), v.float()
        elif invalid == "missing_cache":
            metadata.kv_cache_params = None
        elif invalid == "inactive_cache":
            metadata.kv_cache_params.use_cache = False
        elif invalid == "cross_attention":
            metadata.seq_lens_kv = metadata.seq_lens.clone()
            assert metadata.is_cross and not metadata.enable_helix
        else:
            metadata.enable_helix = True
            assert not metadata.is_cross
        with pytest.raises(ValueError, match=error):
            _forward(attention, metadata, q, k, v, scale)
    finally:
        manager.shutdown()
