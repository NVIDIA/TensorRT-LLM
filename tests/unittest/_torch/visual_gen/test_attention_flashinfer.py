# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and routing coverage for the VisualGen FlashInfer backend."""

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention_backend.flashinfer import FlashInferAttention
from tensorrt_llm._torch.visual_gen.attention_backend.utils import get_visual_gen_attention_backend
from tensorrt_llm.visual_gen.args import QuantAttentionConfig


@pytest.fixture
def require_flashinfer_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.fail("FlashInfer attention CI tests require CUDA.", pytrace=False)
    try:
        importlib.import_module("flashinfer")
    except (ImportError, OSError) as error:
        pytest.fail(f"FlashInfer is unavailable: {error}", pytrace=False)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    if query.shape[1] != key.shape[1]:
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    attention_mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        is_causal=is_causal,
    )
    return output.transpose(1, 2)


def _reference_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    is_causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.shape[2] != k.shape[2]:
        k = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    if is_causal:
        causal_mask = torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool).tril()
        scores.masked_fill_(~causal_mask, -torch.inf)
    if key_padding_mask is not None:
        scores.masked_fill_(~key_padding_mask[:, None, None, :], -torch.inf)
    return torch.logsumexp(scores, dim=-1)


def _require_sm12x_nvfp4() -> None:
    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (12, 1)):
        pytest.skip("This FlashInfer NVFP4 recipe requires SM120 or SM121.")
    flashinfer = importlib.import_module("flashinfer")
    required_apis = ("nvfp4_attention_sm120_quantize_qkv", "nvfp4_attention_sm120_fwd")
    missing_apis = [name for name in required_apis if not hasattr(flashinfer, name)]
    if missing_apis:
        pytest.fail(
            "The installed FlashInfer is missing required SM12x NVFP4 APIs: "
            + ", ".join(missing_apis),
            pytrace=False,
        )


@pytest.mark.cpu_only
def test_flashinfer_backend_is_registered() -> None:
    assert get_visual_gen_attention_backend("FLASHINFER") is FlashInferAttention


@pytest.mark.parametrize(
    (
        "dtype",
        "batch_size",
        "query_length",
        "key_value_length",
        "num_kv_heads",
        "attention_mask",
        "use_padding",
    ),
    (
        pytest.param(
            torch.bfloat16, 1, 47, 47, 4, PredefinedAttentionMask.FULL, False, id="bf16_self"
        ),
        pytest.param(
            torch.bfloat16,
            2,
            47,
            47,
            4,
            PredefinedAttentionMask.FULL,
            True,
            id="bf16_self_padding",
        ),
        pytest.param(
            torch.bfloat16,
            2,
            47,
            53,
            4,
            PredefinedAttentionMask.FULL,
            True,
            id="bf16_cross_padding",
        ),
        pytest.param(
            torch.bfloat16,
            2,
            47,
            47,
            2,
            PredefinedAttentionMask.CAUSAL,
            False,
            id="bf16_causal_gqa",
        ),
        pytest.param(
            torch.float16, 2, 47, 47, 4, PredefinedAttentionMask.FULL, False, id="fp16_self"
        ),
    ),
)
@pytest.mark.usefixtures("require_flashinfer_cuda")
def test_flashinfer_attention_matches_sdpa(
    dtype: torch.dtype,
    batch_size: int,
    query_length: int,
    key_value_length: int,
    num_kv_heads: int,
    attention_mask: PredefinedAttentionMask,
    use_padding: bool,
) -> None:
    torch.manual_seed(0)
    q = torch.randn((batch_size, query_length, 4, 64), device="cuda", dtype=dtype) * 0.2
    k = (
        torch.randn(
            (batch_size, key_value_length, num_kv_heads, 64),
            device="cuda",
            dtype=dtype,
        )
        * 0.2
    )
    v = torch.randn_like(k)
    key_padding_mask = None
    if use_padding:
        key_padding_mask = torch.ones(
            (batch_size, key_value_length), device="cuda", dtype=torch.bool
        )
        key_padding_mask[0, -4:] = False
        if batch_size > 1:
            key_padding_mask[1, -2:] = False
    attention = FlashInferAttention(num_heads=4, num_kv_heads=num_kv_heads, head_dim=64)

    output, lse = attention.forward_with_lse(
        q,
        k,
        v,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
    )
    reference = _reference_attention(
        q,
        k,
        v,
        is_causal=attention_mask == PredefinedAttentionMask.CAUSAL,
        key_padding_mask=key_padding_mask,
    )

    torch.testing.assert_close(output, reference, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(
        lse,
        _reference_lse(
            q,
            k,
            is_causal=attention_mask == PredefinedAttentionMask.CAUSAL,
            key_padding_mask=key_padding_mask,
        ),
        rtol=0,
        atol=6e-2,
    )


@pytest.mark.usefixtures("require_flashinfer_cuda")
def test_flashinfer_rejects_causal_cross_attention() -> None:
    q = torch.randn((1, 47, 4, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 53, 4, 64), device="cuda", dtype=torch.bfloat16)
    attention = FlashInferAttention(num_heads=4, head_dim=64)

    with pytest.raises(ValueError, match="equal Q and K/V lengths"):
        attention(q, k, k, attention_mask=PredefinedAttentionMask.CAUSAL)


@pytest.mark.usefixtures("require_flashinfer_cuda")
def test_flashinfer_rejects_causal_attention_with_padding_mask() -> None:
    q = torch.randn((1, 47, 4, 64), device="cuda", dtype=torch.bfloat16)
    key_padding_mask = torch.ones((1, 47), device="cuda", dtype=torch.bool)
    attention = FlashInferAttention(num_heads=4, head_dim=64)

    with pytest.raises(ValueError, match="does not combine causal and padding masks"):
        attention(
            q,
            q,
            q,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            key_padding_mask=key_padding_mask,
        )


@pytest.mark.parametrize(
    ("num_kv_heads", "attention_mask"),
    (
        pytest.param(2, PredefinedAttentionMask.FULL, id="mha"),
        pytest.param(1, PredefinedAttentionMask.CAUSAL, id="causal_gqa"),
    ),
)
@pytest.mark.parametrize("qk_dtype", ("mxfp8", "nvfp4"))
@pytest.mark.usefixtures("require_flashinfer_cuda")
def test_flashinfer_blockscaled_attention_sm10x_matches_sdpa(
    qk_dtype: str,
    num_kv_heads: int,
    attention_mask: PredefinedAttentionMask,
) -> None:
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("This FlashInfer block-scaled recipe requires SM100 or SM103.")

    torch.manual_seed(2)
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn((1, 129, num_kv_heads, 128), device="cuda", dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k)
    attention = FlashInferAttention(
        num_heads=2,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        quant_attention_config=QuantAttentionConfig(qk_dtype=qk_dtype, v_dtype="fp8"),
    )

    output, lse = attention.forward_with_lse(q, k, v, attention_mask=attention_mask)
    is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
    reference = _reference_attention(
        q,
        k,
        v,
        is_causal=is_causal,
    )
    cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)

    assert cosine > 0.98
    torch.testing.assert_close(lse, _reference_lse(q, k, is_causal=is_causal), rtol=0, atol=6e-2)


@pytest.mark.parametrize(
    "attention_mask",
    (
        pytest.param(PredefinedAttentionMask.FULL, id="full"),
        pytest.param(PredefinedAttentionMask.CAUSAL, id="causal"),
    ),
)
@pytest.mark.usefixtures("require_flashinfer_cuda")
def test_flashinfer_nvfp4_attention_sm12x_matches_sdpa(
    attention_mask: PredefinedAttentionMask,
) -> None:
    _require_sm12x_nvfp4()

    torch.manual_seed(3)
    q = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attention = FlashInferAttention(
        num_heads=2,
        head_dim=128,
        quant_attention_config=QuantAttentionConfig(qk_dtype="nvfp4", v_dtype="nvfp4"),
    )

    output, lse = attention.forward_with_lse(q, k, v, attention_mask=attention_mask)
    is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
    reference = _reference_attention(q, k, v, is_causal=is_causal)
    cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)

    assert cosine > 0.98
    # Early causal rows do not average V quantization error; cosine measures output parity.
    if not is_causal:
        torch.testing.assert_close(output, reference, rtol=5e-2, atol=7e-2)
    torch.testing.assert_close(lse, _reference_lse(q, k, is_causal=is_causal), rtol=0, atol=6e-2)


@pytest.mark.usefixtures("require_flashinfer_cuda")
def test_flashinfer_nvfp4_attention_sm12x_rejects_unaligned_sequence_length() -> None:
    _require_sm12x_nvfp4()
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16)
    attention = FlashInferAttention(
        num_heads=2,
        head_dim=128,
        quant_attention_config=QuantAttentionConfig(qk_dtype="nvfp4", v_dtype="nvfp4"),
    )

    with pytest.raises(ValueError, match="multiple of 128"):
        attention(q, q, q)


@pytest.mark.usefixtures("require_flashinfer_cuda")
def test_flashinfer_mxfp8_attention_is_rejected_on_sm12x() -> None:
    capability = torch.cuda.get_device_capability()
    if capability not in ((12, 0), (12, 1)):
        pytest.skip("This FlashInfer MXFP8 rejection test requires SM120 or SM121.")

    q = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    attention = FlashInferAttention(
        num_heads=2,
        head_dim=128,
        quant_attention_config=QuantAttentionConfig(qk_dtype="mxfp8", v_dtype="fp8"),
    )

    with pytest.raises(RuntimeError, match="qk_dtype='mxfp8'.*SM100/SM103"):
        attention(q, q, q)
