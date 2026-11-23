# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MLA paths at the Hy4-Flash sliding-window geometry (qk 192+64, v 256, window 512):
windowed context (packed-QKV repack, grouped batches), windowed decode, FP8 context quantize-copy.

The windowed tests also require the same operands without a window to land outside the band, so a
kernel that ignored the window cannot pass.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import get_sm_version, is_sm_100f, str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

NUM_HEADS = 64
Q_LORA_RANK = 1024
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 256
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
WINDOW = 512
ROPE_THETA = 10_000.0
MAX_POSITIONS = 4096
TOKENS_PER_BLOCK = 32
FP8_MAX = 448.0

# The band `test_attention_mla.py` holds bf16 MLA output to.
ATOL = 3e-2
RTOL = 3e-3


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    sm = get_sm_version()
    if not is_sm_100f(sm):
        pytest.skip(f"the trtllm-gen MLA paths tested here are SM100-family; this device is SM{sm}")


def _rope_rows(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Adjacent-pair (GPT-J) rotation of row i at position i, from the layer's `rotary_cos_sin`."""
    rows, dim = x.shape
    half = dim // 2
    view = table.view(-1, dim, 2)
    cos = view[:rows, :half, 0]
    sin = view[:rows, :half, 1]
    pairs = x.float().reshape(rows, half, 2)
    out = torch.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
    out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
    return out.reshape(rows, dim).to(x.dtype)


def _make_layer(kv_cache_quant_algo: str | None, *, head_dim: int, num_kv_heads: int):
    # `model_type` keeps DeepSeek V3's yarn fallback off: these layers use plain RoPE.
    rope_config = SimpleNamespace(
        hidden_size=QK_HEAD_DIM * NUM_HEADS,
        num_attention_heads=NUM_HEADS,
        max_position_embeddings=MAX_POSITIONS,
        rope_theta=ROPE_THETA,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        model_type="hy_v4_flash",
    )
    return get_attention_backend("TRTLLM")(
        layer_idx=0,
        num_heads=NUM_HEADS,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        quant_config=(
            QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)
            if kv_cache_quant_algo is not None
            else None
        ),
        q_scaling=1.0,
        pos_embd_params=PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams.from_config(rope_config),
            is_neox=False,
        ),
        mla_params=MLAParams(
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            predicted_tokens_per_seq=1,
        ),
    )


def _make_manager(
    total_tokens: int,
    max_seq_len: int,
    *,
    dtype: torch.dtype,
    window: int | None = None,
    max_batch_size: int = 1,
) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=-(-total_tokens // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK,
            enable_block_reuse=False,
            max_attention_window=None if window is None else [window],
            # No host tier: these tests read only the device pool.
            host_cache_size=0,
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=LATENT_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=str_dtype_to_binding(torch_dtype_to_str(dtype)),
    )


def _context_metadata(manager, seq_lens, cached_lens):
    metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(range(len(seq_lens))),
        max_num_requests=len(seq_lens),
        num_contexts=len(seq_lens),
        prompt_lens=list(seq_lens),
        max_num_tokens=sum(seq_lens),
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        enable_context_mla_with_cached_kv=True,
    )
    metadata.prepare()
    return metadata


def _generation_metadata(manager, seq_lens, cached_lens, prompt_lens):
    metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(range(len(seq_lens))),
        max_num_requests=len(seq_lens),
        num_contexts=0,
        prompt_lens=list(prompt_lens),
        max_num_tokens=max(sum(prompt_lens), sum(seq_lens)),
        kv_cache_manager=manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
    )
    metadata.prepare()
    return metadata


def _join_k(k_nope: torch.Tensor, roped_pe: torch.Tensor) -> torch.Tensor:
    """[k_nope | k_pe] per head, the shared rope segment broadcast to every head."""
    rows = k_nope.shape[0]
    return torch.cat(
        (
            k_nope.reshape(rows, NUM_HEADS, QK_NOPE_HEAD_DIM),
            roped_pe.view(rows, 1, QK_ROPE_HEAD_DIM).expand(-1, NUM_HEADS, -1),
        ),
        dim=-1,
    ).reshape(rows, NUM_HEADS * QK_HEAD_DIM)


def _out_of_band(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Fraction of elements outside the accuracy band."""
    diff = (out.float() - ref.float()).abs()
    return (diff > ATOL + RTOL * ref.float().abs()).float().mean().item()


def _context_reference(q, k, v, q_lens, kv_lens, window: int | None) -> torch.Tensor:
    """Causal attention; query i sits at position kv_len - q_len + i and reads the last `window` keys."""
    scale = 1.0 / math.sqrt(QK_HEAD_DIM)
    outs = []
    q_off = kv_off = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        qs = q[q_off : q_off + q_len].view(q_len, NUM_HEADS, QK_HEAD_DIM).float()
        ks = k[kv_off : kv_off + kv_len].view(kv_len, NUM_HEADS, QK_HEAD_DIM).float()
        vs = v[kv_off : kv_off + kv_len].view(kv_len, NUM_HEADS, V_HEAD_DIM).float()
        for i in range(q_len):
            pos = kv_len - q_len + i
            lo = max(0, pos - window + 1) if window else 0
            scores = torch.einsum("hd,khd->hk", qs[i], ks[lo : pos + 1]) * scale
            probs = torch.softmax(scores, dim=-1)
            outs.append(torch.einsum("hk,khv->hv", probs, vs[lo : pos + 1]))
        q_off += q_len
        kv_off += kv_len
    return torch.cat(outs).reshape(-1, NUM_HEADS * V_HEAD_DIM).to(torch.bfloat16)


def _generation_reference(fused_q, k_rows, window: int | None) -> torch.Tensor:
    """Decode over latent rows [cKV | rope(k_pe)] (values: the cKV columns), newest `window` only."""
    scale = 1.0 / math.sqrt(QK_HEAD_DIM)
    q = fused_q.float().view(-1, NUM_HEADS, LATENT_DIM)
    ks = k_rows.float()
    if window is not None:
        ks = ks[-window:]
    vs = ks[:, :KV_LORA_RANK]
    outs = []
    for i in range(q.shape[0]):
        probs = torch.softmax(torch.einsum("hd,kd->hk", q[i], ks) * scale, dim=-1)
        outs.append(torch.einsum("hk,kv->hv", probs, vs))
    return torch.cat(outs).reshape(-1, NUM_HEADS * KV_LORA_RANK).to(torch.bfloat16)


@pytest.mark.parametrize(
    "seed,cached_lens,new_lens",
    [
        # One request, kv 664 > window: a single packed group under the windowed mask.
        (11, [600], [64]),
        # Ten requests overflow one group's row budget: the long one runs windowed, the
        # short ones (history < window) dense in their own group.
        (19, [600] + [200] * 9, [16] * 10),
    ],
    ids=["single", "grouped"],
)
def test_windowed_context(seed, cached_lens, new_lens) -> None:
    _require_sm100()
    torch.manual_seed(seed)
    kv_lens = [cached + new for cached, new in zip(cached_lens, new_lens)]
    total = sum(kv_lens)
    pool_tokens = sum(-(-kv // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK for kv in kv_lens)
    layer = _make_layer(None, head_dim=QK_HEAD_DIM, num_kv_heads=NUM_HEADS)
    manager = _make_manager(
        pool_tokens, max(kv_lens), dtype=torch.bfloat16, max_batch_size=len(new_lens)
    )
    try:
        for rid, kv_len in enumerate(kv_lens):
            kv_cache = manager._create_kv_cache(rid, None, None)
            assert kv_cache.resume(torch.cuda.current_stream().cuda_stream)
            kv_cache.capacity = kv_len
        metadata = _context_metadata(manager, new_lens, cached_lens)

        # With `latent_cache=None` the op applies no RoPE, so K's rope segment is rotated here.
        q = torch.randn(sum(new_lens), NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_nope = torch.randn(
            total, NUM_HEADS * QK_NOPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        k_pe = torch.randn(total, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(total, NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k = _join_k(k_nope, _rope_rows(k_pe, layer.rotary_cos_sin)).contiguous()

        out = layer.forward(
            q.clone(),
            k,
            v,
            metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                attention_window_size=WINDOW,
            ),
        )
        ref = _context_reference(q, k, v, new_lens, kv_lens, WINDOW)
        full = _context_reference(q, k, v, new_lens, kv_lens, None)
        torch.cuda.synchronize()

        assert out.abs().sum() > 0
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
        # Only the first request's history exceeds the window.
        long_rows = new_lens[0]
        assert _out_of_band(out[:long_rows], full[:long_rows]) > 0.05, "window not applied"
    finally:
        manager.shutdown()


def test_windowed_decode() -> None:
    """Decode at position 600 with the window from the KV manager: the pool keeps all 601 rows,
    only the mask drops the aged-out ones."""
    _require_sm100()
    torch.manual_seed(13)
    prefill, total = 600, 601
    layer = _make_layer(None, head_dim=LATENT_DIM, num_kv_heads=1)
    manager = _make_manager(total, total, dtype=torch.bfloat16, window=WINDOW)
    try:
        kv_cache = manager._create_kv_cache(0, None, None)
        assert kv_cache.resume(torch.cuda.current_stream().cuda_stream)
        kv_cache.capacity = prefill

        q_ctx = torch.randn(prefill, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent_ctx = torch.randn(prefill, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        # The append rotates k_pe in place in its `latent_cache` argument; keep a pre-call copy.
        latent_written = latent_ctx.clone()
        layer.mla_rope_append_paged_kv_assign_q(
            q_ctx, latent_ctx, _context_metadata(manager, [prefill], [0])
        )

        kv_cache.capacity = total
        metadata = _generation_metadata(manager, [1], [prefill], [prefill])
        fused_q = torch.randn(1, NUM_HEADS * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        q_pe = torch.randn(1, NUM_HEADS, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent = torch.randn(1, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        latent_new = latent.clone()
        cu_q_seqlens = torch.empty(2, dtype=torch.int32, device="cuda")
        cu_kv_seqlens = torch.empty(2, dtype=torch.int32, device="cuda")
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device="cuda")
        # Rotates q_pe into fused_q in place and appends the new latent row.
        layer.mla_rope_generation(
            fused_q,
            q_pe,
            latent,
            metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            None,
            None,
            None,
        )

        out = layer.forward(
            fused_q,
            None,
            None,
            metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent,
            q_pe=q_pe,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=None,
            mla_bmm2_scale=None,
            quant_q_buffer=None,
        )

        rows = torch.cat([latent_written, latent_new], dim=0)
        k_rows = torch.cat(
            [rows[:, :KV_LORA_RANK], _rope_rows(rows[:, KV_LORA_RANK:], layer.rotary_cos_sin)],
            dim=-1,
        )
        ref = _generation_reference(fused_q, k_rows, WINDOW)
        full = _generation_reference(fused_q, k_rows, None)
        torch.cuda.synchronize()

        assert out.abs().sum() > 0
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
        assert _out_of_band(out, full) > 0.05, "window not applied"
    finally:
        manager.shutdown()


def _to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """The op's unit-scale copy: round to nearest, saturating."""
    return x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def test_fp8_context_quantize_copy() -> None:
    """FP8 context at 192/64/256: the op's own Q/K/V quantize-copy must match caller-quantized
    inputs bit for bit, and track a BF16-KV layer within FP8 noise."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    fp8_layer = _make_layer(QuantAlgo.FP8.value, head_dim=QK_HEAD_DIM, num_kv_heads=NUM_HEADS)
    if not fp8_layer.has_fp8_context_mla:
        pytest.skip("this SM has no FP8 context MLA FMHA")
    bf16_layer = _make_layer(None, head_dim=QK_HEAD_DIM, num_kv_heads=NUM_HEADS)
    scale = fp8_layer.kv_scale_orig_quant
    # `_to_e4m3` assumes unit scale.
    assert isinstance(scale, torch.Tensor) and bool(torch.all(scale == 1.0))

    torch.manual_seed(17)
    cached, new = 256, 128
    total = cached + new
    fp8_manager = _make_manager(total, total, dtype=torch.float8_e4m3fn)
    bf16_manager = _make_manager(total, total, dtype=torch.bfloat16)

    def random_inputs(num_tokens):
        q = torch.randn(num_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent = torch.randn(num_tokens, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        return q, latent

    cached_q, cached_latent = random_inputs(cached)
    new_q, new_latent = random_inputs(new)
    kv_b = (
        torch.randn(
            KV_LORA_RANK,
            NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * KV_LORA_RANK**-0.5
    )

    def prime(layer, manager):
        """Append the cached and new tokens; return the new tokens' rope'd q and metadata."""
        kv_cache = manager._create_kv_cache(0, None, None)
        assert kv_cache.resume(torch.cuda.current_stream().cuda_stream)
        kv_cache.capacity = cached
        layer.mla_rope_append_paged_kv_assign_q(
            cached_q.clone(), cached_latent.clone(), _context_metadata(manager, [cached], [0])
        )
        kv_cache.capacity = total
        metadata = _context_metadata(manager, [new], [cached])
        q = new_q.clone()
        layer.mla_rope_append_paged_kv_assign_q(q, new_latent.clone(), metadata)
        return q, metadata

    try:
        _, fp8_metadata = prime(fp8_layer, fp8_manager)
        q, bf16_metadata = prime(bf16_layer, bf16_manager)

        compressed_kv, k_pe = bf16_layer.load_paged_kv_cache_for_mla(bf16_metadata, torch.bfloat16)
        k_nope, v = (compressed_kv @ kv_b).split(
            [NUM_HEADS * QK_NOPE_HEAD_DIM, NUM_HEADS * V_HEAD_DIM], -1
        )
        k = _join_k(k_nope, k_pe)
        v = v.contiguous()

        def run(layer, key, value, metadata, **fp8_q_args):
            return layer.forward(
                q.clone(),
                key,
                value,
                metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                    **fp8_q_args,
                ),
            )

        bf16_out = run(bf16_layer, k, v, bf16_metadata)
        # BF16 K/V: the op runs its own 192/64/256 quantize-copy.
        fp8_copy_out = run(fp8_layer, k, v, fp8_metadata)
        # Caller-quantized Q/K/V: the op skips the copies.
        fp8_out = run(
            fp8_layer,
            _to_e4m3(k),
            _to_e4m3(v),
            fp8_metadata,
            quant_q_buffer=_to_e4m3(q),
            quant_scale_qkv=scale,
        )
        torch.cuda.synchronize()

        assert bf16_out.abs().sum() > 0
        # A mis-laid-out copy can stay inside a loose band, so require bit equality.
        assert torch.equal(fp8_copy_out.view(torch.int16), fp8_out.view(torch.int16))
        relative_error = (fp8_out.float() - bf16_out.float()).norm() / bf16_out.float().norm()
        assert relative_error < 0.1, f"FP8 context is {relative_error.item():.3f} from BF16-KV"
    finally:
        fp8_manager.shutdown()
        bf16_manager.shutdown()
