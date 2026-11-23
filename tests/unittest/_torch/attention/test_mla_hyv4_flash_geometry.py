# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared MLA engine on a sliding-window MLA geometry (qk 192+64, v 256).

DeepSeek's MLA -- qk 128+64, v 128 -- is the shape the shared MLA paths were
written for. A sliding-window layer of Hy4-Flash runs MLA at qk 192+64, v 256
and kv_lora 512 under a 512-token window, and three engine paths meet a shape
and a mask that only this geometry produces:

* a context call whose kv length exceeds the layer's window. No separate-Q/K/V
  context kernel carries a windowed mask, so such a call has to reach one that
  does (the packed-QKV layout), and the dispatcher's probe has to see this
  call's mask rather than the fixed causal one;
* a decode whose history exceeds the window, whose mask has to drop the keys
  the window aged out;
* an e4m3 context pool, whose Q/K/V quantize-copy has to be instantiated at
  these head widths.

Each test compares the layer's output against a torch reference built from the
call's own operands. The two windowed tests also run the same operands without
a window and require that rival to land outside the band, so a kernel that
ignored the window cannot pass.

These are engine tests of the shared code, not target tests: the target's own
variant test lives under ``tests/unittest/_torch/modeling_v2/``.
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

# One Hy4-Flash sliding-window layer's attention geometry.
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
    """The trtllm-gen MLA context and generation kernels pinned here are SM100-family."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    sm = get_sm_version()
    if not is_sm_100f(sm):
        pytest.skip(f"the trtllm-gen MLA paths pinned here are SM100-family; this device is SM{sm}")


def _rope_rows(x: torch.Tensor, table: torch.Tensor, first_position: int = 0) -> torch.Tensor:
    """Rotate each row of `x` at its own absolute position.

    `table` is the layer's `rotary_cos_sin`: `2 * QK_ROPE_HEAD_DIM` floats per
    position in the duplicated layout, so `view(-1, QK_ROPE_HEAD_DIM, 2)` gives a
    (cos, sin) pair per slot and the rotation reads the first
    `QK_ROPE_HEAD_DIM // 2` slots -- the adjacent-pair (GPT-J) rotation the MLA
    kernels apply to the rope segment.
    """
    rows, dim = x.shape
    half = dim // 2
    view = table.view(-1, dim, 2)
    cos = view[first_position : first_position + rows, :half, 0]
    sin = view[first_position : first_position + rows, :half, 1]
    pairs = x.float().reshape(rows, half, 2)
    out = torch.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
    out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
    return out.reshape(rows, dim).to(x.dtype)


def _make_layer(kv_cache_quant_algo: str | None, *, head_dim: int, num_kv_heads: int):
    # Only the rope fields are read here; qk_rope_head_dim fixes the rotation
    # width and max_position_embeddings the table length. `model_type` is read
    # only to keep DeepSeek V3's yarn fallback off: these layers use plain RoPE.
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
    # A manager built here is outside the executor's overlays, so the gate
    # YAML's `kv_cache_config.host_cache_size` never reaches it: left unset,
    # the manager auto-provisions a host tier from the device quota. These
    # tests read only the paged pool, never the host tier, so they ask for
    # none: 0 runs without a host tier (see the manager's host-quota branch),
    # which also keeps an engine-bearing test from pinning host memory
    # beside another job on a shared node.
    return KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=-(-total_tokens // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK,
            enable_block_reuse=False,
            max_attention_window=None if window is None else [window],
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


def _out_of_band(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Fraction of elements outside the accuracy band."""
    diff = (out.float() - ref.float()).abs()
    limit = ATOL + RTOL * ref.float().abs()
    return (diff > limit).float().mean().item()


def _context_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_lens, kv_lens, window: int | None
) -> torch.Tensor:
    """Causal attention over explicit K/V with a left window of `window` keys.

    Query i of a sequence of `q_len` queries and `kv_len` keys sits at absolute
    position `kv_len - q_len + i` (the keys include the cached prefix) and
    attends keys `max(0, pos - window + 1) .. pos`; `window=None` is full causal.
    """
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


def _generation_reference(
    fused_q: torch.Tensor, k_rows: torch.Tensor, window: int | None
) -> torch.Tensor:
    """Decode attention over the latent pool.

    `fused_q` is `[tokens, num_heads, kv_lora_rank + qk_rope_head_dim]` with the
    rope segment already rotated; `k_rows` is the pool `[history, kv_lora_rank +
    qk_rope_head_dim]`, each row `[cKV | rope(k_pe)]`, and the values are its
    first `kv_lora_rank` columns. With a window, only the newest `window` rows
    are read.
    """
    scale = 1.0 / math.sqrt(QK_HEAD_DIM)
    q = fused_q.float().view(-1, NUM_HEADS, LATENT_DIM)
    ks = k_rows.float()
    if window is not None:
        ks = ks[-window:]
    vs = ks[:, :KV_LORA_RANK]
    outs = []
    for i in range(q.shape[0]):
        scores = torch.einsum("hd,kd->hk", q[i], ks) * scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hk,kv->hv", probs, vs))
    return torch.cat(outs).reshape(-1, NUM_HEADS * KV_LORA_RANK).to(torch.bfloat16)


def test_windowed_context_attends_only_inside_the_window() -> None:
    """A context call whose kv length exceeds the layer's window.

    One request with 600 cached keys and 64 new queries: the oldest query sits
    at position 600, so its window is keys 89..600, and the call's kv length
    (664) exceeds the window (512). The windowed mask has no separate-Q/K/V
    context kernel, so the call is repacked; either way it must attend only
    inside each query's window, and the same operands under a full causal mask
    must land far away.
    """
    _require_sm100()
    torch.manual_seed(11)
    cached, new = 600, 64
    total = cached + new
    layer = _make_layer(None, head_dim=QK_HEAD_DIM, num_kv_heads=NUM_HEADS)
    manager = _make_manager(total, total, dtype=torch.bfloat16)
    try:
        kv_cache = manager._create_kv_cache(0, None, None)
        assert kv_cache.resume(torch.cuda.current_stream().cuda_stream)
        kv_cache.capacity = total
        metadata = _context_metadata(manager, [new], [cached])

        # Explicit K/V: the keys carry the cached prefix, the queries only the
        # new rows. With `latent_cache=None` the op applies no RoPE, so K's rope
        # segment is rotated here with the layer's own table.
        q = torch.randn(new, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_nope = torch.randn(
            total, NUM_HEADS * QK_NOPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        k_pe = torch.randn(total, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(total, NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        roped_pe = _rope_rows(k_pe, layer.rotary_cos_sin)
        k = (
            torch.cat(
                (
                    k_nope.view(total, NUM_HEADS, QK_NOPE_HEAD_DIM),
                    roped_pe.view(total, 1, QK_ROPE_HEAD_DIM).expand(-1, NUM_HEADS, -1),
                ),
                dim=-1,
            )
            .reshape(total, NUM_HEADS * QK_HEAD_DIM)
            .contiguous()
        )

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
        ref = _context_reference(q, k, v, [new], [total], WINDOW)
        full = _context_reference(q, k, v, [new], [total], None)
        torch.cuda.synchronize()

        assert out.abs().sum() > 0
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
        assert _out_of_band(out, full) > 0.05, (
            "the same operands under a full causal mask are inside the band: the "
            "window is not applied"
        )
    finally:
        manager.shutdown()


def test_windowed_context_packs_a_batch_in_groups() -> None:
    """A windowed context call with several requests of different lengths.

    The call's kv length (the batch maximum) exceeds the window, so every request
    is repacked; a repacked block proportional to the batch times the window is
    too large for a prefill, so the requests are packed in groups and each group
    runs its own packed call. Ten requests, the first with 600 cached keys and
    the rest with 200, need more rows than one group holds: the first group's
    block is longer than the window and runs under the windowed mask, the second
    group's blocks are shorter and run dense within themselves. Every request
    must attend only inside its own window, so a request whose history is
    shorter than the window keeps all its keys and the long one drops its aged
    out keys -- and the same operands under a full causal mask must land far
    away.
    """
    _require_sm100()
    torch.manual_seed(19)
    cached_lens = [600] + [200] * 9
    new_lens = [16] * 10
    kv_lens = [cached + new for cached, new in zip(cached_lens, new_lens)]
    total = sum(kv_lens)
    # The manager holds one block per whole or partial block of each request.
    pool_tokens = sum(-(-kv // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK for kv in kv_lens)
    layer = _make_layer(None, head_dim=QK_HEAD_DIM, num_kv_heads=NUM_HEADS)
    manager = _make_manager(
        pool_tokens, max(kv_lens), dtype=torch.bfloat16, max_batch_size=len(new_lens)
    )
    try:
        # Each request's own cache: its block count follows its own KV length.
        for rid, kv_len in enumerate(kv_lens):
            kv_cache = manager._create_kv_cache(rid, None, None)
            assert kv_cache.resume(torch.cuda.current_stream().cuda_stream)
            kv_cache.capacity = kv_len
        metadata = _context_metadata(manager, new_lens, cached_lens)

        # Per request: its queries, and its keys carrying the cached prefix.
        q = torch.randn(sum(new_lens), NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k_nope = torch.randn(
            total, NUM_HEADS * QK_NOPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        k_pe = torch.randn(total, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(total, NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        roped_pe = _rope_rows(k_pe, layer.rotary_cos_sin)
        k = (
            torch.cat(
                (
                    k_nope.view(total, NUM_HEADS, QK_NOPE_HEAD_DIM),
                    roped_pe.view(total, 1, QK_ROPE_HEAD_DIM).expand(-1, NUM_HEADS, -1),
                ),
                dim=-1,
            )
            .reshape(total, NUM_HEADS * QK_HEAD_DIM)
            .contiguous()
        )

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
        # Only the long request's history exceeds the window, so the window's
        # effect is confined to its rows.
        long_rows = new_lens[0]
        assert _out_of_band(out[:long_rows], full[:long_rows]) > 0.05, (
            "the long request under a full causal mask is inside the band: the "
            "window is not applied"
        )
    finally:
        manager.shutdown()


def test_windowed_decode_attends_only_inside_the_window() -> None:
    """A decode whose history exceeds the layer's window.

    600 prefill tokens, then one decode token: the query at position 600 may
    read keys 89..600 only, and the layer's window comes from the KV manager
    (`max_attention_window`), the production configuration for a sliding-window
    layer. The manager gives this request blocks for its whole history (19
    pages), so the aged-out keys stay in the pool and only the mask drops them.
    The reference is built from the inputs rather than from that pool: the
    newest 512 latent rows, rotated with the layer's own table.
    """
    _require_sm100()
    torch.manual_seed(13)
    prefill, total = 600, 601
    layer = _make_layer(None, head_dim=LATENT_DIM, num_kv_heads=1)
    manager = _make_manager(total, total, dtype=torch.bfloat16, window=WINDOW)
    assert manager.max_attention_window_vec == [WINDOW]
    try:
        kv_cache = manager._create_kv_cache(0, None, None)
        assert kv_cache.resume(torch.cuda.current_stream().cuda_stream)
        kv_cache.capacity = prefill

        # Prefill: write 600 rope'd latent rows into the sliding-window pool.
        q_ctx = torch.randn(prefill, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent_ctx = torch.randn(prefill, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        # The append rotates the k_pe half and writes it back into its
        # `latent_cache` argument (the kernel keeps the rotated value there for
        # chunked prefill), so the reference below reads a pre-call copy.
        latent_written = latent_ctx.clone()
        layer.mla_rope_append_paged_kv_assign_q(
            q_ctx, latent_ctx, _context_metadata(manager, [prefill], [0])
        )

        # One decode token: its history (601) exceeds the window (512).
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

        # The pool rows the reference needs, from the inputs: [cKV | rope(k_pe)].
        table = layer.rotary_cos_sin
        rows = torch.cat([latent_written, latent_new], dim=0)
        k_rows = torch.cat(
            [rows[:, :KV_LORA_RANK], _rope_rows(rows[:, KV_LORA_RANK:], table)], dim=-1
        )
        ref = _generation_reference(fused_q, k_rows, WINDOW)
        full = _generation_reference(fused_q, k_rows, None)
        torch.cuda.synchronize()

        assert out.abs().sum() > 0
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
        assert _out_of_band(out, full) > 0.05, (
            "the same query against the whole history is inside the band: the window is not applied"
        )
    finally:
        manager.shutdown()


def _to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """What the op's unit-scale copy writes: round to nearest, saturating."""
    return x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def test_fp8_context_pool_quantizes_the_checkpoint_head_widths() -> None:
    """An e4m3 context pool at qk 192+64 and v 256.

    The FP8 context FMHA reads FP8 Q, K and V, and the op quantizes BF16 K / V
    itself with the cache scale. Those copies cover whole heads, so their kernel
    is instantiated per head width: at DeepSeek's 128/64/128 the checkpoint's
    192/64/256 operands come out mis-laid-out. Handing the op caller-quantized
    FP8 tensors (which skip the copies) must be bit-identical to handing it the
    BF16 inputs, and the FP8 run must track a BF16-KV layer on the same inputs
    within FP8 quantization noise.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    fp8_layer = _make_layer(QuantAlgo.FP8.value, head_dim=QK_HEAD_DIM, num_kv_heads=NUM_HEADS)
    if not fp8_layer.has_fp8_context_mla:
        pytest.skip("this SM has no FP8 context MLA FMHA")
    bf16_layer = _make_layer(None, head_dim=QK_HEAD_DIM, num_kv_heads=NUM_HEADS)
    scale = fp8_layer.kv_scale_orig_quant
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
        """Write this pair's share of the inputs into `manager`; return the
        forward's already rope'd q and its metadata."""
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

        # The unquantized reference K / V come straight out of the BF16 cache.
        compressed_kv, k_pe = bf16_layer.load_paged_kv_cache_for_mla(bf16_metadata, torch.bfloat16)
        assert compressed_kv.shape[0] == total
        k_nope, v = (compressed_kv @ kv_b).split(
            [NUM_HEADS * QK_NOPE_HEAD_DIM, NUM_HEADS * V_HEAD_DIM], -1
        )
        k = torch.cat(
            (
                k_nope.reshape(-1, NUM_HEADS, QK_NOPE_HEAD_DIM),
                k_pe.view(-1, 1, QK_ROPE_HEAD_DIM).expand(-1, NUM_HEADS, -1),
            ),
            dim=-1,
        ).reshape(-1, NUM_HEADS * QK_HEAD_DIM)
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

        def quantize(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)[0]

        bf16_out = run(bf16_layer, k, v, bf16_metadata)
        # BF16 K / V: the op quantizes q, k and v itself with the 192/64/256
        # copy, so this run is what exercises the per-width kernel instantiation.
        fp8_copy_out = run(fp8_layer, k, v, fp8_metadata)
        # Caller-quantized q / k / v with quant_q_buffer: the op skips every
        # copy and the FMHA reads the caller's tensors.
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
        # A mis-laid-out copy moves operands between heads and still lands
        # inside a loose band, so the copy is held to the caller-quantized
        # run's output bit for bit.
        assert torch.equal(fp8_copy_out.view(torch.int16), fp8_out.view(torch.int16))
        # The FP8 path tracks the unquantized reference within FP8 quantization noise.
        diff = fp8_out.float() - bf16_out.float()
        relative_error = diff.norm() / bf16_out.float().norm()
        assert relative_error < 0.1, (
            f"FP8 context at qk 192+64 / v 256 is {relative_error.item():.3f} away from "
            "the BF16-KV reference; its quantize-copy is mis-laid-out"
        )
    finally:
        fp8_manager.shutdown()
        bf16_manager.shutdown()
