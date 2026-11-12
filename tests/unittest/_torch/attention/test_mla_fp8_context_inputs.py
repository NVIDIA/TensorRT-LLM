# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA cached-KV context attention with caller-quantized FP8 Q / K / V.

With an FP8 KV cache, the attention op's FP8 context MLA FMHA reads FP8 Q, K and V. Given BF16
K / V it quantizes them itself with ``kv_scale_orig_quant``, the inverse of the KV cache scale
(round to nearest, saturating). A caller may hand it those FP8 tensors, quantized with the same
scale, directly (``key`` / ``value`` in E4M3, and ``quant_q_buffer`` with ``quant_scale_qkv``);
the output must be bit-identical to handing it the BF16 inputs.
The last test checks such a run against a BF16 KV cache on the same inputs: the
bit-identity alone would let a scaling bug that skews every run equally through.
"""

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
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

NUM_HEADS = 32
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
MAX_POSITIONS = 4096
FP8_MAX = 448.0


def _make_layer(kv_cache_quant_algo: str | None = QuantAlgo.FP8.value):
    rope_config = SimpleNamespace(
        hidden_size=7168,
        num_attention_heads=NUM_HEADS,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        max_position_embeddings=MAX_POSITIONS,
        rope_theta=10000.0,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        model_type="deepseek_v3",
    )
    return get_attention_backend("TRTLLM")(
        layer_idx=0,
        num_heads=NUM_HEADS,
        head_dim=QK_HEAD_DIM,
        num_kv_heads=NUM_HEADS,
        quant_config=(
            QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)
            if kv_cache_quant_algo is not None
            else None
        ),
        q_scaling=1.0,
        pos_embd_params=PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
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


def _to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """What the op's unit-scale copy writes: round to nearest, saturating."""
    return x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


@pytest.mark.parametrize(
    "cached,new", [([256], [128]), ([256, 200], [128, 184])], ids=["one_request", "two_requests"]
)
def test_mla_cached_kv_context_fp8_inputs_match_bf16_inputs(cached, new) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    layer = _make_layer()
    if not layer.has_fp8_context_mla:
        pytest.skip("this SM has no FP8 context MLA FMHA")
    scale = layer.kv_scale_orig_quant
    assert isinstance(scale, torch.Tensor) and bool(torch.all(scale == 1.0))

    torch.manual_seed(7)
    batch = len(cached)
    total = [c + n for c, n in zip(cached, new)]
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    tokens_per_block = 32
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=sum(-(-t // tokens_per_block) * tokens_per_block for t in total),
            enable_block_reuse=False,
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=tokens_per_block,
        max_seq_len=max(total),
        max_batch_size=batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
    )
    metadata_cls = get_attention_backend("TRTLLM").Metadata

    def context_metadata(seq_lens, cached_lens):
        metadata = metadata_cls(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=batch,
            prompt_lens=list(seq_lens),
            max_num_tokens=sum(seq_lens),
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        return metadata

    def random_inputs(num_tokens):
        q = torch.randn(num_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent = torch.randn(
            num_tokens, KV_LORA_RANK + QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        return q, latent

    try:
        stream = torch.cuda.current_stream().cuda_stream
        caches = []
        for req_id, length in enumerate(cached):
            kv_cache = manager._create_kv_cache(req_id, None, None)
            assert kv_cache.resume(stream)
            kv_cache.capacity = length
            caches.append(kv_cache)
        # First chunk: write the cached tokens' latent KV into the FP8 cache.
        q, latent = random_inputs(sum(cached))
        layer.mla_rope_append_paged_kv_assign_q(q, latent, context_metadata(cached, [0] * batch))

        for kv_cache, length in zip(caches, total):
            kv_cache.capacity = length
        metadata = context_metadata(new, cached)
        q, latent = random_inputs(sum(new))
        layer.mla_rope_append_paged_kv_assign_q(q, latent, metadata)
        compressed_kv, k_pe = layer.load_paged_kv_cache_for_mla(metadata, torch.bfloat16)
        assert compressed_kv.shape[0] == sum(total)

        kv_b = (
            torch.randn(
                KV_LORA_RANK,
                NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * KV_LORA_RANK**-0.5
        )
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

        def run(key, value, **fp8_q_args):
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

        bf16_out = run(k, v)
        q_fp8 = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, scale)[0]
        fp8_out = run(_to_e4m3(k), _to_e4m3(v), quant_q_buffer=q_fp8, quant_scale_qkv=scale)
        fp8_kv_only_out = run(_to_e4m3(k), _to_e4m3(v))
        torch.cuda.synchronize()

        assert bf16_out.abs().sum() > 0
        assert torch.equal(fp8_out.view(torch.int16), bf16_out.view(torch.int16))
        # Without quant_q_buffer the op quantizes Q itself; K / V still come from the caller.
        assert torch.equal(fp8_kv_only_out.view(torch.int16), bf16_out.view(torch.int16))

        with pytest.raises(RuntimeError, match="same dtype"):
            run(_to_e4m3(k), v)
        with pytest.raises(RuntimeError, match="same dtype"):
            run(k, _to_e4m3(v))
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "cached,new", [([256], [128]), ([256, 200], [128, 184])], ids=["one_request", "two_requests"]
)
def test_mla_cached_kv_context_fp8_inputs_non_unit_kv_scale(cached, new) -> None:
    """The contract above with a non-unit kv cache scale.

    The FMHA dequantizes Q / K / V with the cache scale, so the caller must quantize with
    `kv_scale_orig_quant`, and the op's own copies of BF16 inputs must use the same scale
    (not 1.0): the runs agree bit-for-bit at any scale, and a caller that quantizes with a
    different scale gets different (wrong) output.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    layer = _make_layer()
    if not layer.has_fp8_context_mla:
        pytest.skip("this SM has no FP8 context MLA FMHA")
    # A non-unit layer scale: kv_scale_quant_orig = 2.0, kv_scale_orig_quant = 0.5.
    layer.kv_cache_scaling_factor = torch.full((1,), 2.0, device="cuda")
    layer.kv_scale_quant_orig = layer.kv_cache_scaling_factor
    layer.kv_scale_orig_quant = 1.0 / layer.kv_cache_scaling_factor
    scale = layer.kv_scale_orig_quant
    assert isinstance(scale, torch.Tensor) and bool(torch.all(scale != 1.0))

    def quantize(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)[0]

    torch.manual_seed(7)
    batch = len(cached)
    total = [c + n for c, n in zip(cached, new)]
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    tokens_per_block = 32
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=sum(-(-t // tokens_per_block) * tokens_per_block for t in total),
            enable_block_reuse=False,
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=tokens_per_block,
        max_seq_len=max(total),
        max_batch_size=batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
    )
    metadata_cls = get_attention_backend("TRTLLM").Metadata

    def context_metadata(seq_lens, cached_lens):
        metadata = metadata_cls(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=batch,
            prompt_lens=list(seq_lens),
            max_num_tokens=sum(seq_lens),
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        return metadata

    def random_inputs(num_tokens):
        q = torch.randn(num_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent = torch.randn(
            num_tokens, KV_LORA_RANK + QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        return q, latent

    try:
        stream = torch.cuda.current_stream().cuda_stream
        caches = []
        for req_id, length in enumerate(cached):
            kv_cache = manager._create_kv_cache(req_id, None, None)
            assert kv_cache.resume(stream)
            kv_cache.capacity = length
            caches.append(kv_cache)
        # First chunk: write the cached tokens' latent KV into the FP8 cache.
        q, latent = random_inputs(sum(cached))
        layer.mla_rope_append_paged_kv_assign_q(q, latent, context_metadata(cached, [0] * batch))

        for kv_cache, length in zip(caches, total):
            kv_cache.capacity = length
        metadata = context_metadata(new, cached)
        q, latent = random_inputs(sum(new))
        layer.mla_rope_append_paged_kv_assign_q(q, latent, metadata)
        compressed_kv, k_pe = layer.load_paged_kv_cache_for_mla(metadata, torch.bfloat16)
        assert compressed_kv.shape[0] == sum(total)

        kv_b = (
            torch.randn(
                KV_LORA_RANK,
                NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * KV_LORA_RANK**-0.5
        )
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

        def run(key, value, **fp8_q_args):
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

        bf16_out = run(k, v)
        fp8_out = run(quantize(k), quantize(v), quant_q_buffer=quantize(q), quant_scale_qkv=scale)
        fp8_kv_only_out = run(quantize(k), quantize(v))
        torch.cuda.synchronize()

        assert bf16_out.abs().sum() > 0
        # The op's copies of the BF16 inputs and the caller's FP8 both quantize with
        # the kv cache scale, so the runs agree at a non-unit scale too.
        assert torch.equal(fp8_out.view(torch.int16), bf16_out.view(torch.int16))
        assert torch.equal(fp8_kv_only_out.view(torch.int16), bf16_out.view(torch.int16))
        # Data quantized with another scale (here: unit) stays wrong: the FMHA
        # dequantizes with the kv cache scale, whatever quant_scale_qkv carries.
        wrong_out = run(_to_e4m3(k), _to_e4m3(v))
        assert not torch.equal(wrong_out.view(torch.int16), bf16_out.view(torch.int16))
    finally:
        manager.shutdown()


@pytest.mark.parametrize("cached,new", [([256], [128])], ids=["one_request"])
def test_mla_cached_kv_context_fp8_inputs_track_the_unquantized_reference(cached, new) -> None:
    """The runs above against a BF16 KV cache, as a numeric reference.

    The tests above only compare runs of the same FP8 layer, so a scaling bug that
    skewed them all equally would pass them. Here a layer with a BF16 KV cache runs
    the same inputs end to end: its output is the reference, the FP8 layer's scaled
    context inputs must track it, and unit-scaled inputs (what the op's own copies
    did on main at a non-unit KV scale) land far away.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    fp8_layer = _make_layer()
    if not fp8_layer.has_fp8_context_mla:
        pytest.skip("this SM has no FP8 context MLA FMHA")
    bf16_layer = _make_layer(kv_cache_quant_algo=None)
    # A non-unit layer scale: kv_scale_quant_orig = 2.0, kv_scale_orig_quant = 0.5.
    fp8_layer.kv_cache_scaling_factor = torch.full((1,), 2.0, device="cuda")
    fp8_layer.kv_scale_quant_orig = fp8_layer.kv_cache_scaling_factor
    fp8_layer.kv_scale_orig_quant = 1.0 / fp8_layer.kv_cache_scaling_factor
    scale = fp8_layer.kv_scale_orig_quant
    assert isinstance(scale, torch.Tensor) and bool(torch.all(scale != 1.0))

    def quantize(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)[0]

    torch.manual_seed(7)
    batch = len(cached)
    total = [c + n for c, n in zip(cached, new)]
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    tokens_per_block = 32
    metadata_cls = get_attention_backend("TRTLLM").Metadata

    def make_manager(dtype: torch.dtype) -> KVCacheManagerV2:
        return KVCacheManagerV2(
            KvCacheConfig(
                max_tokens=sum(-(-t // tokens_per_block) * tokens_per_block for t in total),
                enable_block_reuse=False,
            ),
            CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            tokens_per_block=tokens_per_block,
            max_seq_len=max(total),
            max_batch_size=batch,
            mapping=mapping,
            dtype=str_dtype_to_binding(torch_dtype_to_str(dtype)),
        )

    def context_metadata(seq_lens, cached_lens, manager):
        metadata = metadata_cls(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=batch,
            prompt_lens=list(seq_lens),
            max_num_tokens=sum(seq_lens),
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached_lens),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        return metadata

    def random_inputs(num_tokens):
        q = torch.randn(num_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        latent = torch.randn(
            num_tokens, KV_LORA_RANK + QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        return q, latent

    # One set of inputs drives both pipelines: the FP8 cache holds their latent
    # quantized at the layer scale, the BF16 cache holds it exactly.
    cached_q, cached_latent = random_inputs(sum(cached))
    new_q, new_latent = random_inputs(sum(new))
    kv_b = (
        torch.randn(
            KV_LORA_RANK,
            NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        * KV_LORA_RANK**-0.5
    )

    fp8_manager = make_manager(torch.float8_e4m3fn)
    bf16_manager = make_manager(torch.bfloat16)

    def prime(layer, manager):
        """Write this pair's share of the inputs into `manager`; return the
        forward's already rope'd q and its metadata."""
        stream = torch.cuda.current_stream().cuda_stream
        caches = []
        for req_id, length in enumerate(cached):
            kv_cache = manager._create_kv_cache(req_id, None, None)
            assert kv_cache.resume(stream)
            kv_cache.capacity = length
            caches.append(kv_cache)
        layer.mla_rope_append_paged_kv_assign_q(
            cached_q.clone(), cached_latent.clone(), context_metadata(cached, [0] * batch, manager)
        )
        for kv_cache, length in zip(caches, total):
            kv_cache.capacity = length
        metadata = context_metadata(new, cached, manager)
        # assign_q writes the rope'd q back into its q buffer, as the forwards below expect.
        q = new_q.clone()
        layer.mla_rope_append_paged_kv_assign_q(q, new_latent.clone(), metadata)
        return q, metadata

    try:
        _, fp8_metadata = prime(fp8_layer, fp8_manager)
        q, bf16_metadata = prime(bf16_layer, bf16_manager)

        # The unquantized reference K / V come straight out of the BF16 cache.
        compressed_kv, k_pe = bf16_layer.load_paged_kv_cache_for_mla(bf16_metadata, torch.bfloat16)
        assert compressed_kv.shape[0] == sum(total)
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

        def relative_error(candidate: torch.Tensor, reference_out: torch.Tensor) -> torch.Tensor:
            diff = candidate.float() - reference_out.float()
            return diff.norm() / reference_out.float().norm()

        reference = run(bf16_layer, k, v, bf16_metadata)
        fp8_out = run(
            fp8_layer,
            quantize(k),
            quantize(v),
            fp8_metadata,
            quant_q_buffer=quantize(q),
            quant_scale_qkv=scale,
        )
        wrong_out = run(fp8_layer, _to_e4m3(k), _to_e4m3(v), fp8_metadata)
        torch.cuda.synchronize()

        assert reference.abs().sum() > 0
        assert fp8_out.abs().sum() > 0
        # The FP8 path tracks the unquantized reference within FP8 quantization noise.
        assert relative_error(fp8_out, reference) < 0.1
        # Unit-scale data dequantized at the layer scale (2x here) lands far away.
        assert relative_error(wrong_out, reference) > 0.25
    finally:
        fp8_manager.shutdown()
        bf16_manager.shutdown()
