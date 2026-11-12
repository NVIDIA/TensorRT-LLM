# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family MLA attention ops against the attention-backend calls they replace.

``ops/attention/k3_mla_{state,attention,cache}`` are compared with ``TrtllmAttention`` configured
the way the K3 target configures its context (``mha``) and generation (``mqa``) roles: GPT-NeoX
RoPE parameters with the identity table installed, unit KV scales. Each case builds two identical
KV cache managers, runs the backend on one and the family op on the other with the same inputs,
and requires the outputs and every cache pool to match bit for bit.

``TLLM_FMHA_LIBS`` restricts the backend to the FMHA libraries that serve these calls on SM107
(FP4 MLA, CuTe DSL MLA, fallback), so on SM100 / SM103 the backend launches the kernels the family
ops launch.
"""

from types import SimpleNamespace

import pytest
import torch

H = 96
Q_LORA, LORA, NOPE, ROPE, VDIM = 1536, 512, 128, 64, 128
QK = NOPE + ROPE
MAX_POS = 4096


def _sm_version() -> int:
    if not torch.cuda.is_available():
        return 0
    from tensorrt_llm._utils import get_sm_version

    return get_sm_version()


pytestmark = pytest.mark.skipif(
    _sm_version() not in (100, 103, 107),
    reason="needs an SM100 / SM103 / SM107 GPU (FP8 context MLA FMHA)",
)


@pytest.fixture(autouse=True)
def _sm107_fmha_libraries(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fp4_mla,cute_dsl_mla,fallback")


def _ops():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (
        k3_mla_attention,
        k3_mla_cache,
        k3_mla_state,
    )

    return SimpleNamespace(attn=k3_mla_attention, cache=k3_mla_cache, state=k3_mla_state)


def _quant_config(kind: str):
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    if kind == "bf16":
        return QuantConfig()
    algo = QuantAlgo.FP8 if kind == "fp8" else QuantAlgo.NVFP4
    return QuantConfig(kv_cache_quant_algo=algo.value)


# ---------------------------------------------------------------------------------------------
# The backend side: TrtllmAttention as the K3 target builds it.
# ---------------------------------------------------------------------------------------------


def _install_identity_rope(backend) -> None:
    identity_rope_table = _ops().state.identity_rope_table
    backend.rotary_cos_sin = identity_rope_table(
        backend.rotary_cos_sin.numel(), backend.rotary_cos_sin.device
    )
    rope = backend.rope_params
    floats_per_position = rope.dim * (2 if rope.duplicate_data else 1)

    def ensure_table_size(required_max_positions: int) -> None:
        table_positions = backend.rotary_cos_sin.numel() // floats_per_position
        rope.max_positions = max(rope.max_positions, table_positions, required_max_positions)
        if required_max_positions > table_positions:
            backend.rotary_cos_sin = identity_rope_table(
                rope.max_positions * floats_per_position, backend.rotary_cos_sin.device
            )

    backend._ensure_rope_table_size = ensure_table_size


def _backend_roles(kind: str, layer_idx: int = 0):
    """``(mha, mqa)``: the context and generation ``TrtllmAttention`` of one K3 MLA layer."""
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.utils import create_attention
    from tensorrt_llm.functional import PositionEmbeddingType

    quant_config = _quant_config(kind)
    dims = dict(
        pos_embd_params=PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams(
                dim=ROPE,
                theta=10000.0,
                max_positions=MAX_POS,
                original_max_positions=MAX_POS,
                duplicate_data=True,
            ),
            is_neox=False,
        ),
        quant_config=quant_config,
        q_scaling=1.0,
        is_mla_enable=True,
        q_lora_rank=Q_LORA,
        kv_lora_rank=LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        predicted_tokens_per_seq=1,
        skip_create_weights_in_init=True,
        sparse_params=None,
        skip_correction_threshold=0.0,
    )
    mqa = create_attention(
        "TRTLLM",
        layer_idx,
        H,
        head_dim=LORA + ROPE,
        num_kv_heads=1,
        hidden_size=7168,
        v_head_dim=LORA,
        dtype=torch.bfloat16,
        aux_stream=torch.cuda.Stream(),
        rope_append=True,
        kv_cache_dtype="auto",
        **dims,
    )
    mha = create_attention(
        "TRTLLM", layer_idx, H, head_dim=QK, num_kv_heads=H, v_head_dim=VDIM, **dims
    )
    for role in (mha, mqa):
        _install_identity_rope(role)
        role.update_quant_config(quant_config)
    return mha, mqa


def _family_roles(kind: str, layer_idx: int = 0):
    """``(context, generation)`` family states of the same layer."""
    K3MlaAttnState = _ops().state.K3MlaAttnState
    common = dict(
        layer_idx=layer_idx,
        q_lora_rank=Q_LORA,
        max_position_embeddings=MAX_POS,
        quant_config=_quant_config(kind),
    )
    ctx = K3MlaAttnState.context(
        num_context_heads=H,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=VDIM,
        kv_lora_rank=LORA,
        **common,
    )
    gen = K3MlaAttnState.generation(
        num_heads=H, kv_lora_rank=LORA, qk_rope_head_dim=ROPE, qk_nope_head_dim=NOPE, **common
    )
    return ctx, gen


# ---------------------------------------------------------------------------------------------
# One KV cache manager with its requests and per-step attention metadata.
# ---------------------------------------------------------------------------------------------


class _Bench:
    def __init__(self, kind: str, tokens_per_block: int, max_seq_len: int, batch: int):
        import tensorrt_llm
        from tensorrt_llm.mapping import Mapping

        self.kind = kind
        self.batch = batch
        self.mapping = Mapping(world_size=1, tp_size=1, rank=0)
        pages = -(-max_seq_len // tokens_per_block)
        max_tokens = batch * pages * tokens_per_block
        cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY
        common = dict(
            num_layers=1,
            num_kv_heads=1,
            head_dim=LORA + ROPE,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch,
            mapping=self.mapping,
        )
        if kind in ("fp8", "bf16"):
            from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
            from tensorrt_llm.llmapi.llm_args import KvCacheConfig

            data_type = tensorrt_llm.bindings.DataType
            self.manager = KVCacheManagerV2(
                KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
                cache_type,
                dtype=data_type.FP8 if kind == "fp8" else data_type.BF16,
                **common,
            )
        else:
            from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import (
                Fp4MlaKVCacheManagerV2,
            )
            from tensorrt_llm.llmapi.llm_args import KvCacheConfig

            self.manager = Fp4MlaKVCacheManagerV2(
                KvCacheConfig(
                    max_tokens=max_tokens,
                    dtype="nvfp4",
                    enable_block_reuse=False,
                    host_cache_size=0,
                ),
                cache_type,
                dtype=tensorrt_llm.bindings.DataType.NVFP4,
                max_num_tokens=max_tokens,
                pretrained_config=SimpleNamespace(kv_lora_rank=LORA),
                **common,
            )
        self._caches = []
        for pool in self.pools():
            pool.zero_()

    def pools(self) -> list:
        """Every cache pool a call of these tests may write."""
        if self.kind != "nvfp4":
            return [self.manager.get_buffers(0)]
        manager = self.manager
        kv, sf = manager.get_fp4_mla_cache_buffers(0)
        pools = [kv, sf, manager.get_fp4_mla_hp_pool()]
        for extra in (manager.get_mla_v_scale_pool(), manager.get_mla_v_packed_pool_base()):
            if extra is not None:
                pools.append(extra)
        return pools

    def grow(self, lengths) -> None:
        """Give request ``i`` capacity for ``lengths[i]`` tokens (creating it on first use)."""
        stream = torch.cuda.current_stream().cuda_stream
        for req_id, length in enumerate(lengths):
            if req_id == len(self._caches):
                kv_cache = self.manager._create_kv_cache(req_id, None, None)
                assert kv_cache.resume(stream)
                self._caches.append(kv_cache)
            self._caches[req_id].capacity = length

    def metadata(
        self,
        seq_lens,
        cached,
        *,
        context: bool,
        chunk_size: int = 0,
        cached_kv: bool = False,
        prompt_lens=None,
    ):
        from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
        from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams

        n = len(seq_lens)
        md = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(n)),
            max_num_requests=self.batch,
            num_contexts=n if context else 0,
            prompt_lens=list(seq_lens) if context else list(prompt_lens),
            max_num_tokens=max(sum(seq_lens), 1),
            kv_cache_manager=self.manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached)),
            mapping=self.mapping,
            enable_context_mla_with_cached_kv=cached_kv,
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=chunk_size > 0,
                chunk_size=chunk_size,
                chunked_prefill_buffer_batch_size=1,
            ),
        )
        md.prepare()
        return md

    def shutdown(self) -> None:
        self.manager.shutdown()


def _assert_same(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape, a.dtype, b.dtype)
    a_bits = a.contiguous().view(torch.uint8)
    b_bits = b.contiguous().view(torch.uint8)
    mismatch = (a_bits != b_bits).sum().item()
    detail = ""
    if mismatch and a.is_floating_point() and a.element_size() > 1:
        diff = (a.float() - b.float()).abs()
        rows = (diff.flatten(1).amax(1) > 0).sum().item() if diff.dim() > 1 else "-"
        detail = f", max |diff| {diff.max().item():.3g}, rows differing {rows}"
    assert mismatch == 0, f"{name}: {mismatch} of {a_bits.numel()} bytes differ{detail}"


def _assert_same_pools(backend: _Bench, family: _Bench) -> None:
    torch.cuda.synchronize()
    for i, (a, b) in enumerate(zip(backend.pools(), family.pools())):
        _assert_same(f"pool {i}", a, b)


def _context_inputs(num_tokens: int, seed: int):
    """``q [T, H * 192]``, ``k [T, H * 192]``, ``v`` (``[T, H * 128]`` view of the kv_b output
    rows, as the target hands it) and ``latent [T, 576]``."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.empty(*shape, dtype=torch.bfloat16, device="cuda").uniform_(-1, 1, generator=g)

    q = rand(num_tokens, H * QK)
    k = rand(num_tokens, H * QK)
    kv = rand(num_tokens, H * (NOPE + VDIM))
    v = kv[:, H * NOPE :]
    latent = rand(num_tokens, LORA + ROPE)
    return q, k, v, latent


def _kv_b(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.empty(LORA, H * (NOPE + VDIM), dtype=torch.bfloat16, device="cuda")
    return w.uniform_(-1, 1, generator=g) * LORA**-0.5


# ---------------------------------------------------------------------------------------------
# State.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bf16", "fp8", "nvfp4"])
def test_state_matches_backend_roles(kind):
    mha, mqa = _backend_roles(kind)
    ctx, gen = _family_roles(kind)
    for name, backend, state in (("context", mha, ctx), ("generation", mqa, gen)):
        for attr in (
            "rotary_inv_freq",
            "rotary_cos_sin",
            "kv_cache_scaling_factor",
            "kv_scale_quant_orig",
            "kv_scale_orig_quant",
        ):
            _assert_same(f"{name}.{attr}", getattr(backend, attr), getattr(state, attr))
        for attr in (
            "num_heads",
            "head_dim",
            "num_kv_heads",
            "q_scaling",
            "predicted_tokens_per_seq",
            "rope_append",
            "has_fp8_kv_cache",
            "has_fp4_kv_cache",
            "uses_fp4_mla_attention",
            "has_fp8_context_mla",
            "skip_correction_threshold",
            "is_mla_enable",
        ):
            assert getattr(backend, attr) == getattr(state, attr), (name, attr)
        assert int(backend.quant_mode) == int(state.quant_mode), name
        assert int(backend.position_embedding_type) == int(state.position_embedding_type), name
        rope = backend.rope_params
        assert (
            rope.dim,
            rope.theta,
            rope.max_positions,
            rope.original_max_positions,
            rope.duplicate_data,
        ) == (
            state.rope_dim,
            state.rope_base,
            state.rope_max_positions,
            state.rope_original_max_positions,
            state.rope_duplicate_data,
        )
        assert backend.support_fp4_kv_cache() == state.support_fp4_kv_cache()
    # Growing the identity table past the configured positions. The target builds both backend
    # roles from one RopeParams, so the context role's growth raises the generation role's position
    # count as well; each family state keeps its own (the op reads positions below either count).
    for name, backend, state in (("context", mha, ctx), ("generation", mqa, gen)):
        rope = backend.rope_params
        backend._ensure_rope_table_size(3 * MAX_POS)
        state._ensure_rope_table_size(3 * MAX_POS)
        _assert_same(f"{name}.grown rotary_cos_sin", backend.rotary_cos_sin, state.rotary_cos_sin)
        assert rope.max_positions == state.rope_max_positions


# ---------------------------------------------------------------------------------------------
# FP8 latent cache (K3's FP8 MLA layers): context, cached-prefix context, generation; the BF16
# cache's context and generation.
# ---------------------------------------------------------------------------------------------

# (generation batch, q rows per request, CuTe DSL FP8 decode expected on SM100 / SM103 at 32-token
# pages). On a BF16 cache the CuTe DSL decode takes K3's 96 heads at every batch size.
_GEN_CASES = {
    "op_b2_q1": (2, 1, False),
    "cute_table_b8_q1": (8, 1, True),
    "cute_wide_verify_b2_q2": (2, 2, True),
}


@pytest.mark.parametrize("tokens_per_block", [32, 128])
@pytest.mark.parametrize("gen_case", list(_GEN_CASES))
@pytest.mark.parametrize("kind", ["fp8", "bf16"])
def test_context_and_decode_match_backend(kind, tokens_per_block, gen_case, monkeypatch):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )

    ops = _ops()
    batch, q_len, expect_cute = _GEN_CASES[gen_case]
    if kind == "bf16":
        expect_cute = True
    if tokens_per_block == 128:
        if _sm_version() == 107:
            expect_cute = True  # 128-token pages are the CuTe DSL decode's alone on SM107.
        elif not expect_cute:
            pytest.skip(
                "off SM107 the op reads 128-token FP8 MLA pages through PrimTS, not restricted here"
            )
    ctx_lens = [37, 130, 5, 64, 200, 17, 90, 11][:batch]
    steps = 2
    max_seq_len = max(ctx_lens) + steps * q_len + 1
    benches = [_Bench(kind, tokens_per_block, max_seq_len, batch) for _ in range(2)]
    decode_calls = []
    decode_op = (
        "cute_dsl_mla_decode_fp8_blackwell"
        if kind == "fp8"
        else "cute_dsl_mla_decode_fp16_blackwell"
    )
    real_decode = getattr(torch.ops.trtllm, decode_op)

    def counting_decode(*args, **kwargs):
        decode_calls.append(1)
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(torch.ops.trtllm, decode_op, counting_decode)
    try:
        mha, mqa = _backend_roles(kind)
        ctx, gen = _family_roles(kind)
        for bench in benches:
            bench.grow(ctx_lens)
        mds = [b.metadata(ctx_lens, [0] * batch, context=True) for b in benches]
        q, k, v, latent = _context_inputs(sum(ctx_lens), seed=1)
        out_backend = torch.empty(sum(ctx_lens), H * VDIM, dtype=torch.bfloat16, device="cuda")
        out_family = torch.empty_like(out_backend)
        mha.forward(
            q.clone(),
            k.clone(),
            v,
            mds[0],
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent,
                out_scale=None,
                output=out_backend,
            ),
        )
        ops.attn.mla_context_attention(
            ctx, q.clone(), k.clone(), v, mds[1], out_family, latent_cache=latent
        )
        _assert_same("context output", out_backend, out_family)
        _assert_same_pools(*benches)

        for step in range(steps):
            lengths = [c + (step + 1) * q_len for c in ctx_lens]
            cached = [c + step * q_len for c in ctx_lens]
            for bench in benches:
                bench.grow(lengths)
            mds = [
                b.metadata([q_len] * batch, cached, context=False, prompt_lens=ctx_lens)
                for b in benches
            ]
            num_tokens = batch * q_len
            g = torch.Generator(device="cuda").manual_seed(100 + step)
            fused_q = torch.empty(num_tokens, H, LORA + ROPE, dtype=torch.bfloat16, device="cuda")
            fused_q.uniform_(-1, 1, generator=g)
            q_pe = torch.empty(num_tokens, H, ROPE, dtype=torch.bfloat16, device="cuda")
            q_pe.uniform_(-1, 1, generator=g)
            gen_latent = torch.empty(num_tokens, LORA + ROPE, dtype=torch.bfloat16, device="cuda")
            gen_latent.uniform_(-1, 1, generator=g)
            results = []
            decode_calls.clear()
            for side, md in enumerate(mds):
                bufs = dict(
                    fused_q=fused_q.clone(),
                    cu_q_seqlens=torch.zeros(batch + 1, dtype=torch.int32, device="cuda"),
                    cu_kv_seqlens=torch.zeros(batch + 1, dtype=torch.int32, device="cuda"),
                    fmha_scheduler_counter=torch.zeros(1, dtype=torch.uint32, device="cuda"),
                    mla_bmm1_scale=torch.zeros(2, dtype=torch.float32, device="cuda"),
                    mla_bmm2_scale=torch.zeros(1, dtype=torch.float32, device="cuda"),
                    quant_q_buffer=torch.zeros(
                        num_tokens, H, LORA + ROPE, dtype=torch.uint8, device="cuda"
                    ),
                )
                rope_args = (
                    bufs["fused_q"],
                    q_pe,
                    gen_latent,
                    md,
                    bufs["cu_q_seqlens"],
                    bufs["cu_kv_seqlens"],
                    bufs["fmha_scheduler_counter"],
                    bufs["mla_bmm1_scale"],
                    bufs["mla_bmm2_scale"],
                    bufs["quant_q_buffer"],
                )
                if side == 0:
                    mqa.mla_rope_generation(*rope_args)
                else:
                    ops.cache.mla_rope_generation(gen, *rope_args)
                fq = bufs["fused_q"].view(num_tokens, H * (LORA + ROPE))
                out = torch.empty(num_tokens, H * LORA, dtype=torch.bfloat16, device="cuda")
                decode_args = dict(
                    latent_cache=gen_latent,
                    q_pe=q_pe,
                    cu_q_seqlens=bufs["cu_q_seqlens"],
                    cu_kv_seqlens=bufs["cu_kv_seqlens"],
                    fmha_scheduler_counter=bufs["fmha_scheduler_counter"],
                    mla_bmm1_scale=bufs["mla_bmm1_scale"],
                    mla_bmm2_scale=bufs["mla_bmm2_scale"],
                    quant_q_buffer=bufs["quant_q_buffer"],
                )
                if side == 0:
                    mqa.forward(
                        fq,
                        None,
                        None,
                        md,
                        forward_args=AttentionForwardArgs(
                            attention_input_type=AttentionInputType.generation_only,
                            out_scale=None,
                            output=out,
                            **decode_args,
                        ),
                    )
                else:
                    assert ops.attn.cute_dsl_decode_applies(gen, fq, md, out, None) == expect_cute
                    ops.attn.mla_decode(gen, fq, md, out, **decode_args)
                results.append((bufs, out))
            assert len(decode_calls) == (2 if expect_cute else 0), decode_calls
            (b_bufs, b_out), (f_bufs, f_out) = results
            for name in b_bufs:
                _assert_same(f"step {step} {name}", b_bufs[name], f_bufs[name])
            _assert_same(f"step {step} decode output", b_out, f_out)
            _assert_same_pools(*benches)
    finally:
        for bench in benches:
            bench.shutdown()


class _BackendCalls:
    """The cached-prefix context calls of the target's chunked-prefill / cached-KV chains, on the
    backend role."""

    def __init__(self, mha):
        self.mha = mha
        self.kv_scale = mha.kv_scale_orig_quant

    def rope_append(self, q, latent, md):
        self.mha.mla_rope_append_paged_kv_assign_q(q, latent, md)

    def load_paged(self, md):
        return self.mha.load_paged_kv_cache_for_mla(md, torch.bfloat16)

    def load_chunked(self, md, loop_idx):
        return self.mha.load_chunked_kv_cache_for_mla(
            metadata=md,
            num_ctx_cached_tokens=md.host_cu_chunked_seq_len[loop_idx, md.num_contexts],
            cu_chunked_seq_len=md.cu_chunked_seq_len[loop_idx],
            chunked_global_offset=md.chunked_global_offset[loop_idx],
            chunked_max_seq_len=md.max_chunk_len_per_loop[loop_idx],
            out_dtype=torch.bfloat16,
        )

    def attend(self, q, k, v, md, output, *, full_mask=False, softmax_stats=None, q_fp8=None):
        from tensorrt_llm._torch.attention.backends.interface import (
            AttentionForwardArgs,
            AttentionInputType,
            PredefinedAttentionMask,
        )

        kwargs = dict(
            attention_input_type=AttentionInputType.context_only,
            latent_cache=None,
            out_scale=None,
            output=output,
        )
        if full_mask:
            kwargs["attention_mask"] = PredefinedAttentionMask.FULL
        if softmax_stats is not None:
            kwargs["softmax_stats_tensor"] = softmax_stats
            kwargs["chunked_prefill_buffer_batch_size"] = 1
        if q_fp8 is not None:
            kwargs["quant_q_buffer"] = q_fp8
            kwargs["quant_scale_qkv"] = self.kv_scale
        return self.mha.forward(q, k, v, md, forward_args=AttentionForwardArgs(**kwargs))

    def merge(self, output, temp, stats, temp_stats, merge_op, md):
        self.mha.merge_attention_for_mla(output, temp, stats, temp_stats, merge_op, md)


class _FamilyCalls:
    def __init__(self, state):
        self.state = state
        self.kv_scale = state.kv_scale_orig_quant
        self.ops = _ops()

    def rope_append(self, q, latent, md):
        self.ops.cache.mla_rope_append_paged_kv_assign_q(self.state, q, latent, md)

    def load_paged(self, md):
        return self.ops.cache.load_paged_kv_cache_for_mla(self.state, md, torch.bfloat16)

    def load_chunked(self, md, loop_idx):
        return self.ops.cache.load_chunked_kv_cache_for_mla(
            self.state,
            md,
            md.host_cu_chunked_seq_len[loop_idx, md.num_contexts],
            md.cu_chunked_seq_len[loop_idx],
            md.chunked_global_offset[loop_idx],
            md.max_chunk_len_per_loop[loop_idx],
            torch.bfloat16,
        )

    def attend(self, q, k, v, md, output, *, full_mask=False, softmax_stats=None, q_fp8=None):
        return self.ops.attn.mla_context_attention(
            self.state,
            q,
            k,
            v,
            md,
            output,
            full_mask=full_mask,
            softmax_stats=softmax_stats,
            chunked_prefill_buffer_batch_size=1,
            quant_q_buffer=q_fp8,
            quant_scale_qkv=self.kv_scale if q_fp8 is not None else None,
        )

    def merge(self, output, temp, stats, temp_stats, merge_op, md):
        self.ops.cache.merge_attention_for_mla(
            self.state, output, temp, stats, temp_stats, merge_op, md
        )


def _expand(compressed_kv, k_pe, kv_b, fp8):
    """K / V of the context FMHA from latent rows (the target's kv_b expansion)."""
    k_nope, v = (compressed_kv @ kv_b).split([H * NOPE, H * VDIM], -1)
    k = torch.cat(
        (k_nope.reshape(-1, H, NOPE), k_pe.reshape(-1, 1, ROPE).expand(-1, H, -1)), dim=-1
    ).reshape(-1, H * QK)
    v = v.contiguous()
    if fp8:
        k = k.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        v = v.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return k, v


def _cached_context(calls, q, compressed_kv, latent, md, kv_b, fp8, record):
    """The target's cached-prefix context chains (``context_chunked_prefill`` /
    ``context_cached_kv``) over ``calls``; ``record`` collects every intermediate."""
    calls.rope_append(q, latent, md)
    record.append(("q after rope append", q.clone()))
    q_fp8 = None
    if fp8:
        q_fp8 = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(q, calls.kv_scale)[0]
    # Zero-filled: a partition leaves the rows of requests without KV in it unwritten (the merge
    # skips them), and the test compares every row.
    output = torch.zeros(q.shape[0], H * VDIM, dtype=q.dtype, device=q.device)
    if not md.runtime_features.chunked_prefill:
        full_kv, full_pe = calls.load_paged(md)
        record += [("paged latent", full_kv), ("paged k_pe", full_pe)]
        k, v = _expand(full_kv, full_pe, kv_b, fp8)
        return calls.attend(q, k, v, md, output, q_fp8=q_fp8)
    stats = torch.zeros((md.num_ctx_tokens, H, 2), dtype=torch.float, device="cuda")
    temp_stats = torch.zeros_like(stats)
    temp = torch.zeros_like(output)
    origin = (md.kv_lens_cuda_runtime, md.kv_lens_runtime, md.host_total_kv_lens[0].item())
    for loop_idx in range(md.chunked_loop_num):
        chunk_kv, chunk_pe = calls.load_chunked(md, loop_idx)
        record += [(f"chunk {loop_idx} latent", chunk_kv), (f"chunk {loop_idx} k_pe", chunk_pe)]
        k, v = _expand(chunk_kv, chunk_pe, kv_b, fp8)
        md.kv_lens_runtime = md.host_chunked_seq_len[loop_idx]
        md.kv_lens_cuda_runtime = md.chunked_seq_len[loop_idx]
        md.host_total_kv_lens[0] = md.host_cu_chunked_seq_len[loop_idx, md.num_contexts]
        temp = calls.attend(
            q, k, v, md, temp, full_mask=True, softmax_stats=temp_stats, q_fp8=q_fp8
        )
        record += [
            (f"chunk {loop_idx} output", temp.clone()),
            (f"chunk {loop_idx} stats", temp_stats.clone()),
        ]
        calls.merge(output, temp, stats, temp_stats, md.merge_op_tensor[loop_idx], md)
    k_pe = latent.view(-1, LORA + ROPE)[:, LORA:]
    k, v = _expand(compressed_kv, k_pe, kv_b, fp8)
    md.kv_lens_runtime = md.prompt_lens_cpu_runtime
    md.kv_lens_cuda_runtime = md.prompt_lens_cuda_runtime
    md.host_total_kv_lens[0] = md.prompt_lens_cpu_runtime[: md.num_contexts].sum().item()
    temp = calls.attend(q, k, v, md, temp, softmax_stats=temp_stats, q_fp8=q_fp8)
    calls.merge(output, temp, stats, temp_stats, md.merge_op_tensor[md.chunked_loop_num], md)
    md.kv_lens_cuda_runtime, md.kv_lens_runtime, total = origin
    md.host_total_kv_lens[0] = total
    return output


@pytest.mark.parametrize("chunk_size", [0, 64], ids=["cached_kv", "chunked_prefill"])
@pytest.mark.parametrize("fp8_inputs", [False, True], ids=["bf16_qkv", "fp8_qkv"])
def test_fp8_cached_prefix_context_matches_backend(chunk_size, fp8_inputs):
    cached = [256, 200]
    new = [128, 184]
    total = [c + n for c, n in zip(cached, new)]
    batch = len(cached)
    benches = [_Bench("fp8", 32, max(total), batch) for _ in range(2)]
    try:
        mha, _ = _backend_roles("fp8")
        ctx, _ = _family_roles("fp8")
        sides = [_BackendCalls(mha), _FamilyCalls(ctx)]
        # The prefix: one plain context chunk per side.
        for bench in benches:
            bench.grow(cached)
        q, _, _, latent = _context_inputs(sum(cached), seed=2)
        for calls, bench in zip(sides, benches):
            md = bench.metadata(cached, [0] * batch, context=True, cached_kv=True)
            calls.rope_append(q.clone(), latent, md)
        _assert_same_pools(*benches)

        for bench in benches:
            bench.grow(total)
        q, _, _, latent = _context_inputs(sum(new), seed=3)
        compressed_kv = latent[:, :LORA]
        kv_b = _kv_b(seed=4)
        outputs, records = [], []
        for calls, bench in zip(sides, benches):
            md = bench.metadata(new, cached, context=True, chunk_size=chunk_size, cached_kv=True)
            if chunk_size:
                assert md.chunked_loop_num > 1
            record = []
            outputs.append(
                _cached_context(
                    calls, q.clone(), compressed_kv, latent, md, kv_b, fp8_inputs, record
                )
            )
            records.append(record)
        for (name, a), (_, b) in zip(*records):
            _assert_same(name, a, b)
        assert outputs[0].abs().sum() > 0
        _assert_same("context output", outputs[0], outputs[1])
        _assert_same_pools(*benches)
    finally:
        for bench in benches:
            bench.shutdown()


# ---------------------------------------------------------------------------------------------
# NVFP4 latent cache (K3's other MLA layers): context without a prefix, the split-K context's
# append and prefix gather, generation cache update.
# ---------------------------------------------------------------------------------------------


def _fp4_q_buffers(md):
    state = md.fp4_mla_state
    out = []
    for name in ("prequantized_q", "prequantized_q_sf"):
        t = getattr(state, name, None)
        if t is not None:
            out.append((name, t.clone()))
    return out


def _nvfp4_context(benches, ctx_lens):
    """The no-prefix NVFP4 context on both benches; checks outputs and pools."""
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )

    ops = _ops()
    batch = len(ctx_lens)
    mha, mqa = _backend_roles("nvfp4")
    ctx, gen = _family_roles("nvfp4")
    assert mha.uses_fp4_mla_attention and ctx.uses_fp4_mla_attention
    for bench in benches:
        bench.grow(ctx_lens)
    mds = [b.metadata(ctx_lens, [0] * batch, context=True) for b in benches]
    q, k, v, latent = _context_inputs(sum(ctx_lens), seed=5)
    out_backend = torch.empty(sum(ctx_lens), H * VDIM, dtype=torch.bfloat16, device="cuda")
    out_family = torch.empty_like(out_backend)
    mha.forward(
        q.clone(),
        k.clone(),
        v,
        mds[0],
        forward_args=AttentionForwardArgs(
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent,
            out_scale=None,
            output=out_backend,
        ),
    )
    ops.cache.fp4_mla_context_attention(ctx, q.clone(), k.clone(), v, mds[1], out_family, latent)
    torch.cuda.synchronize()
    assert out_backend.abs().sum() > 0
    _assert_same("context output", out_backend, out_family)
    _assert_same_pools(*benches)
    return mha, mqa, ctx, gen


def test_nvfp4_context_matches_backend():
    ctx_lens = [100, 300, 7]
    benches = [_Bench("nvfp4", 128, max(ctx_lens) + 4, len(ctx_lens)) for _ in range(2)]
    try:
        _nvfp4_context(benches, ctx_lens)
    finally:
        for bench in benches:
            bench.shutdown()


@pytest.mark.skipif(
    _sm_version() != 107, reason="the fused FP4 generation scatter (Q quantization) is SM107-only"
)
def test_nvfp4_generation_scatter_matches_backend():
    ops = _ops()
    ctx_lens = [100, 300, 7]
    batch = len(ctx_lens)
    q_len = 1
    benches = [_Bench("nvfp4", 128, max(ctx_lens) + 4, batch) for _ in range(2)]
    try:
        _, mqa, _, gen = _nvfp4_context(benches, ctx_lens)
        lengths = [c + q_len for c in ctx_lens]
        for bench in benches:
            bench.grow(lengths)
        mds = [
            b.metadata([q_len] * batch, ctx_lens, context=False, prompt_lens=ctx_lens)
            for b in benches
        ]
        num_tokens = batch * q_len
        g = torch.Generator(device="cuda").manual_seed(6)
        fused_q = torch.empty(num_tokens, H, LORA + ROPE, dtype=torch.bfloat16, device="cuda")
        fused_q.uniform_(-1, 1, generator=g)
        q_pe = torch.empty(num_tokens, H, ROPE, dtype=torch.bfloat16, device="cuda")
        q_pe.uniform_(-1, 1, generator=g)
        gen_latent = torch.empty(num_tokens, LORA + ROPE, dtype=torch.bfloat16, device="cuda")
        gen_latent.uniform_(-1, 1, generator=g)
        results = []
        for side, md in enumerate(mds):
            fq = fused_q.clone()
            unused = dict(
                cu_q_seqlens=torch.zeros(batch + 1, dtype=torch.int32, device="cuda"),
                cu_kv_seqlens=torch.zeros(batch + 1, dtype=torch.int32, device="cuda"),
                fmha_scheduler_counter=torch.zeros(1, dtype=torch.uint32, device="cuda"),
                mla_bmm1_scale=torch.zeros(2, dtype=torch.float32, device="cuda"),
                mla_bmm2_scale=torch.zeros(1, dtype=torch.float32, device="cuda"),
                quant_q_buffer=torch.zeros(
                    num_tokens, H, LORA + ROPE, dtype=torch.uint8, device="cuda"
                ),
            )
            args = (fq, q_pe, gen_latent, md, *unused.values())
            if side == 0:
                mqa.mla_rope_generation(*args, fuse_fp4_q_quant=True)
            else:
                ops.cache.mla_rope_generation(gen, *args, fuse_fp4_q_quant=True)
            torch.cuda.synchronize()
            results.append([("fused_q", fq)] + _fp4_q_buffers(md))
        assert len(results[0]) > 1
        for (name, a), (_, b) in zip(*results):
            _assert_same(f"generation {name}", a, b)
        _assert_same_pools(*benches)
    finally:
        for bench in benches:
            bench.shutdown()


def test_nvfp4_split_k_context_append_and_prefix_gather_match_backend():
    ops = _ops()
    # The shared-tile FP4 context update needs 16-token-aligned chunk starts (K3 chunks are).
    cached = [256, 192]
    new = [128, 176]
    total = [c + n for c, n in zip(cached, new)]
    batch = len(cached)
    benches = [_Bench("nvfp4", 128, max(total), batch) for _ in range(2)]
    try:
        mha, _ = _backend_roles("nvfp4")
        ctx, _ = _family_roles("nvfp4")

        def append(side, q, latent, md):
            if side == 0:
                mha.append_fp4_mla_context_cache(q, latent, md)
            else:
                ops.cache.append_fp4_mla_context_cache(ctx, q, latent, md)

        for bench in benches:
            bench.grow(cached)
        q, _, _, latent = _context_inputs(sum(cached), seed=7)
        qs = []
        for side, bench in enumerate(benches):
            md = bench.metadata(cached, [0] * batch, context=True, cached_kv=True)
            qs.append(q.clone())
            append(side, qs[-1], latent, md)
        torch.cuda.synchronize()
        _assert_same("q after prefix append", *qs)
        _assert_same_pools(*benches)

        for bench in benches:
            bench.grow(total)
        q, _, _, latent = _context_inputs(sum(new), seed=8)
        sides = [_BackendCalls(mha), _FamilyCalls(ctx)]
        records = []
        for side, (calls, bench) in enumerate(zip(sides, benches)):
            md = bench.metadata(new, cached, context=True, chunk_size=64, cached_kv=True)
            assert md.chunked_loop_num > 1
            record = []
            for loop_idx in range(md.chunked_loop_num):
                chunk_kv, chunk_pe = calls.load_chunked(md, loop_idx)
                record += [
                    (f"chunk {loop_idx} latent", chunk_kv.clone()),
                    (f"chunk {loop_idx} k_pe", chunk_pe.clone()),
                ]
            q_side = q.clone()
            append(side, q_side, latent, md)
            torch.cuda.synchronize()
            record.append(("q after chunk append", q_side))
            records.append(record)
        for (name, a), (_, b) in zip(*records):
            _assert_same(name, a, b)
        _assert_same_pools(*benches)
    finally:
        for bench in benches:
            bench.shutdown()
