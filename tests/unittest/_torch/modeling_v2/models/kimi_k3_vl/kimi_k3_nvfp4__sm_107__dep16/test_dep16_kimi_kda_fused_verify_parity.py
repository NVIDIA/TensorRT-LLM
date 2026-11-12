# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-level parity: the dep16 K3KDALinearAttention fused verify vs a sequential FLA
reference.

Simulates two chained speculative-verification rounds in both worlds:

* Sequential world (``_reference_verify``): per step, the depthwise convolution on the gathered
  window and ``fla.ops.kda.fused_recurrent_kda``, each step's conv window and state written to
  intermediate buffers, plus the legacy promotion (copy the accepted step's conv window / SSM
  state into the live pools).
* Fused world: ``K3KDALinearAttention.forward_verify`` on per-slot replay caches (the
  ``kda_mtp_decode`` entry), in-place state commit after the golden token, and
  only the accepted-draft count recorded between rounds.

Both worlds share the layer's FP8 projections (the fused QKVG and o_proj weight-read GEMMs, the K3
low-M GEMM on the low-M rows as dep16 runs them) and take identical hidden states; the fused world
runs its ``F.linear`` [f_a | b] -> f_b gate projection on the aux stream (dep16 never takes the
f_a-mode fab kernels: they serve at most 16 heads, dep16 runs all 96 on a rank). With
mixed per-request acceptance between rounds, matching round-2 outputs proves the packed row/slot
bookkeeping, conv-window seeding and the in-kernel gated RMSNorm epilogue reproduce the
promoted-state semantics.

Requires 1 Blackwell GPU, fla-core, nvidia-cutlass-dsl. Skips otherwise.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401
    from fla.ops.kda import fused_recurrent_kda  # noqa: F401

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__dep16.modeling import (  # noqa: E501
        K3KDALinearAttention,
        _Fp8BlockScaleWeightReadLinear,
    )

    # The model module transitively imports the optional deps above, so it
    # must stay behind the guard too or collection fails instead of skipping.
    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor in (100, 103, 107)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_blackwell(), reason="needs sm100/sm103/sm107"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

HIDDEN = 512
H = 8  # per-rank head count outside the tuned set {2, 12, 32} — general variant
K = 128
W = 4
M = 2  # draft tokens per round
LB = -5.0


def _make_runtime(seed, aux_stream=None, *, num_heads=H):
    """A finalized layer on seeded FP8 checkpoint-like weights, with its fused QKVG and o_proj
    swapped to dep16's FP8 weight-read GEMM as the model loader does."""
    # A real KimiLinearConfig (not a SimpleNamespace) so the runtime sees the
    # same config surface it does in production. ``linear_attn_config`` carries
    # the per-layer KDA params the runtime reads plus the (unused here)
    # kda_layers/full_attn_layers schedule the config's own validation requires.
    cfg = KimiLinearConfig(
        hidden_size=HIDDEN,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            kda_layers=[1],
            full_attn_layers=[],
            num_heads=num_heads,
            head_dim=K,
            short_conv_kernel_size=W,
            use_full_rank_gate=True,
            gate_lower_bound=LB,
        ),
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    model_config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES))
    rt = K3KDALinearAttention(
        cfg,
        layer_idx=0,
        aux_stream=aux_stream,
        model_config=model_config,
    ).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for name, p in rt.named_parameters():
        if p.dtype == torch.float8_e4m3fn:
            p.copy_(torch.randint(-32, 33, p.shape, generator=gen, device="cuda").to(p.dtype))
        elif name.endswith("weight_scale"):
            p.fill_(0.001)
        elif name.endswith("input_scale"):
            p.fill_(1.0)
        elif name.endswith("A_log"):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda", dtype=torch.float32) * 0.5)
        elif name.endswith("dt_bias"):
            p.copy_(torch.randn(p.shape, generator=gen, device="cuda", dtype=torch.float32) * 0.1)
        else:
            p.copy_(
                (torch.randn(p.shape, generator=gen, device="cuda", dtype=torch.float32) * 0.03).to(
                    p.dtype
                )
            )
    # Weight-load finalize as in production (after the random init above, which the fused
    # projections and kernel constants snapshot).
    rt.finalize_decode_weights()
    rt._build_mtp_conv_weights()
    for name in ("qkvg_proj", "o_proj"):
        module = _Fp8BlockScaleWeightReadLinear.from_linear(getattr(rt, name))
        setattr(rt, name, module)
        module.post_load_weights()
    return rt


def _make_pools(B, seed, *, num_heads=H):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    d = num_heads * K
    conv_pool = (
        torch.randn(B, 3 * d, W - 1, generator=gen, device="cuda", dtype=torch.float32) * 0.5
    ).to(torch.bfloat16)
    ssm_pool = torch.randn(B, num_heads, K, K, generator=gen, device="cuda", dtype=torch.float32)
    ssm_pool *= torch.linspace(0.5, 1.5, K, device="cuda").view(1, 1, K, 1)
    return conv_pool, ssm_pool


def _make_fused_layer_cache(B, conv_pool, *, num_heads=H):
    """Replay caches shaped like PythonMambaCacheManager's KDA allocation,
    with the committed conv window seeded from the base pool (the prefill
    seeding contract: the base pool stores committed columns directly)."""
    d = num_heads * K
    S = W - 1 + M

    def _conv_cache(section):
        cache = torch.zeros(B, S, d, device="cuda", dtype=torch.float32).transpose(-1, -2)
        cache[:, :, : W - 1] = conv_pool[:, section * d : (section + 1) * d].float()
        return cache

    return SimpleNamespace(
        kda_conv_q=_conv_cache(0),
        kda_conv_k=_conv_cache(1),
        kda_conv_v=_conv_cache(2),
        kda_k_cache=torch.zeros(B, M, d, device="cuda", dtype=torch.bfloat16),
        kda_g_cache=torch.zeros(B, M, d, device="cuda", dtype=torch.float32),
        kda_v_cache=torch.zeros(B, M, d, device="cuda", dtype=torch.bfloat16),
        kda_beta_cache=torch.zeros(B, M, num_heads, device="cuda", dtype=torch.bfloat16),
        prev_num_accepted_tokens=torch.zeros(B, dtype=torch.int32, device="cuda"),
        has_kda_replay_caches=True,
        intermediate_conv_window=None,
        intermediate_ssm=None,
    )


def _make_seq_layer_cache(B, *, num_heads=H):
    d = num_heads * K
    return SimpleNamespace(
        kda_k_cache=None,
        has_kda_replay_caches=False,
        intermediate_conv_window=torch.zeros(
            B, M + 1, 3 * d, W - 1, device="cuda", dtype=torch.bfloat16
        ),
        intermediate_ssm=torch.zeros(B, M + 1, num_heads, K, K, device="cuda", dtype=torch.float32),
    )


@torch.no_grad()
def _reference_verify(rt, x2d, num_steps, layer_cache, conv_pool, ssm_pool, slot_indices):
    """Per-step FLA verification on gathered copies of the live rows (the pools stay read-only),
    the layer's own projections in front and its gated output RMSNorm behind. Each step's conv
    window and state go to ``layer_cache``'s intermediate buffers. Returns the normalized core
    ``[tokens, H, D]``."""
    from fla.ops.kda import fused_recurrent_kda

    d, hd, heads = rt.proj_size, rt.head_dim, rt.num_heads
    batch = x2d.shape[0] // num_steps
    x = x2d.view(batch, num_steps, -1)
    q_in, k_in, v_in, onorm_g = rt.qkvg_proj(x).split(d, dim=-1)
    bfa = F.linear(x, rt._bfa_proj_weight)
    beta = bfa[..., hd : hd + heads].float()
    gate = F.linear(bfa[..., :hd], rt.f_b_proj.weight).unflatten(-1, (heads, hd))
    rows = slot_indices.long()
    window = conv_pool.index_select(0, rows)  # [B, 3D, W - 1]
    state = ssm_pool.index_select(0, rows).float()
    conv_weight = (
        torch.cat([rt.q_conv1d.weight, rt.k_conv1d.weight, rt.v_conv1d.weight]).squeeze(1).float()
    )  # [3D, W]
    outputs = []
    for t in range(num_steps):
        step = torch.cat([q_in[:, t], k_in[:, t], v_in[:, t]], dim=-1).to(window.dtype)
        full = torch.cat([window, step.unsqueeze(-1)], dim=-1)  # [B, 3D, W]
        window = full[..., 1:]
        conv = F.silu((full.float() * conv_weight).sum(-1)).to(x2d.dtype)
        q_t, k_t, v_t = (
            part.unflatten(-1, (heads, hd)).unsqueeze(1) for part in conv.split(d, dim=-1)
        )
        o_t, state = fused_recurrent_kda(
            q=q_t,
            k=k_t,
            v=v_t,
            g=gate[:, t : t + 1],
            beta=beta[:, t : t + 1],
            A_log=rt.A_log,
            dt_bias=rt.dt_bias,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=rt.gate_lower_bound,
            state_v_first=True,
        )
        outputs.append(o_t)
        layer_cache.intermediate_conv_window[:batch, t] = window.to(
            layer_cache.intermediate_conv_window.dtype
        )
        layer_cache.intermediate_ssm[:batch, t] = state.to(layer_cache.intermediate_ssm.dtype)
    o = torch.cat(outputs, dim=1)  # [B, T, H, D]
    return rms_norm_gated_token_major(
        o.reshape(-1, hd),
        onorm_g.reshape(-1, heads, hd),
        rt.o_norm.weight,
        rt.o_norm.eps,
        gate_activation="sigmoid",
    ).reshape(-1, heads, hd)


def _promote_sequential(layer_cache, conv_pool, ssm_pool, accept):
    """The manager's legacy promotion: accepted step's states -> pools."""
    B = conv_pool.shape[0]
    rows = torch.arange(B, device="cuda")
    conv_pool.copy_(layer_cache.intermediate_conv_window[rows, accept])
    ssm_pool.copy_(layer_cache.intermediate_ssm[rows, accept])


def _capture_mtp_verify(runtime):
    """Record the kwargs of every fused verify launch, then run it."""
    calls = []
    mtp_verify = runtime._dispatch.mtp_verify

    def _capturing_mtp_verify(**kwargs):
        calls.append(kwargs)
        return mtp_verify(**kwargs)

    runtime._dispatch.mtp_verify = _capturing_mtp_verify
    return calls


def _capture_mtp_verify_fused_fb(runtime):
    """Record the kwargs of every fused verify launch with the forget gate from ``f_a``, then
    run it."""
    calls = []
    mtp_verify_fused_fb = runtime._dispatch.mtp_verify_fused_fb

    def _capturing_mtp_verify_fused_fb(**kwargs):
        calls.append(kwargs)
        return mtp_verify_fused_fb(**kwargs)

    runtime._dispatch.mtp_verify_fused_fb = _capturing_mtp_verify_fused_fb
    return calls


def _assert_fused_epilogue_call(call, runtime):
    """The fused verify ran with the in-kernel output norm on packed rows."""
    assert call["fuse_output_norm"]
    assert call["packed_token_layout"]
    assert call["cu_seqlens"] is None
    assert not call["mxfp8_output"]
    onorm_g = call["onorm_g"]
    assert onorm_g.stride(-1) == 1 and onorm_g.stride(-2) == K
    # The gate and q/k/v stay strided views of the fused QKVG output.
    row_stride = 4 * runtime.proj_size
    assert onorm_g.stride(1) == row_stride
    for name in ("x_q", "x_k", "x_v"):
        assert call[name].stride(1) == row_stride
        assert call[name].stride(-1) == 1
    assert call["beta"].stride(-1) == 1


def _rep(name, a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
    print(f"  {name}: cos={cos:.6f} rel_l2={rel:.3e}")
    return cos > 0.999 and rel < 3e-2


def _make_kvcm_replay_manager(num_heads):
    """The target's V2 hybrid manager, whose KVCM pools hold the KDA states with their replay caches in the same
    slots, with a host tier to offload to."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.cache.k3_sm107_cache_manager import (  # noqa: E501
        KimiK3Sm107CacheManager,
    )
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    return KimiK3Sm107CacheManager(
        mamba_d_state=K,
        mamba_d_conv=W,
        mamba_num_heads=num_heads,
        mamba_n_groups=num_heads,
        mamba_head_dim=K,
        mamba_num_layers=2,
        mamba_layer_mask=[True, True],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=128, host_cache_size=64 << 20, enable_block_reuse=False
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=0,
        num_kv_heads=0,
        head_dim=K,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        spec_config=MTPDecodingConfig(max_draft_len=M),
        layer_mask=[False, False],
        conv_state_layout="q_k_v",
        kda_replay_num_spec=M,
    )


def _offload_and_resume(manager, request_ids):
    """Suspend the requests' KV caches, push their pages to the host tier with newer caches, and resume them; returns
    the slots they resume in."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import CACHE_LEVEL1, LayerId, _introspection

    stream = torch.cuda.current_stream().cuda_stream
    caches = [manager.kv_cache_map[rid] for rid in request_ids]
    for cache in caches:
        cache.suspend()
    group = manager.impl.get_layer_group_id(LayerId(0))
    pool = next(
        pool
        for pool in manager.impl.pool_group_descs
        if any(variant.layer_group_id == group for variant in pool.slot_desc.variants)
    )
    pressure = []
    try:
        for _ in range(pool.num_slots + len(caches)):
            other = manager.impl.create_kv_cache()
            pressure.append(other)
            assert other.resume(stream)
            other.suspend()
    finally:
        for other in pressure:
            other.close()
    # The suspended requests' pages must have been evicted to the host tier, or
    # the resume below would not exercise the offload path at all.
    assert max(_introspection.storage_utilization(manager.impl, CACHE_LEVEL1)) > 0
    new_slots = []
    for cache in reversed(caches):
        assert cache.resume(stream)
        new_slots.append(cache.get_ssm_block_base_index(group))
    return torch.tensor(list(reversed(new_slots)), dtype=torch.int32, device="cuda")


@torch.no_grad()
@pytest.mark.parametrize(
    ("num_heads", "large_state_stride", "kvcm_replay"),
    [
        (H, False, False),
        (64, False, False),
        (96, False, False),
        (64, True, False),
        (H, False, True),
    ],
    ids=["h8", "h64", "h96", "large-state-stride", "h8-kvcm-offload"],
)
def test_fused_vs_sequential_two_rounds(request, num_heads, large_state_stride, kvcm_replay):
    """With ``kvcm_replay``, the fused world keeps its states and replay caches in a V2 manager's KVCM pools, and
    the requests are offloaded to the host tier and resumed in other slots between the two rounds.

    The [f_a | b] rows go through ``F.linear`` [f_a | b] -> f_b (on the aux stream) and
    ``kda_mtp_decode`` with the fused output norm, at every head count: dep16 never launches the
    f_a-mode verify (``mtp_verify_fused_fb``)."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    torch.manual_seed(0)
    B = 4
    T = M + 1
    rt_seq = _make_runtime(seed=1, num_heads=num_heads)
    rt_fused = _make_runtime(seed=1, aux_stream=torch.cuda.Stream(), num_heads=num_heads)
    assert rt_fused.qkvg_proj is not None
    assert rt_fused._bfa_proj_weight is not None
    assert rt_fused._onorm_w_f32 is not None
    mtp_calls = _capture_mtp_verify(rt_fused)
    fb_calls = _capture_mtp_verify_fused_fb(rt_fused)
    slot_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    conv_pool_seq, ssm_pool_seq = _make_pools(B, seed=2, num_heads=num_heads)
    conv_pool_fused = conv_pool_seq.clone()
    if large_state_stride:
        # Only four small states are populated. The gaps reproduce V2's
        # coalesced layer layout, with the last slot beyond INT32_MAX elements.
        state_stride = ((2**31 // (B - 1)) // (K * K) + 1) * K * K
        storage_size = (B - 1) * state_stride + num_heads * K * K
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < storage_size * ssm_pool_seq.element_size() + 2**30:
            pytest.skip("large-stride regression needs 9 GiB of free GPU memory")
        storage = torch.empty(storage_size, dtype=ssm_pool_seq.dtype, device="cuda")
        ssm_pool_fused = storage.as_strided(ssm_pool_seq.shape, (state_stride, K * K, K, 1))
        ssm_pool_fused.copy_(ssm_pool_seq)
    else:
        ssm_pool_fused = ssm_pool_seq.clone()
    cache_seq = _make_seq_layer_cache(B, num_heads=num_heads)
    cache_fused = _make_fused_layer_cache(B, conv_pool_fused, num_heads=num_heads)
    fused_slots = slot_indices
    replay_manager = _make_kvcm_replay_manager(num_heads) if kvcm_replay else None
    if replay_manager is not None:
        request.addfinalizer(replay_manager.shutdown)
        request_ids = list(range(101, 101 + B))
        replay_manager.add_dummy_requests(request_ids, token_nums=[32] * B, is_gen=False)
        fused_slots = torch.tensor(
            replay_manager.get_state_indices(request_ids), dtype=torch.int32, device="cuda"
        )
        cache_fused = replay_manager.mamba_layer_cache(0)
        conv_pool_fused = cache_fused.conv
        ssm_pool_fused = cache_fused.temporal
        conv_pool_fused.index_copy_(0, fused_slots.long(), conv_pool_seq)
        ssm_pool_fused.index_copy_(0, fused_slots.long(), ssm_pool_seq)
        replay_manager.on_state_transfer_complete(request_ids)

    gen = torch.Generator(device="cuda").manual_seed(3)

    def tokens(scale=0.5):
        return (
            torch.randn(B * T, HIDDEN, generator=gen, device="cuda", dtype=torch.float32) * scale
        ).to(torch.bfloat16)

    ok = True
    # ---- Round 1 (no pending drafts) ----
    x1 = tokens()
    projected = rt_fused._project_verify_inputs(x1.view(B, T, -1), B * T)
    assert len(projected) == 6
    beta_view = projected[4]
    # beta is a view of the [f_a | b] projection on the F.linear chain.
    assert beta_view.stride(-2) == rt_fused._bfa_proj_weight.shape[0]
    assert beta_view.stride(-1) == 1
    out1_seq = rt_seq._project_output(
        _reference_verify(rt_seq, x1, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices)
    )
    with with_multi_stream(True):
        out1_fused = rt_fused._project_output(
            rt_fused.forward_verify(x1, T, cache_fused, ssm_pool_fused, fused_slots)
        )
    assert not fb_calls and len(mtp_calls) == 1
    assert mtp_calls[-1]["use_pdl"]
    _assert_fused_epilogue_call(mtp_calls[-1], rt_fused)
    print("round 1:")
    ok &= _rep("out", out1_fused, out1_seq)

    # ---- Acceptance: 0, 1, 2, 0 drafts across the 4 requests ----
    accept = torch.tensor([0, 1, 2, 0], dtype=torch.long, device="cuda")
    _promote_sequential(cache_seq, conv_pool_seq, ssm_pool_seq, accept)
    cache_fused.prev_num_accepted_tokens.index_copy_(0, fused_slots.long(), accept.to(torch.int32))
    if replay_manager is not None:
        fused_slots = _offload_and_resume(replay_manager, request_ids)

    # ---- Round 2 (fused path replays the accepted drafts) ----
    x2 = tokens()
    out2_seq = rt_seq._project_output(
        _reference_verify(rt_seq, x2, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices)
    )
    core2_fused = x2.new_empty(B * T, num_heads, K)
    with with_multi_stream(True):
        result2_fused = rt_fused.forward_verify(
            x2, T, cache_fused, ssm_pool_fused, fused_slots, output=core2_fused
        )
    assert result2_fused is core2_fused
    assert len(mtp_calls) == 2 and not fb_calls
    _assert_fused_epilogue_call(mtp_calls[-1], rt_fused)
    out2_fused = rt_fused._project_output(core2_fused)
    print("round 2 (mixed replay):")
    ok &= _rep("out", out2_fused, out2_seq)

    # Committed pool state cross-check: fused pool holds the state after
    # round-2's golden token; reproduce it in the sequential world by
    # promoting with accept=0 (golden only).
    _promote_sequential(
        cache_seq, conv_pool_seq, ssm_pool_seq, torch.zeros(B, dtype=torch.long, device="cuda")
    )
    ok &= _rep("committed ssm", ssm_pool_fused.index_select(0, fused_slots.long()), ssm_pool_seq)

    assert ok


def _run_one_verify_round(runtime, x, batch, seed, **verify_kwargs):
    conv_pool, ssm_pool = _make_pools(batch, seed=seed)
    layer_cache = _make_fused_layer_cache(batch, conv_pool)
    slot_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    return runtime.forward_verify(x, M + 1, layer_cache, ssm_pool, slot_indices, **verify_kwargs)


@torch.no_grad()
def test_fused_verify_requires_finalized_weights():
    """The verify runs on the kernel constants finalize_decode_weights builds; without them it
    raises instead of computing the output gate another way."""
    batch = 2
    runtime = _make_runtime(seed=5)
    runtime._onorm_w_f32 = None
    x = (torch.randn(batch * (M + 1), HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="finalize_decode_weights"):
        _run_one_verify_round(runtime, x, batch, seed=6)


class _PrequantizedProjectionStub(torch.nn.Module):
    """o_proj stand-in that records the prequantized pair it receives."""

    supports_prequantized_input = True

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.calls = []

    def forward(self, x):
        return x.new_zeros(x.shape[0], self.out_features)

    def forward_prequantized(self, activation, activation_scale):
        self.calls.append((activation, activation_scale))
        return torch.zeros(
            activation.shape[0], self.out_features, dtype=torch.bfloat16, device="cuda"
        )


@torch.no_grad()
@pytest.mark.parametrize("allow, with_output", [(True, False), (False, False), (True, True)])
def test_fused_verify_mxfp8_output_gating(allow, with_output):
    """MXFP8 output only when allowed, unbuffered and accepted by o_proj."""
    batch = 2
    rows = batch * (M + 1)
    runtime = _make_runtime(seed=9)
    stub = _PrequantizedProjectionStub(HIDDEN)
    runtime.o_proj = stub
    calls = []

    def _fake_mtp_verify(**kwargs):
        calls.append(kwargs)
        if kwargs["mxfp8_output"]:
            activation = torch.zeros(rows, H * K, device="cuda").to(torch.float8_e4m3fn)
            return activation, torch.zeros(128 * H * 4, dtype=torch.uint8, device="cuda")
        return torch.zeros(1, rows, H, K, dtype=torch.bfloat16, device="cuda")

    fb_calls = []

    def _fake_mtp_verify_fused_fb(**kwargs):
        fb_calls.append(kwargs)
        return torch.zeros(1, rows, H, K, dtype=torch.bfloat16, device="cuda")

    runtime._dispatch.mtp_verify = _fake_mtp_verify
    runtime._dispatch.mtp_verify_fused_fb = _fake_mtp_verify_fused_fb
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    output = x.new_empty(rows, H, K) if with_output else None
    core = _run_one_verify_round(
        runtime, x, batch, seed=10, output=output, allow_prequantized_output=allow
    )
    out = runtime._project_output(core)

    expect_mxfp8 = allow and not with_output
    assert len(calls) == 1 and not fb_calls
    assert calls[-1]["fuse_output_norm"]
    assert calls[-1]["mxfp8_output"] is expect_mxfp8
    assert len(stub.calls) == int(expect_mxfp8)
    assert out.shape == (rows, HIDDEN)
    if with_output:
        assert core is output


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 7),
    reason="the MXFP8 verify epilogue feeds the Rubin (SM107) CuTe MXFP8 GEMM",
)
def test_fused_verify_mxfp8_output_matches_bf16_projection():
    """SM107: epilogue-quantized o_proj input matches the BF16 core path."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("needs the Rubin CuTe DSL")
    batch = 4
    runtime = _make_runtime(seed=11)
    assert isinstance(runtime.o_proj, _Fp8BlockScaleWeightReadLinear)
    assert runtime.o_proj.supports_prequantized_input
    calls = _capture_mtp_verify(runtime)
    fb_calls = _capture_mtp_verify_fused_fb(runtime)
    x = (torch.randn(batch * (M + 1), HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)

    bf16_out = runtime._project_output(_run_one_verify_round(runtime, x, batch, seed=12))
    # The BF16 core: the fused-epilogue verify with BF16 output.
    assert [call["mxfp8_output"] for call in calls] == [False]
    mxfp8_core = _run_one_verify_round(runtime, x, batch, seed=12, allow_prequantized_output=True)
    mxfp8_out = runtime._project_output(mxfp8_core)

    assert [call["mxfp8_output"] for call in calls] == [False, True]
    assert not fb_calls
    assert mxfp8_core.activation.dtype is torch.float8_e4m3fn
    assert mxfp8_core.scale.dtype is torch.uint8
    assert _rep("mxfp8 vs bf16 o_proj input", mxfp8_out, bf16_out)


def _is_sm107():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


@pytest.mark.skipif(not _is_sm107(), reason="SM107 MXFP8 verify epilogue")
@pytest.mark.parametrize("bf16_rows", [False, True])
@torch.no_grad()
def test_fused_verify_with_fp8_weight_read_o_proj(bf16_rows, monkeypatch):
    """FP8-resident verify on SM107 with the K3 FP8 weight-read o_proj: the verify kernel hands
    o_proj its MXFP8 operand, or BF16 rows where o_proj runs them on the low-M GEMM."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_gemm import (
        k3_fp8w_lowm_gemm,
    )
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    B = 4
    T = M + 1
    monkeypatch.setitem(
        k3_fp8w_lowm_gemm._SM107_BF16_OVER_PREQUANTIZED_MAX_M,
        (HIDDEN, H * K),
        B * T if bf16_rows else 0,
    )
    rt_seq = _make_runtime(seed=1)
    rt_fused = _make_runtime(seed=1, aux_stream=torch.cuda.Stream())
    assert rt_fused.o_proj.lowm_weight_scale is not None
    assert rt_fused.o_proj.takes_prequantized_input(B * T) != bf16_rows
    mtp_calls = _capture_mtp_verify(rt_fused)
    fb_calls = _capture_mtp_verify_fused_fb(rt_fused)
    slot_indices = torch.arange(B, dtype=torch.int32, device="cuda")
    conv_pool_seq, ssm_pool_seq = _make_pools(B, seed=2)
    conv_pool_fused = conv_pool_seq.clone()
    ssm_pool_fused = ssm_pool_seq.clone()
    cache_seq = _make_seq_layer_cache(B)
    cache_fused = _make_fused_layer_cache(B, conv_pool_fused)
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(B * T, HIDDEN, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)

    out_seq = rt_seq._project_output(
        _reference_verify(rt_seq, x, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices)
    )
    with with_multi_stream(True):
        core_fused = rt_fused.forward_verify(
            x, T, cache_fused, ssm_pool_fused, slot_indices, allow_prequantized_output=True
        )
        out_fused = rt_fused._project_output(core_fused)
    assert len(mtp_calls) == 1 and not fb_calls
    assert mtp_calls[0]["mxfp8_output"] != bf16_rows
    assert _rep("out", out_fused, out_seq)
