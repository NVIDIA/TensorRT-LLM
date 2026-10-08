# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime-level parity: KimiKDALinearAttention fused verify vs sequential verify.

Simulates two chained speculative-verification rounds through
``KimiKDALinearAttention.forward_verify`` in both worlds:

* Sequential world: the legacy intermediate-buffer path
  (``forward_verify_sequential``) plus the manager's legacy promotion
  (copy the accepted step's conv window / SSM state into the live pools).
* Fused world: the ``trtllm::kda_mtp_decode`` replay path
  (``forward_verify_fused``) with per-slot replay caches, in-place state
  commit after the golden token, and only the accepted-draft count
  recorded between rounds.

Identical hidden states are fed to both worlds; the fused world additionally
uses fused QKV(G) and full- or low-rank gate projections with multi-stream overlap. With mixed
per-request acceptance between rounds, matching round-2 outputs proves the
projection fusion, packed row/slot bookkeeping, conv-window seeding and the
in-kernel gated RMSNorm epilogue reproduce the promoted-state semantics.

Requires 1 Blackwell GPU, fla-core, nvidia-cutlass-dsl. Skips otherwise.
"""

from types import SimpleNamespace

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401
    from fla.ops.kda import fused_recurrent_kda  # noqa: F401

    # The model module transitively imports the optional deps above, so it
    # must stay behind the guard too or collection fails instead of skipping.
    from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
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
H = 8  # per-rank head count outside the drop's tuned set — general variant
K = 128
W = 4
M = 2  # draft tokens per round
LB = -5.0


@torch.no_grad()
def _make_runtime(
    seed, aux_stream=None, checkpoint_fp8=False, *, num_heads=H, use_full_rank_gate=True
):
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
            use_full_rank_gate=use_full_rank_gate,
            gate_lower_bound=LB,
        ),
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    model_config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES if checkpoint_fp8 else None)
    )
    rt = KimiKDALinearAttention(
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
    # The fused-verify conv constants are prebuilt at weight-load finalize
    # time in production; the runtime never computes them lazily. Mirror
    # that here (after the random init above, which they snapshot).
    rt._build_mtp_conv_weights()
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


def _assert_fused_epilogue_call(call, runtime, use_full_rank_gate):
    """The fused verify ran with the in-kernel output norm on packed rows."""
    assert call["fuse_output_norm"]
    assert call["packed_token_layout"]
    assert call["cu_seqlens"] is None
    assert not call["mxfp8_output"]
    onorm_g = call["onorm_g"]
    assert onorm_g.stride(-1) == 1 and onorm_g.stride(-2) == K
    if use_full_rank_gate:
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


@torch.no_grad()
@pytest.mark.parametrize(
    ("num_heads", "use_full_rank_gate", "checkpoint_fp8", "large_state_stride"),
    [
        (H, True, False, False),
        (H, True, True, False),
        (16, False, False, False),
        (64, False, False, False),
        (64, False, False, True),
    ],
    ids=["full-rank-bf16", "full-rank-fp8", "low-rank-h16", "low-rank-h64", "large-state-stride"],
)
def test_fused_vs_sequential_two_rounds(
    num_heads, use_full_rank_gate, checkpoint_fp8, large_state_stride
):
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    torch.manual_seed(0)
    B = 4
    T = M + 1
    rt_seq = _make_runtime(
        seed=1,
        checkpoint_fp8=checkpoint_fp8,
        num_heads=num_heads,
        use_full_rank_gate=use_full_rank_gate,
    )
    rt_fused = _make_runtime(
        seed=1,
        aux_stream=torch.cuda.Stream(),
        checkpoint_fp8=checkpoint_fp8,
        num_heads=num_heads,
        use_full_rank_gate=use_full_rank_gate,
    )
    rt_fused.finalize_decode_weights()
    if checkpoint_fp8:
        from tensorrt_llm._torch.modules.linear import Linear

        for runtime in (rt_seq, rt_fused):
            for module in runtime.modules():
                if isinstance(module, Linear):
                    module.post_load_weights()
        assert rt_fused.qkvg_proj is not None
        assert rt_fused._bfa_proj_weight is not None
    else:
        assert rt_fused._qkvg_proj_weight is not None
        assert rt_fused._bfa_proj_weight is not None
    assert rt_fused._onorm_w_f32 is not None
    mtp_calls = _capture_mtp_verify(rt_fused)
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

    gen = torch.Generator(device="cuda").manual_seed(3)

    def tokens(scale=0.5):
        return (
            torch.randn(B * T, HIDDEN, generator=gen, device="cuda", dtype=torch.float32) * scale
        ).to(torch.bfloat16)

    ok = True
    # ---- Round 1 (no pending drafts) ----
    x1 = tokens()
    out1_seq = rt_seq._project_output(
        rt_seq.forward_verify_sequential(
            x1, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices
        )
    )
    with with_multi_stream(True):
        out1_fused = rt_fused._project_output(
            rt_fused.forward_verify(
                x1, T, cache_fused, conv_pool_fused, ssm_pool_fused, slot_indices
            )
        )
    _assert_fused_epilogue_call(mtp_calls[-1], rt_fused, use_full_rank_gate)
    print("round 1:")
    ok &= _rep("out", out1_fused, out1_seq)

    # ---- Acceptance: 0, 1, 2, 0 drafts across the 4 requests ----
    accept = torch.tensor([0, 1, 2, 0], dtype=torch.long, device="cuda")
    _promote_sequential(cache_seq, conv_pool_seq, ssm_pool_seq, accept)
    cache_fused.prev_num_accepted_tokens.copy_(accept.to(torch.int32))

    # ---- Round 2 (fused path replays the accepted drafts) ----
    x2 = tokens()
    out2_seq = rt_seq._project_output(
        rt_seq.forward_verify_sequential(
            x2, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices
        )
    )
    core2_fused = x2.new_empty(B * T, num_heads, K)
    with with_multi_stream(True):
        result2_fused = rt_fused.forward_verify(
            x2,
            T,
            cache_fused,
            conv_pool_fused,
            ssm_pool_fused,
            slot_indices,
            output=core2_fused,
        )
    assert result2_fused is core2_fused
    assert len(mtp_calls) == 2
    _assert_fused_epilogue_call(mtp_calls[-1], rt_fused, use_full_rank_gate)
    out2_fused = rt_fused._project_output(core2_fused)
    print("round 2 (mixed replay):")
    ok &= _rep("out", out2_fused, out2_seq)

    # Committed pool state cross-check: fused pool holds the state after
    # round-2's golden token; reproduce it in the sequential world by
    # promoting with accept=0 (golden only).
    _promote_sequential(
        cache_seq, conv_pool_seq, ssm_pool_seq, torch.zeros(B, dtype=torch.long, device="cuda")
    )
    ok &= _rep("committed ssm", ssm_pool_fused, ssm_pool_seq)

    assert ok


def _run_one_verify_round(runtime, x, batch, seed, **verify_kwargs):
    conv_pool, ssm_pool = _make_pools(batch, seed=seed)
    layer_cache = _make_fused_layer_cache(batch, conv_pool)
    slot_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    return runtime.forward_verify(
        x, M + 1, layer_cache, conv_pool, ssm_pool, slot_indices, **verify_kwargs
    )


@torch.no_grad()
def test_fused_verify_falls_back_without_output_norm_weights():
    """Without finalized o_norm weights the verify keeps the Python output gate."""
    batch = 2
    runtime = _make_runtime(seed=5)
    runtime.finalize_decode_weights()
    x = (torch.randn(batch * (M + 1), HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)
    fused_out = runtime._project_output(_run_one_verify_round(runtime, x, batch, seed=6))

    runtime._onorm_w_f32 = None
    calls = _capture_mtp_verify(runtime)
    fallback_out = runtime._project_output(_run_one_verify_round(runtime, x, batch, seed=6))

    assert len(calls) == 1
    assert "fuse_output_norm" not in calls[0]
    assert calls[0]["cu_seqlens"] is not None
    assert _rep("fallback vs fused epilogue", fallback_out, fused_out)


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
    runtime.finalize_decode_weights()
    stub = _PrequantizedProjectionStub(HIDDEN)
    runtime.o_proj = stub
    calls = []

    def _fake_mtp_verify(**kwargs):
        calls.append(kwargs)
        if kwargs["mxfp8_output"]:
            activation = torch.zeros(rows, H * K, device="cuda").to(torch.float8_e4m3fn)
            return activation, torch.zeros(128 * H * 4, dtype=torch.uint8, device="cuda")
        return torch.zeros(1, rows, H, K, dtype=torch.bfloat16, device="cuda")

    runtime._dispatch.mtp_verify = _fake_mtp_verify
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    output = x.new_empty(rows, H, K) if with_output else None
    core = _run_one_verify_round(
        runtime, x, batch, seed=10, output=output, allow_prequantized_output=allow
    )
    out = runtime._project_output(core)

    expect_mxfp8 = allow and not with_output
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
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("needs the Rubin CuTe DSL")
    batch = 4
    runtime = _make_runtime(seed=11)
    runtime.finalize_decode_weights()
    runtime.o_proj = _Fp8BlockScaleWeightReadLinear.from_linear(runtime.o_proj)
    assert runtime.o_proj.supports_prequantized_input
    calls = _capture_mtp_verify(runtime)
    x = (torch.randn(batch * (M + 1), HIDDEN, device="cuda") * 0.5).to(torch.bfloat16)

    bf16_out = runtime._project_output(_run_one_verify_round(runtime, x, batch, seed=12))
    mxfp8_core = _run_one_verify_round(runtime, x, batch, seed=12, allow_prequantized_output=True)
    mxfp8_out = runtime._project_output(mxfp8_core)

    assert [call["mxfp8_output"] for call in calls] == [False, True]
    assert mxfp8_core.activation.dtype is torch.float8_e4m3fn
    assert mxfp8_core.scale.dtype is torch.uint8
    assert _rep("mxfp8 vs bf16 o_proj input", mxfp8_out, bf16_out)
