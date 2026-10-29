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
uses fused QKVG and [f_a|b] projections with multi-stream overlap. With mixed
per-request acceptance between rounds, matching round-2 outputs proves the
projection fusion, packed row/slot bookkeeping, and in-kernel BF16 gated
RMSNorm reproduce the promoted-state semantics.

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
def _make_runtime(seed, aux_stream=None, checkpoint_fp8=False):
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
            num_heads=H,
            head_dim=K,
            short_conv_kernel_size=W,
            use_full_rank_gate=True,
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


def _make_pools(B, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    d = H * K
    conv_pool = (
        torch.randn(B, 3 * d, W - 1, generator=gen, device="cuda", dtype=torch.float32) * 0.5
    ).to(torch.bfloat16)
    ssm_pool = torch.randn(B, H, K, K, generator=gen, device="cuda", dtype=torch.float32)
    ssm_pool *= torch.linspace(0.5, 1.5, K, device="cuda").view(1, 1, K, 1)
    return conv_pool, ssm_pool


def _make_fused_layer_cache(B, conv_pool):
    """Replay caches shaped like PythonMambaCacheManager's KDA allocation,
    with the committed conv window seeded from the base pool (the prefill
    seeding contract: the base pool stores committed columns directly)."""
    d = H * K
    S = W - 1 + M

    def _conv_cache(section):
        cache = torch.zeros(B, S, d, device="cuda", dtype=torch.bfloat16).transpose(-1, -2)
        cache[:, :, : W - 1] = conv_pool[:, section * d : (section + 1) * d]
        return cache

    return SimpleNamespace(
        kda_conv_q=_conv_cache(0),
        kda_conv_k=_conv_cache(1),
        kda_conv_v=_conv_cache(2),
        kda_k_cache=torch.zeros(B, M, d, device="cuda", dtype=torch.bfloat16),
        kda_g_cache=torch.zeros(B, M, d, device="cuda", dtype=torch.float32),
        kda_v_cache=torch.zeros(B, M, d, device="cuda", dtype=torch.bfloat16),
        kda_beta_cache=torch.zeros(B, M, H, device="cuda", dtype=torch.bfloat16),
        prev_num_accepted_tokens=torch.zeros(B, dtype=torch.int32, device="cuda"),
        intermediate_conv_window=None,
        intermediate_ssm=None,
    )


def _make_seq_layer_cache(B):
    d = H * K
    return SimpleNamespace(
        kda_k_cache=None,
        intermediate_conv_window=torch.zeros(
            B, M + 1, 3 * d, W - 1, device="cuda", dtype=torch.bfloat16
        ),
        intermediate_ssm=torch.zeros(B, M + 1, H, K, K, device="cuda", dtype=torch.float32),
    )


def _promote_sequential(layer_cache, conv_pool, ssm_pool, accept):
    """The manager's legacy promotion: accepted step's states -> pools."""
    B = conv_pool.shape[0]
    rows = torch.arange(B, device="cuda")
    conv_pool.copy_(layer_cache.intermediate_conv_window[rows, accept])
    ssm_pool.copy_(layer_cache.intermediate_ssm[rows, accept])


def _rep(name, a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
    print(f"  {name}: cos={cos:.6f} rel_l2={rel:.3e}")
    return cos > 0.999 and rel < 3e-2


def test_plain_decode_direct_pool_matches_fallback():
    """The production decode path updates packed live pools by slot."""
    torch.manual_seed(4)
    batch, slots = 4, 7
    runtime = _make_runtime(seed=5)
    runtime.finalize_decode_weights()
    slot_indices = torch.tensor([6, 1, 4, 2], dtype=torch.long, device="cuda")
    state_indices = slot_indices.to(torch.int32)
    conv_direct, ssm_direct = _make_pools(slots, seed=6)
    conv_fallback = conv_direct.clone()
    ssm_fallback = ssm_direct.clone()
    hidden = torch.randn(batch, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1

    with torch.no_grad():
        expected = runtime.forward_decode_fallback(
            hidden,
            conv_fallback,
            ssm_fallback,
            slot_indices,
        )
        actual = runtime.forward_decode(
            hidden,
            conv_direct,
            ssm_direct,
            slot_indices,
            mamba_metadata=SimpleNamespace(
                _arange_buffer=torch.arange(batch + 1, dtype=torch.int32, device="cuda")
            ),
            ssm_state_indices=state_indices,
        )

    assert _rep("plain decode output", actual, expected)
    torch.testing.assert_close(
        conv_direct.index_select(0, slot_indices),
        conv_fallback.index_select(0, slot_indices),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        ssm_direct.index_select(0, slot_indices),
        ssm_fallback.index_select(0, slot_indices),
        rtol=2e-4,
        atol=2e-4,
    )
    untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
    untouched[slot_indices] = False
    assert torch.equal(conv_direct[untouched], conv_fallback[untouched])
    assert torch.equal(ssm_direct[untouched], ssm_fallback[untouched])


@pytest.fixture(params=[False, True], ids=["standalone", "kvcm_offload"])
def replay_manager(request):
    if not request.param:
        yield None
        return
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import KDAHybridCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    manager = KDAHybridCacheManagerV2(
        mamba_d_state=K,
        mamba_d_conv=W,
        mamba_num_heads=H,
        mamba_n_groups=H,
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
    try:
        yield manager
    finally:
        manager.shutdown()


@pytest.mark.parametrize("fab_fused", [False, True])
@pytest.mark.parametrize("checkpoint_fp8", [False, True])
@torch.no_grad()
def test_fused_vs_sequential_two_rounds(checkpoint_fp8, fab_fused, replay_manager, monkeypatch):
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    # fab_fused: beta and the forget gate come from trtllm::k3_kda_fab_proj (TRTLLM_K3_KDA_FAB_FUSED).
    monkeypatch.setattr(kimi_kda_mixer, "_KDA_FAB_FUSED", fab_fused)

    torch.manual_seed(0)
    B = 4
    T = M + 1
    rt_seq = _make_runtime(seed=1, checkpoint_fp8=checkpoint_fp8)
    rt_fused = _make_runtime(seed=1, aux_stream=torch.cuda.Stream(), checkpoint_fp8=checkpoint_fp8)
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
    mtp_calls = []
    mtp_verify = rt_fused._dispatch.mtp_verify

    def _capturing_mtp_verify(**kwargs):
        mtp_calls.append(kwargs)
        return mtp_verify(**kwargs)

    rt_fused._dispatch.mtp_verify = _capturing_mtp_verify
    slot_indices = torch.arange(B, dtype=torch.int32, device="cuda")

    conv_pool_seq, ssm_pool_seq = _make_pools(B, seed=2)
    conv_pool_fused = conv_pool_seq.clone()
    ssm_pool_fused = ssm_pool_seq.clone()
    cache_seq = _make_seq_layer_cache(B)
    cache_fused = _make_fused_layer_cache(B, conv_pool_fused)
    fused_slots = slot_indices
    if replay_manager is not None:
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
        replay_manager.seed_kda_replay_caches_for_disagg_gen(request_ids)

    gen = torch.Generator(device="cuda").manual_seed(3)

    def tokens(scale=0.5):
        return (
            torch.randn(B * T, HIDDEN, generator=gen, device="cuda", dtype=torch.float32) * scale
        ).to(torch.bfloat16)

    ok = True
    # ---- Round 1 (no pending drafts) ----
    x1 = tokens()
    projected = rt_fused._project_verify_inputs(
        x1.view(B, T, -1), B * T, preserve_section_strides=True
    )
    q_view, k_view, v_view, _, beta_view, onorm_view = projected
    for view in (q_view, k_view, v_view, onorm_view):
        assert view.stride(-2) == 4 * rt_fused.proj_size
        assert view.stride(-1) == 1
    assert (
        kimi_kda_mixer._use_fused_fab(
            x1.view(B, T, -1),
            rt_fused._bfa_proj_weight,
            rt_fused.f_b_proj.weight,
            rt_fused.num_heads,
        )
        == fab_fused
    )
    # beta is a view of the [f_a | b] projection, or its own contiguous tensor when fused.
    beta_rows = rt_fused.num_heads if fab_fused else rt_fused._bfa_proj_weight.shape[0]
    assert beta_view.stride(-2) == beta_rows
    assert beta_view.stride(-1) == 1
    out1_seq = rt_seq.forward_verify_sequential(
        x1, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices
    )
    with with_multi_stream(True):
        out1_fused = rt_fused.forward_verify(
            x1, T, cache_fused, conv_pool_fused, ssm_pool_fused, fused_slots
        )
    assert mtp_calls[-1]["fuse_output_norm"]
    assert not mtp_calls[-1]["mxfp8_output"]
    assert mtp_calls[-1]["onorm_g"].stride(-2) == K
    assert mtp_calls[-1]["onorm_g"].stride(1) == 4 * rt_fused.proj_size
    print("round 1:")
    ok &= _rep("out", out1_fused, out1_seq)

    # ---- Acceptance: 0, 1, 2, 0 drafts across the 4 requests ----
    accept = torch.tensor([0, 1, 2, 0], dtype=torch.long, device="cuda")
    _promote_sequential(cache_seq, conv_pool_seq, ssm_pool_seq, accept)
    cache_fused.prev_num_accepted_tokens.index_copy_(0, fused_slots.long(), accept.to(torch.int32))
    if replay_manager is not None:
        from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId

        caches = [replay_manager.kv_cache_map[rid] for rid in request_ids]
        for cache in caches:
            cache.suspend()
        group = replay_manager.impl.get_layer_group_id(LayerId(0))
        pool = next(
            pool
            for pool in replay_manager.impl.pool_group_descs
            if any(variant.layer_group_id == group for variant in pool.slot_desc.variants)
        )
        pressure = []
        try:
            for _ in range(pool.num_slots + B):
                other = replay_manager.impl.create_kv_cache()
                pressure.append(other)
                assert other.resume(torch.cuda.current_stream().cuda_stream)
                other.suspend()
        finally:
            for other in pressure:
                other.close()
        new_slots = []
        for cache in reversed(caches):
            assert cache.resume(torch.cuda.current_stream().cuda_stream)
            new_slots.append(cache.get_ssm_block_base_index(group))
        fused_slots = torch.tensor(list(reversed(new_slots)), dtype=torch.int32, device="cuda")

    # ---- Round 2 (fused path replays the accepted drafts) ----
    x2 = tokens()
    out2_seq = rt_seq.forward_verify_sequential(
        x2, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices
    )
    with with_multi_stream(True):
        out2_fused = rt_fused.forward_verify(
            x2, T, cache_fused, conv_pool_fused, ssm_pool_fused, fused_slots
        )
    assert len(mtp_calls) == 2
    assert all(call["fuse_output_norm"] for call in mtp_calls)
    assert all(not call["mxfp8_output"] for call in mtp_calls)
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


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("output_norm", "finalized output-norm weights"),
        ("mxfp8_projection", "requires o_proj.forward_prequantized"),
    ],
)
@torch.no_grad()
def test_fused_verify_fails_fast_when_epilogue_is_incomplete(missing, message):
    runtime = _make_runtime(seed=7)
    runtime.finalize_decode_weights()
    if missing == "output_norm":
        runtime._onorm_w_f32 = None
    else:
        runtime._fp8_resident_projections = True
        runtime.o_proj.supports_prequantized_input = True

    batch = 1
    hidden = torch.zeros(batch * (M + 1), HIDDEN, dtype=torch.bfloat16, device="cuda")
    conv_pool, ssm_pool = _make_pools(batch, seed=8)
    layer_cache = _make_fused_layer_cache(batch, conv_pool)
    slot_indices = torch.arange(batch, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match=message):
        runtime.forward_verify_fused(hidden, M + 1, layer_cache, ssm_pool, slot_indices)


@pytest.mark.parametrize("two_ctas", ["0", "1"])
@torch.no_grad()
def test_fused_verify_compact_layout_is_bitwise_identical(two_ctas, monkeypatch):
    """Compact layout through the runtime: TRTLLM_KDA_MTP_COMPACT=1 vs the in-tree kernel.

    Two chained ``forward_verify`` rounds (mixed acceptance before round 2) on a
    bf16 state pool (production ``mamba_ssm_cache_dtype``) with more (head, request)
    tiles than SMs, so the vendored compact layout serves every call of the compact
    world. Outputs, the committed state, the conv windows and the replay caches must
    match the in-tree kernel bit for bit. (The MXFP8-output variant is covered at the
    kernel level by test_kda_mtp_decode_compact_vendored.py.)
    """
    from tensorrt_llm._torch.custom_ops import (
        cute_dsl_kimi_k3_kda_mtp_compact_policy as compact_policy,
    )
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as kda_ops
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    B = max(num_sms, 64) // H + 1  # one CTA per tile and more tiles than SMs
    T = M + 1
    slots = torch.randperm(B, generator=torch.Generator().manual_seed(20)).to(
        device="cuda", dtype=torch.int32
    )
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "0")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", "1")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", two_ctas)
    monkeypatch.setattr(kda_ops, "_compiled_cache", {})
    # record the layout decision of every kda_mtp_decode call
    decisions = []
    select = kda_ops.select_compact_layout

    def recording_select(**kwargs):
        decisions.append(select(**kwargs))
        return decisions[-1]

    monkeypatch.setattr(kda_ops, "select_compact_layout", recording_select)

    def run_world(compact):
        monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", "1" if compact else "0")
        monkeypatch.setattr(compact_policy, "_kda_compact_policy_cache", {})
        decisions.clear()
        rt = _make_runtime(seed=21, aux_stream=torch.cuda.Stream())
        rt.finalize_decode_weights()
        conv_pool, ssm_pool = _make_pools(B, seed=22)
        ssm_pool = ssm_pool.to(torch.bfloat16)
        layer_cache = _make_fused_layer_cache(B, conv_pool)
        gen = torch.Generator(device="cuda").manual_seed(23)
        outs = []
        for rnd in range(2):
            if rnd:
                accept = torch.arange(B, device="cuda", dtype=torch.int32) % (M + 1)
                layer_cache.prev_num_accepted_tokens.index_copy_(0, slots.long(), accept)
            x = torch.randn(B * T, HIDDEN, generator=gen, device="cuda", dtype=torch.float32)
            with with_multi_stream(True):
                out = rt.forward_verify(
                    (x * 0.5).to(torch.bfloat16), T, layer_cache, conv_pool, ssm_pool, slots
                )
            outs.append(out.clone())
        torch.cuda.synchronize()
        tensors = {"conv_pool": conv_pool, "ssm_pool": ssm_pool}
        for name in (
            "kda_conv_q",
            "kda_conv_k",
            "kda_conv_v",
            "kda_k_cache",
            "kda_g_cache",
            "kda_v_cache",
            "kda_beta_cache",
        ):
            tensors[name] = getattr(layer_cache, name)
        return outs, tensors, list(decisions)

    ref_outs, ref_tensors, ref_layouts = run_world(compact=False)
    new_outs, new_tensors, new_layouts = run_world(compact=True)
    assert ref_layouts == [(False, False)] * 2, ref_layouts
    assert new_layouts == [(True, two_ctas == "1")] * 2, new_layouts
    for rnd, (new, ref) in enumerate(zip(new_outs, ref_outs)):
        assert torch.equal(new.view(torch.uint8), ref.view(torch.uint8)), f"round {rnd + 1} output"
    for name, ref in ref_tensors.items():
        new = new_tensors[name]
        assert torch.equal(
            new.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
        ), name


@torch.no_grad()
def test_fused_verify_prequantized_qkvg_is_bitwise_identical():
    """TRTLLM_K3_ATTN_RES_FP8_OUT through the runtime: feeding the fused q|k|v|g projection the
    FP8 1x128 rows (``hidden_fp8``) equals letting it quantize the BF16 rows itself.

    The fused projection is swapped to the FP8 weight-read GEMM as KimiLinearForCausalLM does
    after loading (Rubin: CuTe DSL MXFP8 with a prequantized entry point). Two chained rounds
    with mixed acceptance; outputs and committed state must match bit for bit.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8Activation,
        _swap_linear_to_fp8_weight_read,
    )
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    B, T = 4, M + 1

    def run_world(prequantized: bool):
        rt = _make_runtime(seed=31, aux_stream=torch.cuda.Stream(), checkpoint_fp8=True)
        rt.finalize_decode_weights()
        # Only the fused projection is swapped (the consumer under test); o_proj stays the
        # checkpoint Linear, so the verify kernel keeps its BF16 output as in the other tests.
        _swap_linear_to_fp8_weight_read(rt, "qkvg_proj", (Linear,))
        rt.qkvg_proj.transform_weights()
        for module in rt.modules():
            if isinstance(module, Linear):
                module.post_load_weights()
        if not rt.accepts_prequantized_input():
            pytest.skip("needs the Rubin CuTe DSL MXFP8 GEMM (SM107)")
        conv_pool, ssm_pool = _make_pools(B, seed=32)
        layer_cache = _make_fused_layer_cache(B, conv_pool)
        slots = torch.arange(B, dtype=torch.int32, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(33)
        outs = []
        for rnd in range(2):
            if rnd:
                accept = torch.tensor([0, 1, 2, 0], dtype=torch.int32, device="cuda")
                layer_cache.prev_num_accepted_tokens.copy_(accept)
            x = (torch.randn(B * T, HIDDEN, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            x_fp8 = None
            if prequantized:
                x_fp8 = _Fp8Activation(*torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x))
            with with_multi_stream(True):
                out = rt.forward_verify(
                    x, T, layer_cache, conv_pool, ssm_pool, slots, x2d_fp8=x_fp8
                )
            outs.append(out.clone())
        torch.cuda.synchronize()
        return outs, ssm_pool

    ref_outs, ref_ssm = run_world(False)
    new_outs, new_ssm = run_world(True)
    for rnd, (new, ref) in enumerate(zip(new_outs, ref_outs)):
        assert torch.equal(new.view(torch.uint8), ref.view(torch.uint8)), f"round {rnd + 1}"
    assert torch.equal(new_ssm.view(torch.uint8), ref_ssm.view(torch.uint8))
