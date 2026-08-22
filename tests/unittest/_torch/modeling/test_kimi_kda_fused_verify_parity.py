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

"""Runtime-level parity: KimiKDARuntime fused verify vs sequential verify.

Simulates two chained speculative-verification rounds through
``KimiKDARuntime._forward_verify`` in both worlds:

* Sequential world: the legacy intermediate-buffer path
  (``_forward_verify_sequential``) plus the manager's legacy promotion
  (copy the accepted step's conv window / SSM state into the live pools).
* Fused world: the ``trtllm::kda_mtp_decode`` replay path
  (``_forward_verify_fused``) with per-slot replay caches, in-place state
  commit after the golden token, and only the accepted-draft count
  recorded between rounds.

Identical hidden states are fed to both worlds; the fused world additionally
uses fused QKVG and [f_a|b] projections with multi-stream overlap. With mixed
per-request acceptance between rounds, matching round-2 outputs proves the
projection fusion and replay bookkeeping (shifted ``cu_seqlens`` layout,
conv-window seeding, pending-count plumbing) reproduce the promoted-state
semantics.

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
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiKDARuntime
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor in (100, 103)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_blackwell(), reason="needs sm100/sm103"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

HIDDEN = 512
H = 8  # per-rank head count outside the drop's tuned set — general variant
K = 128
W = 4
M = 2  # draft tokens per round
LB = -5.0


@torch.no_grad()
def _make_runtime(seed, aux_stream=None):
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
    rt = KimiKDARuntime(cfg, layer_idx=0, aux_stream=aux_stream).to("cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for name, p in rt.named_parameters():
        if name.endswith("A_log"):
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
        torch.randn(B, 3 * d, W, generator=gen, device="cuda", dtype=torch.float32) * 0.5
    ).to(torch.bfloat16)
    ssm_pool = torch.randn(B, H, K, K, generator=gen, device="cuda", dtype=torch.float32)
    ssm_pool *= torch.linspace(0.5, 1.5, K, device="cuda").view(1, 1, K, 1)
    return conv_pool, ssm_pool


def _make_fused_layer_cache(B, conv_pool):
    """Replay caches shaped like PythonMambaCacheManager's KDA allocation,
    with the committed conv window seeded from the base pool (the prefill
    seeding contract: FLA window columns [1, W) -> committed columns)."""
    d = H * K
    S = W - 1 + M

    def _conv_cache(section):
        cache = torch.zeros(B, S, d, device="cuda", dtype=torch.float32).transpose(-1, -2)
        cache[:, :, : W - 1] = conv_pool[:, section * d : (section + 1) * d, 1:].float()
        return cache

    return SimpleNamespace(
        kda_conv_q=_conv_cache(0),
        kda_conv_k=_conv_cache(1),
        kda_conv_v=_conv_cache(2),
        kda_qkg_cache=torch.zeros(B, M, 3, d, device="cuda", dtype=torch.float32),
        kda_v_cache=torch.zeros(B, M, d, device="cuda", dtype=torch.float32),
        kda_beta_cache=torch.zeros(B, M, H, device="cuda", dtype=torch.float32),
        prev_num_accepted_tokens=torch.zeros(B, dtype=torch.int32, device="cuda"),
        has_kda_replay_caches=True,
        intermediate_conv_window=None,
        intermediate_ssm=None,
    )


def _make_seq_layer_cache(B):
    d = H * K
    return SimpleNamespace(
        kda_qkg_cache=None,
        has_kda_replay_caches=False,
        intermediate_conv_window=torch.zeros(
            B, M + 1, 3 * d, W, device="cuda", dtype=torch.bfloat16
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


@torch.no_grad()
def test_fused_vs_sequential_two_rounds():
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    torch.manual_seed(0)
    B = 4
    T = M + 1
    rt_seq = _make_runtime(seed=1)
    rt_fused = _make_runtime(seed=1, aux_stream=torch.cuda.Stream())
    rt_fused.finalize_decode_weights()
    assert rt_fused._qkvg_proj_weight is not None
    assert rt_fused._bfa_proj_weight is not None
    slot_indices = torch.arange(B, dtype=torch.long, device="cuda")

    conv_pool_seq, ssm_pool_seq = _make_pools(B, seed=2)
    conv_pool_fused = conv_pool_seq.clone()
    ssm_pool_fused = ssm_pool_seq.clone()
    cache_seq = _make_seq_layer_cache(B)
    cache_fused = _make_fused_layer_cache(B, conv_pool_fused)

    gen = torch.Generator(device="cuda").manual_seed(3)

    def tokens(scale=0.5):
        return (
            torch.randn(B * T, HIDDEN, generator=gen, device="cuda", dtype=torch.float32) * scale
        ).to(torch.bfloat16)

    ok = True
    # ---- Round 1 (no pending drafts) ----
    x1 = tokens()
    out1_seq = rt_seq._forward_verify_sequential(
        x1, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices
    )
    with with_multi_stream(True):
        out1_fused = rt_fused._forward_verify(
            x1, T, cache_fused, conv_pool_fused, ssm_pool_fused, slot_indices
        )
    print("round 1:")
    ok &= _rep("out", out1_fused, out1_seq)

    # ---- Acceptance: 0, 1, 2, 0 drafts across the 4 requests ----
    accept = torch.tensor([0, 1, 2, 0], dtype=torch.long, device="cuda")
    _promote_sequential(cache_seq, conv_pool_seq, ssm_pool_seq, accept)
    cache_fused.prev_num_accepted_tokens.copy_(accept.to(torch.int32))

    # ---- Round 2 (fused path replays the accepted drafts) ----
    x2 = tokens()
    out2_seq = rt_seq._forward_verify_sequential(
        x2, T, cache_seq, conv_pool_seq, ssm_pool_seq, slot_indices
    )
    with with_multi_stream(True):
        out2_fused = rt_fused._forward_verify(
            x2, T, cache_fused, conv_pool_fused, ssm_pool_fused, slot_indices
        )
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
