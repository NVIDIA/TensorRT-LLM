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

"""``trtllm::kda_mtp_decode`` compact (two CTAs per SM) layout vs the default layout.

The compact layout only re-places shared-memory / TMEM buffers (time-shared
region, Kbar written straight into the B2 tiles, S0 hi aliased with the GEMM2
accumulator); every value-producing instruction is unchanged. So the outputs,
the committed recurrent state, the conv windows and the replay caches must be
BITWISE identical to the default layout, over two chained verify rounds with
mixed per-slot accepted counts (round 2 replays round-1 drafts).

Production variant of the K3 decode step: packed token layout, bf16 state
pool (V-first, strided per-layer slice), bf16 conv caches, fused gated RMSNorm,
num_spec = 0..3, with bf16 output (SM100/SM103) or MXFP8 output (SM107). The tile
count (H * N) must exceed the SM count for the compact layout to be selected;
the test checks that both layouts were actually compiled.

Requires 1 SM100/SM103/SM107 GPU and nvidia-cutlass-dsl.
"""

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401

    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as kda_ops
except ImportError as e:
    _HAVE_DEPS = False
    _DEP_ERR = str(e)


def _is_sm100_family():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in ((10, 0), (10, 3), (10, 7))


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not _is_sm100_family(), reason="needs sm100/sm103/sm107"),
    pytest.mark.skipif(not _HAVE_DEPS, reason=f"deps: {_DEP_ERR}"),
]

K = V = 128
W = 4

MUTATED = ("cs_q", "cs_k", "cs_v", "recurrent_state", "kg_cache", "v_cache", "beta_cache")


def _make_inputs(N, H, num_spec, pool, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    d = H * K
    T = N * (1 + num_spec)
    layers = 2
    # fused [q | k | v | output-gate] projection: strided section views
    qkvg = torch.randn(T, 4 * d, dtype=torch.bfloat16, device=dev)
    bfa = torch.randn(T, 128 + H, dtype=torch.bfloat16, device=dev)
    S = W - 1 + num_spec

    def conv():
        return torch.randn(layers, pool, S, d, dtype=torch.bfloat16, device=dev).transpose(-1, -2)[
            1
        ]

    kg = torch.randn(layers, pool, num_spec, 2, d, dtype=torch.float32, device=dev)[1]
    kg[:, :, 1] = -torch.rand_like(kg[:, :, 1]) * 2.0
    beta_cache = torch.rand(layers, pool, num_spec, H + 4, dtype=torch.float32, device=dev)[
        1, ..., :H
    ]
    return dict(
        x_q=qkvg[:, 0 * d : 1 * d].view(1, T, H, K),
        x_k=qkvg[:, 1 * d : 2 * d].view(1, T, H, K),
        x_v=qkvg[:, 2 * d : 3 * d].view(1, T, H, V),
        onorm_g=qkvg[:, 3 * d : 4 * d].view(1, T, H, V),
        g=torch.randn(T, d, dtype=torch.bfloat16, device=dev).view(1, T, H, K),
        beta=bfa[:, 128 : 128 + H].view(1, T, H),
        w_q=torch.randn(d, W, dtype=torch.float32, device=dev) * 0.5,
        w_k=torch.randn(d, W, dtype=torch.float32, device=dev) * 0.5,
        w_v=torch.randn(d, W, dtype=torch.float32, device=dev) * 0.5,
        A_log=torch.randn(H, dtype=torch.float32, device=dev) * 0.5,
        dt_bias=torch.randn(d, dtype=torch.float32, device=dev),
        cs_q=conv(),
        cs_k=conv(),
        cs_v=conv(),
        recurrent_state=(
            torch.randn(layers, pool, H, V, K, dtype=torch.float32, device=dev) * 0.1
        ).to(torch.bfloat16)[1],
        kg_cache=kg,
        v_cache=torch.randn(layers, pool, num_spec, d, dtype=torch.float32, device=dev)[1],
        beta_cache=beta_cache,
        ssm_state_indices=torch.randperm(pool, device=dev)[:N].to(torch.int32),
        cu_seqlens=None,
        num_spec=num_spec,
        num_accepted_tokens=torch.randint(0, num_spec + 1, (pool,), dtype=torch.int32, device=dev),
        lower_bound=-5.0,
        scale=K**-0.5,
        packed_token_layout=True,
        onorm_weight=torch.rand(V, dtype=torch.float32, device=dev) + 0.5,
        onorm_eps=1e-5,
        fuse_output_norm=True,
    )


def _clone(inp):
    out = dict(inp)
    for name in MUTATED:
        t = inp[name]
        base = t._base if t._base is not None else t
        out[name] = torch.as_strided(base.clone(), t.shape, t.stride(), t.storage_offset())
    return out


def _clone_out(o):
    if isinstance(o, tuple):
        # MXFP8 scales pad rows to 128; padding from torch.empty is not written.
        # Unpack the scale layout and compare every logical row/head/replica.
        act, scales = o
        rows, hidden = act.shape
        heads = hidden // V
        scales = scales.view(-1, heads, 32, 4, 4).permute(0, 3, 2, 1, 4)
        return act.clone(), scales.reshape(-1, heads, 4)[:rows].clone()
    return o.clone()


def _assert_out_equal(a, b):
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
    else:
        assert torch.equal(a, b)


def _spy_compiled_layouts(monkeypatch):
    """Record the COMPACT flag of every kernel variant compiled from here on."""
    layouts = []
    real_compile = kda_ops.cute.compile

    def spy(*args, **kwargs):
        layouts.append((kwargs.get("COMPACT"), kwargs.get("COMPACT_TWO_CTAS")))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(kda_ops, "_compiled_cache", {})
    monkeypatch.setattr(kda_ops.cute, "compile", spy)
    return layouts


def _run_two_rounds(inp, compact, monkeypatch, quantize_output=False):
    # Dispatch uses the startup snapshot, not later environment mutations.
    monkeypatch.setenv("TRTLLM_KDA_MTP_COMPACT", "0" if compact else "1")
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT", compact)
    # Reset only startup policy; keep compiled variants for cache separation tests.
    monkeypatch.setattr(kda_ops, "_kda_compact_policy_cache", {})
    x = _clone(inp)
    x["quantize_output"] = quantize_output
    o1 = _clone_out(kda_ops.kda_mtp_decode_impl(**x))
    # round 2: the sampler accepted a different number of drafts per slot
    x["num_accepted_tokens"] = torch.roll(x["num_accepted_tokens"], 1)
    o2 = _clone_out(kda_ops.kda_mtp_decode_impl(**x))
    torch.cuda.synchronize()
    return o1, o2, x


@pytest.mark.parametrize("num_spec", [0, 1, 2, 3])
@pytest.mark.parametrize("two_ctas", [False, True])
@pytest.mark.parametrize("quantize_output", [False, True])
@pytest.mark.parametrize("N,H", [(40, 96), (24, 16), (64, 12), (72, 96)])
def test_compact_layout_is_bitwise_identical(
    N, H, num_spec, quantize_output, two_ctas, monkeypatch
):
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_QUANTIZED", "1")
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_TWO_CTAS", "1" if two_ctas else "0")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if H * N <= num_sms:
        pytest.skip(f"H*N={H * N} <= {num_sms} SMs: single wave, compact path not taken")
    inp = _make_inputs(N, H, num_spec=num_spec, pool=N + 24, seed=1234 + N)
    layouts = _spy_compiled_layouts(monkeypatch)
    ref1, ref2, ref_x = _run_two_rounds(inp, False, monkeypatch, quantize_output)
    new1, new2, new_x = _run_two_rounds(inp, True, monkeypatch, quantize_output)
    # one variant per layout: the comparison is not vacuous on this GPU
    assert layouts == [(False, False), (True, two_ctas)], layouts
    _assert_out_equal(new1, ref1)
    _assert_out_equal(new2, ref2)
    for name in MUTATED:
        assert torch.equal(new_x[name], ref_x[name]), name
    # sanity: the kernel did run and produced finite values
    ref_act = ref2[0] if isinstance(ref2, tuple) else ref2
    assert torch.isfinite(ref_act.float()).all()


def test_compact_policy_switches_without_clearing_cache(monkeypatch):
    """Policy changes select distinct compiled variants and preserve results."""
    inp = _make_inputs(40, 96, num_spec=3, pool=64, seed=1234)
    layouts = _spy_compiled_layouts(monkeypatch)
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_QUANTIZED", "0")
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_TWO_CTAS", "0")
    ref1, ref2, ref_x = _run_two_rounds(inp, True, monkeypatch, quantize_output=True)
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_QUANTIZED", "1")
    for two_ctas in (False, True):
        monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_TWO_CTAS", "1" if two_ctas else "0")
        new1, new2, new_x = _run_two_rounds(inp, True, monkeypatch, quantize_output=True)
        _assert_out_equal(new1, ref1)
        _assert_out_equal(new2, ref2)
        for name in MUTATED:
            assert torch.equal(new_x[name], ref_x[name]), name
    assert layouts == [(False, False), (True, False), (True, True)]
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_QUANTIZED", "0")
    _run_two_rounds(inp, True, monkeypatch, quantize_output=True)
    assert len(layouts) == 3, "disabled policy must reuse the original compiled variant"


def test_compact_policy_architecture_defaults(monkeypatch):
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_QUANTIZED", None)
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_TWO_CTAS", None)
    inp = _make_inputs(40, 96, num_spec=3, pool=64, seed=1234)
    layouts = _spy_compiled_layouts(monkeypatch)
    _run_two_rounds(inp, True, monkeypatch, quantize_output=True)
    is_rubin = torch.cuda.get_device_capability(0) == (10, 7)
    assert layouts == [(is_rubin, is_rubin)]


@pytest.mark.parametrize("override", [None, "0", "1", "", "false"])
def test_compact_policy_is_cached_per_device(override, monkeypatch):
    monkeypatch.setattr(kda_ops, "_kda_compact_policy_cache", {})
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT", True)
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_QUANTIZED", override)
    monkeypatch.setattr(kda_ops, "_KDA_MTP_COMPACT_TWO_CTAS", override)
    calls = []

    def capability(idx):
        calls.append(idx)
        return [(10, 3), (10, 7)][idx]

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    for idx, sm in enumerate(((10, 3), (10, 7))):
        expected = sm == (10, 7) if override is None else override != "0"
        assert kda_ops._kda_compact_policy(torch.device("cuda", idx)) == (
            sm,
            True,
            expected,
            expected,
        )
    for name in ("COMPACT", "COMPACT_QUANTIZED", "COMPACT_TWO_CTAS"):
        monkeypatch.setenv("TRTLLM_KDA_MTP_" + name, "0")
    assert kda_ops._kda_compact_policy(torch.device("cuda")) is kda_ops._kda_compact_policy(
        torch.device("cuda", 1)
    )
    kda_ops._kda_compact_policy(torch.device("cuda", 0))
    assert calls == [0, 1]
