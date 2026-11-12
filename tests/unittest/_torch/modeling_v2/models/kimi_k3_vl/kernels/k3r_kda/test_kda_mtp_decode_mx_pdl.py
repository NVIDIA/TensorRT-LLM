# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MXFP8-output KDA verify as a PDL launch with the early trigger (``mxfp8_pdl_early_trigger``).

The o_proj push with the mid-M GEMM runs the MXFP8-output verify on the two-CTAs-per-SM kernel as a
PDL launch that releases its dependent right after ``griddepcontrol.wait``. Scheduling only: the
E4M3 values, the UE8M0 scales and every mutated buffer (recurrent state, conv windows, replay
caches) must equal those of the plain launch over two chained verify rounds with mixed per-slot
accepted counts, eagerly and under CUDA graph replay.

Inputs follow the runtime contract: packed token layout (strided QKVG section views), bf16 state
pool (V-first, strided per-layer slice), bf16 conv / k / v / beta replay caches, fp32 gate cache,
fused gated RMSNorm with the MXFP8 operand of the Rubin o_proj. The two-CTA MXFP8 kernel runs by
default on SM107 only; the tests select it on SM100 / SM103 as well.

Requires 1 SM100/SM103/SM107 GPU and nvidia-cutlass-dsl.
"""

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_kda import (  # noqa: E501
        mtp_ops as kda_ops,
    )
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
POOL = 80
MUTATED = ("cs_q", "cs_k", "cs_v", "recurrent_state", "k_cache", "g_cache", "v_cache", "beta_cache")


def _make_inputs(N, H, num_spec, seed, zero_state):
    gen = torch.Generator().manual_seed(seed)
    dev = "cuda"
    d = H * K
    T = N * (1 + num_spec)
    S = W - 1 + num_spec

    def rn(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype).to(dev)

    def conv():
        # [layers, slots, window, dim] -> dim-contiguous view of layer 1
        return rn(2, POOL, S, d, dtype=torch.bfloat16).transpose(-1, -2)[1]

    # fused [q | k | v | output-gate] projection: strided section views
    qkvg = rn(T, 4 * d, dtype=torch.bfloat16, scale=0.5)
    # [f_a | b | pad] projection rows (beta is a strided view)
    bfa = rn(T, 128 + (H + 7) // 8 * 8, dtype=torch.bfloat16)
    state = rn(2, POOL, H, V, K, dtype=torch.bfloat16, scale=0.05)
    if zero_state:
        state.zero_()
    slots = torch.randperm(POOL, generator=gen)[:N].to(torch.int32)
    accepted = torch.zeros(POOL, dtype=torch.int32)
    if not zero_state:
        accepted[slots.long()] = torch.randint(
            0, num_spec + 1, (N,), generator=gen, dtype=torch.int32
        )
    heads_padded = (H + 7) // 8 * 8
    return dict(
        x_q=qkvg[:, 0 * d : 1 * d].view(1, T, H, K),
        x_k=qkvg[:, 1 * d : 2 * d].view(1, T, H, K),
        x_v=qkvg[:, 2 * d : 3 * d].view(1, T, H, V),
        onorm_g=qkvg[:, 3 * d : 4 * d].view(1, T, H, V),
        g=rn(T, d, dtype=torch.bfloat16, scale=0.5).view(1, T, H, K),
        beta=bfa[:, 128 : 128 + H].view(1, T, H),
        w_q=rn(d, W, scale=0.5),
        w_k=rn(d, W, scale=0.5),
        w_v=rn(d, W, scale=0.5),
        A_log=rn(H, scale=0.5),
        dt_bias=rn(d, scale=0.5),
        cs_q=conv(),
        cs_k=conv(),
        cs_v=conv(),
        recurrent_state=state[1],
        k_cache=rn(2, POOL, num_spec, d, dtype=torch.bfloat16, scale=0.08)[1],
        g_cache=(-torch.rand(2, POOL, num_spec, d, generator=gen) * 0.5).to(dev)[1],
        v_cache=rn(2, POOL, num_spec, d, dtype=torch.bfloat16)[1],
        beta_cache=torch.rand(2, POOL, num_spec, heads_padded, generator=gen)
        .to(torch.bfloat16)
        .to(dev)[1, ..., :H],
        ssm_state_indices=slots.to(dev),
        cu_seqlens=None,
        num_spec=num_spec,
        num_accepted_tokens=accepted.to(dev),
        lower_bound=-5.0,
        scale=K**-0.5,
        packed_token_layout=True,
        onorm_weight=(0.75 + 0.5 * torch.rand(V, generator=gen)).to(dev),
        onorm_eps=1e-5,
        fuse_output_norm=True,
    )


def _clone(inp):
    out = dict(inp)
    for name in MUTATED:
        t = inp[name]
        base = t._base if t._base is not None else t
        out[name] = torch.as_strided(base.clone(), t.shape, t.stride(), t.storage_offset())
    out["num_accepted_tokens"] = inp["num_accepted_tokens"].clone()
    return out


def _call_kwargs(x, use_pdl):
    return {**x, "use_pdl": use_pdl}


def _next_round(x, rnd):
    """The sampler accepted a different number of drafts per slot."""
    acc = x["num_accepted_tokens"]
    slots = x["ssm_state_indices"].long()
    acc[slots] = (
        acc[slots] + 1 + rnd + torch.arange(len(slots), device=acc.device, dtype=acc.dtype) % 3
    ) % (x["num_spec"] + 1)


def _scale_bytes(scales, rows, heads):
    """The scale bytes of rows [0, rows) (four copies per (row, head)); the padding rows of the
    R128c4 buffer are never written."""
    r = torch.arange(rows, device=scales.device)[:, None]
    h = torch.arange(heads, device=scales.device)[None, :]
    off = (r // 128) * heads * 512 + h * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
    off = (off[..., None] + torch.arange(4, device=scales.device)).reshape(-1)
    return scales.view(torch.uint8)[off]


@pytest.mark.parametrize("H,N,num_spec", [(12, 16, 5), (12, 12, 7), (12, 4, 7)])
def test_mxfp8_pdl_early_trigger_bitwise(H, N, num_spec, monkeypatch):
    """The MXFP8-output verify as a PDL launch with the early trigger
    (two-CTA kernel) gives the plain launch's E4M3 values, scales and mutated buffers."""
    monkeypatch.setattr(kda_ops, "_mxfp8_two_cta_enabled", lambda device: True)
    inp = _make_inputs(N, H, num_spec, seed=3 * N + H, zero_state=False)
    rows = N * (1 + num_spec)

    def run(mx_pdl):
        x = _clone(inp)
        outs = []
        keys_before = set(kda_ops._compiled_cache)
        for rnd in range(2):
            with kda_ops.mxfp8_pdl_early_trigger(mx_pdl):
                act, sf = kda_ops.kda_mtp_decode_impl(
                    **_call_kwargs(x, mx_pdl), quantize_output=True
                )
            outs.append((act.view(torch.uint8).clone(), _scale_bytes(sf, rows, H).clone()))
            _next_round(x, rnd)
        torch.cuda.synchronize()
        new_keys = [k for k in kda_ops._compiled_cache if k not in keys_before]
        return outs, x, new_keys

    ref_outs, ref_x, _ = run(False)
    new_outs, new_x, keys = run(True)
    trig = [k for k in kda_ops._compiled_cache if k[0] == "2cta" and k[-1] is True]
    assert trig, "the early-trigger two-CTA variant did not run"
    for rnd, ((a, sa), (b, sb)) in enumerate(zip(new_outs, ref_outs)):
        assert torch.equal(a, b), f"round {rnd} E4M3 values"
        assert torch.equal(sa, sb), f"round {rnd} UE8M0 scales"
    for name in MUTATED:
        assert torch.equal(new_x[name], ref_x[name]), name


def test_mxfp8_pdl_early_trigger_cuda_graph(monkeypatch):
    """The early-trigger launch in a CUDA graph, replayed, matches eager plain launches."""
    monkeypatch.setattr(kda_ops, "_mxfp8_two_cta_enabled", lambda device: True)
    H, N, num_spec = 12, 16, 5
    inp = _make_inputs(N, H, num_spec, seed=99, zero_state=False)
    rows = N * (1 + num_spec)
    x_eager = _clone(inp)
    act, sf = kda_ops.kda_mtp_decode_impl(**_call_kwargs(x_eager, False), quantize_output=True)
    ref = (act.view(torch.uint8).clone(), _scale_bytes(sf, rows, H).clone())
    x_graph = _clone(inp)
    snap = _clone(x_graph)
    with kda_ops.mxfp8_pdl_early_trigger(True):
        kda_ops.kda_mtp_decode_impl(**_call_kwargs(_clone(inp), True), quantize_output=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_act, g_sf = kda_ops.kda_mtp_decode_impl(
                **_call_kwargs(x_graph, True), quantize_output=True
            )
    for name in MUTATED:
        x_graph[name].copy_(snap[name])
    x_graph["num_accepted_tokens"].copy_(snap["num_accepted_tokens"])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(g_act.view(torch.uint8), ref[0])
    assert torch.equal(_scale_bytes(g_sf, rows, H), ref[1])
    for name in MUTATED:
        assert torch.equal(x_graph[name], x_eager[name]), name
