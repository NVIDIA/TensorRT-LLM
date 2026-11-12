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
"""``trtllm::kda_mtp_decode`` two-CTAs-per-SM variants vs the one-CTA kernel, bit for bit.

* The compact layout of ``kda_decode_mtp_kernel`` (``mtp_ops._compact_layout_enabled``, on
  SM107) only re-places shared-memory / TMEM buffers; every value-producing instruction is
  unchanged.
* The MXFP8 epilogue of ``kda_decode_mtp_2cta_kernel`` (``mtp_ops._mxfp8_two_cta_enabled``, on
  SM107) applies ``kda_decode_mtp_kernel``'s quantization to the same normalized values.

The tests force each selection on or off, so both kernels of a pair run on SM100 / SM103 too.

So the outputs (E4M3 values and the UE8M0 scales of the valid rows for MXFP8 output), the committed
recurrent state, the conv windows and the replay caches must equal those of the default kernel,
over two chained verify rounds with mixed per-slot accepted counts (round 2 replays round-1
drafts), from a zero or a populated recurrent state. Each test checks which kernel variant served
the calls, so that it cannot pass vacuously.

Inputs follow the runtime contract: packed token layout (strided QKVG section views), bf16 state
pool (V-first, strided per-layer slice), bf16 conv / k / v / beta replay caches, fp32 gate cache,
fused gated RMSNorm; MXFP8 output is what SM107 requests for the prequantized o_proj.

Requires 1 SM100/SM103/SM107 GPU and nvidia-cutlass-dsl.
"""

from collections.abc import MutableMapping

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

    # This GPU's default kernel selections (SM107: on), before any test forces them.
    _DEFAULT_SELECTORS = {
        name: getattr(kda_ops, name)
        for name in ("_compact_layout_enabled", "_mxfp8_two_cta_enabled")
    }
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
COMPACT = "_compact_layout_enabled"
MXFP8_2CTA = "_mxfp8_two_cta_enabled"

# (heads per rank, requests, drafts): helix8 (12 heads), helix16 (6), dep16 (96) at batch 1 / 4 /
# 8 / 16 with seven drafts and batch 16 / 32 with five drafts.
DECODE_SHAPES = [
    pytest.param(h, n, s, id=f"{name}_b{n}_mtp{s}")
    for name, h in (("helix8", 12), ("helix16", 6), ("dep16", 96))
    for n, s in ((1, 7), (4, 7), (8, 7), (16, 7), (16, 5), (32, 5))
]


def _num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


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
    bfa = rn(T, 128 + H, dtype=torch.bfloat16)
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
    return out


def _clone_out(o):
    # quantize_output=True returns (E4M3 activations, R128c4 UE8M0 scales)
    return tuple(t.clone() for t in o) if isinstance(o, tuple) else o.clone()


def _scale_bytes(scales, rows, heads):
    """The scale bytes of rows [0, rows) (four copies per (row, head)); the padding rows of the
    R128c4 buffer are never written."""
    r = torch.arange(rows, device=scales.device)[:, None]
    h = torch.arange(heads, device=scales.device)[None, :]
    off = (r // 128) * heads * 512 + h * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
    off = (off[..., None] + torch.arange(4, device=scales.device)).reshape(-1)
    return scales.view(torch.uint8)[off]


def _assert_out_equal(a, b, heads):
    if isinstance(a, tuple):
        assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8)), "E4M3 values"
        rows = a[0].shape[0]
        assert torch.equal(_scale_bytes(a[1], rows, heads), _scale_bytes(b[1], rows, heads)), (
            "UE8M0 scales"
        )
    else:
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), "bf16 output"


class _RecordingCache(MutableMapping):
    """Pass-through view of the op's compile cache that records the keys looked up."""

    def __init__(self, cache):
        self.cache = cache
        self.used = []

    def __getitem__(self, key):
        self.used.append(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __delitem__(self, key):
        del self.cache[key]

    def __iter__(self):
        return iter(self.cache)

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache


def _variant(key):
    """Kernel variant of a compile-cache key."""
    if key[0] == "2cta":
        return "2cta_mxfp8"
    if key[0] == "v2":
        return "v2"
    return "v1_compact" if key[-1] else "v1"


def _run_rounds(inp, quantize_output, selectors, monkeypatch, attrs=None):
    """Two chained verify rounds with the kernel selections ``selectors`` (name -> forced on /
    off, ``None``: this GPU's default) and the module settings ``attrs`` (name -> value);
    returns the outputs, the mutated inputs and the kernel variants that served the calls."""
    for name, value in selectors.items():
        default = _DEFAULT_SELECTORS[name]
        monkeypatch.setattr(
            kda_ops, name, default if value is None else (lambda device, value=value: value)
        )
    for name, value in (attrs or {}).items():
        monkeypatch.setattr(kda_ops, name, value)
    recorder = _RecordingCache(kda_ops._compiled_cache)
    monkeypatch.setattr(kda_ops, "_compiled_cache", recorder)
    x = _clone(inp)
    x["num_accepted_tokens"] = x["num_accepted_tokens"].clone()
    x["quantize_output"] = quantize_output
    o1 = _clone_out(kda_ops.kda_mtp_decode_impl(**x))
    # round 2: the sampler accepted a different number of drafts per slot
    acc = x["num_accepted_tokens"]
    slots = x["ssm_state_indices"].long()
    acc[slots] = (
        acc[slots] + 1 + torch.arange(len(slots), device=acc.device, dtype=acc.dtype) % 3
    ) % (x["num_spec"] + 1)
    o2 = _clone_out(kda_ops.kda_mtp_decode_impl(**x))
    torch.cuda.synchronize()
    monkeypatch.setattr(kda_ops, "_compiled_cache", recorder.cache)
    return o1, o2, x, sorted({_variant(k) for k in recorder.used})


def _assert_bitwise(ref, new, heads):
    ref1, ref2, ref_x, _ = ref
    new1, new2, new_x, _ = new
    _assert_out_equal(new1, ref1, heads)
    _assert_out_equal(new2, ref2, heads)
    for name in MUTATED:
        assert torch.equal(new_x[name], ref_x[name]), name
    # sanity: the kernel ran and produced finite values
    act = ref2[0] if isinstance(ref2, tuple) else ref2
    assert torch.isfinite(act.float()).all()


@pytest.mark.parametrize("initial_state", ["zero", "populated"])
@pytest.mark.parametrize("quantize_output", [False, True], ids=["bf16_out", "mxfp8_out"])
@pytest.mark.parametrize(
    "N,H,num_spec", [(40, 96, 3), (72, 96, 3), (16, 96, 2), (24, 16, 3), (64, 12, 3)]
)
def test_compact_layout_is_bitwise_identical(
    N, H, num_spec, quantize_output, initial_state, monkeypatch
):
    if H * N <= _num_sms():
        pytest.skip(f"H*N={H * N} <= {_num_sms()} SMs: single wave, compact layout not taken")
    inp = _make_inputs(N, H, num_spec, seed=1234 + N, zero_state=initial_state == "zero")
    # BF16 output reaches kda_decode_mtp_kernel only with the v2 kernel off
    attrs = {"_MTP_V2": False}
    ref = _run_rounds(inp, quantize_output, {MXFP8_2CTA: False, COMPACT: False}, monkeypatch, attrs)
    new = _run_rounds(inp, quantize_output, {MXFP8_2CTA: False, COMPACT: True}, monkeypatch, attrs)
    assert ref[3] == ["v1"] and new[3] == ["v1_compact"], (ref[3], new[3])
    _assert_bitwise(ref, new, H)


@pytest.mark.parametrize("initial_state", ["zero", "populated"])
@pytest.mark.parametrize(
    "H,N,num_spec", DECODE_SHAPES + [pytest.param(96, 16, 3, id="dep16_b16_mtp3")]
)
def test_mxfp8_two_cta_is_bitwise_identical(H, N, num_spec, initial_state, monkeypatch):
    # (grids of up to SPLIT_MAX_TILES tiles: one CTA per tile vs the cluster pairs of the reference)
    inp = _make_inputs(N, H, num_spec, seed=4321 + 7 * N + H, zero_state=initial_state == "zero")
    ref = _run_rounds(inp, True, {COMPACT: False, MXFP8_2CTA: False}, monkeypatch)
    new = _run_rounds(inp, True, {COMPACT: False, MXFP8_2CTA: True}, monkeypatch)
    assert ref[3] == ["v1"] and new[3] == ["2cta_mxfp8"], (ref[3], new[3])
    _assert_bitwise(ref, new, H)


@pytest.mark.parametrize("initial_state", ["zero", "populated"])
@pytest.mark.parametrize("quantize_output", [False, True], ids=["bf16_out", "mxfp8_out"])
@pytest.mark.parametrize("H,N,num_spec", DECODE_SHAPES)
def test_decode_shapes_default_matches_one_cta_kernels(
    H, N, num_spec, quantize_output, initial_state, monkeypatch
):
    """The variants selected by default on this GPU reproduce the one-CTA kernels."""
    inp = _make_inputs(N, H, num_spec, seed=99 + 7 * N + H, zero_state=initial_state == "zero")
    ref = _run_rounds(inp, quantize_output, {COMPACT: False, MXFP8_2CTA: False}, monkeypatch)
    new = _run_rounds(inp, quantize_output, {COMPACT: None, MXFP8_2CTA: None}, monkeypatch)
    assert ref[3] == (["v1"] if quantize_output else ["v2"]), ref[3]
    # SM107: MXFP8 output runs the two-CTA kernel; other calls are unchanged
    two_cta = quantize_output and torch.cuda.get_device_capability(0) == (10, 7)
    assert new[3] == (["2cta_mxfp8"] if two_cta else ref[3]), new[3]
    _assert_bitwise(ref, new, H)
