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

"""Compact KDA MTP verify layout (``kda_mtp_decode_compact.py``) vs the in-tree kernel.

``TRTLLM_KDA_MTP_COMPACT=1`` sends multi-wave ``trtllm::kda_mtp_decode`` calls to
the compact (two CTAs per SM) kernel derived from feat/k3_26.08
(``kda_mtp_decode_compact.py``). That kernel only re-places shared-memory / TMEM
buffers (time-shared region, Kbar written straight into the B2 tiles, S0 hi
aliased with the GEMM2 accumulator) and reads and writes the split
replay caches (bf16 k / v / beta, fp32 g) exactly like the in-tree kernel. So the
outputs, the committed recurrent state, the conv windows and the replay caches
must be BITWISE identical to the in-tree kernel's default (3xTF32) path, over two
chained verify rounds with mixed per-slot accepted counts (round 2 replays the
round-1 drafts).

Production variant of the K3 decode step: packed token layout, bf16 state pool
(V-first), bf16 conv caches, fused gated RMSNorm, num_spec = 0..7, BF16 output
(SM100/SM103) or MXFP8 output (SM107), with the pools either as per-layer slices
or as KVCM-v2 per-layer sub-pages of coalesced slots (the serving layout: the
state then takes the strided (slot, head) path). H * N must exceed the SM count
for the compact layout to be selected; every device test records which layout
served each call, so none can pass vacuously. num_spec 4..7 (DSpark draft lengths
5..7) run the compact layout with the Kb hi tile in the dead A3 rows and the
cumulative gates in registers, which is what makes them fit two CTAs per SM.

Port of feat/k3_26.08's ``test_kda_mtp_decode_compact.py``. Device tests
need 1 SM100/SM103/SM107 GPU and nvidia-cutlass-dsl; the policy tests mock the
device and run wherever ``tensorrt_llm`` imports.
"""

import itertools
import math
import subprocess
import sys

import pytest
import torch

try:
    from tensorrt_llm._torch.custom_ops import (
        cute_dsl_kimi_k3_kda_mtp_compact_policy as compact_policy,
    )
except ImportError as e:
    compact_policy = None
    _POLICY_ERR = str(e)
else:
    _POLICY_ERR = None

try:
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401

    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as kda_ops
except ImportError as e:
    kda_ops = None
    _KERNEL_ERR = str(e)
else:
    _KERNEL_ERR = None

_VENDORED_KERNEL = (
    "tensorrt_llm._torch.cute_dsl_kernels.blackwell.kimi_k3_kda.kda_mtp_decode_compact"
)


def _device_skip_reason():
    if compact_policy is None:
        return f"deps: {_POLICY_ERR}"
    if kda_ops is None:
        return f"deps: {_KERNEL_ERR}"
    if not torch.cuda.is_available():
        return "needs a GPU"
    if torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        return "needs sm100/sm103/sm107"
    return None


_DEVICE_SKIP = _device_skip_reason()
requires_device = pytest.mark.skipif(_DEVICE_SKIP is not None, reason=_DEVICE_SKIP or "")
requires_policy = pytest.mark.skipif(compact_policy is None, reason=f"deps: {_POLICY_ERR}")
requires_kernel_module = pytest.mark.skipif(kda_ops is None, reason=f"deps: {_KERNEL_ERR}")

K = V = 128
W = 4
# One state-pool size for every case: the pool extent is part of the compiled
# layout, so sharing it lets every (H, num_spec, output, layout) variant compile once.
POOL = 160
MUTATED = ("cs_q", "cs_k", "cs_v", "recurrent_state", "k_cache", "g_cache", "v_cache", "beta_cache")


class _KeyLog(dict):
    """``kda_ops._compiled_cache`` stand-in that records the variant key of every launch."""

    def __init__(self):
        super().__init__()
        self.used = []

    def __getitem__(self, key):
        self.used.append(key)
        return super().__getitem__(key)


# Shared by the device tests of this module: each compiled variant is built once.
_SESSION_CACHE = _KeyLog()


@pytest.fixture
def layouts(monkeypatch):
    """Record the ``(compact, two_ctas)`` decision of every ``kda_mtp_decode_impl`` call.

    Also routes launches through the shared key-logging compile cache
    (``_SESSION_CACHE.used``) and pins the in-tree math to its default (3xTF32).
    """
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "0")
    monkeypatch.setattr(kda_ops, "_compiled_cache", _SESSION_CACHE)
    _SESSION_CACHE.used = []
    decisions = []
    select = kda_ops.select_compact_layout

    def recording_select(**kwargs):
        decision = select(**kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(kda_ops, "select_compact_layout", recording_select)
    return decisions


def _set_policy(monkeypatch, compact, quantized="1", two_ctas="1"):
    """Override the startup snapshot (the switches are read once, at import)."""
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", compact)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", quantized)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", two_ctas)
    monkeypatch.setattr(compact_policy, "_kda_compact_policy_cache", {})


def _pool_buffer(shape, dtype, state_layout, make):
    """A ``[POOL, *shape]`` per-layer pool in one of the two serving layouts.

    ``layer_slice``: layer 1 of a ``[layers, POOL, *shape]`` buffer.
    ``kvcm``: layer 1's sub-page of KVCM-v2 slots that coalesce two layers
    (``KVCacheManagerV2`` ``_get_state_buffer``): slot stride = 2 x the per-layer size.
    """
    if state_layout == "layer_slice":
        return make((2, POOL, *shape)).to(dtype)[1]
    raw = make((2 * POOL, *shape)).to(dtype)
    per_layer = math.prod(shape)
    return raw.as_strided(
        (POOL, *shape), (2 * raw.stride(0), *raw.stride()[1:]), raw.storage_offset() + per_layer
    )


def _make_inputs(N, H, num_spec, seed, state_layout="layer_slice"):
    torch.manual_seed(seed)
    dev = "cuda"
    d = H * K
    T = N * (1 + num_spec)
    S = W - 1 + num_spec
    # fused [q | k | v | output-gate] projection: strided section views
    qkvg = torch.randn(T, 4 * d, dtype=torch.bfloat16, device=dev)
    bfa = torch.randn(T, 128 + H, dtype=torch.bfloat16, device=dev)

    def randn(shape):
        return torch.randn(shape, dtype=torch.float32, device=dev)

    def rand(shape):
        return torch.rand(shape, dtype=torch.float32, device=dev)

    def conv():
        # dim-contiguous: [pool, S, d] storage, viewed as [pool, d, S]
        return _pool_buffer((S, d), torch.bfloat16, state_layout, randn).transpose(-1, -2)

    # KVCM replay layout (``_kda_replay_layout``): bf16 k / v / beta, fp32 gate;
    # beta rows padded to 16 bytes (``_allocate_kda_beta_cache``).
    padded_heads = (H + 7) // 8 * 8
    beta_cache = _pool_buffer((num_spec, padded_heads), torch.bfloat16, state_layout, rand)
    return dict(
        x_q=qkvg[:, 0 * d : 1 * d].view(1, T, H, K),
        x_k=qkvg[:, 1 * d : 2 * d].view(1, T, H, K),
        x_v=qkvg[:, 2 * d : 3 * d].view(1, T, H, V),
        onorm_g=qkvg[:, 3 * d : 4 * d].view(1, T, H, V),
        g=torch.randn(T, d, dtype=torch.bfloat16, device=dev).view(1, T, H, K),
        beta=bfa[:, 128 : 128 + H].view(1, T, H),
        w_q=randn((d, W)) * 0.5,
        w_k=randn((d, W)) * 0.5,
        w_v=randn((d, W)) * 0.5,
        A_log=randn((H,)) * 0.5,
        dt_bias=randn((d,)),
        cs_q=conv(),
        cs_k=conv(),
        cs_v=conv(),
        recurrent_state=_pool_buffer(
            (H, V, K), torch.bfloat16, state_layout, lambda s: randn(s) * 0.1
        ),
        k_cache=_pool_buffer((num_spec, d), torch.bfloat16, state_layout, randn),
        g_cache=_pool_buffer((num_spec, d), torch.float32, state_layout, lambda s: -rand(s) * 2.0),
        v_cache=_pool_buffer((num_spec, d), torch.bfloat16, state_layout, randn),
        beta_cache=beta_cache[..., :H],
        ssm_state_indices=torch.randperm(POOL, device=dev)[:N].to(torch.int32),
        cu_seqlens=None,
        num_spec=num_spec,
        num_accepted_tokens=torch.randint(0, num_spec + 1, (POOL,), dtype=torch.int32, device=dev),
        lower_bound=-5.0,
        scale=K**-0.5,
        packed_token_layout=True,
        onorm_weight=torch.rand(V, dtype=torch.float32, device=dev) + 0.5,
        onorm_eps=1e-5,
        fuse_output_norm=True,
    )


def _clone(inp):
    """Deep-copy the mutated pools, keeping each view's strides into a cloned base."""
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


def _assert_bitwise(name, new, ref):
    assert new.shape == ref.shape and new.dtype == ref.dtype, name
    new_bytes = new.contiguous().view(torch.uint8)
    ref_bytes = ref.contiguous().view(torch.uint8)
    if torch.equal(new_bytes, ref_bytes):
        return
    mismatched = int((new_bytes != ref_bytes).sum())
    detail = ""
    if new.is_floating_point():
        detail = f"; max |diff| {(new.float() - ref.float()).abs().max().item():.3e}"
    pytest.fail(f"{name}: {mismatched} of {new_bytes.numel()} bytes differ{detail}")


def _assert_out_equal(name, new, ref):
    if isinstance(ref, tuple):
        assert isinstance(new, tuple) and len(new) == len(ref), name
        for part, (x, y) in zip(("act", "scale"), zip(new, ref)):
            _assert_bitwise(f"{name}.{part}", x, y)
    else:
        _assert_bitwise(name, new, ref)


def _assert_pools_equal(new_x, ref_x):
    for name in MUTATED:
        _assert_bitwise(name, new_x[name], ref_x[name])
        # whole backing buffers: other layers / padded beta heads must be untouched alike
        if ref_x[name]._base is not None:
            _assert_bitwise(f"{name} (backing buffer)", new_x[name]._base, ref_x[name]._base)


def _run_round(x):
    return _clone_out(kda_ops.kda_mtp_decode_impl(**x))


def _run_two_rounds(inp, compact, monkeypatch, quantize_output=False, two_ctas="1", quantized="1"):
    _set_policy(monkeypatch, "1" if compact else "0", quantized=quantized, two_ctas=two_ctas)
    x = _clone(inp)
    x["quantize_output"] = quantize_output
    o1 = _run_round(x)
    # round 2: the sampler accepted a different number of drafts per slot
    x["num_accepted_tokens"] = torch.roll(x["num_accepted_tokens"], 1)
    o2 = _run_round(x)
    torch.cuda.synchronize()
    return o1, o2, x


def _skip_single_wave(N, H):
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if H * N <= num_sms:
        pytest.skip(f"H*N={H * N} <= {num_sms} SMs: single wave, compact path not taken")


def _skip_unvalidated_num_spec(num_spec):
    if num_spec > 3 and torch.cuda.get_device_capability(0) != (10, 7):
        pytest.skip(f"num_spec={num_spec}: the policy selects the compact layout on SM107 only")


# ----------------------------------------------------------------------------
# Device tests
# ----------------------------------------------------------------------------


@requires_device
@pytest.mark.parametrize("num_spec", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("two_ctas", [False, True])
@pytest.mark.parametrize("quantize_output", [False, True])
@pytest.mark.parametrize("N,H", [(40, 96), (24, 16), (64, 12), (72, 96)])
def test_compact_layout_is_bitwise_identical(
    N, H, num_spec, quantize_output, two_ctas, monkeypatch, layouts
):
    _skip_single_wave(N, H)
    _skip_unvalidated_num_spec(num_spec)
    inp = _make_inputs(N, H, num_spec=num_spec, seed=1234 + N)
    flag = "1" if two_ctas else "0"
    ref1, ref2, ref_x = _run_two_rounds(inp, False, monkeypatch, quantize_output, flag)
    new1, new2, new_x = _run_two_rounds(inp, True, monkeypatch, quantize_output, flag)
    # one layout per world: the comparison is not vacuous on this GPU
    assert layouts == [(False, False)] * 2 + [(True, two_ctas)] * 2, layouts
    _assert_out_equal("round-1 output", new1, ref1)
    _assert_out_equal("round-2 output", new2, ref2)
    _assert_pools_equal(new_x, ref_x)
    # sanity: the kernel did run and produced finite values
    ref_act = ref2[0] if isinstance(ref2, tuple) else ref2
    assert torch.isfinite(ref_act.float()).all()


@requires_device
@pytest.mark.parametrize("num_spec", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("quantize_output", [False, True])
@pytest.mark.parametrize("two_ctas", [False, True])
def test_compact_layout_kvcm_pools_are_bitwise_identical(
    quantize_output, two_ctas, num_spec, monkeypatch, layouts
):
    """Serving layout: KVCM-v2 sub-page pools, so the state takes the (slot, head) TMA path."""
    N, H = 64, 96
    _skip_single_wave(N, H)
    _skip_unvalidated_num_spec(num_spec)
    inp = _make_inputs(N, H, num_spec=num_spec, seed=4321, state_layout="kvcm")
    with pytest.raises(RuntimeError):
        inp["recurrent_state"].view(-1, V, K)  # not flattenable: strided per-slot state
    flag = "1" if two_ctas else "0"
    ref1, ref2, ref_x = _run_two_rounds(inp, False, monkeypatch, quantize_output, flag)
    new1, new2, new_x = _run_two_rounds(inp, True, monkeypatch, quantize_output, flag)
    assert layouts == [(False, False)] * 2 + [(True, two_ctas)] * 2, layouts
    _assert_out_equal("round-1 output", new1, ref1)
    _assert_out_equal("round-2 output", new2, ref2)
    _assert_pools_equal(new_x, ref_x)


@requires_device
@pytest.mark.parametrize("counts", ["every", "oversized"])
@pytest.mark.parametrize("num_spec", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("quantize_output", [False, True])
@pytest.mark.parametrize("two_ctas", [False, True])
def test_compact_layout_every_accepted_count_is_bitwise_identical(
    two_ctas, quantize_output, num_spec, counts, monkeypatch, layouts
):
    """Every accepted count 0..num_spec occurs in both rounds, plus counts above num_spec.

    The count picks the TMEM columns of the new-token outputs and which Kbar column pair a
    thread owns, so each value is a distinct code path; random counts reach them only by
    chance. Both kernels clamp a count above num_spec to num_spec.
    """
    N, H = 72, 96
    _skip_single_wave(N, H)
    _skip_unvalidated_num_spec(num_spec)
    inp = _make_inputs(N, H, num_spec=num_spec, seed=2468)
    dev = inp["num_accepted_tokens"].device
    inp["ssm_state_indices"] = torch.arange(N, dtype=torch.int32, device=dev)
    accepted = torch.arange(POOL, dtype=torch.int32, device=dev) % (num_spec + 1)
    if counts == "oversized":
        accepted[::5] = num_spec + 3
    inp["num_accepted_tokens"] = accepted
    flag = "1" if two_ctas else "0"
    ref1, ref2, ref_x = _run_two_rounds(inp, False, monkeypatch, quantize_output, flag)
    new1, new2, new_x = _run_two_rounds(inp, True, monkeypatch, quantize_output, flag)
    assert layouts == [(False, False)] * 2 + [(True, two_ctas)] * 2, layouts
    _assert_out_equal("round-1 output", new1, ref1)
    _assert_out_equal("round-2 output", new2, ref2)
    _assert_pools_equal(new_x, ref_x)


@requires_device
@pytest.mark.parametrize("num_spec", [3, 7])
@pytest.mark.parametrize("first_compact", [False, True])
def test_slots_can_switch_layout_between_rounds(first_compact, num_spec, monkeypatch, layouts):
    """A slot's state / replay caches written by one kernel are continued by the other.

    Serving switches layout with the batch size (single wave -> in-tree kernel), so
    round 2 may replay drafts that the other kernel cached in round 1.
    """
    N, H = 40, 96
    _skip_single_wave(N, H)
    _skip_unvalidated_num_spec(num_spec)
    inp = _make_inputs(N, H, num_spec=num_spec, seed=99)
    ref1, ref2, ref_x = _run_two_rounds(inp, False, monkeypatch, quantize_output=True)
    x = _clone(inp)
    x["quantize_output"] = True
    _set_policy(monkeypatch, "1" if first_compact else "0")
    new1 = _run_round(x)
    x["num_accepted_tokens"] = torch.roll(x["num_accepted_tokens"], 1)
    _set_policy(monkeypatch, "0" if first_compact else "1")
    new2 = _run_round(x)
    torch.cuda.synchronize()
    chosen = [compact for compact, _ in layouts]
    assert chosen == [False, False, first_compact, not first_compact], layouts
    _assert_out_equal("round-1 output", new1, ref1)
    _assert_out_equal("round-2 output", new2, ref2)
    _assert_pools_equal(x, ref_x)


@requires_device
@pytest.mark.parametrize("num_spec", [3, 7])
@pytest.mark.parametrize("quantize_output", [False, True])
def test_vendored_default_layout_matches_in_tree_kernel(
    quantize_output, num_spec, monkeypatch, layouts
):
    """Diagnostic: the vendored kernel with COMPACT=False vs the in-tree kernel.

    Isolates the replay-cache port of the vendored kernel (split k / g caches, bf16
    stores, Int64 slot) from its layout change: if this passes and the compact test
    fails, the difference is in the compact placement itself.
    """
    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_compact_ops as compact_ops

    N, H = 40, 96
    _skip_single_wave(N, H)
    _skip_unvalidated_num_spec(num_spec)
    # private cache: the doctored compile below must not leak into the shared one
    monkeypatch.setattr(kda_ops, "_compiled_cache", {})
    launchers = []
    real_compile = kda_ops.cute.compile

    def compile_vendored_default_layout(*args, **kwargs):
        launchers.append(args[0])
        if args[0] is compact_ops._run_kda_decode_mtp_compact:
            assert kwargs["COMPACT"] is True
            kwargs = dict(kwargs, COMPACT=False, COMPACT_TWO_CTAS=False)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(kda_ops.cute, "compile", compile_vendored_default_layout)
    inp = _make_inputs(N, H, num_spec=num_spec, seed=7)
    ref1, ref2, ref_x = _run_two_rounds(inp, False, monkeypatch, quantize_output)
    new1, new2, new_x = _run_two_rounds(inp, True, monkeypatch, quantize_output)
    assert launchers == [kda_ops._run_kda_decode_mtp, compact_ops._run_kda_decode_mtp_compact]
    assert [compact for compact, _ in layouts] == [False, False, True, True], layouts
    _assert_out_equal("round-1 output", new1, ref1)
    _assert_out_equal("round-2 output", new2, ref2)
    _assert_pools_equal(new_x, ref_x)


@requires_device
def test_compact_policy_switches_without_clearing_cache(monkeypatch, layouts):
    """Policy changes select distinct compiled variants and preserve results."""
    _skip_single_wave(40, 96)
    inp = _make_inputs(40, 96, num_spec=3, seed=1234)
    # MXFP8 output with _QUANTIZED=0 keeps the in-tree kernel even with the switch on
    ref1, ref2, ref_x = _run_two_rounds(inp, True, monkeypatch, True, "0", quantized="0")
    for two_ctas in ("0", "1"):
        new1, new2, new_x = _run_two_rounds(inp, True, monkeypatch, True, two_ctas)
        _assert_out_equal("round-1 output", new1, ref1)
        _assert_out_equal("round-2 output", new2, ref2)
        _assert_pools_equal(new_x, ref_x)
    assert layouts == [(False, False)] * 2 + [(True, False)] * 2 + [(True, True)] * 2, layouts
    keys = list(dict.fromkeys(_SESSION_CACHE.used))
    assert len(keys) == 3, "each policy must map to its own compiled variant"
    _set_policy(monkeypatch, "1", quantized="0", two_ctas="1")
    _SESSION_CACHE.used = []
    _run_round(dict(_clone(inp), quantize_output=True))
    assert _SESSION_CACHE.used == [keys[0]], "disabled policy must reuse the original variant"


@requires_device
def test_compact_policy_architecture_defaults(monkeypatch, layouts):
    """Unset _QUANTIZED / _TWO_CTAS follow the device: on for SM107 only."""
    _skip_single_wave(40, 96)
    inp = _make_inputs(40, 96, num_spec=3, seed=1234)
    _run_two_rounds(inp, True, monkeypatch, True, two_ctas=None, quantized=None)
    is_rubin = torch.cuda.get_device_capability(0) == (10, 7)
    assert layouts == [(is_rubin, is_rubin)] * 2, layouts


@requires_device
def test_compact_with_bf16_math_raises_before_launch(monkeypatch, layouts):
    _skip_single_wave(40, 96)
    inp = _make_inputs(40, 96, num_spec=3, seed=5)
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "1")
    _set_policy(monkeypatch, "1")
    x = _clone(inp)
    with pytest.raises(ValueError, match="mutually exclusive"):
        kda_ops.kda_mtp_decode_impl(**x, quantize_output=True)
    assert layouts == [] and _SESSION_CACHE.used == []
    for name in MUTATED:
        assert torch.equal(x[name], inp[name]), f"{name} mutated by a rejected call"


# ----------------------------------------------------------------------------
# Import isolation (no GPU needed)
# ----------------------------------------------------------------------------


@requires_kernel_module
def test_default_path_does_not_import_vendored_kernel():
    """The in-tree op module must not load the vendored kernel until compact is selected."""
    probe = (
        "import sys\n"
        "import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops\n"
        f"assert {_VENDORED_KERNEL!r} not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


# ----------------------------------------------------------------------------
# Policy tests (no GPU: the device is mocked)
# ----------------------------------------------------------------------------


def _mock_devices(monkeypatch, capabilities, num_sms=148, current=0):
    calls = []

    def capability(idx):
        calls.append(idx)
        return capabilities[idx]

    class _Props:
        multi_processor_count = num_sms

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: _Props())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(compact_policy, "_kda_compact_policy_cache", {})
    monkeypatch.setattr(compact_policy, "_num_sms_cache", {})
    return calls


def _feat_k3_26_08_policy(
    sm, enabled, quantized, split_v, tiles, num_sms, dtype, fuse, num_spec, quantize
):
    """Selection expression of feat/k3_26.08 ``kda_mtp_decode_impl``, verbatim except
    for the num_spec bound: 7 on SM107 (the chunk-form limit; feat/k3_26.08 stops at 3,
    where its compact smem layout ran out for two CTAs per SM), 3 on SM100/103, where
    num_spec 4..7 is not validated."""
    return (
        split_v == 1
        and tiles > num_sms
        and dtype is torch.bfloat16
        and bool(fuse)
        and num_spec <= (7 if sm == (10, 7) else 3)
        and (not quantize or quantized)
        and sm in ((10, 0), (10, 3), (10, 7))
        and enabled
    )


@requires_policy
@pytest.mark.parametrize("sm", [(9, 0), (10, 0), (10, 3), (10, 7), (12, 0)])
@pytest.mark.parametrize("quantized_env", [None, "0", "1"])
@pytest.mark.parametrize("two_ctas_env", [None, "0", "1"])
def test_policy_matches_feat_k3_26_08(sm, quantized_env, two_ctas_env, monkeypatch):
    num_sms = 148
    _mock_devices(monkeypatch, {0: sm}, num_sms=num_sms)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", "1")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", quantized_env)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", two_ctas_env)
    rubin = sm == (10, 7)
    quantized = rubin if quantized_env is None else quantized_env == "1"
    two_ctas = rubin if two_ctas_env is None else two_ctas_env == "1"
    grid = itertools.product(
        (1, 2),  # split_v
        (num_sms - 1, num_sms, num_sms + 1, 96 * 64),  # tiles
        (torch.bfloat16, torch.float32),  # state dtype
        (False, True),  # fused output norm
        (0, 3, 4, 7, 8),  # num_spec
        (False, True),  # quantize_output
    )
    for split_v, tiles, dtype, fuse, num_spec, quantize in grid:
        expect = _feat_k3_26_08_policy(
            sm, True, quantized, split_v, tiles, num_sms, dtype, fuse, num_spec, quantize
        )
        got = compact_policy.select_compact_layout(
            device=torch.device("cuda", 0),
            split_v=split_v,
            num_tiles=tiles,
            state_dtype=dtype,
            fuse_output_norm=fuse,
            num_spec=num_spec,
            quantize_output=quantize,
            bf16_math=False,
        )
        assert got == (expect, expect and two_ctas), (split_v, tiles, dtype, fuse, num_spec)


_PRODUCTION_CALL = dict(
    split_v=1,
    num_tiles=96 * 64,
    state_dtype=torch.bfloat16,
    fuse_output_norm=True,
    num_spec=3,
    quantize_output=True,
)


@requires_policy
@pytest.mark.parametrize("sub_switches", [None, "1", "true"])
@pytest.mark.parametrize("compact_env", [None, "0"])
@pytest.mark.parametrize("bf16_math", [False, True])
def test_default_keeps_in_tree_kernel(compact_env, bf16_math, sub_switches, monkeypatch):
    """Unlike feat/k3_26.08 (default on), unset means the in-tree kernel on this branch.

    With the compact layout off, _QUANTIZED / _TWO_CTAS are not read at all (not even
    validated), so the default path depends on TRTLLM_KDA_MTP_COMPACT alone.
    """
    _mock_devices(monkeypatch, {0: (10, 7)})
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", compact_env)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", sub_switches)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", sub_switches)
    got = compact_policy.select_compact_layout(
        device=torch.device("cuda", 0), bf16_math=bf16_math, **_PRODUCTION_CALL
    )
    assert got == (False, False)


@requires_policy
@pytest.mark.parametrize("num_tiles", [32, 96 * 64])
def test_compact_and_bf16_math_are_mutually_exclusive(num_tiles, monkeypatch):
    """Both switches on is rejected for every shape, not only where compact would run."""
    _mock_devices(monkeypatch, {0: (10, 7)})
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", "1")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", None)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", None)
    call = dict(_PRODUCTION_CALL, num_tiles=num_tiles, split_v=2 if num_tiles <= 64 else 1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        compact_policy.select_compact_layout(device=torch.device("cuda", 0), bf16_math=True, **call)
    expect = num_tiles > 148
    assert compact_policy.select_compact_layout(
        device=torch.device("cuda", 0), bf16_math=False, **call
    ) == (expect, expect)


@requires_policy
@pytest.mark.parametrize("override", [None, "0", "1"])
def test_compact_policy_is_cached_per_device(override, monkeypatch):
    calls = _mock_devices(monkeypatch, {0: (10, 3), 1: (10, 7)}, current=1)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", "1")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", override)
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", override)
    for idx, sm in enumerate(((10, 3), (10, 7))):
        expected = sm == (10, 7) if override is None else override == "1"
        assert compact_policy._kda_compact_policy(torch.device("cuda", idx)) == (
            sm,
            True,
            expected,
            expected,
        )
    # startup settings: later environment changes are not observed
    for name in ("COMPACT", "COMPACT_QUANTIZED", "COMPACT_TWO_CTAS"):
        monkeypatch.setenv("TRTLLM_KDA_MTP_" + name, "0")
    assert compact_policy._kda_compact_policy(
        torch.device("cuda")
    ) is compact_policy._kda_compact_policy(torch.device("cuda", 1))
    compact_policy._kda_compact_policy(torch.device("cuda", 0))
    assert calls == [0, 1]


@requires_policy
@pytest.mark.parametrize(
    "attr", ["_KDA_MTP_COMPACT", "_KDA_MTP_COMPACT_QUANTIZED", "_KDA_MTP_COMPACT_TWO_CTAS"]
)
@pytest.mark.parametrize("value", ["", "false", "true", "2", " 1"])
def test_invalid_switch_values_raise(attr, value, monkeypatch):
    _mock_devices(monkeypatch, {0: (10, 7)})
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", "1")
    monkeypatch.setattr(compact_policy, attr, value)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        compact_policy._kda_compact_policy(torch.device("cuda", 0))
