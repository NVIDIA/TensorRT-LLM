# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MK1 port: programmatic dependent launch (PDL) of the KDA MTP verify kernels.

``TRTLLM_K3_KDA_VERIFY_PDL=1`` launches ``trtllm::kda_mtp_decode{,_fp8}`` (the in-tree kernel and
the vendored compact layout) with programmatic dependent launch. A CTA then reads only what the
projections do not produce (slot / accepted-count tables, the recurrent state, the conv windows,
the replay caches, the conv / gate / norm weights) before ``griddepcontrol.wait``, and the
projections' outputs after it; it releases its dependents once it holds its tensor-memory columns
and marks the state copies L2 evict-first. Nothing changes numerically, so the outputs, the
committed state, the conv windows and the replay caches must be BITWISE identical to the default
launch, over two chained verify rounds with mixed per-slot accepted counts, for every ``num_spec``
0..7.

Device tests (SM107 with the CuTe DSL Rubin kernels; the switch is ignored elsewhere):

* parity: the PDL launch against the default launch on every layout the K3 packed path can select
  (a cluster pair per tile, one CTA per tile in one wave and in several waves, the compact layout
  with one and two CTAs per SM) and on the state / math / norm variants;
* hazard: the projections' outputs are written by producer kernels that finish only after the HOST
  releases them (it does so tens of milliseconds after it has enqueued the verify), eagerly and
  inside a CUDA graph with the producers of the mixer on two streams. A verify CTA that read an
  output before its ``griddepcontrol.wait`` sees a stale buffer, and the result differs from the
  default launch's. A probe kernel in the verify's place shows that a PDL launch in such a chain
  starts before the release (a hazard test that cannot overlap would pass on a kernel without the
  wait: the host needs about a millisecond to enqueue the verify, longer than any timer-based
  delay of the producers).
"""

import math
import time

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
    from tensorrt_llm._torch.modules import kimi_k3_mk1_gate as mk1_gate
except ImportError as e:
    kda_ops = None
    mk1_gate = None
    _KERNEL_ERR = str(e)
else:
    _KERNEL_ERR = None

PDL_ENV = "TRTLLM_K3_KDA_VERIFY_PDL"


def _device_skip_reason():
    if kda_ops is None or compact_policy is None:
        return f"deps: {_KERNEL_ERR or _POLICY_ERR}"
    if not torch.cuda.is_available():
        return "needs a GPU"
    if torch.cuda.get_device_capability(0) != (10, 7):
        return "the MK1 PDL variant is validated on SM107 only"
    if not mk1_gate.mk1_platform_supported():
        return "needs the CuTe DSL Rubin helpers (rubin_helpers)"
    return None


_DEVICE_SKIP = _device_skip_reason()
requires_device = pytest.mark.skipif(_DEVICE_SKIP is not None, reason=_DEVICE_SKIP or "")
requires_kernel_module = pytest.mark.skipif(kda_ops is None, reason=f"deps: {_KERNEL_ERR}")

K = V = 128
W = 4
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
def key_log(monkeypatch):
    """Route launches through the shared key-logging compile cache; pin the in-tree math."""
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "0")
    monkeypatch.setattr(kda_ops, "_compiled_cache", _SESSION_CACHE)
    _SESSION_CACHE.used = []
    return _SESSION_CACHE


def _set_compact(monkeypatch, compact, two_ctas=True):
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT", "1" if compact else "0")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_QUANTIZED", "1")
    monkeypatch.setattr(compact_policy, "_KDA_MTP_COMPACT_TWO_CTAS", "1" if two_ctas else "0")
    monkeypatch.setattr(compact_policy, "_kda_compact_policy_cache", {})


def _set_pdl(monkeypatch, pdl):
    monkeypatch.delenv(PDL_ENV, raising=False)
    if pdl:
        monkeypatch.setenv(PDL_ENV, "1")


def _pool_buffer(shape, dtype, state_layout, make):
    """A ``[POOL, *shape]`` per-layer pool: layer 1 of ``[layers, POOL, *shape]`` (``layer_slice``)
    or layer 1's sub-page of KVCM-v2 slots that coalesce two layers (``kvcm``)."""
    if state_layout == "layer_slice":
        return make((2, POOL, *shape)).to(dtype)[1]
    raw = make((2 * POOL, *shape)).to(dtype)
    per_layer = math.prod(shape)
    return raw.as_strided(
        (POOL, *shape), (2 * raw.stride(0), *raw.stride()[1:]), raw.storage_offset() + per_layer
    )


def _make_inputs(N, H, num_spec, seed, state_layout="layer_slice", state_dtype=torch.bfloat16):
    torch.manual_seed(seed)
    dev = "cuda"
    d = H * K
    T = N * (1 + num_spec)
    S = W - 1 + num_spec
    qkvg = torch.randn(T, 4 * d, dtype=torch.bfloat16, device=dev)
    bfa = torch.randn(T, 128 + H, dtype=torch.bfloat16, device=dev)

    def randn(shape):
        return torch.randn(shape, dtype=torch.float32, device=dev)

    def rand(shape):
        return torch.rand(shape, dtype=torch.float32, device=dev)

    def conv():
        return _pool_buffer((S, d), torch.bfloat16, state_layout, randn).transpose(-1, -2)

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
            (H, V, K), state_dtype, state_layout, lambda s: randn(s) * 0.1
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
        # MXFP8 scales pad rows to 128; the padding is never written. Compare every logical
        # row / head / replica.
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
    pytest.fail(f"{name}: {mismatched} of {new_bytes.numel()} bytes differ")


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
        if ref_x[name]._base is not None:
            _assert_bitwise(f"{name} (backing buffer)", new_x[name]._base, ref_x[name]._base)


def _two_rounds(inp, quantize_output):
    """Two chained verify rounds (round 2 replays round 1's drafts with other accepted counts)."""
    x = _clone(inp)
    x["quantize_output"] = quantize_output
    o1 = _clone_out(kda_ops.kda_mtp_decode_impl(**x))
    x["num_accepted_tokens"] = torch.roll(x["num_accepted_tokens"], 1)
    o2 = _clone_out(kda_ops.kda_mtp_decode_impl(**x))
    torch.cuda.synchronize()
    return o1, o2, x


def _pdl_flags_of(key_log):
    """``use_pdl`` of every recorded launch (the last element of the compile key)."""
    return [k[-1] for k in key_log.used]


# (N, H, compact): the in-tree kernel as one cluster pair per tile (H * N <= 64 tiles), as one CTA
# per tile in a single wave (<= 192 tiles at TP8 MBS16) and in several waves (more tiles than SMs,
# compact layout off), and the compact layout (more tiles than SMs: two CTAs per SM).
LAYOUT_SHAPES = [
    pytest.param(2, 12, False, id="intree-split2"),
    pytest.param(16, 12, False, id="intree-onewave"),
    pytest.param(24, 12, False, id="intree-multiwave"),
    pytest.param(40, 12, True, id="compact"),
]


def _skip_wrong_layout(compact, N, H):
    """The compact layout needs more tiles than SMs: skip where a shape does not select it."""
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if compact and H * N <= num_sms:
        pytest.skip(f"H*N={H * N} <= {num_sms} SMs: the compact layout is not selected")


# ----------------------------------------------------------------------------
# Switch handling (no GPU needed)
# ----------------------------------------------------------------------------


@requires_kernel_module
def test_switch_off_by_default(monkeypatch):
    _set_pdl(monkeypatch, False)
    assert kda_ops._verify_pdl_enabled() is False


@requires_kernel_module
def test_switch_ignored_off_sm107(monkeypatch):
    monkeypatch.setattr(mk1_gate, "mk1_platform_supported", lambda: False)
    _set_pdl(monkeypatch, True)
    assert kda_ops._verify_pdl_enabled() is False


@requires_kernel_module
def test_switch_on_where_supported(monkeypatch):
    monkeypatch.setattr(mk1_gate, "mk1_platform_supported", lambda: True)
    _set_pdl(monkeypatch, True)
    assert kda_ops._verify_pdl_enabled() is True


@requires_kernel_module
def test_stack_wide_pdl_switch_wins(monkeypatch):
    """TRTLLM_ENABLE_PDL=0 turns every PDL launch of the stack off, this one included."""
    monkeypatch.setattr(mk1_gate, "mk1_platform_supported", lambda: True)
    _set_pdl(monkeypatch, True)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    assert kda_ops._verify_pdl_enabled() is False
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    assert kda_ops._verify_pdl_enabled() is True


@requires_kernel_module
@pytest.mark.parametrize("value", ["2", "true", "on"])
def test_invalid_switch_values_raise(monkeypatch, value):
    monkeypatch.setattr(mk1_gate, "mk1_platform_supported", lambda: True)
    monkeypatch.setenv(PDL_ENV, value)
    with pytest.raises(ValueError):
        kda_ops._verify_pdl_enabled()


@requires_kernel_module
@pytest.mark.parametrize(
    "packed,converted,expected",
    [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
)
def test_per_call_decision(monkeypatch, packed, converted, expected):
    """PDL needs the packed layout (every CTA waits) and accepted counts that are not the output of
    a kernel launched right before the verify (the wrapper's int64 -> int32 conversion)."""
    monkeypatch.setattr(mk1_gate, "mk1_platform_supported", lambda: True)
    _set_pdl(monkeypatch, True)
    assert kda_ops._verify_pdl_for_call(packed, converted) is expected


@requires_kernel_module
def test_per_call_decision_off_without_the_switch(monkeypatch):
    monkeypatch.setattr(mk1_gate, "mk1_platform_supported", lambda: True)
    _set_pdl(monkeypatch, False)
    assert kda_ops._verify_pdl_for_call(True, False) is False


# ----------------------------------------------------------------------------
# Parity
# ----------------------------------------------------------------------------


def _check_parity(
    N,
    H,
    compact,
    num_spec,
    quantize_output,
    state_layout,
    monkeypatch,
    key_log,
    two_ctas=True,
    state_dtype=torch.bfloat16,
    edit=None,
):
    _skip_wrong_layout(compact, N, H)
    inp = _make_inputs(
        N, H, num_spec=num_spec, seed=4321 + N, state_layout=state_layout, state_dtype=state_dtype
    )
    if edit is not None:
        edit(inp)
    _set_compact(monkeypatch, compact, two_ctas)
    _set_pdl(monkeypatch, False)
    ref1, ref2, ref_x = _two_rounds(inp, quantize_output)
    assert set(_pdl_flags_of(key_log)) == {False}
    assert all(bool(k[-3]) == compact for k in key_log.used), "not the layout the case names"
    assert not compact or all(bool(k[-2]) == two_ctas for k in key_log.used)
    key_log.used = []
    _set_pdl(monkeypatch, True)
    new1, new2, new_x = _two_rounds(inp, quantize_output)
    assert set(_pdl_flags_of(key_log)) == {True}, "the PDL variant did not run"
    _assert_out_equal("round-1 output", new1, ref1)
    _assert_out_equal("round-2 output", new2, ref2)
    _assert_pools_equal(new_x, ref_x)
    ref_act = ref2[0] if isinstance(ref2, tuple) else ref2
    assert torch.isfinite(ref_act.float()).all()


@requires_device
@pytest.mark.parametrize("num_spec", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("N,H,compact", LAYOUT_SHAPES)
def test_pdl_is_bitwise_identical(N, H, compact, num_spec, monkeypatch, key_log):
    """MXFP8 output (the SM107 production variant), per-layer pools."""
    _check_parity(N, H, compact, num_spec, True, "layer_slice", monkeypatch, key_log)


@requires_device
@pytest.mark.parametrize("num_spec", [3, 7])
@pytest.mark.parametrize("N,H,compact", LAYOUT_SHAPES)
def test_pdl_bf16_output_is_bitwise_identical(N, H, compact, num_spec, monkeypatch, key_log):
    _check_parity(N, H, compact, num_spec, False, "layer_slice", monkeypatch, key_log)


@requires_device
@pytest.mark.parametrize("num_spec", [3, 5, 7])
@pytest.mark.parametrize("N,H,compact", LAYOUT_SHAPES[1:])
def test_pdl_kvcm_pools_are_bitwise_identical(N, H, compact, num_spec, monkeypatch, key_log):
    """The serving layout: per-layer sub-pages of KVCM-v2 slots (strided state rows)."""
    _check_parity(N, H, compact, num_spec, True, "kvcm", monkeypatch, key_log)


@requires_device
@pytest.mark.parametrize("num_spec", [3, 7])
def test_pdl_compact_one_cta_per_sm_is_bitwise_identical(num_spec, monkeypatch, key_log):
    """The compact layout with TRTLLM_KDA_MTP_COMPACT_TWO_CTAS=0 (one CTA per SM, other launch)."""
    _check_parity(40, 12, True, num_spec, True, "layer_slice", monkeypatch, key_log, two_ctas=False)


def _no_output_norm(inp):
    inp.update(fuse_output_norm=False, onorm_g=None, onorm_weight=None)


def _zero_accepted(inp):
    inp.update(
        zero_accepted_hint=True, num_accepted_tokens=torch.zeros_like(inp["num_accepted_tokens"])
    )


@requires_device
@pytest.mark.parametrize("N,H,compact", LAYOUT_SHAPES[:2])
@pytest.mark.parametrize("variant", ["fp32-state", "bf16-math", "no-output-norm", "zero-accepted"])
def test_pdl_variants_are_bitwise_identical(variant, N, H, compact, monkeypatch, key_log):
    """State dtype, tensor-core math, the optional output norm and the no-replay variant."""
    num_spec, quantize_output, edit, extra = 3, True, None, {}
    if variant == "fp32-state":
        extra["state_dtype"] = torch.float32
    elif variant == "bf16-math":
        monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "1")
    elif variant == "no-output-norm":
        quantize_output, edit = False, _no_output_norm
    else:
        num_spec, edit = 2, _zero_accepted
    _check_parity(
        N,
        H,
        compact,
        num_spec,
        quantize_output,
        "layer_slice",
        monkeypatch,
        key_log,
        edit=edit,
        **extra,
    )


@requires_device
def test_pdl_is_skipped_for_int64_accepted_counts(monkeypatch, key_log):
    """int64 counts are converted by a kernel launched right before the verify: no PDL, and the
    results equal the default launch's."""
    inp = _make_inputs(16, 12, num_spec=3, seed=77)
    inp["num_accepted_tokens"] = inp["num_accepted_tokens"].to(torch.int64)
    _set_compact(monkeypatch, False)
    _set_pdl(monkeypatch, False)
    ref1, ref2, ref_x = _two_rounds(inp, True)
    key_log.used = []
    _set_pdl(monkeypatch, True)
    new1, new2, new_x = _two_rounds(inp, True)
    assert set(_pdl_flags_of(key_log)) == {False}
    _assert_out_equal("round-1 output", new1, ref1)
    _assert_out_equal("round-2 output", new2, ref2)
    _assert_pools_equal(new_x, ref_x)


# ----------------------------------------------------------------------------
# Hazard: the producers' outputs are written after the verify CTAs have started
# ----------------------------------------------------------------------------
# The producers are Triton kernels that release their dependents at once, wait for their own
# predecessors, spin until the HOST stores a flag in pinned memory (tens of milliseconds after it
# has enqueued the verify) and only then write their outputs. Each is a single wave with a small
# grid (at most _PRODUCER_CTAS CTAs, a grid-stride loop inside, as the projection GEMMs run tens of
# CTAs): a dependent launches only after every CTA of its primary has triggered or exited, so a
# producer with more CTAs than the GPU holds delays the verify's launch until its own spin has
# ended and no hazard can occur. The release is a plain host store, not a kernel on another
# stream (such a release kernel did not run while the spinning chain held the GPU).
# A probe kernel in the verify's place reads the flag at its entry and after its own wait.

_KERNELS = {}
_SPIN_TIMEOUT_NS = 3_000_000_000  # a missing release ends the spin (and fails the comparison)
_RELEASE_AFTER_S = 0.05
_BLOCK = 1024
_PRODUCER_CTAS = 16


def _hazard_kernels():
    """Triton kernels of the hazard tests (built once)."""
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def _now():
        return tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )

    @triton.jit
    def _wait_for_predecessors():
        tl.inline_asm_elementwise(
            "griddepcontrol.wait; // dummy $0",
            "=r,~{memory}",
            [],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _spin_copy(
        flag_addr, timeout_ns, src, dst, nbytes, BLOCK: tl.constexpr, ITERS: tl.constexpr
    ):
        flag = flag_addr.to(tl.int64).to(tl.pointer_type(tl.int32))
        tl.extra.cuda.gdc_launch_dependents()
        _wait_for_predecessors()
        t0 = _now()
        while (tl.load(flag, volatile=True) == 0) & (_now() - t0 < timeout_ns):
            pass
        for i in range(ITERS):
            offs = (i * tl.num_programs(0) + tl.program_id(0)) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < nbytes
            tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    @triton.jit
    def _probe(flag_addr, stamps):
        flag = flag_addr.to(tl.int64).to(tl.pointer_type(tl.int32))
        at_entry = tl.load(flag, volatile=True)
        _wait_for_predecessors()
        after_wait = tl.load(flag, volatile=True)
        tl.store(stamps + 2 * tl.program_id(0), at_entry)
        tl.store(stamps + 2 * tl.program_id(0) + 1, after_wait)

    _KERNELS.update(triton=triton, spin_copy=_spin_copy, probe=_probe)
    return _KERNELS


def _spin_copy_into(flag_addr, src, dst):
    """Copy the bytes of ``src`` into ``dst`` (contiguous, same size) once the host flag is set."""
    k = _hazard_kernels()
    nbytes = dst.numel() * dst.element_size()
    tiles = k["triton"].cdiv(nbytes, _BLOCK)
    ctas = min(tiles, _PRODUCER_CTAS)
    k["spin_copy"][(ctas,)](
        flag_addr,
        _SPIN_TIMEOUT_NS,
        src.view(torch.uint8),
        dst.view(torch.uint8),
        nbytes,
        BLOCK=_BLOCK,
        ITERS=k["triton"].cdiv(tiles, ctas),
        num_warps=4,
        launch_pdl=True,
    )


def _stale_inputs(inp):
    """Inputs whose producer outputs hold other values; the producers' copies make them correct."""
    stale = dict(inp)
    qkvg = inp["x_q"]._base
    assert qkvg is not None
    stale_qkvg = torch.randn_like(qkvg)
    H = inp["x_q"].shape[2]
    T = inp["x_q"].shape[1]
    d = H * K
    stale.update(
        x_q=stale_qkvg[:, 0 * d : 1 * d].view(1, T, H, K),
        x_k=stale_qkvg[:, 1 * d : 2 * d].view(1, T, H, K),
        x_v=stale_qkvg[:, 2 * d : 3 * d].view(1, T, H, V),
        onorm_g=stale_qkvg[:, 3 * d : 4 * d].view(1, T, H, V),
        g=torch.randn_like(inp["g"]),
        beta=torch.randn_like(inp["beta"]),
    )
    return stale, stale_qkvg, qkvg


class _Rig:
    """Producers that finish only after the host releases them, then a consumer (verify or probe).

    ``consumer(rig)`` is called right after the producers are enqueued. Eagerly all producers run
    on the current stream, a PDL chain. With ``graph`` the producers and the consumer are captured
    once with the mixer's structure (the projection on the main stream, the gate and beta on a
    side stream, joined by an event; the captured edges into the consumer are programmatic) and
    replayed per run.
    """

    def __init__(self, inp, consumer, graph, keep_graph=False):
        self.consumer, self.graph_mode = consumer, graph
        # The release flag: pinned host memory, written by the host, polled by the producers.
        self.flag = torch.zeros(1, dtype=torch.int32).pin_memory()
        self.flag_addr = self.flag.data_ptr()
        self.stamps = None
        stale, self.stale_qkvg, self.qkvg = _stale_inputs(inp)
        self.work = _clone(stale)
        self.work["quantize_output"] = True
        self.gate_src = inp["g"].clone()
        self.beta_src = inp["beta"].contiguous().clone()
        self.snapshot = _clone(self.work)
        self.aux = torch.cuda.Stream()
        self.fork, self.join = torch.cuda.Event(), torch.cuda.Event()
        self.result = None
        self.graph = None
        # compile and warm up (outside any capture, on a side stream) with the producers released
        self.flag.fill_(1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._enqueue()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if graph:
            self.graph = (
                torch.cuda.CUDAGraph(keep_graph=True) if keep_graph else torch.cuda.CUDAGraph()
            )
            with torch.cuda.graph(self.graph):
                self.result = self._enqueue()

    def _enqueue(self):
        _spin_copy_into(self.flag_addr, self.qkvg, self.stale_qkvg)
        if self.graph_mode:
            self.fork.record()
            with torch.cuda.stream(self.aux):
                self.fork.wait()
                _spin_copy_into(self.flag_addr, self.gate_src, self.work["g"])
                _spin_copy_into(self.flag_addr, self.beta_src, self.work["beta"])
                self.join.record()
            self.join.wait()
        else:
            _spin_copy_into(self.flag_addr, self.gate_src, self.work["g"])
            _spin_copy_into(self.flag_addr, self.beta_src, self.work["beta"])
        return self.consumer(self)

    def run(self, release_after_s=_RELEASE_AFTER_S):
        """Reset the pools and the stale outputs, start the chain, release it after a host delay."""
        for name in MUTATED:
            self.work[name].copy_(self.snapshot[name])
        self.stale_qkvg.normal_()
        self.work["g"].normal_()
        self.work["beta"].normal_()
        self.flag.zero_()
        if self.stamps is not None:
            self.stamps.zero_()
        torch.cuda.synchronize()
        if self.graph is not None:
            self.graph.replay()
        else:
            self.result = self._enqueue()
        time.sleep(release_after_s)
        self.flag.fill_(1)
        torch.cuda.synchronize()
        return self.result


def _verify_consumer(rig):
    return kda_ops.kda_mtp_decode_impl(**rig.work)


def _probe_consumer(grid):
    stamps = torch.zeros(grid, 2, dtype=torch.int32, device="cuda")

    def consumer(rig):
        rig.stamps = stamps
        _hazard_kernels()["probe"][(grid,)](rig.flag_addr, stamps, num_warps=16, launch_pdl=True)

    return consumer


def _edge_kinds_into_sinks(graph):
    """Edge types (0 default, 1 programmatic) of the edges into the sink nodes of a graph that was
    captured with ``keep_graph=True``; None where the driver bindings cannot list them."""
    try:
        from cuda.bindings import driver as drv

        g = drv.CUgraph(graph.raw_cuda_graph())
        n_edges = drv.cuGraphGetEdges(g, 0)[-1]
        res = drv.cuGraphGetEdges(g, n_edges)
        assert int(res[0]) == 0
        src, dst, data = res[1:4]
    except (ImportError, AttributeError, AssertionError, ValueError):
        return None
    sources = {int(a) for a in src}
    return [int(d.type) for b, d in zip(dst, data) if int(b) not in sources]


@requires_device
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("N,H", [(2, 12), (16, 12), (40, 12)], ids=["n2", "n16", "n40"])
def test_pdl_consumer_overlaps_its_producers(N, H, graph):
    """The hazard tests below only mean something if a PDL launch of a 512-thread kernel starts
    before its producers have finished, in this arrangement of producers."""
    inp = _make_inputs(N, H, num_spec=5, seed=3 + N)
    grid = min(2 * H * N, 2 * torch.cuda.get_device_properties(0).multi_processor_count)
    rig = _Rig(inp, _probe_consumer(grid), graph, keep_graph=True)
    if graph:
        kinds = _edge_kinds_into_sinks(rig.graph)
        assert kinds is None or (kinds and set(kinds) == {1}), (
            f"edge types into the consumer (0 default, 1 programmatic): {kinds}"
        )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    for _ in range(3):
        rig.run()
        at_entry, after_wait = rig.stamps[:, 0], rig.stamps[:, 1]
        early = int((at_entry == 0).sum())  # entered before the host's release
        assert early >= min(grid, num_sms // 2), (
            f"only {early} of {grid} probe CTAs entered before the release: the chain does not "
            "overlap, a verify without its wait would pass the hazard tests"
        )
        assert bool((after_wait == 1).all()), "a probe's wait returned before the release"


def _reference_round(inp, monkeypatch):
    """One verify round with the default launch (output and pools)."""
    _set_pdl(monkeypatch, False)
    ref = _clone(inp)
    ref["quantize_output"] = True
    out = _clone_out(kda_ops.kda_mtp_decode_impl(**ref))
    torch.cuda.synchronize()
    return out, ref


@requires_device
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("N,H,compact", LAYOUT_SHAPES)
def test_pdl_waits_for_late_producers(N, H, compact, graph, monkeypatch, key_log):
    """Every producer output of the verify is written only after the host's release, well after the
    verify CTAs have started; the verify must still read the final values."""
    _skip_wrong_layout(compact, N, H)
    inp = _make_inputs(N, H, num_spec=5, seed=2024 + N)
    _set_compact(monkeypatch, compact)
    ref_out, ref_x = _reference_round(inp, monkeypatch)
    key_log.used = []
    _set_pdl(monkeypatch, True)
    rig = _Rig(inp, _verify_consumer, graph)
    for run in range(3):
        out = rig.run()
        _assert_out_equal(f"run {run} output", _clone_out(out), ref_out)
        _assert_pools_equal(rig.work, ref_x)
    assert set(_pdl_flags_of(key_log)) == {True}, "the PDL variant did not run"
