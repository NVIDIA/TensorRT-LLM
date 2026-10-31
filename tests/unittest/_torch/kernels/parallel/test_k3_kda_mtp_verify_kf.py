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
"""``trtllm::kimi_k3_kda_mtp_verify_kf`` against ``trtllm::kda_mtp_decode_fp8``.

The Kernel Factory verify kernel (BF16 state, MXFP8 output) sums in a different order and uses fast-math
transcendentals, so it is not bit-identical to the CuTe DSL op. On deployed layouts (section-strided projection rows,
beta a slice of [f_a | b], a state pool larger than the batch with the padded slot stride of the hybrid cache and random
slots, padded beta-cache rows, and the replay buffers and accepted counts of KV cache manager v2, which keeps them in
per-slot blocks, or the dense tensors of the Python manager) every call must pass the Kernel Factory package's checker
with the production op as the reference: MXFP8 scales replicated and at most one exponent step apart on a few blocks,
codes at most one step apart on a few elements, the state within two BF16 ulps with few differing elements and no
magnitude bias (round to nearest even), the replay caches likewise, conv windows bit-identical, and every other byte of
the in-place buffers untouched: slots outside the batch, the padding between slots and poisoned guard bytes around each
buffer. This also holds for batches large enough that the kernel runs more than one wave of CTAs. Over chained rounds
the two ops drift apart only by rounding, either op continues from the other's state and caches, requests that share a
padding slot do not disturb the others, CUDA graph replay (with and without PDL) reproduces the eager bytes, unsupported
inputs are rejected, and the dispatch routes only eligible calls, only when enabled, keeping the others (route, dtype,
layout) on the CuTe DSL kernels. Built the way the engine builds them (KimiKDALinearAttention.forward_verify with its
BF16 A_log / dt_bias parameters, on a real KDAHybridCacheManagerV2), the verify calls take the Kernel Factory route and
pass the same checker against the CuTe DSL op, without touching other slots or layers.

In images whose libraries bring their own C++ exception-handling runtime (seen with cuRAND in the Kimi K3 images), a
TORCH_CHECK failing inside a trtllm op aborts the process instead of raising RuntimeError, which ends the session at
test_rejects_unsupported_inputs; preload the system libstdc++ and libgcc_s there (LD_PRELOAD).
"""

import pytest
import torch

_HAVE_DEPS = True
_DEP_ERR = None
try:  # the reference op (trtllm::kda_mtp_decode_fp8) is a CuTe DSL kernel
    import cuda.bindings.driver  # noqa: F401
    import cutlass  # noqa: F401
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

BF16 = torch.bfloat16
MUTATED = ("cs_q", "cs_k", "cs_v", "recurrent_state", "k_cache", "g_cache", "v_cache", "beta_cache")
NEW_TOKEN_INPUTS = ("x_q", "x_k", "x_v", "onorm_g", "g", "beta", "num_accepted_tokens")
# (heads per rank, num_spec, requests): attention-DP and TP8 ranks, 3 and 5 drafts, one request to a few waves
CASES = [(96, 3, 6), (96, 3, 64), (96, 5, 33), (12, 3, 1), (12, 3, 24), (12, 5, 8)]

# The Kernel Factory package's checker (2-5x the production kernel's error against an FP32 reference), applied to the
# requests' slots only (the package divides by the whole pool).
SCALE_FRAC, SCALE_FLOOR = 0.002, 3
CODE_FRAC, CODE_FLOOR = 0.01, 128
FAR_FRAC, FAR_FLOOR = 0.0015, 16
BLOCK_ERR, OUT_L2 = 0.3, 0.01
STATE_SOFT_FRAC, STATE_DIFF_FRAC, STATE_BIAS = 0.001, 0.02, 1e-4
CACHE_TOL = {
    "k_cache": (2e-2, 4e-3),
    "v_cache": (2e-2, 4e-3),
    "beta_cache": (2e-2, 4e-3),
    "g_cache": (1e-5, 1e-5),
}
CACHE_DIFF_FRAC, CACHE_BIAS = 0.01, 1e-4
# Poisoned bytes before and after every in-place buffer (the padding between its slots is poisoned too).
GUARD_BYTES = 4096
_INT_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401

    if not hasattr(torch.ops.trtllm, "kimi_k3_kda_mtp_verify_kf"):
        pytest.skip("trtllm::kimi_k3_kda_mtp_verify_kf is not built")


def _in_place(values, stride=None):
    """values in a view with the given strides (default: their own) of a buffer whose other bytes are NaN, with
    GUARD_BYTES more NaN bytes before and after it."""
    stride = values.stride() if stride is None else stride
    guard = GUARD_BYTES // values.element_size()
    extent = 1 + sum((n - 1) * st for n, st in zip(values.shape, stride))
    buf = torch.empty(2 * guard + extent, dtype=values.dtype, device=values.device)
    buf.view(torch.uint8).fill_(0xFF)
    view = buf.as_strided(values.shape, stride, guard)
    view.copy_(values)
    return view


def _storage_bytes(t):
    """The whole storage under t (padding and guards included) as a flat uint8 tensor."""
    storage = t.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _counts(values, num_spec, kvcm_v2):
    """Accepted counts by slot. KV cache manager v2 keeps each slot's count in a 16-byte block (stride 4); the rest of
    each block holds a different valid count, so a kernel that reads a neighbour gets a wrong one."""
    if not kvcm_v2:
        return values
    blocks = ((values + 1) % (num_spec + 1)).unsqueeze(1).repeat(1, 4)
    blocks[:, 0] = values
    return blocks[:, 0]


def _case(heads, num_spec, batch, seed, pool=None, kvcm_v2=True):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = heads * 128
    rows = batch * (num_spec + 1)
    window = 3 + num_spec
    pool = batch + 19 if pool is None else pool
    # KV cache manager v2 (deployed) keeps each slot's state, conv state and replay buffers in per-slot blocks: slot
    # strides beyond the dense ones (here: the state padded, the replay buffers twice the dense stride). The Python
    # manager (MixedMambaHybridCacheManager) allocates dense per-layer tensors.
    spread = 2 if kvcm_v2 else 1

    def rn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(dtype)

    # Mixer layouts: q | k | v | output-gate sections of one projection; beta a slice of [f_a | b], whose rows the
    # projection pads to 16 bytes.
    qkvg = rn(rows, 4 * dim, dtype=BF16)
    bfa = rn(rows, (128 + heads + 7) // 8 * 8, dtype=BF16)
    padded = (heads + 7) // 8 * 8  # the cache manager pads beta-cache rows to 16 bytes
    beta_cache = _in_place(
        torch.sigmoid(rn(pool, num_spec, heads)).to(BF16), (spread * num_spec * padded, padded, 1)
    )
    slot_stride = heads * (128 * 128 + (3 * 128 * window if kvcm_v2 else 0))
    state = _in_place(
        rn(pool, heads, 128, 128, scale=0.5, dtype=BF16), (slot_stride, 128 * 128, 128, 1)
    )
    k_cache = rn(pool, num_spec, heads, 128)
    conv_strides = (spread * window * dim, 1, dim)  # dim-contiguous [pool, dim, window]
    cache_strides = (spread * num_spec * dim, dim, 1)
    return {
        "x_q": qkvg[:, :dim].view(1, rows, heads, 128),
        "x_k": qkvg[:, dim : 2 * dim].view(1, rows, heads, 128),
        "x_v": qkvg[:, 2 * dim : 3 * dim].view(1, rows, heads, 128),
        "onorm_g": qkvg[:, 3 * dim :].view(1, rows, heads, 128),
        "g": rn(rows, dim, dtype=BF16).view(1, rows, heads, 128),
        "beta": bfa[:, 128 : 128 + heads].view(1, rows, heads),
        "w_q": rn(dim, 4, scale=0.5),
        "w_k": rn(dim, 4, scale=0.5),
        "w_v": rn(dim, 4, scale=0.5),
        "A_log": rn(heads, scale=0.5),
        "dt_bias": rn(dim, scale=0.5),
        "onorm_weight": torch.linspace(0.75, 1.25, 128, device="cuda"),
        "ssm_state_indices": torch.randperm(pool, generator=gen, device="cuda")[:batch].int(),
        "num_accepted_tokens": _counts(
            torch.randint(
                0, num_spec + 1, (pool,), generator=gen, device="cuda", dtype=torch.int32
            ),
            num_spec,
            kvcm_v2,
        ),
        "cs_q": _in_place(rn(pool, window, dim, dtype=BF16).transpose(1, 2), conv_strides),
        "cs_k": _in_place(rn(pool, window, dim, dtype=BF16).transpose(1, 2), conv_strides),
        "cs_v": _in_place(rn(pool, window, dim, dtype=BF16).transpose(1, 2), conv_strides),
        "recurrent_state": state,
        "k_cache": _in_place(
            (k_cache / k_cache.norm(dim=-1, keepdim=True)).reshape(pool, num_spec, dim).to(BF16),
            cache_strides,
        ),
        "g_cache": _in_place(-5.0 * torch.sigmoid(rn(pool, num_spec, dim)), cache_strides),
        "v_cache": _in_place(
            torch.nn.functional.silu(rn(pool, num_spec, dim)).to(BF16), cache_strides
        ),
        "beta_cache": beta_cache,
        "num_spec": num_spec,
    }


def _clone(case):
    """The case with copies of the in-place tensors over copies of their whole buffers (padding and guards)."""
    out = dict(case)
    for name in MUTATED:
        src = case[name]
        out[name] = (
            _storage_bytes(src)
            .clone()
            .view(src.dtype)
            .as_strided(src.shape, src.stride(), src.storage_offset())
        )
    return out


def _check_untouched(before, after, slots):
    """Every byte of the in-place buffers outside the batch slots' elements is unchanged (bitwise; NaN poison)."""
    index = slots.long()
    for name in MUTATED:
        t = after[name]
        size = t.element_size()
        owned = torch.zeros(t.untyped_storage().nbytes() // size, dtype=torch.bool, device="cuda")
        owned.as_strided(t.shape, t.stride(), t.storage_offset())[index] = True
        old = _storage_bytes(before[name]).view(_INT_OF_SIZE[size])[~owned]
        new = _storage_bytes(t).view(_INT_OF_SIZE[size])[~owned]
        changed = int((old != new).sum())
        assert changed == 0, f"{name}: {changed} elements outside the batch slots changed"


def _next_round(cases, seed):
    """Fresh new-token inputs and accepted counts for every chain (same values in all)."""
    ref = cases[0]
    fresh = _case(
        ref["A_log"].shape[0],
        ref["num_spec"],
        ref["ssm_state_indices"].shape[0],
        seed,
        pool=ref["recurrent_state"].shape[0],
    )
    for case in cases:
        for name in NEW_TOKEN_INPUTS:
            case[name] = fresh[name]


def _resync(dst, src):
    for name in MUTATED:
        dst[name].copy_(src[name])


def _reference(case):
    kwargs = {k: v for k, v in case.items() if k != "num_spec"}
    kwargs.update(
        num_spec=case["num_spec"],
        cu_seqlens=None,
        packed_token_layout=True,
        lower_bound=-5.0,
        scale=128**-0.5,
        onorm_eps=1e-5,
    )
    return torch.ops.trtllm.kda_mtp_decode_fp8(**kwargs)


def _kf(case, enable_pdl=False, outputs=None, lower_bound=-5.0, bf16_output=False):
    rows, heads = case["x_q"].shape[1], case["A_log"].shape[0]
    if outputs is None:
        if bf16_output:
            outputs = (torch.empty(1, rows, heads, 128, dtype=BF16, device="cuda"), None)
        else:  # poisoned: every byte the kernel owns must be written
            scale_bytes = (rows + 127) // 128 * 128 * heads * 4
            outputs = (
                torch.full((rows, heads * 128), 0xA5, dtype=torch.uint8, device="cuda").view(
                    torch.float8_e4m3fn
                ),
                torch.full((scale_bytes,), 0xA5, dtype=torch.uint8, device="cuda"),
            )
    torch.ops.trtllm.kimi_k3_kda_mtp_verify_kf(
        case["x_q"],
        case["x_k"],
        case["x_v"],
        case["g"],
        case["beta"],
        case["onorm_g"],
        case["w_q"],
        case["w_k"],
        case["w_v"],
        case["A_log"],
        case["dt_bias"],
        case["onorm_weight"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        case["cs_q"],
        case["cs_k"],
        case["cs_v"],
        case["recurrent_state"],
        case["k_cache"],
        case["g_cache"],
        case["v_cache"],
        case["beta_cache"],
        outputs[0],
        outputs[1],
        case["num_spec"],
        lower_bound,
        128**-0.5,
        1e-5,
        enable_pdl,
    )
    return outputs


def _scale_offsets(rows, heads):
    """Byte offset of (row, K32 slot) in the R128c4 scale layout [ceil(rows / 128), heads, 32, 4, 4]."""
    r = torch.arange(rows, device="cuda").view(-1, 1)
    c = torch.arange(heads * 4, device="cuda").view(1, -1)
    return ((((r // 128) * heads + c // 4) * 32 + r % 32) * 4 + (r % 128) // 32) * 4 + c % 4


def _dequant(q, s, rows, heads):
    slots = s.view(torch.uint8)[_scale_offsets(rows, heads).flatten()].view(rows, heads, 4)
    e8 = slots[..., 0].int()
    deq = q.view(torch.uint8).view(rows, heads, 128).view(torch.float8_e4m3fn).float()
    return deq * torch.exp2((e8 - 127).float()).unsqueeze(-1), e8, slots


def _ordered_codes(q, rows, heads):
    """Monotonic integer key of the E4M3 bytes (both zeros map to 0)."""
    b = q.view(torch.uint8).view(rows, heads, 128).int()
    mag = b & 0x7F
    return torch.where((b & 0x80) != 0, -mag, mag)


def _check_out(ref, kf, rows, heads, rows_used=None, strict=True):
    dq_r, e_r, _ = _dequant(*ref, rows, heads)
    dq_u, e_u, slots = _dequant(*kf, rows, heads)
    k_r, k_u = _ordered_codes(ref[0], rows, heads), _ordered_codes(kf[0], rows, heads)
    if rows_used is not None:
        dq_r, e_r, dq_u, e_u, slots, k_r, k_u = (
            t[rows_used] for t in (dq_r, e_r, dq_u, e_u, slots, k_r, k_u)
        )
    assert bool((slots == slots[..., :1]).all()), "the four K32 scale slots of a block differ"
    assert bool(torch.isfinite(dq_u).all()), "non-finite output"
    assert int((e_r - e_u).abs().max()) <= 1, "a block scale is more than one exponent step off"
    blk = ((dq_u - dq_r).abs().amax(-1) / dq_r.abs().amax(-1).clamp_min(1e-30)).max().item()
    assert blk <= BLOCK_ERR, f"an element is {blk:.3f} of its block amax off"
    l2 = ((dq_u - dq_r).square().sum().sqrt() / dq_r.square().sum().sqrt().clamp_min(1e-30)).item()
    assert l2 <= OUT_L2, f"dequantized relative L2 error {l2:.2e}"
    if not strict:
        return
    mism = e_r != e_u
    blocks = e_r.numel()
    assert int(mism.sum()) <= max(SCALE_FLOOR, SCALE_FRAC * blocks), (
        f"{int(mism.sum())} of {blocks} block scales differ"
    )
    same = (~mism).unsqueeze(-1).expand_as(k_r)
    n_same = max(int(same.sum()), 1)
    code_diff = (k_r != k_u) & same
    far = code_diff & ((k_r - k_u).abs() > 1)
    assert int(code_diff.sum()) <= max(CODE_FLOOR, CODE_FRAC * n_same), (
        f"{int(code_diff.sum())} of {n_same} codes differ"
    )
    assert int(far.sum()) <= max(FAR_FLOOR, FAR_FRAC * n_same), (
        f"{int(far.sum())} codes differ by more than one step"
    )


def _rne_stats(r, u):
    """(fraction of elements that differ, signed magnitude bias sum(|u| - |r|) / sum(|r|))."""
    r, u = r.double(), u.double()
    diff = (r != u).double().mean().item()
    bias = ((u.abs() - r.abs()).sum() / r.abs().sum().clamp_min(1e-30)).item()
    return diff, bias


def _check_mutated(ref, kf, slots, check_slots=None, strict=True):
    used = torch.zeros(ref["recurrent_state"].shape[0], dtype=torch.bool, device="cuda")
    used[slots.long()] = True
    check = used if check_slots is None else check_slots
    for name in MUTATED:
        a, b = kf[name], ref[name]
        assert torch.equal(a[~used], b[~used]), f"{name}: slots outside the batch changed"
        a, b = a[check].float(), b[check].float()
        assert bool(torch.isfinite(a).all()), f"{name}: non-finite values"
        err = (a - b).abs()
        if name.startswith("cs_"):
            assert torch.equal(a, b), f"{name}: conv windows must be bit-identical"
        elif name == "recurrent_state":
            # Two BF16 ulps also bound the rounding that chained rounds carry over (strict=False).
            assert not bool((err > 2.0**-6 * b.abs() + 5e-4).any()), (
                "state off by more than two BF16 ulps"
            )
            if not strict:
                continue
            soft = (err > 2.0**-7 * b.abs() + 5e-4).double().mean().item()
            diff, bias = _rne_stats(b, a)
            assert soft <= STATE_SOFT_FRAC, f"{soft:.2e} of the state beyond one ulp"
            assert diff <= STATE_DIFF_FRAC, f"{diff:.2e} of the state differs"
            assert abs(bias) <= STATE_BIAS, (
                f"state magnitude bias {bias:.2e} (rounding toward zero?)"
            )
        else:  # the caches depend on the new tokens and the conv windows only, never on the state
            atol, rtol = CACHE_TOL[name]
            assert not bool((err > atol + rtol * b.abs()).any()), (
                f"{name}: max |diff| {err.max().item():.3e}"
            )
            if name != "g_cache":
                diff, bias = _rne_stats(b, a)
                assert diff <= CACHE_DIFF_FRAC and abs(bias) <= CACHE_BIAS, (
                    f"{name}: {diff:.2e} differ, bias {bias:.2e}"
                )


@pytest.mark.parametrize("heads,num_spec,batch", CASES)
def test_each_call_passes_the_checker(heads, num_spec, batch):
    """Three rounds, each starting both ops from the reference's state and caches of the previous round."""
    base = _case(heads, num_spec, batch, seed=heads * 1000 + num_spec * 100 + batch)
    ref, kf = _clone(base), _clone(base)
    rows = batch * (num_spec + 1)
    for rnd in range(3):
        if rnd:
            _next_round([ref, kf], seed=1000 * heads + 10 * batch + rnd)
            _resync(kf, ref)
        before = _clone(kf)
        ref_out = _reference(ref)
        kf_out = _kf(kf)
        torch.cuda.synchronize()
        _check_out(ref_out, kf_out, rows, heads)
        _check_mutated(ref, kf, base["ssm_state_indices"])
        _check_untouched(before, kf, base["ssm_state_indices"])


@pytest.mark.parametrize(
    "heads,num_spec,waves,extra",
    [
        (96, 3, 1, 0),
        (96, 3, 1, 1),
        (96, 3, 1, 5),
        (96, 3, 2, 0),
        (96, 3, 2, 1),
        (96, 5, 1, 0),
        (96, 5, 1, 1),
        (96, 5, 1, 5),
        (12, 3, 1, 0),
        (12, 3, 1, 1),
        (12, 3, 1, 5),
    ],
)
def test_wave_edges(heads, num_spec, waves, extra):
    """Around B * H = waves * 2 * SMs * 64: the last batch of a wave count runs CTAs with up to 64 items (a full
    schedule table), the next batches one more wave of equally loaded CTAs."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    batch = waves * 128 * sms // heads + extra
    base = _case(heads, num_spec, batch, seed=heads * 100 + num_spec * 10 + waves + extra)
    ref, kf = _clone(base), _clone(base)
    before = _clone(kf)
    ref_out = _reference(ref)
    kf_out = _kf(kf)
    torch.cuda.synchronize()
    _check_out(ref_out, kf_out, batch * (num_spec + 1), heads)
    _check_mutated(ref, kf, base["ssm_state_indices"])
    _check_untouched(before, kf, base["ssm_state_indices"])


@pytest.mark.parametrize("heads,num_spec,batch", [(96, 3, 64), (12, 3, 24)])
def test_dense_cache_layout(heads, num_spec, batch):
    """The Python cache manager's layout: dense per-layer state and replay tensors, contiguous accepted counts."""
    base = _case(heads, num_spec, batch, seed=heads + 10 * batch, kvcm_v2=False)
    ref, kf = _clone(base), _clone(base)
    before = _clone(kf)
    ref_out = _reference(ref)
    kf_out = _kf(kf)
    torch.cuda.synchronize()
    _check_out(ref_out, kf_out, batch * (num_spec + 1), heads)
    _check_mutated(ref, kf, base["ssm_state_indices"])
    _check_untouched(before, kf, base["ssm_state_indices"])


@pytest.mark.parametrize("heads,num_spec,batch", [(96, 3, 64), (96, 5, 33), (12, 3, 24)])
def test_chained_rounds_drift_by_rounding_only(heads, num_spec, batch):
    """Three rounds on each op's own state: the chains diverge only by the rounding the recurrence carries over."""
    base = _case(heads, num_spec, batch, seed=7 * heads + num_spec + batch)
    ref, kf = _clone(base), _clone(base)
    rows = batch * (num_spec + 1)
    for rnd in range(3):
        if rnd:
            _next_round([ref, kf], seed=2000 * heads + 10 * batch + rnd)
        ref_out = _reference(ref)
        kf_out = _kf(kf)
        torch.cuda.synchronize()
        _check_out(ref_out, kf_out, rows, heads, strict=rnd == 0)
        _check_mutated(ref, kf, base["ssm_state_indices"], strict=rnd == 0)


@pytest.mark.parametrize("first", ["kf", "reference"])
def test_rounds_continue_across_ops(first):
    """Either op continues from the conv windows, state and replay caches the other one left."""
    base = _case(96, 3, 16, seed=7)
    ref, mixed = _clone(base), _clone(base)
    _reference(ref)
    _kf(mixed) if first == "kf" else _reference(mixed)
    _next_round([ref, mixed], seed=8)
    ref_out = _reference(ref)
    mixed_out = _reference(mixed) if first == "kf" else _kf(mixed)
    torch.cuda.synchronize()
    # After a reference first round both chains start the second from the same state: the per-call checker holds.
    strict = first == "reference"
    _check_out(ref_out, mixed_out, 64, 96, strict=strict)
    _check_mutated(ref, mixed, base["ssm_state_indices"], strict=strict)


def test_padding_requests_share_a_slot():
    """CUDA-graph padding requests all use one slot; the real requests must not notice."""
    base = _case(96, 3, 24, seed=9, pool=40)
    slots = base["ssm_state_indices"].clone()
    slots[16:] = slots[
        15
    ]  # requests 15..23 share request 15's slot: requests 0..14 are the real ones
    base["ssm_state_indices"] = slots
    ref, kf = _clone(base), _clone(base)
    ref_out = _reference(ref)
    kf_out = _kf(kf)
    torch.cuda.synchronize()
    _check_out(ref_out, kf_out, 96, 96, rows_used=torch.arange(15 * 4, device="cuda"))
    real = torch.zeros(40, dtype=torch.bool, device="cuda")
    real[slots[:15].long()] = True
    _check_mutated(ref, kf, slots, check_slots=real)


def test_deterministic_graph_replay_and_pdl():
    base = _case(96, 3, 48, seed=13)
    eager = _clone(base)
    eager_out = _kf(eager)
    offsets = _scale_offsets(48 * 4, 96).flatten()
    for enable_pdl in (False, True):
        graphed = _clone(base)
        outputs = (torch.empty_like(eager_out[0]), torch.empty_like(eager_out[1]))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _kf(graphed, enable_pdl, outputs)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[0].view(torch.uint8), eager_out[0].view(torch.uint8))
        assert torch.equal(outputs[1][offsets], eager_out[1][offsets])
        for name in MUTATED:
            assert torch.equal(graphed[name], eager[name]), name


def test_rejects_unsupported_inputs():
    base = _case(96, 3, 8, seed=17)
    with pytest.raises(RuntimeError):
        _kf(
            dict(base, num_spec=2)
        )  # T is no longer B * (num_spec + 1), and num_spec 2 is unsupported
    with pytest.raises(RuntimeError):
        _kf(dict(base, g_cache=base["g_cache"].to(BF16)))  # g_cache must be FP32
    with pytest.raises(RuntimeError):
        _kf(dict(base, recurrent_state=base["recurrent_state"].float()))  # BF16 state only
    with pytest.raises(RuntimeError):
        _kf(_clone(base), bf16_output=True)  # MXFP8 output only
    with pytest.raises(RuntimeError):
        _kf(_clone(base), lower_bound=-4.0)  # the kernel bakes in the Kimi K3 constants
    rows = base["x_q"].shape[1]
    wide = torch.zeros(rows, 96 * 128 + 4, dtype=BF16, device="cuda")
    misaligned = wide[:, 4:].view(1, rows, 96, 128)  # rows start 8 bytes off a 16-byte boundary
    misaligned.copy_(base["x_q"])
    with pytest.raises(RuntimeError):
        _kf(dict(base, x_q=misaligned))
    # beta and the beta cache are read through tensor maps with 8-head boxes: their rows must be padded to a multiple
    # of 8 heads (16 bytes), as the [f_a | b] projection and the cache manager do.
    narrow = _case(12, 3, 8, seed=18)
    rows12 = narrow["x_q"].shape[1]
    unpadded = torch.empty(rows12, 12, dtype=BF16, device="cuda")
    unpadded.copy_(narrow["beta"].view(rows12, 12))
    with pytest.raises(RuntimeError):
        _kf(dict(narrow, beta=unpadded.view(1, rows12, 12)))  # row stride 12
    pool12 = narrow["beta_cache"].shape[0]
    tight_cache = _in_place(narrow["beta_cache"][..., :12].contiguous(), (3 * 12, 12, 1))
    assert tight_cache.shape == (pool12, 3, 12)
    with pytest.raises(RuntimeError):
        _kf(dict(narrow, beta_cache=tight_cache))  # row stride 12
    shifted = torch.zeros(rows12 * 144 + 8, dtype=BF16, device="cuda")[1 : 1 + rows12 * 144].view(
        rows12, 144
    )
    shifted[:, 128:140].copy_(narrow["beta"].view(rows12, 12))
    with pytest.raises(RuntimeError):
        _kf(
            dict(narrow, beta=shifted[:, 128:140].unsqueeze(0))
        )  # row stride 144, base 2 bytes off a 16-byte boundary


def _dispatch_module():
    from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels

    if not hasattr(_kda_kernels, "_kf_verify_ineligibility"):
        pytest.skip("dispatch module without the Kernel Factory verify path")
    return _kda_kernels


def _dispatch_kwargs(case):
    kwargs = {k: v for k, v in case.items() if k != "num_spec"}
    kwargs.update(
        num_spec=case["num_spec"],
        cu_seqlens=None,
        packed_token_layout=True,
        lower_bound=-5.0,
        scale=128**-0.5,
        onorm_eps=1e-5,
        fuse_output_norm=True,
    )
    return kwargs


def test_eligibility_policy(monkeypatch):
    kk = _dispatch_module()

    def reason(heads, num_spec, mxfp8=True, **changes):
        kwargs = dict(_dispatch_kwargs(_case(heads, num_spec, 4, seed=1)), **changes)
        return kk._kf_verify_ineligibility(kwargs, mxfp8)

    assert reason(96, 3) is None
    assert reason(96, 5) is None
    # TP8 ranks (12 heads) stay on the CuTe DSL kernels, whose PDL variant is as fast there.
    assert reason(12, 3) == "12 heads with num_spec 3"
    assert reason(12, 5) == "12 heads with num_spec 5"
    # beta and the beta cache are read through 8-head tensor-map boxes: unpadded rows stay on the CuTe DSL kernels. The
    # 12-head layouts are the ones that can be unpadded (96 heads are a multiple of 8): check them with the route open.
    monkeypatch.setattr(kk, "_KF_VERIFY_ROUTES", kk._KF_VERIFY_ROUTES | {(12, 3)})
    assert reason(12, 3) is None
    heads12 = _case(12, 3, 4, seed=1)
    rows = heads12["x_q"].shape[1]
    assert reason(12, 3, beta=heads12["beta"].reshape(1, rows, 12).clone()) == "beta alignment"
    wide = heads12["beta_cache"]
    tight = _in_place(wide[..., :12].contiguous(), (3 * 12, 12, 1))
    assert reason(12, 3, beta_cache=tight) == "beta_cache alignment"
    assert (
        reason(96, 3, beta=_case(96, 3, 4, seed=1)["beta"].reshape(1, -1, 96).clone()) is None
    )  # stride 96 = 12 x 8
    assert reason(96, 3, mxfp8=False) == "BF16 output"
    assert reason(96, 3, onorm_eps=1e-6) == "constants"
    assert reason(96, 3, packed_token_layout=False) is not None
    state = _case(96, 3, 4, seed=1)["recurrent_state"]
    assert reason(96, 3, recurrent_state=state.float()) == "dtype"
    # The engine keeps A_log and dt_bias as BF16 parameters: those calls need the mixer's FP32 copies.
    params = _case(96, 3, 4, seed=1)
    a_bf16, dt_bf16 = params["A_log"].to(BF16), params["dt_bias"].to(BF16)
    assert reason(96, 3, A_log=a_bf16) == "A_log is torch.bfloat16, not torch.float32"
    assert reason(96, 3, dt_bias=dt_bf16) == "dt_bias is torch.bfloat16, not torch.float32"
    engine = dict(_dispatch_kwargs(params), A_log=a_bf16, dt_bias=dt_bf16)
    with_copies = kk._with_fp32_gate_constants(engine, a_bf16.float(), dt_bf16.float())
    assert kk._kf_verify_ineligibility(with_copies, True) is None
    assert with_copies["A_log"].dtype == torch.float32 and engine["A_log"] is a_bf16
    assert kk._with_fp32_gate_constants(engine, None, None) is engine
    assert kk._with_fp32_gate_constants(engine, a_bf16[:8].float(), dt_bf16.float()) is engine
    # Layouts the CuTe DSL kernels may take but the Kernel Factory op rejects stay on the CuTe DSL kernels.
    assert reason(96, 3, recurrent_state=state.transpose(2, 3)) == "layout"
    slots = _case(96, 3, 4, seed=1)["ssm_state_indices"]
    strided = torch.zeros(2 * slots.numel(), dtype=torch.int32, device="cuda")[::2]
    strided.copy_(slots)
    assert reason(96, 3, ssm_state_indices=strided) == "layout"


@pytest.mark.parametrize("heads,num_spec", [(96, 3), (12, 3), (12, 5)])
@pytest.mark.parametrize("enabled", [False, True])
def test_dispatch_routes_only_when_enabled(enabled, heads, num_spec, monkeypatch):
    kk = _dispatch_module()
    monkeypatch.setenv("TRTLLM_K3_KDA_VERIFY_KF", "1" if enabled else "0")
    dispatch = kk.KDAKernelDispatch()
    if dispatch.verify_kernel_path != "optimized":
        pytest.skip("optimized verify path unavailable")
    assert dispatch._kf_verify is enabled
    base = _case(heads, num_spec, 16, seed=21)
    via_dispatch = _clone(base)
    out, scale = dispatch.mtp_verify(mxfp8_output=True, **_dispatch_kwargs(via_dispatch))
    direct = _clone(base)
    routed = enabled and (heads, num_spec) in kk._KF_VERIFY_ROUTES
    expected = _kf(direct) if routed else _reference(direct)
    torch.cuda.synchronize()
    offsets = _scale_offsets(16 * (num_spec + 1), heads).flatten()
    assert torch.equal(out.view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(scale[offsets], expected[1][offsets])
    for name in MUTATED:
        assert torch.equal(via_dispatch[name], direct[name]), name


def test_switch_parsing(monkeypatch):
    kk = _dispatch_module()
    monkeypatch.delenv("TRTLLM_K3_KDA_VERIFY_KF", raising=False)
    assert kk.KDAKernelDispatch()._kf_verify is False  # default off
    monkeypatch.setenv("TRTLLM_K3_KDA_VERIFY_KF", "yes")
    with pytest.raises(ValueError):
        kk.KDAKernelDispatch()


# The verify call as the engine builds it: KimiKDALinearAttention.forward_verify on a real KDAHybridCacheManagerV2.
ENGINE_BUFFERS = {
    "cs_q": "kda_conv_q",
    "cs_k": "kda_conv_k",
    "cs_v": "kda_conv_v",
    "recurrent_state": "temporal",
    "k_cache": "kda_k_cache",
    "g_cache": "kda_g_cache",
    "v_cache": "kda_v_cache",
    "beta_cache": "kda_beta_cache",
}


class _PrequantizedProjection(torch.nn.Module):
    """o_proj stand-in for the resident MXFP8 projection of SM107: returns the verify kernel's codes and scales."""

    supports_prequantized_input = True

    def forward_prequantized(self, activation, activation_scale):
        return activation, activation_scale


def _engine_buffers(layer_cache):
    return {name: getattr(layer_cache, attr) for name, attr in ENGINE_BUFFERS.items()}


def _engine_setup(heads, num_spec, batch, seed, num_layers=3, max_batch=None):
    """(runtime, manager, slots, hidden rows) for KimiKDALinearAttention.forward_verify as the engine runs it.

    The mixer keeps A_log / dt_bias as BF16 parameters, projects q | k | v | output gate as sections of one GEMM
    output (the kernel reads strided views at offsets 0, D, 2D, 3D) and beta as a slice of [f_a | b] (row stride
    ceil8(128 + H), offset 128). The state, conv windows, replay caches and accepted counts are the views of a
    KDAHybridCacheManagerV2 with num_layers KDA layers: per-slot blocks, a BF16 state, counts at stride 4. o_proj is the
    stand-in above, so a call returns the MXFP8 codes and scales. Every slot of every layer holds finite values.
    """
    try:
        from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
        from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import KDAHybridCacheManagerV2
        from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
        from tensorrt_llm.mapping import Mapping
    except ImportError as e:
        pytest.skip(f"mixer or KV cache manager v2 unavailable: {e}")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = heads * 128
    max_batch = batch if max_batch is None else max_batch

    def rn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    manager = KDAHybridCacheManagerV2(
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_num_heads=heads,
        mamba_n_groups=heads,
        mamba_head_dim=128,
        mamba_num_layers=num_layers,
        mamba_layer_mask=[True] * num_layers,
        mamba_cache_dtype=BF16,
        mamba_ssm_cache_dtype=BF16,
        kv_cache_config=KvCacheConfig(
            max_tokens=max(128, 32 * max_batch),
            host_cache_size=64 << 20,
            enable_block_reuse=False,
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=0,
        num_kv_heads=0,
        head_dim=128,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=max_batch,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        spec_config=MTPDecodingConfig(max_draft_len=num_spec),
        layer_mask=[False] * num_layers,
        conv_state_layout="q_k_v",
        kda_replay_num_spec=num_spec,
    )
    request_ids = list(range(101, 101 + batch))
    manager.add_dummy_requests(request_ids, token_nums=[32] * batch, is_gen=False)
    slots = torch.tensor(manager.get_state_indices(request_ids), dtype=torch.int32, device="cuda")
    for layer in range(num_layers):
        bufs = _engine_buffers(manager.mamba_layer_cache(layer))
        pool = bufs["recurrent_state"].shape[0]
        bufs["recurrent_state"].copy_(rn(pool, heads, 128, 128, scale=0.5))
        for name in ("cs_q", "cs_k", "cs_v"):
            bufs[name].copy_(rn(pool, dim, 3 + num_spec, scale=0.5))
        k = rn(pool, num_spec, heads, 128)
        bufs["k_cache"].copy_((k / k.norm(dim=-1, keepdim=True)).reshape(pool, num_spec, dim))
        bufs["g_cache"].copy_(-5.0 * torch.sigmoid(rn(pool, num_spec, dim)))
        bufs["v_cache"].copy_(torch.nn.functional.silu(rn(pool, num_spec, dim)))
        bufs["beta_cache"].copy_(torch.sigmoid(rn(pool, num_spec, heads)))
    counts = manager.mamba_layer_cache(0).prev_num_accepted_tokens
    counts.copy_(torch.randint(0, num_spec + 1, counts.shape, generator=gen, device="cuda"))

    cfg = KimiLinearConfig(
        hidden_size=512,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            kda_layers=[1],
            full_attn_layers=[],
            num_heads=heads,
            head_dim=128,
            short_conv_kernel_size=4,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
    )
    runtime = KimiKDALinearAttention(cfg, layer_idx=0, model_config=ModelConfig()).to("cuda")
    with torch.no_grad():
        for name, p in runtime.named_parameters():
            if name.endswith("A_log"):
                p.copy_(rn(*p.shape, scale=0.5))
            elif name.endswith("dt_bias"):
                p.copy_(rn(*p.shape, scale=0.1))
            elif name.endswith("o_norm.weight"):
                p.copy_(1.0 + rn(*p.shape, scale=0.1))
            else:
                p.copy_(rn(*p.shape, scale=0.03))
    runtime.finalize_decode_weights()  # fused projections and the FP32 decode constants, as after the weight load
    runtime._fp8_resident_projections = True  # the SM107 deployment: the verify kernel emits MXFP8
    runtime.o_proj = _PrequantizedProjection()
    hidden = rn(batch * (num_spec + 1), 512, scale=0.5).to(BF16)
    return runtime, manager, slots, hidden


@pytest.mark.parametrize("heads,num_spec,batch", [(96, 3, 5), (96, 5, 3)])
@torch.no_grad()
def test_engine_layout_takes_the_kf_route(heads, num_spec, batch, monkeypatch):
    from tensorrt_llm._torch.modules.kimi_kda import kimi_kda_mixer

    kk = _dispatch_module()
    # beta: a slice of [f_a | b], as deployed
    monkeypatch.setattr(kimi_kda_mixer, "_KDA_FAB_FUSED", False)
    monkeypatch.setenv("TRTLLM_K3_KDA_VERIFY_KF", "1")
    runtime, manager, slots, hidden = _engine_setup(heads, num_spec, batch, seed=7 * heads + batch)
    try:
        dispatch = runtime._dispatch
        if dispatch.verify_kernel_path != "optimized":
            pytest.skip("optimized verify path unavailable")
        assert dispatch._kf_verify
        # The mixer casts its parameters to BF16; the FP32 copies hold the same values.
        assert runtime.A_log.dtype == BF16 and runtime.dt_bias.dtype == BF16
        assert torch.equal(runtime._A_log_f32, runtime.A_log.float())
        assert torch.equal(runtime._dt_bias_f32, runtime.dt_bias.float())

        calls, routed = [], []
        mtp_verify, kf_verify = dispatch.mtp_verify, kk._kf_mtp_verify

        def recording_mtp_verify(**kwargs):
            calls.append(kwargs)
            return mtp_verify(**kwargs)

        def recording_kf_verify(**kwargs):
            routed.append(kwargs)
            return kf_verify(**kwargs)

        dispatch.mtp_verify = recording_mtp_verify
        monkeypatch.setattr(kk, "_kf_mtp_verify", recording_kf_verify)

        caches = [manager.mamba_layer_cache(layer) for layer in range(3)]
        cache = caches[0]
        bufs = _engine_buffers(cache)
        before = {name: t.clone() for name, t in bufs.items()}
        others = [{name: t.clone() for name, t in _engine_buffers(c).items()} for c in caches[1:]]
        counts = cache.prev_num_accepted_tokens.clone()
        steps, rows = num_spec + 1, batch * (num_spec + 1)

        def verify():
            return runtime.forward_verify(hidden, steps, cache, cache.conv, cache.temporal, slots)

        dispatch._kf_verify = False
        ref_out = verify()
        torch.cuda.synchronize()
        ref = {name: t.clone() for name, t in bufs.items()}
        assert not routed
        for name, t in bufs.items():
            t.copy_(before[name])
        dispatch._kf_verify = True
        kf_out = verify()
        torch.cuda.synchronize()
        kf = {name: t.clone() for name, t in bufs.items()}
        assert len(routed) == 1, "the call stayed on the CuTe DSL kernels"

        # The engine's layouts reached both ops: q | k | v | output-gate sections of one projection, beta a slice of
        # [f_a | b], per-slot state blocks, counts at stride 4; the CuTe DSL op read the BF16 parameters. These pin
        # what a deployed engine passes (the sections come from _project_verify_inputs; the counts are the per-slot
        # view KDAHybridCacheManagerV2._get_state_buffer gives DataRole("prev_num_accepted_tokens")): a manager or
        # projection change that moves them fails here, and the eligibility check has to follow it.
        for call in calls:
            x_q = call["x_q"]
            assert x_q.stride(1) == 4 * heads * 128
            for i, name in enumerate(("x_k", "x_v", "onorm_g"), start=1):
                t = call[name]
                assert t.untyped_storage().data_ptr() == x_q.untyped_storage().data_ptr()
                assert t.storage_offset() - x_q.storage_offset() == i * heads * 128
            assert call["beta"].stride(1) == (128 + heads + 7) // 8 * 8
            assert call["beta"].storage_offset() == 128
            assert call["recurrent_state"].stride(0) > heads * 128 * 128
            assert call["num_accepted_tokens"].stride(0) == 4
            assert call["A_log"].dtype == BF16 and call["dt_bias"].dtype == BF16
        assert routed[0]["A_log"].dtype == torch.float32
        assert torch.equal(routed[0]["A_log"], calls[-1]["A_log"].float())
        assert torch.equal(routed[0]["dt_bias"], calls[-1]["dt_bias"].float())

        _check_out(ref_out, kf_out, rows, heads)
        _check_mutated(ref, kf, slots)
        unused = torch.ones(bufs["recurrent_state"].shape[0], dtype=torch.bool, device="cuda")
        unused[slots.long()] = False
        for name in MUTATED:
            assert torch.equal(kf[name][unused], before[name][unused]), (
                f"{name}: other slots changed"
            )
        for layer, old in enumerate(others, start=1):
            for name, t in _engine_buffers(caches[layer]).items():
                assert torch.equal(t, old[name]), f"layer {layer} {name} changed"
        assert torch.equal(cache.prev_num_accepted_tokens, counts)
    finally:
        manager.shutdown()
