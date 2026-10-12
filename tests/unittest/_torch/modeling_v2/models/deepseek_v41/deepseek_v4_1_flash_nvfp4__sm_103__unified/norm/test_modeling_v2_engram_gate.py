# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the engram_gate entry of the DeepSeek-V4.1 target.

What is certified: one Triton launch computes the DeepSeek-V4.1 Engram gate per
(token, hc copy) and the gated residual update ``out = x + gate * value`` --
``Engram.forward`` of the checkpoint's ``inference/model.py`` after the ``wkv``
projection, with ``token_mask`` False shutting the gate.

The entry owns what is certified: ``CELLS`` is the list, ``reference`` and
``compare`` are the gate. This file turns a cell's spec into tensors and keeps
the checks a cell cannot carry: the prefix sweeps over every T up to 16384, the
CUDA-graph captures, in-place and strided layouts, the layout guards run against
the unguarded launch, and the golden layers.

Two sources of truth:

* synthetic inputs, against the entry's ``reference`` (torch ops in the
  reference's own order: per-copy ``mean`` over ``D``, ``rsqrt``, the product
  sum, the signed sqrt with clamp, ``sigmoid``, ``masked_fill``, one cast);
* the golden layer dumps of layers 1 and 14: the key and value are rebuilt from
  the checkpoint (the golden hash ids -> the rows of the Engram table read
  through a memory map -> FP8 1x32 activation quantize -> IEEE fp32 GEMM with
  the dequantized ``wkv``), the rebuilt value must equal the golden
  ``engram_value`` bit for bit, and the kernel on the golden ``h_in`` with the
  real ``q_weight`` / ``k_weight`` must reproduce the golden ``engram_gate``
  and ``h_after_engram``. This part skips, naming the directories, when the
  golden dumps or the checkpoint are absent.

Gates. The kernel differs from the torch reference only in the order of its
three fp32 sums over ``D`` (the pointwise tail uses the CUDA math library
functions torch's float kernels call), so ``op.compare`` holds the gate at the
default fp32 tolerance and the output at its dtype's default tolerance, and
``_check`` adds the output bit for bit against ``x + gate * value`` formed in
torch from the kernel's own gate. Where only exact arithmetic is involved (rows
with one or two nonzero terms), the gate is compared bit for bit. Deliberately
wrong variants -- the RMS taken jointly over the 4 copies, no signed sqrt, the
gate not zeroed on masked tokens, no clamp -- go through the same gates and
must fail them; how far outside each lands is printed.
"""

import functools
import json
import os
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.norm.engram_gate import (  # noqa: E501
    _NUM_WARPS,
    _engram_gate_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.norm.engram_gate import (  # noqa: E501
    engram_gate as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "engram_gate requires a CUDA device"

HC = 4  # hc_mult
DIM = 5120  # hidden_size
EPS = 1e-20  # norm_eps
CLAMP = 1e-6  # Engram.clamp_value

# Reference activations dumped from the checkpoint's own inference code; no default
# location, so the cases that compare against them skip unless this is set.
_GOLDEN_DIR = os.environ.get("MODELING_V2_DSV41_GOLDEN_DIR", "")
# The checkpoint resolves under LLM_MODELS_ROOT, as the whole-model gates' does;
# MODELING_V2_DSV41_CKPT_DIR overrides it. A case that needs it skips when it is absent.
_MODELS_ROOT = llm_models_root()
_CKPT_DIR = os.environ.get("MODELING_V2_DSV41_CKPT_DIR") or (
    str(_MODELS_ROOT / "DeepSeek-V4.1-Flash-NVFP4") if _MODELS_ROOT else ""
)
_GOLDEN_CASES = ("capital_fr", "repeat_ngram", "long_retrieval")
_ENGRAM_LAYERS = (1, 14)

DEV = torch.device("cuda")


def test_device_and_launch_config():
    cap = torch.cuda.get_device_capability()
    print(
        f"\n  device {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}; launch: grid T*HC, "
        f"BLOCK next_pow2(D) = {1 << (DIM - 1).bit_length()}, num_warps {_NUM_WARPS}, no autotune"
    )


# ─── reference: the entry's own; controls: deliberately wrong variants of it ───


def _ref(x, key, value, q_weight, k_weight, token_mask=None, variant=None):
    """Returns (out in x.dtype, gate fp32 [T, HC]). With ``variant`` None this is the
    entry's ``reference``; a ``variant`` names a deliberately wrong reference used as a
    control (``mask_ignored`` is the entry's reference without the mask)."""
    if variant in (None, "mask_ignored"):
        mask = None if variant == "mask_ignored" else token_mask
        return op.reference(x, key, value, q_weight, k_weight, EPS, CLAMP, mask)
    key = key.float()
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    if variant == "joint_rms":
        rstd = torch.rsqrt(h.square().mean((-2, -1), keepdim=True) + EPS) * torch.rsqrt(
            key.square().mean((-2, -1), keepdim=True) + EPS
        )
        rstd = rstd.squeeze(-1)
    else:
        rstd = torch.rsqrt(h.square().mean(-1) + EPS) * torch.rsqrt(key.square().mean(-1) + EPS)
    dot = (h * weight * key).sum(-1) * rstd * DIM**-0.5
    if variant == "no_signed_sqrt":
        gate = torch.sigmoid(dot)
    elif variant == "no_clamp":
        gate = torch.sigmoid(torch.copysign(dot.abs().sqrt(), dot))
    else:
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.bool().unsqueeze(-1), 0)
    out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    return out, gate


def _ulps(a, b):
    """Elementwise distance in units of the last place (bf16 / fp16 / fp32), via
    the order-preserving integer image of the bit pattern."""
    assert a.dtype == b.dtype
    itype = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[
        a.dtype
    ]

    def ordered(t):
        i = t.contiguous().view(itype).long()
        top = torch.iinfo(itype).min
        return torch.where(i < 0, top - i, i)

    return (ordered(a) - ordered(b)).abs()


def _check(out, gate, ref_out, ref_gate, x, value):
    """The certification gates.

    1. and 3. ``op.compare``: the gate at the default fp32 tolerance (the kernel's
       three sums over D run in a different order than torch's, nothing else
       differs) and the output against the reference at the default tolerance of
       its dtype.
    2. The output bit for bit against ``x + gate * value`` formed in fp32 from the
       kernel's own gate and rounded once, in torch: the update itself admits no
       tolerance (no FMA contraction, one rounding).
    """
    own = (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(out.dtype)
    assert torch.equal(out, own), f"out != x + gate * value on {(out != own).sum().item()} elements"
    op.compare((out, gate), (ref_out, ref_gate))


def _stats(out, gate, ref_out, ref_gate):
    exact = (out.reshape(-1) == ref_out.reshape(-1)).float().mean().item()
    d = (out.float() - ref_out.float()).abs()
    return (
        f"gate max|diff| {(gate - ref_gate).abs().max().item():.3g} (bit-exact "
        f"{(gate == ref_gate).float().mean().item():.4f}), out bit-exact {exact:.6f}, max|diff| {d.max().item():.3g}, "
        f"elements > 1 ulp {(_ulps(out, ref_out) > 1).float().mean().item():.2e}"
    )


def _assert_controls_fail(out, gate, x, key, value, qw, kw, mask, variants, tag):
    for v in variants:
        w_out, w_gate = _ref(x, key, value, qw, kw, mask, variant=v)
        with pytest.raises(AssertionError):
            _check(out, gate, w_out, w_gate, x, value)
        with pytest.raises(
            AssertionError
        ):  # the output alone, at bf16's default tolerance, also sees it
            torch.testing.assert_close(out, w_out)
        print(f"  control {tag} {v}: rejected; {_stats(out, gate, w_out, w_gate)}")


# ─── synthetic inputs shaped like the checkpoint's ───


def _inputs(t_len, seed, x_dtype=torch.bfloat16, kv_dtype=torch.bfloat16, w_dtype=torch.bfloat16):
    """x [T, HC, D]; key/value as the two split views of one fused wkv output
    [T, (HC + 1) * D]; q/k weights [HC, D]. The copies get different scales (as
    the real stream's copies do), so a joint RMS is visibly wrong; each key is
    noise plus a small random multiple of its x row, so the gates spread over (0, 1)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x_scale = torch.tensor([0.25, 1.0, 3.0, 10.0], device=DEV).view(1, HC, 1)
    k_scale = torch.tensor([2.0, 0.5, 1.0, 0.1], device=DEV).view(1, HC, 1)
    xn = torch.randn(t_len, HC, DIM, device=DEV, generator=g)
    x = (xn * x_scale * torch.exp(torch.randn(t_len, 1, 1, device=DEV, generator=g))).to(x_dtype)
    rho = (torch.rand(t_len, HC, 1, device=DEV, generator=g) - 0.5) * 0.1
    keyf = (torch.randn(t_len, HC, DIM, device=DEV, generator=g) + rho * xn) * k_scale
    valf = torch.randn(t_len, DIM, device=DEV, generator=g)
    kv = torch.cat([keyf.reshape(t_len, HC * DIM), valf], 1).to(kv_dtype)
    key = kv[:, : HC * DIM].view(t_len, HC, DIM)
    value = kv[:, HC * DIM :]
    qw = (1.0 + 0.3 * torch.randn(HC, DIM, device=DEV, generator=g)).to(w_dtype)
    kw = (1.0 + 0.3 * torch.randn(HC, DIM, device=DEV, generator=g)).to(w_dtype)
    return x, key, value, qw, kw


def _mask(t_len, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    m = torch.rand(t_len, device=DEV, generator=g) > 0.2
    if t_len > 1:
        m[0] = False
        m[-1] = True
    return m


def _call_cell_seeds(spec):
    """The input and mask seeds the configuration was measured with."""
    if spec["mask_stride"] != 1:
        return 11, 12
    if (spec["x_dtype"], spec["kv_dtype"], spec["w_dtype"]) != (torch.bfloat16,) * 3:
        return 7, 8
    return spec["T"], spec["T"] + 1


def _drive_call_cell(spec):
    """Plain call form: a new ``out``, ``gate_out`` given, key and value the split views of
    one fused wkv buffer, the dtypes and mask layout of the cell. Masked rows: gate exactly 0
    and the output rows bit-identical to x; the mask may be bool or integer (nonzero = keep)
    and strided."""
    t_len, stride = spec["T"], spec["mask_stride"]
    seed, mask_seed = _call_cell_seeds(spec)
    x, key, value, qw, kw = _inputs(t_len, seed, spec["x_dtype"], spec["kv_dtype"], spec["w_dtype"])
    keep = _mask(t_len, seed=mask_seed) if spec["masked"] else None
    mask = None
    if keep is not None:
        buf = torch.zeros(t_len, stride, dtype=spec["mask_dtype"], device=DEV)
        buf[:, stride // 2] = keep.to(spec["mask_dtype"])
        mask = buf[:, stride // 2]
        assert mask.stride() == (stride,)
    gate = torch.full((t_len, HC), float("nan"), device=DEV)
    out = op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask, out=None, gate_out=gate)
    assert out.dtype == x.dtype and out.shape == x.shape and out.is_contiguous()
    ref_out, ref_gate = op.reference(x, key, value, qw, kw, EPS, CLAMP, mask)
    _check(out, gate, ref_out, ref_gate, x, value)
    print(
        f"\n  T={t_len} mask={None if mask is None else (mask.dtype, mask.stride())} "
        f"x {x.dtype} kv {key.dtype} w {qw.dtype}: {_stats(out, gate, ref_out, ref_gate)}; "
        f"gate range [{ref_gate.min().item():.4f}, {ref_gate.max().item():.4f}]"
    )
    if keep is not None:
        assert torch.equal(gate[~keep], torch.zeros_like(gate[~keep]))
        assert torch.equal(out[~keep], x[~keep])
        assert (gate[keep] > 0).all()
    if stride == 1:
        variants = ["joint_rms", "no_signed_sqrt"] + (
            ["mask_ignored"] if keep is not None and not bool(keep.all()) else []
        )
        _assert_controls_fail(out, gate, x, key, value, qw, kw, mask, variants, f"T={t_len}")


def _drive_edge_cell():
    """Rows built so every sum has one nonzero term (exact in any order): x = e0,
    key = c*e0 + e1 (fp32), so dot = c * sqrt(D) * (1 + c^2)^-1/2 exactly as torch
    computes it. |dot| sweeps across the 1e-6 clamp on both signs, plus zero rows
    (dot = 0 -> sigmoid(+1e-3)) and |dot| up to 1e2. The gate must equal the torch
    reference bit for bit; the no-clamp and no-signed-sqrt controls must not."""
    targets = [0.0, 1e-12, 1e-9, 1e-7, 5e-7, 1e-6, 2e-6, 1e-5, 1e-4, 1e-2, 0.1, 1.0, 4.0, 100.0]
    cs = [s * v / DIM**0.5 for v in targets for s in (1.0, -1.0)]
    n = len(cs) + 3
    x = torch.zeros(n, HC, DIM, dtype=torch.bfloat16, device=DEV)
    key = torch.zeros(n, HC, DIM, dtype=torch.float32, device=DEV)
    x[:, :, 0] = 1.0
    for i, c in enumerate(cs):
        key[i, :, 0] = c
        key[i, :, 1] = 1.0
    x[len(cs)] = 0.0  # x row zero: rstd_x = 1e10, dot = 0
    key[len(cs), :, 1] = 1.0
    key[len(cs) + 1] = 0.0  # key row zero
    x[len(cs) + 2] = 0.0  # both zero
    value = torch.linspace(-3, 3, n * DIM, device=DEV).view(n, DIM)
    qw = torch.ones(HC, DIM, dtype=torch.bfloat16, device=DEV)
    kw = torch.ones(HC, DIM, dtype=torch.bfloat16, device=DEV)
    gate = torch.empty(n, HC, dtype=torch.float32, device=DEV)
    out = op(x, key, value, qw, kw, EPS, CLAMP, gate_out=gate)
    ref_out, ref_gate = op.reference(x, key, value, qw, kw, EPS, CLAMP)
    assert torch.equal(gate, ref_gate), (gate[:, 0] - ref_gate[:, 0]).abs().max().item()
    assert torch.equal(out, ref_out)
    _check(out, gate, ref_out, ref_gate, x, value)
    lo = torch.sigmoid(torch.tensor([CLAMP], device=DEV).sqrt()).item()
    print(
        f"\n  {n} rows: gate bit-exact incl. dot = +-{targets[1]}..{targets[-1]}; "
        f"zero rows gate {gate[-3:, 0].tolist()} "
        f"(sigmoid(sqrt(fp32(1e-6))) = {lo})"
    )
    assert torch.equal(gate[-3:], torch.full_like(gate[-3:], lo))
    for v in ("no_clamp", "no_signed_sqrt"):
        w_out, w_gate = _ref(x, key, value, qw, kw, variant=v)
        with pytest.raises(AssertionError):
            _check(out, gate, w_out, w_gate, x, value)
        print(f"  control {v}: rejected; gate max|diff| {(gate - w_gate).abs().max().item():.3g}")


def _drive_target_masked_cell(kind, t_len):
    """The multimodal target's masked call at T = 16384 with one mask: in place, split wkv
    views, a contiguous bool keep. (a) It equals the same call with ``gate_out`` (for the
    gate) bit for bit, and passes the certification gates against the reference; (b) masked
    rows: gate exactly 0 and the output rows equal x bit for bit; (c) keep rows equal the
    UNMASKED call's rows bit for bit (the mask perturbs no live row); an all-True mask
    tensor equals ``token_mask=None`` bitwise. Controls against the reference built with a
    wrong mask -- ignored, shifted by one row, inverted -- are rejected by the same gates
    wherever the wrong mask differs from the right one."""
    x, kv, key3, value, qw, kw = _mm_state(t_len, seed=9100 + _MM_MASKS.index(kind))
    keep = _mm_mask(kind, t_len)
    assert keep.dtype == torch.bool and keep.is_contiguous() and keep.shape == (t_len,)
    out = _mm_call(x, key3, value, qw, kw, keep)
    gate = torch.full((t_len, HC), float("nan"), device=DEV)
    with_gate = x.clone()
    op(with_gate, key3, value, qw, kw, EPS, CLAMP, keep, with_gate, gate_out=gate)
    assert torch.equal(with_gate, out)
    ref_out, ref_gate = op.reference(x, key3, value, qw, kw, EPS, CLAMP, keep)
    _check(out, gate, ref_out, ref_gate, x, value)
    dead = ~keep
    assert torch.equal(gate[dead], torch.zeros_like(gate[dead]))
    assert torch.equal(out[dead], x[dead])
    plain = _mm_call(x, key3, value, qw, kw, None)
    assert torch.equal(out[keep], plain[keep]), (
        f"{int((out[keep] != plain[keep]).sum())} live elements differ"
    )
    if bool(keep.all()):
        assert torch.equal(out, plain)
    n_dead = int(dead.sum())
    print(
        f"\n  T={t_len} {kind}: {n_dead} masked rows; {_stats(out, gate, ref_out, ref_gate)}; masked rows == x, "
        f"live rows == unmasked call, bit for bit"
    )
    wrong = {"mask_ignored": None, "mask_shifted": torch.roll(keep, 1), "mask_inverted": ~keep}
    for name, w_mask in wrong.items():
        w_keep = torch.ones_like(keep) if w_mask is None else w_mask
        n_diff = int((w_keep != keep).sum())
        if n_diff == 0:  # a constant mask is its own shift; all-True ignored is itself
            print(f"  control {kind} {name}: identical to the right mask, nothing to reject")
            continue
        w_out, w_gate = op.reference(x, key3, value, qw, kw, EPS, CLAMP, w_mask)
        with pytest.raises(AssertionError):
            _check(out, gate, w_out, w_gate, x, value)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out, w_out)
        print(
            f"  control {kind} {name} ({n_diff} rows differ): rejected; {_stats(out, gate, w_out, w_gate)}"
        )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every cell through ``op`` (guarded by ``is_valid``), ``op.reference`` and ``op.compare``,
    with the structural checks and wrong-variant controls its configuration carries."""
    certifying(op)
    spec = cell.spec
    assert (spec["HC"], spec["D"], spec["eps"], spec["clamp_value"]) == (HC, DIM, EPS, CLAMP)
    if spec["kind"] == "call":
        _drive_call_cell(spec)
    elif spec["kind"] == "edges":
        _drive_edge_cell()
    else:
        assert spec["kind"] == "target_masked", spec["kind"]
        _drive_target_masked_cell(spec["mask"], spec["T"])


def test_in_place_and_strided_views():
    """out = x (in place), x / out as views into wider buffers, key / value as split
    views of one fused buffer, gate_out strided: all bit-identical to contiguous
    copies, and nothing outside the views is written."""
    t_len = 96
    x, key, value, qw, kw = _inputs(t_len, seed=21)
    mask = _mask(t_len, seed=22)
    base = op(
        x.contiguous(), key.contiguous(), value.contiguous(), qw, kw, EPS, CLAMP, token_mask=mask
    )

    xi = x.clone()
    res = op(xi, key, value, qw, kw, EPS, CLAMP, token_mask=mask, out=xi)
    assert res.data_ptr() == xi.data_ptr() and torch.equal(xi, base)

    wide = torch.randn(t_len, HC + 2, DIM + 64, device=DEV).to(torch.bfloat16)
    wide_before = wide.clone()
    xv = wide[:, 1 : HC + 1, 32 : 32 + DIM]
    xv.copy_(x)
    wide_before[:, 1 : HC + 1, 32 : 32 + DIM] = x
    ob = torch.full((t_len, HC, DIM + 16), 7.0, dtype=torch.bfloat16, device=DEV)
    ov = ob[:, :, 8 : 8 + DIM]
    gb = torch.full((HC, 2 * t_len), 7.0, device=DEV)
    gv = gb[:, ::2].t()  # [T, HC], strides (2, 2T)
    op(xv, key, value, qw, kw, EPS, CLAMP, token_mask=mask, out=ov, gate_out=gv)
    assert torch.equal(ov, base)
    assert torch.equal(wide, wide_before)  # x view is read only
    assert (ob[:, :, :8] == 7).all() and (ob[:, :, 8 + DIM :] == 7).all()
    assert (gb[:, 1::2] == 7).all()
    _, ref_gate = _ref(x, key, value, qw, kw, mask)
    torch.testing.assert_close(gv, ref_gate)
    # inputs are not modified by an out-of-place call
    snap = [t.clone() for t in (x, key, value, qw, kw)]
    op(x, key, value, qw, kw, EPS, CLAMP)
    assert all(torch.equal(a, b) for a, b in zip(snap, (x, key, value, qw, kw)))
    print(
        "\n  in-place, strided x/out/gate_out, split key/value views: bit-identical, no stray writes"
    )


def test_batch_invariance():
    """Each (token, copy) row is computed by one program with a fixed block, so a
    token's result does not depend on how many tokens share the call: rows of a
    1024-token call equal the same rows computed alone (T=1) and in a 37-token call."""
    x, key, value, qw, kw = _inputs(1024, seed=31)
    mask = _mask(1024, seed=32)
    full = op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask)
    for t in (0, 1, 500, 1023):
        one = op(
            x[t : t + 1],
            key[t : t + 1],
            value[t : t + 1],
            qw,
            kw,
            EPS,
            CLAMP,
            token_mask=mask[t : t + 1],
        )
        assert torch.equal(one, full[t : t + 1])
    part = op(
        x[300:337], key[300:337], value[300:337], qw, kw, EPS, CLAMP, token_mask=mask[300:337]
    )
    assert torch.equal(part, full[300:337])
    again = op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask)
    assert torch.equal(again, full)
    print("\n  rows of T=1024 == the same rows at T=1 and T=37; repeat call bit-identical")


@pytest.mark.parametrize("t_len", [1, 5, 160])
def test_cuda_graph_capture_replay(t_len):
    """Captured once, replayed on new data written into the same buffers (in place,
    with a mask): bit-identical to an eager call on that data."""
    x, key, value, qw, kw = _inputs(t_len, seed=41 + t_len)
    mask = _mask(t_len, seed=42)
    sx, skey, sval, smask = x.clone(), key.clone(), value.clone(), mask.clone()
    sgate = torch.empty(t_len, HC, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(
            sx.clone(), skey, sval, qw, kw, EPS, CLAMP, token_mask=smask, gate_out=sgate
        )  # warm-up / compile
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(sx, skey, sval, qw, kw, EPS, CLAMP, token_mask=smask, out=sx, gate_out=sgate)
    for seed in (1, 2):
        x2, key2, value2, _, _ = _inputs(t_len, seed=100 + seed)
        mask2 = _mask(t_len, seed=200 + seed)
        sx.copy_(x2)
        skey.copy_(key2)
        sval.copy_(value2)
        smask.copy_(mask2)
        graph.replay()
        g2 = torch.empty(t_len, HC, device=DEV)
        eager = op(x2, key2, value2, qw, kw, EPS, CLAMP, token_mask=mask2, gate_out=g2)
        assert torch.equal(sx, eager) and torch.equal(sgate, g2)
    print(f"\n  T={t_len}: graph replay bit-identical to eager on 2 fresh inputs")


# ─── every token count 1..16384 is the same per-row function ───
#
# The core target calls this op on one rank's T tokens, T = 1..max_num_tokens, and
# max_num_tokens reaches 16384 per rank (every member gates only its own tokens, so no
# gathered count reaches it; Engram layers 1 and 14 are before the decoder bounded-replay
# pass). The grid is T * HC programs, one per (token, copy) row, and T is not a kernel
# argument at all: it only sizes the grid. So every prefix of a 16384-token call must
# return exactly the first r rows of the 16384-token result, which is held to the reference.
_N_MAX = 16384


def _jit_variants(fn) -> int:
    """Compiled variants Triton holds for ``fn`` on every device (Triton >= 3.2 keeps them
    in ``device_caches[device][0]``, older releases in ``cache[device]``)."""
    dc = getattr(fn, "device_caches", None)
    if dc:
        return sum(len(v[0]) for v in dc.values())
    return sum(len(v) for v in fn.cache.values())


_POISON = 0x7FA5  # a bf16 NaN bit pattern; the kernel writes finite values for finite inputs
_GUARD = 64  # token rows past the prefix that must still hold the poison after each call


def test_every_token_count_is_a_prefix_of_16384():
    """16384 tokens in the core's layout (key and value the split views of one fused
    [T, 25600] wkv buffer, no mask). The 16384-token call, run with ``gate_out``, passes
    the certification gates against the reference, and the same call without
    ``gate_out`` (the core's form) is bit-identical to it. Then every prefix r = 1..16384
    is called into a token-row-prefix view of one poisoned output buffer: its r rows
    must equal the 16384-token rows bit for bit and the next 64 token rows keep the
    poison (mismatches summed on the device, one synchronization). The kernel's
    compiled variants are counted: none may appear after the first prefix call."""
    n_max = _N_MAX
    x, key, value, qw, kw = _inputs(n_max, seed=8192)
    gate = torch.empty(n_max, HC, dtype=torch.float32, device=DEV)
    with_gate = op(x, key, value, qw, kw, EPS, CLAMP, gate_out=gate)
    ref_out, ref_gate = _ref(x, key, value, qw, kw)
    _check(with_gate, gate, ref_out, ref_gate, x, value)
    full = op(x, key, value, qw, kw, EPS, CLAMP)
    assert torch.equal(full, with_gate)
    print(f"\n  T={n_max}: {_stats(full, gate, ref_out, ref_gate)}")
    full_bits = full.view(torch.int16)
    buf = torch.empty(n_max + _GUARD, HC, DIM, dtype=torch.bfloat16, device=DEV)
    bits = buf.view(torch.int16)
    bits.fill_(_POISON)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    over = torch.zeros((), dtype=torch.int64, device=DEV)
    before = _jit_variants(_engram_gate_kernel)
    settled = None
    for r in range(1, n_max + 1):
        res = op(x[:r], key[:r], value[:r], qw, kw, EPS, CLAMP, None, buf[:r])
        assert res.data_ptr() == buf.data_ptr()
        bad += (bits[:r] != full_bits[:r]).sum()
        over += (bits[r : r + _GUARD] != _POISON).sum()
        if r == 1:
            settled = _jit_variants(_engram_gate_kernel)
    torch.cuda.synchronize()
    after = _jit_variants(_engram_gate_kernel)
    print(
        f"  r = 1..{n_max}: {bad.item()} elements differ from the {n_max}-token rows, {over.item()} guard "
        f"elements written; kernel variants {before} before, {settled} after r = 1, {after} after r = {n_max}"
    )
    assert bad.item() == 0 and over.item() == 0
    assert after == settled and settled - before <= 1, (before, settled, after)


def test_row_sweep_harness_sees_a_one_token_shift():
    """Control for the sweep above: the same bitwise per-row check rejects a call that
    reads its inputs one token row off, at every r tried, including r = 1."""
    x, key, value, qw, kw = _inputs(65, seed=65)
    full = op(x[:64], key[:64], value[:64], qw, kw, EPS, CLAMP)
    for r in (1, 2, 7, 63, 64):
        shifted = op(x[1 : r + 1], key[1 : r + 1], value[1 : r + 1], qw, kw, EPS, CLAMP)
        n_bad = int((shifted.view(torch.int16) != full.view(torch.int16)[:r]).sum().item())
        print(f"  [control row shift] r={r}: {n_bad} / {r * HC * DIM} elements differ")
        assert n_bad > r * HC * DIM // 2


# ─── the multimodal target's masked call: every T 1..16384, eager ───
#
# The multimodal target calls, on one rank's T = 1..16384 tokens of a step that holds a
# context chunk of an image request,
#     engram_gate(h, key3, value, q_weight, k_weight, 1e-20, 1e-6, keep, h)
# in place on the bf16 stream h [T, 4, 5120], key3 / value the split views of ONE fused
# [T, 25600] wkv buffer, and keep a contiguous bool [T] that is False on image-span rows
# (gate 0) and True on every other row (text, delimiters, generation rows of a mixed step).
# Such steps are never captured: the graph decision is all-rank and captured steps are
# generation-only, which carry no image row, so the masked form runs eagerly only. T only
# sizes the grid and the mask is read at row t alone, so the prefix rule applies: the 16384-row
# masked call against the reference, and every prefix r (with the mask's prefix) bitwise
# equal to the first r rows of that call.

_MM_T = _N_MAX


def _spans(t_len, spans):
    """keep [T] bool: False on each [a, b) of ``spans`` (image-span rows)."""
    m = torch.ones(t_len, dtype=torch.bool, device=DEV)
    for a, b in spans:
        m[a:b] = False
    return m


def _random_spans(t_len, g, n_spans, max_len):
    """``n_spans`` image spans of 1..max_len rows at random starts (may overlap / touch)."""
    out = []
    for _ in range(n_spans):
        ln = int(torch.randint(1, max_len + 1, (1,), generator=g))
        a = int(torch.randint(0, max(1, t_len - ln + 1), (1,), generator=g))
        out.append((a, min(t_len, a + ln)))
    return out


def _mm_mask(kind, t_len=_MM_T):
    """The named keep masks (False = image-span row). ``rand<k>`` are seeded random mixes of
    span counts, span lengths and Bernoulli densities."""
    if (
        kind == "none_masked"
    ):  # an image request's chunk that holds no span row: all-True tensor, not None
        return torch.ones(t_len, dtype=torch.bool, device=DEV)
    if kind == "all_masked":  # the whole step is image-span rows
        return torch.zeros(t_len, dtype=torch.bool, device=DEV)
    if kind == "span_at_row0":
        return _spans(t_len, [(0, 1517)])
    if kind == "span_at_end":
        return _spans(t_len, [(t_len - 2611, t_len)])
    if kind == "one_span_mid":
        return _spans(t_len, [(1000, 1000 + 2305)])
    if kind == "multi_span":  # several images, delimiters / text rows between them, mixed-step tail
        return _spans(
            t_len,
            [
                (0, 3),
                (5, 1524),
                (1530, 1531),
                (2000, 4611),
                (4613, 7046),
                (8190, 8194),
                (9000, 13311),
                (t_len - 2, t_len),
            ],
        )
    if kind == "alternating":
        return torch.arange(t_len, device=DEV) % 2 == 0
    g = torch.Generator().manual_seed(4200 + int(kind[4:]))
    k = int(kind[4:])
    if k % 3 == 0:  # Bernoulli rows at a random density
        p = float(torch.rand((), generator=g))
        return (torch.rand(t_len, generator=g) > p).to(DEV)
    n_spans = int(torch.randint(1, 17, (1,), generator=g))
    return _spans(
        t_len, _random_spans(t_len, g, n_spans, 1 + int(torch.randint(1, 3000, (1,), generator=g)))
    )


_MM_MASKS = [
    "none_masked",
    "all_masked",
    "span_at_row0",
    "span_at_end",
    "one_span_mid",
    "multi_span",
    "alternating",
] + [f"rand{k}" for k in range(9)]


def _mm_state(t_len, seed):
    """The target's buffers: the stream x [T, HC, D], ONE fused wkv buffer [T, 25600] and its
    split views key3 [T, HC, D] (row stride 25600) and value [T, D] (row stride 25600)."""
    x, key0, value0, qw, kw = _inputs(t_len, seed=seed)
    kv = torch.empty(t_len, (HC + 1) * DIM, dtype=torch.bfloat16, device=DEV)
    kv[:, : HC * DIM].copy_(key0.reshape(t_len, HC * DIM))
    kv[:, HC * DIM :].copy_(value0)
    key3 = kv[:, : HC * DIM].view(t_len, HC, DIM)
    value = kv[:, HC * DIM :]
    assert key3.stride() == ((HC + 1) * DIM, DIM, 1) and value.stride() == ((HC + 1) * DIM, 1)
    return x, kv, key3, value, qw, kw


def _mm_call(x, key3, value, qw, kw, keep):
    """The target's call on a copy of ``x``: in place, keep as token_mask, no gate_out."""
    h = x.clone()
    res = op(h, key3, value, qw, kw, EPS, CLAMP, keep, h)
    assert res.data_ptr() == h.data_ptr()
    return h


@pytest.mark.parametrize("kind", ["multi_span", "rand3"])
def test_multimodal_masked_every_token_count_is_a_prefix_of_16384(kind):
    """The prefix rule in the target's masked form. The 16384-row call (in place, split views of
    one fused wkv buffer, contiguous bool keep) passes the reference gates. Then every
    r = 1..16384 is run in the same form on one poisoned stream buffer: rows [0, r) get x's
    rows, the call runs in place on buf[:r] with key3[:r], value[:r], keep[:r]; its rows
    must equal the 16384-row result bit for bit and the next 64 rows keep the poison
    (mismatches summed on the device, one synchronization). The masked variant is compiled
    by the 16384-row call in exactly this form, so the sweep must compile nothing."""
    x, kv, key3, value, qw, kw = _mm_state(_MM_T, seed=9300 + len(kind))
    keep = _mm_mask(kind)
    buf = torch.empty(_MM_T + _GUARD, HC, DIM, dtype=torch.bfloat16, device=DEV)
    bits = buf.view(torch.int16)
    bits.fill_(_POISON)
    buf[:_MM_T].copy_(x)
    op(buf[:_MM_T], key3, value, qw, kw, EPS, CLAMP, keep, buf[:_MM_T])  # the 16384-row call
    full = buf[:_MM_T].clone()
    assert (bits[_MM_T:] == _POISON).all()
    gate = torch.empty(_MM_T, HC, dtype=torch.float32, device=DEV)
    with_gate = x.clone()
    op(with_gate, key3, value, qw, kw, EPS, CLAMP, keep, with_gate, gate_out=gate)
    assert torch.equal(with_gate, full)
    ref_out, ref_gate = _ref(x, key3, value, qw, kw, keep)
    _check(full, gate, ref_out, ref_gate, x, value)
    assert torch.equal(full[~keep], x[~keep])
    print(
        f"\n  T={_MM_T} {kind} ({int((~keep).sum())} masked rows): {_stats(full, gate, ref_out, ref_gate)}"
    )
    full_bits = full.view(torch.int16)
    bits.fill_(_POISON)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    over = torch.zeros((), dtype=torch.int64, device=DEV)
    before = _jit_variants(_engram_gate_kernel)
    for r in range(1, _MM_T + 1):
        h = buf[:r]
        h.copy_(x[:r])
        res = op(h, key3[:r], value[:r], qw, kw, EPS, CLAMP, keep[:r], h)
        assert res.data_ptr() == buf.data_ptr()
        bad += (bits[:r] != full_bits[:r]).sum()
        over += (bits[r : r + _GUARD] != _POISON).sum()
    torch.cuda.synchronize()
    after = _jit_variants(_engram_gate_kernel)
    print(
        f"  masked r = 1..{_MM_T}: {bad.item()} elements differ from the {_MM_T}-row result, {over.item()} guard "
        f"elements written; kernel variants {before} before the sweep, {after} after"
    )
    assert bad.item() == 0 and over.item() == 0
    assert after == before, (before, after)


def test_multimodal_masked_sweep_harness_sees_a_shifted_mask():
    """Control for the masked sweep: the same bitwise per-row check rejects a prefix call whose
    mask is read one row off (keep[1:r+1]) at every r tried where the shifted mask differs."""
    x, kv, key3, value, qw, kw = _mm_state(_MM_T, seed=9300 + len("rand3"))
    keep = _mm_mask("rand3")
    full = _mm_call(x, key3, value, qw, kw, keep)
    for r in (2, 7, 64, 1517, 8191, 16383):
        n_rows = int((keep[1 : r + 1] != keep[:r]).sum())
        shifted = _mm_call(x[:r], key3[:r], value[:r], qw, kw, keep[1 : r + 1].contiguous())
        n_bad = int((shifted.view(torch.int16) != full.view(torch.int16)[:r]).sum().item())
        print(
            f"  [control shifted mask] r={r}: mask differs on {n_rows} rows, {n_bad} / {r * HC * DIM} elements differ"
        )
        assert n_rows > 0 and n_bad > 0


# ─── capture at every graph batch the gates capture: the core's exact call ───

# The generation-only CUDA-graph batches an engine captures per rank, Gset: every batch
# 1..32 plus TRT-LLM's padded default list for cuda_graph_config {enable_padding: true,
# max_batch_size: 256} (1, 2, 4, 8, 16..128 step 8, 192, 256; the padding-off list at 256,
# 1..32, 64, 128, 256, is a subset). A decode step gates G tokens and a DSpark target-verify
# step 6 G (1 + 5 drafts), so T = G and 6 G: 82 counts up to 1536. Every member (tep4, dep4,
# tp4) gates only its own tokens, so no gathered row count reaches this op.
_GRAPH_BATCHES = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_T = sorted(set(_GRAPH_BATCHES) | {6 * b for b in _GRAPH_BATCHES})
assert len(_GRAPH_BATCHES) == 46 and len(_CAPTURE_T) == 82 and _CAPTURE_T[-1] == 1536


@pytest.mark.parametrize("t_len", _CAPTURE_T)
def test_cuda_graph_at_every_capture_batch(t_len):
    """The core target's call: key and value are the two split views of ONE static fused
    ``[T, 25600]`` wkv buffer, no mask, no ``gate_out``, in place on the stream. Warmed
    up eagerly at the same T, captured once, then replayed on 3 fresh inputs written in
    place: each replay equals the same eager call on the same data bit for bit, and
    passes the certification gates against the reference (the gate taken from an eager
    call with ``gate_out``, whose ``out`` must equal the call without it bit for bit).
    The joint-RMS and no-signed-sqrt controls are run against the first replay."""
    kv_n = (HC + 1) * DIM
    x0, key0, value0, qw, kw = _inputs(t_len, seed=700 + t_len)
    sx = x0.clone()
    skv = torch.empty(t_len, kv_n, dtype=torch.bfloat16, device=DEV)
    skv[:, : HC * DIM].copy_(key0.reshape(t_len, HC * DIM))
    skv[:, HC * DIM :].copy_(value0)
    skey = skv[:, : HC * DIM].view(t_len, HC, DIM)  # reshape(split(kv)[0], [T, HC, D]) in the core
    sval = skv[:, HC * DIM :]  # split(kv)[1]: a [T, D] view with row stride 25600
    assert skey.data_ptr() == skv.data_ptr() and sval.stride(0) == kv_n
    out0 = x0.clone()
    op(out0, skey, sval, qw, kw, EPS, CLAMP, None, out0)  # the capture-time data's result
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        warm = sx.clone()
        op(warm, skey, sval, qw, kw, EPS, CLAMP, None, warm)  # engine warm-up at this T
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(sx, skey, sval, qw, kw, EPS, CLAMP, None, sx)
    for seed in (1, 2, 3):
        x2, key2, value2, _, _ = _inputs(t_len, seed=800 + 10 * t_len + seed)
        sx.copy_(x2)
        skv[:, : HC * DIM].copy_(key2.reshape(t_len, HC * DIM))
        skv[:, HC * DIM :].copy_(value2)
        graph.replay()
        torch.cuda.synchronize()
        eager = x2.clone()
        op(eager, skey, sval, qw, kw, EPS, CLAMP, None, eager)  # the same call form, eagerly
        assert torch.equal(sx, eager), f"replay != eager on {(sx != eager).sum().item()} elements"
        gate = torch.empty(t_len, HC, dtype=torch.float32, device=DEV)
        with_gate = x2.clone()
        op(with_gate, skey, sval, qw, kw, EPS, CLAMP, None, with_gate, gate_out=gate)
        assert torch.equal(with_gate, eager)
        ref_out, ref_gate = _ref(x2, key2, value2, qw, kw)
        _check(sx, gate, ref_out, ref_gate, x2, value2)
        stale = (sx != out0).float().mean().item()
        assert stale > 0.5, "a replay on new data returned the capture-time data's result"
        if seed == 1:
            print(
                f"\n  T={t_len} graph: replay == eager bit for bit; {_stats(sx, gate, ref_out, ref_gate)}; "
                f"differs from the capture-time data's output on {stale:.4f} of elements"
            )
            _assert_controls_fail(
                sx,
                gate,
                x2,
                key2,
                value2,
                qw,
                kw,
                None,
                ["joint_rms", "no_signed_sqrt"],
                f"graph T={t_len}",
            )


def test_empty_batch():
    x, key, value, qw, kw = _inputs(4, seed=51)
    out = op(
        x[:0],
        key[:0],
        value[:0],
        qw,
        kw,
        EPS,
        CLAMP,
        token_mask=torch.ones(0, dtype=torch.bool, device=DEV),
        gate_out=torch.empty(0, HC, device=DEV),
    )
    assert out.shape == (0, HC, DIM) and out.dtype == torch.bfloat16


# ─── is_valid: each rejected layout, run through the unguarded launch first ───


def _interleaved(t):
    """A view with ``t``'s values and stride 2 along the last dim (other, deterministic data between)."""
    return torch.stack([t, (t.float() * -1.7 + 0.3).to(t.dtype)], -1).flatten(-2)[..., ::2]


def _call(args, **kwargs):
    return op(args["x"], args["key"], args["value"], args["qw"], args["kw"], EPS, CLAMP, **kwargs)


def test_is_valid_rejects_layouts_the_kernel_misreads():
    """Every rejected layout below is run through ``op`` outside ``validating`` first (the
    launch alone, no guard), and it returns without an error: a stride-2 view is read as if
    contiguous (the result is wrong against the reference on the tensor as given), and a
    view with fewer rows / columns / copies than ``x`` implies is read or written past its
    end (the result equals the intended call's on the parent buffer the view was cut from).
    Inside ``validating`` the entry's ``is_valid`` refuses each one."""
    t_len = 16
    x, key, value, qw, kw = _inputs(t_len, seed=61)
    mask = _mask(t_len, seed=62)
    base = dict(x=x, key=key, value=value, qw=qw, kw=kw)
    with validating(op):
        intended = op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask)  # the intended tensors
    strided = {
        "x inner stride 2": ("x", "x must be"),
        "key inner stride 2": ("key", "key must be"),
        "value inner stride 2": ("value", "value must be"),
        "q_weight inner stride 2": ("qw", "q_weight must be"),
    }
    for name, (arg, match) in strided.items():
        args = dict(base)
        args[arg] = _interleaved(base[arg])
        got = _call(args, token_mask=mask)
        torch.cuda.synchronize()
        worst = _ulps(got, intended).max().item()  # the view holds the intended values
        assert worst > 1, name
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                _call(args, token_mask=mask)
        print(
            f"  rejected: {name} (unguarded: no error, bit-exact vs the view's values "
            f"{(got.view(-1) == intended.view(-1)).float().mean().item():.4f}, max ulps {worst})"
        )
    short = {
        "key with HC-1 copies": ("key", key[:, : HC - 1], "key must be"),
        "value with D/2 columns": ("value", value[:, : DIM // 2], "value must be"),
        "q_weight with HC-1 rows": ("qw", qw[: HC - 1], "q_weight must be"),
    }
    for name, (arg, view, match) in short.items():
        args = dict(base)
        args[arg] = view
        got = _call(args, token_mask=mask)
        torch.cuda.synchronize()
        assert torch.equal(got, intended), name  # it used the parent's data beyond the view
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                _call(args, token_mask=mask)
        print(f"  rejected: {name} (unguarded: no error, read the parent buffer past the view)")
    got = op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask[: t_len // 2])
    torch.cuda.synchronize()
    assert torch.equal(got, intended)
    with pytest.raises(AssertionError, match="token_mask must be"):
        with validating(op):
            op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask[: t_len // 2])
    print(
        f"  rejected: token_mask with T/2 entries (unguarded read {t_len - t_len // 2} entries past it)"
    )
    ob = torch.full((t_len * 2, HC, DIM), 7.0, dtype=torch.bfloat16, device=DEV)
    op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask, out=ob[: t_len // 2])
    torch.cuda.synchronize()
    assert torch.equal(ob[:t_len], intended) and (ob[t_len:] == 7).all()
    with pytest.raises(AssertionError, match="out must be"):
        with validating(op):
            op(x, key, value, qw, kw, EPS, CLAMP, token_mask=mask, out=ob[: t_len // 2])
    print(f"  rejected: out with T/2 rows (unguarded wrote {t_len - t_len // 2} rows past it)")
    gb = torch.full((t_len * 2, HC), 7.0, device=DEV)
    op(x, key, value, qw, kw, EPS, CLAMP, gate_out=gb[: t_len // 2])
    torch.cuda.synchronize()
    assert (gb[t_len // 2 : t_len] != 7).all() and (gb[t_len:] == 7).all()
    with pytest.raises(AssertionError, match="gate_out must be"):
        with validating(op):
            op(x, key, value, qw, kw, EPS, CLAMP, gate_out=gb[: t_len // 2])
    print("  rejected: gate_out with T/2 rows (unguarded wrote past it)")
    # CPU tensors are refused by Triton's own launcher, loudly
    with pytest.raises(ValueError, match="cannot be accessed from Triton") as err:
        op(x.cpu(), key, value, qw, kw, EPS, CLAMP)
    print(f"  CPU x: ValueError: {err.value}")


# ─── golden layer dumps (layers 1 and 14), key/value rebuilt from the checkpoint ───


def _golden_available() -> bool:
    return all(
        os.path.isfile(os.path.join(_GOLDEN_DIR, c, "layers", "L14.pt")) for c in _GOLDEN_CASES
    ) and os.path.isfile(os.path.join(_CKPT_DIR, "model.safetensors.index.json"))


def _skip_without_golden():
    if not _golden_available():
        pytest.skip(f"golden dumps ({_GOLDEN_DIR}) or checkpoint ({_CKPT_DIR}) not present")


@functools.lru_cache(maxsize=None)
def _st_entry(name: str):
    """(file, absolute byte offset, dtype string, shape) of one safetensors tensor."""
    with open(os.path.join(_CKPT_DIR, "model.safetensors.index.json")) as f:
        fname = os.path.join(_CKPT_DIR, json.load(f)["weight_map"][name])
    with open(fname, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    m = hdr[name]
    return fname, 8 + n + m["data_offsets"][0], m["dtype"], tuple(m["shape"])


def _st_rows(name: str, rows: np.ndarray) -> torch.Tensor:
    """Rows of a 2-D tensor as raw bytes, through a memory map (only the touched pages are read)."""
    fname, off, _, shape = _st_entry(name)
    table = np.memmap(
        fname, dtype=np.uint8, mode="r", offset=off, shape=(shape[0], _row_bytes(name))
    )
    return torch.from_numpy(np.ascontiguousarray(table[rows]))


def _row_bytes(name):
    _, _, dt, shape = _st_entry(name)
    return shape[1] * {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2}[dt]


def _st_tensor(name: str) -> torch.Tensor:
    fname, off, dt, shape = _st_entry(name)
    rows = np.arange(shape[0])
    raw = _st_rows(name, rows)
    return raw.view(
        {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "BF16": torch.bfloat16}[dt]
    ).reshape(shape)


@contextmanager
def _ieee_fp32_matmul():
    """The golden reference runs its fp32 GEMMs with TF32 off."""
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    if prev_prec is not None:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        if prev_prec is not None:
            torch.backends.cuda.matmul.fp32_precision = prev_prec


def _e8m0(u8):
    """UE8M0 byte b -> 2^(b-127) from the fp32 exponent bits (b=0 -> 2^-127)."""
    b = u8.to(torch.int32)
    bits = torch.where(b == 0, torch.full_like(b, 0x00400000), b << 23)
    return bits.view(torch.float32)


def _fp8_act_qdq(x):
    """The reference's FP8 1x32 activation quantize-dequantize: amax floored at 1e-4,
    scale 2^ceil(log2(amax / 448)) from the fp32 bits, E4M3 of the clamped quotient."""
    xf = x.float().unflatten(-1, (-1, 32))
    v = xf.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = v.contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    sc = ((e + 127) << 23).view(torch.float32)
    q = (xf / sc.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * sc.unsqueeze(-1)).flatten(-2)


@functools.lru_cache(maxsize=None)
def _layer_weights(layer: int):
    """(wkv dequantized to fp32 on the GPU, q_weight, k_weight) of one Engram layer."""
    p = f"layers.{layer}.engram."
    ww = _st_tensor(p + "wkv.weight").to(DEV)
    ws = _st_tensor(p + "wkv.scale").to(DEV)
    n, k = ww.shape
    sc = _e8m0(ws).repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, :k]
    wkv = ww.float() * sc
    return wkv, _st_tensor(p + "q_weight").to(DEV), _st_tensor(p + "k_weight").to(DEV)


def _golden_layer(case: str, layer: int):
    """(h_in, key, value, golden gate, golden h_after_engram) with key/value rebuilt
    from the checkpoint and value checked bit-exact against the golden dump."""
    li = _ENGRAM_LAYERS.index(layer)
    hids = torch.load(os.path.join(_GOLDEN_DIR, case, "engram_hash.pt"), weights_only=True)[
        "hash_ids"
    ][:, li, :]
    t_len, lookups = hids.shape
    uniq, inv = torch.unique(hids.reshape(-1), return_inverse=True)
    p = f"layers.{layer}.engram.embed."
    w = _st_rows(p + "weight", uniq.numpy()).view(torch.float8_e4m3fn).to(DEV)
    s = _e8m0(_st_rows(p + "scale", uniq.numpy()).to(DEV))
    rows = (w.float().unflatten(-1, (-1, 32)) * s.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    gathered = rows[inv.to(DEV)].view(t_len, lookups * rows.shape[-1])
    wkv, _, _ = _layer_weights(layer)
    with _ieee_fp32_matmul():
        kv = torch.matmul(_fp8_act_qdq(gathered), wkv.t()).to(torch.bfloat16)
    key = kv[:, : HC * DIM].view(t_len, HC, DIM)
    value = kv[:, HC * DIM :]
    g = torch.load(
        os.path.join(_GOLDEN_DIR, case, "layers", f"L{layer:02d}.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    gold_value = g["engram_value"].to(DEV)
    assert torch.equal(value, gold_value), f"{case} L{layer}: rebuilt value != golden engram_value"
    return g["h_in"].to(DEV), key, value, g["engram_gate"].to(DEV), g["h_after_engram"].to(DEV)


@pytest.mark.parametrize("layer", _ENGRAM_LAYERS)
@pytest.mark.parametrize("case", _GOLDEN_CASES)
def test_golden_layer(case, layer):
    _skip_without_golden()
    h_in, key, value, gold_gate, gold_out = _golden_layer(case, layer)
    _, qw, kw = _layer_weights(layer)
    t_len = h_in.shape[0]
    gate = torch.empty(t_len, HC, device=DEV)
    out = op(h_in, key, value, qw, kw, EPS, CLAMP, gate_out=gate)
    ref_out, ref_gate = _ref(h_in, key, value, qw, kw)
    # the torch reference rebuilt here is the golden computation (same ops, same inputs)
    same = torch.equal(ref_gate, gold_gate) and torch.equal(ref_out, gold_out)
    torch.testing.assert_close(ref_gate, gold_gate, rtol=0, atol=0)
    torch.testing.assert_close(ref_out, gold_out, rtol=0, atol=0)
    _check(out, gate, gold_out, gold_gate, h_in, value)
    rms = h_in.float().square().mean(-1).sqrt()
    print(
        f"\n  {case} L{layer:02d} T={t_len}: value bit-exact vs golden, "
        f"reference rebuilt here bit-exact vs golden {same}; "
        f"kernel vs golden {_stats(out, gate, gold_out, gold_gate)}; "
        f"gate range [{gold_gate.min().item():.4f}, {gold_gate.max().item():.4f}]; per-copy RMS of h_in (token 1) "
        f"{[round(v, 3) for v in rms[min(1, t_len - 1)].tolist()]}"
    )
    _assert_controls_fail(
        out,
        gate,
        h_in,
        key,
        value,
        qw,
        kw,
        None,
        ["joint_rms", "no_signed_sqrt"],
        f"golden {case} L{layer}",
    )
    # masking on real data: every third token shut, those rows must be h_in exactly
    mask = torch.arange(t_len, device=DEV) % 3 != 1
    gm = torch.empty(t_len, HC, device=DEV)
    om = op(h_in, key, value, qw, kw, EPS, CLAMP, token_mask=mask, gate_out=gm)
    assert torch.equal(om[~mask], h_in[~mask]) and torch.equal(om[mask], out[mask])
    assert torch.equal(gm[~mask], torch.zeros_like(gm[~mask]))
    with pytest.raises(AssertionError):
        _check(
            om, gm, gold_out, gold_gate, h_in, value
        )  # the unmasked golden = the mask-ignored control
    print(
        f"  control golden {case} L{layer} mask_ignored: rejected; {_stats(om, gm, gold_out, gold_gate)}"
    )
