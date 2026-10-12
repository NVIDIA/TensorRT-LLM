# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mhc_hc_head_apply catalog entry.

What is certified is the cross-stream use: ``mixes`` and ``sqrsum`` are the
statistics of one stream (the *producer*) and ``x`` is a different stream (the
*consumer*). That is DeepSeek-V4.1's lagged ``hc_pre``: the attention input of
layer L is collapsed with the ``pre`` computed on the stream before layer L-1's
FFN, the FFN input with the ``pre`` computed on the stream before this layer's
attention, and the final-norm input with layer 39's FFN ``pre``.

Two sources of truth:

* synthetic producer/consumer pairs drawn independently, so nothing of ``x``
  can hide inside ``pre`` -- the entry's ``reference`` and ``compare`` are the
  gate, and its ``CELLS`` the certified configurations;
* the golden layer dumps of the checkpoint (``golden/``, a pure-torch port of
  the vendor reference forward), with the real ``hc_*`` parameters read from
  the checkpoint. These cases need two external directories and skip, naming
  them, when either is absent.

This file also keeps the checks a cell cannot carry: the deliberately wrong
variants (fresh instead of lagged mixes, the consuming sublayer's parameters,
the consumer's RMS statistic, no ``+eps``, copies rolled) rejected by the same
gate, the parameter-slicing and in-place bounds, CUDA-graph replay, the guard
and the op's dtype rejections, and the DeepSeek-V4.1 call at every row count
1..32768 (prefix sweep and its control, one compiled kernel per call, the
CUDA-graph capture union).
"""

import json
import os
import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.norm.mhc_hc_head_apply import (  # noqa: E501
    collapse_error_ratio,
    pre_weights,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.norm.mhc_hc_head_apply import (  # noqa: E501
    mhc_hc_head_apply as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mhc_hc_head_apply requires a CUDA device"

DEV = "cuda"
MULT = 4
HIDDEN = 5120
K = MULT * HIDDEN
NORM_EPS = 1e-20  # DeepSeek-V4.1 rms_norm_eps, used by hc_mixes
HC_EPS = 1e-6  # DeepSeek-V4.1 hc_eps
# The op is not bit-identical to torch: it is built with --use_fast_math, so
# sigmoid's exp and reciprocal and the rsqrt are the approximate hardware
# versions, and its 4-term fp32 sum is a fused-multiply-add chain. pre then
# differs from torch's by about one fp32 ulp, which moves a bf16 rounding
# decision on a small fraction of elements. Measured on sm_103: between 0.99988
# and 1.0 of elements bit-identical, synthetic and golden cases alike. The hard
# gate below sits under that and far above every wrong variant the tests reject,
# the closest of which kept 0.40 of elements bit-identical. (Dropping +eps at
# the production hc_eps kept 0.998: under this gate, but too near it to call
# that discrimination -- see test_eps_is_added_after_sigmoid.)
MIN_BITEXACT = 0.999

# Reference activations dumped from the checkpoint's own inference code; no default
# location, so the cases that compare against them skip unless this is set.
_GOLDEN_DIR = os.environ.get("MODELING_V2_DSV41_GOLDEN_DIR", "")
# The checkpoint resolves under LLM_MODELS_ROOT, as the whole-model gates' does;
# MODELING_V2_DSV41_CKPT_DIR overrides it. A case that needs it skips when it is absent.
_MODELS_ROOT = llm_models_root()
_CKPT_DIR = os.environ.get("MODELING_V2_DSV41_CKPT_DIR") or (
    str(_MODELS_ROOT / "DeepSeek-V4.1-Flash-NVFP4") if _MODELS_ROOT else ""
)
_ENGRAM_LAYERS = (1, 14)


def _producer_stats(stream: torch.Tensor, fn: torch.Tensor):
    """What the hc mixes GEMM hands the collapse: raw mixes (no rsqrt applied)
    for the first `MULT` rows of `fn`, as a contiguous [T, MULT] copy, and the
    sum of squares of the flattened stream, both fp32."""
    flat = stream.reshape(stream.shape[0], -1).float()
    with true_fp32_matmul():
        y = flat @ fn.float().t()
    return y[:, :MULT].contiguous(), flat.square().sum(-1)


def _pre(mixes, sqrsum, scale, base, eps=HC_EPS):
    return pre_weights(mixes, sqrsum, scale, base, MULT, K, NORM_EPS, eps)


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


def _args(mixes, sqrsum, x, scale, base, *, k=K, norm_eps=NORM_EPS, eps=HC_EPS):
    """The op's arguments in order, collapsing every row of `x` into a NaN-filled `out`."""
    m = x.shape[0]
    out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=DEV)
    return (mixes, sqrsum, x, out, scale, base, m, MULT, HIDDEN, k, norm_eps, eps)


def _run(mixes, sqrsum, x, scale, base, **kw):
    return op(*_args(mixes, sqrsum, x, scale, base, **kw))


def _bitexact(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a == b).float().mean().item()


def _row_rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.float(), b.float()
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)


def _assert_matches(out: torch.Tensor, ref: torch.Tensor) -> float:
    """The golden comparisons: default bf16 assert_close plus the bit-exact floor."""
    torch.testing.assert_close(out, ref)
    frac = _bitexact(out, ref)
    assert frac >= MIN_BITEXACT, f"bit-exact fraction {frac} < {MIN_BITEXACT}"
    return frac


def _assert_rejected(tag: str, out: torch.Tensor, ref: tuple) -> None:
    """A deliberately wrong pairing of output and reference must fail the entry's gate."""
    rel = _row_rel(out, ref[0])
    print(
        f"  control {tag}: max|d|={(out.float() - ref[0].float()).abs().max().item():.4g} "
        f"row-rel median={rel.median().item():.4g} max={rel.max().item():.4g} "
        f"bit-exact={_bitexact(out, ref[0]):.4f}"
    )
    with pytest.raises(AssertionError):
        op.compare(out, ref)


def _hc_params(seed: int, saturated: bool = False):
    """Checkpoint-shaped parameters: fn [24, K], scale [3], base [24] (fp32)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    fn = torch.randn(24, K, device="cuda", generator=g) * (1.0 / K**0.5)
    scale = torch.rand(3, device="cuda", generator=g) * 1.5 + 0.25
    base = torch.randn(24, device="cuda", generator=g)
    if saturated:
        # sigmoid pinned near 0 and 1 on two copies: pre ~ eps and ~ 1 + eps
        base[0], base[3] = -40.0, 40.0
    return fn, scale, base


def _stream(m: int, seed: int, row_scale: float) -> torch.Tensor:
    """bf16 [m, MULT, HIDDEN] with per-row magnitudes spread over two decades;
    for m >= 3 row 1 is all zeros (a padded token: sqrsum 0) and row 2 is an
    attention-sink-like row with a ~2e6 stream norm."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(m, MULT, HIDDEN, device="cuda", generator=g)
    s *= row_scale * torch.logspace(-1, 1, m, device="cuda").view(m, 1, 1)
    if m >= 3:
        s[1] = 0.0
        s[2] *= 2.0e6 / s[2].norm()
    return s.to(torch.bfloat16)


def _lagged_inputs(m: int, seed: int, saturated: bool = False):
    fn, scale, base = _hc_params(seed=seed, saturated=saturated)
    producer = _stream(m, seed=seed + 1, row_scale=3.0)
    consumer = _stream(m, seed=seed + 2, row_scale=1.0)
    mixes, sqrsum = _producer_stats(producer, fn)
    return mixes, sqrsum, consumer, scale, base


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    mixes, sqrsum, x, scale, base = _lagged_inputs(
        s["M"], seed=zlib.crc32(cell.why.encode()) % 2**31, saturated=s["saturated"]
    )
    if s["saturated"]:
        pre = _pre(mixes, sqrsum, scale, base)
        assert pre[:, 0].max().item() < 2 * HC_EPS and pre[:, 3].min().item() > 1.0 - 1e-6
    args = _args(mixes, sqrsum, x, scale, base, eps=s["eps"])
    out = op(*args)
    assert out is args[3], "the op writes out in place and returns it"
    op.compare(out, op.reference(*args))


# ─── synthetic: independent producer and consumer streams ──────────────────


@pytest.mark.parametrize("m", [5, 6, 32, 1000, 8192])
def test_lagged_collapse_rejects_fresh_wiring(m: int) -> None:
    """The gate the op passes rejects the V4 (fresh) wirings of the same collapse."""
    mixes, sqrsum, consumer, scale, base = _lagged_inputs(m, seed=10 * m)
    args = _args(mixes, sqrsum, consumer, scale, base)
    out = op(*args)
    op.compare(out, op.reference(*args))
    fn, _, _ = _hc_params(seed=10 * m)
    fm, fs = _producer_stats(consumer, fn)
    # fresh: the consumer's own mixes and statistic (V4 semantics)
    _assert_rejected("fresh", out, op.reference(fm, fs, *args[2:]))
    # the producer's mixes normalized by the consumer's RMS statistic
    _assert_rejected("rstd-from-x", out, op.reference(mixes, fs, *args[2:]))


def test_eps_is_added_after_sigmoid() -> None:
    m, big_eps = 64, 0.25
    mixes, sqrsum, consumer, scale, base = _lagged_inputs(m, seed=3)
    args = _args(mixes, sqrsum, consumer, scale, base, eps=big_eps)
    out = op(*args)
    op.compare(out, op.reference(*args))
    _assert_rejected("no-eps", out, op.reference(*args[:11], 0.0))
    # At the production hc_eps the same omission is below bf16 resolution, so no
    # output-level gate can see it; record how close it lands.
    args6 = _args(mixes, sqrsum, consumer, scale, base)
    out6 = op(*args6)
    ref6_noeps = op.reference(*args6[:11], 0.0)[0]
    print(
        f"  at eps={HC_EPS}: no-eps variant max|d|="
        f"{(out6.float() - ref6_noeps.float()).abs().max().item():.4g} "
        f"bit-exact={_bitexact(out6, ref6_noeps):.6f}"
    )


def test_reads_scale0_and_first_mult_bases_only() -> None:
    m = 32
    fn, scale, base = _hc_params(seed=5)
    producer = _stream(m, seed=15, row_scale=3.0)
    consumer = _stream(m, seed=25, row_scale=1.0)
    mixes, sqrsum = _producer_stats(producer, fn)
    full = _run(mixes, sqrsum, consumer, scale, base)
    trimmed = _run(mixes, sqrsum, consumer, scale[:1].contiguous(), base[:MULT].contiguous())
    assert torch.equal(full, trimmed)


def test_in_place_bounds() -> None:
    m, rows = 40, 48
    fn, scale, base = _hc_params(seed=9)
    producer = _stream(rows, seed=19, row_scale=3.0)
    consumer = _stream(rows, seed=29, row_scale=1.0)
    mixes, sqrsum = _producer_stats(producer, fn)
    snap = [t.clone() for t in (mixes, sqrsum, consumer, scale, base)]
    out = torch.full((rows, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    args = (mixes, sqrsum, consumer, out, scale, base, m, MULT, HIDDEN, K, NORM_EPS, HC_EPS)
    op(*args)
    op.compare(out, op.reference(*args))
    assert bool((out[m:] == 7.0).all()), "rows >= M of out were written"
    for before, after in zip(snap, (mixes, sqrsum, consumer, scale, base)):
        assert torch.equal(before, after), "an input was modified"
    # M = 0 launches an empty grid: out untouched, and later work in the process is unaffected
    op(mixes, sqrsum, consumer, out, scale, base, 0, MULT, HIDDEN, K, NORM_EPS, HC_EPS)
    torch.cuda.synchronize()
    assert bool((out[m:] == 7.0).all())


def test_cuda_graph_replay_reads_current_inputs() -> None:
    m = 8
    fn, scale, base = _hc_params(seed=11)
    mixes = torch.empty(m, MULT, device="cuda")
    sqrsum = torch.empty(m, device="cuda")
    x = torch.empty(m, MULT, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def fill(step: int) -> None:
        pm, ps = _producer_stats(_stream(m, seed=100 + step, row_scale=3.0), fn)
        mixes.copy_(pm)
        sqrsum.copy_(ps)
        x.copy_(_stream(m, seed=200 + step, row_scale=1.0))

    fill(0)
    args = (mixes, sqrsum, x, out, scale, base, m, MULT, HIDDEN, K, NORM_EPS, HC_EPS)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args)
    for step in range(1, 4):
        fill(step)
        graph.replay()
        torch.cuda.synchronize()
        op.compare(out, op.reference(*args))


def test_is_valid_rejects_layouts_the_op_miscomputes_silently() -> None:
    m = 16
    fn, scale, base = _hc_params(seed=13)
    producer = _stream(m, seed=33, row_scale=3.0)
    consumer = _stream(m, seed=43, row_scale=1.0)
    flat = producer.reshape(m, -1).float()
    with true_fp32_matmul():
        y24 = flat @ fn.t()  # the GEMM's full [M, 24] output
    sqrsum = flat.square().sum(-1)
    good_args = _args(y24[:, :MULT].contiguous(), sqrsum, consumer, scale, base)
    good_ref = op.reference(*good_args)
    op.compare(op(*good_args), good_ref)

    # Evidence for the guard: outside validation the op takes the natural slice y24[:, :4]
    # (row stride 24) without complaint and computes something else.
    raw = op(*_args(y24[:, :MULT], sqrsum, consumer, scale, base))
    _assert_rejected("op-on-strided-mixes", raw, good_ref)

    out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    wide = torch.empty(m, 2 * HIDDEN, dtype=torch.bfloat16, device="cuda")
    sq2 = torch.stack([sqrsum, sqrsum], dim=1)
    x_t = consumer.transpose(0, 1).contiguous().transpose(0, 1)  # same values, non-contiguous
    bad_calls = {
        "strided mixes": (y24[:, :MULT], sqrsum, consumer, out, m),
        "[M,24] mixes": (y24, sqrsum, consumer, out, m),
        "strided sqrsum": (y24[:, :MULT].contiguous(), sq2[:, 0], consumer, out, m),
        "non-contiguous x": (y24[:, :MULT].contiguous(), sqrsum, x_t, out, m),
        "strided out": (y24[:, :MULT].contiguous(), sqrsum, consumer, wide[:, :HIDDEN], m),
        "M > out rows": (y24[:, :MULT].contiguous(), sqrsum, consumer, out[: m - 2], m),
    }
    for name, (mx, sq, xx, oo, mm) in bad_calls.items():
        with pytest.raises(AssertionError, match="must be a contiguous"):
            with validating(op):
                op(mx, sq, xx, oo, scale, base, mm, MULT, HIDDEN, K, NORM_EPS, HC_EPS)
        print(f"  is_valid rejected: {name}")


def test_wrong_dtype_raises_before_launch() -> None:
    m = 4
    fn, scale, base = _hc_params(seed=17)
    mixes, sqrsum = _producer_stats(_stream(m, seed=37, row_scale=3.0), fn)
    x = _stream(m, seed=47, row_scale=1.0)
    out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    cases = {
        "mixes bf16": (
            dict(mixes=mixes.bfloat16()),
            "expected scalar type Float but found BFloat16",
        ),
        "sqrsum bf16": (
            dict(sqrsum=sqrsum.bfloat16()),
            "expected scalar type Float but found BFloat16",
        ),
        "scale bf16": (
            dict(scale=scale.bfloat16()),
            "expected scalar type Float but found BFloat16",
        ),
        "x fp32": (dict(x=x.float()), "expected scalar type BFloat16 but found Float"),
        "out fp32": (dict(out=out.float()), "expected scalar type BFloat16 but found Float"),
    }
    for name, (override, text) in cases.items():
        a = dict(mixes=mixes, sqrsum=sqrsum, x=x, out=out, scale=scale, base=base)
        a.update(override)
        with pytest.raises(RuntimeError, match=text):
            op(
                a["mixes"],
                a["sqrsum"],
                a["x"],
                a["out"],
                a["scale"],
                a["base"],
                m,
                MULT,
                HIDDEN,
                K,
                NORM_EPS,
                HC_EPS,
            )
        print(f"  {name}: raised {text!r}")
    torch.cuda.synchronize()


# ─── golden: the checkpoint's real streams and hc parameters ───────────────

_SITES = {
    # attention collapse of layer 1: pre from the stream before layer 0's FFN; x after layer 1's Engram
    "attn@1": ("attn", 1),
    "attn@20": ("attn", 20),
    # FFN collapse of layer 14: pre from the stream entering layer 14's attention (after its Engram)
    "ffn@14": ("ffn", 14),
    "ffn@39": ("ffn", 39),
    # final-norm input: pre from the stream before layer 39's FFN; x = the last layer's output
    "final": ("final", 39),
}
_GOLDEN_CASES = [(c, s) for c in ("capital_fr", "long_retrieval") for s in _SITES] + [
    ("boot_passkey", "final")
]


class _Ckpt:
    def __init__(self, root: str):
        from safetensors import safe_open

        self._root = root
        self._open = safe_open
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self._map = json.load(f)["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        with self._open(
            os.path.join(self._root, self._map[name]), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(name).cuda()

    def hc(self, layer: int, which: str):
        p = f"layers.{layer}.hc_{which}_"
        return self.get(p + "fn"), self.get(p + "scale"), self.get(p + "base")


def _golden_layer(case: str, layer: int) -> dict:
    return torch.load(
        os.path.join(_GOLDEN_DIR, case, "layers", f"L{layer:02d}.pt"),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )


def _block_input(d: dict, layer: int) -> torch.Tensor:
    return d["h_after_engram"] if layer in _ENGRAM_LAYERS else d["h_in"]


@pytest.mark.parametrize("case,site", _GOLDEN_CASES, ids=[f"{c}-{s}" for c, s in _GOLDEN_CASES])
def test_golden_lagged_collapse(case: str, site: str) -> None:
    if not os.path.isdir(os.path.join(_GOLDEN_DIR, case)) or not os.path.isfile(
        os.path.join(_CKPT_DIR, "model.safetensors.index.json")
    ):
        pytest.skip(f"golden dumps ({_GOLDEN_DIR}) or checkpoint ({_CKPT_DIR}) not present")
    ck = _Ckpt(_CKPT_DIR)
    kind, layer = _SITES[site]
    cur = _golden_layer(case, layer)
    if kind == "attn":
        prev = _golden_layer(case, layer - 1)
        producer, x = prev["h_mid"], _block_input(cur, layer)
        golden_pre = cur["pre_mix_in"]
        lag_params, own_params = ck.hc(layer - 1, "ffn"), ck.hc(layer, "attn")
        golden_after = cur["attn_in"]
        norm_w = ck.get(f"layers.{layer}.attn_norm.weight")
    elif kind == "ffn":
        producer, x = _block_input(cur, layer), cur["h_mid"]
        golden_pre = cur["attn_hc_pre"]
        lag_params, own_params = ck.hc(layer, "attn"), ck.hc(layer, "ffn")
        golden_after = cur["ffn_in"]
        norm_w = ck.get(f"layers.{layer}.ffn_norm.weight")
    else:
        fin = torch.load(
            os.path.join(_GOLDEN_DIR, case, "final.pt"),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        producer, x = cur["h_mid"], fin["h_last_out"]
        golden_pre = cur["pre_mix_out"]
        # there are no hc_head parameters: the collapse reuses layer 39's FFN pre
        lag_params, own_params = ck.hc(layer, "ffn"), ck.hc(layer, "attn")
        golden_after = fin["h_final"]
        norm_w = None
    producer, x, golden_pre, golden_after = (
        t.cuda() for t in (producer, x, golden_pre, golden_after)
    )
    fn, scale, base = lag_params
    assert fn.dtype == scale.dtype == base.dtype == torch.float32
    assert tuple(fn.shape) == (24, K) and tuple(scale.shape) == (3,) and tuple(base.shape) == (24,)

    mixes, sqrsum = _producer_stats(producer, fn)
    # the recipe reproduces the pre the reference carried to this site
    torch.testing.assert_close(_pre(mixes, sqrsum, scale, base), golden_pre)

    args = _args(mixes, sqrsum, x, scale, base)
    out = op(*args)
    ref = op.reference(*args)
    op.compare(out, ref)
    expected = ref[0]  # the torch collapse with the recipe's pre, checked equal to the golden pre
    frac = _assert_matches(out, expected)
    rel = _row_rel(out, expected)
    msg = (
        f"\n{case} {site} T={x.shape[0]}: bit-exact vs hc_pre(x, pre) {frac:.6f}, "
        f"row-rel max {rel.max().item():.3g}"
    )
    if norm_w is None:
        frac_g = _assert_matches(out, golden_after)  # golden h_final itself
        msg += f"; bit-exact vs golden h_final {frac_g:.6f}"
    else:
        normed = _ref_rmsnorm(out, norm_w, NORM_EPS)
        torch.testing.assert_close(normed, golden_after)
        msg += (
            f"; rmsnorm(out) vs golden {'attn_in' if kind == 'attn' else 'ffn_in'} "
            f"bit-exact {_bitexact(normed, golden_after):.6f}"
        )
    print(msg)

    # controls: each wires one thing the way a V4 port would, and must be rejected
    fm, fs = _producer_stats(x, own_params[0] if kind != "final" else fn)
    fresh_scale, fresh_base = (own_params[1], own_params[2]) if kind != "final" else (scale, base)
    _assert_rejected("fresh-mixes(op)", _run(fm, fs, x, fresh_scale, fresh_base), ref)
    _assert_rejected(
        "consumer-params(op)", _run(mixes, sqrsum, x, own_params[1], own_params[2]), ref
    )


# ─── The target's call at every row count 1..32768, and the capture union ──
#
# DeepSeek-V4.1's core target collapses the stream with this op twice per layer
# and once before the final norm, at M = the rows of the pass: the tokens the
# rank holds this step (any integer up to the engine's max_num_tokens, 16384 at
# most in the envelope) or, in layers 21-39 and the final collapse of a step
# that runs the decoder bounded-replay pass, that pass's rows (up to 32768).
# The DSpark drafter makes the same call at N = 5G rows. The launch is one CTA
# of 256 threads per token (CTA t reads row t of mixes, sqrsum and x and writes
# row t of out), so M only sizes the grid. The tests below check that on the
# GPU at every M from 1 to 32768.

_N_MAX = 32768
# The row counts of the target's call that the entry's cells certify (its runtime-value dumps
# plus the tops of the step and decoder-pass ranges).
_TARGET_M = tuple(sorted({c.spec["M"] for c in op.CELLS if c.why.startswith("target call")}))
# The capture union per rank. G ranges over every CUDA-graph batch size an
# envelope engine captures: every G 1..32, plus TRT-LLM's padded default list
# for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list at 256 (1..32, 64, 128, 256)
# is a subset. A captured step is generation-only, so the trunk runs at G (a
# decode step) and 6G (the DSpark target-verify step, 1 + 5 draft tokens per
# request), and the drafter at 5G.
_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_M = tuple(sorted({k * g for g in _GSET for k in (1, 5, 6)}))
assert len(_GSET) == 46 and len(_CAPTURE_M) == 114 and max(_CAPTURE_M) == 1536


def _target_args(mixes, sqrsum, x, out, scale, base, m):
    return (mixes, sqrsum, x, out, scale, base, m, MULT, HIDDEN, K, NORM_EPS, HC_EPS)


def _target_call(mixes, sqrsum, x, out, scale, base, m):
    return op(*_target_args(mixes, sqrsum, x, out, scale, base, m))


def test_target_call_every_row_count_is_the_same_per_row_function() -> None:
    """Every prefix M = 1..32768 of the target's call returns exactly the first M
    rows of the 32768-row result, which is held to the reference.

    The output buffer is NaN-filled before every call and the row after the
    last is a guard row that must stay NaN: a skipped row or a write past row
    M - 1 is a mismatch.
    """
    mixes, sqrsum, x, scale, base = _lagged_inputs(_N_MAX, seed=4242)
    full = torch.full((_N_MAX, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
    full_args = _target_args(mixes, sqrsum, x, full, scale, base, _N_MAX)
    op.compare(op(*full_args), op.reference(*full_args))
    fb = full.view(torch.int16)
    buf = torch.empty(_N_MAX + 1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    mismatch = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    guard = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    differs = torch.empty(
        _N_MAX, HIDDEN, dtype=torch.bool, device="cuda"
    )  # one buffer; .any() keeps the reduction free of a widened copy
    for m in range(1, _N_MAX + 1):
        buf[: m + 1].fill_(float("nan"))
        _target_call(mixes[:m], sqrsum[:m], x[:m], buf[:m], scale, base, m)
        mismatch[m] = torch.ne(buf[:m].view(torch.int16), fb[:m], out=differs[:m]).any()
        guard[m] = (~buf[m].isnan()).sum()
    bad = mismatch.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_N_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the guard row, first {over[:10]}"


def test_target_call_row_sweep_harness_sees_a_one_row_shift() -> None:
    """Control for the sweep: its bitwise per-row check rejects a read one row off."""
    n = _N_MAX
    mixes, sqrsum, x, scale, base = _lagged_inputs(n + 1, seed=4243)
    full = torch.empty(n, HIDDEN, dtype=torch.bfloat16, device="cuda")
    _target_call(mixes[:n], sqrsum[:n], x[:n], full, scale, base, n)
    for m in (1, 2, 7, 1024, 2048, 16384, n):
        out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        _target_call(mixes[1 : m + 1], sqrsum[1 : m + 1], x[1 : m + 1], out, scale, base, m)
        n_bad = int((out.view(torch.int16) != full.view(torch.int16)[:m]).sum().item())
        print(f"[control row shift] M={m}: {n_bad} / {m * HIDDEN} elements differ")
        assert n_bad > m * HIDDEN // 2


def test_target_call_term_gate_rejects_wrong_variants() -> None:
    """Discrimination of the entry's gate: the op passes it, and each deliberately
    wrong collapse lands far outside it.

    (a) fresh: the consumer's own mixes and statistic; (b) the producer's mixes
    with the consumer's RMS statistic; (c) another sublayer's scale / base;
    (d) pre applied to the wrong copies (rolled by one).
    """
    m = 2048
    fn, scale, base = _hc_params(seed=4245)
    producer = _stream(m, seed=4246, row_scale=3.0)
    x = _stream(m, seed=4247, row_scale=1.0)
    mixes, sqrsum = _producer_stats(producer, fn)
    out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    args = _target_args(mixes, sqrsum, x, out, scale, base, m)
    op(*args)
    ref = op.reference(*args)
    op.compare(out, ref)
    fm, fs = _producer_stats(x, fn)
    _, scale_o, base_o = _hc_params(seed=4248)
    rest = args[6:]
    wrong = {
        "fresh": op.reference(fm, fs, x, out, scale, base, *rest),
        "rstd-from-x": op.reference(mixes, fs, x, out, scale, base, *rest),
        "other-sublayer-params": op.reference(mixes, sqrsum, x, out, scale_o, base_o, *rest),
        # copy j weighted with pre[j - 1]: the mixes and bases rolled by one copy
        "copies-rolled": op.reference(
            mixes.roll(1, dims=1).contiguous(), sqrsum, x, out, scale, base[:MULT].roll(1), *rest
        ),
    }
    print(f"\ncorrect: max err / bound {collapse_error_ratio(out, ref[1], ref[2]):.3f}")
    for name, ref_w in wrong.items():
        r_w = collapse_error_ratio(out, ref_w[1], ref_w[2])
        print(f"  control {name}: max err / bound {r_w:.4g}")
        assert r_w > 100.0, (name, r_w)


_PROBE_M = tuple(
    sorted(
        set(_TARGET_M)
        | set(_CAPTURE_M)
        | {9, 64, 65, 129, 4096, 8191, 8192, 8193, 12288, 16383, 16385, 24576, 32767, _N_MAX}
    )
)


def test_target_call_one_compiled_kernel_at_every_row_count() -> None:
    """Probe: the target's call launches the same single kernel at every M (compiled ahead of time, no JIT)."""
    mixes, sqrsum, x, scale, base = _lagged_inputs(_N_MAX, seed=4244)
    out = torch.empty(_N_MAX, HIDDEN, dtype=torch.bfloat16, device="cuda")
    names = set()
    for m in _PROBE_M:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _target_call(mixes[:m], sqrsum[:m], x[:m], out[:m], scale, base, m)
            torch.cuda.synchronize()
        launched = [
            e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(launched) == 1, (m, launched)
        names.add(launched[0])
    print(f"{len(_PROBE_M)} row counts probed; kernels launched: {sorted(names)}")
    assert len(names) == 1 and "mhcHcHeadApplyKernel" in next(iter(names)), names


@pytest.mark.parametrize("m", _CAPTURE_M)
def test_target_call_capture_union_replay_is_bitwise_eager(m: int) -> None:
    """The target's call captured at M after an eager call at M, replayed on new
    data: bitwise equal to an eager call on that data.

    The replay on the capture-time data must equal the eager call before the
    capture and differ from the replay on new data, so a replay that reused
    stale inputs would fail.
    """
    mixes, sqrsum, x, scale, base = _lagged_inputs(m, seed=6000 + m)
    out = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
    _target_call(mixes, sqrsum, x, out, scale, base, m)  # the eager call at this shape
    torch.cuda.synchronize()
    out0 = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _target_call(mixes, sqrsum, x, out, scale, base, m)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, out0), m
    out_old = out.clone()
    mixes2, sqrsum2, x2, _, _ = _lagged_inputs(m, seed=7000 + m)
    mixes.copy_(mixes2)
    sqrsum.copy_(sqrsum2)
    x.copy_(x2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    eager = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
    _target_call(mixes2, sqrsum2, x2, eager, scale, base, m)
    torch.cuda.synchronize()
    assert torch.equal(out, eager), m
    assert not torch.equal(out, out_old), "the replay did not read the new inputs"
    args = _target_args(mixes2, sqrsum2, x2, out, scale, base, m)
    op.compare(out, op.reference(*args))  # the replay's output
