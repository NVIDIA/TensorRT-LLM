# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 1.3 capability search: which `mhc_*` call serves DeepSeek-V4.1-Flash's Hyper-Connections.

The installed package carries a whole family of `torch.ops.trtllm.mhc_*` ops, and
the question this probe answers is not "which of them exist" but "which of them
compute what THIS checkpoint computes". Those are different questions because the
family was built for a hyper-connection variant whose `pre` coefficients are
consumed by the sublayer that produced them, and V4.1's are not:

    residual = x
    attn_pre, attn_post, attn_comb = hc_mixes(x, hc_attn_*)   # mixes from x
    x = hc_pre(x, pre_mix)                                    # the INCOMING pre
    ...
    ffn_pre, ffn_post, ffn_comb = hc_mixes(x, hc_ffn_*)
    x = hc_pre(x, attn_pre)                                   # ATTENTION's pre
    return x, ffn_pre                                         # for the NEXT block

So a fusion that derives `pre` and applies it in the same call is a different
model, no matter how well it matches term by term. That is the distinction this
probe is built to measure rather than infer, and it is measured by driving the
ops -- the per-op verdicts below are what an entry decision has to rest on.

Section 0 also records something a reader needs before trusting any of this: the
built library exports `mhc_pre_mapping` and `mhc_split_sinkhorn`, which this
branch's `cpp/` does not declare. That mismatch is reported, not worked around.

References come from `refmods.py`, which Goal 1.1 aligned to the native
implementation, so a disagreement here is the op's and not the reference's.

Reads nothing from the catalog and asserts nothing: it prints measurements.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

sys.path.insert(
    0,
    str(
        Path(
            "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw"
            "/TensorRT-LLM/tensorrt_llm/_torch/staircase/models/deepseek_v41/targets"
            "/v41_flash/sm_103/dep4"
        )
    ),
)
import refmods as R  # noqa: E402  # ty: ignore[unresolved-import]

CKPT = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw"
    "/models/DeepSeek-V4.1-Flash"
)

#: config.json text_config: hidden_size, hc_mult, hc_sinkhorn_iters, hc_eps,
#: rms_norm_eps. `mix_hc` is `(2 + hc) * hc`.
HIDDEN = 5120
HC = 4
MIX_HC = (2 + HC) * HC
STREAM = HC * HIDDEN
SINKHORN_ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-20

#: Row counts a served target reaches.
ROWS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]

#: The candidate calls, with what each would have to serve.
CANDIDATES = (
    "mhc_gemm_sqrsum_fma",
    "mhc_split_sinkhorn",
    "mhc_pre_mapping",
    "mhc_hc_head_apply",
    "mhc_post_mapping",
    "mhc_big_fuse",
    "mhc_fused_hc",
    "mhc_fused_hc_mma_enabled",
)


def _operands(m: int, seed: int):
    """A residual stream and the three HC parameter tensors, at checkpoint dtypes."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HC, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(MIX_HC, STREAM, generator=gen, device="cuda", dtype=torch.float32) * 0.02
    scale = torch.randn(3, generator=gen, device="cuda", dtype=torch.float32)
    base = torch.randn(MIX_HC, generator=gen, device="cuda", dtype=torch.float32)
    return x, fn.contiguous(), scale.contiguous(), base.contiguous()


def _ref_mixes(x: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """`refmods.hc_mix_projection_ref` on a `[M, hc, d]` stream."""
    return R.hc_mix_projection_ref(x.unsqueeze(0), fn, NORM_EPS).squeeze(0)


def _rel(got: torch.Tensor, want: torch.Tensor) -> str:
    diff = (got.float() - want.float()).abs()
    scale = want.float().abs().max().item()
    return f"max_abs={diff.max().item():.3e} rel_scale={diff.max().item() / max(scale, 1e-30):.3e}"


# ── 0. what the build actually carries ───────────────────────────────────────


def probe_availability() -> None:
    print("=" * 104)
    print("0. which mhc ops this build registers, and whether the source tree agrees")
    print("=" * 104)
    repo = Path(
        "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM"
    )
    declared = (repo / "cpp/tensorrt_llm/thop/mhcOp.cpp").read_text()
    for name in CANDIDATES:
        op = getattr(torch.ops.trtllm, name, None)
        callable_ = op is not None and hasattr(op, "default")
        in_cpp = f'"{name}(' in declared or f'm.def("{name}' in declared
        note = ""
        if callable_ and not in_cpp:
            note = "  <-- REGISTERED BY THE BUILT LIBRARY BUT NOT DECLARED IN THIS BRANCH'S cpp/"
        print(f"  {name:28s} callable={str(callable_):5s} declared_in_cpp={str(in_cpp):5s}{note}")
        if callable_:
            # The REGISTERED schema, not the one a recovered wrapper assumes. An
            # earlier revision of this probe called `mhc_split_sinkhorn` with
            # run-1's argument list and got `missing value for argument
            # 'hc_post_mult_value'` -- the built op takes an argument that
            # wrapper does not pass, which is exactly the kind of drift a
            # recovered artifact hides.
            print(f"      schema: {torch.ops.trtllm.__getattr__(name).default._schema}")
    print(
        "\n  A name that is callable but undeclared means the mounted .so was built from a tree\n"
        "  carrying sources this branch does not. It runs today and vanishes on the next rebuild,\n"
        "  so any entry depending on one needs those sources recovered onto this branch first."
    )


# ── 1. checkpoint geometry ───────────────────────────────────────────────────


def probe_checkpoint_geometry() -> None:
    print()
    print("=" * 104)
    print("1. the HC parameters as the raw checkpoint stores them")
    print("=" * 104)
    index = json.loads((CKPT / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    keys = [k for k in wm if k.startswith("layers.2.hc_") or k.startswith("layers.3.hc_")]
    headers: dict[str, dict] = {}
    for key in keys:
        shard = wm[key]
        if shard not in headers:
            with open(CKPT / shard, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                headers[shard] = json.loads(fh.read(size))
    for key in sorted(keys):
        meta = headers[wm[key]][key]
        print(f"  {key:34s} {str(meta['shape']):18s} {meta['dtype']}")
    print(
        f"\n  derived: hc_mult={HC}, hidden={HIDDEN}, flattened stream={STREAM}, "
        f"mix_hc=(2+{HC})*{HC}={MIX_HC}, sinkhorn_iters={SINKHORN_ITERS}, "
        f"hc_eps={HC_EPS}, norm_eps={NORM_EPS}"
    )


# ── 2. the projection + square sum ───────────────────────────────────────────


def probe_gemm_sqrsum() -> None:
    print()
    print("=" * 104)
    print("2. mhc_gemm_sqrsum_fma against hc_mix_projection_ref (note: op returns the")
    print("   UNNORMALIZED projection plus the square sum; the reference folds rsqrt in)")
    print("=" * 104)
    print(f"  {'M':>6s} {'y vs unnormalized':>34s} {'r vs sum(x^2)':>30s}")
    for m in (1, 129, 4096):
        x, fn, _, _ = _operands(m, 300 + m)
        y = torch.empty(m, MIX_HC, device="cuda", dtype=torch.float32)
        r = torch.empty(m, device="cuda", dtype=torch.float32)
        flat = x.reshape(m, STREAM).contiguous()
        torch.ops.trtllm.mhc_gemm_sqrsum_fma(flat, fn, y, r, m, MIX_HC, STREAM, 0, 0)
        want_y = torch.nn.functional.linear(flat.float(), fn)
        want_r = flat.float().square().sum(-1)
        print(f"  {m:6d} {_rel(y, want_y):>34s} {_rel(r, want_r):>30s}")
    # And the composite the reference actually wants.
    x, fn, _, _ = _operands(129, 429)
    flat = x.reshape(129, STREAM).contiguous()
    y = torch.empty(129, MIX_HC, device="cuda", dtype=torch.float32)
    r = torch.empty(129, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(flat, fn, y, r, 129, MIX_HC, STREAM, 0, 0)
    mixes = y * torch.rsqrt(r / STREAM + NORM_EPS).unsqueeze(-1)
    print(f"  composite y*rsqrt(r/K+eps) vs reference mixes: {_rel(mixes, _ref_mixes(x, fn))}")


# ── 3. the split, and the two fused shapes ───────────────────────────────────


def probe_split_and_pre() -> None:
    print()
    print("=" * 104)
    print("3. the pre/post/comb split, and what each candidate does with `pre`")
    print("=" * 104)
    m = 129
    x, fn, scale, base = _operands(m, 707)
    flat = x.reshape(m, STREAM).contiguous()
    y = torch.empty(m, MIX_HC, device="cuda", dtype=torch.float32)
    r = torch.empty(m, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(flat, fn, y, r, m, MIX_HC, STREAM, 0, 0)
    mixes = y * torch.rsqrt(r / STREAM + NORM_EPS).unsqueeze(-1)
    ref_pre, ref_post, ref_comb = R.hc_split_sinkhorn_ref(
        mixes.unsqueeze(0), scale, base, HC, SINKHORN_ITERS, HC_EPS
    )
    ref_pre, ref_post, ref_comb = ref_pre.squeeze(0), ref_post.squeeze(0), ref_comb.squeeze(0)

    op = getattr(torch.ops.trtllm, "mhc_split_sinkhorn", None)
    if op is None:
        print("  mhc_split_sinkhorn: NOT CALLABLE on this build")
    else:
        pre = torch.empty(m, HC, device="cuda", dtype=torch.float32)
        post = torch.empty(m, HC, device="cuda", dtype=torch.float32)
        comb = torch.empty(m, HC, HC, device="cuda", dtype=torch.float32)
        try:
            # `hc_post_mult_value` is the `2 *` in `post = 2 * sigmoid(.)`; the
            # reference hard-codes it, the op takes it as an argument.
            # THREE separate eps, which the reference keeps distinct too:
            # `rms_eps` is the 1e-20 inside the rsqrt, `hc_pre_eps` is the
            # `+ eps` added AFTER pre's sigmoid, and `hc_sinkhorn_eps` is the
            # one the doubly-stochastic loop divides by. This checkpoint uses
            # norm_eps for the first and hc_eps for the other two.
            torch.ops.trtllm.mhc_split_sinkhorn(
                y,
                r,
                scale,
                base,
                pre,
                post,
                comb,
                m,
                STREAM,
                NORM_EPS,
                HC_EPS,
                HC_EPS,
                2.0,
                SINKHORN_ITERS,
            )
            print(f"  mhc_split_sinkhorn pre : {_rel(pre, ref_pre)}")
            print(f"  mhc_split_sinkhorn post: {_rel(post, ref_post)}")
            print(f"  mhc_split_sinkhorn comb: {_rel(comb, ref_comb)}")
            print(
                f"  comb row sums (want ~1): {comb.sum(-1).min().item():.6f} .. "
                f"{comb.sum(-1).max().item():.6f}"
            )
            print(
                f"  comb col sums (want ~1): {comb.sum(-2).min().item():.6f} .. "
                f"{comb.sum(-2).max().item():.6f}"
            )
        except Exception as exc:  # noqa: BLE001 — the signature mismatch IS the measurement
            print(f"  mhc_split_sinkhorn RAISED {type(exc).__name__}: {str(exc)[:200]}")

    # THE DECISIVE ONE. `mhc_hc_head_apply` derives `pre` from the mixes it is
    # given and immediately collapses `x` with it. Driven twice, with the SAME
    # stream but two different mix sources, to show the op is a pure function of
    # its arguments rather than hard-wired to immediate-pre -- which is what
    # decides whether V4.1's delayed schedule can use it.
    # The kernel reads `mixes[token * mult + tid]`, i.e. it strides by `mult`
    # and not by `mix_hc` -- so it wants the PRE SLICE of the projection, not the
    # whole 24-wide row. Driving it with the full row (which an earlier revision
    # of this probe did) reads the wrong elements and reports a mismatch that is
    # the caller's, not the op's.
    pre_slice = y[:, :HC].contiguous()
    out_same = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.mhc_hc_head_apply(
        y[:, :HC].contiguous(),
        r,
        x.contiguous(),
        out_same,
        scale,
        base,
        m,
        HC,
        HIDDEN,
        STREAM,
        NORM_EPS,
        HC_EPS,
    )
    del pre_slice
    want_same = R.hc_pre_ref(x.unsqueeze(0).float(), ref_pre.unsqueeze(0)).squeeze(0)
    print(f"\n  mhc_hc_head_apply(mixes(x), x) vs hc_pre(x, pre(x)):   {_rel(out_same, want_same)}")

    x2, fn2, _, _ = _operands(m, 808)
    flat2 = x2.reshape(m, STREAM).contiguous()
    y2 = torch.empty(m, MIX_HC, device="cuda", dtype=torch.float32)
    r2 = torch.empty(m, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(flat2, fn2, y2, r2, m, MIX_HC, STREAM, 0, 0)
    mixes2 = y2 * torch.rsqrt(r2 / STREAM + NORM_EPS).unsqueeze(-1)
    prev_pre, _, _ = R.hc_split_sinkhorn_ref(
        mixes2.unsqueeze(0), scale, base, HC, SINKHORN_ITERS, HC_EPS
    )
    out_delayed = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.mhc_hc_head_apply(
        y2[:, :HC].contiguous(),
        r2,
        x.contiguous(),
        out_delayed,
        scale,
        base,
        m,
        HC,
        HIDDEN,
        STREAM,
        NORM_EPS,
        HC_EPS,
    )
    want_delayed = R.hc_pre_ref(x.unsqueeze(0).float(), prev_pre).squeeze(0)
    print(
        f"  mhc_hc_head_apply(mixes(x2), x) vs hc_pre(x, pre(x2)): {_rel(out_delayed, want_delayed)}"
    )
    print(
        "  -> if BOTH rows match, the op is a pure function of (mixes, sqrsum, x) and V4.1's\n"
        "     DELAYED pre is expressible by threading the previous sublayer's accumulators in."
    )

    op = getattr(torch.ops.trtllm, "mhc_pre_mapping", None)
    if op is None:
        print("  mhc_pre_mapping: NOT CALLABLE on this build")
    else:
        out = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        try:
            torch.ops.trtllm.mhc_pre_mapping(x.contiguous(), ref_pre, out, m, HIDDEN)
            print(f"  mhc_pre_mapping(x, pre) vs hc_pre_ref:                {_rel(out, want_same)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  mhc_pre_mapping RAISED {type(exc).__name__}: {str(exc)[:200]}")


# ── 4. the post map ──────────────────────────────────────────────────────────


def probe_post_mapping() -> None:
    print()
    print("=" * 104)
    print("4. mhc_post_mapping against hc_post_ref, including the comb orientation")
    print("=" * 104)
    print(f"  {'M':>6s} {'out vs reference':>34s} {'transposed-comb control':>34s}")
    for m in (1, 129, 4096):
        gen = torch.Generator(device="cuda").manual_seed(900 + m)
        residual = torch.randn(m, HC, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
        sub = torch.randn(m, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
        post = torch.rand(m, HC, generator=gen, device="cuda", dtype=torch.float32) * 2
        comb = torch.rand(m, HC, HC, generator=gen, device="cuda", dtype=torch.float32)
        comb = comb / comb.sum(-1, keepdim=True)
        out = torch.empty(m, HC, HIDDEN, device="cuda", dtype=torch.bfloat16)
        torch.ops.trtllm.mhc_post_mapping(
            residual.contiguous(), sub.contiguous(), post, comb.contiguous(), out, m, HIDDEN
        )
        want = R.hc_post_ref(
            sub.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
        ).squeeze(0)
        wrong = R.hc_post_ref(
            sub.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.transpose(-1, -2).contiguous().unsqueeze(0),
        ).squeeze(0)
        print(f"  {m:6d} {_rel(out, want):>34s} {_rel(out, wrong):>34s}")
    print(
        "  -> the right column must be LARGE. If transposing comb changes nothing the\n"
        "     orientation is unobservable here and a control built on it would be blind."
    )


# ── 5. the verdict this search reached ───────────────────────────────────────


def probe_verdict() -> None:
    """State the selection, so a reader gets the conclusion without re-deriving it.

    Every number quoted here comes from the sections above, on this run.
    """
    print()
    print("=" * 104)
    print("5. SELECTION for the Hyper-Connections vocabulary")
    print("=" * 104)
    rows = [
        ("mhc_gemm_sqrsum_fma", "SELECTED", "projection + square sum; composite matches ref mixes"),
        ("mhc_split_sinkhorn", "SELECTED", "pre/post/comb + Sinkhorn; matches at ~1e-7 (fp32)"),
        ("mhc_pre_mapping", "SELECTED", "delayed pre collapse; takes `pre` as state, not mixes"),
        ("mhc_post_mapping", "SELECTED", "expand + residual mix; V4.1's comb orientation"),
        ("mhc_hc_head_apply", "rejected", "fits, but see note 1"),
        ("mhc_big_fuse", "rejected", "immediate-pre by construction; see note 2"),
        ("mhc_fused_hc", "rejected", "immediate-pre by construction; see note 2"),
    ]
    for name, verdict_, why in rows:
        print(f"  {name:24s} {verdict_:9s} {why}")
    print("")
    print(
        "  NOTE 1 -- `mhc_hc_head_apply` is a real candidate and it MATCHES: it derives"
        " `pre` from whatever (mixes, sqrsum) it is handed and applies it to whatever"
        " stream it is handed, so V4.1's delayed schedule is expressible by threading"
        " the PREVIOUS sublayer's accumulators in -- measured above on both the"
        " immediate and the delayed drive. It loses to `mhc_pre_mapping` on STATE,"
        " which is the catalog's stated tie-breaker: carrying `pre` is [M,4] fp32,"
        " carrying (y_acc, r_acc) is [M,24]+[M], and `mhc_split_sinkhorn` has to run"
        " anyway for post/comb -- so the fusion saves no work, it only recomputes a"
        " `pre` the split already produced."
    )
    print(
        "  NOTE 2 -- the two big fusions derive `pre` AND apply it to the stream inside"
        " one call, from the same accumulators that produce post/comb. That is"
        " immediate-pre, and it is not a parameterisation this checkpoint can opt out"
        " of: V4.1 needs the CURRENT sublayer's post/comb together with the PREVIOUS"
        " sublayer's pre, which one call reading one (y_acc, r_acc) cannot express."
        " Rejected on semantics, not on fit."
    )


def main() -> int:
    assert torch.cuda.is_available(), "this is a GPU measurement"
    cap = torch.cuda.get_device_capability()
    import tensorrt_llm

    print(
        f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]} "
        f"trtllm={tensorrt_llm.__version__} torch={torch.__version__}"
    )
    print(f"tensorrt_llm from {tensorrt_llm.__file__}")
    failed = []
    for fn in (
        probe_availability,
        probe_checkpoint_geometry,
        probe_gemm_sqrsum,
        probe_split_and_pre,
        probe_post_mapping,
        probe_verdict,
        probe_split_sinkhorn_domain,
        probe_fused_candidates,
        probe_guard_branches,
        probe_gemm_sqrsum_domain,
        probe_pre_mapping_domain,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — one section dying must not hide the rest
            print(f"\n!! {fn.__name__} raised {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            failed.append(fn.__name__)
    if failed:
        print(f"\nprobe INCOMPLETE: {len(failed)} section(s) raised: {', '.join(failed)}")
        return 1
    print("\nprobe complete")
    return 0


# ── 6. the domain of mhc_split_sinkhorn, driven not read ─────────────────────


def probe_split_sinkhorn_domain() -> None:
    """Which violations this op reports and which it accepts, so guards are earned.

    A wrapper guard is only justified when the violation is metadata-only AND was
    observed to go wrong SILENTLY here. Everything the op rejects itself stays
    out of the wrapper.
    """
    print()
    print("=" * 104)
    print("6. mhc_split_sinkhorn domain: what raises, what is accepted")
    print("=" * 104)
    m = 64
    x, fn, scale, base = _operands(m, 6006)
    flat = x.reshape(m, STREAM).contiguous()
    y = torch.empty(m, MIX_HC, device="cuda", dtype=torch.float32)
    r = torch.empty(m, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(flat, fn, y, r, m, MIX_HC, STREAM, 0, 0)

    def out_bufs(n=m):
        return (
            torch.empty(n, HC, device="cuda", dtype=torch.float32),
            torch.empty(n, HC, device="cuda", dtype=torch.float32),
            torch.empty(n, HC, HC, device="cuda", dtype=torch.float32),
        )

    def call(**kw):
        yy = kw.get("y", y)
        rr = kw.get("r", r)
        sc = kw.get("scale", scale)
        ba = kw.get("base", base)
        pre, post, comb = kw.get("outs") or out_bufs()
        torch.ops.trtllm.mhc_split_sinkhorn(
            yy,
            rr,
            sc,
            ba,
            pre,
            post,
            comb,
            kw.get("M", m),
            kw.get("K", STREAM),
            NORM_EPS,
            HC_EPS,
            HC_EPS,
            kw.get("mult", 2.0),
            kw.get("iters", SINKHORN_ITERS),
        )
        return pre, post, comb

    ref = R.hc_split_sinkhorn_ref(
        (y * torch.rsqrt(r / STREAM + NORM_EPS).unsqueeze(-1)).unsqueeze(0),
        scale,
        base,
        HC,
        SINKHORN_ITERS,
        HC_EPS,
    )
    ref = tuple(t.squeeze(0) for t in ref)

    cases = [
        ("y_acc bf16 instead of fp32", dict(y=y.to(torch.bfloat16))),
        ("hc_base too short (16 of 24)", dict(base=base[:16].contiguous())),
        ("hc_scale too short (2 of 3)", dict(scale=scale[:2].contiguous())),
        (
            "y_acc non-contiguous (column slice)",
            dict(y=torch.empty(m, 2 * MIX_HC, device="cuda")[:, :MIX_HC]),
        ),
        ("M larger than the buffers (M=2m)", dict(M=2 * m)),
        ("M=0", dict(M=0, outs=out_bufs(0))),
        ("sinkhorn_repeat=0", dict(iters=0)),
        ("K=0 (divides the square sum)", dict(K=0)),
    ]
    for name, kw in cases:
        try:
            got = call(**kw)
            same = (
                all(torch.equal(g, w) for g, w in zip(got, ref))
                if kw.get("M", m) == m and kw.get("outs") is None
                else None
            )
            finite = all(torch.isfinite(t).all().item() for t in got)
            print(f"  {name:40s} ACCEPTED  finite={finite}  bit-equal-to-correct={same}")
        except Exception as exc:  # noqa: BLE001 — the raise IS the measurement
            first = str(exc).strip().splitlines()[0]
            print(f"  {name:40s} RAISED {type(exc).__name__}: {first[:110]}")

    # `hc_post_mult_value` is a real knob, so its effect is measured against an
    # INDEPENDENT expectation. An earlier revision printed
    # `post_one / post_one.clamp_min(...)`, which is 1.0 by construction and
    # measured nothing -- it read as a passing check while asserting the
    # identity of a tensor with itself.
    _, post_one, _ = call(mult=1.0)
    _, post_two, _ = call(mult=2.0)
    ratio = post_two / post_one.clamp_min(1e-30)
    print(
        f"  hc_post_mult_value: post(2.0)/post(1.0) in "
        f"[{ratio.min().item():.6f}, {ratio.max().item():.6f}] (want exactly 2.0); "
        f"post(1.0) max={post_one.max().item():.6f} (want <= 1.0, it is a sigmoid)"
    )


# ── 7. the two fused candidates, DRIVEN ──────────────────────────────────────


def _mixes_and_acc(x: torch.Tensor, fn: torch.Tensor):
    """`(y_acc, r_acc, normalized mixes)` for a `[M, hc, d]` stream."""
    m = x.shape[0]
    flat = x.reshape(m, STREAM).contiguous()
    y = torch.empty(m, MIX_HC, device="cuda", dtype=torch.float32)
    r = torch.empty(m, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(flat, fn, y, r, m, MIX_HC, STREAM, 0, 0)
    return y, r, y * torch.rsqrt(r / STREAM + NORM_EPS).unsqueeze(-1)


def probe_fused_candidates() -> None:
    """Drive `mhc_big_fuse` and `mhc_fused_hc`; do not assert their rejection from source.

    Both were rejected in an earlier revision on a source reading alone. A
    rejection that has not been driven is a guess, and this is where it stops
    being one: each op is called at V4.1 geometry, its outputs compared against
    the reference for the IMMEDIATE-pre reading and against the DELAYED-pre one
    this checkpoint needs, and the observed numbers printed either way.
    """
    print()
    print("=" * 104)
    print("7. mhc_big_fuse and mhc_fused_hc driven at V4.1 geometry, immediate vs delayed pre")
    print("=" * 104)
    m = 129
    x, fn, scale, base = _operands(m, 1500)
    x2, fn2, _, _ = _operands(m, 1600)  # a DIFFERENT sublayer's stream
    y, r, mixes = _mixes_and_acc(x, fn)
    y2, r2, mixes2 = _mixes_and_acc(x2, fn2)

    ref_pre, ref_post, ref_comb = (
        t.squeeze(0)
        for t in R.hc_split_sinkhorn_ref(
            mixes.unsqueeze(0), scale, base, HC, SINKHORN_ITERS, HC_EPS
        )
    )
    prev_pre = R.hc_split_sinkhorn_ref(
        mixes2.unsqueeze(0), scale, base, HC, SINKHORN_ITERS, HC_EPS
    )[0]
    want_immediate = R.hc_pre_ref(x.unsqueeze(0).float(), ref_pre.unsqueeze(0)).squeeze(0)
    want_delayed = R.hc_pre_ref(x.unsqueeze(0).float(), prev_pre).squeeze(0)

    # -- mhc_big_fuse ---------------------------------------------------------
    post_mix = torch.empty(m, HC, device="cuda", dtype=torch.float32)
    comb_mix = torch.empty(m, HC, HC, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    try:
        torch.ops.trtllm.mhc_big_fuse(
            y,
            r,
            x.contiguous(),
            scale,
            base,
            post_mix,
            comb_mix,
            layer_input,
            m,
            STREAM,
            HIDDEN,
            NORM_EPS,
            HC_EPS,
            HC_EPS,
            2.0,
            SINKHORN_ITERS,
            1,
            0,
        )
        print(f"  big_fuse post_mix vs reference      {_rel(post_mix, ref_post)}")
        print(f"  big_fuse comb_mix vs reference      {_rel(comb_mix, ref_comb)}")
        print(f"  big_fuse layer_input vs IMMEDIATE   {_rel(layer_input, want_immediate)}")
        print(f"  big_fuse layer_input vs DELAYED     {_rel(layer_input, want_delayed)}")
        print(
            "  -> post/comb match and layer_input matches the IMMEDIATE reading. There is no\n"
            "     argument that redirects the collapse: `residual` is the stream it writes and\n"
            "     (y_acc, r_acc) is the only mix source, so the delayed pairing is unreachable."
        )
    except Exception as exc:  # noqa: BLE001 — an unsupported result is also a result
        print(f"  big_fuse RAISED {type(exc).__name__}: {str(exc)[:180]}")

    # -- mhc_fused_hc ---------------------------------------------------------
    # It consumes the PREVIOUS sublayer's post/comb, applies them to get
    # residual_cur, projects that, splits it, and collapses with the pre it just
    # derived. Driven with the reference's own previous-sublayer state.
    prev_post = torch.rand(m, HC, device="cuda", dtype=torch.float32) * 2
    prev_comb = torch.rand(m, HC, HC, device="cuda", dtype=torch.float32)
    prev_comb = (prev_comb / prev_comb.sum(-1, keepdim=True)).contiguous()
    x_prev_sub = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)

    residual_cur = torch.empty(m, HC, HIDDEN, device="cuda", dtype=torch.bfloat16)
    post_cur = torch.empty(m, HC, device="cuda", dtype=torch.float32)
    comb_cur = torch.empty(m, HC, HC, device="cuda", dtype=torch.float32)
    layer_in_cur = torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    y_ws = torch.empty(m, MIX_HC, device="cuda", dtype=torch.float32)
    r_ws = torch.empty(m, device="cuda", dtype=torch.float32)
    done_ws = torch.zeros(max(1, m), device="cuda", dtype=torch.int32)
    mma = bool(torch.ops.trtllm.mhc_fused_hc_mma_enabled())
    print(f"\n  mhc_fused_hc_mma_enabled() -> {mma}  (a CAPABILITY QUERY, not a compute candidate)")
    try:
        torch.ops.trtllm.mhc_fused_hc(
            x_prev_sub,
            x.contiguous(),
            prev_post,
            prev_comb,
            fn,
            scale,
            base,
            residual_cur,
            post_cur,
            comb_cur,
            layer_in_cur,
            y_ws,
            r_ws,
            done_ws,
            m,
            HIDDEN,
            HC,
            NORM_EPS,
            HC_EPS,
            HC_EPS,
            2.0,
            SINKHORN_ITERS,
            # backend 3 = fused_all_fma. `tile_n` must divide SHAPE_N=24 and may
            # NOT be 0 on this path -- driven: `tile_n=0` raises
            # `mhcFusedHcFmaAllInOneLaunch: SHAPE_N=24 not divisible by
            # tile_n=0`, so the heuristic default the other backends accept is
            # unavailable here and the caller has to pick a divisor.
            3,
            24,
            1,
            0,
            1,
            None,
            0.0,
        )
        want_residual = R.hc_post_ref(
            x_prev_sub.unsqueeze(0), x.unsqueeze(0), prev_post.unsqueeze(0), prev_comb.unsqueeze(0)
        ).squeeze(0)
        print(f"  fused_hc residual_cur vs post_ref   {_rel(residual_cur, want_residual)}")
        _, _, mixes_cur = _mixes_and_acc(residual_cur, fn)
        cur_pre, cur_post, cur_comb = (
            t.squeeze(0)
            for t in R.hc_split_sinkhorn_ref(
                mixes_cur.unsqueeze(0), scale, base, HC, SINKHORN_ITERS, HC_EPS
            )
        )
        print(f"  fused_hc post_mix_cur vs reference  {_rel(post_cur, cur_post)}")
        print(f"  fused_hc comb_mix_cur vs reference  {_rel(comb_cur, cur_comb)}")
        imm = R.hc_pre_ref(residual_cur.unsqueeze(0).float(), cur_pre.unsqueeze(0)).squeeze(0)
        dly = R.hc_pre_ref(residual_cur.unsqueeze(0).float(), prev_pre).squeeze(0)
        print(f"  fused_hc layer_input vs IMMEDIATE   {_rel(layer_in_cur, imm)}")
        print(f"  fused_hc layer_input vs DELAYED     {_rel(layer_in_cur, dly)}")
        print(
            "  -> the same shape of result: it derives `pre` from the residual it just built\n"
            "     and collapses with it, so the pre it applies and the post/comb it emits come\n"
            "     from one (y_acc, r_acc). V4.1 needs them from two different sublayers."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  fused_hc RAISED {type(exc).__name__}: {str(exc)[:180]}")

    # The MMA backends are statically instantiated per hidden size, and this
    # checkpoint's 5120 is not one of them. Driven rather than read off the
    # Python table, because a domain limit that only exists in a helper module
    # is not a limit on the op.
    try:
        torch.ops.trtllm.mhc_fused_hc(
            x_prev_sub,
            x.contiguous(),
            prev_post,
            prev_comb,
            fn,
            scale,
            base,
            residual_cur,
            post_cur,
            comb_cur,
            layer_in_cur,
            y_ws,
            r_ws,
            torch.zeros(max(1, (m + 63) // 64), device="cuda", dtype=torch.int32),
            m,
            HIDDEN,
            HC,
            NORM_EPS,
            HC_EPS,
            HC_EPS,
            2.0,
            SINKHORN_ITERS,
            2,
            0,
            1,
            0,
            1,
            None,
            0.0,
        )
        print("  fused_hc backend=2 (all_mma) at hidden=5120: ACCEPTED")
    except Exception as exc:  # noqa: BLE001
        print(
            f"  fused_hc backend=2 (all_mma) at hidden=5120 RAISED {type(exc).__name__}: "
            f"{str(exc)[:150]}"
        )
    print(
        "  -> the MMA paths are statically instantiated for a fixed hidden-size set; this\n"
        "     checkpoint's 5120 is outside it, so those backends are unavailable here on\n"
        "     domain grounds as well as on the delayed-pre semantics above."
    )


# ── 8. every wrapper guard, driven through the RAW op ────────────────────────


def probe_guard_branches() -> None:
    """One row per guard branch: does the raw op raise, or accept and go wrong?

    A guard is justified only by a SILENT failure IN TENSOR METADATA -- both
    halves, and the second half is the one two revisions of this wrapper got
    wrong in opposite directions. It once asserted exact buffer sizes and
    rejected OVERSIZED scale/base/r_acc, which a pointer kernel simply ignores;
    it later asserted `k >= 1` and `sinkhorn_repeat >= 1`, which are silent but
    are computation values rather than metadata and so authorize no assert at
    all. Each branch is driven here, and the wrapper keeps only the rows that
    are BOTH silently wrong AND metadata.

    EVERY MALFORMED ARGUMENT BELOW DIFFERS FROM THE CORRECT ONE IN METADATA
    ONLY. An earlier revision built the wrong-width and non-contiguous cases out
    of fresh random or uninitialized tensors, so their mismatch was explained by
    the values and said nothing about the stride; and it built the short `r_acc`
    out of a fresh allocation whose tail happened to hold the right bytes, which
    printed ACCEPTED/bit-equal and read as harmlessness. Both are fixed here:
    the logical values are copied from the correct call and any storage the
    kernel reads past them is POISONED to a value chosen here, so an
    accepted-and-wrong row is deterministic and attributable.

    This table is the exploratory record. The pass/fail form of the same
    evidence lives in the entry's GPU test, which asserts acceptance before it
    asserts the guard.
    """
    print()
    print("=" * 104)
    print("8. mhc_split_sinkhorn guard branches, driven through the raw op")
    print("=" * 104)
    m = 64
    x, fn, scale, base = _operands(m, 8008)
    y, r, _ = _mixes_and_acc(x, fn)
    # Finite, and far from anything a correct call reads.
    poison_y, poison_r, poison_s, poison_b = 0.5, 1.0, -3.0, -9.0

    def raw(y_a=None, r_a=None, sc=None, ba=None, k=STREAM, iters=SINKHORN_ITERS, mm=None):
        n = mm if mm is not None else m
        pre = torch.empty(n, HC, device="cuda", dtype=torch.float32)
        post = torch.empty(n, HC, device="cuda", dtype=torch.float32)
        comb = torch.empty(n, HC, HC, device="cuda", dtype=torch.float32)
        torch.ops.trtllm.mhc_split_sinkhorn(
            y if y_a is None else y_a,
            r if r_a is None else r_a,
            scale if sc is None else sc,
            base if ba is None else ba,
            pre,
            post,
            comb,
            n,
            k,
            NORM_EPS,
            HC_EPS,
            HC_EPS,
            2.0,
            iters,
        )
        return pre, post, comb

    ok = raw()

    # ---- metadata-only malformations: correct values, poisoned surroundings --
    wide_y = torch.full((m, 32), poison_y, device="cuda", dtype=torch.float32)
    wide_y[:, :MIX_HC] = y
    arena_y = torch.full((m, 2 * MIX_HC), poison_y, device="cuda", dtype=torch.float32)
    arena_y[:, :MIX_HC] = y
    arena_r = torch.full((m,), poison_r, device="cuda", dtype=torch.float32)
    arena_r[: m // 2] = r[: m // 2]
    arena_s = torch.full((3,), poison_s, device="cuda", dtype=torch.float32)
    arena_s[:2] = scale[:2]
    arena_b = torch.full((MIX_HC,), poison_b, device="cuda", dtype=torch.float32)
    arena_b[:16] = base[:16]

    # ---- oversized buffers, tails poisoned so "ignored" means ignored --------
    wide_s = torch.cat([scale, torch.full((5,), poison_s, device="cuda")]).contiguous()
    wide_b = torch.cat([base, torch.full((8,), poison_b, device="cuda")]).contiguous()
    wide_r = torch.cat([r, torch.full((8,), poison_r, device="cuda")]).contiguous()

    # ---- non-contiguous views carrying exactly the right logical values ------
    nc_y = arena_y[:, :MIX_HC]
    nc_s = torch.stack([scale, scale], dim=1)[:, 0]
    nc_b = torch.stack([base, base], dim=1)[:, 0]
    nc_r = torch.stack([r, r], dim=1)[:, 0]

    cases = [
        ("y_acc width 32, same 24 values", dict(y_a=wide_y)),
        ("y_acc non-contiguous, same values", dict(y_a=nc_y)),
        ("r_acc short (M/2), poisoned tail", dict(r_a=arena_r[: m // 2])),
        ("hc_scale short (2), poisoned tail", dict(sc=arena_s[:2])),
        ("hc_base short (16), poisoned tail", dict(ba=arena_b[:16])),
        ("hc_scale non-contiguous", dict(sc=nc_s)),
        ("hc_base non-contiguous", dict(ba=nc_b)),
        ("r_acc non-contiguous", dict(r_a=nc_r)),
        ("k zero", dict(k=0)),
        ("k negative (-20480)", dict(k=-STREAM)),
        ("sinkhorn_repeat 0", dict(iters=0)),
        ("sinkhorn_repeat negative (-1)", dict(iters=-1)),
        ("r_acc LONGER than M (M+8)", dict(r_a=wide_r)),
        ("hc_scale LONGER than 3 (8)", dict(sc=wide_s)),
        ("hc_base LONGER than 24 (32)", dict(ba=wide_b)),
    ]
    one_pass = raw(iters=1)
    print(f"  {'branch':36s} {'outcome':10s} {'finite':7s} {'==correct':10s} {'==1-pass'}")
    for name, kw in cases:
        try:
            got = raw(**kw)
            finite = all(torch.isfinite(t).all().item() for t in got)
            same = all(torch.equal(g, w) for g, w in zip(got, ok))
            clamp = all(torch.equal(g, w) for g, w in zip(got, one_pass))
            print(f"  {name:36s} {'ACCEPTED':10s} {str(finite):7s} {str(same):10s} {clamp}")
        except Exception as exc:  # noqa: BLE001 — the raise IS the measurement
            print(f"  {name:36s} RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:70]}")
    print(
        "  -> ==correct=True on an ACCEPTED row means the call was HARMLESS, and a wrapper\n"
        "     guard that rejects it is rejecting a valid call (the three LONGER rows).\n"
        "  -> SILENCE ALONE DOES NOT EARN A GUARD. The catalog authorizes a wrapper assert\n"
        "     only for a pure tensor-METADATA violation (shape/stride/dtype/device) that\n"
        "     was observed to be accepted silently -- so the eight shape/stride rows above\n"
        "     are guarded, and the four scalar-domain rows (k 0/negative, sinkhorn_repeat\n"
        "     0/-1) are NOT: `k` is the square sum's divisor and `sinkhorn_repeat` is a\n"
        "     loop bound, both computation values. They are documented in the contract's\n"
        "     Preconditions and measured in the GPU test, where the wrapper is required to\n"
        "     pass them through unchanged rather than reject them.\n"
        "  -> the silent outcomes still differ and the contract names which: finite-and-wrong\n"
        "     (the eight guarded rows and k=0), NON-finite (negative k), and silently clamped\n"
        "     to the 1-pass result (sinkhorn_repeat <= 1, whose column pass is outside the loop)."
    )


def _drive_isolated(body: str) -> None:
    """Run a CONTEXT-POISONING drive in a child process, so this one survives it.

    A misaligned or out-of-bounds fault kills the CUDA context, and every later
    launch in the same process then reports it whatever its own shapes are.
    Measured the hard way twice: once inside section 9 (a legal K=2048 read as
    "misaligned" because it followed K=1025), and once across sections, when
    section 10 inherited section 9's fault and reported a misaligned address on
    a perfectly legal M=1 call.

    Ordering the sections around these drives is not a fix -- it only works
    until someone appends another section, which is precisely what happened. So
    every such drive is isolated here instead, and section order stops
    mattering.
    """
    out = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        timeout=900,
        env=os.environ.copy(),
    )
    for line in out.stdout.splitlines():
        print(f"  {line}")
    if out.returncode != 0:
        print(f"  (child exited {out.returncode}; the fault killed its context, as documented)")


# ── 9. the domain of mhc_gemm_sqrsum_fma, driven not read ────────────────────


def _gemm_operands(m: int, n: int, k: int, seed: int):
    """A flat `[M,K]` bf16 stream and a `[N,K]` fp32 weight, the op's actual layout."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, generator=gen, device="cuda", dtype=torch.float32) * 0.02
    return x.contiguous(), w.contiguous()


def _gemm_raw(x, w, m, n, k, tile_n=0, tile_m=0):
    y = torch.empty(m, n, device="cuda", dtype=torch.float32)
    r = torch.empty(m, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(x, w, y, r, m, n, k, tile_n, tile_m)
    torch.cuda.synchronize()
    return y, r


def probe_gemm_sqrsum_domain() -> None:
    """What `mhc_gemm_sqrsum_fma` accepts, what it reports, and whether its tactic moves bits.

    The load-bearing question is the LAST one. `selectFmaTileN` returns 1 when
    `M <= 32` and 8 otherwise, so the default `tile_n=0` picks a DIFFERENT kernel
    instantiation depending on how many rows were submitted. If the instantiation
    changed a row's result, then a row's result would depend on the batch it rode
    in with -- no covering rule over `M` would exist, and the entry could only
    ever certify a sampled list. Measured here rather than argued from the source.
    """
    print()
    print("=" * 104)
    print("9. mhc_gemm_sqrsum_fma domain: references, tactics, guards")
    print("=" * 104)
    n, k = MIX_HC, STREAM

    # --- 9a. against two independent references -----------------------------
    # The kernel accumulates in fp32 (fmaf) over K terms from exactly-converted
    # bf16. fp64 is the unambiguous truth; fp32-with-TF32-off is the same
    # arithmetic in a different summation order, and the gap between them is what
    # a tolerance has to cover.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"  {'M':>6s} {'y vs fp64':>28s} {'y vs fp32(no-tf32)':>28s} {'r vs fp64':>28s}")
    for m in (1, 2, 32, 33, 129, 4096):
        x, w = _gemm_operands(m, n, k, 900 + m)
        y, r = _gemm_raw(x, w, m, n, k)
        y64 = (x.double() @ w.double().T).float()
        y32 = x.float() @ w.float().T
        r64 = x.double().square().sum(-1).float()
        print(f"  {m:6d} {_rel(y, y64):>28s} {_rel(y, y32):>28s} {_rel(r, r64):>28s}")
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # --- 9b. does the tactic move bits? -------------------------------------
    m = 129
    x, w = _gemm_operands(m, n, k, 4242)
    base = _gemm_raw(x, w, m, n, k, tile_n=1)
    print(
        f"\n  {'tile_n':>8s} {'outcome':10s} {'y bit-equal to tile_n=1':>26s} {'r bit-equal':>13s}"
    )
    for tn in (0, -1, 1, 2, 3, 4, 6, 8, 12, 24, 5, 7, 16):
        try:
            got = _gemm_raw(x, w, m, n, k, tile_n=tn)
            ye = torch.equal(got[0], base[0])
            re_ = torch.equal(got[1], base[1])
            print(f"  {tn:8d} {'ACCEPTED':10s} {str(ye):>26s} {str(re_):>13s}")
        except Exception as exc:  # noqa: BLE001 — the raise IS the measurement
            print(f"  {tn:8d} RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:66]}")

    # `tile_m` is `(void) tile_m` in the launcher -- reserved, never read.
    tm0 = _gemm_raw(x, w, m, n, k, tile_m=0)
    tm7 = _gemm_raw(x, w, m, n, k, tile_m=7)
    same = [torch.equal(a, b) for a, b in zip(tm7, tm0)]
    print(f"  tile_m=7 vs tile_m=0 bit-equal (launcher does `(void) tile_m`): {same}")

    # --- 9c. the covering rule under the M-dependent heuristic ---------------
    # M=32 and M=33 straddle selectFmaTileN's threshold, so this is where a
    # tactic-dependent result would show itself.
    big = 4096
    x, w = _gemm_operands(big, n, k, 777)
    ref = _gemm_raw(x, w, big, n, k)
    bad = []
    for mm in (0, 1, 2, 3, 31, 32, 33, 64, 128, 129, 255, 256, 1024, 4095, 4096):
        got = _gemm_raw(x[:mm].contiguous(), w, mm, n, k)
        if not (torch.equal(got[0], ref[0][:mm]) and torch.equal(got[1], ref[1][:mm])):
            bad.append(mm)
    print(
        f"  rows independent of M across the tile_n=0 threshold: {not bad} (mismatching M: {bad})"
    )

    # --- 9d. guard branches, metadata-only ----------------------------------
    m = 64
    x, w = _gemm_operands(m, n, k, 8080)
    ok = _gemm_raw(x, w, m, n, k)
    poison = 0.5
    ax = torch.full((m, 2 * k), poison, device="cuda", dtype=torch.bfloat16)
    ax[:, :k] = x
    aw = torch.full((n, 2 * k), poison, device="cuda", dtype=torch.float32)
    aw[:, :k] = w
    narrow_w = torch.full((n, k), poison, device="cuda", dtype=torch.float32)
    narrow_w[:, : k // 2] = w[:, : k // 2]
    cases = [
        ("x non-contiguous, same values", dict(x=ax[:, :k])),
        ("w non-contiguous, same values", dict(w=aw[:, :k])),
        ("w only K/2 real columns", dict(w=narrow_w[:, : k // 2], k=k // 2)),
        ("x LONGER than M rows", dict(x=torch.cat([x, x[:8]]).contiguous())),
    ]
    print(f"\n  {'branch':34s} {'outcome':10s} {'finite':7s} {'==correct'}")
    for name, kw in cases:
        try:
            got = _gemm_raw(
                kw.get("x", x), kw.get("w", w), m, n, kw.get("k", k), tile_n=kw.get("tile_n", 0)
            )
            fin = all(torch.isfinite(t).all().item() for t in got)
            eq = all(torch.equal(a, b) for a, b in zip(got, ok))
            print(f"  {name:34s} {'ACCEPTED':10s} {str(fin):7s} {eq}")
        except Exception as exc:  # noqa: BLE001 — the raise IS the measurement
            print(f"  {name:34s} RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")

    # --- 9e. dtypes and a K the vector loads cannot align --------------------
    for name, kw in (
        ("x fp32 instead of bf16", dict(x=x.float())),
        ("w bf16 instead of fp32", dict(w=w.bfloat16())),
    ):
        try:
            _gemm_raw(kw.get("x", x), kw.get("w", w), m, n, k)
            print(f"  {name:34s} ACCEPTED  <-- would need a wrapper guard")
        except Exception as exc:  # noqa: BLE001 — the raise IS the measurement
            print(f"  {name:34s} RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
    # A `K` the vectorized loads cannot align. ORDER MATTERS HERE and an earlier
    # revision of this section got it wrong: a misaligned-address fault POISONS
    # THE CUDA CONTEXT, so every launch after it reports the same error whatever
    # its own shapes are. That run showed K=2048 "misaligned" -- 2048 is a
    # multiple of 4 and perfectly legal; it was only inheriting K=1025's fault.
    # So every legal K is driven FIRST, and the illegal one is driven LAST and
    # once, after which nothing in this process is measurable.
    print()
    for kk in (1023, 2048, 4096, 20480):
        xk, wk = _gemm_operands(8, n, kk, 55 + kk)
        yk, _ = _gemm_raw(xk, wk, 8, n, kk)
        ref64 = (xk.double() @ wk.double().T).float()
        print(
            f"  K={kk:6d} (K%4={kk % 4}, K>=1024={kk >= 1024}) ACCEPTED  y vs fp64 {_rel(yk, ref64)}"
        )

    # WHERE does the fault surface -- at the op call, or at a later sync? That
    # decides whether this is a metadata violation the op accepts SILENTLY (the
    # call returns, something unrelated later takes the blame) or one it
    # reports. The catalog only authorizes a wrapper guard for the former.
    # Isolated, because the answer costs this process its CUDA context.
    _drive_isolated(
        "import torch, tensorrt_llm._torch.custom_ops  # noqa: F401\n"
        "k = 1025\n"
        "g = torch.Generator(device='cuda').manual_seed(7)\n"
        "x = torch.randn(8, k, generator=g, device='cuda', dtype=torch.bfloat16).contiguous()\n"
        "w = torch.randn(24, k, generator=g, device='cuda', dtype=torch.float32).contiguous()\n"
        "y = torch.empty(8, 24, device='cuda', dtype=torch.float32)\n"
        "r = torch.empty(8, device='cuda', dtype=torch.float32)\n"
        "try:\n"
        "    torch.ops.trtllm.mhc_gemm_sqrsum_fma(x, w, y, r, 8, 24, k, 0, 0)\n"
        "    print('K=1025 (K%4=1) the OP CALL ITSELF returned without raising')\n"
        "except Exception as exc:\n"
        "    print('K=1025 the OP CALL raised', type(exc).__name__)\n"
        "try:\n"
        "    torch.cuda.synchronize()\n"
        "    print('K=1025 and the following synchronize() was clean')\n"
        "except Exception as exc:\n"
        "    print('K=1025 the SYNC raised', type(exc).__name__, str(exc).splitlines()[0][:52])\n"
    )
    print(
        "  -> op-call silent: the call is accepted and an unrelated later line takes the\n"
        "     blame, and that child's context is dead from there on. Both halves of the\n"
        "     guard rule hold, so the wrapper asserts this one."
    )


# ── 10. the domain of mhc_pre_mapping, driven not read ───────────────────────


def _pre_operands(m: int, hidden: int, seed: int):
    """The `[M, hc, hidden]` residual copies and the `[M, hc]` coefficients the caller holds."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HC, hidden, generator=gen, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(m, HC, generator=gen, device="cuda", dtype=torch.float32) + 0.5
    return x.contiguous(), pre.contiguous()


def _pre_raw(x, pre, m, hidden):
    out = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.mhc_pre_mapping(x, pre, out, m, hidden)
    torch.cuda.synchronize()
    return out


def _pre_ref(x, pre):
    """fp32 accumulation across the HC copies, then one round to bf16 -- as the kernel does."""
    return (pre.float().unsqueeze(-1) * x.float()).sum(dim=1).to(torch.bfloat16)


def probe_pre_mapping_domain() -> None:
    """What `mhc_pre_mapping` accepts and where its vectorized path stops being legal.

    The shape of the risk is the same one `mhc_gemm_sqrsum_fma` had and it is
    worth stating before the numbers: the inner loop reads one `uint4` (16 bytes,
    8 bf16) per step and has NO scalar tail, so a `hidden_size` that is not a
    multiple of 8 runs off the end of each row. That is driven LAST here, and
    once, because a misaligned or out-of-bounds fault poisons the CUDA context
    and every measurement after it in this process is worthless.
    """
    print()
    print("=" * 104)
    print("10. mhc_pre_mapping domain: reference, covering rule, guards, vector tail")
    print("=" * 104)

    # --- 10a. against a native-torch reference ------------------------------
    print(f"  {'M':>6s} {'out vs fp32-accum-then-round':>34s} {'bit-exact':>11s}")
    for m in (1, 2, 32, 129, 4096):
        x, pre = _pre_operands(m, HIDDEN, 1200 + m)
        got = _pre_raw(x, pre, m, HIDDEN)
        want = _pre_ref(x, pre)
        print(f"  {m:6d} {_rel(got, want):>34s} {str(torch.equal(got, want)):>11s}")

    # --- 10b. the covering rule --------------------------------------------
    big = 4096
    x, pre = _pre_operands(big, HIDDEN, 606)
    ref = _pre_raw(x, pre, big, HIDDEN)
    bad = [
        mm
        for mm in (0, 1, 2, 3, 7, 8, 31, 32, 33, 64, 128, 129, 255, 256, 1024, 4095, 4096)
        if not torch.equal(
            _pre_raw(x[:mm].contiguous(), pre[:mm].contiguous(), mm, HIDDEN), ref[:mm]
        )
    ]
    print(f"  rows independent of M: {not bad} (mismatching M: {bad})")

    # --- 10c. guard branches, metadata-only --------------------------------
    m = 64
    x, pre = _pre_operands(m, HIDDEN, 8181)
    ok = _pre_raw(x, pre, m, HIDDEN)
    px, pp = 0.5, -3.0
    ax = torch.full((m, HC, 2 * HIDDEN), px, device="cuda", dtype=torch.bfloat16)
    ax[:, :, :HIDDEN] = x
    ap = torch.full((m, 2 * HC), pp, device="cuda", dtype=torch.float32)
    ap[:, :HC] = pre
    cases = [
        ("x non-contiguous, same values", dict(x=ax[:, :, :HIDDEN])),
        ("pre_mix non-contiguous, same values", dict(pre=ap[:, :HC])),
        ("x LONGER than M rows", dict(x=torch.cat([x, x[:8]]).contiguous())),
        ("pre_mix LONGER than M rows", dict(pre=torch.cat([pre, pre[:8]]).contiguous())),
    ]
    print(f"\n  {'branch':38s} {'outcome':10s} {'finite':7s} {'==correct'}")
    for name, kw in cases:
        try:
            got = _pre_raw(kw.get("x", x), kw.get("pre", pre), m, HIDDEN)
            fin = torch.isfinite(got.float()).all().item()
            print(f"  {name:38s} {'ACCEPTED':10s} {str(fin):7s} {torch.equal(got, ok)}")
        except Exception as exc:  # noqa: BLE001 — the raise IS the measurement
            print(f"  {name:38s} RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:52]}")

    for name, kw in (
        ("x fp32 instead of bf16", dict(x=x.float())),
        ("pre_mix bf16 instead of fp32", dict(pre=pre.bfloat16())),
    ):
        try:
            _pre_raw(kw.get("x", x), kw.get("pre", pre), m, HIDDEN)
            print(f"  {name:38s} ACCEPTED  <-- would need a wrapper guard")
        except Exception as exc:  # noqa: BLE001 — the raise IS the measurement
            print(f"  {name:38s} RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:52]}")

    # --- 10d. the vector step, legal widths first --------------------------
    print()
    for hidden in (8, 64, 1024, 5120):
        xh, ph = _pre_operands(4, hidden, 70 + hidden)
        got = _pre_raw(xh, ph, 4, hidden)
        print(
            f"  hidden={hidden:6d} (h%8={hidden % 8}) ACCEPTED  "
            f"{_rel(got, _pre_ref(xh, ph))}  bit-exact={torch.equal(got, _pre_ref(xh, ph))}"
        )

    # No scalar tail exists, so a hidden_size that is not a multiple of 8 reads
    # past every row. Isolated for the same reason as section 9's K drive.
    _drive_isolated(
        "import torch, tensorrt_llm._torch.custom_ops  # noqa: F401\n"
        "h = 5124\n"
        "g = torch.Generator(device='cuda').manual_seed(99)\n"
        "x = torch.randn(4, 4, h, generator=g, device='cuda', dtype=torch.bfloat16).contiguous()\n"
        "p = (torch.rand(4, 4, generator=g, device='cuda', dtype=torch.float32) + 0.5).contiguous()\n"
        "o = torch.empty(4, h, device='cuda', dtype=torch.bfloat16)\n"
        "try:\n"
        "    torch.ops.trtllm.mhc_pre_mapping(x, p, o, 4, h)\n"
        "    print('hidden=5124 (h%8=4) the OP CALL ITSELF returned without raising')\n"
        "except Exception as exc:\n"
        "    print('hidden=5124 the OP CALL raised', type(exc).__name__)\n"
        "try:\n"
        "    torch.cuda.synchronize()\n"
        "    ref = (p.float().unsqueeze(-1) * x.float()).sum(dim=1).to(torch.bfloat16)\n"
        "    d = (o.float() - ref.float()).abs().max().item()\n"
        "    print('hidden=5124 sync CLEAN; max_abs vs reference', f'{d:.3e}',"
        " 'bit-exact', torch.equal(o, ref))\n"
        "except Exception as exc:\n"
        "    print('hidden=5124 the SYNC raised', type(exc).__name__,"
        " str(exc).splitlines()[0][:50])\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
