# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive `torch.ops.trtllm.flashinfer_rmsnorm` at DeepSeek-V4.1-Flash's norm geometry.

The existing catalog entry is certified at eps 1e-5 / 1e-6 and widths
4096/5120/2048/128. This checkpoint uses **`rms_norm_eps = 1e-20`** (read from
its own `config.json`) at four distinct widths, and 1e-20 is fourteen orders
below anything the entry has tested. That is not a cosmetic difference: epsilon
is the one term in RMS norm whose handling a kernel can quietly get wrong --
dropped, clamped to a minimum, or downcast on its way through the binding -- and
on ordinary activations the mistake is invisible because `mean(x^2)` is O(1) and
swamps any epsilon at all.

So the questions measured here are:

  * does the kernel agree with an fp32 reference at eps 1e-20 across every V4.1
    width, row count and dtype;
  * is eps honoured AT ITS STATED MAGNITUDE, not merely nonzero -- driven by
    shrinking the input until `mean(x^2)` is comparable to 1e-20, where an
    eps-dropping kernel and an eps-honouring one disagree by a large factor;
  * what an all-zero row does, since `rsqrt(0 + 0)` is `inf` and `0 * inf` is
    NaN -- an eps-dropping kernel turns a zero row into NaN;
  * whether PDL (on by default in the trtllm op) changes any of it.

Reads nothing from the catalog and asserts nothing: it prints measurements.
"""

from __future__ import annotations

import os

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

# The reference is the thing a tolerance is judged against, so it must be true
# fp32 and never tf32.
torch.backends.cuda.matmul.allow_tf32 = False

#: This checkpoint's own epsilon, from `config.json` -> `text_config.rms_norm_eps`.
V41_EPS = 1e-20

#: Every distinct RMSNorm width in the V4.1 text backbone, derived from the
#: checkpoint's `inference/model.py` rather than assumed:
#:   RMSNorm(args.dim)            -> attn_norm, ffn_norm (per layer), final norm
#:   RMSNorm(q_lora_rank)         -> q_norm
#:   RMSNorm(head_dim)            -> kv_norm, and the per-layer compressor norm
#:   RMSNorm(index_head_dim)      -> indexer k_norm
V41_WIDTHS = [
    (5120, "attn_norm / ffn_norm / final norm (args.dim)"),
    (1280, "q_norm (q_lora_rank)"),
    (512, "kv_norm + compressor norm (head_dim)"),
    (128, "indexer k_norm (index_head_dim)"),
]

#: Row counts a served target reaches: single decode, captured decode batches
#: and prefill chunks.
ROW_BUCKETS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]


def _ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32-accumulated RMS norm, exactly as the checkpoint's own `RMSNorm.forward` does it.

    Written from the reference's algebra (upcast, square-mean, rsqrt, scale,
    downcast) in native torch. It shares nothing with the op under test.
    """
    xf = x.float()
    normed = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (normed * w.float()).to(x.dtype)


def _call(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.ops.trtllm.flashinfer_rmsnorm(x, w, eps)


def probe_widths_and_rows() -> None:
    print("=" * 100)
    print("1. every V4.1 norm width at eps 1e-20, across the engine's row buckets")
    print("=" * 100)
    for width, what in V41_WIDTHS:
        worst = 0.0
        worst_m = -1
        for m in ROW_BUCKETS:
            gen = torch.Generator(device="cuda").manual_seed(width * 8191 + m)
            x = torch.randn(m, width, generator=gen, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(width, generator=gen, device="cuda", dtype=torch.bfloat16)
            got = _call(x, w, V41_EPS)
            ref = _ref(x, w, V41_EPS)
            d = (got.float() - ref.float()).abs().max().item()
            if d > worst:
                worst, worst_m = d, m
        print(
            f"  width {width:5d}  {what:46s} worst max_abs over "
            f"{len(ROW_BUCKETS)} row buckets = {worst:.3e} (at M={worst_m})"
        )


def probe_dtypes() -> None:
    print()
    print("=" * 100)
    # Every V4.1 norm input is BF16. fp16/fp32 are extra coverage, NOT source
    # semantics: `Compressor.forward` computes the ratio>1 pooling in fp32 but
    # captures `dtype = x.dtype` on entry and returns `self.norm(kv.to(dtype))`,
    # so the norm sees bf16 on both branches. Reading the `Linear(dtype=float32)`
    # declaration without the call site is what produced the opposite claim.
    print("2. dtypes at eps 1e-20 (V4.1 uses bf16 everywhere; fp16/fp32 are extra coverage)")
    print("=" * 100)
    for width, _what in V41_WIDTHS:
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            gen = torch.Generator(device="cuda").manual_seed(width + 17)
            x = torch.randn(64, width, generator=gen, device="cuda", dtype=dt)
            w = torch.randn(width, generator=gen, device="cuda", dtype=dt)
            try:
                got = _call(x, w, V41_EPS)
            except Exception as exc:  # noqa: BLE001 — the rejection is the measurement
                print(
                    f"  width {width:5d} {str(dt).replace('torch.', ''):>9s} "
                    f"REJECTED {type(exc).__name__}: {str(exc)[:60]}"
                )
                continue
            ref = _ref(x, w, V41_EPS)
            d = (got.float() - ref.float()).abs().max().item()
            scale = ref.float().abs().max().item()
            print(
                f"  width {width:5d} {str(dt).replace('torch.', ''):>9s} ok  "
                f"max_abs={d:.3e} scale={scale:.3e} rel={d / max(scale, 1e-30):.3e}"
            )


def probe_epsilon_is_honoured() -> None:
    """The load-bearing question: is 1e-20 USED, and at its stated magnitude?

    On ordinary activations `mean(x^2)` is O(1), so eps 1e-20 and eps 0 give
    bit-identical answers and the case proves nothing. Shrinking the input until
    `mean(x^2)` is comparable to 1e-20 separates them by a large factor, and that
    is where a dropped or clamped epsilon becomes visible.
    """
    print()
    print("=" * 100)
    print("3. IS eps=1e-20 ACTUALLY USED? kernel vs an eps-honouring and an eps-dropping reference")
    print("=" * 100)
    print("   rows are constant-magnitude c, so mean(x^2) = c^2 exactly")
    print(
        f"  {'width':>6s} {'c':>10s} {'mean(x^2)':>11s} {'|kern-ref(1e-20)|':>18s} "
        f"{'|kern-ref(0)|':>14s}  verdict"
    )
    for width, _what in V41_WIDTHS:
        for c in (1.0, 1e-9, 1e-10, 1e-11, 1e-12):
            x = torch.full((8, width), c, device="cuda", dtype=torch.bfloat16)
            w = torch.ones(width, device="cuda", dtype=torch.bfloat16)
            got = _call(x, w, V41_EPS).float()
            with_eps = _ref(x, w, V41_EPS).float()
            no_eps = _ref(x, w, 0.0).float()
            d_eps = (got - with_eps).abs().max().item()
            d_zero = (got - no_eps).abs().max().item()
            sep = (with_eps - no_eps).abs().max().item()
            if sep == 0.0:
                verdict = "cannot discriminate (eps negligible here)"
            elif d_eps < d_zero:
                verdict = f"HONOURS eps (references differ by {sep:.3e})"
            else:
                verdict = f"** IGNORES eps ** (references differ by {sep:.3e})"
            print(f"  {width:6d} {c:10.1e} {c * c:11.1e} {d_eps:18.3e} {d_zero:14.3e}  {verdict}")


def probe_zero_rows() -> None:
    """A row of exact zeros. With eps it is 0; without eps it is `0 * inf` = NaN."""
    print()
    print("=" * 100)
    print("4. all-zero rows -- the case where a dropped epsilon produces NaN instead of 0")
    print("=" * 100)
    for width, _what in V41_WIDTHS:
        for dt in (torch.bfloat16, torch.float32):
            x = torch.zeros(4, width, device="cuda", dtype=dt)
            w = torch.ones(width, device="cuda", dtype=dt)
            got = _call(x, w, V41_EPS)
            ref = _ref(x, w, V41_EPS)
            print(
                f"  width {width:5d} {str(dt).replace('torch.', ''):>9s}: "
                f"kernel nan={int(got.isnan().sum().item()):4d} "
                f"inf={int(got.isinf().sum().item()):4d} max_abs={got.float().abs().max():.3e} "
                f"| reference nan={int(ref.isnan().sum().item()):4d} "
                f"max_abs={ref.float().abs().max():.3e}"
            )
    # A single zero row mixed into ordinary rows: the kernel normalizes per row,
    # so one degenerate row must not disturb its neighbours.
    gen = torch.Generator(device="cuda").manual_seed(4242)
    x = torch.randn(8, 5120, generator=gen, device="cuda", dtype=torch.bfloat16)
    x[3] = 0.0
    w = torch.randn(5120, generator=gen, device="cuda", dtype=torch.bfloat16)
    got, ref = _call(x, w, V41_EPS), _ref(x, w, V41_EPS)
    other = [r for r in range(8) if r != 3]
    print(
        f"  mixed batch (row 3 zeroed, width 5120): zero row nan="
        f"{int(got[3].isnan().sum().item())} max_abs={got[3].float().abs().max():.3e}; "
        f"other rows max_abs_err={(got[other].float() - ref[other].float()).abs().max():.3e}"
    )


def probe_pdl() -> None:
    """The contract says PDL affects scheduling only. Driven at this checkpoint's eps."""
    print()
    print("=" * 100)
    print("5. PDL on vs off at eps 1e-20 (contract claims scheduling-only)")
    print("=" * 100)
    prev = os.environ.get("TRTLLM_ENABLE_PDL")
    try:
        results = {}
        for setting in ("1", "0"):
            os.environ["TRTLLM_ENABLE_PDL"] = setting
            for width, _what in V41_WIDTHS:
                gen = torch.Generator(device="cuda").manual_seed(width + 99)
                x = torch.randn(129, width, generator=gen, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(width, generator=gen, device="cuda", dtype=torch.bfloat16)
                results[(setting, width)] = _call(x, w, V41_EPS)
        for width, _what in V41_WIDTHS:
            on, off = results[("1", width)], results[("0", width)]
            same = torch.equal(on, off)
            print(
                f"  width {width:5d}: PDL on vs off "
                f"{'BIT-IDENTICAL' if same else f'DIFFER by {(on.float() - off.float()).abs().max():.3e}'}"
            )
    finally:
        if prev is None:
            os.environ.pop("TRTLLM_ENABLE_PDL", None)
        else:
            os.environ["TRTLLM_ENABLE_PDL"] = prev


def probe_tolerance() -> None:
    """Which gate holds at eps 1e-20, and how far a wrong variant sits outside it."""
    print()
    print("=" * 100)
    print("6. tolerance at eps 1e-20: default gate, and the distance of two wrong variants")
    print("=" * 100)
    # Header order follows the print order below, and that is not a cosmetic
    # detail: the first draft of this table labelled the two wrong variants the
    # other way round, which read as "moving epsilon off the mean square is
    # caught by every element" when the truth is the opposite -- at eps 1e-20 on
    # ordinary data that variant is not discriminable at all.
    print(
        f"  {'width':>6s} {'M':>5s} {'out':>9s} {'scale':>10s} {'max_abs':>10s} "
        f"{'>default':>9s} {'sum-not-mean>def':>17s} {'eps-misplaced>def':>18s}"
    )
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
    for width, _what in V41_WIDTHS:
        for m in (1, 4096):
            for dt in (torch.bfloat16, torch.float32):
                gen = torch.Generator(device="cuda").manual_seed(width * 31 + m)
                x = torch.randn(m, width, generator=gen, device="cuda", dtype=dt)
                w = torch.randn(width, generator=gen, device="cuda", dtype=dt)
                got = _call(x, w, V41_EPS).float()
                ref = _ref(x, w, V41_EPS).float()
                bound = 1e-5 + rtol[dt] * ref.abs()
                diff = (got - ref).abs()
                # Wrong variant A: normalizing by sum instead of mean (a very
                # ordinary transcription slip) -- same shape, same values.
                xf = x.float()
                bad_mean = xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + V41_EPS)
                bad_mean = (bad_mean * w.float()).to(dt).float()
                # Wrong variant B: eps applied to the norm instead of the mean
                # square, i.e. sqrt(mean) + eps.
                bad_eps = xf / (xf.square().mean(-1, keepdim=True).sqrt() + V41_EPS)
                bad_eps = (bad_eps * w.float()).to(dt).float()
                print(
                    f"  {width:6d} {m:5d} {str(dt).replace('torch.', ''):>9s} "
                    f"{ref.abs().max().item():10.3e} {diff.max().item():10.3e} "
                    f"{int((diff > bound).sum().item()):9d} "
                    f"{int(((bad_mean - ref).abs() > bound).sum().item()):17d} "
                    f"{int(((bad_eps - ref).abs() > bound).sum().item()):18d}"
                )
    print("  NOTE: `eps-misplaced` is 0 everywhere BY CONSTRUCTION, not by luck --")
    print("  at eps 1e-20 with mean(x^2) ~ 1, `rsqrt(mean+eps)` and `1/(sqrt(mean)+eps)`")
    print("  agree to well inside bf16. Epsilon PLACEMENT is only separable in the")
    print("  tiny-magnitude regime of section 3, which is why that section exists.")


def probe_dtype_domain() -> None:
    """Every dtype the contract makes a claim about, driven, with the exact text quoted.

    The previous contract promised that `float64`, `int8`, `uint8` and
    `float8_e5m2` raise `KeyError` while `float8_e4m3fn` is silently ACCEPTED --
    six claims, none of them driven here, and the accepted one is the dangerous
    kind: it means a caller who forgets to dequantize gets a plausible-looking
    answer rather than an error. Each is driven below and the observed behaviour
    printed verbatim, so the contract quotes a measurement instead of a memory.
    """
    print()
    print("=" * 100)
    print("7. dtype domain -- every claim the contract makes about an input dtype, driven")
    print("=" * 100)
    width = 512
    for dt in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ):
        name = str(dt).replace("torch.", "")
        # Seeded: the accepted-dtype rows below print distances the contract
        # quotes, and an unseeded draw makes those numbers wobble run to run.
        gen = torch.Generator(device="cuda").manual_seed(2026)
        if dt in (torch.int8, torch.uint8):
            x = torch.ones(4, width, device="cuda", dtype=dt)
            w = torch.ones(width, device="cuda", dtype=dt)
        else:
            x = torch.randn(4, width, generator=gen, device="cuda").to(dt)
            w = torch.randn(width, generator=gen, device="cuda").to(dt)
        try:
            got = _call(x, w, V41_EPS)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — the rejection is the measurement
            text = " ".join(str(exc).split())
            print(f"  {name:16s} REJECTED  {type(exc).__name__}: {text[:90]}")
            continue
        # For an ACCEPTED dtype, how far the answer is from an fp32 reference
        # matters as much as the fact it returned: "accepted" and "correct" are
        # different claims and the contract must not merge them.
        ref = _ref(x.float(), w.float(), V41_EPS)
        d = (got.float() - ref).abs().max().item()
        print(
            f"  {name:16s} ACCEPTED  out_dtype={str(got.dtype).replace('torch.', ''):12s} "
            f"nan={int(got.float().isnan().sum().item()):3d} "
            f"max_abs_vs_fp32_ref={d:.3e} scale={ref.abs().max().item():.3e}"
        )


def main() -> int:
    assert torch.cuda.is_available(), "this is a GPU measurement"
    cap = torch.cuda.get_device_capability()
    import tensorrt_llm

    print(
        f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]} trtllm={tensorrt_llm.__version__}"
    )
    print(f"tensorrt_llm from {tensorrt_llm.__file__}")
    print(f"V41 rms_norm_eps = {V41_EPS:g}")
    failed = []
    for fn in (
        probe_widths_and_rows,
        probe_dtypes,
        probe_epsilon_is_honoured,
        probe_zero_rows,
        probe_pdl,
        probe_tolerance,
        probe_dtype_domain,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — one probe dying must not hide the rest
            print(f"\n!! {fn.__name__} raised {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            failed.append(fn.__name__)
    if failed:
        print(f"\nprobe INCOMPLETE: {len(failed)} section(s) raised: {', '.join(failed)}")
        return 1
    print("\nprobe complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
