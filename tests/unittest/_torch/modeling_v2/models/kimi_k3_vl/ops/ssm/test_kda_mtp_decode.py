# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the kda_mtp_decode entry: every cell of ``kda_mtp_decode.CELLS``, built as the KDA
layer's verify calls it (row-strided QKVG and [f_a | b] sections, dim-contiguous conv caches,
scattered slots with slot-indexed accepted counts, the variant's own arguments), against the
entry's fp64 reference: the output, the committed state and conv window, and the cached drafts."""

import zlib

import pytest
import torch

pytest.importorskip("cutlass")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_mtp_decode import (  # noqa: E501, E402
    kda_mtp_decode,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

K = 128
W = 4
E4M3_TOP_STEP = 32.0  # one E4M3 step in [256, 448], the binade a scaled head maximum lands in


def _driven_here() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in kda_mtp_decode.ARCHS}


pytestmark = pytest.mark.skipif(not _driven_here(), reason="kda_mtp_decode.ARCHS")


def _call_args(spec: dict, seed: int) -> dict:
    """The arguments K3KDAKernelDispatch passes for this variant, on a pool that already went
    through several verify rounds."""
    heads, requests, num_spec, variant = (
        spec["heads"],
        spec["requests"],
        spec["num_spec"],
        spec["variant"],
    )
    gen = torch.Generator().manual_seed(seed)
    dim = heads * K
    rows = requests * (1 + num_spec)
    pool = max(2 * requests, 8)

    def rn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype).cuda()

    def conv_cache():  # dim-contiguous [slots, dim, W - 1 + num_spec]
        return rn(pool, W - 1 + num_spec, dim, dtype=torch.bfloat16).transpose(1, 2)

    qkvg = rn(rows, 4 * dim, scale=0.5, dtype=torch.bfloat16)
    bfa = rn(rows, (K + heads + 7) // 8 * 8, dtype=torch.bfloat16)  # [f_a | b | pad]
    # V-first state rows whose scales spread over two orders of magnitude.
    state = rn(pool, heads, K, K, scale=0.05) * torch.logspace(-1, 1, K, device="cuda").view(
        1, 1, K, 1
    )
    slots = torch.randperm(pool, generator=gen)[:requests]
    accepted = torch.zeros(pool, dtype=torch.int32)
    accepted[slots] = torch.randint(0, num_spec + 1, (requests,), generator=gen, dtype=torch.int32)
    args = dict(
        x_q=qkvg[:, :dim].view(1, rows, heads, K),
        x_k=qkvg[:, dim : 2 * dim].view(1, rows, heads, K),
        x_v=qkvg[:, 2 * dim : 3 * dim].view(1, rows, heads, K),
        w_q=rn(dim, W, scale=0.5),
        w_k=rn(dim, W, scale=0.5),
        w_v=rn(dim, W, scale=0.5),
        cs_q=conv_cache(),
        cs_k=conv_cache(),
        cs_v=conv_cache(),
        g=None,
        beta=bfa[:, K : K + heads].view(1, rows, heads),
        A_log=rn(heads, scale=0.5),
        dt_bias=rn(dim, scale=0.5),
        recurrent_state=state,
        k_cache=rn(pool, num_spec, dim, scale=0.08, dtype=torch.bfloat16),
        g_cache=-torch.rand(pool, num_spec, dim, generator=gen).cuda() * 0.5,
        v_cache=rn(pool, num_spec, dim, dtype=torch.bfloat16),
        beta_cache=torch.rand(pool, num_spec, heads, generator=gen).to(torch.bfloat16).cuda(),
        ssm_state_indices=slots.to(torch.int32).cuda(),
        cu_seqlens=None,
        num_spec=num_spec,
        num_accepted_tokens=accepted.cuda(),
        lower_bound=-5.0,
        scale=K**-0.5,
        packed_token_layout=True,
        onorm_g=qkvg[:, 3 * dim :].view(1, rows, heads, K),
        onorm_weight=(0.75 + 0.5 * torch.rand(K, generator=gen)).cuda(),
        onorm_eps=1e-5,
        fuse_output_norm=True,
    )
    if variant == "fused_fb":
        # f_a in {-1, 0, 1} / 4 and f_b in {-2, ..., 2} / 16: every gate dot product is an
        # integer of magnitude at most 128 * 2 = 256 over 64, exact in any fp32 summation order
        # and in bf16, so the kernel's gate GEMM and the reference's agree bit for bit and the
        # state comparison measures the recurrence alone.
        fa = torch.randint(-1, 2, (rows, K), generator=gen) / 4
        bfa[:, :K] = fa.to(torch.bfloat16).cuda()
        w_fb = torch.randint(-2, 3, (dim, K), generator=gen) / 16
        args.update(fa=bfa[:, :K].view(1, rows, K), w_fb=w_fb.to(torch.bfloat16).cuda())
        args["use_pdl"] = True
    else:
        args["g"] = rn(rows, dim, dtype=torch.bfloat16).view(1, rows, heads, K)
        if variant == "bf16":
            args["use_pdl"] = True
        else:
            args["quantize_output"] = True
    return args


def _scale_offsets(rows: int, heads: int) -> torch.Tensor:
    """``[rows, heads, 4]`` byte offsets of the R128c4 UE8M0 scales of rows ``[0, rows)``."""
    r = torch.arange(rows, device="cuda")[:, None]
    h = torch.arange(heads, device="cuda")[None, :]
    base = (r // 128) * heads * 512 + h * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
    return base[..., None] + torch.arange(4, device="cuda")


def _check_mxfp8(out, ref, rows: int, heads: int, why: str) -> None:
    act, scales = out
    ref_act, ref_scales = ref
    assert act.shape == ref_act.shape and act.dtype == ref_act.dtype, why
    offsets = _scale_offsets(rows, heads)
    got_sf = scales.view(torch.uint8).reshape(-1)[offsets]
    want_sf = ref_scales[offsets]
    # The four replicas of each (row, head) scale, and the round-up power of two itself.
    assert torch.equal(got_sf, want_sf), f"{why}: UE8M0 scales"
    unit = torch.exp2(want_sf[..., 0].float() - 127)[..., None]
    got = act.float().view(rows, heads, K) * unit
    want = ref_act.float().view(rows, heads, K) * unit
    # The kernel's bf16 normalized value can land one bf16 ulp from the reference's (fp32 vs fp64
    # recurrence); re-rounded to E4M3 that is at most one step of the head's top binade, and it
    # happens to a small fraction of the elements.
    assert ((got - want).abs() <= E4M3_TOP_STEP * unit).all(), f"{why}: E4M3 values"
    differing = (act.view(torch.uint8) != ref_act.view(torch.uint8)).float().mean().item()
    assert differing <= 1e-2, f"{why}: {differing:.2%} of E4M3 codes differ"


def _check_post(args: dict, post: dict, why: str) -> None:
    # Raw bf16 conv inputs are moved, never computed: bitwise.
    for name in ("cs_q", "cs_k", "cs_v"):
        assert torch.equal(args[name], post[name]), f"{why}: {name}"
    # The fp32 state after 3xTF32 / fp32 recurrence steps, the fp32 golden's band.
    torch.testing.assert_close(
        args["recurrent_state"], post["recurrent_state"], rtol=0.0, atol=1e-4, msg=why
    )
    # The fp32 log-decay: a few fast-math exp / reciprocal roundings of values in [-5, 0].
    torch.testing.assert_close(args["g_cache"], post["g_cache"], rtol=0.0, atol=2e-5, msg=why)
    # bf16 k / v / beta rounded once from values that agree to fp32 accuracy: an element lands at
    # most one bf16 ulp away, which is two ULP units of its row's scale.
    for name in ("k_cache", "v_cache", "beta_cache"):
        assert_within_ulp(args[name], post[name], element_ulp=2.0, rms_ulp=0.25)


@pytest.mark.parametrize(
    "cell", kda_mtp_decode.CELLS, ids=[c.why[:72] for c in kda_mtp_decode.CELLS]
)
@torch.no_grad()
def test_cells(cell) -> None:
    spec = cell.spec
    args = _call_args(spec, zlib.crc32(cell.why.encode()) % 2**31)
    ref, post = kda_mtp_decode.reference(**args)
    with validating(kda_mtp_decode):
        out = kda_mtp_decode(**args)
    torch.cuda.synchronize()

    rows = spec["requests"] * (1 + spec["num_spec"])
    if spec["variant"] == "mxfp8":
        _check_mxfp8(out, ref, rows, spec["heads"], cell.why)
    else:
        assert out.shape == (1, rows, spec["heads"], K) and out.dtype == torch.bfloat16, cell.why
        # Two bf16 roundings: bf16(o) may land one ulp of its own value from the reference's,
        # which moves the normalized element by as much, and the output rounds once more. Each
        # is at most 2**-7 of the head's largest magnitude, two ULP units of the row scale
        # (2.5 measured on GB300).
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=0.25)
    _check_post(args, post, cell.why)
