# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_kda_fab entry and the verify with the forget gate computed in the kernel.

The cells run the fused [f_a | b] -> f_b projection against the entry's reference. The fused path
(`rubin_kda_fab` in f_a mode, then `rubin_kda_mtp_verify` with `fa` / `w_fb`, optionally as a PDL
launch) must reproduce the pair `rubin_kda_fab` (both outputs) + `rubin_kda_mtp_verify` with `g`
bit for bit: output, committed state, conv windows and replay caches, over chained rounds.
"""

import zlib

import pytest
import torch
from _rubin_kda_verify_inputs import MUTATED, K, bits, clone_mutated, fab_args, verify_args

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_fab import (
    rubin_kda_fab,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    rubin_kda_mtp_verify,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)


def _close_bf16(name, got, want, rel=2.0**-7):
    """Split-K fp32 accumulation against one fp32 product: one bf16 rounding apart at most."""
    scale = want.float().abs().amax().clamp_min(1e-6)
    torch.testing.assert_close(
        got.float(), want.float(), rtol=rel, atol=1e-3 * scale.item(), msg=lambda m: f"{name}: {m}"
    )


@pytest.mark.parametrize("cell", rubin_kda_fab.CELLS, ids=[c.why[:60] for c in rubin_kda_fab.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(rubin_kda_fab)
    args = fab_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = rubin_kda_fab(**args)
    ref = rubin_kda_fab.reference(**args)
    torch.cuda.synchronize()
    if args["fa_only"]:
        _close_bf16("bfa", out[:, : K + args["num_heads"]], ref[:, : K + args["num_heads"]])
        return
    (beta, gate), (ref_beta, ref_gate) = out, ref
    _close_bf16("beta", beta, ref_beta)
    # A one-ulp flip of f_a moves the f_b product by about 2^-8 of one term.
    _close_bf16("forget gate", gate, ref_gate, rel=1e-2)


@pytest.mark.parametrize("rows", [1, 8, 96, 144])
@torch.inference_mode()
def test_fa_mode_matches_the_full_kernel(rows) -> None:
    """f_a mode's [f_a | b] is bitwise what the full kernel uses (same tiling and order)."""
    spec = dict(rows=rows, heads=12, hidden=7168, fa_only=False, multi_tile=rows > 128)
    args = fab_args(spec, seed=rows)
    beta, _ = rubin_kda_fab(**args)
    bfa = rubin_kda_fab(**dict(args, fa_only=True))
    torch.cuda.synchronize()
    assert torch.equal(bits(bfa[:, K : K + 12].contiguous()), bits(beta))


def _layer(batch: int, num_spec: int, seed: int) -> tuple[dict, dict]:
    """A 12-head verify layer and its fab inputs, the token rows the fab rows."""
    args = verify_args(dict(N=batch, H=12, num_spec=num_spec, pool=2 * batch), seed)
    fab = fab_args(
        dict(rows=args["x_q"].shape[1], heads=12, hidden=7168, fa_only=False, multi_tile=False),
        seed,
    )
    return args, fab


def _pair(args: dict, fab: dict) -> torch.Tensor:
    """The unfused pair: the full fab kernel, then the verify fed its forget gate."""
    beta, gate = rubin_kda_fab(**fab)
    rows = beta.shape[0]
    return rubin_kda_mtp_verify(
        **dict(args, g=gate.view(1, rows, 12, K), fa=None, w_fb=None, beta=beta.view(1, rows, 12))
    )


def _fused(args: dict, fab: dict, use_pdl: bool) -> torch.Tensor:
    bfa = rubin_kda_fab(**dict(fab, fa_only=True))
    rows = bfa.shape[0]
    return rubin_kda_mtp_verify(
        **dict(
            args,
            g=None,
            fa=bfa[:, :K].view(1, rows, K),
            w_fb=fab["w_fb"],
            beta=bfa[:, K : K + 12].view(1, rows, 12),
            use_pdl=use_pdl,
        )
    )


@pytest.mark.parametrize("use_pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("batch,num_spec", [(1, 7), (16, 5)], ids=["TEP8-b1k7", "helix8-b16k5"])
@torch.inference_mode()
def test_fused_gate_verify_is_bitwise_the_pair(batch, num_spec, use_pdl) -> None:
    args, fab = _layer(batch, num_spec, seed=100 * batch + num_spec)
    pair, fused = clone_mutated(args), clone_mutated(args)
    gen = torch.Generator(device="cpu").manual_seed(1)
    with validating(rubin_kda_fab, rubin_kda_mtp_verify):
        for rnd in range(3):
            out_pair = _pair(pair, fab)
            out_fused = _fused(fused, fab, use_pdl)
            torch.cuda.synchronize()
            assert torch.equal(bits(out_fused), bits(out_pair)), f"round {rnd} output"
            for name in MUTATED:
                assert torch.equal(bits(fused[name]), bits(pair[name])), f"round {rnd} {name}"
            slots = args["ssm_state_indices"].long()
            accepted = torch.randint(0, num_spec + 1, (batch,), generator=gen).to(torch.int32)
            for state in (pair, fused):
                state["num_accepted_tokens"][slots] = accepted.cuda()


@torch.inference_mode()
def test_fused_gate_reads_w_fb() -> None:
    """The in-kernel forget gate comes from w_fb: another w_fb gives another output."""
    args, fab = _layer(1, 7, seed=5)
    first = _fused(clone_mutated(args), fab, use_pdl=False)
    other = dict(fab, w_fb=(fab["w_fb"].float() * -1.5).to(torch.bfloat16))
    second = _fused(clone_mutated(args), other, use_pdl=False)
    torch.cuda.synchronize()
    assert not torch.equal(bits(first), bits(second))
