# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_gate_up_swiglu_nvfp4 entry.

Cells against the reference (fp32 GEMM, torch's SiLU, native-torch NVFP4) in the band
of `_nvfp4_checks.assert_nvfp4_matches_reference`. The exactness claim on its own:
bitwise the unfused path built from in-tree ops on the *unpermuted* weight -- the
CUTLASS FP4 GEMM (BF16 out), `silu_and_mul`, `fp4_quantize` -- scale padding
included; and the three tiles produce the same bytes.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm import (  # noqa: E501
    _nvfp4_gemm_runner as nvfp4_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.nvfp4_gate_up_swiglu_nvfp4 import (  # noqa: E501
    nvfp4_gate_up_swiglu_nvfp4 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.quantization import (  # noqa: E501
    _nvfp4_recipes as recipes,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import assert_nvfp4_matches_reference, seed_of  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
N, K = 34816, 5120
D = N // 2

#: Block scales allowed to differ from the reference: the kernel's BF16 GEMM output
#: and the fp32 reference's differ in about 5e-4 of the elements (fp32 summation order
#: crossing a BF16 rounding midpoint), so ~1% of the 16-blocks carry one moved
#: intermediate; it moves the block's scale only if it is the block maximum and crosses
#: an E4M3 rounding boundary.
MAX_SF_FRAC = 1e-2


_WEIGHTS: dict = {}


def _weights():
    """gate_up as loaded (unpermuted NVFP4) and as the epilogue GEMM takes it."""
    if not _WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(5)
        w = (torch.randn(N, K, generator=g, device=DEV) * K**-0.5).to(torch.bfloat16)
        w_global = (2688.0 / w.abs().amax().float()).reshape(1)
        w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(w, w_global, 16, False, True)
        w_perm, w_sf_perm = recipes.permute_gate_up_for_swiglu_epilogue(w_fp4, w_sf, N, K)
        _WEIGHTS.update(
            w_fp4=w_fp4, w_sf=w_sf, w_perm=w_perm, w_sf_perm=w_sf_perm, w_global=w_global
        )
    return _WEIGHTS


def _operands(m: int, seed: int):
    """``(act_fp4, weight, act_sf, weight_scale, alpha, out_scale)`` for the permuted
    weight, and the unpermuted weight pair."""
    wt = _weights()
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, K, generator=g, device=DEV).to(torch.bfloat16)
    act_global = (2688.0 / x.abs().amax().float()).reshape(1)
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(x, act_global, 16, False, True)
    alpha = (1.0 / (act_global * wt["w_global"])).reshape(1)
    # down_proj's activation global scale; the checkpoint's are 30-112.
    out_scale = torch.tensor([33.6 if seed % 2 else 112.0], dtype=torch.float32, device=DEV)
    args = (act_fp4, wt["w_perm"], act_sf, wt["w_sf_perm"], alpha, out_scale)
    return args, (wt["w_fp4"], wt["w_sf"])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    args, _ = _operands(s["m"], seed_of(cell.why))
    out = op(*args, s["tile"])
    assert out[0].shape == (s["m"], D // 2) and out[0].dtype == torch.uint8
    ref = op.reference(*args, s["tile"])
    assert_nvfp4_matches_reference(out, ref, s["m"], D, max_sf_frac=MAX_SF_FRAC)


@pytest.mark.parametrize(
    "m,tile", [(41, 2), (129, 1), (448, 0)], ids=["m41-t2", "m129-t1", "m448-t0"]
)
def test_bitwise_the_unfused_path_on_the_unpermuted_weight(m, tile) -> None:
    """The epilogue's whole contract: the CUTLASS FP4 GEMM's BF16 output of the weight
    as loaded, `silu_and_mul`, `fp4_quantize(.., out_scale, 16)` -- same FP4 bytes and
    the same full scale buffer, padding rows included."""
    (act_fp4, _, act_sf, _, alpha, out_scale), (w_fp4, w_sf) = _operands(m, seed=m)
    fp4, sf = op(
        act_fp4, _weights()["w_perm"], act_sf, _weights()["w_sf_perm"], alpha, out_scale, tile
    )
    runner = nvfp4_runner.runner_for(torch.bfloat16)
    h = runner([act_fp4, w_fp4, act_sf, w_sf, alpha], tactic=("cutlass", -1))
    want_fp4, want_sf = torch.ops.trtllm.fp4_quantize(
        torch.ops.trtllm.silu_and_mul(h), out_scale, 16, False, True
    )
    assert torch.equal(fp4, want_fp4.view(torch.uint8))
    assert torch.equal(sf.view(torch.uint8), want_sf.view(torch.uint8))


@pytest.mark.parametrize("m,tiles", [(64, (0, 1, 2)), (447, (0, 1))], ids=["m64", "m447"])
def test_tiles_produce_the_same_bytes(m, tiles) -> None:
    args, _ = _operands(m, seed=1000 + m)
    outs = [op(*args, tile) for tile in tiles]
    for fp4, sf in outs[1:]:
        assert torch.equal(fp4, outs[0][0]) and torch.equal(sf, outs[0][1])


def test_guard_refuses_a_per_token_alpha() -> None:
    args, _ = _operands(48, seed=2)
    args = args[:4] + (args[4].repeat(48),) + args[5:]
    with validating(op), pytest.raises(AssertionError, match="exactly one element"):
        op(*args, 2)
