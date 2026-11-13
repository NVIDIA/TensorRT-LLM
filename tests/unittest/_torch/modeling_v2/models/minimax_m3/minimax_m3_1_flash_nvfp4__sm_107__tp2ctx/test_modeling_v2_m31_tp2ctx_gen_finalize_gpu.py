# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_GEN_FINALIZE: the TRTLLM-gen NVFP4 runner's unfinalized parts finalized with the shared expert's add by the
native op equal, bit for bit, the runner's own finalize followed by shared_output.add_(routed_output); on the TEP2 CTX
expert shard (64 local experts of 128, top-4, INTER 3072) with in-kernel MiniMax2 routing, at decode-to-prefill rows."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._utils import get_sm_version

H, INTER, E, E_LOCAL, TOP_K = 6144, 3072, 128, 64, 4
MINIMAX2_ROUTING = 5
ROUTED_SCALING = 2.0
FC31_INPUT_SCALE = 448.0 * 6.0 / 8.0
FC2_INPUT_SCALE = 448.0 * 6.0 / 24.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)


def _rand_e4m3(shape, gen):
    return torch.randint(0x30, 0x40, shape, dtype=torch.uint8, device="cuda", generator=gen).view(
        torch.float8_e4m3fn
    )


@pytest.fixture(scope="module")
def experts():
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEMethod

    gen = torch.Generator(device="cuda").manual_seed(20261109)
    w31 = torch.randint(
        0, 256, (E_LOCAL, 2 * INTER, H // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    w2 = torch.randint(
        0, 256, (E_LOCAL, H, INTER // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    module = SimpleNamespace(
        w3_w1_weight=w31,
        w2_weight=w2,
        w3_w1_weight_scale=_rand_e4m3((E_LOCAL, 2 * INTER, H // 16), gen),
        w2_weight_scale=_rand_e4m3((E_LOCAL, H, INTER // 16), gen),
        is_gated_activation=True,
        bias=False,
    )
    NVFP4TRTLLMGenFusedMoEMethod()._shuffle_all_experts(module)
    base = 1.5 / (FC31_INPUT_SCALE * 2.93 * 1.3 * H**0.5)
    fc31_alpha = base * (0.8 + 0.4 * torch.rand(E_LOCAL, device="cuda", generator=gen))
    fc2_alpha = (0.8 + 0.4 * torch.rand(E_LOCAL, device="cuda", generator=gen)) / (
        FC2_INPUT_SCALE * 2.93 * 1.3 * INTER**0.5
    )
    return SimpleNamespace(
        m=module,
        fc31_alpha=fc31_alpha,
        fc2_alpha=fc2_alpha,
        bias=(0.02 * torch.randn(E, device="cuda", generator=gen)).float(),
    )


def _moe(ex, x, logits, do_finalize):
    import tensorrt_llm._torch.custom_ops  # noqa: F401

    m = ex.m
    xq, xsf = torch.ops.trtllm.fp4_quantize(
        x, torch.tensor([FC31_INPUT_SCALE], device="cuda"), 16, False, False
    )
    out = torch.ops.trtllm.fp4_block_scale_moe_runner(
        logits, ex.bias, xq, xsf.view(torch.float8_e4m3fn).flatten(),
        m.w3_w1_weight, m.w3_w1_weight_scale, None, torch.full((E_LOCAL,), 1.702, device="cuda"),
        1.0 / ex.fc31_alpha, 7.0 / ex.fc31_alpha, m.w2_weight, m.w2_weight_scale, None,
        FC2_INPUT_SCALE * ex.fc31_alpha, ex.fc31_alpha, ex.fc2_alpha,
        E, TOP_K, None, None, INTER, 0, E_LOCAL, ROUTED_SCALING, MINIMAX2_ROUTING, do_finalize, 0,
    )  # fmt: skip
    return out


@pytest.mark.parametrize("rows", [1, 37, 448, 2048, 2560, 16384])
def test_gen_finalize_bitwise(experts, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2ctx import (  # noqa: E501
        modeling,
    )

    assert hasattr(family_ops(), "minimax_m31_moe_finalize"), (
        "the family library lacks minimax_m31_moe_finalize"
    )
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, H, device="cuda", generator=gen).to(torch.bfloat16)
    logits = torch.randn(rows, E, device="cuda", generator=gen).float()
    shared = (0.5 * torch.randn(rows, H, device="cuda", generator=gen)).to(torch.bfloat16)
    finalized = _moe(experts, x, logits, True)
    finalized = finalized[0] if isinstance(finalized, (list, tuple)) else finalized
    parts = _moe(experts, x, logits, False)
    ref = shared.clone().add_(finalized)
    got = modeling._gen_finalize(rows, parts, shared.clone())
    assert got.dtype == torch.bfloat16 and got.shape == (rows, H)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_gen_finalize_rejects_finalized_output():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2ctx import (  # noqa: E501
        modeling,
    )

    shared = torch.zeros(4, H, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="TP2CTX_GEN_FINALIZE"):
        modeling._gen_finalize(4, shared.clone(), shared)


def test_kill_switch_reason(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2ctx import (  # noqa: E501
        modeling,
    )

    monkeypatch.setattr(modeling, "_GEN_FINALIZE", False)
    assert modeling._gen_finalize_reason(SimpleNamespace(), 16384, None) == "killed"


def _finalize_operands(rows, w_dtype, ld, drop, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pairs = rows * TOP_K
    fc2 = torch.randn(pairs, ld, device="cuda", generator=gen).to(torch.bfloat16)[:, :H]
    weights = torch.rand(rows, TOP_K, device="cuda", generator=gen).to(w_dtype)
    idx = torch.randperm(pairs, device="cuda", generator=gen).to(torch.int32).view(rows, TOP_K)
    if drop:
        idx[torch.rand(rows, TOP_K, device="cuda", generator=gen) < 0.2] = -1
    shared = (0.5 * torch.randn(rows, H, device="cuda", generator=gen)).to(torch.bfloat16)
    return fc2, weights, idx.contiguous(), shared


@pytest.mark.parametrize("cfg", [-1, 0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize(
    "rows,w_dtype,ld,drop,with_shared",
    [
        (1, torch.float32, H, False, True),
        (37, torch.bfloat16, H, True, True),
        (2560, torch.float32, H + 64, True, True),
        (2560, torch.float32, H, False, False),
        (16384, torch.float32, H, False, True),
    ],
)
def test_finalize_into_matches_native_op(cfg, rows, w_dtype, ld, drop, with_shared):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels import (
        m31_moe_finalize_out as fin,
    )

    assert hasattr(family_ops(), "minimax_m31_moe_finalize"), (
        "the family library lacks minimax_m31_moe_finalize"
    )
    fc2, weights, idx, shared = _finalize_operands(rows, w_dtype, ld, drop, rows + cfg)
    shared = shared if with_shared else None
    ref = family_ops().minimax_m31_moe_finalize(fc2, weights, idx, shared, cfg)
    out = torch.full((rows, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    assert fin.outside_contract(fc2, weights, idx, shared, out, cfg) is None
    assert fin.finalize_into(fc2, weights, idx, shared, out, cfg), fin.failure(
        0, w_dtype, TOP_K, with_shared, cfg
    )
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_finalize_into_contract():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels import (
        m31_moe_finalize_out as fin,
    )

    fc2, weights, idx, shared = _finalize_operands(8, torch.float32, H, False, 1)
    out = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
    assert fin.outside_contract(fc2, weights, idx, shared, out[:, :-8], 4) is not None
    assert fin.outside_contract(fc2, weights, idx, shared, shared, 4) == "out aliases shared"
    assert fin.outside_contract(fc2, weights, idx.long(), shared, out, 4) is not None
    # fc2 is [8 * TOP_K, H] contiguous here, so any [8, H] row block of it is an otherwise valid out.
    assert fin.outside_contract(fc2, weights, idx, shared, fc2[:8], 4) == "out overlaps fc2"
    assert fin.outside_contract(fc2, weights, idx, shared, fc2[8:16], 4) == "out overlaps fc2"


@pytest.mark.parametrize("rows", [448, 16384])
def test_gen_finalize_into_slot_bitwise(experts, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2ctx import (  # noqa: E501
        modeling,
    )

    assert hasattr(family_ops(), "minimax_m31_moe_finalize"), (
        "the family library lacks minimax_m31_moe_finalize"
    )
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, H, device="cuda", generator=gen).to(torch.bfloat16)
    logits = torch.randn(rows, E, device="cuda", generator=gen).float()
    shared = (0.5 * torch.randn(rows, H, device="cuda", generator=gen)).to(torch.bfloat16)
    finalized = _moe(experts, x, logits, True)
    finalized = finalized[0] if isinstance(finalized, (list, tuple)) else finalized
    parts = _moe(experts, x, logits, False)
    ref = shared.clone().add_(finalized)
    slot = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    got = modeling._gen_finalize(rows, parts, shared.clone(), out=slot)
    assert got.data_ptr() == slot.data_ptr()
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
