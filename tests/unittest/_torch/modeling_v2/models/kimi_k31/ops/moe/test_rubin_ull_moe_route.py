# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's moe_route entry: every cell's launch prepared as the target
prepares it (``SharedFront`` + ``moe.route_launches``), the front mailbox rows written as the 16
ranks' front GEMMs would publish them, and the routing checked against the FP64 reference: the
ids (exact but for near ties), the weights within BF16 rounding, the latent copy bitwise, the
NVFP4 plan against the reference quantization and bitwise against the experts' own plan writer,
the mailbox re-armed and the experts' route mailbox signalled."""

import zlib
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_ull_moe_route import (  # noqa: E501
    EXPERTS,
    LATENT,
    SENTINEL,
    TOP_K,
    WIDTH,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_ull_moe_route import (  # noqa: E501
    moe_route as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the MR8 routing kernel needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

SCALE = 2.5  # routed scaling factor
RANK = 3
# Near ties: the kernel's keys are FP32 (a correctly rounded sigmoid plus the bias); two keys
# closer than this may rank either way.
TIE = 2.0**-20


@pytest.fixture(scope="module")
def target():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_mr8 import (  # noqa: E501
        module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1 import (  # noqa: E501
        modeling as moe,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    return moe, module, moe.NvScratch(device)


def _launch(target, rows, bias, gsf):
    """``prepare_moe``'s SharedFront for ``rows`` and its routing launch. The shared expert is
    given as native (only its shape is read) so that no shared-expert GEMM is compiled: the
    routing kernel's configuration does not depend on it on GB300 (Rubin's ``release_early``
    recompiles it only where the BF16 shared expert runs beside the experts)."""
    moe, module, scratch = target
    device = scratch.device
    state = module("cutedsl.moe.prepared").MoEScratch(16, device, (rows,))
    box = NS(
        buffer=torch.empty((1, 64, WIDTH), dtype=torch.bfloat16, device=device), mc=0, rank=RANK
    )
    box.buffer.view(torch.int32).fill_(SENTINEL)
    metadata = NS(
        routed_scaling_factor=SCALE, moe_renormalize=True, gate=NS(e_score_correction_bias=bias)
    )
    packed = torch.zeros((288, 7168), dtype=torch.bfloat16, device=device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    front = module("cutedsl.moe.front").SharedFront(
        metadata, packed, box, state.front, (rows,), 4.0, 25.0,
        plan=scratch.plan, q=8, routes=scratch.routes,
        nv=moe.NvFront(gsf, wide_routing=sms <= 152),
        native_shared=NS(gate=NS(codes=torch.empty((384, 7168), device="meta"))),
    )  # fmt: skip
    weights = NS(bias=bias, scale=SCALE, renormalize=True, routed=dict(x_gsf=torch.tensor([gsf])))
    launches = moe.route_launches(
        front, state.front, box.buffer, weights, scratch.plan, scratch.routes
    )
    return launches[rows], state


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, target) -> None:
    _, module, scratch = target
    rows = cell.spec["rows"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    ties = cell.spec["logits"] == "ties"
    if ties:
        bias = torch.zeros(EXPERTS, device="cuda")
        logits = torch.full((rows, EXPERTS), 0.5, device="cuda")
    else:
        bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
        logits = torch.randn(rows, EXPERTS, generator=g, device="cuda") * 2
    x = torch.randn(rows, LATENT, generator=g, device="cuda").bfloat16()
    x[x == 0] = 1e-3  # an exact +0 / -0 pair could read as the mailbox sentinel
    gsf = 1.0
    launch, _ = _launch(target, rows, bias, gsf)
    launch.mailbox[0, :rows, :LATENT] = x
    launch.mailbox[0, :rows, LATENT:] = logits.contiguous().view(torch.bfloat16)
    ref = op.reference(launch, None)  # before the kernel re-arms the cells
    scratch.plan.zero_()
    scratch.routes.fill_(-1)
    with validating(op):
        op(launch, module("cutedsl.common.host").stream(launch.mailbox.device))
    torch.cuda.synchronize()

    # ids: by rank, exact unless two of the top 17 keys sit within TIE of each other
    ids = launch.ids.long()
    for t in range(rows):
        keys = ref["keys"][t]
        top = keys.sort(descending=True, stable=True).values[: TOP_K + 1]
        gaps = top[:-1] - top[1:]
        near = bool(((gaps > 0) & (gaps <= TIE)).any())
        if not near:
            assert torch.equal(ids[t], ref["ids"][t]), f"row {t}: ids differ"
        else:
            picked = keys[ids[t]]
            assert bool((picked[:-1] >= picked[1:] - TIE).all()), f"row {t}: not ranked"
            assert bool((picked.min() >= top[TOP_K - 1] - TIE)), f"row {t}: not the top 16"
    # weights: BF16 of the renormalized scaled sigmoids (FP32 sum and div.rn below 2^-20)
    got = launch.weights.double()
    want = ref["weights"]
    assert bool(((got - want).abs() <= 2.0**-8 * want.abs() + 2.0**-20).all()), "weights"
    # the latent copy, bitwise
    assert torch.equal(launch.latent.view(torch.int16), ref["latent"].view(torch.int16))
    # the plan: the reference quantization but where a value sits on a rounding boundary
    # (the kernel's multiplier is an approximate reciprocal)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_ull_experts_nv_experts_runtime as rt,  # noqa: E501
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_ull_nv_routed_experts import (  # noqa: E501
        latent,
    )

    plan = latent(launch.plan, rows, launch.image)
    differs = plan != ref["plan"]
    if bool(differs.any()):
        blocks = x.double().view(rows, -1, 16)
        amax = blocks.abs().amax(-1, keepdim=True)
        scale = (amax / 6).to(torch.float8_e4m3fn).double()
        v = (blocks / scale).abs()
        bounds = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), device="cuda").double()
        near = ((v[..., None] - bounds).abs() <= 2.0**-18 * v[..., None]).any(-1)
        assert bool(near.view(rows, LATENT)[differs].all()), (
            f"{int(differs.sum())} plan values differ off a rounding boundary"
        )
    # ... and bitwise the experts' own plan writer of the same latent
    other = torch.zeros_like(scratch.plan)
    write = rt.write_image if launch.image else rt.write_plan
    write(x, other[: scratch.image.numel()] if launch.image else other[: scratch.rows.numel()], gsf)
    torch.cuda.synchronize()
    assert torch.equal(plan, latent(other, rows, launch.image)), "plan differs from the writer's"
    # every mailbox cell it read re-armed
    assert bool((launch.mailbox[0, :rows].view(torch.int32) == SENTINEL).all()), "not re-armed"
    if launch.routes is not None:  # the experts' route mailbox: ids, then the image count
        routes = launch.routes
        assert torch.equal(routes[: rows * TOP_K], launch.ids.reshape(-1)), "route mailbox ids"
        assert int(routes[TOP_K * 8]) == 2 * rows - 1, "image count"
        routes.fill_(-1)


def test_is_valid_refuses_an_unarmed_route_mailbox(target) -> None:
    _, _, scratch = target
    bias = torch.zeros(EXPERTS, device="cuda")
    launch, _ = _launch(target, 8, bias, 1.0)
    assert launch.routes is not None, "8 rows signal the experts"
    scratch.routes.fill_(-1)
    op.is_valid(launch, None)
    scratch.routes[0] = 5
    with pytest.raises(ValueError, match="not armed"):
        op.is_valid(launch, None)
    scratch.routes.fill_(-1)
