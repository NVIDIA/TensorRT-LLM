# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_decode entry.

Each cell builds the KDA layer's one-token decode call: q/k/v/gate column views of the fused [q | k | v | g]
GEMM output, beta a column view of the merged [f_a | b | pad] projection, the three section views of a padded-slot
packed conv pool (updated in place) and the indexed fp32 state pool. Output and state rows within the decode bands
of an fp32 evaluation, conv windows bit for bit, every other slot untouched.
"""

import math

import pytest
import torch
from _kda_ops_util import (
    EPS,
    HEAD_DIM,
    LOWER_BOUND,
    SCALE,
    assert_scaled_close,
    bits,
    clone_strided,
    generator,
    padded_slots,
    requires_sm103,
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_decode import (
    kda_decode,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

# Approximate exp / rcp in the kernel (main's decode entry bands).
TOL_OUT, TOL_STATE = 2e-2, 1e-3


def _sections(conv, dim, update_conv_cache):
    """q / k / v windows: views of the packed pool (fused path) or dense copies (the fallback)."""
    views = (conv[:, :dim], conv[:, dim : 2 * dim], conv[:, 2 * dim :])
    return views if update_conv_cache else tuple(v.contiguous() for v in views)


def _call(spec, gen):
    heads, batch, slots = spec["heads"], spec["batch"], spec["pool_slots"]
    dim = heads * HEAD_DIM

    def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(dtype)

    qkvg = rnd(batch, 4 * dim)
    merged = rnd(batch, (HEAD_DIM + heads + 7) // 8 * 8)
    gate = rnd(batch, dim)
    conv_pool = padded_slots(slots, (3 * dim, 3), torch.bfloat16)
    conv_pool.copy_(torch.randn(conv_pool.shape, generator=gen, device="cuda") * 0.5)
    update_conv_cache = spec.get("update_conv_cache", True)
    state = padded_slots(slots, (heads, HEAD_DIM, HEAD_DIM), torch.float32)
    indices = torch.randperm(slots, generator=gen, device="cuda")[:batch].to(torch.int32)
    # The fused path addresses the live pool by slot; the fallback hands dense windows per request.
    conv_src = conv_pool if update_conv_cache else conv_pool.index_select(0, indices.long())
    state.copy_(torch.randn(state.shape, generator=gen, device="cuda") * 0.05)

    def heads_view(t):
        return t.unflatten(-1, (heads, HEAD_DIM)).unsqueeze(0)

    zeros = torch.zeros(dim, dtype=torch.bfloat16, device="cuda")
    args = dict(
        x_q=heads_view(qkvg[:, :dim]),
        x_k=heads_view(qkvg[:, dim : 2 * dim]),
        x_v=heads_view(qkvg[:, 2 * dim : 3 * dim]),
        w_q_t=rnd(4, dim, scale=0.3),
        w_k_t=rnd(4, dim, scale=0.3),
        w_v_t=rnd(4, dim, scale=0.3),
        bias_q=zeros,
        bias_k=zeros,
        bias_v=zeros,
        conv_state_q=_sections(conv_src, dim, update_conv_cache)[0],
        conv_state_k=_sections(conv_src, dim, update_conv_cache)[1],
        conv_state_v=_sections(conv_src, dim, update_conv_cache)[2],
        a_log=torch.log(torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen)),
        g=heads_view(gate),
        dt_bias=torch.empty(dim, device="cuda").uniform_(
            math.log(1e-3), math.log(1e-1), generator=gen
        ),
        beta=merged[:, HEAD_DIM : HEAD_DIM + heads].unsqueeze(0),
        onorm_g=heads_view(qkvg[:, 3 * dim :]),
        onorm_weight=1 + torch.randn(HEAD_DIM, generator=gen, device="cuda") * 0.1,
        ssm_state_indices=indices,
        state=state,
        apply_onorm=True,
        update_conv_cache=update_conv_cache,
        use_lower_bound=True,
        apply_beta_sigmoid=True,
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        onorm_eps=EPS,
        output=torch.empty(batch, 1, heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
    )
    return args, conv_src


def _reference_args(args, conv_src):
    """Same-layout copies of everything the op writes."""
    ref_conv = clone_strided(conv_src)
    dim = args["x_q"].shape[-2] * HEAD_DIM
    if not args["update_conv_cache"]:
        # Dense windows are read-only: the reference reads the op's own (unchanged) copies.
        return dict(
            args, state=clone_strided(args["state"]), output=torch.empty_like(args["output"])
        ), ref_conv
    return dict(
        args,
        conv_state_q=ref_conv[:, :dim],
        conv_state_k=ref_conv[:, dim : 2 * dim],
        conv_state_v=ref_conv[:, 2 * dim :],
        state=clone_strided(args["state"]),
        output=torch.empty_like(args["output"]),
    ), ref_conv


@pytest.mark.parametrize("cell", kda_decode.CELLS, ids=[c.why[:60] for c in kda_decode.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(kda_decode)
    args, conv_src = _call(cell.spec, generator(cell))
    ref_args, ref_conv = _reference_args(args, conv_src)
    windows_before = [args[f"conv_state_{n}"].clone() for n in "qkv"]
    assert kda_decode(**args) is None
    kda_decode.reference(**ref_args)
    assert_scaled_close(args["output"], ref_args["output"], TOL_OUT, "output")
    assert torch.equal(bits(conv_src), bits(ref_conv)), (
        "conv windows differ (selected or other slots)"
    )
    if not args["update_conv_cache"]:
        for n, before in zip("qkv", windows_before):
            assert torch.equal(bits(args[f"conv_state_{n}"]), bits(before)), (
                f"read-only {n} window written"
            )
    selected = set(args["ssm_state_indices"].tolist())
    for slot in range(args["state"].shape[0]):
        got, ref = args["state"][slot], ref_args["state"][slot]
        if slot in selected:
            assert_scaled_close(got, ref, TOL_STATE, f"state slot {slot}")
        else:
            assert torch.equal(bits(got), bits(ref)), f"unselected slot {slot} changed"


@torch.inference_mode()
def test_cuda_graph_replay_matches_eager() -> None:
    """Captured once, replayed on rewritten inputs and restored pools: the eager call's bits."""
    spec = dict(heads=12, batch=4, pool_slots=8)
    args, conv_src = _call(spec, torch.Generator("cuda").manual_seed(11))
    conv_seed, state_seed = conv_src.clone(), args["state"].clone()
    kda_decode(**args)
    eager = (args["output"].clone(), conv_src.clone(), args["state"].clone())
    conv_src.copy_(conv_seed)
    args["state"].copy_(state_seed)
    kda_decode(**args)  # warm with identical inputs
    conv_src.copy_(conv_seed)
    args["state"].copy_(state_seed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kda_decode(**args)
    conv_src.copy_(conv_seed)
    args["state"].copy_(state_seed)
    args["output"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for name, got, ref in zip(
        ("output", "conv", "state"), (args["output"], conv_src, args["state"]), eager
    ):
        assert torch.equal(bits(got), bits(ref)), f"replayed {name} differs from eager"


@torch.inference_mode()
def test_guard_refuses_repeated_slots() -> None:
    args, _ = _call(dict(heads=6, batch=2, pool_slots=4), torch.Generator("cuda").manual_seed(4))
    args["ssm_state_indices"] = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="each slot once"):
        with validating(kda_decode):
            kda_decode(**args)
