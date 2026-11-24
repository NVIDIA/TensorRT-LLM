# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kda_mtp_decode_v2.k31_kda_mtp_verify`` (``kda_decode_mtp_v2_kernel``) is bitwise the engine verify.

On the packed / fused-norm contract the engine ``trtllm::kda_mtp_decode`` runs
``kda_decode_mtp_kernel``; the family kernel has the same arithmetic in the same order, so every
output -- the normed core, the committed state, the conv windows and pending columns, the replay
caches -- must match bit for bit, over two chained rounds (the second replays the drafts each
kernel cached in the first). Shapes cover the cluster-pair form (tiles <= 64), one wave of single
CTAs, and more tiles than SMs (where the op runs ``kda_decode_mtp_2cta_kernel``, two tiles per SM).
The two-CTA kernel is also checked bit for bit against the v2 kernel on every shape, including the
attention-DP bar shapes.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401 — engine op
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm import (
    kda_mtp_decode_v2,
)

__extra_import_path__ = ["."]
from _kda_verify_inputs import MUTATED, build  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not kda_mtp_decode_v2.supports_device(torch.device("cuda", torch.cuda.current_device())),
    reason="needs SM100 / SM103 / SM107",
)

_ENGINE_ONLY = dict(cu_seqlens=None, packed_token_layout=True, fuse_output_norm=True)


def _engine(args: dict) -> torch.Tensor:
    return torch.ops.trtllm.kda_mtp_decode(**args, **_ENGINE_ONLY)


def _family(args: dict) -> torch.Tensor:
    return kda_mtp_decode_v2.k31_kda_mtp_verify(**args)


def _assert_bitwise(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape, a.dtype, b.dtype)
    if not torch.equal(a, b):
        diff = (a.float() - b.float()).abs()
        raise AssertionError(
            f"{name}: {int((a != b).sum())} of {a.numel()} elements differ (max {diff.max():.3e})"
        )


def _next_round(args: dict, seed: int) -> dict:
    """Round-2 inputs on the same pools: fresh new-token projections and accepted counts."""
    fresh = build(
        N=args["ssm_state_indices"].shape[0],
        H=args["A_log"].shape[0],
        num_spec=args["num_spec"],
        pool=args["recurrent_state"].shape[0],
        seed=seed,
    )
    nxt = dict(args)
    for name in ("x_q", "x_k", "x_v", "g", "beta", "onorm_g"):
        nxt[name] = fresh[name]
    accepted = torch.zeros_like(args["num_accepted_tokens"])
    slots = args["ssm_state_indices"].long()
    m = args["num_spec"]
    accepted[slots] = ((torch.arange(slots.numel(), device=slots.device) * 3 + 1) % (m + 1)).to(
        accepted.dtype
    )
    nxt["num_accepted_tokens"] = accepted.to(torch.int32)
    return nxt


SHAPES = [
    # (N, H, num_spec, pool, state_layer_view)
    pytest.param(1, 6, 7, 4, False, id="tp16-n1-cluster_pair"),
    pytest.param(4, 12, 4, 8, True, id="tp8-n4-cluster_pair-layer_view"),
    pytest.param(8, 12, 4, 8, False, id="tp8-n8-one_wave"),
    pytest.param(8, 24, 7, 16, False, id="tp4-n8-192_tiles"),
    pytest.param(8, 96, 7, 8, True, id="dp-n8-768_tiles-layer_view"),
    pytest.param(64, 12, 4, 96, False, id="tp8-n64-768_tiles"),
]


@pytest.mark.parametrize("N,H,num_spec,pool,layer_view", SHAPES)
def test_bitwise_vs_engine_verify(N, H, num_spec, pool, layer_view) -> None:
    seed = 1000 * H + 10 * N + num_spec
    engine_args = build(N, H, num_spec, pool=pool, seed=seed, state_layer_view=layer_view)
    family_args = build(N, H, num_spec, pool=pool, seed=seed, state_layer_view=layer_view)
    for round_index in range(2):
        out_engine = _engine(engine_args)
        out_family = _family(family_args)
        torch.cuda.synchronize()
        assert torch.isfinite(out_family.float()).all()
        _assert_bitwise(f"round {round_index} out", out_family, out_engine)
        for name in MUTATED:
            _assert_bitwise(f"round {round_index} {name}", family_args[name], engine_args[name])
        engine_args = _next_round(engine_args, seed + 1)
        family_args = _next_round(family_args, seed + 1)


def _two_kernels(args: dict, kernel: str) -> torch.Tensor:
    return kda_mtp_decode_v2.k31_kda_mtp_verify_impl(**args, kernel=kernel)


TWO_CTA_SHAPES = SHAPES + [
    pytest.param(14, 96, 4, 20, True, id="dp-n14-1344_tiles-4_drafts-layer_view"),
    pytest.param(14, 96, 7, 16, False, id="dp-n14-1344_tiles-7_drafts"),
    pytest.param(25, 96, 4, 32, False, id="dp-n25-2400_tiles-4_drafts"),
]


@pytest.mark.parametrize("N,H,num_spec,pool,layer_view", TWO_CTA_SHAPES)
def test_two_cta_bitwise_vs_v2(N, H, num_spec, pool, layer_view) -> None:
    """``kda_decode_mtp_2cta_kernel`` (any tile count) gives the v2 kernel's bits, over two rounds."""
    seed = 7000 * H + 10 * N + num_spec
    v2_args = build(N, H, num_spec, pool=pool, seed=seed, state_layer_view=layer_view)
    two_args = build(N, H, num_spec, pool=pool, seed=seed, state_layer_view=layer_view)
    for round_index in range(2):
        out_v2 = _two_kernels(v2_args, "v2")
        out_two = _two_kernels(two_args, "two_cta")
        torch.cuda.synchronize()
        assert torch.isfinite(out_two.float()).all()
        _assert_bitwise(f"round {round_index} out", out_two, out_v2)
        for name in MUTATED:
            _assert_bitwise(f"round {round_index} {name}", two_args[name], v2_args[name])
        v2_args = _next_round(v2_args, seed + 1)
        two_args = _next_round(two_args, seed + 1)


@pytest.mark.parametrize(
    "N,H,num_spec,pool",
    [
        pytest.param(16, 96, 4, 20, id="dp-n16-1536_tiles-4_drafts"),
        pytest.param(32, 96, 4, 40, id="dp-n32-3072_tiles-4_drafts"),
        pytest.param(14, 96, 7, 16, id="dp-n14-1344_tiles-7_drafts"),
    ],
)
def test_two_cta_state_evict_last_bitwise(N, H, num_spec, pool) -> None:
    """The evict-last cache policy on the state loads changes no output, state or cache bit."""
    seed = 9100 + N + num_spec
    plain = build(N, H, num_spec, pool=pool, seed=seed)
    hinted = build(N, H, num_spec, pool=pool, seed=seed)
    for round_index in range(2):
        out_plain = _two_kernels(plain, "two_cta")
        out_hinted = kda_mtp_decode_v2.k31_kda_mtp_verify_impl(
            **hinted, kernel="two_cta", state_evict_last=True
        )
        torch.cuda.synchronize()
        _assert_bitwise(f"round {round_index} out", out_hinted, out_plain)
        for name in MUTATED:
            _assert_bitwise(f"round {round_index} {name}", hinted[name], plain[name])
        plain = _next_round(plain, seed + 1)
        hinted = _next_round(hinted, seed + 1)


def test_two_cta_replays_in_a_cuda_graph() -> None:
    """The two-CTA launch keeps no host state: a captured call replays to the eager bits."""
    eager = build(14, 96, 4, pool=16, seed=11)
    graphed = build(14, 96, 4, pool=16, seed=11)
    out_eager = _two_kernels(eager, "two_cta")
    _two_kernels(build(14, 96, 4, pool=16, seed=12), "two_cta")  # compile outside the capture
    torch.cuda.synchronize()
    snapshot = {name: graphed[name].clone() for name in MUTATED}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = _two_kernels(graphed, "two_cta")
    for name in MUTATED:
        graphed[name].copy_(snapshot[name])
    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise("out", out_graph, out_eager)
    for name in MUTATED:
        _assert_bitwise(name, graphed[name], eager[name])


def test_rejects_a_bf16_state() -> None:
    args = build(2, 12, 4)
    args["recurrent_state"] = args["recurrent_state"].to(torch.bfloat16)
    with pytest.raises(ValueError, match="fp32 recurrent state"):
        _family(args)


def test_rejects_unpacked_rows() -> None:
    """The op derives every row from the request index: the row count must be N * (1 + num_spec)."""
    args = build(2, 12, 4)
    for name in ("x_q", "x_k", "x_v", "g", "beta", "onorm_g"):
        args[name] = args[name][:, :-1]
    with pytest.raises(ValueError, match="packed rows"):
        _family(args)
