# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 ``bf16_routed_experts`` entry.

The entry's claim is that it is the packed call ``replicated_experts`` makes for the two BF16
layers (``FlashinferOpBackend.run_bf16_moe`` on ``topk_ids`` / ``topk_weights``)
without the packing, so every cell is gated bitwise against that call over the
same 64-expert window: the routed FC2 rows (read through each call's own
``expanded_idx_to_permuted_idx``) and the row map itself; the caller's ids and
weights, which the entry hands to the kernels, must come back unwritten. (The
returned expert weights are not compared: in the unpacked form they are the
caller's ``topk_weights``, and the model's finalize never reads them.) The
entry's ``reference`` is then held to a band, which certifies what the shared
kernels compute (layout, gating order, activation).

The band, declared before measuring: the reference is FP32 throughout except
for the BF16 FC2 input the kernel also rounds to, so the two differ by FC1 /
FC2 accumulation order over K = 4096, the BF16 output rounding (at most half
an ulp of the row's largest magnitude) and the occasional one-ulp flip of an
FC2 input element that FC1's accumulation order moves across a BF16 rounding
boundary (relative RMS of at most one ulp even if every element flipped).
Gate: every element within 2 BF16 ulp of its row's largest magnitude,
relative RMS within 1 ulp.
"""

import random
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.bf16_routed_experts import (  # noqa: E501
    EXPERTS,
    HIDDEN,
    INTER,
    TOP_K,
    WINDOW,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.bf16_routed_experts import (  # noqa: E501
    bf16_routed_experts as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "bf16_routed_experts requires a CUDA device"

#: Declared before measuring (module docstring).
ELEMENT_ULP, RMS_ULP = 2.0, 1.0


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


@pytest.fixture(scope="module")
def window():
    """The 64 experts of one rank's ring-buddy window, in the layout the BF16 leaf stores."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import get_op_backend
    from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType

    method = BF16TRTLLMGenFusedMoEMethod()
    gen = torch.Generator(device="cuda").manual_seed(2038)
    w13 = w2 = None
    for e in range(WINDOW):
        raw13 = (torch.randn(2 * INTER, HIDDEN, device="cuda", generator=gen) * 0.02).to(
            torch.bfloat16
        )
        raw2 = (torch.randn(HIDDEN, INTER, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        p13 = method._prepare_bf16_weight_for_trtllm_gen(
            raw13, method._get_w3_w1_permute_indices(raw13)
        )
        p2 = method._prepare_bf16_weight_for_trtllm_gen(raw2, method._get_w2_permute_indices(raw2))
        if w13 is None:
            w13 = torch.empty((WINDOW, *p13.shape), dtype=p13.dtype, device="cuda")
            w2 = torch.empty((WINDOW, *p2.shape), dtype=p2.dtype, device="cuda")
        w13[e].copy_(p13)
        w2[e].copy_(p2)
    op_backend = get_op_backend("flashinfer")
    # MuseSparkOpenRoutingMethod is a MiniMax2 routing method; SwiGLU is TRT-LLM's
    # gated activation 0 (ring_buddy._act_type).
    routing = RoutingMethodType.MiniMax2
    return dict(
        w13=w13,
        w2=w2,
        shuffled=method.use_shuffled_weight,
        layout=int(method.weight_layout),
        op_backend=op_backend,
        routing=routing,
        fi_routing=int(op_backend.cvt_routing_method_type(routing)),
        fi_act=int(op_backend.cvt_activation_type(0)),
    )


def _inputs(rows: int, active: int, seed: int):
    """``rows`` top-8 rows whose window ids name exactly ``active`` distinct local experts.

    The other slots carry ids outside the window, as ``ring_buddy_assign`` writes
    them for the slots another rank runs; each row's ids are distinct.
    """
    rng = random.Random(seed)
    local = rng.sample(range(WINDOW), active)
    per_row = [[] for _ in range(rows)]
    for i, e in enumerate(local):
        per_row[i % rows].append(e)
    ids = []
    for row in per_row:
        extra = [e for e in local if e not in row]
        row = row + rng.sample(extra, min(len(extra), rng.randint(0, 2), TOP_K - len(row)))
        row = row + rng.sample(range(WINDOW, EXPERTS), TOP_K - len(row))
        rng.shuffle(row)
        ids.append(row)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, HIDDEN), device="cuda", generator=g).to(torch.bfloat16)
    w = torch.softmax(torch.randn((rows, TOP_K), device="cuda", generator=g), -1)
    return x, torch.tensor(ids, dtype=torch.int32, device="cuda"), w.to(torch.bfloat16)


def _args(window, x, ids, w):
    return (
        x, window["w13"], window["w2"], ids, w, INTER, 0, WINDOW,
        window["fi_routing"], window["fi_act"], window["shuffled"], window["layout"],
        EXPERTS, TOP_K,
    )  # fmt: skip


def _packed(window, x, ids, w):
    """The call ``ring_buddy._bf16_call`` makes without unpacked routing."""
    return window["op_backend"].run_bf16_moe(
        router_logits=None, routing_bias=None, hidden_states=x,
        gemm1_weights=window["w13"], gemm2_weights=window["w2"],
        num_experts=EXPERTS, top_k=TOP_K, n_group=None, topk_group=None,
        intermediate_size=INTER, local_expert_offset=0, local_num_experts=WINDOW,
        routed_scaling_factor=None, routing_method_type=window["routing"],
        topk_weights=w, topk_ids=ids, gated_act_type=0, output=None,
        use_shuffled_weight=window["shuffled"], weight_layout=window["layout"],
        do_finalize=False,
    )  # fmt: skip


def _routed(out, ids):
    """The local slots and their FC2 rows, read through the call's own row map."""
    gemm2, perm = out[0], out[2].reshape(-1).to(torch.int64)
    local = (ids.reshape(-1) < WINDOW).nonzero().reshape(-1)
    return local, gemm2[perm[local]]


def _check_bitwise(out, ref, ids):
    local, rows = _routed(out, ids)
    _, ref_rows = _routed(ref, ids)
    assert torch.equal(rows.view(torch.int16), ref_rows.view(torch.int16)), (
        f"{int((rows != ref_rows).sum())} routed FC2 elements differ from the packed call"
    )
    assert torch.equal(out[2], ref[2]), "the row map differs from the packed call's"
    return local, rows


def _check_untouched(ids, w, ids_before, w_before):
    """The entry hands the caller's ``topk_ids`` / ``topk_weights`` to the kernels: never written."""
    assert torch.equal(ids, ids_before), "the call wrote topk_ids"
    assert torch.equal(w.view(torch.int16), w_before.view(torch.int16)), (
        "the call wrote topk_weights"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[str(c.spec) for c in op.CELLS])
def test_cells(cell, window, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    for trial in range(2):
        x, ids, w = _inputs(spec["rows"], spec["active"], _seed(f"{spec}-{trial}"))
        local_slots = int((ids < WINDOW).sum())
        assert len(set(ids[ids < WINDOW].tolist())) == spec["active"]
        ids_before, w_before = ids.clone(), w.clone()
        out = op(*_args(window, x, ids, w))
        torch.cuda.synchronize()
        _check_untouched(ids, w, ids_before, w_before)
        local, rows = _check_bitwise(out, _packed(window, x, ids, w), ids)
        assert local.numel() == local_slots
        ref = op.reference(*_args(window, x, ids, w))
        assert not ref[(ids.reshape(-1) >= WINDOW).nonzero().reshape(-1)].any()
        if local_slots:
            assert_within_ulp(
                rows.float(), ref[local], ELEMENT_ULP, RMS_ULP, ulp=ULP[torch.bfloat16]
            )


def test_graph_replay_follows_inputs(window):
    """Captured once, replayed on new routing and rows: bitwise the packed call each time."""
    x, ids, w = _inputs(8, 5, _seed("graph"))
    args = _args(window, x, ids, w)
    with validating(op):
        op(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    for trial, active in enumerate((4, 6, 0, 8, 1)):
        new_x, new_ids, new_w = _inputs(8, active, _seed(f"graph-{trial}"))
        x.copy_(new_x)
        ids.copy_(new_ids)
        w.copy_(new_w)
        graph.replay()
        torch.cuda.synchronize()
        _check_untouched(ids, w, new_ids, new_w)
        _check_bitwise(out, _packed(window, x, ids, w), ids)


@pytest.mark.parametrize(
    "bad, match",
    [
        (dict(weights_dtype=torch.float32), "BF16"),
        (dict(ids_dtype=torch.int64), "int32"),
        (dict(window_end=EXPERTS + 1), "inside the expert range"),
    ],
)
def test_is_valid_rejects(window, bad, match):
    x, ids, w = _inputs(8, 4, 5)
    args = list(_args(window, x, ids.to(bad.get("ids_dtype", torch.int32)), w))
    args[4] = w.to(bad.get("weights_dtype", torch.bfloat16))
    if "window_end" in bad:
        args[7] = bad["window_end"]
    with pytest.raises(AssertionError, match=match):
        op.is_valid(*args)
