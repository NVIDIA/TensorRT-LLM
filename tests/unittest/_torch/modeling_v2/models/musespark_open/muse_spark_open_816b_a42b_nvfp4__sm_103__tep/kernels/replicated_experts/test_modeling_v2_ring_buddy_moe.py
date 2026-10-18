# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``replicated_experts`` end to end on one GPU: eight virtual expert-parallel ranks.

Each virtual rank runs the TRTLLM-gen expert call -- NVFP4 (the 60 quantized
layers) and FlashInfer BF16 (the two unquantized layers) -- the way the fast path
does -- over a stacked ``[own block | predecessor's block]`` weight tensor with
ids from ``ring_buddy_assign`` -- and the way the built-in layer does -- over
its own block with global ids. Both feed the built-in per-expert finalize
(``post_expert_norm_combine``), and the all-reduce is reproduced as the fused
finalize all-reduce computes it: each rank's FP32 partial rounded to BF16,
the ranks summed in FP32 in rank order, rounded to BF16.

Claims gated here:

* Every expert row a rank computes from the stacked tensor is bitwise the row
  its owner computes from its own block (same bytes, same kernel per row).
* ``balance=False`` (every expert on its owner) is bitwise the built-in layer.
* ``balance=True`` stays inside the error budget declared for the fast path:
  relative RMS <= 1 BF16 ulp, every element within
  8 ulp of its row's largest magnitude. The only difference is which rows
  share a BF16-rounded partial.
"""

import random
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner as moe_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    BLOCK,
    EXPERTS,
    RANKS,
    TOP_K,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    ring_buddy_assign as assign,
)
from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine

__extra_import_path__ = ["../../../../.."]
from _validating import assert_within_ulp  # noqa: E402

assert torch.cuda.is_available(), "the expert op requires a CUDA device"

HIDDEN = 4096
INTER = 4096
SV = 16
EPS = 1e-6
#: The declared error budget of the balanced placement.
ELEMENT_ULP, RMS_ULP = 8.0, 1.0


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _experts(gen):
    """256 NVFP4 experts: codes uniform over e2m1, power-of-two e4m3 block scales.

    The bytes need not follow the kernel's shuffled layout: both arms read the
    same bytes, so whatever function they define is the same function. Scales
    of 2^-2..2^0 keep every intermediate finite. Per-expert scalars differ
    by expert so a stacked call that read its predecessor's scalars for its
    own experts (or the reverse) would be caught.
    """
    w1 = torch.randint(
        0, 256, (EXPERTS, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    s1 = torch.exp2(
        torch.randint(
            -2, 1, (EXPERTS, 2 * INTER, HIDDEN // SV), device="cuda", generator=gen
        ).float()
    ).to(torch.float8_e4m3fn)
    w2 = torch.randint(
        0, 256, (EXPERTS, HIDDEN, INTER // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    s2 = torch.exp2(
        torch.randint(-2, 1, (EXPERTS, HIDDEN, INTER // SV), device="cuda", generator=gen).float()
    ).to(torch.float8_e4m3fn)
    ids = torch.arange(EXPERTS, device="cuda", dtype=torch.float32)
    alpha1 = (1.0 + ids / EXPERTS) * 2.0**-6
    scale_c = alpha1 * 0.5
    alpha2 = (1.0 + (EXPERTS - ids) / EXPERTS) * 2.0**-6
    return dict(w1=w1, s1=s1, w2=w2, s2=s2, scale_c=scale_c, alpha1=alpha1, alpha2=alpha2)


def _blocks(all_experts, blocks):
    """Concatenate the given 32-expert blocks of every tensor, in order."""
    return {
        name: torch.cat([t[BLOCK * b : BLOCK * (b + 1)] for b in blocks]).contiguous()
        for name, t in all_experts.items()
    }


def _call(weights, x, x_sf, ids, w_bf16, offset, num_local):
    return moe_op(
        None, None, x, x_sf,
        weights["w1"], weights["s1"], None, None, None, None,
        weights["w2"], weights["s2"], None,
        weights["scale_c"], weights["alpha1"], weights["alpha2"],
        EXPERTS, TOP_K, None, None, INTER, offset, num_local, None, 1, False, 0,
        topk_weights=w_bf16, topk_ids=ids, output=None, tune_max_num_tokens=8192, use_dp=False,
    )  # fmt: skip


def _routing(rows, rng, keep):
    out, prev = [], None
    for _ in range(rows):
        if prev is None or keep == 0:
            row = rng.sample(range(EXPERTS), TOP_K)
        else:
            kept = rng.sample(prev, keep)
            row = kept + rng.sample([e for e in range(EXPERTS) if e not in kept], TOP_K - keep)
        out.append(row)
        prev = row
    return out


def _all_reduce_like_finalize(partials):
    """Each rank's FP32 partial rounded to BF16, summed in FP32 in rank order, rounded once."""
    total = torch.zeros_like(partials[0])
    for p in partials:
        total += p.to(torch.bfloat16).float()
    return total.to(torch.bfloat16)


def _rows_by_slot(outputs, rows):
    """slot -> FC2 row for the slots this call computed."""
    gemm2, _, index = outputs
    index = index.reshape(-1).tolist()
    return {s: gemm2[i, :HIDDEN] for s, i in enumerate(index) if 0 <= i < gemm2.shape[0]}


@pytest.fixture(scope="module")
def layer():
    gen = torch.Generator(device="cuda").manual_seed(2026)
    experts = _experts(gen)
    norm_weight = (torch.randn(HIDDEN, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    own = [_blocks(experts, [r]) for r in range(RANKS)]
    stacked = [_blocks(experts, [r, (r - 1) % RANKS]) for r in range(RANKS)]
    return experts, norm_weight, own, stacked


def _check_arms(rows, keep, selected, weights, norm_weight, run_own, run_stacked):
    """Gate the stacked arm against the built-in arm over the eight virtual ranks.

    ``run_own(r, ids, w_bf16)`` is rank ``r``'s built-in call over its own
    block (offset ``32 r``, 32 local experts, global ids); ``run_stacked`` the
    feature's call over the stack (offset 0, 64 local experts, assigned ids).
    """
    w_bf16 = weights.to(torch.bfloat16)

    def finalize(outputs):
        gemm2, _, index = outputs
        return post_expert_norm_combine(
            gemm2, index, weights, norm_weight, EPS,
            num_tokens=rows, top_k=TOP_K, hidden_size=HIDDEN,
        )  # fmt: skip

    built_in, built_in_rows = [], {}
    for r in range(RANKS):
        out = run_own(r, selected, w_bf16)
        built_in.append(finalize(out))
        for slot, row in _rows_by_slot(out, rows).items():
            assert slot not in built_in_rows
            built_in_rows[slot] = row
    assert len(built_in_rows) == rows * TOP_K
    reference = _all_reduce_like_finalize(built_in)

    for balance in (False, True):
        partials, seen, loads = [], set(), None
        for r in range(RANKS):
            ids, ids_w, stats = assign(selected, weights, r, balance)
            assert torch.equal(ids_w, w_bf16)
            out = run_stacked(r, ids, ids_w)
            for slot, row in _rows_by_slot(out, rows).items():
                assert slot not in seen, f"slot {slot} computed twice"
                seen.add(slot)
                assert torch.equal(row, built_in_rows[slot]), (
                    f"rank {r} computed slot {slot} differently from its owner"
                )
            partials.append(finalize(out))
            loads = stats
        assert len(seen) == rows * TOP_K
        result = _all_reduce_like_finalize(partials)
        if not balance:
            assert torch.equal(result, reference), "owner-only placement must be the built-in layer"
        else:
            before, after = loads[:RANKS].tolist(), loads[RANKS:].tolist()
            assert max(after) <= max(before)
            assert_within_ulp(result, reference, ELEMENT_ULP, RMS_ULP)
            diff = (result.float() - reference.float()).abs()
            print(
                f"rows={rows} keep={keep} loads {before} -> {after}: "
                f"{(diff > 0).float().mean().item() * 100:.2f} % elements differ, "
                f"rel RMS {((result.float() - reference.float()).pow(2).mean().sqrt() / reference.float().pow(2).mean().sqrt()).item() / 2**-8:.3f} ulp"  # noqa: E501
            )


@pytest.mark.parametrize("rows,keep", [(1, 0), (8, 5), (8, 0), (16, 3)])
def test_ring_buddy_matches_built_in(layer, rows, keep):
    _, norm_weight, own, stacked = layer
    rng = random.Random(_seed(f"{rows}-{keep}"))
    g = torch.Generator(device="cuda").manual_seed(_seed(f"x{rows}-{keep}"))
    x = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g)
    x_sf = torch.exp2(torch.randint(-2, 1, (rows * HIDDEN // SV,), device="cuda", generator=g)
                      .float()).to(torch.float8_e4m3fn)  # fmt: skip
    selected = torch.tensor(_routing(rows, rng, keep), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(rows, TOP_K, device="cuda", generator=g), -1)
    _check_arms(
        rows, keep, selected, weights, norm_weight,
        lambda r, ids, w: _call(own[r], x, x_sf, ids, w, BLOCK * r, BLOCK),
        lambda r, ids, w: _call(stacked[r], x, x_sf, ids, w, 0, 2 * BLOCK),
    )  # fmt: skip


@pytest.fixture(scope="module")
def bf16_layer():
    """256 BF16 experts in the TRTLLM-gen shuffled BlockMajorK layout the BF16 leaf stores."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import get_op_backend
    from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod

    method = BF16TRTLLMGenFusedMoEMethod()
    gen = torch.Generator(device="cuda").manual_seed(2027)
    w13 = w2 = None
    for e in range(EXPERTS):
        raw13 = (torch.randn(2 * INTER, HIDDEN, device="cuda", generator=gen) * 0.02).to(
            torch.bfloat16
        )
        raw2 = (torch.randn(HIDDEN, INTER, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        p13 = method._prepare_bf16_weight_for_trtllm_gen(
            raw13, method._get_w3_w1_permute_indices(raw13)
        )
        p2 = method._prepare_bf16_weight_for_trtllm_gen(raw2, method._get_w2_permute_indices(raw2))
        if w13 is None:
            w13 = torch.empty((EXPERTS, *p13.shape), dtype=p13.dtype, device="cuda")
            w2 = torch.empty((EXPERTS, *p2.shape), dtype=p2.dtype, device="cuda")
        w13[e].copy_(p13)
        w2[e].copy_(p2)
    norm_weight = (torch.randn(HIDDEN, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    layout = (method.use_shuffled_weight, method.weight_layout)
    return dict(w1=w13, w2=w2), norm_weight, get_op_backend("flashinfer"), layout


def _bf16_call(op_backend, layout, weights, x, ids, w_bf16, offset, num_local):
    return op_backend.run_bf16_moe(
        router_logits=None, routing_bias=None, hidden_states=x,
        gemm1_weights=weights["w1"], gemm2_weights=weights["w2"],
        num_experts=EXPERTS, top_k=TOP_K, n_group=None, topk_group=None,
        intermediate_size=INTER, local_expert_offset=offset, local_num_experts=num_local,
        routed_scaling_factor=None, routing_method_type=1, topk_weights=w_bf16, topk_ids=ids,
        gated_act_type=0, output=None, use_shuffled_weight=layout[0], weight_layout=layout[1],
        do_finalize=False,
    )  # fmt: skip


@pytest.mark.parametrize("rows,keep", [(1, 0), (8, 5), (16, 3)])
def test_ring_buddy_bf16_matches_built_in(bf16_layer, rows, keep):
    experts, norm_weight, op_backend, layout = bf16_layer
    rng = random.Random(_seed(f"bf16-{rows}-{keep}"))
    g = torch.Generator(device="cuda").manual_seed(_seed(f"bf16-x{rows}-{keep}"))
    x = torch.randn(rows, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    selected = torch.tensor(_routing(rows, rng, keep), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(rows, TOP_K, device="cuda", generator=g), -1)
    cache = {}

    def blocks(key, which):
        # One rank's tensors at a time: the full stacked set would hold 51 GB.
        if key not in cache:
            cache.clear()
            cache[key] = _blocks(experts, which)
        return cache[key]

    _check_arms(
        rows, keep, selected, weights, norm_weight,
        lambda r, ids, w: _bf16_call(op_backend, layout, blocks(("own", r), [r]), x, ids, w,
                                     BLOCK * r, BLOCK),
        lambda r, ids, w: _bf16_call(op_backend, layout,
                                     blocks(("stack", r), [r, (r - 1) % RANKS]), x, ids, w,
                                     0, 2 * BLOCK),
    )  # fmt: skip
