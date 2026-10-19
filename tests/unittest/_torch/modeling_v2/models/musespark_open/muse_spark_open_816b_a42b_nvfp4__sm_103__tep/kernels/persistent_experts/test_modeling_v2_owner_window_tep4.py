# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep4 expert paths on one GPU: four virtual expert-parallel ranks of 64 experts each.

Each virtual rank runs one MoE layer's experts three ways and the built-in
per-expert finalize (``post_expert_norm_combine``) on each; the cross-rank sum
is reproduced as the fused finalize all-reduce computes it (each rank's FP32
partial rounded to BF16, the ranks summed in FP32 in rank order, rounded once):

* **built-in**: the built-in tep4 layer, TRTLLM-gen over the rank's own
  64-expert block with global ids (``local_expert_offset = 64 r``);
* **owner window**: ``ring_buddy.install(windows=1)``'s stacked call, the same
  TRTLLM-gen call with offset 0 over the own block and the ids
  ``ring_buddy_assign(..., balance=False, ranks=4)`` rebases to it;
* **persistent**: ``persistent_experts`` on the same window and ids (NVFP4).

For the BF16 layers the owner window's call is ``bf16_routed_experts`` (unpacked
routing) against the built-in packed FlashInfer call.

Tolerances, declared before measuring. The owner window's TRTLLM-gen call reads
the same bytes with the same kernels as the built-in call, so its rows and the
layer output are gated bitwise. The persistent kernel and ``bf16_routed_experts``
claim bitwise rows as well (their own op tests gate that); here they are held to
the band the fast paths declare against the built-in layer: every element within
8 BF16 ulp of its row's largest magnitude and a relative RMS within 1 ulp, and
the test reports whether they came out bitwise.

Also: the tep4 two-window stack (``replicated_experts`` at tep4, not composed by
default) runs its balanced placement through TRTLLM-gen at a 128-expert window
within the same band, so the code path is not broken at tep4.
"""

import random
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    fp4_block_scale_moe_runner as moe_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.persistent_experts import (  # noqa: E501
    persistent_experts,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    EXPERTS,
    TOP_K,
    block_of,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    ring_buddy_assign as assign,
)
from tensorrt_llm._torch.models.modeling_musespark_open import post_expert_norm_combine

__extra_import_path__ = ["../../../../.."]
from _validating import assert_within_ulp  # noqa: E402

assert torch.cuda.is_available(), "the expert ops require a CUDA device"

RANKS = 4
BLOCK = block_of(RANKS)
HIDDEN = 4096
INTER = 4096
SV = 16
EPS = 1e-6
#: The band declared for the fast paths against the built-in layer.
ELEMENT_ULP, RMS_ULP = 8.0, 1.0


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _e4m3(shape, lo_exp: int, hi_exp: int, gen: torch.Generator) -> torch.Tensor:
    e = torch.randint(lo_exp, hi_exp + 1, shape, device="cuda", generator=gen).float()
    m = 1.0 + torch.rand(shape, device="cuda", generator=gen)
    return (torch.exp2(e) * m).clamp(max=448.0).to(torch.float8_e4m3fn)


def _scalars(n: int, gen: torch.Generator):
    e = torch.randint(-7, -4, (n,), device="cuda", generator=gen).float()
    return torch.exp2(e) * (1.0 + torch.rand(n, device="cuda", generator=gen))


@pytest.fixture(scope="module")
def nvfp4_layer():
    """256 NVFP4 experts (random bytes in the stock layout both kernels read) and a norm weight."""
    gen = torch.Generator(device="cuda").manual_seed(4004)
    experts = dict(
        w1=torch.randint(
            0, 256, (EXPERTS, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device="cuda",
            generator=gen,
        ),
        s1=_e4m3((EXPERTS, 2 * INTER, HIDDEN // SV), -3, 0, gen),
        w2=torch.randint(
            0, 256, (EXPERTS, HIDDEN, INTER // 2), dtype=torch.uint8, device="cuda", generator=gen
        ),
        s2=_e4m3((EXPERTS, HIDDEN, INTER // SV), -3, 0, gen),
        scale_c=_scalars(EXPERTS, gen),
        alpha1=_scalars(EXPERTS, gen),
        alpha2=_scalars(EXPERTS, gen),
    )  # fmt: skip
    norm_weight = (torch.randn(HIDDEN, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    return experts, norm_weight


def _blocks(experts, blocks):
    return {
        name: torch.cat([t[BLOCK * b : BLOCK * (b + 1)] for b in blocks]).contiguous()
        for name, t in experts.items()
    }


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


def _trtllm_gen(w, x, x_sf, ids, w_bf16, offset, num_local):
    return moe_op(
        None, None, x, x_sf, w["w1"], w["s1"], None, None, None, None, w["w2"], w["s2"], None,
        w["scale_c"], w["alpha1"], w["alpha2"], EXPERTS, TOP_K, None, None, INTER, offset,
        num_local, None, 1, False, 0, topk_weights=w_bf16, topk_ids=ids, output=None,
        tune_max_num_tokens=8192, use_dp=False,
    )  # fmt: skip


def _persistent(w, x, x_sf, ids, w_bf16):
    return persistent_experts(
        x, x_sf, ids, w_bf16, w["w1"], w["s1"], w["w2"], w["s2"],
        w["scale_c"], w["alpha1"], w["alpha2"],
    )  # fmt: skip


def _rows_by_slot(outputs):
    gemm2, _, index = outputs
    return {
        s: gemm2[i, :HIDDEN]
        for s, i in enumerate(index.reshape(-1).tolist())
        if 0 <= i < gemm2.shape[0]
    }


def _all_reduce_like_finalize(partials):
    total = torch.zeros_like(partials[0], dtype=torch.float32)
    for p in partials:
        total += p.to(torch.bfloat16).float()
    return total.to(torch.bfloat16)


def _layer_output(outputs_per_rank, weights, norm_weight, rows):
    partials = []
    for gemm2, _, index in outputs_per_rank:
        partials.append(
            post_expert_norm_combine(
                gemm2,
                index,
                weights,
                norm_weight,
                EPS,
                num_tokens=rows,
                top_k=TOP_K,
                hidden_size=HIDDEN,
            )  # fmt: skip
        )
    return _all_reduce_like_finalize(partials)


def _report(name, result, reference):
    diff = (result.float() - reference.float()).abs()
    rms = (diff.pow(2).mean().sqrt() / reference.float().pow(2).mean().sqrt()).item() / 2**-8
    bitwise = torch.equal(result, reference)
    print(
        f"{name}: {'bitwise' if bitwise else 'not bitwise'}, "
        f"{(diff > 0).float().mean().item() * 100:.3f} % elements differ, rel RMS {rms:.3f} ulp"
    )
    return bitwise


@pytest.mark.parametrize("rows,keep", [(1, 0), (3, 5), (8, 5), (8, 0)])
def test_owner_window_nvfp4_matches_built_in_tep4(nvfp4_layer, rows, keep):
    experts, norm_weight = nvfp4_layer
    rng = random.Random(_seed(f"tep4-{rows}-{keep}"))
    g = torch.Generator(device="cuda").manual_seed(_seed(f"tep4-x-{rows}-{keep}"))
    x = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g)
    x_sf = _e4m3((rows * HIDDEN // SV,), -4, 0, g)
    selected = torch.tensor(_routing(rows, rng, keep), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(rows, TOP_K, device="cuda", generator=g), -1)
    w_bf16 = weights.to(torch.bfloat16)

    built_in, owner, persistent = [], [], []
    built_in_rows, loads = {}, []
    for r in range(RANKS):
        own = _blocks(experts, [r])
        out = _trtllm_gen(own, x, x_sf, selected, w_bf16, BLOCK * r, BLOCK)
        built_in.append(out)
        built_in_rows.update(_rows_by_slot(out))
        ids, ids_w, stats = assign(selected, weights, r, False, ranks=RANKS)
        assert torch.equal(ids_w, w_bf16)
        loads = stats[:RANKS].tolist()
        owner_out = _trtllm_gen(own, x, x_sf, ids, ids_w, 0, BLOCK)
        owner.append(owner_out)
        for slot, row in _rows_by_slot(owner_out).items():
            assert torch.equal(row, built_in_rows[slot]), f"rank {r} slot {slot}: owner window row"
        persistent.append(_persistent(own, x, x_sf, ids, ids_w))
        del own
    assert len(built_in_rows) == rows * TOP_K
    reference = _layer_output(built_in, weights, norm_weight, rows)
    owner_result = _layer_output(owner, weights, norm_weight, rows)
    assert torch.equal(owner_result, reference), "the owner window must be the built-in layer"
    persistent_result = _layer_output(persistent, weights, norm_weight, rows)
    print(f"rows={rows} keep={keep} per-rank distinct experts {loads}")
    _report("persistent vs built-in", persistent_result, reference)
    assert_within_ulp(persistent_result, reference, ELEMENT_ULP, RMS_ULP)


@pytest.mark.parametrize("rows,keep", [(8, 5), (16, 3)])
def test_two_windows_balanced_tep4_within_band(nvfp4_layer, rows, keep):
    """``replicated_experts`` at tep4 (not composed by default): the 128-expert two-window
    stack with the balanced 4-rank placement, through TRTLLM-gen, partitions every slot
    and stays within the band of the built-in layer."""
    experts, norm_weight = nvfp4_layer
    rng = random.Random(_seed(f"tep4-2w-{rows}-{keep}"))
    g = torch.Generator(device="cuda").manual_seed(_seed(f"tep4-2w-x-{rows}-{keep}"))
    x = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g)
    x_sf = _e4m3((rows * HIDDEN // SV,), -4, 0, g)
    selected = torch.tensor(_routing(rows, rng, keep), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(rows, TOP_K, device="cuda", generator=g), -1)
    w_bf16 = weights.to(torch.bfloat16)
    built_in, stacked, seen = [], [], set()
    built_in_rows = {}
    for r in range(RANKS):
        own = _blocks(experts, [r])
        out = _trtllm_gen(own, x, x_sf, selected, w_bf16, BLOCK * r, BLOCK)
        built_in.append(out)
        built_in_rows.update(_rows_by_slot(out))
        del own
    for r in range(RANKS):
        stack = _blocks(experts, [r, (r - 1) % RANKS])
        ids, ids_w, stats = assign(selected, weights, r, True, ranks=RANKS)
        out = _trtllm_gen(stack, x, x_sf, ids, ids_w, 0, 2 * BLOCK)
        for slot, row in _rows_by_slot(out).items():
            assert slot not in seen, f"slot {slot} computed twice"
            seen.add(slot)
            assert torch.equal(row, built_in_rows[slot])
        stacked.append(out)
        del stack
    assert len(seen) == rows * TOP_K
    before, after = stats[:RANKS].tolist(), stats[RANKS:].tolist()
    assert max(after) <= max(before)
    reference = _layer_output(built_in, weights, norm_weight, rows)
    result = _layer_output(stacked, weights, norm_weight, rows)
    print(f"rows={rows} keep={keep} loads {before} -> {after}")
    _report("two windows balanced vs built-in", result, reference)
    assert_within_ulp(result, reference, ELEMENT_ULP, RMS_ULP)


@pytest.fixture(scope="module")
def bf16_layer():
    """256 BF16 experts in the TRTLLM-gen shuffled BlockMajorK layout the BF16 leaf stores."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import get_op_backend
    from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod

    method = BF16TRTLLMGenFusedMoEMethod()
    gen = torch.Generator(device="cuda").manual_seed(4005)
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


@pytest.mark.parametrize("rows,keep", [(1, 0), (8, 5)])
def test_owner_window_bf16_unpacked_matches_built_in_tep4(bf16_layer, rows, keep):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.bf16_routed_experts import (  # noqa: E501
        bf16_routed_experts,
    )

    experts, norm_weight, op_backend, layout = bf16_layer
    rng = random.Random(_seed(f"tep4-bf16-{rows}-{keep}"))
    g = torch.Generator(device="cuda").manual_seed(_seed(f"tep4-bf16-x-{rows}-{keep}"))
    x = torch.randn(rows, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    selected = torch.tensor(_routing(rows, rng, keep), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(rows, TOP_K, device="cuda", generator=g), -1)
    w_bf16 = weights.to(torch.bfloat16)
    routing_type = int(op_backend.cvt_routing_method_type(1))
    activation_type = int(op_backend.cvt_activation_type(0))
    built_in, owner = [], []
    for r in range(RANKS):
        own = _blocks(experts, [r])
        built_in.append(
            op_backend.run_bf16_moe(
                router_logits=None,
                routing_bias=None,
                hidden_states=x,
                gemm1_weights=own["w1"],
                gemm2_weights=own["w2"],
                num_experts=EXPERTS,
                top_k=TOP_K,
                n_group=None,
                topk_group=None,
                intermediate_size=INTER,
                local_expert_offset=BLOCK * r,
                local_num_experts=BLOCK,
                routed_scaling_factor=None,
                routing_method_type=1,
                topk_weights=w_bf16,
                topk_ids=selected,
                gated_act_type=0,
                output=None,
                use_shuffled_weight=layout[0],
                weight_layout=layout[1],
                do_finalize=False,
            )  # fmt: skip
        )
        ids, ids_w, _ = assign(selected, weights, r, False, ranks=RANKS)
        owner.append(
            bf16_routed_experts(
                x,
                own["w1"],
                own["w2"],
                ids,
                ids_w,
                INTER,
                0,
                BLOCK,
                routing_type,
                activation_type,
                layout[0],
                layout[1],
                EXPERTS,
                TOP_K,
            )  # fmt: skip
        )
        del own
    reference = _layer_output(built_in, weights, norm_weight, rows)
    result = _layer_output(owner, weights, norm_weight, rows)
    _report(f"bf16 unpacked owner window rows={rows}", result, reference)
    assert_within_ulp(result, reference, ELEMENT_ULP, RMS_ULP)
