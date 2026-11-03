# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash MegaMoE end to end on four GPUs (attention DP, EP4) at the mid (150 rows per rank) and hi
(1,280) decode shapes on the delivered path: the FP8 shared expert fused into the kernel, router logits as K-chunk
partials, routing and quantization straight into the launch bucket and the default tactics; and the DSpark draft
bank re-encoded from MXFP4 to NVFP4.

Against the reference: the routed experts' per-expert NVFP4 math, the fused shared expert within 2% of the separate
one's error against FP32, the partials' ordered sum against the router GEMM, and the draft layer against FP32 math on
the same quantization. Bit for bit: the other scheduler with separate staging and no evict-first hint, a repeat and
an epoch-shifted pass, one CUDA graph of the whole schedule, and the one-round push dispatch (its EP16 gate lowered)
against the metadata push.
"""

import pickle
import sys
from typing import Optional

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from _torch.moe.quantize_utils import (
    FP8BlockScalesQuantizeUtil,
    NVFP4QuantizeUtil,
    NVFP4RefMegaMoECuteDsl,
)
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.custom_ops.router_logit_partials import RouterLogitPartials
from tensorrt_llm._torch.custom_ops.torch_custom_ops import router_gemm_autotuned
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE
from tensorrt_llm._torch.models.modeling_dspark_v41 import convert_draft_experts_to_nvfp4
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op
from tensorrt_llm._torch.moe.fused_moe import SwigluActivation, create_moe
from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl, mega_moe_cute_dsl
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import get_sm_version, mpi_allgather, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_WORLD_SIZE = 4
_NUM_EXPERTS = 384
_TOP_K = 6
_HIDDEN_SIZE = 5120
_INTERMEDIATE_SIZE = 2304
_CLAMP = 10.0
_SCALING = 1.5
_MAX_NUM_TOKENS = 1280
# Per-rank rows: the full mid and hi shapes, then each with an idle rank and a one-row (or short) rank.
_SHAPES = {"mid": ([150] * 4, [150, 0, 75, 1]), "hi": ([1280] * 4, [1280, 640, 0, 33])}
# Gate-bias variants of the full shapes: two hot experts, no rows for rank 2's experts, and only the first 24 of every
# rank's 96 experts live, so the phase-interleaved streams end in front of empty experts.
_VARIANTS = ("base", "hot", "no_rank2", "trailing_empty")
_LIVE_EXPERTS_PER_RANK = 24
# The schedule each bucket is compared with: phase interleave (cluster 2) for the grouped mid bucket; the grouped
# fallback for the phase-interleaved hi bucket.
_PHASE_INTERLEAVE_TACTIC = (
    [256, 128, 256],
    [2, 1, 1],
    None,
    ("phase_interleave", 2),
    "atomic_counter",
    "epi_warps",
    True,
    1,
    1,
    (1, 1),
)
_REPLAYS = 5
# Largest fused / separate ratio of the rel-RMS error against the FP32 reference.
_ERROR_RATIO = 1.02
# DSpark draft bank: 128 experts, top-3, at the mid (25 requests x 5) and hi (320 x 3) draft rows.
_DRAFT_EXPERTS = 128
_DRAFT_TOP_K = 3
_DRAFT_ROWS = (125, 960)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths


def _assert_on_all_ranks(problem: Optional[str]) -> None:
    """Fail on every rank together, so no rank is left waiting in the next collective."""
    problems = mpi_allgather(problem)
    assert not any(problems), problems


def _rel_rms(value: torch.Tensor, reference: torch.Tensor) -> float:
    error = (value.float() - reference).pow(2).mean().sqrt()
    return (error / reference.pow(2).mean().sqrt()).item()


def _summed(partials: RouterLogitPartials) -> torch.Tensor:
    """The logits route+quant routes on: the partials summed s = 0 .. S-1 in FP32."""
    total = partials.parts[0].clone()
    for part in partials.parts[1:]:
        total += part
    return total


def _biases(world_size: int) -> dict:
    base = torch.randn(_NUM_EXPERTS, dtype=torch.float32) * 0.05
    per_rank = _NUM_EXPERTS // world_size
    hot, no_rank2, trailing = base.clone(), base.clone(), base.clone()
    hot[[0, per_rank + 1]] += 10.0
    no_rank2[2 * per_rank : 3 * per_rank] = -1e4
    for r in range(world_size):
        trailing[r * per_rank + _LIVE_EXPERTS_PER_RANK : (r + 1) * per_rank] = -1e4
    return {"base": base, "hot": hot, "no_rank2": no_rank2, "trailing_empty": trailing}


def _reference(reference, routing, x, logits) -> torch.Tensor:
    """Per-expert NVFP4 math with the routing weights applied before FC2, as DeepSeek-V4 does."""
    experts, weights = routing.apply(logits)
    out = torch.zeros(x.shape, dtype=torch.float32)
    for expert_id in experts.unique().tolist():
        rows, slots = torch.where(experts == expert_id)
        mlp = reference.experts[expert_id]
        gate, up = mlp.gate_up_proj(x[rows]).float().chunk(2, dim=-1)
        hidden = F.silu(gate) * up * weights[rows, slots, None]
        out[rows] += mlp.down_proj(hidden.to(x.dtype)).float()
    return out.to(x.dtype)


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    p = packed.view(torch.uint8)
    codes = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(p.shape[0], -1).long()
    magnitude = torch.tensor(_E2M1, device=p.device)[codes & 7]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def _dequant_mxfp4(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(exponents.view(torch.uint8).float() - 127.0).repeat_interleave(32, dim=1)
    return _unpack_e2m1(codes) * scale


def _dequant_nvfp4(codes, scale, scale_2) -> torch.Tensor:
    block = scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return _unpack_e2m1(codes) * block * float(scale_2)


def _round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid, ties to even, saturating at 6."""
    a = v.abs().clamp(max=6.0)
    idx = torch.bucketize(a, torch.tensor(_E2M1_MIDPOINTS, device=v.device))
    idx = idx + ((a == 0.75) | (a == 1.75) | (a == 3.5)).long()
    return torch.sign(v) * torch.tensor(_E2M1, device=v.device)[idx]


def _nvfp4_roundtrip(a: torch.Tensor, global_scale: float) -> torch.Tensor:
    """The kernel's NVFP4 quantize + dequantize of fp32 rows (16-element E4M3 block scales)."""
    rows, k = a.shape
    blocks = a.view(rows, k // 16, 16)
    vmax = blocks.abs().amax(-1)
    scale = (global_scale * vmax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    inverse = torch.where(vmax != 0, global_scale / scale, torch.zeros_like(scale))
    q = _round_e2m1(blocks * inverse.unsqueeze(-1))
    return (q * (scale / global_scale).unsqueeze(-1)).reshape(rows, k)


def _mxfp4_draft_experts(device) -> dict:
    """Checkpoint-style MXFP4 draft experts, identical on every rank."""
    gen = torch.Generator(device=device).manual_seed(4101)
    weights = {}
    for expert in range(_DRAFT_EXPERTS):
        for proj, (n, k) in (
            ("w1", (_INTERMEDIATE_SIZE, _HIDDEN_SIZE)),
            ("w3", (_INTERMEDIATE_SIZE, _HIDDEN_SIZE)),
            ("w2", (_HIDDEN_SIZE, _INTERMEDIATE_SIZE)),
        ):
            codes = torch.randint(
                0, 256, (n, k // 2), generator=gen, device=device, dtype=torch.uint8
            )
            # Block scales 2**-8 .. 2**-2: activations of the size the converter's bounds expect.
            exponents = torch.randint(
                119, 126, (n, k // 32), generator=gen, device=device, dtype=torch.int32
            )
            # All-zero blocks may carry any exponent; they must not limit the re-encoding.
            codes[:4, :16] = 0
            exponents[:4, 0] = 0
            weights[f"layers.0.ffn.experts.{expert}.{proj}.weight"] = codes.view(torch.int8)
            weights[f"layers.0.ffn.experts.{expert}.{proj}.scale"] = exponents.to(torch.uint8)
    norm = torch.rand(_HIDDEN_SIZE, generator=gen, device=device) * 0.1 + 0.15
    weights["layers.0.ffn_norm.weight"] = norm.to(torch.bfloat16)
    return weights


def _check_draft_bank(mapping: Mapping, rank: int, world_size: int) -> None:
    """The draft bank's NVFP4 re-encoding is exact; its MegaMoE layer matches FP32 math on the same quantization at
    the mid and hi draft rows, repeats bit for bit and replays its eager bits in a CUDA graph."""
    device = torch.device("cuda", rank)
    mxfp4 = _mxfp4_draft_experts(device)
    nvfp4 = convert_draft_experts_to_nvfp4(mxfp4, _HIDDEN_SIZE, _CLAMP, _SCALING)
    exact, weights, dense = True, {}, {}
    for expert in range(_DRAFT_EXPERTS):
        for proj in ("w1", "w2", "w3"):
            prefix = f"layers.0.ffn.experts.{expert}.{proj}"
            value = _dequant_nvfp4(
                nvfp4[f"{prefix}.weight"],
                nvfp4[f"{prefix}.weight_scale"],
                nvfp4[f"{prefix}.weight_scale_2"],
            )
            exact &= torch.equal(
                value, _dequant_mxfp4(mxfp4[f"{prefix}.weight"], mxfp4[f"{prefix}.scale"])
            )
            dense[(expert, proj)] = value
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                weights[f"{expert}.{proj}.{kind}"] = nvfp4[f"{prefix}.{kind}"]
    _assert_on_all_ranks(None if exact else "draft NVFP4 re-encoding is not exact")
    fc1_amax = float(nvfp4["layers.0.ffn.experts.0.w1.input_scale"]) * 2688.0
    fc2_amax = float(nvfp4["layers.0.ffn.experts.0.w2.input_scale"]) * 2688.0
    bias = torch.randn(_DRAFT_EXPERTS, generator=torch.Generator().manual_seed(7), device="cpu")
    bias = bias.to(device) * 0.1
    routing = DeepSeekV4MoeRoutingMethod(
        top_k=_DRAFT_TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=_SCALING,
        callable_e_score_correction_bias=lambda: bias,
        callable_tid2eid=lambda: None,
        is_hashed=False,
    )
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["DeepseekV41ForCausalLM"],
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=_INTERMEDIATE_SIZE,
            num_experts=_DRAFT_EXPERTS,
            torch_dtype=torch.bfloat16,
        ),
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        moe_backend="MEGAMOE_CUTEDSL",
        max_num_tokens=max(_DRAFT_ROWS),
        moe_max_num_tokens=max(_DRAFT_ROWS),
    )
    moe = create_moe(
        routing_method=routing,
        model_config=model_config,
        reduce_results=False,
        layer_idx=3,
        activation=SwigluActivation(clamp=_CLAMP),
    )
    moe.load_weights([weights])
    moe.post_load_weights()
    norm = mxfp4["layers.0.ffn_norm.weight"].float()
    del mxfp4, nvfp4, weights

    gen = torch.Generator(device=device).manual_seed(100 + rank)
    for rows in _DRAFT_ROWS:
        # RMSNorm-shaped input under the converter's bound; a few hot experts in the router.
        raw = torch.randn(rows, _HIDDEN_SIZE, generator=gen, device=device)
        x = (raw / raw.pow(2).mean(-1, keepdim=True).sqrt() * norm).to(torch.bfloat16)
        hot = torch.randn(_DRAFT_EXPERTS, generator=gen, device=device)
        logits = (
            torch.randn(rows, _DRAFT_EXPERTS, generator=gen, device=device) * 2.0 + hot
        ).float()
        counts = [rows] * world_size
        out = moe(x, logits, all_rank_num_tokens=counts).clone()
        again = moe(x, logits, all_rank_num_tokens=counts).clone()
        _assert_on_all_ranks(None if torch.equal(out, again) else f"draft {rows}: no repeat")
        ids, topk_weights = routing.apply(logits)
        x_q = _nvfp4_roundtrip(x.float(), 2688.0 / fc1_amax)
        ref = torch.zeros(rows, _HIDDEN_SIZE, device=device)
        for expert in ids.unique().tolist():
            hit, ks = (ids == expert).nonzero(as_tuple=True)
            gate = (x_q[hit] @ dense[(expert, "w1")].t()).clamp(max=_CLAMP)
            up = (x_q[hit] @ dense[(expert, "w3")].t()).clamp(min=-_CLAMP, max=_CLAMP)
            act = gate * torch.sigmoid(gate) * up * topk_weights[hit, ks].float().unsqueeze(-1)
            act = _nvfp4_roundtrip(act, 2688.0 / fc2_amax)
            ref.index_add_(0, hit, act @ dense[(expert, "w2")].t())
        rel = (out.float() - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-12)
        p99, worst = float(torch.quantile(rel, 0.99)), float(rel.max())
        # MegaMoE target layers sit as far from this reference; a misplaced routing weight or a
        # skipped quantization lands several times further out.
        _assert_on_all_ranks(
            None if p99 < 3e-2 and worst < 5e-2 else f"draft {rows}: rel L2 {p99}, {worst}"
        )

    rows = _DRAFT_ROWS[0]
    counts = [rows] * world_size
    x_static = torch.randn(rows, _HIDDEN_SIZE, generator=gen, device=device).to(torch.bfloat16)
    logits_static = torch.randn(rows, _DRAFT_EXPERTS, generator=gen, device=device)
    graph, slot = torch.cuda.CUDAGraph(), torch.empty_like(x_static)
    with torch.cuda.graph(graph):
        slot.copy_(moe(x_static, logits_static, all_rank_num_tokens=counts))
    for _ in range(2):
        x_static.normal_(generator=gen).mul_(0.2)
        logits_static.normal_(generator=gen)
        graph.replay()
        eager = moe(x_static, logits_static, all_rank_num_tokens=counts).clone()
        torch.cuda.synchronize()
        _assert_on_all_ranks(None if torch.equal(slot, eager) else "draft graph differs")


def _run(one_round: bool) -> list:
    """This rank's checks; returns the default path's mid outputs on the CPU for the one-round comparison."""
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import dsv4_route_nvfp4_quant
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.nvlink_domain import (
        one_round_router,
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega import (  # noqa: E501
        block_scaled_swap_ab_mega_moe_kernel as kernel_module,
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.schedulers.fc12_scheduler import (  # noqa: E501
        PhaseInterleavedFc12Scheduler,
    )

    routers = []
    if one_round:
        # The one-round router engages from EP16 (the mid case's DEP32); lower its gate so EP4 takes it.
        kernel_module._ONE_ROUND_MIN_WORLD_SIZE = 1
        init = one_round_router._OneRoundRouter.__init__

        def counted_init(self, *args, **kwargs):
            routers.append(1)
            init(self, *args, **kwargs)

        one_round_router._OneRoundRouter.__init__ = counted_init
    rank, world_size = mpi_rank(), mpi_world_size()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    pretrained = PretrainedConfig(
        architectures=["DeepseekV41ForCausalLM"],
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_experts=_NUM_EXPERTS,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=_SCALING,
        n_hash_layers=0,
        n_shared_experts=1,
        vocab_size=129280,
        swiglu_limit=_CLAMP,
        torch_dtype=torch.bfloat16,
    )
    shared_quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
    routed_quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    model_config = ModelConfig(
        pretrained_config=pretrained,
        mapping=mapping,
        quant_config=shared_quant_config,
        moe_backend="MEGAMOE_CUTEDSL",
        max_num_tokens=_MAX_NUM_TOKENS,
        moe_max_num_tokens=_MAX_NUM_TOKENS,
        megamoe_fuse_shared_expert=True,
    )

    with torch.device("cuda", rank), torch.inference_mode():
        torch.manual_seed(1701)
        quantizer = NVFP4QuantizeUtil(
            num_experts=_NUM_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=routed_quant_config,
        )
        routed_weights = quantizer.create_weights(x_sf_global=torch.tensor(448.0 * 6.0))
        shared_weights = FP8BlockScalesQuantizeUtil(
            num_experts=1,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=shared_quant_config,
        ).create_weights(use_e8m0_scale=True)
        gate_weight = torch.randn(_NUM_EXPERTS, _HIDDEN_SIZE) / _HIDDEN_SIZE**0.5
        biases = _biases(world_size)

        def build(layer_idx: int, fuse: bool) -> DeepseekV4MoE:
            moe = DeepseekV4MoE(
                num_experts=_NUM_EXPERTS,
                top_k=_TOP_K,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                shared_expert_intermediate_size=_INTERMEDIATE_SIZE,
                aux_stream_dict={
                    AuxStreamType.MoeShared: torch.cuda.Stream(),
                    AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
                },
                dtype=torch.bfloat16,
                model_config=model_config,
                override_quant_config=routed_quant_config,
                layer_idx=layer_idx,
            )
            assert isinstance(moe.experts.backend, MegaMoECuteDsl)
            moe.experts.load_weights([routed_weights])
            moe.experts.post_load_weights()
            moe.gate.load_weights(
                [{"weight": gate_weight, "e_score_correction_bias": biases["base"]}]
            )
            moe.shared_experts.gate_up_proj.load_weights(
                [
                    {k: shared_weights[f"0.{w}.{k}"] for k in ("weight", "weight_scale")}
                    for w in ("w1", "w3")
                ]
            )
            moe.shared_experts.down_proj.load_weights(
                [{k: shared_weights[f"0.w2.{k}"] for k in ("weight", "weight_scale")}]
            )
            if fuse:
                # Before the Linear transforms, as DeepseekV4ForCausalLM.post_load_weights does.
                assert moe.experts.fuse_shared_expert(moe.shared_experts)
                moe.shared_experts = None
            else:
                moe.shared_experts.gate_up_proj.post_load_weights()
                moe.shared_experts.down_proj.post_load_weights()
            return moe

        # Two layers share the symmetric workspace, so consecutive calls alternate epoch parity.
        layers = [build(i, True) for i in range(2)]

        torch.manual_seed(1702 + rank)
        calls = []
        for shape, patterns in _SHAPES.items():
            if one_round and shape != "mid":
                continue
            for pattern, counts in enumerate(patterns):
                for variant in _VARIANTS if pattern == 0 else ("base",):
                    x = torch.randn((counts[rank], _HIDDEN_SIZE), dtype=torch.bfloat16)
                    label = f"{shape} {counts} {variant}"
                    calls.append((label, layers[len(calls) % 2], biases[variant], x, counts))

        def run(call):
            _, moe, bias, x, counts = call
            moe.gate.e_score_correction_bias.copy_(bias)
            return moe(x, all_rank_num_tokens=counts)

        if one_round:
            # The output aliases the bucket's combine buffer, which later calls reuse.
            outputs = [run(call).clone() for call in calls]
            _assert_on_all_ranks(None if routers else "no one-round router was built")
            return [out.cpu() for out in outputs]

        separate = build(2, False)
        routing = layers[0].experts.routing_method
        reference = quantizer.create_ref_module(routing, ref_cls=NVFP4RefMegaMoECuteDsl)
        reference.load_weights([routed_weights])

        def dequantize(projection: str) -> torch.Tensor:
            weight = shared_weights[f"0.{projection}.weight"]
            scale = shared_weights[f"0.{projection}.weight_scale"]
            scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
            return weight.float() * scale[: weight.shape[0], : weight.shape[1]]

        w1, w3, w2 = dequantize("w1"), dequantize("w3"), dequantize("w2")

        def shared_reference(x: torch.Tensor) -> torch.Tensor:
            gate = (x.float() @ w1.t()).clamp(max=_CLAMP)
            up = (x.float() @ w3.t()).clamp(-_CLAMP, _CLAMP)
            return (F.silu(gate) * up) @ w2.t()

        def check_reference(call) -> Optional[str]:
            """Routed experts vs per-expert NVFP4 math; the fused shared expert within 2% of the separate one's
            error; the partials' ordered sum vs the router GEMM, and its route+quant bit for bit."""
            label, moe, bias, x, counts = call
            separate.gate.e_score_correction_bias.copy_(bias)
            separate_out = separate(x, all_rank_num_tokens=counts).clone()
            routed = separate.compute_routed_output(x, None, None, counts, True).clone()
            fused_out = run(call).clone()
            partials = moe.gate(x, partials=True)
            if not isinstance(partials, RouterLogitPartials):
                return f"{label}: no router partials"
            summed = _summed(partials)
            problems = []
            try:
                reference.check_accuracy(
                    routed, _reference(reference, moe.experts.routing_method, x, summed)
                )
            except Exception as error:  # noqa: BLE001 - check_accuracy raises a bare Exception
                problems.append(f"routed vs reference: {error}")
            expected = routed.float() + shared_reference(x)
            fused_error, separate_error = (
                _rel_rms(fused_out, expected),
                _rel_rms(separate_out, expected),
            )
            if fused_error > separate_error * _ERROR_RATIO:
                problems.append(f"fused rel-RMS {fused_error} vs separate {separate_error}")
            gemm = router_gemm_autotuned(x, moe.gate.weight)
            if not torch.allclose(summed, gemm, atol=2e-4, rtol=1e-5):
                problems.append("partials' sum differs from the router GEMM")
            scale = torch.tensor([448.0 * 6.0 / 20.0])
            for rows in (x.shape[0], x.shape[0] - 7):
                # A row-sliced view (attention-DP padding stripped) keeps its partials a full row count apart.
                routes = [
                    dsv4_route_nvfp4_quant(
                        logits[:rows],
                        moe.gate.e_score_correction_bias,
                        x[:rows],
                        scale,
                        _TOP_K,
                        _SCALING,
                        torch.float32,
                    )[:4]
                    for logits in (partials, summed)
                ]
                if not all(map(torch.equal, *routes)):
                    problems.append(f"route+quant of the partials differs at {rows} rows")
            return f"{label}: {problems}" if problems else None

        staged, kernels = [], []
        stage = mega_moe_cute_dsl.stage_megamoe_inputs
        resolve = op.Sm100MegaMoENvfp4Runner._resolve_kernel_descriptor

        def recording(self, *args, **kwargs):
            resolved = resolve(self, *args, **kwargs)
            kernels.append(resolved[0])
            return resolved

        def launched():
            schedules = [
                (isinstance(k.scheduler, PhaseInterleavedFc12Scheduler), k.shared_fc2_claims)
                for k in kernels
            ]
            kernels.clear()
            return schedules

        first, default_kinds = [], set()
        for call in calls:
            label, _, _, x, counts = call
            if label.endswith("base") and len(set(counts)) == 1:
                _assert_on_all_ranks(check_reference(call))
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    mega_moe_cute_dsl,
                    "stage_megamoe_inputs",
                    lambda *a, **k: staged.append(1) or stage(*a, **k),
                )
                patch.setattr(op.Sm100MegaMoENvfp4Runner, "_resolve_kernel_descriptor", recording)
                staged.clear()
                launched()
                default = run(call).clone()
                schedule = launched()
                default_kinds.update(interleaved for interleaved, _ in schedule)
                problem = f"{label}: default path staged" if staged and counts[rank] else None
                _assert_on_all_ranks(problem)
                # The plain path: the other scheduler, separate staging and no evict-first hint.
                tactic = (
                    _PHASE_INTERLEAVE_TACTIC
                    if max(counts) <= 256
                    else op.default_megamoe_tactic(256)
                )
                patch.setattr(
                    op, "_default_megamoe_tactic_for_key", lambda key, *, num_tokens: tactic
                )
                # A rejected tactic must fail here, not fall back to another one.
                patch.setattr(op, "_megamoe_default_tactic_candidates", lambda t: (t,))
                patch.setattr(op, "_megamoe_weight_evict_first", lambda *args: False)
                patch.setattr(MegaMoECuteDsl, "_route_quant_bucket", lambda *a, **k: None)
                plain = run(call).clone()
                plain_schedule = launched()
            _assert_on_all_ranks(
                None
                if schedule and plain_schedule and schedule != plain_schedule
                else f"{label}: schedules {schedule} vs {plain_schedule}"
            )
            _assert_on_all_ranks(None if torch.isfinite(default).all() else f"{label}: non-finite")
            _assert_on_all_ranks(
                None if torch.equal(default, plain) else f"{label}: differs from the plain path"
            )
            first.append(default)
        # The default path runs both schedulers: grouped at the mid bucket, phase interleave at hi.
        _assert_on_all_ranks(
            None if default_kinds == {True, False} else f"default schedules {default_kinds}"
        )

        for name, schedule in (("again", calls), ("shifted", calls[1:] + calls[:1])):
            if name == "shifted":
                run(calls[0])  # one more call flips the epoch parity of every later call
            outs = [run(call).clone() for call in schedule]
            if name == "shifted":
                outs = outs[-1:] + outs[:-1]
            for i, out in enumerate(outs):
                same = torch.equal(out, first[i])
                _assert_on_all_ranks(None if same else f"{name} pass: {calls[i][0]} differs")

        graph = torch.cuda.CUDAGraph()
        slots = [torch.empty_like(out) for out in first]
        with torch.cuda.graph(graph):
            for slot, call in zip(slots, calls):
                slot.copy_(run(call))
        for replay in range(_REPLAYS):
            for slot in slots:
                slot.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for i, slot in enumerate(slots):
                same = torch.equal(slot, first[i])
                _assert_on_all_ranks(None if same else f"replay {replay}: {calls[i][0]} differs")
        del graph

        _check_draft_bank(mapping, rank, world_size)
        return [out.cpu() for (label, *_), out in zip(calls, first) if label.startswith("mid")]


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason=f"Requires {_WORLD_SIZE} GPUs",
)
@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="MegaMoE CuteDSL on SM100/SM103")
def test_megamoe_e2e() -> None:
    if mpi_world_size() > 1:
        pytest.skip("Spawns its own worker processes")
    outputs = {}
    for one_round in (False, True):
        with MPIPoolExecutor(
            max_workers=_WORLD_SIZE, initializer=_init_worker, initargs=(sys.path,)
        ) as executor:
            futures = [executor.submit(_run, one_round) for _ in range(_WORLD_SIZE)]
            outputs[one_round] = [future.result() for future in futures]
    for rank, (pushed, one_round) in enumerate(zip(outputs[False], outputs[True])):
        assert len(pushed) == len(one_round) > 0
        for i, (expected, got) in enumerate(zip(pushed, one_round)):
            assert torch.equal(got, expected), (
                f"rank {rank}, mid call {i}: one-round output differs"
            )
