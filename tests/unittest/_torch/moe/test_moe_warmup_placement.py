# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE expert placement learned online, on four GPUs (attention DP, EP4) with DeepSeek-V4.1-Flash MoE shapes at
the mid (25 requests x 6 rows per rank) and hi (320 x 4) decode steps.

Three MegaMoE layers on static balancers: a one-copy layer, one under a drafter's row pass (its own rows per
generation request, as DSpark draft stages run) and one with an extra slot per rank. Traffic phases put the hot
experts on a different rank each, with an idle attention-DP rank, a dummy request and half of the requests on their
first decode step. The routing launch counts only the live rows past that step (each pass its own), straight into
the kernel's bucket. A skipped round keeps the placement; every round moves at most a tenth of a layer's slots on
halved counters (swaps, and on the layer with copies copy moves that re-choose its extra copies), the ranks agree and
every per-slot tensor moves. After every round the eager outputs and those of CUDA graphs captured before it stay
bitwise the one-copy identity placement's; after the last one the counters stop.
"""

import dataclasses
import functools
import pickle
import sys
from types import SimpleNamespace
from typing import Optional

import cloudpickle
import pytest
import torch
from _torch.moe.quantize_utils import NVFP4QuantizeUtil
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import SwigluActivation, create_moe
from tensorrt_llm._torch.moe.fused_moe import moe_warmup_placement as mwp
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod
from tensorrt_llm._utils import get_sm_version, mpi_allgather, mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_WORLD_SIZE = 4
_NUM_EXPERTS = 384
_TOP_K = 6
_HIDDEN_SIZE = 5120
_INTERMEDIATE_SIZE = 2304
_LAYER = 3
_PHASES = 4

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths


def _assert_on_all_ranks(problem: Optional[str]) -> None:
    """Fail on every rank together, so no rank is left waiting in the next collective."""
    problems = mpi_allgather(problem)
    assert not any(problems), problems


def _routing(bias: torch.Tensor) -> DeepSeekV4MoeRoutingMethod:
    return DeepSeekV4MoeRoutingMethod(
        top_k=_TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.5,
        callable_e_score_correction_bias=lambda: bias,
        callable_tid2eid=lambda: None,
        is_hashed=False,
    )


def _build(weights, bias, settings, tokens: int, passes):
    """One layer per (row pass, rows per request, extra slots per rank) of ``passes`` on a static balancer, with a
    warmup placement controller for ``settings``."""
    rank, world_size = mpi_rank(), mpi_world_size()
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )

    def lb_config(extra_slots):
        config = MoeLoadBalancerConfig(
            num_slots=_NUM_EXPERTS + extra_slots * world_size, layer_updates_per_iter=0
        )
        config.setup(ep_rank=rank, ep_size=world_size)
        return config

    model_config = dict(
        pretrained_config=PretrainedConfig(
            architectures=["DeepseekV41ForCausalLM"],
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=_INTERMEDIATE_SIZE,
            num_experts=_NUM_EXPERTS,
            torch_dtype=torch.bfloat16,
        ),
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        moe_backend="MEGAMOE_CUTEDSL",
        max_num_tokens=tokens,
        moe_max_num_tokens=tokens,
    )
    controller = mwp.WarmupPlacementController(settings, ep_rank=rank, ep_size=world_size)
    balancer = MoeLoadBalancer(
        ep_rank=rank, ep_size=world_size, layer_updates_per_iter=0, warmup_placement=controller
    )
    layers = []
    with balancer:
        for i, (row_space, rows_per_request, extra_slots) in enumerate(passes):
            layers.append(
                create_moe(
                    routing_method=_routing(bias),
                    model_config=ModelConfig(
                        **model_config,
                        moe_load_balancer=lb_config(extra_slots),
                        moe_row_space=row_space,
                        moe_row_space_rows_per_request=rows_per_request,
                    ),
                    reduce_results=False,
                    layer_idx=_LAYER + i,
                    activation=SwigluActivation(clamp=10.0),
                    allow_backend_degradation=False,
                )
            )
    for moe in layers:
        moe.load_weights([weights])
        moe.post_load_weights()
    balancer.register_weight_slots_after_to_cuda()
    balancer.finalize_model()
    return layers, balancer, controller


def _forward(moe, balancer, x, logits):
    with MoeLoadBalancerIterContext(balancer):
        return moe(x, logits, all_rank_num_tokens=[x.shape[0]] * mpi_world_size())


def _slot_tensors(backend, slots_per_rank):
    """Every non-empty CUDA parameter or buffer of the backend with a leading per-slot dim."""
    named = list(backend.named_parameters(recurse=False)) + list(
        backend.named_buffers(recurse=False)
    )
    return {
        name: t.detach()
        for name, t in named
        if t.is_cuda and t.dim() > 0 and t.shape[0] == slots_per_rank and t.numel() > 0
    }


def _check_moved_state(controller, moe) -> Optional[str]:
    """The controller moves every per-slot tensor of the backend."""
    layer = controller.layers[moe._warmup_placement_index]
    moved = {t.data_ptr() for t in layer.slot_tensors}
    missed = [
        name
        for name, t in _slot_tensors(moe.backend, layer.slots_per_rank).items()
        if t.data_ptr() not in moved
    ]
    return f"per-slot tensors not moved: {missed}" if missed else None


def _phase_inputs(phase: int, tokens: int):
    """This rank's input in a traffic phase whose hot experts are rank ``phase``'s first 24; the
    last rank is idle (attention-DP dummy rows, zero logits)."""
    gen = torch.Generator(device="cpu").manual_seed(1900 + 13 * mpi_rank() + phase)
    x = torch.randn((tokens, _HIDDEN_SIZE), generator=gen, device="cpu") * 0.2
    logits = torch.randn((tokens, _NUM_EXPERTS), generator=gen, device="cpu")
    hot = phase * (_NUM_EXPERTS // mpi_world_size())
    logits[:, hot : hot + 24] += 4.0
    if mpi_rank() == mpi_world_size() - 1:
        logits.zero_()
    return x.to(torch.bfloat16).cuda(), logits.cuda()


def _phase_requests(gens: int, idle: bool = False):
    """Live generation requests of a forward: half past their first decode step, half on it, then an
    attention-DP dummy; only the dummy on the idle last rank (or anywhere when ``idle``)."""
    dummy = [SimpleNamespace(is_dummy=True)]
    if idle or mpi_rank() == mpi_world_size() - 1:
        return SimpleNamespace(context_requests=[], generation_requests=dummy)
    half = gens // 2
    live = [
        SimpleNamespace(is_dummy=False, max_beam_num_tokens=1100, orig_prompt_len=100)
        for _ in range(half)
    ]
    live += [
        SimpleNamespace(is_dummy=False, max_beam_num_tokens=101, orig_prompt_len=100)
        for _ in range(gens - half)
    ]
    return SimpleNamespace(context_requests=[], generation_requests=live + dummy)


def _check_live_round(controller, layers, balancer, inputs, reference, graphs, before, draft_len):
    """After a round: one table on every rank, copies on distinct ranks, at most the swap budget
    moved, halved counters, and identity outputs eager and replayed."""
    world = mpi_world_size()
    for i, (moe, old) in enumerate(zip(layers, before["assign"])):
        new = moe.initial_global_assignments
        tables = mpi_allgather(new)
        _assert_on_all_ranks(
            None if all(t == tables[0] for t in tables) else f"layer {i}: ranks disagree"
        )
        spr = moe.expert_size_per_partition
        if len(new) > _NUM_EXPERTS:
            cover = mwp.is_replicated_cover(new, _NUM_EXPERTS, spr, world)
            _assert_on_all_ranks(None if cover else f"layer {i}: two copies on one rank")
        moved = sum(a != b for a, b in zip(old, new))
        budget = 2 * (int(0.1 * len(new)) // 2)
        _assert_on_all_ranks(
            f"layer {i}: {moved} slots moved > {budget}" if moved > budget else None
        )
        before["moved"][i] += moved
    halved = torch.equal(
        controller._hist, torch.floor(before["hist"].double() * 0.5).to(torch.int32)
    )
    _assert_on_all_ranks(None if halved else "counters not halved after the round")
    controller.set_live_rows(_phase_requests(0, idle=True), draft_len=draft_len)
    for i, moe in enumerate(layers):
        for phase, (x, logits) in enumerate(inputs):
            out = _forward(moe, balancer, x, logits).clone()
            graph, graph_out = graphs[i * _PHASES + phase]
            graph_out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            ref = reference[i][phase]
            _assert_on_all_ranks(
                None if torch.equal(out, ref) else f"layer {i} eager phase {phase}"
            )
            _assert_on_all_ranks(
                None if torch.equal(graph_out, ref) else f"layer {i} graph phase {phase}"
            )


def _run_placement(gens: int, draft_len: int) -> None:
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    rows_per_request = 1 + draft_len
    # The live requests' rows, then one dummy request's.
    tokens = (gens + 1) * rows_per_request
    patch = pytest.MonkeyPatch()
    with torch.device("cuda", rank), torch.inference_mode():
        torch.manual_seed(1801)
        bias = torch.randn(_NUM_EXPERTS, dtype=torch.float32) * 0.05
        quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
        weights = NVFP4QuantizeUtil(
            num_experts=_NUM_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=quant_config,
        ).create_weights(x_sf_global=torch.tensor(448.0 * 6.0))
        settings = mwp.WarmupPlacementSettings(
            start_step=0,
            interval_steps=1,
            max_migrations=_PHASES,
            min_rows=1,
            min_pos=2,
            decay=0.5,
            max_moved_fraction=0.1,
            min_gain=0.0,
        )
        layers, balancer, controller = _build(
            weights,
            bias,
            settings,
            tokens,
            passes=(("main", None, 0), ("draft", draft_len, 0), ("main", None, 1)),
        )
        del weights
        default = list(layers[2].initial_global_assignments)
        assert all(moe.backend.maps_expert_slots for moe in layers)
        inputs = [_phase_inputs(phase, tokens) for phase in range(_PHASES)]
        # The output aliases the bucket's combine buffer, which the next call reuses.
        reference = [[_forward(moe, balancer, x, lg).clone() for x, lg in inputs] for moe in layers]
        for phase in range(_PHASES):
            # Copies hold the same bytes: the layer with extra slots gives the one-copy layer's outputs.
            same = torch.equal(reference[2][phase], reference[0][phase])
            _assert_on_all_ranks(None if same else f"copies change phase {phase}")
        graphs = []
        for moe in layers:
            for x, logits in inputs:
                graph, out = torch.cuda.CUDAGraph(), torch.empty_like(reference[0][0])
                with torch.cuda.graph(graph):
                    out.copy_(_forward(moe, balancer, x, logits))
                graphs.append((graph, out))
        staging, separate_counts = [], []
        stage = mega_moe_cute_dsl.stage_megamoe_inputs
        patch.setattr(
            mega_moe_cute_dsl,
            "stage_megamoe_inputs",
            lambda *a, **k: staging.append(1) or stage(*a, **k),
        )
        patch.setattr(mwp, "count_routing", lambda *a, **k: separate_counts.append(1))
        moved = [0] * len(layers)
        # Step 1 plans under a min_round_gain no plan meets: it moves nothing but halves the
        # counters as a round that moves does; step 2 then places phase 0 from both of its forwards.
        skip = dataclasses.replace(settings, min_round_gain=1e9)
        for step in range(_PHASES + 2):
            torch.cuda.synchronize()
            before = {
                "hist": controller._hist.clone(),
                "assign": [list(moe.initial_global_assignments) for moe in layers],
                "moved": moved,
            }
            rounds = controller.migrations_done
            controller.settings = skip if step == 1 else settings
            controller.on_step(step)
            ran = controller.migrations_done > rounds
            _assert_on_all_ranks(None if ran == (step > 1) else f"step {step}: round ran {ran}")
            if step == 1:
                kept = all(
                    list(moe.initial_global_assignments) == old
                    for moe, old in zip(layers, before["assign"])
                )
                _assert_on_all_ranks(None if kept else "a skipped round changed the placement")
                halved = torch.equal(
                    controller._hist, torch.floor(before["hist"].double() * 0.5).to(torch.int32)
                )
                _assert_on_all_ranks(None if halved else "a skipped round kept its counters")
            if ran:
                _check_live_round(
                    controller, layers, balancer, inputs, reference, graphs, before, draft_len
                )
            if step == 2:
                for moe in layers:
                    _assert_on_all_ranks(_check_moved_state(controller, moe))
                # The first moving round re-chooses the experts with a second copy.
                assign = layers[2].initial_global_assignments
                rechosen = {e for e in assign if assign.count(e) > 1} != {
                    e for e in default if default.count(e) > 1
                }
                _assert_on_all_ranks(None if rechosen else "no copy moved")
            if step == _PHASES + 1:
                break
            controller.set_live_rows(_phase_requests(gens), draft_len=draft_len)
            x, logits = inputs[max(0, step - 1)]
            for moe in layers:
                _forward(moe, balancer, x, logits)
            torch.cuda.synchronize()
            if step == 0:
                # Only the requests past their first decode step count, each pass its own rows.
                for moe in layers:
                    draft = controller.layers[moe._warmup_placement_index].row_space != 0
                    rows = (gens // 2) * (draft_len if draft else rows_per_request)
                    ids = moe.routing_method.apply(logits)[0][:rows]
                    live = logits[:rows].abs().amax(dim=1) > 0
                    expected = torch.bincount(ids[live].reshape(-1).long(), minlength=_NUM_EXPERTS)
                    counted = controller._hist[moe._warmup_placement_index, :_NUM_EXPERTS].long()
                    same = torch.equal(counted, expected)
                    _assert_on_all_ranks(None if same else "counts differ from the counted rows")
        _assert_on_all_ranks(None if controller.frozen else "not frozen after the last round")
        _assert_on_all_ranks(None if all(moved) else f"a layer never moved: {moved}")
        frozen_counts = controller._hist.clone()
        _forward(layers[0], balancer, *inputs[1])
        torch.cuda.synchronize()
        _assert_on_all_ranks(
            None if torch.equal(controller._hist, frozen_counts) else "counted after the freeze"
        )
        patch.undo()
        _assert_on_all_ranks(f"re-staged launches: {len(staging)}" if staging else None)
        _assert_on_all_ranks(
            f"separate count launches: {len(separate_counts)}" if separate_counts else None
        )


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason=f"Requires {_WORLD_SIZE} GPUs",
)
@pytest.mark.skipif(get_sm_version() not in (100, 103), reason="MegaMoE CuteDSL on SM100/SM103")
@pytest.mark.parametrize("gens,draft_len", [(25, 5), (320, 3)], ids=["mid", "hi"])
def test_megamoe_placement(gens: int, draft_len: int) -> None:
    run = functools.partial(_run_placement, gens, draft_len)
    if mpi_world_size() == _WORLD_SIZE:
        run()
        return
    if mpi_world_size() > 1:
        pytest.skip(f"Requires MPI world size {_WORLD_SIZE} or a single process")
    with MPIPoolExecutor(
        max_workers=_WORLD_SIZE, initializer=_init_worker, initargs=(sys.path,)
    ) as executor:
        results = [executor.submit(run) for _ in range(_WORLD_SIZE)]
        assert all(result.result() is None for result in results)
