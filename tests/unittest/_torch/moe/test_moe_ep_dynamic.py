# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic expert parallelism on the DS-V4.1 MoE at attention TP against the static expert split, with
either route kernel, eager and replayed in a CUDA graph."""

import itertools
import pickle
import sys
from contextlib import ExitStack

import cloudpickle
import pytest
import torch
from _torch.moe import test_deepseek_v41_parallel_moe as parallel_moe
from _torch.moe.quantize_utils import NVFP4QuantizeUtil
from _torch.moe.test_deepseek_v41_parallel_moe import (
    _CLAMP,
    _HIDDEN_SIZE,
    _INTERMEDIATE_SIZE,
    _NUM_EXPERTS,
    _ROUTED_SCALE,
    _TOP_K,
    _local_mpi_topology,
)
from _torch.moe.test_deepseek_v41_shared_moe import _nvfp4_input
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE
from tensorrt_llm._torch.models.modeling_deepseekv41 import _MOE_CUTE_FC2_ROWS
from tensorrt_llm._torch.moe.fused_moe import configurable_moe
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MOE_DECODE_OPTIONS, MoEDecodeOptions
from tensorrt_llm._torch.moe.kernels.blackwell import dsv4_route_one
from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import is_blackwell_datacenter, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# Workers unpickle this module's functions by value: they must not reference the parallel_moe module.
parallel_moe_executor = parallel_moe.parallel_moe_executor
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_INPUT_RANGE = 4.0


def _rows(num_rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_rows, _HIDDEN_SIZE, device="cuda", generator=generator)
    return x.clamp(-_INPUT_RANGE, _INPUT_RANGE).to(torch.bfloat16)


def _routed(moe: DeepseekV4MoE, x: torch.Tensor, fp4_input: bool, do_finalize: bool):
    fp4 = _nvfp4_input(x, moe.experts.backend.fc31_input_scale) if fp4_input else None
    return moe.compute_routed_output(x, fp4, None, None, do_finalize=do_finalize)


def _partial(output) -> torch.Tensor:
    """An un-finalized run's rows summed per token on this rank (permutation -1: another rank's)."""
    rows, weights = output[0], output[1]
    permutation = output[2].flatten()[: weights.numel()].view(weights.shape)
    gathered = rows[permutation.clamp(min=0).long()].float()
    gathered = torch.where((permutation >= 0)[..., None], gathered, 0.0)
    return (gathered * weights.float()[..., None]).sum(1).to(rows.dtype)


def _moves(moe: DeepseekV4MoE, x: torch.Tensor) -> bool:
    """Whether the home-first split moves an expert: one rank's half holds over ceil(used / 2)."""
    used = moe.gate.routing_method.apply(moe.gate(x))[0].unique()
    lower = int((used < _NUM_EXPERTS // 2).sum())
    return max(lower, used.numel() - lower) > (used.numel() + 1) // 2


def _close(label: str, got: torch.Tensor, want: torch.Tensor) -> None:
    """Moved experts round the per-rank BF16 partials in another grouping."""
    scale = want.float().abs().max().item()
    max_abs = (got.float() - want.float()).abs().max().item()
    assert torch.isfinite(got).all() and 0 < scale and max_abs <= 2**-6 * scale, (label, max_abs)


def _run_dynamic_ep() -> None:
    local_rank, local_size = _local_mpi_topology()
    torch.cuda.set_device(local_rank)
    world_size = mpi_world_size()
    mapping = Mapping(
        world_size=world_size,
        rank=mpi_rank(),
        gpus_per_node=local_size,
        tp_size=world_size,
        moe_ep_size=2,
        moe_tp_size=world_size // 2,
    )
    routed_quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=_INTERMEDIATE_SIZE,
            num_experts=_NUM_EXPERTS,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=_ROUTED_SCALE,
            n_hash_layers=0,
            vocab_size=129280,
            swiglu_limit=_CLAMP,
            torch_dtype=torch.bfloat16,
        ),
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128),
        moe_backend="TRTLLM",
        max_num_tokens=256,
        max_batch_size=1,
    )
    # DeepSeek-V4.1's opt-ins: small un-finalized NVFP4 FC2s as the CuTe DSL kernel, the one-cluster
    # route kernel, dynamic EP.
    model_config.extra_attrs[MOE_DECODE_OPTIONS] = MoEDecodeOptions(
        cute_fc2_rows=_MOE_CUTE_FC2_ROWS, one_cluster_route=True
    )
    model_config.extra_attrs["moe_dynamic_ep"] = True

    with (
        torch.device(f"cuda:{local_rank}"),
        torch.inference_mode(),
        pytest.MonkeyPatch.context() as env,
        ExitStack() as stack,
    ):
        torch.manual_seed(1701)
        gate_weight = torch.randn(_NUM_EXPERTS, _HIDDEN_SIZE) / _HIDDEN_SIZE**0.5
        gate_bias = torch.randn(_NUM_EXPERTS, dtype=torch.float32) * 0.05
        routed_weights = NVFP4QuantizeUtil(
            num_experts=_NUM_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=routed_quant_config,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=_CLAMP,
        ).create_weights(x_sf_global=torch.tensor(448.0 * 6.0 / _INPUT_RANGE))
        modules = []
        # The production gate wants MoE TP 4 or more: this world (MoE TP 2) loads the static split,
        # then the dynamic one with the bound lowered.
        for min_moe_tp in (4, 2):
            env.setattr(configurable_moe, "_DYNAMIC_EP_MIN_MOE_TP", min_moe_tp)
            moe = DeepseekV4MoE(
                num_experts=_NUM_EXPERTS,
                top_k=_TOP_K,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                shared_expert_intermediate_size=_INTERMEDIATE_SIZE,
                aux_stream_dict={AuxStreamType.MoeShared: torch.cuda.Stream()},
                dtype=torch.bfloat16,
                model_config=model_config,
                override_quant_config=routed_quant_config,
                layer_idx=0,
            )
            stack.enter_context(moe.experts)
            moe.gate.weight.copy_(gate_weight)
            moe.gate.e_score_correction_bias.copy_(gate_bias)
            moe.experts.load_weights([dict(routed_weights)])
            moe.experts.post_load_weights()
            modules.append(moe)
        static, dynamic = modules
        assert static.experts.dynamic_ep is None and dynamic.experts.dynamic_ep is not None

        def all_reduced(output) -> torch.Tensor:
            return static.allreduce(output if torch.is_tensor(output) else _partial(output))

        # At 1 and 6 rows (routing tables) an input whose split keeps every expert home and one that
        # moves some; 16 rows run the static split in both modules.
        cases = [
            next(x for x in (_rows(n, s) for s in range(1, 200)) if _moves(static, x) == moves)
            for n, moves in itertools.product((1, 6), (False, True))
        ] + [_rows(16, seed=1)]
        x = torch.empty(16, _HIDDEN_SIZE, dtype=torch.bfloat16)
        with nccl_window_tensor_scope(x):
            with autotune():
                for case in cases:
                    x[: len(case)].copy_(case)
                    for moe in modules:
                        _routed(moe, x[: len(case)], True, False)
            # Captured once, replayed on new rows written into the captured input.
            rows = x[:6]
            rows.copy_(_rows(6, seed=1))
            _routed(dynamic, rows, True, False)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _routed(dynamic, rows, True, False)
            for seed in (2, 3, 4):
                rows.copy_(_rows(6, seed))
                graph.replay()
                want = all_reduced(_routed(static, rows, True, False))
                _close(f"graph replay, seed {seed}", all_reduced(captured), want)

            for one_cluster in (True, False):
                if not one_cluster:
                    env.setattr(dsv4_route_one, "applies", lambda *args: False)
                for case, fp4_input, do_finalize in itertools.product(
                    cases, (True, False), (True, False)
                ):
                    rows = x[: len(case)]
                    rows.copy_(case)
                    label = f"{len(case)} rows, one-cluster {one_cluster}, fp4 {fp4_input}, "
                    label += f"finalized {do_finalize}"
                    want = all_reduced(_routed(static, rows, fp4_input, do_finalize))
                    got = all_reduced(_routed(dynamic, rows, fp4_input, do_finalize))
                    if len(case) > 8 or not _moves(static, rows):
                        assert torch.equal(got, want), label
                    else:
                        _close(label, got, want)
            torch.cuda.synchronize()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_blackwell_datacenter(),
    reason="dynamic expert parallelism runs on SM100/SM103",
)
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "parallel_moe_executor", [pytest.param(4, id="tp4-moetp2-ep2")], indirect=True
)
def test_moe_ep_dynamic_matches_static_split(parallel_moe_executor: MPIPoolExecutor | None) -> None:
    if parallel_moe_executor is None:
        _run_dynamic_ep()
        return
    results = [
        parallel_moe_executor.submit(_run_dynamic_ep)
        for _ in range(parallel_moe_executor.num_workers)
    ]
    assert all(result.result() is None for result in results)
