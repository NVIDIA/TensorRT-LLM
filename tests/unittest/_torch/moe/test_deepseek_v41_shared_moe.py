# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantized DS-V4.1 routed/shared MoE parity with real MPI reductions."""

import pickle
import sys
from itertools import repeat

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import per_token_cast_to_fp8_e8m0
from _torch.moe import test_deepseek_v41_parallel_moe as parallel_moe
from _torch.moe.quantize_utils import FP8BlockScalesQuantizeUtil, MXFP4MXFP8QuantizeUtil
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

parallel_moe_executor = parallel_moe.parallel_moe_executor
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _load_shared_weights(mlp: GatedMLP, weights: dict[str, torch.Tensor]) -> None:
    mlp.gate_up_proj.load_weights(
        [
            {key: weights[f"0.{projection}.{key}"] for key in ("weight", "weight_scale")}
            for projection in ("w1", "w3")
        ]
    )
    mlp.down_proj.load_weights(
        [{key: weights[f"0.w2.{key}"] for key in ("weight", "weight_scale")}]
    )
    mlp.gate_up_proj.post_load_weights()
    mlp.down_proj.post_load_weights()


def _ue8m0_activation_qdq(x: torch.Tensor) -> torch.Tensor:
    quantized, scales = per_token_cast_to_fp8_e8m0(x)
    return quantized.float() * scales.repeat_interleave(128, dim=1)


def _shared_fp8_reference(
    x: torch.Tensor, weights: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the shared output and global pre-clamp gate/up projection."""
    import torch
    from _torch.moe import test_deepseek_v41_parallel_moe as parallel_moe

    def dequantize_weight(projection: str) -> torch.Tensor:
        scales = weights[f"0.{projection}.weight_scale"]
        scales = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        return weights[f"0.{projection}.weight"].float() * scales

    quantized_input = _ue8m0_activation_qdq(x)
    gate = F.linear(quantized_input, dequantize_weight("w1")).to(x.dtype)
    up = F.linear(quantized_input, dequantize_weight("w3")).to(x.dtype)
    hidden = (
        F.silu(gate.float().clamp(max=parallel_moe._CLAMP))
        * up.float().clamp(-parallel_moe._CLAMP, parallel_moe._CLAMP)
    ).to(x.dtype)
    quantized_hidden = _ue8m0_activation_qdq(hidden)
    native_quantized, native_scales = torch.ops.trtllm.fp8_quantize_1x128(
        hidden.contiguous(), use_ue8m0=True
    )
    native_hidden = native_quantized.float() * native_scales.t().repeat_interleave(128, dim=1)
    torch.testing.assert_close(quantized_hidden, native_hidden, rtol=0, atol=0)
    # Clamp10 produces BF16 100: UE8M0 quantization rounds it to 96, not 100.
    saturated = hidden == parallel_moe._CLAMP**2
    assert saturated.any()
    assert (quantized_hidden[saturated] == 96).all()
    output = F.linear(quantized_hidden, dequantize_weight("w2")).to(x.dtype)
    return output, torch.cat((gate, up), dim=-1)


def _run_shared_moe(ep_size: int, attention_dp: bool) -> None:
    from _torch.moe import test_deepseek_v41_parallel_moe as parallel_moe

    rank = mpi_rank()
    world_size = mpi_world_size()
    local_rank, local_size = parallel_moe._local_mpi_topology()
    torch.cuda.set_device(local_rank)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=local_size,
        tp_size=world_size,
        moe_ep_size=ep_size,
        moe_tp_size=world_size // ep_size,
        enable_attention_dp=attention_dp,
    )
    routed_quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
    shared_quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            hidden_size=parallel_moe._HIDDEN_SIZE,
            intermediate_size=parallel_moe._INTERMEDIATE_SIZE,
            num_experts=parallel_moe._NUM_EXPERTS,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=parallel_moe._ROUTED_SCALE,
            n_hash_layers=0,
            vocab_size=129280,
            swiglu_limit=parallel_moe._CLAMP,
            torch_dtype=torch.bfloat16,
        ),
        mapping=mapping,
        quant_config=shared_quant_config,
        moe_backend="CUTLASS",
        max_num_tokens=256,
    )

    with torch.device(f"cuda:{local_rank}"), torch.inference_mode():
        torch.manual_seed(1701)
        moe = DeepseekV4MoE(
            num_experts=parallel_moe._NUM_EXPERTS,
            top_k=parallel_moe._TOP_K,
            hidden_size=parallel_moe._HIDDEN_SIZE,
            intermediate_size=parallel_moe._INTERMEDIATE_SIZE,
            shared_expert_intermediate_size=parallel_moe._INTERMEDIATE_SIZE,
            aux_stream_dict={AuxStreamType.MoeShared: torch.cuda.Stream()},
            dtype=torch.bfloat16,
            model_config=model_config,
            override_quant_config=routed_quant_config,
            layer_idx=0,
        )
        with moe.experts:
            assert isinstance(moe.experts.backend, CutlassFusedMoE)
            assert not moe.experts.reduce_results
            expected_shared_tp = 1 if attention_dp else 2
            expected_shared_rank = rank % expected_shared_tp
            assert moe.shared_experts.gate_up_proj.mapping.tp_size == expected_shared_tp
            assert moe.shared_experts.down_proj.mapping.tp_size == expected_shared_tp
            assert moe.shared_experts.gate_up_proj.mapping.tp_rank == expected_shared_rank
            assert moe.shared_experts.down_proj.mapping.tp_rank == expected_shared_rank
            assert moe.shared_output_scale == (None if attention_dp else 2 / world_size)
            assert moe.shared_experts.down_proj.weight.dtype == torch.float8_e4m3fn

            moe.gate.weight.copy_(
                torch.randn_like(moe.gate.weight) / parallel_moe._HIDDEN_SIZE**0.5
            )
            moe.gate.e_score_correction_bias.copy_(
                torch.randn(parallel_moe._NUM_EXPERTS, dtype=torch.float32) * 0.05
            )
            routed_quantizer = MXFP4MXFP8QuantizeUtil(
                num_experts=parallel_moe._NUM_EXPERTS,
                dtype=torch.bfloat16,
                intermediate_size=parallel_moe._INTERMEDIATE_SIZE,
                hidden_size=parallel_moe._HIDDEN_SIZE,
                quant_config=routed_quant_config,
                swiglu_alpha=1.0,
                swiglu_beta=0.0,
                swiglu_limit=parallel_moe._CLAMP,
            )
            routed_weights = routed_quantizer.create_weights(input_hidden_alignment=128)
            reference_routed = routed_quantizer.create_ref_module(
                parallel_moe._ReferenceRouting(moe.gate.e_score_correction_bias)
            )
            reference_routed.load_weights([routed_weights])
            moe.experts.load_weights([routed_weights])
            moe.experts.post_load_weights()
            del routed_weights
            if not attention_dp:
                # TP4's 576-wide routed shards need 128-element alignment.
                assert parallel_moe._INTERMEDIATE_SIZE // mapping.moe_tp_size == 576
                weight_elements = torch.iinfo(moe.experts.backend.w2_weight.dtype).bits // 4
                assert moe.experts.backend.w2_weight.shape[-1] * weight_elements == 640

            shared_quantizer = FP8BlockScalesQuantizeUtil(
                num_experts=1,
                dtype=torch.bfloat16,
                intermediate_size=parallel_moe._INTERMEDIATE_SIZE,
                hidden_size=parallel_moe._HIDDEN_SIZE,
                quant_config=shared_quant_config,
                swiglu_alpha=1.0,
                swiglu_beta=0.0,
                swiglu_limit=parallel_moe._CLAMP,
            )
            shared_weights = shared_quantizer.create_weights(use_e8m0_scale=True)
            # The helper's repeating FP8 values otherwise make the two 1152-row
            # shards identical. Distinct block scales expose wrong shard loads.
            # Powers of two preserve UE8M0 scales in both GEMM backends.
            for projection in ("w1", "w2", "w3"):
                scales = shared_weights[f"0.{projection}.weight_scale"]
                row_exponent = torch.arange(scales.shape[0]) // (scales.shape[0] // 2) - 1
                column_exponent = torch.arange(scales.shape[1]) // (scales.shape[1] // 2) - 1
                row_gain = torch.exp2(row_exponent.float())[:, None]
                column_gain = torch.exp2(column_exponent.float())[None, :]
                scales.mul_(row_gain * column_gain)
                if projection == "w2":
                    # Keep both branches visible in the combined-output check.
                    scales.div_(1024)
                elif projection == "w3":
                    # Distinguish gate/up weights so their swap cannot pass.
                    scales.mul_(2)
            _load_shared_weights(moe.shared_experts, shared_weights)

            counts = list(range(1, world_size + 1))
            # Binary rows stay exact under FP8 quantization and distinguish DP slices.
            # Nonnegative inputs keep the shared expert out of the zero SiLU tail.
            torch.manual_seed(1702)
            global_x = torch.randint(
                0, 2, (sum(counts), parallel_moe._HIDDEN_SIZE), dtype=torch.int32
            ).to(torch.bfloat16)
            start = sum(counts[:rank])
            x = global_x[start : start + counts[rank]] if attention_dp else global_x
            all_rank_num_tokens = counts if attention_dp else None
            reference_logits = F.linear(x.float(), moe.gate.weight.float())
            expected_routed = reference_routed(x, reference_logits)
            expected_shared, expected_gate_up = _shared_fp8_reference(x, shared_weights)
            del shared_weights
            assert expected_routed.abs().max() > 0
            assert expected_shared.abs().max() > 0

            local_width = parallel_moe._INTERMEDIATE_SIZE // expected_shared_tp
            shard_start = expected_shared_rank * local_width
            shard_end = shard_start + local_width
            expected_gate, expected_up = expected_gate_up.chunk(2, dim=-1)
            expected_local_gate_up = torch.cat(
                (expected_gate[:, shard_start:shard_end], expected_up[:, shard_start:shard_end]),
                dim=-1,
            )
            # Saturation hides wrong positive FC1 shards from the final output.
            torch.testing.assert_close(
                moe.shared_experts.gate_up_proj(x),
                expected_local_gate_up,
                rtol=1e-2,
                atol=0.1,
            )
            actual_shared = moe.shared_experts(x)
            if moe.shared_output_scale is not None:
                actual_shared *= moe.shared_output_scale
            if moe.allreduce is not None:
                actual_shared = moe.allreduce(actual_shared)
            torch.testing.assert_close(actual_shared, expected_shared, rtol=1e-2, atol=0.1)

            actual_routed = moe.compute_routed_output(
                x, None, None, all_rank_num_tokens, do_finalize=True
            )
            if moe.allreduce is not None:
                actual_routed = moe.allreduce(actual_routed)
            reference_routed.check_accuracy(actual_routed, expected_routed)

            actual = moe(x, all_rank_num_tokens=all_rank_num_tokens)
            torch.cuda.synchronize()
            assert actual.shape == x.shape
            assert torch.isfinite(actual).all()
            expected = expected_routed + expected_shared
            reference_routed.check_accuracy(actual, expected)
            if attention_dp:
                # Send rank 0's prefix to the last rank to check row-order sensitivity.
                misordered = torch.cat(mpi_comm().allgather(actual.cpu())).roll(
                    sum(counts[:-1]), dims=0
                )
                misordered = misordered[start : start + counts[rank]].to(actual.device)
                with pytest.raises(Exception, match="Mismatch percentage"):
                    reference_routed.check_accuracy(misordered, expected)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "parallel_moe_executor,ep_size,attention_dp",
    [
        pytest.param(4, 1, False, id="tp4-ep1-sharedtp2"),
        pytest.param(4, 4, True, id="tp4-ep4-dp-sharedtp1"),
    ],
    indirect=["parallel_moe_executor"],
)
def test_deepseek_v41_shared_moe(
    parallel_moe_executor: MPIPoolExecutor | None, ep_size: int, attention_dp: bool
) -> None:
    if parallel_moe_executor is None:
        _run_shared_moe(ep_size, attention_dp)
        return
    world_size = parallel_moe_executor.num_workers
    results = parallel_moe_executor.map(
        _run_shared_moe, repeat(ep_size, world_size), repeat(attention_dp, world_size)
    )
    assert all(result is None for result in results)
