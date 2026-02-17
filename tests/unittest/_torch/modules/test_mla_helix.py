# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle
import sys
import weakref
from dataclasses import dataclass
from functools import partial
from typing import Optional

import _torch.modules.helix_test_utils as helix_utils
import cloudpickle
import pytest
import torch
from _torch.modules.helix_test_utils import (
    CACHE_TYPE_SELFKONLY,
    activate_all_ranks_for_context,
    compute_mismatch_ratio,
    copy_weights_for_cp,
    create_helix_gen_metadata,
    run_helix_test,
    setup_kv_and_metadata,
    split_inputs_for_rank,
)
from mpi4py import MPI

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionMetadata,
    KVCacheParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(helix_utils)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# Values for deepseek_v3_lite.
@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 32
    q_lora_rank: int = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 2560
    rope_theta: float = 10000.0
    rope_scaling: bool = False
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 32
    # TODO only 1 is supported for now here
    predicted_tokens_per_seq: int = 1
    bias: bool = False
    batch: int = 8
    ctx_len: int = 1024
    # note: need to use fairly high tolerances because the softmax stats can lose
    # a lot of precision and we're using bf16 here.
    atol: float = 1e-1
    rtol: float = 5e-2

    @property
    def max_position_embeddings(self) -> int:
        # Ensure that max_position_embeddings is set large enough for every scenario.
        return self.ctx_len + 1


all_scenarios = [
    Scenario(batch=1, ctx_len=64),
    Scenario(batch=1, ctx_len=512),
    Scenario(batch=1, ctx_len=1024),
    Scenario(batch=1, ctx_len=2048),
    Scenario(batch=1, ctx_len=4096),
    Scenario(batch=1, ctx_len=8192),
    Scenario(batch=1, ctx_len=16384),
    Scenario(batch=1, ctx_len=32768),
    Scenario(batch=1, ctx_len=65536),
    Scenario(batch=1, ctx_len=131072),
    Scenario(batch=1, ctx_len=262144),
    Scenario(batch=1, ctx_len=524288),
    Scenario(batch=1, ctx_len=1048576),
    Scenario(batch=8, ctx_len=1024),
    Scenario(batch=8, ctx_len=2048),
    Scenario(batch=8, ctx_len=4096),
    Scenario(batch=8, ctx_len=8192),
    Scenario(batch=8, ctx_len=16384),
    Scenario(batch=8, ctx_len=32768),
    Scenario(batch=8, ctx_len=65536),
    Scenario(batch=8, ctx_len=131072),
    Scenario(batch=16, ctx_len=1024),
    Scenario(batch=16, ctx_len=2048),
    Scenario(batch=16, ctx_len=4096),
    Scenario(batch=16, ctx_len=8192),
    Scenario(batch=16, ctx_len=16384),
    Scenario(batch=16, ctx_len=32768),
    Scenario(batch=16, ctx_len=65536),
]

# Limit the number of test scenarios to avoid taking too long.
test_scenarios = [
    all_scenarios[0],
    all_scenarios[1],
    all_scenarios[4],
    all_scenarios[7],
    all_scenarios[14],
    all_scenarios[17],
    all_scenarios[23],
    all_scenarios[24],
]


# Default values from deepseek_v3, but will be overwritten by scenario.
@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = (
        {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
    )
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def _generate_random_weights(mla: MLA):
    # Helpers to init a tensor
    def init_low_precision(t, op):
        if t.dtype.itemsize <= 1:
            t2 = torch.empty_like(t, dtype=torch.float32)
            op(t2)
            t.copy_(t2)
        else:
            op(t)

    def init_uniform(tensor, a=-1.0, b=1.0, use_kaiming=False):
        if tensor is not None:
            if use_kaiming:
                tv = tensor.view(-1, tensor.shape[-2], tensor.shape[-1])
                for t in tv:
                    init_low_precision(t, torch.nn.init.kaiming_uniform_)
            else:
                init_low_precision(tensor, partial(torch.nn.init.uniform_, a=a, b=b))

    def init_block_scale(tensor, orig_tensor):
        if tensor is None or orig_tensor is None:
            return
        b1, b2 = 128, 128
        orig_tensor = orig_tensor.contiguous().to(tensor.dtype)
        exp1 = (orig_tensor.shape[-2] + b1 - 1) // b1
        exp2 = (orig_tensor.shape[-1] + b2 - 1) // b2
        if tensor.shape[-2] != exp1 or tensor.shape[-1] != exp2:
            # for some fused weights, this can happen
            # we simply adapt the size of the blocks and use that for the scale
            b1 = (orig_tensor.shape[-2] + tensor.shape[-2] - 1) // tensor.shape[-2]
            b2 = (orig_tensor.shape[-1] + tensor.shape[-1] - 1) // tensor.shape[-1]
        e1 = orig_tensor.shape[-2] // b1
        e2 = orig_tensor.shape[-1] // b2
        x = orig_tensor[..., : e1 * b1, : e2 * b2].view(*orig_tensor.shape[:-2], e1, b1, e2, b2)
        scale = x.abs().amax(dim=(-3, -1)) / 448.0
        if e1 * b1 != orig_tensor.shape[-2]:
            x2 = orig_tensor[..., e1 * b1 :, : e2 * b2].view(*orig_tensor.shape[:-2], 1, -1, e2, b2)
            scale2 = x2.abs().amax(dim=(-3, -1)) / 448.0
            scale = torch.cat([scale, scale2], dim=-2)
        if e2 * b2 != orig_tensor.shape[-1]:
            x3 = orig_tensor[..., : e1 * b1, e2 * b2 :].view(*orig_tensor.shape[:-2], e1, b1, 1, -1)
            scale3 = x3.abs().amax(dim=(-3, -1)) / 448.0
            if scale.shape[-2] == e1 + 1:
                x4 = orig_tensor[..., e1 * b1 :, e2 * b2 :].view(
                    *orig_tensor.shape[:-2], 1, -1, 1, -1
                )
                scale4 = x4.abs().amax(dim=(-3, -1)) / 448.0
                scale3 = torch.cat([scale3, scale4], dim=-2)
            scale = torch.cat([scale, scale3], dim=-1)
        tensor.copy_(scale)

    def init_linear(mod):
        if mod is None:
            return
        init_uniform(mod.weight, use_kaiming=True)
        if hasattr(mod, "weight_scale"):
            init_block_scale(mod.weight_scale, mod.weight)
        if hasattr(mod, "bias"):
            init_uniform(mod.bias)

    # Linear modules
    for name in ["kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj"]:
        init_linear(getattr(mla, name))

    # RMSNorm modules
    for name in ["kv_a_layernorm", "q_a_layernorm"]:
        if name == "q_a_layernorm":
            mod = getattr(mla, name, None)
        else:
            mod = getattr(mla, name)
        if mod is not None and hasattr(mod, "weight"):
            init_uniform(mod.weight, a=0.9, b=1.1)

    # k_b_proj_trans (created in create_weights)
    init_uniform(mla.k_b_proj_trans, use_kaiming=True)
    # k_b_proj_trans_scale (optional)
    if hasattr(mla, "k_b_proj_trans_scale"):
        init_block_scale(mla.k_b_proj_trans_scale, mla.k_b_proj_trans)
    init_uniform(mla.v_b_proj)
    # v_b_proj_scale (optional)
    if hasattr(mla, "v_b_proj_scale"):
        init_block_scale(mla.v_b_proj_scale, mla.v_b_proj)


def _make_latent_cache_gen(
    mla: MLA,
    rank: int,
    world_size: int,
    ctx_len_per_gpu: int,
    input_ctx_bs: torch.Tensor,
    ref_attn_metadata: Optional[AttentionMetadata],
):
    if rank == 0:
        assert ref_attn_metadata is not None
        kv_cache_block_offsets = ref_attn_metadata.kv_cache_manager.host_kv_cache_block_offsets
        kv_buffer = ref_attn_metadata.kv_cache_manager.get_buffers(0)
        ret = input_ctx_bs.new_empty(
            (world_size - 1, input_ctx_bs.shape[0], mla.kv_lora_rank + mla.qk_rope_head_dim)
        )
        # the RoPE values in the KV cache are embedded and we need to get the
        # original values instead for the latent cache
        # so we first get the cos/sin cache used in MLA
        _, cos_sin_cache = mla.pos_embd_params.rope.create_rope_const_params()
        cos_sin_cache = cos_sin_cache.reshape(-1, mla.qk_rope_head_dim, 2)
        assert cos_sin_cache.dtype == torch.float32

        def rotate_half_inv(x):
            """Rotates half the hidden dims of the input (inverse)."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((x2, -x1), dim=-1)

        for r in range(world_size - 1):
            for b in range(input_ctx_bs.shape[0]):
                block, t = divmod(
                    (r + 1) * ctx_len_per_gpu, ref_attn_metadata.kv_cache_manager.tokens_per_block
                )
                kv_block = kv_cache_block_offsets[0, b, 0, block].item()
                ret[r, b] = kv_buffer[kv_block, 0, t, 0, :]
        rope_values = ret[:, :, mla.kv_lora_rank :].clone().to(dtype=torch.float32)
        # now we apply the inverse of RoPE embedding to get the original values
        # rope_values has shape (world_size - 1, batch_size, rope_dim)
        # cos_sin_cache has shape (max_pos, rope_dim, 2)

        # Setup position and cos/sin values
        positions = torch.arange(1, world_size, device=rope_values.device) * ctx_len_per_gpu
        cos_sin_cache_pos = torch.index_select(cos_sin_cache, 0, positions)
        cos = cos_sin_cache_pos[..., 0].unsqueeze(1)
        sin = cos_sin_cache_pos[..., 1].unsqueeze(1)
        # cos/sin shape is (world_size - 1, 1, rope_dim) to broadcast with batch

        # Reshape for pairwise rotation
        rope_values_reshaped = (
            rope_values.unflatten(-1, [-1, 2]).transpose(-1, -2).flatten(start_dim=-2)
        )
        orig_rope_values = rope_values_reshaped * cos + rotate_half_inv(rope_values_reshaped) * sin
        orig_rope_values_reshaped = (
            orig_rope_values.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
        )

        ret[:, :, mla.kv_lora_rank :] = orig_rope_values_reshaped.to(dtype=ret.dtype)
    else:
        ret = input_ctx_bs.new_empty(
            (world_size - 1, input_ctx_bs.shape[0], mla.kv_lora_rank + mla.qk_rope_head_dim)
        )

    mapping = Mapping(
        world_size=world_size, rank=rank, cp_size=world_size, cp_config={"cp_type": CpType.HELIX}
    )
    # use cp_allgather here to broadcast from rank 0 to all other ranks
    ret_all = cp_allgather(ret, mapping=mapping, dim=0)
    ret = ret_all.view(world_size, *ret.shape)[0]
    if rank == world_size - 1:
        return None
    return ret[rank]


def _run_mla_distributed(
    rank: int,
    world_size: int,
    scenario: Scenario,
    mapping: Mapping,
    test_params: tuple,
    ref_output: torch.Tensor,
):
    input_ctx, input_gen, position_ids_ctx, weights, pos_embd_params, ref_attn_metadata = (
        test_params
    )
    position_ids_gen = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )
    extra_attrs = dict()
    config = ModelConfig(mapping=mapping)
    config.extra_attrs = extra_attrs
    mla = MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        config=config,
    ).cuda()
    copy_weights_for_cp(weights, "o_proj.weight", 1, rank, world_size)
    copy_weights_for_cp(weights, "v_b_proj", 0, rank, world_size)
    mla.load_state_dict(weights)

    # Set up KVCacheManager and attn_metadata for distributed.
    kv_cache_manager, attn_metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
    )
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    ctx_len_per_gpu = scenario.ctx_len // world_size

    input_ctx_rank, position_ids_ctx_rank = split_inputs_for_rank(
        input_ctx, position_ids_ctx, scenario, rank, world_size
    )

    # Context step — populate KV cache only; output is discarded.
    # We call forward_impl directly instead of mla.forward() because
    # o_proj is sized for helix-gen (num_heads_tp_cp * v_head_dim) and would
    # crash on the full-width context attention output.
    activate_all_ranks_for_context(attn_metadata, position_ids_ctx_rank)
    ctx_output = input_ctx_rank.new_empty(
        [input_ctx_rank.shape[0], mla.num_heads_tp * mla.v_head_dim], dtype=input_ctx_rank.dtype
    )
    mla.forward_impl(position_ids_ctx_rank, input_ctx_rank, attn_metadata, output=ctx_output)

    # For non-last rank, generate the right latent cache for generation.
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.ctx_len, scenario.hidden_size)
    latent_cache_gen = _make_latent_cache_gen(
        mla, rank, world_size, ctx_len_per_gpu, input_ctx_bs, ref_attn_metadata
    )

    # Single generation step.
    for req_id in range(scenario.batch):
        kv_cache_manager.impl.add_token(req_id)
    helix_is_inactive_rank = [rank != world_size - 1] * scenario.batch
    attn_metadata = create_helix_gen_metadata(
        scenario.batch,
        ctx_len_per_gpu,
        kv_cache_manager,
        helix_is_inactive_rank,
        position_ids_gen,
        enable_context_mla_with_cached_kv=True,
    )
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with model_extra_attrs(extra_attrs):
        output = mla(position_ids_gen, input_gen, attn_metadata, latent_cache_gen=latent_cache_gen)
    print(f"Rank {rank} {world_size}-GPU: result: {output[0, :8]} / {output[-1, -8:]}")

    kv_cache_manager.shutdown()
    if ref_attn_metadata is not None:
        ref_attn_metadata.kv_cache_manager.shutdown()

    return compute_mismatch_ratio(
        output, ref_output, scenario.atol, scenario.rtol, rank, world_size
    )


@torch.inference_mode
def _full_test_multi_gpu(
    rank: int,
    world_size: int,
    scenario: Scenario,
    use_nccl_for_alltoall: bool = False,
    fifo_version: int = 2,
):
    if scenario.rope_scaling:
        rope_scaling = {
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        }
    else:
        rope_scaling = None
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=rope_scaling,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    torch.manual_seed(42)
    input_ctx = torch.empty(
        scenario.batch * scenario.ctx_len, scenario.hidden_size, dtype=scenario.dtype, device="cuda"
    ).uniform_(-1, 1)
    input_gen = torch.empty(
        scenario.batch * scenario.predicted_tokens_per_seq,
        scenario.hidden_size,
        dtype=scenario.dtype,
        device="cuda",
    ).uniform_(-1, 1)
    position_ids_ctx = torch.arange(scenario.ctx_len, dtype=torch.int, device="cuda").repeat(
        scenario.batch
    )
    position_ids_gen = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )

    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    mla = MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
    ).cuda()
    _generate_random_weights(mla)
    weights = mla.state_dict()

    # Up to this point, all ranks should have same tensors because the seed is
    # the same.  Now we run the reference MLA on rank 0.
    if rank == 0:
        ref_mapping = Mapping(world_size=1, tp_size=1, rank=0)
        ref_kv_cache_manager, ref_attn_metadata = setup_kv_and_metadata(
            scenario,
            ref_mapping,
            cache_type=CACHE_TYPE_SELFKONLY,
            num_kv_heads=1,
            head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
        )
        # Context step.
        mla(position_ids_ctx, input_ctx, ref_attn_metadata)

        # Single generation step.
        for req_id in range(scenario.batch):
            ref_kv_cache_manager.impl.add_token(req_id)
        ref_attn_metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
            request_ids=list(range(scenario.batch)),
            max_num_requests=scenario.batch,
            num_contexts=0,
            prompt_lens=[scenario.ctx_len] * scenario.batch,
            max_num_tokens=scenario.ctx_len,
            kv_cache_manager=ref_kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[scenario.ctx_len] * scenario.batch,
            ),
            enable_context_mla_with_cached_kv=True,
        )
        ref_attn_metadata.prepare()
        ref_output = mla(position_ids_gen, input_gen, ref_attn_metadata)
        print(f"Ref result: {ref_output[0, :8]} / {ref_output[-1, -8:]}")
    else:
        ref_output = torch.empty(
            scenario.batch,
            scenario.hidden_size,
            dtype=scenario.dtype,
            device="cuda",
        )
        ref_attn_metadata = None

    # Distributed mapping for helix.
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={
            "cp_type": CpType.HELIX,
            "use_nccl_for_alltoall": use_nccl_for_alltoall,
            "fifo_version": fifo_version,
        },
    )
    # Broadcast reference output from rank 0 to all ranks.
    ref_output_all = cp_allgather(ref_output, mapping=mapping, dim=0)
    ref_output = ref_output_all.view(world_size, *ref_output.shape)[0]

    test_params = (
        input_ctx,
        input_gen,
        position_ids_ctx,
        weights,
        pos_embd_params,
        ref_attn_metadata,
    )
    return _run_mla_distributed(rank, world_size, scenario, mapping, test_params, ref_output)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("scenario", test_scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v1", "fifo_v2"])
def test_mla_helix_distributed(
    scenario: Scenario,
    comms_medium: str,
):
    run_helix_test(_full_test_multi_gpu, scenario, comms_medium)
