# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import pickle
import sys
import weakref
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Optional

import _torch.attention.multi_gpu.helix_test_utils as helix_utils
import cloudpickle
import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import (
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
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionMetadata,
    KVCacheParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.model_config import ModelConfig
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
    # Give the module every head's V weight, so generation applies the V
    # up-projection before the CP all-to-all.
    helix_v_up: bool = False

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
    full_v_b_proj = weights["v_b_proj"]
    copy_weights_for_cp(weights, "o_proj.weight", 1, rank, world_size)
    copy_weights_for_cp(weights, "v_b_proj", 0, rank, world_size)
    mla.load_state_dict(weights)
    if scenario.helix_v_up:
        mla._bind_helix_v_up_weight(full_v_b_proj.to(device="cuda", dtype=scenario.dtype))

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

    # This decode fixture populates each rank's cache from a different token
    # chunk. Complete, replicated-input prefill outputs are checked separately.
    activate_all_ranks_for_context(attn_metadata, position_ids_ctx_rank)
    ctx_output = input_ctx_rank.new_empty(
        [input_ctx_rank.shape[0], mla.mha.num_heads * mla.v_head_dim], dtype=input_ctx_rank.dtype
    )
    mla.forward_impl(position_ids_ctx_rank, input_ctx_rank, attn_metadata, attn_output=[ctx_output])

    # For non-last rank, generate the right latent cache for generation.
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.ctx_len, scenario.hidden_size)
    latent_cache_gen = _make_latent_cache_gen(
        mla, rank, world_size, ctx_len_per_gpu, input_ctx_bs, ref_attn_metadata
    )

    # Single generation step.
    for req_id in range(scenario.batch):
        assert kv_cache_manager.kv_cache_map[req_id].capacity >= (ctx_len_per_gpu + 1) * world_size
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
            assert ref_kv_cache_manager.kv_cache_map[req_id].capacity >= scenario.ctx_len + 1
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
@skip_pre_blackwell
@pytest.mark.parametrize("scenario", test_scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v1", "fifo_v2"])
def test_mla_helix_distributed(
    scenario: Scenario,
    comms_medium: str,
):
    run_helix_test(_full_test_multi_gpu, scenario, comms_medium)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
def test_mla_helix_fifo_v2_unaligned_softmax_stats():
    # Six heads over CP2 produce three local heads and a 24-byte softmax-stats
    # row, exercising the float2 fallback used by Kimi K3 over CP32.
    scenario = Scenario(num_heads=6, num_kv_heads=6, batch=1, ctx_len=64)
    run_helix_test(_full_test_multi_gpu, scenario, "fifo_v2")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize(
    "scenario",
    [dataclasses.replace(s, helix_v_up=True) for s in test_scenarios],
    ids=lambda x: f"scenario: {x}",
)
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v1", "fifo_v2"])
def test_mla_helix_v_up_before_alltoall(scenario: Scenario, comms_medium: str):
    run_helix_test(_full_test_multi_gpu, scenario, comms_medium)


def _v_up_projector(num_heads: int, kv_lora_rank: int, v_head_dim: int, v_full: torch.Tensor):
    """The fields of an MLA module that ``MLA._helix_project_v_partials`` reads."""
    module = SimpleNamespace(
        num_heads_tp=num_heads,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        helix_v_b_proj_full=v_full,
        use_cute_dsl_bf16_bmm=False,
    )
    module._bmm_bf16_out = partial(MLA._bmm_bf16_out, module)
    return module


def _v_up_builder(kv_b_weight, kv_b_scale, *, num_heads, cp_size, cp_rank, nope, v, k):
    """The fields of an MLA module that ``MLA._build_helix_v_up_weight`` reads."""
    module = SimpleNamespace(
        mapping=SimpleNamespace(has_cp_helix=lambda: True, cp_rank=cp_rank),
        dtype=torch.bfloat16,
        kv_b_proj=SimpleNamespace(
            weight=kv_b_weight,
            weight_scale=kv_b_scale,
            partition_plan=SimpleNamespace(enabled=False, op_kind=None),
        ),
        num_heads_tp=num_heads,
        num_heads_tp_cp=num_heads // cp_size,
        qk_nope_head_dim=nope,
        v_head_dim=v,
        kv_lora_rank=k,
        v_b_proj_dequant=None,
        helix_v_b_proj_full=None,
        _helix_v_up_before_exchange=False,
    )
    module._bind_helix_v_up_weight = partial(MLA._bind_helix_v_up_weight, module)
    return module


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_helix_v_up_weight_from_fp8_kv_b(cp_rank: int):
    """MLA dequantizes every tp-local head's V weight from the loaded FP8 kv_b_proj; this rank's
    dequantized V chunk becomes a view of it."""
    h, n, v, k, cp = 4, 128, 128, 256, 2
    generator = torch.Generator().manual_seed(cp_rank)
    weight = (torch.randn(h * (n + v), k, generator=generator) * 64).to(torch.float8_e4m3fn)
    scale = torch.rand(h * (n + v) // 128, k // 128, generator=generator) + 0.5
    module = _v_up_builder(
        weight, scale, num_heads=h, cp_size=cp, cp_rank=cp_rank, nope=n, v=v, k=k
    )
    module.v_b_proj_dequant = torch.empty(h // cp, v, k, dtype=torch.bfloat16)
    MLA._build_helix_v_up_weight(module)

    value_codes = weight[h * n :].view(h, v, k)
    value_scale = scale[h * n // 128 :].view(h, v // 128, k // 128)
    expanded = value_scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    full = (value_codes.float() * expanded).to(torch.bfloat16)
    assert module._helix_v_up_before_exchange
    assert torch.equal(module.helix_v_b_proj_full, full)
    chunk = slice(cp_rank * h // cp, (cp_rank + 1) * h // cp)
    assert torch.equal(module.v_b_proj_dequant, full[chunk])
    assert (
        module.v_b_proj_dequant.untyped_storage().data_ptr()
        == module.helix_v_b_proj_full.untyped_storage().data_ptr()
    )


def test_helix_v_up_weight_views_bf16_kv_b():
    h, n, v, k = 4, 64, 128, 256
    weight = torch.randn(h * (n + v), k, dtype=torch.bfloat16)
    module = _v_up_builder(weight, None, num_heads=h, cp_size=2, cp_rank=0, nope=n, v=v, k=k)
    MLA._build_helix_v_up_weight(module)
    assert module._helix_v_up_before_exchange
    assert torch.equal(module.helix_v_b_proj_full, weight[h * n :].view(h, v, k))
    assert module.helix_v_b_proj_full.data_ptr() == weight[h * n :].data_ptr()


def test_helix_v_up_weight_needs_loaded_fp8_scales():
    """A kv_b_proj whose scales are already in a GEMM layout keeps V after the exchange."""
    h, n, v, k = 4, 128, 128, 256
    weight = torch.zeros(h * (n + v), k, dtype=torch.float8_e4m3fn)
    packed_scale = torch.zeros(k // 128 // 4 + 1, h * (n + v), dtype=torch.int32)
    module = _v_up_builder(
        weight, packed_scale, num_heads=h, cp_size=2, cp_rank=0, nope=n, v=v, k=k
    )
    MLA._build_helix_v_up_weight(module)
    assert not module._helix_v_up_before_exchange
    assert module.helix_v_b_proj_full is None


@skip_pre_blackwell
@pytest.mark.parametrize("num_tokens", [1, 37, 224])
def test_helix_v_up_commutes_with_combine(num_tokens: int):
    """Projecting V before the CP exchange equals projecting the combined latent.

    Emulates the fifo_v2 exchange of ``cp_size`` ranks on one GPU and runs the
    combine kernel on both the latent partials (followed by the rank's V
    shard) and the V-projected partials. The two differ only by the BF16
    rounding of the reassociated sums.
    """
    torch.manual_seed(0)
    cp_size, num_heads, kv_lora_rank, v_head_dim = 4, 16, 512, 128
    heads_per_rank = num_heads // cp_size
    v_full = torch.randn(num_heads, v_head_dim, kv_lora_rank, device="cuda", dtype=torch.bfloat16)
    v_full = v_full / kv_lora_rank**0.5
    projector = _v_up_projector(num_heads, kv_lora_rank, v_head_dim, v_full)

    # Partial outputs and (max, sum) softmax stats of every CP rank for all heads.
    partials = [
        torch.randn(num_tokens, num_heads * kv_lora_rank, device="cuda", dtype=torch.bfloat16)
        for _ in range(cp_size)
    ]
    stats = []
    for _ in range(cp_size):
        s = torch.empty(num_tokens, num_heads, 2, device="cuda", dtype=torch.float32)
        s[..., 0].uniform_(-2.0, 2.0)
        s[..., 1].uniform_(0.5, 4.0)
        stats.append(s)
    projected = [MLA._helix_project_v_partials(projector, p) for p in partials]

    def gather(per_rank, rank, value_dim):
        # What rank ``rank`` holds after the exchange: its head chunk from every rank.
        chunks = [
            t.view(num_tokens, cp_size, heads_per_rank * value_dim)[:, rank] for t in per_rank
        ]
        return torch.stack(chunks, dim=1).view(num_tokens, cp_size, heads_per_rank, value_dim)

    for rank in range(cp_size):
        gathered_stats = torch.stack(
            [s.view(num_tokens, cp_size, heads_per_rank * 2)[:, rank] for s in stats], dim=1
        ).view(num_tokens, cp_size, heads_per_rank, 2)
        latent = torch.ops.trtllm.helix_post_process_native(
            gather(partials, rank, kv_lora_rank).contiguous(), gathered_stats.contiguous(), 1.0, 1
        ).view(num_tokens, heads_per_rank, kv_lora_rank)
        v_shard = v_full[rank * heads_per_rank : (rank + 1) * heads_per_rank]
        reference = torch.empty(
            num_tokens, heads_per_rank, v_head_dim, device="cuda", dtype=torch.bfloat16
        )
        torch.ops.trtllm.bmm_out(
            latent.transpose(0, 1), v_shard.transpose(1, 2), reference.transpose(0, 1)
        )
        result = torch.ops.trtllm.helix_post_process_native(
            gather(projected, rank, v_head_dim).contiguous(), gathered_stats.contiguous(), 1.0, 1
        ).view(num_tokens, heads_per_rank, v_head_dim)
        torch.testing.assert_close(result.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _alltoall_after_v_up_gemm(rank: int, world_size: int, num_replays: int) -> int:
    """Count wrong (token, source rank) rows of the fifo_v2 exchange of V-projected partials.

    The V up-projection GEMM and the all-to-all are captured in one CUDA graph,
    so the all-to-all launches while the GEMM still runs on the SMs
    (programmatic dependent launch). Every replay checks the received partials
    and softmax stats bit for bit.
    """
    from tensorrt_llm._torch.distributed.ops import HelixAllToAllNative

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False, "fifo_version": 2},
    )
    helix = HelixAllToAllNative.get(mapping)
    comm = MPI.COMM_WORLD
    heads_per_rank, kv_lora_rank, v_head_dim = 24, 512, 128
    num_heads = heads_per_rank * world_size
    torch.manual_seed(rank)
    v_full = (
        torch.randn(num_heads, v_head_dim, kv_lora_rank, device="cuda") / kv_lora_rank**0.5
    ).to(torch.bfloat16)
    projector = _v_up_projector(num_heads, kv_lora_rank, v_head_dim, v_full)

    num_bad = 0
    for num_tokens in (224, 448):
        latent = torch.empty(
            num_tokens, num_heads * kv_lora_rank, device="cuda", dtype=torch.bfloat16
        )
        stats = torch.empty(num_tokens, num_heads, 2, device="cuda")
        zero_kv_mask = torch.empty(num_tokens, device="cuda", dtype=torch.bool)

        def fill_inputs():
            latent.copy_(torch.randn_like(latent, dtype=torch.float32))
            stats.copy_(torch.randn_like(stats))
            zero_kv_mask.copy_(torch.rand(num_tokens, device="cuda") < 0.25)

        def exchange():
            partial_v = MLA._helix_project_v_partials(projector, latent)
            partial_v_out, stats_out = helix.alltoall_native(
                partial_v.view(num_tokens, world_size, heads_per_rank * v_head_dim),
                stats.view(num_tokens, world_size, heads_per_rank * 2),
                zero_kv_mask,
            )
            return partial_v, partial_v_out, stats_out

        fill_inputs()
        exchange()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            partial_v, partial_v_out, stats_out = exchange()
        comm.Barrier()
        for _ in range(num_replays):
            fill_inputs()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            # What this rank must hold: its head chunk from every rank, with the
            # rows a source rank owns no KV for replaced by the no-op contribution.
            sent = comm.allgather((partial_v.cpu(), stats.cpu(), zero_kv_mask.cpu()))
            expected_v = torch.stack(
                [p.view(num_tokens, world_size, -1)[:, rank] for p, _, _ in sent], dim=1
            )
            expected_stats = torch.stack(
                [s.view(num_tokens, world_size, heads_per_rank, 2)[:, rank] for _, s, _ in sent],
                dim=1,
            )
            for src, (_, _, mask) in enumerate(sent):
                expected_v[mask, src] = 0
                expected_stats[mask, src, :, 0] = float("-inf")
                expected_stats[mask, src, :, 1] = 0
            bad_rows = (partial_v_out.cpu() != expected_v).any(-1) | (
                stats_out.cpu().view_as(expected_stats) != expected_stats
            ).flatten(2).any(-1)
            num_bad += int(bad_rows.sum())
        del graph
    return num_bad


@skip_pre_blackwell
@pytest.mark.parametrize("world_size", [2, 4])
def test_helix_alltoall_after_v_up_gemm_in_cuda_graph(world_size: int):
    """The exchange stays exact when it overlaps the V up-projection GEMM.

    The receiver restores each 128-byte line's flagged u64 with ordinary
    shared-memory stores and then writes the line out with a bulk copy, which
    only sees those stores after a proxy fence. The overlap with the GEMM makes
    a missing fence show up as flag values in the received partials or stats.
    """
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs to run this test")
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            helix_utils.run_single_rank,
            *zip(*[(_alltoall_after_v_up_gemm, world_size, 20)] * world_size),
        )
        num_bad = list(results)
    assert num_bad == [0] * world_size, f"wrong (token, source) rows per rank: {num_bad}"
