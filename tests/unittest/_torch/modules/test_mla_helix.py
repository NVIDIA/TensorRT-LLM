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
import time
import traceback
import weakref
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
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
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.sampling_params import SamplingParams

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# values for deepseek_v3_lite
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
    kv_cache_tokens_per_block: int = 64
    # TODO only 1 is supported for now here
    predicted_tokens_per_seq: int = 1
    bias: bool = False
    batch: int = 8
    ctx_len: int = 1024
    ref_steps: int = 1
    # note: need to use fairly high tolerances because the softmax stats can lose
    # a lot of precision and we're using bf16 here
    atol: float = 1e-1
    rtol: float = 5e-2

    @property
    def max_position_embeddings(self) -> int:
        # ensure that max_position_embeddings is set large enough for every scenario
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

# limit the number of test scenarios to avoid taking too long
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


# default values from deepseek_v3, but will be overwritten by scenario
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


def _setup_kv_and_metadata(scenario: Scenario, mapping: Mapping, gen_steps: int):
    # Set up KVCacheManager and attn_metadata for MLA
    n_gpu = mapping.world_size
    assert scenario.ctx_len % n_gpu == 0
    ctx_len_per_gpu = scenario.ctx_len // n_gpu
    max_tokens = (
        (ctx_len_per_gpu + gen_steps + scenario.kv_cache_tokens_per_block - 1)
        // scenario.kv_cache_tokens_per_block
        * scenario.kv_cache_tokens_per_block
        * scenario.batch
    )
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        # Use SELFKONLY for MLA
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=scenario.num_layers,
        num_kv_heads=1,
        head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
        tokens_per_block=scenario.kv_cache_tokens_per_block,  # for test, just use seq_len
        max_seq_len=ctx_len_per_gpu + gen_steps,
        max_batch_size=scenario.batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kv_cache_dtype)),
    )
    for req_id in range(scenario.batch):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=1,
            input_tokens=[1] * ctx_len_per_gpu,  # all requests have the same length here
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.is_dummy_request = True
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, ctx_len_per_gpu, beam_width, req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        req.prompt_len = ctx_len_per_gpu
        req.py_prompt_len = req.prompt_len
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([ctx_len_per_gpu] * scenario.batch, dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=scenario.batch,
        prompt_lens=[ctx_len_per_gpu] * scenario.batch,
        max_num_tokens=ctx_len_per_gpu,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in range(scenario.batch)],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()
    return kv_cache_manager, attn_metadata


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


def _copy_to_cp(weights, param_name, dim, rank, world_size):
    w_dim_per_rank = weights[param_name].shape[dim] // world_size
    w_dim_start = rank * w_dim_per_rank
    w_dim_end = w_dim_start + w_dim_per_rank
    slices = [slice(None)] * weights[param_name].ndim
    slices[dim] = slice(w_dim_start, w_dim_end)
    weights[param_name] = weights[param_name][slices]


def _error_report(output, ref_output, atol, rtol, prefix):
    err = torch.abs(output - ref_output)
    ref_abs = torch.abs(ref_output)
    ref_abs[ref_abs == 0] = torch.finfo(ref_abs.dtype).smallest_normal
    rel_err = err / ref_abs
    # always print largest error and its index
    max_err_idx = torch.unravel_index(torch.argmax(err - atol - rtol * ref_abs), err.shape)
    values_err = (output[max_err_idx].item(), ref_output[max_err_idx].item())
    max_abs_err_idx = torch.unravel_index(torch.argmax(err), err.shape)
    values_abs = (output[max_abs_err_idx].item(), ref_output[max_abs_err_idx].item())
    max_rel_err_idx = torch.unravel_index(torch.argmax(rel_err), rel_err.shape)
    values_rel = (output[max_rel_err_idx].item(), ref_output[max_rel_err_idx].item())
    max_abs_err = err[max_abs_err_idx].item()
    max_rel_err = rel_err[max_rel_err_idx].item()
    max_err_idx = [x.item() for x in max_err_idx]
    max_abs_err_idx = [x.item() for x in max_abs_err_idx]
    max_rel_err_idx = [x.item() for x in max_rel_err_idx]
    isclose = err < atol + rtol * ref_abs
    n_error = (~isclose).sum().item()
    print(
        f"{prefix}: {n_error} errors, max error index: {max_err_idx} "
        f"(test/ref values: {values_err}), max abs error index: {max_abs_err_idx} "
        f"(test/ref values: {values_abs}, err: {max_abs_err}), max rel error index: {max_rel_err_idx} "
        f"(test/ref values: {values_rel}, err: {max_rel_err}), atol: {atol}, rtol: {rtol}"
    )
    return n_error


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
        kv_cache_block_offsets = ref_attn_metadata.host_kv_cache_block_offsets
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

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def rotate_half_inv(x):
            """Rotates half the hidden dims of the input."""
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
    gen_steps: int,
):
    input_ctx, input_gen, position_ids_ctx, weights, pos_embd_params, ref_attn_metadata = (
        test_params
    )
    # we start with the same position_ids as the reference MLA.
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
        enable_unit_test=True,
    ).cuda()
    # above should have the same config as the reference MLA except for the mapping
    # we update the weights accordingly and should be able to load them
    _copy_to_cp(weights, "o_proj.weight", 1, rank, world_size)
    _copy_to_cp(weights, "v_b_proj", 0, rank, world_size)
    mla.load_state_dict(weights)
    # Set up KVCacheManager and attn_metadata for distributed
    kv_cache_manager, attn_metadata = _setup_kv_and_metadata(scenario, mapping, gen_steps)
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    ctx_len_per_gpu = scenario.ctx_len // world_size
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.ctx_len, scenario.hidden_size)

    # split inputs into chunks for each rank
    input_ctx_bs_rank = input_ctx_bs[:, rank * ctx_len_per_gpu : (rank + 1) * ctx_len_per_gpu, :]
    input_ctx_rank = input_ctx_bs_rank.reshape(
        scenario.batch * ctx_len_per_gpu, scenario.hidden_size
    ).contiguous()
    position_ids_ctx_bs = position_ids_ctx.view(scenario.batch, scenario.ctx_len)
    position_ids_ctx_bs_rank = position_ids_ctx_bs[
        :, rank * ctx_len_per_gpu : (rank + 1) * ctx_len_per_gpu
    ]
    position_ids_ctx_rank = position_ids_ctx_bs_rank.reshape(
        scenario.batch * ctx_len_per_gpu
    ).contiguous()
    # this represents the context step
    with model_extra_attrs(extra_attrs):
        mla(position_ids_ctx_rank, input_ctx_rank, attn_metadata)

    # for non-last rank, generate the right latent cache for generation
    latent_cache_gen = _make_latent_cache_gen(
        mla, rank, world_size, ctx_len_per_gpu, input_ctx_bs, ref_attn_metadata
    )

    outputs = []
    start = time.time()

    # CUDA graph setup for timing
    use_cuda_graph = gen_steps > scenario.ref_steps
    graph = None
    graph_output = None
    start = time.time()

    for step in range(gen_steps):
        helix_is_inactive_rank = []
        for req_id in range(scenario.batch):
            kv_cache_manager.impl.add_token(req_id)
            # Assume last rank is active for all gen steps.
            if rank == world_size - 1:
                helix_is_inactive_rank.append(False)
                cache_add = step
            else:
                helix_is_inactive_rank.append(True)
                cache_add = 0
        cached_tokens_per_seq = [ctx_len_per_gpu + cache_add for _ in range(scenario.batch)]
        if step == 0:
            attn_metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
                request_ids=list(range(scenario.batch)),
                max_num_requests=scenario.batch,
                num_contexts=0,
                prompt_lens=[ctx_len_per_gpu] * scenario.batch,
                max_num_tokens=ctx_len_per_gpu,
                kv_cache_manager=kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=cached_tokens_per_seq,
                ),
                enable_context_mla_with_cached_kv=True,
                helix_is_inactive_rank=torch.tensor(
                    helix_is_inactive_rank, dtype=torch.bool, device="cuda"
                ),
            )
        else:
            attn_metadata.kv_cache_params = KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=cached_tokens_per_seq,
            )
            attn_metadata.helix_is_inactive_rank = torch.tensor(
                helix_is_inactive_rank, dtype=torch.bool, device="cuda"
            )
        attn_metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
        if not use_cuda_graph:
            # Original non-graph execution
            with model_extra_attrs(extra_attrs):
                result = mla(
                    position_ids_gen, input_gen, attn_metadata, latent_cache_gen=latent_cache_gen
                )
            if step < scenario.ref_steps:
                outputs.append(result)
            print(f"Rank {rank} {world_size}-GPU: result: {result[0, :8]} / {result[-1, -8:]}")
            # update position_ids_gen
            position_ids_gen += 1
            continue

        # CUDA graph capture on first step when timing
        if step == 0:
            print(f"Rank {rank} {world_size}-GPU: Creating CUDA graph and capturing")
            # Create CUDA graph metadata for capture
            attn_metadata = attn_metadata.create_cuda_graph_metadata(max_batch_size=scenario.batch)
            attn_metadata.prepare()
            extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)

            # Warm-up runs before graph capture
            for _ in range(2):
                with model_extra_attrs(extra_attrs):
                    result = mla(
                        position_ids_gen,
                        input_gen,
                        attn_metadata,
                        latent_cache_gen=latent_cache_gen,
                    )

            # Capture the graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with model_extra_attrs(extra_attrs):
                    graph_output = mla(
                        position_ids_gen,
                        input_gen,
                        attn_metadata,
                        latent_cache_gen=latent_cache_gen,
                    )
            result = graph_output
        elif step == scenario.ref_steps:
            start = time.time()

        graph.replay()
        result = graph_output

        # update position_ids_gen
        position_ids_gen += 1

        # Collect outputs for reference comparison
        if step < scenario.ref_steps:
            outputs.append(result)

    end = time.time()
    if gen_steps == scenario.ref_steps:
        avg_gen_time = float("inf")
    else:
        avg_gen_time = (end - start) / (gen_steps - scenario.ref_steps)
    throughput = scenario.batch / avg_gen_time
    print(
        f"Rank {rank} {world_size}-GPU: time taken for "
        f"{gen_steps - scenario.ref_steps} steps: "
        f"{end - start} s, throughput: {throughput} MLA/s"
    )
    output = torch.stack(outputs, dim=0)
    kv_cache_manager.shutdown()
    if ref_attn_metadata is not None:
        ref_attn_metadata.kv_cache_manager.shutdown()

    # every rank should have the same output and checks against the reference output
    atol, rtol = scenario.atol, scenario.rtol
    mismatch_count = 0
    for ref_step in range(scenario.ref_steps):
        for b in range(scenario.batch):
            mismatch_count += _error_report(
                output[ref_step, b],
                ref_output[ref_step, b],
                atol,
                rtol,
                f"Rank {rank} {world_size}-GPU step {ref_step}, batch {b}",
            )

    ratio_mismatch = mismatch_count / output.numel()
    print(
        f"Rank {rank} {world_size}-GPU: {mismatch_count}/{output.numel()} mismatches: {ratio_mismatch}"
    )
    return ratio_mismatch


@torch.inference_mode
def _full_test_multi_gpu(rank: int, world_size: int, scenario: Scenario, gen_steps: int):
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
        enable_unit_test=True,
    ).cuda()
    _generate_random_weights(mla)
    weights = mla.state_dict()
    # up to this point, all ranks should have same tensors because the seed is the same
    # now we run the reference MLA on rank 0
    if rank == 0:
        # Reference output (single GPU, but with correct KV/metadata setup)
        ref_mapping = Mapping(world_size=1, tp_size=1, rank=0)
        ref_kv_cache_manager, ref_attn_metadata = _setup_kv_and_metadata(
            scenario, ref_mapping, gen_steps
        )
        # this represents the context step
        mla(position_ids_ctx, input_ctx, ref_attn_metadata)
        ref_outputs = []
        start = time.time()

        # CUDA graph setup for timing
        use_cuda_graph = gen_steps > scenario.ref_steps
        graph = None
        graph_output = None

        for step in range(gen_steps):
            for req_id in range(scenario.batch):
                ref_kv_cache_manager.impl.add_token(req_id)
            if step == 0:
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
                        num_cached_tokens_per_seq=[
                            scenario.ctx_len + step for _ in range(scenario.batch)
                        ],
                    ),
                    enable_context_mla_with_cached_kv=True,
                )
            else:
                ref_attn_metadata.kv_cache_params = KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[
                        scenario.ctx_len + step for _ in range(scenario.batch)
                    ],
                )
            ref_attn_metadata.prepare()

            if not use_cuda_graph:
                result = mla(position_ids_gen, input_gen, ref_attn_metadata)
                if step < scenario.ref_steps:
                    ref_outputs.append(result)
                print(f"Ref result: {result[0, :8]} / {result[-1, -8:]}")
                # update position_ids_gen
                position_ids_gen += 1
                continue

            # CUDA graph capture on first step when timing
            if step == 0:
                print("Creating CUDA graph and capturing")
                # Create CUDA graph metadata for capture
                ref_attn_metadata = ref_attn_metadata.create_cuda_graph_metadata(
                    max_batch_size=scenario.batch
                )
                ref_attn_metadata.prepare()

                # Warm-up runs before graph capture
                for _ in range(2):
                    result = mla(position_ids_gen, input_gen, ref_attn_metadata)

                # Capture the graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = mla(position_ids_gen, input_gen, ref_attn_metadata)
                result = graph_output
            elif step == scenario.ref_steps:
                # Start timing with CUDA graph
                start = time.time()
            graph.replay()
            result = graph_output
            # update position_ids_gen
            position_ids_gen += 1
            if step < scenario.ref_steps:
                ref_outputs.append(result)
        end = time.time()
        if gen_steps == scenario.ref_steps:
            avg_gen_time = float("inf")
        else:
            avg_gen_time = (end - start) / (gen_steps - scenario.ref_steps)
        throughput = scenario.batch / avg_gen_time
        print(
            f"Time taken for {gen_steps - scenario.ref_steps} steps: "
            f"{end - start} s, throughput: {throughput} MLA/s"
        )
        ref_output = torch.stack(ref_outputs, dim=0)
    else:
        ref_output = torch.empty(
            scenario.ref_steps,
            scenario.batch,
            scenario.hidden_size,
            dtype=scenario.dtype,
            device="cuda",
        )
        ref_attn_metadata = None

    # Distributed mapping for helix
    mapping = Mapping(
        world_size=world_size, rank=rank, cp_size=world_size, cp_config={"cp_type": CpType.HELIX}
    )
    # we use cp_allgather here because there is no broadcast op across CP group
    ref_output_all = cp_allgather(ref_output, mapping=mapping, dim=0)
    # we only need the values from rank 0
    ref_output = ref_output_all.view(world_size, *ref_output.shape)[0]
    test_params = (
        input_ctx,
        input_gen,
        position_ids_ctx,
        weights,
        pos_embd_params,
        ref_attn_metadata,
    )
    return _run_mla_distributed(
        rank, world_size, scenario, mapping, test_params, ref_output, gen_steps
    )


def _run_single_rank(func, *args, **kwargs):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    print(f"rank {rank} starting")
    try:
        ret = func(rank, *args, **kwargs)
        print(f"rank {rank} done")
        return ret
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        raise Exception(f"\n\nError occurred. Original traceback is\n{tb}\n")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("scenario", test_scenarios, ids=lambda x: f"scenario: {x}")
def test_mla_helix_distributed(
    scenario: Scenario,
    gen_steps: Optional[int] = None,
    max_mismatch_ratio: float = 0.02,
    mismatch_ratios: Optional[List[float]] = None,
):
    world_size = 2
    gen_steps = scenario.ref_steps if gen_steps is None else gen_steps
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            _run_single_rank,
            *zip(*[(_full_test_multi_gpu, world_size, scenario, gen_steps)] * world_size),
        )
        if mismatch_ratios is None:
            for ratio_mismatch in results:
                assert ratio_mismatch <= max_mismatch_ratio
        else:
            mismatch_ratios.extend(results)


if __name__ == "__main__":
    for scenario in all_scenarios[:11]:
        timing_steps = 256
        gen_steps = scenario.ref_steps + timing_steps
        print(f"Running scenario: {scenario} and timing {timing_steps} steps")
        mismatch_ratios = []
        test_mla_helix_distributed(scenario, gen_steps=gen_steps, mismatch_ratios=mismatch_ratios)
        if any(mismatch > 0 for mismatch in mismatch_ratios):
            print(f"Numerical test failed with mismatch ratios: {mismatch_ratios}")
        else:
            print("Numerical test passed")
