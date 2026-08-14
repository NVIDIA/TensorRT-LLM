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

import pickle
import sys
import weakref
from dataclasses import dataclass

import _torch.modules.helix_test_utils as helix_utils
import cloudpickle
import pytest
import torch
from _torch.modules.helix_test_utils import (
    CACHE_TYPE_SELF,
    activate_all_ranks_for_context,
    compute_mismatch_ratio,
    copy_weights_for_cp,
    create_helix_gen_metadata,
    run_helix_test,
    setup_kv_and_metadata,
    split_inputs_for_rank,
)
from mpi4py import MPI
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.interface import (
    KVCacheParams,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import Attention
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


# Values inspired by a small LLaMA-like model.
@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    hidden_size: int = 2048  # num_heads * head_dim
    rope_theta: float = 10000.0
    kv_cache_tokens_per_block: int = 32
    bias: bool = False
    batch: int = 8
    ctx_len: int = 1024
    # note: need to use fairly high tolerances because the softmax stats can
    # lose a lot of precision and we're using bf16 here.
    atol: float = 1e-1
    rtol: float = 5e-2

    @property
    def max_position_embeddings(self) -> int:
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
    Scenario(batch=8, ctx_len=1024),
    Scenario(batch=8, ctx_len=2048),
    Scenario(batch=8, ctx_len=4096),
    Scenario(batch=8, ctx_len=8192),
    Scenario(batch=8, ctx_len=16384),
    Scenario(batch=8, ctx_len=32768),
    Scenario(batch=8, ctx_len=65536),
    Scenario(batch=16, ctx_len=1024),
    Scenario(batch=16, ctx_len=2048),
    Scenario(batch=16, ctx_len=4096),
    Scenario(batch=16, ctx_len=8192),
    Scenario(batch=16, ctx_len=16384),
    Scenario(batch=16, ctx_len=32768),
]

# Limit the number of test scenarios to avoid taking too long.
test_scenarios = [
    all_scenarios[0],
    all_scenarios[1],
    all_scenarios[4],
    all_scenarios[7],
    all_scenarios[10],
    all_scenarios[13],
    all_scenarios[17],
    all_scenarios[18],
]


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 2048
    num_attention_heads: int = 16
    rope_scaling: dict = None
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    model_type: str = "llama"


def _generate_random_weights(attn: Attention):
    """Initialize Attention weights with random values."""
    for name, param in attn.named_parameters():
        if param.dtype.itemsize <= 1:
            t2 = torch.empty_like(param, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(t2)
            param.data.copy_(t2)
        else:
            torch.nn.init.kaiming_uniform_(param.data)


def _run_attention_distributed(
    rank: int,
    world_size: int,
    scenario: Scenario,
    mapping: Mapping,
    test_params: tuple,
    ref_output: torch.Tensor,
):
    input_ctx, input_gen, position_ids_ctx, weights, pos_embd_params = test_params
    position_ids_gen = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )

    extra_attrs = dict()
    config = ModelConfig(mapping=mapping)
    config.extra_attrs = extra_attrs
    attn = Attention(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        config=config,
    ).cuda()

    # Split o_proj weight along input dimension for CP.
    copy_weights_for_cp(weights, "o_proj.weight", 1, rank, world_size)
    attn.load_state_dict(weights)

    # Set up KVCacheManager and attn_metadata for distributed.
    kv_cache_manager, attn_metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELF,
        num_kv_heads=scenario.num_kv_heads,
        head_dim=scenario.head_dim,
    )
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    ctx_len_per_gpu = scenario.ctx_len // world_size

    input_ctx_rank, position_ids_ctx_rank = split_inputs_for_rank(
        input_ctx, position_ids_ctx, scenario, rank, world_size
    )

    # Context step — populate KV cache only; output is discarded.
    # We call internal methods directly instead of attn.forward() because
    # o_proj is sized for helix-gen (num_heads_tp_cp * head_dim) and would
    # crash on the full-width context attention output.
    activate_all_ranks_for_context(attn_metadata, position_ids_ctx_rank)
    qkv = attn.qkv_proj(input_ctx_rank)
    q, k, v = qkv, None, None
    q, k, v = attn.apply_rope(q, k, v, position_ids_ctx_rank)
    q, k, v = attn.convert_qkv(q, k, v)
    attn.forward_impl(q, k, v, attn_metadata, PredefinedAttentionMask.CAUSAL, None, None, None)

    # Single generation step.
    for req_id in range(scenario.batch):
        kv_cache_manager.impl.add_token(req_id)
    helix_is_inactive_rank = [rank != world_size - 1] * scenario.batch
    attn_metadata = create_helix_gen_metadata(
        scenario.batch, ctx_len_per_gpu, kv_cache_manager, helix_is_inactive_rank, position_ids_gen
    )
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    with model_extra_attrs(extra_attrs):
        output = attn(position_ids_gen, input_gen, attn_metadata)
    print(f"Rank {rank} {world_size}-GPU: result: {output[0, :8]} / {output[-1, -8:]}")

    kv_cache_manager.shutdown()
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
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=None,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
    )
    torch.manual_seed(42)
    input_ctx = torch.empty(
        scenario.batch * scenario.ctx_len,
        scenario.hidden_size,
        dtype=scenario.dtype,
        device="cuda",
    ).uniform_(-1, 1)
    input_gen = torch.empty(
        scenario.batch,
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
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams.from_config(rope_config),
        is_neox=True,
    )

    attn = Attention(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
    ).cuda()

    _generate_random_weights(attn)
    weights = attn.state_dict()

    # Up to this point, all ranks should have the same tensors because the seed
    # is the same. Now we run the reference Attention on rank 0.
    if rank == 0:
        ref_mapping = Mapping(world_size=1, tp_size=1, rank=0)
        ref_kv_cache_manager, ref_attn_metadata = setup_kv_and_metadata(
            scenario,
            ref_mapping,
            cache_type=CACHE_TYPE_SELF,
            num_kv_heads=scenario.num_kv_heads,
            head_dim=scenario.head_dim,
        )

        # Context step.
        attn(position_ids_ctx, input_ctx, ref_attn_metadata)

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
        )
        ref_attn_metadata.prepare()
        ref_output = attn(position_ids_gen, input_gen, ref_attn_metadata)
        print(f"Ref result: {ref_output[0, :8]} / {ref_output[-1, -8:]}")
        ref_kv_cache_manager.shutdown()
    else:
        ref_output = torch.empty(
            scenario.batch,
            scenario.hidden_size,
            dtype=scenario.dtype,
            device="cuda",
        )

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
    )
    return _run_attention_distributed(rank, world_size, scenario, mapping, test_params, ref_output)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize("scenario", test_scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v1", "fifo_v2"])
def test_mha_helix_distributed(
    scenario: Scenario,
    comms_medium: str,
):
    run_helix_test(_full_test_multi_gpu, scenario, comms_medium)
