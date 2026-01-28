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

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

from visual_gen.configs.op_manager import AttentionOpManager
from visual_gen.configs.parallel import DiTParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.attention import ditAttnProcessor
from visual_gen.utils import get_logger

logger = get_logger(__name__)


def sample_tensors(batch_size, num_heads, seq_len, head_dim, world_size):
    """Create sample tensors for attention testing."""

    shape = (batch_size, num_heads, seq_len, head_dim)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Prepare inputs
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = q.chunk(world_size, dim=2)[rank]
    local_k = k.chunk(world_size, dim=2)[rank]
    local_v = v.chunk(world_size, dim=2)[rank]
    return q, k, v, local_q, local_k, local_v


def sample_joint_tensors(batch_size, num_heads, seq_len, head_dim, world_size):
    """Create sample tensors for attention testing."""

    shape = (batch_size, num_heads, seq_len, head_dim)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Prepare inputs
    origin_q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    origin_k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)
    origin_v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=False)

    dist.broadcast(origin_q, src=0)
    dist.broadcast(origin_k, src=0)
    dist.broadcast(origin_v, src=0)

    joint_seq_length = max(seq_len // 8, 1)
    valid_joint_seq_length = torch.randint(1, joint_seq_length, (batch_size,), device=device)
    dist.broadcast(valid_joint_seq_length, src=0)

    joint_q = torch.narrow(origin_q, dim=2, start=seq_len - joint_seq_length, length=joint_seq_length)
    joint_k = torch.narrow(origin_k, dim=2, start=seq_len - joint_seq_length, length=joint_seq_length)
    joint_v = torch.narrow(origin_v, dim=2, start=seq_len - joint_seq_length, length=joint_seq_length)
    q = torch.narrow(origin_q, dim=2, start=0, length=seq_len - joint_seq_length)
    k = torch.narrow(origin_k, dim=2, start=0, length=seq_len - joint_seq_length)
    v = torch.narrow(origin_v, dim=2, start=0, length=seq_len - joint_seq_length)

    local_q = torch.cat((q.chunk(world_size, dim=2)[rank], joint_q), dim=2)
    local_k = torch.cat((k.chunk(world_size, dim=2)[rank], joint_k), dim=2)
    local_v = torch.cat((v.chunk(world_size, dim=2)[rank], joint_v), dim=2)
    return origin_q, origin_k, origin_v, local_q, local_k, local_v, joint_seq_length, valid_joint_seq_length.cpu()


def test_attn_parallel(
    batch_size, num_heads, seq_len, head_dim, world_size, ulysses_size, ring_size, attn_type, tensor_layout="HND"
):
    """Test basic parallel attention functionality."""
    PipelineConfig.reset()
    query, key, value, local_query, local_key, local_value = sample_tensors(
        batch_size, num_heads, seq_len, head_dim, world_size
    )

    if tensor_layout == "NHD":
        local_query = local_query.permute(0, 2, 1, 3).contiguous()
        local_key = local_key.permute(0, 2, 1, 3).contiguous()
        local_value = local_value.permute(0, 2, 1, 3).contiguous()

    dit_config = DiTParallelConfig()
    dit_config.set_config(
        tp_size=1,
        cfg_size=1,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
    )
    attn = ditAttnProcessor()
    AttentionOpManager.set_attn_config(attn_type=attn_type)
    local_output = attn.visual_gen_attn(local_query, local_key, local_value, tensor_layout=tensor_layout)
    if tensor_layout == "NHD":
        local_output = local_output.permute(0, 2, 1, 3)

    ref_output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
    local_ref_output = ref_output.chunk(world_size, dim=2)[dist.get_rank()]

    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        raise RuntimeError("Accuracy test failed")


def test_joint_attn_parallel(batch_size, num_heads, seq_len, head_dim, world_size, ulysses_size, ring_size, attn_type):
    """Test basic parallel attention functionality."""
    PipelineConfig.reset()
    query, key, value, local_query, local_key, local_value, joint_seq_length, valid_joint_seq_length = (
        sample_joint_tensors(batch_size, num_heads, seq_len, head_dim, world_size)
    )
    dit_config = DiTParallelConfig()
    dit_config.set_config(
        tp_size=1,
        cfg_size=1,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
    )
    attn = ditAttnProcessor()
    AttentionOpManager.set_attn_config(attn_type=attn_type)
    for _ in range(10):
        with torch.cuda.nvtx.range(f"visual_gen attn u{ulysses_size} r{ring_size}"):
            local_output = attn.visual_gen_attn(
                local_query,
                local_key,
                local_value,
                tensor_layout="HND",
                joint_seq_length=joint_seq_length,
                valid_joint_seq_length=valid_joint_seq_length,
                joint_strategy="rear",
            )
    torch.cuda.synchronize()

    num_iter = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iter):
        with torch.cuda.nvtx.range(f"visual_gen attn u{ulysses_size} r{ring_size}"):
            local_output = attn.visual_gen_attn(
                local_query,
                local_key,
                local_value,
                tensor_layout="HND",
                joint_seq_length=joint_seq_length,
                valid_joint_seq_length=valid_joint_seq_length,
                joint_strategy="rear",
            )
    end_event.record()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        logger.info(f"visual_gen attn u{ulysses_size} r{ring_size} time: {start_event.elapsed_time(end_event) / num_iter} ms")

    for batch_idx in range(batch_size):
        logger.info(f"evaluating batch_idx: {batch_idx}")
        valid_seq_length = seq_len - (joint_seq_length - valid_joint_seq_length[batch_idx])
        ref_output = F.scaled_dot_product_attention(
            query[batch_idx, :, :valid_seq_length, :].unsqueeze(0),
            key[batch_idx, :, :valid_seq_length, :].unsqueeze(0),
            value[batch_idx, :, :valid_seq_length, :].unsqueeze(0),
            is_causal=False,
        )
        ref_image_output = ref_output[:, :, : (valid_seq_length - valid_joint_seq_length[batch_idx]), :].chunk(
            world_size, dim=2
        )[dist.get_rank()]
        ref_joint_output = ref_output[:, :, (valid_seq_length - valid_joint_seq_length[batch_idx]) :, :]
        ref_output = torch.cat((ref_image_output, ref_joint_output), dim=2)

        cur_local_output = local_output[batch_idx, :, : ref_output.shape[2], :]

        cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_similarity = cos_sim(
            cur_local_output.reshape(-1).to(torch.float32), ref_output.reshape(-1).to(torch.float32)
        )
        print("cos_similarity total: ", cos_similarity)
        if cos_similarity < 0.99:
            raise RuntimeError("Accuracy test failed")


def test_uneven_varlen_attn_parallel(
    num_heads, seq_len_list, head_dim, world_size, ulysses_size, ring_size, attn_type
):
    """Test uneven parallel attention functionality."""

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    
    cu_seqlens_q = [0]
    for seq_len in seq_len_list:
        cu_seqlens_q.append(cu_seqlens_q[-1] + seq_len)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device).to(torch.int32)
    cu_seqlens_k = cu_seqlens_q.clone()
    max_seqlen_q = max(seq_len_list)
    max_seqlen_k = max_seqlen_q

    total_seq_len = sum(seq_len_list)
    seq_len_padded = math.ceil(total_seq_len / world_size) * world_size
    uneven_number = seq_len_padded - total_seq_len
    PipelineConfig.reset()
    query, key, value, local_query, local_key, local_value = sample_tensors(
        1, num_heads, seq_len_padded, head_dim, world_size
    )

    seq_len_cur_rank = torch.tensor([local_query.shape[2]], dtype=torch.int32, device=device)
    if dist.get_rank() == world_size - 1:
        seq_len_cur_rank = seq_len_cur_rank - uneven_number

    dit_config = DiTParallelConfig()
    dit_config.set_config(
        tp_size=1,
        cfg_size=1,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
    )
    
    PipelineConfig.set_uneven_cp_config(total_seq_len, seq_len_padded, seq_len_cur_rank, dit_config)
    attn = ditAttnProcessor()
    AttentionOpManager.set_attn_config(attn_type=attn_type)
    local_output = attn.visual_gen_attn(local_query, local_key, local_value, tensor_layout="HND", cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)

    local_ref_output_list = []
    for i in range(len(seq_len_list)):
        q_tmp = query[:, :, cu_seqlens_q[i]:cu_seqlens_q[i+1], :]
        k_tmp = key[:, :, cu_seqlens_k[i]:cu_seqlens_k[i+1], :]
        v_tmp = value[:, :, cu_seqlens_k[i]:cu_seqlens_k[i+1], :]
        tmp_output = F.scaled_dot_product_attention(q_tmp, k_tmp, v_tmp, is_causal=False)
        local_ref_output_list.append(tmp_output)

    ref_output = torch.cat(local_ref_output_list, dim=2) 

    local_ref_output = ref_output.chunk(world_size, dim=2)[dist.get_rank()]

    if dist.get_rank() == world_size - 1 and seq_len_padded > total_seq_len:
        local_output = local_output[ :, :, :-uneven_number, :]

    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        raise RuntimeError("Accuracy test failed")


def test_uneven_attn_parallel(
    batch_size, num_heads, seq_len_padded, head_dim, world_size, ulysses_size, ring_size, attn_type
):
    """Test uneven parallel attention functionality."""
    PipelineConfig.reset()
    query, key, value, local_query, local_key, local_value = sample_tensors(
        batch_size, num_heads, seq_len_padded, head_dim, world_size
    )

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    uneven_number = world_size - 1

    seq_len_cur_rank = torch.tensor([local_query.shape[2]], dtype=torch.int32, device=device)
    if dist.get_rank() == world_size - 1:
        seq_len_cur_rank = seq_len_cur_rank - uneven_number
    dit_config = DiTParallelConfig()
    dit_config.set_config(
        tp_size=1,
        cfg_size=1,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
    )
    PipelineConfig.set_uneven_cp_config(seq_len_padded - uneven_number, seq_len_padded, seq_len_cur_rank, dit_config)
    attn = ditAttnProcessor()
    AttentionOpManager.set_attn_config(attn_type=attn_type)
    local_output = attn.visual_gen_attn(local_query, local_key, local_value, tensor_layout="HND")

    query = query[:, :, :-uneven_number, :]
    key = key[:, :, :-uneven_number, :]
    value = value[:, :, :-uneven_number, :]

    ref_output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
    local_ref_output = ref_output.chunk(world_size, dim=2)[dist.get_rank()]

    if dist.get_rank() == world_size - 1:
        local_output = local_output[:, :, :-uneven_number, :]

    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_similarity = cos_sim(local_output.reshape(-1).to(torch.float32), local_ref_output.reshape(-1).to(torch.float32))
    print("cos_similarity total: ", cos_similarity)
    if cos_similarity < 0.99:
        raise RuntimeError("Accuracy test failed")


if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    if world_size not in [2, 4, 8]:
        logger.warning("Only support world_size in [2, 4, 8]")
        exit(0)

    test_sage_attn = False
    test_flash_attn3 = False
    test_flash_attn4 = False
    test_fivx = False

    capability = torch.cuda.get_device_capability(0)
    sm = f"{capability[0]}{capability[1]}"
    try:
        import sageattention  # noqa: F401
        if sm in ["89", "90", "120"]:
            test_sage_attn = True
    except ImportError:
        logger.warning("SageAttention is not installed. test sage-attn will be skipped.")

    try:
        import flash_attn_interface  # noqa: F401

        test_flash_attn3 = True
    except ImportError:
        logger.warning("FlashAttn3 is not installed. test flash-attn3 will be skipped.")

    try:
        import flash_attn.cute.interface  # noqa: F401

        test_flash_attn4 = True
    except ImportError:
        logger.warning("FlashAttn4 is not installed. test flash-attn4 will be skipped.")

    try:
        import flashinfer_vx  # noqa: F401

        test_fivx = True
    except ImportError:
        print("FlashInfer-VX (SageAttn for Blackwell) is not installed, test fivx will be skipped.")

    if test_sage_attn:
        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="sage-attn",
        )
        test_uneven_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len_padded=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="sage-attn",
        )
        if world_size // 2 >= 1:
            test_attn_parallel(
                batch_size=1,
                num_heads=24,
                seq_len=6 * 8 * 1024,
                head_dim=128,
                world_size=world_size,
                ulysses_size=2,
                ring_size=world_size // 2,
                attn_type="sage-attn",
            )
        if world_size // 4 >= 1:
            test_uneven_attn_parallel(
                batch_size=1,
                num_heads=24,
                seq_len_padded=6 * 8 * 1024,
                head_dim=128,
                world_size=world_size,
                ulysses_size=4,
                ring_size=world_size // 4,
                attn_type="sage-attn",
            )

        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="sage-attn",
        )
        test_uneven_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len_padded=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="sage-attn",
        )
        if world_size // 2 >= 1:
            test_joint_attn_parallel(
                batch_size=1,
                num_heads=24,
                seq_len=4096,
                head_dim=128,
                world_size=world_size,
                ulysses_size=world_size,
                ring_size=1,
                attn_type="sage-attn",
            )

    if test_flash_attn3:
        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="flash-attn3",
        )
        test_uneven_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len_padded=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="flash-attn3",
        )
        if world_size // 2 >= 1:
            test_attn_parallel(
                batch_size=1,
                num_heads=24,
                seq_len=6 * 8 * 1024,
                head_dim=128,
                world_size=world_size,
                ulysses_size=2,
                ring_size=world_size // 2,
                attn_type="flash-attn3",
            )
        if world_size // 4 >= 1:
            test_uneven_attn_parallel(
                batch_size=1,
                num_heads=24,
                seq_len_padded=6 * 8 * 1024,
                head_dim=128,
                world_size=world_size,
                ulysses_size=4,
                ring_size=world_size // 4,
                attn_type="flash-attn3",
            )

        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="flash-attn3",
        )
        test_uneven_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len_padded=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="flash-attn3",
        )

        test_uneven_varlen_attn_parallel(
            num_heads=24,
            seq_len_list=[1 * 8 * 1024 - 1, 3 * 8 * 1024],
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="flash-attn3",
        )

    if test_flash_attn4:
        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="flash-attn4",
            tensor_layout="HND",
        )

        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="flash-attn4",
            tensor_layout="NHD",
        )

        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="flash-attn4",
            tensor_layout="HND",
        )

        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=2,
            ring_size=2,
            attn_type="flash-attn4",
            tensor_layout="NHD",
        )

    if test_fivx:
        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=world_size,
            ring_size=1,
            attn_type="fivx",
        )

        test_attn_parallel(
            batch_size=1,
            num_heads=24,
            seq_len=6 * 8 * 1024,
            head_dim=128,
            world_size=world_size,
            ulysses_size=1,
            ring_size=world_size,
            attn_type="fivx",
        )

    dist.destroy_process_group()
