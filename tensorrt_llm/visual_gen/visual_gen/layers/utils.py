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

import torch
import torch.distributed as dist
import torch.nn.functional as F

from visual_gen.configs.parallel import get_dit_parallel_config
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.quant_per_token_block128 import (
    dequantize_per_token_block128_packed,
    quantize_per_token_block128_packed,
)
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def all_to_all(tensor, scatter_idx, gather_idx, tensor_layout, group=None, int8_comm=False):
    """Perform all-to-all communication on a tensor.

    Args:
        tensor (torch.Tensor): Input tensor for all-to-all communication
        scatter_idx (int): Dimension to scatter, will split along this dimension and then scatter to all processes
        gather_idx (int): Dimension to gather, will gather from all processes and then concatenate along this dimension
        group (ProcessGroup, optional): Process group to use for communication
        int8_comm (bool): Whether to use 8bit quantization for communication.
                       Each rank quantizes its tensor once (one scale per rank).
                       After all_to_all, each received chunk will have the scale from its source rank.

    Returns:
        torch.Tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    if scatter_idx == gather_idx:
        raise ValueError("scatter_idx and gather_idx must be different")

    def chunk_tensor(tensor, scatter_idx):
        t_shape = list(tensor.shape)
        if t_shape[scatter_idx] % world_size != 0:
            raise ValueError(
                f"Dimension {scatter_idx} of tensor {tensor.shape} must be divisible by world size {world_size}"
            )
        chunk_size = t_shape[scatter_idx] // world_size
        new_shape = list()
        for i in range(len(t_shape)):
            if i != scatter_idx:
                new_shape.append(t_shape[i])
            else:
                new_shape.extend([world_size, chunk_size])
        tensor = tensor.reshape(*new_shape)
        # move scatter_idx to front
        tensor = tensor.permute(scatter_idx, *[i for i in range(len(new_shape)) if i != scatter_idx]).contiguous()
        return tensor

    # chunk tensor for all_to_all
    if int8_comm:
        original_dtype = tensor.dtype
        # Quantize and pack in one step: output shape [..., seq_len, 132]
        # 132 = 128 bytes (int8 data) + 4 bytes (float32 scale)
        packed_tensor = quantize_per_token_block128_packed(tensor, tensor_layout=tensor_layout)
        packed_tensor = chunk_tensor(packed_tensor, scatter_idx)

        # Perform single all_to_all communication on packed tensor
        output_packed = torch.empty_like(packed_tensor)
        dist.all_to_all_single(output_packed, packed_tensor, group=group)

    else:
        tensor = chunk_tensor(tensor, scatter_idx)

        # Perform all2all
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, group=group)

    # output: e.g., [world_size, B, chunked_H, chunked_S, D] if scatter_idx == 1, gather_idx == 2 -> [B, chunked_H, S, D]
    def reorder_tensor(tensor, gather_idx):
        t_shape = list(tensor.shape)
        world_size = t_shape[0]
        # insert front to gather_idx + 1
        permute_idx = list()
        for i in range(1, len(t_shape)):
            if i != gather_idx + 1:
                permute_idx.append(i)
            else:
                permute_idx.extend([0, i])
        tensor = tensor.permute(*permute_idx).contiguous()

        # reshape tensor
        new_shape = list()
        for i in range(1, len(t_shape)):
            if i != gather_idx + 1:
                new_shape.append(t_shape[i])
            else:
                new_shape.append(world_size * t_shape[i])

        tensor = tensor.reshape(*new_shape)

        return tensor

    if int8_comm:
        # Reorder packed tensor
        output_packed = reorder_tensor(output_packed, gather_idx)
        # Dequantize packed tensor in one step
        output = dequantize_per_token_block128_packed(output_packed, tensor_layout=tensor_layout)
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
    else:
        output = reorder_tensor(output, gather_idx)

    return output


def ulysses_a2a_in(
    query,
    key,
    value,
    attn_mask,
    tensor_layout,
    only_split_query=False,
    ulysses_size=1,
    ulysses_rank=0,
    ulysses_group=None,
    int8_all2all=False,
    fuse_qkv=False,
):
    if ulysses_size == 1:
        return query, key, value, attn_mask

    if attn_mask is not None:
        raise NotImplementedError("Attn mask not supported for ulysses_a2a_in")

    if tensor_layout == "HND":
        scatter_idx = 1
        gather_idx = 2
    elif tensor_layout == "NHD":
        scatter_idx = 2
        gather_idx = 1
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")

    # [B, H, S/N, D] -> [B, H/N, S, D]
    if only_split_query:
        # Only query needs all-to-all, k/v just get chunked locally
        query = all_to_all(
            query,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
            int8_comm=int8_all2all,
        )
        key = torch.chunk(key, ulysses_size, dim=scatter_idx)[ulysses_rank]
        value = torch.chunk(value, ulysses_size, dim=scatter_idx)[ulysses_rank]
    elif fuse_qkv:
        # Fused communication: concatenate q/k/v into [3*B, H, S/N, D], single all-to-all, then split
        # This reduces 3 NCCL calls to 1, improving efficiency
        qkv = torch.cat([query, key, value], dim=0)
        qkv = all_to_all(
            qkv,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
            int8_comm=int8_all2all,
        )
        query, key, value = torch.chunk(qkv, 3, dim=0)
    else:
        # Independent communication: 3 separate all-to-all operations (default, safe)
        query = all_to_all(
            query,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
            int8_comm=int8_all2all,
        )
        key = all_to_all(
            key,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
            int8_comm=int8_all2all,
        )
        value = all_to_all(
            value,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=ulysses_group,
            tensor_layout=tensor_layout,
            int8_comm=int8_all2all,
        )

    return query, key, value, attn_mask


def ulysses_a2a_out(output, tensor_layout, ulysses_size=1, ulysses_group=None, int8_all2all=False):
    if ulysses_size == 1:
        return output

    assert tensor_layout in ["NHD", "HND"], f"tensor_layout must be NHD or HND, but got {tensor_layout}"
    if tensor_layout == "HND":
        scatter_idx = 2
        gather_idx = 1
    elif tensor_layout == "NHD":
        scatter_idx = 1
        gather_idx = 2
    else:
        raise ValueError(f"Invalid tensor layout: {tensor_layout}")
    # [B, H/N, S, D] -> [B, H, S/N, D]
    output = all_to_all(
        output,
        scatter_idx=scatter_idx,
        gather_idx=gather_idx,
        tensor_layout=tensor_layout,
        group=ulysses_group,
        int8_comm=int8_all2all,
    )
    return output


# @torch.compile
def ring_fwd_out_correction(
    out: torch.Tensor, out_per_step: torch.Tensor, softmax_lse: torch.Tensor, softmax_lse_per_step: torch.Tensor
):
    """Merge partial outputs of each step in ring attention"""
    new_out = out - F.sigmoid(softmax_lse_per_step.unsqueeze(-1) - softmax_lse.unsqueeze(-1)) * (out - out_per_step)
    out.copy_(new_out)


# @torch.compile
def ring_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in ring attention"""
    new_lse = softmax_lse - F.logsigmoid(softmax_lse - softmax_lse_per_step)
    softmax_lse.copy_(new_lse)


def ring_attn_p2p_communicate(rank, send_tensor, send_dst, recv_tensor, recv_src, ring_group):
    """Point-to-point communications of KV and dKV in ring attention"""
    send_recv_ops = []
    if rank % 2 == 0:
        send_op = torch.distributed.P2POp(torch.distributed.isend, send_tensor, group_peer=send_dst, group=ring_group)
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_tensor, group_peer=recv_src, group=ring_group)
        send_recv_ops.append(send_op)
        send_recv_ops.append(recv_op)
    else:
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_tensor, group_peer=recv_src, group=ring_group)
        send_op = torch.distributed.P2POp(torch.distributed.isend, send_tensor, group_peer=send_dst, group=ring_group)
        send_recv_ops.append(recv_op)
        send_recv_ops.append(send_op)
    send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)

    return send_recv_reqs


def ulysses_wrapper(func):
    # @torch.cuda.nvtx.range("ditAttn.ulysses_wrapper")
    def wrapper(self, query, key, value, tensor_layout, attn_mask=None, **kwargs):
        only_split_query = kwargs.pop("only_split_query", False)
        # if ulysses_size == 1, no need to do ulysses_a2a_in and ulysses_a2a_out
        if get_dit_parallel_config().ulysses_size() == 1:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        ulysses_size = get_dit_parallel_config().ulysses_size()
        ulysses_rank = get_dit_parallel_config().ulysses_rank()
        ulysses_group = get_dit_parallel_config().ulysses_group()
        ring_size = get_dit_parallel_config().ring_size()
        cp_size = get_dit_parallel_config().cp_size()

        assert tensor_layout in ["NHD", "HND"], f"tensor_layout must be NHD or HND, but got {tensor_layout}"
        if tensor_layout == "HND":
            seq_dim = 2
            head_dim = 1
        elif tensor_layout == "NHD":
            seq_dim = 1
            head_dim = 2
        else:
            raise ValueError(f"Invalid tensor layout: {tensor_layout}")

        if query.shape[head_dim] % ulysses_size != 0:
            raise ValueError(
                f"Head dim {head_dim} of query {query.shape} must be divisible by ulysses size {ulysses_size}"
            )
        if key.shape[head_dim] % ulysses_size != 0:
            raise ValueError(f"Head dim {head_dim} of key {key.shape} must be divisible by ulysses size {ulysses_size}")
        if value.shape[head_dim] % ulysses_size != 0:
            raise ValueError(
                f"Head dim {head_dim} of value {value.shape} must be divisible by ulysses size {ulysses_size}"
            )

        joint_seq_length = kwargs.get("joint_seq_length", 0)
        if joint_seq_length > 0:
            if ring_size == 1:
                joint_strategy = kwargs.get("joint_strategy", "none")
                assert joint_strategy != "none", "joint_strategy can not be none when joint_seq_length > 0"
                if joint_strategy == "rear":
                    joint_query = torch.narrow(
                        query, seq_dim, query.shape[seq_dim] - joint_seq_length, joint_seq_length
                    ).contiguous()
                    joint_key = torch.narrow(
                        key, seq_dim, key.shape[seq_dim] - joint_seq_length, joint_seq_length
                    ).contiguous()
                    joint_value = torch.narrow(
                        value, seq_dim, value.shape[seq_dim] - joint_seq_length, joint_seq_length
                    ).contiguous()
                    query = torch.narrow(query, seq_dim, 0, query.shape[seq_dim] - joint_seq_length).contiguous()
                    key = torch.narrow(key, seq_dim, 0, key.shape[seq_dim] - joint_seq_length).contiguous()
                    value = torch.narrow(value, seq_dim, 0, value.shape[seq_dim] - joint_seq_length).contiguous()
                else:
                    joint_query = torch.narrow(query, seq_dim, 0, joint_seq_length).contiguous()
                    joint_key = torch.narrow(key, seq_dim, 0, joint_seq_length).contiguous()
                    joint_value = torch.narrow(value, seq_dim, 0, joint_seq_length).contiguous()
                    query = torch.narrow(
                        query, seq_dim, joint_seq_length, query.shape[seq_dim] - joint_seq_length
                    ).contiguous()
                    key = torch.narrow(
                        key, seq_dim, joint_seq_length, key.shape[seq_dim] - joint_seq_length
                    ).contiguous()
                    value = torch.narrow(
                        value, seq_dim, joint_seq_length, value.shape[seq_dim] - joint_seq_length
                    ).contiguous()
                # split joint query, key and value along head dimension to follow the same logic as query, key and value.
                joint_query = torch.chunk(joint_query, ulysses_size, dim=head_dim)[ulysses_rank]
                joint_key = torch.chunk(joint_key, ulysses_size, dim=head_dim)[ulysses_rank]
                joint_value = torch.chunk(joint_value, ulysses_size, dim=head_dim)[ulysses_rank]
            else:
                # k/v don't have joint part if ring attn is enabled. The joint part is handled by ring attn.
                joint_query = torch.narrow(
                    query, seq_dim, query.shape[seq_dim] - joint_seq_length, joint_seq_length
                ).contiguous()
                joint_query = torch.chunk(joint_query, ulysses_size, dim=head_dim)[ulysses_rank]
                query = torch.narrow(query, seq_dim, 0, query.shape[seq_dim] - joint_seq_length).contiguous()

        # Apply ulysses_a2a_in before the function call
        int8_all2all = PipelineConfig.int8_ulysses
        fuse_qkv = PipelineConfig.fuse_qkv_in_ulysses
        query, key, value, attn_mask = ulysses_a2a_in(
            query,
            key,
            value,
            attn_mask,
            tensor_layout,
            only_split_query=only_split_query,
            ulysses_size=ulysses_size,
            ulysses_rank=ulysses_rank,
            ulysses_group=ulysses_group,
            int8_all2all=int8_all2all,
            fuse_qkv=fuse_qkv,
        )

        # truncate and pad if cp is uneven
        truncate_and_pad = PipelineConfig.seq_len_all_ranks is not None

        if truncate_and_pad:
            assert (
                joint_seq_length == 0
            ), "joint_seq_length > 0 is not supported by ulysses wrapper when truncate_and_pad is True"
            if ring_size == 1:
                # there is no ring, so we can use PipelineConfig.seq_len to do truncate and pad
                seq_len = PipelineConfig.ulysses_seq_all_ring_ranks[0]

                # Truncate key and value tensors using torch.narrow
                key = torch.narrow(key, seq_dim, 0, seq_len).contiguous()
                value = torch.narrow(value, seq_dim, 0, seq_len).contiguous()
            else:
                # there is ring, so we use the q_seq and kv_seq passed in by ring in kwargs to do truncate and pad
                q_seq = kwargs.get("q_seq_len")
                kv_seq = kwargs.get("kv_seq_len")

                # Truncate key and value tensors using torch.narrow
                key = torch.narrow(key, seq_dim, 0, kv_seq).contiguous()
                value = torch.narrow(value, seq_dim, 0, kv_seq).contiguous()
                kwargs.pop("q_seq_len")
                kwargs.pop("kv_seq_len")

        if joint_seq_length > 0:
            if ring_size == 1 and cp_size == 1:
                if joint_strategy == "rear":
                    query = torch.cat((query, joint_query), dim=seq_dim)
                    key = torch.cat((key, joint_key), dim=seq_dim)
                    value = torch.cat((value, joint_value), dim=seq_dim)
                else:
                    query = torch.cat((joint_query, query), dim=seq_dim)
                    key = torch.cat((joint_key, key), dim=seq_dim)
                    value = torch.cat((joint_value, value), dim=seq_dim)
            else:
                raise NotImplementedError("joint_seq_length > 0 is not supported when ring size > 1 or cp size > 1")

        # Call the original function
        result = func(self, query, key, value, tensor_layout, attn_mask, **kwargs)
        if ring_size == 1:
            if joint_seq_length > 0:
                if joint_strategy == "rear":
                    joint_result = torch.narrow(
                        result, seq_dim, result.shape[seq_dim] - joint_seq_length, joint_seq_length
                    ).contiguous()
                    result = torch.narrow(result, seq_dim, 0, result.shape[seq_dim] - joint_seq_length).contiguous()
                else:
                    joint_result = torch.narrow(result, seq_dim, 0, joint_seq_length).contiguous()
                    result = torch.narrow(
                        result, seq_dim, joint_seq_length, result.shape[seq_dim] - joint_seq_length
                    ).contiguous()

                joint_result_gathered = torch.empty(
                    ulysses_size, *joint_result.shape, device=joint_result.device, dtype=joint_result.dtype
                )
                work = torch.distributed.all_gather_into_tensor(
                    joint_result_gathered, joint_result, group=ulysses_group, async_op=True
                )
                work.wait()
                if tensor_layout == "HND":
                    # [ulysses_size, B, H, S, D] -> [B, ulysses_size * H, S, D]
                    _, B, H, S, D = joint_result_gathered.shape
                    joint_result = joint_result_gathered.permute(1, 0, 2, 3, 4).reshape(B, ulysses_size * H, S, D)
                elif tensor_layout == "NHD":
                    # [ulysses_size, B, S, H, D] -> [B, S, ulysses_size * H, D]
                    _, B, S, H, D = joint_result_gathered.shape
                    joint_result = joint_result_gathered.permute(1, 2, 0, 3, 4).reshape(B, S, ulysses_size * H, D)
                else:
                    raise ValueError(f"Invalid tensor layout: {tensor_layout}")

            # if ring size is 1, return_lse is false, result only has output.
            if truncate_and_pad:
                # Zero out padding using torch.narrow
                if result.shape[seq_dim] > seq_len:
                    padding_part = torch.narrow(result, seq_dim, seq_len, result.shape[seq_dim] - seq_len)
                    padding_part.zero_()
            result = ulysses_a2a_out(
                result, tensor_layout, ulysses_size=ulysses_size, ulysses_group=ulysses_group, int8_all2all=int8_all2all
            )
            if joint_seq_length > 0:
                if joint_strategy == "rear":
                    result = torch.cat((result, joint_result), dim=seq_dim)
                else:
                    result = torch.cat((joint_result, result), dim=seq_dim)
            return result
        else:
            # if ring size is not 1, return_lse is true, result has output and softmax_lse.
            block_out = result[0]
            block_lse = result[1]
            if truncate_and_pad:
                # Zero out padding using torch.narrow
                if block_out.shape[seq_dim] > q_seq:
                    padding_part = torch.narrow(block_out, seq_dim, q_seq, block_out.shape[seq_dim] - q_seq)
                    padding_part.zero_()

                if block_lse.shape[seq_dim] > q_seq:
                    padding_part = torch.narrow(block_lse, seq_dim, q_seq, block_lse.shape[seq_dim] - q_seq)
                    padding_part.zero_()
            block_out = ulysses_a2a_out(
                block_out,
                tensor_layout,
                ulysses_size=ulysses_size,
                ulysses_group=ulysses_group,
                int8_all2all=int8_all2all,
            )
            block_lse = ulysses_a2a_out(
                block_lse,
                tensor_layout,
                ulysses_size=ulysses_size,
                ulysses_group=ulysses_group,
                int8_all2all=int8_all2all,
            )
            return (block_out, block_lse)

    return wrapper


def get_kv_rank(ring_size, ring_rank, cur_iter):
    # get the the source rank of kv tensor in current iter
    return (ring_size + ring_rank - cur_iter) % ring_size


def ring_wrapper(func):
    # @torch.cuda.nvtx.range("ditAttn.ring_wrapper")
    def wrapper(self, query, key, value, tensor_layout, attn_mask=None, **kwargs):

        ring_size = get_dit_parallel_config().ring_size()
        ring_group = get_dit_parallel_config().ring_group()
        ulysses_size = get_dit_parallel_config().ulysses_size()

        if ring_size == 1:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        self.select_attn_impl()
        if self.attn_impl.__class__.__name__ not in ["SageAttn", "FlashAttn3", "FlashAttn4", "FlashInferVx"]:
            raise NotImplementedError(
                "Ring wrapper only supports SageAttn/FlashAttn3/FlashAttn4, but got "
                + self.attn_impl.__class__.__name__
            )

        cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
        if cu_seqlens_q is not None:
            raise NotImplementedError("var_len_attention is not supported by ring wrapper")

        joint_seq_length = kwargs.get("joint_seq_length", 0)
        valid_joint_seq_length = kwargs.get("valid_joint_seq_length", None)

        rank = get_dit_parallel_config().ring_rank()
        send_dst = (rank + 1) % ring_size
        recv_src = (rank - 1) % ring_size

        # Determine sequence dimension based on tensor layout
        if tensor_layout == "HND":
            seq_dim = 2  # kv_inputs is [2, B, H, S, D], so seq_dim is 3
        elif tensor_layout == "NHD":
            seq_dim = 1  # kv_inputs is [2, B, S, H, D], so seq_dim is 2
        else:
            # Default to dimension 3 for backward compatibility
            seq_dim = 2

        if joint_seq_length > 0:
            # Joint sequence will be handled outside ring attn
            kwargs["joint_seq_length"] = 0
            kwargs["valid_joint_seq_length"] = None
            joint_strategy = kwargs.get("joint_strategy", "none")
            assert joint_strategy != "none", "joint_strategy can not be none when joint_seq_length > 0"
            if joint_strategy == "rear":
                joint_key = torch.narrow(
                    key, seq_dim, key.shape[seq_dim] - joint_seq_length, joint_seq_length
                ).contiguous()
                joint_value = torch.narrow(
                    value, seq_dim, value.shape[seq_dim] - joint_seq_length, joint_seq_length
                ).contiguous()
                key = torch.narrow(key, seq_dim, 0, key.shape[seq_dim] - joint_seq_length).contiguous()
                value = torch.narrow(value, seq_dim, 0, value.shape[seq_dim] - joint_seq_length).contiguous()
            else:
                joint_key = torch.narrow(key, seq_dim, 0, joint_seq_length).contiguous()
                joint_value = torch.narrow(value, seq_dim, 0, joint_seq_length).contiguous()
                key = torch.narrow(key, seq_dim, joint_seq_length, key.shape[seq_dim] - joint_seq_length).contiguous()
                value = torch.narrow(
                    value, seq_dim, joint_seq_length, value.shape[seq_dim] - joint_seq_length
                ).contiguous()

        p2p_comm_buffers = [None, None]
        p2p_comm_buffers[0] = torch.cat((key.unsqueeze(0), value.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        out = None
        softmax_lse = None
        for i in range(ring_size):
            kv_rank = get_kv_rank(ring_size, rank, i)
            # wait until KV is received
            for req in send_recv_reqs[(i + 1) % 2]:
                req.wait()

            if i < (ring_size - 1):
                p2p_comm_buffers[(i + 1) % 2] = torch.empty_like(p2p_comm_buffers[i % 2])
                send_recv_reqs[i % 2] = ring_attn_p2p_communicate(
                    rank, p2p_comm_buffers[i % 2], send_dst, p2p_comm_buffers[(i + 1) % 2], recv_src, ring_group
                )
            kv_inputs = p2p_comm_buffers[i % 2]

            # do truncate and pad if cp is uneven and ulysses size is 1,
            # if ulysses size > 1, truncate and pad will be done in ulysses_wrapper.
            if PipelineConfig.seq_len_cur_ring_group is not None:
                if ulysses_size == 1:
                    # seq_dim+1 because kv_inputs is concated to [2, B, H, S, D] or [2, B, S, H, D]
                    if kv_inputs.shape[seq_dim + 1] != PipelineConfig.seq_len_cur_ring_group[kv_rank]:
                        # Truncate kv_inputs using torch.narrow
                        kv_inputs = torch.narrow(
                            kv_inputs, seq_dim + 1, 0, PipelineConfig.seq_len_cur_ring_group[kv_rank]
                        )
                else:
                    kwargs["q_seq_len"] = PipelineConfig.ulysses_seq_all_ring_ranks[rank]
                    kwargs["kv_seq_len"] = PipelineConfig.ulysses_seq_all_ring_ranks[kv_rank]

            kwargs["return_lse"] = True
            with torch.cuda.device(
                query.device.index
            ):  # we need this line because a bug in flash-attn4 https://github.com/Dao-AILab/flash-attention/pull/1793
                block_out = func(self, query, kv_inputs[0], kv_inputs[1], tensor_layout, attn_mask, **kwargs)

            out_per_step = block_out[0]
            softmax_lse_per_step = block_out[1]

            if i == 0:
                softmax_lse = torch.clone(softmax_lse_per_step).to(torch.float)
                out = torch.clone(out_per_step)
            else:
                ring_fwd_out_correction(out, out_per_step, softmax_lse, softmax_lse_per_step)
                ring_fwd_softmax_lse_correction(softmax_lse, softmax_lse_per_step)

        if joint_seq_length > 0:
            kwargs["joint_seq_length"] = joint_seq_length
            kwargs["valid_joint_seq_length"] = valid_joint_seq_length
            kwargs["only_split_query"] = (
                True  # joint key and joint value are not splited by sequence parallel, so only split query.
            )
            block_out = func(self, query, joint_key, joint_value, tensor_layout, attn_mask, **kwargs)
            ring_fwd_out_correction(out, block_out[0], softmax_lse, block_out[1])
            ring_fwd_softmax_lse_correction(softmax_lse, block_out[1])

        # Determine output sequence dimension based on tensor layout (for output tensor)
        if tensor_layout == "HND":
            out_seq_dim = 2  # out is [B, H, S, D], so seq_dim is 2
        elif tensor_layout == "NHD":
            out_seq_dim = 1  # out is [B, S, H, D], so seq_dim is 1
        else:
            # Default to dimension 2 for backward compatibility
            out_seq_dim = 2

        if (
            PipelineConfig.seq_len_cur_ring_group is not None
            and ulysses_size == 1
            and out.shape[out_seq_dim] > PipelineConfig.seq_len_cur_ring_group[rank]
        ):
            # Zero out padding using torch.narrow
            start_pos = PipelineConfig.seq_len_cur_ring_group[rank]
            padding_length = out.shape[out_seq_dim] - start_pos
            padding_part = torch.narrow(out, out_seq_dim, start_pos, padding_length)
            padding_part.zero_()

        return out

    return wrapper


def cp_wrapper(func):
    """
    `all_gather` kv from other ranks thus only need one communication but may have larger message size compared to `ulysses` and can't overlap compared to `ring`.
    This is usually suitable for small kv, such as in image generation models.
    """

    # @torch.cuda.nvtx.range("ditAttn.cp_wrapper")
    def wrapper(self, query, key, value, tensor_layout, attn_mask=None, **kwargs):

        cp_size = get_dit_parallel_config().cp_size()
        cp_group = get_dit_parallel_config().cp_group()

        if cp_size == 1:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
        if cu_seqlens_q is not None:
            raise NotImplementedError("var_len_attention is not supported by cp wrapper")

        joint_seq_length = kwargs.get("joint_seq_length", 0)

        # Determine sequence dimension based on tensor layout
        assert len(query.shape) == 4, f"Query shape must be (B, H, S, D), but got {query.shape}"
        assert len(key.shape) == 4, f"Key shape must be (B, H, S, D), but got {key.shape}"
        assert len(value.shape) == 4, f"Value shape must be (B, H, S, D), but got {value.shape}"
        if tensor_layout == "HND":
            seq_dim = 2
        elif tensor_layout == "NHD":
            seq_dim = 1
        else:
            raise ValueError(f"Invalid tensor layout: {tensor_layout}")

        if joint_seq_length > 0:
            ulysses_size = get_dit_parallel_config().ulysses_size()
            ring_size = get_dit_parallel_config().ring_size()
            if ulysses_size > 1 or ring_size > 1:
                raise NotImplementedError(
                    "joint_seq_length > 0 is not supported by cp wrapper when ulysses size > 1 or ring size > 1"
                )
            joint_strategy = kwargs.get("joint_strategy", "none")
            assert joint_strategy != "none", "joint_strategy can not be none when joint_seq_length > 0"
            if joint_strategy == "rear":
                joint_key = torch.narrow(
                    key, seq_dim, key.shape[seq_dim] - joint_seq_length, joint_seq_length
                ).contiguous()
                joint_value = torch.narrow(
                    value, seq_dim, value.shape[seq_dim] - joint_seq_length, joint_seq_length
                ).contiguous()
                key = torch.narrow(key, seq_dim, 0, key.shape[seq_dim] - joint_seq_length).contiguous()
                value = torch.narrow(value, seq_dim, 0, value.shape[seq_dim] - joint_seq_length).contiguous()
            else:
                joint_key = torch.narrow(key, seq_dim, 0, joint_seq_length).contiguous()
                joint_value = torch.narrow(value, seq_dim, 0, joint_seq_length).contiguous()
                key = torch.narrow(key, seq_dim, joint_seq_length, key.shape[seq_dim] - joint_seq_length).contiguous()
                value = torch.narrow(
                    value, seq_dim, joint_seq_length, value.shape[seq_dim] - joint_seq_length
                ).contiguous()

        kv = torch.cat((key.unsqueeze(0), value.unsqueeze(0)), dim=0)

        kv_gathered = torch.empty(cp_size, *kv.shape, device=kv.device, dtype=kv.dtype)
        work = torch.distributed.all_gather_into_tensor(kv_gathered, kv, group=cp_group, async_op=True)
        work.wait()
        if tensor_layout == "HND":
            _, B, H, S, D = kv.shape
            # [cp_size, 2, B, H, S, D] -> [2, B, H, cp_size * S, D]
            kv = kv_gathered.permute(1, 2, 3, 0, 4, 5).reshape(2, B, H, cp_size * S, D).contiguous()
        elif tensor_layout == "NHD":
            _, B, S, H, D = kv.shape
            # [cp_size, 2, B, S, H, D] -> [2, B, cp_size * S, H, D]
            kv = kv_gathered.permute(1, 2, 0, 3, 4, 5).reshape(2, B, cp_size * S, H, D).contiguous()
        else:
            raise ValueError(f"Invalid tensor layout: {tensor_layout}")

        if joint_seq_length > 0:
            joint_kv = torch.cat((joint_key.unsqueeze(0), joint_value.unsqueeze(0)), dim=0)
            if joint_strategy == "rear":
                kv = torch.cat((kv, joint_kv), dim=seq_dim + 1)
            else:
                kv = torch.cat((joint_kv, kv), dim=seq_dim + 1)
        out = func(self, query, kv[0], kv[1], tensor_layout, attn_mask, **kwargs)

        return out

    return wrapper


def joint_sequence_wrapper(func):
    # @torch.cuda.nvtx.range("ditAttn.joint_sequence_wrapper")
    def wrapper(self, query, key, value, tensor_layout, attn_mask=None, **kwargs):
        """
        Joint sequence wrapper is used to remove the invalid(padding) tokens in joint sequence.
        For example, a query/key/value with size (B, S, H, D) will be split into (B, S-J, H, D) and (B, J, H, D) if joint_seq_length is J.
        The joint sequence is not splited by sequence parallel and concatenated with the splited one.
        This is often used in double-stream models, such as Flux and Hunyuan.
        In this case, image and text are used and the image part is splited but text part is not. So the shape is (B, S_splited_img + S_full_text, H, D)
        valid_joint_seq_length is the valid sequence length of the joint sequence, since the joint sequence is usually padded.
        """
        
        joint_seq_length = kwargs.pop("joint_seq_length", 0)
        valid_joint_seq_length = kwargs.pop("valid_joint_seq_length", None)
        joint_strategy = kwargs.pop("joint_strategy", "none")
        if joint_strategy not in ["none", "front", "rear"]:
            raise ValueError(f"Invalid joint strategy: {joint_strategy}")

        if joint_strategy == "none" or valid_joint_seq_length is None or joint_seq_length == 0:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        if query.shape[0] == 1 and valid_joint_seq_length == joint_seq_length:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

        return_lse = kwargs.get("return_lse", False)
        if tensor_layout == "HND":
            seq_dim = 2
            B, H, S, D = query.shape
            if return_lse:
                lse = torch.zeros((B, H, S), dtype=query.dtype, device=query.device)
        elif tensor_layout == "NHD":
            seq_dim = 1
            B, S, H, D = query.shape
            if return_lse:
                lse = torch.zeros((B, S, H), dtype=query.dtype, device=query.device)
        else:
            raise ValueError(f"Invalid tensor layout: {tensor_layout}")

        if joint_seq_length > 0:
            output = torch.empty_like(query)
            for batch_idx in range(B):
                valid_q_length = S - joint_seq_length + valid_joint_seq_length[batch_idx]
                valid_kv_length = key.shape[seq_dim] - joint_seq_length + valid_joint_seq_length[batch_idx]
                if tensor_layout == "HND":
                    cur_out = func(
                        self,
                        query[batch_idx, :, :valid_q_length, :].unsqueeze(0),
                        key[batch_idx, :, :valid_kv_length, :].unsqueeze(0),
                        value[batch_idx, :, :valid_kv_length, :].unsqueeze(0),
                        tensor_layout,
                        attn_mask,
                        **kwargs,
                    )
                    if return_lse:
                        output[batch_idx, :, :valid_q_length, :] = cur_out[0]
                        lse[batch_idx, :, :valid_q_length] = cur_out[1]  # (B,H,N)
                    else:
                        output[batch_idx, :, :valid_q_length, :] = cur_out
                elif tensor_layout == "NHD":
                    cur_out = func(
                        self,
                        query[batch_idx, :valid_q_length, :, :].unsqueeze(0),
                        key[batch_idx, :valid_kv_length, :, :].unsqueeze(0),
                        value[batch_idx, :valid_kv_length, :, :].unsqueeze(0),
                        tensor_layout,
                        attn_mask,
                        **kwargs,
                    )
                    if return_lse:
                        output[batch_idx, :valid_q_length, :, :] = cur_out[0]
                        lse[batch_idx, :valid_q_length, :] = cur_out[1]  # (B,N,H)
                    else:
                        output[batch_idx, :valid_q_length, :, :] = cur_out
                else:
                    raise ValueError(f"Invalid tensor layout: {tensor_layout}")
            if return_lse:
                return (output, lse)
            else:
                return output
        else:
            return func(self, query, key, value, tensor_layout, attn_mask, **kwargs)

    return wrapper
