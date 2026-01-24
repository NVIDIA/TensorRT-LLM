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
from typing import Tuple

import torch

from visual_gen.configs.parallel import get_dit_parallel_config
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


def _dit_sp_split(
    parallel_config,
    x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    dim: int,
    allow_uneven: bool = True,
) -> torch.Tensor:
    if parallel_config.sp_size() == 1:
        return x

    # Get parallel config parameters
    ulysses_size = parallel_config.ulysses_size()
    ring_size = parallel_config.ring_size()
    ulysses_rank = parallel_config.ulysses_rank()
    ring_rank = parallel_config.ring_rank()
    cp_size = parallel_config.cp_size()
    cp_rank = parallel_config.cp_rank()

    # Calculate total splits and current split index
    total_splits = ulysses_size * ring_size * cp_size
    assert total_splits == parallel_config.sp_size(), (
        "Total splits do not match sequence parallel size"
    )
    split_idx = cp_size * ulysses_size * ring_rank + cp_size * ulysses_rank + cp_rank

    if not allow_uneven:
        assert x.shape[dim] % total_splits == 0, "Sequence length must be divisible by total splits"

    seq_len = x.shape[dim]
    seq_len_padded = int(math.ceil(seq_len / total_splits)) * total_splits
    # Split hidden states along sequence dimension
    logger.debug(
        f"Split hidden states: {x.shape}, total_splits={total_splits}, split_dim={dim}, chunk_idx={split_idx}"
    )

    if seq_len_padded > seq_len:
        pad_size = [x.size(i) for i in range(x.ndim)]
        pad_size[dim] = seq_len_padded - seq_len
        x = torch.cat([x, x.new_zeros(*pad_size)], dim=dim).contiguous()

        seq_len_cur_rank = int(math.ceil(seq_len / total_splits))
        if split_idx + 1 == total_splits:
            seq_len_cur_rank = int(math.ceil(seq_len / total_splits)) - (seq_len_padded - seq_len)
        if PipelineConfig.seq_len is None:
            PipelineConfig.set_uneven_cp_config(
                seq_len, seq_len_padded, seq_len_cur_rank, parallel_config
            )

    x_splits = torch.chunk(x, total_splits, dim=dim)
    return x_splits[split_idx]


def _all_gather(
    x: torch.Tensor, group_size: int, group: torch.distributed.ProcessGroup, cat_dim: int
) -> torch.Tensor:
    x_gathered = torch.empty(group_size, *x.shape, device=x.device, dtype=x.dtype)
    work = torch.distributed.all_gather_into_tensor(x_gathered, x, group=group, async_op=True)
    work.wait()
    # [group_size, a, b, ..., cat_dim, c, d, ...] -> [a, b, ..., group_size, cat_dim, c, d, ...]
    if cat_dim != 0:
        permute_dim = [i + 1 for i in range(cat_dim)]
        permute_dim.append(0)  # group_size
        permute_dim.extend([i + 1 for i in range(cat_dim, x.ndim)])
        x_gathered = x_gathered.permute(permute_dim)

        x_shape = list(x.shape)
        new_shape = list()
        new_shape.extend([x_shape[i] for i in range(cat_dim)])
        new_shape.append(-1)  # group_size * shape[cat_dim]
        new_shape.extend([x_shape[i] for i in range(cat_dim + 1, x.ndim)])
        x = x_gathered.reshape(*new_shape).contiguous()
    else:
        x = x_gathered.reshape(-1, *x.shape[1:]).contiguous()
    return x


def _dit_sp_gather(parallel_config, hidden_states: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if parallel_config.sp_size() == 1:
        return hidden_states

    # Get parallel config parameters
    ulysses_size = parallel_config.ulysses_size()
    ring_size = parallel_config.ring_size()
    cp_size = parallel_config.cp_size()
    hidden_states = hidden_states.contiguous()
    # Gather from cp group
    if cp_size > 1:
        logger.debug(
            f"Gather hidden states from cp group: {hidden_states.shape}, cp_size={cp_size}"
        )
        hidden_states = _all_gather(hidden_states, cp_size, parallel_config.cp_group(), dim)

    # Gather from ulysses group
    if ulysses_size > 1:
        logger.debug(
            f"Gather hidden states from ulysses group: {hidden_states.shape}, ulysses_size={ulysses_size}"
        )
        hidden_states = _all_gather(
            hidden_states, ulysses_size, parallel_config.ulysses_group(), dim
        )

    # Then gather from ring group
    if ring_size > 1:
        logger.debug(
            f"Gather hidden states from ring group: {hidden_states.shape}, ring_size={ring_size}"
        )
        hidden_states = _all_gather(hidden_states, ring_size, parallel_config.ring_group(), dim)
    # # todo: sync to make sure hidden_states are ready, it should be removed
    # torch.cuda.synchronize()
    hidden_states = hidden_states[:, : PipelineConfig.seq_len, :]

    return hidden_states


def dit_sp_split(
    x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], dim: int, allow_uneven: bool = True
) -> torch.Tensor:
    dit_parallel_config = get_dit_parallel_config()
    return _dit_sp_split(dit_parallel_config, x, dim, allow_uneven)


def dit_sp_gather(hidden_states: torch.Tensor, dim: int = 1) -> torch.Tensor:
    dit_parallel_config = get_dit_parallel_config()
    return _dit_sp_gather(dit_parallel_config, hidden_states, dim)


def _dit_dp_split(parallel_config, x: torch.Tensor, dim: int) -> torch.Tensor:
    if parallel_config.dp_size() == 1:
        return x

    dp_size = parallel_config.dp_size()
    dp_rank = parallel_config.dp_rank()

    assert x.shape[dim] % dp_size == 0, (
        f"batch size must be divisible by total splits, {x.shape[dim]} % {dp_size} != 0"
    )

    return torch.chunk(x, dp_size, dim=dim)[dp_rank]


def _dit_dp_gather(parallel_config, x: torch.Tensor, dim: int) -> torch.Tensor:
    if parallel_config.dp_size() == 1:
        return x

    dp_size = parallel_config.dp_size()

    x_gathered = [torch.empty_like(x) for _ in range(dp_size)]
    torch.distributed.all_gather(x_gathered, x, group=parallel_config.dp_group())

    return torch.cat(x_gathered, dim=dim)


def dit_dp_split(x: torch.Tensor, dim: int) -> torch.Tensor:
    dit_parallel_config = get_dit_parallel_config()
    return _dit_dp_split(dit_parallel_config, x, dim)


def dit_dp_gather(x: torch.Tensor, dim: int) -> torch.Tensor:
    dit_parallel_config = get_dit_parallel_config()
    return _dit_dp_gather(dit_parallel_config, x, dim)


def enable_tensor_parallel(model: torch.nn.Module) -> torch.nn.Module:
    dit_parallel_config = get_dit_parallel_config()
    if dit_parallel_config.tp_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dit_parallel_config.tp_rank()]
        )
    return model
