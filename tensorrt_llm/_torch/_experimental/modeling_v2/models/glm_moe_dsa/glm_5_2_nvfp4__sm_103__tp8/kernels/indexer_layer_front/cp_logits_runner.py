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
"""Compile cache and launcher of the indexer_layer_front context-partitioned logits kernel.

Same configuration, arguments, output arena and values as cross_rank_topk's
``indexer_topk/cross_rank_logits_runner.py``; the kernel
(``indexer_layer_front/fp8_paged_mqa_logits_cp.py``) waits for its programmatic primary before any
global access and is launched with programmatic dependent launch when ``TRTLLM_ENABLE_PDL`` is on
(it then starts during the generation rope, which triggers right after its last read).
"""

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

from ..indexer_topk.cross_rank_logits_runner import (
    COMPUTE_BLOCK_KV,
    LOGITS_BUFFER_NAME,
    NUM_W_IN_REG,
    SPLIT_KV,
    TRIGGER_DEPENDENTS,
    local_capacity,
)
from .fp8_paged_mqa_logits_cp import FP8MQALogitsCPZ14Kernel

_KERNEL_CACHE: Dict[Tuple, object] = {}
_NUM_SMS: Dict[int, int] = {}
_UNUSED_SCHEDULE: Dict[int, torch.Tensor] = {}


def use_pdl() -> bool:
    """Whether the kernel is launched as a programmatic dependent (TensorRT-LLM's PDL switch)."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return bool(get_env_enable_pdl())


def _num_sms(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _NUM_SMS:
        _NUM_SMS[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _NUM_SMS[index]


def _unused_schedule(device: torch.device) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    t = _UNUSED_SCHEDULE.get(index)
    if t is None:
        t = torch.zeros(1, 2, dtype=torch.int32, device=device)
        _UNUSED_SCHEDULE[index] = t
    return t


def _compile(
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    num_sms: int,
    world: int,
    wait_at_entry: bool,
    pdl: bool,
):
    n_cols = next_n * num_heads
    block_bytes = phys_block_kv * (head_dim + 4)
    sym_num_phys_blocks = cute.sym_int()
    sym_b = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks_per_seq = cute.sym_int()
    num_ctas = cute.sym_int()
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (sym_num_phys_blocks, block_bytes), stride=(cute.sym_int64(), 1)
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n_cols, head_dim, sym_b), stride_order=(1, 0, 2)
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (n_cols, sym_b), stride_order=(0, 1)
    )
    logits_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32, (cute.sym_int(), max_ctx), stride=(cute.sym_int64(), 1)
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_b, max_blocks_per_seq), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_b,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = FP8MQALogitsCPZ14Kernel(
        block_kv=COMPUTE_BLOCK_KV,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_sms=num_sms,
        num_w_in_reg=NUM_W_IN_REG,
        world=world,
        trigger_dependents=TRIGGER_DEPENDENTS,
        wait_at_entry=wait_at_entry,
        use_pdl=pdl,
    )
    return cute.compile(
        kernel,
        kv_fake,
        q_fake,
        w_fake,
        logits_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(0),
        fake_stream,
        options="--enable-tvm-ffi",
    )


def _key(
    device: torch.device,
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    world: int,
    wait_at_entry: bool,
    pdl: bool,
) -> Tuple:
    return (phys_block_kv, num_heads, head_dim, next_n, _num_sms(device), world, wait_at_entry, pdl)


def ensure_compiled(
    device: torch.device,
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    world: int,
    wait_at_entry: bool = True,
    pdl: bool = None,
) -> None:
    """Compile the kernel for this shape now (host only, no device work)."""
    pdl = use_pdl() if pdl is None else pdl
    key = _key(device, phys_block_kv, num_heads, head_dim, next_n, world, wait_at_entry, pdl)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _compile(*key)


def paged_mqa_logits_cp(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    kv_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
    rank: int,
    world: int,
    wait_at_entry: bool = True,
    pdl: bool = None,
) -> torch.Tensor:
    """cross_rank_topk's ``paged_mqa_logits_cp`` with the indexer_layer_front kernel (same arguments, arena and values).

    ``wait_at_entry`` / ``pdl`` default to the served indexer_layer_front configuration (wait
    at entry, PDL launch when TensorRT-LLM's PDL switch is on); tests pass
    other values.
    """
    batch, next_n, num_heads, head_dim = q.shape
    assert batch == 1
    pdl = use_pdl() if pdl is None else pdl
    phys_block_kv = kv_fused.shape[1]
    num_phys_blocks = kv_fused.shape[0]
    width = local_capacity(max_seq_len, world)
    aligned = (width + SPLIT_KV - 1) // SPLIT_KV * SPLIT_KV
    logits = get_memory_buffers().get_buffer(
        [next_n, aligned],
        torch.float32,
        buffer_name=LOGITS_BUFFER_NAME,
        reserve_buffer=torch.cuda.is_current_stream_capturing(),
    )[:, :width]
    key = _key(q.device, phys_block_kv, num_heads, head_dim, next_n, world, wait_at_entry, pdl)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        compiled = _compile(*key)
        _KERNEL_CACHE[key] = compiled
    q_3d = q.reshape(batch, next_n * num_heads, head_dim).permute(1, 2, 0)
    w_2d = weights.reshape(batch, next_n * num_heads).t()
    compiled(
        kv_fused.reshape(num_phys_blocks, -1),
        q_3d.view(torch.uint8),
        w_2d,
        logits,
        block_table,
        kv_lens,
        _unused_schedule(q.device),
        num_phys_blocks,
        batch,
        rank,
    )
    return logits
