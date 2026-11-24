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
"""The MLA DSpark drafter's helix layer on the target's exchange, with fewer launches.

Under helix the MLA drafter keeps one latent per token, so its context pages are sharded over the
whole repurposed group (``Mapping.helix_mapping_for_kv_heads(1)``): every rank attends all heads
over its pages and the per-rank partials are merged by a helix exchange. Switches (environment,
default off; each falls back to the current path where it does not apply):

* ``TRTLLM_K3_ULL2_DRAFT_A2A=1``: the drafter's helix mapping takes the cp_config of the target's
  helix mapping (``inherit_cp_config``), so its exchange runs on the target's transport, e.g. the
  MNNVL FIFO all-to-all + ``helix_post_process_native`` for ``use_nccl_for_alltoall: false``,
  instead of the NCCL all-to-all with its torch reformats. Same values: both combines run the
  same arithmetic on the same records.
* ``TRTLLM_K3_ULL2_DRAFT_TAIL=1`` (with ``_A2A`` and an MNNVL cp_config): the V up-projection, the
  exchange and the combine of a layer are one launch of the Kimi-K3 helix tail kernel
  (``k3_helix_tail``, rows copied, no fold), on the receive buffers of the target's tail when the
  mappings match. The V up-projection sums in another FP32 order than the cuBLAS BMM: drafter
  rounding only.
* ``TRTLLM_K3_ULL2_DRAFT_STORE=1`` (with ``TLLM_DSPARK_TAIL_V2=1``): under helix too, one kernel
  rotates the query's rope half, normalizes and rotates the block latents into a scratch row
  buffer, and one masked scatter stores the rows this rank's pages hold
  (``scatter_rows_to_pool(rows_mask=owned)``), instead of the separate RoPE launches, an FP8 cast
  and a gather / where / index_put round trip of the pool; the FP8 query of the decode is cast in
  one launch (``to_draft_fp8_1k``) instead of a clamp and a cast. Same values.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..features import active as _features


def draft_helix_a2a_enabled() -> bool:
    """Whether the drafter's helix mapping inherits the target's cp_config."""
    return _features().ull2_draft_a2a


def draft_helix_tail_enabled() -> bool:
    """Whether the drafter runs its helix V up-projection + exchange + combine as one kernel."""
    return draft_helix_a2a_enabled() and _features().ull2_draft_tail


def draft_helix_store_enabled() -> bool:
    """Whether a helix drafter takes the fused block RoPE with an owner-masked pool store."""
    return _features().ull2_draft_store


@triton.jit
def _draft_fp8_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    v = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
    # clamp(-448, 448) keeping NaN (min / max would drop it), then round to nearest even.
    clamped = tl.minimum(tl.maximum(v, -448.0), 448.0)
    v = tl.where(v != v, v, clamped)
    tl.store(dst_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def to_draft_fp8_1k(x: torch.Tensor) -> torch.Tensor:
    """``to_draft_fp8(x)`` (saturated E4M3 at unit scale) of a contiguous tensor in one launch."""
    assert x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    numel = x.numel()
    if numel:
        block = 4096
        _draft_fp8_kernel[(triton.cdiv(numel, block),)](x, out, numel, BLOCK=block, num_warps=4)
    return out


def draft_helix_tail(
    mapping: Mapping,
    rows: torch.Tensor,
    stats: torch.Tensor,
    w_uv: torch.Tensor,
    zero_kv_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """One drafter layer's helix V up-projection, exchange and combine in one launch, or None.

    Args:
        mapping: The drafter's helix mapping (tp 1, cp = the repurposed group).
        rows: BF16 ``[T, H, kv_lora_rank]`` (or any view with T * H contiguous rows) latent
            attention output of this rank's pages, every head.
        stats: FP32 ``[T, H, 2]`` ``(max, sum)`` of ``rows``, as ``_helix_post_process`` takes them.
        w_uv: BF16 ``[H, v_head_dim, kv_lora_rank]`` V up-projection.
        zero_kv_mask: Optional bool ``[T]``, True where this rank holds no key of the token.

    Returns:
        BF16 ``[T, H / cp * v_head_dim]``: this rank's heads after the merge, what the BMM +
        ``_helix_post_process`` return; None when the switch is off or the call does not fit the
        kernel (the caller keeps that path). The decision depends only on shapes and settings, so
        every rank takes the same branch.
    """
    if not draft_helix_tail_enabled() or torch.compiler.is_compiling():
        return None
    if not mapping.has_cp_helix() or mapping.cp_config.get("use_nccl_for_alltoall", True):
        return None
    if get_sm_version() not in (100, 103, 107):
        return None
    from ..kernels import k3_helix_tail as kernel
    from ..mla.helix_tail import K3HelixTail

    if w_uv.dim() != 3 or rows.dtype != torch.bfloat16 or w_uv.dtype != torch.bfloat16:
        return None
    num_heads, v_dim, latent_dim = w_uv.shape
    cp_size = mapping.cp_size
    if not kernel.can_run(cp_size, num_heads, latent_dim, v_dim):
        return None
    if not rows.is_contiguous() or not stats.is_contiguous() or not w_uv.is_contiguous():
        return None
    num_tokens = rows.numel() // (num_heads * latent_dim)
    if num_tokens == 0 or stats.numel() != num_tokens * num_heads * 2:
        return None
    tail = K3HelixTail._cache.get(mapping)
    if tail is None:
        if torch.cuda.is_current_stream_capturing():
            # The receive buffer is allocated collectively, outside graph capture; an eager
            # forward on every rank creates it first.
            logger.warning_once(
                "DSpark drafter helix tail: no receive buffer before graph capture, the "
                "captured drafter keeps the separate exchange.",
                key="dspark_ull2_tail_capture",
            )
            return None
        tail = K3HelixTail.get(mapping)
    if num_tokens * cp_size * kernel.record_bytes(num_heads // cp_size, v_dim) > tail.half_bytes:
        return None
    return tail(
        rows.reshape(num_tokens, num_heads, latent_dim),
        stats.reshape(num_tokens, num_heads, 2),
        w_uv,
        zero_kv_mask,
    )
