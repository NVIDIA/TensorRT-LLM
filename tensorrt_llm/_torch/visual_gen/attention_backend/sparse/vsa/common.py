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

"""VisualGen Video Sparse Attention (VSA) algorithm and metadata.

The public tensor contract in this module is compact BSHD. The hierarchical
algorithm combines dense attention between mean-pooled 4x4x4 token cubes with
a backend-provided block-sparse fine stage. When a fine-stage implementation
cannot run, dense SDPA provides the existing functional fallback.
"""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from typing import Iterator, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F

# A 4x4x4 cube is one 64-token sparse block for every VSA fine-stage backend.
VSA_TILE_SIZE: Tuple[int, int, int] = (4, 4, 4)
VSA_BLOCK_SIZE = VSA_TILE_SIZE[0] * VSA_TILE_SIZE[1] * VSA_TILE_SIZE[2]


def _get_tile_partition_indices(
    dit_seq_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    time, height, width = dit_seq_shape
    tile_time, tile_height, tile_width = tile_size
    num_time = ceil(time / tile_time)
    num_height = ceil(height / tile_height)
    num_width = ceil(width / tile_width)

    block_time = torch.arange(num_time, device=device).view(num_time, 1, 1, 1, 1, 1)
    block_height = torch.arange(num_height, device=device).view(1, num_height, 1, 1, 1, 1)
    block_width = torch.arange(num_width, device=device).view(1, 1, num_width, 1, 1, 1)
    local_time = torch.arange(tile_time, device=device).view(1, 1, 1, tile_time, 1, 1)
    local_height = torch.arange(tile_height, device=device).view(1, 1, 1, 1, tile_height, 1)
    local_width = torch.arange(tile_width, device=device).view(1, 1, 1, 1, 1, tile_width)

    global_time = block_time * tile_time + local_time
    global_height = block_height * tile_height + local_height
    global_width = block_width * tile_width + local_width
    valid = (global_time < time) & (global_height < height) & (global_width < width)
    flat = global_time * (height * width) + global_height * width + global_width
    indices = torch.where(valid, flat, torch.full_like(flat, -1))
    return indices.reshape(-1).to(torch.long)


def _construct_variable_block_sizes(
    dit_seq_shape: Tuple[int, int, int],
    num_tiles: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    time, height, width = dit_seq_shape
    tile_time, tile_height, tile_width = tile_size
    num_time, num_height, num_width = num_tiles

    block_time = torch.arange(num_time, device=device)
    block_height = torch.arange(num_height, device=device)
    block_width = torch.arange(num_width, device=device)
    valid_time = (time - block_time * tile_time).clamp(max=tile_time)
    valid_height = (height - block_height * tile_height).clamp(max=tile_height)
    valid_width = (width - block_width * tile_width).clamp(max=tile_width)
    sizes = (
        valid_time.view(num_time, 1, 1)
        * valid_height.view(1, num_height, 1)
        * valid_width.view(1, 1, num_width)
    )
    return sizes.reshape(-1).to(torch.long)


@dataclass(frozen=True, slots=True)
class VSAMetadata:
    """Shape-dependent metadata required by the VSA sparse path."""

    num_cubes: int
    padded_seq_length: int
    variable_block_sizes: torch.LongTensor
    kv_token_mask: torch.BoolTensor
    non_pad_index: torch.LongTensor
    gather_idx: torch.LongTensor
    untile_idx: torch.LongTensor


class VSAMetadataBuilder:
    """Build VSA metadata while caching shape-dependent index tensors."""

    def __init__(self) -> None:
        self._cache: dict[Tuple[Tuple[int, int, int], torch.device], VSAMetadata] = {}

    def _build_metadata(
        self,
        dit_seq_shape: Tuple[int, int, int],
        device: torch.device,
    ) -> VSAMetadata:
        time, height, width = dit_seq_shape
        tile_time, tile_height, tile_width = VSA_TILE_SIZE
        num_tiles = (
            ceil(time / tile_time),
            ceil(height / tile_height),
            ceil(width / tile_width),
        )
        total_seq_length = time * height * width
        padded_seq_length = (
            num_tiles[0] * num_tiles[1] * num_tiles[2] * tile_time * tile_height * tile_width
        )
        num_cubes = num_tiles[0] * num_tiles[1] * num_tiles[2]
        tokens_per_cube = VSA_BLOCK_SIZE

        tile_partition_indices = _get_tile_partition_indices(dit_seq_shape, VSA_TILE_SIZE, device)
        gather_idx = tile_partition_indices[tile_partition_indices >= 0]

        variable_block_sizes = _construct_variable_block_sizes(
            dit_seq_shape, num_tiles, VSA_TILE_SIZE, device
        )
        local_offsets = torch.arange(tokens_per_cube, device=device).expand(
            num_cubes, tokens_per_cube
        )
        cube_offsets = torch.arange(num_cubes, device=device).unsqueeze(1) * tokens_per_cube
        non_pad_index = (cube_offsets + local_offsets)[
            local_offsets < variable_block_sizes.unsqueeze(1)
        ]

        untile_idx = torch.empty(total_seq_length, dtype=torch.long, device=device)
        untile_idx[gather_idx] = non_pad_index

        kv_token_mask = torch.zeros(padded_seq_length, dtype=torch.bool, device=device)
        kv_token_mask[non_pad_index] = True

        return VSAMetadata(
            num_cubes=num_cubes,
            padded_seq_length=padded_seq_length,
            variable_block_sizes=variable_block_sizes,
            kv_token_mask=kv_token_mask,
            non_pad_index=non_pad_index,
            gather_idx=gather_idx,
            untile_idx=untile_idx,
        )

    def build(
        self,
        raw_latent_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        device: torch.device,
    ) -> VSAMetadata:
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )
        cache_key = (dit_seq_shape, device)
        metadata = self._cache.get(cache_key)
        if metadata is None:
            metadata = self._build_metadata(dit_seq_shape, device)
            self._cache[cache_key] = metadata

        return metadata

    def clear(self) -> None:
        """Release cached tensors after CUDA Graphs that reference them are cleared."""

        self._cache.clear()


_vsa_forward_context_var: contextvars.ContextVar[Optional[VSAMetadata]] = contextvars.ContextVar(
    "_vsa_forward_context", default=None
)


@contextmanager
def set_vsa_forward_context(metadata: VSAMetadata) -> Iterator[None]:
    """Make VSA metadata visible to attention layers for one model forward."""

    token = _vsa_forward_context_var.set(metadata)
    try:
        yield
    finally:
        _vsa_forward_context_var.reset(token)


def get_vsa_forward_context() -> Optional[VSAMetadata]:
    """Return the metadata for the active VSA model forward, if any."""

    return _vsa_forward_context_var.get(None)


def _mean_pool_cubes(
    x_tiled: torch.Tensor,
    variable_block_sizes: torch.LongTensor,
    prod_tile: int,
    num_cubes: int,
) -> torch.Tensor:
    batch_size, _padded, num_heads, head_dim = x_tiled.shape
    x_cubes = x_tiled.view(batch_size, num_cubes, prod_tile, num_heads, head_dim)
    # FP32 accumulation avoids perturbing the coarse softmax when inputs are BF16.
    x_sum = x_cubes.float().sum(dim=2)
    valid_counts = variable_block_sizes.float().clamp(min=1).view(1, num_cubes, 1, 1)
    return (x_sum / valid_counts).to(x_tiled.dtype)


class VSAPreprocessor:
    """Convert compact BSHD tensors between sequence-major and tile-major order."""

    @staticmethod
    def tile(
        x: torch.Tensor,
        non_pad_index: torch.LongTensor,
        gather_idx: torch.LongTensor,
        padded_seq_len: int,
    ) -> torch.Tensor:
        # index_select + index_copy_ keeps this path traceable by torch.compile.
        batch_size, _seq_len, num_heads, head_dim = x.shape
        x_valid = x.index_select(1, gather_idx)
        x_padded = x.new_zeros(batch_size, padded_seq_len, num_heads, head_dim)
        x_padded.index_copy_(1, non_pad_index, x_valid)
        return x_padded

    @staticmethod
    def untile(
        x: torch.Tensor,
        untile_idx: torch.LongTensor,
    ) -> torch.Tensor:
        return torch.index_select(x, 1, untile_idx)


class VSAFineStage(Protocol):
    """Backend capability consumed by the VSA fine stage."""

    def __call__(
        self,
        q_tiled: torch.Tensor,
        k_tiled: torch.Tensor,
        v_tiled: torch.Tensor,
        topk_indices: torch.Tensor,
        variable_block_sizes: torch.LongTensor,
        kv_token_mask: torch.BoolTensor,
        cur_topk: int,
        num_cubes: int,
    ) -> Optional[torch.Tensor]: ...


class VSAAlgorithm:
    """Composable hierarchical VSA strategy over compact BSHD tensors.

    A VisualGen backend supplies the fine-stage capability to :meth:`forward`.
    Returning ``None`` from that capability selects the common dense SDPA
    fallback. Callers must install a :class:`VSAMetadata` context around each
    model forward.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        *,
        vsa_sparsity: float,
    ) -> None:
        resolved_num_kv_heads = num_kv_heads or num_heads
        if resolved_num_kv_heads != num_heads:
            raise ValueError(
                "VSA coarse mean-pool assumes MHA (num_kv_heads == num_heads), "
                f"got num_kv_heads={resolved_num_kv_heads}, num_heads={num_heads}. "
                "GQA/MQA is not supported."
            )
        self.vsa_sparsity = vsa_sparsity

    @torch.compiler.disable
    def _get_vsa_metadata(self) -> VSAMetadata:
        metadata = get_vsa_forward_context()
        if metadata is None:
            raise RuntimeError(
                "VSAAlgorithm.forward called without an active VSA forward context. "
                "Wrap each transformer call with set_vsa_forward_context()."
            )
        return metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        fine_stage: VSAFineStage,
        gate_compress: Optional[torch.Tensor] = None,
        gate_fine: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run hierarchical VSA on compact BSHD Q/K/V tensors."""

        del kwargs
        if gate_compress is None:
            raise ValueError(
                "VSA requires gate_compress. "
                "Ensure to_gate_compress is wired in the transformer block."
            )
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("VSA requires Q, K, and V with the same BSHD shape.")
        if any(tensor.device != q.device or tensor.dtype != q.dtype for tensor in (k, v)):
            raise ValueError("VSA requires Q, K, and V to share device and dtype.")
        if not isinstance(gate_compress, torch.Tensor):
            raise TypeError("VSA gate_compress must be a torch.Tensor.")
        if gate_compress.shape != q.shape or gate_compress.device != q.device:
            raise ValueError("VSA gate_compress must share Q's shape and device.")
        if gate_compress.dtype != q.dtype:
            raise ValueError("VSA gate_compress must share Q's dtype.")
        if gate_fine is not None and (
            not isinstance(gate_fine, torch.Tensor)
            or gate_fine.shape != q.shape
            or gate_fine.device != q.device
            or gate_fine.dtype != q.dtype
        ):
            raise ValueError("VSA gate_fine must share Q's shape, device, and dtype.")

        metadata = self._get_vsa_metadata()
        non_pad_index = metadata.non_pad_index
        gather_idx = metadata.gather_idx
        untile_idx = metadata.untile_idx
        variable_block_sizes = metadata.variable_block_sizes
        kv_token_mask = metadata.kv_token_mask
        padded_len = metadata.padded_seq_length
        num_cubes = metadata.num_cubes

        batch_size, _seq_len, num_heads, head_dim = q.shape
        prod_tile = VSA_BLOCK_SIZE
        cur_topk = max(1, ceil((1.0 - self.vsa_sparsity) * num_cubes))

        q_tiled = VSAPreprocessor.tile(q, non_pad_index, gather_idx, padded_len)
        k_tiled = VSAPreprocessor.tile(k, non_pad_index, gather_idx, padded_len)
        v_tiled = VSAPreprocessor.tile(v, non_pad_index, gather_idx, padded_len)

        q_coarse = _mean_pool_cubes(q_tiled, variable_block_sizes, prod_tile, num_cubes)
        k_coarse = _mean_pool_cubes(k_tiled, variable_block_sizes, prod_tile, num_cubes)
        v_coarse = _mean_pool_cubes(v_tiled, variable_block_sizes, prod_tile, num_cubes)

        scale = head_dim**-0.5
        coarse_scores = torch.einsum("bnhd,bmhd->bhnm", q_coarse, k_coarse) * scale
        coarse_probs = coarse_scores.softmax(dim=-1)
        coarse_output = torch.einsum("bhnm,bmhd->bnhd", coarse_probs, v_coarse)
        topk_indices = coarse_probs.topk(cur_topk, dim=-1).indices.to(torch.int32)

        coarse_output_tiled = (
            coarse_output.unsqueeze(2)
            .expand(
                batch_size,
                num_cubes,
                prod_tile,
                num_heads,
                head_dim,
            )
            .reshape(
                batch_size,
                padded_len,
                num_heads,
                head_dim,
            )
        )

        fine_output_tiled = fine_stage(
            q_tiled,
            k_tiled,
            v_tiled,
            topk_indices,
            variable_block_sizes,
            kv_token_mask,
            cur_topk,
            num_cubes,
        )
        if fine_output_tiled is not None:
            gate_compress_tiled = VSAPreprocessor.tile(
                gate_compress, non_pad_index, gather_idx, padded_len
            )
            if gate_fine is not None:
                gate_fine_tiled = VSAPreprocessor.tile(
                    gate_fine, non_pad_index, gather_idx, padded_len
                )
                combined_tiled = (
                    gate_compress_tiled * coarse_output_tiled + gate_fine_tiled * fine_output_tiled
                )
            else:
                combined_tiled = gate_compress_tiled * coarse_output_tiled + fine_output_tiled
            return VSAPreprocessor.untile(combined_tiled, untile_idx)

        coarse_output_full = VSAPreprocessor.untile(
            coarse_output_tiled,
            untile_idx,
        )
        fine_output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        ).transpose(1, 2)
        if gate_fine is not None:
            output = gate_compress * coarse_output_full + gate_fine * fine_output
        else:
            output = gate_compress * coarse_output_full + fine_output
        return output


__all__ = [
    "VSA_TILE_SIZE",
    "VSAAlgorithm",
    "VSAFineStage",
    "VSAMetadata",
    "VSAMetadataBuilder",
    "VSAPreprocessor",
    "get_vsa_forward_context",
    "set_vsa_forward_context",
]
