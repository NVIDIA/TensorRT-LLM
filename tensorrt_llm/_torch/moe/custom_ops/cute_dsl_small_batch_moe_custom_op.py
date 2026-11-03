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
"""``trtllm::cute_dsl_nvfp4_small_batch_moe_rubin``: the small-batch NVFP4 routed-experts kernel.

Wraps :class:`RubinSmallBatchFusedMoeKernel` (FC1 + SiTU + NVFP4 requantization + FC2 + combine
of at most 32 tokens in one persistent grid). The caller passes the tile-128 ``moe_sort`` tables
and a counter buffer from :func:`new_small_batch_moe_counters` that it keeps for the layer (every
launch leaves it zero again).
"""

import functools

import torch

from tensorrt_llm._utils import get_sm_version

__all__ = [
    "SMALL_BATCH_MOE_MAX_TOKENS",
    "die_major_layout",
    "new_small_batch_moe_counters",
    "small_batch_moe_route_word",
    "small_batch_moe_supports",
]

#: Largest token count of the kernel (MMA N 32).
SMALL_BATCH_MOE_MAX_TOKENS = 32
#: Tiles the counter buffer holds (moe_sort's padded rows / 128 at <= 32 tokens).
_MAX_TILES = 1024
#: Persistent CTAs per token tile by SM version (others: one per SM). On Rubin the grid leaves
#: SMs free for the shared expert, which runs on a concurrent stream.
_PERSISTENT_CTAS = {107: {8: 204, 16: 196, 32: 180}}
#: (claim_free, max_slices) of the expert-parallel (permuted) layout and of the token-indexed
#: layout of MoE-TP slices. Small intermediates make FC2 tasks short, so the token-indexed schedule
#: claims an FC2 task only once 4 ring stages are free for it and splits the tiles into more
#: slices.
_PERMUTED_SCHEDULE = (0, 16)
_TOKEN_INDEXED_SCHEDULE = (4, 24)
#: Ring stage cap: die-major weights stream from the SM's own domain and keep more stages busy.
_RING_STAGES = 8
_DIE_MAJOR_RING_STAGES = 12


def small_batch_moe_supports(intermediate_size: int, hidden_size: int) -> bool:
    """Whether the kernel serves experts of this per-rank shape (NVFP4, gated FC1)."""
    from ..kernels.rubin.rubin_small_batch_fused_moe import RubinSmallBatchFusedMoeKernel

    # whole 256-wide FC2 k-tiles: the FC1-output scale buffer then needs no zero padding
    return intermediate_size % 256 == 0 and RubinSmallBatchFusedMoeKernel.supports(
        SMALL_BATCH_MOE_MAX_TOKENS, 2 * intermediate_size, hidden_size
    )


def die_major_layout(t: torch.Tensor) -> torch.Tensor:
    """``[L, R, C]`` expert weights or block scales -> ``[2 L, R / 2, C]``: slot e holds rows
    ``[0, R / 2)`` of expert e, slot L + e rows ``[R / 2, R)``. The halves keep whole 128-row scale
    blocks, so the 128x4-blocked scales split the same way."""
    rows = t.size(1)
    assert rows % 256 == 0, "die-major halves need whole 128-row blocks"
    return torch.cat([t[:, : rows // 2], t[:, rows // 2 :]], dim=0).contiguous()


def new_small_batch_moe_counters(device: torch.device) -> torch.Tensor:
    """A zeroed counter buffer for one caller."""
    from ..kernels.rubin.rubin_small_batch_fused_moe import counter_words

    return torch.zeros(counter_words(_MAX_TILES), dtype=torch.int32, device=device)


def small_batch_moe_route_word(counters: torch.Tensor) -> torch.Tensor:
    """The int32 word of ``counters`` into which a routing kernel publishes the step's tokens
    for a launch with ``published_routing``."""
    from ..kernels.rubin.rubin_small_batch_fused_moe import CTR_ROUTE

    return counters[CTR_ROUTE : CTR_ROUTE + 1]


@functools.cache
def _persistent_ctas(device_index: int, n_tok: int) -> int:
    num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    return min(_PERSISTENT_CTAS.get(get_sm_version(), {}).get(n_tok, num_sms), num_sms)


class _SmallBatchMoeRunner:
    """Compiles (once per configuration) and launches :class:`RubinSmallBatchFusedMoeKernel`."""

    kernel_cache: dict = {}

    @classmethod
    def run(
        cls,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        alpha1: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        alpha2: torch.Tensor,
        out: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        permuted_idx_to_expanded_idx: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        global_sf: torch.Tensor,
        token_final_scales: torch.Tensor,
        counters: torch.Tensor,
        top_k: int,
        situ_beta: float,
        situ_linear_beta: float,
        token_indexed: bool,
        num_ctas: int,
        deterministic_combine: bool,
        die_major: bool,
        published_routing: bool,
        expert_offset: int,
    ) -> None:
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        from ..kernels.rubin.rubin_small_batch_fused_moe import (
            MAX_SLICES,
            RubinSmallBatchFusedMoeKernel,
            counter_words,
            n_tok_for,
            pack_sm_domain_words,
            workspace_words,
        )
        from ..kernels.rubin.utils import TRTLLM_ENABLE_PDL, make_ptr

        orig_m = x.size(0)
        num_local, n1, k1 = w1.size(0), w1.size(1), w2.size(1)
        if die_major:
            num_local, n1, k1 = num_local // 2, 2 * n1, 2 * k1
        sm_domain_words = ()
        if die_major:
            from ...locality_domain.placement import sm_locality_domains

            sm_domain_words = pack_sm_domain_words(sm_locality_domains(x.device.index))
        k2 = n1 // 2
        assert k2 % 256 == 0, "the FC1-output scale buffer is not zero-padded"
        m = permuted_idx_to_expanded_idx.size(0)
        if published_routing:
            from ..kernels.blackwell.kimi_k3_routing import sort_buffer_sizes

            m = sort_buffer_sizes(orig_m, num_local)[1]
        assert m // 128 <= _MAX_TILES and counters.numel() >= counter_words(m // 128)
        assert expanded_idx_to_permuted_idx.numel() == orig_m * top_k
        n_tok = n_tok_for(orig_m)
        if num_ctas <= 0:
            num_ctas = _persistent_ctas(x.device.index, n_tok)
        h = torch.empty(m, k2 // 2, dtype=torch.uint8, device=x.device)
        h_sf = torch.empty(
            (m // 128) * RubinSmallBatchFusedMoeKernel.h_sf_tile_bytes(n1),
            dtype=torch.uint8,
            device=x.device,
        )
        ws = h_sf  # the red combine reads no workspace
        if deterministic_combine:
            ws = torch.empty(
                workspace_words(k1, SMALL_BATCH_MOE_MAX_TOKENS, MAX_SLICES),
                dtype=torch.int32,
                device=x.device,
            )

        def ptr(dtype, t, align=None):
            if align is None:
                return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem)
            return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

        ptrs = [
            ptr(cutlass.Float4E2M1FN, x, 32),
            ptr(cutlass.Float4E2M1FN, w1, 32),
            ptr(cutlass.Float8E4M3FN, x_sf, 16),
            ptr(cutlass.Float8E4M3FN, w1_sf, 16),
            ptr(cutlass.Float4E2M1FN, h, 32),
            ptr(cutlass.Float8E4M3FN, h_sf, 16),
            ptr(cutlass.Float32, alpha1),
            ptr(cutlass.Float4E2M1FN, w2, 32),
            ptr(cutlass.Float8E4M3FN, w2_sf, 16),
            ptr(cutlass.Float32, alpha2),
            ptr(cutlass.BFloat16, out, 16),
            ptr(cutlass.Int32, tile_idx_to_group_idx),
            ptr(cutlass.Int32, tile_idx_to_mn_limit),
            ptr(cutlass.Int32, permuted_idx_to_expanded_idx),
            ptr(cutlass.Int32, num_non_exiting_tiles),
            ptr(cutlass.Float32, global_sf),
            ptr(cutlass.Float32, token_final_scales),
            ptr(cutlass.Int32, expanded_idx_to_permuted_idx),
            ptr(cutlass.Int32, counters),
            ptr(cutlass.Int32, ws, 16),
        ]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        key = (
            n_tok,
            n1,
            k1,
            top_k,
            situ_beta,
            situ_linear_beta,
            token_indexed,
            num_ctas,
            deterministic_combine,
            die_major,
            sm_domain_words,
            published_routing,
            TRTLLM_ENABLE_PDL,
        )
        compiled = cls.kernel_cache.get(key)
        if compiled is None:
            claim_free, max_slices = (
                _TOKEN_INDEXED_SCHEDULE if token_indexed else _PERMUTED_SCHEDULE
            )
            kernel = RubinSmallBatchFusedMoeKernel(
                n_tok,
                n1,
                k1,
                top_k,
                situ_beta,
                situ_linear_beta,
                token_indexed=token_indexed,
                claim_free=claim_free,
                max_slices=max_slices,
                max_ab_stage=_DIE_MAJOR_RING_STAGES if die_major else _RING_STAGES,
                deterministic_combine=deterministic_combine,
                use_pdl=TRTLLM_ENABLE_PDL,
                die_major=die_major,
                sm_domain_words=sm_domain_words,
                published_routing=published_routing,
            )
            compiled = cute.compile(
                kernel.wrapper, *ptrs, orig_m, m, num_local, expert_offset, num_ctas, stream
            )
            cls.kernel_cache[key] = compiled
        compiled(*ptrs, orig_m, m, num_local, expert_offset, stream)


@torch.library.custom_op(
    "trtllm::cute_dsl_nvfp4_small_batch_moe_rubin",
    mutates_args=("output", "counters"),
    device_types="cuda",
)
def cute_dsl_nvfp4_small_batch_moe_rubin(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc1_alpha: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    fc2_alpha: torch.Tensor,
    output: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    global_sf: torch.Tensor,
    token_final_scales: torch.Tensor,
    counters: torch.Tensor,
    top_k: int,
    situ_beta: float,
    situ_linear_beta: float,
    token_indexed: bool,
    num_ctas: int = 0,
    deterministic_combine: bool = False,
    die_major: bool = False,
    published_routing: bool = False,
    expert_offset: int = 0,
) -> None:
    """Routed NVFP4 SiTU MoE of at most 32 tokens: writes the sum of this rank's experts'
    weighted down-projections into every element of ``output`` [tokens, hidden].

    ``input`` / ``input_scale``: NVFP4 tokens and their linear scales; ``fc1_*`` / ``fc2_*``: the
    local experts' NVFP4 weights, 128x4-blocked scales and per-expert alphas (FC1 rows in the Rubin
    ``[gate16, up16]`` order); the six ``moe_sort`` tables of tile size 128; ``global_sf``: the FC2
    input scale; ``counters``: from :func:`new_small_batch_moe_counters`, zero again afterwards.
    ``token_indexed``: the token-indexed FC1 output layout (MoE-TP slices of the intermediate).
    ``num_ctas`` > 0 overrides the persistent CTA count. ``deterministic_combine``: the
    schedule-independent slice combine instead of BF16 vector reds (a test reference).
    ``die_major``: the four weight tensors hold :func:`die_major_layout` bytes under their
    ``[L, rows, cols]`` shapes (see ``RubinSmallBatchFusedMoeKernel``). ``published_routing``
    (``token_indexed``): instead of the tables, ``expanded_idx_to_permuted_idx`` holds the top-k
    ids, which a routing kernel publishes into :func:`small_batch_moe_route_word` of ``counters``
    (the other table arguments are not read); ``expert_offset``: this rank's first expert.
    """
    if die_major:
        fc1_weight, fc1_weight_scale, fc2_weight, fc2_weight_scale = (
            t.view(2 * t.size(0), t.size(1) // 2, *t.shape[2:])
            for t in (fc1_weight, fc1_weight_scale, fc2_weight, fc2_weight_scale)
        )
    _SmallBatchMoeRunner.run(
        input,
        input_scale,
        fc1_weight,
        fc1_weight_scale,
        fc1_alpha,
        fc2_weight,
        fc2_weight_scale,
        fc2_alpha,
        output,
        tile_idx_to_group_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        expanded_idx_to_permuted_idx,
        num_non_exiting_tiles,
        global_sf,
        token_final_scales,
        counters,
        top_k,
        float(situ_beta),
        float(situ_linear_beta),
        token_indexed,
        num_ctas,
        deterministic_combine,
        die_major,
        published_routing,
        expert_offset,
    )


@torch.library.register_fake("trtllm::cute_dsl_nvfp4_small_batch_moe_rubin")
def _(
    input,
    input_scale,
    fc1_weight,
    fc1_weight_scale,
    fc1_alpha,
    fc2_weight,
    fc2_weight_scale,
    fc2_alpha,
    output,
    tile_idx_to_group_idx,
    tile_idx_to_mn_limit,
    permuted_idx_to_expanded_idx,
    expanded_idx_to_permuted_idx,
    num_non_exiting_tiles,
    global_sf,
    token_final_scales,
    counters,
    top_k,
    situ_beta,
    situ_linear_beta,
    token_indexed,
    num_ctas=0,
    deterministic_combine=False,
    die_major=False,
    published_routing=False,
    expert_offset=0,
) -> None:
    return None
