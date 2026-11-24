# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA over an NVFP4 latent cache, context rows: the selected latent keys into an FP8 scratch.

A context step's rows select up to Top-K keys each among their request's cached and new keys.
Over an NVFP4 cache the step dequantizes the union of those keys, each key once, into a compact
E4M3 scratch of its own and hands the sparse MLA the scratch row of every selected key. The row
format and the rounding are the generation gather's (``rubin_dsa_nvfp4_gather_fp8``).
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .rubin_dsa_nvfp4_gather_fp8 import LATENT_DIM, nvfp4_rows_to_e4m3

# The engine DSA backend's default global dequantization scale (amax 100 over 448 x 6).
_AMAX100 = 100.0 / (448.0 * 6.0)
# The largest pool row the op addresses (int32).
_INT32_MAX = 2**31 - 1


@functools.cache
def _trtllm_ops():
    """``torch.ops.trtllm`` with the compiled ops registered, on the first call."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*

    return torch.ops.trtllm


def scratch_rows(total_kv_tokens: int, rows: int, topk: int) -> int:
    """Rows of the scratch: the union holds at most ``min(total_kv_tokens, rows * topk)`` keys,
    and never fewer than ``topk`` rows, since the sparse attention may prefetch up to Top-K rows
    of a row whose context has fewer keys."""
    return max(min(total_kv_tokens, rows * topk), topk)


def _spec(cached, new, **overrides) -> dict:
    spec = dict(
        cached=cached,
        new=new,
        topk=2048,
        tokens_per_block=64,
        page_scale=1,
        layer_offset=0,
        scale=1.0,
        scales="normal",
        pools=1,
    )
    spec.update(overrides)
    return spec


class RubinDsaNvfp4CtxGatherFp8(OpWrapper):
    """Gather the keys a context step's rows select into a new E4M3 scratch; return ``(scratch,
    compact_indices)``.

    Row ``r`` belongs to context request ``q = query_req_indices[r]``, whose keys are the packed
    tokens ``cu_kv_lengths[q] + pos`` for ``0 <= pos < cu_kv_lengths[q + 1] - cu_kv_lengths[q]``.
    A column ``pos = local_topk_indices[r, c]`` selects that key when ``q`` is a request, ``pos``
    lies in it, the packed token is below ``total_kv_tokens``, its page ``block_table[q, pos //
    tokens_per_block]`` is mapped and its pool row ``page * page_stride + pos % tokens_per_block +
    layer_id * tokens_per_block`` lies in ``[0, num_pool_tokens)``. ``scratch[k, 0]`` is the k-th
    selected key in packed-token order (request, then position), in E4M3 as
    ``nvfp4_rows_to_e4m3`` gives it; ``compact_indices[r, c]`` is the scratch row of its key, or -1.
    Scratch rows past the selected count are unspecified (``scratch_rows`` gives its size).

    ``host_pool_pointers`` / ``host_pool_mapping`` / ``layer_idx`` / ``residual_dim`` /
    ``num_pool_tokens`` / ``global_dequant_scale`` as in the generation gather;
    ``local_topk_indices`` int32 ``[rows, topk]`` (request-local, -1 for none),
    ``query_req_indices`` int32 ``[rows]``, ``block_table`` int32 ``[>= requests, pages]``, all
    contiguous; ``cu_kv_lengths`` int64 ``[requests + 1]``; ``total_kv_tokens`` its last entry;
    ``page_stride`` the pool's page index scale x ``tokens_per_block``; ``layer_id`` the layer's
    offset in a page.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one prompt, no cached prefix: 300 keys, fewer than the Top-K (the scratch floor)",
            spec=_spec([0], [300]),
        ),
        Cell(
            why="a 256-row chunk on a 6000-key cached prefix at the amax-100 scale",
            spec=_spec([6000], [256], scale=_AMAX100),
        ),
        Cell(
            why="four requests of mixed lengths, one long enough to fill the Top-K",
            spec=_spec([0, 100, 3000, 0], [37, 64, 300, 2100]),
        ),
        Cell(
            why="24 layers per page, the layer at offset 7: the pool row's layer term",
            spec=_spec([100], [200], page_scale=24, layer_offset=7),
        ),
        Cell(why="32-token pages", spec=_spec([500], [100], tokens_per_block=32)),
        Cell(
            why="block scales over the whole finite E4M3 range at the amax-100 scale",
            spec=_spec([0], [512], scales="wide", scale=_AMAX100),
        ),
        Cell(
            why="two pools, the layer mapped to the second: the op follows host_pool_mapping",
            spec=_spec([0], [128], pools=2),
        ),
        Cell(
            why="an 8192-row prefill chunk of one request: 16.8M pairs onto 8192 keys",
            spec=_spec([0], [8192], scale=_AMAX100),
        ),
    )
    note = """
    The family entry over the engine's ``trtllm::nvfp4_mla_context_kv_cache_gather``, the op the
    engine DSA backend runs for an NVFP4 cache's context rows, with the scratch the backend sizes.
    The scratch and the compact indices are new per call and the op allocates a workspace, so this
    runs eagerly, as context steps do; ``total_kv_tokens`` is a host integer, so nothing waits on
    the GPU. The rounding is the generation gather's, so the cells gate bit for bit.

    The NVFP4 write of the step's new keys comes first and is not this entry's: the catalog
    ``mla_rope_append_paged_kv_assign_q`` in the NVFP4 KV quant mode with ``residual_dim`` 0 and
    the reciprocal of ``global_dequant_scale``. A layer whose latent pages are FP8 maps to a pool
    with no block-scale region and is refused by the op.
    """

    def __call__(
        self,
        host_pool_pointers: torch.Tensor,
        host_pool_mapping: torch.Tensor,
        local_topk_indices: torch.Tensor,
        query_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        cu_kv_lengths: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        total_kv_tokens: int,
        tokens_per_block: int,
        page_stride: int,
        layer_id: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, topk = local_topk_indices.shape
        scratch = torch.empty(
            (scratch_rows(total_kv_tokens, rows, topk), 1, LATENT_DIM),
            dtype=torch.float8_e4m3fn,
            device=local_topk_indices.device,
        )
        compact_indices = torch.empty_like(local_topk_indices)
        _trtllm_ops().nvfp4_mla_context_kv_cache_gather(
            host_pool_pointers,
            host_pool_mapping,
            local_topk_indices,
            query_req_indices,
            block_table,
            cu_kv_lengths,
            scratch,
            compact_indices,
            global_dequant_scale,
            layer_idx,
            total_kv_tokens,
            tokens_per_block,
            page_stride,
            layer_id,
            residual_dim,
            num_pool_tokens,
        )
        return scratch, compact_indices

    def reference(
        self,
        data_pool: torch.Tensor,
        scale_pool: torch.Tensor,
        local_topk_indices: torch.Tensor,
        query_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        cu_kv_lengths: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        total_kv_tokens: int,
        tokens_per_block: int,
        page_stride: int,
        layer_id: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(scratch, compact_indices)`` in int64 arithmetic; scratch rows past the selected
        count are zero (the op leaves them unspecified).

        As in the generation gather, the pool the two host tables address for ``layer_idx`` comes
        in their place: ``data_pool`` uint8 ``[num_pool_tokens, 288]``, ``scale_pool``
        float8_e4m3fn ``[num_pool_tokens, 36]``. ``layer_idx`` and ``residual_dim`` (0) are not
        used.
        """
        rows, topk = local_topk_indices.shape
        device = local_topk_indices.device
        starts = cu_kv_lengths.long()
        requests = starts.numel() - 1
        pos = local_topk_indices.long()
        req = query_req_indices.long()[:, None].expand(rows, topk)
        req_ok = (req >= 0) & (req < requests)
        req = req.clamp(0, max(requests - 1, 0))
        start = starts[req]
        packed = start + pos
        keyed = req_ok & (pos >= 0) & (pos < starts[req + 1] - start)
        keyed &= (packed >= 0) & (packed < total_kv_tokens)
        block = pos.clamp_min(0) // tokens_per_block
        pages = block_table.shape[1]
        page = block_table.long()[req, block.clamp(max=pages - 1)]
        pool_row = page * page_stride + pos.clamp_min(0) % tokens_per_block
        pool_row += layer_id * tokens_per_block
        chosen = keyed & (block < pages) & (page >= 0) & (pool_row >= 0)
        chosen &= (pool_row < num_pool_tokens) & (pool_row <= _INT32_MAX)

        selected = torch.zeros(total_kv_tokens, dtype=torch.bool, device=device)
        key_row = torch.zeros(total_kv_tokens, dtype=torch.long, device=device)
        selected[packed[chosen]] = True
        key_row[packed[chosen]] = pool_row[chosen]
        order = torch.cumsum(selected.long(), 0) - selected.long()
        slot = packed.clamp(0, total_kv_tokens - 1)
        compact = torch.where(keyed & selected[slot], order[slot], torch.full_like(slot, -1))

        scratch = torch.zeros(
            (scratch_rows(total_kv_tokens, rows, topk), 1, LATENT_DIM),
            dtype=torch.uint8,
            device=device,
        )
        picked = key_row[selected]
        codes = data_pool.view(torch.uint8)[picked]
        scales = scale_pool.view(torch.uint8)[picked].view(torch.float8_e4m3fn)
        rows_e4m3 = nvfp4_rows_to_e4m3(codes, scales, global_dequant_scale)
        scratch[: picked.numel(), 0] = rows_e4m3.view(torch.uint8)
        return scratch.view(torch.float8_e4m3fn), compact.to(torch.int32)

    def is_valid(
        self,
        host_pool_pointers: torch.Tensor,
        host_pool_mapping: torch.Tensor,
        local_topk_indices: torch.Tensor,
        query_req_indices: torch.Tensor,
        block_table: torch.Tensor,
        cu_kv_lengths: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        total_kv_tokens: int,
        tokens_per_block: int,
        page_stride: int,
        layer_id: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> None:
        # The family cache stores no residual groups (mla_kv_cache_residual_dim 0).
        assert residual_dim == 0, (
            f"residual_dim must be 0 for the family NVFP4 latent cache; got {residual_dim}"
        )
        # Keys at or past total_kv_tokens are never selected: a total below the packed keys drops
        # the last requests' keys without a word.
        assert int(cu_kv_lengths[-1]) == total_kv_tokens, (
            f"total_kv_tokens {total_kv_tokens} is not the packed keys' {int(cu_kv_lengths[-1])}"
        )
        if local_topk_indices.numel() == 0:
            return
        requests = cu_kv_lengths.numel() - 1
        req = query_req_indices.long()
        # A row of no request selects nothing.
        assert int(req.min()) >= 0 and int(req.max()) < requests, "a row of no context request"
        # Request-local positions only: a pool row or a packed token past the request's length
        # (the remapped indices, say) selects nothing, so the row attends to fewer keys.
        lengths = (cu_kv_lengths[1:] - cu_kv_lengths[:-1]).long()[req]
        pos = local_topk_indices.long()
        assert bool((pos < lengths[:, None]).all()), (
            "a Top-K position past its request's keys: pass the request-local Top-K, not the remap"
        )


rubin_dsa_nvfp4_ctx_gather_fp8 = RubinDsaNvfp4CtxGatherFp8()
