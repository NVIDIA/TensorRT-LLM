# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only sparse policy shared by group recommendation and launch planning.

Reuse common FMHA geometry, split and scheduler selection. Explicit G remains
caller-owned; only the recommendation evaluates alternative groups.
"""

from dataclasses import dataclass

MAX_GROUP_SIZE = 8
SUPPORTED_GROUP_SIZES = tuple(range(1, MAX_GROUP_SIZE + 1))
MAX_TILE_SIZE_Q = 128
TILE_SIZE_KV = 128
# Standalone reducer capacity; work and the first SM wave bound actual fanout.
MAX_SPLITS_KV = 128


@dataclass(frozen=True)
class SparseLaunch:
    tile_size_q: int
    num_insts_kv: int
    splits_kv: int
    head_dim_per_stage_kv: int
    o_stages: int
    use_persistent_scheduler: bool

    @property
    def use_keeps_mma_ab(self) -> bool:
        return self.tile_size_q >= 64


def select_sparse_launch(
    *,
    group_size: int,
    heads_q_per_kv: int,
    head_dim: int,
    q_dtype_key: str,
    num_routes: int,
    num_kv_heads: int,
    route_kv_tokens: int,
    multi_processor_count: int,
    split_kv: bool = True,
) -> SparseLaunch:
    """Choose the smallest fitting dense Q recipe, then useful one-wave splits.

    G is already fixed by the caller. This function never changes it, queries
    the device, reads tensors, or applies a batch/context tuning table.
    """
    from .kernels.fmha_decode.fmha_decode_config import (
        MIN_LOOP_ITERS_PER_SPLIT,
        _select_auto_launch_mode,
        enumerate_grouped_q_mma_candidates,
        select_splits_kv,
    )

    # Splits must also cover non-shared KV in the union. Retain the route's
    # capacity bound here; runtime masks the inactive split prefix. The G
    # recommendation below scores useful work with a shared-KV estimate.
    work_tokens = max(1, route_kv_tokens)
    # D256 FP8 G1 requires the staged Keeps recipe. D64/D128 also have
    # qualified unstaged Swaps recipes.
    minimum_q = (
        64
        if head_dim == 256 and q_dtype_key == "float8_e4m3fn" and group_size == 1
        else 8
    )
    candidates = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=heads_q_per_kv, seq_len_q=group_size
    )

    def recipe(minimum: int):
        return next(
            c
            for c in candidates
            if c.tile_size_q >= minimum and c.q_tokens_per_cta >= group_size
        )

    def fanout(instances: int) -> int:
        maximum = MAX_SPLITS_KV
        if group_size == 1:
            # A partial final iteration must not buy an otherwise empty split.
            maximum = min(maximum, max(1, work_tokens // (TILE_SIZE_KV * instances)))
        return select_splits_kv(
            seq_len_kv=work_tokens,
            batch_size=num_routes,
            num_heads_kv=num_kv_heads,
            tile_size_kv=TILE_SIZE_KV,
            num_insts_kv=instances,
            service_capacity=multi_processor_count,
            max_splits_kv=maximum,
            min_loop_iters_per_split=1 if group_size == 1 else MIN_LOOP_ITERS_PER_SPLIT,
            split_kv=split_kv,
        )

    candidate = recipe(minimum_q)
    instances = 1 if head_dim == 256 and candidate.variant == "keeps_mma_ab" else 2
    splits = fanout(instances)
    # Other D256 FP8 groups may use Swaps only when split-KV is active.
    if (
        head_dim == 256
        and q_dtype_key == "float8_e4m3fn"
        and splits == 1
        and candidate.tile_size_q < 64
    ):
        candidate = recipe(64)
        instances = 1
        splits = fanout(instances)
    return SparseLaunch(
        tile_size_q=candidate.tile_size_q,
        num_insts_kv=instances,
        splits_kv=splits,
        # Full FP8 stages preserve the 128-byte swizzled planes while issuing
        # PV at N256. BF16 and Swaps retain their existing D128 head bands.
        head_dim_per_stage_kv=(
            256
            if head_dim == 256
            and q_dtype_key == "float8_e4m3fn"
            and candidate.variant == "keeps_mma_ab"
            else 128
            if head_dim == 256
            else 0
        ),
        o_stages=1 if head_dim == 256 and candidate.variant == "keeps_mma_ab" else 2,
        use_persistent_scheduler=(
            splits == 1
            and _select_auto_launch_mode(
                batch_size=num_routes,
                num_heads_kv=num_kv_heads,
                seq_len_kv=work_tokens,
                tile_size_kv=TILE_SIZE_KV,
                split_kv=False,
                service_capacity=multi_processor_count,
            )
            == "persistent"
        ),
    )


def candidate_union_tokens(
    group_size: int, selected_kv_tokens: int, block_size: int = 4
) -> int:
    """Bound a union in semantic blocks; G1 retains its exact tail length."""
    if group_size == 1:
        return selected_kv_tokens
    return (
        group_size * ((selected_kv_tokens + block_size - 1) // block_size) * block_size
    )


def suggest_sparse_group(
    *,
    batch_size: int,
    seq_len_q: int,
    selected_kv_tokens: int,
    heads_q_per_kv: int,
    num_kv_heads: int,
    multi_processor_count: int,
    head_dim: int,
    q_dtype_key: str,
    split_kv: bool,
    block_size: int = 4,
) -> int:
    """Prefer grouping, then splitting; shrink G only for better SM coverage.

    Bounds and dtype validation belong to the public caller. A final short Q
    group contributes only its live rows. No SQ % G divisibility is required.
    A fully-shared per-query KV proxy estimates useful active splits, rather
    than counting every capacity-based split as active. Padding breaks ties
    in wave coverage; it must not keep an otherwise under-filled GPU grouped.
    TODO: model actual union overlap and scattered loading/metadata costs.
    """
    from .kernels.fmha_decode.fmha_decode_config import (
        compute_runtime_active_splits_kv,
        enumerate_grouped_q_mma_candidates,
    )

    largest = min(MAX_GROUP_SIZE, seq_len_q, MAX_TILE_SIZE_Q // heads_q_per_kv)
    dense_candidates = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=heads_q_per_kv, seq_len_q=largest
    )
    # Retain the largest group fitting each dense Q tile. Smaller groups
    # sharing that tile only add partially occupied CTAs. At the smallest
    # tile, still allow further ungrouping to expose otherwise idle SMs.
    groups = {
        min(largest, candidate.q_tokens_per_cta) for candidate in dense_candidates
    }
    smallest_capacity = min(
        candidate.q_tokens_per_cta for candidate in dense_candidates
    )
    groups.update(range(1, min(largest, smallest_capacity) + 1))
    best_group, best_coverage, best_density = largest, -1.0, -1.0
    for group in sorted(groups, reverse=True):
        full_groups, tail = divmod(seq_len_q, group)
        routes = batch_size * (full_groups + int(tail > 0))
        length = candidate_union_tokens(group, selected_kv_tokens, block_size)
        launch = select_sparse_launch(
            group_size=group,
            heads_q_per_kv=heads_q_per_kv,
            head_dim=head_dim,
            q_dtype_key=q_dtype_key,
            num_routes=routes,
            num_kv_heads=num_kv_heads,
            route_kv_tokens=length,
            multi_processor_count=multi_processor_count,
            split_kv=split_kv,
        )

        # Score active work, not the configured union-capacity split grid.
        active_splits = compute_runtime_active_splits_kv(
            valid_k=length // group,
            tile_size_kv=TILE_SIZE_KV,
            num_insts_kv=launch.num_insts_kv,
            configured_splits_kv=launch.splits_kv,
        )
        per_request = (full_groups + int(tail > 0)) * active_splits
        grid = batch_size * num_kv_heads * per_request
        waves = (grid + multi_processor_count - 1) // multi_processor_count
        coverage = grid / waves
        # A short final group still pays for the full Q tile. Prefer less
        # padding only when active SM coverage is equal.
        live_rows = batch_size * num_kv_heads * seq_len_q * heads_q_per_kv
        density = live_rows / (routes * num_kv_heads * launch.tile_size_q)
        if coverage > best_coverage or (
            coverage == best_coverage and density > best_density
        ):
            best_group, best_coverage, best_density = group, coverage, density
        if group == largest and grid >= multi_processor_count:
            # The grouped Q grid already supplies a full first wave.
            break
        if coverage == multi_processor_count:
            break
    return best_group
