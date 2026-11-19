# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""L2 policy construction for weight and weight-scale TMA loads."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims


@dsl_user_op
def _make_l2_cache_policy(priority: str, *, loc=None, ip=None) -> Int64:
    # PTX defines the policy as an opaque value; let createpolicy encode it.
    return prims.inline_ptx(
        f"createpolicy.fractional.L2::{priority}.b64 {{$w0}}, 1.0;",
        write_only_types=[Int64],
        loc=loc,
        ip=ip,
    )


def _is_single_expert_tile(expert_idx, previous_expert_idx, next_expert_idx):
    """Routing stores each expert's token tiles in a contiguous run."""
    return (previous_expert_idx != expert_idx) & (next_expert_idx != expert_idx)


@cute.jit
def make_weight_l2_cache_hint(
    enabled: cutlass.Constexpr, tile_idx_view, num_token_tiles_view, token_tile
):
    """Choose a policy once per work tile from this expert's routing span.

    Only an expert with neither neighbor in its contiguous routing run can
    evict first. Bound neighbor reads by the live routing count, not the
    metadata allocation or the early-exit launch capacity.
    """
    if cutlass.const_expr(enabled):
        num_token_tiles = num_token_tiles_view.load(idx=Int32(0), vector_size=1)[0]
        expert_idx = tile_idx_view.load(idx=token_tile, vector_size=1)[0]
        previous_expert_idx = Int32(-1)
        next_expert_idx = Int32(-1)
        if token_tile > Int32(0):
            previous_expert_idx = tile_idx_view.load(
                idx=token_tile - Int32(1), vector_size=1
            )[0]
        if token_tile + Int32(1) < num_token_tiles:
            next_expert_idx = tile_idx_view.load(
                idx=token_tile + Int32(1), vector_size=1
            )[0]
        policy = _make_l2_cache_policy("evict_normal")
        if _is_single_expert_tile(expert_idx, previous_expert_idx, next_expert_idx):
            policy = _make_l2_cache_policy("evict_first")
        return policy
    return None
