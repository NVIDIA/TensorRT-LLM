# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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
"""Parity of the Kimi K3 helix_a2a token-sliced decoder layer with the full-row layer.

N ranks run in threads; ``_SimulatedHelixA2A`` keeps the production row split
(``scatter_rows``) and replaces only the collectives.
"""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear as kimi
from tensorrt_llm._torch.models.modeling_kimi_linear import (
    KimiK3HelixA2A,
    KimiK3RMSNorm,
    KimiLinearDecoderLayer,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import is_nccl_window_tensor_scoped_module

pytestmark = pytest.mark.cpu_only

HIDDEN = 16


class _World:
    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.slots = [None] * world_size

    def exchange(self, rank, tensor):
        self.slots[rank] = tensor
        self.barrier.wait()
        gathered = list(self.slots)
        self.barrier.wait()
        return gathered


class _SimulatedHelixA2A(KimiK3HelixA2A):
    def __init__(self, world, rank):
        nn.Module.__init__(self)
        self.world, self.world_size, self.rank = world, world.world_size, rank

    def all_gather(self, x, num_tokens):
        return torch.cat(self.world.exchange(self.rank, x), dim=0)[:num_tokens]

    def reduce_scatter(self, x, num_tokens):
        return self.scatter_rows(sum(self.world.exchange(self.rank, x)), num_tokens)


class _Capture:
    def __init__(self):
        self.captured = {}

    def maybe_capture_hidden_states(self, layer_id, hidden_states, _):
        self.captured[layer_id] = hidden_states.clone()


def _make_layer(layer_idx, attn_weight, moe_weight, rank, token_sliced):
    torch.manual_seed(0)
    layer = KimiLinearDecoderLayer.__new__(KimiLinearDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_idx, layer.attn_res_block_size = layer_idx, 2
    layer.is_kda, layer.is_moe, layer.token_sliced = True, True, token_sliced
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attention_res_norm",
        "mlp_res_norm",
    ):
        setattr(layer, name, KimiK3RMSNorm(HIDDEN))
    layer.self_attention_res_proj = nn.Linear(HIDDEN, 1, bias=False)
    layer.mlp_res_proj = nn.Linear(HIDDEN, 1, bias=False)
    # Head-TP attention: rank 0 holds the whole row-parallel partial sum.
    owns_output = not token_sliced or rank == 0
    layer.linear_attn = lambda x, _md: x @ attn_weight if owns_output else torch.zeros_like(x)
    layer.seen_all_rank_num_tokens = []

    def moe(x, all_rank_num_tokens):
        layer.seen_all_rank_num_tokens.append(all_rank_num_tokens)
        return torch.tanh(x @ moe_weight)

    layer.block_sparse_moe = moe
    return layer


def test_decoder_layer_is_nccl_window_scoped():
    # The reduce-scatter frees an NCCL-window AllReduce output inside the layer.
    assert is_nccl_window_tensor_scoped_module(KimiLinearDecoderLayer)


@pytest.mark.parametrize("num_tokens,world_size", [(13, 4), (8, 4), (3, 4), (7, 2)])
@pytest.mark.parametrize("layer_idx", [3, 4])  # 4 % 2 == 0 writes a new snapshot
@pytest.mark.parametrize("capture", [False, True])
def test_token_sliced_layer_matches_full_rows(num_tokens, world_size, layer_idx, capture):
    torch.manual_seed(1)
    attn_weight = torch.randn(HIDDEN, HIDDEN)
    moe_weight = torch.randn(HIDDEN, HIDDEN)
    num_snapshots, bank_rows = 2, 4
    hidden = torch.randn(num_tokens, HIDDEN)
    bank = torch.zeros(bank_rows, num_tokens, HIDDEN)
    bank[:num_snapshots] = torch.randn(num_snapshots, num_tokens, HIDDEN)
    md = SimpleNamespace(num_tokens=num_tokens, all_rank_num_tokens=None)

    with patch.object(kimi, "_FUSED_ATTN_RES_ENABLED", False):
        ref_capture = _Capture()
        ref_layer = _make_layer(layer_idx, attn_weight, moe_weight, 0, token_sliced=False)
        ref_bank = bank.clone()
        ref_out, ref_snapshots = ref_layer(
            hidden.clone(),
            ref_bank,
            num_snapshots,
            md,
            capture=(ref_capture, 7) if capture else None,
        )

        world = _World(world_size)
        results, errors = [None] * world_size, []
        # Built on this thread: nn.Linear init draws from the shared global RNG.
        layers = [
            _make_layer(layer_idx, attn_weight, moe_weight, rank, token_sliced=True)
            for rank in range(world_size)
        ]

        def run(rank):
            try:
                helix = _SimulatedHelixA2A(world, rank)
                layer = layers[rank]
                rank_capture = _Capture()
                rank_bank = helix.scatter_rows(bank.clone(), num_tokens)
                out, snapshots = layer(
                    helix.scatter_rows(hidden.clone(), num_tokens),
                    rank_bank,
                    num_snapshots,
                    md,
                    capture=(rank_capture, 7) if capture else None,
                    helix_a2a=helix,
                )
                results[rank] = (out, snapshots, rank_bank, rank_capture, layer, helix)
            except BaseException as exc:  # surface thread failures in the test
                errors.append(exc)
                world.barrier.abort()

        threads = [threading.Thread(target=run, args=(r,)) for r in range(world_size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert not errors, errors

    chunk = -(-num_tokens // world_size)
    out = torch.cat([r[0] for r in results])[:num_tokens]
    # Default fp32 tolerances: chunked GEMMs accumulate in a different order.
    torch.testing.assert_close(out, ref_out)
    sliced_bank = torch.cat([r[2] for r in results], dim=-2)[..., :num_tokens, :]
    torch.testing.assert_close(sliced_bank, ref_bank)
    for _, snapshots, _, rank_capture, layer, helix in results:
        assert snapshots == ref_snapshots
        assert layer.seen_all_rank_num_tokens == [[chunk] * world_size]
        if capture:
            # The DSpark tap sees every row on every rank, as in the full-row layer.
            torch.testing.assert_close(rank_capture.captured[7], ref_capture.captured[7])
