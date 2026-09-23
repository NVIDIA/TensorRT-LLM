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
"""Tensor-parallel sharding of Mamba2 mixers, including group replication
(``tp_size > n_groups``)."""

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm
from tensorrt_llm._torch.modules.mamba.mamba2_tp import (
    Mamba2TpShard,
    ReplicatedGroupRMSNormGated,
    mamba2_valid_tp_sizes,
    validate_mamba2_tp,
)
from tensorrt_llm._torch.utils import split
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

# Nemotron-3 Ultra: 256 Mamba heads x 64, 8 groups, d_state 128.
ULTRA = dict(nheads=256, n_groups=8, head_dim=64, d_state=128)


def test_valid_tp_sizes_ultra():
    assert mamba2_valid_tp_sizes(256, 8) == [1, 2, 4, 8, 16, 32, 64, 128, 256]


@pytest.mark.parametrize("tp", [3, 5, 6, 12, 24, 512])
def test_validate_rejects_bad_tp(tp):
    with pytest.raises(ValueError, match="Valid tp_size values"):
        validate_mamba2_tp(256, 8, tp)


def test_validate_rejects_group_indivisible():
    # 24 heads split evenly at tp=3, but 8 groups neither divide tp=3 nor are
    # a multiple of it.
    with pytest.raises(ValueError, match="Valid tp_size values"):
        validate_mamba2_tp(24, 8, 3)


def test_validate_rejects_zero_n_groups():
    # n_groups must be >= 1, otherwise Mamba2TpShard.replication would divide
    # by zero. The suggestion list is computed from n_groups, so a degenerate
    # geometry must not advertise one: with n_groups=0 every tp_size passes
    # `tp % n_groups == 0`'s counterpart check and the list would read
    # [1, 2, 4, 8], none of which is actually usable.
    with pytest.raises(ValueError) as excinfo:
        validate_mamba2_tp(8, 0, 1)
    assert "Valid tp_size values" not in str(excinfo.value)
    assert "n_groups=0" in str(excinfo.value)


@pytest.mark.parametrize(
    "tp, tp_ngroups, replication",
    [(1, 8, 1), (2, 4, 1), (4, 2, 1), (8, 1, 1), (16, 1, 2), (32, 1, 4)],
)
def test_shard_dims_ultra(tp, tp_ngroups, replication):
    s = Mamba2TpShard(tp_size=tp, **ULTRA)
    assert s.tp_ngroups == tp_ngroups
    assert s.replication == replication
    assert s.replicated == (replication > 1)
    assert s.tp_nheads == 256 // tp
    assert s.tp_d_inner == 16384 // tp
    assert s.tp_grouped_state_dim == tp_ngroups * 128
    assert s.tp_conv_dim == s.tp_d_inner + 2 * tp_ngroups * 128
    assert s.tp_d_in_proj == 2 * s.tp_d_inner + 2 * tp_ngroups * 128 + 256 // tp
    assert s.padded_n_groups == tp_ngroups * tp
    assert s.padded_conv_dim == s.tp_conv_dim * tp
    assert s.padded_d_in_proj == s.tp_d_in_proj * tp
    assert s.group_size == 2048
    if s.replicated:
        assert [s.group_index(r) for r in range(tp)] == [r // replication for r in range(tp)]
    else:
        # Even layouts are exactly today's arithmetic.
        assert s.padded_n_groups == 8
        assert s.padded_conv_dim == 16384 + 2 * 8 * 128
        assert s.padded_d_in_proj == 2 * 16384 + 2 * 8 * 128 + 256


def test_ultra_tp16_matches_design_numbers():
    s = Mamba2TpShard(tp_size=16, **ULTRA)
    assert (s.tp_nheads, s.tp_d_inner, s.tp_conv_dim, s.tp_d_in_proj) == (
        16,
        1024,
        1024 + 256,
        2320,
    )


def _row_ids(rows: int, cols: int = 2) -> torch.Tensor:
    """Tensor whose every element in row i equals i (row identity)."""
    return torch.arange(rows, dtype=torch.int64).unsqueeze(1).expand(rows, cols).clone()


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_rearrange_in_proj_rows_by_group(tp):
    # 8 heads x 4, 2 groups x d_state 3: tp=2 splits groups evenly,
    # tp=4/8 replicate each group across 2/4 ranks.
    s = Mamba2TpShard(tp_size=tp, nheads=8, n_groups=2, head_dim=4, d_state=3)
    d_inner, gs, nheads = 32, 6, 8
    w = _row_ids(2 * d_inner + 2 * gs + nheads)
    out = s.rearrange_in_proj_rows(w)
    assert out.shape == (s.padded_d_in_proj, 2)
    z0, x0, b0, c0, dt0 = 0, d_inner, 2 * d_inner, 2 * d_inner + gs, 2 * d_inner + 2 * gs

    def heads(base, width, r):
        return list(range(base + r * width, base + (r + 1) * width))

    def groups(base, r):
        if s.replicated:
            g = s.group_index(r)
            return list(range(base + g * 3, base + (g + 1) * 3))
        width = gs // tp
        return list(range(base + r * width, base + (r + 1) * width))

    for r in range(tp):
        rows = out[r * s.tp_d_in_proj : (r + 1) * s.tp_d_in_proj, 0].tolist()
        expected = (
            heads(z0, s.tp_d_inner, r)
            + heads(x0, s.tp_d_inner, r)
            + groups(b0, r)
            + groups(c0, r)
            + heads(dt0, s.tp_nheads, r)
        )
        assert rows == expected, f"rank {r}"


def test_rearrange_conv1d_rows_replicates_single_group():
    s = Mamba2TpShard(tp_size=4, nheads=8, n_groups=1, head_dim=4, d_state=3)
    d_inner = 32
    w = _row_ids(d_inner + 2 * 3)
    out = s.rearrange_conv1d_rows(w)
    assert out.shape == (s.padded_conv_dim, 2)
    for r in range(4):
        rows = out[r * s.tp_conv_dim : (r + 1) * s.tp_conv_dim, 0].tolist()
        x_rows = list(range(r * 8, (r + 1) * 8))
        assert rows == x_rows + [32, 33, 34] + [35, 36, 37], f"rank {r}"


def test_rearrange_matches_legacy_split_when_groups_divide():
    """tp <= n_groups must reproduce the historical per-segment split exactly."""
    s = Mamba2TpShard(tp_size=4, nheads=8, n_groups=8, head_dim=4, d_state=3)
    w = torch.randn(2 * 32 + 2 * 24 + 8, 5)
    z, x, b, c, dt = torch.split(w, [32, 32, 24, 24, 8])
    legacy = torch.cat(
        [
            torch.cat(
                [split(z, 4, r), split(x, 4, r), split(b, 4, r), split(c, 4, r), split(dt, 4, r)]
            )
            for r in range(4)
        ]
    )
    torch.testing.assert_close(s.rearrange_in_proj_rows(w), legacy)

    wc = torch.randn(32 + 24 + 24, 4)
    xx, bb, cc = torch.split(wc, [32, 24, 24])
    legacy_c = torch.cat(
        [torch.cat([split(xx, 4, r), split(bb, 4, r), split(cc, 4, r)]) for r in range(4)]
    )
    torch.testing.assert_close(s.rearrange_conv1d_rows(wc), legacy_c)


def test_rearrange_is_identity_for_tp1():
    s = Mamba2TpShard(tp_size=1, nheads=8, n_groups=2, head_dim=4, d_state=3)
    w = torch.randn(2 * 32 + 2 * 6 + 8, 3)
    torch.testing.assert_close(s.rearrange_in_proj_rows(w), w)


# --------------------------------------------------------------------------
# ReplicatedGroupRMSNormGated
# --------------------------------------------------------------------------
skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@skip_no_cuda
@pytest.mark.parametrize("tp, n_groups", [(2, 1), (4, 2), (4, 1)])
def test_replicated_norm_matches_full_group_norm(tp, n_groups):
    torch.manual_seed(0)
    nheads, head_dim, d_state, tokens = 8, 16, 8, 6
    shard = Mamba2TpShard(
        tp_size=tp, nheads=nheads, n_groups=n_groups, head_dim=head_dim, d_state=d_state
    )
    assert shard.replicated
    d_inner = shard.d_inner
    x = torch.randn(tokens, d_inner, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.rand(d_inner, device="cuda", dtype=torch.bfloat16) + 0.5

    ref_norm = RMSNorm(
        d_inner, eps=1e-5, group_size=shard.group_size, norm_before_gate=False, dtype=torch.bfloat16
    ).cuda()
    ref_norm.weight.data.copy_(weight)
    ref = ref_norm(x, z)

    # The exact per-group sum of squares the all-reduce would deliver.
    xz = x.float() * torch.nn.functional.silu(z.float())
    group_ss = xz.view(tokens, n_groups, shard.group_size).square().sum(-1)

    for rank in range(tp):
        norm = ReplicatedGroupRMSNormGated(
            shard,
            rank,
            Mapping(world_size=tp, tp_size=tp, rank=rank),
            eps=1e-5,
            dtype=torch.bfloat16,
        ).cuda()
        cols = slice(rank * shard.tp_d_inner, (rank + 1) * shard.tp_d_inner)
        norm.weight.data.copy_(weight[cols])
        g = shard.group_index(rank)

        def fake_reduce(local_ss, cols=cols, g=g):
            torch.testing.assert_close(local_ss, xz[:, cols].square().sum(-1), atol=1e-3, rtol=1e-3)
            return group_ss[:, g]

        norm.reduce_sum_of_squares = fake_reduce
        out = norm(x[:, cols], z[:, cols])
        assert out.dtype == torch.bfloat16 and out.shape == ref[:, cols].shape
        torch.testing.assert_close(out, ref[:, cols], atol=2e-2, rtol=2e-2)


@skip_no_cuda
def test_reduce_sum_of_squares_fills_own_group_column():
    shard = Mamba2TpShard(tp_size=4, nheads=8, n_groups=2, head_dim=16, d_state=8)
    norm = ReplicatedGroupRMSNormGated(
        shard, tp_rank=3, mapping=Mapping(world_size=4, tp_size=4, rank=3), dtype=torch.bfloat16
    ).cuda()
    seen = {}

    def fake_exchange(buf):
        seen["buf"] = buf.clone()
        return buf * 2  # stands in for the sum over the two replicas

    norm._exchange = fake_exchange
    local = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    out = norm.reduce_sum_of_squares(local)
    assert seen["buf"].shape == (3, 2) and seen["buf"].dtype == torch.float32
    assert torch.equal(seen["buf"][:, 1], local)  # rank 3 -> group 1
    assert seen["buf"][:, 0].abs().sum() == 0
    torch.testing.assert_close(out, local * 2)


# --------------------------------------------------------------------------
# Mamba2Mixer construction
# --------------------------------------------------------------------------
def _make_mixer(tp, rank, nheads, n_groups, head_dim=16, d_state=8, d_model=32):
    # Deferred: Mamba2Mixer needs flashinfer's SSDCombined, not always installed; keeps this file importable without it.
    from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer

    # NCCL strategy: no all-reduce workspace is allocated at construction,
    # so a single process can instantiate every rank's module.
    cfg = ModelConfig(
        mapping=Mapping(world_size=tp, tp_size=tp, rank=rank),
        allreduce_strategy=AllReduceStrategy.NCCL,
    )
    return Mamba2Mixer(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        nheads=nheads,
        n_groups=n_groups,
        head_dim=head_dim,
        chunk_size=8,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=cfg,
    )


@skip_no_cuda
def test_mixer_replicated_layout_shapes():
    tp, nheads, n_groups, head_dim, d_state, d_model = 4, 8, 2, 16, 8, 32
    for rank in range(tp):
        mixer = _make_mixer(tp, rank, nheads, n_groups, head_dim, d_state, d_model)
        s = mixer.shard
        assert s.replicated and s.replication == 2
        assert (mixer.tp_ngroups, mixer.tp_nheads, mixer.tp_d_inner, mixer.tp_conv_dim) == (
            1,
            2,
            32,
            32 + 2 * d_state,
        )
        assert mixer.in_proj.weight.shape == (s.tp_d_in_proj, d_model)
        assert mixer.conv1d.weight.shape == (s.tp_conv_dim, 4)
        assert mixer.conv1d.bias.shape == (s.tp_conv_dim,)
        assert mixer.out_proj.weight.shape == (d_model, s.tp_d_inner)
        assert isinstance(mixer.norm, ReplicatedGroupRMSNormGated)
        assert mixer.norm.weight.shape == (s.tp_d_inner,)
        assert mixer.norm.group_index == rank // 2
        assert mixer.A.shape == mixer.D.shape == mixer.dt_bias.shape == (2,)
        assert mixer.is_nvfp4 is False and mixer.norm.is_nvfp4 is False


@skip_no_cuda
def test_mixer_even_groups_keep_legacy_norm():
    mixer = _make_mixer(tp=2, rank=1, nheads=8, n_groups=2)
    s = mixer.shard
    assert not s.replicated and mixer.tp_ngroups == 1 and mixer.tp_nheads == 4
    assert type(mixer.norm) is RMSNorm
    assert not isinstance(mixer.norm, ReplicatedGroupRMSNormGated)
    assert mixer.norm.group_size == mixer.tp_d_inner // mixer.tp_ngroups == 64
    assert mixer.in_proj.weight.shape == (2 * 64 + 2 * 8 + 4, 32)
    assert mixer.conv1d.weight.shape == (64 + 2 * 8, 4)


@skip_no_cuda
def test_mixer_tp1_is_legacy_layout():
    mixer = _make_mixer(tp=1, rank=0, nheads=8, n_groups=2)
    s = mixer.shard
    assert not s.replicated
    assert type(mixer.norm) is RMSNorm
    assert mixer.in_proj.weight.shape == (2 * 128 + 2 * 16 + 8, 32)
    assert mixer.conv1d.weight.shape == (128 + 2 * 16, 4)
    assert mixer.A.shape == (8,)
