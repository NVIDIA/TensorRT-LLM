# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import torch

from tensorrt_llm._torch.speculative.interface import SpecMetadata


def test_device_window_repacking_follows_fresh_token_owners():
    owners = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    meta = types.SimpleNamespace(
        request_temperatures=torch.tensor([0.2, 0.7, 1.1]),
        temperatures=torch.empty(6),
        request_top_ks=torch.tensor([2, 7, 11], dtype=torch.int32),
        top_ks=torch.empty(6, dtype=torch.int32),
        request_top_ps=torch.tensor([0.3, 0.8, 0.95]),
        top_ps=torch.empty(6),
        request_seeds=torch.tensor([101, 202, 303], dtype=torch.int64),
        seeds=torch.empty(6, dtype=torch.int64),
        request_offsets=torch.tensor([10, 20, 30], dtype=torch.int64),
        offsets=torch.empty(6, dtype=torch.int64),
        _sampling_params_signature=[("request",), ("expanded",)],
    )

    SpecMetadata.remap_expanded_sampling_params(meta, owners, owners.numel())

    assert torch.equal(meta.temperatures, meta.request_temperatures.index_select(0, owners))
    assert torch.equal(meta.top_ks, meta.request_top_ks.index_select(0, owners))
    assert torch.equal(meta.top_ps, meta.request_top_ps.index_select(0, owners))
    assert torch.equal(meta.seeds, meta.request_seeds.index_select(0, owners))
    assert torch.equal(meta.offsets, meta.request_offsets.index_select(0, owners))
    assert meta._sampling_params_signature[0] == ("request",)
    assert meta._sampling_params_signature[1] is None


def test_device_window_repacking_ignores_unallocated_optional_buffers():
    meta = types.SimpleNamespace(
        request_temperatures=None,
        temperatures=None,
        request_top_ks=None,
        top_ks=None,
        request_top_ps=None,
        top_ps=None,
        request_seeds=None,
        seeds=None,
        request_offsets=None,
        offsets=None,
        _sampling_params_signature=[None, None],
    )
    SpecMetadata.remap_expanded_sampling_params(meta, torch.tensor([0]), 1)


def test_device_window_repacking_forces_next_host_expansion():
    values = ((0.2, 2, 0.3), (0.7, 7, 0.8))
    per_request = [(0.2, 2, 0.3, 2), (0.7, 7, 0.8, 4)]
    meta = types.SimpleNamespace(
        request_temperatures=torch.tensor([0.2, 0.7]),
        temperatures=torch.empty(6),
        request_top_ks=torch.tensor([2, 7], dtype=torch.int32),
        top_ks=torch.empty(6, dtype=torch.int32),
        request_top_ps=torch.tensor([0.3, 0.8]),
        top_ps=torch.empty(6),
        request_seeds=None,
        seeds=None,
        request_offsets=None,
        offsets=None,
        _sampling_params_signature=[values, (values, (2, 4))],
    )
    SpecMetadata.remap_expanded_sampling_params(meta, torch.tensor([0, 1, 1, 1, 0, 0]), 6)

    need_request, need_expanded = SpecMetadata._sampling_params_buffers_need_update(
        meta, per_request
    )
    assert need_request is False
    assert need_expanded is True
