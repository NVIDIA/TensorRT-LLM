# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MegaMoE CuteDSL backend's methods for adaptive EPLB."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import TimeModel
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import LayerBlockLayout, SlotPlane
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mm
from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.geometry import BundleLayout

pytestmark = pytest.mark.cpu_only

IMPL = mm.TrtllmCutedslMegaMoeNvfp4Impl


def test_slot_weight_planes_are_the_quant_methods_live_planes():
    bundle = BundleLayout.create(hidden=512, intermediate=256)
    slots = 4
    params = {}
    for plane in bundle.planes:
        shape, stride, dtype_name, _, _ = plane.storage_metadata(slots)
        params[plane.name] = torch.empty_strided(shape, stride, dtype=getattr(torch, dtype_name))
    backend = SimpleNamespace(
        quant_method=SimpleNamespace(live_weight_plane_spec=lambda module: bundle),
        compute_slot_count=slots,
        layer_idx=3,
        get_parameter=params.__getitem__,
    )
    declared = IMPL.slot_weight_planes(backend)
    assert [name for name, _ in declared] == [plane.name for plane in bundle.planes]
    planes = []
    for name, alignment in declared:
        param = params[name]
        shape, stride = tuple(param.shape), tuple(param.stride())
        planes.append(SlotPlane(name, shape, stride, param.dtype, alignment))
    assert [plane.slot_bytes for plane in planes] == [plane.nbytes for plane in bundle.planes]
    layout = LayerBlockLayout(planes, slots)
    assert all(layout.offsets[plane.name] % plane.alignment == 0 for plane in planes)
    first = bundle.planes[0].name
    params[first] = torch.empty(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"plane {first!r}"):
        IMPL.slot_weight_planes(backend)


def test_weights_stop_being_movable_once_the_kernel_holds_views():
    backend = SimpleNamespace(_cute_weight_cache={}, layer_idx=5)
    IMPL.check_weights_movable(backend)
    backend._cute_weight_cache["views"] = object()
    with pytest.raises(RuntimeError, match="MoE layer 5 has launched its kernel"):
        IMPL.check_weights_movable(backend)
    assert IMPL.slot_id_dtype.fget(SimpleNamespace(_topk_idx_dtype=torch.int32)) is torch.int32


def test_decode_time_model_streams_the_expert_bytes_and_prices_the_row_flops():
    bandwidth, peak, scale = IMPL._DECODE_DEVICES["GB300"]
    hidden, expand, inner = 7168, 6144, 3072
    backend = SimpleNamespace(
        _DECODE_DEVICES=IMPL._DECODE_DEVICES,
        hidden_size=hidden,
        expand_intermediate_size_per_partition=expand,
        intermediate_size_per_partition=inner,
    )
    model = TimeModel(**IMPL.decode_time_model(backend, 37158924, "NVIDIA GB300"))
    # A slot with rows streams its expert once; a row costs its FC1 and FC2 FLOPs at the peak.
    assert model.per_active_slot_us == pytest.approx(37158924 / bandwidth * 1e6)
    row_us = 2 * hidden * (expand + inner) / peak * 1e6
    assert model.per_row_us == pytest.approx(row_us)
    # A tile is 256 rows; every extra tile of the busiest slot costs half a tile more.
    assert model.tile_rows == 256 and model.per_extra_tile_us == pytest.approx(256 * row_us)
    assert model.hot_slot_extra_tile_us == pytest.approx(128 * row_us)
    assert model.scale == scale
    assert IMPL.has_decode_time_model("NVIDIA GB300") and not IMPL.has_decode_time_model("A100")
    with pytest.raises(ValueError, match="A100"):
        IMPL.decode_time_model(backend, 37158924, "NVIDIA A100")


def test_a_calibration_prices_any_gpu_and_its_identity_resolves_tactics_without_a_launch(
    monkeypatch,
):
    backend = SimpleNamespace(
        hidden_size=64,
        expand_intermediate_size_per_partition=64,
        intermediate_size_per_partition=32,
        _select_launch_max_tokens=lambda tokens: 256,
        _use_in_kernel_fc2_reduce=lambda max_tokens: False,
        apply_topk_in_fc1=False,
        combine_format="16e2m1xbf16",
        tactic_autotune=False,
        routing_method=SimpleNamespace(experts_per_token=6),
    )
    model = IMPL.decode_time_model(backend, 1000, "NVIDIA B200", (1e12, 1e15, 0.7))
    assert model["per_active_slot_us"] == pytest.approx(1e-3) and model["scale"] == 0.7
    monkeypatch.setattr(mm, "get_sm_version", lambda: 103)
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

    tactic = repr(op._unpack_tactic(op._MEGAMOE_NONBULK_STANDALONE_TACTIC))
    identity = IMPL.decode_calibration_identity(backend, [32, 1024])
    assert identity["tactics"] == {"32": tactic, "1024": tactic} and identity["top_k"] == 6
    backend.tactic_autotune = True
    assert IMPL.decode_calibration_identity(backend, [32])["tactics"] == {"32": "autotune"}


def test_route_staging_hands_out_the_routes_and_weights_of_the_staging():
    """``route_staging`` hands out the index and weight staging of the bucket that
    ``_staging_of_next_run`` finds for the routes' ``top_k``, and nothing where it finds none."""
    idx, weights = torch.full((4, 2), -1, dtype=torch.int32), torch.zeros(4, 2)
    bufs = SimpleNamespace(topk_idx_local=idx, topk_weights=weights)
    asked = []

    def staging_of_next_run(rows, top_k):
        asked.append((rows, top_k))
        return (bufs, 4) if top_k == 2 else None

    backend = SimpleNamespace(_staging_of_next_run=staging_of_next_run)
    got = IMPL.route_staging(backend, 3, 2)
    assert got.topk_idx is idx and got.topk_weights is weights and got.bucket == 4
    assert backend._handed_out_route_staging == idx.data_ptr()
    assert IMPL.route_staging(backend, 3, 6) is None
    assert asked == [(3, 2), (3, 6)]


def test_staged_routes_are_taken_only_after_a_route_staging_hand_out():
    """run_moe reads the gate weights of staged routes from the staging too, so it takes routes found
    there only from a caller of route_staging (who writes the weights beside them), once per
    hand-out, and never together with a replica plan, whose route wait the staged path skips."""
    bufs = SimpleNamespace(topk_idx_local=torch.full((16, 6), -1, dtype=torch.int32))
    backend = SimpleNamespace()
    routes = bufs.topk_idx_local[:4]
    with pytest.raises(ValueError, match="route_staging"):
        IMPL._routes_staged(backend, bufs, routes, 4, 6, None)
    backend._handed_out_route_staging = bufs.topk_idx_local.data_ptr()
    assert IMPL._routes_staged(backend, bufs, routes, 4, 6, None)
    with pytest.raises(ValueError, match="route_staging"):
        IMPL._routes_staged(backend, bufs, routes, 4, 6, None)
    backend._handed_out_route_staging = bufs.topk_idx_local.data_ptr()
    with pytest.raises(ValueError, match="replica plan"):
        IMPL._routes_staged(backend, bufs, routes, 4, 6, object())
    # Routes in a tensor of the caller's own are staged by run_moe as before.
    own = torch.zeros(4, 6, dtype=torch.int32)
    assert not IMPL._routes_staged(backend, bufs, own, 4, 6, None)
