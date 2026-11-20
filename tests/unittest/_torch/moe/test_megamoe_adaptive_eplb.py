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


def test_decode_time_model_streams_the_expert_bytes_and_prices_the_row_flops():
    bandwidth, peak = 8.0e12, 15.0e15  # any rates: a calibration gives them
    hidden, expand, inner = 7168, 6144, 3072
    backend = SimpleNamespace(
        hidden_size=hidden,
        expand_intermediate_size_per_partition=expand,
        intermediate_size_per_partition=inner,
    )
    model = TimeModel(**IMPL.decode_time_model(backend, 37158924, (bandwidth, peak), tile_rows=256))
    # A slot with rows streams its expert once; a row costs its FC1 and FC2 FLOPs at the peak.
    assert model.per_active_slot_us == pytest.approx(37158924 / bandwidth * 1e6)
    row_us = 2 * hidden * (expand + inner) / peak * 1e6
    assert model.per_row_us == pytest.approx(row_us)
    # A tile is 256 rows; every extra tile of the busiest slot costs half a tile more.
    assert model.tile_rows == 256 and model.per_extra_tile_us == pytest.approx(256 * row_us)
    assert model.hot_slot_extra_tile_us == pytest.approx(128 * row_us)
    assert model.scale == 1.0  # the calibration's, per bucket


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
        ep_size=4,
        compute_slot_count=8,
        num_experts=6,
    )
    backend._decode_tactic = lambda tokens: IMPL._decode_tactic(backend, tokens)
    monkeypatch.setattr(mm, "get_sm_version", lambda: 103)
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

    # A quantized (NVFP4) combine resolves to the quantized-combine tactic, and from the bulk
    # threshold's tokens per rank on to the bulk-store tactic, as the launch does.
    small = repr(op._unpack_tactic(op._MEGAMOE_QUANTIZED_COMBINE_TACTIC))
    bulk = repr(op._unpack_tactic(op._MEGAMOE_NVFP4_COMBINE_BULK_TACTIC))
    tokens = [32, 1024, op._MEGAMOE_NVFP4_COMBINE_BULK_MIN_TOKENS]
    identity = IMPL.decode_calibration_identity(backend, tokens)
    assert identity["tactics"] == dict(zip(map(str, tokens), (small, small, bulk)))
    assert (identity["num_experts"], identity["expand_intermediate_size"]) == (6, 64)  # geometry
    assert identity["top_k"] == 6 and identity["launch_buckets"] == dict.fromkeys(
        map(str, tokens), 256
    )

    def ladder(tokens):  # as the backend's: no launch takes more tokens than its top bucket
        if tokens > 4096:
            raise RuntimeError("exceeds the top adaptive bucket")
        return 256

    backend._select_launch_max_tokens = ladder
    assert list(IMPL.decode_calibration_identity(backend, [32, 8192])["tactics"]) == ["32"]
    # Autotuned, a bucket takes the tactic its last launch ran (the autotuner's choice), tile and
    # all; before any launch it has none.
    backend.tactic_autotune = True
    identity = IMPL.decode_calibration_identity(backend, [32])
    assert (identity["tactics"], identity["tile_rows"]) == ({"32": "autotune"}, {"32": None})
    key = op.megamoe_launch_key(4, 6, 8, 64, 32, 256, "16e2m1xbf16")
    tuned = op._unpack_tactic(op._MEGAMOE_NVFP4_COMBINE_BULK_TACTIC)
    monkeypatch.setitem(op._MEGAMOE_LAUNCHED_TACTICS, key, tuned)
    identity = IMPL.decode_calibration_identity(backend, [32])
    assert (identity["tactics"], identity["tile_rows"]) == ({"32": bulk}, {"32": tuned[0][0]})


def test_an_adaptive_route_launch_builds_its_dsl_scalars_once():
    """An adaptive route launch's DSL scalars are made once per value, not per launch: building them
    per launch cost about 80 us of host time per layer call (cProfile of the eager path)."""
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

    first = op._adaptive_route_scalar(float, 1 << 40)  # float() makes a new object on every call
    assert op._adaptive_route_scalar(float, 1 << 40) is first and first == float(1 << 40)
    assert op._adaptive_route_scalar(float, 8) == 8.0
