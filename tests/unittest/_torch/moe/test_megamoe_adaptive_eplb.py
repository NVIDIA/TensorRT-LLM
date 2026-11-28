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
    """An adaptive route launch's DSL scalars are made once per value, not per launch: a layer
    launches with the same device addresses and few row counts."""
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

    first = op._adaptive_route_scalar(float, 1 << 40)  # float() makes a new object on every call
    assert op._adaptive_route_scalar(float, 1 << 40) is first and first == float(1 << 40)
    assert op._adaptive_route_scalar(float, 8) == 8.0


def test_only_a_bucket_whose_kernel_has_the_adaptive_route_maps_routes_in_its_router(monkeypatch):
    """A bucket whose kernel has no adaptive route in its dispatch router (an SM107 generic
    bucket on the dedup combine, for one): its adaptive layers route through the adaptive EPLB
    route op, with a log line naming the bucket."""
    logged = []
    monkeypatch.setattr(mm, "IS_CUTLASS_DSL_AVAILABLE", True)
    monkeypatch.setattr(mm, "get_sm_version", lambda: 107)
    monkeypatch.setattr(mm.logger, "info_once", lambda *msg, key: logged.append((msg, key)))
    backend = SimpleNamespace(
        _genphase_direct_input_buckets=set(),
        _padding_drop_buckets=set(),
        _adaptive_router_buckets=set(),
        layer_load_balancer=SimpleNamespace(router_route=object()),
        _slot_of_expert=None,
        _route_quant_fuses=lambda: True,
        _router_launch_max_tokens=lambda num_tokens: 2048,
    )
    for bucket, has_route in ((2048, False), (4096, True)):
        properties = SimpleNamespace(
            supports_direct_inputs=False, masks_invalid_routes=True, routes_adaptive=has_route
        )
        IMPL._record_bucket_kernel(backend, bucket, properties)
    assert backend._adaptive_router_buckets == {4096}
    assert backend._padding_drop_buckets == {2048, 4096}
    assert [key for _, key in logged] == [("megamoe_no_adaptive_router", 2048)]
    assert "SM107 kernel of bucket 2048 has no adaptive route" in logged[0][0][0]
    assert not IMPL.routes_adaptive_in_router(backend, 8)
    backend._router_launch_max_tokens = lambda num_tokens: 4096
    assert IMPL.routes_adaptive_in_router(backend, 8)


def test_home_slots_launch_the_kernel_and_weights_of_a_layer_without_a_balancer():
    """A layer whose balancer is off launches as without one: the kernel of the experts per rank,
    its bucket properties, and leading views (no copy) of the weight tensors, whose leading axis
    is the slot: the home slots."""
    backend = SimpleNamespace(
        num_experts=8,
        ep_size=2,
        expert_size_per_partition=6,
        layer_load_balancer=None,
        _home_slots=IMPL._home_slots,  # the class default a subclass without __init__ reads
        _maxt_buckets=[256, 1024],
        _genphase_direct_input_buckets={256},
        _padding_drop_buckets={256},
        _adaptive_router_buckets={256},
    )
    slot_count = lambda: IMPL.compute_slot_count.fget(backend)  # noqa: E731
    assert slot_count() == 6
    asked = []

    def properties(max_T, world_size):
        asked.append((max_T, world_size, slot_count()))
        big = max_T > 256
        return SimpleNamespace(
            supports_direct_inputs=big, masks_invalid_routes=big, routes_adaptive=False
        )

    backend._kernel_properties = properties
    backend._record_bucket_kernel = lambda max_T, p: IMPL._record_bucket_kernel(backend, max_T, p)
    IMPL.serve_home_slots(backend)
    assert slot_count() == 4 and asked == [(256, 2, 4), (1024, 2, 4)]
    assert backend._genphase_direct_input_buckets == backend._padding_drop_buckets == {1024}
    assert not backend._adaptive_router_buckets
    names = ("mega_fc1_weight", "mega_fc1_weight_sf", "mega_fc2_weight", "mega_fc2_weight_sf")
    names += ("fc31_alpha", "fc2_alpha", "fc1_norm_const")
    for name in names:
        setattr(backend, name, torch.arange(6 * 3).view(6, 3))
    view = IMPL._build_weight_view(backend)
    for name, tensor in zip(names, vars(view).values()):
        full = getattr(backend, name)
        assert tensor.shape[0] == 4 and tensor.data_ptr() == full.data_ptr()
        assert torch.equal(tensor, full[:4])


def test_an_sm107_generation_phase_bucket_takes_the_adaptive_route_in_its_kernel(monkeypatch):
    """An SM107 generation-phase bucket reads its inputs where the fused route+quant leaves them and
    its kernel maps and counts the expert ids itself: its adaptive layers skip the route op too. A
    staged bucket the fused launch cannot write still routes through it."""
    monkeypatch.setattr(mm, "IS_CUTLASS_DSL_AVAILABLE", True)
    backend = SimpleNamespace(
        _active_launch_max_tokens=128,
        tactic_autotune=False,
        _select_launch_max_tokens=lambda hint: hint,
        _genphase_direct_input_buckets={128},
        _adaptive_router_buckets=set(),
        _padding_drop_buckets=set(),
        _slot_of_expert=None,
        _replicated=False,
        _route_quant_fuses=lambda: True,
        _route_quant_bucket=lambda *args: None,
        _shared_activation_target=lambda num_tokens: None,
        _adaptive_route_launch=None,
        routing_method=SimpleNamespace(
            top_k=6, e_score_correction_bias=None, routed_scaling_factor=2.5
        ),
        fc31_input_scale=None,
    )
    for name in ("_router_launch_max_tokens", "routes_adaptive_in_router"):
        setattr(backend, name, getattr(IMPL, name).__get__(backend))
    assert backend._router_launch_max_tokens(64) == 128
    assert backend._router_launch_max_tokens(256) is None
    assert not backend.routes_adaptive_in_router(64)  # the bucket's kernel has no adaptive route
    backend._adaptive_router_buckets.add(128)
    assert backend.routes_adaptive_in_router(64) and not backend.routes_adaptive_in_router(0)

    def route_quant(*args, staged=None, slot_of_expert=None, **kwargs):
        # Expert ids, left where the kernel reads them.
        assert staged is None and slot_of_expert is None
        ids = torch.zeros((args[2].shape[0], 6), dtype=torch.int32)
        return ids, torch.zeros(ids.shape), args[2], args[2], None

    monkeypatch.setattr(mm, "dsv4_route_nvfp4_quant", route_quant, raising=False)
    x, logits = torch.zeros((64, 8)), torch.zeros((64, 384))
    route = ("table", "rows", "fw", "on", True)
    result = IMPL.try_fused_route_quant(backend, x, logits, adaptive_route=route)
    assert result is not None and result.routes_mapped
    assert backend._adaptive_route_launch == ("table", "rows", "fw", "on", 64, True)
    # A staged bucket the fused launch cannot write (no hint yet, say) keeps the route op.
    backend._genphase_direct_input_buckets = set()
    backend._adaptive_route_launch = None
    assert IMPL.try_fused_route_quant(backend, x, logits, adaptive_route=route) is None
    assert backend._adaptive_route_launch is None
