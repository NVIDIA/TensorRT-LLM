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
"""Engine names the Kimi K3 SM107 family code probes behind what its Rubin deployment runs.

The family code reaches main's engine through duck-typed probes (``hasattr``,
``getattr(..., None)``, ``inspect.signature``). A probe for a name the runtime object does not have
returns its fallback, so the path behind it stops working without an error. Each test pins the
probes of one path against the class the family code gets at runtime (the leaf ``create_moe``
builds, the KV cache manager the executor creates, ...), so a rename on either side fails here
rather than as a slower step in a perf run.

Here: the probes of the shared modules (``dspark/``, ``common/``, ``ops/``, ``kernels/``), the
target classes and constants records of all four SM107 targets, and the probes of the glue that the
tp1 target (``kimi_k3_nvfp4__sm_107__tp1``) and the helix8 targets (each its own copy of the helix8
model) share in shape (the routing tail, the MLA projection reads). The dep16 glue's probes are in
``kimi_k3_nvfp4__sm_107__dep16/test_dep16_k3_engine_probes.py``; each target's own hooks and
declarations are in its directory.

CPU only: classes and signatures, no kernels.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest
from torch import nn

K3_RUBIN = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
TARGETS = ("tp1", "helix8_moeep8", "helix8_moetp2ep4", "dep16")
#: The glue module of each SM107 target: the target's modeling.py.
_GLUE = {t: f"kimi_k3_nvfp4__sm_107__{t}.modeling" for t in TARGETS}
#: The glue modules with the CuTe routing tail and the FP8 weight-read MLA projections whose probes
#: this file pins (dep16's are in its own file).
_ROUTED_GLUES = (_GLUE["tp1"], _GLUE["helix8_moeep8"], _GLUE["helix8_moetp2ep4"])


def _k3_module(name):
    """A module of the Kimi K3 family code (``dspark.drafter``, ``kernels.k3r_spec...``)."""
    return importlib.import_module(f"{K3_RUBIN}.{name}")


def _bare(cls):
    """An instance of the ``nn.Module`` ``cls`` without running its constructor."""
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    return instance


def _is_property(cls, name):
    return isinstance(inspect.getattr_static(cls, name, None), property)


def _constants(target):
    """The ``K3Sm107Constants`` record of the SM107 target ``target``."""
    return _k3_module(f"kimi_k3_nvfp4__sm_107__{target}.modeling").CONSTANTS


_TARGET_CLASSES = {
    "tp1": "ModelingV2KimiK3Nvfp4Sm107Tp1",
    "helix8_moeep8": "ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8",
    "helix8_moetp2ep4": "ModelingV2KimiK3Nvfp4Sm107Helix8Moetp2ep4",
    "dep16": "ModelingV2KimiK3Nvfp4Sm107Dep16",
}


def _target_class(target):
    return getattr(_k3_module(f"kimi_k3_nvfp4__sm_107__{target}.modeling"), _TARGET_CLASSES[target])


def test_loader_does_not_take_model_defaults_from_k3():
    """The model loader takes model defaults from the class registered for the checkpoint's
    architecture, before it routes a Modeling V2 target. For Kimi K3 that class declares none, so
    every engine setting of a K3 deployment is the recipe's; the targets declare none either."""
    from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class

    builtin = get_registered_model_class("KimiK3ForConditionalGeneration")
    assert builtin is not None
    assert not hasattr(builtin, "get_model_defaults")
    for target in TARGETS:
        assert not hasattr(_target_class(target), "get_model_defaults"), target


@pytest.mark.parametrize("target", TARGETS)
def test_target_class_declares_its_record(target):
    """Each target class declares its own constants record as ``K3_CONSTANTS``."""
    assert _target_class(target).K3_CONSTANTS is _constants(target)


def test_spec_decode_head_reads_main_names():
    """``kernels.spec_decode_head`` (``decode_prologue`` on the generation targets) reads main's preparer
    and metadata directly: the overlap switch, the previous step's offsets, the helix buffers and
    the per-token validity flag. ``preprocess_inputs`` must run the prologue before its own head
    and read the same names, so the two write the same step."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    source = inspect.getsource(DecoderInputPreparer.preprocess_inputs)
    for name in (
        "self.decode_prologue",
        "self._config.disable_overlap_scheduler",
        "self._buffers.previous_pos_id_offsets_cuda",
        "self._buffers.previous_kv_lens_offsets_cuda",
        "num_chunked_ctx_requests",
        "recompute_helix_spec_buffers",
        "_helix_spec_tokens_valid",
    ):
        assert name in source, name
    assert source.index("self.decode_prologue") < source.index(
        "self._config.disable_overlap_scheduler"
    )
    assert "get_runtime_tokens_per_gen_step" in inspect.signature(DecoderInputPreparer).parameters
    assert "self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step" in (
        inspect.getsource(DecoderInputPreparer.__init__)
    )
    for name in (
        "_helix_spec_tokens_valid",
        "helix_position_offsets",
        "helix_local_slots",
        "helix_kv_bounds",
        "on_update_kv_lens",
        "recompute_helix_spec_buffers",
    ):
        assert hasattr(TrtllmAttentionMetadata, name), name


def test_routed_experts_tune_the_routing_tile_like_mains_runners():
    """The routed experts (every target but dep16) run their own kernel chains; the AutoTuner picks
    the routing tile through the family's copies of main's two CuteDSL outer runners
    (``ops/moe/k3r_experts``). The copies keep main's AutoTuner keys, ``unique_id``, routing tiles
    and tuning config, so a tactic cache written by either serves the other."""
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.fused_fc12 import CuteDslFc12FusedMoENvfp4Runner
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.runners import CuteDslFusedMoENvfp4Runner
    from tensorrt_llm._torch.utils import ActivationType

    kx = _k3_module("ops.moe.k3r_experts")
    assert kx.TILE_TUNER_KEYS == {
        "CUTEDSL": "CuteDslFusedMoE::run_moe_nvfp4",
        "CUTEDSL_FC12": "CuteDslFc12FusedMoE::run_moe_nvfp4",
    }
    geometry = dict(num_experts=896, top_k=16, num_local_experts=112, local_expert_offset=224)
    for kernels, mains, identity in (
        ("CUTEDSL", CuteDslFusedMoENvfp4Runner, None),
        ("CUTEDSL_FC12", CuteDslFc12FusedMoENvfp4Runner, (int(ActivationType.SiTu), 4.0, 25.0)),
    ):
        ours = kx._tile_runner_class(kernels)(
            forward_impl=None, workload_identity=identity, **geometry
        )
        theirs = mains(
            forward_impl=None,
            enable_finalize_fusion=True,
            enable_alltoall=False,
            workload_identity=identity,
            **geometry,
        )
        assert ours.unique_id() == theirs.unique_id(), kernels
        assert ours.tile_size_of(-1) == theirs.tile_size_of(-1) == 128
        assert ours.tile_size_of(256) == theirs.tile_size_of(256) == 256
        a, b = ours.get_tuning_config(), theirs.get_tuning_config()
        for spec_a, spec_b in zip(a.dynamic_tensor_specs, b.dynamic_tensor_specs, strict=True):
            assert (spec_a.input_idx, spec_a.dim_idx) == (spec_b.input_idx, spec_b.dim_idx)
            assert spec_a.gen_tuning_buckets is spec_b.gen_tuning_buckets
            assert spec_a.map_to_tuning_buckets is spec_b.map_to_tuning_buckets
        assert [(c.input_idx, c.dim_idx) for c in a.constraint_specs] == [
            (c.input_idx, c.dim_idx) for c in b.constraint_specs
        ]
        assert a.use_cold_l2_cache == b.use_cold_l2_cache
    assert kx._tile_runner_class("CUTEDSL_FC12").tile_sizes() == (
        CuteDslFc12FusedMoENvfp4Runner._tile_sizes()
    )


def test_latent_down_fusion_feeds_the_fc12_experts():
    """``ld_fused`` (both helix8 targets, folded into their helix8 model): the latent down
    projection hands the routed experts its own NVFP4 output only where they run the FC12 kernels,
    whose input is quantized with ``fp4_quantize`` against ``fc31_input_scale``, as
    k3_latent_down_fp4 quantizes it. tp1 and dep16 do not fuse it."""
    assert {t for t in TARGETS if _constants(t).ld_fused} == {"helix8_moeep8", "helix8_moetp2ep4"}
    for target in ("helix8_moeep8", "helix8_moetp2ep4"):
        source = inspect.getsource(_k3_module(_GLUE[target]).K3MoERuntime._fused_latent_input)
        assert '"CUTEDSL_FC12"' in source and "fc31_input_scale" in source, target
    for glue in (_GLUE["tp1"], _GLUE["dep16"]):
        runtime = _k3_module(glue).K3MoERuntime
        assert not hasattr(runtime, "_fused_latent_input"), glue


def test_router_precision_and_megamoe_follow_the_target():
    """BF16 router GEMM on tp1 and both helix8 targets, FP32 on dep16 (attention DP); only dep16
    builds MegaMoE routed experts, from its own module."""
    assert {t for t in TARGETS if _constants(t).router_bf16_gate} == {
        "tp1",
        "helix8_moeep8",
        "helix8_moetp2ep4",
    }
    assert {t for t in TARGETS if _constants(t).megamoe_experts is not None} == {"dep16"}
    megamoe = _constants("dep16").megamoe_experts
    assert megamoe.__module__ == f"{K3_RUBIN}.kimi_k3_nvfp4__sm_107__dep16.modeling"
    params = inspect.signature(megamoe.__init__).parameters
    for name in (
        "num_experts",
        "top_k",
        "hidden_size",
        "intermediate_size",
        "mapping",
        "layer_idx",
        "situ_beta",
        "situ_linear_beta",
        "max_num_tokens",
        "dtype",
        "skip_create_weights_in_init",
    ):
        assert name in params, name


def test_private_persistent_grids_take_the_shared_sm_reserve(monkeypatch):
    """``TRTLLM_DWDP_PREFETCH_GATHER`` (tp1 context): the DWDP gather's resident SMs reach
    persistent grids through ``cute_dsl_utils.sm_reserve()`` (``DwdpManager`` sets it before the
    model is built). The KDA prefill grid sizes itself by it and the persistent attention-residual
    op of every target's glue takes it as its ``sm_reserve`` argument."""
    import torch

    from tensorrt_llm._torch import cute_dsl_utils

    for glue in sorted(set(_GLUE.values())):
        assert _k3_module(glue).sm_reserve is cute_dsl_utils.sm_reserve, glue
    schema = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd.default._schema
    assert [arg.name for arg in schema.arguments][-1] == "sm_reserve"
    prefill_ops = pytest.importorskip(f"{K3_RUBIN}.ops.ssm._kda_prefill")
    monkeypatch.setattr(cute_dsl_utils, "_sm_reserve", 16)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=212),
    )
    assert prefill_ops._kda_persistent_grid_blocks(torch.device("cuda", 0)) == 196


def test_fp8_kv_layer_selector_on_the_cache_policy():
    """The MLA layers of every target's glue take their FP8 KV-cache fallback layers from the
    selector of ``kimi_k3_cache_policy`` that the cache manager reads too."""
    from tensorrt_llm._torch import kimi_k3_cache_policy

    selector = getattr(kimi_k3_cache_policy, "get_kimi_k3_fp8_kv_layer_ids", None)
    assert callable(selector)
    assert "nvfp4_enabled" in inspect.signature(selector).parameters
    for glue in sorted(set(_GLUE.values())):
        assert _k3_module(glue)._kimi_k3_cache_policy is kimi_k3_cache_policy, glue


def test_fp4_mla_probe_on_the_family_mla_state():
    """The FP4 MLA paths (packed decode, split-K context) key off ``uses_fp4_mla_attention`` of
    the layer's ``K3MlaAttnState`` roles, the family's in every target's glue."""
    state = _k3_module("ops.attention.k3_mla_state").K3MlaAttnState
    assert _is_property(state, "uses_fp4_mla_attention")
    for glue in sorted(set(_GLUE.values())):
        assert _k3_module(glue).K3MlaAttnState is state, glue


def test_dspark_unified_draft_cache_probes_on_the_k3_hybrid_manager():
    """DSpark's drafter context KV in the target manager (``TRTLLM_DISAGG_SHARED_DRAFT_KV``).
    The engine appends the drafter's layers to
    ``KimiK3HybridCacheManager`` only when the class opts in, and only when the loaded drafter
    says it reads the manager's pages (``reads_managed_ctx_pages`` /
    ``reads_helix_sharded_ctx_pages``, which the family drafter declares). The family
    worker finds the layers through ``draft_layer_ids`` and ``get_buffers(..., kv_layout=...)``.
    Main's ``DFlashSpecMetadata.prepare`` hands a disaggregated generation request's new slot to
    the worker's ``_seed_generation_slot``, and the family worker seeds the slot's context
    length only when it reports ``_ctx_unified_cache``. The KDA bookkeeping after accept runs
    only when the manager is a ``MambaHybridCacheManager``."""
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    assert issubclass(KimiK3HybridCacheManager, MambaHybridCacheManager)
    assert issubclass(KimiK3HybridCacheManager, KVCacheManagerV2)
    if not hasattr(KimiK3HybridCacheManager, "accepts_appended_draft_layers"):
        pytest.skip("main has no drafter layers in the target manager (unified draft KV)")
    assert KimiK3HybridCacheManager.accepts_appended_draft_layers() is True
    assert _is_property(KVCacheManagerV2, "draft_layer_ids")
    assert "kv_layout" in inspect.signature(KVCacheManagerV2.get_buffers).parameters
    for name in ("get_layer_kv_factor", "get_layer_page_index_scale", "get_draft_kv_cache_view"):
        assert callable(getattr(KVCacheManagerV2, name, None)), name
    assert "_seed_generation_slot" in inspect.getsource(DFlashSpecMetadata.prepare)
    worker = _k3_module("dspark.worker").K3DSparkWorker
    assert "_ctx_unified_cache" in inspect.getsource(worker._seed_generation_slot)
    drafter = _k3_module("dspark.drafter").K3DSparkDrafter
    assert callable(drafter.reads_managed_ctx_pages)
    assert _is_property(drafter, "reads_helix_sharded_ctx_pages")


def test_kda_inforward_snapshot_plan_on_the_private_metadata():
    """In-forward KDA snapshots (context, block reuse): ``KimiK3HybridCacheManager`` plans them
    whenever main's KDA kernels are present, whatever the model, and
    ``finish_inforward_snapshots`` then commits the prefix from the live slot and copies the
    staging rows back. ``Mamba2Metadata.prepare`` hands the plan to
    ``_prepare_inforward_split``, which the family's step metadata (``common/kda_metadata``, the
    ``mamba_metadata_cls`` of every target) turns into a ``kda_snapshot`` plan on the live slots
    with the plan's staging rows; the KDA layers read those rows through
    ``inforward_snapshot_stage_views``."""
    import torch

    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        InforwardSnapshotEntry,
        MambaHybridCacheManagerV2,
    )

    assert issubclass(KimiK3HybridCacheManager, MambaHybridCacheManagerV2)
    for name in ("inforward_snapshot_stage_views", "state_index", "finish_inforward_snapshots"):
        assert callable(getattr(MambaHybridCacheManagerV2, name, None)), name
    assert "_prepare_inforward_split(" in inspect.getsource(Mamba2Metadata.prepare)
    metadata_cls = _k3_module("common.kda_metadata").K3MambaMetadata
    for target in TARGETS:
        text_cls = _k3_module(_GLUE[target]).KimiK3DefaultForCausalLM
        assert text_cls.mamba_metadata_cls is metadata_cls, target
        assert _target_class(target).mamba_metadata_cls is metadata_cls, target
    assert list(inspect.signature(metadata_cls._prepare_inforward_split).parameters) == list(
        inspect.signature(Mamba2Metadata._prepare_inforward_split).parameters
    )

    metadata = metadata_cls.__new__(metadata_cls)
    metadata._state_indices_long = torch.zeros(4, dtype=torch.long)
    metadata.kda_snapshot = None
    manager = SimpleNamespace(conv_state_shape=(12, 3), state_index={11: 7, 13: 2}.__getitem__)
    attn_metadata = SimpleNamespace(seq_lens=torch.tensor([200, 130, 96]), request_ids=[11, 12, 13])
    plan = {
        11: InforwardSnapshotEntry(row=1, point=1152, num_before=128, num_after=72),
        13: InforwardSnapshotEntry(row=0, point=64, num_before=64, num_after=32),
    }
    metadata._prepare_inforward_split(attn_metadata, manager, plan, 3, 3)
    snapshot = metadata.kda_snapshot
    assert snapshot.num_snapshots == 2
    assert snapshot.slots.tolist() == [7, 2]
    assert snapshot.stage_rows.tolist() == [1, 0]
    assert snapshot.token_indices.tolist() == [128, 330 + 64]
    assert snapshot.chunks.tolist() == [2, -1, 1]
    with pytest.raises(RuntimeError, match="generation requests"):
        metadata._prepare_inforward_split(attn_metadata, manager, plan, 3, 4)
    misaligned = {13: InforwardSnapshotEntry(row=0, point=48, num_before=48, num_after=48)}
    with pytest.raises(RuntimeError, match="does not match the plan"):
        metadata._prepare_inforward_split(attn_metadata, manager, misaligned, 3, 3)


def test_dspark_latent_cache_helix_mapping():
    """The DSpark drafter under Helix with ``TRTLLM_DISAGG_SHARED_DRAFT_KV=1`` (generation): main
    unifies the draft KV with a CP > 1 target only when the drafter reports
    ``reads_helix_sharded_ctx_pages`` (``_dflash_unify_blocker``), which the family drafter does
    when its latent cache is page-sharded. The sharding comes from the target's helix CP mapping:
    tp = 1 and cp over the repurposed TP group, on the target's CP ranks, with the NCCL all-to-all
    exchange."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
    from tensorrt_llm.mapping import CpType, Mapping

    shard = _k3_module("dspark.drafter")._latent_cache_helix_mapping
    if hasattr(KvCacheCreator, "_unified_drafter_reads_helix_pages"):
        assert "_unified_drafter_reads_helix_pages()" in inspect.getsource(
            KvCacheCreator._dflash_unify_blocker
        )
        assert "reads_helix_sharded_ctx_pages" in inspect.getsource(
            KvCacheCreator._unified_drafter_reads_helix_pages
        )
    else:
        # Main without the helix unified draft KV refuses to unify a CP > 1 target; the drafter
        # then keeps its own manager and the sharding below serves its latent cache alone.
        assert "cp_size" in inspect.getsource(KvCacheCreator._dflash_unify_blocker)
    for rank in (0, 5):
        helix = Mapping(
            world_size=8,
            rank=rank,
            gpus_per_node=4,
            tp_size=1,
            cp_size=8,
            cp_config={"cp_type": CpType.HELIX, "use_nccl_for_alltoall": False},
        )
        repurposed = helix.repurpose_helix_cp_to_tp()
        sharded = shard(repurposed, helix)
        assert (sharded.tp_size, sharded.cp_size, sharded.pp_size) == (1, 8, 1)
        assert sharded.has_cp_helix()
        assert (sharded.cp_rank, sharded.rank) == (helix.cp_rank, rank)
        assert sharded.cp_config == {"cp_type": CpType.HELIX}
        assert shard(repurposed, None) is None
    plain = Mapping(world_size=8, rank=0, gpus_per_node=4, tp_size=8)
    assert shard(plain, None) is None


@pytest.mark.parametrize("glue", _ROUTED_GLUES)
def test_router_picks_the_routing_kernel(glue):
    """The routing each forward hands the routed experts (``K3MoERuntime._router``): the CuTe
    routing tail where a target is set, with its ``moe_sort`` tables where they are of the experts
    asked for; else ``noaux_tc_op`` (the experts then run ``moe_sort``). dep16 always routes with
    ``noaux_tc_op`` (its own test)."""
    import torch

    moe_runtime = _k3_module(glue)
    routing_types = _k3_module("ops.moe.k3r_experts")
    assert _k3_module("kernels.k3r_moe.k3_route_tail").TILE == 128
    assert not hasattr(moe_runtime.K3MoEGate, "route_with_sort")
    ids, weights = torch.zeros(2, 16, dtype=torch.int32), torch.ones(2, 16)
    tables = routing_types.SortTables(
        128, tuple(torch.zeros(1, dtype=torch.int32) for _ in range(6))
    )
    calls = []

    def route_tail(logits, offset, num):
        calls.append(("tail", offset, num))
        return routing_types.SortedRouting(ids, weights, tables)

    def route(logits):
        calls.append(("topk",))
        return ids, weights

    runtime = _bare(moe_runtime.K3MoERuntime)
    runtime.gate = SimpleNamespace(route=route, route_tail=route_tail)
    runtime.routed_experts = SimpleNamespace(consumes_sort_tables=True)
    logits = torch.empty(2, 896)

    assert runtime._router((0, 896))(logits, 0, 896).sort_tables is tables
    assert runtime._router((0, 896))(logits, 448, 448).sort_tables is None
    assert runtime._router(None)(logits, 0, 896).sort_tables is None
    assert calls == [("tail", 0, 896), ("tail", 0, 896), ("topk",)]
    calls.clear()
    runtime.routed_experts = SimpleNamespace(consumes_sort_tables=False)
    assert runtime._router(None)(logits, 0, 896).sort_tables is None
    assert calls == [("topk",)]


#: This rank's routed experts, per target with the routing tail (none runs attention DP; dep16, the
#: attention-DP target, never uses the tail and has its own test).
_ROUTED_LOCAL = {"tp1": 896, "helix8_moeep8": 112, "helix8_moetp2ep4": 224}


@pytest.mark.parametrize("target", sorted(_ROUTED_LOCAL))
def test_route_tail_serves_the_reference_sizes(monkeypatch, target):
    """Which forwards route with ``k3_route_tail`` (``K3MoERuntime._route_tail_target``): the routed
    experts not attention-DP, 1 to 2048 tokens, up to 256 on one cluster and above that where the
    multi-CTA sort fits this rank's experts. Every other forward routes with ``noaux_tc_op`` and the
    experts' ``moe_sort``. So tp1 (896 local experts) routes 1..256 tokens with the tail and more
    with ``noaux_tc_op`` + ``moe_sort``; the helix8 targets use the tail up to the sort's limit."""
    import torch

    moe_runtime = _k3_module(_GLUE[target])
    route_tail = _k3_module("kernels.k3r_moe.k3_route_tail")
    experts_cls = _k3_module("ops.moe.k3r_experts").K3RoutedExperts
    monkeypatch.setattr(route_tail, "supports_device", lambda device: True)
    monkeypatch.setattr(route_tail, "wide_workspace", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    local = _ROUTED_LOCAL[target]
    runtime = _bare(moe_runtime.K3MoERuntime)
    runtime.gate = SimpleNamespace(num_experts=896, e_score_correction_bias=torch.empty(896))
    runtime.routed_experts = _bare(experts_cls)
    runtime.routed_experts.backend = SimpleNamespace(
        slot_start=0, expert_size_per_partition=local, num_slots=896
    )
    # tp1's glue still reads the attention-DP flag (its context servers may run either); the
    # helix8 glue refuses attention DP at construction and reads none.
    runtime._attention_dp = False
    for m in (1, 16, 64, 256, 257, 512, 1024, 2048, 2049):
        tail = 1 <= m <= 2048 and (m <= route_tail.MAX_M or route_tail.wide_sort_fits(m, local))
        assert runtime._route_tail_target(m) == ((0, local) if tail else None), m
    if target == "tp1":
        assert runtime._route_tail_target(256) == (0, 896)
        assert runtime._route_tail_target(257) is None
        runtime._attention_dp = True
        assert runtime._route_tail_target(1) is None


def test_dspark_split_plan_planner():
    """The DSpark drafter's split-KV plan (generation) runs the planner kernel of
    ``kernels/k3r_spec/mla_split_plan.py`` at the drafter's piece cost (``_launch_mla_split_plan``)
    on the CuTe DSL MLA decode runner's geometry (``_mla_split_plan_geometry``): the kernel takes
    the piece cost as an argument, and the runner exposes what the geometry reads."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    drafter = _k3_module("dspark.drafter")
    plan = _k3_module("kernels.k3r_spec.mla_split_plan")
    params = inspect.signature(plan._mla_split_kv_plan_kernel.fn).parameters
    for name in ("PIECE_COST", "MAX_PIECES", "BLOCK", "KV_TILE", "MIN_TILES"):
        assert name in params, name
    for name in (
        "_SPLIT_PLAN_MAX_REQUESTS",
        "_SPLIT_PLAN_MAX_SPLIT",
        "_SPLIT_PLAN_KV_TILE",
        "_SPLIT_PLAN_MIN_TILES",
    ):
        assert isinstance(getattr(plan, name, None), int), name
    runner = CuteDSLNVMlaDecodeBlackwellRunner
    assert list(inspect.signature(runner.get_num_q_tiles).parameters) == ["num_heads", "seq_len_q"]
    assert callable(runner._get_max_active_blocks)
    assert runner._CLUSTER_SHAPE_MNK[0] >= 1
    assert isinstance(drafter._MLA_SPLIT_PLAN_PIECE_COST_TILES, int)
    assert drafter._MLA_SPLIT_PLAN_PIECE_COST_TILES > 0


def test_dspark_drafter_declares_its_slot_convention():
    """DSpark's ``shift_label`` slot convention (generation): the engine sizes the unified draft
    KV from ``draft_model.shift_label`` (``get_draft_block_width``, K slots instead of K + 1),
    which the drafter's DSpark heads set."""
    drafter = _k3_module("dspark.drafter")
    assert issubclass(drafter.K3DSparkDrafter, drafter._DSparkHeadMixin)
    heads = drafter._DSparkHeadMixin.__new__(drafter._DSparkHeadMixin)
    heads._init_dspark_heads(SimpleNamespace())
    assert heads.shift_label is True
    heads._init_dspark_heads(SimpleNamespace(dflash_config={"shift_label": False}))
    assert heads.shift_label is False


def test_dspark_worker_takes_the_shared_slot_calls():
    """Main's ``DFlashSpecMetadata.prepare()`` drives the drafter's context slots by calling
    methods on the worker it is bound to, and on K3 that is the family ``K3DSparkWorker``: every
    ``worker.<method>(...)`` call there binds to the family method of that name. A missing
    method or a changed signature raises ``TypeError`` on the first step after the context
    buffers exist, so no DSpark run starts."""
    import ast
    import textwrap

    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata

    tree = ast.parse(textwrap.dedent(inspect.getsource(DFlashSpecMetadata.prepare)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "worker"
    ]
    names = {call.func.attr for call in calls}
    assert {"_assign_slot", "_write_ctx_len"} <= names, names

    worker = _k3_module("dspark.worker").K3DSparkWorker
    unbound = []
    for call in calls:
        method = getattr(worker, call.func.attr, None)
        if not callable(method):
            unbound.append((call.func.attr, "missing"))
            continue
        args = [object()] * (1 + len(call.args))
        kwargs = {kw.arg: object() for kw in call.keywords if kw.arg is not None}
        try:
            inspect.signature(method).bind(*args, **kwargs)
        except TypeError as error:
            unbound.append((call.func.attr, str(error)))
    assert not unbound, unbound


# The FP8 block-scale projections of K3's MLA layer and the KDA layer's o_proj; the weight load
# swaps each one for the glue's ``_Fp8BlockScaleWeightReadLinear``.
_K3_MLA_PROJECTIONS = ("kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj", "g_proj")
# Read through ``getattr(proj, name, None)``; the bias-free FP8 projection has none.
_OPTIONAL_PROJECTION_READS = frozenset({"bias"})
#: Reads each glue must make (the paths its MLA / KDA chains still run).
_EXPECTED_READS = {
    _GLUE["tp1"]: {
        "weight",
        "weight_scale",
        "in_features",
        "supports_prequantized_input",
        "forward_prequantized",
        "takes_prequantized_input",
    },
    **{
        _GLUE[target]: {
            "prepare_output_slice",
            "forward_slice",
            "forward_mxfp8",
            "forward_gated_mxfp8",
            "gated_mxfp8_applies",
            "takes_mid_mxfp8_input",
            "supports_prequantized_input",
            "fp8w_lowm_partial_n_tile",
            "forward_prequantized",
            "takes_prequantized_input",
            "next_ar_push",
        }
        for target in ("helix8_moeep8", "helix8_moetp2ep4")
    },
}


def _projection_reads(glue):
    import re

    source = inspect.getsource(_k3_module(glue))
    names = "|".join(_K3_MLA_PROJECTIONS)
    reads = set(re.findall(rf"(?:self|layer)\.(?:{names})\.(\w+)\b", source))
    reads |= set(re.findall(r"\bproj\.(\w+)\b", source))
    reads |= set(
        re.findall(rf"getattr\((?:(?:self|layer)\.(?:{names})|kv_b_proj), \"(\w+)\"", source)
    )
    kda = source
    reads |= set(re.findall(r"self\.o_proj\.(\w+)\b", kda))
    reads |= set(re.findall(r"getattr\((?:self\.)?o_proj, \"(\w+)\"", kda))
    return reads


@pytest.mark.parametrize("glue", _ROUTED_GLUES)
def test_fp8_weight_read_linear_serves_the_k3_projection_reads(glue):
    """The glue's MLA chains and the KDA MTP verify's o_proj read attributes and methods of their
    projections, and on K3 those are the glue's ``_Fp8BlockScaleWeightReadLinear``: every one read
    exists on it. ``K3MLAAttention.cache_derived_state`` calls ``prepare_output_slice`` at
    post-load on every Helix rank, so one missing there stops every Helix GEN worker before it
    serves; elsewhere a missing one silently drops the path behind a ``getattr`` probe."""
    import torch

    reads = _projection_reads(glue)
    assert _EXPECTED_READS[glue] <= reads, reads

    linear = _k3_module(glue)._Fp8BlockScaleWeightReadLinear
    projection = linear(torch.zeros(256, 128, dtype=torch.float8_e4m3fn), torch.ones(2, 1), 256)
    # Set by ``from_linear``, the constructor K3's swap uses.
    projection.quant_config, projection.tp_size, projection.tp_rank = None, 1, 0
    missing = sorted(
        attr
        for attr in reads
        if attr not in _OPTIONAL_PROJECTION_READS and not hasattr(projection, attr)
    )
    assert not missing, missing
