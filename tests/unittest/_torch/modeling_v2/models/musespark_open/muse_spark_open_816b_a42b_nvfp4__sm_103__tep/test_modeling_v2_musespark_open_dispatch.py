# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's fast-path plan (``dispatch.py``): the paths, the compositions, ``compose``.

Host-only. ``FAST_PATHS`` is closed and acyclic; each composition is closed and
carries its topology; a ``Plan`` is always closed under ``depends_on``. A
deployment without a composition (GPU architecture, tensor-parallel size,
topology, speculative decoding, checkpoint layout) gets ``NO_PATHS`` under
``auto`` and an error under
``require``; a deployment with one gets all of it or, when a precondition is
missing, an error under both modes, never a part: every combination of missing
preconditions, and every combination of the public switches they read (with
leftover target knobs set to garbage), is checked under ``auto`` and
``require``. A windowed drafter context arena leaves out exactly the fast paths
written for a full-length one. The shell raises under ``require`` before the
built-in constructor runs.
"""

from __future__ import annotations

import dataclasses
import itertools
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch as tp_agreement,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    runtime as _runtime,
)
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)
from tensorrt_llm.llmapi.llm_args import (
    DFlashDecodingConfig,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


_SM103, _SM107 = (10, 3), (10, 7)


@pytest.fixture(autouse=True)
def _no_runtime_hooks(monkeypatch):
    """The executor hooks pin host staging blocks; these shells are built without them."""
    monkeypatch.setattr(_runtime, "make_runtime_hooks", lambda model: None)


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """The shells here are built as on an sm103 device, whatever device runs the test."""
    monkeypatch.setattr(modeling, "device_compute_capability", lambda: _SM103)


_AUTO, _REQUIRE = ModelingV2Mode.AUTO, ModelingV2Mode.REQUIRE
_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
_HOOKS = {"construct", "load", "draft", "executor"}
_DUMMY_MODEL = "/tmp/dummy_model"


def _tep(tp: int, gpus_per_node: int = 4) -> Mapping:
    return Mapping(world_size=tp, tp_size=tp, moe_ep_size=tp, gpus_per_node=gpus_per_node)


def _dflash(max_draft_len: int = 7, **kwargs) -> DFlashDecodingConfig:
    return DFlashDecodingConfig(max_draft_len=max_draft_len, **kwargs)


def _config(tp: int = 8, spec=None, gpus_per_node: int = 4, quant=QuantAlgo.NVFP4):
    return SimpleNamespace(
        mapping=_tep(tp, gpus_per_node),
        spec_config=spec,
        quant_config=SimpleNamespace(quant_algo=quant),
    )


_SERVED = _config(8, _dflash(7))


# --------------------------------------------------------------------------
# The fast paths and the composition
# --------------------------------------------------------------------------
def test_the_fast_paths_are_named_once_with_known_hooks_and_dependencies():
    names = [path.name for path in dispatch.FAST_PATHS]
    assert len(names) == len(set(names)) == len(dispatch.FAST_PATH_NAMES)
    assert set(names) == dispatch.FAST_PATH_NAMES
    for path in dispatch.FAST_PATHS:
        assert path.hook in _HOOKS, path
        assert path.summary, path
        assert set(path.depends_on) <= dispatch.FAST_PATH_NAMES, path
        assert path.name not in path.depends_on, path


def test_the_dependencies_are_acyclic():
    by_name = {path.name: path for path in dispatch.FAST_PATHS}
    done, active = set(), set()

    def visit(name):
        assert name not in active, f"a cycle through {name}"
        if name in done:
            return
        active.add(name)
        for dependency in by_name[name].depends_on:
            visit(dependency)
        active.discard(name)
        done.add(name)

    for name in by_name:
        visit(name)
    assert done == dispatch.FAST_PATH_NAMES


def test_sm103_tensor_parallel_size_8_serves_every_fast_path_and_only_4_and_8_any():
    assert set(modeling.TARGET_PATHS) == {(103, 8), (107, 8), (103, 4), (107, 4)}
    served = modeling.TARGET_PATHS[103, 8].served
    assert served == dispatch.FAST_PATH_NAMES
    assert len(served) == 39
    assert dispatch.closure(served) == served
    assert modeling.TARGET_PATHS[103, 8].topology is dispatch.TWO_TRAYS_OF_4
    for capability in (_SM103, _SM107):
        for tp in (1, 2, 16):
            config = _config(tp, _dflash(7))
            assert dispatch.composition(config, modeling.TARGET_PATHS, capability) is None


def test_sm107_tensor_parallel_size_8_serves_every_fast_path():
    sm107 = modeling.TARGET_PATHS[107, 8]
    assert sm107.served == dispatch.FAST_PATH_NAMES
    assert dispatch.closure(sm107.served) == sm107.served
    assert sm107.topology is dispatch.TWO_TRAYS_OF_4
    assert dispatch.composition(_SERVED, modeling.TARGET_PATHS, _SM107) is sm107
    assert (
        dispatch.composition(_SERVED, modeling.TARGET_PATHS, _SM103)
        is modeling.TARGET_PATHS[103, 8]
    )


def test_each_architecture_serves_its_own_tep4_set():
    sm103, sm107 = modeling.TARGET_PATHS[103, 4], modeling.TARGET_PATHS[107, 4]
    assert sm103.served == _SM103_TEP4 and sm107.served == _TEP4
    assert dispatch.closure(sm107.served) == sm107.served
    assert sm103.topology is sm107.topology is dispatch.ONE_TRAY_OF_4
    assert dispatch.composition(_config(4, _dflash(7)), modeling.TARGET_PATHS, _SM107) is sm107


def test_the_sm107_compositions_need_the_builtin_fused_kernels_opted_in():
    for tp in (4, 8):
        assert dispatch.SM107_FUSED_KERNELS in modeling.TARGET_PATHS[107, tp].preconditions
        assert dispatch.SM107_FUSED_KERNELS not in modeling.TARGET_PATHS[103, tp].preconditions


@pytest.mark.parametrize("value, holds", [(None, False), ("0", False), ("1", True), ("on", True)])
def test_the_sm107_opt_in_is_read_through_the_builtin_helper(monkeypatch, value, holds):
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin

    if value is None:
        monkeypatch.delenv("MUSESPARK_OPEN_SM107_FUSED_KERNELS", raising=False)
    else:
        monkeypatch.setenv("MUSESPARK_OPEN_SM107_FUSED_KERNELS", value)
    assert dispatch.SM107_FUSED_KERNELS.holds(None, None) is holds
    assert builtin.fused_kernels_supported((10, 3))
    assert builtin.fused_kernels_supported((10, 7)) is holds
    for other in ((10, 0), (9, 0), (12, 0)):
        assert not builtin.fused_kernels_supported(other)


@pytest.mark.parametrize("tp", [4, 8])
def test_an_sm107_deployment_without_the_opt_in_raises_in_both_modes(
    monkeypatch, quiet, agreement, tp
):
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    monkeypatch.delenv("MUSESPARK_OPEN_SM107_FUSED_KERNELS", raising=False)
    model = SimpleNamespace(_prefill_sp_enabled=False)
    config = _config(tp, _dflash(7))
    for mode in (_AUTO, _REQUIRE):
        with pytest.raises(ValueError, match="MUSESPARK_OPEN_SM107_FUSED_KERNELS=1"):
            dispatch.compose(model, config, mode, modeling.TARGET_PATHS, _SM107)
    monkeypatch.setenv("MUSESPARK_OPEN_SM107_FUSED_KERNELS", "1")
    plan = dispatch.compose(model, config, _REQUIRE, modeling.TARGET_PATHS, _SM107)
    assert plan.paths == modeling.TARGET_PATHS[107, tp].served
    assert quiet[-1][1].startswith(f"[musespark_open] sm107 tep{tp} on ")


@pytest.mark.parametrize("capability", [(10, 0), (9, 0), (12, 0), (10, 1), None])
def test_an_architecture_without_an_entry_has_no_composition(capability):
    for tp in (4, 8):
        config = _config(tp, _dflash(7))
        assert dispatch.composition(config, modeling.TARGET_PATHS, capability) is None
        with pytest.raises(ValueError, match="serves no fast path") as raised:
            dispatch.require_served(config, _REQUIRE, modeling.TARGET_PATHS, capability)
        arch = "no CUDA device" if capability is None else f"sm{10 * capability[0] + capability[1]}"
        assert str(raised.value).split(": ", 2)[2].startswith(f"{arch}, tp {tp}, ")


def test_sm_of_reads_the_compute_capability():
    assert dispatch.sm_of((10, 3)) == 103
    assert dispatch.sm_of((10, 7)) == 107
    assert dispatch.sm_of((9, 0)) == 90
    assert dispatch.sm_of(None) is None


#: The fast paths written for any rank count whose output is bitwise the built-in
#: model's.
_TEP4_BITWISE = {
    "preattn_prep",
    "full_attention_prep_append",
    "post_attention_hop",
    "deferred_input_prep",
    "sampler_store",
    "replicated_embedding",
    "fused_accept",
    "verify_fold",
    "draft_head",
    "path_walk_tail",
    "drafter_block_conv",
    "context_kv_fork",
    "split_sampler",
}

#: The TEP4 composition on either architecture: the bitwise paths and the sharded
#: drafter projection.
_TEP4 = _TEP4_BITWISE | {"drafter_fc_shard"}

#: sm103 TEP4's composition adds the attention paths validated at a tep4 rank's 32
#: query / 4 KV heads.
_SM103_TEP4 = _TEP4 | {
    "swa_paged_decode",
    "swa_block",
    "full_attention_split_kv",
    "full_attention_prep_fold",
    "drafter_block_decode",
}


def test_tensor_parallel_size_4_serves_the_bitwise_paths_on_one_tray():
    tep4 = modeling.TARGET_PATHS[103, 4]
    assert tep4.served == _SM103_TEP4
    assert dispatch.closure(tep4.served) == tep4.served
    assert tep4.topology is dispatch.ONE_TRAY_OF_4
    assert tep4.preconditions == (dispatch.NO_PREFILL_SEQUENCE_PARALLEL,)
    assert dispatch.composition(_config(4, _dflash(7)), modeling.TARGET_PATHS, _SM103) is tep4
    for gpus_per_node in (1, 2, 8):
        config = _config(4, _dflash(7), gpus_per_node=gpus_per_node)
        assert dispatch.composition(config, modeling.TARGET_PATHS, _SM103) is None, gpus_per_node
    for spec in (None, _dflash(3), _dflash(8)):
        assert dispatch.composition(_config(4, spec), modeling.TARGET_PATHS, _SM103) is None


#: Requirements only the TP8 composition's paths have: a TP4 deployment cannot (the
#: sharded projections raise in the built-in model at tp 4) or need not meet them.
_TP8_ONLY = (
    "TRTLLM_LOW_M_GEMM_BACKEND",
    "MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL",
    "MUSESPARK_OPEN_SHARDED_PROJECTIONS",
    "MUSESPARK_OPEN_ROUTING_MEMO",
    "MUSESPARK_OPEN_FUSED_PRE_EXPERT_QUANT",
    "finalize transport",
    "projection gathers",
    "MoE prologue",
    "pool transport",
    "unpool transport",
    "drafter hops",
)


def test_tensor_parallel_size_4_needs_no_tp8_only_setting():
    tep4 = modeling.TARGET_PATHS[103, 4]
    required = [p.requirement for p in dispatch.preconditions(tep4.served, tep4)]
    for word in _TP8_ONLY:
        assert not any(word in requirement for requirement in required), word
    for word in (
        "TRTLLM_ENABLE_PDL",
        "MUSESPARK_OPEN_ALLREDUCE=MNNVL",
        "draft head's key exchange",
        "MUSESPARK_OPEN_FUSED_RESIDUAL",
        "MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL",
        "MUSESPARK_OPEN_FUSED_PREATTN",
        "MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT",
        "CuTe-DSL RMSNorm",
        "post-attention hop",
        "head sampler",
        "MUSESPARK_OPEN_PREFILL_SEQUENCE_PARALLEL",
    ):
        assert any(word in requirement for requirement in required), word


def test_a_tp4_model_without_the_tp8_switches_composes_the_whole_tep4_set(
    monkeypatch, quiet, agreement
):
    """Every precondition of a TP8-only path fails here, every other one holds: TP4
    still gets all of its composition."""
    tep4 = modeling.TARGET_PATHS[103, 4]
    held = tuple(
        dataclasses.replace(p, holds=lambda model, config, keep=p.needed(tep4.served): keep)
        for p in dispatch.PRECONDITIONS
    )
    monkeypatch.setattr(dispatch, "PRECONDITIONS", held)
    model = SimpleNamespace(_prefill_sp_enabled=False)
    for mode in (_AUTO, _REQUIRE):
        plan = dispatch.compose(model, _config(4, _dflash(7)), mode, modeling.TARGET_PATHS, _SM103)
        assert plan.paths == _SM103_TEP4
    assert quiet[-1] == (
        "info",
        "[musespark_open] sm103 tep4 on one node of 4 in one NVLink domain: "
        + dispatch.Plan(_SM103_TEP4).describe(),
    )


def test_prefill_sequence_parallelism_at_tp4_raises_in_both_modes(monkeypatch, quiet, agreement):
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    model = SimpleNamespace(_prefill_sp_enabled=True)
    for mode in (_AUTO, _REQUIRE):
        with pytest.raises(ValueError, match="MUSESPARK_OPEN_PREFILL_SEQUENCE_PARALLEL"):
            dispatch.compose(model, _config(4, _dflash(7)), mode, modeling.TARGET_PATHS, _SM103)


@pytest.mark.parametrize("gpus_per_node, served", [(1, False), (2, False), (4, True), (8, False)])
def test_the_topology_is_part_of_the_composition(gpus_per_node, served):
    """TP8 is served over two trays of four GPUs only; one node of eight (or TP8 under
    Ray, where ``gpus_per_node`` is the tensor-parallel size) has no composition."""
    config = _config(8, _dflash(7), gpus_per_node=gpus_per_node)
    expected = modeling.TARGET_PATHS[103, 8] if served else None
    assert dispatch.composition(config, modeling.TARGET_PATHS, _SM103) is expected


def test_the_topologies_count_the_nodes_of_the_tp_group():
    assert dispatch.TWO_TRAYS_OF_4.holds(_tep(8))
    assert not dispatch.ONE_TRAY_OF_4.holds(_tep(8))
    assert dispatch.ONE_TRAY_OF_4.holds(_tep(4))
    assert not dispatch.TWO_TRAYS_OF_4.holds(_tep(4))
    assert not dispatch.ONE_TRAY_OF_4.holds(_tep(4, gpus_per_node=8))
    assert not dispatch.TWO_TRAYS_OF_4.holds(_tep(8, gpus_per_node=8))


@pytest.mark.parametrize("quant", [None, QuantAlgo.FP8, QuantAlgo.W4A16_AWQ])
def test_a_checkpoint_that_is_not_nvfp4_has_no_composition(quant):
    config = _config(8, _dflash(7), quant=quant)
    assert not dispatch.nvfp4_checkpoint(config)
    assert dispatch.composition(config, modeling.TARGET_PATHS, _SM103) is None


def test_closure_adds_every_fast_path_built_on_and_rejects_unknown_names():
    assert dispatch.closure(()) == frozenset()
    assert dispatch.closure({"preattn_prep"}) == {"preattn_prep"}
    assert dispatch.closure({"full_attention_prep_fold"}) == {
        "full_attention_prep_fold",
        "full_attention_split_kv",
        "full_attention_prep_append",
        "preattn_prep",
    }
    for path in dispatch.FAST_PATHS:
        closed = dispatch.closure({path.name})
        assert set(path.depends_on) | {path.name} <= closed
        assert dispatch.closure(closed) == closed
    with pytest.raises(KeyError):
        dispatch.closure({"not_a_path"})


def test_a_composition_is_closed_and_names_only_fast_paths():
    one = dispatch.ONE_TRAY_OF_4
    with pytest.raises(ValueError, match="not fast paths"):
        dispatch.TargetPaths(served=frozenset({"not_a_path"}), topology=one)
    with pytest.raises(ValueError, match="builds on unserved"):
        dispatch.TargetPaths(served=frozenset({"dense_gemm_prefetch"}), topology=one)
    assert dispatch.TargetPaths(served=frozenset(), topology=one).served == frozenset()


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
def test_a_plan_is_closed_and_names_only_fast_paths():
    with pytest.raises(ValueError, match="not fast paths"):
        dispatch.Plan(frozenset({"not_a_path"}))
    with pytest.raises(ValueError, match="builds on"):
        dispatch.Plan(frozenset({"swa_block"}))
    assert dispatch.plan_of("swa_block").paths == {"swa_block", "preattn_prep", "swa_paged_decode"}
    assert dispatch.plan_of() == dispatch.NO_PATHS == dispatch.Plan()


def test_enabled_answers_for_fast_paths_only():
    plan = dispatch.plan_of("dense_gemm_prefetch")
    assert plan.enabled("dense_gemm_prefetch") and plan.enabled("dense_gemms")
    assert not plan.enabled("preattn_prep")
    for name in dispatch.FAST_PATH_NAMES:
        assert not dispatch.NO_PATHS.enabled(name)
    with pytest.raises(KeyError, match="not a fast path"):
        plan.enabled("not_a_path")
    with pytest.raises(KeyError, match="not a fast path"):
        dispatch.NO_PATHS.enabled("not_a_path")


def test_only_no_paths_is_false_and_the_description_follows_fast_paths_order():
    assert not dispatch.NO_PATHS
    assert dispatch.plan_of("preattn_prep")
    assert dispatch.NO_PATHS.describe() == "no fast path"
    plan = dispatch.plan_of("swa_block")
    assert plan.describe() == "3 fast paths: preattn_prep, swa_paged_decode, swa_block"
    full = dispatch.Plan(modeling.TARGET_PATHS[103, 8].served)
    assert full.describe() == "39 fast paths: " + ", ".join(
        path.name for path in dispatch.FAST_PATHS
    )


# --------------------------------------------------------------------------
# Which deployments have a composition
# --------------------------------------------------------------------------
_SPECULATION = {
    "none": (lambda: None, False),
    "dflash3": (lambda: _dflash(3), False),
    "dflash7": (lambda: _dflash(7), True),
    "dflash8": (lambda: _dflash(8), False),
    "dflash7-trtllm-attention": (lambda: _dflash(7, attention_backend="TRTLLM"), False),
    "ngram": (lambda: NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2), False),
    "eagle3": (
        lambda: Eagle3DecodingConfig(max_draft_len=7, speculative_model=_DUMMY_MODEL),
        False,
    ),
    "mtp": (lambda: MTPDecodingConfig(max_draft_len=3), False),
}


@pytest.mark.parametrize("name", list(_SPECULATION))
def test_only_dflash_drafting_a_linear_chain_of_seven_is_served(name):
    make, linear = _SPECULATION[name]
    spec = make()
    assert dispatch.dflash_linear(spec) is linear
    served = dispatch.composition(_config(8, spec), modeling.TARGET_PATHS, _SM103)
    assert served is (modeling.TARGET_PATHS[103, 8] if linear else None)


def test_a_draft_length_schedule_or_a_tree_is_not_served():
    assert not dispatch.dflash_linear(_dflash(7, draft_len_schedule={1: 7}))
    tree = _dflash(7)
    object.__setattr__(tree, "max_total_draft_tokens", 14)
    assert not dispatch.dflash_linear(tree)
    stand_in = SimpleNamespace(
        **{name: getattr(_dflash(7), name) for name in type(_dflash(7)).model_fields},
        spec_dec_mode=_dflash(7).spec_dec_mode,
    )
    assert not dispatch.dflash_linear(stand_in), "not a DFlashDecodingConfig"


def test_a_config_without_a_mapping_has_no_composition():
    config = SimpleNamespace(spec_config=_dflash(7))
    assert dispatch.composition(config, modeling.TARGET_PATHS, _SM103) is None


@pytest.mark.parametrize("tp, spec", [(2, _dflash(7)), (8, None), (8, _dflash(3))])
def test_require_raises_before_construction_naming_every_fact_read(tp, spec):
    config = _config(tp, spec)
    with pytest.raises(ValueError, match="modeling_v2='require'") as raised:
        dispatch.require_served(config, _REQUIRE, modeling.TARGET_PATHS, _SM103)
    message = str(raised.value)
    assert f"tp {tp}" in message and "speculative decoding" in message
    assert "4 GPUs per node, the TP group over" in message
    assert "checkpoint quantization NVFP4" in message
    assert message.split(": ", 2)[2].startswith(f"sm103, tp {tp}, ")
    assert (
        "sm103 tp 4 on one node of 4 in one NVLink domain, "
        "sm103 tp 8 on two nodes of 4 in one NVLink domain, "
        "sm107 tp 4 on one node of 4 in one NVLink domain, "
        "sm107 tp 8 on two nodes of 4 in one NVLink domain"
    ) in message
    assert "each with DFlash drafting a linear chain of 7 tokens" in message
    dispatch.require_served(config, _AUTO, modeling.TARGET_PATHS, _SM103)
    dispatch.require_served(_SERVED, _REQUIRE, modeling.TARGET_PATHS, _SM103)


# --------------------------------------------------------------------------
# compose: all of the composition, or NO_PATHS
# --------------------------------------------------------------------------
@pytest.fixture
def quiet(monkeypatch):
    """The log lines ``compose`` writes, recorded."""
    lines = []
    for level in ("info", "warning"):
        monkeypatch.setattr(
            dispatch.logger, level, lambda message, level=level: lines.append((level, message))
        )
    return lines


@pytest.fixture
def agreement(monkeypatch):
    """``agreed_union`` as a group of one rank, recording each call."""
    calls = []

    def union(mapping, local):
        calls.append((mapping, list(local)))
        return list(dict.fromkeys(local))

    monkeypatch.setattr(tp_agreement, "agreed_union", union)
    return calls


def _stand_in_preconditions(monkeypatch, holds):
    """``PRECONDITIONS`` replaced by one stand-in per entry of ``holds``, needed everywhere."""
    preconditions = tuple(
        dispatch.Precondition(f"precondition {i}", lambda model, config, on=on: on)
        for i, on in enumerate(holds)
    )
    monkeypatch.setattr(dispatch, "PRECONDITIONS", preconditions)
    return [p.requirement for p, on in zip(preconditions, holds) if not on]


def test_every_precondition_names_its_requirement_and_the_fast_paths_it_serves():
    requirements = [p.requirement for p in dispatch.PRECONDITIONS]
    assert len(set(requirements)) == len(requirements) and all(requirements)
    for precondition in dispatch.PRECONDITIONS:
        assert set(precondition.needed_by) <= dispatch.FAST_PATH_NAMES, precondition
    # tp 8 serves every fast path, so it needs every precondition.
    full = modeling.TARGET_PATHS[103, 8]
    assert dispatch.preconditions(full.served, full) == dispatch.PRECONDITIONS


def test_a_precondition_is_needed_by_a_plan_with_one_of_its_paths():
    everywhere = dispatch.Precondition("a", lambda m, c: True)
    hop = dispatch.Precondition("b", lambda m, c: True, ("post_attention_hop",))
    assert everywhere.needed(frozenset()) and everywhere.needed(frozenset({"preattn_prep"}))
    assert hop.needed(frozenset({"post_attention_hop", "preattn_prep"}))
    assert not hop.needed(frozenset({"preattn_prep"}))


_COMBINATIONS = 10


def test_every_combination_of_missing_preconditions_raises_in_both_modes(
    monkeypatch, quiet, agreement
):
    """All 2**10 combinations, each under ``auto`` and ``require``: the whole composition
    or ``ValueError`` naming every missing requirement, never ``NO_PATHS`` or a part."""
    full = dispatch.Plan(modeling.TARGET_PATHS[103, 8].served)
    for holds in itertools.product((True, False), repeat=_COMBINATIONS):
        missing = _stand_in_preconditions(monkeypatch, holds)
        for mode in (_AUTO, _REQUIRE):
            if missing:
                with pytest.raises(ValueError) as raised:
                    dispatch.compose(object(), _SERVED, mode, modeling.TARGET_PATHS, _SM103)
                for requirement in missing:
                    assert requirement in str(raised.value)
            else:
                assert (
                    dispatch.compose(object(), _SERVED, mode, modeling.TARGET_PATHS, _SM103) == full
                )
    # One agreement per call, whatever the outcome: every rank reaches the same collectives.
    assert len(agreement) == 2 * 2**_COMBINATIONS


def test_a_precondition_missing_on_a_peer_only_raises_in_both_modes(monkeypatch, quiet):
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    peer = "precondition 3 (on TP ranks [7] of 8)"
    monkeypatch.setattr(tp_agreement, "agreed_union", lambda mapping, local: list(local) + [peer])
    for mode in (_AUTO, _REQUIRE):
        with pytest.raises(ValueError, match=r"precondition 3 \(on TP ranks \[7\] of 8\)"):
            dispatch.compose(object(), _SERVED, mode, modeling.TARGET_PATHS, _SM103)


class _Group:
    """A TP group of 8 whose last rank answers ``peer``."""

    def __init__(self, peer):
        self.peer = peer
        self.gathered = []

    def get(self, mapping):
        return self

    def tp_allgather(self, obj, small_payload=False):
        self.gathered.append(obj)
        return [obj] * 7 + [self.peer]


def test_the_preconditions_are_agreed_over_the_tp_group(monkeypatch, quiet):
    """The real agreement over a stand-in group: a requirement only a peer misses is named
    with its ranks, on every rank, in both modes."""
    group = _Group(["precondition 0"])
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    for mode in (_AUTO, _REQUIRE):
        with pytest.raises(ValueError, match=r"precondition 0 \(on TP ranks \[7\] of 8\)"):
            dispatch.compose(object(), _SERVED, mode, modeling.TARGET_PATHS, _SM103)
    assert group.gathered == [[], []]


@pytest.mark.parametrize(
    "config",
    [
        _config(2, _dflash(7)),
        _config(16, _dflash(7)),
        _config(8, None),
        _config(8, _dflash(8)),
        _config(8, _dflash(7), gpus_per_node=8),
        _config(8, _dflash(7), quant=None),
    ],
    ids=["tp2", "tp16", "no-spec", "dflash8", "one-node-of-8", "not-nvfp4"],
)
def test_an_unserved_deployment_gets_no_paths_without_a_collective(
    monkeypatch, quiet, agreement, config
):
    _stand_in_preconditions(monkeypatch, (False,) * 4)
    assert (
        dispatch.compose(object(), config, _AUTO, modeling.TARGET_PATHS, _SM103)
        == dispatch.NO_PATHS
    )
    assert agreement == []
    assert [level for level, _ in quiet] == ["info"]
    with pytest.raises(ValueError, match="serves no fast path"):
        dispatch.compose(object(), config, _REQUIRE, modeling.TARGET_PATHS, _SM103)
    assert agreement == []


def test_missing_preconditions_reads_the_built_model_and_its_config():
    seen = []
    required = (
        dispatch.Precondition("a", lambda model, config: seen.append((model, config)) or True),
        dispatch.Precondition("b", lambda model, config: False),
        dispatch.Precondition("c", lambda model, config: False),
    )
    assert dispatch.missing_preconditions("model", "config", required) == ["b", "c"]
    assert seen == [("model", "config")]


def test_the_plan_line_names_the_composition_on_every_rank(monkeypatch, quiet, agreement):
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    dispatch.compose(object(), _SERVED, _AUTO, modeling.TARGET_PATHS, _SM103)
    assert quiet == [
        (
            "info",
            "[musespark_open] sm103 tep8 on two nodes of 4 in one NVLink domain: "
            + dispatch.Plan(modeling.TARGET_PATHS[103, 8].served).describe(),
        )
    ]


# --------------------------------------------------------------------------
# The environment picks nothing
# --------------------------------------------------------------------------
_ENV_CHOICES = {
    "TRTLLM_ENABLE_PDL": (None, "1", "0"),
    "MUSESPARK_OPEN_FUSED_RESIDUAL": (None, "0"),
    "MUSESPARK_OPEN_ROUTING_MEMO": (None, "off"),
    "MUSESPARK_OPEN_FUSED_PRE_EXPERT_QUANT": (None, "0"),
    "MUSESPARK_OPEN_FUSED_PREATTN": (None, "1"),
    "MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL": (None, "1"),
    "MODELING_V2_FEATURES": (None, "unknown_path,stale_knob,DISABLE"),
    "MODELING_V2_RESEARCH": (None, "paged_decode_full"),
}
_ENV_DEFAULTS_HOLD = {
    "TRTLLM_ENABLE_PDL": (None, "1"),
    "MUSESPARK_OPEN_FUSED_RESIDUAL": (None,),
    "MUSESPARK_OPEN_ROUTING_MEMO": (None,),
    "MUSESPARK_OPEN_FUSED_PRE_EXPERT_QUANT": (None,),
    "MUSESPARK_OPEN_FUSED_PREATTN": (None,),
    "MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL": (None,),
}


def _reads_a_switch(precondition) -> bool:
    """Whether ``precondition`` reads a public switch (through a built-in helper or
    ``get_env_enable_pdl``); the others read the built model."""
    holds = precondition.holds
    return holds is dispatch._pdl_enabled or holds.__qualname__.startswith("_builtin_switch.")


def test_every_environment_gives_the_whole_composition_or_an_error(monkeypatch, quiet, agreement):
    """Every combination of the public switches the preconditions read, with leftover
    target knobs set to garbage or not: under ``auto`` and ``require`` alike the plan is
    the whole composition or ``ValueError``, never ``NO_PATHS`` or a part. The
    preconditions that read the built model are held true here."""
    switched = tuple(
        p if _reads_a_switch(p) else dataclasses.replace(p, holds=lambda model, config: True)
        for p in dispatch.PRECONDITIONS
    )
    assert sum(p is q for p, q in zip(switched, dispatch.PRECONDITIONS)) == 6
    monkeypatch.setattr(dispatch, "PRECONDITIONS", switched)
    full = dispatch.Plan(modeling.TARGET_PATHS[103, 8].served)
    names = list(_ENV_CHOICES)
    for values in itertools.product(*(_ENV_CHOICES[name] for name in names)):
        for name, value in zip(names, values):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        defaults = all(
            value in _ENV_DEFAULTS_HOLD.get(name, (value,)) for name, value in zip(names, values)
        )
        for mode in (_AUTO, _REQUIRE):
            if defaults:
                assert (
                    dispatch.compose(object(), _SERVED, mode, modeling.TARGET_PATHS, _SM103) == full
                )
            else:
                with pytest.raises(ValueError, match="the fast paths need"):
                    dispatch.compose(object(), _SERVED, mode, modeling.TARGET_PATHS, _SM103)


# --------------------------------------------------------------------------
# A windowed drafter context arena
# --------------------------------------------------------------------------
_WINDOWED = SimpleNamespace(windowed_ctx_arena=True)
_LEFT_OUT = {
    "context_kv_fork",
    "verify_fold",
}
#: The drafter paths a windowed arena keeps: the block decode reads the physical
#: length the built-in prepare hands it, and the hops and the PDL chain build on it.
_KEPT_DRAFTER = {"drafter_block_decode", "drafter_hops", "drafter_pdl_chain"}


def test_dependents_are_every_fast_path_built_on_the_names():
    assert dispatch.dependents(()) == frozenset()
    assert dispatch.dependents({"drafter_block_decode"}) == {
        "drafter_block_decode",
        "drafter_hops",
        "drafter_pdl_chain",
    }
    assert dispatch.dependents(dispatch.FULL_CONTEXT_ARENA_PATHS) == _LEFT_OUT


@pytest.mark.parametrize(
    "worker", [None, SimpleNamespace(), SimpleNamespace(windowed_ctx_arena=False)]
)
def test_a_full_length_arena_keeps_the_whole_composition(monkeypatch, quiet, agreement, worker):
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    plan = dispatch.compose(
        object(), _SERVED, _AUTO, modeling.TARGET_PATHS, _SM103, spec_worker=worker
    )
    assert plan == dispatch.Plan(modeling.TARGET_PATHS[103, 8].served)
    assert agreement == [(_SERVED.mapping, [])]


@pytest.mark.parametrize("key", sorted(modeling.TARGET_PATHS))
def test_a_windowed_arena_leaves_out_the_full_length_arena_paths(
    monkeypatch, quiet, agreement, key
):
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    monkeypatch.setenv("MUSESPARK_OPEN_SM107_FUSED_KERNELS", "1")
    sm, tp = key
    config = _config(tp, _dflash(7))
    served = modeling.TARGET_PATHS[key].served
    capability = divmod(sm, 10)
    for mode in (_AUTO, _REQUIRE):
        plan = dispatch.compose(
            object(), config, mode, modeling.TARGET_PATHS, capability, spec_worker=_WINDOWED
        )
        assert plan.paths == served - _LEFT_OUT
        assert dispatch.closure(plan.paths) == plan.paths
        if tp == 8:
            assert _KEPT_DRAFTER <= plan.paths
    assert all(local == ["a windowed drafter context arena"] for _, local in agreement)
    assert quiet[-1][1].startswith(f"[musespark_open] sm{sm} tep{tp} on ")
    assert "with a windowed drafter context arena: " in quiet[-1][1]


def test_ranks_that_disagree_on_the_arena_raise(monkeypatch, quiet):
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    monkeypatch.setattr(tp_agreement, "Distributed", _Group([]))
    for mode in (_AUTO, _REQUIRE):
        with pytest.raises(ValueError, match="the same drafter context arena on every rank"):
            dispatch.compose(
                object(), _SERVED, mode, modeling.TARGET_PATHS, _SM103, spec_worker=_WINDOWED
            )


def test_the_arena_is_read_off_the_worker():
    assert not dispatch.windowed_context_arena(None)
    assert not dispatch.windowed_context_arena(SimpleNamespace())
    assert dispatch.windowed_context_arena(_WINDOWED)


# --------------------------------------------------------------------------
# The preconditions on stand-in models
# --------------------------------------------------------------------------
def _layer(**feed_forward):
    return SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=None), feed_forward=SimpleNamespace(**feed_forward)
    )


def test_the_sharded_projections_are_read_from_the_built_moes():
    holds = dispatch._sharded_projections
    both = dict(_pool_gather_workspace=object(), _unpool_gather_workspace=object())
    assert holds(SimpleNamespace(layers=[_layer(**both), _layer(**both)]), None)
    one = dict(both, _unpool_gather_workspace=None)
    assert not holds(SimpleNamespace(layers=[_layer(**both), _layer(**one)]), None)
    assert not holds(SimpleNamespace(layers=[_layer()]), None)


def test_the_fabric_finalize_is_the_models_workspace_on_every_layer():
    holds = dispatch._fabric_finalize
    workspace = object()
    model = SimpleNamespace(
        _finalize_ar_workspace=workspace,
        layers=[_layer(_finalize_ar_workspace=workspace) for _ in range(3)],
    )
    assert holds(model, None)
    model.layers[1].feed_forward._finalize_ar_workspace = object()
    assert not holds(model, None)
    assert not holds(SimpleNamespace(_finalize_ar_workspace=None, layers=[]), None)


def test_the_o_proj_all_reduce_is_mnnvl_on_every_layer():
    from tensorrt_llm.functional import AllReduceStrategy

    holds = dispatch._mnnvl_o_proj_all_reduce

    def layer(strategy, mnnvl=object()):
        all_reduce = SimpleNamespace(strategy=strategy, mnnvl_allreduce=mnnvl)
        return SimpleNamespace(
            self_attn=SimpleNamespace(o_proj=SimpleNamespace(all_reduce=all_reduce))
        )

    mnnvl = AllReduceStrategy.MNNVL
    assert holds(SimpleNamespace(layers=[layer(mnnvl), layer(mnnvl)]), None)
    assert not holds(SimpleNamespace(layers=[layer(mnnvl), layer(AllReduceStrategy.NCCL)]), None)
    assert not holds(SimpleNamespace(layers=[layer(mnnvl, mnnvl=None)]), None)


def test_the_checkpoint_is_nvfp4():
    holds = dispatch.nvfp4_checkpoint
    assert holds(SimpleNamespace(quant_config=SimpleNamespace(quant_algo=QuantAlgo.NVFP4)))
    assert not holds(SimpleNamespace(quant_config=SimpleNamespace(quant_algo=QuantAlgo.FP8)))
    assert not holds(SimpleNamespace())


def test_the_qkv_weight_layout_is_read_from_the_built_attention():
    holds = dispatch._default_qkv_weight_layout

    def model(*attentions):
        return SimpleNamespace(layers=[SimpleNamespace(self_attn=a) for a in attentions])

    plain = SimpleNamespace(_qkv_weight_layout=False, _qkv_layout_inverse=None)
    assert holds(model(plain, SimpleNamespace()), None)
    assert not holds(model(plain, SimpleNamespace(_qkv_weight_layout=True)), None)
    assert not holds(model(SimpleNamespace(_qkv_layout_inverse=object())), None)


def test_the_prefill_sequence_parallelism_is_read_from_the_built_model():
    holds = dispatch._no_prefill_sequence_parallel
    assert holds(SimpleNamespace(), None)
    assert holds(SimpleNamespace(_prefill_sp_enabled=False), None)
    assert not holds(SimpleNamespace(_prefill_sp_enabled=True), None)


def test_the_key_exchange_asks_the_builtin_mnnvl_check_without_building(monkeypatch):
    import torch

    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    asked = []

    def is_mnnvl(mapping, dtype, explicitly_requested=False):
        asked.append((mapping, dtype, explicitly_requested))
        return mapping == "fabric"

    monkeypatch.setattr(MNNVLAllReduce, "is_mnnvl", staticmethod(is_mnnvl))
    assert dispatch._mnnvl_fp32_all_reduce(None, SimpleNamespace(mapping="fabric"))
    assert not dispatch._mnnvl_fp32_all_reduce(None, SimpleNamespace(mapping="nccl"))
    assert not dispatch._mnnvl_fp32_all_reduce(None, SimpleNamespace())
    assert asked == [("fabric", torch.float32, True), ("nccl", torch.float32, True)]


def test_the_cute_dsl_checks_import_their_op_modules(monkeypatch):
    import types as _types

    module = _types.ModuleType("stand_in")
    module.HAS_CUTE_DSL = True
    module.available = lambda: False
    monkeypatch.setattr(dispatch.importlib, "import_module", lambda name: module)
    assert dispatch._cute_dsl_kernels(("ops.x", "HAS_CUTE_DSL"))(None, None)
    assert not dispatch._cute_dsl_kernels(("ops.x", "HAS_CUTE_DSL"), ("ops.y", "available"))(
        None, None
    )


def test_the_head_sampler_toolchain_is_ninja_and_nvcc(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    (tmp_path / "bin").mkdir()
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(cpp_extension, "is_ninja_available", lambda: True)
    assert not dispatch._head_sampler_toolchain(None, None)
    (tmp_path / "bin" / "nvcc").write_text("")
    assert dispatch._head_sampler_toolchain(None, None)
    monkeypatch.setattr(cpp_extension, "is_ninja_available", lambda: False)
    assert not dispatch._head_sampler_toolchain(None, None)
    monkeypatch.setattr(cpp_extension, "is_ninja_available", lambda: True)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    assert not dispatch._head_sampler_toolchain(None, None)


def test_the_built_in_switches_are_read_through_the_built_in_helpers(monkeypatch):
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin

    on = dispatch._builtin_switch("_fused_residual_enabled", True)
    off = dispatch._builtin_switch("_fused_preattention_enabled", False)
    monkeypatch.setattr(builtin, "_fused_residual_enabled", lambda: True)
    monkeypatch.setattr(builtin, "_fused_preattention_enabled", lambda: False)
    assert on(None, None) and off(None, None)
    monkeypatch.setattr(builtin, "_fused_residual_enabled", lambda: False)
    monkeypatch.setattr(builtin, "_fused_preattention_enabled", lambda: True)
    assert not on(None, None) and not off(None, None)


def test_the_low_m_dispatcher_and_pdl_are_read_through_their_public_switches(monkeypatch):
    from tensorrt_llm._torch.modules import low_m_gemm

    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", True)
    assert dispatch._low_m_gemm_active(None, None)
    monkeypatch.setattr(low_m_gemm, "LOW_M_GEMM_ACTIVE", False)
    assert not dispatch._low_m_gemm_active(None, None)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    assert dispatch._pdl_enabled(None, None)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    assert not dispatch._pdl_enabled(None, None)


# --------------------------------------------------------------------------
# The shell
# --------------------------------------------------------------------------
@pytest.fixture
def shell_parts(monkeypatch):
    """The shell's built-in constructor and adoption replaced by recorders."""
    events = []

    def builtin_init(self, model_config, *args, **kwargs):
        events.append("built-in model")
        import torch.nn as nn

        nn.Module.__init__(self)
        self.llm = SimpleNamespace(model=nn.Module())

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(
        modeling.MuseSparkOpenTepModelingV2,
        "adopt",
        classmethod(
            lambda cls, model, plan=dispatch.NO_PATHS, compute_capability=None: events.append(
                ("adopt", plan, compute_capability)
            )
        ),
    )
    return events


@pytest.mark.parametrize("tp, spec", [(2, _dflash(7)), (8, None), (8, _dflash(3))])
def test_the_shell_raises_under_require_before_the_built_in_constructor(
    monkeypatch, shell_parts, tp, spec
):
    monkeypatch.setattr(modeling, "resolved_mode", lambda target: _REQUIRE)
    with pytest.raises(ValueError, match="serves no fast path"):
        _SHELL(_config(tp, spec))
    assert shell_parts == []


def test_an_unserved_shell_under_auto_adopts_with_no_paths(monkeypatch, shell_parts, quiet):
    monkeypatch.setattr(modeling, "resolved_mode", lambda target: _AUTO)
    shell = _SHELL(_config(2, _dflash(7)))
    assert shell._plan == dispatch.NO_PATHS
    assert shell_parts == ["built-in model", ("adopt", dispatch.NO_PATHS, _SM103)]


def test_a_served_shell_adopts_with_the_plan_compose_made(monkeypatch, shell_parts):
    plan = dispatch.plan_of("preattn_prep")
    composed = []

    def compose(text_model, model_config, mode, compositions, compute_capability, spec_worker=None):
        composed.append(
            (text_model, model_config, mode, compositions, compute_capability, spec_worker)
        )
        return plan

    monkeypatch.setattr(modeling, "resolved_mode", lambda target: _REQUIRE)
    monkeypatch.setattr(modeling._dispatch, "compose", compose)
    shell = _SHELL(_SERVED)
    assert shell._plan is plan
    assert composed == [(shell.llm.model, _SERVED, _REQUIRE, modeling.TARGET_PATHS, _SM103, None)]
    assert shell_parts == ["built-in model", ("adopt", plan, _SM103)]


def test_the_shell_composes_with_the_built_in_dflash_worker(monkeypatch, shell_parts, quiet):
    """The arena is read off the worker the built-in model built: windowed here."""
    worker = SimpleNamespace(windowed_ctx_arena=True)

    def builtin_init(self, model_config, *args, **kwargs):
        import torch.nn as nn

        nn.Module.__init__(self)
        self.llm = SimpleNamespace(model=nn.Module(), spec_worker=worker)

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(modeling, "resolved_mode", lambda target: _REQUIRE)
    monkeypatch.setattr(tp_agreement, "agreed_union", lambda mapping, local: list(local))
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    shell = _SHELL(_SERVED)
    assert shell._plan.paths == modeling.TARGET_PATHS[103, 8].served - _LEFT_OUT


def test_the_shell_composes_for_the_device_it_reads_once(monkeypatch, shell_parts, quiet):
    """The capability is read once per shell and keys the composition: on sm107 the plan
    is the sm107 TEP8 composition."""
    reads = []

    def capability():
        reads.append(_SM107)
        return _SM107

    monkeypatch.setattr(modeling, "device_compute_capability", capability)
    monkeypatch.setenv("MUSESPARK_OPEN_SM107_FUSED_KERNELS", "1")
    monkeypatch.setattr(modeling, "resolved_mode", lambda target: _REQUIRE)
    monkeypatch.setattr(tp_agreement, "agreed_union", lambda mapping, local: list(local))
    _stand_in_preconditions(monkeypatch, (True,) * 4)
    shell = _SHELL(_SERVED)
    assert reads == [_SM107]
    assert shell._plan.paths == modeling.TARGET_PATHS[107, 8].served
    # The core adopts for the capability read here (the decode FMHA selection).
    assert shell_parts[-1] == ("adopt", shell._plan, _SM107)
    assert "attention_output_into_hop" in shell._plan.paths


def test_a_shell_on_an_architecture_without_an_entry_raises_under_require(monkeypatch, shell_parts):
    monkeypatch.setattr(modeling, "device_compute_capability", lambda: (10, 0))
    monkeypatch.setattr(modeling, "resolved_mode", lambda target: _REQUIRE)
    with pytest.raises(ValueError, match="serves no fast path for this deployment: sm100, tp 8"):
        _SHELL(_SERVED)
    assert shell_parts == []
