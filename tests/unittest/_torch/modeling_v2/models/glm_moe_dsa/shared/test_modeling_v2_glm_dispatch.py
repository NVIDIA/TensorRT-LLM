# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The composition of the GLM-5.2 tp8 target: the fast-path table, the DISABLE knob, the
fallback policy and the rank-uniform checks.

Host-only: the knobs are passed as an explicit environment and the collective is a fake
two-rank group or a fake MPI communicator, so nothing here needs a GPU or MPI.
"""

from __future__ import annotations

import pytest

import tensorrt_llm._utils as trtllm_utils
from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
)

pytestmark = pytest.mark.cpu_only

AUTO, REQUIRE = ModelingV2Mode.AUTO, ModelingV2Mode.REQUIRE
DISABLE = dispatch.DISABLE_ENV
ACCUMULATION = dispatch.KNOB_PREFIX + "ROUTED_MOE_INTO_HOP_ACCUMULATION"

# The feature vocabulary, in this order: the fast-path names, the DISABLE tokens and the
# knob infixes are the same words.
VOCABULARY = (
    "absorbed_query",
    "draft_absorbed_query",
    "shared_indexer_absorbed_query",
    "fused_kv_indexer_projection",
    "indexer_layer_front",
    "deferred_norm",
    "deferred_norm_evict_hints",
    "rope_kv_append",
    "indexer_topk",
    "cross_rank_topk",
    "cross_rank_topk_cluster_merge",
    "cross_rank_topk_global_emit",
    "sparse_mla_decode",
    "attention_output_tail",
    "attention_output_tail_prefetch",
    "attention_output_into_hop",
    "oproj_weight_staging",
    "post_attention_hop",
    "hop_routing",
    "router_gemm",
    "routed_moe",
    "routed_moe_into_hop",
    "shared_expert_prefetch",
    "post_moe_hop",
    "cross_layer_expert_prefetch",
    "dense_mlp",
    "draft_front",
    "draft_routed_moe",
    "draft_live_rows",
    "draft_tail",
    "draft_loop",
    "decode_prologue",
    "deferred_input_prep",
    "sampler_store",
    "execution_stream_loop",
    "host_page_retention",
)


@pytest.fixture(autouse=True)
def _fresh_fallback_state():
    dispatch.set_require(False)
    dispatch.reset_fallback_state()
    yield
    dispatch.set_require(False)
    dispatch.reset_fallback_state()


def _off(plan: dispatch.Plan) -> set:
    return {name for name in dispatch.FAST_PATH_NAMES if not plan.enabled(name)}


# --- the table --------------------------------------------------------------------------


def test_the_fast_paths_are_the_vocabulary_in_order():
    """The DISABLE tokens are these words; the log line lists them in this order."""
    assert dispatch.FAST_PATH_NAMES == VOCABULARY


def test_dependencies_name_fast_paths_and_have_no_cycle():
    names = set(dispatch.FAST_PATH_NAMES)
    for path in dispatch.FAST_PATHS:
        assert set(path.depends_on) <= names, path
        assert path.name not in path.depends_on, path
        # A base that is also a (transitive) dependent of the path would be a cycle.
        dependents = dispatch.disable_closure(frozenset({path.name})) - {path.name}
        assert not set(path.depends_on) & dependents, path


# --- DISABLE ----------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", " , ,"], ids=["unset", "empty", "blank"])
def test_every_fast_path_runs_by_default(raw):
    environ = {} if raw is None else {DISABLE: raw}
    plan = dispatch.resolve(AUTO, environ)
    assert _off(plan) == set()
    assert plan.disabled == plan.disabled_dependents == frozenset()


def test_disabling_a_path_without_dependents_turns_off_only_it():
    plan = dispatch.resolve(AUTO, {DISABLE: "draft_tail"})
    assert _off(plan) == {"draft_tail"}
    assert plan.disabled_dependents == frozenset()


@pytest.mark.parametrize(
    "disabled, dependents",
    [
        (
            "routed_moe",
            {
                "routed_moe_into_hop",
                "hop_routing",
                "shared_expert_prefetch",
                "cross_layer_expert_prefetch",
            },
        ),
        (
            "post_attention_hop",
            {
                "attention_output_into_hop",
                "oproj_weight_staging",
                "hop_routing",
                "shared_expert_prefetch",
                "draft_live_rows",
                "host_page_retention",
            },
        ),
        (
            "cross_rank_topk",
            {
                "cross_rank_topk_cluster_merge",
                "cross_rank_topk_global_emit",
                "indexer_layer_front",
            },
        ),
        ("draft_loop", {"draft_live_rows", "decode_prologue"}),
    ],
)
def test_disabling_a_path_turns_off_every_path_that_depends_on_it(disabled, dependents):
    plan = dispatch.resolve(AUTO, {DISABLE: disabled})
    assert plan.disabled == {disabled}
    assert plan.disabled_dependents == dependents
    assert _off(plan) == {disabled} | dependents


def test_disable_ignores_case_and_blanks():
    plan = dispatch.resolve(AUTO, {DISABLE: " Draft_Tail ,, DENSE_MLP "})
    assert _off(plan) == {"draft_tail", "dense_mlp"}


def test_disable_naming_something_else_raises():
    with pytest.raises(ValueError, match=r"not_a_fast_path") as excinfo:
        dispatch.resolve(AUTO, {DISABLE: "draft_tail,not_a_fast_path"})
    assert DISABLE in str(excinfo.value)


def test_a_call_site_asking_for_a_misspelled_path_fails():
    plan = dispatch.resolve(AUTO, {})
    with pytest.raises(KeyError, match="draft_tails"):
        plan.enabled("draft_tails")


def test_only_this_targets_knobs_are_recorded():
    plan = dispatch.resolve(
        AUTO, {DISABLE: "draft_tail", ACCUMULATION: "identity", "UNRELATED": "1"}
    )
    assert plan.knobs == ((DISABLE, "draft_tail"), (ACCUMULATION, "identity"))


# --- missed and uncovered ---------------------------------------------------------------


def test_a_miss_raises_under_require():
    dispatch.resolve(REQUIRE, {})
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'routed_moe'.*at 6 rows.*workspace"):
        dispatch.missed("routed_moe", 6, "its workspace is not built")


def test_a_miss_falls_back_and_is_counted_under_auto():
    dispatch.resolve(AUTO, {})
    dispatch.missed("routed_moe", None, "its workspace is not built")
    dispatch.missed("routed_moe", 6, "its workspace is not built")
    assert dispatch.missed_counts() == {"routed_moe": 2}


@pytest.mark.parametrize("mode", [AUTO, REQUIRE])
def test_an_uncovered_step_never_raises(mode):
    dispatch.resolve(mode, {})
    dispatch.uncovered("sparse_mla_decode", "more rows than the kernel takes")
    assert dispatch.missed_counts() == {}


@pytest.mark.parametrize("mode", [AUTO, REQUIRE])
def test_a_miss_decided_per_call_never_raises_and_is_counted(mode):
    dispatch.resolve(mode, {})
    dispatch.missed("routed_moe", 6, "not warmed up before CUDA graph capture", per_call=True)
    dispatch.missed("routed_moe", 6, "not warmed up before CUDA graph capture", per_call=True)
    assert dispatch.missed_counts() == {"routed_moe": 2}


# --- rank uniformity --------------------------------------------------------------------


def _two_rank_group(peer_environ: dict, calls: list):
    """A fake allgather over two ranks: this rank first, then a peer with ``peer_environ``."""

    def allgather(knobs, world_size):
        calls.append((knobs, world_size))
        return [knobs, dispatch.read_knobs(peer_environ)]

    return allgather


def test_ranks_that_agree_pass_with_one_collective():
    env = {DISABLE: "draft_tail", ACCUMULATION: "identity"}
    calls = []
    knobs = dispatch.read_knobs(env)
    dispatch.check_rank_uniform(knobs, 2, _two_rank_group(dict(env), calls))
    assert calls == [(knobs, 2)]


def test_unrelated_environment_does_not_count():
    dispatch.check_rank_uniform(
        dispatch.read_knobs({"UNRELATED": "a"}), 2, _two_rank_group({"UNRELATED": "b"}, [])
    )


def test_ranks_that_would_run_different_fast_paths_raise_with_the_rank_table():
    with pytest.raises(RuntimeError) as excinfo:
        dispatch.check_rank_uniform(
            dispatch.read_knobs({}), 2, _two_rank_group({DISABLE: "routed_moe"}, [])
        )
    message = str(excinfo.value)
    rank0, rank1 = message.split("rank 1:")
    assert "rank 0: knobs" in rank0 and f"{DISABLE}='<unset>'" in rank0
    assert f"{DISABLE}='routed_moe'" in rank1


def test_ranks_that_read_different_knob_values_raise():
    with pytest.raises(RuntimeError, match=rf"{ACCUMULATION}='identity'"):
        dispatch.check_rank_uniform(
            dispatch.read_knobs({ACCUMULATION: "fp32"}),
            2,
            _two_rank_group({ACCUMULATION: "identity"}, []),
        )


def test_a_value_one_rank_cannot_parse_fails_the_check_on_every_rank():
    """The check compares the raw values, so the rank with a valid value raises too."""
    with pytest.raises(RuntimeError, match=rf"{DISABLE}='not_a_fast_path'"):
        dispatch.check_rank_uniform(
            dispatch.read_knobs({DISABLE: "draft_tail"}),
            2,
            _two_rank_group({DISABLE: "not_a_fast_path"}, []),
        )


def test_without_a_communicator_the_check_is_skipped():
    dispatch.check_rank_uniform(dispatch.read_knobs({}), 2, lambda knobs, world_size: None)


def test_the_ranks_agree_on_the_knobs_the_shared_switches_and_the_mode():
    env = {
        DISABLE: "draft_tail",
        "TRTLLM_FORCE_MNNVL_AR": "1",
        "TLLM_TEST_MNNVL": "0",
        "TRTLLM_ENABLE_PDL": "1",
        "TRTLLM_MODELING_V2_VALIDATE": "1",
        "UNRELATED": "1",
    }
    inputs = dispatch.rank_uniform_inputs(dispatch.read_knobs(env), REQUIRE, env)
    assert inputs == (
        ("TLLM_TEST_MNNVL", "0"),
        ("TRTLLM_ENABLE_PDL", "1"),
        ("TRTLLM_FORCE_MNNVL_AR", "1"),
        ("TRTLLM_MODELING_V2_VALIDATE", "1"),
        (DISABLE, "draft_tail"),
        (dispatch.MODE_KEY, "require"),
    )
    assert dispatch.rank_uniform_inputs((), AUTO, {}) == ((dispatch.MODE_KEY, "auto"),)


@pytest.mark.parametrize(
    "peer_env, peer_mode",
    [({"TRTLLM_FORCE_MNNVL_AR": "1"}, REQUIRE), ({}, AUTO)],
    ids=["shared-switch", "mode"],
)
def test_ranks_that_differ_in_a_shared_switch_or_the_mode_raise(peer_env, peer_mode):
    mine = dispatch.rank_uniform_inputs((), REQUIRE, {})
    peer = dispatch.rank_uniform_inputs((), peer_mode, peer_env)
    with pytest.raises(RuntimeError, match="rank 1: knobs") as excinfo:
        dispatch.check_rank_uniform(mine, 2, lambda value, world_size: [value, peer])
    assert dispatch.MODE_KEY in str(excinfo.value)


_STATE = (
    ("model.layers.0", (("post_attn_hop", "PostAttentionHop"),)),
    ("model.layers.1", (("post_moe_hop", "PostMoeHop", "engaged=True"),)),
)


def test_ranks_that_attached_the_same_state_pass_with_one_collective():
    calls = []

    def allgather(value, world_size):
        calls.append(world_size)
        return [value, value]

    dispatch.check_rank_uniform_adopted(_STATE, 2, allgather)
    assert calls == [2]


def test_ranks_that_attached_different_state_raise_naming_the_first_module():
    peer = (_STATE[0], ("model.layers.1", ()))
    with pytest.raises(RuntimeError) as excinfo:
        dispatch.check_rank_uniform_adopted(_STATE, 2, lambda value, world_size: [value, peer])
    message = str(excinfo.value)
    assert (
        "rank 0: ('model.layers.1', (('post_moe_hop', 'PostMoeHop', 'engaged=True'),))" in message
    )
    assert "rank 1: ('model.layers.1', ())" in message
    assert "model.layers.0" not in message


def test_a_rank_with_fewer_modules_is_named():
    with pytest.raises(RuntimeError, match=r"rank 1: <no module>"):
        dispatch.check_rank_uniform_adopted(_STATE, 2, lambda value, world_size: [value, value[:1]])


def test_without_a_communicator_the_adopted_check_is_skipped():
    dispatch.check_rank_uniform_adopted(_STATE, 2, lambda value, world_size: None)


class _Communicator:
    def __init__(self, size: int) -> None:
        self.size = size
        self.gathered = []

    def Get_size(self) -> int:  # noqa: N802 - the mpi4py name
        return self.size

    def allgather(self, knobs):
        self.gathered.append(knobs)
        return [knobs] * self.size


def test_the_session_communicator_of_this_model_is_gathered_once(monkeypatch):
    comm = _Communicator(8)
    monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: False)
    monkeypatch.setattr(trtllm_utils, "mpi_comm", lambda: comm)
    knobs = dispatch.read_knobs({DISABLE: "draft_tail"})
    dispatch.check_rank_uniform(knobs, 8)
    assert comm.gathered == [knobs]


@pytest.mark.parametrize("size", [1, 16], ids=["singleton", "larger"])
def test_a_communicator_that_is_not_this_models_ranks_is_not_gathered(monkeypatch, size):
    """A singleton proves nothing; a larger one would wait on ranks that never build the model."""
    comm = _Communicator(size)
    monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: False)
    monkeypatch.setattr(trtllm_utils, "mpi_comm", lambda: comm)
    dispatch.check_rank_uniform(dispatch.read_knobs({}), 8)
    assert comm.gathered == []


def test_without_mpi_nothing_is_gathered(monkeypatch):
    monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: True)
    monkeypatch.setattr(trtllm_utils, "mpi_comm", lambda: pytest.fail("no communicator expected"))
    dispatch.check_rank_uniform(dispatch.read_knobs({}), 8)
