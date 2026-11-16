# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spec's deployments, each stated as the options its engine takes."""

from __future__ import annotations

import shlex

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

#: What each side of a disaggregated deployment states to move the cache.
MOVES = {"cache_transceiver_config": {"backend": "DEFAULT"}}


def problems(block) -> list[str]:
    errors: list[str] = []
    deployments.validate(block, errors)
    return errors


def resolve(block, accuracy=None):
    assert problems(block) == []
    data = {"deployments": block}
    if accuracy is not None:
        data["accuracy"] = accuracy
    return deployments.resolve(data)


def test_an_aggregated_deployment_keeps_every_field_it_states(tmp_path):
    options = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 2,
        "kv_cache_config": {"dtype": "nvfp4"},
        "future_native_key": 17,
    }
    (tmp_path / "engine.yaml").write_text(yaml.safe_dump(options))
    data = {
        "deployments": [
            {
                "name": "first",
                "options": "engine.yaml",
                "env": {"UCX_TLS": "^ib", "OMP_NUM_THREADS": 2},
                "timeout": 5400,
                "inputs": ["text", "image"],
                "probes": ["chunk"],
                "tests": ["tests/test_native.py::test_acceptance"],
                "benchmarks": [
                    {"name": "gsm8k"},
                    {
                        "name": "aa_lcr",
                        "anchor": 0.62,
                        "source": "measured baseline",
                        "unit": "fraction",
                        "env": {"JUDGE": "local"},
                        "args": ["--limit", "8"],
                        "through": "eval",
                    },
                ],
            }
        ]
    }
    deployments.resolve_paths(data, tmp_path)
    assert problems(data["deployments"]) == []
    [dep] = deployments.resolve(data)
    assert dep.first and dep.kind == "aggregated"
    assert dep.server.options == options
    assert dep.server.options_path == str(tmp_path / "engine.yaml")
    assert dep.timeout == 5400
    assert dep.env == {"UCX_TLS": "^ib", "OMP_NUM_THREADS": "2"}
    assert dep.inputs == ("text", "image") and dep.asked() == ("chunk",)
    assert dep.tests == ("tests/test_native.py::test_acceptance",)
    assert [(b.name, b.anchor, b.through) for b in dep.benchmarks] == [
        ("gsm8k", None, "endpoint"),
        ("aa_lcr", 0.62, "eval"),
    ]
    assert dep.benchmarks[1].source == "measured baseline"
    assert dep.benchmarks[1].env == {"JUDGE": "local"}
    # Written back, it resolves to the same deployment.
    again = deployments.resolve({"deployments": [dep.as_dict()]})
    assert [d.as_dict() for d in again] == [dep.as_dict()]


def test_a_disaggregated_deployment_has_unequal_roles_and_instances():
    ctx = {"tensor_parallel_size": 1, **MOVES}
    gen = {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2, **MOVES}
    first, pd = resolve(
        [
            {"name": "first", "options": {}},
            {
                "name": "pd",
                "ctx": {"options": ctx},
                "gen": {"options": gen, "instances": 2},
                "relative_to": "first",
                "env": {"UCX_TLS": "^ib"},
                "timeout": 100,
            },
        ]
    )
    assert first.first and not pd.first and pd.relative_to == "first"
    assert pd.kind == "disaggregated" and pd.ctx.options == ctx and pd.gen.options == gen
    assert [(i.name, i.server.ranks) for i in pd.instances()] == [
        ("ctx0", 1),
        ("gen0", 2),
        ("gen1", 2),
    ]
    assert pd.ranks == 5 and pd.shape == "pd5"
    assert pd.as_dict()["gen"] == {"options": gen, "instances": 2}


@pytest.mark.parametrize(
    ("options", "shape"),
    [
        ({}, "tp1"),
        ({"tensor_parallel_size": 2}, "tp2"),
        ({"tensor_parallel_size": 4, "moe_expert_parallel_size": 4}, "tep4"),
        ({"tensor_parallel_size": 4, "enable_attention_dp": True}, "dep4"),
        ({"tensor_parallel_size": 2, "pipeline_parallel_size": 2}, "tp4"),
        # Every parallel size the engine multiplies is a factor; experts add no rank.
        (
            {"tensor_parallel_size": 4, "pipeline_parallel_size": 2, "context_parallel_size": 3},
            "tp24",
        ),
    ],
)
def test_a_deployment_has_a_one_word_shape_with_its_devices(options, shape):
    [dep] = resolve([{"name": "first", "options": options}])
    assert dep.shape == shape and dep.ranks == int(shape.lstrip("tedp"))


def test_probes_follow_the_options_unless_the_deployment_states_them():
    on = {"kv_cache_config": {"enable_block_reuse": True}, "enable_chunked_prefill": True}
    derived, stated, none = resolve(
        [
            {"name": "derived", "options": on},
            {"name": "stated", "options": on, "probes": ["chunk"]},
            {"name": "none", "options": on, "probes": []},
        ]
    )
    assert derived.asked() == ("reuse", "chunk")
    assert stated.asked() == ("chunk",)
    assert none.asked() == ()


def test_a_drafter_is_read_from_the_generating_engine():
    drafting = {"speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    plain, drafts, pd = resolve(
        [
            {"name": "plain", "options": {"speculative_config": None}},
            {"name": "drafts", "options": drafting},
            {"name": "pd", "ctx": {"options": MOVES}, "gen": {"options": {**drafting, **MOVES}}},
        ]
    )
    assert not plain.drafts and drafts.drafts and pd.drafts


def test_benchmark_arguments_split_as_a_shell_would():
    args = '--stop "END OF ANSWER" --path "/data/a b.json" --empty ""'
    [dep] = resolve([{"name": "first", "options": {}, "benchmarks": [{"name": "g", "args": args}]}])
    assert dep.benchmarks[0].args == tuple(shlex.split(args))


def test_the_accuracy_block_gives_the_default_benchmark_and_its_defaults():
    accuracy = {
        "model_key": "org/toy",
        "reference_yaml": "references.yaml",
        "discriminators": {"quant_algo": "NVFP4"},
        "dataset_path": "/offline/data",
    }
    default, listed, none = resolve(
        [
            {"name": "first", "options": {}},
            {
                "name": "listed",
                "options": {},
                "benchmarks": [{"name": "aa_lcr", "anchor": 0.62, "source": "measured"}],
            },
            {"name": "none", "options": {}, "benchmarks": []},
        ],
        accuracy,
    )
    assert [b.name for b in default.benchmarks] == ["gsm8k"]
    assert default.benchmarks[0].model_key == "org/toy"
    assert default.benchmarks[0].discriminators == {"quant_algo": "NVFP4"}
    assert [(b.name, b.anchor, b.dataset_path) for b in listed.benchmarks] == [
        ("aa_lcr", 0.62, "/offline/data")
    ]
    assert none.benchmarks == ()
    # Without an `accuracy` block a deployment that lists nothing is judged by its boot.
    [bare] = resolve([{"name": "first", "options": {}}])
    assert bare.benchmarks == ()


def test_a_side_that_cannot_move_the_cache_is_one_line_with_its_fix():
    sides = {"ctx": {"options": {"cache_transceiver_config": {}}}, "gen": {"options": MOVES}}
    assert problems([{"name": "pd", **sides}]) == [
        "missing deployments.pd.ctx options cache_transceiver_config: a disaggregated server "
        "refuses its first request without a backend to move the cache over; fix: "
        "`cache_transceiver_config: {backend: DEFAULT}`"
    ]
    # An aggregated deployment moves no cache and owes no such key.
    assert problems([{"name": "tp1", "options": {}}]) == []


def test_a_deployment_states_no_launch_string():
    said = problems([{"name": "first", "options": {}, "launch": "srun -n 2"}])
    assert len(said) == 1 and "unknown key 'deployments[0].launch'" in said[0]
    said = problems(
        [{"name": "pd", "ctx": {"options": MOVES, "launch": "x"}, "gen": {"options": MOVES}}]
    )
    assert len(said) == 1 and "unknown key 'deployments.pd.ctx.launch'" in said[0]


@pytest.mark.parametrize(
    "block",
    [
        [],
        [{"name": "only_ctx", "ctx": {"options": MOVES}}],
        [{"name": "mixed", "options": {}, "ctx": {"options": MOVES}, "gen": {"options": MOVES}}],
        [{"name": "default", "options": {}}],
        [{"name": "Upper", "options": {}}],
        [{"name": "anchor", "options": {}, "benchmarks": [{"name": "gsm8k", "anchor": 80}]}],
        [{"name": "source", "options": {}, "benchmarks": [{"name": "gsm8k", "source": "x"}]}],
        [{"name": "badsize", "options": {"tensor_parallel_size": "many"}}],
        [{"name": "probe", "options": {}, "probes": ["warp"]}],
        [{"name": "notext", "options": {}, "inputs": ["image"]}],
        [{"name": "through", "options": {}, "benchmarks": [{"name": "g", "through": "magic"}]}],
        [{"name": "count", "ctx": {"options": MOVES, "instances": 0}, "gen": {"options": MOVES}}],
        [{"name": "test", "options": {}, "tests": ["/abs/test_x.py::test_y"]}],
        [{"name": "time", "options": {}, "timeout": 0}],
    ],
)
def test_a_block_that_is_not_a_deployment_list_is_refused(block):
    assert problems(block)


def test_a_reference_to_a_deployment_of_another_run_says_what_to_state_instead():
    """The second run of a large model: what it was judged against was the first run's."""
    (said,) = problems([{"name": "dep4", "options": {}, "relative_to": "tep4"}])
    assert said.startswith(
        "'deployments.dep4.relative_to' is 'tep4', which names no deployment before this one."
    )
    assert "(`onboarded: <ref>`) is not measured here" in said
    assert (
        "state the score that run measured as the benchmark's `anchor`, with its `source`" in said
    )
    # And what it tells to state is a block the spec takes.
    entry = {"name": "gsm8k", "anchor": 41.9, "source": "the first run's report"}
    assert not problems([{"name": "dep4", "options": {}, "benchmarks": [entry]}])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1])
def test_an_anchor_is_a_finite_score(score):
    entry = {"name": "gsm8k", "anchor": score, "source": "not a measurement"}
    assert problems([{"name": "first", "options": {}, "benchmarks": [entry]}])


def test_the_one_named_is_found_and_the_first_is_the_default():
    data = {"deployments": [{"name": "first", "options": {}}, {"name": "later", "options": {}}]}
    assert deployments.named(data).name == "first"
    assert deployments.named(data, "later").name == "later"
    with pytest.raises(KeyError, match="names no deployment of this spec \\(first, later\\)"):
        deployments.named(data, "absent")


def test_each_deployment_is_described_in_one_line():
    found = resolve(
        [
            {"name": "first", "options": {"tensor_parallel_size": 2}},
            {
                "name": "pd",
                "ctx": {"options": MOVES},
                "gen": {"options": MOVES, "instances": 2},
                "tests": ["tests/a.py::t"],
            },
        ],
        {"model_key": "org/toy"},
    )
    lines = deployments.describe(found, {"first:gsm8k": "reference row"}).splitlines()
    assert lines[0].startswith("- `first` (first): aggregated, 2 rank(s); one session")
    assert "gsm8k (reference row)" in lines[0]
    assert "ctx 1 rank(s); gen 2 x 1 rank(s)" in lines[1] and "test tests/a.py::t" in lines[1]
