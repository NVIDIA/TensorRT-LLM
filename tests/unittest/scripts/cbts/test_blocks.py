#!/usr/bin/env python3
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
"""Tests for CBTS test-db loading and narrowed test-db writing when a file holds several contexts."""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest
import yaml

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.blocks import (
    Stage,
    YAMLIndex,
    _estimate_entries_seconds,
    parse_stages_from_groovy,
    write_filtered_test_db,
)
from cbts.command.report_decision import _stage_group
from cbts.coverage.selection.touch_db import stage_family
from cbts.coverage.tier import write_coverage_test_db
from cbts.rules import test_list as test_list_mod
from cbts.rules._helpers import lookup_paths_into_block_filters
from cbts.rules.base import PRInputs
from cbts.rules.spec_dec import SpecDecRule
from cbts.rules.waives import WAIVES_FILE, WaivesRule

pytestmark = pytest.mark.cpu_only

_CONDITION = {"terms": {"stage": "post_merge", "backend": "pytorch"}}


@pytest.mark.parametrize("include_post_merge", [False, True])
@pytest.mark.parametrize("suffix", ["", "-{split}_of_8"])
def test_parse_generated_stages(tmp_path: Path, include_post_merge: bool, suffix: str) -> None:
    """Three splits in an eight-build cycle create only three SBSA stages."""
    groovy = tmp_path / "L0_Test.groovy"
    groovy.write_text(
        "multiNodesSBSAConfigs += buildStageConfigs(\n"
        f'    "GB300-12_GPUs-PerfSanity-Post-Merge{suffix}",\n'
        '    "auto:gb300-flex", "perf_context", 3, 12, 3\n'
        ")\n"
        '// x86SlurmTestConfigs += buildStageConfigs("Disabled", "x86", "disabled", 1, 8, 1)\n'
    )
    stages = parse_stages_from_groovy(groovy, include_post_merge=include_post_merge)
    assert list(stages) == (
        [
            f"GB300-12_GPUs-PerfSanity-Post-Merge-{k}" + ("_of_8" if suffix else "")
            for k in range(1, 4)
        ]
        if include_post_merge
        else []
    )
    assert all(stage.cpu_arch == "sbsa" and stage.total_splits == 3 for stage in stages.values())
    assert [stage.split_id for stage in stages.values()] == (
        [1, 2, 3] if include_post_merge else []
    )


@pytest.mark.parametrize("suffix", ["-1", "-2_of_8", "-3_of_8-cbts"])
def test_scheduled_stage_family(suffix: str) -> None:
    """Cadence names remain shards of one family in both CBTS readers."""
    prefix = "GB300-PyTorch-PerfSanity-Post-Merge"
    assert _stage_group(prefix + suffix) == prefix
    assert stage_family(prefix + suffix) == prefix


def _write_test_db(test_db_dir: Path) -> None:
    test_db_dir.mkdir()
    (test_db_dir / "l0_x_multi_nodes_perf_sanity.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_x_multi_nodes_perf_sanity_a": [
                    {
                        "condition": _CONDITION,
                        "tests": ["perf/test_a.py::test_one", "perf/test_a.py::test_two"],
                    }
                ],
                "l0_x_multi_nodes_perf_sanity_b": [
                    {"condition": _CONDITION, "tests": ["perf/test_b.py::test_one"]}
                ],
            },
            sort_keys=False,
        )
    )
    (test_db_dir / "l0_x.yml").write_text(
        yaml.safe_dump(
            {"version": "0.0.1", "l0_x": [{"condition": _CONDITION, "tests": ["unit/test_c.py"]}]}
        )
    )


def test_yaml_index_loads_every_context_of_a_file(tmp_path: Path) -> None:
    """Each context of a multi-context file gets its own blocks and waive lookup."""
    _write_test_db(tmp_path / "test-db")
    index = YAMLIndex.load(tmp_path / "test-db")

    by_context = {block.yaml_stem: block.tests for block in index.blocks}
    assert by_context == {
        "l0_x_multi_nodes_perf_sanity_a": ["perf/test_a.py::test_one", "perf/test_a.py::test_two"],
        "l0_x_multi_nodes_perf_sanity_b": ["perf/test_b.py::test_one"],
        "l0_x": ["unit/test_c.py"],
    }
    matches = index.find_matches_for_waive("perf/test_b.py::test_one")
    assert matches is not None and [(b.yaml_stem, lvl) for b, lvl in matches] == [
        ("l0_x_multi_nodes_perf_sanity_b", "perf/test_b.py::test_one")
    ]


def test_narrowed_test_db_has_one_file_per_context(tmp_path: Path) -> None:
    """Both narrowed test-dbs write each affected context alone to `<context>.yml`."""
    _write_test_db(tmp_path / "test-db")

    filtered = tmp_path / "filtered"
    write_filtered_test_db(
        tmp_path / "test-db",
        filtered,
        {("l0_x_multi_nodes_perf_sanity_b", 0): {"perf/test_b.py": {"perf/test_b.py::test_one"}}},
    )
    assert sorted(p.name for p in filtered.glob("*.yml")) == ["l0_x_multi_nodes_perf_sanity_b.yml"]
    assert yaml.safe_load((filtered / "l0_x_multi_nodes_perf_sanity_b.yml").read_text()) == {
        "version": "0.0.1",
        "l0_x_multi_nodes_perf_sanity_b": [
            {"condition": _CONDITION, "tests": ["perf/test_b.py::test_one"]}
        ],
    }

    coverage = tmp_path / "coverage"
    write_coverage_test_db(
        tmp_path / "test-db",
        coverage,
        {("l0_x_multi_nodes_perf_sanity_a", 0): {"perf/test_a.py::test_two"}},
    )
    assert sorted(p.name for p in coverage.glob("*.yml")) == ["l0_x_multi_nodes_perf_sanity_a.yml"]
    assert yaml.safe_load((coverage / "l0_x_multi_nodes_perf_sanity_a.yml").read_text()) == {
        "version": "0.0.1",
        "l0_x_multi_nodes_perf_sanity_a": [
            {"condition": _CONDITION, "tests": ["perf/test_a.py::test_one"]}
        ],
    }


def test_yaml_index_strips_ignore_options_from_directory_targets(tmp_path: Path) -> None:
    """Directory entries carrying `--ignore` / `--ignore-glob` exclude ignored subtrees in waive and path lookups."""
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    entry = (
        "unittest/_torch/attention/sparse "
        "--import-mode=importlib "
        "--ignore=unittest/_torch/attention/sparse/dsa "
        "--ignore unittest/_torch/attention/sparse/multi_gpu "
        '--ignore-glob="unittest/_torch/attention/sparse/msa/*" '
        '--ignore-glob="*/parent_ignored" '
        '--ignore-glob="*/test_*/*" '
        '--ignore-glob="*slow*"'
    )
    (test_db / "l0_b200.yml").write_text(
        yaml.safe_dump(
            {"version": "0.0.1", "l0_b200": [{"condition": _CONDITION, "tests": [entry]}]},
            sort_keys=False,
        )
    )
    (test_db / "l0_h100.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_h100": [
                    {
                        "condition": _CONDITION,
                        "tests": ["unittest/_torch/attention/sparse/msa/test_proxy.py"],
                    }
                ],
            },
            sort_keys=False,
        )
    )
    index = YAMLIndex.load(test_db)
    for retained in (
        "unittest/_torch/attention/sparse/csa2/test_backend.py::test_foo",
        "unittest/_torch/attention/sparse/csa2/test_backend.py::test_slow",
        "unittest/_torch/attention/sparse/test_cache.py::test_eviction",
    ):
        matches = index.find_matches_for_waive(retained)
        assert matches is not None and [(b.yaml_stem, lvl) for b, lvl in matches] == [
            ("l0_b200", "unittest/_torch/attention/sparse")
        ]
        assert index.find_match_for_path(retained) is not None
    for excluded in (
        "unittest/_torch/attention/sparse/dsa/test_bar.py",
        "unittest/_torch/attention/sparse/multi_gpu/test_baz.py",
        "unittest/_torch/attention/sparse/parent_ignored/deep/test_nested.py",
        "unittest/_torch/attention/sparse/test_dir/test_inside.py",
        "unittest/_torch/attention/sparse/csa2/test_slow_ops.py::test_x",
    ):
        assert index.find_matches_for_waive(excluded) is None
        assert index.find_match_for_path(excluded) is None
    # `msa/*` is excluded on l0_b200 for both test files and walked-up data files, selecting only l0_h100.
    for msa_path in (
        "unittest/_torch/attention/sparse/msa/test_proxy.py",
        "unittest/_torch/attention/sparse/msa/config.yaml",
    ):
        path_matches = index.find_match_for_path(msa_path)
        assert path_matches is not None and [b.yaml_stem for b, _ in path_matches] == ["l0_h100"]


def test_waives_rule_resolves_per_block_across_file_and_directory_registrations(
    tmp_path: Path,
) -> None:
    """A file registration on one platform must not shadow a directory batch on another."""
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    pre_cond = {"terms": {"stage": "pre_merge", "backend": "pytorch"}}
    dir_batch = (
        "unittest/_torch/modules/kimi_kda "
        "--ignore=unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py"
    )
    prefill_file = "unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py"
    verify_file = "unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py"
    (test_db / "l0_gb300.yml").write_text(
        yaml.safe_dump(
            {"version": "0.0.1", "l0_gb300": [{"condition": pre_cond, "tests": [verify_file]}]},
            sort_keys=False,
        )
    )
    (test_db / "l0_b200.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_b200": [{"condition": pre_cond, "tests": [prefill_file, dir_batch]}],
            },
            sort_keys=False,
        )
    )
    index = YAMLIndex.load(test_db)
    stages = {
        "GB300-4_GPUs-PyTorch-1": Stage("GB300-4_GPUs-PyTorch-1", "l0_gb300", "sbsa", 1, 1, {}),
        "DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1, {}),
    }
    rule = WaivesRule(index, stages)

    waive_diff = f"@@ -1,1 +1,2 @@\n+{verify_file}::test_two_cta SKIP (https://nvbugs/1)\n"
    res = rule.apply(PRInputs(changed_files={WAIVES_FILE}, diffs={WAIVES_FILE: waive_diff}))
    assert res is not None and res.scope == "waiveonly"
    assert res.affected_stages == {"GB300-4_GPUs-PyTorch-1", "DGX_B200-PyTorch-1"}
    filtered = tmp_path / "filtered_verify"
    write_filtered_test_db(test_db, filtered, res.block_filters)
    assert yaml.safe_load((filtered / "l0_gb300.yml").read_text())["l0_gb300"][0]["tests"] == [
        verify_file
    ]
    assert yaml.safe_load((filtered / "l0_b200.yml").read_text())["l0_b200"][0]["tests"] == [
        dir_batch
    ]

    ignore_diff = f"@@ -1,1 +1,2 @@\n+{prefill_file}::test_prefill SKIP (https://nvbugs/2)\n"
    res2 = rule.apply(PRInputs(changed_files={WAIVES_FILE}, diffs={WAIVES_FILE: ignore_diff}))
    assert res2 is not None and res2.affected_stages == {"DGX_B200-PyTorch-1"}
    filtered2 = tmp_path / "filtered_prefill"
    write_filtered_test_db(test_db, filtered2, res2.block_filters)
    assert yaml.safe_load((filtered2 / "l0_b200.yml").read_text())["l0_b200"][0]["tests"] == [
        prefill_file
    ]


def test_deselect_preserves_parameter_boundaries(tmp_path: Path) -> None:
    """`--deselect=file::test[a]` excludes `[a]` while retaining `[b]` and unparameterized waives."""
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    entry = 'unit/test_x.py -k "test_case" --deselect=unit/test_x.py::test_case[a]'
    (test_db / "l0_b200.yml").write_text(
        yaml.safe_dump(
            {"version": "0.0.1", "l0_b200": [{"condition": _CONDITION, "tests": [entry]}]},
            sort_keys=False,
        )
    )
    index = YAMLIndex.load(test_db)
    assert index.find_matches_for_waive("unit/test_x.py::test_case[a]") is None
    assert index.find_matches_for_waive("unit/test_x.py::test_case[b]") is not None
    assert index.find_matches_for_waive("unit/test_x.py::test_case") is not None


def test_waives_rule_handles_dotted_file_and_negated_k_filters(tmp_path: Path) -> None:
    """`-k "a.py or b.py"` rejects `c.py`, while `-k "not NVFP4"` over-includes non-NVFP4 waives."""
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    pre_cond = {"terms": {"stage": "pre_merge", "backend": "pytorch"}}
    sampler_k = (
        'unittest/_torch/sampler -k "test_in_graph_sampling.py or test_finish_reasons_fused.py"'
    )
    kv_not_k = 'unittest/disaggregated/test_deepseek_v4_kv_transfer.py -k "not NVFP4"'
    (test_db / "l0_a10.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_a10": [{"condition": pre_cond, "tests": ["unittest/_torch/sampler"]}],
            },
            sort_keys=False,
        )
    )
    (test_db / "l0_b200.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_b200": [{"condition": pre_cond, "tests": [sampler_k, kv_not_k]}],
            },
            sort_keys=False,
        )
    )
    index = YAMLIndex.load(test_db)
    stages = {
        "A10-PyTorch-1": Stage("A10-PyTorch-1", "l0_a10", "x86_64", 1, 1, {}),
        "DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1, {}),
    }
    rule = WaivesRule(index, stages)

    penalties_diff = "@@ -1,1 +1,2 @@\n+unittest/_torch/sampler/test_penalties.py::test_x SKIP (https://nvbugs/1)\n"
    res_pen = rule.apply(PRInputs(changed_files={WAIVES_FILE}, diffs={WAIVES_FILE: penalties_diff}))
    assert res_pen is not None and res_pen.affected_stages == {"A10-PyTorch-1"}

    sampler_diff = (
        "@@ -1,1 +1,2 @@\n"
        "+unittest/_torch/sampler/test_in_graph_sampling.py::test_x SKIP (https://nvbugs/3)\n"
    )
    res_sampler = rule.apply(
        PRInputs(changed_files={WAIVES_FILE}, diffs={WAIVES_FILE: sampler_diff})
    )
    assert res_sampler is not None and res_sampler.affected_stages == {
        "A10-PyTorch-1",
        "DGX_B200-PyTorch-1",
    }

    non_fp4_diff = (
        "@@ -1,1 +1,2 @@\n"
        "+unittest/disaggregated/test_deepseek_v4_kv_transfer.py::test_transfer[bf16] SKIP (https://nvbugs/2)\n"
    )
    res_kv = rule.apply(PRInputs(changed_files={WAIVES_FILE}, diffs={WAIVES_FILE: non_fp4_diff}))
    assert res_kv is not None and res_kv.affected_stages == {"DGX_B200-PyTorch-1"}

    # Unparameterized function unwaive must retain parameter-level `-k "NVFP4"` batches.
    kv_fp4_only = 'unittest/disaggregated/test_deepseek_v4_kv_transfer.py -k "NVFP4"'
    (test_db / "l0_b200.yml").write_text(
        yaml.safe_dump(
            {"version": "0.0.1", "l0_b200": [{"condition": pre_cond, "tests": [kv_fp4_only]}]},
            sort_keys=False,
        )
    )
    rule_fp4 = WaivesRule(YAMLIndex.load(test_db), stages)
    unwaive_func_diff = (
        "@@ -1,2 +1,1 @@\n"
        "-unittest/disaggregated/test_deepseek_v4_kv_transfer.py::test_deepseek_v4_native_sparse_host_transfer SKIP (https://nvbugs/4)\n"
    )
    res_unwaive = rule_fp4.apply(
        PRInputs(changed_files={WAIVES_FILE}, diffs={WAIVES_FILE: unwaive_func_diff})
    )
    assert res_unwaive is not None and res_unwaive.affected_stages == {"DGX_B200-PyTorch-1"}
    filtered_fp4 = tmp_path / "filtered_fp4"
    write_filtered_test_db(test_db, filtered_fp4, res_unwaive.block_filters)
    assert yaml.safe_load((filtered_fp4 / "l0_b200.yml").read_text())["l0_b200"][0]["tests"] == [
        kv_fp4_only
    ]


def test_lookup_paths_preserves_walked_up_config_matches_alongside_direct_test(
    tmp_path: Path,
) -> None:
    """A walked-up support-file match must survive alongside a direct test match in the same block."""
    test_db = tmp_path / "test-db"
    test_db.mkdir()
    entries = [
        "disaggregated/test_disaggregated.py::test_disaggregated_cache_aware_balance",
        "other_suite/test_unrelated.py::test_keep_when_touched",
        "other_suite/test_untouched.py::test_drop",
    ]
    (test_db / "l0_a10.yml").write_text(
        yaml.safe_dump(
            {"version": "0.0.1", "l0_a10": [{"condition": _CONDITION, "tests": entries}]},
            sort_keys=False,
        )
    )
    index = YAMLIndex.load(test_db)
    bf, unmatched = lookup_paths_into_block_filters(
        index,
        [
            "disaggregated/test_configs/disagg_config_cache_aware_balance.yaml",
            "other_suite/test_unrelated.py::test_keep_when_touched",
        ],
    )
    assert unmatched == []
    filtered = tmp_path / "filtered"
    write_filtered_test_db(test_db, filtered, bf)
    assert yaml.safe_load((filtered / "l0_a10.yml").read_text())["l0_a10"][0]["tests"] == [
        "disaggregated/test_disaggregated.py::test_disaggregated_cache_aware_balance",
        "other_suite/test_unrelated.py::test_keep_when_touched",
    ]

    # Editing one file under a `--ignore` directory batch must not re-include the ignored sibling file.
    dir_batch = (
        "unittest/_torch/modules/kimi_kda "
        "--ignore=unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py"
    )
    prefill = "unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py"
    (test_db / "l0_b200.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_b200": [{"condition": _CONDITION, "tests": [prefill, dir_batch]}],
            },
            sort_keys=False,
        )
    )
    bf_kda, unm_kda = lookup_paths_into_block_filters(
        YAMLIndex.load(test_db),
        ["unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py"],
    )
    assert unm_kda == []
    filtered_kda = tmp_path / "filtered_kda"
    write_filtered_test_db(test_db, filtered_kda, bf_kda)
    assert yaml.safe_load((filtered_kda / "l0_b200.yml").read_text())["l0_b200"][0]["tests"] == [
        dir_batch
    ]

    # SpecDecRule selects standalone `_mtp` entries without dragging in `--ignore=..._mtp...` batches.
    mtp_file = "unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py"
    kda_batch = f"{dir_batch} --ignore={mtp_file}"
    (test_db / "l0_b200.yml").write_text(
        yaml.safe_dump(
            {
                "version": "0.0.1",
                "l0_b200": [
                    {
                        "condition": {"terms": {"stage": "pre_merge", "backend": "pytorch"}},
                        "tests": [prefill, mtp_file, kda_batch],
                    }
                ],
            },
            sort_keys=False,
        )
    )
    stages = {"DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1, {})}
    sd_res = SpecDecRule(YAMLIndex.load(test_db), stages).apply(
        PRInputs(
            changed_files={"tensorrt_llm/_torch/speculative/mtp.py"},
            diffs={"tensorrt_llm/_torch/speculative/mtp.py": "@@ -1,1 +1,2 @@\n+# x\n"},
        )
    )
    assert sd_res is not None and sd_res.affected_stages == {"DGX_B200-PyTorch-1"}
    filtered_sd = tmp_path / "filtered_sd"
    write_filtered_test_db(test_db, filtered_sd, sd_res.block_filters)
    assert yaml.safe_load((filtered_sd / "l0_b200.yml").read_text())["l0_b200"][0]["tests"] == [
        mtp_file
    ]


def test_test_list_rule_b200_addition_with_cpu_deletion_leaves_cpu_unfiltered(
    tmp_path: Path,
) -> None:
    """Adding a B200 test while only deleting from l0_cpu.yml must not narrow l0_cpu."""
    test_db = tmp_path / "tests/integration/test_lists/test-db"
    test_db.mkdir(parents=True)
    pre_cond = {"terms": {"stage": "pre_merge", "backend": "pytorch"}}
    post_cond = {"terms": {"stage": "post_merge", "backend": "pytorch"}}
    added_test = "unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py"
    b200_rel = "tests/integration/test_lists/test-db/l0_b200.yml"
    cpu_rel = "tests/integration/test_lists/test-db/l0_cpu.yml"
    b200_pre = yaml.safe_dump(
        {
            "version": "0.0.1",
            "l0_b200": [
                {"condition": pre_cond, "tests": ["unittest/test_keep.py"]},
                {"condition": post_cond, "tests": [added_test]},
            ],
        },
        sort_keys=False,
    )
    b200_post = yaml.safe_dump(
        {
            "version": "0.0.1",
            "l0_b200": [
                {"condition": pre_cond, "tests": ["unittest/test_keep.py", added_test]},
                {"condition": post_cond, "tests": [added_test]},
            ],
        },
        sort_keys=False,
    )
    cpu_pre = yaml.safe_dump(
        {
            "version": "0.0.1",
            "l0_cpu": [
                {
                    "condition": pre_cond,
                    "tests": [
                        "unittest/_torch/modeling",
                        "examples/test_removed.py::test_old",
                        "unittest/scripts",
                    ],
                }
            ],
        },
        sort_keys=False,
    )
    cpu_post = yaml.safe_dump(
        {
            "version": "0.0.1",
            "l0_cpu": [
                {
                    "condition": pre_cond,
                    "tests": ["unittest/_torch/modeling", "unittest/scripts"],
                }
            ],
        },
        sort_keys=False,
    )
    (tmp_path / b200_rel).write_text(b200_post)
    (tmp_path / cpu_rel).write_text(cpu_post)
    index = YAMLIndex.load(test_db)
    stages = {
        "DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1, {}),
        "CPU-Generic-1": Stage("CPU-Generic-1", "l0_cpu", "x86_64", 1, 1, {}),
    }
    rule = test_list_mod.TestListRule(index, stages, tmp_path)
    b200_diff = "".join(difflib.unified_diff(b200_pre.splitlines(True), b200_post.splitlines(True)))
    cpu_diff = "".join(difflib.unified_diff(cpu_pre.splitlines(True), cpu_post.splitlines(True)))
    res = rule.apply(
        PRInputs(
            changed_files={b200_rel, cpu_rel},
            diffs={b200_rel: b200_diff, cpu_rel: cpu_diff},
        )
    )
    assert res is not None and res.scope == "testlistonly"
    assert res.affected_stages == {"DGX_B200-PyTorch-1"}
    assert set(res.block_filters) == {("l0_b200", 0)}
    filtered = tmp_path / "filtered"
    write_filtered_test_db(test_db, filtered, res.block_filters)
    assert sorted(p.name for p in filtered.glob("*.yml")) == ["l0_b200.yml"]
    assert yaml.safe_load((filtered / "l0_b200.yml").read_text())["l0_b200"] == [
        {"condition": pre_cond, "tests": [added_test]}
    ]


def test_test_list_rule_multiline_wrapped_selector_schedules_stage(tmp_path: Path) -> None:
    """Wrapped multiline YAML test additions resolve to the full folded selector."""
    test_db = tmp_path / "tests/integration/test_lists/test-db"
    test_db.mkdir(parents=True)
    b200_rel = "tests/integration/test_lists/test-db/l0_b200.yml"
    pre_text = (
        "version: 0.0.1\n"
        "l0_b200:\n"
        "- condition:\n"
        "    terms:\n"
        "      stage: pre_merge\n"
        "      backend: pytorch\n"
        "  tests:\n"
        "  - unittest/test_keep.py\n"
    )
    post_text = (
        pre_text
        + '  - unittest/_torch/moe/test_moe_backend.py -k "test_cutlass_moe_routing_round_trip or\n'
        + '    test_cutlass_moe_public_activation_ids"\n'
    )
    (tmp_path / b200_rel).write_text(post_text)
    index = YAMLIndex.load(test_db)
    stages = {"DGX_B200-PyTorch-1": Stage("DGX_B200-PyTorch-1", "l0_b200", "x86_64", 1, 1, {})}
    rule = test_list_mod.TestListRule(index, stages, tmp_path)
    diff = "".join(difflib.unified_diff(pre_text.splitlines(True), post_text.splitlines(True)))
    res = rule.apply(PRInputs(changed_files={b200_rel}, diffs={b200_rel: diff}))
    assert res is not None and res.scope == "testlistonly"
    assert res.affected_stages == {"DGX_B200-PyTorch-1"}
    filtered = tmp_path / "filtered_multiline"
    write_filtered_test_db(test_db, filtered, res.block_filters)
    expected = (
        'unittest/_torch/moe/test_moe_backend.py -k "test_cutlass_moe_routing_round_trip or '
        'test_cutlass_moe_public_activation_ids"'
    )
    assert yaml.safe_load((filtered / "l0_b200.yml").read_text())["l0_b200"][0]["tests"] == [
        expected
    ]


def test_estimate_entries_seconds_reads_wrapped_unittest_batch_durations() -> None:
    """`_estimate_entries_seconds` hits `test_unittests_v2[...]` keys before subtree fallback."""
    batch = "unittest/_torch/modules/kimi_kda --ignore=unittest/_torch/modules/kimi_kda/test_a.py"
    durations = {
        f"test_unittests.py::test_unittests_v2[{batch}]": 416.52,
        "unittest/_torch/modules/kimi_kda/test_b.py::test_slow": 1700.0,
    }
    assert _estimate_entries_seconds([batch], durations, avg=100.0) == pytest.approx(416.52)
