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
"""Unit tests for the gen_only device-step-time bucket selection (nvbug 6843918).

expected_gen_only_ngen and _select_ngen_bucket choose which
num_generation_tokens bucket the published perf-sanity metric
mean_gen_worker_per_iter_device_step_time describes. A change to either can
silently move the published number, so their branches are pinned here.
"""

import pathlib
import sys
import types

import pytest
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "integration"))

from defs.perf import test_perf_sanity as tps  # noqa: E402


def _cfg(llm_args):
    """A stand-in gen ServerConfig exposing only merged_llm_api_config_data()."""
    return types.SimpleNamespace(merged_llm_api_config_data=lambda: dict(llm_args))


def _raising_cfg(exc):
    def _load():
        raise exc

    return types.SimpleNamespace(merged_llm_api_config_data=_load)


# ---------------------------------------------------------------------------
# expected_gen_only_ngen
# ---------------------------------------------------------------------------


def test_expected_ngen_nvbug_6843918_case():
    # gb300 DSv4-Pro con4301 gen dep8 mtp1: min(ceil(4301/8), 512, 1024//2) * 2.
    cfg = _cfg(
        {
            "tensor_parallel_size": 8,
            "enable_attention_dp": True,
            "max_batch_size": 512,
            "max_num_tokens": 1024,
            "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 1},
        }
    )
    assert tps.expected_gen_only_ngen(cfg, 4301, 1) == 1024


@pytest.mark.parametrize(
    ("enable_attention_dp", "expected"),
    [
        # ADP: each of the 4 ranks holds ceil(12/4) = 3 requests.
        (True, 3 * 4),
        # No ADP: every TP rank sees the whole batch of 12.
        (False, 12 * 4),
    ],
)
def test_expected_ngen_attention_dp(enable_attention_dp, expected):
    cfg = _cfg(
        {
            "tensor_parallel_size": 4,
            "enable_attention_dp": enable_attention_dp,
            "max_batch_size": 64,
            "max_num_tokens": 8192,
            "speculative_config": {"max_draft_len": 3},
        }
    )
    assert tps.expected_gen_only_ngen(cfg, 12, 1) == expected


def test_expected_ngen_max_draft_len_takes_precedence_over_nextn():
    cfg = _cfg(
        {
            "max_batch_size": 64,
            "max_num_tokens": 8192,
            "speculative_config": {"max_draft_len": 3, "num_nextn_predict_layers": 1},
        }
    )
    assert tps.expected_gen_only_ngen(cfg, 10, 1) == 10 * (1 + 3)


@pytest.mark.parametrize("spec_config", [None, {}, {"max_draft_len": None}])
def test_expected_ngen_without_speculation_is_one_token_per_request(spec_config):
    llm_args = {"max_batch_size": 64, "max_num_tokens": 8192}
    if spec_config is not None:
        llm_args["speculative_config"] = spec_config
    assert tps.expected_gen_only_ngen(_cfg(llm_args), 10, 1) == 10


def test_expected_ngen_capped_by_max_batch_size():
    cfg = _cfg({"max_batch_size": 16, "max_num_tokens": 8192})
    assert tps.expected_gen_only_ngen(cfg, 100, 1) == 16


def test_expected_ngen_capped_by_max_num_tokens():
    # max_num_tokens // (1 + max_draft_len) = 100 // 4 = 25 requests per rank.
    cfg = _cfg(
        {
            "max_batch_size": 512,
            "max_num_tokens": 100,
            "speculative_config": {"max_draft_len": 3},
        }
    )
    assert tps.expected_gen_only_ngen(cfg, 1000, 1) == 25 * 4


def test_expected_ngen_split_across_gen_servers():
    cfg = _cfg({"max_batch_size": 512, "max_num_tokens": 8192})
    assert tps.expected_gen_only_ngen(cfg, 10, 3) == 4  # ceil(10 / 3)


@pytest.mark.parametrize("missing", ["max_batch_size", "max_num_tokens"])
def test_expected_ngen_none_without_batch_or_token_budget(missing):
    llm_args = {"max_batch_size": 64, "max_num_tokens": 8192}
    del llm_args[missing]
    assert tps.expected_gen_only_ngen(_cfg(llm_args), 10, 1) is None


@pytest.mark.parametrize(
    ("gen_config", "concurrency", "num_gen_servers"),
    [
        (None, 10, 1),
        (_cfg({"max_batch_size": 64, "max_num_tokens": 8192}), 0, 1),
        (_cfg({"max_batch_size": 64, "max_num_tokens": 8192}), 10, 0),
        # A token budget smaller than one request's tokens gives 0 per rank.
        (
            _cfg(
                {
                    "max_batch_size": 64,
                    "max_num_tokens": 2,
                    "speculative_config": {"max_draft_len": 3},
                }
            ),
            10,
            1,
        ),
    ],
)
def test_expected_ngen_none_for_degenerate_inputs(gen_config, concurrency, num_gen_servers):
    assert tps.expected_gen_only_ngen(gen_config, concurrency, num_gen_servers) is None


@pytest.mark.parametrize("exc", [OSError("missing"), yaml.YAMLError("bad yaml")])
def test_expected_ngen_none_when_config_cannot_load(exc):
    assert tps.expected_gen_only_ngen(_raising_cfg(exc), 10, 1) is None


def test_expected_ngen_reads_merged_external_config(tmp_path):
    # The batch/token budgets live in the external extra_llm_api_config file and
    # the perf YAML overrides max_batch_size; the merged view must be used.
    external = tmp_path / "extra.yaml"
    external.write_text(yaml.safe_dump({"max_batch_size": 512, "max_num_tokens": 8192}))
    cfg = object.__new__(tps.ServerConfig)
    cfg.extra_llm_api_config_data = {"max_batch_size": 8}
    cfg.extra_llm_api_config_path = str(external)
    assert tps.expected_gen_only_ngen(cfg, 100, 1) == 8


# ---------------------------------------------------------------------------
# _select_ngen_bucket
# ---------------------------------------------------------------------------


def _buckets(**counts):
    """{ngen: [ngen-valued samples] * count}, so the chosen bucket is identifiable."""
    return {int(k[1:]): [float(k[1:])] * n for k, n in counts.items()}


def test_select_uses_mode_without_expectation():
    assert tps._select_ngen_bucket(_buckets(n64=441, n1024=440), None) == [64.0] * 441


def test_select_mode_ties_break_to_largest_ngen():
    assert tps._select_ngen_bucket(_buckets(n64=10, n1024=10), None) == [1024.0] * 10


def test_select_expected_equal_to_mode():
    assert tps._select_ngen_bucket(_buckets(n64=10, n1024=20), 1024) == [1024.0] * 20


def test_select_expected_overrides_near_tied_mode():
    # nvbug 6843918: 441 iterations at ngen=64 vs 440 at the expected 1024.
    assert tps._select_ngen_bucket(_buckets(n64=441, n1024=440), 1024) == [1024.0] * 440


def test_select_expected_at_trust_floor_is_used():
    assert tps._EXPECTED_NGEN_MIN_MODE_FRACTION == 0.5
    assert tps._select_ngen_bucket(_buckets(n64=100, n1024=50), 1024) == [1024.0] * 50


def test_select_expected_below_trust_floor_falls_back_to_mode(capsys):
    assert tps._select_ngen_bucket(_buckets(n64=100, n1024=49), 1024) == [64.0] * 100
    assert "falling back to the modal bucket" in capsys.readouterr().out


def test_select_absent_expected_bucket_falls_back_to_mode(capsys):
    assert tps._select_ngen_bucket(_buckets(n64=100), 1024) == [64.0] * 100
    assert "falling back to the modal bucket" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _stats_at_mode_ngen: the published statistic follows the selected bucket
# ---------------------------------------------------------------------------


def _rows(**counts):
    rows = []
    for key, n in counts.items():
        ngen = int(key[1:])
        rows += [tps._IterRow(ngen=ngen, device_step_time=float(ngen) / 10)] * n
    return rows


def test_stats_follow_expected_bucket():
    per_file_rows = [_rows(n64=441, n1024=440)]
    assert tps._stats_at_mode_ngen(per_file_rows).mean == pytest.approx(6.4)
    assert tps._stats_at_mode_ngen(per_file_rows, 1024).mean == pytest.approx(102.4)


def test_stats_without_parseable_ngen_use_whole_sample():
    rows = [tps._IterRow(ngen=None, device_step_time=t) for t in (1.0, 3.0)]
    assert tps._stats_at_mode_ngen([rows], 1024).mean == pytest.approx(2.0)
