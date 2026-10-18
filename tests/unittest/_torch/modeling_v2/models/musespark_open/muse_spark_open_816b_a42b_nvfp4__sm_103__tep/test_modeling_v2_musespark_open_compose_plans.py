# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The composition harnesses' arm ids as dispatch plans (``_compose_plans.py``), host-only."""

from __future__ import annotations

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)

__extra_import_path__ = ["."]
import _compose_plans as plans  # noqa: E402

pytestmark = pytest.mark.cpu_only


def test_every_fast_path_carries_ids_and_no_id_is_shared_but_the_staging_pair():
    assert set(plans.PATH_IDS) == dispatch.FAST_PATH_NAMES
    owners = {}
    for path, ids in plans.PATH_IDS.items():
        assert ids, path
        for token in ids:
            owners.setdefault(token, set()).add(path)
    shared = {token: paths for token, paths in owners.items() if len(paths) > 1}
    assert shared == {"input_staging": {"deferred_input_prep", "sampler_store"}}


def test_the_tp8_composition_carries_the_served_ids():
    ids = plans.ids_of(dispatch.Plan(modeling.TARGET_PATHS[103, 8].served))
    # 47 served ids, and the two the served ones imply (the dense GEMMs, the split sampler).
    assert len(ids) == 49 and {"dense_gemms", "split_sampler"} <= ids


@pytest.mark.parametrize(
    "value, paths",
    [
        ("", set()),
        ("hop", {"post_attention_hop"}),
        ("hop_early_trigger,stale_poll", {"post_attention_hop"}),
        ("gemm_prefetch", {"dense_gemms", "dense_gemm_prefetch"}),
        ("swa_block", {"preattn_prep", "swa_paged_decode", "swa_block"}),
        ("input_staging", {"deferred_input_prep", "sampler_store"}),
        ("PREATTN_PREP, gathers", {"preattn_prep", "projection_gathers"}),
    ],
)
def test_an_arm_is_the_plan_of_its_ids_paths(value, paths):
    plan = plans.plan_of_ids(value)
    assert plan.paths == dispatch.closure(paths)
    assert plans.selection(value) == plans.ids_of(plan)


def test_the_built_in_arm_has_no_plan():
    assert plans.plan_of_ids(None) is None
    assert plans.selection(None) == frozenset()


@pytest.mark.parametrize(
    "value",
    [
        "paged_decode",
        "preattn_prep,paged_decode",
        "gemm_prefetch_pinned",
        "ring_buddy_static",
        "gathers_unpool_only",
        "stale_poll_any",
    ],
)
def test_an_id_no_fast_path_carries_is_not_a_plan(value):
    with pytest.raises(plans.NotAPlan):
        plans.plan_of_ids(value)
    assert not plans.is_plan(value)


def test_served_arms_leaves_out_the_arms_that_are_not_plans():
    arms = (
        ("builtin", None, None),
        ("a", "paged_decode", None),
        ("b", "paged_decode_q8", "a"),
        ("c", "preattn_prep", "builtin"),
    )
    assert plans.served_arms(arms) == (
        ("builtin", None, None),
        ("b", "paged_decode_q8", None),
        ("c", "preattn_prep", "builtin"),
    )
