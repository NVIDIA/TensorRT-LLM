# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp4ep4 target's construction-time checks, host-side: it requires its own parallel layout, its routed
experts' checkpoint spec meets the shared loader and its DSpark worker's markov bias the shared worker."""

import pytest

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_construction import (  # noqa: E402, F401
    check_dspark_markov_bias_meets_the_shared_worker_contract,
    check_expert_ckpt_spec_meets_the_shared_loader_contract,
    config,
    sm_100,
)

try:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        modeling as target,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        spec_worker,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.usefixtures("sm_100")


def test_requires_its_layout():
    target._check_construction(config(4, 4))
    with pytest.raises(AssertionError, match="the tp16_moetp4ep4 target needs"):
        target._check_construction(config(16, 1))
    with pytest.raises(AssertionError, match="enable_attention_dp false"):
        target._check_construction(config(4, 4, attention_dp=True))


def test_expert_ckpt_spec_meets_the_shared_loader_contract():
    check_expert_ckpt_spec_meets_the_shared_loader_contract(target)


def test_dspark_markov_bias_meets_the_shared_worker_contract(monkeypatch):
    check_dspark_markov_bias_meets_the_shared_worker_contract(spec_worker, monkeypatch)
