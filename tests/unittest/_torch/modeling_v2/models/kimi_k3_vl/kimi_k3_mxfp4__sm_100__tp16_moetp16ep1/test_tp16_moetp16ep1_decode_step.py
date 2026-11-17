# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target's step wiring (host-side, no GPU): its text model keeps no per-token KDA states, and runs
one path for a step through every layer. The family's ``test_kimi_k3_step.py`` and ``test_kimi_k3_attn_res.py`` check
how a step is classified."""

import pytest

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_decode_step import (  # noqa: E402
    STEP_PATHS,
    check_each_step_takes_one_path_through_every_layer,
)

try:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        modeling as target,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


def test_kda_draft_records_off():
    """The hybrid cache manager keeps no per-token KDA states, and the latent-push decision reads that."""
    assert object.__new__(target.KimiLinearModel).kda_draft_records is False


@pytest.mark.parametrize("seq_lens,num_contexts,rows,path", STEP_PATHS)
def test_each_step_takes_one_path_through_every_layer(seq_lens, num_contexts, rows, path):
    check_each_step_takes_one_path_through_every_layer(target, seq_lens, num_contexts, rows, path)
