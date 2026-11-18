# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-GSM8K gate of kimi_k3_nvfp4__sm_107__dep16 on SM107.

Its generation server (attention DP16, MoE EP16), in the disaggregated cases that run it.

The deployment and the harness are in ``../_sm107_gsm8k.py``.
"""

import pytest

from .._sm107_gsm8k import CASES, Sm107Gsm8kHarness, requires_sm107, run_disaggregated_gsm8k

_CASES = dict([CASES["dep16"]])


@requires_sm107
@pytest.mark.timeout(7200)
class TestModelingV2KimiK3Nvfp4Sm107Dep16(Sm107Gsm8kHarness):
    """GSM8K against the disaggregated deployments that run this target."""

    @pytest.mark.parametrize("test_case", list(_CASES))
    def test_gsm8k(self, output_dir: str, test_case: str) -> None:
        run_disaggregated_gsm8k(output_dir, test_case, max_draft_len=_CASES[test_case])
