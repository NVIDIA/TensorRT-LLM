#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-export of tests/test_common/cluster_env.py for the PerfSanity scripts.

The rule table lives with the test helpers so pytest sessions and PerfSanity
submission share one copy. The submit scripts import this module by name
from their own directory.
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[3] / "tests" / "test_common" / "cluster_env.py"
_spec = importlib.util.spec_from_file_location("test_common_cluster_env", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

BASE_UCX_UNSET = _module.BASE_UCX_UNSET
UCX_ENV_RULES = _module.UCX_ENV_RULES
get_ucx_tls_cmd = _module.get_ucx_tls_cmd
gpu_type_from_stage_name = _module.gpu_type_from_stage_name
gpu_type_from_supported_gpus = _module.gpu_type_from_supported_gpus
