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

import argparse
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "examples" / "llm-api" / "quickstart_advanced.py"
_SPEC = importlib.util.spec_from_file_location("quickstart_advanced", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    ("cli_value", "expected"),
    [
        ("auto", "auto"),
        ("true", True),
        ("false", False),
    ],
)
def test_use_kv_cache_manager_v2_cli_values(cli_value: str, expected: str | bool) -> None:
    parser = _MODULE.add_llm_args(argparse.ArgumentParser())

    args = parser.parse_args(["--model_dir", "dummy-model", "--use_kv_cache_manager_v2", cli_value])

    assert args.use_kv_cache_manager_v2 == expected


def test_use_kv_cache_manager_v2_cli_default_is_auto() -> None:
    parser = _MODULE.add_llm_args(argparse.ArgumentParser())

    args = parser.parse_args(["--model_dir", "dummy-model"])

    assert args.use_kv_cache_manager_v2 == "auto"
