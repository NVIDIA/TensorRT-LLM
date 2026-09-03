# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyExecutor construction ownership tests for DWDP."""

from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager, set_global_dwdp_manager


def test_create_py_executor_rolls_back_new_dwdp_manager_on_error():
    set_global_dwdp_manager(None)
    manager = MagicMock()
    manager.__exit__.side_effect = lambda *_args: set_global_dwdp_manager(None)

    def fail_after_registration(**_kwargs):
        set_global_dwdp_manager(manager)
        raise RuntimeError("construction failed")

    with patch.object(
        py_executor_creator,
        "_create_py_executor_impl",
        side_effect=fail_after_registration,
    ):
        with pytest.raises(RuntimeError, match="construction failed"):
            py_executor_creator.create_py_executor(llm_args=MagicMock())

    manager.__exit__.assert_called_once_with(None, None, None)
    assert get_global_dwdp_manager() is None
