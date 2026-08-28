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

import pytest

from tensorrt_llm._torch.utils import is_torch_compiling, torch_compiling

pytestmark = pytest.mark.cpu_only


def test_torch_compiling_restores_flag_after_exception() -> None:
    with torch_compiling(False):
        with pytest.raises(RuntimeError), torch_compiling(True):
            raise RuntimeError
        assert not is_torch_compiling()
