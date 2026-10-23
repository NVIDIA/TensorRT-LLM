# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU binding regressions for both profiling build configurations.

Private adapter and four-vectorcall coverage also require a pinned native fixture;
these tests exercise the normal product module, not a replacement extension.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_CHECK = """
import os
import torch
import tensorrt_llm.bindings as bindings

flag = bindings.BuildInfo.ENABLE_NANOBIND_PROFILING
assert type(flag) is bool
class TensorSubclass(torch.Tensor):
    pass

tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
if os.environ['PR6_SUBCLASS'] == '1':
    tensor = tensor.as_subclass(TensorSubclass)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
) as prof:
    assert bindings.QuantMode.none().value == 0
    if os.environ['PR6_KEYWORD'] == '1':
        config = bindings.executor.PromptTuningConfig(embedding_table=tensor)
    else:
        config = bindings.executor.PromptTuningConfig(tensor)
    result = config.embedding_table
    torch.testing.assert_close(result, tensor, check_type=False)
    assert result.data_ptr() == tensor.data_ptr()
    try:
        bindings.QuantMode.none(tensor)
    except TypeError as error:
        assert 'incompatible function arguments' in str(error)
    else:
        raise AssertionError('wrong arity accepted')
    assert bindings.QuantMode.none().value == 0

selected = os.environ.get('TLLM_PROFILE_CUSTOM_REGIONS')
enabled = flag and selected in (None, 'nanobind')
events = [event.name for event in prof.events()
          if event.name.startswith('trtllm::nanobind::')]
assert bool(events) == enabled, (flag, selected, events)
if enabled:
    assert events.count('trtllm::nanobind::tensorrt_llm.bindings.QuantMode.none') == 3
    assert events.count(
        'trtllm::nanobind::tensorrt_llm.bindings.executor.PromptTuningConfig.__init__'
    ) == 1
"""


@pytest.mark.parametrize("selector", [None, "nanobind", "none", "mpi"])
@pytest.mark.parametrize("keyword,subclass", [(False, False), (True, False), (True, True)])
def test_nanobind_results_errors_and_selection(
    selector: str | None, keyword: bool, subclass: bool
) -> None:
    """Preserve CPU tensor storage and TypeError while respecting startup selection."""
    environment = os.environ.copy()
    if selector is None:
        environment.pop("TLLM_PROFILE_CUSTOM_REGIONS", None)
    else:
        environment["TLLM_PROFILE_CUSTOM_REGIONS"] = selector
    environment["PR6_KEYWORD"] = str(int(keyword))
    environment["PR6_SUBCLASS"] = str(int(subclass))
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
