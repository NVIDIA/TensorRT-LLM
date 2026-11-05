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

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_CHECK = """
import json
import os
import tempfile
from pathlib import Path

import torch
from profiler_test_utils import decode_profiler_metadata

import tensorrt_llm.bindings as bindings

flag = bindings.BuildInfo.ENABLE_NANOBIND_PROFILING
assert type(flag) is bool
class TensorSubclass(torch.Tensor):
    pass

tensor = torch.arange(6, dtype=torch.int32).reshape(2, 3)
if os.environ['PR6_SUBCLASS'] == '1':
    tensor = tensor.as_subclass(TensorSubclass)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'execution.json'
    observer = torch.profiler.ExecutionTraceObserver().register_callback(str(path))
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True,
            execution_trace_observer=observer,
        ) as prof:
            assert bindings.QuantMode.none().value == 0
            if os.environ['PR6_KEYWORD'] == '1':
                config = bindings.executor.LoraConfig(task_id=1, config=tensor)
            else:
                config = bindings.executor.LoraConfig(1, None, tensor)
            sampling = bindings.executor.SamplingConfig()
            sampling.top_k = 3
            assert sampling.top_k == 3
            result = config.config
            torch.testing.assert_close(result, tensor, allow_subclasses=True)
            assert result.data_ptr() == tensor.data_ptr()
            try:
                bindings.QuantMode.none(tensor)
            except TypeError as error:
                assert 'incompatible function arguments' in str(error)
            else:
                raise AssertionError('wrong arity accepted')
            assert bindings.QuantMode.none().value == 0
    finally:
        observer.unregister_callback()
    nodes = [node for node in json.loads(path.read_text())['nodes']
             if node['name'].startswith('trtllm::nanobind::')]

selected = os.environ.get('TLLM_PROFILE_CUSTOM_REGIONS')
enabled = flag and selected in (None, 'nanobind')
events = [event.name for event in prof.events()
          if event.name.startswith('trtllm::nanobind::')]
assert bool(events) == enabled, (flag, selected, events)
assert bool(nodes) == enabled, (flag, selected, nodes)
if enabled:
    assert events.count('trtllm::nanobind::tensorrt_llm.bindings.QuantMode.none') == 3
    assert events.count(
        'trtllm::nanobind::tensorrt_llm.bindings.executor.LoraConfig.__init__'
    ) == 1
    assert events.count(
        'trtllm::nanobind::tensorrt_llm.bindings.executor.SamplingConfig.top_k'
    ) == 2
    assert '<anonymous>' not in ' '.join(events)
    constructor = [node for node in nodes if node['name'] ==
                   'trtllm::nanobind::tensorrt_llm.bindings.executor.LoraConfig.__init__']
    assert len(constructor) == 1
    region, metadata = decode_profiler_metadata(constructor[0]['inputs']['values'][0])
    assert region == 'nanobind'
    assert metadata == {
        'nanobind.tensor_count': 1,
        'nanobind.tensor.0.shape': [2, 3],
        'nanobind.tensor.0.stride': [3, 1],
        'nanobind.tensor.0.dtype': 'Int',
        'nanobind.tensor.0.device': 'cpu',
    }
"""


@pytest.mark.parametrize("selector", [None, "nanobind", "none", "mpi"])
@pytest.mark.parametrize(
    "keyword,subclass", [(False, False), (False, True), (True, False), (True, True)]
)
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
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_nanobind_property_stubs_are_valid_python() -> None:
    """Generate callable property accessors instead of aliases beneath decorators."""
    import ast

    from nanobind.stubgen import StubGen

    from tensorrt_llm.bindings import executor

    generator = StubGen(executor, include_docstrings=False)
    generator.put_type(executor.SamplingConfig, "SamplingConfig")
    module = ast.parse(generator.output)
    config = next(node for node in module.body if isinstance(node, ast.ClassDef))
    accessors = [
        node for node in config.body if isinstance(node, ast.FunctionDef) and node.name == "top_k"
    ]
    assert [ast.unparse(node.decorator_list[0]) for node in accessors] == [
        "property",
        "top_k.setter",
    ]


@pytest.mark.parametrize(
    "selector,option",
    [
        ("nanobind", "ENABLE_NANOBIND_PROFILING"),
        ("nccl", "ENABLE_NCCL_PROFILING"),
    ],
)
def test_explicit_omitted_region_warns_once(selector: str, option: str) -> None:
    """An explicit unavailable native probe explains an otherwise empty trace."""
    script = f"""
import warnings

warnings.filterwarnings('error', message=r'.*{option}=OFF')
import tensorrt_llm.bindings as bindings
import tensorrt_llm.bindings
print('capability=' + str(int(getattr(bindings.BuildInfo, '{option}'))))
"""
    environment = os.environ.copy()
    environment["TLLM_PROFILE_CUSTOM_REGIONS"] = selector
    environment["PYTHONWARNINGS"] = "default"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = int("capability=0" in result.stdout)
    assert (result.stdout + result.stderr).count(f"{option}=OFF") == expected


@pytest.mark.parametrize(
    "selector,option",
    [
        ("none", None),
        ("unknown", None),
        ("nanobind", "ENABLE_NANOBIND_PROFILING"),
        ("nccl", "ENABLE_NCCL_PROFILING"),
    ],
)
def test_native_module_initialization_warnings_do_not_require_cuda(
    selector: str, option: str | None, tmp_path: Path
) -> None:
    """Initialize the actual extension without the separate package/MPI bootstrap."""
    import tensorrt_llm.bindings as bindings

    parent_extension = Path(bindings.__file__).resolve()
    expected_hash = hashlib.sha256(parent_extension.read_bytes()).hexdigest()
    script = """
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import torch
assert torch.cuda.device_count() == 0
print('visible_cuda_devices=0', flush=True)
package = importlib.machinery.PathFinder.find_spec('tensorrt_llm')
spec = importlib.machinery.PathFinder.find_spec(
    'tensorrt_llm.bindings', package.submodule_search_locations
)
assert isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
bindings = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bindings
spec.loader.exec_module(bindings)
assert importlib.import_module(spec.name) is bindings
assert bindings.QuantMode.none().value == 0
print(json.dumps({
    'file': bindings.__file__,
    'sha256': hashlib.sha256(Path(bindings.__file__).read_bytes()).hexdigest(),
    'flags': {name: getattr(bindings.BuildInfo, name) for name in (
        'ENABLE_NANOBIND_PROFILING', 'ENABLE_NCCL_PROFILING'
    )},
    'visible_cuda_devices': torch.cuda.device_count(),
}))
"""
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TLLM_PROFILE_CUSTOM_REGIONS=selector,
        TLLM_LOG_LEVEL="INFO",
        TLLM_LOG_FIRST_RANK_ONLY="ON",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (tmp_path / "native-extension-init.stdout.txt").write_text(result.stdout)
    (tmp_path / "native-extension-init.stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads(result.stdout.strip().splitlines()[-1])
    (tmp_path / "native-extension-init.json").write_text(
        json.dumps({"selector": selector, "scope": "native-extension initialization", **info})
    )
    assert info["visible_cuda_devices"] == 0
    assert Path(info["file"]).resolve() == parent_extension
    assert info["sha256"] == expected_hash
    assert info["flags"] == {
        name: getattr(bindings.BuildInfo, name)
        for name in ("ENABLE_NANOBIND_PROFILING", "ENABLE_NCCL_PROFILING")
    }
    assert all(type(flag) is bool for flag in info["flags"].values())
    if selector == "unknown":
        assert result.stderr.count("Invalid TLLM_PROFILE_CUSTOM_REGIONS region:") == 1
    elif option is not None:
        assert result.stderr.count(f"{option}=OFF") == int(not info["flags"][option])
    else:
        assert "TLLM_PROFILE_CUSTOM_REGIONS" not in result.stderr
