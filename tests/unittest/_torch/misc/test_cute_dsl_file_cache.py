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
"""Processes share their ``cute.compile`` results through TRTLLM_CUTE_DSL_CACHE_DIR."""

import json
import os
import re
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

# Compiles one TVM-FFI kernel with symbolic sizes and the scale in argv, runs it and prints a digest
# of its output.
_CHILD = r"""
import hashlib
import json
import sys

import cutlass
import cutlass.cute as cute
import torch

import tensorrt_llm._torch.cute_dsl_utils  # noqa: F401

SCALE = float(sys.argv[1])


@cute.kernel
def axpy_kernel(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < cute.size(x):
        out[i] = x[i] * SCALE + y[i]


@cute.jit
def axpy(x: cute.Tensor, y: cute.Tensor, out: cute.Tensor, stream):
    n = cute.size(x)
    axpy_kernel(x, y, out).launch(grid=((n + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream)


n = cute.sym_int()
fake = [
    cute.runtime.make_fake_compact_tensor(cutlass.Float32, (n,), stride_order=(0,))
    for _ in range(3)
]
stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
compiled = cute.compile(axpy, *fake, stream, options="--enable-tvm-ffi")
x = torch.arange(-2000, 3011, dtype=torch.float32, device="cuda")
y = torch.arange(5011, dtype=torch.float32, device="cuda").flip(0)
out = torch.empty_like(x)
compiled(x, y, out)
torch.cuda.synchronize()
assert torch.equal(out, x * SCALE + y)
print("RESULT " + json.dumps({"digest": hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()}))
"""

_OUTCOME = re.compile(r"\[cute_dsl_file_cache\] (hit|miss) \S*axpy")
_RESULT = re.compile(r"^RESULT (.*)$", re.MULTILINE)


def _start(script, cache_dir, scale=3.0):
    """Start a child running the kernel with ``scale``; ``cache_dir=None`` turns the cache off."""
    env = dict(os.environ, TLLM_LOG_LEVEL="DEBUG")
    env.pop("TRTLLM_CUTE_DSL_CACHE_DIR", None)
    if cache_dir is not None:
        env["TRTLLM_CUTE_DSL_CACHE_DIR"] = str(cache_dir)
    return subprocess.Popen(
        [sys.executable, str(script), str(scale)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _finish(*procs):
    """Each child's cache outcomes for the kernel, its output digest, and its output."""
    outs = {}
    try:
        for proc in procs:
            try:
                outs[proc] = proc.communicate(timeout=900)[0]
            except subprocess.TimeoutExpired:
                proc.kill()
                raise AssertionError(f"child timed out:\n{proc.communicate()[0][-6000:]}") from None
    finally:
        # A child left running would keep its GPU memory after the test.
        for proc in procs:
            if proc not in outs and proc.poll() is None:
                proc.kill()
                proc.communicate()
    results = []
    for proc in procs:
        out = outs[proc]
        assert proc.returncode == 0, out[-6000:]
        result = _RESULT.search(out)
        assert result, out[-6000:]
        results.append((_OUTCOME.findall(out), json.loads(result.group(1))["digest"], out))
    return results


@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="needs the CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cute_compile_reused_across_processes(tmp_path):
    script = tmp_path / "axpy.py"
    script.write_text(_CHILD)
    cache = tmp_path / "cache"

    # Two processes race on the empty cache while a third runs without it.
    *cold, stock = _finish(_start(script, cache), _start(script, cache), _start(script, None))
    for outcomes, _, out in cold:
        assert outcomes in (["miss"], ["hit"]), out[-6000:]
    assert any(outcomes == ["miss"] for outcomes, _, _ in cold), cold[0][2][-6000:]
    assert stock[0] == [], stock[2][-6000:]
    entries = list(cache.iterdir())
    assert len(entries) == 1, entries

    # A later process loads the stored compile; a changed kernel compiles and stores its own.
    warm, changed = _finish(_start(script, cache), _start(script, cache, scale=4.0))
    assert warm[0] == ["hit"], warm[2][-6000:]
    assert changed[0] == ["miss"], changed[2][-6000:]
    assert changed[1] != stock[1]

    # A broken entry is compiled again and replaced, and the next process loads the new one.
    data = entries[0].read_bytes()
    entries[0].write_bytes(data[: len(data) // 2])
    (repaired,) = _finish(_start(script, cache))
    assert repaired[0] == ["miss"], repaired[2][-6000:]
    assert f"entry {entries[0].name} unusable" in repaired[2], repaired[2][-6000:]
    (reloaded,) = _finish(_start(script, cache))
    assert reloaded[0] == ["hit"], reloaded[2][-6000:]

    assert {digest for _, digest, _ in [*cold, warm, repaired, reloaded]} == {stock[1]}
    # Only whole entries are left behind: the kernel's and the changed kernel's.
    names = [p.name for p in cache.iterdir()]
    assert len(names) == 2, names
    assert all(re.fullmatch(r"cute_dsl_[0-9a-f]{64}\.mlir", name) for name in names), names
