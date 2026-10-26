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
"""FlashInfer's sampling module built by the CPU-only helper is what warmup loads.

``start_flashinfer_build`` builds the module in a helper with no visible GPU,
into a fresh FlashInfer workspace; ``wait_flashinfer_build`` joins it. The GPU
process then loads the module from that workspace without rebuilding it, and
it samples. The same holds for the TRTLLM-GEN attention launcher (``fmha_gen``)
an attention layer on FlashInfer's TRTLLM-GEN FMHA library builds, started
before the sampling module as the model loader starts it: a paged decode
call then loads it without a rebuild. Without prefetch, or with the module
already built, nothing is started. A build still running at the join's timeout is stopped together with
its compilers, and the module is then built in place; so is one whose
starting process exits. Each case runs in a subprocess: FlashInfer fixes its
workspace directory at import time.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

pytestmark = pytest.mark.skipif(
    not IS_FLASHINFER_AVAILABLE or not torch.cuda.is_available(),
    reason="needs FlashInfer and a GPU",
)

_CASE = textwrap.dedent(
    """
    import json, os, time
    import torch
    from tensorrt_llm._torch import jit_prefetch as jp
    from flashinfer.jit.sampling import gen_sampling_module

    started = jp.start_flashinfer_build()
    waited = jp.wait_flashinfer_build()
    so = gen_sampling_module().jit_library_path
    mtime = os.stat(so).st_mtime_ns if so.exists() else None
    # get_sampling_module() runs ninja on a JIT module; on the helper's build
    # it must be a no-op (the library is not rebuilt) and only load it.
    import flashinfer.sampling as s
    t = time.time()
    s.get_sampling_module()
    load_s = time.time() - t
    probs = torch.softmax(torch.randn(4, 1000, device="cuda"), -1)
    tok = s.sampling_from_probs(probs)
    print(json.dumps({"started": started, "waited": waited, "built_before_load": mtime is not None,
                      "rebuilt": mtime is not None and os.stat(so).st_mtime_ns != mtime,
                      "load_s": load_s, "sampled": list(tok.shape)}))
    """
)


# The sampling module started by the model loader, while the weights load:
# warmup finds that helper pending (it must not start another nor build the
# module in place), joins it, and loads the helper's build.
_LOAD_TIME_CASE = textwrap.dedent(
    """
    import json, os, time
    import torch
    from tensorrt_llm._torch import jit_prefetch as jp
    from flashinfer.jit.sampling import gen_sampling_module

    started = jp.start_flashinfer_build(("sampling",))
    pending = [jp.flashinfer_build_pending("sampling"), jp.flashinfer_build_pending("fmha_gen")]
    again = jp.start_flashinfer_build()
    waited = jp.wait_flashinfer_build()
    so = gen_sampling_module().jit_library_path
    mtime = os.stat(so).st_mtime_ns if so.exists() else None
    import flashinfer.sampling as s
    t = time.time()
    s.get_sampling_module()
    load_s = time.time() - t
    tok = s.sampling_from_probs(torch.softmax(torch.randn(4, 1000, device="cuda"), -1))
    print(json.dumps({"started": started, "pending": pending, "again": again,
                      "waited": waited, "built_before_load": mtime is not None,
                      "rebuilt": mtime is not None and os.stat(so).st_mtime_ns != mtime,
                      "load_s": load_s, "after": jp.flashinfer_build_pending("sampling"),
                      "sampled": list(tok.shape)}))
    """
)


_SESSION = textwrap.dedent(
    """
    import os, time

    _COMPILERS = {"nvcc", "cicc", "ptxas", "cudafe++"}

    def session(sid):
        # Commands of the live (non-zombie) processes in session sid. The
        # helper leads its session; ninja runs each compiler in a process
        # group of its own, but in the same session.
        procs = []
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    comm, rest = f.read().rsplit(")", 1)
            except OSError:
                continue
            state, _, _, sess = rest.split()[:4]
            if int(sess) == sid and state != "Z":
                procs.append(comm.split("(", 1)[1])
        return procs

    def wait_compiling(sid, timeout_s=180):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            compilers = sorted(set(session(sid)) & _COMPILERS)
            if compilers:
                return compilers
            time.sleep(0.2)
        return []

    def wait_gone(sid, timeout_s=30):
        deadline = time.time() + timeout_s
        left = session(sid)
        while left and time.time() < deadline:
            time.sleep(0.5)
            left = session(sid)
        return left
    """
)

_TIMEOUT_CASE = _SESSION + textwrap.dedent(
    """
    import json
    import torch
    from tensorrt_llm._torch import jit_prefetch as jp
    from flashinfer.jit.sampling import gen_sampling_module

    started = jp.start_flashinfer_build()
    sid = jp._fi_builds[0][1].pid
    # Join once the helper's nvcc is running, with a timeout far shorter
    # than the compile.
    compilers = wait_compiling(sid)
    waited = jp.wait_flashinfer_build(timeout_s=1.0)
    left = wait_gone(sid)
    built_by_helper = gen_sampling_module().is_compiled
    # Nothing is left to join or stop, and the module builds in place.
    again = jp.wait_flashinfer_build()
    jp.cancel_flashinfer_build()
    import flashinfer.sampling as s
    s.get_sampling_module()
    probs = torch.softmax(torch.randn(4, 1000, device="cuda"), -1)
    tok = s.sampling_from_probs(probs)
    print(json.dumps({"started": started, "compilers": compilers, "waited": waited, "left": left,
                      "built_by_helper": built_by_helper, "again": again,
                      "sampled": list(tok.shape)}))
    """
)

# A process that starts the build and exits mid-compile, without joining or
# cancelling it, after printing the helper's session id. It exits normally:
# under an MPI launcher, a process that skips MPI finalization (os._exit)
# aborts the whole job step.
_ORPHAN_CASE = _SESSION + textwrap.dedent(
    """
    import json
    from tensorrt_llm._torch import jit_prefetch as jp

    assert jp.start_flashinfer_build()
    sid = jp._fi_builds[0][1].pid
    assert wait_compiling(sid)
    print(json.dumps({"sid": sid}), flush=True)
    """
)

_ORPHAN_WATCH = _SESSION + textwrap.dedent(
    """
    import json, sys
    from flashinfer.jit.sampling import gen_sampling_module

    left = wait_gone(int(sys.argv[1]))
    print(json.dumps({"left": left, "built": gen_sampling_module().is_compiled}))
    """
)


# FlashInfer's TRTLLM-GEN attention launcher (fmha_gen), started before the
# sampling module as the model loader starts it, and joined with it: the GPU
# process then loads it without rebuilding it, and FlashInfer's TRTLLM-GEN
# paged decode runs on it.
_FMHA_GEN_CASE = textwrap.dedent(
    """
    import json, os, time
    import torch
    from tensorrt_llm._torch import jit_prefetch as jp
    from flashinfer.jit import env as jit_env

    so = jit_env.FLASHINFER_JIT_DIR / "fmha_gen" / "fmha_gen.so"
    started_model = jp.start_flashinfer_build(("fmha_gen",))
    started_sampling = jp.start_flashinfer_build()
    pending = jp.flashinfer_builds_pending()
    waited = jp.wait_flashinfer_build()
    mtime = os.stat(so).st_mtime_ns if so.exists() else None
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    # One decode token of a GQA layer (32 query and 8 KV heads of 128) over
    # a 64-token paged KV cache: two pages of 32 tokens, HND layout.
    dev = torch.device("cuda")
    q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(2, 2, 8, 32, 128, dtype=torch.bfloat16, device=dev)
    block_tables = torch.arange(2, dtype=torch.int32, device=dev).view(1, 2)
    seq_lens = torch.full((1,), 64, dtype=torch.int32, device=dev)
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=dev)
    t = time.time()
    out = trtllm_batch_decode_with_kv_cache(
        q, kv, workspace, block_tables, seq_lens, max_seq_len=64,
        bmm1_scale=128 ** -0.5, bmm2_scale=1.0, backend="trtllm-gen")
    torch.cuda.synchronize()
    first_call_s = time.time() - t
    print(json.dumps({"started": [started_model, started_sampling], "pending": pending,
                      "waited": waited, "built_before_load": mtime is not None,
                      "rebuilt": mtime is not None and os.stat(so).st_mtime_ns != mtime,
                      "first_call_s": first_call_s, "finite": bool(torch.isfinite(out).all()),
                      "again": [jp.start_flashinfer_build(("fmha_gen",)),
                                jp.flashinfer_builds_pending()]}))
    """
)


def _env(tmp_path, prefetch: bool):
    env = dict(os.environ, FLASHINFER_WORKSPACE_BASE=str(tmp_path / "fi"), PYTHONNOUSERSITE="1")
    env.pop("TLLM_JIT_AHEAD", None)
    if prefetch:
        env["TLLM_JIT_AHEAD"] = "1"
    return env


def _run(tmp_path, prefetch: bool, case: str = _CASE, args=()):
    out = subprocess.run(
        [sys.executable, "-c", case, *args],
        env=_env(tmp_path, prefetch),
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    return out.stdout.strip().splitlines()[-1]


def _run_json(tmp_path, prefetch: bool, case: str = _CASE, args=()):
    return json.loads(_run(tmp_path, prefetch, case, args))


def test_helper_build_is_what_warmup_loads(tmp_path):
    r = _run_json(tmp_path, prefetch=True)
    assert r["started"] is True
    assert r["waited"] is not None
    # The helper built the library, and loading it in this process did not
    # rebuild it (a compile takes about a minute; a load well under 10 s).
    assert r["built_before_load"] and not r["rebuilt"], r
    assert r["load_s"] < 10, r
    assert r["sampled"] == [4]


def test_load_time_sampling_build_is_joined_not_restarted(tmp_path):
    r = _run_json(tmp_path, prefetch=True, case=_LOAD_TIME_CASE)
    assert r["started"] is True and r["again"] is False, r
    # The loader's helper builds the sampling module and nothing else.
    assert r["pending"] == [True, False], r
    assert r["waited"] is not None and r["after"] is False, r
    assert r["built_before_load"] and not r["rebuilt"], r
    assert r["load_s"] < 10, r
    assert r["sampled"] == [4]


def test_layer_module_built_by_helper_is_what_the_layer_loads(tmp_path):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FlashInfer's TRTLLM-GEN FMHA needs SM100 / SM103")
    r = _run_json(tmp_path, prefetch=True, case=_FMHA_GEN_CASE)
    # Two helpers (the model's module, then the sampling module), both joined.
    assert r["started"] == [True, True] and r["pending"] is True, r
    assert r["waited"] is not None
    # The helper built the launcher; the decode call loaded it without a
    # rebuild (a build takes about a minute) and ran on it.
    assert r["built_before_load"] and not r["rebuilt"], r
    assert r["first_call_s"] < 30, r
    assert r["finite"], r
    # Built and joined: nothing is started again, nothing left to join.
    assert r["again"] == [False, False], r


def test_nothing_started_without_prefetch(tmp_path):
    case = (
        "from tensorrt_llm._torch import jit_prefetch as jp;"
        "print(jp.start_flashinfer_build(), jp.wait_flashinfer_build())"
    )
    assert _run(tmp_path, prefetch=False, case=case) == "False None"


def test_nothing_started_when_already_built(tmp_path):
    # The first case builds the module into this workspace; a second process
    # finds it on disk and starts no helper.
    first = _run_json(tmp_path, prefetch=True)
    assert first["started"] is True
    again = _run_json(tmp_path, prefetch=True)
    assert again["started"] is False
    assert again["built_before_load"] and not again["rebuilt"], again


def test_build_stops_when_its_process_exits(tmp_path):
    # The process that started the build exits mid-compile (a server that
    # dies in warmup): the helper and its compilers stop on their own.
    sid = _run_json(tmp_path, prefetch=True, case=_ORPHAN_CASE)["sid"]
    r = _run_json(tmp_path, prefetch=False, case=_ORPHAN_WATCH, args=(str(sid),))
    assert r == {"left": [], "built": False}, r


def test_timed_out_build_is_stopped_with_its_compilers(tmp_path):
    r = _run_json(tmp_path, prefetch=True, case=_TIMEOUT_CASE)
    assert r["started"] is True
    # Joined mid-compile: the join gave up after its timeout and stopped the
    # helper and every compiler it ran, before the module was linked.
    assert r["compilers"], r
    assert 1.0 <= r["waited"] < 30, r
    assert r["left"] == [], r
    assert r["built_by_helper"] is False, r
    assert r["again"] is None
    # The module was then built in place, in the same build directory.
    assert r["sampled"] == [4]
