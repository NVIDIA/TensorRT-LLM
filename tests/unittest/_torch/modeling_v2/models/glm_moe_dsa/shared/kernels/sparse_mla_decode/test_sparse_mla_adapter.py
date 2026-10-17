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
"""Checks of the sparse-MLA decode library declaration on the DSA backend.

These run on a GPU only on request (opt-in:
``GLM_MOE_DSA_SPARSE_MLA_DECODE_ADAPTER_APPLIED=1``). Each check starts a fresh interpreter that
installs the plan of the fast paths it is given (``plan_testlib.FAST_PATHS_ENV``). With
sparse_mla_decode running, the layer declares the library ahead of the registry's, as ``attention.GlmMLA.adopt``
does, and rebuilds its FMHA libraries, as ``create_weights`` does; the registry is unchanged.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plan_testlib  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("GLM_MOE_DSA_SPARSE_MLA_DECODE_ADAPTER_APPLIED") != "1",
    reason="opt-in GPU check",
)

_HERE = Path(__file__).resolve().parent

_FORWARD_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import conftest  # builds the dev op when needed
import plan_testlib
import torch
from sparse_mla_harness import DsaGenHarness
from tensorrt_llm._torch.attention.backends.fmha import registry
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (
    dispatch,
)
plan_testlib.select_from_env(dispatch)
h = DsaGenHarness([20000], seq_len_q=6, seed=5)
if dispatch.is_enabled("sparse_mla_decode"):
    for layer in h.layers:
        layer.prepended_fmha_lib_classes = (dispatch.get_impl("sparse_mla_decode", "fmha_lib"),)
        layer.update_quant_config(layer.quant_config)
try:
    topk = h.random_topk(seed=6, overlap=0.7, recent=32)
    step = h.make_step(0, seed=7)
    out = h.forward(0, step, topk)
    libs = [type(lib).__name__ for lib in h.layers[0]._fmha_manager.fmha_libs]
    print(json.dumps({"registry": list(registry.FMHA_LIBS), "layer_libs": libs,
                      "selected": h.selected_library(0),
                      "out_finite": bool(torch.isfinite(out).all())}))
finally:
    h.shutdown()
"""


def _run(features: str) -> dict:
    env = dict(os.environ)
    env[plan_testlib.FAST_PATHS_ENV] = features
    proc = subprocess.run(
        [sys.executable, "-c", _FORWARD_SCRIPT, str(_HERE)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_default_off_keeps_native_libraries():
    result = _run("")
    assert "glm_moe_dsa_sparse_mla_decode" not in result["registry"]
    assert "SparseMlaDecodeFmha" not in result["layer_libs"]
    assert result["selected"] == ["FallbackFmha"]


def test_sparse_mla_selected_when_enabled():
    result = _run("sparse_mla_decode")
    assert "glm_moe_dsa_sparse_mla_decode" not in result["registry"]
    assert result["layer_libs"][0] == "SparseMlaDecodeFmha"
    assert result["layer_libs"].count("SparseMlaDecodeFmha") == 1
    assert result["selected"] == ["SparseMlaDecodeFmha"]
    assert result["out_finite"]
