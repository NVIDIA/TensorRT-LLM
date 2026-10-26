# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_QKV_FUSED (#20629 QKV_FUSED, A1) gating on the sm_107 tp2ctx target, from source and module state.

The fused QKV GEMM + producer takes steps above 1024 rows only; its kill switch is TRTLLM_M31_DISABLE_TP2CTX_QKV_FUSED;
the compiled prefill reaches it inside the fused-producer op (whose body runs eagerly), never in a traced region; and
the runner comes from this target's own copy of #20629's kernel. The kernel's bitwise gate (#20629's
test_m3_qkv_fusion.py, against the stock producer on the kernel's own GEMM output) runs on sm_107. CPU only.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

pytestmark = pytest.mark.cpu_only

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"


def test_the_switch_and_the_row_floor():
    src = (_TARGET / "modeling.py").read_text()
    # sm_107: default ON after gates b + c (rule 1b, disclosed); kill TRTLLM_M31_DISABLE_TP2CTX_QKV_FUSED.
    assert '_CTX_QKV_FUSED = not m31_disabled("TP2CTX_QKV_FUSED")' in src
    assert "_CTX_QKV_FUSED_MIN_ROWS = 1024" in src
    assert "return rows > _CTX_QKV_FUSED_MIN_ROWS" in src


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"TRTLLM_M31_ENABLE_TP2CTX_QKV_FUSED": "1"}, True),
        (
            {"TRTLLM_M31_ENABLE_TP2CTX_QKV_FUSED": "1", "TRTLLM_M31_DISABLE_TP2CTX_QKV_FUSED": "1"},
            False,
        ),
    ],
    ids=["default-off", "opt-in", "disable-wins"],
)
def test_the_switch_is_off_by_default_on_sm_107(env, expected):
    """The switch line's value per environment, from the target's m31_switches in a fresh process (read once at
    import)."""
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx"
        ".m31_switches import m31_disabled, m31_enabled; "
        "print('SWITCH=' + str(m31_enabled('TP2CTX_QKV_FUSED') and not m31_disabled('TP2CTX_QKV_FUSED')))"
    )
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TRTLLM_M31_ENABLE_", "TRTLLM_M31_DISABLE_"))
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**clean, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    # Importing tensorrt_llm prints its version banner to stdout first, so read the sentinel line.
    switch = [line for line in out.stdout.splitlines() if line.startswith("SWITCH=")]
    assert switch == [f"SWITCH={expected}"]


def test_the_fused_path_never_runs_traced():
    src = (_TARGET / "modeling.py").read_text()
    start = src.index("    def _ctx_fused_qkv_gemm_producer(")
    gate = src[start : src.index("        runner = _ctx_qkv_fused_runner(", start)]
    for condition in (
        "not _CTX_QKV_FUSED",
        "is_torch_compiling()",
        "torch.compiler.is_compiling()",
        "get_sm_version() != 107",
        "not _ctx_qkv_fused_takes_rows(num_tokens)",
        "_autotuner_tuning()",
    ):
        assert condition in gate, f"the fused QKV gate lost {condition!r}"


def test_the_runner_is_this_targets_copy_of_the_kernel():
    src = (_TARGET / "modeling.py").read_text()
    assert "minimax_m31_nvfp4__sm_107__tp2ctx.kernels.qkv_fusion.kernel import" in src
    kernel = (_TARGET / "kernels" / "qkv_fusion" / "kernel.py").read_text()
    assert "class M3QkvFusedRunner" in kernel


def test_the_kernel_loads_its_shipped_aot_object_before_compiling():
    kernel = (_TARGET / "kernels" / "qkv_fusion" / "kernel.py").read_text()
    body = kernel[kernel.index("def compile(") : kernel.index("def _cache_strides")]
    assert "cute_aot.compile_or_load(" in body and "tvm_ffi=True" in body
    assert "compiled = cute.compile(" not in body
