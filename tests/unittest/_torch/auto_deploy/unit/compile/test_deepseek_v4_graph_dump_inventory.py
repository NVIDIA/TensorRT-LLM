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
"""DeepSeek V4 five-layer graph dump inventory checks."""

import os
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[6]
_GRAPH_DUMP_ENV = "DEEPSEEK_V4_GRAPH_DUMP"
_DEFAULT_GRAPH_DUMP = (
    _REPO_ROOT
    / "ad_run_logs"
    / "graph_dumps"
    / "deepseek_v4_flash_5layer_sampling_workspace_20260425_131939"
    / "084_compile_compile_model.txt"
)
_DEFAULT_GRAPH_FIXTURE_IS_STALE = _GRAPH_DUMP_ENV not in os.environ
_STALE_DEFAULT_FIXTURE_XFAIL = pytest.mark.xfail(
    _DEFAULT_GRAPH_FIXTURE_IS_STALE,
    reason=(
        "default five-layer compile graph fixture still records torch_* wrapper "
        "selection; remove this xfail when the fixture is regenerated with Triton "
        "attention/router ops"
    ),
    strict=True,
)
_ATTENTION_OP_MARKERS = (
    "triton_deepseek_v4_sparse_attention_v2_with_cache.default",
    "torch_deepseek_v4_sparse_attention_v2_with_cache.default",
)
_ROUTER_OP_MARKERS = (
    "triton_deepseek_v4_router.default",
    "torch_deepseek_v4_router.default",
)
_DEVICE_OP_GATES = (
    pytest.param(
        "cached sparse attention",
        "auto_deploy.torch_deepseek_v4_sparse_attention_v2_with_cache.default(",
        "auto_deploy.triton_deepseek_v4_sparse_attention_v2_with_cache.default(",
        5,
        "Wave 4",
        marks=_STALE_DEFAULT_FIXTURE_XFAIL,
    ),
    pytest.param(
        "router",
        "auto_deploy.torch_deepseek_v4_router.default(",
        "auto_deploy.triton_deepseek_v4_router.default(",
        5,
        "Wave 2/4",
        marks=_STALE_DEFAULT_FIXTURE_XFAIL,
    ),
)


def _graph_dump_path() -> Path:
    return Path(os.environ.get(_GRAPH_DUMP_ENV, _DEFAULT_GRAPH_DUMP))


def _read_graph_dump() -> str:
    graph_dump = _graph_dump_path()
    if not graph_dump.is_file():
        pytest.skip(f"DeepSeek V4 graph dump fixture is not present: {graph_dump}")
    return graph_dump.read_text(encoding="utf-8")


def _layer_block(graph_dump: str, layer_idx: int) -> str:
    layer_marker = f"%layers_{layer_idx}_"
    start = graph_dump.index(layer_marker)
    next_start = graph_dump.find(f"%layers_{layer_idx + 1}_", start + len(layer_marker))
    if next_start == -1:
        return graph_dump[start:]
    return graph_dump[start:next_start]


def _single_line(block: str, marker: str) -> str:
    matches = [line for line in block.splitlines() if marker in line]
    assert len(matches) == 1, f"expected one line containing {marker!r}, found {len(matches)}"
    return matches[0]


def _single_dsv4_op_line(block: str, markers: tuple[str, ...]) -> str:
    matches = [line for line in block.splitlines() if any(marker in line for marker in markers)]
    assert len(matches) == 1, f"expected one DSV4 op line for {markers!r}, found {len(matches)}"
    return matches[0]


def _op_call_count(graph_dump: str, op_call: str) -> int:
    return graph_dump.count(f" = {op_call}")


def _assert_has_regex(text: str, pattern: str) -> None:
    assert re.search(pattern, text), pattern


@pytest.mark.parametrize(
    ("path_name", "reference_op", "device_op", "expected_count", "wave"),
    _DEVICE_OP_GATES,
)
def test_deepseek_v4_5layer_graph_dump_uses_device_custom_ops_for_supported_paths(
    path_name: str,
    reference_op: str,
    device_op: str,
    expected_count: int,
    wave: str,
) -> None:
    graph_dump = _read_graph_dump()
    reference_count = _op_call_count(graph_dump, reference_op)
    device_count = _op_call_count(graph_dump, device_op)

    assert reference_count == 0, (
        f"{wave}: {path_name} still exposes {reference_count} PyTorch reference "
        f"wrapper call(s) in {_graph_dump_path()}; expected zero {reference_op} "
        f"call(s)."
    )
    assert device_count == expected_count, (
        f"{wave}: {path_name} exposes {device_count} device op call(s) in "
        f"{_graph_dump_path()}; expected {expected_count} {device_op} call(s)."
    )


def test_deepseek_v4_5layer_graph_dump_attention_inventory() -> None:
    graph_dump = _read_graph_dump()
    assert (
        sum(
            _op_call_count(graph_dump, f"auto_deploy.{marker}(") for marker in _ATTENTION_OP_MARKERS
        )
        == 5
    )

    expected_by_layer = {
        0: {
            "topk_width": 128,
            "compress_ratio": 0,
            "max_compressed_len": "None",
            "compressor_fragments": ("s72xs70x0 : torch.bfloat16", "0x0 : torch.bfloat16"),
        },
        1: {
            "topk_width": 128,
            "compress_ratio": 0,
            "max_compressed_len": "None",
            "compressor_fragments": ("s72xs70x0 : torch.bfloat16", "0x0 : torch.bfloat16"),
        },
        2: {
            "topk_width": 640,
            "compress_ratio": 4,
            "max_compressed_len": "2048",
            "compressor_fragments": (
                "s72xs70x1024 : torch.bfloat16",
                "4x1024 : torch.float32",
                "512 : torch.float32",
            ),
        },
        3: {
            "topk_width": 192,
            "compress_ratio": 128,
            "max_compressed_len": "64",
            "compressor_fragments": (
                "s72xs70x512 : torch.bfloat16",
                "128x512 : torch.float32",
                "512 : torch.float32",
            ),
        },
        4: {
            "topk_width": 640,
            "compress_ratio": 4,
            "max_compressed_len": "2048",
            "compressor_fragments": (
                "s72xs70x1024 : torch.bfloat16",
                "4x1024 : torch.float32",
                "512 : torch.float32",
            ),
        },
    }

    for layer_idx, expected in expected_by_layer.items():
        block = _layer_block(graph_dump, layer_idx)
        attention_line = _single_dsv4_op_line(block, _ATTENTION_OP_MARKERS)
        assert f"s72xs70x{expected['topk_width']} : torch.int32" in attention_line
        assert (
            f", 128, {expected['compress_ratio']}, {expected['max_compressed_len']}, 1e-06, 64)"
            in attention_line
        )
        for fragment in expected["compressor_fragments"]:
            assert fragment in attention_line


def test_deepseek_v4_5layer_graph_dump_ratio4_indexer_inventory() -> None:
    graph_dump = _read_graph_dump()

    for layer_idx in (2, 4):
        block = _layer_block(graph_dump, layer_idx)
        _assert_has_regex(
            block,
            rf"%layers_{layer_idx}_attn_indexer_compressor_wkv_torch_linear_simple_\d+ = "
            rf"auto_deploy\.torch_linear_simple\.default\([^\n]*s72xs70x256 : torch\.bfloat16",
        )
        _assert_has_regex(
            block,
            rf"%layers_{layer_idx}_attn_indexer_compressor_wgate_torch_linear_simple_\d+ = "
            rf"auto_deploy\.torch_linear_simple\.default\([^\n]*s72xs70x256 : torch\.bfloat16",
        )
        _assert_has_regex(
            block,
            rf"%layers_{layer_idx}_attn_indexer_compressor_torch_rope_with_complex_freqs_\d+ = "
            r"auto_deploy\.torch_rope_with_complex_freqs\.default",
        )
        _assert_has_regex(
            block,
            rf"%layers_{layer_idx}_attn_indexer_clamp_\d+ = aten\.clamp\.default"
            rf"\([^\n]*2048\*s72x4x32 : torch\.float32, -6\.0, 6\.0\)",
        )
        _assert_has_regex(
            block,
            rf"%layers_{layer_idx}_attn_indexer_topk(?:_\d+)? = aten\.topk\.default"
            rf"\([^\n]*s72xs70x2048 : torch\.float32, 512\)",
        )
        _assert_has_regex(
            block,
            rf"%layers_{layer_idx}_attn_indexer_all_reduce_\d+ = "
            rf"auto_deploy\.trtllm_dist_all_reduce\.default"
            rf"\([^\n]*s72xs70x2048 : torch\.float32, NCCL\)",
        )


def test_deepseek_v4_5layer_graph_dump_moe_and_collective_inventory() -> None:
    graph_dump = _read_graph_dump()
    all_reduce_lines = [
        line
        for line in graph_dump.splitlines()
        if " = auto_deploy.trtllm_dist_all_reduce.default(" in line
    ]
    assert len(all_reduce_lines) == 17

    for layer_idx in range(5):
        block = _layer_block(graph_dump, layer_idx)
        grouped_wo_a_line = _single_line(
            block,
            "torch_fake_quant_deepseek_v4_wo_a_grouped_finegrained_fp8_linear.default",
        )
        assert "s72xs70x1x4096 : torch.bfloat16" in grouped_wo_a_line
        assert ") : s72xs70x1x1024 : torch.bfloat16" in grouped_wo_a_line

        router_line = _single_dsv4_op_line(block, _ROUTER_OP_MARKERS)
        assert ") : (s70*s72x6, s70*s72x6) : (torch.int64, torch.float32)" in router_line
        if layer_idx <= 2:
            assert f"%layers_{layer_idx}_ffn_gate_tid2eid : 129280x6 : torch.int64" in router_line
            assert ", 6, 1.5, True)" in router_line
        else:
            assert f"%layers_{layer_idx}_ffn_gate_bias : 256 : torch.float32" in router_line
            assert ", None, 6, 1.5, False)" in router_line

        mxfp4_moe_line = _single_line(block, "triton_deepseek_v4_mxfp4_moe_from_routing.default")
        for fragment in (
            "s70*s72x6 : torch.int64",
            "s70*s72x6 : torch.float32",
            "32x4096x128x16 : torch.uint8",
            "32x4096x128 : torch.uint8",
            "1.0, 10.0",
            "32x4096x64x16 : torch.uint8",
            "32x4096x64 : torch.uint8",
        ):
            assert fragment in mxfp4_moe_line

        for weight_name, weight_shape in (
            ("w1", "256x4096"),
            ("w3", "256x4096"),
            ("w2", "4096x256"),
        ):
            assert (
                f"%layers_{layer_idx}_ffn_shared_experts_{weight_name}_weight : "
                f"{weight_shape} : torch.float8_e4m3fn"
            ) in block

        _assert_has_regex(
            block,
            rf"%layers_{layer_idx}_attn_all_reduce(?:_\d+)? = "
            rf"auto_deploy\.trtllm_dist_all_reduce\.default"
            rf"\(%layers_{layer_idx}_attn_wo_b_torch_linear_simple_\d+ : "
            rf"s72xs70x4096 : torch\.bfloat16, NCCL\)",
        )
        _assert_has_regex(
            block,
            r"%trtllm_dist_all_reduce_default(?:_\d+)? = "
            r"auto_deploy\.trtllm_dist_all_reduce\.default"
            r"\(%triton_deepseek_v4_mxfp4_moe_from_routing_default(?:_\d+)? : "
            r"s70\*s72x4096 : torch\.bfloat16, NCCL\)",
        )
        _assert_has_regex(
            block,
            r"%all_reduce_default(?:_\d+)? = auto_deploy\.trtllm_dist_all_reduce\.default"
            r"\(%torch_fake_quant_finegrained_fp8_linear_default(?:_\d+)? : "
            r"s70\*s72x4096 : torch\.bfloat16, NCCL\)",
        )
