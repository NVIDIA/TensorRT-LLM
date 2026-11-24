# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax-H3 static-FP8 four-GPU Ulysses transport."""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest

    pytestmark = pytest.mark.cpu_only
except ImportError:  # pragma: no cover - bare unittest invocation without pytest
    pytest = None

REPO = Path(__file__).resolve().parents[6]
U4 = REPO / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_fp8__sm_103_107__u4"


class _FakeFP8Method:
    def get_static_input_scale(self, module):
        if module.input_scale is not None and not module.force_dynamic_quantization:
            return module.input_scale
        return None


class _FakeUnquantMethod:
    pass


class _FakeScale:
    def __init__(
        self,
        value: float,
        *,
        dtype="fp32",
        numel: int = 1,
        is_cuda: bool = True,
        ptr: int = 1024,
        version: int = 0,
    ):
        self.value = value
        self.dtype = dtype
        self._numel = numel
        self.is_cuda = is_cuda
        self._ptr = ptr
        self._version = version

    def numel(self) -> int:
        return self._numel

    def data_ptr(self) -> int:
        return self._ptr

    def __gt__(self, other):
        return SimpleNamespace(all=lambda: self.value > other)


class _FakeTensor:
    def __init__(self, shape, *, dtype="bf16", is_cuda=True, stride_last=1):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._stride_last = stride_last

    def stride(self, dim):
        assert dim == -1
        return self._stride_last

    def view(self, *shape):
        return _FakeTensor(shape, dtype=self.dtype, is_cuda=self.is_cuda)

    def flatten(self, start_dim):
        head = self.shape[:start_dim]
        tail = 1
        for d in self.shape[start_dim:]:
            tail *= d
        return _FakeTensor((*head, tail), dtype=self.dtype, is_cuda=self.is_cuda)


def _load_output_exchange_namespace():
    tree = ast.parse((U4 / "output_exchange.py").read_text())
    keep_names = {
        "_ULYSSES_SIZE",
        "_LOCAL_HEADS",
        "_HEAD_DIM",
        "_LOCAL_WIDTH",
        "_FULL_WIDTH",
        "_CHUNK_COLS",
        "_NUM_CHUNKS",
        "static_fp8_output_scale",
        "unpack_u4_output",
        "exchange_fp8_output",
        "exchange_bf16_output",
    }
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in keep_names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in keep_names for t in node.targets)
        )
    ]
    ns = {
        "torch": SimpleNamespace(
            Tensor=object,
            bfloat16="bf16",
            float32="fp32",
            float8_e4m3fn="fp8",
            uint8="u8",
        ),
        "FP8QDQLinearMethod": _FakeFP8Method,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(U4 / "output_exchange.py"), "exec"), ns
    )
    return ns


def _load_pipeline_namespace(output_ns):
    tree = ast.parse((U4 / "pipeline.py").read_text())
    cls_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Fp8UlyssesAttention"
    )
    finite_calls = []

    def fake_isfinite(scale):
        finite_calls.append(scale)
        return SimpleNamespace(all=lambda: math.isfinite(scale.value))

    exchanged = []
    exchanged_bf16 = []
    fused_qk = []

    def fake_exchange(raw_out, scale, group):
        exchanged.append((raw_out.shape, scale.value, group))
        return _FakeTensor((1, raw_out.shape[1] // 4, 56 * 128), dtype="fp8")

    def fake_exchange_bf16(raw_out, group):
        exchanged_bf16.append((raw_out.shape, group))
        return _FakeTensor((1, raw_out.shape[1] // 4, 56 * 128), dtype="bf16")

    def fake_qk_norm_rope_pack(qkv, wq, wk, cos, sin, eps, valid_rows, group):
        fused_qk.append((qkv.shape, valid_rows, group))
        return ("SEND_FUSED", "DESCALES_FUSED")

    ns = {
        "torch": SimpleNamespace(
            Tensor=object,
            bfloat16="bf16",
            float32="fp32",
            float8_e4m3fn="fp8",
            isfinite=fake_isfinite,
            compiler=SimpleNamespace(disable=lambda fn: fn),
        ),
        "AttentionTensorLayout": SimpleNamespace(NHD="NHD", HND="HND"),
        "MiniMaxH3Attention": type(
            "MiniMaxH3Attention", (), {"__init__": lambda self, *a, **k: None}
        ),
        "static_fp8_output_scale": output_ns["static_fp8_output_scale"],
        "exchange_fp8_output": fake_exchange,
        "exchange_bf16_output": fake_exchange_bf16,
        "pack_qkv": lambda q, k, v, valid_rows, group: ("SEND", "DESCALES"),
        "qk_norm_rope_pack_qkv": fake_qk_norm_rope_pack,
    }
    exec(compile(ast.Module(body=[cls_node], type_ignores=[]), str(U4 / "pipeline.py"), "exec"), ns)
    ns["_finite_calls"] = finite_calls
    ns["_exchanged"] = exchanged
    ns["_exchanged_bf16"] = exchanged_bf16
    ns["_fused_qk"] = fused_qk
    return ns


def _make_linear(
    *,
    quant_method=None,
    dtype="bf16",
    all_reduce=None,
    weight_shape=(5376, 7168),
    scale=None,
    force_dynamic=False,
):
    if quant_method is None:
        quant_method = _FakeFP8Method()
    if scale is None:
        scale = _FakeScale(0.25)
    return SimpleNamespace(
        quant_method=quant_method,
        dtype=dtype,
        all_reduce=all_reduce,
        weight=_FakeTensor(weight_shape, dtype="fp8"),
        input_scale=scale,
        force_dynamic_quantization=force_dynamic,
    )


class TestFp8U4OutputExchangeContracts(unittest.TestCase):
    def setUp(self):
        self.out_ns = _load_output_exchange_namespace()
        self.pipe_ns = _load_pipeline_namespace(self.out_ns)

    def test_static_fp8_output_scale_accepts_valid_static_linear(self):
        linear = _make_linear()
        self.assertIs(self.out_ns["static_fp8_output_scale"](linear), linear.input_scale)

    def test_static_fp8_output_scale_rejects_unqualified_linears(self):
        cases = {
            "unquantized_method": _make_linear(quant_method=_FakeUnquantMethod()),
            "subclass_method": _make_linear(quant_method=type("SubFP8", (_FakeFP8Method,), {})()),
            "wrong_dtype_none": _make_linear(dtype=None),
            "wrong_dtype_fp16": _make_linear(dtype="fp16"),
            "tp_all_reduce": _make_linear(all_reduce=object()),
            "wrong_in_features": _make_linear(weight_shape=(5376, 4096)),
            "force_dynamic": _make_linear(force_dynamic=True),
            "non_fp32_scale": _make_linear(scale=_FakeScale(0.25, dtype="bf16")),
            "non_scalar_scale": _make_linear(scale=_FakeScale(0.25, numel=2)),
            "cpu_scale": _make_linear(scale=_FakeScale(0.25, is_cuda=False)),
        }
        for name, linear in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self.out_ns["static_fp8_output_scale"](linear))

    def test_unpack_and_exchange_reject_bad_shapes(self):
        with self.assertRaises(ValueError):
            self.out_ns["unpack_u4_output"](_FakeTensor((2, 4, 1792), dtype="u8"))
        with self.assertRaises(ValueError):
            self.out_ns["unpack_u4_output"](_FakeTensor((4, 4, 1024), dtype="u8"))
        with self.assertRaises(ValueError):
            self.out_ns["exchange_fp8_output"](
                _FakeTensor((1, 13, 14, 128), dtype="bf16"),
                _FakeScale(1.0),
                None,
            )

    def test_attention_guard_caches_finite_positive_scale_and_rejects_bad_values(self):
        cls = self.pipe_ns["Fp8UlyssesAttention"]
        attn = cls()
        attn.attn = SimpleNamespace(
            world_size=4,
            inner_backend=SimpleNamespace(preferred_layout="NHD"),
        )
        good = _make_linear(scale=_FakeScale(0.5, ptr=2048))
        attn.to_out = [good]
        self.assertIs(attn._validated_fp8_output_scale(), good.input_scale)
        self.assertEqual(len(self.pipe_ns["_finite_calls"]), 1)
        # Second call reuses cached validation without re-syncing scale.
        self.assertIs(attn._validated_fp8_output_scale(), good.input_scale)
        self.assertEqual(len(self.pipe_ns["_finite_calls"]), 1)

        for idx, bad_val in enumerate((0.0, -1.0, float("nan"), float("inf")), start=1):
            with self.subTest(bad_val=bad_val):
                bad = _make_linear(scale=_FakeScale(bad_val, ptr=4096 + idx * 16))
                attn.to_out = [bad]
                self.assertIsNone(attn._validated_fp8_output_scale())

    def test_packed_forward_routes_fp8_and_bf16_blocks_and_counts_calls(self):
        cls = self.pipe_ns["Fp8UlyssesAttention"]
        attn = cls()
        bf16_calls = []

        def fake_forward_packed_fp8(send, *, fp8_descales, return_pre_a2a=False, **kwargs):
            if return_pre_a2a:
                return _FakeTensor((1, 16, 14, 128), dtype="bf16")
            bf16_calls.append((send, fp8_descales))
            return _FakeTensor((1, 4, 56, 128), dtype="bf16")

        attn.attn = SimpleNamespace(
            world_size=4,
            _ulysses_rank=0,
            process_group="PG",
            inner_backend=SimpleNamespace(preferred_layout="NHD"),
            forward_packed_fp8=fake_forward_packed_fp8,
        )
        q = _FakeTensor((1, 4, 56 * 128), dtype="bf16")
        k = _FakeTensor((1, 4, 56 * 128), dtype="bf16")
        v = _FakeTensor((1, 4, 56 * 128), dtype="bf16")

        # Static-FP8 block takes FP8 output exchange.
        attn.to_out = [_make_linear(scale=_FakeScale(0.25, ptr=8192))]
        out_fp8 = attn._packed_forward(q, k, v)
        self.assertEqual(out_fp8.dtype, "fp8")
        self.assertEqual(out_fp8.shape, (1, 4, 56 * 128))
        self.assertEqual(attn._fp8_output_exchange_calls, 1)
        self.assertEqual(attn._bf16_output_exchange_calls, 0)
        self.assertEqual(len(self.pipe_ns["_exchanged"]), 1)

        # Protected BF16 block uses fast BF16 unpack on NHD.
        attn.to_out = [_make_linear(quant_method=_FakeUnquantMethod())]
        out_bf16 = attn._packed_forward(q, k, v)
        self.assertEqual(out_bf16.dtype, "bf16")
        self.assertEqual(out_bf16.shape, (1, 4, 56 * 128))
        self.assertEqual(attn._fp8_output_exchange_calls, 1)
        self.assertEqual(attn._bf16_output_exchange_calls, 1)
        self.assertEqual(len(self.pipe_ns["_exchanged_bf16"]), 1)
        self.assertEqual(len(bf16_calls), 0)

        # HND layout falls back cleanly to stock BF16 output A2A.
        attn.attn.inner_backend.preferred_layout = "HND"
        attn.to_out = [_make_linear(scale=_FakeScale(0.25, ptr=8192))]
        out_hnd = attn._packed_forward(q, k, v)
        self.assertEqual(out_hnd.dtype, "bf16")
        self.assertEqual(attn._bf16_output_exchange_calls, 2)
        self.assertEqual(len(bf16_calls), 1)

        # Fused QK/RoPE + pack routes cleanly when enabled.
        attn.attn.inner_backend.preferred_layout = "NHD"
        attn.norm_q = SimpleNamespace(weight="WQ", variance_epsilon=1e-6)
        attn.norm_k = SimpleNamespace(weight="WK", variance_epsilon=1e-6)
        attn.to_out = [_make_linear(scale=_FakeScale(0.25, ptr=8192))]
        qkv = _FakeTensor((1, 4, 3 * 56 * 128), dtype="bf16")
        qkv.is_contiguous = lambda: True
        out_fused = attn._fused_packed_forward(qkv, ("COS", "SIN"))
        self.assertEqual(out_fused.dtype, "fp8")
        self.assertEqual(attn._fused_qk_pack_calls, 1)
        self.assertEqual(len(self.pipe_ns["_fused_qk"]), 1)


if __name__ == "__main__":
    unittest.main()
