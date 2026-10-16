# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only DSpark routing contracts; no Torch, CuTe, or CUDA runtime is loaded.

Execute the production Python guards with tensor metadata and mocked launchers.
The AST extraction follows the existing K2/K3 routing-test prototype. Scalar
mask checks are not a substitute for compiling and testing the GPU kernels.
"""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

_ROOT = Path(__file__).resolve().parents[5]
_OPS = _ROOT / "tensorrt_llm/_torch/custom_ops"
_ATTENTION = _OPS / "dspark_attention_custom_op.py"
_PREPARATION = _OPS / "dspark_rmsnorm_rope_custom_op.py"
_KERNELS = _ROOT / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dspark"
_MODEL = _ROOT / "tensorrt_llm/_torch/models/modeling_dspark.py"
_PHYSICAL_KS = tuple(range(1, 9))


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _load(path, name, namespace, owner=None):
    nodes = _tree(path).body
    if owner is not None:
        nodes = next(node for node in nodes if getattr(node, "name", None) == owner).body
    node = next(node for node in nodes if getattr(node, "name", None) == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _constants(path):
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in _tree(path).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_DSV4_DSPARK_")
    }


def _tensor(shape, dtype="bf16", *, cuda=True, contiguous=True, strides=None):
    if strides is None:
        strides = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        is_cuda=cuda,
        is_contiguous=lambda: contiguous,
        numel=lambda: math.prod(shape),
        stride=lambda dim=None: strides if dim is None else strides[dim],
        data_ptr=lambda: 0,
    )


def _guards(sm=100):
    namespace = {
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32", int32="i32", int64="i64"),
        "get_sm_version": lambda: sm,
        "_log_unsupported": lambda *args: False,
    }
    namespace.update(_constants(_PREPARATION))
    for name in (
        "_get_dspark_arch_str",
        "is_fused_dspark_rmsnorm_rope_supported",
        "_is_dspark_cache_write_supported",
        "_is_dspark_draft_block_supported",
        "is_fused_dspark_attention_preparation_supported",
    ):
        _load(_PREPARATION, name, namespace)
    namespace.update(_constants(_ATTENTION))
    for name in (
        "is_dsv4_dspark_attention_config_supported",
        "is_fused_dsv4_dspark_attention_supported",
        "_is_fused_dsv4_dspark_attention_input_supported",
    ):
        _load(_ATTENTION, name, namespace)
    return namespace


def _inputs(k, batch=2):
    return [
        _tensor((batch, k, 128, 512)),
        _tensor((batch, 8, 512)),
        _tensor((batch + 1, 128, 512), contiguous=False, strides=(196608, 512, 1)),
        _tensor((batch,), "i32"),
        _tensor((batch,), "i32"),
        _tensor((batch,), "i64"),
        _tensor((128,), "fp32"),
        _tensor((batch, k, 32, 2), "fp32"),
    ]


class TestDSparkFusedAttentionRoute(unittest.TestCase):
    def test_storage_bounded_range_is_consistent(self):
        for path in (_ATTENTION, _PREPARATION):
            constants = _constants(path)
            self.assertEqual(constants["_DSV4_DSPARK_BLOCK_SIZES"], _PHYSICAL_KS)
            self.assertEqual(constants["_DSV4_DSPARK_DRAFT_BLOCK_STORAGE_SIZE"], 8)
        kernel = next(
            node
            for node in _tree(_KERNELS / "attention.py").body
            if isinstance(node, ast.ClassDef) and node.name == "DSparkAttention"
        )
        page_size = next(
            ast.literal_eval(node.value)
            for node in kernel.body
            if isinstance(node, ast.Assign) and node.targets[0].id == "page_size_draft"
        )
        self.assertEqual(_PHYSICAL_KS, tuple(range(1, page_size + 1)))

    def test_config_routes_every_k_and_rejects_out_of_range(self):
        supported = _guards()["is_dsv4_dspark_attention_config_supported"]
        for k in (-1, 0, *_PHYSICAL_KS, 9, 16, 128):
            with self.subTest(k=k):
                self.assertEqual(supported(k, 128, 512, 128), k in _PHYSICAL_KS)
        for k in _PHYSICAL_KS:
            self.assertFalse(supported(k, 64, 512, 128))
            self.assertFalse(supported(k, 128, 256, 128))
            self.assertFalse(supported(k, 128, 512, 64))

    def test_kernel_constructor_keeps_geometry_and_storage_guards(self):
        class KernelBase:
            def __init__(self, *args, **kwargs):
                self.seq_len_q = kwargs["seq_len_q"]

        namespace = {
            "cutlass": SimpleNamespace(Float32="fp32"),
            "cute": SimpleNamespace(jit=lambda f: f),
            "DSparkAttentionKernel": KernelBase,
        }
        kernel = _load(_KERNELS / "attention.py", "DSparkAttention", namespace)
        args = ("fp32", (128, 128), (128, 256), 1, 8, 128, 0.0)
        for k in _PHYSICAL_KS:
            with self.subTest(k=k):
                op = kernel(*args, arch_str="sm_100", seq_len_q=k, inverse_rope_dim=64)
                self.assertEqual(op.seq_len_q, k)
                self.assertEqual(op.fixed_cache_seq_len, 128 + k)
                self.assertEqual(op.tma_page_size_draft, 128)
        for k in (-1, 0, 9, 16):
            with self.subTest(k=k), self.assertRaises(ValueError):
                kernel(*args, arch_str="sm_100", seq_len_q=k)
        with self.assertRaises(ValueError):
            kernel(*args, arch_str="sm_100", seq_len_q=8, inverse_rope_dim=32)
        with self.assertRaises(ValueError):
            kernel(*args[:4], 16, *args[5:], arch_str="sm_100", seq_len_q=8)

    def test_attention_tensor_contract_for_every_k_and_architecture(self):
        for sm in (90, 100, 101, 103, 109, 120, 121):
            gate = _guards(sm)["_is_fused_dsv4_dspark_attention_input_supported"]
            for k in (0, *_PHYSICAL_KS, 9, 16):
                with self.subTest(sm=sm, k=k):
                    self.assertEqual(gate(*_inputs(k)), sm in (100, 103) and k in _PHYSICAL_KS)

    def test_attention_dtype_shape_and_layout_guards_survive(self):
        gate = _guards()["_is_fused_dsv4_dspark_attention_input_supported"]
        bad_inputs = (
            (0, _tensor((2, 4, 128, 512), dtype="fp32")),
            (0, _tensor((2, 4, 64, 512))),
            (0, _tensor((2, 4, 128, 256))),
            (0, _tensor((2, 4, 512))),
            (1, _tensor((2, 7, 512))),
            (1, _tensor((2, 9, 512))),
            (1, _tensor((2, 8, 512), dtype="fp32")),
            (2, _tensor((3, 64, 512))),
            (2, _tensor((3, 128, 512), dtype="fp32")),
            (2, _tensor((3, 128, 512), strides=(0, 512, 1))),
            (2, _tensor((3, 128, 512), strides=(196609, 512, 1))),
            (2, _tensor((3, 128, 512), strides=(196608, 513, 1))),
            (2, _tensor((3, 128, 512), strides=(196608, 512, 2))),
            (3, _tensor((2,), "i64")),
            (4, _tensor((2,), "i64")),
            (4, _tensor((3,), "i32")),
            (5, _tensor((2,), "i32")),
            (5, _tensor((3,), "i64")),
            (6, _tensor((128,), "bf16")),
            (6, _tensor((64,), "fp32")),
            (7, _tensor((2, 4, 31, 2), "fp32")),
        )
        for index, bad in bad_inputs:
            with self.subTest(index=index, shape=bad.shape, dtype=bad.dtype):
                inputs = _inputs(4)
                inputs[index] = bad
                self.assertFalse(gate(*inputs))
        for k in _PHYSICAL_KS:
            for index in range(8):
                inputs = _inputs(k)
                inputs[index].is_cuda = False
                self.assertFalse(gate(*inputs))
                if index != 2:  # The worker's strided cache-page view is supported.
                    inputs = _inputs(k)
                    inputs[index].is_contiguous = lambda: False
                    self.assertFalse(gate(*inputs))

    def test_preparation_bounds_and_guards_for_every_k(self):
        gate = _guards()["is_fused_dspark_attention_preparation_supported"]
        for k in (0, *_PHYSICAL_KS, 9, 16):
            with self.subTest(k=k):
                inputs = [
                    _tensor((2, 1, 512)),
                    _tensor((2, k, 512)),
                    _tensor((512,)),
                    _tensor((2, 32, 2), "fp32"),
                    _tensor((2 * k, 32, 2), "fp32"),
                    _inputs(k)[2],
                    _tensor((2,), "i64"),
                    _tensor((2,), "i64"),
                ]
                self.assertEqual(gate(*inputs), k in _PHYSICAL_KS)
                if k in _PHYSICAL_KS:
                    for index in range(len(inputs)):
                        original = inputs[index].is_cuda
                        inputs[index].is_cuda = False
                        self.assertFalse(gate(*inputs))
                        inputs[index].is_cuda = original
                    inputs[4] = _tensor((2 * k + 1, 32, 2), "fp32")
                    self.assertFalse(gate(*inputs))

    def test_masks_cover_short_full_wrapped_and_empty_windows(self):
        namespace = {
            "cutlass": SimpleNamespace(Int32=int, const_expr=lambda x: x),
            "cute": SimpleNamespace(elem_less=lambda a, b: a < b),
        }
        path = _KERNELS / "attention_kernel.py"
        mask = _load(path, "is_score_valid", namespace, "DSparkAttentionKernel")
        valid_length = _load(path, "get_window_valid_len", namespace, "DSparkAttentionKernel")
        kernel = SimpleNamespace(
            mma_qk_tiler=(128, 128, 128),
            window_valid_len_from_tensor=True,
            tma_page_size_win=128,
        )
        for k in _PHYSICAL_KS:
            for end in (0, 5, 127, 128, 257, 390):
                for valid in (-1, 0, 1, 3, 128, 129):
                    with self.subTest(k=k, end=end, valid=valid):
                        length = valid_length(kernel, [valid], 0)
                        expected = {(end - age) % 128 for age in range(min(max(valid, 0), 128))}
                        actual = {
                            col for col in range(128) if mask(kernel, col, 0, 128 + k, length, end)
                        }
                        self.assertEqual(actual, expected)
                        draft = {
                            col for col in range(128) if mask(kernel, col, 1, 128 + k, length, end)
                        }
                        self.assertEqual(draft, set(range(k)))

    def test_cache_keys_separate_k_and_arch_but_not_runtime_shapes(self):
        namespace = _guards()
        cache = {}
        compile_kernel = Mock(side_effect=lambda *args: Mock())
        namespace["_dspark_attention_kernel_cache"] = cache
        namespace["_compile_dspark_attention"] = compile_kernel
        capturing = Mock(return_value=False)
        namespace["torch"].cuda = SimpleNamespace(is_current_stream_capturing=capturing)
        namespace["torch"].empty_like = lambda q: _tensor(q.shape)
        run = _load(_ATTENTION, "_run_dspark_attention", namespace)
        for arch in ("sm_100", "sm_103"):
            namespace["_get_dspark_arch_str"] = lambda arch=arch: arch
            for k in _PHYSICAL_KS:
                with self.subTest(k=k, arch=arch):
                    for batch, scale in ((1, 0.125), (3, 0.25)):
                        self.assertEqual(run(*_inputs(k, batch), scale).shape, (batch, k, 128, 512))
                    self.assertEqual(cache[(k, arch)].call_count, 2)
        self.assertEqual(compile_kernel.call_count, 16)
        self.assertEqual(capturing.call_count, 16)
        self.assertEqual(len({id(compiled) for compiled in cache.values()}), 16)

    def test_graph_capture_only_uses_warmed_k(self):
        for k in _PHYSICAL_KS:
            namespace = _guards()
            namespace["_dspark_attention_kernel_cache"] = {}
            namespace["_compile_dspark_attention"] = Mock(side_effect=AssertionError("compile"))
            namespace["torch"].cuda = SimpleNamespace(is_current_stream_capturing=lambda: True)
            namespace["torch"].empty_like = Mock(side_effect=AssertionError("allocation"))
            run = _load(_ATTENTION, "_run_dspark_attention", namespace)
            with self.subTest(k=k), self.assertRaisesRegex(RuntimeError, "warmed up"):
                run(*_inputs(k), 0.125)

    def test_unsupported_k_warmup_does_not_allocate(self):
        namespace = _guards()
        warmup = _load(_ATTENTION, "warmup_fused_dsv4_dspark_attention", namespace)
        # The metadata-only torch object cannot allocate; early return is required.
        for k in (-1, 0, 9, 16):
            warmup(k, 1e-6)

    def test_query_storage_and_fallback_routes_remain_separate(self):
        attention = ast.unparse(_tree(_KERNELS / "attention.py"))
        self.assertIn("(self.num_heads, self.head_dim, self.seq_len_q, batch_size)", attention)
        self.assertIn("batch_size * self.seq_len_q * self.inverse_rope_dim", attention)
        self.assertIn("self.fixed_cache_seq_len = self.window_size + seq_len_q", attention)
        model = {}
        _load(_MODEL, "dspark_attention_forward_batched", model)
        function = next(
            node
            for node in _tree(_MODEL).body
            if getattr(node, "name", None) == "dspark_attention_forward_batched"
        )
        route = next(
            node
            for node in function.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "use_fused_dsv4_dspark_attention"
        )
        fused = ast.unparse(ast.Module(body=route.body, type_ignores=[]))
        fallback = ast.unparse(ast.Module(body=route.orelse, type_ignores=[]))
        self.assertIn("cute_dsl_dspark_rmsnorm_rope_draft_block(", fused)
        self.assertIn("fused_dsv4_dspark_attention(", fused)
        self.assertNotIn("fused_dsv4_dspark_attention(", fallback)
        self.assertIn(
            "get_dspark_topk_idxs_batched(window_size, block, start_pos, valid_len)", fallback
        )
        self.assertIn("torch.cat([cache_rows, kv], dim=1)", fallback)


if __name__ == "__main__":
    unittest.main()
