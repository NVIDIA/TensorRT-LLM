# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[6]
MODULE = (
    ROOT / "tensorrt_llm" / "_torch" / "attention_backend" / "sparse" / "deepseek_v4" / "module.py"
)
METADATA = (
    ROOT
    / "tensorrt_llm"
    / "_torch"
    / "attention_backend"
    / "sparse"
    / "deepseek_v4"
    / "metadata.py"
)


class _Tensor:
    def __init__(self, values, *, _storage=None, _start=0, _stop=None):
        self._storage = list(values) if _storage is None else _storage
        self._start = _start
        self._stop = len(self._storage) if _stop is None else _stop

    @property
    def values(self):
        return self._storage[self._start : self._stop]

    @property
    def shape(self):
        return (self._stop - self._start,)

    def data_ptr(self):
        return id(self._storage), self._start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            if step != 1:
                raise AssertionError("the production path uses contiguous views")
            return _Tensor(
                [],
                _storage=self._storage,
                _start=self._start + start,
                _stop=self._start + stop,
            )
        return self._storage[self._start + index]

    def copy_(self, values):
        values = values.values if isinstance(values, _Tensor) else list(values)
        if len(values) != self.shape[0]:
            raise AssertionError("copy size mismatch")
        self._storage[self._start : self._stop] = values
        return self


class _Torch:
    int32 = "int32"

    @staticmethod
    def cumsum(source, dim, *, dtype, out):
        if dim != 0 or dtype != _Torch.int32:
            raise AssertionError("unexpected cumsum contract")
        total = 0
        values = []
        for value in source.values:
            total += value
            values.append(total)
        out.copy_(values)


def _load_function(path, name, *, class_name=None, namespace=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = tree.body
    if class_name is not None:
        body = next(
            node.body
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        arg.annotation = None
    function.returns = None
    compiled_namespace = {} if namespace is None else dict(namespace)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        compiled_namespace,
    )
    return compiled_namespace[name], source


_FUSED_Q_ROPE_SPECS, _MODULE_SOURCE = _load_function(MODULE, "_fused_q_rope_specs")
_PREP_GEN_PREFIX, _METADATA_SOURCE = _load_function(
    METADATA,
    "mla_prepare_fused_q_gen_cu_seqlens",
    class_name="DeepseekV4TrtllmAttentionMetadata",
    namespace={"torch": _Torch},
)


class _Mqa:
    def __init__(self):
        self.rotary_cos_sin = object()
        self.ensure_calls = []

    def _ensure_rope_table_size(self, size):
        self.ensure_calls.append(size)


def _prefix(values):
    total = 0
    result = [0]
    for value in values:
        total += value
        result.append(total)
    return result


def _ragged_metadata(
    verify_lens,
    *,
    num_contexts=0,
    context_lens=(),
    kv_lens=None,
    ctx_prefix=None,
):
    num_generations = len(verify_lens)
    num_ctx_tokens = sum(context_lens)
    num_tokens = num_ctx_tokens + sum(verify_lens)
    if kv_lens is None:
        kv_lens = [1000 + index + length for index, length in enumerate(verify_lens)]
    metadata = SimpleNamespace(
        kv_lens_cuda_runtime=_Tensor(kv_lens),
        num_ctx_tokens=num_ctx_tokens,
        num_tokens=num_tokens,
        max_seq_len=max(kv_lens),
        is_ragged_verify=True,
        num_contexts=num_contexts,
        num_generations=num_generations,
        seq_lens_cuda=_Tensor([*context_lens, *verify_lens]),
        fused_q_gen_cu_seqlens=_Tensor([0] * (num_generations + 1)),
        _fused_q_gen_cu_seqlens_valid=False,
    )
    metadata.mla_prepare_fused_q_gen_cu_seqlens = types.MethodType(_PREP_GEN_PREFIX, metadata)
    if ctx_prefix is not None:
        metadata.mla_prepare_ctx_cu_seqlens = lambda: ctx_prefix
    return metadata


class TestRaggedFusedQRoPESpecs(unittest.TestCase):
    def setUp(self):
        self.mla = SimpleNamespace(mqa=_Mqa())

    def _assert_generation_contract(self, verify_lens):
        metadata = _ragged_metadata(verify_lens)
        cos_sin, specs = _FUSED_Q_ROPE_SPECS(
            self.mla, metadata, num_contexts=0, num_generations=len(verify_lens)
        )

        self.assertIs(cos_sin, self.mla.mqa.rotary_cos_sin)
        self.assertEqual(len(specs), 1)
        rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
        self.assertEqual((rows.start, rows.stop), (0, sum(verify_lens)))
        self.assertEqual(cache_lens.values, metadata.kv_lens_cuda_runtime.values)
        self.assertEqual(seq_len, 0)
        self.assertEqual(cu_q_seqlens.values, _prefix(verify_lens))

        # Mirrors the fused CUDA kernel's cu_q position formula. Cache lengths
        # include this step's tokens, so every request must start at its own
        # pre-step cache position rather than at a uniform-V/G approximation.
        for request, query_len in enumerate(verify_lens):
            cached_after_step = cache_lens[request]
            seq_begin = cu_q_seqlens[request]
            for local_token in range(query_len):
                token = seq_begin + local_token
                position = (token - seq_begin) + (cached_after_step - query_len)
                self.assertEqual(position, cached_after_step - query_len + local_token)

    def test_heterogeneous_g128_generation_contract(self):
        for verify_lens in (
            [3] * 64 + [5] * 64,
            [4] * 64 + [6] * 64,
            [5] * 64 + [6] * 64,
        ):
            with self.subTest(global_budget=sum(verify_lens)):
                self._assert_generation_contract(verify_lens)

    def test_uniform_k5_keeps_scalar_generation_spec(self):
        num_generations = 128
        tokens_per_request = 6
        metadata = SimpleNamespace(
            kv_lens_cuda_runtime=_Tensor([1006] * num_generations),
            num_ctx_tokens=0,
            num_tokens=num_generations * tokens_per_request,
            max_seq_len=1006,
            is_ragged_verify=False,
            mla_prepare_fused_q_gen_cu_seqlens=lambda: self.fail(
                "uniform K5 must not prepare ragged prefixes"
            ),
        )

        _, specs = _FUSED_Q_ROPE_SPECS(
            self.mla, metadata, num_contexts=0, num_generations=num_generations
        )

        self.assertEqual(len(specs), 1)
        rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
        self.assertEqual((rows.start, rows.stop), (0, 768))
        self.assertEqual(cache_lens.values, [1006] * num_generations)
        self.assertEqual(seq_len, tokens_per_request)
        self.assertIsNone(cu_q_seqlens)

    def test_mixed_batch_uses_disjoint_context_and_generation_prefixes(self):
        context_lens = [2, 3]
        verify_lens = [2, 1, 3]
        ctx_prefix = _Tensor(_prefix(context_lens))
        kv_lens = [102, 203, 302, 401, 503]
        metadata = _ragged_metadata(
            verify_lens,
            num_contexts=len(context_lens),
            context_lens=context_lens,
            kv_lens=kv_lens,
            ctx_prefix=ctx_prefix,
        )

        _, specs = _FUSED_Q_ROPE_SPECS(
            self.mla,
            metadata,
            num_contexts=len(context_lens),
            num_generations=len(verify_lens),
        )

        self.assertEqual(len(specs), 2)
        ctx_rows, ctx_cache, ctx_seq_len, ctx_cu_q = specs[0]
        gen_rows, gen_cache, gen_seq_len, gen_cu_q = specs[1]
        self.assertEqual((ctx_rows.start, ctx_rows.stop), (0, 5))
        self.assertEqual((gen_rows.start, gen_rows.stop), (5, 11))
        self.assertEqual(ctx_cache.values, kv_lens[:2])
        self.assertEqual(gen_cache.values, kv_lens[2:])
        self.assertEqual((ctx_seq_len, gen_seq_len), (0, 0))
        self.assertIs(ctx_cu_q, ctx_prefix)
        self.assertEqual(gen_cu_q.values, [0, 2, 3, 6])
        self.assertNotEqual(ctx_cu_q.data_ptr(), gen_cu_q.data_ptr())

    def test_context_only_keeps_its_original_prefix_boundary(self):
        ctx_prefix = _Tensor([0, 2, 5])
        metadata = SimpleNamespace(
            kv_lens_cuda_runtime=_Tensor([102, 203]),
            num_ctx_tokens=5,
            num_tokens=5,
            max_seq_len=203,
            mla_prepare_ctx_cu_seqlens=lambda: ctx_prefix,
        )

        _, specs = _FUSED_Q_ROPE_SPECS(self.mla, metadata, num_contexts=2, num_generations=0)

        self.assertEqual(len(specs), 1)
        rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
        self.assertEqual((rows.start, rows.stop), (0, 5))
        self.assertEqual(cache_lens.values, [102, 203])
        self.assertEqual(seq_len, 0)
        self.assertIs(cu_q_seqlens, ctx_prefix)

    def test_ragged_generation_fails_closed_without_stable_prefix_hook(self):
        metadata = SimpleNamespace(
            kv_lens_cuda_runtime=_Tensor([10, 20]),
            num_ctx_tokens=0,
            num_tokens=3,
            max_seq_len=20,
            is_ragged_verify=True,
        )
        self.assertEqual(_FUSED_Q_ROPE_SPECS(self.mla, metadata, 0, 2), (None, []))
        metadata.mla_prepare_fused_q_gen_cu_seqlens = lambda: None
        self.assertEqual(_FUSED_Q_ROPE_SPECS(self.mla, metadata, 0, 2), (None, []))

    def test_prefix_storage_address_is_fixed_across_rebuilds(self):
        metadata = _ragged_metadata([2, 1, 3])
        first = metadata.mla_prepare_fused_q_gen_cu_seqlens()
        first_address = first.data_ptr()
        self.assertEqual(first.values, [0, 2, 3, 6])

        metadata.seq_lens_cuda = _Tensor([1, 4, 1])
        metadata._fused_q_gen_cu_seqlens_valid = False
        second = metadata.mla_prepare_fused_q_gen_cu_seqlens()
        self.assertEqual(second.data_ptr(), first_address)
        self.assertEqual(second.values, [0, 1, 5, 6])

    def test_prefix_builder_fails_closed_without_graph_storage(self):
        metadata = SimpleNamespace(
            is_ragged_verify=True,
            num_contexts=0,
            num_generations=2,
            seq_lens_cuda=_Tensor([1, 2]),
            fused_q_gen_cu_seqlens=None,
            _fused_q_gen_cu_seqlens_valid=False,
        )
        self.assertIsNone(_PREP_GEN_PREFIX(metadata))
        metadata.fused_q_gen_cu_seqlens = _Tensor([0, 0])
        self.assertIsNone(_PREP_GEN_PREFIX(metadata))

    def test_storage_and_rebuild_are_cuda_graph_structural(self):
        tree = ast.parse(_METADATA_SOURCE)
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4TrtllmAttentionMetadata"
        )

        def method_source(name):
            node = next(
                item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == name
            )
            return ast.get_source_segment(_METADATA_SOURCE, node)

        create = method_source("__post_init__")
        prepare = method_source("mla_prepare_fused_q_gen_cu_seqlens")
        invalidate = method_source("_invalidate_mla_scheduler_buffers")
        device_layout = method_source("apply_device_ragged_layout")

        self.assertIn('cache_name="fused_q_gen_cu_seqlens"', create)
        self.assertIn("self.cuda_graph_buffers", create)
        self.assertIn("capture_graph=capture_graph", create)
        self.assertNotIn("torch.empty", prepare)
        self.assertIn("out=cu_seqlens[1 : num_generations + 1]", prepare)
        self.assertIn("self._fused_q_gen_cu_seqlens_valid = False", invalidate)
        self.assertLess(
            device_layout.index("super().apply_device_ragged_layout"),
            device_layout.index("self._invalidate_mla_scheduler_buffers()"),
        )


if __name__ == "__main__":
    unittest.main()
