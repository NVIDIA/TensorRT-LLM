# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU behavior controls against actual TrtllmAttention/PhasedFmha methods."""

import dataclasses
import types
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends import trtllm as backend
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization import QuantMode

A = backend.TrtllmAttention
M = backend.TrtllmAttentionMetadata


class CpuMetadata(M):
    """Real metadata type with CPU views and no device-buffer allocation."""

    def __init__(self, **values: object) -> None:
        object.__setattr__(self, "_qa_values", values)

    def __getattribute__(self, name: str) -> object:
        if name == "_qa_values":
            return object.__getattribute__(self, name)
        values = object.__getattribute__(self, "_qa_values")
        return values[name] if name in values else object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        values = object.__getattribute__(self, "_qa_values")
        if name in values:
            values[name] = value
        else:
            object.__setattr__(self, name, value)


class Layer:
    """Layer carrier for the actual backend methods without model weights."""

    head_dim: int
    local_layer_idx: int

    def out_head_size(self, is_gen_only: bool) -> int:
        return self.head_dim

    def get_local_layer_idx(self, metadata: CpuMetadata) -> int:
        return self.local_layer_idx


def fixture(
    index: int = 0, cls: type[PhasedFmha] = FallbackFmha
) -> tuple[Layer, PhasedFmha, torch.Tensor, CpuMetadata, AttentionForwardArgs]:
    a = Layer()
    for k, v in dict(
        is_mla_enable=False,
        sparse_params=None,
        quant_mode=QuantMode(0),
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
        local_layer_idx=index,
        kv_scale_orig_quant=torch.ones(1),
        kv_scale_quant_orig=torch.ones(1),
        cache_context_plan=True,
        _context_plan_fmha=None,
        _context_plan_fmha_owner=None,
        _context_plan_counter_rows=0,
    ).items():
        setattr(a, k, v)
    for name in [
        "_context_plan_sig",
        "_is_plain_context_call",
        "_context_plan_matches",
        "_build_context_plan",
        "_context_plan_fmha_for",
        "_run_context_plan",
        "_fallback_context_plan_eligible",
    ]:
        if hasattr(A, name):
            setattr(a, name, types.MethodType(getattr(A, name), a))
    a._ensure_rope_table_size = Mock()
    if cls is FlashInferTrtllmGenFmha:
        fmha = object.__new__(cls)
        PhasedFmha.__init__(fmha, a)
    else:
        fmha = cls(a)
    fmha.prepare_workspace = Mock()
    fmha.run_context = Mock()
    fmha._get_total_num_blocks = (
        Mock(return_value=16)
        if cls is FlashInferTrtllmGenFmha
        else Mock(side_effect=AssertionError("fallback extent must never be queried"))
    )
    a._fmha_manager = NS(fmha_libs=[fmha], select=Mock(return_value=fmha))
    a._context_plan_layer_key = (cls, 1, 1, 4)
    q = torch.empty(2, 12, dtype=torch.bfloat16)
    m = CpuMetadata(
        kv_lens_runtime=torch.tensor([3], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2], dtype=torch.int32),
        kv_lens_cuda_runtime=torch.tensor([3], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([2], dtype=torch.int32),
        num_contexts=1,
        num_generations=0,
        num_ctx_tokens=2,
        kv_cache_block_offsets=torch.empty(1, 2, 4, dtype=torch.int32),
        effective_workspace=torch.empty(1, dtype=torch.uint8),
        kv_cache_manager=NS(max_attention_window_vec=[16, 16], dtype=DataType.BF16),
        host_kv_cache_pool_pointers=torch.tensor([123], dtype=torch.int64),
        beam_width=1,
        max_context_q_len_override=None,
        spec_decoding_bl_tree_mask=None,
        spec_bl_tree_first_sparse_mask_offset_kv=None,
        helix_position_offsets=None,
        enable_flash_mla=False,
        max_num_sequences=None,
        max_num_requests=2,
        cache_indirection=None,
        tokens_per_block=32,
        is_cross=False,
        use_paged_context_fmha=True,
        max_seq_len=16,
        cu_q_seqlens=None,
        cu_kv_seqlens=None,
        context_plan=None,
        is_cuda_graph=False,
        workspace_reclaimable=True,
    )
    m.mixed_precision_staging = None
    m.draft_mixed_precision_staging = None
    f = AttentionForwardArgs(output=torch.empty(2, 4, dtype=torch.bfloat16), is_fused_qkv=True)
    return a, fmha, q, m, f


def same(test: unittest.TestCase, x: FmhaParams, y: FmhaParams) -> None:
    for item in dataclasses.fields(x):
        p = getattr(x, item.name)
        q = getattr(y, item.name)
        if isinstance(p, torch.Tensor):
            test.assertIsInstance(q, torch.Tensor)
            test.assertEqual(
                (p.data_ptr(), p.storage_offset(), p.shape, p.stride(), p.dtype, p.device),
                (q.data_ptr(), q.storage_offset(), q.shape, q.stride(), q.dtype, q.device),
            )
        elif item.name in ["attn", "meta", "fwd"]:
            test.assertIs(p, q)
        else:
            test.assertEqual(p, q, item.name)


pytestmark = pytest.mark.cpu_only


class TestContextPlan(unittest.TestCase):
    def test_mixed_precision_uses_regular_path_before_and_after_plan(self) -> None:
        for field in [
            "mixed_precision_staging",
            "draft_mixed_precision_staging",
            "mixed_precision_high_block_table",
            "mixed_precision_low_block_offsets",
            "draft_mixed_precision_high_block_table",
            "draft_mixed_precision_low_block_offsets",
            "manager_high_layers",
        ]:
            for plan_built in [False, True]:
                with self.subTest(field=field, plan_built=plan_built):
                    a, lib, q, m, f = fixture()
                    if plan_built:
                        self.assertTrue(a._run_context_plan(q, None, None, m, f))
                    lib.run_context.reset_mock()
                    if field == "manager_high_layers":
                        m.kv_cache_manager.mixed_precision_high_layers = {0: 1}
                    else:
                        setattr(m, field, object())
                    self.assertFalse(a._run_context_plan(q, None, None, m, f))
                    lib.run_context.assert_not_called()

    def test_plan_hit_rechecks_metadata_admission(self) -> None:
        for field, value in [
            ("beam_width", 2),
            ("max_context_q_len_override", 1),
            ("spec_decoding_bl_tree_mask", object()),
            ("spec_bl_tree_first_sparse_mask_offset_kv", object()),
            ("helix_position_offsets", object()),
            ("enable_flash_mla", True),
        ]:
            for cls in [FallbackFmha, FlashInferTrtllmGenFmha]:
                with self.subTest(field=field, library=cls.__name__):
                    a, lib, q, m, f = fixture(cls=cls)
                    self.assertTrue(a._run_context_plan(q, None, None, m, f))
                    lib.run_context.reset_mock()
                    setattr(m, field, value)
                    self.assertFalse(a._run_context_plan(q, None, None, m, f))
                    lib.run_context.assert_not_called()

    def test_output_allocated_before_attempting_plan(self) -> None:
        a, lib, q, m, f = fixture()
        f.output = None
        allocated = torch.empty(2, 4, dtype=torch.bfloat16)
        a.create_output = Mock(return_value=(allocated,))
        actual = a._run_context_plan

        def require_output(q, k, v, metadata, fwd) -> bool:
            self.assertIsNotNone(
                fwd.output, "ordinary caller output must be allocated before plan admission"
            )
            return actual(q, k, v, metadata, fwd)

        a._run_context_plan = require_output
        # This fixture exercises allocation order without constructing GPU metadata.
        result = A.forward(a, q, None, None, m, f)
        self.assertIs(result, allocated)
        a.create_output.assert_called_once()
        lib.run_context.assert_called_once()

    def test_fi_existing_and_missing_output(self) -> None:
        for missing in [False, True]:
            with self.subTest(missing=missing):
                a, lib, q, m, f = fixture(cls=FlashInferTrtllmGenFmha)
                allocated = f.output
                if missing:
                    f.output = None
                a.create_output = Mock(return_value=(allocated,))
                self.assertIs(A.forward(a, q, None, None, m, f), allocated)
                self.assertEqual(a.create_output.call_count, 1 if missing else 0)
                lib.run_context.assert_called_once()

    def test_forward_keeps_error_contract_and_allocation_order(self) -> None:
        for kind in ["window-pair", "window-generation", "dsv4"]:
            with self.subTest(kind=kind):
                a, lib, q, m, f = fixture()
                f.output = None
                a.create_output = Mock()
                a._run_context_plan = Mock(return_value=False)
                if kind.startswith("window"):
                    f.variable_window_token_starts = torch.tensor([0], dtype=torch.int32)
                    if kind == "window-generation":
                        f.variable_window_token_ends = torch.tensor([2], dtype=torch.int32)
                        f.attention_input_type = AttentionInputType.generation_only
                else:
                    f.enable_dsv4_epilogue_fusion = True
                with self.assertRaises(ValueError):
                    A.forward(a, q, None, None, m, f)
                a.create_output.assert_not_called()
                lib.run_context.assert_not_called()

    def test_disabled_plan_keeps_regular_allocator(self) -> None:
        a, _, q, m, f = fixture()
        a.cache_context_plan = False
        f.output = None

        class StopAtAllocator(Exception):
            pass

        a.create_output = Mock(side_effect=StopAtAllocator)
        a._run_context_plan = Mock(side_effect=AssertionError("disabled plan"))
        with self.assertRaises(StopAtAllocator):
            A.forward(a, q, None, None, m, f)
        a.create_output.assert_called_once()
        a._run_context_plan.assert_not_called()

    def test_fallback_two_layers_and_fresh_parameters(self) -> None:
        a, fmha, q, m, f = fixture()
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        got = fmha.run_context.call_args.args[0]
        expected = fmha.context_params(q, None, None, m, f, m.effective_workspace, 0, 2, 3)
        same(self, got, expected)
        self.assertEqual(got.total_num_blocks, 0)
        b, second, q2, _, f2 = fixture(1)
        self.assertTrue(b._run_context_plan(q2, None, None, m, f2))
        self.assertEqual(a._fmha_manager.select.call_count, 1)
        self.assertEqual(b._fmha_manager.select.call_count, 0)
        new = second.run_context.call_args.args[0]
        self.assertIs(new.attn, b)
        self.assertIs(new.fwd, f2)
        self.assertIs(new.qkv_input, q2)
        self.assertEqual(new.local_layer_idx, 1)
        self.assertIsNot(new, got)
        fmha._get_total_num_blocks.assert_not_called()
        second._get_total_num_blocks.assert_not_called()

    def test_fi_keeps_extent_workspace_and_rebuild(self) -> None:
        a, fmha, q, m, f = fixture(cls=FlashInferTrtllmGenFmha)
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        self.assertEqual(fmha.prepare_workspace.call_count, 1)
        self.assertEqual(fmha.run_context.call_args.args[0].total_num_blocks, 16)
        fmha._get_total_num_blocks.return_value = 32
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        self.assertEqual(a._fmha_manager.select.call_count, 2)
        self.assertEqual(fmha.run_context.call_args.args[0].total_num_blocks, 32)

    def test_fallback_subclass_keeps_regular_forward(self) -> None:
        class Specialized(FallbackFmha):
            def forward(self, *args, **kwargs):
                raise AssertionError("special forward cannot be bypassed")

        a, _, q, m, f = fixture(cls=Specialized)
        self.assertFalse(a._run_context_plan(q, None, None, m, f))

    def test_recipe_and_quant_scope(self) -> None:
        for field, value in [
            ("sage_attn_smooth_v", True),
            ("sage_attn_smooth_k", True),
            ("sage_attn_qk_int8", True),
            ("sage_attn_num_elts_per_blk_q", 1),
            ("sage_quantized", object()),
            ("vc_attention", True),
            ("vc_operands", object()),
            ("out_scale", torch.ones(1)),
            ("qk_fp8_per_tensor", True),
            ("v_fp8_per_tensor", True),
            ("sparse_attn_phase", 1),
            ("timestep", 1.0),
        ]:
            with self.subTest(field=field):
                a, _, q, m, f = fixture()
                setattr(f, field, value)
                self.assertFalse(a._run_context_plan(q, None, None, m, f))
        for mode in [QuantMode.INT8_KV_CACHE, QuantMode.FP8_KV_CACHE, QuantMode.INT8_WEIGHTS]:
            with self.subTest(mode=mode):
                a, _, q, m, f = fixture()
                a.quant_mode = mode
                self.assertFalse(a._run_context_plan(q, None, None, m, f))
        a, _, q, m, f = fixture()
        f.sparse_runtime_params.sparse_kv_indices = torch.ones(1, dtype=torch.int32)
        self.assertFalse(a._run_context_plan(q, None, None, m, f))

    def test_noncovered_states(self) -> None:
        for field, value in [
            ("num_generations", 1),
            ("is_cross", True),
            ("enable_flash_mla", True),
            ("beam_width", 2),
            ("effective_workspace", None),
            ("kv_cache_block_offsets", None),
            ("spec_decoding_bl_tree_mask", torch.ones(1)),
            ("helix_position_offsets", torch.ones(1)),
        ]:
            with self.subTest(field=field):
                a, _, q, m, f = fixture()
                setattr(m, field, value)
                self.assertFalse(a._run_context_plan(q, None, None, m, f))
        a, _, q, m, f = fixture()
        a.is_mla_enable = True
        self.assertFalse(a._run_context_plan(q, None, None, m, f))
        a, _, q, m, f = fixture()
        f.attention_input_type = AttentionInputType.generation_only
        self.assertFalse(a._run_context_plan(q, None, None, m, f))
        a, _, q, m, f = fixture()
        self.assertFalse(a._run_context_plan(q.float(), None, None, m, f))

    def test_plan_hit_cannot_bypass_new_recipe(self) -> None:
        a, lib, q, m, f = fixture()
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        lib.run_context.reset_mock()
        f.vc_attention = True
        self.assertFalse(a._run_context_plan(q, None, None, m, f))
        lib.run_context.assert_not_called()

    def test_workspace_tensor_and_window_rebind(self) -> None:
        for change in ["workspace", "kv_tensor", "manager", "pool", "window", "signature"]:
            with self.subTest(change=change):
                a, lib, q, m, f = fixture()
                self.assertTrue(a._run_context_plan(q, None, None, m, f))
                if change == "workspace":
                    m.effective_workspace = torch.empty(2, dtype=torch.uint8)
                if change == "kv_tensor":
                    m.kv_lens_runtime = m.kv_lens_runtime.clone()
                if change == "manager":
                    m.kv_cache_manager = NS(max_attention_window_vec=[16, 16], dtype=DataType.BF16)
                if change == "pool":
                    m.host_kv_cache_pool_pointers = m.host_kv_cache_pool_pointers.clone()
                if change == "window":
                    f.attention_window_size = 8
                if change == "signature":
                    a._context_plan_layer_key = ("different-heads",)
                self.assertTrue(a._run_context_plan(q, None, None, m, f))
                self.assertEqual(a._fmha_manager.select.call_count, 2)

    def test_manager_replacement_uses_new_library(self) -> None:
        a, lib, q, m, f = fixture()
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        fresh = FallbackFmha(a)
        fresh.prepare_workspace = Mock()
        fresh.run_context = Mock()
        fresh._get_total_num_blocks = Mock(side_effect=AssertionError())
        a._fmha_manager = NS(fmha_libs=[fresh], select=Mock(return_value=fresh))
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        fresh.run_context.assert_called_once()
        fresh.prepare_workspace.assert_called_once()

    def test_existing_lifecycle_invalidates_inplace_lengths(self) -> None:
        meta = object.__new__(M)
        meta.enable_flash_mla = False
        meta.fp4_mla_state = None
        meta.mixed_precision_staging = None
        for method in ["on_update_kv_lens", "update_for_spec_dec"]:
            meta.context_plan = object()
            getattr(meta, method)()
            self.assertIsNone(meta.context_plan)
        meta.context_plan = object()
        meta.draft_replay_swapped_attrs = {}
        meta.foo = 1
        meta.swap_for_draft("foo", 2)
        self.assertIsNone(meta.context_plan)
        meta.context_plan = object()
        meta.restore_draft_swaps()
        self.assertIsNone(meta.context_plan)
        self.assertEqual(meta.foo, 1)
        with patch.object(AttentionMetadata, "restore_from_spec_dec", lambda s: None):
            meta.context_plan = object()
            meta.restore_from_spec_dec()
            self.assertIsNone(meta.context_plan)
        meta.context_plan = object()
        meta.update_spec_dec_param(1, False, False, False, 0, 0)
        self.assertIsNone(meta.context_plan)
        meta.context_plan = object()
        t = torch.tensor([7], dtype=torch.int32)
        meta._bind_runtime_views(
            kv_lens_cuda=t, kv_lens=t, prompt_lens_cuda=t, prompt_lens_cpu=t, host_request_types=t
        )
        self.assertIsNone(meta.context_plan)
        a, _, q, m, f = fixture()
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        m.kv_lens_runtime.fill_(9)
        # Call the actual lifecycle invalidator on this lightweight metadata facade.
        m.enable_flash_mla = False
        m.fp4_mla_state = None
        m.mixed_precision_staging = None
        m._invalidate_mla_scheduler_buffers = types.MethodType(
            M._invalidate_mla_scheduler_buffers, m
        )
        M.on_update_kv_lens(m)
        self.assertTrue(a._run_context_plan(q, None, None, m, f))
        self.assertEqual(m.context_plan.max_kv_len, 9)

    def test_prepare_invalidates_before_h2d(self) -> None:
        meta = object.__new__(M)
        meta.context_plan = object()
        meta.kv_cache_manager = object()
        meta.prompt_lens = [2]
        meta._seq_lens = torch.tensor([2], dtype=torch.int32)
        meta.prompt_lens_cpu = torch.empty(1, dtype=torch.int32)
        meta.prompt_lens_cuda = torch.empty(1, dtype=torch.int32)

        class StopBeforeDeviceCopy(Exception):
            pass

        with (
            patch.object(AttentionMetadata, "prepare", lambda s: None),
            patch.object(backend, "get_model_extra_attrs", return_value={}),
            patch.object(backend, "copy_h2d_pool_shared", side_effect=StopBeforeDeviceCopy),
        ):
            with self.assertRaises(StopBeforeDeviceCopy):
                meta.prepare()
        self.assertIsNone(meta.context_plan)
