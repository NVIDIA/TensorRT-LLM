# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU routing guards and numerical-source contracts without GPU imports."""

import ast
import copy
import fnmatch
import logging
import os
import re
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Callable, Optional, Type
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[7]
MODEL = REPO / "tensorrt_llm/_torch/visual_gen/models/minimax_h3"
FP8 = MODEL / "minimax_h3_fp8__sm_107__u1"


def _routing_namespace():
    """Execute the real routing functions with fake hardware and no CUDA imports."""
    tree = ast.parse((MODEL / "scenarios.py").read_text())
    names = {
        "NoScenario",
        "_transformer_quant_config",
        "_transformer_quant_algo",
        "_transformer_layer_quant",
        "_ulysses_size",
        "route_fp8_sm_107_u1",
        "route_nvfp4_sm_107_u1_u4",
        "resolve_scenario",
    }
    body = [
        n
        for n in tree.body
        if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
        or (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "ROUTES"
        )
    ]
    namespace = {
        "__package__": "_h3_fp8_cpu_contracts",
        "Optional": Optional,
        "Type": Type,
        "Callable": Callable,
        "os": os,
        "QuantAlgo": SimpleNamespace(FP8="FP8", NVFP4="NVFP4"),
        "logger": logging.getLogger(__name__),
        "_MODES": ("off", "auto", "require"),
        "_device_capability": lambda: (10, 7),
        "_device_multiprocessors": lambda: 212,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(MODEL / "scenarios.py"), "exec"),
        namespace,
    )
    return namespace


@lru_cache(maxsize=1)
def _checkpoint_exclusion_matcher():
    """Use the real QuantConfig matcher without importing GPU runtime bindings."""
    path = REPO / "tensorrt_llm/models/modeling_utils.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QuantConfig")
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "is_module_excluded_from_quantization"
    )
    namespace = {"fnmatch": fnmatch, "re": re}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


def _quant_config():
    config = SimpleNamespace(quant_algo="FP8", exclude_modules=None)
    config.is_module_excluded_from_quantization = MethodType(
        _checkpoint_exclusion_matcher(), config
    )
    return config


def _uniform_ignore_patterns():
    return [
        "audio_proj_in",
        "audio_proj_out",
        "context_embedder",
        "norm_out*",
        "proj_in",
        "proj_out",
        "time_embedder*",
        "token_refiner*",
    ] + [f"transformer_blocks.{i}.adaln_proj*" for i in range(50)]


def _earlytop8_ignore_patterns():
    # A ModelOpt FP8 export may keep whole blocks in BF16 through its ignore list.
    return _uniform_ignore_patterns() + [
        "transformer_blocks.1.*",
        "transformer_blocks.16*",
        "transformer_blocks.32*",
        "transformer_blocks.34*",
        "transformer_blocks.35*",
        "transformer_blocks.36*",
        "transformer_blocks.37*",
        "transformer_blocks.42*",
    ]


def _config():
    return SimpleNamespace(
        model_configs={
            "transformer": SimpleNamespace(
                quant_config=_quant_config(),
                quant_config_dict=None,
            )
        },
        quant_config=_quant_config(),
        quant_config_dict=None,
        dynamic_weight_quant=False,
        force_dynamic_quantization=False,
        attention=SimpleNamespace(
            backend="CUDNN",
            quant_attention_config=SimpleNamespace(
                qk_dtype="fp8",
                v_dtype="fp8",
            ),
        ),
        visual_gen_mapping=SimpleNamespace(ulysses_size=1),
        mapping=SimpleNamespace(tp_size=1),
        parallel=SimpleNamespace(parallel_vae_size=1, async_ulysses=False),
        torch_compile=SimpleNamespace(enable=True),
        cuda_graph=SimpleNamespace(enable=False),
        cpu_offload_config=SimpleNamespace(enable=False),
        runtime_lora=None,
        cache=None,
        extra_attrs={"visual_gen_scenario": "require"},
    )


class RoutingContracts(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"NVIDIA_TF32_OVERRIDE": "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_unqualified_tf32_settings_are_rejected(self):
        namespace = _routing_namespace()
        for value in [None, "1"]:
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
                else:
                    os.environ["NVIDIA_TF32_OVERRIDE"] = value
                with self.assertRaises(namespace["NoScenario"]):
                    namespace["route_fp8_sm_107_u1"](_config())

    def test_unsupported_combinations_are_rejected_before_import(self):
        changes = [
            ("dynamic_weight_quant", True),
            ("force_dynamic_quantization", True),
            ("mapping.tp_size", 2),
            ("visual_gen_mapping.ulysses_size", 4),
            ("parallel.parallel_vae_size", 4),
            ("parallel.async_ulysses", True),
            ("torch_compile.enable", False),
            ("cuda_graph.enable", True),
            ("cpu_offload_config.enable", True),
            ("runtime_lora", object()),
            ("cache", object()),
            ("attention.backend", "FA4"),
            ("attention.quant_attention_config.qk_dtype", "bf16"),
            ("attention.quant_attention_config.v_dtype", "bf16"),
        ]
        for name, value in changes:
            with self.subTest(name=name):
                cfg = _config()
                owner = cfg
                fields = name.split(".")
                for field in fields[:-1]:
                    owner = getattr(owner, field)
                setattr(owner, fields[-1], value)
                namespace = _routing_namespace()
                with self.assertRaises(namespace["NoScenario"]):
                    namespace["route_fp8_sm_107_u1"](cfg)

    def test_mixed_and_bf16_checkpoints_are_rejected(self):
        for quant, layers in [("BF16", None), ("FP8", {}), ("FP8", {"block8": "BF16"})]:
            cfg = _config()
            cfg.model_configs["transformer"].quant_config.quant_algo = quant
            cfg.model_configs["transformer"].quant_config_dict = layers
            namespace = _routing_namespace()
            with self.assertRaises(namespace["NoScenario"]):
                namespace["route_fp8_sm_107_u1"](cfg)

    def test_actual_architecture_and_sm_count_are_required(self):
        for capability, count in [(None, None), ((10, 3), 152), ((10, 7), 148)]:
            namespace = _routing_namespace()
            namespace["_device_capability"] = lambda: capability
            namespace["_device_multiprocessors"] = lambda: count
            with self.assertRaises(namespace["NoScenario"]):
                namespace["route_fp8_sm_107_u1"](_config())

    def test_selected_route_configures_before_pipeline_import(self):
        namespace = _routing_namespace()
        calls = []
        module = "_h3_fp8_cpu_contracts.minimax_h3_fp8__sm_107__u1"
        config = ModuleType(module + ".config")
        config.select = lambda: calls.append("selected")
        pipeline = ModuleType(module + ".pipeline")
        expected = type("Pipeline", (), {})
        pipeline.MiniMaxH3Fp8Sm107U1Pipeline = expected
        with patch.dict(sys.modules, {module + ".config": config, module + ".pipeline": pipeline}):
            self.assertIs(namespace["route_fp8_sm_107_u1"](_config()), expected)
        self.assertEqual(calls, ["selected"])

    def test_uniform_non_main_exclusions_still_select_fp8(self):
        cfg = _config()
        cfg.model_configs["transformer"].quant_config.exclude_modules = _uniform_ignore_patterns()
        namespace = _routing_namespace()
        module = "_h3_fp8_cpu_contracts.minimax_h3_fp8__sm_107__u1"
        config = ModuleType(module + ".config")
        config.select = lambda: None
        pipeline = ModuleType(module + ".pipeline")
        expected = type("Pipeline", (), {})
        pipeline.MiniMaxH3Fp8Sm107U1Pipeline = expected
        cfg.extra_attrs["visual_gen_scenario"] = "auto"
        with patch.dict(sys.modules, {module + ".config": config, module + ".pipeline": pipeline}):
            self.assertIs(namespace["resolve_scenario"](object, cfg), expected)

    def test_ignore_based_earlytop8_rejects_before_selection(self):
        cfg = _config()
        quant = cfg.model_configs["transformer"].quant_config
        quant.exclude_modules = _earlytop8_ignore_patterns()
        roles = ("attn.qkv_proj", "attn.to_out.0", "ff.gate_up_proj", "ff.down_proj")
        excluded = [
            f"transformer_blocks.{i}.{role}"
            for i in range(50)
            for role in roles
            if quant.is_module_excluded_from_quantization(f"transformer_blocks.{i}.{role}")
        ]
        self.assertEqual(len(excluded), 32)
        self.assertIsNone(cfg.model_configs["transformer"].quant_config_dict)
        namespace = _routing_namespace()
        with self.assertRaisesRegex(namespace["NoScenario"], "excluded"):
            namespace["route_fp8_sm_107_u1"](cfg)
        with self.assertRaisesRegex(RuntimeError, "excluded"):
            namespace["resolve_scenario"](object, cfg)

    def test_auto_and_off_keep_family_for_unsupported_cases(self):
        cases = [
            (_earlytop8_ignore_patterns(), (10, 7), 212, 1),
            (_earlytop8_ignore_patterns(), (10, 3), 152, 1),
            (_earlytop8_ignore_patterns(), (10, 7), 212, 4),
            (_uniform_ignore_patterns(), (10, 3), 152, 1),
            (_uniform_ignore_patterns(), (10, 7), 212, 4),
        ]
        for mode in ("auto", "off"):
            for ignored, capability, count, ulysses in cases:
                with self.subTest(mode=mode, capability=capability, ulysses=ulysses):
                    cfg = _config()
                    cfg.model_configs["transformer"].quant_config.exclude_modules = ignored
                    cfg.visual_gen_mapping.ulysses_size = ulysses
                    cfg.extra_attrs["visual_gen_scenario"] = mode
                    namespace = _routing_namespace()
                    namespace["_device_capability"] = lambda: capability
                    namespace["_device_multiprocessors"] = lambda: count
                    self.assertIs(namespace["resolve_scenario"](object, cfg), object)

    def test_parent_glob_regex_and_global_exclusions_are_honored(self):
        patterns = [
            "transformer_blocks.1",
            "transformer_blocks.1.*",
            "transformer_blocks.1*",
            r"re:transformer_blocks\.1\..*",
            r"re:transformer_blocks\.1",
            "transformer_blocks.*.ff.down_proj",
        ]
        for pattern in patterns:
            for use_global in (False, True):
                with self.subTest(pattern=pattern, global_config=use_global):
                    cfg = _config()
                    if use_global:
                        cfg.model_configs["transformer"].quant_config = None
                        cfg.quant_config.exclude_modules = [pattern]
                    else:
                        cfg.model_configs["transformer"].quant_config.exclude_modules = [pattern]
                    cfg.extra_attrs["visual_gen_scenario"] = "auto"
                    self.assertIs(_routing_namespace()["resolve_scenario"](object, cfg), object)

    def test_off_does_not_call_any_route(self):
        namespace = _routing_namespace()
        cfg = _config()
        cfg.extra_attrs["visual_gen_scenario"] = "off"
        namespace["ROUTES"] = (lambda cfg: self.fail("off must not inspect or import scenarios"),)
        self.assertIs(namespace["resolve_scenario"](object, cfg), object)


class ForwardContract(unittest.TestCase):
    def test_qkv_gemm_keeps_the_opaque_framework_backend(self):
        tree = ast.parse((FP8 / "ops.py").read_text())
        function = copy.deepcopy(
            next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "qkv_gemm")
        )
        self.assertEqual(ast.literal_eval(function.decorator_list[0].args[0]), "h3_fp8::qkv_gemm")
        function.decorator_list = []
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None
        calls = []
        transposed, result = object(), object()
        weight = SimpleNamespace(t=lambda: transposed)
        activation, input_scale, weight_scale = object(), object(), object()

        def scaled_mm(a, w, **kwargs):
            calls.append((a, w, kwargs))
            return result

        namespace = {"torch": SimpleNamespace(_scaled_mm=scaled_mm, bfloat16="BF16")}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "<actual-qkv-gemm>", "exec"),
            namespace,
        )
        self.assertIs(namespace["qkv_gemm"](activation, weight, input_scale, weight_scale), result)
        self.assertEqual(
            calls,
            [
                (
                    activation,
                    transposed,
                    {
                        "scale_a": input_scale,
                        "scale_b": weight_scale,
                        "out_dtype": "BF16",
                        "use_fast_accum": True,
                    },
                )
            ],
        )

    def test_jit_cuda_sources_are_packaged(self):
        tree = ast.parse((REPO / "setup.py").read_text())
        patterns = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "minimax_h3_fp8__sm_107__u1/kernels/" in node.value
        ]
        sources = {p.name for pattern in patterns for p in (REPO / "tensorrt_llm").glob(pattern)}
        self.assertEqual(
            sources, {"sharedCast.cpp", "sharedCast.cu", "staticQuant.cpp", "staticQuant.cu"}
        )

    def test_shared_head_input_is_the_only_forward_change(self):
        tree = ast.parse((MODEL / "transformer_minimax_h3.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Transformer3DModel"
        )
        native = copy.deepcopy(
            next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        )
        native.body.pop(0)
        clone = next(
            n
            for n in ast.parse((FP8 / "transformer_forward.py").read_text()).body
            if isinstance(n, ast.FunctionDef)
        )
        clone.name, clone.decorator_list = native.name, copy.deepcopy(native.decorator_list)
        shared = [
            n
            for n in clone.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "shared_projection_input" for t in n.targets
            )
        ]
        self.assertEqual(len(shared), 1)
        clone.body.remove(shared[0])
        consumers = []
        for node in ast.walk(clone):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in ("proj_out", "audio_proj_out")
            ):
                self.assertEqual(len(node.args), 1)
                self.assertIsInstance(node.args[0], ast.Name)
                self.assertEqual(node.args[0].id, "shared_projection_input")
                node.args[0] = ast.parse(
                    f"self._projection_input(self.{node.func.attr}, packed_hidden_states)",
                    mode="eval",
                ).body
                consumers.append(node.func.attr)
        self.assertEqual(sorted(consumers), ["audio_proj_out", "proj_out"])
        self.assertEqual(
            ast.dump(clone, include_attributes=False), ast.dump(native, include_attributes=False)
        )


class _LayoutTensor:
    """Allocation-free tensor metadata for dispatch tests, not a numerical model."""

    def __init__(self, shape, dtype="BF16"):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim = len(self.shape)
        self.device = "cpu"

    def dim(self):
        return self.ndim

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def reshape(self, *shape):
        if -1 in shape:
            known = 1
            for size in shape:
                known *= size if size != -1 else 1
            shape = tuple(self.numel() // known if size == -1 else size for size in shape)
        return _LayoutTensor(shape, self.dtype)

    def reshape_as(self, other):
        return self.reshape(*other.shape)

    def t(self):
        return _LayoutTensor(self.shape[::-1], self.dtype)

    def index_select(self, dim, indices):
        shape = list(self.shape)
        shape[dim] = indices.numel()
        return _LayoutTensor(shape, self.dtype)

    def __getitem__(self, slices):
        shape = [len(range(*s.indices(n))) for s, n in zip(slices, self.shape)]
        return _LayoutTensor(shape, self.dtype)

    def __add__(self, other):
        return self if self.ndim >= other.ndim else other

    __mul__ = __add__


def _length_fallback_namespace():
    """Execute actual scenario and family methods; replace only CUDA/linear math."""
    calls = []
    tree = ast.parse((FP8 / "ops.py").read_text())
    lengths = ast.literal_eval(
        next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "FULL_SEQUENCE_LENGTHS" for t in n.targets)
        )
    )

    def mark(name, result):
        def call(*args, **kwargs):
            calls.append(name)
            return result(*args, **kwargs)

        return call

    def qk(qkv, *args, **kwargs):
        projected = _LayoutTensor((*qkv.shape[:2], 7168), "FP8")
        return projected, projected, projected, _LayoutTensor((3,), "FP32")

    def dedicated(name, result):
        def call(first, *args):
            rows = first.numel() // first.shape[-1]
            if rows not in lengths:
                raise AssertionError(f"Unqualified rows reached {name}: {rows}")
            calls.append(name)
            return result(first, *args)

        return call

    ops = SimpleNamespace(
        FULL_SEQUENCE_LENGTHS=lengths,
        qkv_gemm=dedicated("qkv_gemm", lambda a, w, *s: _LayoutTensor((a.shape[0], w.shape[0]))),
        norm=dedicated("norm", lambda h, *a: _LayoutTensor(h.shape, "FP8")),
        attention_projection=dedicated(
            "attention_projection", lambda raw, w, si, sw, h, *a: (h, _LayoutTensor(h.shape, "FP8"))
        ),
        gate_up=dedicated("gate_up", lambda a, *w: _LayoutTensor((a.shape[0], 14336), "FP8")),
        down=dedicated("down", lambda a, w, si, sw, h, *rest: h),
    )
    torch = SimpleNamespace(
        Tensor=_LayoutTensor,
        bfloat16="BF16",
        float8_e4m3fn="FP8",
        ops=SimpleNamespace(
            tensorrt_llm=SimpleNamespace(
                static_quantize_e4m3_per_tensor=mark(
                    "producer_quantize", lambda a, s: (_LayoutTensor(a.shape, "FP8"),)
                )
            ),
            trtllm=SimpleNamespace(
                cublas_scaled_mm=mark(
                    "family_qkv", lambda a, w, **kw: _LayoutTensor((a.shape[0], w.shape[1]))
                )
            ),
            h3_lever8=SimpleNamespace(qk_norm_rope_fp8=mark("family_qk_norm", qk)),
        ),
    )

    def gate_res_norm(hidden, projected, *args):
        if projected.shape != hidden.shape:
            raise AssertionError(f"Family block needs OutProj, got {projected.shape}")
        calls.append("family_gate_res_norm")
        return hidden, hidden

    namespace = dict(
        torch=torch,
        _DispatchModule=type(
            "_DispatchModule", (), {"__call__": lambda self, *a, **kw: self.forward(*a, **kw)}
        ),
        Optional=Optional,
        Linear=SimpleNamespace,
        BufferKind=SimpleNamespace(DEFAULT=0, NCCL_WINDOW=1),
        _should_apply_low_m_gemm=lambda *args: False,
        _fp8_per_tensor_uses_cute_dsl_sm107=lambda: False,
        _static_quantize_e4m3_per_tensor=mark(
            "family_quantize", lambda a, s: _LayoutTensor(a.shape, "FP8")
        ),
        h3_norm_mod=mark("family_norm", lambda h, *a: h),
        h3_gate_res_norm_mod=gate_res_norm,
        OUT_QUANT_ENABLED=False,
        Fp4QuantizedTensor=type("Fp4QuantizedTensor", (), {}),
        fusion_mode=lambda: "qkstatic",
        qk_static_descale=lambda *a: _LayoutTensor((3,), "FP32"),
        fused_fp8_qk=SimpleNamespace(qk_static_descale=lambda *a: _LayoutTensor((3,), "FP32")),
        qk_norm_rope=mark("qk_norm_rope", qk),
        ops=ops,
    )
    sources = [
        (
            REPO / "tensorrt_llm/_torch/modules/linear.py",
            {"FP8QDQLinearMethod": {"get_static_input_scale", "apply"}},
        ),
        (
            MODEL / "transformer_minimax_h3.py",
            {"MiniMaxH3TransformerBlock": {"_forward_fused_adaln"}},
        ),
        (
            MODEL / "minimax_h3_nvfp4__sm_107__u1_u4/modeling.py",
            {"ScenarioAttention": {"forward", "_out_proj"}},
        ),
        (FP8 / "modeling.py", {"Fp8QkvMethod": None, "Fp8Attention": None, "Fp8Block": None}),
    ]
    for path, names in sources:
        classes = []
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef) or node.name not in names:
                continue
            selected = names[node.name]
            if selected is not None:
                node.bases = [ast.Name(id="_DispatchModule", ctx=ast.Load())]
                node.body = [
                    n
                    for n in node.body
                    if (isinstance(n, ast.FunctionDef) and n.name in selected)
                    or isinstance(n, ast.AnnAssign)
                ]
                for index, item in enumerate(node.body):
                    if isinstance(item, ast.AnnAssign):
                        node.body[index] = ast.Assign(targets=[item.target], value=item.value)
            classes.append(node)
        module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace, calls


class LengthFallbackContracts(unittest.TestCase):
    def _block(self, namespace, calls):
        tensor = _LayoutTensor

        class Projection:
            def __init__(self, shape):
                self.weight, self.dtype = tensor(shape, "FP8"), "BF16"
                self.input_scale = self.weight_scale = tensor((1,), "FP32")
                self.force_dynamic_quantization = self.enable_cuda_core = False
                self.all_reduce = self.bias = None

            def __call__(self, hidden):
                if self.weight.shape[0] == 21504:
                    return namespace["Fp8QkvMethod"]().apply(self, hidden, None)
                calls.append("out_projection")
                if hidden.shape[-1] != 7168:
                    raise AssertionError("Attention output projection requires width 7168")
                return tensor((*hidden.shape[:-1], 5376))

        class FeedForward:
            gate_up_proj = Projection((28672, 5376))
            down_proj = Projection((5376, 14336))

            def __call__(self, hidden):
                calls.append("family_ffn")
                return tensor(hidden.shape)

        attention = namespace["Fp8Attention"]()
        attention.qkv_proj = Projection((21504, 5376))
        attention.to_out = [Projection((5376, 7168))]
        attention.use_async_ulysses = False
        attention.attn = SimpleNamespace(world_size=1)
        attention.on_fp8_path = lambda: True
        attention._use_fused_fp8_qk = lambda *args: True
        attention._use_packed_a2a = lambda *args: False
        attention.head_dim, attention.local_num_attention_heads, attention.layer_idx = 128, 56, 0
        attention.norm_q = attention.norm_k = SimpleNamespace(
            weight=tensor((128,)), variance_epsilon=1e-6
        )
        attention._attn_impl = lambda q, *a, **kw: tensor(q.shape)
        block = namespace["Fp8Block"]()
        block.attn, block.ff = attention, FeedForward()
        block.norm1 = block.norm2 = SimpleNamespace(weight=tensor((5376,)), variance_epsilon=1e-6)
        block.adaln_proj = SimpleNamespace(table=lambda temb: tensor((3, 6 * 5376)))
        block._fp8_qkv_reciprocal = block._fp8_gate_up_reciprocal = tensor((1,), "FP32")
        block._fp8_output_quant_factor = tensor((1,), "FP32")
        block._fp8_packed_gate_up = tensor((28672, 5376), "FP8")
        return block

    def _run_block(self, shape, dtype="BF16"):
        namespace, calls = _length_fallback_namespace()
        self.assertEqual(namespace["ops"].FULL_SEQUENCE_LENGTHS, (37776, 37820, 38169))
        block = self._block(namespace, calls)
        rows = shape[-2]
        output = block._forward_fused_adaln(
            _LayoutTensor(shape, dtype),
            _LayoutTensor((1, 5376)),
            _LayoutTensor((rows,), "INT64"),
            (_LayoutTensor((rows, 64)), _LayoutTensor((rows, 64))),
            None,
            None,
            None,
        )
        return output, calls

    def test_loader_warmup_and_other_unknown_lengths_use_complete_family_math(self):
        for rows in (37712, 38170):
            with self.subTest(rows=rows):
                output, calls = self._run_block((1, rows, 5376))
                self.assertEqual(output.shape, (1, rows, 5376))
                self.assertEqual(
                    calls,
                    [
                        "family_norm",
                        "family_quantize",
                        "family_qkv",
                        "family_qk_norm",
                        "out_projection",
                        "family_gate_res_norm",
                        "family_ffn",
                    ],
                )

    def test_qualified_request_keeps_dedicated_fusions_and_raw_attention_output(self):
        for rows in (37776, 37820, 38169):
            with self.subTest(rows=rows):
                output, calls = self._run_block((1, rows, 5376))
                self.assertEqual(output.shape, (1, rows, 5376))
                self.assertEqual(
                    calls,
                    ["norm", "qkv_gemm", "qk_norm_rope", "attention_projection", "gate_up", "down"],
                )

    def test_invalid_batch_width_rank_and_dtype_still_fail_before_dispatch(self):
        for shape, dtype in [
            ((2, 37712, 5376), "BF16"),
            ((1, 37712, 5375), "BF16"),
            ((37712, 5376), "BF16"),
            ((1, 37712, 5376), "FP16"),
        ]:
            with self.subTest(shape=shape, dtype=dtype):
                with self.assertRaisesRegex(ValueError, "qualified BF16"):
                    self._run_block(shape, dtype)


if __name__ == "__main__":
    unittest.main()
