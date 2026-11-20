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


if __name__ == "__main__":
    unittest.main()
