# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU source-contract checks; no CUDA/model import or end-to-end gate implied."""

import ast
import enum
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"
SOURCE = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2"


class Quant(enum.Enum):
    FP8 = "FP8"
    NVFP4 = "NVFP4"
    MXFP8 = "MXFP8"


class NoTarget(Exception):
    pass


def load_source(path, **values):
    """Evaluate exact definitions with only top-level dependency imports supplied by fixtures."""
    tree = ast.parse(path.read_text())
    tree.body = [
        n
        for n in tree.body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
        or (isinstance(n, ast.ImportFrom) and n.module == "__future__")
    ]
    namespace = dict(TYPE_CHECKING=False, enum=enum, os=os, json=json, Path=Path, **values)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def route():
    ns = load_source(
        SOURCE / "models/minimax_m3/routing.py",
        QuantAlgo=Quant,
        NoTarget=NoTarget,
        __package__=PACKAGE + ".models.minimax_m3",
    )
    # The HF quant parser is a framework prerequisite; these tests supply its result.
    ns["_quant_algo"] = lambda config: Quant.MXFP8
    return ns["route"]


def inputs():
    config = types.SimpleNamespace(
        text_config=dict(
            num_hidden_layers=60,
            hidden_size=6144,
            num_local_experts=128,
            intermediate_size=3072,
            num_attention_heads=64,
            num_key_value_heads=4,
            rope_theta=10_000_000,
            sparse_attention_config=dict(use_sparse_attention=True, sparse_attention_freq=[1] * 60),
        ),
        quantization_config=dict(moe_quant_algo="NVFP4", moe_quant_format="nvfp4-pack-quantized"),
    )
    mapping = types.SimpleNamespace(
        world_size=8,
        tp_size=8,
        pp_size=1,
        cp_size=1,
        moe_tp_size=8,
        moe_ep_size=1,
        enable_attention_dp=False,
    )
    return config, mapping


class TestM31Tp8RoutingSource(unittest.TestCase):
    def test_declines_outside_scope(self):
        cases = [("chip", (10, 0)), ("chip", (10, 7))]
        cases += [
            (name, value)
            for name, value in (
                ("world_size", 4),
                ("tp_size", 4),
                ("tp_size", 2),
                ("pp_size", 2),
                ("cp_size", 2),
                ("moe_tp_size", 1),
                ("moe_ep_size", 2),
                ("enable_attention_dp", True),
            )
        ]
        cases += [
            (name, None)
            for name in ("m3_rope", "m3_layers", "wrong_experts", "wrong_shape", "bf16_kv")
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                config, mapping = inputs()
                chip, kv = (10, 3), Quant.NVFP4
                if name == "chip":
                    chip = value
                elif hasattr(mapping, name):
                    setattr(mapping, name, value)
                elif name == "m3_rope":
                    config.text_config["rope_theta"] = 5_000_000
                elif name == "m3_layers":
                    config.text_config["sparse_attention_config"]["sparse_attention_freq"] = [
                        0
                    ] * 3 + [1] * 57
                elif name == "wrong_experts":
                    config.quantization_config["moe_quant_algo"] = "FP8"
                elif name == "wrong_shape":
                    config.text_config["num_key_value_heads"] = 8
                else:
                    kv = None
                with self.assertRaises(NoTarget):
                    route()(config, mapping, chip, None, kv)

    def test_tp8_routes_only_to_tp8(self):
        config, mapping = inputs()
        name = PACKAGE + ".models.minimax_m3.minimax_m31_nvfp4__sm_103__tp8.modeling"
        module = types.ModuleType(name)
        sentinel = type("Target", (), {})
        module.ModelingV2MiniMaxM31Nvfp4Sm103Tp8 = sentinel
        fake_name = PACKAGE + ".models.minimax_m3.fake_ops"
        fake_module = types.ModuleType(fake_name)
        fake_module.register = lambda: None
        with patch.dict(sys.modules, {name: module, fake_name: fake_module}):
            self.assertIs(route()(config, mapping, (10, 3), None, Quant.NVFP4), sentinel)

    def test_router_arity_and_mode(self):
        for arch, arity in [
            ("GptOssForCausalLM", 4),
            ("DeepseekV3ForCausalLM", 4),
            ("MiniMaxM3SparseForConditionalGeneration", 5),
        ]:
            with self.subTest(arch=arch):
                ns = load_source(
                    SOURCE / "_router_index.py", logger=types.SimpleNamespace(info=lambda *a: None)
                )
                calls = []
                target = type("Target", (), {})

                def call(*args):
                    calls.append(args)
                    return target

                ns["routing_module"] = lambda name: types.SimpleNamespace(route=call)
                ns["_device_compute_capability"] = lambda: (10, 3)
                ns["_kv_cache_quant_algo"] = lambda *args: Quant.NVFP4
                config = types.SimpleNamespace(
                    pretrained_config=types.SimpleNamespace(architectures=[arch]),
                    mapping=object(),
                    spec_config=None,
                )
                self.assertIs(
                    ns["modeling_v2_resolve"](config, types.SimpleNamespace(modeling_v2="require")),
                    target,
                )
                self.assertEqual(len(calls[0]), arity)
                if arity == 5:
                    self.assertIs(ns["resolved_mode"](target), ns["ModelingV2Mode"].REQUIRE)
                self.assertEqual(
                    set(ns["MODELING_V2_ROUTERS"]),
                    {
                        "GptOssForCausalLM",
                        "DeepseekV3ForCausalLM",
                        "MiniMaxM3SparseForConditionalGeneration",
                    },
                )

    def test_parent_import_and_late_native_registration(self):
        """Import actual parent with no native ops, then load ops and repeat registration."""
        calls = []
        native = types.SimpleNamespace()
        installed = {}
        torch = types.ModuleType("torch")
        torch.ops = types.SimpleNamespace(trtllm=native)
        torch._C = types.SimpleNamespace(
            _dispatch_has_kernel_for_dispatch_key=lambda name, key: installed.get(name) == "Meta"
        )

        def register_fake(name):
            def decorate(fn):
                self.assertNotIn(name, installed)
                installed[name] = fn
                calls.append(name)
                return fn

            return decorate

        torch.library = types.SimpleNamespace(register_fake=register_fake)
        registry = types.ModuleType("torch._library.simple_registry")
        registry.singleton = types.SimpleNamespace(
            find=lambda name: types.SimpleNamespace(
                fake_impl=types.SimpleNamespace(kernel=installed.get(name))
            )
        )
        family_name = PACKAGE + ".models.minimax_m3"
        catalog_name = PACKAGE + ".catalog.fakes.m31_fake_ops"

        def load_file(name, path, package=False):
            spec = importlib.util.spec_from_file_location(
                name, path, submodule_search_locations=[str(path.parent)] if package else None
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        with patch.dict(sys.modules, {"torch": torch, "torch._library.simple_registry": registry}):
            for fullname in (family_name, catalog_name):
                parts = fullname.split(".")
                for length in range(1, len(parts)):
                    name = ".".join(parts[:length])
                    package = types.ModuleType(name)
                    package.__path__ = [str(ROOT.joinpath(*parts[:length]))]
                    sys.modules[name] = package
            load_file(catalog_name, SOURCE / "catalog/fakes/m31_fake_ops.py")
            parent = load_file(family_name, SOURCE / "models/minimax_m3/__init__.py", package=True)
            self.assertEqual(calls, [])
            self.assertFalse(hasattr(parent, "fake_ops"))
            fake_module = load_file(
                family_name + ".fake_ops", SOURCE / "models/minimax_m3/fake_ops.py"
            )
            self.assertEqual(len(fake_module.FAKES), 7)
            # A route rejected after the parent import can still choose the built-in model.
            config, mapping = inputs()
            with self.assertRaises(NoTarget):
                route()(config, mapping, (10, 7), None, Quant.NVFP4)
            names = list(fake_module.FAKES)
            for name in names:
                setattr(native, name, object())
            installed["trtllm::" + names[0]] = "Meta"
            previous_fake = object()
            installed["trtllm::" + names[1]] = previous_fake
            fake_module.register()
            self.assertEqual(len(calls), 5)
            fake_module.register()
            self.assertEqual(len(calls), 5)
            self.assertIs(installed["trtllm::" + names[1]], previous_fake)
            self.assertEqual(installed["trtllm::" + names[0]], "Meta")


if __name__ == "__main__":
    unittest.main()
