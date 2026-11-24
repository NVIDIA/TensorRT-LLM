# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax-H3 FP8 U4 Video VAE configuration (SM103 & SM107)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

U4 = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_fp8__sm_103_107__u4"
)


def _load_vae_opt(capability: tuple[int, int]):
    tree = ast.parse((U4 / "vae_opt.py").read_text())
    keep = {
        "_SUPPORTED_VAE_CAPABILITIES",
        "_DEFAULT_VAE_GRAPH_MAX_SHAPES",
        "_VAE_FP8_TILE",
        "_VAE_FP8_QTILE_U1",
        "_VAE_FP8_QTILE_U4",
        "_is_supported_fp8_vae_cuda",
        "configure_u4_video_vae_pre_load",
        "configure_u4_video_vae_post_load",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in keep:
            # Replace lazy relative import with injected test helpers
            new_body = [stmt for stmt in node.body if not isinstance(stmt, ast.ImportFrom)]
            fn = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=new_body,
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=node.type_comment,
                type_params=getattr(node, "type_params", []),
            )
            ast.copy_location(fn, node)
            body.append(fn)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in keep for t in node.targets
        ):
            body.append(node)
    events = []
    ns = {
        "torch": SimpleNamespace(
            device=object,
            cuda=SimpleNamespace(get_device_capability=lambda dev: capability),
        ),
        "logger": SimpleNamespace(info=lambda msg: events.append(("log", msg))),
        "set_tiles": lambda mlp, q: events.append(("set_tiles", mlp, q)),
        "convert_decoder_linears_fp8": lambda dec: events.append(("convert", dec)),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "vae_opt.py"), "exec"), ns)
    ns["_events"] = events
    return ns


class TestFp8U4VaeOptContracts(unittest.TestCase):
    def test_module_does_not_eagerly_import_nvfp4_scenario_config(self):
        tree = ast.parse((U4 / "vae_opt.py").read_text())
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for imp in top_imports:
            mod = getattr(imp, "module", "") or ""
            self.assertNotIn("nvfp4", mod)

    def test_unsupported_arch_and_cpu_are_strict_noops(self):
        for cap, dev_type in (((9, 0), "cuda"), ((10, 7), "cpu")):
            ns = _load_vae_opt(cap)
            vae = SimpleNamespace(decoder="DEC", decoder_graph_max_shapes=0)
            dev = SimpleNamespace(type=dev_type)
            self.assertFalse(ns["configure_u4_video_vae_pre_load"](vae, dev))
            self.assertFalse(ns["configure_u4_video_vae_post_load"](vae, dev))
            self.assertEqual(vae.decoder_graph_max_shapes, 0)
            self.assertEqual(ns["_events"], [])

    def test_sm103_and_sm107_configure_chunk_graph_and_fp8_decoder(self):
        for cap in ((10, 3), (10, 7)):
            ns = _load_vae_opt(cap)
            vae = SimpleNamespace(decoder="DEC", decoder_graph_max_shapes=0)
            dev = SimpleNamespace(type="cuda")
            self.assertTrue(ns["configure_u4_video_vae_pre_load"](vae, dev))
            self.assertEqual(vae.decoder_graph_max_shapes, 2)
            self.assertTrue(ns["configure_u4_video_vae_post_load"](vae, dev, ulysses_size=4))
            self.assertIn(("set_tiles", (16, 1024, 8), (64, 4, 8)), ns["_events"])
            self.assertIn(("convert", "DEC"), ns["_events"])


if __name__ == "__main__":
    unittest.main()
