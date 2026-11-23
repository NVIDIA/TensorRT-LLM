# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax-H3 FP8 U4 Video VAE configuration (SM103 & SM107)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

U4 = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_fp8__sm_103_107__u4"
)


def _load_vae_opt(capability: tuple[int, int]):
    tree = ast.parse((U4 / "vae_opt.py").read_text())
    keep = {
        "_SUPPORTED_VAE_CAPABILITIES",
        "_is_supported_fp8_vae_cuda",
        "configure_u4_video_vae_pre_load",
        "configure_u4_video_vae_post_load",
    }
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in keep)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in keep for t in node.targets)
        )
    ]
    events = []
    ns = {
        "torch": SimpleNamespace(
            device=object,
            cuda=SimpleNamespace(get_device_capability=lambda dev: capability),
        ),
        "logger": SimpleNamespace(info=lambda msg: events.append(("log", msg))),
        "SCENARIO": SimpleNamespace(
            vae_graph_max_shapes=2,
            vae_fp8_tiled=True,
            vae_fp8_tile=(8, 512, 4),
            resolved_vae_fp8_qtile=lambda u: (16, 4, 4),
        ),
        "set_tiles": lambda mlp, q: events.append(("set_tiles", mlp, q)),
        "convert_decoder_linears_fp8": lambda dec: events.append(("convert", dec)),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "vae_opt.py"), "exec"), ns)
    ns["_events"] = events
    return ns


class TestFp8U4VaeOptContracts(unittest.TestCase):
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
            self.assertIn(("set_tiles", (8, 512, 4), (16, 4, 4)), ns["_events"])
            self.assertIn(("convert", "DEC"), ns["_events"])


if __name__ == "__main__":
    unittest.main()
