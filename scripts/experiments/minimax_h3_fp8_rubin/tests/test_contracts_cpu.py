# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fixed scenario selection and the upstream-forward transformation."""

import ast
import copy
import hashlib
import json
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def test_frozen_launch_selection_cpu(self) -> None:
        tree = ast.parse((ROOT / "long_round3/long_round3_runtime.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "configure")
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<actual-configure>", "exec"), scope)

        @dataclass(frozen=True)
        class Frozen:
            fp8_fusion_block_h: int = 8
            fp8_fusion_warps: int = 2

        half, original = object(), object()
        qk = SimpleNamespace(
            SCENARIO=Frozen(), _V2=True, _VAMAX_FOLD=False, _tensor_cast_g_kernel=original
        )
        obj = SimpleNamespace(
            qk=qk,
            half_rope=half,
            original_kernel=original,
            original_config=(8, 2, True, False),
            sg=SimpleNamespace(run=object()),
            residual=SimpleNamespace(STATE={}),
            gateup=SimpleNamespace(configure=lambda enabled: None),
            down=SimpleNamespace(configure=lambda enabled: None),
        )
        scope["configure"](obj, True)
        self.assertIs(qk._tensor_cast_g_kernel, half)
        scope["configure"](obj, False)
        self.assertIs(qk._tensor_cast_g_kernel, original)
        qk.SCENARIO = Frozen(4, 2)
        with self.assertRaises(ValueError):
            scope["configure"](obj, True)
        self.assertIs(qk._tensor_cast_g_kernel, original)

    def test_current_forward_inverse_cpu(self) -> None:
        default = (
            ROOT.parents[2]
            / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/transformer_minimax_h3.py"
        )
        path = Path(os.environ.get("H3_NATIVE_TRANSFORMER_SOURCE", str(default)))
        source = path.read_text()
        proof = json.loads((ROOT / "long_round5/round5_shared_forward_proof.json").read_text())
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), proof["native_file_sha256"])
        cls = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Transformer3DModel"
        )
        native = copy.deepcopy(
            next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        )
        native.body.pop(0)
        clone_path = ROOT / "long_round5/round5_shared_forward.py"
        self.assertEqual(hashlib.sha256(clone_path.read_bytes()).hexdigest(), proof["clone_sha256"])
        clone = next(
            n for n in ast.parse(clone_path.read_text()).body if isinstance(n, ast.FunctionDef)
        )
        clone.name, clone.decorator_list = native.name, copy.deepcopy(native.decorator_list)
        assignments = [
            n
            for n in clone.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_h3_shared_fp32_projection_input"
                for t in n.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        clone.body.remove(assignments[0])
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
                self.assertEqual(node.args[0].id, "_h3_shared_fp32_projection_input")
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
