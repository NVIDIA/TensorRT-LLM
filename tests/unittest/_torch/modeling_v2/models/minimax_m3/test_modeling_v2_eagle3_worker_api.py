# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual SM107 dispatch methods against the current parent API.

Only the Python dispatch method is loaded, so this contract runs without CUDA.
The strict parent signature comes from its real source rather than a separately
maintained stub; GPU serving checks cover the model computations themselves.
"""

import ast
import copy
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
MODEL_ROOT = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
PARENT_PATH = ROOT / "tensorrt_llm/_torch/speculative/eagle3.py"


def _dispatch(class_node):
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_draft_loop"
    )


def _load_dispatch(tp, fused):
    parent_tree = ast.parse(PARENT_PATH.read_text())
    parent_class = next(
        n
        for n in parent_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Eagle3OneModelWorker"
    )
    parent = copy.deepcopy(_dispatch(parent_class))
    parent.body = ast.parse(
        "self.parent_arguments = locals().copy()\nreturn self.parent_result"
    ).body
    path = MODEL_ROOT / f"minimax_m3_nvfp4__sm_107__tp{tp}/modeling.py"
    target_tree = ast.parse(path.read_text())
    target_class = next(
        n for n in target_tree.body if isinstance(n, ast.ClassDef) and n.name == "M3Eagle3Worker"
    )
    method = copy.deepcopy(_dispatch(target_class))
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Parent",
                bases=[],
                keywords=[],
                body=[parent],
                decorator_list=[],
                type_params=[],
            ),
            ast.ClassDef(
                name="Worker",
                bases=[ast.Name(id="Parent", ctx=ast.Load())],
                keywords=[],
                body=[method],
                decorator_list=[],
                type_params=[],
            ),
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    worker = namespace["Worker"]()
    worker._m3_fused_loop_ok = lambda *args: fused
    worker._m3_wide_draft_logits_ok = lambda *args: False
    worker.parent_result = (object(), object())
    worker.fused_result = object()
    worker._m3_fused_draft_loop = lambda *args: worker.fused_result
    return worker, namespace["Parent"]._forward_draft_loop


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("fused", [False, True], ids=["stock", "fused"])
def test_dispatch_matches_current_parent_signature_and_return_contract(tp, fused):
    worker, parent = _load_dispatch(tp, fused)
    parameters = list(inspect.signature(parent).parameters)[1:]
    arguments = {name: object() for name in parameters}
    # Use every required parent argument, including the real accepted tokens and
    # pre-draft length reference. The old override rejects this exact call.
    result = worker._forward_draft_loop(**arguments)
    assert isinstance(result, tuple) and len(result) == 2
    if fused:
        assert result[0] is worker.fused_result and result[1] is None
    else:
        assert result is worker.parent_result
        assert all(worker.parent_arguments[name] is value for name, value in arguments.items())


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_external_shared_kv_keeps_the_parent_tensor_return_contract(tp):
    class ExternalWorker:
        __slots__ = ()
        _uses_external_shared_target_kv = True
        tokens = object()

        def _forward_external_shared_target_kv_draft_loop(self, **kwargs):
            return self.tokens

    paths = [PARENT_PATH, MODEL_ROOT / f"minimax_m3_nvfp4__sm_107__tp{tp}/modeling.py"]
    for path in paths:
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name in ("Eagle3OneModelWorker", "M3Eagle3Worker")
        )
        forward = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_impl"
        )
        branch = copy.deepcopy(
            next(
                n
                for n in forward.body
                if isinstance(n, ast.If)
                and ast.unparse(n.test) == "self._uses_external_shared_target_kv"
            )
        )
        # Isolate the actual external branch: it retains a Tensor return, while
        # the normal draft-loop dispatcher now returns a two-item tuple.
        branch.orelse = []
        names = {n.id for n in ast.walk(branch) if isinstance(n, ast.Name)} - {"self"}
        namespace = {name: object() for name in names}
        namespace.update(self=ExternalWorker(), next_new_tokens=None)
        code = ast.Module(body=[branch], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
        assert namespace["next_draft_tokens"] is ExternalWorker.tokens
        assert namespace["next_new_tokens"] is None
