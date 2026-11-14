# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise every MiniMax-M3 target against the current Eagle3 worker API.

Only the dispatch and forward token-buffer selection are loaded, without CUDA.
The strict parent signature comes from its real source rather than a separately
maintained stub; GPU serving checks cover the model computations themselves.
"""

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
MODEL_ROOT = ROOT / "tensorrt_llm/_torch/modeling_v2/models/minimax_m3"
PARENT_PATH = ROOT / "tensorrt_llm/_torch/speculative/eagle3.py"
MODEL_PATHS = sorted(MODEL_ROOT.glob("minimax_m3_nvfp4__*/modeling.py"))
assert MODEL_PATHS, f"No MiniMax-M3 NVFP4 modeling.py files found under {MODEL_ROOT}"


def _dispatch(class_node):
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_draft_loop"
    )


def _load_dispatch(path: Path, fused: bool):
    """Load a target dispatcher with the real parent signature and stubbed GPU work."""
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
    worker._m3_next_new_tokens = None
    worker.fused_next_new = object()

    def fused_loop(*args: object) -> object:
        worker._m3_next_new_tokens = worker.fused_next_new
        return worker.fused_result

    worker._m3_fused_draft_loop = fused_loop
    return worker, namespace["Parent"]._forward_draft_loop


@pytest.mark.parametrize("path", MODEL_PATHS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("fused", [False, True], ids=["stock", "fused"])
def test_dispatch_matches_current_parent_signature_and_return_contract(path: Path, fused: bool):
    """Verify argument forwarding and token-buffer handoff for stock and fused dispatch."""
    worker, parent = _load_dispatch(path, fused)
    parameters = list(inspect.signature(parent).parameters)[1:]
    arguments = {name: object() for name in parameters}
    # Use every required parent argument, including the real accepted tokens and
    # pre-draft length reference. The old override rejects this exact call.
    result = worker._forward_draft_loop(**arguments)
    assert isinstance(result, tuple) and len(result) == 2
    if fused:
        assert result[0] is worker.fused_result
        # Targets may return the fused buffer or leave it on the worker for the caller.
        assert result[1] is None or result[1] is worker.fused_next_new
        assert worker._m3_next_new_tokens is worker.fused_next_new
    else:
        assert result is worker.parent_result
        assert all(worker.parent_arguments[name] is value for name, value in arguments.items())


@pytest.mark.parametrize("path", [PARENT_PATH, *MODEL_PATHS], ids=lambda path: path.parent.name)
def test_external_shared_kv_keeps_the_parent_tensor_return_contract(path: Path):
    """Verify external shared-KV drafting returns tokens without a next-input buffer."""

    class ExternalWorker:
        __slots__ = ()
        _uses_external_shared_target_kv = True
        tokens = object()

        def _forward_external_shared_target_kv_draft_loop(self, **kwargs):
            return self.tokens

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


@pytest.mark.parametrize("path", MODEL_PATHS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("mode", ["fallback", "fallback_buffer", "fused", "external"])
def test_forward_preserves_draft_arguments_and_token_buffers(path: Path, mode: str) -> None:
    """Exercise the caller, dispatch, and buffer selection without unrelated forward globals."""
    worker, _ = _load_dispatch(path, fused=mode == "fused")
    worker._uses_external_shared_target_kv = mode == "external"
    worker.parent_result = (
        sentinel.drafts,
        sentinel.next_new if mode == "fallback_buffer" else None,
    )
    worker.fused_result = sentinel.drafts
    worker.fused_next_new = sentinel.next_new
    metadata = SimpleNamespace(
        _seq_lens_cuda=sentinel.pre_draft_seq_lens,
        all_rank_num_tokens=sentinel.rank_tokens,
    )

    def prepare_metadata(md: SimpleNamespace) -> None:
        md._seq_lens_cuda = sentinel.draft_seq_lens

    worker._prepare_attn_metadata_for_spec_dec = Mock(side_effect=prepare_metadata)
    worker._restore_attn_metadata_from_spec_dec = Mock()
    worker.prepare_1st_drafter_inputs = Mock(return_value=sentinel.inputs)
    worker.get_draft_kv_cache_manager = Mock(return_value=sentinel.cache)
    worker._forward_external_shared_target_kv_draft_loop = Mock(return_value=sentinel.drafts)
    worker._prepare_next_new_tokens = Mock(return_value=sentinel.next_new)

    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "M3Eagle3Worker")
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_impl"
    )
    start = next(
        i
        for i, node in enumerate(forward.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self._uses_external_shared_target_kv"
    )
    # Keep only drafting and next-input buffer selection. Acceptance, diagnostics,
    # model state updates, and their module globals are outside this contract.
    buffer_selection = next(
        node
        for node in forward.body[start + 1 :]
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        in ("next_new_tokens is None", "self._m3_next_new_tokens is not None")
    )
    statements = [forward.body[start], buffer_selection]
    namespace = dict(
        self=worker,
        input_ids=sentinel.input_ids,
        position_ids=Mock(),
        hidden_states=sentinel.hidden_states,
        attn_metadata=metadata,
        spec_metadata=SimpleNamespace(batch_indices_cuda=sentinel.batch_indices),
        draft_model=sentinel.draft_model,
        accepted_tokens=sentinel.accepted,
        num_accepted_tokens=sentinel.accepted_counts,
        num_contexts=1,
        num_gens=1,
        batch_size=2,
        resource_manager=sentinel.resources,
        next_new_tokens=None,
    )
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)

    assert namespace["next_draft_tokens"] is sentinel.drafts
    assert namespace["next_new_tokens"] is sentinel.next_new
    if mode.startswith("fallback"):
        assert worker.parent_arguments["inputs"] is sentinel.inputs
        assert worker.parent_arguments["draft_kv_cache_manager"] is sentinel.cache
        assert worker.parent_arguments["accepted_tokens"] is sentinel.accepted
        assert worker.parent_arguments["pre_draft_seq_lens"] is sentinel.pre_draft_seq_lens
    if mode in ("fallback_buffer", "fused"):
        worker._prepare_next_new_tokens.assert_not_called()
    else:
        worker._prepare_next_new_tokens.assert_called_once_with(
            sentinel.accepted, sentinel.drafts, sentinel.batch_indices, 2, sentinel.accepted_counts
        )
