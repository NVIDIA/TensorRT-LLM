# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MegaMoE metadata-push routers with padded top-k rows.

The fused MoE op launches with a fixed number of token rows; rows past the live tokens carry -1 top-k indices,
which the routers map to a trash bucket. The vendored routers count and sort valid routes only (TensorRT-LLM local
patch in token_comm.py and token_comm_dedup.py).

The CPU test checks that the patch is still in the vendored sources, so that a re-sync cannot drop it silently.
The GPU test runs single-GPU router launches (world size 1, so no peer is involved) and checks every routing
record against a CPU reference: padded launches at several fills, including one live token and none, and unpadded
launches of the same live rows.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_NVLINK = _ROOT / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain"
_PACKAGE = "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe"
_ROUTERS = {
    # kind: (module, router class, method that writes the sorted slab)
    "plain": ("token_comm", "_MetadataPushRouter", "_sort_router_elements"),
    "dedup": ("token_comm_dedup", "_DedupMetadataPushRouter", "_sort_token_routes_and_scatter"),
}
# The test of an `if` that admits valid routes only: `<expert> < expert_count` (not the padded expert count).
_VALID_ROUTE_TEST = re.compile(r"\S+ < (Int32\()?(self\.)?expert_count\)?")


def _guards(function: ast.FunctionDef, is_target) -> list[list[str]]:
    """For every node of ``function`` that ``is_target`` selects, the tests of the `if` bodies enclosing it."""
    found = []

    def visit(node: ast.AST, tests: list[str]) -> None:
        if is_target(node):
            found.append(tests)
        if isinstance(node, ast.If):
            visit(node.test, tests)
            for child in node.body:
                visit(child, tests + [ast.unparse(node.test)])
            for child in node.orelse:
                visit(child, tests)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, tests)

    visit(function, [])
    return found


def _is_histogram_atomic(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.arch.atomic_add"


def _is_sorted_store(node: ast.AST) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Subscript) and ast.unparse(target.value) == "sorted_elements"
        for target in node.targets
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", sorted(_ROUTERS))
def test_router_counts_and_sorts_valid_routes_only(kind: str) -> None:
    module, class_name, sort_name = _ROUTERS[kind]
    tree = ast.parse((_NVLINK / f"{module}.py").read_text())
    router = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {node.name: node for node in router.body if isinstance(node, ast.FunctionDef)}
    for method, is_target in (
        ("_build_histogram", _is_histogram_atomic),
        (sort_name, _is_sorted_store),
    ):
        guards = _guards(methods[method], is_target)
        assert guards, f"{class_name}.{method}: no histogram atomic / sorted-slab store found"
        for tests in guards:
            assert any(_VALID_ROUTE_TEST.fullmatch(test) for test in tests), (
                f"{class_name}.{method}: a histogram atomic or sorted-slab store is not limited to valid "
                f"routes (enclosing tests: {tests})"
            )


def _require_router_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the MegaMoE routers run on SM100, SM103 and SM107")
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("cutlass-dsl is not available")


class _RouterLaunch:
    """One compiled single-GPU router launch (world size 1) and the readback of its routing records."""

    token_padding_block = 128

    def __init__(self, kind, experts, topk, max_tokens, apply_scores, index_dtype):
        import importlib

        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute.typing import AddressSpace

        api = importlib.import_module(f"{_PACKAGE}.api")
        nvlink = f"{_PACKAGE}.communication.nvlink_domain"
        symmetric = importlib.import_module(f"{nvlink}.symmetric_buffer")
        workspace_module = importlib.import_module(f"{_PACKAGE}.helpers.device_workspace")
        module, class_name, _ = _ROUTERS[kind]
        router_class = getattr(importlib.import_module(f"{nvlink}.{module}"), class_name)

        self.dedup = kind == "dedup"
        self.experts, self.topk, self.apply_scores = experts, topk, apply_scores
        self.router = router_class(
            api.ProblemDesc(
                {
                    "world_size": 1,
                    "expert_count": experts,
                    "topk": topk,
                    "max_tokens_per_rank": max_tokens,
                    "apply_topk_at_fc1": apply_scores,
                }
            ),
            api.ImplDesc(
                {
                    "token_padding_block": self.token_padding_block,
                    "promised_launchable_sm_count": torch.cuda.get_device_properties(
                        0
                    ).multi_processor_count,
                }
            ),
        )
        self.workspace = workspace_module.DeviceWorkspace()
        self.router.register_device_workspace(self.workspace)
        self.workspace.finalize()
        self.buffers = {
            space: torch.zeros(
                max(16, self.workspace.total_bytes(space)), dtype=torch.uint8, device="cuda"
            )
            for space in ("local", "shared")
        }
        self.indices = torch.empty((max_tokens, topk), dtype=index_dtype, device="cuda")
        # The dedup router always reads top-k scores.
        self.scores = (
            torch.empty((max_tokens, topk), dtype=torch.float32, device="cuda")
            if apply_scores or self.dedup
            else None
        )

        def pointer(tensor):
            return cute.runtime.make_ptr(
                cutlass.Uint8, tensor.data_ptr(), AddressSpace.gmem, assumed_align=16
            )

        self.arguments = dict(
            topk_indices=from_dlpack(self.indices, assumed_align=16),
            topk_scores=None if self.scores is None else from_dlpack(self.scores, assumed_align=16),
            local_rank=0,
            local_workspace=pointer(self.buffers["local"]),
            shared_workspace=pointer(self.buffers["shared"]),
            peer_rank_ptr_mapper_host=symmetric.SymmetricBufferHost(
                self.buffers["shared"].data_ptr(), (0,), 0, 1
            ),
            device_workspace=self.workspace,
            stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        self.compiled = cute.compile(self.router.launch_router, **self.arguments)

    def _region(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        buffer = self.buffers[self.workspace.region(name).buffer_space]
        offset, size = self.workspace.offset(name), self.workspace.nbytes(name)
        return buffer.narrow(0, offset, size).view(dtype)

    def __call__(self, indices: torch.Tensor, scores: torch.Tensor) -> dict:
        """Route ``indices`` (padded to the launch rows with -1); per expert, its sorted routing records."""
        self.indices.copy_(indices)
        if self.scores is not None:
            self.scores.copy_(scores)
        for buffer in self.buffers.values():
            buffer.zero_()
        self.compiled(**self.arguments)
        torch.cuda.synchronize()

        sizes = self._region(self.router.sizes_region, torch.int32)[: self.experts].tolist()
        records = self._region(self.router.token_src_metadata_region, torch.int64).cpu()
        pooled_scores = None
        if self.apply_scores:
            pooled_scores = self._region(self.router.fc1_topk_scores_region, torch.int32).cpu()
        routes, base = {}, 0
        for expert, count in enumerate(sizes):
            segment = []
            for position in range(base, base + count):
                record = int(records[position])
                high = record >> 32
                segment.append(
                    (
                        record & 0xFFFF,  # source token
                        (record >> 16) & 0xFF,  # top-k slot
                        (record >> 24) & 0xFF,  # source rank
                        high & 0xFF,  # order among the token's routes on this rank
                        ((high >> 8) & 0xFF) + 1,  # the token's routes on this rank
                        None if pooled_scores is None else int(pooled_scores[position]),
                    )
                )
            routes[expert] = sorted(segment)
            base += -(-count // self.token_padding_block) * self.token_padding_block
        return routes


def _reference(indices, scores, experts, dedup, apply_scores) -> dict:
    """Per expert, the sorted records of the valid routes. The dedup router numbers a token's routes on one rank
    (here every route) in ascending expert order; the other router writes order 0 of 1."""
    score_bits = scores.contiguous().view(torch.int32) if apply_scores else None
    routes = {expert: [] for expert in range(experts)}
    for token, row in enumerate(indices.tolist()):
        valid = sorted((expert, slot) for slot, expert in enumerate(row) if 0 <= expert < experts)
        for order, (expert, slot) in enumerate(valid):
            routes[expert].append(
                (
                    token,
                    slot,
                    0,
                    order if dedup else 0,
                    len(valid) if dedup else 1,
                    None if score_bits is None else int(score_bits[token, slot]),
                )
            )
    return {expert: sorted(records) for expert, records in routes.items()}


@pytest.mark.parametrize(
    "kind,index_dtype,apply_scores,max_tokens",
    [
        ("plain", torch.int32, False, 512),  # one DATA CTA
        ("plain", torch.int64, True, 4096),  # several DATA CTAs, padded slots in every one
        ("dedup", torch.int32, True, 512),
        ("dedup", torch.int64, True, 4096),  # whole padded tokens fill the last DATA CTAs
    ],
)
def test_router_records_match_reference_with_padding(kind, index_dtype, apply_scores, max_tokens):
    _require_router_gpu()
    experts, topk = 64, 8
    generator = torch.Generator().manual_seed(2026)
    padded_launch = _RouterLaunch(kind, experts, topk, max_tokens, apply_scores, index_dtype)
    for live in (max_tokens, max_tokens // 4, 1, 0):
        live_indices = torch.rand((live, experts), generator=generator).topk(topk, dim=-1).indices
        if live > 1:
            live_indices[1, 0] = -1  # one invalid route inside a live token
        live_scores = torch.rand((live, topk), generator=generator)
        indices = torch.full((max_tokens, topk), -1, dtype=index_dtype)
        indices[:live] = live_indices.to(index_dtype)
        scores = torch.zeros((max_tokens, topk), dtype=torch.float32)
        scores[:live] = live_scores
        expected = _reference(indices[:live], scores[:live], experts, kind == "dedup", apply_scores)
        padded = padded_launch(indices.cuda(), scores.cuda())
        assert padded == expected, f"{kind} {index_dtype} launch rows {max_tokens}, live {live}"
        if 0 < live < max_tokens:
            unpadded_launch = _RouterLaunch(kind, experts, topk, live, apply_scores, index_dtype)
            unpadded = unpadded_launch(indices[:live].cuda(), scores[:live].cuda())
            assert unpadded == padded, (
                f"{kind} {index_dtype} live {live}: padding changed the records"
            )
