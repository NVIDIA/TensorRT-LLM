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
"""CPU contract tests for the proxy ordering of the MegaMoE NVLink token communication.

Bulk copies and TMA loads access memory through the async proxy; ordinary loads, stores
and the flags the CTAs and ranks synchronize on use the generic proxy. Data that one
proxy wrote and the other reads needs a proxy fence in between:

- Publication: each router CTA writes its per-expert counts to shared memory with
  ordinary stores and copies them to every peer with a bulk copy. Every writing thread
  issues ``fence.proxy.async.shared::cta`` and the CTA synchronizes before the elected
  thread issues the copy; without the fence a peer can receive counts that miss some of
  the writes.
- Consumption: a thread that acquires a flag and then reads the data it publishes with
  TMA issues ``fence.proxy.async.global`` between the two. The acquire orders the data
  for the generic proxy only.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_NVLINK_DOMAIN = (
    _ROOT / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain"
)
_SOURCES = ["token_comm.py", "token_comm_dedup.py", "token_comm_deterministic.py"]
_PUBLISH = "_broadcast_sizes_to_peers"
_PREAMBLE = [
    "cute.arch.fence_proxy('async.shared', space='cta')",
    "cute.arch.sync_threads()",
]
_GLOBAL_FENCE = "cute.arch.fence_proxy('async.global')"
_WAIT_HELPERS = ("wait_for_sizes_ready", "wait_for_metadata_ready")

# Every acquire wait of the three files, as (method, flag): the waits after which the method reads the published
# data with TMA, and the waits after which it reads them with ordinary loads only (and the wait helpers themselves).
# A new wait has to be sorted into one of them.
_TMA_READ_WAITS = {
    ("_compute_push_tables", "sizes_ready"),  # the peers' rows of the size matrix
    ("token_in", "wait_for_metadata_ready"),  # the peers' token rows
    ("_token_back_multi_stage", "fc2_done"),  # the FC2 rows of a local expert
    ("token_back", "fc2_done"),
}
_GENERIC_READ_WAITS = {
    ("_router_multiple_ctas", "source_base_ready"),
    ("_router_helper_multiple_ctas", "histogram_done"),
    ("_router_push_metadata", "sorted_metadata_ready"),
    ("_router_push_metadata", "push_table_ready"),
    ("token_in", "wait_for_sizes_ready"),
    ("token_back", "wait_for_sizes_ready"),
    ("token_back", "wait_for_metadata_ready"),
    ("wait_for_sizes_ready", "sizes_ready"),
    ("wait_for_sizes_ready", "wait_for_sizes_ready"),
    ("wait_for_metadata_ready", "metadata_ready"),
}


def _publishes_counts(statement: ast.stmt) -> bool:
    value = getattr(statement, "value", None)
    return (
        isinstance(statement, (ast.Assign, ast.Expr))
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == _PUBLISH
    )


def _bodies(tree: ast.AST):
    for node in ast.walk(tree):
        for field in ("body", "orelse"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                yield body


def _publication_preambles(source: Path) -> list[tuple[int, list[str]]]:
    """Each statement that publishes counts, with the two statements before it."""
    sites = []
    for body in _bodies(ast.parse(source.read_text())):
        for index, statement in enumerate(body):
            if _publishes_counts(statement):
                preamble = [ast.unparse(s) for s in body[max(0, index - 2) : index]]
                sites.append((statement.lineno, preamble))
    return sites


def _acquired_flag(loop: ast.While) -> str | None:
    """The flag a spin loop acquires (the root name of the loaded address), unless a nested loop does."""
    for node in ast.walk(loop):
        if node is not loop and isinstance(node, ast.While):
            return None
    for node in ast.walk(loop):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and any(
                k.arg == "sem" and isinstance(k.value, ast.Constant) and k.value.value == "acquire"
                for k in node.keywords
            )
        ):
            address = node.args[0]
            while isinstance(address, ast.BinOp):
                address = address.left
            return ast.unparse(address)
    return None


def _helper_wait(statement: ast.stmt) -> str | None:
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr in _WAIT_HELPERS
    ):
        return statement.value.func.attr
    return None


def _acquire_waits(source: Path) -> list[tuple[str, str, int, list[str]]]:
    """Each wait as (method, flag, line, the two statements after it)."""
    waits = []
    for function in ast.walk(ast.parse(source.read_text())):
        if not isinstance(function, ast.FunctionDef):
            continue
        for body in _bodies(function):
            for index, statement in enumerate(body):
                flag = _helper_wait(statement)
                if flag is None and isinstance(statement, ast.While):
                    flag = _acquired_flag(statement)
                if flag is not None:
                    after = [ast.unparse(s) for s in body[index + 1 : index + 3]]
                    waits.append((function.name, flag, statement.lineno, after))
    return waits


@pytest.mark.parametrize("source", _SOURCES)
def test_router_counts_reach_the_async_proxy_before_publication(source: str) -> None:
    sites = _publication_preambles(_NVLINK_DOMAIN / source)

    # The single-CTA router publishes from its data CTA, the multi-CTA one from its helper.
    assert len(sites) == 2
    for lineno, preamble in sites:
        assert preamble == _PREAMBLE, f"{source}:{lineno} publishes counts without the fence"


@pytest.mark.parametrize("source", _SOURCES)
def test_tma_reads_of_published_data_follow_a_proxy_fence(source: str) -> None:
    waits = _acquire_waits(_NVLINK_DOMAIN / source)

    unsorted = {(m, f) for m, f, _, _ in waits} - _TMA_READ_WAITS - _GENERIC_READ_WAITS
    assert not unsorted, f"{source}: sort the new acquire waits {sorted(unsorted)}"
    tma_waits = [w for w in waits if (w[0], w[1]) in _TMA_READ_WAITS]
    # Every file has the size matrix and the token rows; token_back reads the FC2 rows in each.
    assert {(m, f) for m, f, _, _ in tma_waits} >= {
        ("_compute_push_tables", "sizes_ready"),
        ("token_in", "wait_for_metadata_ready"),
        ("token_back", "fc2_done"),
    }
    for method, flag, lineno, after in tma_waits:
        # A warp barrier may first order the lanes that acquired; the fence then orders the TMA after them.
        if after[:1] == ["cute.arch.sync_warp()"]:
            after = after[1:]
        assert after[:1] == [_GLOBAL_FENCE], (
            f"{source}:{lineno} {method} reads {flag}'s data with TMA unfenced"
        )
