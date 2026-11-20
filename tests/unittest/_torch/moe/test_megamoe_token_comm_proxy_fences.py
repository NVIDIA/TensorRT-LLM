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
"""CPU contract test for the proxy ordering of MegaMoE token-back.

TMA loads read memory through the async proxy; the FC2 epilogue writes the rows that
``fc2_done`` publishes with ordinary stores (the data rows on the non-bulk store path, the
scale rows on every path) and releases ``fc2_done`` through the generic proxy. The PTX memory
model orders accesses of two proxies only across a proxy fence, so the acquire of
``fc2_done`` alone does not order those rows before token-back's TMA read of them: every
thread that acquires ``fc2_done`` issues ``fence.proxy.async.global`` before the read.

Token-in has the same order to keep: after ``metadata_ready`` it pulls the peers' rows with TMA,
rows written with generic stores (the one-round plane rows, or the peers' staged inputs), so it
issues the fence after that wait on every dispatch path.

The check reads the sources only (no GPU, no import of the kernels).
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_NVLINK_DOMAIN = (
    _ROOT / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain"
)
_FLAG = "fc2_done"
_GLOBAL_FENCE = "cute.arch.fence_proxy('async.global')"
# The methods that wait for fc2_done, per file: a wait that moves elsewhere has to be listed here.
_WAITING_METHODS = {
    "token_comm.py": {"_token_back_multi_stage", "token_back"},
    "token_comm_dedup.py": {"token_back"},
}


def _acquires_flag(loop: ast.While) -> bool:
    """Whether this spin loop itself acquires fc2_done (the root name of the loaded address)."""
    for node in ast.walk(loop):
        if node is not loop and isinstance(node, ast.While):
            return False
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
            return ast.unparse(address) == _FLAG
    return False


def _fc2_done_waits(source: str) -> list[tuple[str, int, list[str]]]:
    """Each fc2_done wait as (method, line, the two statements after it)."""
    waits = []
    for function in ast.walk(ast.parse((_NVLINK_DOMAIN / source).read_text())):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            for field in ("body", "orelse"):
                body = getattr(node, field, None)
                if not isinstance(body, list):
                    continue
                for index, statement in enumerate(body):
                    if isinstance(statement, ast.While) and _acquires_flag(statement):
                        after = [ast.unparse(s) for s in body[index + 1 : index + 3]]
                        waits.append((function.name, statement.lineno, after))
    return waits


@pytest.mark.parametrize("source", sorted(_WAITING_METHODS))
def test_fc2_done_waits_are_where_the_contract_expects(source: str) -> None:
    methods = {method for method, _, _ in _fc2_done_waits(source)}

    assert methods == _WAITING_METHODS[source], (
        f"{source}: fc2_done is waited for in {sorted(methods)}"
    )


@pytest.mark.parametrize("source", sorted(_WAITING_METHODS))
def test_tma_reads_of_fc2_rows_follow_a_proxy_fence(source: str) -> None:
    unfenced = []
    for method, lineno, after in _fc2_done_waits(source):
        # A warp barrier may first order the lanes that acquired; the fence then orders the TMA after them.
        if after[:1] == ["cute.arch.sync_warp()"]:
            after = after[1:]
        if after[:1] != [_GLOBAL_FENCE]:
            unfenced.append(f"{source}:{lineno} {method} reads the FC2 rows with TMA unfenced")
    assert not unfenced, "\n".join(unfenced)


def _token_in_metadata_waits(source: str) -> list[tuple[int, list[str]]]:
    """Each wait for metadata_ready in the source's token_in as (line, the statement after it)."""
    waits = []
    tree = ast.parse((_NVLINK_DOMAIN / source).read_text())
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name != "token_in":
            continue
        for node in ast.walk(function):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                call = statement.value if isinstance(statement, ast.Expr) else None
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "wait_for_metadata_ready"
                ):
                    after = [ast.unparse(s) for s in body[index + 1 : index + 2]]
                    waits.append((statement.lineno, after))
    return waits


@pytest.mark.parametrize("source", sorted(_WAITING_METHODS))
def test_token_in_fences_the_rows_metadata_ready_publishes(source: str) -> None:
    waits = _token_in_metadata_waits(source)
    assert waits, f"{source}: token_in no longer waits for metadata_ready"
    unfenced = [
        f"{source}:{lineno} token_in pulls the peers' rows with TMA unfenced"
        for lineno, after in waits
        if after != ["fence_proxy_async_global()"]
    ]
    assert not unfenced, "\n".join(unfenced)
