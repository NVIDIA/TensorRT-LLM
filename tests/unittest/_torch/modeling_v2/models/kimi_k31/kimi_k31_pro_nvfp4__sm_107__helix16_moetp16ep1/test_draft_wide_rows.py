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
"""``draft_linear``'s Rubin rows at three and four requests' blocks (``_DRAFT_WIDE_RUBIN``).

The table only holds 24 and 32 rows of the drafter's served projection shapes. With a row (the
table set and the Rubin rows forced on any SM 10.x device), 24 / 32 rows run that kernel bit for bit
and within BF16 rounding of the FP64 product, a CUDA graph replays the eager bits, and a capture
before the engine is prepared keeps cuBLAS. Every other row count (8, 16 and the counts between)
runs what it runs without the table, bit for bit; so does a device that does not take the Rubin
rows. The empty table changes nothing.
"""

import importlib

import pytest
import torch

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
)
#: The drafter's projections above one request's block on the served headtp path: conv_proj and
#: gate_up / 16, [q_a | kv_a], the absorbed query (6 heads), o_proj (6 heads), down / 16, and the
#: five layers' stacked history kv_a.
SHAPES = ((1792, 7168), (2112, 7168), (3456, 1536), (7168, 768), (7168, 896), (2880, 7168))
#: The shape the kernel tests run on ([q_a | kv_a]) and one choice of each kind a row may name there, at
#: both row counts (token tile 32: one token CTA).
SHAPE = (2112, 7168)
CHOICES = (
    ("warp", 16, 32, 256, 6, 2),
    ("warp", 32, 32, 128, 4, 2),
    ("splitk", 64, 32, 4, 6),
    ("splitk", 128, 32, 1, 4),
    ("proj", 64, 32, 7, 6),
)


def _sm100_family():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _sm100_family(), reason="needs an SM 10.x GPU")


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(_PKG + ".modeling")


def _w(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (0.02 * torch.randn(*shape, generator=g, device="cuda")).bfloat16()


def _x(rows, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, k, generator=g, device="cuda").bfloat16()


def _bits(t):
    return t.view(torch.int16)


def _close(got, x, weight):
    ref = x.double() @ weight.double().t()
    err = (got.double() - ref).abs().max() / ref.abs().max()
    return err.item() < 1e-2


@pytest.fixture
def rows_table(mod, monkeypatch):
    """Set ``_DRAFT_WIDE_RUBIN`` to the given table with the Rubin rows forced on."""

    def use(table):
        monkeypatch.setattr(mod, "_DRAFT_WIDE_RUBIN", table)
        monkeypatch.setattr(mod, "_rubin_rows", lambda device: True)

    return use


def test_the_table_holds_only_wide_rows_of_served_shapes(mod):
    assert mod._DRAFT_WIDE_ROWS == (24, 32)
    for shape, rows in mod._DRAFT_WIDE_RUBIN.items():
        assert shape in SHAPES, shape
        assert set(rows) <= set(mod._DRAFT_WIDE_ROWS), (shape, rows)


@pytest.mark.parametrize("rows", [24, 32])
def test_empty_table_keeps_cublas(mod, monkeypatch, rows):
    monkeypatch.setattr(mod, "_DRAFT_WIDE_RUBIN", {})
    monkeypatch.setattr(mod, "_rubin_rows", lambda device: True)
    for i, shape in enumerate(SHAPES):
        w, x = _w(shape, i), _x(rows, shape[1], 10 + i)
        assert mod._draft_wide_choice(x, w) is None
        got = mod.draft_linear(x, w)
        assert torch.equal(_bits(got), _bits(mod.rubin_linear(x, w, False))), shape


@pytest.mark.parametrize("choice", CHOICES, ids=str)
@pytest.mark.parametrize("rows", [24, 32])
def test_rows_run_their_kernel(mod, rows_table, choice, rows):
    shape = SHAPE
    w, x = _w(shape, 1), _x(rows, shape[1], rows)
    rows_table({shape: {24: choice, 32: choice}})
    assert mod._draft_wide_choice(x, w) == choice
    got = mod.draft_linear(x, w)
    want = mod._draft_wide_gemm(x, w, choice)
    assert torch.equal(_bits(got), _bits(want))
    assert _close(got, x, w)


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 15, 16, 17, 23, 25, 31])
def test_other_rows_run_what_they_ran(mod, rows_table, rows):
    tables = {shape: {24: CHOICES[0], 32: CHOICES[2]} for shape in SHAPES}
    for i, shape in enumerate(SHAPES):
        w, x = _w(shape, 20 + i), _x(rows, shape[1], 30 + i)
        rows_table({})
        want = mod.draft_linear(x, w)
        rows_table(tables)
        assert mod._draft_wide_choice(x, w) is None
        got = mod.draft_linear(x, w)
        assert torch.equal(_bits(got), _bits(want)), (shape, rows)


def test_a_device_without_rubin_rows_keeps_cublas(mod, monkeypatch):
    shape = (1792, 7168)
    monkeypatch.setattr(mod, "_DRAFT_WIDE_RUBIN", {shape: {32: CHOICES[0]}})
    monkeypatch.setattr(mod, "_rubin_rows", lambda device: False)
    w, x = _w(shape, 2), _x(32, shape[1], 3)
    assert mod._draft_wide_choice(x, w) is None
    got = mod.draft_linear(x, w)
    assert torch.equal(_bits(got), _bits(mod.rubin_linear(x, w, False)))


def test_unpacked_rows_keep_cublas(mod, rows_table):
    shape = (1792, 7168)
    rows_table({shape: {32: CHOICES[0]}})
    w = _w(shape, 4)
    z = torch.empty(32, 7168 + 8, dtype=torch.bfloat16, device="cuda")[:, :7168]
    z.copy_(_x(32, 7168, 5))
    assert mod._draft_wide_choice(z, w) is None


@pytest.mark.parametrize("choice", [CHOICES[0], CHOICES[2], CHOICES[4]], ids=str)
def test_graph_replay_matches_eager(mod, rows_table, choice):
    shape = SHAPE
    rows_table({shape: {32: choice}})
    w, x = _w(shape, 6), _x(32, shape[1], 7)
    mod.draft_linear(x, w)  # prepares the engine / projection outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mod.draft_linear(x, w)
    for seed in (11, 12, 13):
        x.copy_(_x(32, shape[1], seed))
        graph.replay()
        torch.cuda.synchronize()
        eager = mod.draft_linear(x, w)
        assert torch.equal(_bits(out), _bits(eager))


@pytest.mark.parametrize("choice", [CHOICES[0], CHOICES[4]], ids=str)
def test_capture_before_the_engine_keeps_cublas(mod, rows_table, choice):
    shape = (2880, 7168)
    rows_table({shape: {24: choice}})
    w, x = _w(shape, 8), _x(24, shape[1], 9)  # a fresh weight: no engine yet
    want = mod.rubin_linear(x, w, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mod.draft_linear(x, w)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(want))


def test_served_wide_head_and_indexer_tactics(mod, monkeypatch):
    """The Rubin head and DSA indexer front select their measured wide-row tactics."""
    monkeypatch.setattr(mod, "_rubin_rows", lambda device: True)
    device = torch.device("cuda")
    head = _w((10240, 7168), 40)
    for rows, tactic in (
        (8, mod._HEAD_GEMM_ROWS_RUBIN[0][1]),
        (9, ("warp", 32, 16, 256, 4, 2)),
        (16, ("warp", 32, 16, 256, 4, 2)),
        (17, ("warp", 32, 32, 128, 6, 2)),
        (32, ("warp", 32, 32, 128, 6, 2)),
        (33, None),
    ):
        assert mod._head_gemm_choice(_x(rows, 7168, 4), head) == tactic, rows
    table = mod.front_table(device)
    for rows in (17, 24, 32):
        assert mod.tactic_for(rows, table) == (128, 16, 3, 4)
    assert mod.tactic_for(33, table) is None
