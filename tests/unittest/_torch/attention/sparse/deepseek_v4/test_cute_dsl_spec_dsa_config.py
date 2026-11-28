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
"""specDSA configuration: the spec table, its validation, the support checks and the dispatch table of the op.

Runs without a GPU: the checks refuse an unsupported spec, an unknown or incapable implementation and a missing
production logits op before any device work. Only the last test needs CuTe DSL (not a GPU) and skips without it.
"""

import dataclasses
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import config as C
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import (
    hybrid_op,
    large_batch_op,
    megakernel_op,
    op,
    tail_op,
    topk_multicta_op,
)

FLASH = C.DS_V4_FLASH
FLASH_FP4 = C.DS_V4_FLASH_FP4
# An indexer shape without kernels: top-1024 instead of top-512.
PRO = dataclasses.replace(FLASH, name="DS-V4-Pro", topk=1024)


def test_ds_v4_flash_values():
    """The numbers the kernels and the tests were validated with."""
    s = C.DS_V4_FLASH
    assert s.name == "DS-V4-Flash"
    assert (s.index_heads, s.index_head_dim, s.index_k_dtype) == (64, 128, "fp8")
    assert (s.index_page_tokens, s.compress_ratio, s.topk, s.max_index_len) == (32, 4, 512, 1 << 18)
    assert (s.attn_heads, s.kv_dim, s.kv_format, s.kv_page_rows) == (64, 512, "v4_e4m3_rows", 32)
    assert (s.window, s.swa_page_rows, s.has_sink) == (128, 128, True)
    assert (s.megakernel_max_batch, s.large_batch, s.max_batch) == (47, True, 1024)
    assert s.tail_hybrid_range is None
    assert s.shape_key == (64, 128, 512, 4)
    assert (s.min_index_len, s.selected_rows, s.pages_per_tile) == (512, 640, 4)
    assert s.index_page_bytes == 32 * (128 + 4)
    assert s.default_softmax_scale == 1.0 / math.sqrt(512.0)
    assert (C.NUM_SMS, C.INDEX_TILE) == (148, 128)
    assert C.SUPPORTED_SPECS == {"DS-V4-Flash": s, "DS-V4-Flash-FP4": C.DS_V4_FLASH_FP4}


def test_ds_v4_flash_fp4_values():
    """The same model with the MXFP4 indexer K cache: only the K dtype (and the batch sizes of its kernels) differ."""
    s = C.DS_V4_FLASH_FP4
    assert s.name == "DS-V4-Flash-FP4"
    assert (
        dataclasses.replace(
            s,
            name=FLASH.name,
            index_k_dtype="fp8",
            megakernel_max_batch=FLASH.megakernel_max_batch,
            tail_hybrid_range=None,
        )
        == FLASH
    )
    assert (s.index_k_dtype, s.large_batch, s.max_batch) == ("fp4", True, 1024)
    # The FP4 K stream is twice as fast per byte, the megakernel row cost is not: the large-batch kernels are faster
    # from 21 rows (nsys, megakernel against large batch at 17 to 28 rows). The megakernel stays able to serve 20 rows.
    assert s.megakernel_max_batch == 20
    # The tail-attention hybrid serves 5 to 26 rows: the clusters of its two kernels have to fit one wave of the SMs.
    assert s.tail_hybrid_range == (5, 26)
    assert s.index_page_bytes == 32 * (128 // 2 + 128 // 32)
    assert s.shape_key == FLASH.shape_key and s.kernel_key != FLASH.kernel_key


def test_index_page_bytes():
    """The token bytes of the cache manager of the DeepSeek-V4 attention backend times the tokens of a page."""
    assert FLASH.index_page_bytes == 32 * (
        128 + 4
    )  # E4M3 values, then one float32 scale per 128 elements
    assert dataclasses.replace(FLASH, index_head_dim=256).index_page_bytes == 32 * (256 + 2 * 4)
    fp4 = dataclasses.replace(FLASH, index_k_dtype="fp4")
    assert fp4.index_page_bytes == 32 * (
        128 // 2 + 128 // 32
    )  # packed E2M1 plus one UE8M0 per 32 elements


def test_known_shapes_match_the_adapter_table():
    path = Path(C.__file__).parents[1] / "throwaway_topk" / "adapter" / "indexer_config.py"
    if not path.exists():
        pytest.skip("the top-K adapter's model table is not in this tree")
    modspec = importlib.util.spec_from_file_location("indexer_config_under_test", path)
    table = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(table)
    for name, shape in C.KNOWN_SHAPES.items():
        assert table.MODEL_CONFIGS[name] == shape
    assert FLASH.shape_key == table.MODEL_CONFIGS[FLASH.name]


def test_frozen_and_hashable():
    assert len({FLASH, dataclasses.replace(FLASH), PRO}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        FLASH.topk = 1024


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"name": ""}, "name must not be empty"),
        ({"topk": 0}, "topk must be a positive int"),
        ({"index_heads": -1}, "index_heads must be a positive int"),
        ({"topk": 512.0}, "topk must be a positive int"),
        ({"index_k_dtype": "int8"}, "index_k_dtype"),
        ({"kv_format": "mla"}, "kv_format"),
        ({"index_head_dim": 100}, "multiple of 128"),
        ({"index_k_dtype": "fp4", "index_head_dim": 100}, "multiple of 32"),
        ({"topk": (1 << 18) + 32}, "exceeds max_index_len"),
        ({"max_index_len": (1 << 18) + 1}, "multiple of index_page_tokens"),
        ({"index_page_tokens": 256}, "divide the indexer tile"),
        ({"megakernel_max_batch": -1}, "megakernel_max_batch"),
        ({"megakernel_max_batch": 2048}, "exceeds max_batch"),
        ({"large_batch": False}, "need the large-batch kernels"),
        ({"tail_hybrid_range": (0, 8)}, "1 <= first <= last <= max_batch"),
        ({"tail_hybrid_range": (9, 8)}, "1 <= first <= last <= max_batch"),
        ({"tail_hybrid_range": (5, 1025)}, "1 <= first <= last <= max_batch"),
        ({"tail_hybrid_range": (5,)}, "tuple .first, last. of ints"),
        ({"tail_hybrid_range": [5, 8]}, "tuple .first, last. of ints"),
        ({"tail_hybrid_range": (5.0, 8)}, "tuple .first, last. of ints"),
        ({"tail_hybrid_range": 8}, "tuple .first, last. of ints"),
    ],
)
def test_validation(changes, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(FLASH, **changes)


def test_kernel_key_covers_every_structural_field(monkeypatch):
    """A field the kernels depend on must change ``kernel_key``; the label and the batch-size limits must not."""
    monkeypatch.setattr(C, "KV_FORMATS", ("v4_e4m3_rows", "other"))
    other_value = {
        "index_heads": 32,
        "index_head_dim": 256,
        "index_k_dtype": "fp4",
        "index_page_tokens": 16,
        "compress_ratio": 1,
        "topk": 1024,
        "max_index_len": 1 << 17,
        "attn_heads": 32,
        "kv_dim": 576,
        "kv_format": "other",
        "kv_page_rows": 64,
        "window": 64,
        "swa_page_rows": 64,
        "has_sink": False,
    }
    not_structural = {
        "name",
        "megakernel_max_batch",
        "large_batch",
        "max_batch",
        "tail_hybrid_range",
    }
    assert {f.name for f in dataclasses.fields(C.SpecDsaConfig)} == not_structural | set(
        other_value
    )
    for field, value in other_value.items():
        assert dataclasses.replace(FLASH, **{field: value}).kernel_key != FLASH.kernel_key, field
    relabeled = dataclasses.replace(FLASH, name="alias", megakernel_max_batch=8, max_batch=256)
    assert relabeled.kernel_key == FLASH.kernel_key
    # the hybrid range is host-side dispatch data: no kernel_key, and so no compile or engine cache key of the
    # kernels, changes with it (those are keyed by ``kernel_key``; the hybrid's kernels by cluster size and variant)
    for rng in (None, (1, 1), (5, 26), (8, 12)):
        assert (
            dataclasses.replace(FLASH_FP4, tail_hybrid_range=rng).kernel_key == FLASH_FP4.kernel_key
        )
        assert dataclasses.replace(FLASH, tail_hybrid_range=rng).kernel_key == FLASH.kernel_key


def test_supported_spec():
    assert C.supported_spec() is FLASH
    assert C.supported_spec(FLASH) is FLASH
    # lower batch-size limits are served: more rows go to the large-batch kernels, or fewer rows are accepted
    for lower in (
        dataclasses.replace(FLASH, name="small", megakernel_max_batch=8, max_batch=64),
        dataclasses.replace(FLASH, megakernel_max_batch=0),
        dataclasses.replace(FLASH, large_batch=False, max_batch=47),
    ):
        assert C.supported_spec(lower) is lower
    for higher in (
        dataclasses.replace(FLASH, max_batch=2048),
        dataclasses.replace(FLASH, megakernel_max_batch=64),
    ):
        with pytest.raises(NotImplementedError, match="batch-size limits"):
            C.supported_spec(higher)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        C.supported_spec(PRO)


def test_supported_spec_checks_the_tail_hybrid_range():
    """A narrower range, or none, is served (the other batch sizes go to the other implementations); a range that
    covers a batch size the spec's does not, or any range of a spec without one, is not."""
    for inside in (None, (5, 26), (5, 5), (8, 20), (26, 26)):
        config = dataclasses.replace(FLASH_FP4, tail_hybrid_range=inside)
        assert C.supported_spec(config) is config
    for outside in ((4, 26), (1, 8), (5, 27), (27, 40), (1, 1024)):
        with pytest.raises(NotImplementedError, match="tail-hybrid range"):
            C.supported_spec(dataclasses.replace(FLASH_FP4, tail_hybrid_range=outside))
    with pytest.raises(NotImplementedError, match="tail-hybrid range"):
        C.supported_spec(dataclasses.replace(FLASH, tail_hybrid_range=(5, 26)))


def test_resolve_spec():
    assert C.resolve_spec(64, 128, 512, 4) is FLASH
    assert C.resolve_spec(64, 128, 512, 4, index_k_dtype="fp8", kv_format="v4_e4m3_rows") is FLASH
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        C.resolve_spec(64, 128, 1024, 4)
    with pytest.raises(NotImplementedError, match="DS-V3.2"):
        C.resolve_spec(64, 128, 2048, 1)
    assert C.resolve_spec(64, 128, 512, 4, index_k_dtype="fp4") is FLASH_FP4
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        C.resolve_spec(64, 128, 1024, 4, index_k_dtype="fp4")
    with pytest.raises(NotImplementedError, match="specDSA has no kernels"):
        C.resolve_spec(32, 128, 512, 4)  # another model with the same top-K


def test_require_built_for():
    assert C.require_built_for(None, FLASH) is FLASH
    alias = dataclasses.replace(FLASH, name="alias")
    assert C.require_built_for(alias, FLASH) is alias
    with pytest.raises(NotImplementedError, match="specialized for 'DS-V4-Flash'"):
        C.require_built_for(PRO, FLASH)


def _default_ranges(spec):
    """Batch sizes each implementation serves by default for ``spec``, as the text of the README table ("none" when
    empty). Each one has to be a single range."""
    served = {name: [] for name in op.IMPLEMENTATIONS}
    for batch in range(1, spec.max_batch + 1):
        served[op.implementation_for(batch, spec)].append(batch)
    cells = {}
    for name, sizes in served.items():
        assert sizes == list(range(sizes[0], sizes[-1] + 1)) if sizes else True, (spec.name, name)
        cells[name] = f"{sizes[0]} to {sizes[-1]:,} rows" if sizes else "none"
    return cells


def test_readme_table_lists_the_batch_sizes_of_every_spec():
    """The README's spec table (megakernel, tail hybrid, large-batch ranges) is the dispatch of ``SUPPORTED_SPECS``."""
    readme = (Path(C.__file__).parent / "README.md").read_text()
    for spec in C.SUPPORTED_SPECS.values():
        rows = [ln for ln in readme.splitlines() if ln.startswith(f"| {spec.name} |")]
        assert len(rows) == 1, f"{spec.name}: no row in the README table"
        cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        ranges = _default_ranges(spec)
        assert cells[-3] == ranges["megakernel"], spec.name
        assert cells[-2] == ranges["tail_hybrid"], spec.name
        assert cells[-1] == ranges["large_batch"], spec.name


def test_op_routes_batch_sizes_by_spec():
    assert op.implementation_for(1) == "megakernel"
    assert op.implementation_for(47) == "megakernel"
    assert op.implementation_for(48) == "large_batch"
    assert op.implementation_for(1024) == "large_batch"
    for bad in (-1, 0, 1025):
        with pytest.raises(ValueError, match="batch size"):
            op.implementation_for(bad)
    small = dataclasses.replace(FLASH, megakernel_max_batch=8)
    assert (op.implementation_for(8, small), op.implementation_for(9, small)) == (
        "megakernel",
        "large_batch",
    )
    assert (
        op.implementation_for(1, dataclasses.replace(FLASH, megakernel_max_batch=0))
        == "large_batch"
    )
    no_large = dataclasses.replace(FLASH, large_batch=False, max_batch=47)
    assert op.implementation_for(47, no_large) == "megakernel"
    with pytest.raises(ValueError, match="batch size 48"):
        op.implementation_for(48, no_large)
    assert list(megakernel_op.supported_batch_sizes()) == list(range(1, 48))
    assert list(megakernel_op.supported_batch_sizes(small)) == list(range(1, 9))


def test_op_routes_fp4_batch_sizes():
    """The dispatch table of the FP4 spec: the megakernel for 1 to 4 rows, the tail hybrid for 5 to 26, the large-batch
    kernels from 27 rows. The megakernel still serves up to 20 rows when asked for."""
    assert FLASH_FP4.megakernel_max_batch == 20 and FLASH_FP4.tail_hybrid_range == (5, 26)
    for batch in range(1, FLASH_FP4.max_batch + 1):
        expected = "megakernel" if batch <= 4 else ("tail_hybrid" if batch <= 26 else "large_batch")
        assert op.implementation_for(batch, FLASH_FP4) == expected, batch
    assert list(megakernel_op.supported_batch_sizes(FLASH_FP4)) == list(range(1, 21))
    assert list(hybrid_op.supported_batch_sizes(FLASH_FP4)) == list(range(5, 27))
    assert list(hybrid_op.supported_batch_sizes()) == list(range(5, 27))
    # the FP8 spec keeps its own switch and has no hybrid
    assert op.implementation_for(47, FLASH) == "megakernel"
    assert op.implementation_for(48, FLASH) == "large_batch"
    assert "tail_hybrid" not in {op.implementation_for(b, FLASH) for b in range(1, 1025)}
    assert list(hybrid_op.supported_batch_sizes(FLASH)) == []


def test_op_routes_a_custom_hybrid_range():
    """The hybrid takes the batch sizes of its range out of the megakernel's and the large-batch kernels' ranges."""
    narrow = dataclasses.replace(FLASH_FP4, tail_hybrid_range=(8, 12))
    got = [op.implementation_for(b, narrow) for b in (1, 7, 8, 12, 13, 20, 21, 1024)]
    assert (
        got == ["megakernel"] * 2 + ["tail_hybrid"] * 2 + ["megakernel"] * 2 + ["large_batch"] * 2
    )
    none = dataclasses.replace(FLASH_FP4, tail_hybrid_range=None)
    assert [op.implementation_for(b, none) for b in (5, 20, 21)] == [
        "megakernel",
        "megakernel",
        "large_batch",
    ]
    # a range above the megakernel's last size and inside the large-batch kernels'
    above = dataclasses.replace(FLASH_FP4, tail_hybrid_range=(22, 24))
    assert [op.implementation_for(b, above) for b in (20, 21, 22, 24, 25)] == [
        "megakernel",
        "large_batch",
        "tail_hybrid",
        "tail_hybrid",
        "large_batch",
    ]


def test_tail_hybrid_range_is_inside_the_limits_of_its_kernels():
    """Both kernels of the hybrid serve every batch size of the range (their clusters fit one wave of the SMs)."""
    first, last = FLASH_FP4.tail_hybrid_range
    for batch in range(first, last + 1):
        assert tail_op.is_batch_size_supported(batch)[0], batch
        assert topk_multicta_op.is_batch_size_supported(batch)[0], batch
    assert last <= min(tail_op.TAIL_MAX_BATCH, topk_multicta_op.MAX_BATCH)


def _prepare_args(batch):
    i32 = torch.int32
    return (
        torch.full((batch,), 4096, dtype=i32),
        torch.full((batch,), 16384, dtype=i32),
        torch.zeros((batch, 128), dtype=i32),
        torch.zeros(64),
    )


@pytest.mark.parametrize(
    "spec, batch, implementation, match",
    [
        (FLASH_FP4, 8, "megakernal", "implementation must be one of"),
        (
            FLASH_FP4,
            21,
            "megakernel",
            "megakernel implementation of DS-V4-Flash-FP4 serves 1 to 20 rows",
        ),
        (
            FLASH_FP4,
            4,
            "tail_hybrid",
            "tail_hybrid implementation of DS-V4-Flash-FP4 serves 5 to 26 rows",
        ),
        (
            FLASH_FP4,
            27,
            "tail_hybrid",
            "tail_hybrid implementation of DS-V4-Flash-FP4 serves 5 to 26 rows",
        ),
        (FLASH, 8, "tail_hybrid", "DS-V4-Flash has no tail_hybrid implementation"),
        (FLASH, 48, "megakernel", "megakernel implementation of DS-V4-Flash serves 1 to 47 rows"),
        (
            dataclasses.replace(FLASH, megakernel_max_batch=0),
            8,
            "megakernel",
            "has no megakernel implementation",
        ),
        (
            dataclasses.replace(FLASH, large_batch=False, max_batch=47),
            8,
            "large_batch",
            "has no large_batch implementation",
        ),
        (FLASH_FP4, 1025, "large_batch", "batch size 1025"),
    ],
)
def test_forced_implementation_is_refused_before_any_device_work(
    spec, batch, implementation, match
):
    """A forced implementation that cannot serve the batch size is a ValueError of ``prepare`` and ``precompile``,
    raised before any device work (these run on the CPU)."""
    with pytest.raises(ValueError, match=match):
        op.prepare(*_prepare_args(batch), config=spec, implementation=implementation)
    with pytest.raises(ValueError, match=match):
        op.precompile(torch.device("cpu"), batch, spec, implementation=implementation)


def test_forced_implementation_that_can_serve_the_batch_size_is_accepted():
    """The megakernel at 5 to 20 FP4 rows and the large-batch kernels below their default range are selectable;
    what a forced request resolves to is checked on the GPU (``test_cute_dsl_spec_dsa_op.py``)."""
    for batch, implementation in (
        (5, "megakernel"),
        (20, "megakernel"),
        (5, "large_batch"),
        (21, "large_batch"),
        (26, "large_batch"),
        (5, "tail_hybrid"),
        (26, "tail_hybrid"),
    ):
        assert op._select(batch, FLASH_FP4, implementation) == implementation
    assert op._select(5, FLASH_FP4, None) == "tail_hybrid"
    assert op._select(5, FLASH_FP4, "tail_hybrid") == "tail_hybrid"
    assert op._select(3, FLASH_FP4, "megakernel") == "megakernel"


def test_unsupported_spec_is_refused_before_any_device_work():
    for is_supported in (
        op.is_supported,
        megakernel_op.is_supported,
        large_batch_op.is_supported,
        hybrid_op.is_supported,
    ):
        ok, why = is_supported(config=PRO)
        assert not ok and "DS-V4-Pro" in why
    ok, why = hybrid_op.is_supported(config=FLASH)
    assert not ok and "no tail-hybrid range" in why
    ok, why = megakernel_op.is_supported(config=dataclasses.replace(FLASH, megakernel_max_batch=0))
    assert not ok and "no megakernel" in why
    ok, why = large_batch_op.is_supported(
        config=dataclasses.replace(FLASH, large_batch=False, max_batch=47)
    )
    assert not ok and "no large-batch" in why
    i32 = torch.int32
    ctx, kv = torch.tensor([4096], dtype=i32), torch.tensor([16384], dtype=i32)
    table, sinks = torch.zeros((1, 128), dtype=i32), torch.zeros(64)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        op.precompile(torch.device("cpu"), 4, PRO)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        op.prepare(ctx, kv, table, sinks, config=PRO)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        megakernel_op.prepare(ctx, kv, table, config=PRO)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        large_batch_op.prepare(ctx, sinks, config=PRO)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        hybrid_op.prepare(ctx, kv, table, sinks, config=PRO)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        hybrid_op.precompile(torch.device("cpu"), 8, PRO)
    # the hybrid refuses a batch size outside its range before any device work, whichever spec
    with pytest.raises(ValueError, match="batch size 4 not in .5, 26."):
        hybrid_op.prepare(*_prepare_args(4), config=FLASH_FP4)
    with pytest.raises(ValueError, match="no tail-hybrid range"):
        hybrid_op.prepare(*_prepare_args(8), config=FLASH)
    with pytest.raises(ValueError, match="batch size 27 not in .5, 26."):
        hybrid_op.precompile(torch.device("cpu"), 27, FLASH_FP4)


def test_missing_production_logits_op_is_a_clear_error(monkeypatch):
    """The hybrid resolves the production logits op and its scheduler metadata when first used; a missing one is a
    RuntimeError that names it (no fallback)."""
    # the custom ops of the build do not register the op
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm._torch.custom_ops", types.ModuleType("custom_ops_without_k1")
    )
    monkeypatch.setattr(torch.ops, "trtllm", types.SimpleNamespace())
    with pytest.raises(RuntimeError, match="cute_dsl_fp4_paged_mqa_logits.*not registered"):
        hybrid_op._k1_op()
    # the custom ops cannot be imported at all
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.custom_ops", None)
    with pytest.raises(RuntimeError, match="cute_dsl_fp4_paged_mqa_logits.*failed to import"):
        hybrid_op._k1_op()
    # no scheduler metadata function
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm.deep_gemm", types.ModuleType("deep_gemm_without")
    )
    with pytest.raises(RuntimeError, match="get_paged_mqa_logits_metadata"):
        hybrid_op._metadata_fn()
    ok, why = hybrid_op.is_supported(config=FLASH_FP4)
    assert not ok and why


def test_op_is_supported_asks_the_hybrid_for_its_batch_sizes(monkeypatch):
    """With a ``batch_size`` that defaults to the tail hybrid, ``op.is_supported`` includes the production logits op
    (the device check is stubbed: the answer here is the one of the spec and the batch size)."""
    monkeypatch.setattr(op, "check_device", lambda device=None: (True, ""))
    monkeypatch.setattr(hybrid_op, "check_device", lambda device=None: (True, ""))
    monkeypatch.setattr(hybrid_op, "_k1_op", lambda: None)
    monkeypatch.setattr(hybrid_op, "_metadata_fn", lambda: None)
    for batch in (None, 1, 4, 5, 26, 27, 1024):
        assert op.is_supported(config=FLASH_FP4, batch_size=batch) == (True, "")

    def missing():
        raise RuntimeError("no production logits op")

    monkeypatch.setattr(hybrid_op, "_k1_op", missing)
    for batch in (5, 8, 26):
        assert op.is_supported(config=FLASH_FP4, batch_size=batch) == (
            False,
            "no production logits op",
        )
    # outside the hybrid's range, for the FP8 spec and without a batch size the hybrid is not needed
    for batch in (None, 1, 4, 27, 1024):
        assert op.is_supported(config=FLASH_FP4, batch_size=batch) == (True, "")
    assert op.is_supported(config=FLASH, batch_size=8) == (True, "")
    ok, why = op.is_supported(config=FLASH_FP4, batch_size=1025)
    assert not ok and "batch size 1025" in why
    ok, why = op.is_supported(config=PRO, batch_size=8)
    assert not ok and "DS-V4-Pro" in why


def test_kernels_reject_another_spec():
    pytest.importorskip("cutlass.cute")  # CuTe DSL; no GPU needed
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import large_batch, megakernel

    with pytest.raises(NotImplementedError, match="specialized for"):
        megakernel.SpecDsaMegakernel(config=PRO)
    with pytest.raises(NotImplementedError, match="specialized for"):
        large_batch.IndexerKernel(config=PRO)
    with pytest.raises(NotImplementedError, match="specialized for"):
        large_batch.AttentionTail(config=PRO)
    # the kernel constants follow the spec
    assert megakernel.K_TOP == large_batch.K_TOP == FLASH.topk
    assert megakernel.N_MAX == FLASH.max_index_len
    assert (large_batch.HEADS, large_batch.LATENT, large_batch.WINDOW, large_batch.NROWS) == (
        FLASH.attn_heads,
        FLASH.kv_dim,
        FLASH.window,
        FLASH.selected_rows,
    )
