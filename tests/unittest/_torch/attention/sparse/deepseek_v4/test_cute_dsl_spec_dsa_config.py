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
"""specDSA configuration: the spec table, its validation and the support checks.

Runs without a GPU: the checks refuse an unsupported spec before any device work. Only the last test needs CuTe DSL
(not a GPU) and skips without it.
"""

import dataclasses
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import config as C
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import (
    large_batch_op,
    megakernel_op,
    op,
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
            s, name=FLASH.name, index_k_dtype="fp8", megakernel_max_batch=FLASH.megakernel_max_batch
        )
        == FLASH
    )
    assert (s.index_k_dtype, s.large_batch, s.max_batch) == ("fp4", True, 1024)
    # The FP4 K stream is twice as fast per byte, the megakernel row cost is not: the large-batch kernels are faster
    # from 21 rows (nsys, megakernel against large batch at 17 to 28 rows).
    assert s.megakernel_max_batch == 20
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
    not_structural = {"name", "megakernel_max_batch", "large_batch", "max_batch"}
    assert {f.name for f in dataclasses.fields(C.SpecDsaConfig)} == not_structural | set(
        other_value
    )
    for field, value in other_value.items():
        assert dataclasses.replace(FLASH, **{field: value}).kernel_key != FLASH.kernel_key, field
    relabeled = dataclasses.replace(FLASH, name="alias", megakernel_max_batch=8, max_batch=256)
    assert relabeled.kernel_key == FLASH.kernel_key


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


def test_readme_table_lists_the_batch_sizes_of_every_spec():
    """The README's spec table (megakernel and large-batch ranges) is the one of ``SUPPORTED_SPECS``."""
    readme = (Path(C.__file__).parent / "README.md").read_text()
    for spec in C.SUPPORTED_SPECS.values():
        rows = [ln for ln in readme.splitlines() if ln.startswith(f"| {spec.name} |")]
        assert len(rows) == 1, f"{spec.name}: no row in the README table"
        cells = [c.strip() for c in rows[0].strip().strip("|").split("|")]
        assert cells[-2] == f"1 to {spec.megakernel_max_batch} rows", spec.name
        assert cells[-1] == f"{spec.megakernel_max_batch + 1} to {spec.max_batch:,} rows", spec.name


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
    """The FP4 spec switches to the large-batch kernels one row after its megakernel's last row (20)."""
    last = FLASH_FP4.megakernel_max_batch
    assert last == 20
    assert (op.implementation_for(1, FLASH_FP4), op.implementation_for(last, FLASH_FP4)) == (
        "megakernel",
        "megakernel",
    )
    assert (op.implementation_for(last + 1, FLASH_FP4), op.implementation_for(1024, FLASH_FP4)) == (
        "large_batch",
        "large_batch",
    )
    assert list(megakernel_op.supported_batch_sizes(FLASH_FP4)) == list(range(1, last + 1))
    # the FP8 spec keeps its own switch
    assert op.implementation_for(47, FLASH) == "megakernel"
    assert op.implementation_for(48, FLASH) == "large_batch"


def test_unsupported_spec_is_refused_before_any_device_work():
    for is_supported in (op.is_supported, megakernel_op.is_supported, large_batch_op.is_supported):
        ok, why = is_supported(config=PRO)
        assert not ok and "DS-V4-Pro" in why
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
