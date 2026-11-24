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
"""Two worker switches of the engine DFlash worker.

``_draft_reads_target_metadata``: a worker whose drafter reads none of the
target's attention metadata skips its device refresh in the draft step and keeps
the kv_lens_cuda arithmetic. The default keeps the full refresh.

``_init_unified_draft_tables``: block offsets the device cannot decode (per-layer
page tables, Helix, SWA scratch reuse, index scales) build the draft table on the host.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[4]


class _Metadata:
    """Records how update_for_spec_dec is called; kv_lens_cuda is a CPU tensor."""

    def __init__(self, kv_lens):
        self.kv_lens_cuda = torch.tensor(kv_lens, dtype=torch.long)
        self.calls = []

    def update_for_spec_dec(self, **kwargs):
        # The device refresh reads kv_lens_cuda as the draft step leaves it.
        self.calls.append((kwargs, self.kv_lens_cuda.tolist()))


class _NoTargetMetadataWorker(DFlashWorker):
    def _draft_reads_target_metadata(self) -> bool:
        return False


def _draft_step(worker_cls):
    worker = worker_cls(DFlashDecodingConfig(max_draft_len=3), Mapping())
    metadata = _Metadata([10, 20, 30, 40])
    accepted = torch.tensor([1, 2, 4, 1])
    worker._prepare_kv_for_draft_forward(metadata, accepted, num_contexts=1, batch_size=4)
    during = metadata.kv_lens_cuda.tolist()
    worker._apply_kv_rewind_after_draft(metadata, SimpleNamespace(is_cuda_graph=False))
    return worker, metadata, during


def test_default_worker_refreshes_the_target_metadata():
    worker, metadata, during = _draft_step(DFlashWorker)
    assert worker._draft_reads_target_metadata() is True
    assert metadata.calls == [({}, [10, 21, 31, 41])]
    assert during == [10, 21, 31, 41]


def test_worker_that_reads_no_target_metadata_skips_only_the_device_refresh():
    worker, metadata, during = _draft_step(_NoTargetMetadataWorker)
    assert metadata.calls == [({"refresh_device_metadata": False}, [10, 21, 31, 41])]
    # Same kv_lens_cuda arithmetic as the default: +1 for the draft step, then the rewind.
    _, default_metadata, default_during = _draft_step(DFlashWorker)
    assert during == default_during
    assert metadata.kv_lens_cuda.tolist() == default_metadata.kv_lens_cuda.tolist()
    # The rewind leaves each generation row advanced by its accepted count; the context row stays.
    assert metadata.kv_lens_cuda.tolist() == [10, 20 + 2, 30 + 4, 40 + 1]
    assert torch.equal(worker._kv_rewind_amount, 1 - torch.tensor([2, 4, 1]))


def test_metadata_without_kv_lens_is_left_alone():
    worker = _NoTargetMetadataWorker(DFlashDecodingConfig(max_draft_len=3), Mapping())
    metadata = SimpleNamespace()
    worker._prepare_kv_for_draft_forward(metadata, torch.tensor([1]), num_contexts=0, batch_size=1)
    assert not worker._kv_rewind_pending


def test_engine_metadata_takes_the_switch():
    """A worker that skips the refresh runs on any engine metadata class: the base class takes
    the keyword, so the draft step raises no TypeError and keeps its kv_lens_cuda arithmetic."""
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata

    worker = _NoTargetMetadataWorker(DFlashDecodingConfig(max_draft_len=3), Mapping())
    metadata = object.__new__(AttentionMetadata)  # no buffers: only the hook and kv_lens_cuda
    metadata.kv_lens_cuda = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    accepted = torch.tensor([1, 2, 4, 1])
    worker._prepare_kv_for_draft_forward(metadata, accepted, num_contexts=1, batch_size=4)
    assert metadata.kv_lens_cuda.tolist() == [10, 21, 31, 41]
    worker._apply_kv_rewind_after_draft(metadata, SimpleNamespace(is_cuda_graph=False))
    assert metadata.kv_lens_cuda.tolist() == [10, 22, 34, 41]


_ENGINE_BACKENDS = _REPO / "tensorrt_llm/_torch/attention/backends"


def test_every_engine_metadata_override_takes_the_switch():
    """Each engine metadata class that overrides update_for_spec_dec takes
    refresh_device_metadata, so a worker may pass it whatever class the target uses."""
    if not _ENGINE_BACKENDS.is_dir():
        pytest.skip("the engine attention backends are not in this source tree")
    overrides, missing = [], []
    for path in sorted(_ENGINE_BACKENDS.rglob("*.py")):
        for cls in ast.walk(ast.parse(path.read_text())):
            if not isinstance(cls, ast.ClassDef) or not cls.name.endswith("Metadata"):
                continue
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "update_for_spec_dec":
                    where = f"{path.relative_to(_ENGINE_BACKENDS)}:{cls.name}"
                    overrides.append(where)
                    names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
                    if "refresh_device_metadata" not in names and fn.args.kwarg is None:
                        missing.append(where)
    assert len(overrides) >= 2, overrides  # the base class and at least one backend
    assert not missing, f"update_for_spec_dec without refresh_device_metadata: {missing}"


# --- Per-layer page tables send the draft table to the host ------------------------------


def _manager(**overrides):
    fields = dict(
        draft_layer_ids=[61],
        layer_offsets={61: 3},
        layer_to_pool_mapping_dict={3: 2},
        _index_scale_ints={2: 1},
        get_layer_page_index_scale=lambda layer: 1,
        get_layer_kv_factor=lambda layer: 1,
        _helix_cp_size=1,
        enable_swa_scratch_reuse=False,
        _use_per_layer_page_tables=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _init_tables(manager, monkeypatch):
    warnings = []
    monkeypatch.setattr(logger, "warning", lambda *msg: warnings.append(" ".join(msg)))
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=3), Mapping())
    worker._init_unified_draft_tables(manager)
    return worker, warnings


def test_shared_page_table_decodes_on_the_device(monkeypatch):
    worker, warnings = _init_tables(_manager(), monkeypatch)
    assert worker._device_draft_tables
    assert (worker._ctx_pool_idx, worker._ctx_block_divisor) == (2, 1)
    assert not warnings


def test_manager_without_the_attribute_decodes_on_the_device(monkeypatch):
    manager = _manager()
    del manager._use_per_layer_page_tables
    worker, _ = _init_tables(manager, monkeypatch)
    assert worker._device_draft_tables


@pytest.mark.parametrize(
    "overrides",
    [
        {"_use_per_layer_page_tables": True},
        {"_helix_cp_size": 8},
        {"enable_swa_scratch_reuse": True},
        {"_index_scale_ints": {2: 2}},
    ],
    ids=["per_layer_page_tables", "helix", "swa_scratch_reuse", "index_scale"],
)
def test_undecodable_offsets_build_the_table_on_the_host(overrides, monkeypatch):
    worker, warnings = _init_tables(_manager(**overrides), monkeypatch)
    assert not worker._device_draft_tables
    assert len(warnings) == 1 and "building draft block tables on the host" in warnings[0]
