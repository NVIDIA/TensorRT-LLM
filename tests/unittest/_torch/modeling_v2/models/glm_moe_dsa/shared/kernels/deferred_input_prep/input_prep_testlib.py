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
"""Helpers for the CPU tests of deferred_input_prep's host side (stager and loop domain).

``load()`` returns deferred_input_prep's ``dispatch``, ``compat``, ``packet`` and ``stager``
modules. Without a built ``tensorrt_llm`` it imports them from the source tree
under a stand-in package, so ``PrepStager`` runs with CPU tensors. Logging and
``prefer_pinned`` then need ``tensorrt_llm``: tests replace
``dispatch._log_info`` and ``stager._host_memory_pinned``.

``FakePacket`` replaces ``PrepPacket`` for CPU tensors. It performs the
staged writes at its ``commit``, through NumPy views of the destinations.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[9]
MODELING_V2 = REPO / "tensorrt_llm" / "_torch" / "_experimental" / "modeling_v2"
TARGET = MODELING_V2 / "models" / "glm_moe_dsa" / "shared"
PACKAGES = TARGET / "kernels"
_STANDALONE = "_input_prep_testlib"


def _module_from(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dispatch_standalone(name):
    """The target's ``dispatch.py`` as module ``name``, without importing ``tensorrt_llm``.

    Its absolute imports, ``tensorrt_llm.logger`` and the modeling_v2 router index, use
    only the standard library: when they are not imported yet, they are loaded from their
    source files and registered under their own names only while ``dispatch.py`` runs.
    """
    deps = (
        ("tensorrt_llm.logger", REPO / "tensorrt_llm" / "logger.py"),
        (
            "tensorrt_llm._torch._experimental.modeling_v2._router_index",
            MODELING_V2 / "_router_index.py",
        ),
    )
    added = []
    try:
        for dep, path in deps:
            if dep not in sys.modules:
                added.append(dep)
                _module_from(dep, path)
        return _module_from(name, TARGET / "dispatch.py")
    finally:
        for dep in added:
            sys.modules.pop(dep, None)


def _standalone():
    if _STANDALONE not in sys.modules:
        for name, path in (
            (_STANDALONE, TARGET),
            (f"{_STANDALONE}.kernels", PACKAGES),
            (f"{_STANDALONE}.kernels.deferred_input_prep", PACKAGES / "deferred_input_prep"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        sys.modules[_STANDALONE].dispatch = load_dispatch_standalone(f"{_STANDALONE}.dispatch")
    return SimpleNamespace(
        dispatch=importlib.import_module(f"{_STANDALONE}.dispatch"),
        compat=importlib.import_module(f"{_STANDALONE}.kernels.deferred_input_prep.compat"),
        packet=importlib.import_module(f"{_STANDALONE}.kernels.deferred_input_prep.packet"),
        stager=importlib.import_module(f"{_STANDALONE}.kernels.deferred_input_prep.stager"),
    )


def load():
    """deferred_input_prep's host modules: the installed package, else the source tree standalone."""
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_input_prep import (  # noqa: E501
            compat,
            packet,
            stager,
        )
    except (ImportError, OSError):
        return _standalone()
    return SimpleNamespace(dispatch=dispatch, compat=compat, packet=packet, stager=stager)


class FakePacket:
    """``PrepPacket`` stand-in for CPU tensors (see the module docstring).

    ``max_desc`` descriptors fit in one packet (one per write); a write whose
    bytes overlap an earlier write of the packet raises ``PacketOverlap``.
    """

    def __init__(self, packet_module, max_desc=1 << 20):
        self._full = packet_module.PacketFull
        self._overlap = packet_module.PacketOverlap
        self.max_desc = max_desc
        self._ops = []
        self._ranges = []
        self.num_commits = 0
        self.num_writes = 0

    def begin(self):
        self._ops, self._ranges = [], []

    def abort(self):
        self._ops, self._ranges = [], []

    @property
    def num_descriptors(self):
        return len(self._ops)

    def _claim(self, *dsts):
        if len(self._ops) + 1 > self.max_desc:
            raise self._full("fake packet is full")
        new = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in dsts]
        for lo, hi in new:
            for a, b in self._ranges:
                if lo < b and a < hi:
                    raise self._overlap("overlapping destinations in one packet")
        self._ranges.extend(new)

    def _add(self, fn, *dsts):
        self._claim(*dsts)
        self._ops.append(fn)

    def add_copy(self, dst, src, clamp_min=None):
        def op():
            values = src.reshape(dst.shape).numpy()
            dst.numpy()[...] = values if clamp_min is None else values.clip(min=clamp_min)

        self._add(op, dst)

    def add_values(self, dst, values):
        def op():
            dst.numpy()[...] = list(values)

        self._add(op, dst)

    def add_gather(self, dst, src, indices, addend=0):
        def op():
            flat = src.reshape(-1).numpy()
            out = dst.numpy()
            out[...] = 0
            out[: len(indices)] = flat[list(indices)] + addend

        self._add(op, dst)

    def add_zero(self, dst):
        def op():
            dst.numpy()[...] = 0

        self._add(op, dst)

    def add_clamp_min(self, dst, lo):
        def op():
            out = dst.numpy()
            out[...] = out.clip(min=lo)

        self._add(op, dst)

    def commit(self, stream=None, record_event=True):
        ops, self._ops, self._ranges = self._ops, [], []
        for op in ops:
            op()
        self.num_commits += 1
        self.num_writes += len(ops)
        return len(ops)

    def set_completion_event(self, event):
        pass


def eligibility_case(**overrides):
    """Arguments of ``PrepStager.begin_if_eligible`` for a steady served decode step."""
    mode = SimpleNamespace(use_one_engine=lambda: True)
    cfg = SimpleNamespace(
        spec_config=SimpleNamespace(spec_dec_mode=mode, is_linear_tree=True),
        disable_overlap_scheduler=overrides.pop("disable_overlap_scheduler", False),
        enable_attention_dp=overrides.pop("enable_attention_dp", False),
        use_mrope=False,
    )
    runner = SimpleNamespace(
        _config=cfg,
        use_beam_search=False,
        mapping=SimpleNamespace(has_cp_helix=lambda: False, pp_size=overrides.pop("pp_size", 1)),
        guided_decoder=None,
        _lora=SimpleNamespace(cuda_graph_manager=None),
    )
    rows = overrides.pop("rows", [SimpleNamespace(is_dummy=False, py_batch_idx=0)])
    requests = SimpleNamespace(num_context_requests=0, generation_requests=rows)
    DSAtrtllmAttentionMetadata = type("DSAtrtllmAttentionMetadata", (), {})
    manager = SimpleNamespace(index_mapper=object(), _use_per_layer_page_tables=False)
    attn = DSAtrtllmAttentionMetadata()
    attn.kv_cache_manager = manager
    attn.draft_kv_cache_manager = None
    args = dict(
        runner=runner,
        requests=requests,
        attn_metadata=attn,
        spec_metadata=SimpleNamespace(),
        new_tensors_device=SimpleNamespace(next_draft_tokens=object()),
        graph_replay=True,
        is_dummy=False,
        enable_spec_decode=True,
        promoted=False,
    )
    args.update(overrides)
    return args
