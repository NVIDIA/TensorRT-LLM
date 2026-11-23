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
"""DFlash draft KV placement on a pipeline-parallel context server, on the host only.

A stage of a pipeline-parallel DFlash context server (the model construction records it,
``pp_dflash_context.set_pp_dflash_context_server``) lets a disaggregated context server at
pipeline_parallel_size > 1 share one KV cache manager between the target and a
TRTLLM DFlash drafter, with every draft layer on the last stage. The tests pin:

* the placement on a four-stage split 15/16/16/15 of a 62-layer target with a 5-layer
  drafter: the last stage holds global layers 47-66, the others none of 62-66;
* the pipeline-parallel blocker, lifted only for that placement;
* the draft window check over the full draft list on every stage;
* the drafter block width and generation headroom, equal on every stage;
* the per-stage KV bytes per token by kind;
* on any other server: the default blocker strings, layer specs and widths;
* the drafter context a context server keeps (window, reuse rewind and one block) covers
  every row the generation server's drafter reads, from the attention-window rules both
  derive from the drafter config (``config_utils.py`` and ``modeling_dflash.py``, executed
  from their files);
* FP8 draft KV (``TRTLLM_DFLASH_FP8_DRAFT_KV=1``): the draft layer specs in FP8 and nothing
  else changed, the stage's bytes per token and rescaled ``pool_ratio``, the draft buffers'
  own coalescing group, and the refusal wherever the draft layers do not share an FP8
  target's manager.

``_util.py``, ``kv_cache_manager_v2.py``, ``mapping.py`` and ``dflash_kv_storage.py`` are
executed from their files with every other ``tensorrt_llm`` module stubbed, so the tests run
without the compiled bindings. The stubs are installed in ``sys.modules`` only
while this module's tests run.
"""

from __future__ import annotations

import enum
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[5]
_FP8_KNOB = "TRTLLM_DFLASH_FP8_DRAFT_KV"
_SHARED_DRAFT_KV_ENV = "TRTLLM_DISAGG_SHARED_DRAFT_KV"
_BALANCE_TAG = "[pp_stage_host_balance]"
_STUBBED_ROOTS = ("tensorrt_llm", "triton", "nvtx")
_UTIL = "tensorrt_llm._torch.pyexecutor._util"
_KVM = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
_MAPPING = "tensorrt_llm.mapping"
_SWITCH = "tensorrt_llm._torch.disaggregation.pp_dflash_context"
_BALANCE = "tensorrt_llm._torch.pyexecutor.kv_cache.pp_stage_host_balance"
_STORAGE = "tensorrt_llm._torch.speculative.dflash_kv_storage"

# Released checkpoint geometry: 62 target layers, full attention at l = 1 mod 4,
# sliding window 2048 elsewhere; 16 KV heads x 64, FP8 KV. The drafter has 5
# sliding-window layers of 8 KV heads x 128 in BF16.
TARGET_LAYERS = 62
DRAFT_LAYERS = 5
WINDOW = 2048
MAX_SEQ_LEN = 996_672
PARTITION = [15, 16, 16, 15]
MAX_DRAFT_LEN = 7


# --------------------------------------------------------------------------
# Stubs for the modules the three files import
# --------------------------------------------------------------------------
class _StubMeta(type):
    """Metaclass of a stub: any attribute is another stub; a bare decorator passes through."""

    _cache: dict = {}

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub(f"{cls.__name__}.{name}")

    def __call__(cls, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return super().__call__(*args, **kwargs)

    def __or__(cls, other):
        return cls

    __ror__ = __or__

    def __getitem__(cls, item):
        return cls

    def __iter__(cls):
        return iter(())


class _StubBase(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub(f"{type(self).__name__}().{name}")


def _stub(qualname: str) -> type:
    cache = _StubMeta._cache
    if qualname not in cache:
        cache[qualname] = _StubMeta(qualname, (_StubBase,), {})
    return cache[qualname]


class _StubModule(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        submodule = sys.modules.get(f"{self.__name__}.{name}")
        if submodule is not None:
            return submodule
        return _stub(f"{self.__name__}.{name}")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves a ``_StubModule`` for every stubbed name not already in ``sys.modules``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUBBED_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


class DataType(enum.Enum):
    """The binding's KV data types that the layouts here use."""

    FP8 = "fp8"
    BF16 = "bf16"
    HALF = "half"
    FLOAT = "float"
    NVFP4 = "nvfp4"


_DTYPE_BYTES = {DataType.FP8: 1, DataType.BF16: 2, DataType.HALF: 2, DataType.FLOAT: 4}
_TORCH_DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float16: "float16"}
_BINDING_DTYPES = {"bfloat16": DataType.BF16, "float16": DataType.HALF}


class _Logger:
    def __init__(self):
        self.records = []

    def _record(self, level, message, *args, **kwargs):
        self.records.append((level, message % args if args else str(message)))

    def info(self, message, *args, **kwargs):
        self._record("info", message, *args)

    def info_once(self, message, *args, **kwargs):
        self._record("info", message, *args)

    def warning(self, message, *args, **kwargs):
        self._record("warning", message, *args)

    def warning_once(self, message, *args, **kwargs):
        self._record("warning", message, *args)

    def debug(self, message, *args, **kwargs):
        self._record("debug", message, *args)

    def error(self, message, *args, **kwargs):
        self._record("error", message, *args)


def _clamp_max_attention_window_vec(max_attention_window_vec, max_seq_len):
    windows = [max_seq_len] if max_attention_window_vec is None else list(max_attention_window_vec)
    return [min(max_seq_len, window) for window in windows]


def _project_max_attention_window_vec(max_attention_window_vec, pp_layers):
    pattern_len = len(max_attention_window_vec)
    return [max_attention_window_vec[layer_idx % pattern_len] for layer_idx in pp_layers]


def _get_draft_block_width(spec_config):
    if spec_config is None or not spec_config.spec_dec_mode.is_dflash():
        return 0
    return spec_config.max_draft_len + 1


def _get_num_extra_kv_tokens(spec_config):
    if spec_config is None:
        return 0
    return spec_config.max_draft_len - 1


def _module(name: str, **attributes) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = _StubModule(name)
        sys.modules[name] = module
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _install_stubs(logger: _Logger) -> None:
    _module("tensorrt_llm")
    _module(
        "tensorrt_llm._utils",
        get_size_in_bytes=lambda elements, dtype: int(elements * _DTYPE_BYTES[dtype]),
        str_dtype_to_binding=lambda name: _BINDING_DTYPES[name],
        torch_dtype_to_str=lambda dtype: _TORCH_DTYPE_NAMES[dtype],
        mpi_disabled=lambda: False,
    )
    _module("tensorrt_llm.bindings", DataType=DataType)
    _module("tensorrt_llm.logger", logger=logger)
    _module(
        "tensorrt_llm.runtime.kv_cache_manager_v2",
        _KV_CACHE_ITERATION_STATS_DELTA_FIELDS=(),
        DataRole=str,
    )
    _module(
        "tensorrt_llm._torch.pyexecutor.resource_manager",
        DataType=DataType,
        _clamp_max_attention_window_vec=_clamp_max_attention_window_vec,
        _project_max_attention_window_vec=_project_max_attention_window_vec,
    )
    _module(
        "tensorrt_llm._torch.pyexecutor.config_utils",
        is_mla=lambda config: False,
        is_gemma4_hybrid=lambda config: False,
    )
    _module(
        "tensorrt_llm.llmapi.llm_args",
        disagg_shares_draft_kv=lambda spec_config: True,
    )
    _module(
        "tensorrt_llm._torch.speculative",
        get_draft_block_width=_get_draft_block_width,
        get_num_extra_kv_tokens=_get_num_extra_kv_tokens,
        draft_prompt_lookahead=lambda spec_config: 0,
    )


def _exec_file(name: str, relative_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loaded():
    """The files under test, executed against the stubs; ``sys.modules`` restored after."""
    saved = {
        name: module for name, module in sys.modules.items() if name.split(".")[0] in _STUBBED_ROOTS
    }
    for name in saved:
        del sys.modules[name]
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    logger = _Logger()
    try:
        _install_stubs(logger)
        mapping = _exec_file(_MAPPING, "tensorrt_llm/mapping.py")
        # Pure Python (no torch): the KV layout reads the attention-group TP size from it.
        _exec_file("tensorrt_llm._torch.adp_groups", "tensorrt_llm/_torch/adp_groups.py")
        switch = _exec_file(_SWITCH, "tensorrt_llm/_torch/disaggregation/pp_dflash_context.py")
        balance = _exec_file(
            _BALANCE, "tensorrt_llm/_torch/pyexecutor/kv_cache/pp_stage_host_balance.py"
        )
        _exec_file(_STORAGE, "tensorrt_llm/_torch/speculative/dflash_kv_storage.py")
        kvm = _exec_file(_KVM, "tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py")
        util = _exec_file(_UTIL, "tensorrt_llm/_torch/pyexecutor/_util.py")
        yield SimpleNamespace(
            mapping=mapping, switch=switch, balance=balance, kvm=kvm, util=util, logger=logger
        )
    finally:
        sys.meta_path.remove(finder)
        for name in [name for name in sys.modules if name.split(".")[0] in _STUBBED_ROOTS]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def _shared_draft_kv_requested(loaded, monkeypatch):
    # A context server with the shared draft KV sets it; the layout rules below assume it.
    monkeypatch.setenv(_SHARED_DRAFT_KV_ENV, "1")
    monkeypatch.delenv(_FP8_KNOB, raising=False)
    # No model construction recorded a pipeline-parallel DFlash context server yet.
    loaded.switch.set_pp_dflash_context_server(False)
    yield
    loaded.switch.set_pp_dflash_context_server(False)


def _server(loaded, enabled: bool = True) -> None:
    """Record what the model construction of a pipeline-parallel DFlash context server records."""
    loaded.switch.set_pp_dflash_context_server(enabled)


# --------------------------------------------------------------------------
# A KV cache creator for the released layout
# --------------------------------------------------------------------------
def _quant_config(*, fp8_kv: bool):
    return SimpleNamespace(
        quant_mode=SimpleNamespace(
            has_fp8_kv_cache=lambda: fp8_kv,
            has_fp4_kv_cache=lambda: False,
            has_kv_cache_quant=lambda: fp8_kv,
        )
    )


def _target_config():
    pretrained = SimpleNamespace(
        num_hidden_layers=TARGET_LAYERS,
        num_attention_heads=64,
        num_key_value_heads=16,
        head_dim=64,
        hidden_size=64 * 64,
        torch_dtype=torch.bfloat16,
    )
    return SimpleNamespace(pretrained_config=pretrained, quant_config=_quant_config(fp8_kv=True))


def _draft_config():
    pretrained = SimpleNamespace(
        num_hidden_layers=DRAFT_LAYERS,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=32 * 128,
        torch_dtype=torch.bfloat16,
    )
    return SimpleNamespace(pretrained_config=pretrained, quant_config=_quant_config(fp8_kv=False))


def _target_window_pattern():
    # Repeating pattern anchored at layer 0: layer l is full attention when l = 1 mod 4.
    return [WINDOW, MAX_SEQ_LEN, WINDOW, WINDOW]


def _spec_config(*, backend="TRTLLM", dflash=True):
    return SimpleNamespace(
        spec_dec_mode=SimpleNamespace(
            is_dflash=lambda: dflash,
            is_dspark=lambda: not dflash,
            is_external_drafter=lambda: True,
        ),
        attention_backend=backend,
        draft_is_embedded_in_target=False,
        _allow_separate_draft_kv_cache=True,
        draft_len_schedule=None,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
        tokens_per_gen_step=MAX_DRAFT_LEN + 1,
        use_dynamic_tree=False,
    )


def _mapping(loaded, *, pp_size, rank, partition=None):
    return loaded.mapping.Mapping(
        world_size=pp_size,
        rank=rank,
        gpus_per_node=max(pp_size, 1),
        pp_size=pp_size,
        pp_partition=partition,
    )


def _creator(
    loaded,
    *,
    pp_size=4,
    rank=3,
    partition=PARTITION,
    is_disagg=True,
    spec_config=None,
    draft_windows=None,
):
    util = loaded.util
    creator = object.__new__(util.KvCacheCreator)
    creator._speculative_config = spec_config if spec_config is not None else _spec_config()
    creator._draft_model_engine = None
    draft = _draft_config()
    creator._draft_config = draft
    creator._mapping = _mapping(loaded, pp_size=pp_size, rank=rank, partition=partition)
    creator._max_seq_len = MAX_SEQ_LEN
    creator._unification_max_seq_len = MAX_SEQ_LEN
    windows = [WINDOW] * DRAFT_LAYERS if draft_windows is None else draft_windows
    creator._get_draft_max_attention_window = lambda max_seq_len, kv_cache_config: windows
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=_target_config()), dtype=torch.bfloat16
    )
    creator._kv_cache_manager_cls = loaded.kvm.KVCacheManagerV2
    creator._kv_cache_config = SimpleNamespace(max_attention_window=_target_window_pattern())
    creator._is_disagg = is_disagg
    creator._get_draft_kv_model_config = lambda: draft
    return creator


def _specs(creator):
    return creator._build_unified_dflash_layer_specs(creator._kv_cache_config)


def _spec_rows(specs):
    """Everything a layer spec carries, as plain values."""
    return [
        (
            spec.global_layer_id,
            str(spec.owner),
            spec.dtype,
            spec.num_kv_heads,
            spec.head_dim,
            tuple(spec.buffer_roles),
            spec.sliding_window_size,
            spec.num_sink_tokens,
        )
        for spec in specs
    ]


def _expected_target_rows(layer_ids):
    return [
        (
            layer_id,
            "target",
            DataType.FP8,
            16,
            64,
            ("key", "value"),
            None if layer_id % 4 == 1 else WINDOW,
            None,
        )
        for layer_id in layer_ids
    ]


def _expected_draft_rows(draft_ids):
    return [
        (TARGET_LAYERS + i, "draft", DataType.BF16, 8, 128, ("key", "value"), WINDOW, None)
        for i in draft_ids
    ]


def _stage_target_ids(rank):
    start = sum(PARTITION[:rank])
    return list(range(start, start + PARTITION[rank]))


# --------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------
@pytest.mark.parametrize("enabled", [False, True])
def test_switch_is_what_the_model_construction_recorded(loaded, enabled):
    assert loaded.switch.pp_dflash_context_enabled() is False
    _server(loaded, enabled)
    # The KV cache manager and the KV cache creator read the one shared switch.
    assert loaded.kvm.pp_dflash_context_enabled is loaded.switch.pp_dflash_context_enabled
    assert loaded.util.pp_dflash_context_enabled is loaded.switch.pp_dflash_context_enabled
    assert loaded.kvm.pp_dflash_context_enabled() is enabled


# --------------------------------------------------------------------------
# Placement on a pipeline-parallel DFlash context server
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rank", range(4))
def test_last_stage_holds_every_draft_layer(loaded, monkeypatch, rank):
    _server(loaded)
    creator = _creator(loaded, rank=rank)

    assert creator._pp_dflash_context_placement()
    assert creator._dflash_unify_blocker() is None
    assert creator._should_unify_dflash_kv_cache()
    creator._check_shared_draft_kv_request()

    rows = _spec_rows(_specs(creator))
    draft_ids = list(range(DRAFT_LAYERS)) if rank == 3 else []
    assert rows == _expected_target_rows(_stage_target_ids(rank)) + _expected_draft_rows(draft_ids)


def test_stage_layer_ranges_stay_contiguous(loaded, monkeypatch):
    """The cache transceiver treats stage p as global layers [sum(lpp[:p]), sum(lpp[:p+1]))."""
    _server(loaded)
    per_stage = [
        [spec.global_layer_id for spec in _specs(_creator(loaded, rank=rank))] for rank in range(4)
    ]

    layer_num_per_pp = [len(ids) for ids in per_stage]
    assert layer_num_per_pp == [15, 16, 16, 20]
    starts = [sum(layer_num_per_pp[:rank]) for rank in range(4)]
    for rank, ids in enumerate(per_stage):
        assert ids == list(range(starts[rank], starts[rank] + layer_num_per_pp[rank]))
    assert per_stage[3][-DRAFT_LAYERS:] == [62, 63, 64, 65, 66]


def test_default_partition_places_draft_layers_on_the_last_stage(loaded, monkeypatch):
    """Without pp_partition the 5 draft layers would split 2/1/1/1; the placement keeps them whole."""
    _server(loaded)
    draft_per_stage = []
    for rank in range(4):
        specs = _specs(_creator(loaded, rank=rank, partition=None))
        draft_per_stage.append([s.global_layer_id for s in specs if str(s.owner) == "draft"])
    assert draft_per_stage == [[], [], [], [62, 63, 64, 65, 66]]


def test_draft_window_check_covers_every_draft_layer_on_every_stage(loaded, monkeypatch):
    _server(loaded)
    mixed = [WINDOW, WINDOW, 4096, WINDOW, WINDOW]
    reasons = {
        _creator(loaded, rank=rank, draft_windows=mixed)._dflash_unify_blocker()
        for rank in range(4)
    }
    assert reasons == {"the draft layers use different attention windows"}


# --------------------------------------------------------------------------
# The pipeline-parallel blocker
# --------------------------------------------------------------------------
def _blockers(loaded, *, is_disagg, backend, dflash):
    spec_config = _spec_config(backend=backend, dflash=dflash)
    return [
        _creator(
            loaded, rank=rank, is_disagg=is_disagg, spec_config=spec_config
        )._dflash_unify_blocker()
        for rank in range(4)
    ]


@pytest.mark.parametrize(
    "is_disagg,backend,dflash,expected",
    [
        (False, "TRTLLM", True, "pipeline_parallel_size is 4, not 1"),
        (True, "TRTLLM", False, "pipeline_parallel_size is 4, not 1"),
        (True, "VANILLA", True, "the drafter attention_backend is VANILLA, not TRTLLM"),
        (True, "TRTLLM", True, None),
    ],
)
def test_pipeline_parallel_blocker_lifts_only_for_the_placement(
    loaded, monkeypatch, is_disagg, backend, dflash, expected
):
    _server(loaded)
    assert _blockers(loaded, is_disagg=is_disagg, backend=backend, dflash=dflash) == [expected] * 4


def test_context_parallel_still_blocks_on_the_server(loaded, monkeypatch):
    _server(loaded)
    creator = _creator(loaded)
    creator._mapping = SimpleNamespace(
        pp_size=4,
        cp_size=2,
        tp_size=1,
        enable_attention_dp=False,
        is_last_pp_rank=lambda: True,
        pp_layers=lambda num_layers: list(range(num_layers)),
    )
    assert creator._dflash_unify_blocker() == "context_parallel_size is 2, not 1"


# --------------------------------------------------------------------------
# Any other server: the default behavior
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "is_disagg,backend,dflash,expected",
    [
        (True, "TRTLLM", True, "pipeline_parallel_size is 4, not 1"),
        (False, "TRTLLM", True, "pipeline_parallel_size is 4, not 1"),
        (True, "TRTLLM", False, "pipeline_parallel_size is 4, not 1"),
        (True, "VANILLA", True, "the drafter attention_backend is VANILLA, not TRTLLM"),
    ],
)
def test_other_server_pipeline_parallel_blocker(loaded, is_disagg, backend, dflash, expected):
    assert _blockers(loaded, is_disagg=is_disagg, backend=backend, dflash=dflash) == [expected] * 4


def test_other_server_keeps_the_pp_refusal_and_its_message(loaded):
    for rank in range(4):
        creator = _creator(loaded, rank=rank)
        assert not creator._should_unify_dflash_kv_cache()
        assert _specs(creator) == ()
        with pytest.raises(ValueError) as error:
            creator._check_shared_draft_kv_request()
        assert str(error.value) == (
            "TRTLLM_DISAGG_SHARED_DRAFT_KV=1 requires the target and draft "
            "KV layers to share one KV cache manager, but pipeline_parallel_size "
            "is 4, not 1."
        )


@pytest.mark.parametrize("server", [False, True])
def test_single_stage_layout_on_any_server(loaded, server):
    """A single stage lays out target and draft layers alike on any server."""
    _server(loaded, server)
    creator = _creator(loaded, pp_size=1, rank=0, partition=None)

    assert creator._dflash_unify_blocker() is None
    rows = _spec_rows(_specs(creator))
    assert rows == _expected_target_rows(range(TARGET_LAYERS)) + _expected_draft_rows(
        range(DRAFT_LAYERS)
    )


def test_single_stage_with_a_partition_still_raises_in_the_window_check(loaded):
    """pp_layers(5) against a 62-layer partition raises."""
    creator = _creator(loaded, pp_size=1, rank=0, partition=[TARGET_LAYERS])
    with pytest.raises(ValueError, match="does not match num_layers=5"):
        creator._dflash_unify_blocker()


# --------------------------------------------------------------------------
# Drafter block width and generation headroom on every stage
# --------------------------------------------------------------------------
def _hand_built_stage_specs(loaded, rank):
    """Stage ``rank``'s layout as the placement builds it, written out by hand."""
    kvm = loaded.kvm
    specs = [
        kvm.KVCacheLayerSpec(
            global_layer_id=layer_id,
            owner=kvm.KVCacheLayerOwner.TARGET,
            dtype=DataType.FP8,
            num_kv_heads=16,
            head_dim=64,
            buffer_roles=(kvm.Role.KEY, kvm.Role.VALUE),
            sliding_window_size=None if layer_id % 4 == 1 else WINDOW,
        )
        for layer_id in _stage_target_ids(rank)
    ]
    if rank == 3:
        specs += [
            kvm.KVCacheLayerSpec(
                global_layer_id=TARGET_LAYERS + i,
                owner=kvm.KVCacheLayerOwner.DRAFT,
                dtype=DataType.BF16,
                num_kv_heads=8,
                head_dim=128,
                buffer_roles=(kvm.Role.KEY, kvm.Role.VALUE),
                sliding_window_size=WINDOW,
            )
            for i in range(DRAFT_LAYERS)
        ]
    return tuple(specs)


_BLOCK_WIDTH = MAX_DRAFT_LEN + 1
_POSITION_EXTRA = MAX_DRAFT_LEN - 1


def test_placement_specs_match_the_hand_built_stage_specs(loaded, monkeypatch):
    _server(loaded)
    for rank in range(4):
        assert _specs(_creator(loaded, rank=rank)) == _hand_built_stage_specs(loaded, rank)


def test_block_width_and_headroom_are_equal_on_every_stage(loaded, monkeypatch):
    _server(loaded)
    kvm = loaded.kvm
    spec_config = _spec_config()
    widths, extras, capacities = [], [], []
    for rank in range(4):
        specs = _hand_built_stage_specs(loaded, rank)
        mapping = _mapping(loaded, pp_size=4, rank=rank, partition=PARTITION)
        widths.append(kvm._get_shared_draft_block_width(spec_config, specs, mapping))
        extras.append(kvm._get_kv_alloc_extra_tokens(spec_config, specs, mapping))
        capacities.append(
            kvm._get_generation_kv_capacity(
                spec_config, is_draft=False, layer_specs=specs, mapping=mapping
            )
        )
    assert widths == [_BLOCK_WIDTH] * 4
    assert extras == [_BLOCK_WIDTH] * 4
    assert capacities == [(MAX_DRAFT_LEN, _BLOCK_WIDTH + MAX_DRAFT_LEN + 1)] * 4


def test_other_server_block_width_only_where_the_draft_layers_are(loaded):
    """On any other server only the stage holding draft layers charges the block (no mapping)."""
    kvm = loaded.kvm
    spec_config = _spec_config()
    widths, extras, capacities = [], [], []
    for rank in range(4):
        specs = _hand_built_stage_specs(loaded, rank)
        widths.append(kvm._get_shared_draft_block_width(spec_config, specs))
        extras.append(kvm._get_kv_alloc_extra_tokens(spec_config, specs))
        capacities.append(
            kvm._get_generation_kv_capacity(spec_config, is_draft=False, layer_specs=specs)
        )
    assert widths == [0, 0, 0, _BLOCK_WIDTH]
    assert extras == [_POSITION_EXTRA] * 3 + [_BLOCK_WIDTH]
    assert capacities == [(MAX_DRAFT_LEN, _POSITION_EXTRA + MAX_DRAFT_LEN + 1)] * 3 + [
        (MAX_DRAFT_LEN, _BLOCK_WIDTH + MAX_DRAFT_LEN + 1)
    ]


@pytest.mark.parametrize("server", [False, True])
def test_block_width_needs_a_shared_layout_and_pipeline_parallelism(loaded, server):
    kvm = loaded.kvm
    _server(loaded, server)
    spec_config = _spec_config()
    target_only = _hand_built_stage_specs(loaded, 0)
    pp1 = _mapping(loaded, pp_size=1, rank=0)
    pp4 = _mapping(loaded, pp_size=4, rank=0, partition=PARTITION)

    # No explicit layout: no shared drafter, whatever the server.
    assert kvm._get_shared_draft_block_width(spec_config, None, pp4) == 0
    # A single stage holds its own draft layers or none at all.
    assert kvm._get_shared_draft_block_width(spec_config, target_only, pp1) == 0
    assert kvm._get_shared_draft_block_width(spec_config, target_only, None) == 0
    # A non-DFlash drafter never charges it.
    dspark = _spec_config(dflash=False)
    assert kvm._get_shared_draft_block_width(dspark, target_only, pp4) == 0
    expected = _BLOCK_WIDTH if server else 0
    assert kvm._get_shared_draft_block_width(spec_config, target_only, pp4) == expected


def test_block_width_ignores_the_mapping_on_any_other_server(loaded):
    """Off a pipeline-parallel DFlash context server the mapping argument is never read."""
    kvm = loaded.kvm
    spec_config = _spec_config()
    unreadable = object()
    for layout in (None, _hand_built_stage_specs(loaded, 0), _hand_built_stage_specs(loaded, 3)):
        assert kvm._get_shared_draft_block_width(
            spec_config, layout
        ) == kvm._get_shared_draft_block_width(spec_config, layout, unreadable)
        assert kvm._get_generation_kv_capacity(
            spec_config, is_draft=False, layer_specs=layout
        ) == kvm._get_generation_kv_capacity(
            spec_config, is_draft=False, layer_specs=layout, mapping=unreadable
        )


# --------------------------------------------------------------------------
# Per-stage KV bytes per token
# --------------------------------------------------------------------------
def test_stage_kv_bytes_per_token(loaded):
    util = loaded.util
    stages = [
        util.stage_kv_bytes_per_token(_hand_built_stage_specs(loaded, rank), MAX_SEQ_LEN)
        for rank in range(4)
    ]

    # Every stage holds 4 of the 16 full-attention layers, 2,048 B per token each.
    assert [stage.full for stage in stages] == [8192] * 4
    assert [stage.sliding_target for stage in stages] == [22528, 24576, 24576, 22528]
    assert [stage.sliding_draft for stage in stages] == [0, 0, 0, 20480]
    assert [stage.sliding_windows for stage in stages] == [(WINDOW,)] * 4
    assert [stage.draft_layer_ids for stage in stages] == [()] * 3 + [(62, 63, 64, 65, 66)]
    assert [stage.sliding_growth for stage in stages][:3] == [1.0] * 3
    assert stages[3].sliding_growth == pytest.approx(43008 / 22528)


def test_stage_layout_log_line(loaded, monkeypatch):
    _server(loaded)
    creator = _creator(loaded, rank=3)
    loaded.logger.records.clear()

    creator._log_pp_dflash_stage_layout(_specs(creator))

    assert loaded.logger.records == [
        (
            "info",
            "DFlash on pipeline stage 3 of 4: draft KV layers [62, 63, 64, 65, 66], 20 KV "
            "layers; KV bytes per token: full 8192, sliding target 22528, sliding draft "
            "20480 (windows [2048], sliding growth 1.909).",
        )
    ]


# --------------------------------------------------------------------------
# Per-stage pool_ratio on the stage that holds the draft layers
# --------------------------------------------------------------------------
TOKENS_PER_BLOCK = 32
POOL_RATIO = [0.15, 0.85]
# Sliding-window and full-attention KV bytes per token of a stage without draft
# layers (stage 0), and of the last stage with them.
_SLIDING_TARGET_S3 = 22528
_SLIDING_S3 = 22528 + 20480
_FULL = 8192


class _Config:
    """The two ``KvCacheConfig`` members the per-stage config reads and copies."""

    def __init__(self, pool_ratio):
        self.pool_ratio = pool_ratio
        self.copies = []

    def model_copy(self, update):
        copy = _Config(update.get("pool_ratio", self.pool_ratio))
        self.copies.append(copy)
        return copy


def _stage_creator(loaded, rank):
    creator = _creator(loaded, rank=rank)
    creator._tokens_per_block = TOKENS_PER_BLOCK
    return creator


def test_draft_stage_pool_ratio_keeps_the_target_token_split(loaded):
    """The last stage scales its sliding-window share by the draft growth, then renormalizes."""
    util = loaded.util
    specs = _hand_built_stage_specs(loaded, 3)
    ratio = util.draft_inclusive_pool_ratio(POOL_RATIO, specs, MAX_SEQ_LEN, TOKENS_PER_BLOCK)

    growth = _SLIDING_S3 / _SLIDING_TARGET_S3
    weights = [POOL_RATIO[0] * growth, POOL_RATIO[1]]
    assert ratio == pytest.approx([weight / sum(weights) for weight in weights])
    assert sum(ratio) == pytest.approx(1.0, abs=1e-12)
    # Tokens per byte share: sliding : full equals the target-only stage's split.
    target_tokens = (POOL_RATIO[0] / _SLIDING_TARGET_S3) / (POOL_RATIO[1] / _FULL)
    draft_stage_tokens = (ratio[0] / _SLIDING_S3) / (ratio[1] / _FULL)
    assert draft_stage_tokens == pytest.approx(target_tokens)
    # Both life cycles give up the same fraction of their tokens to the draft bytes.
    assert ratio[1] / POOL_RATIO[1] == pytest.approx(ratio[0] / (POOL_RATIO[0] * growth))


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_stages_without_draft_layers_keep_the_pool_ratio(loaded, rank):
    util = loaded.util
    specs = _hand_built_stage_specs(loaded, rank)
    assert util.draft_inclusive_pool_ratio(POOL_RATIO, specs, MAX_SEQ_LEN, TOKENS_PER_BLOCK) is None


def test_pool_ratio_follows_the_life_cycle_registration_order(loaded):
    """A layout that starts with a full-attention layer lists the full share first."""
    kvm, util = loaded.kvm, loaded.util

    def spec(layer_id, owner, window):
        return kvm.KVCacheLayerSpec(
            global_layer_id=layer_id,
            owner=owner,
            dtype=DataType.FP8,
            num_kv_heads=16,
            head_dim=64,
            buffer_roles=(kvm.Role.KEY, kvm.Role.VALUE),
            sliding_window_size=window,
        )

    target, draft = kvm.KVCacheLayerOwner.TARGET, kvm.KVCacheLayerOwner.DRAFT
    specs = (
        spec(0, target, MAX_SEQ_LEN),
        spec(1, target, WINDOW),
        spec(2, draft, WINDOW),
        spec(3, target, None),
    )
    ratio = util.draft_inclusive_pool_ratio([0.85, 0.15], specs, MAX_SEQ_LEN, TOKENS_PER_BLOCK)
    # Life cycles: full (layers 0 and 3), then sliding (1 and 2); the draft layer doubles sliding.
    weights = [0.85, 0.15 * 2]
    assert ratio == pytest.approx([weight / sum(weights) for weight in weights])


def test_draft_stage_pool_ratio_leaves_unmatched_layouts_alone(loaded):
    kvm, util = loaded.kvm, loaded.util
    specs = _hand_built_stage_specs(loaded, 3)
    # Estimation and an unset ratio: nothing to rescale.
    assert util.draft_inclusive_pool_ratio(None, specs, MAX_SEQ_LEN, TOKENS_PER_BLOCK) is None
    # One entry per life cycle is what the manager accepts; anything else stays as given.
    assert util.draft_inclusive_pool_ratio([1.0], specs, MAX_SEQ_LEN, TOKENS_PER_BLOCK) is None
    # Draft layers with a window of their own form a life cycle no configured share covers.
    target_only = [spec for spec in specs if spec.owner is kvm.KVCacheLayerOwner.TARGET]
    own_window = [
        kvm.KVCacheLayerSpec(
            global_layer_id=spec.global_layer_id,
            owner=spec.owner,
            dtype=spec.dtype,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            buffer_roles=spec.buffer_roles,
            sliding_window_size=WINDOW // 2,
        )
        for spec in specs
        if spec.owner is kvm.KVCacheLayerOwner.DRAFT
    ]
    assert (
        util.draft_inclusive_pool_ratio(
            [0.1, 0.8, 0.1], target_only + own_window, MAX_SEQ_LEN, TOKENS_PER_BLOCK
        )
        is None
    )


def test_draft_stage_config_rescales_only_the_last_stage(loaded, monkeypatch):
    _server(loaded)
    util = loaded.util
    for rank in range(4):
        creator = _stage_creator(loaded, rank)
        config = _Config(list(POOL_RATIO))
        loaded.logger.records.clear()

        stage_config = creator._pp_dflash_stage_kv_cache_config(config, _specs(creator))

        assert config.pool_ratio == POOL_RATIO
        if rank < 3:
            assert stage_config is config
            assert config.copies == []
            assert loaded.logger.records == []
            continue
        expected = util.draft_inclusive_pool_ratio(
            POOL_RATIO, _specs(creator), MAX_SEQ_LEN, TOKENS_PER_BLOCK
        )
        assert config.copies == [stage_config]
        assert stage_config.pool_ratio == expected
        assert loaded.logger.records == [
            (
                "info",
                "DFlash on pipeline stage 3: pool_ratio [0.15, 0.85] -> "
                f"{[round(share, 6) for share in expected]} so the draft KV layers keep the "
                "configured token split.",
            )
        ]


def test_draft_stage_config_keeps_an_unset_pool_ratio(loaded, monkeypatch):
    """The estimation pass runs with no pool_ratio; its config is passed through."""
    _server(loaded)
    creator = _stage_creator(loaded, 3)
    config = _Config(None)
    assert creator._pp_dflash_stage_kv_cache_config(config, _specs(creator)) is config
    assert config.copies == []


def test_other_server_keeps_the_configured_pool_ratio(loaded):
    """Without the placement the config object itself is used, on every stage."""
    for rank in range(4):
        creator = _stage_creator(loaded, rank)
        config = _Config(list(POOL_RATIO))
        assert (
            creator._pp_dflash_stage_kv_cache_config(config, _hand_built_stage_specs(loaded, rank))
            is config
        )
        assert config.copies == []


# --------------------------------------------------------------------------
# Per-stage host tier sizing (pp_stage_host_balance.py)
# --------------------------------------------------------------------------
_GIB = 1 << 30
_HOST = 279 * _GIB
_REMOTE = 180 * _GIB
# Device quotas of the four stages after the cross-rank token clamp.
_DEVICE = [int(gib * _GIB) for gib in (119.03, 119.32, 119.32, 121.91)]


def _host_config(host_cache_size, pool_ratio=POOL_RATIO):
    return SimpleNamespace(host_cache_size=host_cache_size, pool_ratio=pool_ratio)


def _applies(loaded, config, mapping, *, is_draft=False, is_estimating=False):
    return loaded.kvm.KVCacheManagerV2._pp_stage_host_balance_applies(
        config, mapping, is_draft=is_draft, is_estimating=is_estimating
    )


def test_host_balance_needs_the_pipeline_parallel_placement(loaded):
    """Off a pipeline-parallel DFlash context server the host tier keeps its size, silently."""
    loaded.logger.records.clear()
    assert not _applies(loaded, _host_config(_HOST), _mapping(loaded, pp_size=4, rank=3))
    assert loaded.logger.records == []
    _server(loaded)
    assert not _applies(loaded, _host_config(_HOST), _mapping(loaded, pp_size=1, rank=0))


def test_host_balance_applies_only_to_the_target_manager_with_a_host_tier(loaded):
    _server(loaded)
    mapping = _mapping(loaded, pp_size=4, rank=3)
    assert _applies(loaded, _host_config(_HOST), mapping)
    assert not _applies(loaded, _host_config(_HOST), mapping, is_draft=True)
    assert not _applies(loaded, _host_config(_HOST), mapping, is_estimating=True)
    assert not _applies(loaded, _host_config(None), mapping)
    assert not _applies(loaded, _host_config(0), mapping)


class _Gather:
    """``Distributed.get(mapping).allgather``: records what a rank sends."""

    def __init__(self, gathered=None):
        self.gathered = gathered
        self.sent = []

    def get(self, mapping):
        return self

    def allgather(self, obj):
        self.sent.append(obj)
        return list(self.gathered) if self.gathered is not None else [obj] * 4


def _stage_manager(loaded, rank, pool_ratio="draft-inclusive"):
    """The members the balance reads, for one stage of the placement, and its config."""
    specs = _specs(_stage_creator(loaded, rank))
    # The manager's own windows for an explicit layer layout: None for full attention.
    windows = [
        None
        if spec.sliding_window_size is None or spec.sliding_window_size >= MAX_SEQ_LEN
        else spec.sliding_window_size
        for spec in specs
    ]
    manager = SimpleNamespace(
        layer_specs=tuple(specs),
        max_attention_window_vec=windows,
        max_seq_len=MAX_SEQ_LEN,
        tokens_per_block=TOKENS_PER_BLOCK,
    )
    manager._get_runtime_cache_size_layer_components = types.MethodType(
        loaded.kvm.KVCacheManagerV2._get_runtime_cache_size_layer_components, manager
    )
    if pool_ratio == "draft-inclusive":
        pool_ratio = (
            loaded.util.draft_inclusive_pool_ratio(POOL_RATIO, specs, MAX_SEQ_LEN, TOKENS_PER_BLOCK)
            or POOL_RATIO
        )
    return manager, _host_config(_HOST, pool_ratio)


def _balanced_quotas(loaded, monkeypatch, stages, remote_quota=_REMOTE):
    kvm = loaded.kvm
    method = kvm.KVCacheManagerV2._pp_stage_balanced_host_quota

    def call(rank):
        manager, config = stages[rank]
        return method(
            manager,
            config,
            _mapping(loaded, pp_size=4, rank=rank),
            device_quota=_DEVICE[rank],
            host_quota=_HOST,
            remote_quota=remote_quota,
        )

    sent = []
    for rank in range(4):
        gather = _Gather()
        monkeypatch.setattr(kvm, "Distributed", gather)
        call(rank)
        sent.append(gather.sent[0])
    monkeypatch.setattr(kvm, "Distributed", _Gather(sent))
    loaded.logger.records.clear()
    return [call(rank) for rank in range(4)], sent


def test_balanced_host_tiers_compose_with_the_draft_stage_pool_ratio(loaded, monkeypatch):
    """Every stage ends with the same full-attention tokens over device, host and remote tiers."""
    _server(loaded)
    stages = [_stage_manager(loaded, rank) for rank in range(4)]
    quotas, sent = _balanced_quotas(loaded, monkeypatch, stages)

    shares = [config.pool_ratio[1] for _, config in stages]
    assert shares[:3] == [POOL_RATIO[1]] * 3
    assert shares[3] == pytest.approx(0.748, abs=5e-4)
    assert [entry[1] for entry in sent] == pytest.approx([_FULL / share for share in shares])
    assert [entry[2:] for entry in sent] == [(_HOST, _DEVICE[r] + _REMOTE) for r in range(4)]

    granularity = loaded.balance.HOST_TIER_GRANULARITY
    assert all(quota % granularity == 0 for quota in quotas)
    assert 4 * _HOST - 4 * granularity <= sum(quotas) <= 4 * _HOST
    assert quotas[3] > _HOST > max(quotas[:3])
    tokens = [(quota + _DEVICE[r] + _REMOTE) * shares[r] / _FULL for r, quota in enumerate(quotas)]
    assert max(tokens) - min(tokens) <= granularity / _FULL
    before = [(_HOST + _DEVICE[r] + _REMOTE) * shares[r] / _FULL for r in range(4)]
    assert min(before) == before[3] < min(tokens)

    infos = [message for level, message in loaded.logger.records if level == "info"]
    # Rank 0 lists the plan of every rank; every rank names its own stage.
    assert sum(message.startswith(f"{_BALANCE_TAG} rank ") for message in infos) == 4
    assert [message.split(" host tier")[0] for message in infos if "pipeline stage" in message] == [
        f"{_BALANCE_TAG} pipeline stage {rank}" for rank in range(4)
    ]


def test_prefetch_sizes_itself_by_the_claim_caps(loaded, monkeypatch):
    """A prefetch through a transient cache stops where the first chunk's claim will: at the
    stage agreement's cap and at the drafter window guard's limit (asked without counting a
    claim); a zero cap prefetches nothing. Without either it takes the full match."""
    kvm = loaded.kvm
    monkeypatch.setattr(kvm, "ReuseScope", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(kvm, "CACHE_LEVEL1", 1, raising=False)
    limits, created, counted = [], [], []
    guard_cap = {"limit": None}

    class _Guard:
        def claim_limit(self, tokens, prompt_len, seed, count=True, owner=None):
            counted.append(count)
            return guard_cap["limit"]

    class _Cache:
        def prefetch(self, level):
            return True

        def close(self):
            pass

    def create_kv_cache(scope, tokens):
        created.append(tokens)
        return _Cache()

    def context_reuse_tokens(req, reuse_limit=None):
        limits.append(reuse_limit)
        return [0] * (100 if reuse_limit is None else reuse_limit)

    manager = SimpleNamespace(
        enable_block_reuse=True,
        _draft_context_guard=None,
        _draft_context_held_keys=False,
        ctx_diag=None,
        _derive_reuse_salt=lambda salt: 0,
        _draft_context_scope=lambda req: ((), 0),
        _context_reuse_tokens=context_reuse_tokens,
        impl=SimpleNamespace(create_kv_cache=create_kv_cache),
    )
    for name in ("_prefetch_claim_limit", "_draft_context_claim_limit"):
        setattr(manager, name, types.MethodType(getattr(kvm.KVCacheManagerV2, name), manager))
    prefetch = kvm.KVCacheManagerV2.prefetch_for_context_tokens

    def first_chunk(pp_cap=None):
        return SimpleNamespace(
            py_request_id=1,
            is_first_context_chunk=True,
            py_pp_reuse_cap=pp_cap,
            lora_task_id=None,
            cache_salt=None,
            multimodal_hashes=None,
            prompt_len=100,
        )

    assert prefetch(manager, [first_chunk()]) and limits == [None]
    assert prefetch(manager, [first_chunk(64)]) and limits[-1] == 64
    manager._draft_context_guard = _Guard()
    guard_cap["limit"] = 32
    assert prefetch(manager, [first_chunk(64)]) and limits[-1] == 32
    guard_cap["limit"] = None
    assert prefetch(manager, [first_chunk(64)]) and limits[-1] == 64
    assert counted == [False, False]
    created.clear()
    assert prefetch(manager, [first_chunk(0)]) and created == []


def test_balanced_host_tiers_fall_back_together(loaded, monkeypatch):
    """A stage that cannot describe its tiers keeps every rank at the configured size."""
    _server(loaded)
    stages = [_stage_manager(loaded, rank) for rank in range(4)]
    stages[1] = _stage_manager(loaded, 1, pool_ratio=None)
    quotas, sent = _balanced_quotas(loaded, monkeypatch, stages)
    assert sent[1] is None
    assert quotas == [_HOST] * 4
    assert [level for level, _ in loaded.logger.records] == ["warning"] * 4


def test_equal_stages_keep_the_configured_host_tier(loaded, monkeypatch):
    """One full-attention share and one device quota on every stage: nothing moves."""
    _server(loaded)
    stages = [_stage_manager(loaded, rank, pool_ratio=POOL_RATIO) for rank in range(4)]
    monkeypatch.setattr(sys.modules[__name__], "_DEVICE", [_DEVICE[0]] * 4)
    quotas, _ = _balanced_quotas(loaded, monkeypatch, stages)
    assert quotas == [_HOST] * 4


# --------------------------------------------------------------------------
# The kept drafter context covers what a generation server's drafter reads
# --------------------------------------------------------------------------
#: The released drafter's attention layout: non-causal sliding layers of WINDOW tokens.
_RELEASED_DRAFT_LAYOUT = dict(
    is_causal=False,
    layer_types=["sliding_attention"] * DRAFT_LAYERS,
    sliding_window=WINDOW,
    use_sliding_window=True,
    max_window_layers=DRAFT_LAYERS,
)
#: A context server's reuse rewind (``swa_endpoint_rewind_tokens``).
_REWIND = 512
_CONFIG_UTILS = "tensorrt_llm._torch.pyexecutor.config_utils"


@pytest.fixture
def window_rules(loaded, monkeypatch):
    """The attention-window rules both servers derive from the drafter config, executed from
    their files: ``config_utils.py`` (patched into ``_util`` and into the stubbed module) and
    ``modeling_dflash.py`` (the generation server's block attention), loaded after the patch."""
    config_utils = _exec_file(
        "tensorrt_llm._torch.pyexecutor._config_utils_under_test",
        "tensorrt_llm/_torch/pyexecutor/config_utils.py",
    )
    names = (
        "get_layer_attention_window",
        "_is_sliding_attention_layer",
        "uses_vswa_kv_cache_layout",
    )
    for name in names:
        monkeypatch.setattr(sys.modules[_CONFIG_UTILS], name, getattr(config_utils, name))
        monkeypatch.setattr(loaded.util, name, getattr(config_utils, name))
    dflash = _exec_file(
        "tensorrt_llm._torch.models._modeling_dflash_under_test",
        "tensorrt_llm/_torch/models/modeling_dflash.py",
    )
    return dflash


def _context_server_keep(loaded, draft_layout):
    """The last stage's drafter context window and keep count, as the KV cache creator and the
    manager derive them from a drafter of ``draft_layout`` over the released target's window
    pattern, with the ``_REWIND`` reuse rewind and block size."""
    creator = _creator(loaded)
    # The creator's own derivation of the draft windows, from the drafter config.
    del creator._get_draft_max_attention_window
    for key, value in draft_layout.items():
        setattr(creator._draft_config.pretrained_config, key, value)
    window = creator._pp_dflash_draft_window(
        SimpleNamespace(max_attention_window=_target_window_pattern())
    )
    if window is None:
        return None, None
    manager = object.__new__(loaded.kvm.KVCacheManagerV2)
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.enable_block_reuse = False
    manager._swa_endpoint_rewind = _REWIND
    manager.kv_connector_manager = None
    return window, manager.enable_draft_context_window(window)


def _generation_windows(dflash, draft_layout):
    """Each drafter layer's ``(causal, (left, right))`` block attention on a generation server,
    from the drafter's own window resolution (a drafter whose ``dflash_config`` sets no
    ``use_swa``, as the released one)."""
    drafter = object.__new__(dflash.DFlashForCausalLM)
    torch.nn.Module.__init__(drafter)
    drafter.config = SimpleNamespace(num_hidden_layers=DRAFT_LAYERS, **draft_layout)
    drafter._is_dflash2 = True
    drafter._sliding_layers_causal = False
    layer_types = draft_layout.get("layer_types")
    drafter._layer_windows = [
        dflash.dspark_layer_window_size(False, 0, layer_types, layer)
        for layer in range(DRAFT_LAYERS)
    ]
    return [drafter._resolve_block_attention(layer) for layer in range(DRAFT_LAYERS)]


def test_kept_drafter_context_covers_the_generation_window_rewind_and_a_block(loaded, window_rules):
    """The released drafter: the context server keeps at least the generation server's
    drafter window, the reuse rewind and one KV block of every prompt's drafter context, and
    guards reuse with a window that covers every row the generation server reads."""
    window, keep = _context_server_keep(loaded, _RELEASED_DRAFT_LAYOUT)
    generation = _generation_windows(window_rules, _RELEASED_DRAFT_LAYOUT)
    assert (window, keep) == (WINDOW, WINDOW + _REWIND + TOKENS_PER_BLOCK)
    for causal, (left, right) in generation:
        # A block row at the prompt end reads ``left`` context rows below it: a window of
        # ``left + 1`` positions.
        assert (causal, left, right) == (False, WINDOW - 1, WINDOW - 1)
        assert window >= left + 1
        assert keep >= (left + 1) + _REWIND + TOKENS_PER_BLOCK


def test_a_drafter_without_a_window_keeps_every_row(loaded, window_rules):
    """A drafter config without an attention layout over the released target's window pattern:
    the generation server's drafter attends the whole context, and the context server trims
    nothing."""
    layout = dict(layer_types=None)
    window, keep = _context_server_keep(loaded, layout)
    generation = _generation_windows(window_rules, layout)
    assert all(left < 0 for _, (left, _) in generation)
    assert (window, keep) == (None, None)


# --------------------------------------------------------------------------
# FP8 draft KV (TRTLLM_DFLASH_FP8_DRAFT_KV)
# --------------------------------------------------------------------------
_FP8_LOG = (
    "DFlash draft KV layers [62, 63, 64, 65, 66] stored as fp8 e4m3 (scale 1.0, saturating), "
    "coalescing group 3."
)


def _fp8_specs(loaded, monkeypatch, rank):
    _server(loaded)
    monkeypatch.setenv(_FP8_KNOB, "1")
    return _specs(_stage_creator(loaded, rank))


@pytest.mark.parametrize("knob", [None, "0", "true"])
def test_fp8_knob_unset_keeps_bf16_draft_layers(loaded, monkeypatch, knob):
    _server(loaded)
    if knob is not None:
        monkeypatch.setenv(_FP8_KNOB, knob)
    loaded.logger.records.clear()
    for rank in range(4):
        creator = _stage_creator(loaded, rank)
        assert _specs(creator) == _hand_built_stage_specs(loaded, rank)
        creator._check_fp8_draft_kv_request()
    assert not any("fp8 e4m3" in message for _, message in loaded.logger.records)


def test_fp8_draft_layers_change_only_their_dtype(loaded, monkeypatch):
    kvm = loaded.kvm
    loaded.logger.records.clear()
    for rank in range(4):
        specs = _fp8_specs(loaded, monkeypatch, rank)
        bf16 = _hand_built_stage_specs(loaded, rank)
        expected = tuple(
            replace(spec, dtype=DataType.FP8) if spec.owner is kvm.KVCacheLayerOwner.DRAFT else spec
            for spec in bf16
        )
        assert specs == expected
    draft = [spec for spec in specs if spec.owner is kvm.KVCacheLayerOwner.DRAFT]
    assert [spec.global_layer_id for spec in draft] == list(range(62, 67))
    # 8 KV heads x 128 x 1 B for each of K and V; BF16 held 4,096.
    assert [spec.bytes_per_token() for spec in draft] == [2048] * DRAFT_LAYERS
    # One line, on the stage that holds the draft layers.
    assert [message for _, message in loaded.logger.records if "fp8 e4m3" in message] == [_FP8_LOG]


def test_fp8_stage_kv_bytes_and_layout_line(loaded, monkeypatch):
    util = loaded.util
    stages = [
        util.stage_kv_bytes_per_token(_fp8_specs(loaded, monkeypatch, rank), MAX_SEQ_LEN)
        for rank in range(4)
    ]
    assert [stage.full for stage in stages] == [8192] * 4
    assert [stage.sliding_target for stage in stages] == [22528, 24576, 24576, 22528]
    assert [stage.sliding_draft for stage in stages] == [0, 0, 0, 10240]
    assert stages[3].sliding_growth == pytest.approx(32768 / 22528)
    creator = _stage_creator(loaded, 3)
    specs = _specs(creator)
    loaded.logger.records.clear()
    creator._log_pp_dflash_stage_layout(specs)
    assert loaded.logger.records == [
        (
            "info",
            "DFlash on pipeline stage 3 of 4: draft KV layers [62, 63, 64, 65, 66], 20 KV "
            "layers; KV bytes per token: full 8192, sliding target 22528, sliding draft "
            "10240 (windows [2048], sliding growth 1.455).",
        )
    ]


@pytest.mark.parametrize(
    "pool_ratio,expected",
    [([0.125, 0.875], [0.172043, 0.827957]), ([0.15, 0.85], [0.204255, 0.795745])],
)
def test_fp8_draft_stage_pool_ratio(loaded, monkeypatch, pool_ratio, expected):
    specs = _fp8_specs(loaded, monkeypatch, 3)
    ratio = loaded.util.draft_inclusive_pool_ratio(pool_ratio, specs, MAX_SEQ_LEN, TOKENS_PER_BLOCK)
    assert [round(share, 6) for share in ratio] == expected
    for rank in range(3):
        stage = _fp8_specs(loaded, monkeypatch, rank)
        assert (
            loaded.util.draft_inclusive_pool_ratio(pool_ratio, stage, MAX_SEQ_LEN, TOKENS_PER_BLOCK)
            is None
        )


def test_fp8_draft_buffers_have_their_own_coalescing_group(loaded, monkeypatch):
    kvm = loaded.kvm
    specs = _fp8_specs(loaded, monkeypatch, 3)
    bf16 = _hand_built_stage_specs(loaded, 3)
    for spec in specs:
        for role in spec.buffer_roles:
            group = kvm._layer_buffer_coalescing_group(spec, role)
            if spec.owner is kvm.KVCacheLayerOwner.DRAFT:
                assert group == kvm.FP8_DRAFT_COALESCING_GROUP == 3
                assert kvm._coalescing_group_kwargs(group) == {"coalescing_group": 3}
            else:
                assert group == 0
                assert kvm._coalescing_group_kwargs(group) == {}
    # The default layout passes no group to the bindings.
    assert {
        kvm._layer_buffer_coalescing_group(spec, role)
        for spec in bf16
        for role in spec.buffer_roles
    } == {0}
    # Without the knob, FP8 draft layer specs get no group of their own either.
    monkeypatch.delenv(_FP8_KNOB)
    assert {
        kvm._layer_buffer_coalescing_group(spec, role)
        for spec in specs
        for role in spec.buffer_roles
    } == {0}
    # The FP8 draft K/V buffer is exactly the size of the target's FP8 one: without its own
    # group it would coalesce into the target's pool.
    target = next(spec for spec in specs if spec.owner is kvm.KVCacheLayerOwner.TARGET)
    draft = next(spec for spec in specs if spec.owner is kvm.KVCacheLayerOwner.DRAFT)
    assert target.bytes_per_token(kvm.Role.KEY) == draft.bytes_per_token(kvm.Role.KEY) == 1024
    # The pool groups the cost model counts are the same as with BF16 draft layers.
    assert loaded.util._get_num_storage_pool_groups(
        specs, TOKENS_PER_BLOCK
    ) == loaded.util._get_num_storage_pool_groups(bf16, TOKENS_PER_BLOCK)


def test_nvfp4_coalescing_groups_are_unchanged(loaded):
    kvm = loaded.kvm
    roles = (kvm.Role.KEY, kvm.Role.VALUE, kvm.Role.KEY_BLOCK_SCALE, kvm.Role.VALUE_BLOCK_SCALE)
    for owner in (kvm.KVCacheLayerOwner.TARGET, kvm.KVCacheLayerOwner.DRAFT):
        spec = kvm.KVCacheLayerSpec(
            global_layer_id=0,
            owner=owner,
            dtype=DataType.NVFP4,
            num_kv_heads=8,
            head_dim=128,
            buffer_roles=roles,
            sliding_window_size=WINDOW,
        )
        assert [kvm._layer_buffer_coalescing_group(spec, role) for role in roles] == [1, 1, 2, 2]


def test_coalescing_group_needs_the_binding_field(loaded, monkeypatch):
    kvm = loaded.kvm
    monkeypatch.setattr(kvm, "BufferConfig", type("BufferConfig", (), {}))
    assert kvm._coalescing_group_kwargs(0) == {}
    with pytest.raises(NotImplementedError, match="coalescing group 3"):
        kvm._coalescing_group_kwargs(kvm.FP8_DRAFT_COALESCING_GROUP)


def test_fp8_knob_refuses_a_non_fp8_target(loaded, monkeypatch):
    _server(loaded)
    monkeypatch.setenv(_FP8_KNOB, "1")
    creator = _stage_creator(loaded, 3)
    target = creator._model_engine.model.model_config
    creator._model_engine.model.model_config = SimpleNamespace(
        pretrained_config=target.pretrained_config, quant_config=_quant_config(fp8_kv=False)
    )
    with pytest.raises(ValueError, match="TRTLLM_DFLASH_FP8_DRAFT_KV=1 needs .* not FP8"):
        _specs(creator)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("vanilla", "attention_backend is VANILLA"),
        ("no_spec", "speculative decoding is not enabled"),
        ("unshared", "does not share the draft KV"),
    ],
)
def test_fp8_knob_refuses_where_no_draft_layer_shares_the_manager(
    loaded, monkeypatch, case, reason
):
    monkeypatch.setenv(_FP8_KNOB, "1")
    if case == "vanilla":
        creator = _creator(
            loaded,
            pp_size=1,
            rank=0,
            partition=None,
            is_disagg=True,
            spec_config=_spec_config(backend="VANILLA"),
        )
    elif case == "no_spec":
        creator = _creator(loaded, pp_size=1, rank=0, partition=None)
        creator._speculative_config = None
    else:
        creator = _creator(loaded, pp_size=1, rank=0, partition=None)
        monkeypatch.setattr(loaded.util, "disagg_shares_draft_kv", lambda spec_config: False)
    assert not creator._should_unify_dflash_kv_cache()
    with pytest.raises(ValueError, match=f"TRTLLM_DFLASH_FP8_DRAFT_KV=1 needs .*{reason}"):
        creator._check_fp8_draft_kv_request()
    monkeypatch.delenv(_FP8_KNOB)
    creator._check_fp8_draft_kv_request()


def test_fp8_knob_passes_on_every_stage_of_the_placement(loaded, monkeypatch):
    for rank in range(4):
        _fp8_specs(loaded, monkeypatch, rank)
        creator = _stage_creator(loaded, rank)
        assert creator._should_unify_dflash_kv_cache()
        creator._check_fp8_draft_kv_request()
