# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-architecture decode FMHA selection (``attention.select_decode_fmha``).

Host-only. The attention paths of ``attention.TRTLLM_GEN_DECODE_PATHS`` need flashinfer's
trtllm-gen FMHA in each target layer's FMHA manager, and the full-attention ones need the
manager to select it for decode steps. On sm_103 the public library list already does; on
sm_107 the selection puts a ``DecodeTrtllmGenFmha`` in front of each layer's libraries.
These tests drive the selection on stand-in attention backends, the generation-only offer
through the real ``FmhaManager`` selection with stand-in libraries, the two lookups the
attention paths make, and the core's call of the selection per plan and architecture. That
the composition then serves on sm_107 is the GPU test's business
(``test_modeling_v2_musespark_open_compose_served.py``).
"""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)
from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi_gen
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.manager import FmhaManager

pytestmark = pytest.mark.cpu_only

_DECODE = target_attention.DecodeTrtllmGenFmha
_SM103, _SM107 = (10, 3), (10, 7)


class _Backend:
    """A ``TrtllmAttention`` as far as the selection and the library constructor read it."""

    def __init__(self, **overrides) -> None:
        self.num_heads = 16
        self.num_kv_heads = 2
        self.head_dim = 64
        self.v_head_dim = None
        self.kv_lora_rank = None
        self.is_mla_enable = False
        self.skip_correction_threshold = 0.0
        self.sparse_params = None
        self._fmha_manager = SimpleNamespace(fmha_libs=["public"])
        vars(self).update(overrides)


def _layers(n: int = 4, **overrides):
    return [
        SimpleNamespace(self_attn=SimpleNamespace(layer_idx=i, attn=_Backend(**overrides)))
        for i in range(n)
    ]


@pytest.fixture
def flashinfer_ready(monkeypatch):
    """flashinfer installed and the binding complete, whatever this host has."""
    monkeypatch.setattr(target_attention._flashinfer_utils, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(
        fi_gen.FlashInferTrtllmGenFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: [])
    )


def _libs(layers):
    return [layer.self_attn.attn._fmha_manager.fmha_libs for layer in layers]


# --------------------------------------------------------------------------
# The selection, per architecture
# --------------------------------------------------------------------------
def test_sm107_puts_the_decode_library_in_front_of_every_layer(flashinfer_ready):
    layers = _layers()
    assert target_attention.select_decode_fmha(layers, _SM107) == 4
    for layer, libs in zip(layers, _libs(layers)):
        assert len(libs) == 2 and libs[1] == "public"
        assert type(libs[0]) is _DECODE and libs[0].attn is layer.self_attn.attn
    # Idempotent: a manager that holds the library already is left alone.
    assert target_attention.select_decode_fmha(layers, _SM107) == 0
    assert [len(libs) for libs in _libs(layers)] == [2] * 4


@pytest.mark.parametrize("capability", (_SM103, None), ids=("sm103", "no_device"))
def test_sm103_and_no_device_change_nothing(flashinfer_ready, capability):
    layers = _layers()
    assert target_attention.select_decode_fmha(layers, capability) == 0
    assert _libs(layers) == [["public"]] * 4


def test_another_architecture_has_no_selection(flashinfer_ready):
    layers = _layers()
    with pytest.raises(ValueError, match="no decode FMHA selection is written for sm100"):
        target_attention.select_decode_fmha(layers, (10, 0))
    assert _libs(layers) == [["public"]] * 4


def test_sm107_keeps_a_public_trtllm_gen_library(flashinfer_ready):
    """A manager whose public list admitted flashinfer's trtllm-gen FMHA keeps its selection."""
    layers = _layers(1)
    attn = layers[0].self_attn.attn
    public = fi_gen.FlashInferTrtllmGenFmha(attn)
    attn._fmha_manager.fmha_libs = [public]
    assert target_attention.select_decode_fmha(layers, _SM107) == 0
    assert attn._fmha_manager.fmha_libs == [public]


@pytest.mark.parametrize(
    "overrides, reason",
    (
        ({"skip_correction_threshold": 0.5}, "skip-correction"),
        ({"is_mla_enable": True}, "MLA"),
        ({"sparse_params": object()}, "sparse attention"),
        ({"num_kv_heads": 3}, "16 query heads over 3 KV heads"),
    ),
)
def test_sm107_refuses_a_layer_the_library_cannot_serve_and_changes_nothing(
    flashinfer_ready, overrides, reason
):
    layers = _layers()
    layers[2].self_attn.attn = _Backend(**overrides)
    with pytest.raises(ValueError, match=f"layer 2: {reason}"):
        target_attention.select_decode_fmha(layers, _SM107)
    assert _libs(layers) == [["public"]] * 4


def test_sm107_refuses_without_flashinfer_or_its_binding(monkeypatch):
    layers = _layers(2)
    monkeypatch.setattr(target_attention._flashinfer_utils, "IS_FLASHINFER_AVAILABLE", False)
    with pytest.raises(ValueError, match="layer 0: flashinfer is not installed"):
        target_attention.select_decode_fmha(layers, _SM107)
    monkeypatch.setattr(target_attention._flashinfer_utils, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(
        fi_gen.FlashInferTrtllmGenFmha,
        "_missing_fused_nanobind_ops",
        staticmethod(lambda: ["trtllm_gen_generation_preprocess"]),
    )
    with pytest.raises(ValueError, match="lacks trtllm_gen_generation_preprocess"):
        target_attention.select_decode_fmha(layers, _SM107)
    assert _libs(layers) == [["public"]] * 2


# --------------------------------------------------------------------------
# The generation-only offer, through the manager's selection
# --------------------------------------------------------------------------
class _Fallback(Fmha):
    """A stand-in for the public ``thop`` library: one library for the whole request."""

    def _is_supported(self, q, k, v, metadata, forward_args, *, phase=None) -> bool:
        return True

    def forward(self, q, k, v, metadata, forward_args) -> None:
        raise AssertionError("not run")


@pytest.fixture
def library_checks(monkeypatch):
    """The trtllm-gen library's own request checks, recorded and accepting."""
    seen = []

    def checks(self, q, k, v, metadata, forward_args, *, phase=None):
        seen.append(phase)
        return True

    monkeypatch.setattr(fi_gen.FlashInferTrtllmGenFmha, "_is_supported", checks)
    return seen


def _forward_args():
    return SimpleNamespace(sparse_runtime_params=SimpleNamespace(block_sparse_inputs=None))


@pytest.mark.parametrize("phase", (None, FmhaPhase.CONTEXT))
def test_the_decode_library_offers_no_context_phase(library_checks, phase):
    lib = _DECODE(_Backend())
    assert not lib.is_supported(None, None, None, None, _forward_args(), phase=phase)
    assert library_checks == []


def test_the_decode_library_asks_its_own_checks_for_the_generation_phase(library_checks):
    lib = _DECODE(_Backend())
    phase = FmhaPhase.GENERATION
    assert lib.is_supported(None, None, None, None, _forward_args(), phase=phase)
    assert library_checks == [FmhaPhase.GENERATION]


@pytest.mark.parametrize(
    "contexts, generations, picked",
    ((0, 1, "decode"), (0, 8, "decode"), (1, 0, "fallback"), (1, 1, "fallback")),
    ids=("generation_only", "generation_batch", "prefill", "mixed"),
)
def test_the_manager_selects_the_decode_library_for_generation_only_steps(
    library_checks, contexts, generations, picked
):
    attn = _Backend(is_mla_enable=False)
    decode, fallback = _DECODE(attn), _Fallback(attn)
    manager = FmhaManager.__new__(FmhaManager)
    manager.fmha_libs = [decode, fallback]
    metadata = SimpleNamespace(num_contexts=contexts, num_generations=generations)
    got = manager._select_uncached(attn, None, None, None, metadata, _forward_args())
    assert got is {"decode": decode, "fallback": fallback}[picked]


def test_the_manager_falls_through_when_the_library_refuses_the_step(monkeypatch):
    monkeypatch.setattr(
        fi_gen.FlashInferTrtllmGenFmha, "_is_supported", lambda self, *a, phase=None: False
    )
    attn = _Backend()
    fallback = _Fallback(attn)
    manager = FmhaManager.__new__(FmhaManager)
    manager.fmha_libs = [_DECODE(attn), fallback]
    metadata = SimpleNamespace(num_contexts=0, num_generations=1)
    assert manager._select_uncached(attn, None, None, None, metadata, _forward_args()) is fallback


# --------------------------------------------------------------------------
# The lookups the attention paths make
# --------------------------------------------------------------------------
def test_both_lookups_find_the_decode_library():
    attn = _Backend()
    decode = _DECODE(attn)
    attn._fmha_manager.fmha_libs = ["public", decode]
    assert target_attention.trtllm_gen_fmha(attn) is decode
    holder = SimpleNamespace(attn=attn)
    assert target_attention.MuseSparkOpenTepDecodeAttention._total_num_blocks_fmha(holder) is decode
    assert full_attention.FullAppendAttention._prep_append_fmha_lib(holder) is decode


def test_both_lookups_find_nothing_without_it():
    attn = _Backend()
    holder = SimpleNamespace(attn=attn)
    assert target_attention.trtllm_gen_fmha(attn) is None
    assert target_attention.trtllm_gen_fmha(SimpleNamespace()) is None
    assert target_attention.MuseSparkOpenTepDecodeAttention._total_num_blocks_fmha(holder) is None
    assert full_attention.FullAppendAttention._prep_append_fmha_lib(holder) is None


# --------------------------------------------------------------------------
# The core: which plans select, on which architecture
# --------------------------------------------------------------------------
def _prep_append_tests():
    """The prep-append host test module (its stand-in text model), loaded from beside this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        here,
        "kernels",
        "full_attention_prep_append",
        "test_modeling_v2_musespark_open_prep_append.py",
    )
    spec = importlib.util.spec_from_file_location("_prep_append_for_decode_fmha", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _prep_append_tests()


@pytest.fixture
def recorded(monkeypatch):
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)
    monkeypatch.setattr(full_attention, "preprocess_reachable", lambda: None)
    calls = []

    def select(layers, capability):
        calls.append((len(layers), capability))
        return 0

    monkeypatch.setattr(target_attention, "select_decode_fmha", select)
    return calls


def test_the_paths_that_need_the_library_are_the_attention_paths_but_the_prep():
    attention_paths = {
        "preattn_prep",
        "swa_paged_decode",
        "swa_block",
        "full_attention_prep_append",
        "full_attention_split_kv",
        "full_attention_prep_fold",
    }
    assert set(target_attention.TRTLLM_GEN_DECODE_PATHS) == attention_paths - {"preattn_prep"}
    assert set(target_attention.TRTLLM_GEN_DECODE_PATHS) <= dispatch.FAST_PATH_NAMES


@pytest.mark.parametrize("capability", (_SM103, _SM107))
@pytest.mark.parametrize(
    "paths",
    (("full_attention_prep_append",), ("swa_paged_decode",)),
    ids=("tep4_prep_append", "swa_paged_decode"),
)
def test_the_core_selects_for_the_planned_attention_paths(recorded, capability, paths):
    model = _P._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of(*paths), capability)
    assert recorded == [(4, capability)]


def test_the_core_does_not_select_without_those_paths(recorded):
    model = _P._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of("preattn_prep"), _SM107)
    model = _P._Model(4)
    modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.NO_PATHS, _SM107)
    assert recorded == []


def test_every_composition_with_an_attention_path_selects():
    """Each served composition carries a path of ``TRTLLM_GEN_DECODE_PATHS``, so on sm_107
    every one of them puts the decode library in front."""
    for key, composition in modeling.TARGET_PATHS.items():
        assert set(composition.served) & set(target_attention.TRTLLM_GEN_DECODE_PATHS), key
