# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the attention fast paths serve the decode steps of the composed model, not only
compose, on real weights in the TP8 decode shape.

Run like ``test_modeling_v2_musespark_open_decode_forward.py`` (eight MPI ranks on two
NVLink-connected four-GPU nodes, sm_103 or sm_107, the checkpoint in
``MUSESPARK_REFERENCE_CKPT``, the serving configuration's switches), whose model build, KV
cache, decode runs and entry-point counts this test reuses with two arms: the built-in model,
and ``attention`` = every attention fast path (``preattn_prep``, ``swa_paged_decode``,
``swa_block``, ``full_attention_prep_append``, ``full_attention_split_kv``,
``full_attention_prep_fold``), adopted for the device's architecture as the shell adopts it.

An attention path is installed when the model is composed but decides per decode step
whether it runs. Those per-step checks read the attention backend's FMHA libraries (the
trtllm-gen one bounds the KV-pool view, and the full-attention paths intercept its decode
call), so a composition whose layers lack that library composes every path and serves
none. Here, at decode steps of 1..8 rows, eager and replayed from a CUDA graph:

* every attention path runs its kernels at every call site (``_expected_counts``: the
  sliding-window block at the four sliding-window layers, the prep-fold kernel at the two
  full-attention layers);
* no decode step's attention path declines for want of the trtllm-gen FMHA library
  (``no trtllm-gen FMHA library``), and at 1..8 rows no captured step declines at all;
* every captured decode graph of 1..8 rows logs its fold census with both full-attention
  layers folded and no warning;
* each layer's FMHA manager holds flashinfer's trtllm-gen FMHA: the public library on
  sm_103, ``attention.DecodeTrtllmGenFmha`` in front on sm_107, where the built-in arm holds
  none (the state in which the paths composed but did not serve).

At 9 and 16 rows the paths decline by shape and the counts are zero.
"""

import importlib.util
import os

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)
from tensorrt_llm._torch.attention.backends.fmha import flashinfer_trtllm_gen as fi_gen


def _decode_forward():
    """The decode-forward composition test module, loaded from beside this file as a library."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "test_modeling_v2_musespark_open_decode_forward.py")
    spec = importlib.util.spec_from_file_location("_decode_forward_for_served", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _decode_forward()


def _fixture_body(fixture):
    """The generator function behind a pytest fixture, across pytest's wrappings of it."""
    get = getattr(fixture, "_get_wrapped_function", None)
    if get is not None:
        return get()
    wrapped = getattr(fixture, "__pytest_wrapped__", None)
    if wrapped is not None:
        return wrapped.obj
    return getattr(fixture, "__wrapped__", fixture)


#: Every attention fast path (``swa_block`` and ``prep_fold`` close over the other four).
ATTENTION = "swa_block,prep_fold"
#: (arm, the arm's ids (``_compose_plans.plan_of_ids``), the arm whose bits it must equal).
ARMS = (("builtin", None, None), ("attention", ATTENTION, None))
FULL_LAYERS = len(C.FULL_LAYER_INDICES)
#: The decline the per-arch FMHA selection removes.
NO_LIBRARY = "no trtllm-gen FMHA library"
#: Why each decode call's attention path declined: (arm, mode, rows) -> {reason: calls}.
_DECLINES: dict = {}
_WHERE: list = [None]


@pytest.fixture(scope="module")
def env():
    """The decode-forward test's environment with this file's two arms; the sliding-window
    block and the full-attention kernels counted, every decline reason recorded."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.swa_block import (  # noqa: E501
        attention as swa_attention_block,
    )

    saved = (
        C.ARMS,
        C.COMBINED,
        C.DETERMINISM,
        C.SPLIT_KV_SELECTED,
        C.PREP_FOLD_SELECTED,
        C.PREP_APPEND_ARMS,
    )
    C.ARMS, C.COMBINED, C.DETERMINISM = ARMS, {}, ()
    C.SPLIT_KV_SELECTED = C.PREP_FOLD_SELECTED = C.PREP_APPEND_ARMS = True
    block = swa_attention_block.swa_block_attention
    swa_attention_block.swa_block_attention = C._Counted("swa_block_attention", block)
    declines = [
        (target_attention.MuseSparkOpenTepDecodeAttention, "_paged_decode_decline"),
        (full_attention.FullAppendAttention, "_prep_append_decline"),
    ]
    originals = [(cls, name, getattr(cls, name)) for cls, name in declines]

    def recording(fn):
        def decline(self, *args, **kwargs):
            reason = fn(self, *args, **kwargs)
            if reason is not None and _WHERE[0] is not None:
                seen = _DECLINES.setdefault(_WHERE[0], {})
                seen[reason] = seen.get(reason, 0) + 1
            return reason

        return decline

    for cls, name, fn in originals:
        setattr(cls, name, recording(fn))
    try:
        yield from _fixture_body(C.env)()
    finally:
        for cls, name, fn in originals:
            setattr(cls, name, fn)
        swa_attention_block.swa_block_attention = block
        (
            C.ARMS,
            C.COMBINED,
            C.DETERMINISM,
            C.SPLIT_KV_SELECTED,
            C.PREP_FOLD_SELECTED,
            C.PREP_APPEND_ARMS,
        ) = saved


def _compare(env, rows: int, mode: str) -> None:
    """``_compare_arms`` with each arm's decline reasons recorded per run."""
    real_run = C._run

    def run(e, model, rows_, graph=False):
        arm = next(name for name, m in e.models.items() if m is model)
        _WHERE[0] = (arm, "graph" if graph else "eager", rows_)
        try:
            return real_run(e, model, rows_, graph)
        finally:
            _WHERE[0] = None

    C._run = run
    try:
        C._compare_arms(env, rows, mode)
    finally:
        C._run = real_run


def test_every_layer_holds_the_trtllm_gen_library(env):
    """The attention arm's managers hold flashinfer's trtllm-gen FMHA: the public library on
    sm_103, ``DecodeTrtllmGenFmha`` in front on sm_107; the built-in arm on sm_107 holds none."""
    capability = torch.cuda.get_device_capability()
    problems = []
    for arm, model in env.models.items():
        for index, layer in enumerate(model.layers):
            libs = layer.self_attn.attn._fmha_manager.fmha_libs
            found = target_attention.trtllm_gen_fmha(layer.self_attn.attn)
            if arm == "attention" and capability == (10, 7):
                if type(libs[0]) is not target_attention.DecodeTrtllmGenFmha:
                    problems.append(f"{arm} layer {index}: {[type(lib).__name__ for lib in libs]}")
            elif arm == "attention" or capability == (10, 3):
                if type(found) is not fi_gen.FlashInferTrtllmGenFmha:
                    problems.append(f"{arm} layer {index}: {[type(lib).__name__ for lib in libs]}")
            elif found is not None:
                problems.append(f"{arm} layer {index}: the built-in arm holds {found}")
    everyone = env.comm.allgather(problems)
    assert not any(everyone), everyone


@pytest.mark.parametrize("rows", C.ROWS)
def test_the_attention_paths_serve_in_eager_decode(env, rows):
    _compare(env, rows, "eager")


@pytest.mark.parametrize("rows", C.ROWS)
def test_the_attention_paths_serve_in_cuda_graphs(env, rows):
    _compare(env, rows, "graph")


def test_no_step_declines_for_want_of_the_library(env):
    """No decode call of either mode declined for want of the trtllm-gen FMHA library, and no
    captured step of 1..8 rows declined at all."""
    if not _DECLINES and not C._RESULTS:
        pytest.skip("no comparison ran in this session")
    problems = []
    for (arm, mode, rows), reasons in sorted(_DECLINES.items()):
        if arm != "attention":
            continue
        if NO_LIBRARY in reasons:
            problems.append(f"{mode} {rows} rows: {reasons}")
        elif mode == "graph" and rows <= 8:
            problems.append(f"{mode} {rows} rows declined: {reasons}")
    everyone = env.comm.allgather(problems)
    assert not any(everyone), everyone


def test_every_captured_graph_folds_both_full_layers(env):
    """The fold census of the attention arm: every captured decode graph of 1..8 rows logged
    both full-attention layers folded, and no capture warned."""
    captured = sorted(
        rows for (mode, arm, rows) in C._RESULTS if mode == "graph" and arm == "attention"
    )
    if not captured:
        pytest.skip("no CUDA-graph comparison ran in this session")
    census = next(
        layer.self_attn._prep_append_census
        for layer in env.models["attention"].layers
        if not layer.self_attn.is_sliding
    )
    reports = list(census.reports)
    folded = [r for r in reports if r[1] == r[2] == FULL_LAYERS and r[3] == "info"]
    warned = [r[4] for r in reports if r[3] == "warning"]
    want = sum(1 for rows in captured if rows <= 8)
    problems = []
    if warned:
        problems.append(f"warned: {warned[:2]}")
    if len(folded) < want:
        problems.append(f"{len(folded)} fully folded captures of {len(reports)}, want {want}")
    everyone = env.comm.allgather(problems)
    assert not any(everyone), everyone
