# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the folded prep append and its prefetch trim on real weights in the TP8 decode
shape, bit for bit.

Run like ``test_modeling_v2_musespark_open_decode_forward.py`` (eight MPI ranks on two
NVLink-connected four-GPU GB300 nodes, the checkpoint in ``MUSESPARK_REFERENCE_CKPT``, the
serving configuration's switches), whose model build, KV cache, decode runs and comparisons
this test reuses with its own arms: the same six checkpoint layers (two of them
full-attention NoPE layers), the FP8 ``KVCacheManagerV2`` with 32-token pages,
one request prefilled past the window, then decode steps of M = 1..8 rows and the
fallback widths 9 and 16, eagerly and replayed from a CUDA graph per arm.

The folded prep append (``prep_append``) is bit-identical, so every arm with it must equal
the arm without it in every output of every step, every greedy token and every KV cache
byte: alone with ``preattn_prep`` (equal to the built-in model), and on the base set
(``preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch``). The folded prep
must run on both full-attention layers of every decode forward of at most 8 rows, and the
built-in preprocess must stand down for each of them (``FoldedPreprocess.outputs`` counted);
the ``preattn_prep`` prep then runs on the four sliding-window layers only. The prefetch
trim (``prep_append_prefetch_trim``, on the base set with the prep append) changes no bit
either, and the two full-attention layers' QKV projections fork no attention output prefetch
(``l2_prefetch`` counted). Both again on the tier-2 set (``TIER1_TARGET`` + ``TIER2_NEW``).
Every captured graph of an arm with the prep append logs its fold census (``FoldCensus``):
both full-attention layers fold at 1..8 rows, and no capture warns.

``MODELING_V2_COMPOSE_OUT=<dir>`` also writes rank 0's per-arm, per-width result table there.
"""

import importlib.util
import json
import os

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.full_attention_prep_append import (  # noqa: E501
    attention as full_attention,
)


def _decode_forward():
    """The decode-forward composition test module, loaded from beside this file as a library."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "test_modeling_v2_musespark_open_decode_forward.py")
    spec = importlib.util.spec_from_file_location("_decode_forward_for_prep_append", path)
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


#: (arm, the arm's ids (``_compose_plans.plan_of_ids``), the arm whose bits it must equal).
ARMS = (
    ("builtin", None, None),
    ("preattn_prep", "preattn_prep", "builtin"),
    ("preattn_prep_prep_append", "preattn_prep,prep_append", "builtin"),
    ("base", "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch", None),
    (
        "base_prep_append",
        "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch,prep_append",
        "base",
    ),
    (
        "base_prep_append_prefetch_trim",
        "preattn_prep,hop,gathers,ring_buddy,paged_decode_q8,gemm_prefetch,prep_append,prep_append_prefetch_trim",
        "base",
    ),
    ("tier2", C.TIER1_TARGET + "," + C.TIER2_NEW, None),
    (
        "tier2_prep_append_prefetch_trim",
        C.TIER1_TARGET + "," + C.TIER2_NEW + ",prep_append,prep_append_prefetch_trim",
        "tier2",
    ),
)
#: Model layers 1 and 5 (checkpoint 1 and 61) are the full-attention layers.
FULL_LAYERS = len(C.FULL_LAYER_INDICES)
#: Entry point -> (feature ids that run it, call sites per decode forward, widest width):
#: the composition test's, which counts the folded prep and the elided preprocess too.
ENTRIES = dict(C.ENTRIES)
assert ENTRIES["preattn_prep_append"] == ({"prep_append"}, FULL_LAYERS, 8)
assert ENTRIES["preprocess_elided"] == ({"prep_append"}, FULL_LAYERS, 8)


#: The composition test's expected counts: the early-release prefetch drops, and the folded
#: prep in place of the preattn_prep prep on the full-attention layers and the trimmed QKV
#: prefetch there.
_expected_counts = C._expected_counts


@pytest.fixture(scope="module")
def env():
    """The decode-forward test's environment with this file's arms; the folded prep's entry points
    counted."""
    saved = (C.ARMS, C.ENTRIES, C.COMBINED, C.DETERMINISM, C._expected_counts, C._metadata)
    C.ARMS, C.ENTRIES, C.COMBINED, C.DETERMINISM = ARMS, ENTRIES, {}, ("base_prep_append",)
    C._expected_counts = _expected_counts
    original_metadata = C._metadata

    def metadata(*args, **kwargs):
        # Eager steps build fresh metadata, whose attention workspace starts empty and is
        # grown by the first attention call: give it its size up front so the first
        # full-attention layer of an eager step folds too (the serving engine keeps one
        # metadata whose workspace is sized by the warm-up before any graph is captured).
        md = original_metadata(*args, **kwargs)
        for workspace in (md.workspace, md.cuda_graph_workspace):
            if workspace is not None and workspace.numel() * workspace.element_size() < (1 << 30):
                workspace.resize_((1 << 30) // workspace.element_size())
        return md

    C._metadata = metadata
    patched = [
        (full_attention, "preattn_prep_append", "preattn_prep_append"),
    ]
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patched]
    for module, attr, name in patched:
        setattr(module, attr, C._Counted(name, getattr(module, attr)))
    outputs = full_attention.FoldedPreprocess.outputs

    def counted_outputs(self, args):
        C._COUNTS["preprocess_elided"] = C._COUNTS.get("preprocess_elided", 0) + 1
        return outputs(self, args)

    full_attention.FoldedPreprocess.outputs = counted_outputs
    try:
        yield from _fixture_body(C.env)()
    finally:
        full_attention.FoldedPreprocess.outputs = outputs
        for module, attr, original in originals:
            setattr(module, attr, original)
        (C.ARMS, C.ENTRIES, C.COMBINED, C.DETERMINISM, C._expected_counts, C._metadata) = saved


def test_the_arms_have_the_expected_layout(env):
    """The full-attention layers of every prep-append arm carry the mixin in front of the
    preattn_prep class."""
    wrong = []
    for arm, selection, _ in ARMS:
        selected = C._selection(selection)
        for index, layer in enumerate(env.models[arm].layers):
            cls = type(layer.self_attn)
            folded = cls.__mro__[1] is full_attention.FullAppendAttention
            want = "prep_append" in selected and not layer.self_attn.is_sliding
            if folded != want:
                wrong.append(f"{arm} layer {index}: {cls.__name__}")
    assert not wrong, wrong


@pytest.mark.parametrize("rows", C.ROWS)
def test_prep_append_is_bit_identical_in_eager_decode(env, rows):
    C._compare_arms(env, rows, "eager")


@pytest.mark.parametrize("rows", C.ROWS)
def test_prep_append_is_bit_identical_in_cuda_graphs(env, rows):
    C._compare_arms(env, rows, "graph")


def test_every_captured_graph_reports_its_folds(env):
    """The fold census of each prep-append arm: every captured decode graph logged one line,
    both full-attention layers fold in every capture of 1..8 rows, and no capture warned
    (the wider captures say why they do not fold)."""
    captured = {(arm, rows) for (mode, arm, rows) in C._RESULTS if mode == "graph"}
    if not captured:
        pytest.skip("no CUDA-graph comparison ran in this session")
    problems = []
    for arm, selection, _ in ARMS:
        widths = sorted(rows for a, rows in captured if a == arm)
        if "prep_append" not in C._selection(selection) or not widths:
            continue
        census = next(
            layer.self_attn._prep_append_census
            for layer in env.models[arm].layers
            if not layer.self_attn.is_sliding
        )
        reports = list(census.reports)
        folded = [r for r in reports if r[1] == r[2] == FULL_LAYERS and r[3] == "info"]
        warned = [r[4] for r in reports if r[3] == "warning"]
        want = sum(1 for rows in widths if rows <= 8)
        if warned:
            problems.append(f"{arm}: {warned[:2]}")
        if len(folded) < want:
            problems.append(
                f"{arm}: {len(folded)} fully folded captures of {len(reports)}, want {want}"
            )
    everyone = env.comm.allgather(problems)
    assert not any(everyone), everyone


def test_write_the_result_table(env):
    assert C._RESULTS, "the composition tests recorded nothing"
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir and env.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        table = {f"{m}:{a}@{r}": v for (m, a, r), v in sorted(C._RESULTS.items())}
        with open(os.path.join(out_dir, "compose_prep_append.json"), "w") as f:
            json.dump(table, f, indent=1, sort_keys=True, default=str)
    torch.cuda.synchronize()
