# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAILBK on this target: ``TAILBK`` is on by default and ``TRTLLM_M31_DISABLE_TAILBK`` turns it off; ``_runtime_hooks``
passes it as RuntimeHooks.spec_tail_bookkeep; the model logs the armed items once; and the worker's own items, R1
(``M3DSparkWorker._rng_state_per_request``) and Z1 (``M3DSparkWorker._zero_padding_rows``), match the in-tree methods
bit for bit, eagerly and inside ``torch.cuda.graph``.

The constant and the arm line are evaluated from modeling.py's source against a private copy of m31_switches loaded
under each environment, so the heavy modeling module is not reloaded (CPU only). The R1 / Z1 tests import the target's
dspark module and need CUDA. The shared items (B1, S1, A1) are tested in tests/unittest/_torch/speculative/
test_tail_bookkeep.py.
"""

import ast
import importlib
import importlib.util
import os
import types
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

TARGET = "minimax_m31_nvfp4__sm_107__tp8"
CONSTANT = "TAILBK"
FIELD = "spec_tail_bookkeep"
_DIR = Path(_family.__file__).resolve().parent / TARGET
_SWITCHES = _DIR / "m31_switches.py"
_MODELING = _DIR / "modeling.py"
ARM_PREFIX = "[M3 host] tail bookkeeping: "

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target's worker; CUDA buffers"
)


def _module_tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _switches(monkeypatch, env):
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_tailbk_{id(env)}", _SWITCHES)
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    return switches


def _top_level(names) -> list:
    """modeling.py's module-level statements that bind one of ``names`` (assignments and function definitions)."""
    out = []
    for node in _module_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                out.append(node)
    return out


def _value(monkeypatch, env) -> bool:
    switches = _switches(monkeypatch, env)
    statements = _top_level({CONSTANT})
    assert len(statements) == 1
    namespace = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(compile(ast.Module(statements, []), str(_MODELING), "exec"), namespace)
    return namespace[CONSTANT]


@pytest.mark.cpu_only
def test_on_by_default(monkeypatch):
    assert _value(monkeypatch, {}) is True


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["1", "true", "ON", " yes "])
def test_the_kill_switch_turns_it_off(monkeypatch, value):
    assert _value(monkeypatch, {"TRTLLM_M31_DISABLE_TAILBK": value}) is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_an_unset_kill_switch_keeps_it(monkeypatch, value):
    assert _value(monkeypatch, {"TRTLLM_M31_DISABLE_TAILBK": value}) is True


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "other", ["TAILBK_SPEC", "HOSTTAIL", "BLOCKOFF_GUARD", "SLOT_IDS_CHANGED", "R1", "R1VR8", "A1"]
)
def test_other_switches_leave_it_on(monkeypatch, other):
    # A1 is another item's switch on this target (the verify all-gather) and R1VR8 the MSA-rows bound; R1 is the
    # sm_103 targets' MSA-rows switch name. None of them is a TAILBK sub-item switch.
    assert _value(monkeypatch, {f"TRTLLM_M31_DISABLE_{other}": "1"}) is True


MODEL_CLASS = "ModelingV2MiniMaxM31Nvfp4Sm107Tp8"


def _function(name: str, cls: str = None) -> ast.FunctionDef:
    """A module-level function, or a method of the module-level class ``cls``."""
    scope = _module_tree().body
    if cls is not None:
        scope = next(
            node for node in scope if isinstance(node, ast.ClassDef) and node.name == cls
        ).body
    return next(node for node in scope if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.cpu_only
def test_runtime_hooks_passes_the_constant():
    calls = [
        node
        for node in ast.walk(_function("_runtime_hooks"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeHooks"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(keywords.get(FIELD), ast.Name) and keywords[FIELD].id == CONSTANT


@pytest.mark.cpu_only
def test_the_model_logs_the_arm_line_after_building_its_hooks():
    body = _function("__init__", MODEL_CLASS).body
    names = [
        node.value.func.id
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        else None
        for node in body
    ]
    assigns_hooks = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "runtime_hooks" for t in node.targets)
    ]
    assert assigns_hooks and "_log_tailbk" in names
    assert names.index("_log_tailbk") > assigns_hooks[0]


class _DSparkWorker:
    """Stands in for dspark.M3DSparkWorker in the arm line's namespace."""


def _arm_lines(monkeypatch, env, worker) -> list:
    """The lines ``_log_tailbk(worker)`` logs on a fresh copy of the function (its once-per-process flag reset)."""
    switches = _switches(monkeypatch, env)
    statements = _top_level({CONSTANT, "_TAILBK_LOGGED", "_log_tailbk"})
    assert len(statements) == 3
    lines = []
    namespace = {
        "m31_disabled": switches.m31_disabled,
        "m31_enabled": switches.m31_enabled,
        "_dspark": types.SimpleNamespace(M3DSparkWorker=_DSparkWorker),
        "logger": types.SimpleNamespace(info=lines.append),
    }
    exec(compile(ast.Module(statements, []), str(_MODELING), "exec"), namespace)
    namespace["_log_tailbk"](worker)
    namespace["_log_tailbk"](worker)  # once per process
    return lines


@pytest.mark.cpu_only
def test_arm_line_lists_the_dspark_workers_items_once(monkeypatch):
    assert _arm_lines(monkeypatch, {}, _DSparkWorker()) == [ARM_PREFIX + "R1 B1 Z1 S1 A1"]


@pytest.mark.cpu_only
def test_arm_line_lists_only_the_accept_for_another_worker(monkeypatch):
    assert _arm_lines(monkeypatch, {}, object()) == [ARM_PREFIX + "A1"]


@pytest.mark.cpu_only
def test_no_arm_line_when_off_or_without_speculation(monkeypatch):
    assert _arm_lines(monkeypatch, {"TRTLLM_M31_DISABLE_TAILBK": "1"}, _DSparkWorker()) == []
    assert _arm_lines(monkeypatch, {}, None) == []


@needs_cuda
def test_the_in_tree_hook_follows_the_switch():
    modeling = importlib.import_module(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{TARGET}.modeling"
    )
    hooks = modeling._runtime_hooks(None)
    disabled = os.environ.get("TRTLLM_M31_DISABLE_TAILBK", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )
    assert getattr(hooks, FIELD) is getattr(modeling, CONSTANT) is (not disabled)


# ---------------------------------------------------------------------------------------------------------------------
# R1 and Z1 against the in-tree methods (CUDA: the real populate).
# ---------------------------------------------------------------------------------------------------------------------

V = 64
SLOT = 3
# (temperature, top_k, top_p, min_p, seed): unseeded, int64-edge seed, a uint64 seed (negative as int64), seed 0.
REQUESTS = [
    (0.7, 0, 0.9, 0.0, None),
    (1.3, 17, 0.5, 0.05, (1 << 63) - 1),
    (0.25, 3, 1.0, 0.2, (1 << 64) - 1),
    (1.0, 0, 0.95, 0.0, 0),
]


def _dspark():
    return importlib.import_module(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{TARGET}.dspark"
    )


def _interface():
    return importlib.import_module("tensorrt_llm._torch.speculative.interface")


@pytest.fixture(autouse=True)
def _zeroed_padding_rows(monkeypatch):
    # The sm_107 targets keep padding rows zeroed (routing arms interface.arm_sm107_spec_paths).
    monkeypatch.setattr(_interface(), "_CHEAP_PADDING_ROWS", False)


def _worker(attention_dp: bool = False):
    worker = object.__new__(_dspark().M3DSparkWorker)  # the overrides read only these fields
    worker.mapping = types.SimpleNamespace(enable_attention_dp=attention_dp, tp_size=1, tp_rank=0)
    worker._d2t = None
    worker.force_num_accepted_tokens = 0.0
    return worker


def _request(params, request_id: int = 7):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

    temperature, top_k, top_p, min_p, seed = params
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, seed=seed
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=SLOT,
        py_request_id=request_id,
        py_decoding_iter=0,
    )


def _meta(k: int, on: bool):
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    interface = _interface()
    meta = interface.SpecMetadata(
        max_num_requests=8,
        max_draft_len=k,
        max_total_draft_tokens=k,
        spec_dec_mode=interface.SpeculativeDecodingMode.DSPARK,
    )
    meta.runtime_draft_len = k
    meta.advanced_sampling_mode = AdvancedSamplingMode.NO_TOPK
    meta.use_rejection_sampling = True
    meta.vocab_size = V
    meta.draft_vocab_size = V
    meta.tail_bookkeep = on
    return meta


def _populate(meta, params) -> None:
    meta.populate_sampling_params_for_one_model([_request(params)])
    assert not meta.is_all_greedy_sample


def _stock_rng(worker, meta, k):
    base = _interface().SpecWorkerBase._rng_state_per_request
    accept = base(worker, meta, 0, 1, slot=meta.rng_accept_slot)
    block = base(worker, meta, 0, 1, repeat=k, slot=meta.rng_draft_slot(0))
    return accept, block


def _r1_rng(worker, meta, k, stash: bool = True):
    accept = worker._rng_state_per_request(meta, 0, 1, slot=meta.rng_accept_slot)
    if not stash:
        worker._tailbk_win = None
    block = worker._rng_state_per_request(meta, 0, 1, repeat=k, slot=meta.rng_draft_slot(0))
    return accept, block


def _assert_rng_equal(got, want, what):
    for (g_seed, g_off), (w_seed, w_off), part in zip(got, want, ("accept", "block")):
        for g, w, name in ((g_seed, w_seed, "seed"), (g_off, w_off, "offset")):
            assert g.dtype == w.dtype == torch.int64, f"{what} {part} {name} dtype"
            assert g.shape == w.shape and g.is_contiguous(), f"{what} {part} {name} shape"
            assert torch.equal(g, w), f"{what} {part} {name}: {g.tolist()} vs {w.tolist()}"


@needs_cuda
@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("stash", [True, False], ids=["stash", "no_stash"])
def test_r1_eager_matches_the_in_tree_rows(k, stash):
    worker = _worker()
    on, off = _meta(k, True), _meta(k, False)
    for params in REQUESTS:
        _populate(on, params)
        _populate(off, params)
        _assert_rng_equal(
            _r1_rng(worker, on, k, stash), _stock_rng(worker, off, k), f"K={k} {params}"
        )
        assert worker._tailbk_win is None  # the block consumed (or never had) the window


@needs_cuda
@pytest.mark.parametrize("k", [1, 3, 7])
def test_r1_offsets_that_wrap_int64(k):
    """The largest offset a host upload allows (o + K * SPAN = 2^63 - 1): the accept and draft rows wrap past 2^63
    in both, to the same bits (int64 addition is associative mod 2^64)."""
    span = _interface()._RNG_SLOT_SPAN
    worker = _worker()
    on, off = _meta(k, True), _meta(k, False)
    o = (1 << 63) - 1 - k * span
    for meta in (on, off):
        _populate(meta, REQUESTS[1])
        meta.request_offsets[0] = o
        meta.offsets[: k + 1] = torch.tensor(
            [o + r * span for r in range(k + 1)], dtype=torch.int64, device="cuda"
        )
    got, want = _r1_rng(worker, on, k), _stock_rng(worker, off, k)
    _assert_rng_equal(got, want, "wrap")
    assert int(want[0][1][0]) < 0  # the accept row did wrap


@needs_cuda
def test_r1_counter_near_the_int64_edge():
    """A seeded slot whose window counter puts its offset base within one window of 2^63."""
    k = 7
    worker = _worker()
    on, off = _meta(k, True), _meta(k, False)
    window = (on.rng_accept_slot + 1) * _interface()._RNG_SLOT_SPAN
    step = ((1 << 63) - 1 - k * _interface()._RNG_SLOT_SPAN) // window
    for meta in (on, off):
        meta._rng_slot_owner[SLOT] = 7
        meta._rng_window_counter[SLOT] = step
        _populate(meta, REQUESTS[1])
    _assert_rng_equal(_r1_rng(worker, on, k), _stock_rng(worker, off, k), "edge counter")


@needs_cuda
def test_r1_stash_of_another_metadata_is_not_used():
    k = 3
    worker = _worker()
    on, other, off = _meta(k, True), _meta(k, True), _meta(k, False)
    _populate(on, REQUESTS[0])
    _populate(off, REQUESTS[0])
    _populate(other, REQUESTS[1])
    other_accept = worker._rng_state_per_request(other, 0, 1, slot=other.rng_accept_slot)
    assert worker._tailbk_win is not None and worker._tailbk_win[0] is other
    block = worker._rng_state_per_request(on, 0, 1, repeat=k, slot=on.rng_draft_slot(0))
    accept = worker._rng_state_per_request(on, 0, 1, slot=on.rng_accept_slot)
    _assert_rng_equal((accept, block), _stock_rng(worker, off, k), "foreign stash")
    assert other_accept[1].shape == (1,)


@needs_cuda
@pytest.mark.parametrize(
    "case", ["flag_off", "two_requests", "context_row", "step_form_slot", "dynamic_draft_len"]
)
def test_r1_falls_back_to_the_in_tree_rows(case):
    k = 3
    worker = _worker()
    meta = _meta(k, case != "flag_off")
    _populate(meta, REQUESTS[1])
    base = _interface().SpecWorkerBase._rng_state_per_request
    args = {
        "flag_off": ((0, 1), {"slot": meta.rng_accept_slot}),
        "two_requests": ((0, 2), {"slot": meta.rng_accept_slot}),
        "context_row": ((1, 2), {"slot": meta.rng_accept_slot}),
        "step_form_slot": ((0, 1), {"slot": meta.rng_draft_slot(1)}),
        "dynamic_draft_len": ((0, 1), {"repeat": k - 1, "slot": meta.rng_draft_slot(0)}),
    }[case]
    if case == "dynamic_draft_len":
        meta.runtime_draft_len = k - 1
    got = worker._rng_state_per_request(meta, *args[0], **args[1])
    want = base(worker, meta, *args[0], **args[1])
    assert worker._tailbk_win is None
    for g, w in zip(got, want):
        assert torch.equal(g, w)


@needs_cuda
@pytest.mark.parametrize("k", [1, 4, 7])
def test_r1_in_a_cuda_graph(k):
    """Captured on the first request's populate, replayed after each later one (same buffers, new values)."""
    worker = _worker()
    on, off = _meta(k, True), _meta(k, False)
    _populate(on, REQUESTS[0])
    _populate(off, REQUESTS[0])

    def step():
        worker._tailbk_win = None  # the forward's reset
        return _r1_rng(worker, on, k)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for params in REQUESTS[1:]:
        _populate(on, params)
        _populate(off, params)
        graph.replay()
        want = _stock_rng(worker, off, k)
        torch.cuda.synchronize()
        _assert_rng_equal(captured, want, f"replay {params}")


@needs_cuda
def test_the_forward_resets_the_r1_window(monkeypatch):
    dspark = _dspark()
    seen = []
    monkeypatch.setattr(
        dspark.S4WorkerMixin,
        "_forward_impl",
        lambda self, *a, **kw: seen.append(self._tailbk_win) or "out",
    )
    worker = _worker()
    worker._tailbk_win = ("stale", None)
    assert worker._forward_impl(1, x=2) == "out"
    assert seen == [None] and worker._tailbk_win is None


DUMMY_ROW = 16  # spec_metadata.dummy_slot_row


def _z1_meta(slots, on: bool = True, ragged: bool = False, fused: bool = False):
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    return types.SimpleNamespace(
        tail_bookkeep=on,
        batch_slot_ids=torch.tensor(slots, dtype=torch.long, device="cuda"),
        dummy_slot_row=DUMMY_ROW,
        is_ragged_verify=ragged,
        advanced_sampling_mode=AdvancedSamplingMode.FULL if fused else AdvancedSamplingMode.NO_TOPK,
    )


def _z1_logits(rows: int) -> torch.Tensor:
    logits = torch.randn(rows, V, device="cuda")
    logits[:, 0] = float("inf")
    logits[:, 1] = -float("inf")
    logits[:, 2] = -0.0
    return logits


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _stock_zero(*args, **kwargs):
    return _interface().SpecWorkerBase._zero_padding_rows(*args, **kwargs)


@needs_cuda
@pytest.mark.parametrize("rows_per_request", [1, 7, 8])
def test_z1_skips_only_while_capturing_batch_one(rows_per_request):
    worker = _worker()
    meta = _z1_meta([SLOT])
    logits = _z1_logits(rows_per_request)
    eager = worker._zero_padding_rows(logits, meta, 0, 1, rows_per_request)
    assert (
        eager is not logits
    )  # eager steps (the graph warmup's dummy included) keep the in-tree zeroing
    graph, captured = _capture(
        lambda: worker._zero_padding_rows(logits, meta, 0, 1, rows_per_request)
    )
    assert captured is logits
    graph.replay()
    # The in-tree result at the one real request is a bitwise copy of the logits.
    stock = _stock_zero(logits, meta, 0, 1, rows_per_request)
    torch.cuda.synchronize()
    assert torch.equal(stock.view(torch.int32), logits.view(torch.int32))


@needs_cuda
@pytest.mark.parametrize("batch", range(2, 9))
@pytest.mark.parametrize("rows_per_request", [1, 8])
def test_z1_keeps_the_zeroing_in_padded_graphs(batch, rows_per_request):
    """Batches 2..8 can hold dummies: the captured graph zeroes their rows exactly as the in-tree method."""
    worker = _worker()
    real = batch // 2
    meta = _z1_meta([SLOT + i for i in range(real)] + [DUMMY_ROW] * (batch - real))
    logits = _z1_logits(batch * rows_per_request)
    graph, captured = _capture(
        lambda: worker._zero_padding_rows(logits, meta, 0, batch, rows_per_request)
    )
    assert captured is not logits
    graph.replay()
    stock = _stock_zero(logits, meta, 0, batch, rows_per_request)
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int32), stock.view(torch.int32))
    assert bool((captured[real * rows_per_request :] == 0).all())


@needs_cuda
@pytest.mark.parametrize(
    "case",
    [
        "flag_off",
        "attention_dp",
        "fused_mode",
        "ragged",
        "context_row",
        "strided_logits",
        "unaligned_logits",
    ],
)
def test_z1_keeps_the_zeroing_outside_its_condition(case):
    worker = _worker(attention_dp=case == "attention_dp")
    slots = [SLOT, SLOT + 1] if case == "context_row" else [SLOT]
    meta = _z1_meta(
        slots, on=case != "flag_off", ragged=case == "ragged", fused=case == "fused_mode"
    )
    logits = _z1_logits(4)
    if case == "strided_logits":  # the copy would be contiguous: a consumer may dispatch on that
        logits = torch.cat([logits, logits], dim=1)[:, :V]
    elif case == "unaligned_logits":  # the copy would be aligned: a consumer may dispatch on that
        logits = torch.cat([torch.zeros(4, device="cuda"), logits.flatten()])[4:].view(4, V)
    nc, bs = (1, 2) if case == "context_row" else (0, 1)
    if case == "ragged":
        meta.verify_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
        meta.total_verify_tokens = 4
    graph, captured = _capture(
        lambda: worker._zero_padding_rows(logits, meta, nc, bs, 4, packed_verify=True)
    )
    assert captured is not logits
    graph.replay()
    stock = _stock_zero(logits, meta, nc, bs, 4, packed_verify=True)
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int32), stock.view(torch.int32))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
