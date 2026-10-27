# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B1 arming and all-rank preparation agreement, without importing a GPU runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[3]
_CATALOG = "tensorrt_llm._torch._experimental.modeling_v2.catalog"
_SOURCE = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog/spec_decode/m31_draft_bf16.py"
)


@pytest.fixture
def adapter(monkeypatch):
    # Preserve the production module and arm body; mock only runtime dependencies.
    def custom_op(*args, **kwargs):
        def decorate(fn):
            fn.register_fake = lambda fake: fake
            return fn

        return decorate

    torch = SimpleNamespace(
        Tensor=object,
        library=SimpleNamespace(custom_op=custom_op),
        bfloat16=object(),
        zeros=lambda *args, **kwargs: object(),
    )
    ag = SimpleNamespace()
    comm = ModuleType(_CATALOG + ".comm")
    comm.allgather_cast = ag
    logger = ModuleType("tensorrt_llm.logger")
    logger.logger = SimpleNamespace(warning=lambda *args: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, comm.__name__, comm)
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    monkeypatch.delenv("TRTLLM_M31_DISABLE_DRAFT_AG_BF16", raising=False)

    def load():
        spec = importlib.util.spec_from_file_location(
            _CATALOG + ".spec_decode._b1_under_test", _SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load, ag


@pytest.mark.parametrize(
    "mode",
    ["pass", "local_fail", "peer_fail", "stock_fail", "no_agree", "other_rows", "other_cols"],
)
def test_preparation_preserves_all_rank_agreement(adapter, monkeypatch, mode):
    load, ag = adapter
    b1 = load()
    assert not b1._DISABLED
    events = []
    mapping, group = object(), object()
    rows = (7,) if mode == "other_rows" else (154,)
    cols = 999 if mode == "other_cols" else 100032

    def stock(actual_mapping, actual_cols, actual_rows, device, group=None, agree=None):
        assert actual_mapping is mapping and actual_cols == cols and actual_rows == rows
        assert device == "cuda:0"
        events.append("stock_compile")
        ok = mode != "stock_fail"
        if agree is not None:
            ok = agree(ok)
        if ok:
            events.append("stock_launch")
        return ok

    def workspace(actual_mapping, actual_cols, actual_group):
        assert actual_mapping is mapping and actual_cols == cols and actual_group is group
        return {"world": 2}

    def prepare(shard, ws):
        assert ws == {"world": 2}
        events.append("b1_prepare")
        b1._READY.add(0)
        if mode == "local_fail":
            raise RuntimeError("local preparation failure")

    def agree(ok):
        events.append(("agree", ok))
        return ok and mode != "peer_fail"

    ag.precompile, ag.workspace_args = stock, workspace
    monkeypatch.setattr(b1, "prepare", prepare)
    b1.arm_draft_bf16()
    installed = ag.precompile
    b1.arm_draft_bf16()
    assert ag.precompile is installed
    result = ag.precompile(
        mapping, cols, rows, "cuda:0", group=group, agree=None if mode == "no_agree" else agree
    )
    if mode in ("local_fail", "peer_fail", "stock_fail"):
        assert not result and not b1._READY and "stock_launch" not in events
    else:
        assert result and events[-1] == "stock_launch"
    if mode in ("pass", "local_fail", "peer_fail"):
        assert events[:3] == ["stock_compile", "b1_prepare", ("agree", mode != "local_fail")]
    else:
        assert "b1_prepare" not in events
    if mode == "pass":
        assert b1._READY == {0}


def test_kill_switch_prevents_arm_prepare_and_dispatch(adapter, monkeypatch):
    load, ag = adapter
    monkeypatch.setenv("TRTLLM_M31_DISABLE_DRAFT_AG_BF16", "1")
    b1 = load()
    stock = object()
    ag.precompile = stock
    b1.arm_draft_bf16()
    assert ag.precompile is stock and not b1._ARMED
    b1.prepare(object(), {})  # No tensor access is allowed after the kill gate.
    model = SimpleNamespace(logits_processor=object())
    assert b1.try_head_and_refine(object(), model, None, None, None, None, 22, 7) is None
    assert not b1._READY


def test_other_model_and_d8_decline_before_tensor_access(adapter):
    load, _ = adapter
    b1 = load()
    model = SimpleNamespace(logits_processor=object())
    assert b1.try_head_and_refine(object(), model, None, None, None, None, 22, 7) is None
    worker = type("DSparkWorker", (), {"__module__": b1._TARGET + ".dspark"})()
    processor = type("_DraftA1Logits", (), {"__module__": b1._TARGET + ".dspark"})()
    model = SimpleNamespace(logits_processor=processor)
    assert b1.try_head_and_refine(worker, model, None, None, None, None, 1, 7) is None


@pytest.mark.parametrize("fail_row", [None, 154])
def test_all_qualified_rows_prepare_before_agreement(adapter, monkeypatch, fail_row):
    load, ag = adapter
    b1 = load()
    events = []
    monkeypatch.setattr(b1.torch, "zeros", lambda rows, *args, **kwargs: rows)

    def prepare(rows, ws):
        events.append(rows)
        b1._READY.add((0, rows))
        if rows == fail_row:
            raise RuntimeError("middle shape failed")

    def stock(mapping, cols, rows, device, group=None, agree=None):
        return agree(True)

    def agree(ok):
        events.append(("agree", ok))
        return ok

    ag.precompile = stock
    ag.workspace_args = lambda *args: {"world": 2}
    monkeypatch.setattr(b1, "prepare", prepare)
    b1.arm_draft_bf16()
    assert ag.precompile(None, 100032, (84, 91, 154, 196, 203), "cuda:0", agree=agree) == (
        fail_row is None
    )
    if fail_row is None:
        assert events == [91, 154, 196, ("agree", True)]
        assert b1._READY == {(0, 91), (0, 154), (0, 196)}
    else:
        assert events == [91, 154, ("agree", False)]
        assert not b1._READY


@pytest.mark.parametrize("requests", [12, 13, 22, 28, 29])
def test_prepare_marks_only_compiled_device_and_row(adapter, monkeypatch, requests):
    load, _ = adapter
    b1 = load()
    events = []
    device = SimpleNamespace(index=0)
    shard = SimpleNamespace(
        ndim=2,
        shape=(requests * 7, 100032),
        device=device,
        new_empty=lambda shape: events.append(("allocate", shape)),
    )
    monkeypatch.setattr(
        b1.torch, "cuda", SimpleNamespace(get_device_capability=lambda d: (10, 7)), raising=False
    )
    monkeypatch.setattr(b1, "_load", lambda: True)
    monkeypatch.setattr(b1, "_kernel", lambda *args: events.append("compile"))
    epilogue = SimpleNamespace(
        prebuild=lambda d: object(), _splits=lambda n, v: events.append(("splits", n))
    )
    monkeypatch.setattr(b1, "importlib", SimpleNamespace(import_module=lambda name: epilogue))
    ws = {"world": 2, "ws_uc": None, "ws_mc": None, "ws_flags": None, "rank": 0}
    b1.prepare(shard, ws)
    if 13 <= requests <= 28:
        assert b1._READY == {(0, requests * 7)}
        assert events == [("allocate", (requests * 7, 200064)), "compile", ("splits", requests)]
    else:
        assert not b1._READY and not events


@pytest.mark.parametrize("requests", range(13, 29))
@pytest.mark.parametrize("mode", ["pass", "kill", "other_ready_row", "mixed", "a1_missing"])
def test_dispatch_launches_only_ready_rows(adapter, monkeypatch, requests, mode):
    load, _ = adapter
    b1 = load()
    events = []
    device = SimpleNamespace(index=0)
    dtype = b1.torch.bfloat16

    class Tensor:
        def __init__(self, shape):
            self.shape, self.device, self.dtype, self.is_cuda = shape, device, dtype, True

        def is_contiguous(self):
            return True

        def view(self, *shape):
            return Tensor(shape)

    worker = type("DSparkWorker", (), {"__module__": b1._TARGET + ".dspark"})()
    processor = type("_DraftA1Logits", (), {"__module__": b1._TARGET + ".dspark"})()
    head = SimpleNamespace(
        apply_linear=lambda *args: events.append("head") or Tensor((requests * 7, 100032))
    )
    processor._head = [head]
    processor._rows = () if mode == "a1_missing" else (requests * 7,)
    processor._workspace = dict(world=2, rank=0, ws_uc=None, ws_mc=None, ws_flags=None)
    model = SimpleNamespace(
        logits_processor=processor,
        lm_head=head,
        has_markov_head=True,
        markov_w1=Tensor((200064, 256)),
        markov_w2=Tensor((200064, 256)),
    )
    b1._READY.add((0, 7 if mode == "other_ready_row" else requests * 7))
    b1._DISABLED = mode == "kill"
    epilogue = SimpleNamespace(
        prebuild=lambda d: (None, None),
        _splits=lambda n, v: 1,
        _prev_embeddings=lambda *args: Tensor((requests, 256)),
    )
    monkeypatch.setattr(
        b1,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: SimpleNamespace(S4FE=True)
            if name.endswith("dspark_s4")
            else epilogue
        ),
    )
    monkeypatch.setattr(
        b1, "gather", lambda *args: events.append("gather") or Tensor((requests * 7, 200064))
    )
    monkeypatch.setattr(b1.torch, "empty", lambda shape, **kw: Tensor(shape), raising=False)
    monkeypatch.setattr(b1.torch, "empty_like", lambda t: Tensor(t.shape), raising=False)
    monkeypatch.setattr(b1.torch, "float32", object(), raising=False)
    monkeypatch.setattr(b1.torch, "int64", object(), raising=False)
    monkeypatch.setattr(
        b1.torch,
        "nn",
        SimpleNamespace(
            functional=SimpleNamespace(linear=lambda *args: Tensor((requests, 200064)))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        b1.torch,
        "ops",
        SimpleNamespace(
            trtllm=SimpleNamespace(
                minimax_m31_markov_epilogue_bf16=lambda *args: events.append("consumer")
            )
        ),
        raising=False,
    )
    b1._LOGGED.add((0, requests))
    out = b1.try_head_and_refine(
        worker,
        model,
        Tensor((requests * 7, 6144)),
        SimpleNamespace(num_contexts=int(mode == "mixed")),
        {"first_prev_tokens": None},
        None,
        requests,
        7,
    )
    if mode == "pass":
        assert out.shape == (requests, 7, 200064)
        assert events == ["head", "gather"] + ["consumer"] * 7
    else:
        assert out is None and events == []
