# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tail bookkeeping (TAIL_BOOKKEEP) at TP2: the actual family mixin through the actual TP2 worker MRO, with CPU doubles
(uint64 wrap, window reuse / invalidation, exception cleanup, negative guards of the padding-row zeroing)."""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[7]
FAMILY = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"


class Vector:
    adds = 0

    def __init__(self, values, dtype="int64"):
        self.values = list(values)
        self.dtype = dtype

    def __getitem__(self, key):
        return Vector(self.values[key], self.dtype)

    def __add__(self, value):
        Vector.adds += 1
        return Vector(((x + value + 2**63) % 2**64 - 2**63 for x in self.values), self.dtype)

    def dim(self):
        return 1

    def numel(self):
        return len(self.values)

    def is_contiguous(self):
        return True


class Base:
    def _rng_state_per_request(self, *args, **kwargs):
        return "stock_rng"

    def _zero_padding_rows(self, *args, **kwargs):
        return "stock_zero"

    def _forward_impl(self, fail=False):
        assert self._tailbk_win is None
        self._tailbk_win = "used"
        if fail:
            raise ValueError("forward failed")
        return "forward"


def test_actual_module_and_worker_mro():
    torch = types.ModuleType("torch")
    torch.Tensor = Vector
    torch.int64, torch.int32, torch.float32 = "int64", "int32", "float32"
    capture = [False]
    torch.cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: capture[0])
    interface = types.ModuleType("tensorrt_llm._torch.speculative.interface")
    interface._RNG_SLOT_SPAN = 32
    logger_module = types.ModuleType("tensorrt_llm.logger")
    logger_module.logger = types.SimpleNamespace(info=lambda *args: None)
    names = ("torch", interface.__name__, logger_module.__name__)
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules.update(
        {"torch": torch, interface.__name__: interface, logger_module.__name__: logger_module}
    )
    try:
        p = FAMILY / "m31_tail_bookkeep.py"
        spec = importlib.util.spec_from_file_location("_actual_m31_tailbk", p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old
    checks = 0
    for tp in (2,):
        p = FAMILY / "minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark.py"
        cls = next(
            n
            for n in ast.parse(p.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "M3DSparkWorker"
        )
        assert [ast.unparse(b) for b in cls.bases] == [
            "TailBookkeepMixin",
            "S4WorkerMixin",
            "DSparkWorker",
        ]
        assert not any(
            isinstance(n, ast.FunctionDef) and n.name.startswith("_tailbk_") for n in cls.body
        )
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        ns = {
            "TailBookkeepMixin": module.TailBookkeepMixin,
            "S4WorkerMixin": type("S4", (), {}),
            "DSparkWorker": Base,
        }
        unit = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
        exec(compile(unit, str(p), "exec"), ns)
        worker = ns["M3DSparkWorker"]()
        worker.mapping = types.SimpleNamespace(enable_attention_dp=False)
        for k in (1, 3, 7):
            for offset in (0, 2**63 - 40, -(2**63) + 7):
                meta = types.SimpleNamespace(
                    tail_bookkeep=True,
                    is_spec_dec_tree=False,
                    is_spec_dec_dynamic_tree=False,
                    is_ragged_verify=False,
                    max_draft_len=k,
                    runtime_draft_len=k,
                    advanced_sampling_mode=types.SimpleNamespace(is_fused=False),
                    rng_draft_slot=lambda i, k=k: k + 1 + i,
                    rng_accept_slot=2 * k + 1,
                )
                for name, dtype in module._TAIL_BOOKKEEP_ROW_BUFFERS:
                    setattr(meta, name, Vector([11] * (k + 1), dtype))
                meta.offsets = Vector(
                    [(offset + i * 32 + 2**63) % 2**64 - 2**63 for i in range(k + 1)]
                )
                meta.request_seeds, meta.request_offsets = Vector([11]), Vector([offset])
                Vector.adds = 0
                accept = worker._rng_state_per_request(meta, 0, 1, slot=2 * k + 1)
                block = worker._rng_state_per_request(meta, 0, 1, repeat=k, slot=k + 1)

                def wrap(x):
                    return (x + 2**63) % 2**64 - 2**63

                assert accept[1].values == [wrap(offset + (2 * k + 1) * 32)]
                assert block[1].values == [wrap(offset + (k + 1 + j) * 32) for j in range(k)]
                assert Vector.adds == 1 and worker._tailbk_win is None
                assert accept[0].values == [11] and block[0].values == [11] * k
                meta.tail_bookkeep = False
                assert worker._rng_state_per_request(meta, 0, 1, slot=2 * k + 1) == "stock_rng"
                checks += 1
        logits = types.SimpleNamespace(is_contiguous=lambda: True, data_ptr=lambda: 256)
        meta.tail_bookkeep = True
        capture[0] = True
        assert worker._zero_padding_rows(logits, meta, 0, 1, 4) is logits
        for field, value in [("tail_bookkeep", False), ("is_ragged_verify", True)]:
            old = getattr(meta, field)
            setattr(meta, field, value)
            assert worker._zero_padding_rows(logits, meta, 0, 1, 4) == "stock_zero"
            setattr(meta, field, old)
        for nc, bs in ((1, 1), (0, 2)):
            assert worker._zero_padding_rows(logits, meta, nc, bs, 4) == "stock_zero"
        capture[0] = False
        assert worker._zero_padding_rows(logits, meta, 0, 1, 4) == "stock_zero"
        worker._tailbk_win = "stale"
        assert worker._forward_impl() == "forward" and worker._tailbk_win is None
        try:
            worker._forward_impl(fail=True)
        except ValueError:
            pass
        else:
            raise AssertionError("forward exception lost")
        assert worker._tailbk_win is None
        checks += 8
    assert checks == 17
