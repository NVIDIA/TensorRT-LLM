# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the product dispatch and prebuild functions with CPU metadata doubles."""

import ast
import json
import math
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def test_product_adapter_contract(monkeypatch):
    SOURCE = (
        Path(__file__).resolve().parents[9]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/ops/prob1_worker.py"
    )

    class Tensor:
        def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
            self.shape, self.dtype, self.device = shape, dtype, device
            self.ndim = len(shape)
            self.is_cuda = device.startswith("cuda")
            self.contiguous = contiguous
            self.sanitized = False

        def is_contiguous(self):
            return self.contiguous

        def view(self, *shape):
            if -1 in shape:
                size = math.prod(self.shape) // -math.prod(shape)
                shape = tuple(size if dim == -1 else dim for dim in shape)
            out = Tensor(shape, self.dtype, self.device, self.contiguous)
            out.sanitized = self.sanitized
            return out

        def long(self):
            return Tensor(self.shape, "int64", self.device, self.contiguous)

        def __getitem__(self, index):
            if isinstance(index, slice):
                start, stop, stride = index.indices(self.shape[0])
                return Tensor(
                    (len(range(start, stop, stride)), *self.shape[1:]), self.dtype, self.device
                )
            raise AssertionError(index)

    class Metadata(types.SimpleNamespace):
        def __eq__(self, other):
            raise AssertionError("Metadata equality may synchronize tensors; use identity")

        def rng_draft_slot(self, step):
            return 8 + step

    calls = []
    padding_calls = []
    torch = types.SimpleNamespace(
        Tensor=Tensor, bfloat16="bf16", float32="fp32", int32="int32", int64="int64"
    )
    prob1 = types.SimpleNamespace(TAG="P1", supported=lambda *args: None)
    eligibility_source = SOURCE.parent.parent / "kernels/m3ll/prob1/draft_prob1.py"
    eligibility = next(
        node
        for node in ast.parse(eligibility_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "engages"
    )
    eligibility_ns = {"MAX_ROWS": 8}
    exec(
        compile(ast.Module(body=[eligibility], type_ignores=[]), str(eligibility_source), "exec"),
        eligibility_ns,
    )
    prob1.engages = eligibility_ns["engages"]
    namespace = {
        "torch": torch,
        "prob1": prob1,
        "_RNG_SLOT_SPAN": 32,
        "logger": types.SimpleNamespace(info=lambda *args: None),
    }
    tree = ast.parse(SOURCE.read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__", names=[ast.alias("annotations")], level=0
                        ),
                        *funcs,
                    ],
                    type_ignores=[],
                )
            ),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )

    def fixture():
        p1 = types.SimpleNamespace(k=7, max_rows=8, vocab=200064, world=2, w1=Tensor((200064, 256)))

        def run(*args):
            calls.append(args)
            return Tensor((args[0].shape[0] // 7, 7), "int32")

        p1.run = run
        p1.taken_line = lambda g: None
        md = Metadata(
            use_rejection_sampling=True,
            wants_advanced_draft_sampling=True,
            draft_probs=Tensor((6, 7, 200064), "fp32"),
            batch_slot_ids=Tensor((3,), "int32"),
            request_offsets=Tensor((3,), "int64"),
            request_temperatures=Tensor((3,), "fp32"),
            request_seeds=Tensor((3,), "int64"),
            dummy_slot_row=5,
        )
        draft = object()
        state = types.SimpleNamespace(block=7, shard=100032)

        def sanitize(raw, metadata, nc, bs, block):
            padding_calls.append((raw.shape, metadata, nc, bs, block))
            out = Tensor(raw.shape, raw.dtype, raw.device)
            out.sanitized = True
            return out

        worker = types.SimpleNamespace(
            _p1=p1,
            _s4_step=state,
            _s4_step_mode="logits",
            _s4_draft=draft,
            _s4_step_meta=(1, 3, md),
            _s4_common=lambda g, d: True,
            _zero_padding_rows=sanitize,
        )
        return (
            worker,
            draft,
            Tensor((2, 7, 100032)),
            {"first_prev_tokens": Tensor((2,), "int64")},
            md,
            1,
            3,
        )

    fn = namespace["draw_ancestral_block"]
    checks = []
    args = fixture()
    result = fn(*args)
    assert result[1] == 200064 and result[0].shape == (2, 7)
    assert args[4].draft_probs_last_dim == 200064 and len(calls) == 1
    payload = calls[-1]
    assert (
        payload[0].shape == (14, 100032)
        and payload[1].dtype == "int64"
        and payload[2].shape == (2,)
    )
    assert payload[2].dtype == "int64" and payload[3] == 5 and payload[5:7] == (256, 32)
    assert payload[9] is args[4].draft_probs
    assert payload[0].sanitized
    assert padding_calls[-1][0] == (14, 100032) and padding_calls[-1][1] is args[4]
    assert padding_calls[-1][2:] == (1, 3, 7)
    checks.append("mixed-context dispatch; shared native payload/RNG/q publication")

    for world in (2, 4, 8):
        for gens in (1, 2, 3, 4):
            args = list(fixture())
            worker, draft, _, inputs, md, nc, _ = args
            p1 = worker._p1
            p1.world = world
            p1.max_rows = 4 if world == 8 else 8
            worker._s4_step.shard = p1.vocab // world
            worker._s4_step_meta = (nc, nc + gens, md)
            args[2] = Tensor((gens, 7, p1.vocab // world))
            args[6] = nc + gens
            inputs["first_prev_tokens"] = Tensor((gens,), "int32")
            md.batch_slot_ids = Tensor((nc + gens,), "int64")
            md.request_offsets = Tensor((nc + gens,), "int64")
            md.request_temperatures = Tensor((nc + gens,), "fp32")
            md.request_seeds = Tensor((nc + gens,), "int64")
            result = fn(*args)
            assert result[0].shape == (gens, 7) and result[1] == p1.vocab
            assert calls[-1][2].shape == (gens,)
            checks.append(f"shared topology world{world} B{gens} payload")

    mutations = {
        "FP32 raw": lambda a: setattr(a[2], "dtype", "fp32"),
        "CPU raw": lambda a: setattr(a[2], "is_cuda", False),
        "strided raw": lambda a: setattr(a[2], "contiguous", False),
        "other device": lambda a: setattr(a[2], "device", "cuda:1"),
        "wrong K": lambda a: setattr(a[2], "shape", (2, 6, 100032)),
        "wrong shard": lambda a: setattr(a[2], "shape", (2, 7, 100033)),
        "wrong vocab topology": lambda a: setattr(a[0]._p1, "world", 4),
        "no active step": lambda a: setattr(a[0], "_s4_step", None),
        "greedy path": lambda a: setattr(a[0], "_s4_step_mode", "tokens"),
        "foreign draft": lambda a: setattr(a[0], "_s4_draft", object()),
        "no P1": lambda a: setattr(a[0], "_p1", None),
        "foreign metadata": lambda a: setattr(a[0], "_s4_step_meta", (1, 3, Metadata())),
        "wrong context": lambda a: setattr(a[0], "_s4_step_meta", (0, 3, a[4])),
        "wrong batch": lambda a: setattr(a[0], "_s4_step_meta", (1, 4, a[4])),
        "S4 decline": lambda a: setattr(a[0], "_s4_common", lambda *args: False),
        "no rejection": lambda a: setattr(a[4], "use_rejection_sampling", False),
        "no advanced": lambda a: setattr(a[4], "wants_advanced_draft_sampling", False),
        "missing q": lambda a: setattr(a[4], "draft_probs", None),
        "wrong q dtype": lambda a: setattr(a[4].draft_probs, "dtype", "bf16"),
        "strided q": lambda a: setattr(a[4].draft_probs, "contiguous", False),
        "small q vocab": lambda a: setattr(a[4].draft_probs, "shape", (6, 7, 200063)),
        "missing anchor": lambda a: a[3].clear(),
        "float anchor": lambda a: setattr(a[3]["first_prev_tokens"], "dtype", "fp32"),
        "wrong anchor shape": lambda a: setattr(a[3]["first_prev_tokens"], "shape", (2, 1)),
    }
    for label, mutate in mutations.items():
        a = fixture()
        mutate(a)
        before = len(calls)
        before_padding = len(padding_calls)
        assert fn(*a) is None, label
        assert len(calls) == before and len(padding_calls) == before_padding, label
        checks.append(label + " declines before native op")

    # Actual prebuild body: foreign ownership declines collectively before allocation.
    dspark = types.ModuleType("tensorrt_llm._torch.speculative.dspark")
    dspark._ANCESTRAL_BLOCK_OVERRIDE = None

    def setter(hook):
        dspark._ANCESTRAL_BLOCK_OVERRIDE = hook

    dspark.set_ancestral_block_override = setter
    spec = types.ModuleType("tensorrt_llm._torch.speculative")
    spec.dspark = dspark
    ops = types.ModuleType("tensorrt_llm._torch.distributed.ops")
    ops._get_mnnvl_workspace_comm = lambda m: "comm"
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.speculative", spec)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch.distributed.ops", ops)
    created = []
    p1 = types.SimpleNamespace(describe=lambda: "ready")

    def create(*args):
        created.append(args)
        return p1

    prob1.DraftProb1 = types.SimpleNamespace(create=create)
    prebuild = namespace["prebuild"]
    map_ = types.SimpleNamespace(tp_size=2, tp_rank=0)
    for foreign, enabled, expected in [
        (None, True, True),
        (fn, True, True),
        (object(), True, False),
        (None, False, False),
    ]:
        dspark._ANCESTRAL_BLOCK_OVERRIDE = foreign
        votes = []

        def agree(v):
            votes.append(v)
            return v

        w = types.SimpleNamespace(_s4_logits_on=enabled, _s4_ranks_agree=agree)
        count = len(created)
        prebuild(w, map_, Tensor((200064, 256)), Tensor((100032, 256)), 0, 200064, 7)
        assert votes == [expected] and (len(created) == count + 1) == expected
        assert (w._p1 is p1) == expected
        assert dspark._ANCESTRAL_BLOCK_OVERRIDE is (fn if expected else foreign)
        owner_name = "none" if foreign is None else "self" if foreign is fn else "foreign"
        checks.append(f"collective prebuild owner={owner_name} enabled={enabled}")
    print(
        json.dumps(
            {
                "status": "PASS_PRODUCT_ADAPTER_CPU_CONTROL_ONLY",
                "checks": len(checks),
                "cases": checks,
            },
            indent=2,
        )
    )
