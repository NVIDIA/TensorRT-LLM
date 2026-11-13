# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual shared PROB1 metadata protocol and TP2 delegates with CPU doubles."""

import ast
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.cpu_only


def test_shared_prob1_adapter():
    s = Path(__file__).resolve().parents[9]
    base = "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/"
    f = s / base / "ops/prob1_worker.py"
    tree = ast.parse(f.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        or isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
    ]
    logger = types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
    ops = types.ModuleType("tensorrt_llm._torch.distributed.ops")
    ops._get_mnnvl_workspace_comm = lambda m: ("comm", m.tp_size)
    dspark = types.ModuleType("tensorrt_llm._torch.speculative.dspark")
    speculative = types.ModuleType("tensorrt_llm._torch.speculative")
    speculative.dspark = dspark
    with patch.dict(sys.modules, {ops.__name__: ops, speculative.__name__: speculative}):

        class T:
            dtype = "I64"

            def __init__(self, label="T"):
                self.label = label
                self.shape = (7, 100)
                self.last = None

            def __getitem__(self, s):
                return T(self.label + "slice")

            def view(self, *shape):
                self.last = shape
                return self

            def long(self):
                return self

        class P:
            def __init__(self, world):
                self.k = 7
                self.vocab = 200064
                self.max_rows = 4 if world == 8 else 8
                self.calls = []

            def run(self, *args):
                self.calls.append(args)
                return "TOKENS"

            def describe(self):
                return "DRAFT_PROB1"

            def taken_line(self, g):
                return None

        created = []
        decline = [None]

        def create(mapping, comm, *args):
            created.append((mapping.tp_size, comm, args))
            return P(mapping.tp_size)

        prob = types.SimpleNamespace(
            TAG="PROB1",
            supported=lambda *args: decline[0],
            DraftProb1=types.SimpleNamespace(create=create),
        )
        eng_source = s / base / "kernels/m3ll/prob1/draft_prob1.py"
        eng = next(
            n
            for n in ast.parse(eng_source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "engages"
        )
        eng_ns = {"MAX_ROWS": 8}
        exec(compile(ast.Module(body=[eng], type_ignores=[]), "eng", "exec"), eng_ns)
        prob.engages = eng_ns["engages"]
        ns = dict(
            torch=types.SimpleNamespace(int64="I64"), logger=logger, prob1=prob, _RNG_SLOT_SPAN=100
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(f), "exec"), ns)
        checks = []
        for world in [2, 4, 8]:
            mapping = types.SimpleNamespace(tp_size=world, tp_rank=0)
            w = types.SimpleNamespace(
                _s4_logits_on=True, _s4_ranks_agree=lambda local: local, _s4_step_meta=(1, 3, None)
            )
            created.clear()
            decline[0] = None
            ns["prebuild"](w, mapping, "W1", "W2", 0, 200064, 7)
            assert created[0][:2] == (world, ("comm", world))
            md = types.SimpleNamespace(
                use_rejection_sampling=True,
                wants_advanced_draft_sampling=True,
                draft_probs="PROBS",
                batch_slot_ids=T("SLOTS"),
                request_offsets=T("OFF"),
                request_temperatures=T("TEMP"),
                request_seeds=T("SEEDS"),
                rng_draft_slot=lambda step: 9,
            )
            assert ns["engages"](w, md, w._p1.max_rows, types.SimpleNamespace(block=7))
            assert not ns["engages"](w, md, w._p1.max_rows + 1, types.SimpleNamespace(block=7))
            md.use_rejection_sampling = False
            assert not ns["engages"](w, md, 1, types.SimpleNamespace(block=7))
            md.use_rejection_sampling = True
            assert (
                ns["step"](
                    w,
                    types.SimpleNamespace(block=7),
                    T("BASE"),
                    {"first_prev_tokens": T("PREV")},
                    md,
                )
                == "TOKENS"
            )
            args = w._p1.calls[0]
            assert (
                args[5:7] == (900, 100)
                and args[-1] == "PROBS"
                and md.draft_probs_last_dim == 200064
                and args[2].label == "SLOTSslice"
            )
            checks.append(
                f"world{world}_collective_topology_maxrows_verifier_RNG_slots_proposal_contract"
            )
            for why, logits, agree in [
                ("binarymissing", True, True),
                (None, False, True),
                (None, True, False),
            ]:
                decline[0] = why
                w._s4_logits_on = logits
                w._s4_ranks_agree = lambda local: local and agree
                created.clear()
                ns["prebuild"](w, mapping, "W1", "W2", 0, 200064, 7)
                assert w._p1 is None and not created
            checks.append(f"world{world}_decline_binary_switch_peer_without_collectiveallocation")
        for tp in [2]:
            p = s / base / f"minimax_m3_1_flash_nvfp4__sm_107__tp{tp}/dspark_markov_worker.py"
            code = p.read_text()
            compile(code, str(p), "exec")
            t = ast.parse(code)
            assert (
                "prob1_worker.prebuild(" in code
                and ("prob1_worker.engages(" in code)
                and ("prob1_worker.step(" in code)
            )
            # Tokens use the existing shared S4 override, never the stock sampler after the proposal-producing draw.
            assert "tokens = self._s4_tokens" in code and "return tokens if tokens.dtype" in code
            cls = next(
                n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "S4WorkerMixin"
            )
            wrappers = [
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef)
                and n.name in ["_prebuild_prob1", "_p1_engages", "_p1_step"]
            ]
            # The TP2 kill switch reaches the shared prebuild as its target-level decline.
            wrappers += [
                n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "prob1_off_reason"
            ]
            delegate_calls = []
            adapter = types.SimpleNamespace(
                **{
                    name: (lambda *args, _name=name: delegate_calls.append((_name, args)) or _name)
                    for name in ["prebuild", "engages", "step"]
                }
            )
            wn = {
                "prob1_worker": adapter,
                "torch": types.SimpleNamespace(Tensor=object),
                "__builtins__": __builtins__,
                "_PROB1_KILLED": (),
            }
            exec(
                compile(
                    ast.fix_missing_locations(
                        ast.Module(
                            body=[
                                ast.ImportFrom(
                                    module="__future__",
                                    names=[ast.alias(name="annotations")],
                                    level=0,
                                )
                            ]
                            + wrappers,
                            type_ignores=[],
                        )
                    ),
                    "actualdelegates",
                    "exec",
                ),
                wn,
            )
            assert wn["_prebuild_prob1"]("worker", "mapping", "w1", "w2", 0, 100, 7) == "prebuild"
            assert wn["_p1_engages"]("worker", "md", 2, "state") == "engages"
            assert wn["_p1_step"]("worker", "state", "base", "inputs", "md") == "step"
            assert [x[0] for x in delegate_calls] == ["prebuild", "engages", "step"]
            assert delegate_calls[0][1][-1] is None
            wn["_PROB1_KILLED"] = ("TRTLLM_M31_DISABLE_DRAFT_PROB1",)
            wn["_prebuild_prob1"]("worker", "mapping", "w1", "w2", 0, 100, 7)
            assert delegate_calls[-1][1][-1] == "off (TRTLLM_M31_DISABLE_DRAFT_PROB1 set)"
            checks.append(f"TP{tp}_syntax_ACTUALdelegates_and_existing_tokenreturn")
        assert len(checks) == 7
