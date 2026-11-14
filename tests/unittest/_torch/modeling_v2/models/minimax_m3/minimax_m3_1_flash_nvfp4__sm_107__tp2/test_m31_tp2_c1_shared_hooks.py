# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the TP2 FRONT_CAPFORK port: the family ``capture_or_inline`` selection, stream order and source
lifetime, the TP2 worker delegate and kill switches, and the target forward's join, using actual ASTs and
stream / metadata doubles. No GPU.
"""

import pytest

pytestmark = pytest.mark.cpu_only


def test_capture_fork_selection_lifetime_and_delegates():
    import ast
    import contextlib
    import gc
    import io
    import types
    import weakref
    from pathlib import Path

    r = Path(__file__).resolve().parents[7]
    fam = r / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    p = fam / "m31_capture_fork.py"
    tree = ast.parse(p.read_text())
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("wait", self.name, other.name))

    main = Stream("main")
    current = [main]

    class Device:
        def __init__(self, index=0):
            self.index = index

    @contextlib.contextmanager
    def in_stream(stream):
        old = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = old

    cuda = types.SimpleNamespace(
        current_device=lambda: 0,
        current_stream=lambda device=None: current[0],
        Stream=lambda device=None: Stream("aux" + str(device)),
        stream=in_stream,
    )
    torch = types.SimpleNamespace(
        cuda=cuda,
        device=lambda x: x,
        distributed=types.SimpleNamespace(is_available=lambda: False),
    )
    ns = dict(
        torch=torch,
        os=types.SimpleNamespace(environ={}),
        sys=types.SimpleNamespace(stderr=io.StringIO()),
        _STREAMS={},
        _PENDING={},
        _LOGGED=set(),
    )
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    body.insert(
        0,
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(p),
            "exec",
        ),
        ns,
    )

    class Hidden:
        def __init__(self, rows=8, cuda=True, index=0):
            self.shape = (rows, 6144)
            self.is_cuda = cuda
            self.device = Device(index)

    class Metadata:
        def maybe_capture_hidden_states(self, layer, h, residual):
            events.append(("copy", current[0].name, layer, id(h), id(residual)))

    md = Metadata()
    checks = []
    for rows in [1, 4, 8]:
        events.clear()
        h = Hidden(rows)
        ns["capture_or_inline"](md, 3, h, None, "sm107_tp2", True, max_rows=8)
        assert events[0] == ("wait", "aux0", "main") and events[1][:3] == (
            "copy",
            "aux0",
            3,
        )
        assert ns["pending"]() == 1 and ns["_PENDING"][0][0] is h
        ref = weakref.ref(h)
        del h
        gc.collect()
        assert ref() is not None
        assert ns["join"]() == 1 and events[-1] == ("wait", "main", "aux0")
        gc.collect()
        assert ref() is None
        before = len(events)
        assert ns["join"]() == 0 and len(events) == before
    checks += [
        "actual_family_wait_copy_join_order_rows1_4_8",
        "source_lifetime_retained_until_join",
        "empty_join_no_stream_work",
    ]
    for name, rows, cuda_value, layer, residual, enabled in [
        ("kill", 8, True, 3, None, False),
        ("final", 8, True, -1, None, True),
        ("residual", 8, True, 3, object(), True),
        ("cpu", 8, False, 3, None, True),
        ("wide", 9, True, 3, None, True),
        ("empty", 0, True, 3, None, True),
    ]:
        events.clear()
        h = Hidden(rows, cuda_value)
        ns["capture_or_inline"](md, layer, h, residual, "sm107_tp2", enabled, max_rows=8)
        assert events == [("copy", "main", layer, id(h), id(residual))] and ns["pending"]() == 0, (
            name
        )
        checks.append("inline_" + name)
    # Existing TP4/8 have no new row limit; their dispatch conditions remain unchanged.
    for tp in [4, 8]:
        events.clear()
        h = Hidden(16)
        ns["capture_or_inline"](md, 3, h, None, f"sm107_tp{tp}", True)
        assert ns["pending"]() == 1
        ns["join"]()
        checks.append(f"TP{tp}_existing_rowcoverage_preserved")
    for tp in [2]:
        f = fam / "minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark.py"
        t = ast.parse(f.read_text())
        c = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "M3DSparkWorker")
        n = next(
            n
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == "capture_hidden_states"
        )
        wns = dict(
            m31_capture_fork=types.SimpleNamespace(
                capture_or_inline=lambda *args, **kwargs: events.append(("delegate", args, kwargs))
            ),
            FRONT_CAPFORK=True,
            FRONT_CAPFORK_MAX_ROWS=8,
            FRONT_CAPFORK_BASE_ROWS=8,
        )
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.ImportFrom(
                                module="__future__",
                                names=[ast.alias(name="annotations")],
                                level=0,
                            ),
                            n,
                        ],
                        type_ignores=[],
                    )
                ),
                str(f),
                "exec",
            ),
            wns,
        )
        events.clear()
        h = Hidden()
        wns["capture_hidden_states"](types.SimpleNamespace(), md, 3, h, None)
        assert events[0][0] == "delegate" and events[0][1][4] == f"sm107_tp{tp}"
        assert events[0][2] == ({"max_rows": 8} if tp == 2 else {})
        if tp == 2:
            assignment = next(
                a
                for a in t.body
                if isinstance(a, ast.Assign)
                and any(isinstance(z, ast.Name) and z.id == "FRONT_CAPFORK" for z in a.targets)
            )
            # Default ON; its own and the host capture tail's kill switches each turn it off.
            for killed in (set(), {"FRONT_CAPFORK"}, {"HOST_CAPTURE_TAIL"}):
                env = {"m31_disabled": lambda key, s=killed: key in s}
                exec(
                    compile(ast.Module(body=[assignment], type_ignores=[]), "switch", "exec"),
                    env,
                )
                assert env["FRONT_CAPFORK"] == (not killed), killed

            def _assign(name):
                return next(
                    a
                    for a in t.body
                    if isinstance(a, ast.Assign)
                    and any(isinstance(z, ast.Name) and z.id == name for z in a.targets)
                )

            assert ast.literal_eval(_assign("FRONT_CAPFORK_BASE_ROWS").value) == 8
            # TP2WIDE capfork wide: 32 rows by default; its switch or the track group restores the base 8.
            for scoped, want in (
                (set(), 32),
                ({"FRONT_CAPFORK_WIDE"}, 8),
                ({"TP2WIDE_COMMGEMV"}, 8),
            ):
                env = {
                    "m31_disabled": lambda key, s=scoped: key in s,
                    "FRONT_CAPFORK_BASE_ROWS": 8,
                }
                for name in ("FRONT_CAPFORK_WIDE", "FRONT_CAPFORK_MAX_ROWS"):
                    exec(
                        compile(ast.Module(body=[_assign(name)], type_ignores=[]), name, "exec"),
                        env,
                    )
                assert env["FRONT_CAPFORK_MAX_ROWS"] == want, (
                    scoped,
                    env["FRONT_CAPFORK_MAX_ROWS"],
                )
        checks.append(f"TP{tp}_actual_delegation_and_TP2defaultkill")
        f.read_text()
        compile(f.read_text(), str(f), "exec")
    # Verify the actual target forward joins after model() and before every capture/logits/worker consumer.
    p = fam / "minimax_m3_1_flash_nvfp4__sm_107__tp2/modeling.py"
    t = ast.parse(p.read_text())
    c = next(
        n
        for n in t.body
        if isinstance(n, ast.ClassDef) and n.name.startswith("ModelingV2MiniMaxM31")
    )
    fn = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    lines = ast.unparse(fn)
    assert (
        lines.index("hidden_states = self.model(")
        < lines.index("m31_capture_fork.join()")
        < lines.index("getattr(attn_metadata, 'csa2_remote_tail_mode'")
    )
    assert lines.index("m31_capture_fork.join()") < lines.index(
        "capture(spec_metadata, FINAL_HIDDEN_LAYER_ID"
    )
    compile(p.read_text(), str(p), "exec")
    checks.append("actual_TP2_forward_join_before_all_consumers")


if __name__ == "__main__":
    test_capture_fork_selection_lifetime_and_delegates()
    print("TP2_FRONT_CAPFORK_CPU_PASS")
