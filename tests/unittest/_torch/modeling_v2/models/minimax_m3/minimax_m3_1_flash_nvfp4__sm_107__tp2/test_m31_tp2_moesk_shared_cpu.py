# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the small-decode MoE engine's (SMALL_DECODE_MOE_ENGINE) host methods and kernel resource assertions with CPU
doubles. No device algorithm executes."""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only


class _NoImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return ast.copy_location(ast.Pass(), node)

    def visit_ImportFrom(self, node):
        return ast.copy_location(ast.Pass(), node)


def _load(path, scope):
    tree = _NoImports().visit(ast.parse(path.read_text()))
    tree.body.insert(
        0,
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope


def test_shared_moesk_actual_host_and_resource_admission():
    family = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    )
    disabled = set()
    image_calls = []
    logger = SimpleNamespace(info=lambda *a: None)
    torch = SimpleNamespace(
        uint8="u8", int32="i32", bfloat16="bf16", float32="f32", device=lambda d: d
    )
    moe_dir = family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_moe"
    shared = _load(
        moe_dir / "small_decode_engine.py",
        {
            "__file__": str(moe_dir / "small_decode_engine.py"),
            "Path": Path,
            "torch": torch,
            "m31_disabled": disabled.__contains__,
            "fc1_scale_image": lambda t: image_calls.append(t) or "IMAGE",
        },
    )
    api = SimpleNamespace(
        **{k: shared[k] for k in ("engine_path", "configure", "prepare_layer", "scale_words")}
    )
    canonical = Path(api.engine_path())
    # engine_path() resolves symlinks (the package may be a link to an installed tree): compare resolved paths.
    assert canonical == (moe_dir / "small_decode_engine_kernel.py").resolve()
    checks = 0
    for bulk_off in (False, True):
        for k128_off in (False, True):
            disabled.clear()
            if bulk_off:
                disabled.add("MOE_SFA_BULK")
            if k128_off:
                disabled.add("MOE_K128")
            for sfa_cp in (0, 1):
                c = {"sfa_cp": sfa_cp}
                api.configure(c, disabled.__contains__)
                assert bool(c.get("sfa_bulk")) == bool(sfa_cp and not bulk_off)
                assert bool(c.get("k128")) == (not k128_off)
                checks += 1
    disabled.clear()
    cfg = {"sfa_cp": 1}
    api.configure(cfg, disabled.__contains__, enabled=False)
    assert cfg == {"sfa_cp": 1}
    checks += 1

    class StopAtCompile(Exception):
        pass

    class Tensor:
        def __init__(self):
            self.views = []

        def view(self, *args):
            self.views.append(args)
            return self

        def reshape(self, *args):
            return self

        def to(self, *args):
            return self

    for tp in (2,):
        path = moe_dir / "op.py"
        props = SimpleNamespace(multi_processor_count=212)
        torch.cuda = SimpleNamespace(
            is_current_stream_capturing=lambda: False,
            get_device_properties=lambda d: props,
            current_stream=lambda d: SimpleNamespace(cuda_stream=0),
        )
        scope = _load(
            path,
            {
                "__file__": str(path),
                "Path": Path,
                "os": os,
                "sys": sys,
                "torch": torch,
                "logger": logger,
                "_flog": logger,
                "_vrpoll_log": logger,
                "_e0log": logger,
                "_os_log": logger,
                "_ost_log": logger,
                "m31_disabled": disabled.__contains__,
                "m31_moesk": api,
                "m3_report": SimpleNamespace(work=lambda *a: None),
            },
        )
        assert Path(scope["_ENGINE_PATH"]) == canonical
        source_set = scope["_aot_sources"]
        scope["cute_aot"] = SimpleNamespace(
            __file__=str(family / "fake_aot.py"),
            BLACKWELL_DIR=family / "fake_blackwell",
            py_files=lambda *a: ("target.py", "helpers.py"),
        )
        assert str(canonical) in source_set()
        scope["is_supported"] = lambda d: (True, "")
        scope["make_phase_plan"] = lambda *a, **kw: (Tensor(), 2)
        captured = []

        def compiler(config, engine):
            captured.append(dict(config))
            assert Path(engine) == (
                canonical
                if config.get("sfa_bulk") or config.get("k128")
                else path.with_name("m3_moe_kernel.py")
            )
            raise StopAtCompile

        scope["_kernel_module"] = compiler
        cls = scope["M3MoeState"]
        for rows in range(1, 9):
            try:
                cls("cuda", num_tokens=rows, shared=True, num_ctas=188)
            except StopAtCompile:
                pass
            else:
                raise AssertionError("must stop before allocations")
            cfg = captured[-1]
            assert cfg["i_tp"] == cfg["sh_i"] == 3072 // tp
            expected_on = rows == 8
            assert bool(cfg.get("sfa_bulk")) == bool(cfg.get("k128")) == expected_on
            # Execute the existing kernel's actual constants + resource assertions only.
            selected = canonical if expected_on else path.with_name("m3_moe_kernel.py")
            tree = ast.parse(selected.read_text())
            cutoff = min(
                n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.decorator_list
            )
            body = [
                n
                for n in tree.body
                if n.lineno < cutoff and not isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            ks = {"M3_CONFIG": cfg}
            exec(
                compile(ast.Module(body=body, type_ignores=[]), str(canonical), "exec"),
                ks,
            )
            assert (
                ks["I_TP"] == 3072 // tp
                and bool(ks.get("SFA_BULK")) == expected_on
                and bool(ks.get("K128")) == expected_on
            )
            assert ks["STAGES"] >= 3 and ks["SH_NST"] >= 3
            checks += 1
        if tp == 2:
            disabled.update({"MOE_SFA_BULK", "MOE_K128"})
            try:
                cls("cuda", num_tokens=8, shared=True, num_ctas=188)
            except StopAtCompile:
                pass
            else:
                raise AssertionError("both-killed legacy state must stop before allocation")
            assert not captured[-1].get("sfa_bulk") and not captured[-1].get("k128")
            disabled.clear()
            checks += 1
            # TRTLLM_M31_DISABLE_SMALL_DECODE_MOE_ENGINE restores the original TP2 engine at the eight-row cell too.
            assert scope["_MOESK_KILLED"] is False
            scope["_MOESK_KILLED"] = True
            try:
                cls("cuda", num_tokens=8, shared=True, num_ctas=188)
            except StopAtCompile:
                pass
            else:
                raise AssertionError("MOESK-killed state must stop before allocation")
            assert not captured[-1].get("sfa_bulk") and not captured[-1].get("k128")
            scope["_MOESK_KILLED"] = False
            checks += 1
            # TP2WC_MOE_ENGINE builds the reference engine on the plain schedule: the eight-row state with the
            # MOE_HANDOFF tables carries epi_vskip, which the reference engine refuses; handoff=False carries none of
            # the keys it refuses.
            engine_src = ast.parse((family / "kernels/m3ll/tp4_moe_engine.py").read_text())
            not_taken = next(
                ast.literal_eval(n.value)
                for n in engine_src.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "NOT_TAKEN_KEYS" for t in n.targets)
            )
            for handoff, carries in ((None, True), (False, False)):
                try:
                    cls("cuda", num_tokens=8, shared=True, num_ctas=188, handoff=handoff)
                except StopAtCompile:
                    pass
                else:
                    raise AssertionError("must stop before allocations")
                assert bool(captured[-1].get("epi_vskip")) == carries
                if not carries:
                    assert not [k for k in not_taken if captured[-1].get(k)], captured[-1]
            checks += 1
        for rows in (0, 9):
            try:
                cls("cuda", num_tokens=rows, shared=True, num_ctas=188)
            except ValueError:
                checks += 1
            else:
                raise AssertionError("invalid row count admitted")
        # Actual layer method must eagerly select the family image; no allocation or CUDA needed.
        layer = SimpleNamespace(w3_w1_weight_scale=Tensor())
        scope["M3MoeLayer"] = lambda *a: layer
        state = SimpleNamespace(config={"sfa_bulk": 1}, shared=False)
        assert cls.layer(state, *([None] * 11)) is layer and layer.sfa1_img == "IMAGE"
        assert image_calls[-1] is layer.w3_w1_weight_scale
        assert api.scale_words(state.config, layer) == "IMAGE"
        state.config = {}
        assert api.scale_words(state.config, layer) is layer.w3_w1_weight_scale
        assert layer.w3_w1_weight_scale.views[-2:] == [("u8",), ("i32",)]
        checks += 4
        # Exercise actual run argument assembly: the secondary FC1 word-image operand must switch,
        # while the first raw FC1 scale operand remains unchanged in the ABI.
        tensor = Tensor()
        tensor.dtype = "bf16"
        tensor.shape = (4, 6144)
        tensor.is_contiguous = lambda: True
        tensor.device = "cuda"
        for name in (
            "w3_w1_weight",
            "w2_weight",
            "w2_weight_scale",
            "router_weight",
            "fc31_input_scale",
            "bias",
            "escal",
        ):
            setattr(layer, name, tensor)
        for bulk in (False, True):
            calls = []
            state = SimpleNamespace(
                m=4,
                device="cuda",
                shared=False,
                config={"sfa_bulk": int(bulk)},
                items=set(),
                _fn=lambda *a: calls.append(a),
            )
            for name in (
                "_dummy",
                "lgbuf",
                "plan",
                "partials",
                "hbuf32",
                "epochs",
                "route_dbg",
                "tdbg",
                "hx",
                "dnx",
                "_ar_off",
                "img32",
                "cnt",
            ):
                setattr(state, name, tensor)
            assert cls.run(state, layer, tensor, out=tensor) is tensor
            assert calls[0][1] is layer.w3_w1_weight_scale
            # The word-image operand precedes the FC2 scale words, img32, cnt and (hvpf) the flat FC2 weight bytes.
            assert calls[0][-6] == ("IMAGE" if bulk else layer.w3_w1_weight_scale)
            checks += 1
    assert checks == 28


def test_tp2_moesk_kill_switch_env():
    """The TP2 kill is read at import with the target's strict switch parser."""
    family = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    )
    path = family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/m3_moe/op.py"
    tree = ast.parse(path.read_text())
    stmt = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_MOESK_KILLED" for t in n.targets)
    )
    expr = compile(ast.Expression(stmt.value), str(path), "eval")

    def switches(env):
        return _load(
            family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/m31_switches.py",
            {
                "os": SimpleNamespace(environ=env),
                "Dict": dict,
                "FrozenSet": frozenset,
                "logger": SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
            },
        )

    for env, want in (
        ({}, False),
        ({"TRTLLM_M31_DISABLE_SMALL_DECODE_MOE_ENGINE": "1"}, True),
        ({"TRTLLM_M31_DISABLE_SMALL_DECODE_MOE_ENGINE": "0"}, False),
        ({"TRTLLM_M31_DISABLE_SMALL_DECODE_MOE_ENGINE": "true"}, True),
    ):
        assert eval(expr, {"m31_disabled": switches(env)["m31_disabled"]}) is want, env
    try:
        switches({"TRTLLM_M31_DISABLE_SMALL_DECODE_MOE_ENGINE": "maybe"})
    except ValueError:
        pass
    else:
        raise AssertionError("a malformed kill value must raise")


if __name__ == "__main__":
    test_shared_moesk_actual_host_and_resource_admission()
    test_tp2_moesk_kill_switch_env()
    print("TP2_MOESK_SHARED_CPU_PASS")
