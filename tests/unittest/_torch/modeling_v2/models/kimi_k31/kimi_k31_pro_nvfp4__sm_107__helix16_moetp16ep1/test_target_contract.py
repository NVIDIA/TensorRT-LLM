# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contracts of ``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1``: the recipe its ``features.py``
accepts, the one layout it owns, the family router's branch to it, and its profile selection.

The target's recipe module (``ops/common/rubin_ull_features.py``) is loaded from its file into a
private package (with a private copy of the switchboard), so the profile selected here stays out
of the process's real modules. Needs the engine package importable.
"""

import ast
import copy
import dataclasses
import importlib.util
import itertools
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CODING_GUIDELINES.md").is_file())
FAMILY = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k31"
TARGET = "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
SWITCHES = "tensorrt_llm._torch.decode_switches"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


class TargetContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_modules = dict(sys.modules)
        # The engine package first (the target's features module imports from it), then a private copy of the
        # switchboard in its place: the target's profiles are built from that copy's DecodeSwitches class.
        importlib.import_module("tensorrt_llm.quantization")
        cls.switch = load(SWITCHES, ROOT / "tensorrt_llm/_torch/decode_switches.py")
        package("_k31_contract_target", FAMILY / TARGET)
        cls.features = load(
            "_k31_contract_target.features", FAMILY / "ops/common/rubin_ull_features.py"
        )

    @classmethod
    def tearDownClass(cls):
        for name in set(sys.modules) - set(cls.saved_modules):
            del sys.modules[name]
        sys.modules.update(cls.saved_modules)

    def layout(self, **changes):
        fields = dict(
            world_size=16,
            tp_size=1,
            pp_size=1,
            cp_size=16,
            helix=True,
            attention_dp=False,
            moe_tp_size=16,
            moe_ep_size=1,
        )
        fields.update(changes)
        return self.features.Layout(**fields)

    def decision(self, **changes):
        fields = dict(
            indexer_k_dtype="fp8",
            kv_cache_quant_algo="FP8",
            spec_mode="DFLASH",
            max_draft_len=7,
            moe_backend="CUTEDSL",
            layout=self.layout(),
            role="generation",
        )
        fields.update(changes)
        return self.features.decide(**fields)

    def test_ull_recipe(self):
        for backend in self.features.MOE_BACKENDS:
            with self.subTest(backend=backend):
                result = self.decision(moe_backend=backend)
                self.assertIsNone(result.reason)
                self.assertEqual(result.topology, "ull16")
                self.assertEqual(result.target, TARGET)

    def test_exclusive_layout(self):
        for world, tp, cp, pp, helix, adp, moe in itertools.product(
            (8, 16, 32),
            (1, 8, 16),
            (1, 8, 16),
            (1, 2),
            (False, True),
            (False, True),
            ((1, 16), (2, 4), (4, 4), (16, 1)),
        ):
            layout = self.layout(
                world_size=world,
                tp_size=tp,
                cp_size=cp,
                pp_size=pp,
                helix=helix,
                attention_dp=adp,
                moe_tp_size=moe[0],
                moe_ep_size=moe[1],
            )
            expected = (world, tp, cp, pp, helix, adp, moe) == (16, 1, 16, 1, True, False, (16, 1))
            self.assertEqual(self.features.topology(layout) == "ull16", expected, str(layout))

    def test_invalid_recipe_rejected(self):
        for changes in (
            dict(role="context"),
            dict(indexer_k_dtype="fp4"),
            dict(kv_cache_quant_algo="NVFP4"),
            dict(spec_mode=None, max_draft_len=None),
            dict(spec_mode="DSPARK"),
            dict(spec_mode="MTP"),
            dict(max_draft_len=5),
            dict(moe_backend="TRTLLM"),
        ):
            with self.subTest(changes=changes):
                self.assertIsNotNone(self.decision(**changes).reason)
        # A context server is sent to the DFlash2 context targets.
        reason = self.decision(role="context").reason
        for target in ("kimi_k31_pro_nvfp4__sm_107__dep16", "kimi_k31_pro_nvfp4__sm_107__tep8"):
            self.assertIn(target, reason)

    def test_other_topologies_are_not_ull16(self):
        for changes, expected in (
            (
                dict(
                    tp_size=16,
                    cp_size=1,
                    helix=False,
                    attention_dp=True,
                    moe_tp_size=1,
                    moe_ep_size=16,
                ),
                "dep16",
            ),
            (dict(world_size=8, cp_size=8, moe_tp_size=1, moe_ep_size=8), "helix8"),
            (dict(world_size=8, cp_size=8, moe_tp_size=2, moe_ep_size=4), "helix8"),
            (
                dict(world_size=8, tp_size=8, cp_size=1, helix=False, moe_tp_size=1, moe_ep_size=8),
                "tep8",
            ),
        ):
            with self.subTest(layout=expected):
                self.assertNotEqual(self.features.topology(self.layout(**changes)), "ull16")

    def test_family_dispatch_boundary(self):
        # Run the family router's ULL branch (in ``_route``, behind the logging ``route``) with
        # stand-ins for what it calls.
        tree = ast.parse((FAMILY / "routing.py").read_text())
        route = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_route"
        )

        def calls(node, name):
            return any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == name
                for child in ast.walk(node)
            )

        generation = next(
            node
            for node in route.body
            if isinstance(node, ast.If)
            and any(calls(child, "_route_ull_generation") for child in node.body)
        )
        function = ast.parse("def dispatch(model_config, llm_args): return 'existing'").body[0]
        function.body[:0] = [
            ast.parse("mapping = model_config.mapping").body[0],
            copy.deepcopy(generation),
        ]
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = dict(
            compute_capability=(10, 7),
            _ULL_COMPUTE_CAPABILITIES=((10, 3), (10, 7)),
            _indexer_k_dtype=lambda *args: "fp8",
            _route_ull_generation=lambda *args: TARGET,
        )
        exec(compile(module, "<family ULL dispatch>", "exec"), namespace)

        def model(layout):
            fields = dataclasses.asdict(layout)
            fields["enable_attention_dp"] = fields.pop("attention_dp")
            fields["has_cp_helix"] = lambda: layout.helix
            return types.SimpleNamespace(mapping=types.SimpleNamespace(**fields))

        for changes, expected in (
            ({}, TARGET),
            ({"cp_size": 8, "world_size": 8}, "existing"),
            ({"moe_tp_size": 4, "moe_ep_size": 4}, "existing"),
            ({"tp_size": 16, "cp_size": 1}, "existing"),
            ({"attention_dp": True}, "existing"),
            ({"pp_size": 2}, "existing"),
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    namespace["dispatch"](model(self.layout(**changes)), None), expected
                )
        tep8 = model(
            self.layout(
                world_size=8, tp_size=8, cp_size=1, helix=False, moe_tp_size=1, moe_ep_size=8
            )
        )
        self.assertEqual(namespace["dispatch"](tep8, None), "existing")

    def test_generation_profile_has_no_batch_dependent_subprofile(self):
        for batch, draft in itertools.product((1, 2, 4, 8, 24), (0, 5, 7)):
            self.assertEqual(
                str(self.features.profile_key("ull16", "generation", batch, draft)),
                "ull16/generation",
            )
        with self.assertRaises(ValueError):
            self.features.ProfileKey("ull16", "context")
        with self.assertRaises(ValueError):
            self.features.ProfileKey("helix8", "generation")

    def test_profile_selection_is_private(self):
        features = self.features
        saved_active, saved_key = features._ACTIVE, features._SELECTED
        try:
            selected = features.select(features.ULL16_GENERATION)
            self.assertTrue(selected.router_bf16)
            self.assertFalse(selected.kda_h8_mx_pdl)
            # select() configures the switchboard with the profile's switches (the Helix key-shard
            # indexer among them).
            self.assertEqual(self.switch.active(), features.SWITCHES[features.ULL16_GENERATION])
            self.assertTrue(self.switch.active().dsa_helix_key_shard)
        finally:
            features._ACTIVE, features._SELECTED = saved_active, saved_key

    def test_select_reads_no_environment(self):
        """The flat chain's switches and the retired engine variables have no effect: select()
        reads no environment variable (the switchboard's retired-variable warning is skipped)."""
        features = self.features
        saved_active, saved_key = features._ACTIVE, features._SELECTED
        stale = {
            "TLLM_K31_FLAT_DSA_CORE": "fmha",
            "TLLM_K31_FLAT_KDA_HEAD": "0",
            "TRTLLM_DSA_KEY_SHARD_STATS": "1000",
            "TRTLLM_K3_FC12_PREZERO": "1",
        }
        self.switch._reset()
        try:
            with (
                mock.patch.dict(os.environ, stale),
                mock.patch.object(self.switch, "warn_retired_env") as warn,
            ):
                selected = features.select(features.ULL16_GENERATION)
            warn.assert_not_called()
            self.assertEqual(selected, features.PROFILES[features.ULL16_GENERATION])
            self.assertEqual(
                (selected.dsa_core, selected.dsa_stats_every),
                ("sparse", 0),
            )
        finally:
            features._ACTIVE, features._SELECTED = saved_active, saved_key
            self.switch._reset()

    def test_served_defaults_of_the_flat_fields(self):
        served = self.features.K31Features()
        self.assertEqual((served.dsa_core, served.dsa_core_max_rows), ("sparse", 8))
        self.assertFalse(served.dsa_core_check)
        self.assertEqual(served.dsa_stats_every, 0)
        self.assertFalse(served.ring_fill_check)
        self.assertNotIn("moe_grouped_polled", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("kda_head_multi16", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("moe_tail16_nosleep", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("moe_tail_wide_nosleep", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("moe_tail16_late_rms", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("oproj24_carveout", self.features.MODEL_CONFIG_FIELDS)
        # The 17-24-row DSA in-projection is fixed and 25-32 rows use the regular plan.
        self.assertNotIn("dsa_inproj24", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("dsa_inproj32_vendored", self.features.MODEL_CONFIG_FIELDS)
        # The rows > 8 sparse MLA runs one CTA per row; model_kwargs turn it off.
        self.assertNotIn("dsa_fmha_single_cta", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("draft_wide_rows", self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("tailgate_wide_ctas", self.features.MODEL_CONFIG_FIELDS)
        # The fused select-accept's sequence-lengths copy off the main stream (``glue_h1a``).
        self.assertFalse(served.glue_h1a)
        for field in (
            "glue_h1b",
            "glue_pfold",
            "ctx_first_drafts",
            "draft_head_selector_concat",
            "draft_rank_topk_walk",
            "draft_agree_multicast",
            "draft_absorbed_query",
            "draft_conv_fold",
            "draft_query_fold",
            "dsa_fused_index",
            "dsa_merge_decode",
            "kda_head",
            "kda_head_order",
            "draft_attention",
            "draft_rows16",
        ):
            self.assertNotIn(field, self.features.MODEL_CONFIG_FIELDS)
        self.assertNotIn("kda_fold_wide", self.features.MODEL_CONFIG_FIELDS)
        with self.assertRaises(ValueError):
            self.features.K31Features(dsa_core="r9")
        with self.assertRaises(ValueError):
            self.features.K31Features(dsa_stats_every=-1)

    def test_model_config_fields(self):
        """``LlmArgs.model_kwargs`` (pretrained-config attributes ``kimi_k31_<field>``) set the
        diagnostics and experimental levers; nothing else, and
        nothing when none is set."""
        features = self.features
        saved = features._ACTIVE
        try:
            base = features.K31Features(router_bf16=True)
            features._ACTIVE = base
            self.assertIs(features.apply_model_config(types.SimpleNamespace()), base)
            config = types.SimpleNamespace(
                kimi_k31_dsa_stats_every=1000,
                kimi_k31_dsa_core_check=True,
                kimi_k31_ring_fill_check=True,
                kimi_k31_glue_h1a=False,
                kimi_k31_dsa_core="fmha",  # not a model-config field: ignored
            )
            applied = features.apply_model_config(config)
            self.assertEqual(
                (applied.dsa_stats_every, applied.dsa_core_check),
                (1000, True),
            )
            self.assertTrue(applied.ring_fill_check)
            self.assertFalse(applied.glue_h1a)
            self.assertEqual(applied.dsa_core, "sparse")
            self.assertTrue(applied.router_bf16)
            self.assertIs(features.active(), applied)
            for bad in (
                types.SimpleNamespace(kimi_k31_dsa_stats_every="1000"),
                types.SimpleNamespace(kimi_k31_dsa_stats_every=True),
            ):
                with self.assertRaises(TypeError):
                    features.apply_model_config(bad)
        finally:
            features._ACTIVE = saved

    def test_target_reads_no_environment(self):
        """No file of the target reads (or writes) ``os.environ`` / ``getenv``: the target directory, the
        family kernels and ops it owns (``rubin_ull_*`` and ``kernels/vendor/``), and every family file it
        imports, directly or through those (absolute, relative and dotted-string imports). Of a shared
        family file (one the target does not own: the DFlash2 drafter's ``ops/spec`` entries), only the
        module-level imports are followed; its function-level ones load the torch references of its
        ``reference`` oracles (the family's other targets, ``ops/spec/dflash2_verify.py``).

        The exceptions: the vendored ``symm.py``, which sets torch's TORCH_SYMM_MEM_DISABLE_MULTICAST
        around its own rendezvous (torch's only way to skip the multicast object; not a switch of the
        target), and the shared ``ops/spec/dflash2_sampling.py``, whose proposal store names
        TLLM_K31_DFLASH2_VERIFY in its startup log line (this target always verifies blocks)."""
        allowed = {
            "kernels/vendor/vllm_main/specialized_decode/symm.py",
            # The family's shared DFlash2 sampling entry (not this target's file): its read of
            # TLLM_K31_DFLASH2_VERIFY only labels the proposal store's startup log line; helix8
            # makes the same read.
            "ops/spec/dflash2_sampling.py",
        }
        package = ".".join(FAMILY.relative_to(ROOT).parts)
        target = FAMILY / TARGET

        def module_file(name):
            base = ROOT.joinpath(*name.split("."))
            for path in (base.with_suffix(".py"), base / "__init__.py"):
                if path.is_file():
                    return path
            return None

        def imported(path, tree, shared):
            here = ".".join(path.relative_to(ROOT).parent.parts)  # where relative imports start
            nodes = ast.walk(tree)
            if shared:  # module level only: not inside a function
                local = {
                    id(n)
                    for f in ast.walk(tree)
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for n in ast.walk(f)
                }
                nodes = [n for n in ast.walk(tree) if id(n) not in local]
            for node in nodes:
                if isinstance(node, ast.Import):
                    yield from (alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        anchor = here.split(".")[: len(here.split(".")) - node.level + 1]
                        base = ".".join(anchor + ([node.module] if node.module else []))
                    yield base
                    yield from (f"{base}.{alias.name}" for alias in node.names)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.startswith(package + "."):
                        yield node.value

        owned = [*target.rglob("*.py"), *(FAMILY / "kernels/vendor").rglob("*.py")]
        owned += [*FAMILY.glob("kernels/*/rubin_ull_*.py"), *FAMILY.glob("ops/*/rubin_ull_*.py")]
        owned_set = set(owned)
        todo, scanned, found = list(owned), {}, []
        while todo:
            path = todo.pop()
            rel = path.relative_to(FAMILY).as_posix()
            if rel in scanned:
                continue
            scanned[rel] = tree = ast.parse(path.read_text(), rel)
            for name in imported(path, tree, path not in owned_set):
                parts = name.split(".")
                for end in range(len(package.split(".")) + 1, len(parts) + 1):
                    if ".".join(parts[:end]).startswith(package + "."):
                        dep = module_file(".".join(parts[:end]))
                        if dep is not None:
                            todo.append(dep)
        self.assertIn("kernels/vendor/vllm_main/specialized_decode/symm.py", scanned)
        self.assertIn("ops/spec/dflash2_sampling.py", scanned)
        for rel, tree in sorted(scanned.items()):
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.alias):  # from os import environ / getenv
                    name = node.name
                else:
                    name = getattr(node, "id", None)
                if name in ("environ", "getenv", "putenv", "environb") and rel not in allowed:
                    found.append(f"{rel}:{getattr(node, 'lineno', 0)} {name}")
        self.assertEqual(found, [])

    def test_shared_ops_oracle_imports_stay_in_reference(self):
        """The environment scan above follows only the module-level imports of the shared DFlash2 ops.
        Their function-level imports of family modules that read the environment (the torch oracles,
        ``ops/spec/dflash2_verify.py``) sit in ``reference`` or in module functions that only
        ``reference`` reaches, so no served call (``__call__``, ``is_valid``, the module functions the
        target imports) loads them. Read off the source: each shared ``ops/spec`` file's call graph
        among its own functions and methods."""
        package = ".".join(FAMILY.relative_to(ROOT).parts)

        def reads_environment(path):
            for node in ast.walk(ast.parse(path.read_text())):
                name = getattr(node, "attr", None) or getattr(node, "id", None)
                if name in ("environ", "getenv"):
                    return True
            return False

        def resolve(path, node):
            here = ".".join(path.relative_to(ROOT).parent.parts).split(".")
            base = node.module or ""
            if node.level:
                base = ".".join(here[: len(here) - node.level + 1] + ([base] if base else []))
            names = [base] + [f"{base}.{a.name}" for a in node.names]
            for name in names:
                if name.startswith(package + "."):
                    file = ROOT.joinpath(*name.split(".")).with_suffix(".py")
                    if file.is_file():
                        yield file

        found, checked = [], 0
        for path in sorted((FAMILY / "ops/spec").glob("*.py")):
            if path.name.startswith(("rubin_ull_", "_rubin_ull_")):
                continue
            tree = ast.parse(path.read_text())
            functions = {
                f.name: f
                for f in ast.walk(tree)
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            calls = {
                name: {
                    getattr(c.func, "attr", None) or getattr(c.func, "id", None)
                    for c in ast.walk(f)
                    if isinstance(c, ast.Call)
                }
                for name, f in functions.items()
            }
            served = {n for n in functions if n != "reference"}
            # The functions that run outside reference: every function except reference and those
            # only reference (directly or through others only it reaches) calls.
            oracle = {"reference"}
            grew = True
            while grew:
                grew = False
                for name in functions:
                    if name in oracle:
                        continue
                    callers = {c for c, called in calls.items() if name in called}
                    if callers and callers <= oracle and not name.startswith(("__", "is_valid")):
                        oracle.add(name)
                        grew = True
            served -= oracle
            for name in sorted(functions):
                for node in ast.walk(functions[name]):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    for dep in resolve(path, node):
                        if reads_environment(dep):
                            checked += 1
                            if name in served:
                                found.append(f"{path.name}:{node.lineno} {name} imports {dep.name}")
        self.assertGreater(checked, 0, "no oracle import found: the scan no longer sees them")
        self.assertEqual(found, [])

    def test_target_dynamic_imports_resolve(self):
        """Every ``import_module`` target in the target and the family files it owns (``rubin_ull_*``
        and ``kernels/vendor/``) names a module of the tree: string constants, f-strings of module-level
        string constants, and relative names against ``__package__``. Names built from runtime values
        are skipped. Such imports sit in ``try`` / ``except ImportError``, so a string that a move left
        pointing nowhere only falls back silently at run time."""
        known_unresolved = set()
        owned = [*(FAMILY / TARGET).rglob("*.py"), *(FAMILY / "kernels/vendor").rglob("*.py")]
        owned += [*FAMILY.glob("kernels/*/rubin_ull_*.py"), *FAMILY.glob("ops/*/rubin_ull_*.py")]
        resolved, unresolved = [], set()
        for path in owned:
            rel = path.relative_to(FAMILY).as_posix()
            tree = ast.parse(path.read_text(), rel)
            strings = {}  # module-level names bound to a string constant
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target, value = getattr(node.targets[0], "id", None), node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        strings[target] = value.value
                    elif isinstance(value, ast.Name) and value.id in strings:
                        strings[target] = strings[value.id]

            def text(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    return node.value
                if isinstance(node, ast.JoinedStr):
                    parts = []
                    for part in node.values:
                        if isinstance(part, ast.FormattedValue):
                            part = getattr(part.value, "id", None)
                            if part not in strings:
                                return None
                            parts.append(strings[part])
                        else:
                            parts.append(part.value)
                    return "".join(parts)
                return None

            package = ".".join(path.relative_to(ROOT).parent.parts)  # __package__ of the file
            for node in ast.walk(tree):
                func = node.func if isinstance(node, ast.Call) else None
                if getattr(func, "attr", getattr(func, "id", None)) != "import_module":
                    continue
                name = text(node.args[0]) if node.args else None
                if name is None:
                    continue
                full = name
                if name.startswith("."):
                    self.assertEqual(
                        getattr(node.args[1], "id", None), "__package__", f"{rel}: {name}"
                    )
                    level = len(name) - len(name.lstrip("."))
                    anchor = package.split(".")[: len(package.split(".")) - level + 1]
                    full = ".".join(anchor + [name.lstrip(".")])
                if not full.startswith("tensorrt_llm."):
                    continue
                base = ROOT.joinpath(*full.split("."))
                if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
                    resolved.append((rel, name))
                else:
                    unresolved.add((rel, name))
        self.assertEqual(unresolved, known_unresolved)
        # The scan still resolves a relative string: the KDA kernels' verify op loader.
        self.assertIn(
            ("kernels/ssm/rubin_ull_kda_kernels.py", ".rubin_ull_kda_cute_dsl_kimi_k3_kda_mtp_ops"),
            resolved,
        )


if __name__ == "__main__":
    unittest.main()
