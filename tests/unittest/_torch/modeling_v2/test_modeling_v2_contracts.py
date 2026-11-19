# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code contracts of Modeling V2 code: the catalog entries and their cells, and the
multimodal contract and naming rule of the targets of vision-language architectures.

An entry computes a pure function of its arguments, so its ``CELLS`` cover its
whole input space. A vision-language target runs media through the multimodal
runtime or refuses it at admission, and a vision-language model subclasses the
shared base and its name ends with ``ForConditionalGeneration``. These checks
keep that true. They read the source and import no target, so they run on any
CPU machine; a text-only target's admission hook runs on its own, with the shared
helper. Which architectures take media is the router's own check, from the
built-in input processors.
"""

from __future__ import annotations
import __future__

import ast
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/modeling_v2"
_UNIT_ROOT = Path(__file__).resolve().parent


def _entry_files() -> list[Path]:
    """Every entry module: the shared catalog, each family's ``ops/`` and each target's."""
    shared = (_ROOT / "catalog").glob("*/*.py")
    family = (_ROOT / "models").glob("*/ops/*/*.py")
    owned = (_ROOT / "models").glob("*/*/ops/*/*.py")
    return sorted(p for p in (*shared, *family, *owned) if p.name != "__init__.py")


def test_entry_discovery_covers_family_level_ops(tmp_path, monkeypatch):
    entries = {
        tmp_path / "catalog/attention/shared.py",
        tmp_path / "models/family/ops/attention/family.py",
        tmp_path / "models/family/target/ops/attention/target.py",
    }
    for path in entries:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        (path.parent / "__init__.py").touch()
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    assert set(_entry_files()) == entries


def _catalog_entries() -> list[tuple[Path, ast.ClassDef]]:
    """Every OpWrapper subclass, shared or owned, as (path, class node)."""
    found = []
    for path in _entry_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id == "OpWrapper" for b in node.bases
            ):
                found.append((path, node))
    return found


def test_call_reads_no_instance_state():
    """`__call__` takes every argument explicitly.

    Once it reads an attribute of `self`, the arguments a cell drives it with are
    no longer its whole input, and `CELLS` covers only what that instance holds.
    """
    offenders = []
    for path, cls in _catalog_entries():
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef) or node.name != "__call__":
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self"
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{inner.lineno} self.{inner.attr}")
    assert not offenders, (
        "__call__ must take every argument explicitly; these read instance state: "
        + ", ".join(offenders)
    )


def test_is_valid_accepts_everything_call_accepts():
    """`is_valid` accepts every argument `__call__` accepts.

    `validating()` forwards the call's full argument set to `is_valid`, so
    `is_valid` names each of them or absorbs the rest with `**kwargs`. An entry
    without its own `is_valid` inherits the base class's `(*args, **kwargs)`.
    `reference` is not checked: a cell calls it with the cell's own arguments.
    The first parameter of `__call__`, `reference` and `is_valid` is `self`, so
    the comparison does not drop a real argument.
    """
    problems = []
    for path, cls in _catalog_entries():
        funcs = {
            n.name: n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("__call__", "reference", "is_valid")
        }
        call = funcs.get("__call__")
        if call is None:
            continue
        for name, fn in funcs.items():
            params = [a.arg for a in fn.args.args]
            if not params or params[0] != "self":
                problems.append(
                    f"{path.relative_to(_ROOT)}: {cls.name}.{name} first param is not self"
                )

        is_valid = funcs.get("is_valid")
        if is_valid is None or is_valid.args.kwarg is not None:
            continue
        call_names = {a.arg for a in call.args.args + call.args.kwonlyargs} - {"self"}
        valid_names = {a.arg for a in is_valid.args.args + is_valid.args.kwonlyargs} - {"self"}
        missing = sorted(call_names - valid_names)
        if missing:
            problems.append(
                f"{path.relative_to(_ROOT)}: {cls.name}.is_valid does not accept "
                f"{missing} that __call__ accepts, and has no ** absorber"
            )
    assert not problems, "; ".join(problems)


# `CELLS = certified_cells(op, TARGET_CELLS)`; the test function's own name
# `test_certified_cells(` does not count as driving the cells.
_DRIVES_TARGET_CELLS = re.compile(r"^CELLS\s*=\s*certified_cells\(\w+,\s*TARGET_CELLS\)", re.M)


def test_every_entry_with_cells_has_a_test_that_drives_them():
    """A cell list no test runs reads as coverage that does not exist.

    An entry that declares `Cell(` needs a test under this tree that names both
    the entry and `.CELLS`, or assigns `CELLS = certified_cells(op, TARGET_CELLS)`,
    which drives the entry's cells and its targets'. This reads the source, so it
    shows that some test refers to the cells, not that they ran.
    """
    sources = [t.read_text() for t in Path(__file__).resolve().parent.rglob("*.py")]
    undriven = []
    for entry in _entry_files():
        if "Cell(" not in entry.read_text():
            continue
        if not any(
            entry.stem in src and (".CELLS" in src or _DRIVES_TARGET_CELLS.search(src))
            for src in sources
        ):
            undriven.append(str(entry.relative_to(_ROOT)))
    assert not undriven, (
        f"{len(undriven)} entry(ies) declare cells that no test drives: "
        f"{', '.join(undriven)}. Parametrize the entry's test off its CELLS."
    )


def _target_cells_tests() -> list[Path]:
    return sorted(
        p
        for area in _UNIT_ROOT.iterdir()
        if area.is_dir() and area.name != "models"
        for p in area.rglob("test_modeling_v2_*.py")
        if "TARGET_CELLS" in p.read_text()
    )


@pytest.mark.parametrize(
    "test", _target_cells_tests(), ids=lambda p: str(p.relative_to(_UNIT_ROOT))
)
def test_target_cells_feed_entries_the_target_calls(test):
    """An entry test's `TARGET_CELLS` hold the shapes each target calls the entry with.

    The test sits under `<area>/`, including component subdirectories such as
    `moe/ops/`, and mirrors `catalog/<area>/<entry>.py`. Each key is a target
    directory under `models/`, and that target's code imports the entry: a cell
    for an entry the target never calls certifies nothing it runs.
    """
    area = test.relative_to(_UNIT_ROOT).parts[0]
    entry = test.stem.removeprefix("test_modeling_v2_")
    assert (_ROOT / "catalog" / area / f"{entry}.py").is_file(), (
        f"{test.name}: declares TARGET_CELLS but mirrors no catalog/{area}/{entry}.py"
    )
    text = test.read_text()
    assert _DRIVES_TARGET_CELLS.search(text), (
        f"{test.name}: declares TARGET_CELLS but never passes them to certified_cells"
    )
    assert re.search(r'parametrize\(\s*"cell",\s*CELLS\b', text), (
        f"{test.name}: test_certified_cells does not parametrize over CELLS"
    )
    targets = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "TARGET_CELLS":
            targets = [ast.literal_eval(k) for k in node.value.keys]
    assert targets, f"{test.name}: TARGET_CELLS is not a dict literal keyed by target directory"
    for target in targets:
        target_dir = _ROOT / "models" / target
        assert target_dir.is_dir(), f"{test.name}: {target!r} is not a directory under models/"
        used = "".join(
            p.read_text() for p in target_dir.rglob("*.py") if "__pycache__" not in p.parts
        )
        assert re.search(rf"catalog\.{area}\.{entry} import", used), (
            f"{test.name}: declares cells for {target}, which never imports {area}/{entry}"
        )


# --- the multimodal contract of vision-language targets ----------------------------


def _routed_classes() -> list[tuple[str, Path, str]]:
    """`(architecture, modeling.py, class name)` for every class a family's `route` returns.

    Each `return` of `route` names a class that `route` imports from a target's `modeling`
    module, as every route does today. Any other return fails here, so no routed class is
    skipped unread."""
    found, unread = [], []
    for routing in sorted((_ROOT / "models").glob("*/routing.py")):
        for fn in ast.parse(routing.read_text()).body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            archs = [
                arg.value
                for dec in fn.decorator_list
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "routes"
                for arg in dec.args
                if isinstance(arg, ast.Constant)
            ]
            if not archs:
                continue
            imported = {
                alias.asname or alias.name: Path(routing.parent, *node.module.split("."))
                for node in ast.walk(fn)
                if isinstance(node, ast.ImportFrom)
                and node.level == 1
                and (node.module or "").endswith(".modeling")
                for alias in node.names
            }
            for node in ast.walk(fn):
                if not isinstance(node, ast.Return):
                    continue
                name = getattr(node.value, "id", None)
                if name in imported:
                    found += [(arch, imported[name].with_suffix(".py"), name) for arch in archs]
                else:
                    unread.append(f"{routing.relative_to(_ROOT)}:{node.lineno}")
    assert not unread, (
        f"{unread}: return the class that route imports from the target's modeling module "
        "directly, so these checks can read it"
    )
    return found


def _takes_media(architecture: str) -> bool:
    """Whether requests for `architecture` can carry media, as the router decides it."""
    from tensorrt_llm._torch.modeling_v2._router_index import _architecture_takes_media

    return _architecture_takes_media(architecture)


def _media_routes() -> list[tuple[Path, str]]:
    """The classes routed for an architecture whose requests can carry media."""
    return sorted({(path, name) for arch, path, name in _routed_classes() if _takes_media(arch)})


def _calls(fn: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == name
        for n in ast.walk(fn)
    )


def _raises(fn: ast.AST) -> bool:
    return any(isinstance(n, ast.Raise) for n in ast.walk(fn))


def _refuses_anything(fn: ast.AST) -> bool:
    """Whether the method calls or raises anything: a docstring or `pass` alone refuses nothing."""
    return _raises(fn) or any(isinstance(n, ast.Call) for n in ast.walk(fn))


#: A text-only target's admission error, as the shared helper's users word it.
_TEXT_ONLY_MESSAGE = "{} serves text only; send no images, videos or embeddings"
#: `multimodal_data` a text-only target refuses (the last key is one no helper knows), and the
#: ones it admits: none, or M-RoPE positions alone.
_REFUSED = (
    {"image": {"pixel_values": 0}},
    {"video": {"pixel_values_videos": 0}},
    {"multimodal_embedding": 0},
    {"new_modality": 0},
)
_ADMITTED = ({}, {"mrope_config": {"mrope_position_deltas": 0}})


def _admission_problems(where: str, name: str, hook: ast.FunctionDef) -> list[str]:
    """Run a text-only target's `validate_multimodal_request_data` on its own, as a method of a
    class named `name`, with the real `has_media_or_embeddings`: it refuses exactly what the
    helper flags, with the shared message. The target module itself is not imported."""
    from tensorrt_llm.inputs.multimodal import has_media_or_embeddings

    namespace = {"has_media_or_embeddings": has_media_or_embeddings}
    code = compile(
        ast.Module(body=[hook], type_ignores=[]),
        where,
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    model = type(name, (), {hook.name: namespace[hook.name]})()
    wrong = []
    for data in _ADMITTED:
        try:
            model.validate_multimodal_request_data(data)
        except Exception as error:
            wrong.append(f"refuses {sorted(data)}: {error!r}")
    for data in _REFUSED:
        try:
            model.validate_multimodal_request_data(data)
        except ValueError as error:
            if str(error) != _TEXT_ONLY_MESSAGE.format(name):
                wrong.append(f"refuses {sorted(data)} with {str(error)!r}")
        except Exception as error:
            wrong.append(f"refuses {sorted(data)} with {error!r}, not a ValueError")
        else:
            wrong.append(f"admits {sorted(data)}")
    if not wrong:
        return []
    return [
        f"{where}: validate_multimodal_request_data must refuse what has_media_or_embeddings "
        f"flags with {_TEXT_ONLY_MESSAGE.format(name)!r} and admit the rest, but {wrong}"
    ]


#: The shared base of vision-language models, and what a subclass declares: the base does
#: the rest. Its tower is `mm_encoder_class`, or an override of `build_mm_encoder`.
_BASE = "ModelingV2ForConditionalGeneration"
_SUBCLASS_ATTRIBUTES = ("language_model_class", "mm_encoder_weight_prefix")
_SUBCLASS_METHODS = ("encode_multimodal_inputs",)


def _class_def(path: Path, name: str) -> ast.ClassDef | None:
    """The top-level class `name` of the module at `path`, or None."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _base_names(cls: ast.ClassDef) -> set[str]:
    return {getattr(b, "id", getattr(b, "attr", None)) for b in cls.bases}


def _assignments(cls: ast.ClassDef) -> dict[str, ast.expr | None]:
    """The names the class body assigns, each with its value (None for a bare annotation)."""
    return {
        t.id: n.value
        for n in cls.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Name)
    }


#: Targets that encode media in their own forward instead of on the shared base, as
#: `(target directory under models/, class name)`: the qwen4_exp FP8 target encodes only the
#: frames of the current prefill chunk. Each refuses at admission, in a `validate_request` of
#: its own class, what its tower cannot take; the router accepts it for that alone. Add no
#: entry: a new vision-language target subclasses the base. An entry whose class moves onto
#: the base or serves text only fails until it is deleted, so the set only shrinks.
_OWN_ENCODER_TARGETS = frozenset(
    {
        (
            "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified",
            "ModelingV2Qwen38FlashNextFp8Sm103Unified",
        ),
    }
)


def _target_key(path: Path, name: str) -> tuple[str, str]:
    return path.parent.relative_to(_ROOT / "models").as_posix(), name


def _multimodal_contract_problems(path: Path, name: str) -> list[str]:
    """A class routed for media is either a `ModelingV2ForConditionalGeneration` subclass that
    declares `_SUBCLASS_ATTRIBUTES` and its tower and defines `_SUBCLASS_METHODS`, or serves
    text only and refuses media at admission as `_admission_problems` checks, or is listed in
    `_OWN_ENCODER_TARGETS`, still needs to be, and defines a `validate_request` that refuses
    something. A `forward` that raises on multimodal input uses `has_media_or_embeddings`."""
    where = f"{path.relative_to(_ROOT)}:{name}"
    cls = _class_def(path, name)
    if cls is None:
        return [f"{where}: no such class"]
    defs = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assigned = _assignments(cls)
    problems = []
    listed = _target_key(path, name) in _OWN_ENCODER_TARGETS
    if listed and (_BASE in _base_names(cls) or "validate_multimodal_request_data" in defs):
        # The class meets the contract another way now: drop the entry, check it as any other.
        problems.append(f"{where}: no longer encodes media itself; delete its entry")
        listed = False
    if listed:
        hook = defs.get("validate_request")
        if hook is None:
            problems.append(f"{where}: encodes media itself and does not define validate_request")
        elif not _refuses_anything(hook):
            problems.append(
                f"{where}: encodes media itself and its validate_request refuses nothing"
            )
    elif _BASE in _base_names(cls):
        problems += [
            f"{where}: the subclass does not declare {m}"
            for m in _SUBCLASS_ATTRIBUTES
            if assigned.get(m) is None
        ]
        problems += [
            f"{where}: the subclass does not define {m}" for m in _SUBCLASS_METHODS if m not in defs
        ]
        # A bare annotation or `None` builds no tower.
        tower = assigned.get("mm_encoder_class")
        if "build_mm_encoder" not in defs and (
            tower is None or (isinstance(tower, ast.Constant) and tower.value is None)
        ):
            problems.append(
                f"{where}: the subclass neither sets mm_encoder_class to a tower class nor "
                "defines build_mm_encoder"
            )
    else:
        hook = defs.get("validate_multimodal_request_data")
        if hook is None:
            problems.append(
                f"{where}: neither a {_BASE} subclass nor a text-only target with "
                "validate_multimodal_request_data"
            )
        else:
            problems += _admission_problems(where, name, hook)
    forward = defs.get("forward")
    if (
        forward is not None
        and _raises(forward)
        and "multimodal_params" in ast.unparse(forward)
        and not _calls(forward, "has_media_or_embeddings")
    ):
        problems.append(
            f"{where}: forward refuses multimodal input without has_media_or_embeddings"
        )
    return problems


def test_vision_language_targets_keep_the_multimodal_contract():
    """Every class routed for an architecture whose requests can carry media either runs media
    through the multimodal runtime or refuses it at admission, where only that request fails.
    Each text-only target's admission hook runs on media, embeddings and M-RoPE positions. The
    router checks the contract's shape at startup; this finds a broken target without a GPU."""
    routed = _media_routes()
    families = {path.relative_to(_ROOT / "models").parts[0] for path, _ in routed}
    assert {"glm5_next", "kimi_k3_vl", "minimax_m3", "qwen4_exp"} <= families, families
    text_only = {
        path.relative_to(_ROOT / "models").parts[0]
        for path, name in routed
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "validate_multimodal_request_data"
            for node in getattr(_class_def(path, name), "body", ())
        )
    }
    assert {"glm5_next", "kimi_k3_vl", "minimax_m3"} <= text_only, text_only
    stale = _OWN_ENCODER_TARGETS - {_target_key(path, name) for path, name in routed}
    assert not stale, f"{sorted(stale)}: no longer routed for media; delete the entry"
    problems = [p for path, name in routed for p in _multimodal_contract_problems(path, name)]
    assert not problems, problems


_FIXTURE_ROUTING = """
from ..._router_index import routes


@routes("{arch}")
def route(model_config, llm_args, compute_capability):
    from .target.modeling import {cls}

    return {cls}
"""

_TEXT_ONLY_OK = """
class Good(Base):
    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        if has_media_or_embeddings(mm_data):
            raise ValueError(
                f"{type(self).__name__} serves text only; send no images, videos or embeddings"
            )

    def forward(self, *args, **kwargs):
        params = kwargs.pop("multimodal_params", None) or []
        if any(has_media_or_embeddings(p.multimodal_data) for p in params):
            raise ValueError("text only")
"""

_SUBCLASS_OK = """
class Good(ModelingV2ForConditionalGeneration):
    language_model_class = Llm
    mm_encoder_class = Tower
    mm_encoder_weight_prefix = "model.visual."

    def encode_multimodal_inputs(self, multimodal_params):
        return None
"""
_BUILT_TOWER = """
    def build_mm_encoder(self, pretrained_config):
        return Tower(pretrained_config.audio_config)
"""


@pytest.mark.parametrize(
    "source, problem",
    [
        (_TEXT_ONLY_OK, None),
        (_SUBCLASS_OK, None),
        ("class Good(Base):\n    pass\n", "neither a ModelingV2ForConditionalGeneration"),
        (
            _TEXT_ONLY_OK.replace("if has_media_or_embeddings(mm_data)", "if mm_data"),
            "but [\"refuses ['mrope_config']",
        ),
        (
            _TEXT_ONLY_OK.replace("if has_media_or_embeddings", "if not has_media_or_embeddings"),
            'but ["refuses []',
        ),
        (
            _TEXT_ONLY_OK.replace("raise ValueError(", "ValueError("),
            "but [\"admits ['image']\"",
        ),
        (
            _TEXT_ONLY_OK.replace("serves text only", "is text only"),
            "but [\"refuses ['image'] with 'Good is text only",
        ),
        (
            _TEXT_ONLY_OK.replace("raise ValueError(", "raise RuntimeError("),
            "not a ValueError",
        ),
        (
            _TEXT_ONLY_OK.replace(
                "any(has_media_or_embeddings(p.multimodal_data) for p in params)", "params"
            ),
            "forward refuses multimodal input",
        ),
        (
            _SUBCLASS_OK.replace("    language_model_class = Llm\n", ""),
            "does not declare language_model_class",
        ),
        (
            _SUBCLASS_OK.replace("def encode_multimodal_inputs", "def other"),
            "does not define encode_multimodal_inputs",
        ),
        (_SUBCLASS_OK.replace("    mm_encoder_class = Tower\n", "") + _BUILT_TOWER, None),
        (
            _SUBCLASS_OK.replace("    mm_encoder_class = Tower\n", ""),
            "neither sets mm_encoder_class to a tower class nor defines build_mm_encoder",
        ),
        (
            _SUBCLASS_OK.replace("mm_encoder_class = Tower", "mm_encoder_class = None"),
            "neither sets mm_encoder_class to a tower class nor defines build_mm_encoder",
        ),
        (
            _SUBCLASS_OK.replace("mm_encoder_class = Tower", "mm_encoder_class: type"),
            "neither sets mm_encoder_class to a tower class nor defines build_mm_encoder",
        ),
    ],
    ids=[
        "text_only_ok",
        "subclass_ok",
        "no_hook",
        "truthiness_hook",
        "inverted_hook",
        "hook_never_raises",
        "other_message",
        "other_error_type",
        "truthiness_forward",
        "subclass_without_language_model_class",
        "subclass_without_encoder_hook",
        "subclass_building_its_tower",
        "subclass_without_tower",
        "subclass_with_a_none_tower",
        "subclass_with_a_bare_tower_annotation",
    ],
)
def test_the_multimodal_contract_check_reports_a_broken_target(
    tmp_path, monkeypatch, source, problem
):
    """The architecture's input processor picks the classes to check, not its name: the
    media architecture is named like a text one and the text one like a vision-language
    one."""
    for arch, family in (("FooForCausalLM", "foo"), ("BarForConditionalGeneration", "bar")):
        target = tmp_path / "models" / family / "target"
        target.mkdir(parents=True)
        (target.parent / "routing.py").write_text(_FIXTURE_ROUTING.format(arch=arch, cls="Good"))
        # The text architecture's class is never checked.
        (target / "modeling.py").write_text(
            source if family == "foo" else "class Good:\n    pass\n"
        )
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module, "_takes_media", lambda arch: arch == "FooForCausalLM")
    routed = _media_routes()
    assert routed == [(tmp_path / "models/foo/target/modeling.py", "Good")]
    problems = [p for path, name in routed for p in _multimodal_contract_problems(path, name)]
    if problem is None:
        assert not problems, problems
    else:
        assert len(problems) == 1 and problem in problems[0], problems


_OWN_ENCODER_OK = """
class Good(Base):
    def validate_request(self, request):
        \"\"\"Refuse a media request the tower would assert on.\"\"\"
        vision.validate_request(self.vision_tower, request)
"""


@pytest.mark.parametrize(
    "source, problem",
    [
        (_OWN_ENCODER_OK, None),
        ("class Good(Base):\n    pass\n", "does not define validate_request"),
        (
            _OWN_ENCODER_OK.replace("vision.validate_request(self.vision_tower, request)", "pass"),
            "its validate_request refuses nothing",
        ),
        (_SUBCLASS_OK, "no longer encodes media itself; delete its entry"),
        (_TEXT_ONLY_OK, "no longer encodes media itself; delete its entry"),
    ],
    ids=[
        "own_encoder_ok",
        "own_encoder_without_validate_request",
        "own_encoder_refusing_nothing",
        "moved_onto_the_base",
        "now_text_only",
    ],
)
def test_a_listed_own_encoder_target_must_refuse_at_admission(
    tmp_path, monkeypatch, source, problem
):
    """A listed target needs a `validate_request` of its own that refuses something, and the
    entry must still be needed: a class that moved onto the base or serves text only is checked
    as such, and its entry is reported for deletion."""
    target = tmp_path / "models" / "foo" / "target"
    target.mkdir(parents=True)
    (target / "modeling.py").write_text(source)
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_ROOT", tmp_path)
    monkeypatch.setattr(module, "_OWN_ENCODER_TARGETS", frozenset({("foo/target", "Good")}))
    problems = _multimodal_contract_problems(target / "modeling.py", "Good")
    if problem is None:
        assert not problems, problems
    else:
        assert len(problems) == 1 and problem in problems[0], problems


def test_a_route_return_the_checks_cannot_read_fails(tmp_path, monkeypatch):
    """A route that returns its class through a local name fails instead of being skipped."""
    target = tmp_path / "models" / "foo" / "target"
    target.mkdir(parents=True)
    routing = _FIXTURE_ROUTING.format(arch="FooForConditionalGeneration", cls="Good")
    (target.parent / "routing.py").write_text(
        routing.replace("    return Good", "    target_class = Good\n    return target_class")
    )
    (target / "modeling.py").write_text(_SUBCLASS_OK)
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    with pytest.raises(AssertionError, match=r"models/foo/routing\.py:10\b"):
        _routed_classes()


# --- the naming rule of vision-language models ----------------------------------------

#: A class with one of these bases is a multimodal model.
_MULTIMODAL_BASES = {_BASE, "MultimodalModelMixin"}


def _routed_multimodal_models() -> list[tuple[Path, ast.ClassDef]]:
    """Every routed class that is a multimodal model, whatever its architecture."""
    found = []
    for path, name in sorted({(path, name) for _, path, name in _routed_classes()}):
        cls = _class_def(path, name)
        if cls is not None and _base_names(cls) & _MULTIMODAL_BASES:
            found.append((path, cls))
    return found


def _naming_rule_problems() -> list[str]:
    """A routed multimodal model subclasses `ModelingV2ForConditionalGeneration` and its name
    ends with `ForConditionalGeneration`; no class of a target's `modeling.py` is named
    `...VLM`."""
    problems = []
    for path, cls in _routed_multimodal_models():
        where = f"{path.relative_to(_ROOT)}:{cls.name}"
        if _BASE not in _base_names(cls):
            problems.append(f"{where}: a multimodal model that does not subclass {_BASE}")
        if not cls.name.endswith("ForConditionalGeneration"):
            problems.append(f"{where}: a multimodal model not named ...ForConditionalGeneration")
    for path in sorted((_ROOT / "models").glob("*/*/modeling.py")):
        problems += [
            f"{path.relative_to(_ROOT)}:{node.name}: a target class named ...VLM"
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ClassDef) and node.name.upper().endswith("VLM")
        ]
    return problems


def test_vision_language_models_follow_the_naming_rule():
    """Every routed multimodal model subclasses the shared base and its name ends with
    `ForConditionalGeneration`, and no target class is named `...VLM`. The rest of the name,
    `ModelingV2<Target>`, is left to review. A language model target is not a multimodal
    model, so its `ModelingV2<Target>` name is not checked here."""
    families = {
        path.relative_to(_ROOT / "models").parts[0] for path, _ in _routed_multimodal_models()
    }
    assert "qwen4_exp" in families, families
    problems = _naming_rule_problems()
    assert not problems, problems


_NAMED_OK = """
class ModelingV2Foo(Base):
    pass


class ModelingV2FooForConditionalGeneration(ModelingV2ForConditionalGeneration):
    language_model_class = ModelingV2Foo
"""


@pytest.mark.parametrize(
    "routed, source, problem",
    [
        ("ModelingV2FooForConditionalGeneration", _NAMED_OK, None),
        # A language model target routed on its own is not a multimodal model.
        ("ModelingV2Foo", _NAMED_OK, None),
        (
            "ModelingV2FooForConditionalGeneration",
            _NAMED_OK.replace(f"({_BASE})", "(MultimodalModelMixin, PreTrainedModel)"),
            f"does not subclass {_BASE}",
        ),
        (
            "ModelingV2FooModel",
            _NAMED_OK.replace("ModelingV2FooForConditionalGeneration", "ModelingV2FooModel"),
            "not named ...ForConditionalGeneration",
        ),
        (
            "ModelingV2FooForConditionalGeneration",
            _NAMED_OK.replace("ModelingV2Foo(", "ModelingV2FooVLM(").replace(
                "= ModelingV2Foo\n", "= ModelingV2FooVLM\n"
            ),
            "ModelingV2FooVLM: a target class named ...VLM",
        ),
    ],
    ids=[
        "model_ok",
        "language_model_ok",
        "mixin_not_the_base",
        "model_suffix",
        "vlm_suffix",
    ],
)
def test_the_naming_rule_check_reports_each_broken_part(
    tmp_path, monkeypatch, routed, source, problem
):
    """Each stand-in breaks one part of the rule and gets exactly that problem."""
    target = tmp_path / "models" / "foo" / "target"
    target.mkdir(parents=True)
    (target.parent / "routing.py").write_text(
        _FIXTURE_ROUTING.format(arch="FooForConditionalGeneration", cls=routed)
    )
    (target / "modeling.py").write_text(source)
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    problems = _naming_rule_problems()
    if problem is None:
        assert not problems, problems
    else:
        assert len(problems) == 1 and problem in problems[0], problems
