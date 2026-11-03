# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Completeness, laziness and layering of the ``tensorrt_llm.serve.openai_protocol`` facade.

``tensorrt_llm/serve/openai_protocol/`` holds one submodule per API behind a
lazy facade (``__init__.py``) that re-exports, on first access, every name the
former single-file module had. These tests keep the facade's name table
complete and frozen, the facade lazy, and the submodule import graph acyclic.
The VisualGen names are checked by a sibling test under the VisualGen test
directory, so that CI selection for VisualGen-only pull requests runs it.
"""

import ast
import importlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_common.serve.shims import ShimRow, check_identity

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

_PACKAGE = "tensorrt_llm.serve.openai_protocol"
_PACKAGE_DIR = "tensorrt_llm/serve/openai_protocol"
_FACADE_PATH = f"{_PACKAGE_DIR}/__init__.py"

# Every name the former single-file module defined, with the module that defines it now, plus
# the one name it imported that code in the repository imports through it. Frozen: a new model
# is imported from its submodule (tensorrt_llm/serve/AGENTS.md), never added here.
_FORMER_NAMES: dict[str, str] = {
    # common
    "ConversationParams": "tensorrt_llm.serve.openai_protocol.common",
    "DisaggregatedParams": "tensorrt_llm.serve.openai_protocol.common",
    "EarlyStopping": "tensorrt_llm.serve.openai_protocol.common",
    "ErrorResponse": "tensorrt_llm.serve.openai_protocol.common",
    "OpenAIBaseModel": "tensorrt_llm.serve.openai_protocol.common",
    "PromptTokensDetails": "tensorrt_llm.serve.openai_protocol.common",
    "ResponseFormat": "tensorrt_llm.serve.openai_protocol.common",
    "SpeculativeDecodingStats": "tensorrt_llm.serve.openai_protocol.common",
    "StreamOptions": "tensorrt_llm.serve.openai_protocol.common",
    "UsageInfo": "tensorrt_llm.serve.openai_protocol.common",
    "_OmitsAbsentSpecDecodeStats": "tensorrt_llm.serve.openai_protocol.common",
    "ensure_request_chat_template_allowed": "tensorrt_llm.serve.openai_protocol.common",
    # sampling
    "_LOGIT_BIAS_MAX": "tensorrt_llm.serve.openai_protocol.sampling",
    "_LOGIT_BIAS_MIN": "tensorrt_llm.serve.openai_protocol.sampling",
    "_early_stopping_to_int": "tensorrt_llm.serve.openai_protocol.sampling",
    "_logit_bias_to_embedding_bias": "tensorrt_llm.serve.openai_protocol.sampling",
    "_record_sampling_params_request_fields": "tensorrt_llm.serve.openai_protocol.sampling",
    "_response_format_text_config_to_guided_decoding_params": (
        "tensorrt_llm.serve.openai_protocol.sampling"
    ),
    "_response_format_to_guided_decoding_params": "tensorrt_llm.serve.openai_protocol.sampling",
    # tools
    "ChatCompletionNamedFunction": "tensorrt_llm.serve.openai_protocol.tools",
    "ChatCompletionNamedToolChoiceParam": "tensorrt_llm.serve.openai_protocol.tools",
    "ChatCompletionToolsParam": "tensorrt_llm.serve.openai_protocol.tools",
    "DeltaFunctionCall": "tensorrt_llm.serve.openai_protocol.tools",
    "DeltaToolCall": "tensorrt_llm.serve.openai_protocol.tools",
    "FunctionCall": "tensorrt_llm.serve.openai_protocol.tools",
    "FunctionDefinition": "tensorrt_llm.serve.openai_protocol.tools",
    "NamespacedFunctionToolParam": "tensorrt_llm.serve.openai_protocol.tools",
    "TOOL_PARAM_MAX_ENUM_VALUES": "tensorrt_llm.serve.openai_protocol.tools",
    "ToolCall": "tensorrt_llm.serve.openai_protocol.tools",
    "_SCHEMA_INSTANCE_KEYWORDS": "tensorrt_llm.serve.openai_protocol.tools",
    "_SCHEMA_MAP_KEYWORDS": "tensorrt_llm.serve.openai_protocol.tools",
    "_count_schema_enum_values": "tensorrt_llm.serve.openai_protocol.tools",
    # models
    "ModelCard": "tensorrt_llm.serve.openai_protocol.models",
    "ModelList": "tensorrt_llm.serve.openai_protocol.models",
    # tokenize
    "TokenizeRequest": "tensorrt_llm.serve.openai_protocol.tokenize",
    "TokenizeResponse": "tensorrt_llm.serve.openai_protocol.tokenize",
    # embeddings
    "EmbeddingRequest": "tensorrt_llm.serve.openai_protocol.embeddings",
    "EmbeddingResponse": "tensorrt_llm.serve.openai_protocol.embeddings",
    "EmbeddingResponseData": "tensorrt_llm.serve.openai_protocol.embeddings",
    "EmbeddingUsageInfo": "tensorrt_llm.serve.openai_protocol.embeddings",
    # completions
    "CompletionLogProbs": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionRequest": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionResponse": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionResponseChoice": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionResponseStreamChoice": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionStreamResponse": "tensorrt_llm.serve.openai_protocol.completions",
    # chat
    "ChatCompletionContentPartParam": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionLogProb": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionLogProbs": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionLogProbsContent": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionMessageParam": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionRequest": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionResponse": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionResponseChoice": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionResponseStreamChoice": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionStreamResponse": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionThinkingParam": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatMessage": "tensorrt_llm.serve.openai_protocol.chat",
    "CustomChatCompletionContentPartParam": "tensorrt_llm.serve.openai_protocol.chat",
    "CustomChatCompletionMessageParam": "tensorrt_llm.serve.openai_protocol.chat",
    "DeltaMessage": "tensorrt_llm.serve.openai_protocol.chat",
    "DynamicToolsSystemMessageParam": "tensorrt_llm.serve.openai_protocol.chat",
    "ReasoningAssistantMessage": "tensorrt_llm.serve.openai_protocol.chat",
    # kv_cache
    "KVCacheTruncateRequest": "tensorrt_llm.serve.openai_protocol.kv_cache",
    "KVCacheTruncateTokensRequest": "tensorrt_llm.serve.openai_protocol.kv_cache",
    # responses
    "InputTokensDetails": "tensorrt_llm.serve.openai_protocol.responses",
    "OutputTokensDetails": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponseInputOutputItem": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponseUsage": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponsesRequest": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponsesResponse": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponsesStreamResponse": "tensorrt_llm.serve.openai_protocol.responses",
    "StreamingResponsesResponse": "tensorrt_llm.serve.openai_protocol.responses",
    "_ID_STRIPPED_ROLES": "tensorrt_llm.serve.openai_protocol.responses",
    "_RESPONSES_HANDOFF_FIELDS": "tensorrt_llm.serve.openai_protocol.responses",
    "_drop_explicit_nulls": "tensorrt_llm.serve.openai_protocol.responses",
    "_materialize_validator_iterators": "tensorrt_llm.serve.openai_protocol.responses",
    # control
    "MemoryUpdateRequest": "tensorrt_llm.serve.openai_protocol.control",
    "ProfileActivity": "tensorrt_llm.serve.openai_protocol.control",
    "RuntimeMemoryUpdateRequest": "tensorrt_llm.serve.openai_protocol.control",
    "StartProfileRequest": "tensorrt_llm.serve.openai_protocol.control",
    "UpdateWeightsRequest": "tensorrt_llm.serve.openai_protocol.control",
    # disagg
    "UCompletionRequest": "tensorrt_llm.serve.openai_protocol.disagg",
    "UCompletionResponse": "tensorrt_llm.serve.openai_protocol.disagg",
    "_deserialize_first_gen_log_probs": "tensorrt_llm.serve.openai_protocol.disagg",
    "_deserialize_first_gen_logits": "tensorrt_llm.serve.openai_protocol.disagg",
    "_serialize_first_gen_log_probs": "tensorrt_llm.serve.openai_protocol.disagg",
    "_serialize_first_gen_logits": "tensorrt_llm.serve.openai_protocol.disagg",
    "decode_opaque_state": "tensorrt_llm.serve.openai_protocol.disagg",
    "encode_opaque_state": "tensorrt_llm.serve.openai_protocol.disagg",
    "to_disaggregated_params": "tensorrt_llm.serve.openai_protocol.disagg",
    "to_llm_conversation_params": "tensorrt_llm.serve.openai_protocol.disagg",
    "to_llm_disaggregated_params": "tensorrt_llm.serve.openai_protocol.disagg",
    # imported by the former module and imported from it by code in the repository
    "DisaggScheduleStyle": "tensorrt_llm.llmapi",
}

# The VisualGen models the former module also defined. Their identity rows live with the other
# VisualGen serve tests; here they are names only, so that this file names no VisualGen module.
_VISUAL_GEN_NAMES = frozenset(
    {
        "ImageGenerationRequest",
        "ImageEditRequest",
        "ImageObject",
        "ImageGenerationResponse",
        "MediaReferenceItem",
        "VideoGenerationRequest",
        "VideoJob",
        "VideoJobList",
    }
)

# Private helpers that tests import through the package, with one importing test file each.
_TEST_IMPORTED_PRIVATE_NAMES = {
    "_serialize_first_gen_log_probs": "tests/unittest/disaggregated/test_openai_disagg_service.py",
    "_deserialize_first_gen_log_probs": (
        "tests/unittest/disaggregated/test_openai_disagg_service.py"
    ),
    "_serialize_first_gen_logits": "tests/unittest/disaggregated/test_openai_disagg_service.py",
    "_deserialize_first_gen_logits": "tests/unittest/disaggregated/test_openai_disagg_service.py",
    "_logit_bias_to_embedding_bias": "tests/unittest/llmapi/apps/test_harmony_parsing.py",
    "_response_format_to_guided_decoding_params": (
        "tests/unittest/llmapi/apps/test_serving_extensions.py"
    ),
}

# Every name the facade re-exports: the former module's definitions and DisaggScheduleStyle.
_EXPECTED = frozenset(_FORMER_NAMES) | _VISUAL_GEN_NAMES

# The facade is the package's own interface, not a compatibility shim: it never warns and the
# old path is not banned, so these rows check identity only.
_FACADE_ROWS = tuple(
    ShimRow(_PACKAGE, name, module, retire_in="not retired", ban_old_path=False, warns=False)
    for name, module in _FORMER_NAMES.items()
)

# Which submodules each submodule may import ("a: {b}" means a imports from b). Acyclic.
_ALLOWED_SIBLING_IMPORTS = {
    "common": frozenset(),
    "sampling": frozenset({"common"}),
    "tools": frozenset({"common"}),
    "models": frozenset({"common"}),
    "tokenize": frozenset({"common"}),
    "embeddings": frozenset({"common"}),
    "control": frozenset({"common"}),
    "completions": frozenset({"common", "sampling"}),
    "responses": frozenset({"common", "sampling"}),
    "chat": frozenset({"common", "sampling", "tools"}),
    "kv_cache": frozenset({"common", "tools", "chat"}),
    "disagg": frozenset({"common", "completions", "chat"}),
}


@pytest.fixture(scope="module")
def facade():
    return importlib.import_module(_PACKAGE)


@pytest.fixture(scope="module")
def package_dir(facade) -> Path:
    return Path(facade.__file__).resolve().parent


def _resolve(node: ast.ImportFrom, package: str) -> str:
    """The absolute module an ``ImportFrom`` names, relative imports resolved against ``package``."""
    if node.level == 0:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)]
    if node.module:
        base.append(node.module)
    return ".".join(base)


def _run_in_fresh_interpreter(code: str) -> dict:
    """Run ``code`` in a new interpreter and return the JSON it prints last."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"the probe interpreter exited with {result.returncode}; stdout tail: "
        f"{result.stdout[-1000:]!r}; stderr tail: {result.stderr[-2000:]!r}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _with_facade_fix(problem: str) -> str:
    """Prefix a shim-identity message with what to change in the facade."""
    match = re.search(rf"`{re.escape(_PACKAGE)}\.(\w+)`", problem)
    name = match.group(1) if match else "<name>"
    origin = _FORMER_NAMES.get(name, "<module>")
    if "no longer exists" in problem:
        fix = (
            f"`{name}` was a top-level name of tensorrt_llm/serve/openai_protocol.py and must "
            f"stay importable from {_PACKAGE}. Define it in {origin} and add '{name}': "
            f"'{origin}' to _NAME_TO_MODULE in {_FACADE_PATH}."
        )
    else:
        fix = (
            f"{_PACKAGE}.{name} is not the object defined in {origin}; the facade must "
            "re-export it, not copy or redefine it."
        )
    return f"{fix} ({problem})"


def test_every_former_name_is_reachable_and_identical():
    problems = check_identity(_FACADE_ROWS)
    assert problems == [], "\n".join(_with_facade_fix(problem) for problem in problems)


def test_facade_table_is_frozen(facade):
    table = set(facade._NAME_TO_MODULE)
    problems = [
        f"`{name}` is in the facade table but was not a name of the former module. New models "
        f"are imported from their submodule (tensorrt_llm/serve/AGENTS.md); remove it from "
        f"_NAME_TO_MODULE."
        for name in sorted(table - _EXPECTED)
    ]
    problems += [
        f"`{name}` was a name of the former module but is not in _NAME_TO_MODULE. Add "
        f"'{name}': '<module>' to _NAME_TO_MODULE; underscore names belong in the table too."
        for name in sorted(_EXPECTED - table)
    ]
    problems += [
        f"`{name}` is both a submodule of {_PACKAGE} and a row of _NAME_TO_MODULE; importing "
        "the submodule would bind the package attribute to the module and shadow the "
        "re-exported name."
        for name in sorted(table & set(_ALLOWED_SIBLING_IMPORTS))
    ]
    assert problems == [], "\n".join(problems)


def test_all_dir_and_type_checking_block(facade):
    public = {name for name in facade._NAME_TO_MODULE if not name.startswith("_")}
    problems = []
    if set(facade.__all__) != public:
        problems.append(
            f"__all__ differs from the public names of _NAME_TO_MODULE: "
            f"{sorted(set(facade.__all__) ^ public)}"
        )
    missing_from_dir = sorted(_EXPECTED - set(dir(facade)))
    if missing_from_dir:
        problems.append(f"dir({_PACKAGE}) lacks {missing_from_dir}")

    tree = ast.parse(Path(facade.__file__).read_text())
    type_checking_imports = set()
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom):
                    module = _resolve(statement, _PACKAGE)
                    type_checking_imports |= {(module, alias.name) for alias in statement.names}
    expected_imports = {
        (facade._NAME_TO_MODULE[name], name)
        for name in facade.__all__
        if name not in _VISUAL_GEN_NAMES
    }
    if type_checking_imports != expected_imports:
        problems.append(
            f"the TYPE_CHECKING imports and __all__ of {_FACADE_PATH} disagree with "
            f"_NAME_TO_MODULE: {sorted(type_checking_imports ^ expected_imports)}. The block "
            "imports every name of __all__ except the VisualGen names, which this file names "
            "only as strings."
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [_resolve(node, _PACKAGE)]
        else:
            continue
        for module in modules:
            if module not in ("importlib", "typing") and not module.startswith(_PACKAGE + "."):
                problems.append(
                    f"{_FACADE_PATH} imports {module} at line {node.lineno}; the facade imports "
                    "only importlib, typing and its own submodules (the latter under "
                    "TYPE_CHECKING)."
                )
    assert problems == [], "\n".join(problems)


def test_unknown_name_raises_attribute_error(facade):
    assert getattr(facade, "NoSuchModel", None) is None
    with pytest.raises(ImportError):
        exec(f"from {_PACKAGE} import NoSuchModel", {})


def test_submodule_import_graph(package_dir):
    present = {path.stem for path in package_dir.glob("*.py")} - {"__init__"}
    problems = [
        f"{_PACKAGE_DIR}/{name}.py is not in the allowed import graph of this test; a new API's "
        "submodule is added to _ALLOWED_SIBLING_IMPORTS here (importing common.py only, unless "
        "it needs more) and the graph must stay acyclic."
        for name in sorted(present - set(_ALLOWED_SIBLING_IMPORTS))
    ]
    problems += [
        f"{_PACKAGE_DIR}/{name}.py is missing; the allowed import graph of this test lists it."
        for name in sorted(set(_ALLOWED_SIBLING_IMPORTS) - present)
    ]
    for name in sorted(present & set(_ALLOWED_SIBLING_IMPORTS)):
        tree = ast.parse((package_dir / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [_resolve(node, _PACKAGE)]
            else:
                continue
            for module in modules:
                if module == _PACKAGE:
                    problems.append(
                        f"{_PACKAGE_DIR}/{name}.py imports the package {_PACKAGE} itself at "
                        f"line {node.lineno}; import the defining submodule, so that the facade "
                        "never sits in an import chain between submodules."
                    )
                elif module.startswith(_PACKAGE + "."):
                    other = module[len(_PACKAGE) + 1 :].split(".")[0]
                    if other not in _ALLOWED_SIBLING_IMPORTS[name]:
                        problems.append(
                            f"{_PACKAGE_DIR}/{name}.py imports {other}; allowed sibling imports "
                            f"are {sorted(_ALLOWED_SIBLING_IMPORTS[name])}. Move the shared "
                            "definition to common.py or sampling.py; the graph must stay acyclic."
                        )
    assert problems == [], "\n".join(problems)


def test_importing_the_package_loads_no_submodule():
    out = _run_in_fresh_interpreter(
        f"""
        import json, sys
        import tensorrt_llm.serve.openai_protocol
        prefix = {_PACKAGE!r} + "."
        print(json.dumps(sorted(m for m in sys.modules if m.startswith(prefix))))
        """
    )
    assert out == [], (
        f"importing {_PACKAGE} loaded {out}; the facade imports a submodule only inside "
        "__getattr__."
    )


def test_tools_does_not_load_chat_or_responses():
    out = _run_in_fresh_interpreter(
        f"""
        import json, sys
        import tensorrt_llm.serve.openai_protocol.tools
        prefix = {_PACKAGE!r} + "."
        print(json.dumps(sorted(m[len(prefix):] for m in sys.modules if m.startswith(prefix))))
        """
    )
    heavy = sorted(set(out) & {"chat", "responses", "completions"})
    assert heavy == [], (
        f"importing {_PACKAGE}.tools loaded {heavy}; tools.py may import only common.py."
    )


def test_chat_models_do_not_load_extensions(package_dir):
    out = _run_in_fresh_interpreter(
        f"""
        import json, sys
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
        prefix = {_PACKAGE!r} + "."
        print(json.dumps({{
            "extensions": [m for m in sys.modules if m.startswith("tensorrt_llm.serve.extensions")],
            "loaded": sorted(m[len(prefix):] for m in sys.modules if m.startswith(prefix)),
        }}))
        """
    )
    culprits: list = []
    if out["extensions"]:
        importers = sorted(
            name
            for name in out["loaded"]
            if "tensorrt_llm.serve.extensions" in (package_dir / f"{name}.py").read_text()
        )
        culprits = importers or out["loaded"]
    assert out["extensions"] == [], (
        f"importing ChatCompletionRequest through {_PACKAGE} loaded {out['extensions']}; "
        f"{[f'{_PACKAGE_DIR}/{name}.py' for name in culprits]} must not import "
        "tensorrt_llm.serve.extensions: the built-in extensions import the protocol models, and "
        "serving_extensions imports them only on the first lookup to keep that cycle open."
    )


def test_private_helpers_that_tests_import():
    problems = []
    for name, test_file in _TEST_IMPORTED_PRIVATE_NAMES.items():
        namespace: dict = {}
        try:
            exec(f"from {_PACKAGE} import {name}", namespace)
        except ImportError as exc:
            problems.append(
                f"{test_file} imports {name} from {_PACKAGE}; keep it in _NAME_TO_MODULE (the "
                f"table keeps underscore names too). Import failed with: {exc}"
            )
            continue
        expected = getattr(importlib.import_module(_FORMER_NAMES[name]), name)
        if namespace[name] is not expected:
            problems.append(
                f"{test_file} imports {name} from {_PACKAGE}, which is not the object defined "
                f"in {_FORMER_NAMES[name]}."
            )
    assert problems == [], "\n".join(problems)
