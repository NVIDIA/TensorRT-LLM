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
"""``tensorrt_llm/serve/envs.py``: the one place serving code reads environment variables.

Three groups of checks:

- the lint: no module in ``tensorrt_llm/serve/`` (except ``scripts/``) or the
  trtllm-serve CLI reads ``os.environ`` directly, and the lint itself sees every
  read form and ignores writes (so an empty scan cannot pass vacuously);
- the declarations: every reader reproduces the parsing its call site used
  before the declaration moved into ``envs.py`` (a table written out here, per
  environment state), values are read at access time, secrets are marked, the
  launcher enums and the ``*_ENV`` name constants agree with the declarations;
- the module contract: undeclared names raise ``AttributeError``, required
  variables raise ``KeyError``, ``consume()`` reads once.
"""

import ast
import os
import textwrap
from pathlib import Path

import pytest
from test_common.serve.env_lint import ENVS_IMPORTABLE_NAMES, find_direct_env_reads, scope_files

from tensorrt_llm.serve import envs

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
ENVS_PATH = REPO_ROOT / "tensorrt_llm" / "serve" / "envs.py"
AGENTS_MD = REPO_ROOT / "tensorrt_llm" / "serve" / "AGENTS.md"

# ---------------------------------------------------------------------------
# The lint over the serve tree and the CLI.
# ---------------------------------------------------------------------------


def test_no_direct_environment_reads():
    files = scope_files(REPO_ROOT, visual_gen=False)
    assert files, "the env-lint scanned no file; scope_files() lost the serve tree"
    findings = find_direct_env_reads(files, root=REPO_ROOT)
    assert not findings, "\n".join(finding.message() for finding in findings)


def test_scope_files_cover_the_serve_tree_and_the_cli():
    relative = {
        path.relative_to(REPO_ROOT).as_posix() for path in scope_files(REPO_ROOT, visual_gen=False)
    }
    assert "tensorrt_llm/commands/serve.py" in relative
    assert "tensorrt_llm/serve/openai_server.py" in relative
    assert any(name.startswith("tensorrt_llm/commands/serve_cli/") for name in relative)
    assert "tensorrt_llm/serve/envs.py" not in relative, (
        "the module reading os.environ for everyone is not linted"
    )
    assert not any(name.startswith("tensorrt_llm/serve/scripts/") for name in relative), (
        "serve/scripts/ is a published client package and stays outside the lint"
    )
    assert not any(name.startswith("tensorrt_llm/serve/api/visual_gen/") for name in relative), (
        "the VisualGen package is checked by tests/unittest/serve/api/visual_gen/test_envs_lint_visual_gen.py"
    )


def _findings(tmp_path, source):
    module = tmp_path / "mod.py"
    module.write_text(textwrap.dedent(source))
    return find_direct_env_reads([module], root=tmp_path)


@pytest.mark.parametrize(
    ("source", "line", "text"),
    [
        ('import os\nx = os.getenv("A")\n', 2, 'os.getenv("A")'),
        ('import os\nx = os.getenv("A", "0") == "1"\n', 2, 'os.getenv("A", "0")'),
        ('from os import getenv\nx = getenv("A")\n', 2, 'getenv("A")'),
        ('from os import getenv as read_env\nx = read_env("A")\n', 2, 'read_env("A")'),
        ('import os\nx = os.environ.get("A")\n', 2, 'os.environ.get("A")'),
        ('import os\nx = os.environ.get("A", "1") == "1"\n', 2, 'os.environ.get("A", "1")'),
        ('import os\nx = os.environ["A"]\n', 2, 'os.environ["A"]'),
        ('import os\nx = "A" in os.environ\n', 2, '"A" in os.environ'),
        ('import os\nx = "A" not in os.environ\n', 2, '"A" not in os.environ'),
        ('import os\nx = os.environ.pop("A", None)\n', 2, 'os.environ.pop("A", None)'),
        ('import os\nif os.environ.pop("A", None):\n    pass\n', 2, 'os.environ.pop("A", None)'),
        ('import os\nx = os.environ.setdefault("A", "1")\n', 2, 'os.environ.setdefault("A", "1")'),
        ('from os import environ\nx = environ.get("A")\n', 2, 'environ.get("A")'),
        ('from os import environ as env\nx = env["A"]\n', 2, 'env["A"]'),
        ('import os as _os\nx = _os.getenv("A")\n', 2, '_os.getenv("A")'),
        ('import os.path\nx = os.getenv("A")\n', 2, 'os.getenv("A")'),
        ('import os.path\nx = os.environ["A"]\n', 2, 'os.environ["A"]'),
        ('import os\n\ndef f():\n    return os.environ.get("A")\n', 4, 'os.environ.get("A")'),
        ("import os\nx = int(os.environ[NAME_ENV])\n", 2, "os.environ[NAME_ENV]"),
        (
            "from tensorrt_llm.serve.envs import TRTLLM_SERVER_DISABLE_GC\n",
            1,
            "from tensorrt_llm.serve.envs import TRTLLM_SERVER_DISABLE_GC",
        ),
        (
            "from tensorrt_llm.serve.envs import consume, TRTLLM_SERVER_DISABLE_GC\n",
            1,
            "from tensorrt_llm.serve.envs import consume, TRTLLM_SERVER_DISABLE_GC",
        ),
        (
            "from .envs import TRTLLM_SERVER_DISABLE_GC\n",
            1,
            "from .envs import TRTLLM_SERVER_DISABLE_GC",
        ),
        ("from ..serve.envs import SLURM_PROCID\n", 1, "from ..serve.envs import SLURM_PROCID"),
    ],
)
def test_lint_flags_every_read_form(tmp_path, source, line, text):
    findings = _findings(tmp_path, source)
    assert [(finding.line, finding.source) for finding in findings] == [(line, text)]
    message = findings[0].message()
    assert message.startswith(f"mod.py:{line} ")
    assert "tensorrt_llm/serve/envs.py" in message and "tensorrt_llm/serve/AGENTS.md" in message


@pytest.mark.parametrize(
    "source",
    [
        'import os\nos.environ["A"] = "1"\n',
        'import os\ndel os.environ["A"]\n',
        'import os\nos.environ.pop("A", None)\n',
        'import os\nos.environ.setdefault("A", "1")\n',
        'import os\nos.environ.update({"A": "1"})\n',
        "import os\nenv = os.environ.copy()\n",
        "import os\nenv = dict(os.environ)\n",
        'import os\nenv = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}\n',
        "import os\nfor name in os.environ:\n    pass\n",
        'import os\nx = os.path.join("a", "b")\n',
        'import os.path\nx = os.path.join("a", "b")\n',
        'import os.path as osp\nx = osp.join("a", "b")\n',
        "from tensorrt_llm.serve import envs\nx = envs.TRTLLM_SERVER_DISABLE_GC\n",
        'from tensorrt_llm.serve import envs\nx = envs.consume("TLLM_FRONTEND_READY_FD")\n',
        'from tensorrt_llm.serve.envs import consume\nx = consume("TLLM_FRONTEND_READY_FD")\n',
        "from tensorrt_llm.serve.envs import EnvVar, environment_variables\n",
        "from . import envs\nx = envs.SLURM_PROCID\n",
        'def getenv(name):\n    return name\n\nx = getenv("A")\n',
        'environ = {}\nx = environ.get("A")\n',
    ],
)
def test_lint_allows_writes_and_copies(tmp_path, source):
    assert _findings(tmp_path, source) == []


def test_lint_allows_exactly_the_module_api():
    assert ENVS_IMPORTABLE_NAMES == set(envs.__all__), (
        "the names the lint lets code import from tensorrt_llm/serve/envs.py are the module's "
        "__all__; keep ENVS_IMPORTABLE_NAMES in tests/test_common/serve/env_lint.py in step with it"
    )


def test_lint_reports_every_read_in_file_order(tmp_path):
    source = """\
        import os

        FIRST = os.getenv("A")


        def second():
            if "B" in os.environ:
                return os.environ["B"]
            return None
        """
    findings = _findings(tmp_path, source)
    assert [(finding.line, finding.source) for finding in findings] == [
        (3, 'os.getenv("A")'),
        (7, '"B" in os.environ'),
        (8, 'os.environ["B"]'),
    ]


# ---------------------------------------------------------------------------
# The declarations reproduce the parsing of the call sites they replaced.
# ---------------------------------------------------------------------------

# Environment states a reader is checked in: unset, empty, "0", "1", text, an integer.
STATES = ("unset", "", "0", "1", "abc", "12")

# What today's call sites compute in each state; an exception class means it raises.
FLAG_OFF_UNLESS_1 = (False, False, False, True, False, False)
FLAG_ON_UNLESS_SET_OTHERWISE = (True, False, False, True, False, False)
NON_EMPTY = (False, False, True, True, True, True)
PRESENT = (False, True, True, True, True, True)
RAW = (None, "", "0", "1", "abc", "12")
RAW_EMPTY_DEFAULT = ("", "", "0", "1", "abc", "12")
RAW_ZERO_DEFAULT = ("0", "", "0", "1", "abc", "12")
REQUIRED = (KeyError, "", "0", "1", "abc", "12")
REQUIRED_INT = (KeyError, ValueError, 0, 1, ValueError, 12)
INT_DEFAULT_180 = (180, ValueError, 0, 1, ValueError, 12)
OPTIONAL_INT = (None, ValueError, 0, 1, ValueError, 12)
FLOAT_DEFAULT_180 = (180.0, ValueError, 0.0, 1.0, ValueError, 12.0)
FLOAT_DEFAULT_300 = (300.0, ValueError, 0.0, 1.0, ValueError, 12.0)
NAMES = ((), (), ("0",), ("1",), ("abc",), ("12",))

EXPECTED = {
    # Deployment switches.
    "TRTLLM_SERVER_DISABLE_GC": FLAG_OFF_UNLESS_1,
    "TRTLLM_DISAGG_SERVER_DISABLE_GC": FLAG_ON_UNLESS_SET_OTHERWISE,
    "TRTLLM_RESPONSES_API_DISABLE_STORE": NON_EMPTY,
    "TRTLLM_RUNTIME_CONTROL_API_KEY": RAW,
    "TRTLLM_MEDIA_STORAGE_PATH": RAW,
    "DISABLE_HARMONY_ADAPTER": FLAG_OFF_UNLESS_1,
    "TRTLLM_KVCACHE_TIME_OUTPUT_PATH": RAW_EMPTY_DEFAULT,
    "TRTLLM_KIMI_PARAM_POLICY": FLAG_OFF_UNLESS_1,
    "TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR": FLAG_OFF_UNLESS_1,
    "TRTLLM_DISAGG_NO_RETRY": FLAG_OFF_UNLESS_1,
    "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": FLAG_OFF_UNLESS_1,
    "TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT": FLOAT_DEFAULT_180,
    "TRTLLM_DISAGG_COORDINATOR_PORT": OPTIONAL_INT,
    "TRTLLM_DISAGG_COORDINATOR_UDS": FLAG_ON_UNLESS_SET_OTHERWISE,
    "TLLM_FRONTEND_READY_TIMEOUT": FLOAT_DEFAULT_300,
    "TRTLLM_ANTHROPIC_BATCH_CONCURRENCY": RAW,
    "TRTLLM_ANTHROPIC_BATCH_MAX_RETAINED": RAW,
    "TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK": OPTIONAL_INT,
    "TRTLLM_RESPONSES_DEBUG": FLAG_OFF_UNLESS_1,
    "TRTLLM_ENABLE_RENDER_ENDPOINTS": FLAG_OFF_UNLESS_1,
    "TRTLLM_RENDER_IGNORED_FIELDS": NAMES,
    "TRTLLM_DISALLOW_LOCAL_MEDIA_PATH": RAW_ZERO_DEFAULT,
    "TRTLLM_WEB_SEARCH": RAW,
    "TRTLLM_WEB_SEARCH_MAX_RESULTS": RAW,
    "TRTLLM_WEB_SEARCH_TIMEOUT_S": RAW,
    "TRTLLM_WEB_SEARCH_MAX_USES": RAW,
    "TRTLLM_WEB_SEARCH_RETRIES": RAW,
    "TRTLLM_WEB_SEARCH_UNAVAILABLE": RAW,
    "BRAVE_SEARCH_API_KEY": RAW,
    "TAVILY_API_KEY": RAW,
    "SEARXNG_URL": RAW,
    "TRTLLM_TRUST_KV_HINTS": FLAG_OFF_UNLESS_1,
    "TRTLLM_DISAGG_SHARED_DRAFT_KV": RAW,
    "PROMETHEUS_MULTIPROC_DIR": RAW,
    # Launcher-internal variables.
    "TRTLLM_DISAGG_CONFIG_FILE": REQUIRED,
    "TRTLLM_DISAGG_COORDINATOR_URL": REQUIRED,
    "TRTLLM_DISAGG_METADATA_CONFIG_FILE": RAW,
    "TRTLLM_DISAGG_REQUEST_TIMEOUT": INT_DEFAULT_180,
    "TRTLLM_DISAGG_SERVER_START_TIMEOUT": INT_DEFAULT_180,
    "TRTLLM_DISAGG_SCHEDULE_STYLE": RAW,
    "TRTLLM_DISAGG_LOG_LEVEL": RAW,
    "TRTLLM_DISAGG_WORKER_PROCESS_ID": RAW_ZERO_DEFAULT,
    "TLLM_DISAGG_INSTANCE_IDX": RAW,
    "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT": RAW,
    "TLLM_EXECUTOR_ATTACH_INFO": PRESENT,
    "TLLM_EXECUTOR_FRONTEND_ID": REQUIRED_INT,
    "TLLM_FRONTEND_READY_FD": RAW,
    "TLLM_FRONTEND_AFFINITY_FD": RAW,
    "TLLM_SERVE_LAUNCHER_UDS": RAW,
    "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": REQUIRED,
    "TRTLLM_DISAGG_ROLE": RAW,
    "TRTLLM_DISAGG_DEPLOYMENT_ID": RAW,
    # External variables.
    "OMPI_COMM_WORLD_RANK": RAW,
    "SLURM_PROCID": RAW,
}


# Where each variable was read before its declaration moved into envs.py (the parsing the
# reader reproduces). Several files: the ones that shared one expression.
CALL_SITES = {
    "TRTLLM_SERVER_DISABLE_GC": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_SERVER_DISABLE_GC": "tensorrt_llm/commands/serve.py",
    "TRTLLM_RESPONSES_API_DISABLE_STORE": "tensorrt_llm/commands/serve.py, tensorrt_llm/serve/openai_server.py",
    "TRTLLM_RUNTIME_CONTROL_API_KEY": "tensorrt_llm/commands/serve.py",
    "TRTLLM_MEDIA_STORAGE_PATH": "tensorrt_llm/serve/openai_server.py",
    "DISABLE_HARMONY_ADAPTER": (
        "tensorrt_llm/serve/chat_tokenization.py, tensorrt_llm/serve/openai_server.py, "
        "tensorrt_llm/serve/render/resources.py"
    ),
    "TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "tensorrt_llm/serve/openai_server.py, tensorrt_llm/serve/responses_utils.py",
    "TRTLLM_KIMI_PARAM_POLICY": "tensorrt_llm/serve/extensions/kimi_k3.py",
    "TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR": "tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py",
    "TRTLLM_DISAGG_NO_RETRY": "tensorrt_llm/serve/openai_client.py",
    "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": (
        "tensorrt_llm/serve/disagg_coordinator.py, tensorrt_llm/serve/openai_disagg_service.py"
    ),
    "TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT": "tensorrt_llm/serve/disagg_coordinator.py",
    "TRTLLM_DISAGG_COORDINATOR_PORT": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_COORDINATOR_UDS": "tensorrt_llm/commands/serve.py",
    "TLLM_FRONTEND_READY_TIMEOUT": "tensorrt_llm/serve/_frontend_processes.py",
    "TRTLLM_ANTHROPIC_BATCH_CONCURRENCY": "tensorrt_llm/serve/anthropic_batches.py",
    "TRTLLM_ANTHROPIC_BATCH_MAX_RETAINED": "tensorrt_llm/serve/anthropic_batches.py",
    "TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK": "tensorrt_llm/serve/router_utils.py",
    "TRTLLM_RESPONSES_DEBUG": "tensorrt_llm/serve/responses_utils.py",
    "TRTLLM_ENABLE_RENDER_ENDPOINTS": "tensorrt_llm/serve/render/api_router.py",
    "TRTLLM_RENDER_IGNORED_FIELDS": "tensorrt_llm/serve/render/serving.py",
    "TRTLLM_DISALLOW_LOCAL_MEDIA_PATH": "tensorrt_llm/serve/visual_gen_utils.py",
    "TRTLLM_WEB_SEARCH": "tensorrt_llm/serve/web_search.py",
    "TRTLLM_WEB_SEARCH_MAX_RESULTS": "tensorrt_llm/serve/web_search.py",
    "TRTLLM_WEB_SEARCH_TIMEOUT_S": "tensorrt_llm/serve/web_search.py",
    "TRTLLM_WEB_SEARCH_MAX_USES": "tensorrt_llm/serve/web_search.py",
    "TRTLLM_WEB_SEARCH_RETRIES": "tensorrt_llm/serve/web_search.py",
    "TRTLLM_WEB_SEARCH_UNAVAILABLE": "tensorrt_llm/serve/responses_web_search.py",
    "BRAVE_SEARCH_API_KEY": "tensorrt_llm/serve/web_search.py",
    "TAVILY_API_KEY": "tensorrt_llm/serve/web_search.py",
    "SEARXNG_URL": "tensorrt_llm/serve/web_search.py",
    "TRTLLM_TRUST_KV_HINTS": "tensorrt_llm/serve/openai_server.py, tensorrt_llm/serve/openai_disagg_server.py",
    "TRTLLM_DISAGG_SHARED_DRAFT_KV": "tensorrt_llm/commands/serve.py",
    "PROMETHEUS_MULTIPROC_DIR": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_CONFIG_FILE": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_COORDINATOR_URL": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_METADATA_CONFIG_FILE": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_REQUEST_TIMEOUT": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_SERVER_START_TIMEOUT": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_SCHEDULE_STYLE": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_LOG_LEVEL": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_WORKER_PROCESS_ID": "tensorrt_llm/commands/serve.py",
    "TLLM_DISAGG_INSTANCE_IDX": "tensorrt_llm/commands/serve.py",
    "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT": "tensorrt_llm/commands/serve.py",
    "TLLM_EXECUTOR_ATTACH_INFO": "tensorrt_llm/serve/_frontend_processes.py",
    "TLLM_EXECUTOR_FRONTEND_ID": "tensorrt_llm/serve/_frontend_processes.py",
    "TLLM_FRONTEND_READY_FD": "tensorrt_llm/serve/_frontend_processes.py (consumed)",
    "TLLM_FRONTEND_AFFINITY_FD": "tensorrt_llm/serve/_frontend_processes.py (consumed)",
    "TLLM_SERVE_LAUNCHER_UDS": "tensorrt_llm/commands/serve.py (consumed)",
    "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_ROLE": "tensorrt_llm/commands/serve.py",
    "TRTLLM_DISAGG_DEPLOYMENT_ID": (
        "tensorrt_llm/commands/serve.py (written there; read by tensorrt_llm/usage/usage_lib.py)"
    ),
    "OMPI_COMM_WORLD_RANK": "tensorrt_llm/commands/serve.py",
    "SLURM_PROCID": "tensorrt_llm/commands/serve.py",
}


def _set_state(monkeypatch, name, state):
    if state == "unset":
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, state)


def _describe(state):
    return "unset" if state == "unset" else f"set to {state!r}"


def test_parsing_table_covers_every_declaration():
    assert set(EXPECTED) == set(envs.environment_variables), (
        "every declared variable has a row in EXPECTED (and every row a declaration): add the "
        "parsing today's call site does for each state in STATES."
    )
    assert set(CALL_SITES) == set(envs.environment_variables), (
        "every declared variable names the call site its reader reproduces in CALL_SITES"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_readers_reproduce_todays_parsing(monkeypatch, name):
    for state, expected in zip(STATES, EXPECTED[name]):
        _set_state(monkeypatch, name, state)
        if isinstance(expected, type) and issubclass(expected, BaseException):
            with pytest.raises(expected):
                getattr(envs, name)
            continue
        actual = getattr(envs, name)
        assert actual == expected and type(actual) is type(expected), (
            f"envs.{name} {_describe(state)} gave {actual!r}; today's code gives {expected!r}. "
            f"The reader must reproduce the parsing at {CALL_SITES[name]}; changing it is a "
            "behavior change."
        )


def test_render_ignored_fields_are_split_and_stripped(monkeypatch):
    monkeypatch.setenv("TRTLLM_RENDER_IGNORED_FIELDS", " a ,b,, c ,")
    assert envs.TRTLLM_RENDER_IGNORED_FIELDS == ("a", "b", "c")


def test_disable_store_is_any_non_empty_value(monkeypatch):
    # Not a boolean: TRTLLM_RESPONSES_API_DISABLE_STORE=0 disables the store too.
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "0")
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is True
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "")
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is False


def test_unset_and_empty_differ_where_they_did(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_SERVER_DISABLE_GC", raising=False)
    assert envs.TRTLLM_DISAGG_SERVER_DISABLE_GC is True
    monkeypatch.setenv("TRTLLM_DISAGG_SERVER_DISABLE_GC", "")
    assert envs.TRTLLM_DISAGG_SERVER_DISABLE_GC is False


def test_values_are_read_at_access_time(monkeypatch):
    monkeypatch.delenv("TRTLLM_RESPONSES_API_DISABLE_STORE", raising=False)
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is False
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "1")
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is True
    monkeypatch.delenv("TRTLLM_RESPONSES_API_DISABLE_STORE")
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is False


# ---------------------------------------------------------------------------
# The module contract.
# ---------------------------------------------------------------------------


def test_consume_reads_once(monkeypatch):
    monkeypatch.setenv("TLLM_FRONTEND_READY_FD", "7")
    assert envs.consume("TLLM_FRONTEND_READY_FD") == "7"
    assert "TLLM_FRONTEND_READY_FD" not in os.environ
    assert envs.consume("TLLM_FRONTEND_READY_FD") is None


def test_consume_rejects_names_that_are_not_internal(monkeypatch):
    monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", "1")
    with pytest.raises(KeyError, match="not a declared internal variable"):
        envs.consume("TRTLLM_SERVER_DISABLE_GC")
    assert os.environ["TRTLLM_SERVER_DISABLE_GC"] == "1", (
        "a rejected consume() must not remove the variable"
    )
    with pytest.raises(KeyError, match="not a declared internal variable"):
        envs.consume("TRTLLM_NOT_DECLARED")


def test_undeclared_name_raises_attribute_error():
    with pytest.raises(AttributeError, match=r"tensorrt_llm/serve/envs\.py"):
        envs.TRTLLM_NOT_DECLARED
    assert not hasattr(envs, "TRTLLM_NOT_DECLARED")


def test_required_internal_variable_raises_key_error(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_CONFIG_FILE", raising=False)
    with pytest.raises(KeyError):
        envs.TRTLLM_DISAGG_CONFIG_FILE


def test_dir_lists_the_declared_variables():
    assert set(envs.environment_variables) <= set(dir(envs))


# ---------------------------------------------------------------------------
# The declarations agree with the code around them.
# ---------------------------------------------------------------------------

KINDS_IN_ORDER = ("deploy", "internal", "external")
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def test_declarations_are_complete():
    problems = []
    kinds_seen = []
    for name, var in envs.environment_variables.items():
        if var.name != name:
            problems.append(f"{name}: declared under the key {name!r} but named {var.name!r}")
        if var.kind not in KINDS_IN_ORDER:
            problems.append(f"{name}: kind {var.kind!r} is not one of {KINDS_IN_ORDER}")
        if not var.default.strip():
            problems.append(f"{name}: default is empty; say what an unset variable means")
        if not var.description.strip():
            problems.append(f"{name}: description is empty")
        for reader in var.also_read_by:
            path = REPO_ROOT / reader
            if not path.is_file():
                problems.append(f"{name}: also_read_by names {reader}, which does not exist")
            elif name not in path.read_text(errors="replace"):
                problems.append(
                    f"{name}: also_read_by names {reader}, which does not mention the variable"
                )
        if not kinds_seen or kinds_seen[-1] != var.kind:
            kinds_seen.append(var.kind)
    if kinds_seen != list(KINDS_IN_ORDER):
        problems.append(
            f"declarations are listed in the order {kinds_seen}; keep the three blocks deploy, "
            "internal, external, in that order"
        )
    assert not problems, "\n".join(problems)


def test_secrets_are_marked():
    unmarked = [
        name
        for name, var in envs.environment_variables.items()
        if (name.endswith(SECRET_SUFFIXES) or name == "TLLM_EXECUTOR_ATTACH_INFO")
        and not var.secret
    ]
    assert not unmarked, (
        f"{unmarked} carry credentials; declare them with secret=True in tensorrt_llm/serve/envs.py"
    )


def test_type_declarations_match_the_declared_names():
    tree = ast.parse(ENVS_PATH.read_text())
    annotated = set()
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    annotated.add(statement.target.id)
    assert annotated == set(envs.environment_variables), (
        "the TYPE_CHECKING block of tensorrt_llm/serve/envs.py declares the type of every variable "
        f"and nothing else; missing {sorted(set(envs.environment_variables) - annotated)}, "
        f"stale {sorted(annotated - set(envs.environment_variables))}"
    )


def _module_level_string_assignments(path):
    """``NAME = "value"`` statements at module level, as name -> value."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _envs_enums(path):
    """``class <Name>Envs(...)`` definitions in a module, as class name -> {member: value}."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("Envs"):
            members = {}
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            members[target.id] = statement.value.value
            out[node.name] = members
    return out


def test_launcher_enums_are_declared():
    from tensorrt_llm.executor.utils import LlmLauncherEnvs

    enums = {}
    for path in scope_files(REPO_ROOT, visual_gen=False):
        enums.update(_envs_enums(path))
    assert {"DisaggLauncherEnvs", "DisaggWorkerEnvs"} <= set(enums), (
        f"the disaggregated launcher enums were not found in the linted directories (found {sorted(enums)})"
    )
    undeclared = [
        f"{enum}.{member} ({value}) is not declared in tensorrt_llm/serve/envs.py; add it to the "
        "internal block (or the deploy block if users set it)."
        for enum, members in sorted(enums.items())
        for member, value in members.items()
        if value not in envs.environment_variables
    ]
    assert not undeclared, "\n".join(undeclared)
    # The one LlmLauncherEnvs member the serve CLI reads.
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR.value in envs.environment_variables


def test_env_name_constants_are_declared():
    constants = {}
    for path in scope_files(REPO_ROOT, visual_gen=False):
        for name, value in _module_level_string_assignments(path).items():
            if name.endswith("_ENV"):
                constants[(path.relative_to(REPO_ROOT).as_posix(), name)] = value
    assert constants, "no <NAME>_ENV constant found in the linted directories; the scan is broken"
    undeclared = [
        f"{path}: {name} = {value!r} names a variable that tensorrt_llm/serve/envs.py does not declare"
        for (path, name), value in sorted(constants.items())
        if value not in envs.environment_variables
    ]
    assert not undeclared, "\n".join(undeclared)


def test_agents_md_states_the_rule():
    text = AGENTS_MD.read_text()
    assert "`serve/envs.py`" in text and "`tests/unittest/serve/test_envs.py`" in text, (
        "tensorrt_llm/serve/AGENTS.md must tell contributors to read environment variables through "
        "serve/envs.py and name this test as the check"
    )
