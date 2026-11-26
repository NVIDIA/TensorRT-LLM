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
  trtllm-serve CLI reads ``os.environ`` directly, and the lint recognizes the
  documented read forms and ignores writes (so an empty scan cannot pass vacuously);
- the declarations: the table states every reader's parsing contract per
  environment state, values are read at access time, secrets are marked, and
  launcher enums and ``*_ENV`` name constants agree with the declarations;
- the module contract: undeclared names raise ``AttributeError``, required
  variables raise ``KeyError``, ``consume()`` reads once.
"""

import ast
import inspect
import os
import pydoc
import textwrap
from pathlib import Path

import pytest
from test_common.serve.env_lint import (
    ENVS_IMPORTABLE_NAMES,
    Finding,
    declared_env_names,
    find_direct_env_reads,
    scope_files,
)

from tensorrt_llm.serve import envs

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# The lint over the serve tree and the CLI.
# ---------------------------------------------------------------------------


def test_no_direct_environment_reads() -> None:
    files = scope_files(REPO_ROOT, visual_gen=False)
    assert files, "the env-lint scanned no file; scope_files() lost the serve tree"
    findings = find_direct_env_reads(files, root=REPO_ROOT)
    assert not findings, "\n".join(finding.message() for finding in findings)


def test_scope_files_cover_the_serve_tree_and_the_cli() -> None:
    relative = {
        path.relative_to(REPO_ROOT).as_posix() for path in scope_files(REPO_ROOT, visual_gen=False)
    }
    assert "tensorrt_llm/commands/serve.py" in relative
    assert "tensorrt_llm/serve/openai_server.py" in relative
    assert "tensorrt_llm/serve/envs.py" not in relative, (
        "the module reading os.environ for everyone is not linted"
    )
    assert not any(name.startswith("tensorrt_llm/serve/scripts/") for name in relative), (
        "serve/scripts/ is a published client package and stays outside the lint"
    )
    visual_gen_relative = {
        path.relative_to(REPO_ROOT).as_posix() for path in scope_files(REPO_ROOT, visual_gen=True)
    }
    assert visual_gen_relative == {
        "tensorrt_llm/serve/openai_video_routes.py",
        "tensorrt_llm/serve/visual_gen_utils.py",
        "tensorrt_llm/serve/visual_gen_metrics.py",
        "tensorrt_llm/serve/_visual_gen_errors.py",
    }
    assert visual_gen_relative <= relative


def _findings(tmp_path: Path, source: str) -> list[Finding]:
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
        ('import os\nenv = os.environ\nx = env.get("A")\n', 3, 'env.get("A")'),
        ('import os\nenv = os.environ.copy()\nx = env["A"]\n', 3, 'env["A"]'),
        ('import os\nx = os.environ.copy().get("A")\n', 2, 'os.environ.copy().get("A")'),
        ('import os\nx = dict(os.environ).get("A")\n', 2, 'dict(os.environ).get("A")'),
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
        (
            "from tensorrt_llm.serve import envs\nx = envs.TRTLLM_SERVER_DISABLE_GC\n",
            2,
            "envs.TRTLLM_SERVER_DISABLE_GC",
        ),
        (
            "from . import envs\nx = envs.SLURM_PROCID\n",
            2,
            "envs.SLURM_PROCID",
        ),
        (
            "from tensorrt_llm.serve import envs\n\nclass C:\n    X = envs.TRTLLM_SERVER_DISABLE_GC\n",
            4,
            "envs.TRTLLM_SERVER_DISABLE_GC",
        ),
        (
            "from tensorrt_llm.serve import envs\n\ndef f(x=envs.TRTLLM_SERVER_DISABLE_GC):\n    return x\n",
            3,
            "envs.TRTLLM_SERVER_DISABLE_GC",
        ),
        (
            'from tensorrt_llm.serve import envs\nx = envs.consume("TLLM_FRONTEND_READY_FD")\n',
            2,
            'envs.consume("TLLM_FRONTEND_READY_FD")',
        ),
        (
            'from tensorrt_llm.serve import envs\ncb = envs.consume\nx = cb("TLLM_FRONTEND_READY_FD")\n',
            3,
            'cb("TLLM_FRONTEND_READY_FD")',
        ),
        (
            'from tensorrt_llm.serve.envs import consume\nx = consume("TLLM_FRONTEND_READY_FD")\n',
            2,
            'consume("TLLM_FRONTEND_READY_FD")',
        ),
        (
            "from tensorrt_llm.serve import envs\n\ndef f():\n    return envs.TRTLLM_NOT_DECLARED\n",
            4,
            "envs.TRTLLM_NOT_DECLARED",
        ),
        (
            'import os\nenv = os.environ\nif cond:\n    env = {}\nx = env.get("A")\n',
            5,
            'env.get("A")',
        ),
        (
            (
                "import os\n"
                "from typing import TYPE_CHECKING\n"
                "env = os.environ\n"
                "if TYPE_CHECKING:\n"
                "    env = {}\n"
                'x = env.get("A")\n'
            ),
            6,
            'env.get("A")',
        ),
        (
            'import os\n\ndef f():\n    return env.get("A")\n\nenv = os.environ\n',
            4,
            'env.get("A")',
        ),
        (
            'import os\n\ndef f(x: os.getenv("A")):\n    return x\n',
            3,
            'os.getenv("A")',
        ),
        (
            'import os\nx: os.getenv("A") = 1\n',
            2,
            'os.getenv("A")',
        ),
        (
            "from tensorrt_llm.serve import envs\nxs = [envs.SLURM_PROCID for _ in rows]\n",
            2,
            "envs.SLURM_PROCID",
        ),
        (
            "from tensorrt_llm.serve import envs\nxs = (x for x in envs.TRTLLM_RENDER_IGNORED_FIELDS)\n",
            2,
            "envs.TRTLLM_RENDER_IGNORED_FIELDS",
        ),
        (
            'import os\n\nclass C:\n    os = None\n    def f(self):\n        return os.getenv("A")\n',
            6,
            'os.getenv("A")',
        ),
        (
            (
                "import os\n"
                "\n"
                "def outer():\n"
                "    def inner():\n"
                '        return env.get("A")\n'
                "    env = os.environ\n"
                "    return inner()\n"
            ),
            5,
            'env.get("A")',
        ),
        (
            'import os\nenv = {}\nfor _ in range(2):\n    x = env.get("A")\n    env = os.environ\n',
            4,
            'env.get("A")',
        ),
        (
            (
                "import os\n"
                "env = {}\n"
                "try:\n"
                "    env = os.environ\n"
                "    might_fail()\n"
                "    env = {}\n"
                "except ValueError:\n"
                '    x = env.get("A")\n'
            ),
            8,
            'env.get("A")',
        ),
        (
            'import os\nenv = os.environ if cond else {}\nx = env.get("A")\n',
            3,
            'env.get("A")',
        ),
        (
            (
                "import os\n"
                "env = {}\n"
                "try:\n"
                "    env = os.environ\n"
                "    might_fail()\n"
                "    env = {}\n"
                "finally:\n"
                '    x = env.get("A")\n'
            ),
            8,
            'env.get("A")',
        ),
        (
            (
                "import os\n"
                "a = b = c = d = e = {}\n"
                "for _ in range(6):\n"
                '    x = a.get("A")\n'
                "    a = b\n"
                "    b = c\n"
                "    c = d\n"
                "    d = e\n"
                "    e = os.environ\n"
            ),
            4,
            'a.get("A")',
        ),
        (
            (
                "import os\n"
                "env = {}\n"
                "for _ in range(1):\n"
                "    env = os.environ\n"
                "    break\n"
                "else:\n"
                "    env = {}\n"
                'x = env.get("A")\n'
            ),
            8,
            'env.get("A")',
        ),
        (
            (
                "def outer():\n"
                "    def inner():\n"
                '        return os.getenv("A")\n'
                "    import os\n"
                "    res = inner()\n"
                "    os = None\n"
                "    return res\n"
            ),
            3,
            'os.getenv("A")',
        ),
        (
            (
                "import os\n"
                "env = {}\n"
                "try:\n"
                "    if enabled:\n"
                "        env = os.environ\n"
                "        risky()\n"
                "        env = {}\n"
                "finally:\n"
                '    x = env.get("A")\n'
            ),
            9,
            'env.get("A")',
        ),
        (
            (
                "import os\n"
                "env = {}\n"
                "for i in range(2):\n"
                '    x = env.get("A")\n'
                "    if i == 0:\n"
                "        env = os.environ\n"
                "        continue\n"
                "    env = {}\n"
            ),
            4,
            'env.get("A")',
        ),
        (
            'import os\n\ndef f():\n    return env.get("A")\n\nif (env := os.environ) is not None:\n    f()\n',
            4,
            'env.get("A")',
        ),
        (
            'import os\nenv, other = os.environ, {}\nx = env.get("A")\n',
            3,
            'env.get("A")',
        ),
        (
            'import os as Config\n\nclass Config:\n    value = Config.getenv("A")\n',
            4,
            'Config.getenv("A")',
        ),
        (
            (
                "import os\n"
                "env = {}\n"
                "for _ in [1]:\n"
                "    try:\n"
                "        break\n"
                "    finally:\n"
                "        env = os.environ\n"
                'x = env.get("A")\n'
            ),
            8,
            'env.get("A")',
        ),
        (
            'import os\nenv = {}\nwhile not (env := os.environ):\n    break\nelse:\n    x = env.get("A")\n',
            6,
            'env.get("A")',
        ),
        (
            'import os\na, b = os.environ, {}\na, b = b, a\nx = b.get("A")\n',
            4,
            'b.get("A")',
        ),
        (
            'import os\nenv = os.environ\nflag and (env := {})\nx = env.get("A")\n',
            4,
            'env.get("A")',
        ),
        (
            'import os\nenv = {}\n[(env := os.environ) for _ in [1]]\nx = env.get("A")\n',
            4,
            'env.get("A")',
        ),
        (
            (
                "import os\n"
                "env = {}\n"
                "match mode:\n"
                "    case 1:\n"
                "        env = os.environ\n"
                "    case _:\n"
                "        env = {}\n"
                'x = env.get("A")\n'
            ),
            8,
            'env.get("A")',
        ),
        (
            'import os\n\nclass C:\n    x = os.getenv("A")\n    os = None\n',
            4,
            'os.getenv("A")',
        ),
        (
            (
                "import os\n"
                "\n"
                "def outer():\n"
                "    env = {}\n"
                "    def mid():\n"
                "        nonlocal env\n"
                "        env = {}\n"
                "        def inner():\n"
                "            nonlocal env\n"
                "            env = os.environ\n"
                "        inner()\n"
                "    mid()\n"
                '    return env.get("A")\n'
            ),
            13,
            'env.get("A")',
        ),
        (
            'import os\n\ndef read(env=os.environ):\n    return env.get("A")\n',
            4,
            'env.get("A")',
        ),
        (
            'import os\n\ndef read(*, env=os.environ):\n    return env.get("A")\n',
            4,
            'env.get("A")',
        ),
        (
            'import os\nread = lambda env=os.environ: env.get("A")\n',
            2,
            'env.get("A")',
        ),
        (
            "from tensorrt_llm.serve import envs\nxs = list(envs.SLURM_PROCID for _ in range(1))\n",
            2,
            "envs.SLURM_PROCID",
        ),
        (
            'import os\nsnapshot = dict(**os.environ)\nx = snapshot.get("A")\n',
            3,
            'snapshot.get("A")',
        ),
        (
            'from os import getenv\nTYPE_CHECKING = True\nif TYPE_CHECKING:\n    x = getenv("A")\n',
            4,
            'getenv("A")',
        ),
        (
            'from os import getenv\nif config.TYPE_CHECKING:\n    x = getenv("A")\n',
            3,
            'getenv("A")',
        ),
        (
            "from tensorrt_llm.serve import envs\nrank = next((envs.SLURM_PROCID for _ in range(1)), None)\n",
            2,
            "envs.SLURM_PROCID",
        ),
        (
            "from tensorrt_llm.serve import envs\nfor rank in (envs.SLURM_PROCID for _ in range(1)):\n    pass\n",
            2,
            "envs.SLURM_PROCID",
        ),
        (
            'import os\n\ndef read(get=os.environ.get):\n    return get("A")\n',
            4,
            'get("A")',
        ),
        (
            'import os\nget = os.environ.get\nx = get("A")\n',
            3,
            'get("A")',
        ),
        (
            (
                "import os as cfg\n"
                "\n"
                "def make(cfg):\n"
                "    class C:\n"
                '        value = cfg.getenv("A")\n'
                "        cfg = None\n"
                "    return C\n"
            ),
            5,
            'cfg.getenv("A")',
        ),
        (
            'import os\nchild_env = {}\nchild_env |= os.environ\nx = child_env.get("A")\n',
            4,
            'child_env.get("A")',
        ),
        (
            'import os\nchild = {}\n\ndef fill():\n    child.update(os.environ)\n\nfill()\nx = child.get("A")\n',
            8,
            'child.get("A")',
        ),
        (
            "from tensorrt_llm.serve import envs\nranks = [v for v in (envs.SLURM_PROCID for _ in range(1))]\n",
            2,
            "envs.SLURM_PROCID",
        ),
        (
            (
                "import os\n"
                "\n"
                "def outer():\n"
                "    from typing import TYPE_CHECKING as TC\n"
                "    def mid():\n"
                "        def inner():\n"
                "            nonlocal TC\n"
                "            TC = True\n"
                "        inner()\n"
                "    mid()\n"
                "    if TC:\n"
                '        return os.getenv("A")\n'
                "    return None\n"
            ),
            12,
            'os.getenv("A")',
        ),
        (
            'import os\nchild = {}\nalias = child\nalias.update(os.environ)\nx = child.get("A")\n',
            5,
            'child.get("A")',
        ),
        (
            'import os\nx = dict(os.environ.items()).get("A")\n',
            2,
            'dict(os.environ.items()).get("A")',
        ),
        (
            "from tensorrt_llm.serve import envs\nfound = 0 in (envs.SLURM_PROCID for _ in range(1))\n",
            2,
            "envs.SLURM_PROCID",
        ),
        (
            (
                "import os\n"
                "from typing import TYPE_CHECKING as TC\n"
                "\n"
                "def f():\n"
                "    import typing as TC\n"
                "    if TC:\n"
                '        return os.getenv("A")\n'
                "    return None\n"
            ),
            7,
            'os.getenv("A")',
        ),
    ],
)
def test_lint_flags_every_read_form(tmp_path: Path, source: str, line: int, text: str) -> None:
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
        'import os\nenv = os.environ.copy()\nenv["A"] = "1"\nenv.pop("B", None)\n',
        "import os\nenv = dict(os.environ)\n",
        'import os\nenv = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}\n',
        "import os\nfor name in os.environ:\n    pass\n",
        'import os\nenv = os.environ\nxs = [env.get("A") for env in rows]\n',
        (
            "import os\n"
            "env = {}\n"
            "try:\n"
            "    cb = lambda: (env := os.environ)\n"
            "    raise ValueError\n"
            "except ValueError:\n"
            '    x = env.get("A")\n'
        ),
        (
            "import os\n"
            "from typing import TYPE_CHECKING as TC\n"
            "def helper(TC):\n"
            "    return TC\n"
            "if TC:\n"
            '    x = os.getenv("A")\n'
        ),
        "import os\nchild_env = {}\nchild_env |= os.environ\n",
        'import os\nx = os.path.join("a", "b")\n',
        'import os.path\nx = os.path.join("a", "b")\n',
        'import os.path as osp\nx = osp.join("a", "b")\n',
        "from tensorrt_llm.serve import envs\n\ndef f():\n    return envs.TRTLLM_SERVER_DISABLE_GC\n",
        "from tensorrt_llm.serve import envs\n\ndef f(envs):\n    return envs.value\n",
        "from tensorrt_llm.serve import envs\ncallback = envs.consume\n",
        "from tensorrt_llm.serve import envs\nxs = (envs.SLURM_PROCID for _ in rows)\n",
        "from tensorrt_llm.serve import envs\nxs = next(iter(()), (envs.SLURM_PROCID for _ in range(1)))\n",
        "from tensorrt_llm.serve import envs\nxs = Builder().join(envs.SLURM_PROCID for _ in range(1))\n",
        'from tensorrt_llm.serve import envs\n\ndef f():\n    return envs.consume("TLLM_FRONTEND_READY_FD")\n',
        'from tensorrt_llm.serve.envs import consume\n\ndef f():\n    return consume("TLLM_FRONTEND_READY_FD")\n',
        "from tensorrt_llm.serve.envs import EnvVar, environment_variables\n",
        "from tensorrt_llm.serve import envs\nX = envs.environment_variables\n",
        "from . import envs\n\ndef f():\n    return envs.SLURM_PROCID\n",
        'def getenv(name):\n    return name\n\nx = getenv("A")\n',
        'environ = {}\nx = environ.get("A")\n',
    ],
)
def test_lint_allows_writes_and_copies(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == []


def test_lint_resolves_relative_imports_against_package_path(tmp_path: Path) -> None:
    # Include an outer tensorrt_llm segment (like a worktree under /code/tensorrt_llm/.worktrees/)
    # to verify both root-relative and absolute path invocations resolve the inner package.
    outer = tmp_path / "tensorrt_llm" / "worktree"
    sub = outer / "tensorrt_llm" / "serve" / "render" / "foo.py"
    sub.parent.mkdir(parents=True)
    # `from . import envs` in serve/render/foo.py points to serve.render.envs, not serve.envs.
    sub.write_text("from . import envs\n\ndef f():\n    return envs.ANY_ATTR\n")
    assert find_direct_env_reads([sub], root=outer) == []
    assert find_direct_env_reads([sub]) == []
    # `from .. import envs` in serve/render/foo.py resolves to tensorrt_llm.serve.envs.
    sub.write_text("from .. import envs\n\ndef f():\n    return envs.TRTLLM_NOT_DECLARED\n")
    for findings in (find_direct_env_reads([sub], root=outer), find_direct_env_reads([sub])):
        assert [(f.line, f.source, f.kind) for f in findings] == [
            (4, "envs.TRTLLM_NOT_DECLARED", "unknown")
        ]


def test_lint_allows_exactly_the_module_api() -> None:
    assert ENVS_IMPORTABLE_NAMES == set(envs.__all__), (
        "the names the lint lets code import from tensorrt_llm/serve/envs.py are the module's "
        "__all__; keep ENVS_IMPORTABLE_NAMES in tests/test_common/serve/env_lint.py in step with it"
    )


def test_lint_reports_every_read_in_file_order(tmp_path: Path) -> None:
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
# The declarations follow the per-state reader contracts.
# ---------------------------------------------------------------------------

# Environment states a reader is checked in: unset, empty, "0", "1", text, an integer.
STATES = ("unset", "", "0", "1", "abc", "12")

# Expected reader values in each state; an exception class means the reader raises.
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
    "TRTLLM_DISALLOW_LOCAL_MEDIA_PATH": RAW,
    "TRTLLM_MEDIA_DISABLE_PATH_RESPONSE": RAW_EMPTY_DEFAULT,
    "TLLM_PREFIX_TOKEN_CACHE_ROUTE_BY_CONVERSATION": RAW,
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
    "TRTLLM_DISAGG_ROLE": RAW,
    "TRTLLM_DISAGG_DEPLOYMENT_ID": RAW,
    # External variables.
    "OMPI_COMM_WORLD_RANK": RAW,
    "SLURM_PROCID": RAW,
}


def _set_state(monkeypatch: pytest.MonkeyPatch, name: str, state: str) -> None:
    if state == "unset":
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, state)


def _describe(state: str) -> str:
    return "unset" if state == "unset" else f"set to {state!r}"


def test_declared_names_are_unique() -> None:
    declared = [var.name for var in (*envs._DEPLOY, *envs._INTERNAL, *envs._EXTERNAL)]
    assert len(declared) == len(set(declared)) == len(envs.environment_variables), (
        "duplicate environment variable names in tensorrt_llm/serve/envs.py would silently shadow"
    )


def test_parsing_table_covers_every_declaration() -> None:
    assert set(EXPECTED) == set(envs.environment_variables), (
        "every declared variable has a row in EXPECTED (and every row a declaration): add the "
        "reader contract for each state in STATES."
    )


@pytest.mark.parametrize(
    ("name", "state", "expected"),
    [
        (name, state, expected)
        for name in sorted(EXPECTED)
        for state, expected in zip(STATES, EXPECTED[name], strict=True)
    ],
    ids=lambda val: str(val) if not isinstance(val, type) else val.__name__,
)
def test_readers_follow_declared_parsing_contract(
    monkeypatch: pytest.MonkeyPatch, name: str, state: str, expected: object
) -> None:
    _set_state(monkeypatch, name, state)
    if isinstance(expected, type) and issubclass(expected, BaseException):
        with pytest.raises(expected):
            getattr(envs, name)
        return
    actual = getattr(envs, name)
    assert actual == expected and type(actual) is type(expected), (
        f"envs.{name} {_describe(state)} gave {actual!r}; its declared contract is "
        f"{expected!r}. Changing the reader or default is a behavior change."
    )


def test_render_ignored_fields_are_split_and_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTLLM_RENDER_IGNORED_FIELDS", " a ,b,, c ,")
    assert envs.TRTLLM_RENDER_IGNORED_FIELDS == ("a", "b", "c")


def test_values_are_read_at_access_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRTLLM_RESPONSES_API_DISABLE_STORE", raising=False)
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is False
    monkeypatch.setenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "1")
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is True
    monkeypatch.delenv("TRTLLM_RESPONSES_API_DISABLE_STORE")
    assert envs.TRTLLM_RESPONSES_API_DISABLE_STORE is False


# ---------------------------------------------------------------------------
# The module contract.
# ---------------------------------------------------------------------------


def test_consumable_variables_are_explicitly_marked() -> None:
    consumable = {name for name, var in envs.environment_variables.items() if var.consumable}
    assert consumable == {
        "TLLM_FRONTEND_READY_FD",
        "TLLM_FRONTEND_AFFINITY_FD",
        "TLLM_SERVE_LAUNCHER_UDS",
    }
    assert all(envs.environment_variables[name].kind == "internal" for name in consumable)
    positional = envs.EnvVar("X", "internal", lambda: None, "unset", "desc", False, ("a.py",))
    assert positional.also_read_by == ("a.py",)
    assert positional.consumable is False
    with pytest.raises(TypeError):
        envs.EnvVar("X", "internal", lambda: None, "unset", "desc", False, ("a.py",), True)  # type: ignore[misc]


def test_consume_reads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_FRONTEND_READY_FD", "7")
    assert envs.consume("TLLM_FRONTEND_READY_FD") == "7"
    assert "TLLM_FRONTEND_READY_FD" not in os.environ
    assert envs.consume("TLLM_FRONTEND_READY_FD") is None


def test_consume_rejects_non_consumable_and_undeclared_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", "1")
    with pytest.raises(KeyError, match="not a declared consumable internal variable"):
        envs.consume("TRTLLM_SERVER_DISABLE_GC")
    assert os.environ["TRTLLM_SERVER_DISABLE_GC"] == "1", (
        "a rejected consume() must not remove the variable"
    )
    monkeypatch.setenv("TLLM_EXECUTOR_FRONTEND_ID", "2")
    with pytest.raises(KeyError, match="not a declared consumable internal variable"):
        envs.consume("TLLM_EXECUTOR_FRONTEND_ID")
    assert os.environ["TLLM_EXECUTOR_FRONTEND_ID"] == "2", (
        "consume() must not remove non-consumable internal variables"
    )
    with pytest.raises(KeyError, match="not a declared consumable internal variable"):
        envs.consume("TRTLLM_NOT_DECLARED")


def test_undeclared_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match=r"tensorrt_llm/serve/envs\.py"):
        envs.TRTLLM_NOT_DECLARED
    assert not hasattr(envs, "TRTLLM_NOT_DECLARED")


def test_introspection_does_not_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)

    members = dict(inspect.getmembers(envs))
    assert members["environment_variables"] is envs.environment_variables
    assert not set(envs.environment_variables).intersection(members)
    assert "environment_variables" in pydoc.render_doc(envs, renderer=pydoc.plaintext)


# ---------------------------------------------------------------------------
# The declarations agree with the code around them.
# ---------------------------------------------------------------------------

KINDS_IN_ORDER = ("deploy", "internal", "external")
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
GATED_ALSO_READ_BY_PREFIXES = ("tensorrt_llm/serve/", "tensorrt_llm/commands/")


def _check_also_read_by(name: str, reader: str, repo_root: Path = REPO_ROOT) -> str | None:
    rel = Path(reader)
    if not reader.strip() or rel.is_absolute() or ".." in rel.parts:
        return f"{name}: also_read_by entry {reader!r} must be a non-empty repo-relative path"
    normalized = rel.as_posix()
    if not normalized.startswith(GATED_ALSO_READ_BY_PREFIXES):
        return None
    path = repo_root / rel
    if not path.is_file():
        return f"{name}: also_read_by names {reader}, which does not exist"
    if name not in path.read_text(errors="replace"):
        return f"{name}: also_read_by names {reader}, which does not mention the variable"
    return None


def test_declarations_are_complete() -> None:
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
            problem = _check_also_read_by(name, reader)
            if problem is not None:
                problems.append(problem)
        if not kinds_seen or kinds_seen[-1] != var.kind:
            kinds_seen.append(var.kind)
    if kinds_seen != list(KINDS_IN_ORDER):
        problems.append(
            f"declarations are listed in the order {kinds_seen}; keep the three blocks deploy, "
            "internal, external, in that order"
        )
    assert not problems, "\n".join(problems)


def test_also_read_by_gates_only_serve_and_commands_paths(tmp_path: Path) -> None:
    serve_file = tmp_path / "tensorrt_llm/serve/sample.py"
    serve_file.parent.mkdir(parents=True)
    serve_file.write_text("# does not mention var\n")
    commands_file = tmp_path / "tensorrt_llm/commands/sample.py"
    commands_file.parent.mkdir(parents=True)
    commands_file.write_text("# does not mention var\n")

    assert (
        _check_also_read_by("TRTLLM_FOO", "tensorrt_llm/grpc/openengine/missing.py", tmp_path)
        is None
    )
    assert (
        _check_also_read_by("TRTLLM_FOO", "tensorrt_llm/_torch/visual_gen/missing.py", tmp_path)
        is None
    )
    assert _check_also_read_by("TRTLLM_FOO", "/abs/path.py", tmp_path) is not None
    assert _check_also_read_by("TRTLLM_FOO", "../escape.py", tmp_path) is not None
    assert _check_also_read_by("TRTLLM_FOO", "tensorrt_llm/serve/missing.py", tmp_path) is not None
    assert _check_also_read_by("TRTLLM_FOO", "tensorrt_llm//serve/missing.py", tmp_path) is not None
    assert (
        _check_also_read_by("TRTLLM_FOO", "./tensorrt_llm/commands/missing.py", tmp_path)
        is not None
    )
    assert _check_also_read_by("TRTLLM_FOO", "tensorrt_llm/serve/sample.py", tmp_path) is not None
    assert (
        _check_also_read_by("TRTLLM_FOO", "tensorrt_llm/commands/sample.py", tmp_path) is not None
    )

    serve_file.write_text('os.getenv("TRTLLM_FOO")\n')
    commands_file.write_text('os.getenv("TRTLLM_FOO")\n')
    assert _check_also_read_by("TRTLLM_FOO", "tensorrt_llm/serve/sample.py", tmp_path) is None
    assert _check_also_read_by("TRTLLM_FOO", "./tensorrt_llm/commands/sample.py", tmp_path) is None


def test_secrets_are_marked() -> None:
    unmarked = [
        name
        for name, var in envs.environment_variables.items()
        if (name.endswith(SECRET_SUFFIXES) or name == "TLLM_EXECUTOR_ATTACH_INFO")
        and not var.secret
    ]
    assert not unmarked, (
        f"{unmarked} carry credentials; declare them with secret=True in tensorrt_llm/serve/envs.py"
    )


def test_type_declarations_match_the_declared_names() -> None:
    annotated = declared_env_names(REPO_ROOT)
    assert annotated == set(envs.environment_variables), (
        "the TYPE_CHECKING block of tensorrt_llm/serve/envs.py declares the type of every variable "
        f"and nothing else; missing {sorted(set(envs.environment_variables) - annotated)}, "
        f"stale {sorted(annotated - set(envs.environment_variables))}"
    )


def _module_level_string_assignments(path: Path) -> dict[str, str]:
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


def _envs_enums(path: Path) -> dict[str, dict[str, str]]:
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


def test_launcher_enums_are_declared() -> None:
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


def test_env_name_constants_are_declared() -> None:
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
