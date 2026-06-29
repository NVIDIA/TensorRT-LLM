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

"""Tests for the generated LLMC compatibility redirect."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CREATE_SCRIPT = REPO_ROOT / "examples" / "auto_deploy" / "llmc" / "create_standalone_package.py"
REDIRECT_ENV_VAR = "TRTLLM_REDIRECT_AD_TO_LLMC"


def _stub_preamble(package_dir: Path) -> str:
    bundled_torch_dir = package_dir / "bundled_torch"
    child_script = package_dir / "redirect_child.py"
    return textwrap.dedent(
        f"""
        import importlib.machinery
        import sys
        import types

        sys.path.insert(0, {str(package_dir)!r})
        REDIRECT_CHILD_SCRIPT = {str(child_script)!r}

        def make_module(name, *, is_package=False):
            module = types.ModuleType(name)
            module.__package__ = name if is_package else name.rpartition('.')[0]
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
            if is_package:
                module.__path__ = []
            sys.modules[name] = module
            return module

        trtllm = make_module('tensorrt_llm', is_package=True)
        trtllm_torch = make_module('tensorrt_llm._torch', is_package=True)
        trtllm_torch.__path__ = [{str(bundled_torch_dir)!r}]
        trtllm._torch = trtllm_torch

        for name in ('llmc.compile', 'llmc.custom_ops', 'llmc.export', 'llmc.models'):
            make_module(name, is_package=True)
        compat = make_module('llmc._compat')
        compat.TRTLLM_AVAILABLE = False
        make_module('modelopt')
        """
    )


@pytest.fixture(scope="module")
def generated_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    package_dir = tmp_path_factory.mktemp("llmc_redirect")
    subprocess.run(
        [sys.executable, str(CREATE_SCRIPT), "--output-dir", str(package_dir)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (package_dir / "llmc" / "trtllm_compat.py").is_file()
    (package_dir / "llmc" / "redirect_test_module.py").write_text(
        "class RedirectProbe:\n    pass\n\nVALUE = 42\n"
    )

    bundled_ad_dir = package_dir / "bundled_torch" / "auto_deploy"
    bundled_ad_dir.mkdir(parents=True)
    (bundled_ad_dir / "__init__.py").write_text(
        "raise RuntimeError('bundled AutoDeploy must not be imported')\n"
    )

    child_script = package_dir / "redirect_child.py"
    child_script.write_text(
        _stub_preamble(package_dir)
        + textwrap.dedent(
            """
            import base64
            import importlib
            import pickle
            import sys

            probe = pickle.loads(base64.b64decode(sys.argv[1]))
            canonical_module = importlib.import_module('llmc.redirect_test_module')
            legacy_module = importlib.import_module(
                'tensorrt_llm._torch.auto_deploy.redirect_test_module'
            )
            assert type(probe) is canonical_module.RedirectProbe
            assert legacy_module is canonical_module
            """
        )
    )
    return package_dir


def _run_generated(
    package_dir: Path, script: str, env_value: str | None
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_value is None:
        env.pop(REDIRECT_ENV_VAR, None)
    else:
        env[REDIRECT_ENV_VAR] = env_value

    return subprocess.run(
        [sys.executable, "-c", _stub_preamble(package_dir) + textwrap.dedent(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )


def _assert_success(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.parametrize("env_value", [None, "", "0", "false", "NO", "off"])
def test_redirect_is_disabled_by_default(generated_package: Path, env_value: str | None) -> None:
    result = _run_generated(
        generated_package,
        """
        import importlib
        import sys
        import llmc

        assert 'tensorrt_llm._torch.auto_deploy' not in sys.modules
        try:
            importlib.import_module('tensorrt_llm._torch.auto_deploy')
        except RuntimeError as exc:
            assert 'bundled AutoDeploy must not be imported' in str(exc)
        else:
            raise AssertionError('redirect unexpectedly enabled for falsey environment value')
        """,
        env_value=env_value,
    )
    _assert_success(result)


@pytest.mark.parametrize("env_value", ["1", "true", "YES", "on"])
def test_redirect_uses_canonical_llmc_modules(generated_package: Path, env_value: str) -> None:
    result = _run_generated(
        generated_package,
        """
        import importlib
        import sys
        import llmc

        legacy_root = importlib.import_module('tensorrt_llm._torch.auto_deploy')
        legacy_module = importlib.import_module(
            'tensorrt_llm._torch.auto_deploy.redirect_test_module'
        )
        canonical_module = importlib.import_module('llmc.redirect_test_module')

        assert legacy_root is llmc
        assert legacy_module is canonical_module
        assert legacy_module.__name__ == 'llmc.redirect_test_module'
        assert legacy_module.__package__ == 'llmc'
        assert legacy_module.__spec__.name == 'llmc.redirect_test_module'
        assert legacy_module.__loader__ is canonical_module.__loader__
        assert sys.modules['tensorrt_llm._torch.auto_deploy'] is llmc

        from llmc.trtllm_compat import install_autodeploy_redirect
        install_autodeploy_redirect()
        """,
        env_value=env_value,
    )
    _assert_success(result)


def test_redirect_can_be_installed_explicitly(generated_package: Path) -> None:
    result = _run_generated(
        generated_package,
        """
        import importlib
        import sys
        import llmc

        assert 'tensorrt_llm._torch.auto_deploy' not in sys.modules
        from llmc.trtllm_compat import install_autodeploy_redirect
        install_autodeploy_redirect()

        legacy_root = importlib.import_module('tensorrt_llm._torch.auto_deploy')
        assert legacy_root is llmc
        """,
        env_value=None,
    )
    _assert_success(result)


def test_redirect_bootstraps_during_child_unpickle(generated_package: Path) -> None:
    result = _run_generated(
        generated_package,
        """
        import base64
        import os
        import pickle
        import subprocess
        import sys

        import llmc
        from llmc.redirect_test_module import RedirectProbe

        payload = base64.b64encode(pickle.dumps(RedirectProbe())).decode('ascii')
        child = subprocess.run(
            [sys.executable, REDIRECT_CHILD_SCRIPT, payload],
            check=False,
            capture_output=True,
            env=os.environ.copy(),
            text=True,
            timeout=30,
        )
        assert child.returncode == 0, (child.stdout, child.stderr)
        """,
        env_value="true",
    )
    _assert_success(result)


def test_redirects_real_trtllm_runtime_entrypoints(generated_package: Path) -> None:
    env = os.environ.copy()
    env[REDIRECT_ENV_VAR] = "true"
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.util
        import pathlib
        import sys

        sys.path.insert(0, {str(generated_package)!r})

        def reject_bundled_ad(event, args):
            if event != 'open' or not args or not isinstance(args[0], (str, bytes)):
                return
            path = str(args[0]).replace('\\\\', '/')
            if '/tensorrt_llm/_torch/auto_deploy/' in path:
                raise RuntimeError(f'bundled AutoDeploy source opened: {{path}}')

        sys.addaudithook(reject_bundled_ad)
        if importlib.util.find_spec('tensorrt_llm') is None:
            print('TensorRT-LLM unavailable')
            raise SystemExit(77)

        import tensorrt_llm
        import llmc

        legacy_args = importlib.import_module('tensorrt_llm._torch.auto_deploy.llm_args')
        legacy_executor = importlib.import_module(
            'tensorrt_llm._torch.auto_deploy.shim.ad_executor'
        )
        assert legacy_args is importlib.import_module('llmc.llm_args')
        assert legacy_executor is importlib.import_module('llmc.shim.ad_executor')
        assert legacy_args.LlmArgs.__module__ == 'llmc.llm_args'
        assert pathlib.Path(legacy_args.__file__).is_relative_to({str(generated_package)!r})
        assert pathlib.Path(legacy_executor.__file__).is_relative_to({str(generated_package)!r})
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        cwd=generated_package,
        env=env,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    _assert_success(result)


def test_redirect_rejects_invalid_environment_value(generated_package: Path) -> None:
    result = _run_generated(generated_package, "import llmc", env_value="sometimes")
    assert result.returncode != 0
    assert f"{REDIRECT_ENV_VAR} must be a boolean value" in result.stderr


def test_redirect_rejects_preloaded_bundled_modules(generated_package: Path) -> None:
    result = _run_generated(
        generated_package,
        """
        import sys
        import types

        sys.modules['tensorrt_llm._torch.auto_deploy'] = types.ModuleType(
            'tensorrt_llm._torch.auto_deploy'
        )
        import llmc
        """,
        env_value="true",
    )
    assert result.returncode != 0
    assert "after bundled modules were loaded" in result.stderr
