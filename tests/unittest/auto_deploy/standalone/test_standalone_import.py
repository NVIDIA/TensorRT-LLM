# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CI test that validates auto_deploy works without TensorRT-LLM installed.

This test creates an isolated environment by building a stub tensorrt_llm package
in a temp directory and running Python subprocesses that use ONLY the stub
(via sys.path manipulation), NOT the real tensorrt_llm.

The stub provides:
  tensorrt_llm/__init__.py          (empty — just __version__)
  tensorrt_llm/_torch/__init__.py   (empty)
  tensorrt_llm/_torch/auto_deploy/  (symlink → real auto_deploy source)

This means `from tensorrt_llm.mapping import Mapping` will fail with
ModuleNotFoundError (no `mapping` module in the stub), causing _compat.py
to set TRTLLM_AVAILABLE = False — exactly the standalone scenario.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
AUTO_DEPLOY_SRC = os.path.join(REPO_ROOT, "tensorrt_llm", "_torch", "auto_deploy")


@pytest.fixture(scope="module")
def standalone_env(tmp_path_factory):
    """Create a stub tensorrt_llm package that makes _compat.TRTLLM_AVAILABLE = False.

    Returns a dict with 'stub_root' and a helper to get the right PYTHONPATH.
    """
    stub_base = str(tmp_path_factory.mktemp("standalone_stub"))

    # Create stub tensorrt_llm package
    stub_trtllm = os.path.join(stub_base, "tensorrt_llm")
    os.makedirs(os.path.join(stub_trtllm, "_torch"), exist_ok=True)

    with open(os.path.join(stub_trtllm, "__init__.py"), "w") as f:
        f.write('__version__ = "standalone"\n')

    with open(os.path.join(stub_trtllm, "_torch", "__init__.py"), "w") as f:
        f.write("")

    # Symlink the real auto_deploy source
    os.symlink(AUTO_DEPLOY_SRC, os.path.join(stub_trtllm, "_torch", "auto_deploy"))

    return {"stub_base": stub_base, "python": sys.executable}


def _run_standalone(env_info, script: str, timeout=120):
    """Run a Python script that sees the stub tensorrt_llm instead of the real one.

    We write a bootstrap script that:
    1. Prepends the stub directory to sys.path so `tensorrt_llm` resolves to the
       stub before any real install on PYTHONPATH, site-packages, or pth-added
       directories. We deliberately do NOT filter the rest of sys.path: in CI
       images, `torch` and `tensorrt_llm` live in the same site-packages, and
       stripping that directory would also remove torch (the real failure we
       hit before this change).
    2. Purges any cached tensorrt_llm modules so the first import resolves
       through the stub we just placed at the front.
    3. Executes the user script.
    """
    stub_base = env_info["stub_base"]
    preamble = textwrap.dedent(f"""\
import sys, importlib

_stub = {stub_base!r}
if _stub in sys.path:
    sys.path.remove(_stub)
sys.path.insert(0, _stub)

# Purge any cached tensorrt_llm modules so the stub is found on first import
for k in list(sys.modules):
    if k.startswith("tensorrt_llm"):
        del sys.modules[k]

# Clear import caches so Python re-discovers packages
importlib.invalidate_caches()

""")
    full_script = preamble + textwrap.dedent(script)

    result = subprocess.run(
        [env_info["python"], "-c", full_script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )
    return result


class TestStandaloneImport:
    """Tests that auto_deploy imports work when TRT-LLM is not available."""

    def test_trtllm_not_available(self, standalone_env):
        """Verify that _compat.TRTLLM_AVAILABLE is False in the standalone env."""
        result = _run_standalone(
            standalone_env,
            """
            from tensorrt_llm._torch.auto_deploy._compat import TRTLLM_AVAILABLE
            assert not TRTLLM_AVAILABLE, f"Expected TRTLLM_AVAILABLE=False, got {TRTLLM_AVAILABLE}"
            print("OK: TRTLLM_AVAILABLE is False")
            """,
        )
        assert result.returncode == 0, f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_auto_deploy_imports(self, standalone_env):
        """Verify that the auto_deploy package imports successfully."""
        result = _run_standalone(
            standalone_env,
            """
            import tensorrt_llm._torch.auto_deploy
            from tensorrt_llm._torch.auto_deploy._compat import TRTLLM_AVAILABLE
            assert not TRTLLM_AVAILABLE
            print("OK: auto_deploy imported without TRT-LLM")
            """,
        )
        assert result.returncode == 0, f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_custom_ops_register(self, standalone_env):
        """Verify standalone custom ops register under torch.ops.auto_deploy."""
        result = _run_standalone(
            standalone_env,
            """
            import torch
            import tensorrt_llm._torch.auto_deploy
            # Check that at least some standalone ops are registered
            ad_ops = [name for name in dir(torch.ops.auto_deploy) if not name.startswith('_')]
            assert len(ad_ops) > 0, f"No auto_deploy ops registered. Available: {ad_ops}"
            print(f"OK: {len(ad_ops)} ops registered")
            """,
        )
        assert result.returncode == 0, f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_transforms_register(self, standalone_env):
        """Verify transform library loads and registers transforms."""
        result = _run_standalone(
            standalone_env,
            """
            from tensorrt_llm._torch.auto_deploy.transform.interface import TransformRegistry
            registry = TransformRegistry._registry
            assert len(registry) > 0, f"No transforms registered: {registry}"
            print(f"OK: {len(registry)} transforms registered")
            """,
        )
        assert result.returncode == 0, f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_export_module_loads(self, standalone_env):
        """Verify the export library loads."""
        result = _run_standalone(
            standalone_env,
            """
            from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
            print("OK: export module loaded")
            """,
        )
        assert result.returncode == 0, f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_model_factory_loads(self, standalone_env):
        """Verify ModelFactoryRegistry loads."""
        result = _run_standalone(
            standalone_env,
            """
            from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
            entries = ModelFactoryRegistry.entries()
            assert len(entries) > 0, f"No model factories registered: {entries}"
            print(f"OK: {len(entries)} model factories registered")
            """,
        )
        assert result.returncode == 0, f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_compat_types_available(self, standalone_env):
        """Verify _compat types work in standalone mode."""
        result = _run_standalone(
            standalone_env,
            """
            from tensorrt_llm._torch.auto_deploy._compat import (
                TRTLLM_AVAILABLE, ActivationType, KvCacheConfig,
                get_free_port, str_dtype_to_torch, make_weak_ref,
            )
            from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
            assert not TRTLLM_AVAILABLE
            # Test enum
            assert ActivationType.Silu == 4
            # Test KvCacheConfig
            cfg = KvCacheConfig()
            assert cfg.tokens_per_block == 32
            # Test DistConfig (replaces the old Mapping shim)
            dc = DistConfig(world_size=1, rank=0, tp_size=1)
            assert dc.tp_rank == 0
            # Test utility
            import torch
            assert str_dtype_to_torch("float16") == torch.float16
            print("OK: all _compat types work in standalone mode")
            """,
        )
        assert result.returncode == 0, f"Failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
