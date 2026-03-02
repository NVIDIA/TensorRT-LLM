import os
from functools import partial
from pathlib import Path
from subprocess import run

from tests.unittest.utils.util import getSMVersion


def test_fmha():
    build_run = partial(run, shell=True, check=True)

    current_dir = Path.cwd()
    project_dir = Path(__file__).parent.resolve().parent.parent.parent
    fmha_v2_dir = project_dir / "cpp/kernels/fmha_v2"

    try:
        os.chdir(fmha_v2_dir)

        test_arch = getSMVersion()
        # SM70 is deprecated in TRTLLM, so we don't need to test it
        all_archs = [80, 86, 89, 90, 100, 120]

        # Select the family we belong to (e.g. 103 -> 100)
        test_arch = max(filter(lambda x: x <= test_arch, all_archs))

        build_only_on_archs = filter(lambda x: x != test_arch, all_archs)

        env = os.environ.copy()
        env.update({
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "ENABLE_SM89_QMMA": "1",
            "ENABLE_HMMA_FP32": "1",
            "SCHEDULING_MODE": "1",
            "ENABLE_SM100": "1",
            "ENABLE_SM120": "1",
            "DISABLE_SKIP_SOFTMAX":
            "1",  # Do not run tests with skip-softmax feature.
        })

        # The test executable is too large if we build all the architectures, so we must build architectures individually
        def build_arch(arch):
            env["FMHA_FILTER_ARCH"] = str(arch)
            build_run(
                "rm -rf generated temp obj .pytest_cache __pycache__ bin cubin")
            build_run("python3 setup.py", env=env)
            build_run("make -j 16", env=env)

        # As part of the test we should compile all the architectures, even the ones we dont have executors for
        for arch in build_only_on_archs:
            build_arch(arch)

        build_arch(test_arch)
        build_run("pytest fmha_test.py", env=env)

    finally:
        os.chdir(current_dir)
