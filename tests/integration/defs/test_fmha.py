import os
from functools import partial
from pathlib import Path
from subprocess import run


def test_fmha():
    build_run = partial(run, shell=True, check=True)

    current_dir = Path.cwd()
    project_dir = Path(__file__).parent.resolve().parent.parent.parent
    fmha_v2_dir = project_dir / "cpp/kernels/fmha_v2"

    try:
        os.chdir(fmha_v2_dir)

        env = os.environ.copy()
        env.update({
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "ENABLE_SM89_QMMA": "1",
            "ENABLE_HMMA_FP32": "1",
            "SCHEDULING_MODE": "1",
            "ENABLE_SM100": "1",
            "ENABLE_SM120": "1",
        })

        build_run(
            "rm -rf generated temp obj .pytest_cache __pycache__ bin cubin")
        build_run("python3 setup.py", env=env)
        build_run("make -j 16", env=env)
        build_run("pytest fmha_test.py", env=env)

    finally:
        os.chdir(current_dir)
