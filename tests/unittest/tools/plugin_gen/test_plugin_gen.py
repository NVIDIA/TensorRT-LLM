import os
from importlib.metadata import version

import pytest

from .kernel_config import get_fmha_kernel_meta_data

KERNEL_META_DATA = get_fmha_kernel_meta_data()

try:
    from tensorrt_llm.tools.plugin_gen.plugin_gen import (TRITON_COMPILE_BIN,
                                                          gen_trt_plugins)
except ImportError:
    TRITON_COMPILE_BIN = "does_not_exist"

    def gen_trt_plugins(*args, **kwargs):
        pass


WORKSPACE = './tmp/'


def is_triton_installed() -> bool:
    # the triton detection does not work in PyTorch NGC 23.10 container
    try:
        if version('triton') != "2.1.0+440fd1b":
            return False
    except Exception:
        return False

    return os.path.exists(TRITON_COMPILE_BIN)


def is_trt_automation() -> bool:
    return os.path.exists("/build/config.yml")


@pytest.mark.skipif(
    not is_triton_installed() or is_trt_automation(),
    reason=
    'triton is not installed, this test is not supported in trt automation')
def test_end_to_end():
    gen_trt_plugins(workspace=WORKSPACE, metas=[KERNEL_META_DATA])
