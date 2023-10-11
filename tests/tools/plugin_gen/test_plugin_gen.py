import os

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
    return os.path.exists(TRITON_COMPILE_BIN)


@pytest.mark.skipif(not is_triton_installed(), reason='triton is not installed')
def test_end_to_end():
    gen_trt_plugins(workspace=WORKSPACE, metas=[KERNEL_META_DATA])
