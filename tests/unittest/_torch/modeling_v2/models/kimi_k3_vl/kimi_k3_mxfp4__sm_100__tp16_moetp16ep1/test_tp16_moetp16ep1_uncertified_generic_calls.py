# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target's ``UNCERTIFIED_GENERIC_CALLS`` names every stock tensorrt_llm import of the module
that declares it (its ``modeling.py`` and any other module of the target), and nothing else.

The check is ``_kimi_k3_generic_calls``; it reads the source and imports nothing."""

import pytest

__extra_import_path__ = [".."]
from _kimi_k3_generic_calls import (  # noqa: E402
    FAMILY_DIR,
    check_names_every_stock_import,
    declaring_modules,
)

pytestmark = pytest.mark.cpu_only

TARGET = "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"
MODULES = declaring_modules(TARGET)


def test_modeling_declares_the_generic_calls():
    assert FAMILY_DIR / TARGET / "modeling.py" in MODULES, (
        f"{TARGET}/modeling.py does not declare UNCERTIFIED_GENERIC_CALLS"
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(FAMILY_DIR / TARGET)))
def test_uncertified_generic_calls_name_every_stock_import(path):
    check_names_every_stock_import(path, TARGET)
