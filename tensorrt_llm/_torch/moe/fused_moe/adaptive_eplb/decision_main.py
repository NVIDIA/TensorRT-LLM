# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point of the adaptive EPLB decision process.

``python decision_main.py <socket fd> <op library>...`` runs in a fresh interpreter without MPI or
a GPU. It loads ``policy.py`` and ``runtime.py`` of this directory from their files under their
package names, so the process imports numpy and torch but not ``tensorrt_llm``, whose import would
cost each rank about 7 s and 1.3 GB more. Then it answers its rank's runtime (``runtime.serve``).
"""

import ctypes
import importlib.util
import os
import signal
import sys
from multiprocessing.connection import Connection

import torch

PACKAGE = "tensorrt_llm._torch.moe.fused_moe.adaptive_eplb"


def load(name: str):
    """Import the module ``name`` of this directory from its file, under its package name.

    ``runtime`` imports ``policy`` relative to the package, which resolves to the module loaded
    here first; the parent packages are not imported.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> None:
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: end with the parent
    for library in argv[2:]:
        torch.ops.load_library(library)
    load("policy")
    load("runtime").serve(Connection(int(argv[1])))


if __name__ == "__main__":
    main(sys.argv)
