# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interface stub for the fused MuseSpark expert-finalize + TP all-reduce kernel.

The optimized kernel is being reworked and is not present in this build. This
stub keeps the module importable and reports the kernel unavailable, so every
caller selects the unfused finalize + AllReduce fallback automatically:

* ``HAS_CUTEDSL`` is ``False``: workspace construction is skipped and the
  cluster-boundary kernel's helper imports stay un-executed.
* ``FinalizeIpcWorkspace`` raises on direct construction; the guarded builder
  in ``modeling_musespark_open`` never reaches it while ``HAS_CUTEDSL`` is
  ``False``.

The multi-rank integration tests in
``tests/unittest/_torch/modeling/test_musespark_open_finalize_allreduce.py`` skip
while the kernel is unavailable and double as the acceptance harness for the
reworked implementation.
"""

HAS_CUTEDSL = False


class FinalizeIpcWorkspace:
    """Placeholder for the fused finalize + all-reduce workspace."""

    def __init__(self, mapping=None, *, use_mnnvl: bool = False):
        raise NotImplementedError(
            "The fused finalize+all-reduce kernel is not available in this "
            "build; the unfused finalize + AllReduce path is selected "
            "automatically."
        )
