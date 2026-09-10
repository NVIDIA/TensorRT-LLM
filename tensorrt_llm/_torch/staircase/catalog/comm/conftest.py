# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep pytest from collecting the collective entries' rank bodies.

``allgather_test.py`` and ``reducescatter_test.py`` are their own launchers:
their ``test_*`` functions read module-global rank state that only
``_run_one_rank`` binds, and several of them assert on communicator state the
previous test left behind, so they are a fixed sequence inside one 4-rank job
rather than independent cases. Collected directly they would run at world size
1 against unbound globals.

``tests/unittest/_torch/staircase/comm/test_staircase_*_op_matrix.py`` are the
collected entry points; each starts the 4-rank job and reports its result.
These two files stay here rather than moving with them because the launcher
re-execs them as ``python -m`` and the ranks need this package context for
their relative imports.
"""

collect_ignore = ["allgather_test.py", "reducescatter_test.py"]
