# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-merge pack: what a delivered branch still owes before it is a PR.

Every step here was first done by hand at the end of an onboarding, after
every unit was green, and every one of them found something:

- :mod:`.lint` — the repository's pre-commit hooks over the changed files,
  twice; the formatting is committed only when the second pass is clean, and
  only with an AST proof that it changed nothing but formatting.
- :mod:`.ci_case` — a test-db line run the way CI runs it (the line itself as
  the test list, CI's pytest flags, CI's working directory), recording the
  ``HEAD`` it measured at start and at end.
- :mod:`.copyright` — NVIDIA headers on new source files, current year on
  modified ones.
- :mod:`.split` — the branch split by path into reviewable PR stacks, with
  the checks that the stacks together reproduce the branch.
- :mod:`.resign` — a non-destructive DCO re-sign that signs each commit as
  its own author.
- :mod:`.pr_open` — opening a draft PR; opt-in, and dry-run unless told
  otherwise. Nothing in this package pushes.

``staircase-onboard premerge <step> ...`` runs any of them; :mod:`.pack`
writes the cheap ones into a report at the end of delivery.

Target-path flattening (``flatten_target.sh`` in the run that produced this
pack) is not ported: the workflow now places targets at
``models/<family>/<checkpoint>__<arch>__<parallel>/`` from the start, so there
is no nested layout left to flatten.
"""

from __future__ import annotations

__all__: list[str] = []
