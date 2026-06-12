# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose the session-scoped ``verl_setup`` fixture to every test in this directory.

The fixture is defined in ``test_verl_cases.py`` with ``autouse=True``, but
pytest's autouse semantics only apply within the defining module, so tests
in sibling files (``test_verl_E2E_*.py``) would hit KeyError on env vars
like ``TRTLLM_TEST_MODEL_PATH_ROOT`` and ModuleNotFoundError on
``verl.trainer.main_ppo`` without this re-export. Re-importing here makes
the fixture (and its autouse) visible to every test in this package, so
the E2E files inherit the verl clone/install + env-var export from
``verl_config.yml`` without having to import it themselves.
"""

from .test_verl_cases import verl_setup  # noqa: F401
