# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A generated wiki of one TensorRT-LLM checkout, for the agents that work on it.

The wiki holds what agents otherwise grep for or what a workflow would hard-code:
the LLM API fields (from the generated serve schema), ModelingV2 families,
routed architectures, targets and operation wrappers, the CI test lists that
run each test and its waives, the accuracy tasks and reference scores, the
serving configurations the checkout ships, and the environment variables it
reads. Each fact is extracted from the checkout's files (nothing is imported
from TensorRT-LLM) into a *pack* keyed by commit and uncommitted state, so it
follows the tree as it moves. The one hand-written table is
:data:`features.FEATURES`, held to the schema by tests.

``pack.ensure(repo, out)`` builds or reuses a pack, :class:`query.Wiki` answers
from it, and ``trtllm-wiki`` is the command line.
"""

from .pack import WIKI_SCHEMA, build, ensure
from .query import Wiki

__all__ = ["WIKI_SCHEMA", "Wiki", "build", "ensure"]
