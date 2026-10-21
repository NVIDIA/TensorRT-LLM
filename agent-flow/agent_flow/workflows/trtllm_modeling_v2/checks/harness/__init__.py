# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Programs the gates run on the compute side, under the interpreter of the tree being measured.

Nothing in this package imports ``agent_flow``: each script runs as
``<tree's python> <this directory>/<script>.py`` inside a gate's job, where
``agent_flow`` is generally not installed. Each imports its siblings by
path, and the orchestrator imports the same modules as a package to parse
what they print.
"""
